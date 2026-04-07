# optimization/greedy_solver.py (或所在的相应文件)
import numpy as np # 💡 新增 numpy 导入
from objective import compute_cost
from latency import estimate_aoi
from constraints.resource import check_memory_constraint
from constraints.bandwidth import check_bandwidth_constraint
from utils.logger import logger


def select_best_node(simulator, network, task_chain, user, alpha, beta, request_time):
    best_score = float('inf')
    best_node = None
    best_metrics = None  # 用于记录最优解的具体指标，方便打印
    
    edge_nodes = network.get_edge_nodes()

    max_comp = max(n.compute_power for n in edge_nodes)
    min_comp = min(n.compute_power for n in edge_nodes)
    min_c = min(n.cost for n in edge_nodes)
    max_c = max(n.cost for n in edge_nodes)
    
    all_bws = [data['edge'].bandwidth for u, v, data in network.graph.edges(data=True)]
    min_bw = min(all_bws) if all_bws else 1.0

    workload = sum(t.workload for t in task_chain.tasks)
    mig_cost = sum(t.deployment_cost for t in task_chain.tasks)
    
    sensor_id = task_chain.sensor_id
    sensor = network.get_node(sensor_id)
    sensor_data_size = sensor.data_size

    # 理论 Cost 边界
    min_env_cost = min_c * (workload / max_comp)
    max_env_cost = max_c * (workload / min_comp) + mig_cost

    # 理论 AoI 边界
    min_env_aoi = 0.0
    worst_sense = sensor_data_size / min_bw
    worst_comp = workload / min_comp
    worst_queue = worst_comp * 5.0
    worst_res = 1.0 / min_bw
    max_env_aoi = worst_sense + worst_comp + worst_queue + worst_res


    # ==========================================
    # 开始遍历所有边缘节点，寻找最优解
    # ==========================================
    for node in edge_nodes:
        # C2 资源约束
        if not check_memory_constraint(node, task_chain, simulator):
            logger.debug(f"    [剪枝] Node {node.id} 内存不足 (剩余 {node.available_memory}G < 需求)")
            continue

        # C4 带宽约束
        raw_data_size = sensor.data_size
        path_up = network.get_path(sensor_id, node.id, raw_data_size)
        if not check_bandwidth_constraint(network, sensor_id, node.id, raw_data_size, task_chain.required_bandwidth):
            if not path_up:
                logger.debug(f"    [剪枝] Node {node.id} 带宽不足 (无上行路径）")
            else:
                bw = network.get_path_bandwidth(path_up)
                logger.debug(f"    [剪枝] Node {node.id} 带宽不足 (当前 {bw}M < 需求 {task_chain.required_bandwidth}M)")
            continue

        if not network.get_path(node.id, user.id, 1.0):
            logger.debug(f"    [剪枝] 用户 {user.id} 到 {node.id} 无路径")
            continue

        # 1. 试探性评估
        sense, q_time, comp, res, migration = simulator.evaluate_step(node, task_chain, user, request_time)

        # 2. 计算目标函数的真实值
        aoi = estimate_aoi(sense, q_time, comp, res)
        cost = compute_cost(node, comp, task_chain, migration)
        
        # Min-Max 归一化
        norm_aoi = float(np.clip((aoi - min_env_aoi) / (max_env_aoi - min_env_aoi + 1e-8), 0.0, 1.0))
        norm_cost = float(np.clip((cost - min_env_cost) / (max_env_cost - min_env_cost + 1e-8), 0.0, 1.0))

        # Score 现在基于归一化后的数据计算 (值域 0~1)
        score = alpha * norm_aoi + beta * norm_cost

        # 打印日志 (如果你在测 1000 个真实请求，建议把这里的 info 改成 debug，不然控制台会卡死)
        # logger.debug(f"    [评估] Node {node.id}: Score={score:.3f} (Norm AoI={norm_aoi:.2f}, Norm Cost={norm_cost:.2f})")

        # 3. 记录最优
        if score < best_score:
            best_score = score
            best_node = node
            # 返回出去的还是纯物理数值，确保后续画图和统计完全正常
            best_metrics = {"aoi": aoi, "cost": cost, "migration": migration}

    return best_node, best_score, best_metrics
