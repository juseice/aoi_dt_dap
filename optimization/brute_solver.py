# optimization/brute_solver.py
import itertools
import numpy as np
from objective import compute_cost
from latency import estimate_aoi
from constraints.resource import check_memory_constraint
from constraints.bandwidth import check_bandwidth_constraint
from utils.logger import logger


def select_best_node(simulator, network, task_chain, user, alpha, beta, request_time):
    """
    贪心穷举策略 (Greedy Brute-Force)

    枚举任务链在所有边缘节点上的分布式部署组合，选出归一化联合得分最低的方案。
    搜索空间：|nodes|^K，K 为任务链长度。
    """
    tasks = task_chain.tasks
    edge_nodes = network.get_edge_nodes()
    sensor_id = task_chain.sensor_id
    sensor = network.get_node(sensor_id)

    # ==========================================
    # 预计算归一化边界
    # ==========================================
    max_comp_power = max(n.compute_power for n in edge_nodes)
    min_comp_power = min(n.compute_power for n in edge_nodes)
    min_node_cost = min(n.cost for n in edge_nodes)
    max_node_cost = max(n.cost for n in edge_nodes)

    all_bws = [data['edge'].bandwidth for u, v, data in network.graph.edges(data=True)]
    min_bw = min(all_bws) if all_bws else 1.0

    total_mig_cost = sum(t.deployment_cost for t in tasks)

    min_env_cost = sum(min_node_cost * t.workload / max_comp_power for t in tasks)
    max_env_cost = sum(max_node_cost * t.workload / min_comp_power for t in tasks) + total_mig_cost

    worst_sense = sensor.data_size / min_bw
    worst_comp = sum(t.workload / min_comp_power for t in tasks)
    worst_inter = sum(t.output_data_size / min_bw for t in tasks[:-1])
    worst_queue = worst_comp * 5.0
    worst_res = 1.0 / min_bw
    max_env_aoi = worst_sense + worst_comp + worst_inter + worst_queue + worst_res

    logger.debug(
        f"    [搜索空间] {len(edge_nodes)}^{len(tasks)} = "
        f"{len(edge_nodes) ** len(tasks)} 个候选组合"
    )

    # ==========================================
    # 遍历所有部署组合，寻找最优解
    # ==========================================
    best_score = float('inf')
    best_node_list = None
    best_metrics = None

    for node_list in itertools.product(edge_nodes, repeat=len(tasks)):
        # --- 剪枝：逐任务检查约束 ---

        feasible = True
        for i, (task, node) in enumerate(zip(tasks, node_list)):
            # C2 内存约束
            if not check_memory_constraint(node, task, task_chain.id, i, simulator):
                logger.debug(f"    [剪枝] 组合 {[n.id for n in node_list]}: task {i} 节点 {node.id} 内存不足")
                feasible = False
                break

            # 第一个任务：C4 传感器上行带宽约束
            if i == 0 and not check_bandwidth_constraint(
                network, sensor_id, node.id, sensor.data_size, task_chain.required_bandwidth
            ):
                logger.debug(f"    [剪枝] 组合 {[n.id for n in node_list]}: 首节点 {node.id} 上行带宽不足")
                feasible = False
                break

            # 最后一个任务：下行连通性约束（节点 → 用户）
            if i == len(tasks) - 1 and not network.get_path(node.id, user.id, 1.0):
                logger.debug(f"    [剪枝] 组合 {[n.id for n in node_list]}: 末节点 {node.id} 无下行路径至用户")
                feasible = False
                break

        if not feasible:
            continue

        # 相邻任务节点间的连通性
        for i in range(len(tasks) - 1):
            src, dst = node_list[i], node_list[i + 1]
            if src.id != dst.id and not network.get_path(src.id, dst.id, tasks[i].output_data_size):
                logger.debug(
                    f"    [剪枝] 组合 {[n.id for n in node_list]}: "
                    f"task {i}→{i+1} 节点 {src.id}→{dst.id} 无路径"
                )
                feasible = False
                break

        if not feasible:
            continue

        # --- 评估：计算目标函数得分 ---
        sense, q_time, comp, res, migration = simulator.evaluate_step(
            node_list, task_chain, user, request_time
        )

        aoi = estimate_aoi(sense, q_time, comp, res)
        cost = compute_cost(node_list, task_chain, migration)

        norm_aoi = float(np.clip(aoi / (max_env_aoi + 1e-8), 0.0, 1.0))
        norm_cost = float(np.clip(
            (cost - min_env_cost) / (max_env_cost - min_env_cost + 1e-8), 0.0, 1.0
        ))

        score = alpha * norm_aoi + beta * norm_cost

        if score < best_score:
            best_score = score
            best_node_list = list(node_list)
            best_metrics = {"aoi": aoi, "cost": cost, "migration": migration}

    return best_node_list, best_score, best_metrics
