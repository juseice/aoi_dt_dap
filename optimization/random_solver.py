# optimization/random_solver.py
import random
import numpy as np  # 💡 新增 numpy 用于归一化裁剪
from objective import compute_cost
from latency import estimate_aoi
from constraints.resource import check_memory_constraint
from constraints.bandwidth import check_bandwidth_constraint
from utils.logger import logger


def select_random_node(simulator, network, task_chain, user, alpha, beta, request_time):
    """
    随机分配策略 (Random Baseline)
    从所有满足物理约束（内存、带宽、网络连通性）的候选节点中，等概率随机选择一个。
    """
    feasible_nodes = []
    edge_nodes = network.get_edge_nodes()

    # ==========================================
    # 1. 过滤阶段：找出所有合法的候选节点
    # ==========================================
    for node in edge_nodes:
        # C2 资源约束
        if not check_memory_constraint(node, task_chain, simulator):
            continue

        # C4 带宽约束 (传感器 -> 边缘节点)
        sensor_id = task_chain.sensor_id
        sensor = network.get_node(sensor_id)
        raw_data_size = sensor.data_size
        if not check_bandwidth_constraint(network, sensor_id, node.id, raw_data_size, task_chain.required_bandwidth):
            continue

        # 连通性约束 (边缘节点 -> 用户)
        if not network.get_path(node.id, user.id, 1.0):
            continue

        # 通过所有校验，加入可用候选池
        feasible_nodes.append(node)

    # ==========================================
    # 2. 决策阶段：无脑随机选
    # ==========================================
    if not feasible_nodes:
        # 如果资源枯竭，没有合法节点，返回失败
        return None, float('inf'), None

    # 从合法节点中随机挑一个
    selected_node = random.choice(feasible_nodes)

    # ==========================================
    # 3. 评估阶段：计算指标与归一化 Score
    # ==========================================
    # 此时只是算指标用于记录和打印，依然不会改变物理机状态
    sense, q_time, comp, res, migration = simulator.evaluate_step(selected_node, task_chain, user, request_time)

    # 真实物理指标
    aoi = estimate_aoi(sense, q_time, comp, res)
    cost = compute_cost(selected_node, comp, task_chain, migration)

    # ------------------------------------------
    # 动态计算当前请求的理论上下界，以对齐 RL 的归一化逻辑
    # ------------------------------------------
    max_comp = max(n.compute_power for n in edge_nodes)
    min_comp = min(n.compute_power for n in edge_nodes)
    min_c = min(n.cost for n in edge_nodes)
    max_c = max(n.cost for n in edge_nodes)

    # 提取全网最小带宽（防止极端拥塞）
    all_bws = [data['edge'].bandwidth for u, v, data in network.graph.edges(data=True)]
    min_bw = min(all_bws) if all_bws else 1.0

    workload = sum(t.workload for t in task_chain.tasks)
    mig_cost = sum(t.deployment_cost for t in task_chain.tasks)
    sensor_data_size = network.get_node(task_chain.sensor_id).data_size

    # 理论 Cost 边界
    min_env_cost = min_c * (workload / max_comp)
    max_env_cost = max_c * (workload / min_comp) + mig_cost

    # 理论 AoI 边界
    min_env_aoi = 0.0
    worst_sense = sensor_data_size / min_bw
    worst_comp = workload / min_comp
    worst_queue = worst_comp * 5.0  # 与 RL 保持一致的排队冗余预估
    worst_res = 1.0 / min_bw
    max_env_aoi = worst_sense + worst_comp + worst_queue + worst_res

    # Min-Max 归一化处理
    norm_aoi = float(np.clip((aoi - min_env_aoi) / (max_env_aoi - min_env_aoi + 1e-8), 0.0, 1.0))
    norm_cost = float(np.clip((cost - min_env_cost) / (max_env_cost - min_env_cost + 1e-8), 0.0, 1.0))

    # 最终的归一化联合得分 (值域严格在 0 ~ 1 之间)
    score = alpha * norm_aoi + beta * norm_cost

    # 记录字典里依然保留真实的物理数值供画图使用
    metrics = {"aoi": aoi, "cost": cost, "migration": migration}

    return selected_node, score, metrics
