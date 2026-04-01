# optimization/random_solver.py
import random
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

    # ==========================================
    # 1. 过滤阶段：找出所有合法的候选节点
    # ==========================================
    for node in network.get_edge_nodes():
        # C2 资源约束
        if not check_memory_constraint(node, task_chain, simulator):
            # logger.info(f"内存淘汰 {node.id}")
            continue

        # C4 带宽约束 (传感器 -> 边缘节点)
        sensor_id = task_chain.sensor_id
        sensor = network.get_node(sensor_id)
        raw_data_size = sensor.data_size
        if not check_bandwidth_constraint(network, sensor_id, node.id, raw_data_size, task_chain.required_bandwidth):
            # logger.info(f"上行带宽淘汰 {node.id}")
            continue

        # 连通性约束 (边缘节点 -> 用户)
        if not network.get_path(node.id, user.id, 1.0):
            # logger.info(f"用户联通淘汰 {node.id}")
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
    # 3. 评估阶段：计算选定节点的各项指标
    # ==========================================
    # 此时只是算指标用于记录和打印，依然不会改变物理机状态
    sense, q_time, comp, res, migration = simulator.evaluate_step(selected_node, task_chain, user, request_time)

    aoi = estimate_aoi(sense, q_time, comp, res)
    cost = compute_cost(selected_node, comp, task_chain, migration)
    score = alpha * cost + beta * aoi

    metrics = {"aoi": aoi, "cost": cost, "migration": migration}

    return selected_node, score, metrics
