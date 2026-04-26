# optimization/random_solver.py
import random
import numpy as np
from objective import compute_cost
from latency import estimate_aoi
from constraints.resource import check_memory_constraint
from constraints.bandwidth import check_bandwidth_constraint
from utils.logger import logger


def select_random_node(simulator, network, task_chain, user, alpha, beta, request_time):
    """
    随机分配策略 (Random Baseline)

    对任务链中的每个任务，独立地从满足约束（内存、带宽、连通性）的
    候选节点中等概率随机选择一个，返回 node_list。
    """
    tasks = task_chain.tasks
    edge_nodes = network.get_edge_nodes()
    sensor_id = task_chain.sensor_id
    sensor = network.get_node(sensor_id)

    # ==========================================
    # 1. 过滤阶段：逐任务筛选合法候选节点
    # ==========================================
    feasible_per_task = []

    for i, task in enumerate(tasks):
        feasible = []

        for node in edge_nodes:
            # C2 内存约束（同节点复用时无需额外分配）
            if not check_memory_constraint(node, task, task_chain.id, i, simulator):
                continue

            # 第一个任务：C4 传感器上行带宽约束
            if i == 0 and not check_bandwidth_constraint(
                network, sensor_id, node.id, sensor.data_size, task_chain.required_bandwidth
            ):
                continue

            # 最后一个任务：下行连通性约束（节点 → 用户）
            if i == len(tasks) - 1 and not network.get_path(node.id, user.id, 1.0):
                continue

            feasible.append(node)

        if not feasible:
            return None, float('inf'), None

        feasible_per_task.append(feasible)

    # ==========================================
    # 2. 决策阶段：每个任务独立随机选节点
    # ==========================================
    node_list = [random.choice(f) for f in feasible_per_task]

    # 校验相邻任务节点之间的连通性（不同节点才需要路径）
    for i in range(len(tasks) - 1):
        src, dst = node_list[i], node_list[i + 1]
        if src.id != dst.id and not network.get_path(src.id, dst.id, tasks[i].output_data_size):
            return None, float('inf'), None

    # ==========================================
    # 3. 评估阶段：计算指标与归一化 Score
    # ==========================================
    sense, q_time, comp, res, migration = simulator.evaluate_step(
        node_list, task_chain, user, request_time
    )

    aoi = estimate_aoi(sense, q_time, comp, res)
    cost = compute_cost(node_list, task_chain, migration)

    # ------------------------------------------
    # 动态计算当前请求的理论上下界用于归一化
    # ------------------------------------------
    max_comp_power = max(n.compute_power for n in edge_nodes)
    min_comp_power = min(n.compute_power for n in edge_nodes)
    min_node_cost = min(n.cost for n in edge_nodes)
    max_node_cost = max(n.cost for n in edge_nodes)

    all_bws = [data['edge'].bandwidth for u, v, data in network.graph.edges(data=True)]
    min_bw = min(all_bws) if all_bws else 1.0

    total_mig_cost = sum(t.deployment_cost for t in tasks)
    sensor_data_size = sensor.data_size

    # 理论 Cost 边界（按每个任务独立计算）
    min_env_cost = sum(min_node_cost * t.workload / max_comp_power for t in tasks)
    max_env_cost = sum(max_node_cost * t.workload / min_comp_power for t in tasks) + total_mig_cost

    # 理论 AoI 边界（保守估计，含排队冗余和最差中间传输）
    worst_sense = sensor_data_size / min_bw
    worst_comp = sum(t.workload / min_comp_power for t in tasks)
    worst_inter = sum(t.output_data_size / min_bw for t in tasks[:-1])
    worst_queue = worst_comp * 5.0
    worst_res = 1.0 / min_bw
    max_env_aoi = worst_sense + worst_comp + worst_inter + worst_queue + worst_res

    norm_aoi = float(np.clip(aoi / (max_env_aoi + 1e-8), 0.0, 1.0))
    norm_cost = float(np.clip(
        (cost - min_env_cost) / (max_env_cost - min_env_cost + 1e-8), 0.0, 1.0
    ))

    score = alpha * norm_aoi + beta * norm_cost
    metrics = {"aoi": aoi, "cost": cost, "migration": migration}

    return node_list, score, metrics
