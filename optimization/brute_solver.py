'''
Take decision
'''

from objective import compute_cost
from latency import estimate_aoi
from constraints.resource import check_memory_constraint
from constraints.bandwidth import check_bandwidth_constraint
from utils.logger import logger


def select_best_node(simulator, network, task_chain, user, alpha, beta, request_time):
    best_score = float('inf')
    best_node = None
    best_metrics = None  # 用于记录最优解的具体指标，方便打印

    for node in network.get_edge_nodes():
        # C2 资源约束
        if not check_memory_constraint(node, task_chain, simulator):
            logger.info(f"    [剪枝] Node {node.id} 内存不足 (剩余 {node.available_memory}G < 需求)")
            continue

        # C4 带宽约束
        sensor_id = task_chain.sensor_id
        sensor = network.get_node(sensor_id)
        raw_data_size = sensor.data_size
        path_up = network.get_path(sensor_id, node.id, sensor.data_size)
        if not check_bandwidth_constraint(network, sensor_id, node.id, raw_data_size, task_chain.required_bandwidth):
            if not path_up:
                logger.info(f"    [剪枝] Node {node.id} 带宽不足 (无上行路径）")
            else:
                bw = network.get_path_bandwidth(path_up)
                logger.info(f"    [剪枝] Node {node.id} 带宽不足 (当前 {bw}M < 需求 {task_chain.required_bandwidth}M)")
            continue

        if not network.get_path(node.id, user.id, 1.0):
            logger.info(f"    [剪枝] 用户 {user.id} 到 {node.id} 无路径")
            continue

        # 1. 试探性评估
        sense, q_time, comp, res, migration = simulator.evaluate_step(node, task_chain, user, request_time)

        # 2. 计算目标函数
        aoi = estimate_aoi(sense, q_time, comp, res)
        cost = compute_cost(node, comp, task_chain, migration)
        score = alpha * cost + beta * aoi

        logger.info(f"    [评估] Node {node.id}: Score={score:.2f} (AoI={aoi:.2f}, Cost={cost})")

        # 3. 记录最优
        if score < best_score:
            best_score = score
            best_node = node
            best_metrics = {"aoi": aoi, "cost": cost, "migration": migration}

    return best_node, best_score, best_metrics
