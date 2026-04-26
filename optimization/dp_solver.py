# optimization/dp_solver.py
import itertools
import numpy as np
from utils.logger import logger
from latency.sensing import compute_sensing_delay
from latency.transmission import compute_transmission_delay


def solve_dp_offline(network, request_stream, alpha, beta):
    """
    DP-ODA with distributed chain placement.

    State at each step: tuple of node IDs, one per task in the chain.
    State space per step: |nodes|^K  (K = number of tasks in the chain).
    Migration: charged as full chain cost whenever the placement tuple changes.
    """
    N = len(request_stream)
    if N == 0:
        return []

    nodes = network.get_edge_nodes()

    # ==========================================
    # 辅助函数：计算无排队物理延迟和基础运行成本
    # node_list: tuple/list of EdgeNode, one per task
    # ==========================================
    def get_base_metrics(req, node_list):
        tasks = req.task_chain.tasks
        sensor_id = req.task_chain.sensor_id
        sensor = network.get_node(sensor_id)

        path_up = network.get_path(sensor_id, node_list[0].id, sensor.data_size)
        path_down = network.get_path(node_list[-1].id, req.user.id, 1.0)
        if not path_up or not path_down:
            return float('inf'), float('inf')

        bw_up = network.get_path_bandwidth(path_up)
        if bw_up < req.task_chain.required_bandwidth:
            return float('inf'), float('inf')

        sense_delay = compute_sensing_delay(network, sensor_id, node_list[0].id, sensor.data_size)

        total_comp = 0.0
        total_inter = 0.0
        run_cost = 0.0

        for i, (task, node) in enumerate(zip(tasks, node_list)):
            comp = task.workload / node.compute_power
            total_comp += comp
            run_cost += node.cost * comp

            # 中间传输时延（任务 i → 任务 i+1，不同节点才计算）
            if i < len(tasks) - 1:
                next_node = node_list[i + 1]
                if node.id != next_node.id:
                    out_size = task.output_data_size
                    path_inter = network.get_path(node.id, next_node.id, out_size)
                    if not path_inter:
                        return float('inf'), float('inf')
                    bw_inter = network.get_path_bandwidth(path_inter)
                    inter = (
                        compute_transmission_delay(out_size, bw_inter)
                        if bw_inter > 0 else float('inf')
                    )
                    if inter == float('inf'):
                        return float('inf'), float('inf')
                    total_inter += inter

        bw_down = network.get_path_bandwidth(path_down)
        res_delay = (
            compute_transmission_delay(1.0, bw_down) if bw_down > 0 else float('inf')
        )

        aoi = sense_delay + total_comp + total_inter + res_delay
        return aoi, run_cost

    # ==========================================
    # 1. 预扫描与缓存 (Pre-computation & Cache)
    # ==========================================
    # State key: tuple of node IDs, one per task
    # metrics_cache[n][placement_key] = (raw_aoi, raw_run_cost)
    metrics_cache = [{} for _ in range(N)]

    min_aoi, max_aoi = float('inf'), -float('inf')
    min_cost, max_cost = float('inf'), -float('inf')

    logger.info("  >> 正在执行预扫描 (Pre-computation Pass)... 计算全局上下界并缓存。")

    for n in range(N):
        req_n = request_stream[n]
        n_tasks = len(req_n.task_chain.tasks)
        mig_cost_val = sum(task.deployment_cost for task in req_n.task_chain.tasks)

        for node_list in itertools.product(nodes, repeat=n_tasks):
            key = tuple(node.id for node in node_list)
            aoi, run_cost = get_base_metrics(req_n, node_list)
            metrics_cache[n][key] = (aoi, run_cost)

            if aoi != float('inf'):
                min_aoi = min(min_aoi, aoi)
                max_aoi = max(max_aoi, aoi)
                min_cost = min(min_cost, run_cost)
                max_cost = max(max_cost, run_cost + mig_cost_val)

    aoi_range = max_aoi - min_aoi + 1e-8
    cost_range = max_cost - min_cost + 1e-8

    n_states = len(metrics_cache[0])
    logger.info(f"     状态空间每步 {n_states} 个，AoI 边界: [{min_aoi:.3f}, {max_aoi:.3f}], Cost 边界: [{min_cost:.3f}, {max_cost:.3f}]")
    logger.info("  >> 正在执行 DP-ODA 正向状态推导...")

    # ==========================================
    # 2. 动态规划正向推导 (Forward Pass)
    # ==========================================
    dp_table = [{} for _ in range(N)]
    pre_placement = [{} for _ in range(N)]
    metrics_record = [{} for _ in range(N)]

    # 初始化首个决策步 (n = 0)：第一步必然发生迁移（从无到有）
    req_0 = request_stream[0]
    mig_cost_base = sum(task.deployment_cost for task in req_0.task_chain.tasks)

    for key, (aoi, run_cost) in metrics_cache[0].items():
        if aoi == float('inf'):
            dp_table[0][key] = float('inf')
            continue
        step_cost = run_cost + mig_cost_base
        norm_aoi = np.clip((aoi - min_aoi) / aoi_range, 0.0, 1.0)
        norm_cost = np.clip((step_cost - min_cost) / cost_range, 0.0, 1.0)
        dp_table[0][key] = alpha * norm_aoi + beta * norm_cost
        pre_placement[0][key] = None
        metrics_record[0][key] = (aoi, step_cost, True)

    # 状态转移方程推导 (n = 1 to N-1)
    for n in range(1, N):
        req_n = request_stream[n]
        mig_cost_val = sum(task.deployment_cost for task in req_n.task_chain.tasks)

        for key, (aoi, run_cost) in metrics_cache[n].items():
            if aoi == float('inf'):
                dp_table[n][key] = float('inf')
                continue

            norm_aoi = np.clip((aoi - min_aoi) / aoi_range, 0.0, 1.0)

            min_cum_cost = float('inf')
            best_pre = None
            best_step_cost = None
            is_mig = False

            for prev_key, prev_cost in dp_table[n - 1].items():
                if prev_cost == float('inf'):
                    continue

                # 整条链的部署位置发生任何变化即视为迁移
                mig_flag = (key != prev_key)
                step_cost = run_cost + (mig_cost_val if mig_flag else 0)
                norm_cost = np.clip((step_cost - min_cost) / cost_range, 0.0, 1.0)

                cum_cost = prev_cost + alpha * norm_aoi + beta * norm_cost
                if cum_cost < min_cum_cost:
                    min_cum_cost = cum_cost
                    best_pre = prev_key
                    best_step_cost = step_cost
                    is_mig = mig_flag

            dp_table[n][key] = min_cum_cost
            pre_placement[n][key] = best_pre
            metrics_record[n][key] = (aoi, best_step_cost, is_mig)

    # ==========================================
    # 3. 动态规划反向回溯 (Backward Pass)
    # ==========================================
    opt_last_key = None
    min_final_val = float('inf')
    for key, val in dp_table[N - 1].items():
        if val < min_final_val:
            min_final_val = val
            opt_last_key = key

    if opt_last_key is None:
        logger.error("  >> DP-ODA 求解失败：全局无合法路径！")
        return []

    opt_path = []
    curr_key = opt_last_key

    for n in range(N - 1, -1, -1):
        aoi, cost, mig = metrics_record[n][curr_key]
        opt_path.append({
            'req': request_stream[n].id,
            'node_ids': list(curr_key),   # list of node IDs, one per task
            'aoi': aoi,
            'cost': cost,
            'migrated': mig,
        })
        curr_key = pre_placement[n][curr_key]

    opt_path.reverse()
    return opt_path
