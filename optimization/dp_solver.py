# optimization/dp_solver.py
from utils.logger import logger
from latency.sensing import compute_sensing_delay
from latency.computation import compute_chain_delay
from latency.transmission import compute_transmission_delay
import numpy as np  # 用于 np.clip


def solve_dp_offline(network, request_stream, alpha, beta):
    """
    基于动态规划的最优部署算法 (DP-ODA) - 对应论文 P2 简化模型
    加入 Min-Max 归一化与预扫描缓存加速机制
    """
    N = len(request_stream)
    if N == 0:
        return []

    nodes = network.get_edge_nodes()

    # ==========================================
    # 辅助函数：计算无排队物理延迟和基础运行成本
    # ==========================================
    def get_base_metrics(req, node):
        sensor_id = req.task_chain.sensor_id
        sensor = network.get_node(sensor_id)

        path_up = network.get_path(sensor_id, node.id, sensor.data_size)
        path_down = network.get_path(node.id, req.user.id, 1.0)

        if not path_up or not path_down:
            return float('inf'), float('inf')

        bw_up = network.get_path_bandwidth(path_up)
        if bw_up < req.task_chain.required_bandwidth:
            return float('inf'), float('inf')

        sense_delay = compute_sensing_delay(network, sensor_id, node.id, sensor.data_size)
        comp_time = compute_chain_delay(req.task_chain, node)

        bw_down = network.get_path_bandwidth(path_down)
        res_delay = compute_transmission_delay(1.0, bw_down) if bw_down > 0 else float('inf')

        aoi = sense_delay + comp_time + res_delay
        run_cost = node.cost * comp_time

        return aoi, run_cost

    logger.info(f"  >> 正在执行预扫描 (Pre-computation Pass)... 计算全局上下界并缓存。")

    # ==========================================
    # 1. 预扫描与缓存 (Pre-computation & Cache)
    # ==========================================
    min_aoi, max_aoi = float('inf'), -float('inf')
    min_cost, max_cost = float('inf'), -float('inf')

    # 建立字典缓存：metrics_cache[n][node_id] = (raw_aoi, raw_run_cost)
    metrics_cache = [{} for _ in range(N)]

    for n in range(N):
        req_n = request_stream[n]
        # 提取迁移成本常量
        mig_cost_val = sum(task.deployment_cost for task in req_n.task_chain.tasks)

        for v_i in nodes:
            aoi, run_cost = get_base_metrics(req_n, v_i)
            metrics_cache[n][v_i.id] = (aoi, run_cost)

            if aoi != float('inf'):
                # 更新 AoI 全局极值
                min_aoi = min(min_aoi, aoi)
                max_aoi = max(max_aoi, aoi)

                # 更新 Cost 全局极值 (最坏情况是发生迁移，最好情况是不迁移)
                min_cost = min(min_cost, run_cost)
                max_cost = max(max_cost, run_cost + mig_cost_val)

    # 防止分母为 0
    aoi_range = max_aoi - min_aoi + 1e-8
    cost_range = max_cost - min_cost + 1e-8

    logger.info(f"     AoI 边界: [{min_aoi:.3f}, {max_aoi:.3f}], Cost 边界: [{min_cost:.3f}, {max_cost:.3f}]")
    logger.info("  >> 正在执行 DP-ODA 正向状态推导...")

    # ==========================================
    # 2. 动态规划正向推导 (Forward Pass)
    # ==========================================
    dp_table = [{} for _ in range(N)]
    pre_node = [{} for _ in range(N)]
    metrics_record = [{} for _ in range(N)]

    # 初始化首个决策步 (n = 0)
    req_0 = request_stream[0]
    mig_cost_base = sum(task.deployment_cost for task in req_0.task_chain.tasks)

    for v_i in nodes:
        aoi, run_cost = metrics_cache[0][v_i.id]  # 直接查表，不重复计算！
        if aoi == float('inf'):
            dp_table[0][v_i.id] = float('inf')
            continue

        step_cost = run_cost + mig_cost_base

        # 利用全局边界进行归一化
        norm_aoi = np.clip((aoi - min_aoi) / aoi_range, 0.0, 1.0)
        norm_cost = np.clip((step_cost - min_cost) / cost_range, 0.0, 1.0)

        # DP 表里存的是“归一化后的无量纲代价值”
        dp_table[0][v_i.id] = alpha * norm_aoi + beta * norm_cost
        pre_node[0][v_i.id] = None
        # record 表里存的必须是“真实的物理值”，方便画图！
        metrics_record[0][v_i.id] = (aoi, step_cost, True)

    # 状态转移方程推导 (n = 1 to N-1)
    for n in range(1, N):
        req_n = request_stream[n]
        mig_cost_val = sum(task.deployment_cost for task in req_n.task_chain.tasks)

        for v_i in nodes:
            aoi, run_cost = metrics_cache[n][v_i.id]  # 查表
            if aoi == float('inf'):
                dp_table[n][v_i.id] = float('inf')
                continue

            # 当前步骤的 AoI 是确定的，先归一化
            norm_aoi = np.clip((aoi - min_aoi) / aoi_range, 0.0, 1.0)

            min_cum_cost = float('inf')
            best_pre = None
            best_step_cost = None
            is_mig = False

            for v_k in nodes:
                prev_cost = dp_table[n - 1].get(v_k.id, float('inf'))
                if prev_cost == float('inf'):
                    continue

                mig_flag = (v_i.id != v_k.id)
                actual_mig_cost = mig_cost_val if mig_flag else 0
                step_cost = run_cost + actual_mig_cost

                # 对总步进成本进行归一化
                norm_cost = np.clip((step_cost - min_cost) / cost_range, 0.0, 1.0)

                # 状态转移方程使用的是归一化后的增量
                cum_cost = prev_cost + alpha * norm_aoi + beta * norm_cost

                if cum_cost < min_cum_cost:
                    min_cum_cost = cum_cost
                    best_pre = v_k.id
                    best_step_cost = step_cost
                    is_mig = mig_flag

            dp_table[n][v_i.id] = min_cum_cost
            pre_node[n][v_i.id] = best_pre
            metrics_record[n][v_i.id] = (aoi, best_step_cost, is_mig)

    # ==========================================
    # 3. 动态规划反向回溯 (Backward Pass)
    # ==========================================
    opt_last_node = None
    min_final_val = float('inf')
    for node_id, val in dp_table[N - 1].items():
        if val < min_final_val:
            min_final_val = val
            opt_last_node = node_id

    if opt_last_node is None:
        logger.error("  >> DP-ODA 求解失败：全局无合法路径！")
        return []

    opt_path = []
    curr_node = opt_last_node

    for n in range(N - 1, -1, -1):
        aoi, cost, mig = metrics_record[n][curr_node]
        opt_path.append({
            'req': request_stream[n].id,
            'node_id': curr_node,
            'aoi': aoi,  # 真实物理意义的数值
            'cost': cost,  # 也是真实数值
            'migrated': mig
        })
        curr_node = pre_node[n][curr_node]

    opt_path.reverse()
    return opt_path
