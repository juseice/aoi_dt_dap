# optimization/dp_solver.py
from utils.logger import logger
from latency.sensing import compute_sensing_delay
from latency.computation import compute_chain_delay
from latency.transmission import compute_transmission_delay


def solve_dp_offline(network, request_stream, alpha, beta):
    """
    基于动态规划的最优部署算法 (DP-ODA) - 对应论文 P2 简化模型
    作为 Offline Oracle，假设资源充足且无排队时延，求取全局理论最优边界。
    """
    N = len(request_stream)
    if N == 0:
        return []

    # 获取候选边缘节点集合 (V)
    nodes = network.get_edge_nodes()

    # 初始化 DP 表格和前驱记录表
    # dp_table[n][node_id] 对应论文中的 f(n, v_i)
    dp_table = [{} for _ in range(N)]
    # pre_node[n][node_id] 对应论文中的 Pre(n, v_i)
    pre_node = [{} for _ in range(N)]

    # 额外记录：用于最终画图的单步指标 (aoi, cost, is_migrated)
    metrics_record = [{} for _ in range(N)]

    # ==========================================
    # 辅助函数：计算无排队物理延迟和基础运行成本
    # ==========================================
    def get_base_metrics(req, node):
        sensor_id = req.task_chain.sensor_id
        sensor = network.get_node(sensor_id)

        # 1. 寻路与带宽校验 (严格遵循物理连通性)
        path_up = network.get_path(sensor_id, node.id, sensor.data_size)
        path_down = network.get_path(node.id, req.user.id, 1.0)

        if not path_up or not path_down:
            return float('inf'), float('inf')

        bw_up = network.get_path_bandwidth(path_up)
        if bw_up < req.task_chain.required_bandwidth:
            return float('inf'), float('inf')

        # 2. 计算无排队时延 (简化模型 P2 的特性)
        sense_delay = compute_sensing_delay(network, sensor_id, node.id, sensor.data_size)
        comp_time = compute_chain_delay(req.task_chain, node)

        bw_down = network.get_path_bandwidth(path_down)
        res_delay = compute_transmission_delay(1.0, bw_down) if bw_down > 0 else float('inf')

        aoi = sense_delay + comp_time + res_delay

        # 3. 计算运行成本 (修复后的量纲：CUT_i * D_comp)
        run_cost = node.cost * comp_time

        return aoi, run_cost

    logger.info("  >> 正在执行 DP-ODA 离线全局优化...")

    # ==========================================
    # 动态规划正向推导 (Forward Pass)
    # ==========================================

    # 1. 初始化首个决策步 (n = 0)
    req_0 = request_stream[0]
    mig_cost_base = sum(task.deployment_cost for task in req_0.task_chain.tasks)

    for v_i in nodes:
        aoi, run_cost = get_base_metrics(req_0, v_i)
        if aoi == float('inf'):
            dp_table[0][v_i.id] = float('inf')
            continue

        # 初始步视为一次必然发生的部署开销
        step_cost = run_cost + mig_cost_base

        # f(1, v_i) = \phi(1, v_i)
        dp_table[0][v_i.id] = alpha * step_cost + beta * aoi
        pre_node[0][v_i.id] = None
        metrics_record[0][v_i.id] = (aoi, step_cost, True)

    # 2. 状态转移方程推导 (n = 1 to N-1)
    for n in range(1, N):
        req_n = request_stream[n]
        mig_cost_val = sum(task.deployment_cost for task in req_n.task_chain.tasks)

        for v_i in nodes:
            aoi, run_cost = get_base_metrics(req_n, v_i)
            if aoi == float('inf'):
                dp_table[n][v_i.id] = float('inf')
                continue

            min_cum_cost = float('inf')
            best_pre = None
            best_step_cost = None
            is_mig = False

            # 遍历上一状态的节点 v_k
            for v_k in nodes:
                prev_cost = dp_table[n - 1].get(v_k.id, float('inf'))
                if prev_cost == float('inf'):
                    continue

                # 计算是否触发迁移 I(v_i != v_k)
                mig_flag = (v_i.id != v_k.id)
                actual_mig_cost = mig_cost_val if mig_flag else 0
                step_cost = run_cost + actual_mig_cost

                # 状态转移：f(n-1, v_k) + \phi(n, v_i) + 迁移
                cum_cost = prev_cost + alpha * step_cost + beta * aoi

                if cum_cost < min_cum_cost:
                    min_cum_cost = cum_cost
                    best_pre = v_k.id
                    best_step_cost = step_cost
                    is_mig = mig_flag

            dp_table[n][v_i.id] = min_cum_cost
            pre_node[n][v_i.id] = best_pre
            metrics_record[n][v_i.id] = (aoi, best_step_cost, is_mig)

    # ==========================================
    # 动态规划反向回溯 (Backward Pass)
    # ==========================================
    # 找出 h_N^*
    opt_last_node = None
    min_final_val = float('inf')
    for node_id, val in dp_table[N - 1].items():
        if val < min_final_val:
            min_final_val = val
            opt_last_node = node_id

    if opt_last_node is None:
        logger.error("  >> DP-ODA 求解失败：全局无合法路径！")
        return []

    # 沿着 Pre 表回溯最优序列
    opt_path = []
    curr_node = opt_last_node

    for n in range(N - 1, -1, -1):
        aoi, cost, mig = metrics_record[n][curr_node]
        opt_path.append({
            'req': request_stream[n].id,
            'node_id': curr_node,
            'aoi': aoi,
            'cost': cost,
            'migrated': mig
        })
        curr_node = pre_node[n][curr_node]

    # 回溯得到的是从后往前的序列，需要反转
    opt_path.reverse()

    return opt_path
