# constraint/bandwidth.py

def check_bandwidth_constraint(network, source_id, target_id, data_size, required_bandwidth):
    """
    校验源节点到目标节点的路径上，是否有足够的带宽
    公式 C4: B_{a,v} >= rate_j
    """
    # 获取两点之间的路由路径
    path = network.get_path(source_id, target_id, data_size)
    if not path:
        return False

    # 找出路径上的瓶颈带宽（最小带宽）
    bottleneck_bw = network.get_path_bandwidth(path)

    return bottleneck_bw >= required_bandwidth
