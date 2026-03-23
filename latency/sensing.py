from latency.transmission import compute_transmission_delay


def compute_sensing_delay(network, sensor_id, target_node_id, data_size):
    path = network.get_path(sensor_id, target_node_id, data_size)
    if not path:
        return float('inf')  # 网络不通，AoI 无限大

    bandwidth = network.get_path_bandwidth(path)

    trans_delay = compute_transmission_delay(data_size, bandwidth)

    t_acq = 1  # TODO: 数据产生时间，后面扩展

    return t_acq + trans_delay
