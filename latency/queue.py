def compute_queue_delay(node, current_time, compute_time):
    start_time = max(current_time, node.available_time)

    queue_delay = start_time - current_time

    finish_time = start_time + compute_time

    # 更新节点状态
    node.available_time = finish_time

    return queue_delay, start_time, finish_time
