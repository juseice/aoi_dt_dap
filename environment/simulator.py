"""
Evaluate
"""

import random
from latency.transmission import compute_transmission_delay
from latency.sensing import compute_sensing_delay
from latency.computation import compute_chain_delay
from latency.queue import compute_queue_delay


class Simulator:
    def __init__(self, network, dt_ttl=25.0):
        self.network = network
        self.dt_ttl = dt_ttl
        self.node_available_time = {node.id: 0.0 for node in network.get_edge_nodes()}
        # 格式: { task_chain.id: {'node_id': node_id, 'memory': mem_req} }
        self.dt_placements = {}
        # 格式: { task_chain.id: last_request_time }
        self.dt_last_access = {}
        self.dt_history_aoi = {}

    # ==========================================
    # 对外提供状态查询接口 (解耦的核心)
    # ==========================================
    def get_dt_placement(self, task_chain_id):
        """查询 DT 当前部署在哪个节点 ID 上，若未部署返回 None"""
        info = self.dt_placements.get(task_chain_id)
        return info['node_id'] if info else None

    def get_dt_last_aoi(self, task_chain_id):
        """查询 DT 上一次访问的真实 AoI，若无记录返回 0.0"""
        return self.dt_history_aoi.get(task_chain_id, 0.0)

    def get_node_available_memory(self, node_id):
        """获取特定节点的剩余内存"""
        node = self.network.get_node(node_id)
        return node.available_memory if node else 0.0

    def cleanup_expired_dts(self, current_time):
        """
        清理长时间未被访问的 DT 实例，回收物理机内存
        返回被清理的实例信息，方便日志打印
        """
        expired_dts = []
        for dt_id, last_time in self.dt_last_access.items():
            if current_time - last_time > self.dt_ttl:
                expired_dts.append(dt_id)

        evicted_info = []
        for dt_id in expired_dts:
            info = self.dt_placements[dt_id]
            node_id = info['node_id']
            mem_req = info['memory']

            # 1. 归还物理机内存
            node = self.network.get_node(node_id)
            node.available_memory += mem_req

            # 2. 从注册表中抹除记录
            del self.dt_placements[dt_id]
            del self.dt_last_access[dt_id]

            evicted_info.append((dt_id, node_id, mem_req))

        return evicted_info

    def evaluate_step(self, node, task_chain, user, request_time):
        # current_time = self.current_time
        sensor_id = task_chain.sensor_id
        sensor = self.network.get_node(sensor_id)
        raw_data_size = sensor.data_size

        sense_delay = compute_sensing_delay(
            self.network,
            sensor_id,
            node.id,
            raw_data_size
        )
        data_arrival_time = request_time + sense_delay

        start_time = max(data_arrival_time, self.node_available_time[node.id])
        queue_delay = start_time - data_arrival_time
        compute_time = compute_chain_delay(task_chain, node)

        # response delay
        result_size = 1.0
        path = self.network.get_path(node.id, user.id, result_size)
        bw_down = self.network.get_path_bandwidth(path) if path else 0
        response_delay = compute_transmission_delay(result_size, bw_down)

        # 计算迁移 flag
        prev_info = self.dt_placements.get(task_chain.id)
        prev_node_id = prev_info['node_id'] if prev_info else None

        is_migrated = 1 if (prev_node_id != node.id) else 0

        return sense_delay, queue_delay, compute_time, response_delay, is_migrated

    def commit_step(self, node, task_chain, user, request_time):
        """真正落子，更新状态字典和 prev_node"""
        # 感知时间（上行时间）
        sensor_id = task_chain.sensor_id
        sensor = self.network.get_node(sensor_id)
        path_up = self.network.get_path(sensor_id, node.id, sensor.data_size)
        bw_up = self.network.get_path_bandwidth(path_up) if path_up else 0
        sense_delay = compute_transmission_delay(sensor.data_size, bw_up) if bw_up > 0 else float('inf')

        # 排队时间+计算时间
        data_arrival_time = request_time + sense_delay
        start_time = max(data_arrival_time, self.node_available_time[node.id])
        queue_delay = start_time - data_arrival_time
        compute_time = compute_chain_delay(task_chain, node)
        finish_time = start_time + compute_time

        # 响应时间（下行时间）
        result_size = 1.0
        path_down = self.network.get_path(node.id, user.id, result_size)
        bw_down = self.network.get_path_bandwidth(path_down) if path_down else 0
        response_delay = compute_transmission_delay(result_size, bw_down) if bw_down > 0 else float('inf')

        prev_info = self.dt_placements.get(task_chain.id)
        prev_node_id = prev_info['node_id'] if prev_info else None
        is_migrated = 1 if (prev_node_id != node.id) else 0

        total_mem_req = sum(task.memory_requirement for task in task_chain.tasks)
        if is_migrated:
            # 释放上一运行内存
            if prev_node_id is not None:
                prev_node = self.network.get_node(prev_node_id)
                prev_node.available_memory += prev_info['memory']

            # 内存不足报错
            if node.available_memory < total_mem_req:
                raise RuntimeError(f"物理环境崩溃：节点 {node.id} 内存不足，无法执行部署！")
            node.available_memory -= total_mem_req

            self.dt_placements[task_chain.id] = {'node_id': node.id, 'memory': total_mem_req}

        # 改变系统状态
        self.dt_last_access[task_chain.id] = request_time
        self.node_available_time[node.id] = finish_time
        real_aoi = sense_delay + queue_delay + compute_time + response_delay
        self.dt_history_aoi[task_chain.id] = real_aoi

        return sense_delay, queue_delay, compute_time, response_delay, is_migrated
