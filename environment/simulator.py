"""
Evaluate
"""

import random
from latency.transmission import compute_transmission_delay
from latency.sensing import compute_sensing_delay


class Simulator:
    def __init__(self, network, dt_ttl=10.0):
        self.network = network
        self.dt_ttl = dt_ttl
        self.node_available_time = {node.id: 0.0 for node in network.get_edge_nodes()}
        # format: { task_chain.id: {'node_ids': [node_id, ...], 'memories': [mem, ...]} }
        # one entry per task in the chain
        self.dt_placements = {}
        # format: { task_chain.id: last_request_time }
        self.dt_last_access = {}
        self.dt_history_aoi = {}

    # ==========================================
    # 对外提供状态查询接口 (解耦的核心)
    # ==========================================
    def get_dt_placement(self, task_chain_id):
        """返回 DT 各任务的节点 ID 列表，若未部署返回 None"""
        info = self.dt_placements.get(task_chain_id)
        return info['node_ids'] if info else None

    def get_dt_last_aoi(self, task_chain_id):
        """查询 DT 上一次访问的真实 AoI，若无记录返回 0.0"""
        return self.dt_history_aoi.get(task_chain_id, 0.0)

    def get_node_available_memory(self, node_id):
        """获取特定节点的剩余内存"""
        node = self.network.get_node(node_id)
        return node.available_memory if node else 0.0

    def cleanup_expired_dts(self, current_time):
        """
        清理长时间未被访问的 DT 实例，回收各节点内存
        """
        expired_dts = []
        for dt_id, last_time in self.dt_last_access.items():
            if current_time - last_time > self.dt_ttl:
                expired_dts.append(dt_id)

        evicted_info = []
        for dt_id in expired_dts:
            info = self.dt_placements[dt_id]
            for node_id, mem_req in zip(info['node_ids'], info['memories']):
                node = self.network.get_node(node_id)
                node.available_memory += mem_req

            del self.dt_placements[dt_id]
            del self.dt_last_access[dt_id]

            evicted_info.append((dt_id, info['node_ids'], info['memories']))

        return evicted_info

    # ==========================================
    # 内部共享：计算链式部署的各阶段时延
    # ==========================================
    def _compute_chain_latency(self, node_list, task_chain, user, request_time):
        """
        对分布在 node_list 上的 task_chain 进行时延推演（不修改状态）。

        node_list: list[EdgeNode], len == len(task_chain.tasks)
        返回:
            sense_delay       — 感知/上行时延 (sensor → node_list[0])
            queue_delays      — 每个任务节点的排队时延列表
            compute_times     — 每个任务的计算时延列表
            inter_delays      — 相邻任务节点间的中间传输时延列表 (len = n_tasks - 1)
            response_delay    — 结果下行时延 (node_list[-1] → user)
        """
        tasks = task_chain.tasks
        sensor_id = task_chain.sensor_id
        sensor = self.network.get_node(sensor_id)
        raw_data_size = sensor.data_size

        # 感知时延：sensor → 第一个任务节点
        sense_delay = compute_sensing_delay(
            self.network, sensor_id, node_list[0].id, raw_data_size
        )
        current_time = request_time + sense_delay

        queue_delays = []
        compute_times = []
        inter_delays = []

        for i, (task, node) in enumerate(zip(tasks, node_list)):
            # 排队时延
            start_time = max(current_time, self.node_available_time[node.id])
            queue_delays.append(start_time - current_time)

            # 计算时延
            comp = task.workload / node.compute_power
            compute_times.append(comp)
            finish_time = start_time + comp

            # 中间传输时延（任务 i → 任务 i+1）
            if i < len(tasks) - 1:
                next_node = node_list[i + 1]
                if node.id != next_node.id:
                    out_size = task.output_data_size
                    path = self.network.get_path(node.id, next_node.id, out_size)
                    bw = self.network.get_path_bandwidth(path) if path else 0
                    inter = compute_transmission_delay(out_size, bw) if bw > 0 else float('inf')
                else:
                    inter = 0.0
                inter_delays.append(inter)
                current_time = finish_time + inter
            else:
                current_time = finish_time

        # 响应时延：最后一个任务节点 → user
        result_size = 1.0
        path_down = self.network.get_path(node_list[-1].id, user.id, result_size)
        bw_down = self.network.get_path_bandwidth(path_down) if path_down else 0
        response_delay = (
            compute_transmission_delay(result_size, bw_down) if bw_down > 0 else float('inf')
        )

        return sense_delay, queue_delays, compute_times, inter_delays, response_delay

    # ==========================================
    # 核心接口
    # ==========================================
    def evaluate_step(self, node_list, task_chain, user, request_time):
        """
        试探性评估：预测指标，不修改任何状态。

        node_list: list[EdgeNode]，长度等于 task_chain.tasks 的数量。
        返回 (sense_delay, queue_delay, compute_time, response_delay, is_migrated)
        其中 compute_time 包含所有中间传输时延，以便 estimate_aoi 仍可直接求和。
        """
        sense_delay, queue_delays, compute_times, inter_delays, response_delay = \
            self._compute_chain_latency(node_list, task_chain, user, request_time)

        prev_info = self.dt_placements.get(task_chain.id)
        prev_node_ids = prev_info['node_ids'] if prev_info else None
        new_node_ids = [n.id for n in node_list]
        is_migrated = 1 if (prev_node_ids != new_node_ids) else 0

        return (
            sense_delay,
            sum(queue_delays),
            sum(compute_times) + sum(inter_delays),
            response_delay,
            is_migrated,
        )

    def commit_step(self, node_list, task_chain, user, request_time):
        """
        真正落子：执行部署决策，更新状态字典。

        node_list: list[EdgeNode]，长度等于 task_chain.tasks 的数量。
        返回 (sense_delay, queue_delay, compute_time, response_delay, is_migrated)
        """
        tasks = task_chain.tasks

        sense_delay, queue_delays, compute_times, inter_delays, response_delay = \
            self._compute_chain_latency(node_list, task_chain, user, request_time)

        # 更新每个节点的可用时间（顺序执行流水线）
        sensor_id = task_chain.sensor_id
        sensor = self.network.get_node(sensor_id)
        current_time = request_time + sense_delay

        for i, (task, node) in enumerate(zip(tasks, node_list)):
            start_time = max(current_time, self.node_available_time[node.id])
            finish_time = start_time + compute_times[i]
            self.node_available_time[node.id] = finish_time
            current_time = finish_time + (inter_delays[i] if i < len(inter_delays) else 0)

        # 迁移检测与内存管理
        prev_info = self.dt_placements.get(task_chain.id)
        prev_node_ids = prev_info['node_ids'] if prev_info else None
        new_node_ids = [n.id for n in node_list]
        is_migrated = 1 if (prev_node_ids != new_node_ids) else 0

        if is_migrated:
            # 释放旧节点内存
            if prev_info is not None:
                for prev_node_id, mem in zip(prev_info['node_ids'], prev_info['memories']):
                    prev_node = self.network.get_node(prev_node_id)
                    prev_node.available_memory += mem

            # 在新节点上分配内存
            new_memories = []
            for task, node in zip(tasks, node_list):
                if node.available_memory < task.memory_requirement:
                    raise RuntimeError(
                        f"物理环境崩溃：节点 {node.id} 内存不足，无法部署任务 {task.id}！"
                    )
                node.available_memory -= task.memory_requirement
                new_memories.append(task.memory_requirement)

            self.dt_placements[task_chain.id] = {
                'node_ids': new_node_ids,
                'memories': new_memories,
            }

        self.dt_last_access[task_chain.id] = request_time
        real_aoi = (
            sense_delay
            + sum(queue_delays)
            + sum(compute_times)
            + sum(inter_delays)
            + response_delay
        )
        self.dt_history_aoi[task_chain.id] = real_aoi

        return (
            sense_delay,
            sum(queue_delays),
            sum(compute_times) + sum(inter_delays),
            response_delay,
            is_migrated,
        )
