# environment/rl_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.logger import logger
from latency import estimate_aoi
from objective import compute_cost
from environment.event import generate_poisson_requests
from environment.simulator import Simulator


class DTEngineEnv(gym.Env):
    """
    边缘网络数字孪生部署的 RL 环境
    """

    def __init__(self, network, users, task_chains, request_stream=None, total_reqs=50, seed=42,
                 alpha=0.5, beta=0.5):  # 【修改1】新增 alpha 和 beta 参数，默认各占一半
        super(DTEngineEnv, self).__init__()

        self.network = network
        self.users = users
        self.task_chains = task_chains
        self.seed = seed

        self.fixed_request_stream = request_stream
        if self.fixed_request_stream is not None:
            self.request_stream = self.fixed_request_stream
            self.total_reqs = len(self.request_stream)
        else:
            self.request_stream = None
            self.total_reqs = total_reqs
        self.seed = seed

        # 提取所有的边缘节点 (动作空间的备选项)
        self.edge_nodes = self.network.get_edge_nodes()
        self.num_nodes = len(self.edge_nodes)

        # 提取实体列表用于 One-Hot 编码索引
        self.user_ids = [u.id for u in self.users]
        self.sensor_ids = list(set([tc.sensor_id for tc in self.task_chains]))

        self.num_users = len(self.user_ids)
        self.num_sensors = len(self.sensor_ids)

        # ----------------------------------------
        # 1. 定义动作空间 (Action Space)
        # ----------------------------------------
        # 动作就是选择把当前的 DT 请求部署在哪个节点上。取值范围: 0 到 num_nodes - 1
        self.action_space = spaces.Discrete(self.num_nodes)

        # ----------------------------------------
        # 2. 定义状态空间 (Observation Space)
        # ----------------------------------------
        # 对应论文 4.1: U_t (用户位置), H_t-1 (上次部署位置), M_t (各节点内存), 等
        # 为了让神经网络好收敛，我们需要把所有信息展平成一维数组 (Box)
        # 假设我们提取的特征长度为：
        # 1 (用户ID) + 1 (传感器ID) + 1 (该DT上次部署节点) + 1 (上次的AoI) + num_nodes (各节点可用内存)
        # 2. 状态空间维度精准计算
        # U_t (One-Hot) + S_t (One-Hot) + H_{t-1} (One-Hot) + M_t (归一化内存) + Delta_{t-1} (归一化AoI) + s_to_n + n_to_u
        self.obs_dim = self.num_users + self.num_sensors + self.num_nodes + self.num_nodes + 1 + self.num_nodes + self.num_nodes

        # 所有的状态值都被严格限制在 0.0 到 1.0 之间 (极其利于神经网络收敛)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        # RL Reward Hyperparameters (对应公式 7)
        self.alpha = alpha
        self.beta = beta

        self.Z = 20.0
        self.R_finish = 10.0
        self.R_penalty = -5.0

        # 预先提取网络中的全局最强属性，用于计算公式(8)的理论极限
        self.max_compute = max([n.compute_power for n in self.edge_nodes])
        self.min_cost = min([n.cost for n in self.edge_nodes])

        # 获取全网最大带宽
        all_edges = []
        for u, v, data in self.network.graph.edges(data=True):
            all_edges.append(data['edge'].bandwidth)
        self.max_bandwidth = max(all_edges) if all_edges else 10.0

    def _compute_lower_bound(self, req):
        """
        计算公式 (8): LB_j (理论最低时延与成本)
        假设没有排队、没有迁移开销，且全都走最大带宽、最强算力节点。
        """
        task_chain = req.task_chain
        total_workload = sum(task.workload for task in task_chain.tasks)

        # 1. 理论最小计算时间 = 总计算量 / 全网最强节点算力
        min_comp_time = total_workload / self.max_compute

        # 2. 理论最低运行成本 = 全网最便宜节点的单价 * 最小计算时间
        min_run_cost = self.min_cost * min_comp_time

        # 3. 理论最小传输时间 (简化估算) = (上行数据量 + 下行结果) / 全网最大带宽
        sensor_data_size = self.network.get_node(task_chain.sensor_id).data_size
        result_size = 1.0
        min_trans_time = (sensor_data_size + result_size) / self.max_bandwidth

        # LB_j = alpha * Cost_min + beta * AoI_min (假设 0 排队时延)
        lb_j = self.alpha * min_run_cost + self.beta * (min_comp_time + min_trans_time)
        return lb_j

    def _get_obs(self, req):
        """
        将物理世界的字典和对象，翻译成神经网络能看懂的 Float 数组
        """
        # 1. U_t: 当前请求的用户特征
        u_vec = np.zeros(self.num_users, dtype=np.float32)
        u_vec[self.user_ids.index(req.user.id)] = 1.0

        # 2. S_t: 传感器 ID 独热编码 (One-Hot)
        s_vec = np.zeros(self.num_sensors, dtype=np.float32)
        s_vec[self.sensor_ids.index(req.task_chain.sensor_id)] = 1.0

        # 3. H_t-1: 该 DT 实例上一次部署在哪里？
        h_vec = np.zeros(self.num_nodes, dtype=np.float32)
        prev_node_id = self.simulator.get_dt_placement(req.task_chain.id)
        if prev_node_id is not None:
            node_idx = [n.id for n in self.edge_nodes].index(prev_node_id)
            h_vec[node_idx] = 1.0

        # 4. M_t: 各节点可用内存归一化 (当前内存 / 最大内存)
        # node.memory 存的是容量上限，available_memory 是当前值
        m_vec = np.zeros(self.num_nodes, dtype=np.float32)
        for i, node in enumerate(self.edge_nodes):
            avail_mem = self.simulator.get_node_available_memory(node.id)
            ratio = avail_mem / max(node.memory, 1e-5)
            m_vec[i] = np.clip(ratio, 0.0, 1.0)

        # 5. Delta_{t-1}: 上一次的 AoI 归一化
        last_aoi = self.simulator.get_dt_last_aoi(req.task_chain.id)
        MAX_AOI_SCALE = 10.0
        aoi_vec = np.array([np.clip(last_aoi / MAX_AOI_SCALE, 0.0, 1.0)], dtype=np.float32)

        # 6. 拓扑距离感知
        s_to_n_dist = np.zeros(self.num_nodes, dtype=np.float32)
        n_to_u_dist = np.zeros(self.num_nodes, dtype=np.float32)
        sensor_id = req.task_chain.sensor_id
        user_id = req.user.id
        for i, node in enumerate(self.edge_nodes):
            # 找传感器到节点的路径长度 (这里简单用跳数 len(path) 估算，如果不可达给个大数值)
            path_up = self.network.get_path(sensor_id, node.id, 1.0)
            s_to_n_dist[i] = len(path_up) if path_up else 10.0

            # 找节点到用户的路径长度
            path_down = self.network.get_path(node.id, user_id, 1.0)
            n_to_u_dist[i] = len(path_down) if path_down else 10.0

        s_to_n_dist = np.clip(s_to_n_dist / 10.0, 0.0, 1.0)
        n_to_u_dist = np.clip(n_to_u_dist / 10.0, 0.0, 1.0)

        # 将所有特征拼接成一个扁平的一维数组
        obs = np.concatenate([u_vec, s_vec, h_vec, m_vec, aoi_vec, s_to_n_dist, n_to_u_dist])
        return obs

    def reset(self, seed=None, options=None):
        """
        环境重置：每跑完一轮 (Episode) 都会调用。
        """
        super().reset(seed=seed)

        # 重置物理机内存
        for node in self.edge_nodes:
            node.available_memory = getattr(node, 'memory', 10.0)

        self.simulator = Simulator(self.network, dt_ttl=10.0)
        if self.fixed_request_stream is not None:
            self.request_stream = self.fixed_request_stream
        else:
            self.request_stream = generate_poisson_requests(
                self.users, self.task_chains, arrival_rate=0.5, total_requests=self.total_reqs, seed=seed
            )

        self.current_step = 0
        self.current_req = self.request_stream[self.current_step]

        return self._get_obs(self.current_req), {}

    def step(self, action):
        """
        步进函数：RL 智能体给出一个 action，环境执行并返回 reward 和下一个状态
        """
        req = self.current_req
        # 1. 将网络输出的整数 (0, 1, 2) 映射回真实的物理节点对象
        target_node = self.edge_nodes[action]

        # 先执行垃圾回收推演时间
        self.simulator.cleanup_expired_dts(req.trigger_time)

        # 2. 在物理引擎中真实执行！
        try:
            real_sense, real_queue, real_comp, real_res, real_mig = self.simulator.commit_step(
                target_node, req.task_chain, req.user, req.trigger_time
            )

            # 计算真实指标
            aoi = estimate_aoi(real_sense, real_queue, real_comp, real_res)
            cost = compute_cost(target_node, real_comp, req.task_chain, real_mig)

            # Reward
            lb_j = self._compute_lower_bound(req)
            actual_obj = max(self.alpha * cost + self.beta * aoi, 1e-5)

            # rt = Z * LB / (alpha*C + beta*D) + R_finish
            reward = (self.Z * lb_j) / actual_obj + self.R_finish
            success = True

        except RuntimeError as e:
            # 【约束惩罚】：如果智能体选了一个内存不够的节点，物理机崩溃
            # 给予极大的负惩罚，并可以选择提前结束这一回合
            # reward = self.R_penalty
            # aoi, cost = float('inf'), float('inf')
            # real_sense, real_queue, real_comp, real_res, real_mig = \
            #     float('inf'), float('inf'), float('inf'), float('inf'), float('inf')
            # # 保证状态流转
            # fallback_node = None
            # for n in self.edge_nodes:
            #     if n.available_memory >= sum(t.memory_requirement for t in req.task_chain.tasks):
            #         fallback_node = n
            #         break
            # if fallback_node:
            #     # 推进时间轴
            #     self.simulator.commit_step(fallback_node, req.task_chain, req.user, req.time)
            # else:
            #     # terminated = True
            #     pass
            reward = self.R_penalty
            aoi, cost = float('inf'), float('inf')
            real_mig = False
            success = False
            # 此时我们不再寻找 fallback_node，物理时间会在下一个请求到来时自然推进

        # 3. 推进时间，获取下一个请求
        self.current_step += 1
        # 检查这一个 Episode (回合) 是否跑完了
        terminated = self.current_step >= self.total_reqs
        truncated = False  # 用于超时截断，这里不用

        if not terminated:
            self.current_req = self.request_stream[self.current_step]
            next_obs = self._get_obs(self.current_req)
        else:
            # 保持张量形状，返回当前观测即可
            next_obs = self._get_obs(req)

        # 用于记录画图的详细信息
        info = {
            'req_id': req.id,
            'aoi': aoi,
            'cost': cost,
            'migrated': bool(real_mig),
            'success': success  # 用于记录这步是否被 Drop
        }
        return next_obs, reward, terminated, truncated, info
