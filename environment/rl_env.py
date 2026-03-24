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
    边缘网络数字孪生部署的 RL 环境 (符合 Gymnasium 标准API)
    """

    def __init__(self, network, users, task_chains, total_reqs=50, seed=42):
        super(DTEngineEnv, self).__init__()

        self.network = network
        self.users = users
        self.task_chains = task_chains
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
        # U_t (One-Hot) + S_t (One-Hot) + H_{t-1} (One-Hot) + M_t (归一化内存) + Delta_{t-1} (归一化AoI)
        self.obs_dim = self.num_users + self.num_sensors + self.num_nodes + self.num_nodes + 1

        # 所有的状态值都被严格限制在 0.0 到 1.0 之间 (极其利于神经网络收敛)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

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
        prev_info = self.simulator.dt_placements.get(req.task_chain.id)
        if prev_info is not None:
            # 找到对应的节点索引置为 1 (如果从未部署过，则全为 0)
            prev_node_id = prev_info['node_id']
            node_idx = [n.id for n in self.edge_nodes].index(prev_node_id)
            h_vec[node_idx] = 1.0

        # 4. M_t: 各节点可用内存归一化 (当前内存 / 最大内存)
        # 假设 node.memory 存的是容量上限，available_memory 是当前值
        m_vec = np.zeros(self.num_nodes, dtype=np.float32)
        for i, node in enumerate(self.edge_nodes):
            # 防止除以 0，加入 clip 保障
            ratio = node.available_memory / max(node.memory, 1e-5)
            m_vec[i] = np.clip(ratio, 0.0, 1.0)

        # 5. Delta_{t-1}: 上一次的 AoI 归一化
        last_aoi = self.simulator.dt_history_aoi.get(req.task_chain.id, 0.0)
        # 设定一个经验最大 AoI 用于缩放 (比如 10 秒)，超过 10 秒的都视为 1.0
        MAX_AOI_SCALE = 10.0
        aoi_norm = np.clip(last_aoi / MAX_AOI_SCALE, 0.0, 1.0)
        aoi_vec = np.array([aoi_norm], dtype=np.float32)

        # 将所有特征拼接成一个扁平的一维数组
        obs = np.concatenate([u_vec, s_vec, h_vec, m_vec, aoi_vec])
        return obs

    def reset(self, seed=None, options=None):
        """
        环境重置：每跑完一轮 (Episode) 都会调用。
        """
        super().reset(seed=seed)

        # 重置物理机内存 (假设 node.memory 是我们设定的最大容量属性)
        for node in self.edge_nodes:
            node.available_memory = getattr(node, 'memory', 10.0)

        self.simulator = Simulator(self.network, dt_ttl=10.0)

        # 2. 生成新一轮的泊松请求流
        self.request_stream = generate_poisson_requests(
            self.users, self.task_chains, arrival_rate=0.5, total_requests=self.total_reqs, seed=seed
        )

        self.current_step = 0
        self.current_req = self.request_stream[self.current_step]

        # 3. 返回第一个状态 (Observation) 和空的 Info 字典
        obs = self._get_obs(self.current_req)
        return obs, {}

    def step(self, action):
        """
        步进函数：RL 智能体给出一个 action，环境执行并返回 reward 和下一个状态
        """
        req = self.current_req
        # 1. 将网络输出的整数 (0, 1, 2) 映射回真实的物理节点对象
        target_node = self.edge_nodes[action]

        # 先执行垃圾回收推演时间
        self.simulator.cleanup_expired_dts(req.time)

        # 2. 在物理引擎中真实执行！
        try:
            real_sense, real_queue, real_comp, real_res, real_mig = self.simulator.commit_step(
                target_node, req.task_chain, req.user, req.time
            )

            # 计算真实指标
            aoi = estimate_aoi(real_sense, real_queue, real_comp, real_res)
            cost = compute_cost(target_node, real_comp, req.task_chain, real_mig)

            # RL 是追求最大化 Reward，所以我们取负数
            alpha, beta = 1.0, 1.0
            reward = - (alpha * cost + beta * aoi)

        except RuntimeError as e:
            # 【约束惩罚】：如果智能体选了一个内存不够的节点，物理机崩溃
            # 给予极大的负惩罚，并可以选择提前结束这一回合
            reward = -100.0
            aoi, cost = float('inf'), float('inf')

        # 3. 推进时间，获取下一个请求
        self.current_step += 1
        # 检查这一个 Episode (回合) 是否跑完了
        terminated = self.current_step >= self.total_reqs
        truncated = False  # 用于超时截断，这里不用

        if not terminated:
            self.current_req = self.request_stream[self.current_step]
            next_obs = self._get_obs(self.current_req)
        else:
            # 如果结束了，返回当前观测即可
            next_obs = self._get_obs(req)

        # 用于记录画图的详细信息
        info = {'req_id': req.id, 'aoi': aoi, 'cost': cost, 'migrated': bool(real_mig)}
        return next_obs, reward, terminated, truncated, info
