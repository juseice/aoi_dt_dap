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
        obs_dim = 4 + self.num_nodes

        # 定义状态向量的上下界 (通常归一化到 0~1 或用一个足够大的范围)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self, req):
        """
        【核心映射】：将物理世界的字典和对象，翻译成神经网络能看懂的 Float 数组
        """
        # 1. U_t: 当前请求的用户特征
        u_feat = req.user.id

        # 2. 当前请求的 DT 关联的传感器特征
        s_feat = req.task_chain.sensor_id

        # 3. H_t-1: 该 DT 实例上一次部署在哪里？
        prev_info = self.simulator.dt_placements.get(req.task_chain.id)
        h_prev = prev_info['node_id'] if prev_info else -1  # -1 表示从未部署过

        # 4. Delta_t-1: 该 DT 上一次的历史 AoI (这里我们简单用 Simulator 里记录的时间差代替，或者先传 0)
        # 真实工程中可以在 Simulator 维护一个 dt_history_aoi 字典
        last_aoi = 0.0

        # 5. M_t: 当前所有边缘节点的可用内存百分比或绝对值
        m_feat = [node.available_memory for node in self.edge_nodes]

        # 拼接成 Numpy 数组
        obs = np.array([u_feat, s_feat, h_prev, last_aoi] + m_feat, dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """
        环境重置：每跑完一轮 (Episode) 都会调用。
        类似于打游戏死掉后重新开局。
        """
        super().reset(seed=seed)

        # 1. 重置物理模拟器和网络节点内存等状态
        for node in self.edge_nodes:
            # TODO: 你可以根据初始配置还原节点内存，这里假设初始满血是 10G
            pass
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

            # 【设计 Reward】：论文里要最小化 (alpha*Cost + beta*AoI)
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
        info = {'req_id': req.id, 'aoi': aoi, 'cost': cost}

        return next_obs, reward, terminated, truncated, info
    