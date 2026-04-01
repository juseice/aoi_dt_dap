# algorithms/random_agent.py
import random
from constraints.resource import check_memory_constraint
from constraints.bandwidth import check_bandwidth_constraint
from utils.logger import logger

class RandomAgent:
    """
    随机分配策略基准 (Random Baseline)
    """
    def __init__(self, env):
        """
        初始化 Agent
        :param env: 你的 DTEngineEnv 实例
        """
        self.env = env

    def predict(self, obs=None, deterministic=False):
        """
        统一预测接口，与 Stable-Baselines3 的 PPO 接口保持一致。
        :param obs: 当前的环境状态 (虽然随机算法不需要看状态，但为了接口统一保留)
        :param deterministic: 是否确定性动作 (兼容参数)
        :return: action (int), states (None)
        """
        # 从环境对象中直接提取当前的物理请求和仿真器
        req = self.env.current_req
        simulator = self.env.simulator
        network = self.env.network

        feasible_node_indices = []

        # ==========================================
        # 1. 过滤阶段：找出所有合法的候选节点索引
        # ==========================================
        for idx, node in enumerate(self.env.edge_nodes):
            # C2 资源约束校验
            if not check_memory_constraint(node, req.task_chain, simulator):
                continue

            # C4 带宽约束校验 (传感器 -> 边缘节点)
            sensor_id = req.task_chain.sensor_id
            sensor = network.get_node(sensor_id)
            if not check_bandwidth_constraint(network, sensor_id, node.id,
                                              sensor.data_size, req.task_chain.required_bandwidth):
                continue

            # 连通性约束校验 (边缘节点 -> 用户)
            if not network.get_path(node.id, req.user.id, 1.0):
                continue

            # 通过所有校验，加入可用候选池
            feasible_node_indices.append(idx)

        # ==========================================
        # 2. 决策阶段：无脑随机选
        # ==========================================
        if not feasible_node_indices:
            # 如果资源枯竭，没有合法节点。
            # 强化学习环境中，为了触发合法的 step 报错并给予惩罚，
            # 我们可以随机返回一个非法动作，或者返回一个特定的越界值。
            # 这里简单处理为在所有节点中随机瞎猜一个，交由 env 惩罚
            action = random.randint(0, self.env.num_nodes - 1)
        else:
            # 从合法的节点索引中随机挑一个
            action = random.choice(feasible_node_indices)

        # 返回动作索引 (对应 env.action_space 的离散整数) 和 空状态 (兼容 RNN)
        return action, None
