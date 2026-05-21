# algorithm/greedy_agent.py
import random
from constraints.resource import check_memory_constraint
from constraints.bandwidth import check_bandwidth_constraint
from latency import estimate_aoi
from objective import compute_cost
from utils.logger import logger


class GreedyAgent:
    """
    贪心分配策略基准 (Greedy Baseline)
    在当前决策步，通过遍历所有合法的候选节点，利用物理仿真器进行无副作用的预评估，
    选择使得当前瞬时 (alpha * Cost + beta * AoI) 最小的节点索引。
    """

    def __init__(self, env):
        """
        初始化 Agent
        :param env: DTEngineEnv 实例，用于提取环境参数和仿真器
        """
        self.env = env

    def predict(self, obs=None, deterministic=True):
        """
        统一预测接口
        :param obs: 当前的环境状态向量 (贪心算法通过解析底层环境直接获取信息，所以可以忽略 obs 张量)
        :param deterministic: 兼容 SB3 接口
        :return: action (int), state (None)
        """
        # 从环境对象中提取当前的物理请求、仿真器和超参数
        req = self.env.current_req
        simulator = self.env.simulator
        network = self.env.network
        alpha = self.env.alpha
        beta = self.env.beta

        best_score = float('inf')
        best_action_idx = None

        # ==========================================
        # 1. 遍历所有节点索引，进行剪枝与预评估
        # ==========================================
        tasks = req.task_chain.tasks
        sensor_id = req.task_chain.sensor_id
        sensor = network.get_node(sensor_id)

        for idx, node in enumerate(self.env.edge_nodes):
            # --- 阶段 A: 物理约束剪枝 (合法性过滤) ---
            # C2 资源约束：链上所有子任务均部署在同一节点，逐任务检查内存
            if not all(
                check_memory_constraint(node, task, req.task_chain.id, i, simulator)
                for i, task in enumerate(tasks)
            ):
                continue

            # C4 带宽约束 (传感器 -> 边缘节点)
            if not check_bandwidth_constraint(network, sensor_id, node.id,
                                              sensor.data_size, req.task_chain.required_bandwidth):
                continue

            # 连通性约束 (边缘节点 -> 用户)
            if not network.get_path(node.id, req.user.id, 1.0):
                continue

            # --- 阶段 B: 试探性评估 (Look-ahead Evaluation) ---
            # RL 单节点动作：同一节点承载链上全部子任务
            node_list = [node] * len(tasks)
            sense, q_time, comp, res, migration = simulator.evaluate_step(
                node_list, req.task_chain, req.user, req.trigger_time
            )

            # --- 阶段 C: 目标函数计算 ---
            aoi = estimate_aoi(sense, q_time, comp, res)
            cost = compute_cost(node_list, req.task_chain, bool(migration))

            # 贪心目标：最小化瞬时成本与 AoI 的加权和
            score = alpha * cost + beta * aoi

            # ==========================================
            # 2. 记录当前最优的动作索引
            # ==========================================
            if score < best_score:
                best_score = score
                best_action_idx = idx

        # ==========================================
        # 3. 异常处理 (资源枯竭时的兜底)
        # ==========================================
        if best_action_idx is None:
            # 当网络严重拥塞或内存全部耗尽时，没有任何合法节点。
            # 按照 RL 环境的规矩，我们随机返回一个动作（或固定返回0），
            # 让环境 (env.step) 去触发 RuntimeError 并给予巨大的 R_penalty 惩罚。
            best_action_idx = random.randint(0, self.env.num_nodes - 1)
            logger.warning(f"[Greedy] 资源枯竭，无法找到合法节点，将被迫执行非法动作 {best_action_idx}")

        return best_action_idx, None
