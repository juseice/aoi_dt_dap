# algorithms/dp_agent.py
from optimization.dp_solver import solve_dp_offline
from utils.logger import logger


class DPAgent:
    """
    动态规划基准智能体 (Dynamic Programming Agent)
    核心特性：
    1. 初始化时，利用“上帝视角”读取完整请求流，调用 DP 求解器进行一次性离线求解。
    2. predict() 接口：作为仿真流水线中的运动员 (DP Real)，按序吐出算好的动作。
    3. get_oracle_history() 接口：作为预言家 (DP Oracle)，直接暴管理论下界数据。
    """

    def __init__(self, env):
        self.env = env
        self.request_stream = env.request_stream

        logger.info("[DPAgent] 正在提取全局请求流，执行离线动态规划 (Oracle) 求解...")

        # 提取全局最优计划
        # 格式预期: [{'req': req_id, 'node_id': id, 'aoi': val, 'cost': val, 'migrated': bool}, ...]
        self.oracle_plan = solve_dp_offline(
            network=self.env.network,
            request_stream=self.request_stream,
            alpha=self.env.alpha,
            beta=self.env.beta
        )

        # 健壮性校验
        if not self.oracle_plan or len(self.oracle_plan) != len(self.request_stream):
            logger.error("[DPAgent] DP 求解失败或返回的计划长度与请求流不匹配！")
            self.valid = False
        else:
            self.valid = True

        self.current_step = 0

    def predict(self, obs=None, deterministic=True):
        """
        在线交互接口：每一步从“锦囊”里掏出一个预先算好的动作交给环境去校验。

        :param obs: 忽略，因为 DP 已经拥有了全局视角
        :param deterministic: 兼容参数
        :return: action_idx (int), state (None)
        """
        # 防止越界或求解失败时的崩溃
        if not self.valid or self.current_step >= len(self.oracle_plan):
            logger.warning("[DPAgent] 预定计划已耗尽或失效，执行无效动作。")
            return 0, None

        # 1. 查阅 DP 锦囊，获取当前步应该部署的节点 ID 列表（每任务一个）
        planned_node_ids = self.oracle_plan[self.current_step]['node_ids']
        # RL env 使用单节点动作，取链首节点 ID 作为动作目标
        primary_node_id = planned_node_ids[0]

        # 2. 将真实的物理节点 ID 转换为 Env 需要的动作索引 (Action Index)
        try:
            action_idx = [n.id for n in self.env.edge_nodes].index(primary_node_id)
        except ValueError:
            logger.error(f"[DPAgent] 严重错误：DP 计算出的节点 {primary_node_id} 不在环境可用节点中！")
            action_idx = 0

        # 3. 步数加一，准备下一次查询
        self.current_step += 1

        return action_idx, None

    def get_oracle_history(self):
        """
        获取理论上的绝对最优数据 (DP Oracle)。
        注意：这些数据无视了物理机的动态内存上限，也没有计算突发流量带来的真实排队拥塞，
        它代表了当前系统拓扑结构下的极限性能下界。
        """
        if not self.valid:
            return []
        return self.oracle_plan
