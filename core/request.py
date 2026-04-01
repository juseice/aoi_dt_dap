from enum import Enum
from typing import Optional


class RequestStatus(Enum):
    """请求的生命周期状态"""
    PENDING = "PENDING"       # 已触发，正在等待调度或排队中
    PROCESSING = "PROCESSING" # 正在边缘节点上执行计算/同步
    COMPLETED = "COMPLETED"   # 执行完毕，结果已返回给用户
    FAILED = "FAILED"         # 部署失败（例如：内存不足或无可用路由）


class Request:
    """定义一个独立的请求事件"""
    def __init__(self, req_id, trigger_time, user, task_chain):
        self.id = req_id
        self.trigger_time = trigger_time
        self.user = user
        self.task_chain = task_chain

        self.status: RequestStatus = RequestStatus.PENDING
        self.assigned_node = None  # 最终被 PPO 算法分配到哪个边缘节点 (EdgeNode)

        self.start_processing_time: Optional[float] = None  # 开始执行的时间
        self.finish_time: Optional[float] = None  # 彻底完成并返回结果的时间

    # ==========================================
    # 辅助计算方法 (为你的 PPO Reward 提供直接的数据接口)
    # ==========================================
    @property
    def queuing_delay(self) -> float:
        """排队时延：从触发到开始处理之间的时间"""
        if self.start_processing_time is None:
            return 0.0
        return self.start_processing_time - self.trigger_time

    @property
    def processing_delay(self) -> float:
        """处理时延：包含计算时间和可能的传输时间"""
        if self.start_processing_time is None or self.finish_time is None:
            return 0.0
        return self.finish_time - self.start_processing_time

    @property
    def total_delay(self) -> float:
        """总响应时延 (Response Time)"""
        if self.finish_time is None:
            return 0.0
        return self.finish_time - self.trigger_time

    # ==========================================
    # 状态更新逻辑
    # ==========================================

    def mark_as_processing(self, current_time: float, assigned_node):
        """当环境的调度器(Agent)决定处理该请求时调用"""
        self.status = RequestStatus.PROCESSING
        self.start_processing_time = current_time
        self.assigned_node = assigned_node

    def mark_as_completed(self, current_time: float):
        """当任务在边缘节点执行完毕时调用"""
        self.status = RequestStatus.COMPLETED
        self.finish_time = current_time

    def mark_as_failed(self):
        """当资源超限，硬约束被打破时调用"""
        self.status = RequestStatus.FAILED

    def __repr__(self):
        """友好的打印格式，方便 Debug"""
        assigned = self.assigned_node.id if self.assigned_node else "None"
        return (f"<Request[{self.id}] "
                f"Time:{self.trigger_time:.2f} | "
                f"User:{self.user.id} -> DT:{self.task_chain.id} | "
                f"Status:{self.status.value} | Node:{assigned}>")