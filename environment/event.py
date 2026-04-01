# environment/event.py
import random
from core.request import Request

def generate_poisson_requests(users, task_chains, arrival_rate=1.0, total_requests=20, seed=None):
    """
    生成基于泊松过程的事件流
    :param users: 可用的用户对象列表
    :param task_chains: 可用的数字孪生服务(TaskChain)列表
    :param arrival_rate: lambda (λ)，平均每秒到达的请求数量。值越大，请求越密集，越容易引发网络拥塞。
    :param total_requests: 需要生成的总请求数
    :param seed: 随机种子，用于复现实验结果
    """
    if seed is not None:
        random.seed(seed)

    requests = []
    current_time = 0.0

    for i in range(1, total_requests + 1):
        # 【核心数学逻辑】：泊松过程的时间间隔服从指数分布
        # random.expovariate(λ) 会根据均值 1/λ 返回下一个事件发生需要等待的时间
        inter_arrival_time = random.expovariate(arrival_rate)
        current_time += inter_arrival_time

        # 随机分配该请求是谁发起的，请求哪个服务
        # (后续如果你想做差异化，比如 User 1 特别活跃，可以在这里改成带权重的 random.choices)
        user = random.choice(users)
        dt = random.choice(task_chains)

        req = Request(
            req_id=i,
            trigger_time=current_time,
            user=user,
            task_chain=dt
        )
        requests.append(req)

    return requests
