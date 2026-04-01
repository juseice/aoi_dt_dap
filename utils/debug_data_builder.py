# utils/debug_dataset_builder.py
import os
import pickle
from core.network import Network, Edge
from core.node import EdgeNode
from core.sensor import Sensor
from core.user import UserNode
from core.task import Task, TaskChain
from environment.event import Request
from utils.logger import logger


def build_debug_dataset():
    """
    手搓一个极小、确定的数据集，专门用于肉眼 Debug 和验证算法逻辑。
    包含 3个节点，2个传感器，2个用户，2个DT，以及 5 个精心设计时间戳的请求。
    """
    net = Network()

    # ==========================================
    # 1. 节点设计：故意制造 Trade-off
    # ==========================================
    # EN_0: 算力极强，但极其昂贵，且内存极小 (只能处理小任务，大任务会爆内存)
    en_0 = EdgeNode(id="EN_0", compute_power=20.0, cost=10.0, memory=4.0)

    # EN_1: 算力中等，价格中等，内存中等 (万金油节点)
    en_1 = EdgeNode(id="EN_1", compute_power=10.0, cost=5.0, memory=8.0)

    # EN_2: 算力极弱，超级便宜，内存巨大 (处理极其慢，容易造成排队灾难)
    en_2 = EdgeNode(id="EN_2", compute_power=2.0, cost=1.0, memory=32.0)

    for n in [en_0, en_1, en_2]:
        net.add_node(n)

    # 节点之间的骨干网 (高带宽)
    net.add_edge(Edge(id="Link_E0_E1", source_node=en_0, terminal_node=en_1, bandwidth=50.0))
    net.add_edge(Edge(id="Link_E1_E0", source_node=en_1, terminal_node=en_0, bandwidth=50.0))
    net.add_edge(Edge(id="Link_E1_E2", source_node=en_1, terminal_node=en_2, bandwidth=50.0))
    net.add_edge(Edge(id="Link_E2_E1", source_node=en_2, terminal_node=en_1, bandwidth=50.0))

    # ==========================================
    # 2. 终端设计：制造距离差异
    # ==========================================
    # SN_0 和 U_0 离 EN_0 近
    sn_0 = Sensor(id="SN_0", data_size=2.0)
    u_0 = UserNode(id="U_0")
    net.add_node(sn_0)
    net.add_node(u_0)
    net.add_edge(Edge(id="L_S0_E0", source_node=sn_0, terminal_node=en_0, bandwidth=10.0))
    net.add_edge(Edge(id="L_E0_U0", source_node=en_0, terminal_node=u_0, bandwidth=10.0))

    # SN_1 和 U_1 离 EN_2 近
    sn_1 = Sensor(id="SN_1", data_size=5.0)
    u_1 = UserNode(id="U_1")
    net.add_node(sn_1)
    net.add_node(u_1)
    net.add_edge(Edge(id="L_S1_E2", source_node=sn_1, terminal_node=en_2, bandwidth=10.0))
    net.add_edge(Edge(id="L_E2_U1", source_node=en_2, terminal_node=u_1, bandwidth=10.0))
    # 为了保证全网连通性，给传感器和用户加上通往中转节点 EN_1 的备用链路(低带宽)
    net.add_edge(Edge(id="L_S0_E1", source_node=sn_0, terminal_node=en_1, bandwidth=2.0))
    net.add_edge(Edge(id="L_E1_U0", source_node=en_1, terminal_node=u_0, bandwidth=2.0))
    net.add_edge(Edge(id="L_S1_E1", source_node=sn_1, terminal_node=en_1, bandwidth=2.0))
    net.add_edge(Edge(id="L_E1_U1", source_node=en_1, terminal_node=u_1, bandwidth=2.0))

    # ==========================================
    # 3. 任务链设计：一大一小
    # ==========================================
    # DT_0: 轻量级服务 (关联 SN_0)，需要 2G 内存，部署费 5.0
    t1 = Task(id="T1", workload=4.0, deployment_cost=5.0, memory_requirement=2.0)
    dt_0 = TaskChain(id="DT_0", tasks=[t1], sensor_id="SN_0", required_bandwidth=1.0)

    # DT_1: 重量级服务 (关联 SN_1)，需要 6G 内存，部署费 10.0
    t2 = Task(id="T2", workload=20.0, deployment_cost=10.0, memory_requirement=6.0)
    dt_1 = TaskChain(id="DT_1", tasks=[t2], sensor_id="SN_1", required_bandwidth=2.0)

    # ==========================================
    # 4. 手搓极度刁钻的请求流 (5 个请求)
    # ==========================================
    request_stream = [
        # Req 1: U_0 请求轻量 DT_0 (期望它去 EN_0，因为近且快，且内存够)
        Request(req_id=1, trigger_time=0.0, user=u_0, task_chain=dt_0),

        # Req 2: 紧接着 U_1 请求重量 DT_1
        # 陷阱：贪心算法如果只看算力，可能会把它塞给 EN_0，但 EN_0 内存只有 4G，DT_1 需要 6G，会直接部署失败！
        # 期望它去 EN_1 (万金油) 或者 EN_2 (便宜但慢)
        Request(req_id=2, trigger_time=0.1, user=u_1, task_chain=dt_1),

        # Req 3: 短时间内，U_1 再次请求重量 DT_1 (测缓存和排队)
        # 期望：如果 Req 2 部署在 EN_1，Req 3 应该原地命中缓存 (migrated=False)，但会产生排队时延
        Request(req_id=3, trigger_time=0.5, user=u_1, task_chain=dt_1),

        # Req 4: 过了很久，U_0 请求 DT_1
        # 测点：此时触发了垃圾回收，之前的实例被清空了。这是一个跨区域请求。
        Request(req_id=4, trigger_time=15.0, user=u_0, task_chain=dt_1),

        # Req 5: U_1 请求 DT_0
        Request(req_id=5, trigger_time=16.0, user=u_1, task_chain=dt_0),
    ]

    dataset = {
        'network': net,
        'users': [u_0, u_1],
        'task_chains': [dt_0, dt_1],
        'request_stream': request_stream,
        'config': {'num_edge_nodes': 3, 'total_requests': 5, 'seed': 'debug'}
    }

    os.makedirs("data", exist_ok=True)
    filename = "data/dataset_debug.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    logger.info(f"Debug 数据集已生成至 {filename}")


if __name__ == "__main__":
    build_debug_dataset()
