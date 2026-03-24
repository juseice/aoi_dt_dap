# utils/data_generator.py
import random
import networkx as nx
import pickle
import os
from core.network import Network, Edge
from core.node import EdgeNode
from core.sensor import Sensor
from core.user import UserNode
from core.task import Task, TaskChain
from environment.event import generate_poisson_requests
from utils.logger import logger


def generate_synthetic_dataset(
        num_edge_nodes=15,
        num_sensors=5,
        num_users=10,
        num_dt_services=5,
        total_requests=200,
        arrival_rate=1.0,
        seed=42
):
    """
    生成一个完整的大型随机边缘计算网络数据集
    """
    random.seed(seed)
    np_seed = seed  # 用于 networkx 等库

    net = Network()

    # ==========================================
    # 1. 生成基础设施：边缘节点与核心 Mesh 网
    # ==========================================
    # 使用随机几何图生成具有空间局部性的边缘节点连接 (距离小于 0.4 的节点互联)
    G_base = nx.random_geometric_graph(num_edge_nodes, radius=0.4, seed=np_seed)

    # 确保图是连通的，如果不连通，添加边直到连通
    while not nx.is_connected(G_base):
        components = list(nx.connected_components(G_base))
        u = random.choice(list(components[0]))
        v = random.choice(list(components[1]))
        G_base.add_edge(u, v)

    edge_nodes = []
    for i in range(num_edge_nodes):
        # 随机化节点的硬件配置
        compute = random.uniform(5.0, 20.0)  # 算力 5~20 GFLOPs
        memory = random.choice([4.0, 8.0, 16.0, 32.0])  # 内存规格
        cost = random.uniform(1.0, 5.0)  # 运行单价

        node = EdgeNode(id=f"EN_{i}", compute_power=compute, cost=cost, memory=memory)
        edge_nodes.append(node)
        net.add_node(node)

    # 链路 (前缀: Link_E2E_)
    global_link_counter = 0
    for u, v in G_base.edges():
        bw = random.uniform(10.0, 50.0)  # 核心网带宽较高
        node_u, node_v = edge_nodes[u], edge_nodes[v]
        net.add_edge(Edge(f"Link_E2E_{global_link_counter}", node_u, node_v, bandwidth=bw))
        net.add_edge(Edge(f"Link_E2E_{global_link_counter + 1}", node_v, node_u, bandwidth=bw))
        global_link_counter += 2

    # ==========================================
    # 2. 生成物理终端：传感器与用户
    # ==========================================
    sensors = []
    for i in range(num_sensors):
        sensor = Sensor(id=f"SN_{i}", data_size=random.uniform(1.0, 5.0))
        sensors.append(sensor)
        net.add_node(sensor)
        # 每个传感器随机连接边缘节点 (AP)
        connected_nodes = random.sample(edge_nodes, k=random.randint(2, 5))
        for node in connected_nodes:
            bw = random.uniform(2.0, 10.0)  # 接入网带宽较低
            net.add_edge(Edge(f"Link_S2E_{global_link_counter}", sensor, node, bandwidth=bw))
            global_link_counter += 1

    users = []
    for i in range(num_users):
        user = UserNode(id=f"U_{i}")
        users.append(user)
        net.add_node(user)
        # 用户通常只连接离自己最近的 1 个边缘节点 (单归属)
        node = random.choice(edge_nodes)
        bw = random.uniform(10.0, 20.0)
        net.add_edge(Edge(f"Link_E2U_{global_link_counter}", node, user, bandwidth=bw))

    # ==========================================
    # 3. 生成数字孪生服务逻辑：任务链
    # ==========================================
    task_chains = []
    task_counter = 0
    for i in range(num_dt_services):
        # 每个 DT 包含 1~3 个子任务
        num_subtasks = random.randint(1, 3)
        tasks = []
        for _ in range(num_subtasks):
            t = Task(
                id=f"Task_{task_counter}",
                workload=random.uniform(5.0, 15.0),
                deployment_cost=random.uniform(2.0, 10.0),
                memory_requirement=random.uniform(1.0, 2.0)
            )
            tasks.append(t)
            task_counter += 1

        # 随机绑定一个对应的物理传感器
        bound_sensor = random.choice(sensors).id
        req_bw = random.uniform(1.0, 5.0)

        dt = TaskChain(id=f"DT_{i + 1}", tasks=tasks, sensor_id=bound_sensor, required_bandwidth=req_bw)
        task_chains.append(dt)

    # ==========================================
    # 4. 生成动态请求流
    # ==========================================
    request_stream = generate_poisson_requests(
        users=users,
        task_chains=task_chains,
        arrival_rate=arrival_rate,
        total_requests=total_requests,
        seed=seed
    )

    dataset = {
        'network': net,
        'users': users,
        'task_chains': task_chains,
        'request_stream': request_stream,
        'config': {
            'num_edge_nodes': num_edge_nodes,
            'total_requests': total_requests,
            'seed': seed
        }
    }

    return dataset


def save_dataset(dataset, filename="data/dataset_default.pkl"):
    """将生成的数据集序列化保存到本地硬盘"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    logger.info(f"数据集已保存至 {filename}")


def load_dataset(filename="data/dataset_default.pkl"):
    """从本地硬盘加载数据集"""
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    logger.info(
        f"成功加载数据集 {filename} (节点数: {dataset['config']['num_edge_nodes']}, 请求数: {dataset['config']['total_requests']})")
    return dataset
