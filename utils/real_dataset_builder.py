import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from geopy.distance import geodesic
import random

from core.network import Network, Edge
from core.node import EdgeNode
from core.user import UserNode
from core.sensor import Sensor
from core.task import Task, TaskChain
from core.request import Request
from utils.logger import logger
from utils.alibaba_trace_loader import AlibabaTraceLoader


def associate_user_to_ap(user_node, lat, lon, edge_nodes, net):
    """
    根据用户当前坐标，动态连接到最近的边缘节点 (AP)。
    注意：在 Dataset 构建阶段调用此函数，会导致 net 对象最终只保留用户最后一次请求时的拓扑状态。
    """
    user_node.lat = lat
    user_node.lon = lon

    # 先清理该用户旧的接入链路 (模拟移动过程中的 AP 切换)
    # edges_to_remove = [e for e in net.edges if e.target_node == user_node or e.source_node == user_node]
    # for e in edges_to_remove:
    #     net.remove_edge(e)

    # 寻找当前最近的基站
    min_dist = float('inf')
    nearest_en = None
    for en in edge_nodes:
        dist = geodesic((lat, lon), (en.lat, en.lon)).meters
        if dist < min_dist:
            min_dist = dist
            nearest_en = en

    # 建立基站到用户的下行链路 (Downlink)，用于传输 DT 查询结果
    downlink_bw = max(5.0, 50.0 - (min_dist / 100.0))

    net.add_edge(Edge(id=f"Downlink_{nearest_en.id}_{user_node.id}",
                      source_node=nearest_en, terminal_node=user_node, bandwidth=downlink_bw))

    return nearest_en


def build_real_dataset(telecom_path: str, alibaba_path: str,
                       num_edge_nodes: int = 20, num_dts: int = 10,
                       num_sensors: int = 15, total_requests: int = 500,
                       seed: int = 42):
    """
    基于真实数据集（上海电信 + 阿里集群）构建边缘数字孪生仿真环境
    """
    np.random.seed(seed)
    random.seed(seed)
    net = Network()

    # 缓存字典与列表
    users_dict = {}
    edge_nodes = []
    sensors = []
    task_chains = []
    request_stream = []

    logger.info("=== 开始构建数据驱动的边缘仿真环境 ===")

    # ==========================================
    # 第一步：解析上海电信数据集 -> 生成 Network & EdgeNodes (物理骨干网)
    # ==========================================
    logger.info("1. 正在从上海电信数据集提取网络拓扑...")
    df_telecom = pd.read_excel(telecom_path)
    df_telecom = df_telecom.dropna(subset=['latitude', 'longitude', 'start time', 'user id'])

    # 提取唯一的基站位置
    unique_locations = df_telecom[['latitude', 'longitude']].drop_duplicates().values
    if len(unique_locations) > num_edge_nodes:
        indices = np.random.choice(len(unique_locations), num_edge_nodes, replace=False)
        selected_locs = unique_locations[indices]
    else:
        selected_locs = unique_locations

    for i, loc in enumerate(selected_locs):
        # 异构节点生成：随机分配 3 个梯队的算力和内存 (20%强，60%中，20%弱)
        tier = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        if tier == 0:
            cp, mem, cost = 20.0, 16.0, 10.0
        elif tier == 1:
            cp, mem, cost = 10.0, 8.0, 5.0
        else:
            cp, mem, cost = 2.0, 4.0, 1.0

        en = EdgeNode(id=f"EN_{i}", compute_power=cp, cost=cost, memory=mem)
        en.lat, en.lon = loc[0], loc[1]
        edge_nodes.append(en)
        net.add_node(en)

    # 构建基站间的骨干网链路
    for i in range(len(edge_nodes)):
        for j in range(i + 1, len(edge_nodes)):
            dist_m = geodesic((edge_nodes[i].lat, edge_nodes[i].lon),
                              (edge_nodes[j].lat, edge_nodes[j].lon)).meters
            bw = max(50.0, 200.0 - (dist_m / 1000.0) * 20.0)
            net.add_edge(
                Edge(id=f"Link_E{i}_E{j}", source_node=edge_nodes[i], terminal_node=edge_nodes[j], bandwidth=bw))
            net.add_edge(
                Edge(id=f"Link_E{j}_E{i}", source_node=edge_nodes[j], terminal_node=edge_nodes[i], bandwidth=bw))

    # ==========================================
    # 第二步：生成物理传感器 (Sensors) 并绑定上行链路
    # ==========================================
    logger.info("2. 正在生成静态物理传感器与上行链路...")
    lats = [en.lat for en in edge_nodes]
    lons = [en.lon for en in edge_nodes]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    for i in range(num_sensors):
        s_lat = random.uniform(min_lat, max_lat)
        s_lon = random.uniform(min_lon, max_lon)
        data_size = random.uniform(1.0, 5.0)

        sensor = Sensor(id=f"Sensor_{i}", data_size=data_size)
        sensor.lat, sensor.lon = s_lat, s_lon
        sensors.append(sensor)
        net.add_node(sensor)

        # 绑定到距离最近的边缘节点
        min_dist = float('inf')
        nearest_en = None
        for en in edge_nodes:
            dist = geodesic((s_lat, s_lon), (en.lat, en.lon)).meters
            if dist < min_dist:
                min_dist = dist
                nearest_en = en

        uplink_bw = random.uniform(10.0, 20.0)
        net.add_edge(Edge(id=f"Uplink_{sensor.id}_{nearest_en.id}",
                          source_node=sensor, terminal_node=nearest_en, bandwidth=uplink_bw))

    # ==========================================
    # 第三步：解析阿里数据集 -> 生成数字孪生实例 (TaskChains)
    # ==========================================
    logger.info("3. 基于阿里巴巴集群轨迹生成数字孪生任务链...")
    ali_loader = AlibabaTraceLoader(file_path=alibaba_path, edge_scale_factor=0.1)
    task_counter = 0

    for i in range(num_dts):
        num_subtasks = random.randint(2, 4)
        tasks = []
        ali_samples = ali_loader.sample_tasks(n=num_subtasks)
        if num_subtasks == 1:
            ali_samples = [ali_samples]

        for sample in ali_samples:
            t = Task(
                id=f"Task_{task_counter}",
                workload=sample['workload'],
                deployment_cost=sample['duration'] * 0.05,
                memory_requirement=sample['memory']
            )
            tasks.append(t)
            task_counter += 1

        # 随机绑定一个对应的物理传感器
        bound_sensor = random.choice(sensors)
        req_bw = bound_sensor.data_size * random.uniform(0.8, 1.2)

        dt = TaskChain(
            id=f"DT_{i}",
            tasks=tasks,
            sensor_id=bound_sensor.id,
            required_bandwidth=req_bw
        )
        task_chains.append(dt)

    # ==========================================
    # 第四步：提取真实请求 -> 生成 Users & RequestStream
    # ==========================================
    logger.info("4. 提取真实移动轨迹，生成用户请求流...")
    df_telecom['start time'] = pd.to_datetime(df_telecom['start time'])
    df_telecom = df_telecom.sort_values(by='start time')
    df_requests = df_telecom.head(total_requests).copy()

    base_time = df_requests['start time'].iloc[0]
    df_requests['relative_time'] = (df_requests['start time'] - base_time).dt.total_seconds()

    # 提前计算 Zipf 分布概率，用于分配请求的目标 DT
    zipf_probs = 1.0 / np.power(np.arange(1, num_dts + 1), 0.8)
    zipf_probs /= np.sum(zipf_probs)

    req_id_counter = 1
    for _, row in df_requests.iterrows():
        uid = str(row['user id'])
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        trigger_time = float(row['relative_time'])

        if uid not in users_dict:
            new_user = UserNode(id=f"U_{uid}")
            users_dict[uid] = new_user
            net.add_node(new_user)
        user = users_dict[uid]

        # 动态接入最近的 AP
        associate_user_to_ap(user, lat, lon, edge_nodes, net)

        # 采用长尾分布随机分配一个目标 DT
        target_dt_idx = np.random.choice(num_dts, p=zipf_probs)
        target_dt = task_chains[target_dt_idx]

        req = Request(
            req_id=req_id_counter,
            trigger_time=trigger_time,
            user=user,
            task_chain=target_dt
        )
        request_stream.append(req)
        req_id_counter += 1

    # ==========================================
    # 组装返回数据
    # ==========================================
    dataset = {
        'network': net,
        'users': list(users_dict.values()),
        'task_chains': task_chains,
        'request_stream': request_stream,
        'config': {
            'num_edge_nodes': len(edge_nodes),
            'num_dts': len(task_chains),
            'num_sensors': len(sensors),
            'total_requests': len(request_stream),
            'seed': seed,
            'source': 'ShanghaiTelecom + AlibabaCluster'
        }
    }

    logger.info(
        f"=== 构建完成！节点:{len(edge_nodes)} | 传感器:{len(sensors)} | DT数:{len(task_chains)} | 用户:{len(users_dict)} | 请求:{len(request_stream)} ===")
    return dataset


if __name__ == "__main__":
    # 测试运行
    # 请替换为你的实际文件路径
    telecom_file = "../datasets/telecom-shanghai-dataset/data_10.1610.31.xlsx"
    alibaba_file = "../datasets/cluster-trace-gpu-v2025/disaggregated_DLRM_trace.csv"

    # 建议先跑一个小规模测试
    ds = build_real_dataset(telecom_path=telecom_file, alibaba_path=alibaba_file,
                            num_edge_nodes=5, num_dts=3, total_requests=10)

    print("\n--- 成功生成的真实请求流 ---")
    for r in ds['request_stream']:
        print(f"Time: {r.trigger_time:.1f}s | User: {r.user.id[:10]}... | Target DT: {r.task_chain.id}")
