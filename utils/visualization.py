# utils/visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import collections
import math
import os
import seaborn as sns
from pathlib import Path

from core import EdgeNode
from core import Sensor
from core import UserNode


def _load_geo_dict_from_csv(csv_path):
    """内部辅助函数：从 CSV 加载坐标字典"""
    geo_dict = {}
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # 建立字典: {'EN_0': (Longitude, Latitude)}
                geo_dict[str(row['Node_ID'])] = (row['Longitude'], row['Latitude'])
            print(f"成功从 CSV 加载了 {len(geo_dict)} 个节点的地理坐标！")
        except Exception as e:
            print(f"读取 CSV 坐标失败: {e}")
    return geo_dict


def plot_network_topology(network):
    """
    绘制异构边缘网络拓扑图，区分传感器、边缘节点和用户
    """
    G = network.graph

    # 我们手动将节点分为三层，实现横向流动。
    layer_map = {}  # node_id -> layer_index (0: Sensor, 1: Edge, 2: User)

    sensors = []
    edge_nodes = []
    users = []

    # 节点分类
    for node_id, data in G.nodes(data=True):
        node = data.get('node')
        if isinstance(node, Sensor):
            sensors.append(node_id)
            layer_map[node_id] = 0
        elif isinstance(node, UserNode):
            users.append(node_id)
            layer_map[node_id] = 2
        else:
            edge_nodes.append(node_id)
            layer_map[node_id] = 1

    # 手动计算松散的层次布局位置
    pos = {}

    # 定义布局参数 (可根据节点数量微调)
    X_SPACE = 10.0  # 层与层之间的横向距离
    Y_SPACE_SENSOR = 4.0  # 传感器之间的垂直距离
    Y_SPACE_EDGE = 3.0  # 边缘节点之间的垂直距离 (Mesh网需要更松散)
    Y_SPACE_USER = 5.0  # 用户之间的垂直距离

    # 排序节点 ID 以保证每次生成的图位置固定，不闪烁
    sensors.sort()
    edge_nodes.sort()
    users.sort()

    # 计算 0 层 (Sensors, 左)
    total_height_s = (len(sensors) - 1) * Y_SPACE_SENSOR
    for i, node_id in enumerate(sensors):
        # 居中排列
        y_coord = (total_height_s / 2.0) - (i * Y_SPACE_SENSOR)
        pos[node_id] = (0, y_coord)

    # 计算 1 层 (Edge Nodes, 中) - 这是 Mesh 网所在层，需要特别松散
    total_height_e = (len(edge_nodes) - 1) * Y_SPACE_EDGE
    # 边缘节点层整体往右移
    for i, node_id in enumerate(edge_nodes):
        y_coord = (total_height_e / 2.0) - (i * Y_SPACE_EDGE)
        pos[node_id] = (X_SPACE, y_coord)

    # 计算 2 层 (Users, 右)
    total_height_u = (len(users) - 1) * Y_SPACE_USER
    for i, node_id in enumerate(users):
        y_coord = (total_height_u / 2.0) - (i * Y_SPACE_USER)
        # 用户层再往右移
        pos[node_id] = (X_SPACE * 2, y_coord)

    # 3. 优化绘图参数 (Optimize Plotting Params)
    # 显著增大画布大小，为“松散”提供空间
    plt.figure(figsize=(16, 10), dpi=100)

    # 调整节点大小，相比之前略微减小以减少重叠感
    NODE_SIZE_BASE = 1600
    EDGE_SIZE = NODE_SIZE_BASE
    SENSOR_SIZE = NODE_SIZE_BASE * 0.8
    USER_SIZE = NODE_SIZE_BASE * 0.8

    # 绘制节点 (使用更现代、对比度更高的颜色组合)
    # edgecolors='black' 增加节点描边，提高清晰度
    nx.draw_networkx_nodes(G, pos, nodelist=sensors, node_color='#4caf50', node_shape='^', node_size=SENSOR_SIZE,
                           label='Sensors', edgecolors='black', linewidths=1.5)  # 鲜艳绿
    nx.draw_networkx_nodes(G, pos, nodelist=edge_nodes, node_color='#2196f3', node_shape='o', node_size=EDGE_SIZE,
                           label='Edge Nodes', edgecolors='black', linewidths=1.5)  # 鲜艳蓝
    nx.draw_networkx_nodes(G, pos, nodelist=users, node_color='#f44336', node_shape='s', node_size=USER_SIZE,
                           label='Users', edgecolors='black', linewidths=1.5)  # 鲜艳红

    # 绘制边 (增加透明度和箭头大小)
    # edge_color 为灰色，避免干扰节点颜色
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=25, edge_color='#757575', width=1.5,
                                   alpha=0.6)  # 半透明灰色有向边

    # 4. 彻底解决标签遮挡 (Fix Label Occlusion)
    # 核心技巧：不要把标签画在节点中心，而是画在节点正下方
    label_pos = {k: (v[0], v[1] - 0.7) for k, v in pos.items()}  # Y轴整体下移 0.7

    # 绘制节点 ID 标签 (字体加粗，颜色调深)
    nx.draw_networkx_labels(G, label_pos, font_size=12, font_family="DejaVu Sans", font_weight='bold',
                            font_color='#212121')

    # 绘制边带宽标签
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        edge_labels[(u, v)] = f"{d['edge'].bandwidth}M"

    # 优化边标签样式：增加白色半透明背景 (bbox)， label_pos调到0.3避免在Mesh网中心重叠
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='#424242',
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7,
                                           boxstyle='round,pad=0.2'),
                                 label_pos=0.3)  # 靠近源端，避免 Mesh 网多条边中心标签重合

    # 5. 绘图修饰 (Plot Polish)
    plt.title("Heterogeneous Edge Network Topology (Layered View)", fontsize=20, fontweight='bold', pad=20)

    # 增加边距，防止边界节点被切掉
    ax = plt.gca()
    ax.margins(0.1)  # 增加 10% 的 margins

    # 调整图例位置到下方，不遮挡网络
    plt.legend(scatterpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12, frameon=False)

    plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()  # 自动调整布局
    plt.show()


def plot_macro_topology(network, geo_csv_path=None):
    """
    绘制用于大规模节点（如 50+ 节点）的宏观全景拓扑图。
    去除了冗余的文字标签、边带宽文字，缩小了节点体积，
    优先采用真实地理坐标布局，展现网络全景。
    """
    G = network.graph

    sensors = []
    edge_nodes = []
    users = []

    pos = {}

    external_geo_dict = _load_geo_dict_from_csv(geo_csv_path)
    # 1. 节点分类与坐标提取
    for node_id, data in G.nodes(data=True):
        node = data.get('node')
        if type(node).__name__ == 'Sensor':  # 兼容不同导入方式
            sensors.append(node.id)
        elif type(node).__name__ == 'UserNode':
            users.append(node.id)
        else:
            edge_nodes.append(node.id)

        # 尝试获取经纬度作为真实地理位置坐标
        node_id_str = str(node.id)
        if node_id_str in external_geo_dict:
            pos[node.id] = external_geo_dict[node_id_str]
        elif hasattr(node, 'lon') and hasattr(node, 'lat'):
            pos[node.id] = (node.lon, node.lat)
        else:
            print(f"警告: 节点 {node.id} 无地理信息！")

    use_geo = len(pos) > 0

    if use_geo:
        pos = nx.spring_layout(G, pos=pos, k=0.05, iterations=15, seed=42)
    else:
        pos = nx.spring_layout(G, k=0.4, iterations=150, seed=42)

    # 3. 优化大规模绘图参数
    # 使用正方形画布，更适合地理分布和弹簧布局
    plt.figure(figsize=(12, 12), dpi=150)

    # 极简绘制节点 (大幅缩小 Size，去除粗黑边框)
    EDGE_SIZE = 120
    SENSOR_SIZE = 60
    USER_SIZE = 60

    nx.draw_networkx_nodes(G, pos, nodelist=edge_nodes, node_color='#2196f3',
                           node_shape='o', node_size=EDGE_SIZE, label='Edge Nodes', alpha=0.9, edgecolors='white',
                           linewidths=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=sensors, node_color='#4caf50',
                           node_shape='^', node_size=SENSOR_SIZE, label='Sensors', alpha=0.9, edgecolors='white',
                           linewidths=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=users, node_color='#f44336',
                           node_shape='s', node_size=USER_SIZE, label='Users', alpha=0.9, edgecolors='white',
                           linewidths=0.5)

    # 极简绘制边 (去箭头、降透明度、减细线条)
    # 对于 50 个节点，边可能多达几百条，浅灰色 + 半透明能画出类似“光纤网络”的高级质感
    nx.draw_networkx_edges(G, pos, edge_color='#BDBDBD', width=0.6, alpha=0.3, arrows=False)

    # 4. 图表修饰
    layout_name = "Real Geo-Location Based" if use_geo else "Force-Directed Layout"
    plt.title(f"Macro Network Topology Overview\n({layout_name}, N={len(edge_nodes)})",
              fontsize=16, fontweight='bold', pad=20)

    # 调整图例到右上角，并增加背景透明度防遮挡
    plt.legend(scatterpoints=1, loc='upper right', fontsize=12, framealpha=0.9, edgecolor='#E0E0E0')

    plt.axis('off')  # 隐藏刻度轴
    plt.tight_layout()
    plt.show()

    return pos


def plot_infrastructure_topology(network, geo_csv_path=None, fixed_pos=None):
    """
    基础设施拓扑图：屏蔽移动用户，仅展示 Edge Nodes 和 Sensors，
    用于体现边缘设备与数据源的绑定关系。
    """
    G = network.graph

    # 1. 筛选保留的节点 (排除 UserNode)
    infra_nodes = []
    for node_id, data in G.nodes(data=True):
        node_type = type(data.get('node')).__name__
        if node_type in ['EdgeNode', 'Sensor']:
            infra_nodes.append(node_id)

    # 2. 生成子图 (这会自动过滤掉所有连接到用户的边)
    G_infra = G.subgraph(infra_nodes)
    external_geo_dict = _load_geo_dict_from_csv(geo_csv_path)

    sensors = []
    edge_nodes = []
    pos = {}

    for node_id, data in G_infra.nodes(data=True):
        node = data.get('node')
        if type(node).__name__ == 'Sensor':
            sensors.append(node.id)
        else:
            edge_nodes.append(node.id)
    #
    #     node_id_str = str(node.id)
    #     if node_id_str in external_geo_dict:
    #         pos[node.id] = external_geo_dict[node_id_str]
    #     elif hasattr(node, 'lon') and hasattr(node, 'lat'):
    #         pos[node.id] = (node.lon, node.lat)
    #     else:
    #         print(f"警告: 节点 {node.id} 无地理信息！")

    if fixed_pos is not None:
        pos = {n: fixed_pos[n] for n in infra_nodes if n in fixed_pos}
        use_geo = True
    else:
        # 如果没提供，才回退到自己瞎算（会导致你遇到的塌缩）
        external_geo_dict = _load_geo_dict_from_csv(geo_csv_path)
        for node_id, data in G_infra.nodes(data=True):
            node = data.get('node')
            node_id_str = str(node.id)
            if node_id_str in external_geo_dict:
                pos[node_id] = external_geo_dict[node_id_str]

        use_geo = len(pos) > 0
        if use_geo:
            pos = nx.spring_layout(G_infra, pos=pos, k=0.05, iterations=15, seed=42)
        else:
            pos = nx.spring_layout(G_infra, k=0.4, iterations=150, seed=42)

    plt.figure(figsize=(10, 10), dpi=150)

    # 节点大小保持适中
    nx.draw_networkx_nodes(G_infra, pos, nodelist=edge_nodes, node_color='#2196f3',
                           node_shape='o', node_size=200, label='Edge Nodes', edgecolors='white', linewidths=1)
    nx.draw_networkx_nodes(G_infra, pos, nodelist=sensors, node_color='#4caf50',
                           node_shape='^', node_size=100, label='Sensors', edgecolors='white', linewidths=1)

    # 因为去掉了大量用户的连线，基础设施的连线可以稍微加深一点，展现骨干质感
    nx.draw_networkx_edges(G_infra, pos, edge_color='#9E9E9E', width=1.0, alpha=0.5, arrows=False)

    layout_name = "Geo-Relaxed Layout" if use_geo else "Force-Directed Layout"
    plt.title(f"Infrastructure Topology\n({layout_name})", fontsize=16, fontweight='bold', pad=20)
    plt.legend(scatterpoints=1, loc='upper right', fontsize=12, framealpha=0.9)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_simulation_results(metrics_history):
    """
    绘制 AoI 和 Cost 的趋势折线图
    :param metrics_history: list of dict, 例如 [{'req': 1, 'aoi': 1.2, 'cost': 5, 'migrated': True}, ...]
    """
    if not metrics_history:
        print("无数据可绘制！")
        return

    reqs = [data['req'] for data in metrics_history]
    aois = [data['aoi'] for data in metrics_history]
    costs = [data['cost'] for data in metrics_history]
    migrations = [data['req'] for data in metrics_history if data['migrated']]
    mig_aois = [data['aoi'] for data in metrics_history if data['migrated']]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 1. 绘制 AoI 曲线
    ax1.plot(reqs, aois, marker='o', linestyle='-', color='#1f77b4', label='AoI')
    # 在发生迁移的点上打上红色的星号标记
    if migrations:
        ax1.scatter(migrations, mig_aois, color='red', s=100, marker='*', zorder=5, label='Migration Triggered')

    ax1.set_xlabel('Request Sequence')
    ax1.set_ylabel('Age of Information (s)')
    ax1.set_title('AoI Evolution')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 2. 绘制 Cost 曲线
    ax2.plot(reqs, costs, marker='s', linestyle='-', color='#ff7f0e', label='System Cost')
    ax2.set_xlabel('Request Sequence')
    ax2.set_ylabel('Cost')
    ax2.set_title('System Cost Evolution')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_comparative_results(histories):
    """
    绘制多种算法的对比折线图
    :param histories: 字典格式，如 {'Random Baseline': history1, 'Greedy Best': history2}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100)

    # 为不同算法分配特定的颜色和标记符号
    style_map = {
        'Random Baseline': {'color': '#f44336', 'marker': 'x', 'linestyle': '--'},  # 红色虚线
        'Greedy Best': {'color': '#2196f3', 'marker': 'o', 'linestyle': '-'},  # 蓝色实线
        # 给 DP 分配同一种绿色，用虚实线和不同的形状区分
        'DP Ideal': {'color': '#4caf50', 'marker': '^', 'linestyle': '--'},  # 绿色虚线 (三角形标记)：理想理论值
        'DP Real': {'color': '#4caf50', 'marker': 's', 'linestyle': '-'},  # 绿色实线 (正方形标记)：现实执行值
        'PPO (DRL)': {'color': '#FF9800', 'marker': 'd', 'linestyle': '-'},  # 橙色实线 (菱形标记)：强化学习结果
    }
    desired_order = ['DP Ideal (Oracle)', 'Random Baseline', 'Greedy Best', 'DP Real (Simulated)', 'PPO (DRL)']
    sorted_labels = sorted(histories.keys(), key=lambda l: desired_order.index(l) if l in desired_order else 999)

    for label in sorted_labels:
        history = histories.get(label)
        if not history:
            continue

        # 如果传入的 label 不在预设里，给个默认灰色
        style = style_map.get(label, {'color': 'gray', 'marker': '.'})

        reqs = []
        aois = []
        costs = []

        for h in history:
            reqs.append(h['req'])
            # 如果是 inf，用 None 替代，这样 matplotlib 会在这里画一个断点，
            # 或者你可以选择在汇总报告中过滤，并在画图时直接跳过非法点，
            # 这里的处理方式会在对应请求ID处留空，直观展现失败
            if math.isinf(h['aoi']):
                aois.append(None)
            else:
                aois.append(h['aoi'])

            # (原代码这里有个reqs.append重复了，且costs重复收集，已修正为单次遍历收集)
            # 处理 Cost 中的 inf 同样可以使用 None
            if math.isinf(h['cost']):
                costs.append(None)
            else:
                costs.append(h['cost'])

        # 1. 绘制 AoI 对比
        ax1.plot(reqs, aois, label=label, color=style['color'], marker=style['marker'],
                 linestyle=style.get('linestyle', '-'), linewidth=2, markersize=8)

        # 2. 绘制 Cost 对比
        ax2.plot(reqs, costs, label=label, color=style['color'], marker=style['marker'],
                 linestyle=style.get('linestyle', '-'), linewidth=2, markersize=8)

    # 装饰 AoI 图表
    ax1.set_title('Average Age of Information (AoI) Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Request Sequence', fontsize=12)
    ax1.set_ylabel('AoI (seconds)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.8)
    ax1.legend(fontsize=12, loc='best')

    # 装饰 Cost 图表
    ax2.set_title('System Operational Cost Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Request Sequence', fontsize=12)
    ax2.set_ylabel('Total Cost', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.8)
    ax2.legend(fontsize=12, loc='best')

    plt.tight_layout()
    plt.show()
