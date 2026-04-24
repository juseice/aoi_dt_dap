from collections import Counter
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_dataset(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"[错误] 找不到文件: {pkl_path}")
        return

    print(f"正在加载数据集: {pkl_path} ...")
    try:
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"[加载失败] 请确保读取环境中有 Request 等自定义类的定义。错误信息: {e}")
        return

    print("\n" + "=" * 50)
    print(" 1. 数据集宏观配置 (Config)")
    print("=" * 50)
    config = dataset.get('config', {})
    for k, v in config.items():
        print(f"  - {k:<15}: {v}")

    print("\n" + "=" * 50)
    print(" 2. 实体规模统计")
    print("=" * 50)
    print(f"  - 边缘节点总数 : {len(dataset.get('edge_nodes', []))}")
    print(f"  - 传感器总数   : {len(dataset.get('sensors', []))}")
    print(f"  - 用户总数     : {len(dataset.get('users', []))}")
    print(f"  - DT 服务总数  : {len(dataset.get('task_chains', []))}")

    print("\n" + "=" * 50)
    print(" 3. 动态请求流分析 (Request Stream)")
    print("=" * 50)
    requests = dataset.get('request_stream', [])
    print(f"  - 总请求数量   : {len(requests)}")

    if requests:
        # ==========================
        # A. 时序特征分析
        # ==========================
        try:
            start_time = requests[0].trigger_time
            end_time = requests[-1].trigger_time
            duration = end_time - start_time
            print(f"  - 首个请求时间 : {start_time:.2f} s")
            print(f"  - 末尾请求时间 : {end_time:.2f} s")
            print(f"  - 仿真时间跨度 : {duration:.2f} s")

            if duration > 0:
                actual_lambda = len(requests) / duration
                print(f"  - 实际平均到达率 (Lambda) : {actual_lambda:.2f} req/s")
        except AttributeError:
            print("  - [提示] 无法读取 trigger_time，跳过时间分析。")

        # ==========================
        # B. 业务负载分布分析 (验证 Zipf 长尾)
        # ==========================
        try:
            # 统计每个 DT 被请求的次数
            dt_counter = Counter([req.task_chain for req in requests])
            print("\n  - Top 5 最热门的 DT 服务 (业务流行度分布):")

            for i, (dt, count) in enumerate(dt_counter.most_common(5)):
                # 尝试获取 DT 的可读标识（如果有 id、name 属性则提取，否则直接打印对象）
                dt_name = getattr(dt, 'dt_id', getattr(dt, 'name', f"DT_Obj_{i}"))
                percentage = (count / len(requests)) * 100
                print(f"      * {dt_name:<10} : {count:^4} 次请求  ({percentage:>5.1f}%)")

        except Exception as e:
            print(f"  - [提示] 无法统计热门 DT 服务，错误: {e}")

        # ==========================
        # C. 活跃用户分析
        # ==========================
        try:
            user_counter = Counter([req.user for req in requests])
            print(f"\n  - 发起过请求的活跃用户数: {len(user_counter)} / {len(dataset.get('users', []))}")
        except Exception:
            pass

    # ==========================
    # 新增：边缘节点详情遍历
    # ==========================
    print("\n" + "=" * 50)
    print(" 4. 边缘节点详情 (Edge Nodes Data)")
    print("=" * 50)
    edge_nodes = dataset.get('edge_nodes', [])
    if edge_nodes:
        # 打印表头
        print(
            f"  {'Node ID':<10} | {'纬度 (Lat)':<12} | {'经度 (Lon)':<12} | {'算力 (Capacity)':<15} | {'内存 (Memory)':<12}")
        print("  " + "-" * 70)

        for i, node in enumerate(edge_nodes):
            # 安全提取属性，如果您的类中变量名不同，请在此处修改（如 'cpu_cores', 'ram' 等）
            n_id = getattr(node, 'node_id', getattr(node, 'id', f'EN_{i}'))
            lat = getattr(node, 'latitude', getattr(node, 'lat', 'N/A'))
            lon = getattr(node, 'longitude', getattr(node, 'lon', 'N/A'))
            cap = getattr(node, 'capacity', getattr(node, 'compute_power', 'N/A'))
            mem = getattr(node, 'memory', getattr(node, 'mem', 'N/A'))

            # 格式化数字输出，保持对齐
            lat_str = f"{lat:.6f}" if isinstance(lat, (int, float)) else str(lat)
            lon_str = f"{lon:.6f}" if isinstance(lon, (int, float)) else str(lon)
            cap_str = f"{cap:.2f}" if isinstance(cap, (int, float)) else str(cap)
            mem_str = f"{mem:.2f}" if isinstance(mem, (int, float)) else str(mem)

            print(f"  {str(n_id):<10} | {lat_str:<12} | {lon_str:<12} | {cap_str:<15} | {mem_str:<12}")
    else:
        print("  - [提示] 数据集中未找到边缘节点数据。")

    print("\n" + "=" * 50)
    print(" 分析完成！")
    print("=" * 50)


def load_and_verify_heterogeneity(pkl_path):
    """
    读取仿真数据集，提取基站与用户属性，并绘制物理帕累托分布图
    以验证边缘网络空间异构性与资源断崖假说。
    """
    if not os.path.exists(pkl_path):
        print(f"[错误] 找不到文件: {pkl_path}")
        return

    print(f">>> 正在加载数据集: {pkl_path} ...")
    try:
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"[加载失败] 确保在读取环境中有相关类的定义。错误: {e}")
        return

    # ==========================================
    # 1. 从自定义对象列表中提取数据构造 DataFrame
    # ==========================================
    print(">>> 正在从对象提取基站与用户特征...")

    edge_nodes = dataset.get('edge_nodes', [])
    users = dataset.get('users', [])

    if not edge_nodes or not users:
        print("[错误] 数据集中未找到 edge_nodes 或 users。")
        return

    # 提取基站属性 (请确保你的 EdgeNode 类包含这些属性，若名字不同请在此修改)
    # 例如：node.id, node.lat, node.lon, node.capacity
    node_data = []
    for i, node in enumerate(edge_nodes):
        node_id = getattr(node, 'node_id', getattr(node, 'id', f'EN_{i}'))
        lat = getattr(node, 'latitude', getattr(node, 'lat', 0.0))
        lon = getattr(node, 'longitude', getattr(node, 'lon', 0.0))
        # 寻找算力属性，可能是 capacity, compute, cpu_cycles 等
        capacity = getattr(node, 'capacity', getattr(node, 'compute_power', 1.0))
        node_data.append([node_id, lat, lon, capacity])

    nodes_df = pd.DataFrame(node_data, columns=['Node_ID', 'Latitude', 'Longitude', 'Capacity'])

    # 提取用户属性
    user_data = []
    for user in users:
        lat = getattr(user, 'latitude', getattr(user, 'lat', 0.0))
        lon = getattr(user, 'longitude', getattr(user, 'lon', 0.0))
        user_data.append([lat, lon])

    users_df = pd.DataFrame(user_data, columns=['Latitude', 'Longitude'])

    # ==========================================
    # 2. 计算空间传输代价 (基站到用户的平均距离)
    # ==========================================
    print(">>> 正在计算基站与用户群的拓扑距离矩阵...")
    node_coords = nodes_df[['Latitude', 'Longitude']].values
    user_coords = users_df[['Latitude', 'Longitude']].values

    # 计算欧式距离矩阵 (反映物理传输代价的基准)
    dist_matrix = cdist(node_coords, user_coords, metric='euclidean')

    # 每个基站到所有用户的平均距离
    nodes_df['Avg_Distance'] = dist_matrix.mean(axis=1)

    # 归一化处理以便于坐标轴同量级对比
    nodes_df['Norm_Capacity'] = nodes_df['Capacity'] / nodes_df['Capacity'].max()
    nodes_df['Norm_Distance'] = nodes_df['Avg_Distance'] / nodes_df['Avg_Distance'].max()

    # ==========================================
    # 3. 绘制基础设施的物理帕累托散点图
    # ==========================================
    print(">>> 正在生成空间异构性与资源断崖验证图...")
    plt.figure(figsize=(9, 7))

    # 根据归一化算力划分层级 (Tier)
    conditions = [
        (nodes_df['Norm_Capacity'] >= 0.7),
        (nodes_df['Norm_Capacity'] >= 0.5) & (nodes_df['Norm_Capacity'] < 0.7),
        (nodes_df['Norm_Capacity'] < 0.5)
    ]
    tiers = ['Tier 0 (核心大容量机房)', 'Tier 1 (汇聚节点)', 'Tier 2 (远端小基站)']
    nodes_df['Tier'] = np.select(conditions, tiers, default='Unknown')

    # 绘制带边缘颜色的高级散点图
    sns.scatterplot(
        data=nodes_df,
        x='Norm_Capacity',
        y='Norm_Distance',
        hue='Tier',
        palette=['#e74c3c', '#f1c40f', '#3498db'],
        s=180, alpha=0.85, edgecolor='black', linewidth=1
    )

    # 画出理论上的“理想线性权衡线”（即 Convex Front）
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='理论线性过渡区')

    # 圈出“非凸空白区”（即导致智能体震荡的物理黑洞）
    # plt.axvspan(0.3, 0.65, color='gray', alpha=0.15, label='物理资源断崖 (无理想过渡节点)')

    plt.title('真实边缘网络的物理分布验证', fontweight='bold', fontsize=16, pad=15)
    plt.xlabel('归一化计算资源容量 (Capacity: 越大越好)', fontweight='bold', fontsize=12)
    plt.ylabel('归一化平均用户物理距离 (传输代价: 越大越差)', fontweight='bold', fontsize=12)

    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)

    # ==========================================
    # 4. 保存与展示
    # ==========================================
    save_path = "plot_infrastructure_non_convexity.pdf"
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f">>> 验证完成！图片已保存至: {save_path}")
    plt.show()


def plot_workload_distribution(pkl_path):
    """
    绘制数字孪生服务的长尾需求分布与流行度特征图。
    验证真实业务场景中的齐普夫(Zipf)长尾效应。
    """
    if not os.path.exists(pkl_path):
        print(f"[错误] 找不到文件: {pkl_path}")
        return

    print(f">>> 正在加载数据集以绘制长尾分布: {pkl_path} ...")
    try:
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"[加载失败] 请确保读取环境中有相关类的定义。错误: {e}")
        return

    requests = dataset.get('request_stream', [])
    if not requests:
        print("[错误] 数据集中未找到 request_stream，无法绘制长尾分布。")
        return

    print(">>> 正在统计 DT 服务请求频次...")
    # 统计每个 DT 被请求的次数
    dt_counter = Counter([req.task_chain for req in requests])

    # 获取所有的请求数量并降序排列
    sorted_counts = sorted(dt_counter.values(), reverse=True)
    ranks = np.arange(1, len(sorted_counts) + 1)

    # 确保输出目录存在
    output_dir = "figure"
    os.makedirs(output_dir, exist_ok=True)

    print(">>> 正在生成长尾需求分布图...")
    plt.figure(figsize=(10, 6))

    # 使用柱状图展示各个 DT 服务的请求量
    sns.barplot(x=ranks, y=sorted_counts, color='#3498db', alpha=0.7, edgecolor='black')

    # 叠加折线图以突出长尾趋势
    plt.plot(ranks - 1, sorted_counts, color='#e74c3c', marker='o', linewidth=2, markersize=5, label='需求趋势线')

    plt.title('数字孪生服务的长尾需求分布与流行度特征', fontweight='bold', fontsize=16, pad=15)
    plt.xlabel('数字孪生服务排名 (按流行度降序)', fontweight='bold', fontsize=12)
    plt.ylabel('请求到达次数', fontweight='bold', fontsize=12)

    # 优化 X 轴刻度显示，防止标签过于拥挤
    if len(ranks) > 15:
        step = max(1, len(ranks) // 10)
        plt.xticks(ticks=np.arange(0, len(ranks), step=step),
                   labels=np.arange(1, len(ranks) + 1, step=step))

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    # 保存图片
    save_path = os.path.join(output_dir, "workload_distribution.pdf")
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f">>> 验证完成！长尾分布图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 替换为你实际想要分析的 pkl 文件路径
    # 例如：上一节生成的 "data/load_dataset_lambda_50.pkl"
    target_file = "../data/dataset_real_30.pkl"
    analyze_dataset(target_file)
    # load_and_verify_heterogeneity(target_file)
    plot_workload_distribution(target_file)
