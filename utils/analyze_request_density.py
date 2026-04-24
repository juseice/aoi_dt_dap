from utils.data_generator import load_dataset
from utils.visualization import plot_macro_topology, plot_infrastructure_topology
from inspect_pkl import inspect_network_nodes
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_request_density(dataset):
    # 提取所有请求的触发时间
    times = [req.trigger_time for req in dataset['request_stream']]

    print(times[:5])
    plt.figure(figsize=(10, 4))
    # 使用直方图观察每 60 秒（1分钟）内的请求数量
    n, bins, patches = plt.hist(times, bins=60, color='#3498db', alpha=0.7, edgecolor='white')

    plt.title("Request Density Analysis (Shanghai Telecom Trace)")
    plt.xlabel("Relative Time (seconds)")
    plt.ylabel("Number of Requests")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    peak_density = max(n)
    avg_density = len(times) / (max(times) - min(times)) * 60

    print(f"统计报告：")
    print(f"-> 最大瞬时密度: {peak_density:.0f} 请求/分钟")
    print(f"-> 平均流量密度: {avg_density:.2f} 请求/分钟")
    plt.show()


if __name__ == "__main__":
    # 加载数据集
    dataset = load_dataset("../data/dataset_real_30.pkl")

    inspect_network_nodes(dataset)
    analyze_request_density(dataset)


