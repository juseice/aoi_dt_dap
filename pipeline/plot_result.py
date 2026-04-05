# pipeline/plot_results.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 定位项目根目录
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 全局统一配色与样式字典
STYLE_MAP = {
    'Random': {'color': '#f44336', 'marker': 'x', 'linestyle': '--'},
    'Greedy': {'color': '#2196f3', 'marker': 'o', 'linestyle': '-'},
    'DP Ideal': {'color': '#4caf50', 'marker': '^', 'linestyle': '--'},
    'DP Real': {'color': '#4caf50', 'marker': 's', 'linestyle': '-'},
    'PPO (Balanced)': {'color': '#FF9800', 'marker': 'd', 'linestyle': '-'},
    'PPO': {'color': '#FF9800', 'marker': 'd', 'linestyle': '-'}  # 兼容不同命名
}


def get_style(algo_name):
    # 模糊匹配样式
    for key, style in STYLE_MAP.items():
        if key in algo_name:
            return style
    return {'color': 'gray', 'marker': '.', 'linestyle': '-'}


def plot_load_sensitivity():
    """图 1：负载敏感性测试 (AoI, Cost, Success Rate)"""
    csv_path = os.path.join(RESULTS_DIR, "exp_load_sensitivity.csv")
    if not os.path.exists(csv_path):
        print(f"[跳过] 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
    ax_aoi, ax_cost, ax_succ = axes

    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo].sort_values(by='Load')
        x = algo_data['Load']
        style = get_style(algo)

        ax_aoi.plot(x, algo_data['Avg_AoI'], label=algo, **style, linewidth=2, markersize=7)
        ax_cost.plot(x, algo_data['Avg_Cost'], label=algo, **style, linewidth=2, markersize=7)
        ax_succ.plot(x, algo_data['Success_Rate'], label=algo, **style, linewidth=2, markersize=7)

    ax_aoi.set(title='(a) Average AoI vs Load', xlabel='Total Requests', ylabel='AoI (s)')
    ax_cost.set(title='(b) Average Cost vs Load', xlabel='Total Requests', ylabel='Cost')
    ax_succ.set(title='(c) Success Rate vs Load', xlabel='Total Requests', ylabel='Success Rate (%)')
    ax_succ.set_ylim(-5, 105)

    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.7)

    handles, labels = ax_aoi.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels), frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(RESULTS_DIR, "fig_load_sensitivity.png"), bbox_inches='tight')
    plt.show()


def plot_scalability():
    """图 2：可扩展性与执行时间测试"""
    csv_path = os.path.join(RESULTS_DIR, "exp_scalability.csv")
    if not os.path.exists(csv_path):
        print(f"[跳过] 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    fig, ax_time = plt.subplots(figsize=(8, 6), dpi=100)

    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo].sort_values(by='Node_Scale')
        x = algo_data['Node_Scale']
        style = get_style(algo)

        # 绘制执行时间 (使用对数坐标 Y 轴)
        ax_time.plot(x, algo_data['Avg_Latency_ms'], label=algo, **style, linewidth=2.5, markersize=8)

    ax_time.set_title('Algorithm Execution Time vs Network Scale', fontsize=14, fontweight='bold')
    ax_time.set_xlabel('Number of Edge Nodes (N)', fontsize=12)
    ax_time.set_ylabel('Average Decision Latency (ms) [Log Scale]', fontsize=12)

    # 开启对数坐标！这是展示 DP 指数爆炸的关键
    ax_time.set_yscale('log')
    ax_time.grid(True, which="both", linestyle=':', alpha=0.7)
    ax_time.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_scalability.png"), bbox_inches='tight')
    plt.show()


def plot_pareto():
    """图 3：帕累托权衡 (Trade-off)"""
    csv_path = os.path.join(RESULTS_DIR, "exp_pareto_tradeoff.csv")
    if not os.path.exists(csv_path):
        print(f"[跳过] 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # 提取 PPO 和 Greedy 的数据
    ppo_df = df[df['Algorithm'].str.contains("PPO")].sort_values(by='Avg_AoI')
    greedy_df = df[df['Algorithm'].str.contains("Greedy")].sort_values(by='Avg_AoI')

    # 绘制帕累托前沿曲线
    if not ppo_df.empty:
        ax.plot(ppo_df['Avg_AoI'], ppo_df['Avg_Cost'], marker='d', color='#FF9800',
                linestyle='-', linewidth=2, markersize=9, label='PPO Pareto Front')
        # 给 PPO 的点打上 alpha 标签
        for _, row in ppo_df.iterrows():
            ax.annotate(f"α={row['Weight_Alpha']}", (row['Avg_AoI'], row['Avg_Cost']),
                        textcoords="offset points", xytext=(10, 5), ha='left', fontsize=9)

    if not greedy_df.empty:
        ax.plot(greedy_df['Avg_AoI'], greedy_df['Avg_Cost'], marker='o', color='#2196f3',
                linestyle='--', linewidth=2, markersize=8, label='Greedy Trade-off')

    ax.set_title('Cost vs AoI Pareto Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average AoI (seconds) [Lower is Better]', fontsize=12)
    ax.set_ylabel('Average Operational Cost [Lower is Better]', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_pareto.png"), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("正在生成实验图表...")
    plot_load_sensitivity()
    plot_scalability()
    plot_pareto()
    print("图表已全部生成并保存在 results/ 目录下！")
