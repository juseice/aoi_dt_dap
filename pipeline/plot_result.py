# pipeline/plot_results.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 定位项目根目录
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 全局统一配色与样式字典
# sns.set_theme(style="whitegrid", palette="muted")
STYLE_MAP = {
    'Random': {'color': '#f44336', 'marker': 'x', 'linestyle': '--'},
    'Greedy': {'color': '#2196f3', 'marker': 'o', 'linestyle': '-'},
    'DP Ideal': {'color': '#4caf50', 'marker': '^', 'linestyle': '--'},
    'DP Real': {'color': '#4caf50', 'marker': 's', 'linestyle': '-'},
    'PPO (Balanced)': {'color': '#FF9800', 'marker': 'd', 'linestyle': '-'},
    'PPO': {'color': '#FF9800', 'marker': 'd', 'linestyle': '-'}  # 兼容不同命名
}
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams.update({
#     'font.size': 8,          # 全局默认基础字体大小 (原默认是10，可改为 8 或 9)
#     'axes.titlesize': 10,     # 子图标题字体大小 (例如 (a) Request Success Rate)
#     'axes.labelsize': 9,     # X/Y 轴说明文字字体大小
#     'xtick.labelsize': 6,     # X 轴刻度数字大小
#     'ytick.labelsize': 6,     # Y 轴刻度数字大小
#     'legend.fontsize': 8,    # 图例字体大小
#     'figure.titlesize': 12,   # 整个大图的总标题字体大小
#     'figure.autolayout': False # 开启自动布局防裁剪 (非常关键！)
# })


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
    localize_label = {
        "Random": "随机基线算法",
        "Greedy": "贪心算法",
        "DP Ideal": "基于动态规划的最优部署算法（理想）",
        "DP Real": "基于动态规划的最优部署算法",
        "PPO (DRL)": "基于近端策略优化的边缘数字孪生信息年龄与成本优化算法"
    }

    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo].sort_values(by='Load')
        x = algo_data['Load']
        style = get_style(algo)
        algo_data['Avg_AoI'] = algo_data['Avg_AoI'] * 1000

        ax_aoi.plot(x, algo_data['Avg_AoI'], label=localize_label[algo], **style, linewidth=2, markersize=7)
        ax_cost.plot(x, algo_data['Avg_Cost'], label=localize_label[algo], **style, linewidth=2, markersize=7)
        ax_succ.plot(x, algo_data['Success_Rate'], label=localize_label[algo], **style, linewidth=2, markersize=7)

    ax_aoi.set(title='(a) 平均信息年龄 vs 负载', xlabel='总计请求', ylabel='信息年龄 (毫秒)')
    ax_cost.set(title='(b) 平均成本 vs 负载', xlabel='总计请求', ylabel='成本')
    ax_succ.set(title='(c) 成功率 vs 负载', xlabel='总计请求', ylabel='成功率 (%)')
    ax_succ.set_ylim(-5, 105)

    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.7)

    handles, labels = ax_aoi.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels), frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(RESULTS_DIR, "fig_load_sensitivity.pdf"), bbox_inches='tight', format="pdf")
    plt.show()


def plot_scalability():
    """图 2：可扩展性与执行时间测试"""
    csv_path = os.path.join(RESULTS_DIR, "exp_scalability.csv")
    if not os.path.exists(csv_path):
        print(f"[跳过] 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    fig, ax_time = plt.subplots(figsize=(8, 6), dpi=100)
    localize_label = {
        "Random": "随机基线算法",
        "Greedy": "贪心算法",
        "DP Ideal": "基于动态规划的最优部署算法（理想）",
        "DP Real": "基于动态规划的最优部署算法",
        "PPO (DRL)": "基于近端策略优化的边缘数字孪生信息年龄与成本优化算法"
    }

    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo].sort_values(by='Node_Scale')
        x = algo_data['Node_Scale']
        style = get_style(algo)

        # 绘制执行时间 (使用对数坐标 Y 轴)
        ax_time.plot(x, algo_data['Avg_Latency_ms'], label=localize_label[algo], **style, linewidth=2.5, markersize=8)

    ax_time.set_title('算法决策时间与网络尺度的关系', fontweight='bold')
    ax_time.set_xlabel('边缘节点数量')
    ax_time.set_ylabel('平均决策延迟 (毫秒) [Log Scale]')

    # 开启对数坐标！这是展示 DP 指数爆炸的关键
    ax_time.set_yscale('log')
    ax_time.grid(True, which="both", linestyle=':', alpha=0.7)
    ax_time.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_scalability.pdf"), bbox_inches='tight', format="pdf")
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
                        textcoords="offset points", xytext=(10, 5), ha='left')

    if not greedy_df.empty:
        ax.plot(greedy_df['Avg_AoI'], greedy_df['Avg_Cost'], marker='o', color='#2196f3',
                linestyle='--', linewidth=2, markersize=8, label='Greedy Trade-off')

    ax.set_title('Cost vs AoI Pareto Trade-off', fontweight='bold')
    ax.set_xlabel('平均信息年龄 (seconds) [Lower is Better]')
    ax.set_ylabel('Average Operational Cost [Lower is Better]')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_pareto.pdf"), bbox_inches='tight', format="pdf")
    plt.show()


def plot_weight_sensitivity(algo_keyword="PPO"):
    """
    动态计算归一化并绘制权重敏感度折线图
    :param csv_path: 实验数据 CSV 的文件路径
    :param algo_keyword: 要提取的算法关键字，例如 "Greedy" 或 "PPO"
    """
    csv_path = os.path.join(RESULTS_DIR, "exp_real_pareto_results.csv")
    if not os.path.exists(csv_path):
        print(f"[跳过] 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 提取 PPO 和 Greedy 的数据
    ppo_df = df[df['Algorithm'].str.contains("PPO")].sort_values(by='Avg_AoI')
    greedy_df = df[df['Algorithm'].str.contains("Greedy")].sort_values(by='Avg_AoI')

    if algo_keyword == "Greedy":
        df_algo = greedy_df
    else:
        df_algo = ppo_df

    # 按照 Alpha 权重从小到大排序，保证折线不打结
    df_algo = df_algo.sort_values(by='Weight_Alpha').reset_index(drop=True)

    # 2. 动态进行 Min-Max 归一化 (核心逻辑)
    aoi_min, aoi_max = df_algo['Avg_AoI'].min(), df_algo['Avg_AoI'].max()
    cost_min, cost_max = df_algo['Avg_Cost'].min(), df_algo['Avg_Cost'].max()

    # 防止分母为 0 (例如所有测试结果完全一致)
    aoi_range = aoi_max - aoi_min if aoi_max != aoi_min else 1.0
    cost_range = cost_max - cost_min if cost_max != cost_min else 1.0

    df_algo['Norm_AoI'] = (df_algo['Avg_AoI'] - aoi_min) / aoi_range
    df_algo['Norm_Cost'] = (df_algo['Avg_Cost'] - cost_min) / cost_range
    df_algo['Norm_Score'] = (df_algo['Norm_AoI'] + df_algo['Norm_Cost']) / 2

    # 3. 开始作图
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 画归一化 AoI 的折线 (蓝色)
    ax.plot(df_algo['Weight_Alpha'], df_algo['Norm_AoI'], marker='o', color='#3498db',
            linestyle='-', linewidth=2.5, markersize=8, label='归一化信息年龄')

    # 画归一化 Cost 的折线 (红色)
    ax.plot(df_algo['Weight_Alpha'], df_algo['Norm_Cost'], marker='s', color='#e74c3c',
            linestyle='--', linewidth=2.5, markersize=8, label='归一化成本')

    ax.plot(df_algo['Weight_Alpha'], df_algo['Norm_Score'], marker='d', color='#FF9800',
            linestyle='--', linewidth=2.5, markersize=8, label='归一化综合代价')

    # 4. 寻找并标注 Trade-off 交叉点 (Sweet Spot)
    # 逻辑：寻找两条归一化曲线数值差距最小的那个点
    diff = np.abs(df_algo['Norm_AoI'] - df_algo['Norm_Cost'])
    best_idx = diff.idxmin()
    best_alpha = df_algo.loc[best_idx, 'Weight_Alpha']
    best_val = (df_algo.loc[best_idx, 'Norm_AoI'] + df_algo.loc[best_idx, 'Norm_Cost']) / 2.0

    # 画一个高亮圆圈和文字提示
    # ax.plot(best_alpha, best_val, 'o', ms=16, mec='#27ae60', mfc='none', mew=2.5)
    # ax.annotate('Optimal Trade-off',
    #             xy=(best_alpha, best_val),
    #             xytext=(best_alpha - 0.25, best_val + 0.15),
    #             arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=1.5, headwidth=8),
    #             fontweight='bold', color='#27ae60', fontsize=11)

    # 5. 美化图表
    ax.set_title(f'基于近端策略优化的边缘数字孪生信息年龄与成本优化算法权重分析', pad=15, fontweight='bold', fontsize=14)
    ax.set_xlabel(r'成本偏好权重 ($\alpha$)', fontsize=12)
    ax.set_ylabel('归一化指标 [0.0 - 1.0]', fontsize=12)

    # 设置坐标轴边界，留出视觉空间
    ax.set_xlim(df_algo['Weight_Alpha'].min() - 0.05, df_algo['Weight_Alpha'].max() + 0.05)
    ax.set_ylim(-0.05, 1.05)

    ax.legend(loc='center right', frameon=True, shadow=True, fontsize=11)

    # 6. 自动保存为 PDF
    # 假设图片保存在 CSV 同级目录下
    save_dir = os.path.dirname(csv_path)
    save_filename = f"plot_weight_sensitivity_{algo_keyword.lower()}.pdf"
    save_path = os.path.join(save_dir, save_filename)

    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    print(f"[{algo_keyword}] 权重敏感度图已成功保存至: {save_path}")
    plt.show()


# ==========================================
# 战役一：帕累托前沿折线散点图 (Pareto Front)
# ==========================================
def plot_pareto_front():
    csv_path = os.path.join(PROJECT_ROOT, "results", "exp_real_pareto_results.csv")
    if not os.path.exists(csv_path):
        print(f"找不到数据文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 6), dpi=300)
    # 2. 动态进行 Min-Max 归一化 (核心逻辑)
    # aoi_min, aoi_max = df_algo['Avg_AoI'].min(), df_algo['Avg_AoI'].max()
    # cost_min, cost_max = df_algo['Avg_Cost'].min(), df_algo['Avg_Cost'].max()
    #
    # # 防止分母为 0 (例如所有测试结果完全一致)
    # aoi_range = aoi_max - aoi_min if aoi_max != aoi_min else 1.0
    # cost_range = cost_max - cost_min if cost_max != cost_min else 1.0
    #
    # df_ppo['Norm_AoI'] = (df_ppo['Avg_AoI'] - aoi_min) / aoi_range
    # df_ppo['Norm_Cost'] = (df_ppo['Avg_Cost'] - cost_min) / cost_range

    # 按照成本从小到大排序，保证折线不打结
    df_ppo = df[df['Method'] == 'PA-PPO'].sort_values('Avg_Cost')
    df_greedy = df[df['Method'] == 'Greedy'].sort_values('Avg_Cost')

    # 绘制折线和散点
    plt.plot(df_ppo['Avg_Cost'], df_ppo['Avg_AoI'], marker='o', markersize=10,
             linewidth=2.5, color='#e74c3c', label='PA-PPO (Ours)')
    plt.plot(df_greedy['Avg_Cost'], df_greedy['Avg_AoI'], marker='s', markersize=8,
             linewidth=2, color='#95a5a6', linestyle='--', label='Greedy Baseline')

    # 为 PPO 的点添加权重标注 (Alpha 值)
    for i, row in df_ppo.iterrows():
        plt.annotate(f"α={row['Weight_Alpha']}",
                     (row['Avg_Cost'], row['Avg_AoI']),
                     textcoords="offset points", xytext=(10, 10), ha='left')

    plt.title('Pareto Front: System Cost vs. Information Freshness (AoI)', pad=15, fontweight='bold')
    plt.xlabel('Average System Cost (Lower is better)')
    plt.ylabel('平均信息年龄 (Lower is better)')
    plt.legend(loc='upper right')

    # 越靠近左下角性能越好，画个箭头提示
    # plt.annotate('Better Performance', xy=(0.05, 0.05), xycoords='axes fraction',
    #              xytext=(0.3, 0.2), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    #              fontweight='bold', color='#2c3e50')

    save_path = os.path.join(PROJECT_ROOT, "results", "plot_pareto_front.pdf")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    print(f"帕累托前沿图已保存至: {save_path}")
    plt.show()


# ==========================================
# 战役二：基准算法全方位对比 (Bar Charts 1x3)
# ==========================================
def plot_baseline_comparison():
    csv_path = os.path.join(PROJECT_ROOT, "results", "exp_baseline_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"找不到数据文件: {csv_path}")
        return

    localize_label_plot = {
        "Random": "随机基线\n(Random)",
        "Greedy": "贪心算法\n(Greedy)",
        "DP Ideal": "最优部署\n(理想边界)",
        "DP Real": "最优部署\n(DP-ODA)",
        "PA-PPO": "PACE\n(本文算法)"
    }
    df = pd.read_csv(csv_path)
    df['Algorithm_CN'] = df['Algorithm'].map(localize_label_plot)

    # 创建 1x3 的并排子图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 自定义颜色，突出 PPO
    colors = ['#e74c3c' if algo == 'PA-PPO' else '#3498db' for algo in df['Algorithm']]
    df['Avg_AoI'] = df['Avg_AoI'] * 1000

    # 1. 成功率对比 (越高越好)
    sns.barplot(x='Algorithm_CN', y='Success_Rate', data=df, ax=axes[0], palette=colors)
    axes[0].set_title('(a) 请求成功率 ↑', fontweight='bold')
    axes[0].set_ylabel('成功率 (%)')
    axes[0].set_ylim(0, 100)

    # 2. 平均 AoI 对比 (越低越好)
    sns.barplot(x='Algorithm_CN', y='Avg_AoI', data=df, ax=axes[1], palette=colors)
    axes[1].set_title('(b) 平均信息年龄 ↓', fontweight='bold')
    axes[1].set_ylabel('信息年龄 (毫秒)')

    # 3. 平均成本对比 (越低越好)
    sns.barplot(x='Algorithm_CN', y='Avg_Cost', data=df, ax=axes[2], palette=colors)
    axes[2].set_title('(c) 平均成本 ↓', fontweight='bold')
    axes[2].set_ylabel('成本')

    # 美化 X 轴标签
    for ax in axes:
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('不同算法的性能对比', fontweight='bold')

    save_path = os.path.join(PROJECT_ROOT, "results", "plot_baseline_comparison.pdf")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', format="pdf")
    print(f"基准对比图已保存至: {save_path}")
    plt.show()


# ==========================================
# 战役三：时序动态响应与抗压分析 (Smoothed Line Plot)
# ==========================================
def plot_timeseries_analysis():
    csv_path = os.path.join(PROJECT_ROOT, "results", "exp_timeseries_stress.csv")
    if not os.path.exists(csv_path):
        print(f"找不到数据文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6), dpi=300)

    # 原始数据是 step 级别的，会有剧烈的毛刺。
    # 论文中为了看清趋势，通常使用滑动平均 (Rolling Mean) 进行平滑处理。
    window_size = 60
    df['Instant_AoI'] = df['Instant_AoI'] * 1000

    df_ppo = df[df['Algorithm'] == 'PA-PPO'].copy()
    df_greedy = df[df['Algorithm'] == 'Greedy'].copy()

    df_ppo['Smoothed_AoI'] = df_ppo['Instant_AoI'].rolling(window=window_size, min_periods=1).mean()
    df_greedy['Smoothed_AoI'] = df_greedy['Instant_AoI'].rolling(window=window_size, min_periods=1).mean()

    # 绘制平滑后的 AoI 曲线
    plt.plot(df_ppo['Step'], df_ppo['Smoothed_AoI'], label='PACE',
             linewidth=2.5, color='#e74c3c')
    plt.plot(df_greedy['Step'], df_greedy['Smoothed_AoI'], label='Greedy',
             linewidth=2, color='#34495e', linestyle='--')

    # 添加半透明的置信带 (展示原始数据的波动范围)
    plt.fill_between(df_ppo['Step'],
                     df_ppo['Smoothed_AoI'] - df_ppo['Instant_AoI'].rolling(window_size).std(),
                     df_ppo['Smoothed_AoI'] + df_ppo['Instant_AoI'].rolling(window_size).std(),
                     color='#e74c3c', alpha=0.15)

    plt.fill_between(df_greedy['Step'],
                     df_greedy['Smoothed_AoI'] - df_greedy['Instant_AoI'].rolling(window_size).std(),
                     df_greedy['Smoothed_AoI'] + df_greedy['Instant_AoI'].rolling(window_size).std(),
                     color='#34495e', alpha=0.1)

    # 标注流量突发区 (假设在 300-500 步之间发生拥塞，可根据真实数据调整阴影位置)
    # plt.axvspan(300, 500, color='yellow', alpha=0.15, label='High Traffic Burst Area')

    plt.title('真实世界流量影响下的动态信息年龄响应', pad=15, fontweight='bold')
    plt.xlabel('请求序列 (每分钟)')
    plt.ylabel('瞬时信息年龄 （毫秒）')

    # 将图例放在外侧防遮挡
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    save_path = os.path.join(PROJECT_ROOT, "results", "plot_timeseries_stress.pdf")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', format="pdf")
    print(f"时序动态响应图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("正在生成实验图表...")
    # === 随机数据 ===
    plot_load_sensitivity()
    plot_scalability()
    # plot_pareto()

    # === 真实数据 ===
    # plot_pareto_front()
    plot_baseline_comparison()
    plot_timeseries_analysis()

    # === 补充实验 ===
    plot_weight_sensitivity("PPO")

    print("图表已全部生成并保存在 results/ 目录下！")
