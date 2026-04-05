import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_load_sensitivity(csv_path="pipeline/results/exp_load_sensitivity.csv"):
    """
    读取负载敏感性 CSV 数据，并绘制出精美的学术级折线图
    """
    if not os.path.exists(csv_path):
        print(f"找不到文件 {csv_path}，请先运行 run_experiments.py")
        return

    # 1. 读取数据
    df = pd.read_csv(csv_path)

    # 2. 全局样式与配色字典 (保持与之前的高度一致)
    style_map = {
        'Random': {'color': '#f44336', 'marker': 'x', 'linestyle': '--'},  # 红色虚线
        'Greedy': {'color': '#2196f3', 'marker': 'o', 'linestyle': '-'},  # 蓝色实线
        'DP Ideal': {'color': '#4caf50', 'marker': '^', 'linestyle': '--'},  # 绿色虚线
        'DP Real': {'color': '#4caf50', 'marker': 's', 'linestyle': '-'},  # 绿色实线
        'PPO': {'color': '#FF9800', 'marker': 'd', 'linestyle': '-'}  # 橙色实线(菱形)
    }

    # 获取所有出现的算法名单，并按我们期望的顺序排列
    desired_order = ['DP Ideal', 'Random', 'Greedy', 'DP Real', 'PPO']
    algorithms = [alg for alg in desired_order if alg in df['Algorithm'].unique()]

    # 3. 创建一块 1x3 的大画布 (一行三个子图)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)
    ax_aoi, ax_cost, ax_succ = axes

    # 4. 遍历算法画线
    for algo in algorithms:
        # 提取该算法的数据，并按 Load (X轴) 排序，防止线乱飞
        algo_data = df[df['Algorithm'] == algo].sort_values(by='Load')

        x = algo_data['Load']
        style = style_map.get(algo, {'color': 'gray', 'marker': '.'})

        # --- 图 A: AoI vs Load ---
        ax_aoi.plot(x, algo_data['Avg_AoI'], label=algo,
                    color=style['color'], marker=style['marker'],
                    linestyle=style.get('linestyle', '-'), linewidth=2.5, markersize=8)

        # --- 图 B: Cost vs Load ---
        ax_cost.plot(x, algo_data['Avg_Cost'], label=algo,
                     color=style['color'], marker=style['marker'],
                     linestyle=style.get('linestyle', '-'), linewidth=2.5, markersize=8)

        # --- 图 C: Success Rate vs Load ---
        ax_succ.plot(x, algo_data['Success_Rate'], label=algo,
                     color=style['color'], marker=style['marker'],
                     linestyle=style.get('linestyle', '-'), linewidth=2.5, markersize=8)

    # ==========================================
    # 5. 图表精细化修饰 (学术论文标准)
    # ==========================================

    # 装饰图 A: AoI
    ax_aoi.set_title('(a) Average AoI vs. Request Load', fontsize=14, fontweight='bold')
    ax_aoi.set_xlabel('Request Load (Total Requests)', fontsize=12)
    ax_aoi.set_ylabel('Average AoI (seconds)', fontsize=12)
    ax_aoi.grid(True, linestyle=':', alpha=0.7)

    # 装饰图 B: Cost
    ax_cost.set_title('(b) Average Cost vs. Request Load', fontsize=14, fontweight='bold')
    ax_cost.set_xlabel('Request Load (Total Requests)', fontsize=12)
    ax_cost.set_ylabel('Average Operational Cost', fontsize=12)
    ax_cost.grid(True, linestyle=':', alpha=0.7)

    # 装饰图 C: Success Rate
    ax_succ.set_title('(c) Success Rate vs. Request Load', fontsize=14, fontweight='bold')
    ax_succ.set_xlabel('Request Load (Total Requests)', fontsize=12)
    ax_succ.set_ylabel('Success Rate (%)', fontsize=12)
    ax_succ.set_ylim(0, 105)  # 成功率最高100%，留点顶部空间
    ax_succ.grid(True, linestyle=':', alpha=0.7)

    # 添加唯一的全局图例 (放在顶部或右侧，避免遮挡数据)
    handles, labels = ax_aoi.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=len(algorithms), fontsize=12, frameon=False)

    plt.tight_layout()
    # 调整整体的顶部边距，给图例留出空间
    plt.subplots_adjust(top=0.85)

    # 保存为高清 PDF 和 PNG 供论文使用
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/fig_load_sensitivity.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("results/fig_load_sensitivity.png", format='png', bbox_inches='tight')

    print("画图完成！图表已保存至 results/fig_load_sensitivity.pdf 和 .png")
    plt.show()


if __name__ == "__main__":
    plot_load_sensitivity()
