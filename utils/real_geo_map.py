import pandas as pd
import folium
import branca.colormap as cm
import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "dataset_real_geo_locations.csv")

# 三张图的独立保存路径
HTML_MACRO_PATH = os.path.join(DATA_DIR, "interactive_macro_map.html")
HTML_INFRA_PATH = os.path.join(DATA_DIR, "interactive_infra_map.html")
HTML_CORE_PATH = os.path.join(DATA_DIR, "interactive_core_map.html")


def _render_folium_map(df, save_path):
    """内部通用渲染函数：负责实际的 Folium 地图绘制逻辑"""
    if df.empty:
        print(f"数据为空，跳过保存: {save_path}")
        return

    # 1. 动态计算当前数据的地图中心
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()

    # 2. 初始化地图对象 (使用极简浅色底图，凸显业务点)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')

    # 样式配置字典
    style_config = {
        'EdgeNode': {'color': 'blue', 'icon': 'cloud', 'icon_color': 'white'},
        'Sensor': {'color': 'green', 'icon': 'bullseye', 'icon_color': 'white'},
        'UserNode': {'color': 'red', 'icon': 'user', 'icon_color': 'white'}
    }

    # 3. 添加节点标记
    for _, row in df.iterrows():
        node_type = row['Type']
        config = style_config.get(node_type, {'color': 'gray', 'icon': 'info-sign'})

        icon = folium.Icon(
            color=config['color'],
            icon=config['icon'],
            icon_color=config['icon_color'],
            prefix='fa'
        )

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"<b>ID:</b> {row['Node_ID']}<br><b>Type:</b> {node_type}",
            tooltip=row['Node_ID'],
            icon=icon
        ).add_to(m)

    m.save(save_path)
    print(f"地图已保存至: {save_path}")


# ==========================================
# 1. 宏观全景图 (包含所有节点)
# ==========================================
def plot_interactive_macro_map(csv_path, save_path):
    print("\n>>> 正在生成: [宏观全景地图] (包含基站、传感器、用户)...")
    try:
        df = pd.read_csv(csv_path)
        _render_folium_map(df, save_path)
    except Exception as e:
        print(f"读取 CSV 文件失败: {e}")


# ==========================================
# 2. 基础设施图 (屏蔽用户)
# ==========================================
def plot_interactive_infra_map(csv_path, save_path):
    print("\n>>> 正在生成: [基础设施地图] (仅含基站和传感器)...")
    try:
        df = pd.read_csv(csv_path)
        # 核心过滤逻辑：仅保留 EdgeNode 和 Sensor
        df_infra = df[df['Type'].isin(['EdgeNode', 'Sensor'])].copy()
        _render_folium_map(df_infra, save_path)
    except Exception as e:
        print(f"读取 CSV 文件失败: {e}")


# ==========================================
# 3. 核心骨干网图 (仅保留基站)
# ==========================================
def plot_interactive_core_map(csv_path, save_path):
    print("\n>>> 正在生成: [核心骨干网地图] (仅含基站)...")
    try:
        df = pd.read_csv(csv_path)
        # 核心过滤逻辑：仅保留 EdgeNode
        df_core = df[df['Type'] == 'EdgeNode'].copy()
        _render_folium_map(df_core, save_path)
    except Exception as e:
        print(f"读取 CSV 文件失败: {e}")


def plot_deployment_bubble_map():
    csv_path = os.path.join(PROJECT_ROOT, "results", "exp_heatmap_data.csv")
    if not os.path.exists(csv_path):
        print("找不到热力数据 CSV，请先在 run_experiment.py 中运行 generate_spatial_heatmap_data！")
        return

    df = pd.read_csv(csv_path)

    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()

    # 使用深色底图 (CartoDB dark_matter) 能让热力气泡展现出极强的科技感和对比度
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB dark_matter')

    max_load = df['Load_Count'].max()

    # 定义丰富的连续渐变色带 (Colormap)
    colormap = cm.LinearColormap(
        colors=['#1e88e5', '#00e5ff', '#00e676', '#ffea00', '#ff1744'],
        vmin=0,           # 最小值
        vmax=max_load     # 最大值
    )
    # 为图例设置标题
    colormap.caption = 'DT Deployment Load Count'

    for _, row in df.iterrows():
        load = row['Load_Count']

        # 如果这个基站 0 负载，我们画一个很小的灰色点作为物理基站的背景参考
        if load == 0:
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=3,
                color='white',
                weight=1,
                fill=True,
                fill_color='gray',
                fill_opacity=0.3,
                popup=f"ID: {row['Node_ID']}<br>Load: 0"
            ).add_to(m)
        else:
            dynamic_color = colormap(load)

            # 半径根据负载量动态计算 (基础大小 + 比例放大)
            radius_size = 5 + (load / max_load) * 20

            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=radius_size,
                color=dynamic_color, # 使用自动计算出的渐变色
                weight=1,            # 边框极细
                fill=True,
                fill_color=dynamic_color,
                fill_opacity=0.6,    # 半透明叠加效果
                popup=f"<b>ID:</b> {row['Node_ID']}<br><b>Deployments:</b> {load}",
                tooltip=f"Load: {load}"
            ).add_to(m)

    # 将图例添加到地图上
    m.add_child(colormap)
    # ==========================================
    # 强制将图例生成的 svg 文本填充色 (fill) 改为白色，并加粗一点增加可读性
    # ==========================================
    css_injection = """
        <style>
            svg text { 
                fill: white !important; 
                font-weight: 500 !important;
                font-size: 13px !important;
            }
        </style>
        """
    m.get_root().header.add_child(folium.Element(css_injection))

    save_path = os.path.join(PROJECT_ROOT, "results", "interactive_deployment_heatmap.html")
    m.save(save_path)
    print(f"空间部署气泡图已生成: {save_path}")


if __name__ == "__main__":
    # if not os.path.exists(CSV_PATH):
    #     print("找不到 CSV 文件，请先运行数据导出脚本生成地理位置数据！")
    # else:
    #     # 一次性生成三张结构图
    #     plot_interactive_macro_map(CSV_PATH, HTML_MACRO_PATH)
    #     plot_interactive_infra_map(CSV_PATH, HTML_INFRA_PATH)
    #     plot_interactive_core_map(CSV_PATH, HTML_CORE_PATH)
    #     print("\n所有维度的交互式地图生成完毕！请在浏览器中打开对应 HTML 查看。")

    # 气泡图
    plot_deployment_bubble_map()

