# utils/generate_real_datasets.py
import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])

from utils.data_generator import save_dataset
from utils.logger import logger
from utils.real_dataset_builder import build_real_dataset


def export_geo_to_csv(dataset, save_dir):
    """
    从数据集中剥离出所有节点的地理位置，并保存为 CSV 格式，
    """
    G = dataset['network'].graph
    geo_data = []

    # 遍历网络图中的所有节点，提取属性
    for node_id, data in G.nodes(data=True):
        node = data.get('node')
        # 检查节点是否具有经纬度属性
        if hasattr(node, 'lat') and hasattr(node, 'lon'):
            # 获取节点的真实类名 (EdgeNode, Sensor, UserNode)
            node_type = type(node).__name__
            geo_data.append({
                'Node_ID': node.id,
                'Type': node_type,
                'Latitude': node.lat,
                'Longitude': node.lon
            })

    # 转换为 DataFrame 并保存
    if geo_data:
        df = pd.DataFrame(geo_data)
        csv_path = os.path.join(save_dir, "dataset_real_geo_locations.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"节点的经纬度报表已单独导出至: {csv_path}")
    else:
        logger.warning("数据集中没有发现带有经纬度属性的节点！")


def generate_real_world_data():
    # 1. 配置文件路径 (请确保这些文件在你的电脑上路径正确)
    telecom_file = os.path.join(PROJECT_ROOT, "datasets", "telecom-shanghai-dataset", "data_10.1610.31.xlsx")
    alibaba_file = os.path.join(PROJECT_ROOT, "datasets", "cluster-trace-gpu-v2025", "disaggregated_DLRM_trace.csv")

    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

    logger.info(">>> 正在基于真实轨迹构建大规模训练集 (30 节点, 1000 请求)...")

    # 构建一个中等规模的真实数据集用于训练
    real_ds_train = build_real_dataset(
        telecom_path=telecom_file,
        alibaba_path=alibaba_file,
        num_edge_nodes=30,
        num_dts=15,
        num_sensors=20,
        total_requests=1000,
        seed=2026
    )

    save_path = os.path.join(PROJECT_ROOT, "data", "dataset_real_30.pkl")
    save_dataset(real_ds_train, save_path)
    logger.info(f"真实数据集已保存至: {save_path}")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    export_geo_to_csv(real_ds_train, data_dir)


if __name__ == "__main__":
    generate_real_world_data()
    # export_geo_to_csv()
