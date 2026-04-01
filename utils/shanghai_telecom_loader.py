# utils/shanghai_telecom_loader.py
import pandas as pd
import numpy as np
import os
import glob
import pickle
from geopy.distance import geodesic
from utils.logger import logger


class ShanghaiTelecomLoader:
    def __init__(self, data_dir: str, cache_file: str = "data/shanghai_bs_cache.pkl"):
        """
        初始化上海电信数据集加载器
        :param data_dir: 存放 12 个 .xlsx 文件的目录
        :param cache_file: 提取后的基站拓扑缓存文件，避免每次重新读取 Excel
        """
        self.data_dir = data_dir
        self.cache_file = cache_file
        self.base_stations = []  # 存储选定的基站坐标 [{'id': 'BS_0', 'lat': 31.2, 'lon': 121.4}, ...]
        self.distance_matrix = None  # 距离矩阵 (米)
        self.bandwidth_matrix = None  # 带宽矩阵 (Mbps)

    def build_topology(self, num_nodes: int = 20, bounding_box: dict = None):
        """
        构建物理拓扑：如果有缓存则读缓存，否则从 Excel 中提取。
        """
        if os.path.exists(self.cache_file):
            logger.info(f"从缓存加载边缘节点拓扑: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self.base_stations = data['base_stations']
                self.distance_matrix = data['distance_matrix']
                self.bandwidth_matrix = data['bandwidth_matrix']
            return

        logger.info("未找到缓存，开始从原始 Excel 文件解析基站拓扑...")
        self._extract_from_raw_data(num_nodes, bounding_box)
        self._calculate_matrices()
        self._save_cache()

    def _extract_from_raw_data(self, num_nodes: int, bounding_box: dict):
        # 设定上海市中心的核心区域 (经纬度范围)
        if bounding_box is None:
            bounding_box = {
                'min_lat': 31.15, 'max_lat': 31.30,
                'min_lon': 121.40, 'max_lon': 121.55
            }

        all_files = glob.glob(os.path.join(self.data_dir, "*.xlsx"))
        if not all_files:
            raise FileNotFoundError(f"目录 {self.data_dir} 下未找到 .xlsx 文件")

        unique_locations = set()
        file_to_read = all_files[0]
        logger.info(f"正在读取 {file_to_read} 提取基站坐标...")

        # 修正1：分别读取 latitude 和 longitude
        df = pd.read_excel(file_to_read, usecols=['latitude', 'longitude'])

        # 修正2：剔除那些经纬度为空 (NaN) 的无效行
        df = df.dropna(subset=['latitude', 'longitude'])

        # 遍历有效数据
        for _, row in df.iterrows():
            try:
                lat = float(row['latitude'])
                lon = float(row['longitude'])

                # 过滤：只保留在目标 Bounding Box 内的基站
                if (bounding_box['min_lon'] <= lon <= bounding_box['max_lon'] and
                        bounding_box['min_lat'] <= lat <= bounding_box['max_lat']):
                    # 使用 round 保留一定精度，将距离极近的坐标视为同一个基站
                    unique_locations.add((round(lon, 4), round(lat, 4)))
            except ValueError:
                continue

            # 提取够了就提前退出，加快速度 (多提取一些作为备选)
            if len(unique_locations) >= num_nodes * 5:
                break

        if len(unique_locations) < num_nodes:
            logger.warning(f"区域内仅找到 {len(unique_locations)} 个基站，不足设定的 {num_nodes} 个")
            num_nodes = len(unique_locations)

        # 随机挑选或取前 N 个作为我们的边缘节点
        selected_locs = list(unique_locations)[:num_nodes]
        for i, (lon, lat) in enumerate(selected_locs):
            self.base_stations.append({
                'id': f"EN_{i}",
                'lon': lon,
                'lat': lat
            })

        logger.info(f"成功提取 {num_nodes} 个基站作为边缘计算节点。")

    def _calculate_matrices(self):
        """根据经纬度计算物理距离矩阵，并映射为带宽矩阵"""
        n = len(self.base_stations)
        self.distance_matrix = np.zeros((n, n))
        self.bandwidth_matrix = np.zeros((n, n))

        # 简单的带宽映射规则：距离越远，可用带宽越小 (模拟光纤铺设成本和路由跳数)
        # 假设最大带宽 200Mbps，每公里衰减 20Mbps，保底 50Mbps
        MAX_BW = 200.0
        MIN_BW = 50.0
        DECAY_PER_KM = 20.0

        for i in range(n):
            for j in range(n):
                if i == j:
                    self.distance_matrix[i][j] = 0.0
                    self.bandwidth_matrix[i][j] = float('inf')  # 内部传输带宽无限大
                else:
                    coords_1 = (self.base_stations[i]['lat'], self.base_stations[i]['lon'])
                    coords_2 = (self.base_stations[j]['lat'], self.base_stations[j]['lon'])
                    # 计算地球表面两点距离（米）
                    dist_m = geodesic(coords_1, coords_2).meters
                    self.distance_matrix[i][j] = round(dist_m, 2)

                    # 计算带宽
                    dist_km = dist_m / 1000.0
                    bw = MAX_BW - (dist_km * DECAY_PER_KM)
                    self.bandwidth_matrix[i][j] = round(max(MIN_BW, bw), 2)

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        data = {
            'base_stations': self.base_stations,
            'distance_matrix': self.distance_matrix,
            'bandwidth_matrix': self.bandwidth_matrix
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info("拓扑数据已缓存。")

    def get_topology(self):
        return self.base_stations, self.distance_matrix, self.bandwidth_matrix


# 测试桩
if __name__ == "__main__":
    loader = ShanghaiTelecomLoader(data_dir="../datasets/telecom-shanghai-dataset")
    # 第一次运行会解析 Excel，后续秒开
    loader.build_topology(num_nodes=50)
    bs, dist, bw = loader.get_topology()
    print(f"节点 EN_0 和 EN_1 的距离为: {dist[0][1]} 米, 链路带宽为: {bw[0][1]} Mbps")
