# utils/alibaba_trace_loader.py
import pandas as pd
import numpy as np
import os
from utils.logger import logger


class AlibabaTraceLoader:
    def __init__(self, file_path: str, edge_scale_factor: float = 0.1):
        """
        初始化阿里数据集加载器
        :param file_path: CSV 文件路径
        :param edge_scale_factor: 边缘缩放因子。数据中心的任务太大，
                                  需要按比例缩小以适应边缘节点 (例如乘以 0.1)
        """
        self.file_path = file_path
        self.edge_scale_factor = edge_scale_factor
        self.df = None
        self._load_and_clean_data()

    def _load_and_clean_data(self):
        if not os.path.exists(self.file_path):
            logger.error(f"找不到数据集文件: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")

        logger.info(f"正在加载阿里集群数据集: {self.file_path} ...")
        # 仅读取我们需要的列，节省内存
        usecols = ['cpu_request', 'memory_request', 'creation_time', 'deletion_time']
        self.df = pd.read_csv(self.file_path, usecols=usecols)

        # 1. 剔除无效值 (NaN) 和请求为 0 的异常数据
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['cpu_request', 'memory_request'])
        self.df = self.df[(self.df['cpu_request'] > 0) & (self.df['memory_request'] > 0)]

        # 2. 计算任务的理论生命周期 (Duration = deletion - creation)
        # 注意：有些任务没有 deletion_time，需要设一个默认的短存活时间
        self.df['duration'] = self.df['deletion_time'] - self.df['creation_time']
        self.df['duration'] = self.df['duration'].fillna(10.0)  # 默认 10 秒
        self.df = self.df[self.df['duration'] > 0]

        # 3. 边缘化缩放 (Edge Scaling)
        # 将数据中心的庞大需求等比例缩放，映射为边缘计算级别的微服务需求
        self.df['edge_cpu'] = self.df['cpu_request']
        self.df['edge_mem'] = self.df['memory_request'] * self.edge_scale_factor

        logger.info(f"数据集清洗完成。有效记录数: {len(self.df)} / {initial_len}")

    def sample_tasks(self, n: int = 1) -> list:
        """
        从真实的经验分布中随机采样 n 个任务的资源需求
        :return: [{'workload': float, 'memory': float, 'duration': float}, ...]
        """
        if self.df is None or len(self.df) == 0:
            raise ValueError("数据集未正确加载或为空")

        # 从 DataFrame 中随机抽取 n 行
        sampled = self.df.sample(n=n, replace=True)

        tasks_info = []
        for _, row in sampled.iterrows():
            workload = row['edge_cpu']
            memory = row['edge_mem']
            duration = row['duration']

            # 为了防止出现极端微小的值，设一个下限
            tasks_info.append({
                'workload': max(0.5, round(workload, 2)),
                'memory': max(0.1, round(memory, 2)),
                'duration': round(duration, 2)
            })

        return tasks_info if n > 1 else tasks_info[0]


# ==========================================
# 简单的测试桩
# ==========================================
if __name__ == "__main__":
    # 假设你的 CSV 放在这个路径
    csv_path = "../datasets/cluster-trace-gpu-v2025/disaggregated_DLRM_trace.csv"

    loader = AlibabaTraceLoader(file_path=csv_path, edge_scale_factor=0.1)

    print("\n随机采样 3 个任务的特征:")
    samples = loader.sample_tasks(3)
    for i, s in enumerate(samples):
        print(f"Task {i + 1}: Workload={s['workload']}, Mem={s['memory']}GB, Duration={s['duration']}s")
