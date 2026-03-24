# build_datasets.py
from utils.data_generator import generate_synthetic_dataset, save_dataset

# 1. 小规模测试集 (方便快速 Debug)
ds_small = generate_synthetic_dataset(num_edge_nodes=5, num_sensors=2, num_users=3, total_requests=50, seed=10)
save_dataset(ds_small, "data/dataset_small.pkl")

# 2. 大规模压测集 (用于放在论文里的主要实验，考察算法的扩展性 Scalability)
ds_large = generate_synthetic_dataset(num_edge_nodes=50, num_sensors=20, num_users=10, num_dt_services=30, total_requests=2000, seed=42)
save_dataset(ds_large, "data/dataset_large.pkl")
