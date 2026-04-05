import os
# 请确保从你的工程中正确导入这两个函数
from utils.data_generator import generate_synthetic_dataset, save_dataset
from utils.logger import logger


def generate_scalability_datasets():
    """
    批量生成用于可扩展性测试的随机数据集
    """
    # 确保 data 文件夹存在
    os.makedirs("data", exist_ok=True)
    # ==========================================
    # 2. 可扩展性测试集矩阵
    # ==========================================
    logger.info("\n>>> 开始生成可扩展性压测数据集矩阵...")

    node_scales = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for n in node_scales:
        # 按比例缩放其他实体，保证网络密度和业务压力相对恒定
        # 你可以根据实际论文需求微调这些缩放系数
        num_sensors = max(4, n // 2)  # 传感器数量为节点数的一半
        num_users = max(5, n // 2)  # 用户数量为节点数的一半
        num_dt_services = max(5, n // 2)  # DT 服务种类

        # 请求总数也可以随着节点数扩大，这里我们给个固定倍数，比如每个规模跑 200 个请求用于测时
        total_requests = 200

        logger.info(f"正在生成 {n} 节点拓扑 | 传感器:{num_sensors} | 用户:{num_users} | DT种类:{num_dt_services} ...")

        ds = generate_synthetic_dataset(
            num_edge_nodes=n,
            num_sensors=num_sensors,
            num_users=num_users,
            num_dt_services=num_dt_services,
            total_requests=total_requests,
            seed=2026  # 使用固定种子，保证同一规模下的拓扑结构可复现
        )

        filename = f"data/dataset_{n}_nodes.pkl"
        save_dataset(ds, filename)
        logger.info(f"{n} 节点数据集已保存至 {filename}")

    logger.info("\n所有测试集生成完毕！")


if __name__ == "__main__":
    generate_scalability_datasets()
