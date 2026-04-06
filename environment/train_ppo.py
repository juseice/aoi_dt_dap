# train_ppo.py
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from utils.logger import logger
from utils.visualization import plot_simulation_results
from pathlib import Path
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing
from tqdm import tqdm

from environment.rl_env import DTEngineEnv
from main import setup_clean_environment  # 暂用建图
from utils.data_generator import load_dataset

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])

# ==========================================
# 训练过程数据记录器 (用于论文 6.4.1 画收敛图)
# ==========================================

class ConvergenceLoggerCallback(BaseCallback):
    def __init__(self, model_name, verbose=0):
        super().__init__(verbose)
        self.model_name = model_name
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        # 累加每一步的 reward
        self.current_reward += self.locals['rewards'][0]

        # 检查回合是否结束 (dones 是一个布尔数组，因为我们用了 vec_env)
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True

    def _on_training_end(self) -> None:
        # 训练结束后，将收敛数据落盘为 CSV
        log_dir = os.path.join(PROJECT_ROOT, "results", "training_logs")
        os.makedirs(log_dir, exist_ok=True)

        df = pd.DataFrame({
            'Episode': range(1, len(self.episode_rewards) + 1),
            'Cumulative_Reward': self.episode_rewards
        })
        csv_path = os.path.join(log_dir, f"{self.model_name}_convergence.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"模型 {self.model_name} 的收敛数据已保存至 {csv_path}")


def train_real_world_pareto():
    logger.info("开始基于真实数据分布训练帕累托模型矩阵...")

    dataset_path = os.path.join(PROJECT_ROOT, "data", "dataset_real_30.pkl")
    if not os.path.exists(dataset_path):
        logger.error("找不到真实数据集文件，请先运行 generate_real_datasets.py")
        return

    dataset = load_dataset(dataset_path)

    # 真实数据环境下，增加训练步数
    TOTAL_TIMESTEPS = 50000
    TRAIN_REQS_PER_EPISODE = 500

    # 帕累托权重组合
    pareto_weights = [(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]
    num_cpus = min(8, multiprocessing.cpu_count())
    logger.info(f"开启 CPU 多进程加速，同时运行 {num_cpus} 个平行仿真环境！")

    # 创建保存目录
    model_dir = os.path.join(PROJECT_ROOT, "environment", "models", "pareto_real")
    os.makedirs(model_dir, exist_ok=True)

    for alpha, beta in tqdm(pareto_weights, desc="总体模型训练进度", colour="green"):
        model_name = f"ppo_real_a{alpha}_b{beta}"
        logger.info(f"\n>>> 正在训练真实数据模型: {model_name} (α={alpha}, β={beta}) <<<")

        # 使用真实数据集的物理网络、用户、任务链
        env_maker = lambda: DTEngineEnv(
            network=dataset['network'],
            users=dataset['users'],
            task_chains=dataset['task_chains'],
            request_stream=None,  # 训练时使用随机抽样
            total_reqs=TRAIN_REQS_PER_EPISODE,
            seed=None,
            alpha=alpha,
            beta=beta
        )

        vec_env = make_vec_env(env_maker, n_envs=num_cpus, vec_env_cls=SubprocVecEnv)

        model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=3e-4)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

        # 保存模型
        save_path = os.path.join(model_dir, model_name)
        model.save(save_path)
        logger.info(f"模型已保存至: {save_path}.zip")

    logger.info("\n真实数据帕累托模型完毕！")


def train_pareto_models():
    logger.info("\n" + "=" * 50)
    logger.info("启动自动化 DRL 训练矩阵 (生成帕累托模型库)")
    logger.info("=" * 50)

    # 1. 加载数据集
    dataset_path = os.path.join(PROJECT_ROOT, "data", "dataset_large.pkl")
    dataset = load_dataset(dataset_path)

    net = dataset['network']
    users = dataset['users']
    task_chains = dataset['task_chains']

    # 训练参数
    TRAIN_REQS_PER_EPISODE = 500  # 让每回合足够长，让智能体吃尽苦头去学习
    TOTAL_TIMESTEPS = 80000       # 每个模型的总训练步数 (如果算力够，可设为 100000)

    # 2. 设定我们要探索的权重组合 (alpha: 成本权重, beta: AoI 权重)
    pareto_weights = [
        (0.9, 0.1), # 极端偏好：省钱 (Cost 优先)
        (0.7, 0.3),
        (0.5, 0.5), # 绝对平衡
        (0.3, 0.7),
        (0.1, 0.9)  # 极端偏好：新鲜度 (AoI 优先)
    ]

    pareto_weights = [
        (0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)
    ]

    model_dir = os.path.join(PROJECT_ROOT, "environment", "models", "pareto")
    os.makedirs(model_dir, exist_ok=True)

    # 3. 开始循环炼丹
    for alpha, beta in pareto_weights:
        model_name = f"ppo_a{alpha}_b{beta}_large"
        logger.info(f"\n>>> 正在训练模型: {model_name} (Cost权={alpha}, AoI权={beta}) <<<")

        # 动态实例化带有特定权重的 Env
        env_maker = lambda: DTEngineEnv(
            network=net,
            users=users,
            task_chains=task_chains,
            request_stream=None, # 训练时传入 None，让环境内部自己随机生成请求，增加样本多样性
            total_reqs=TRAIN_REQS_PER_EPISODE,
            seed=None,
            alpha=alpha,
            beta=beta
        )

        vec_env = make_vec_env(env_maker, n_envs=1)

        # 构建 PPO 模型
        tb_log = os.path.join(PROJECT_ROOT, "results", "tensorboard_logs", model_name)
        model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=3e-4, tensorboard_log=tb_log)

        callback = ConvergenceLoggerCallback(model_name=model_name)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

        # 保存训练好的大脑权重
        save_path = f"models/pareto/{model_name}"
        model.save(save_path)
        logger.info(f" 模型已保存至 {save_path}.zip")

    logger.info("\n 所有帕累托偏好模型训练完毕！")


def train_scalability_models():
    logger.info("\n" + "=" * 50)
    logger.info("启动自动化炼丹：针对不同网络规模训练专属 PPO 模型")
    logger.info("=" * 50)

    node_scales = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    TRAIN_REQS_PER_EPISODE = 500
    TOTAL_TIMESTEPS = 40000  # 规模越大，可能需要的步数越多，这里统一给 4万步

    os.makedirs("models/scalability", exist_ok=True)

    for n in node_scales:
        model_name = f"ppo_scale_{n}"
        logger.info(f"\n>>> 正在训练 {n} 节点规模的专属模型: {model_name} <<<")

        # 1. 加载对应规模的真实拓扑数据集
        dataset_path = os.path.join(PROJECT_ROOT, "data", f"dataset_{n}_nodes.pkl")
        if not os.path.exists(dataset_path):
            logger.error(f"找不到数据集 {dataset_path}，请先生成！")
            continue

        dataset = load_dataset(dataset_path)

        # 2. 实例化专属环境 (权衡参数固定为 0.5 即可)
        env_maker = lambda: DTEngineEnv(
            network=dataset['network'],
            users=dataset['users'],
            task_chains=dataset['task_chains'],
            request_stream=None,  # 训练时用随机请求流
            total_reqs=TRAIN_REQS_PER_EPISODE,
            seed=None,
            alpha=0.5, beta=0.5
        )
        vec_env = make_vec_env(env_maker, n_envs=1)

        # 3. 训练并保存
        model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=3e-4)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        save_path = f"models/scalability/{model_name}"
        model.save(save_path)
        logger.info(f"规模 {n} 的模型已保存至 {save_path}.zip")


def train_and_evaluate_ppo():
    logger.info("\n" + "=" * 50)
    logger.info("1：初始化 DRL 训练环境 (PPO)")
    logger.info("=" * 50)

    # 1. 准备物理网络环境
    # net, users, task_chains = setup_clean_environment()
    dataset = load_dataset("../data/dataset_debug.pkl")
    net = dataset['network']
    users = dataset['users']
    task_chains = dataset['task_chains']

    # 训练时的请求序列可以长一点
    TRAIN_REQS_PER_EPISODE = 500

    # 为了让 SB3 运行更稳，我们用一个 lambda 函数来实例化你的 Env
    env_maker = lambda: DTEngineEnv(
        network=net,
        users=users,
        task_chains=task_chains,
        total_reqs=TRAIN_REQS_PER_EPISODE,
        seed=None  # 训练时不需要固定种子，让它见识各种随机请求的组合！
    )

    # SB3 推荐的环境向量化包装器 (方便以后你想开多线程训练)
    vec_env = make_vec_env(env_maker, n_envs=1)

    # 2. 构建 PPO 神经网络模型
    # "MlpPolicy" 表示使用多层感知机 (全连接神经网络)
    # verbose=1 会在终端打印训练进度 (比如平均奖励、损失函数等)
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4,
                tensorboard_log="./ppo_dt_tensorboard/")

    logger.info("\n" + "=" * 50)
    logger.info("2：试错学习")
    logger.info("=" * 50)

    TOTAL_TIMESTEPS = 50000
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 保存训练好的大脑权重
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_dt_deployment")
    logger.info("模型已保存至 models/ppo_dt_deployment.zip")

    logger.info("\n" + "=" * 50)
    logger.info("3：评估训练好的模型")
    logger.info("=" * 50)

    # 评估时，我们需要用一个【固定 Seed】的干净环境，以保证公平性
    EVAL_REQS = 30
    # eval_net, eval_users, eval_task_chains = setup_clean_environment()
    dataset = load_dataset("../data/dataset_debug.pkl")
    eval_net = dataset['network']
    eval_users = dataset['users']
    eval_task_chains = dataset['task_chains']
    eval_env = DTEngineEnv(
        network=eval_net,
        users=eval_users,
        task_chains=eval_task_chains,
        total_reqs=EVAL_REQS,
        seed=2026  # 用固定的测试卷子
    )

    obs, info = eval_env.reset()
    history = []

    for step in range(EVAL_REQS):
        # 核心：让训练好的模型进行“预测”(给出最佳动作)
        # deterministic=True 表示取消探索，每次都严格选概率最大的最优动作
        action, _states = model.predict(obs, deterministic=True)

        # 将神经网络的指令下发给物理环境
        obs, reward, terminated, truncated, info = eval_env.step(action)

        if info['cost'] != float('inf'):
            logger.info(
                f"[请求 {info['req_id']}] RL 智能体部署 Node {action} | AoI={info['aoi']:.2f}, Cost={info['cost']:.2f}")
            history.append({
                'req': info['req_id'],
                'aoi': info['aoi'],
                'cost': info['cost'],
                'migrated': False  # 此处可视化简写，实际上如果需要可以通过对比动作得知
            })
        else:
            logger.error(f"[请求 {info['req_id']}] 部署崩溃！(模型没学好，选了爆满的节点)")

        if terminated or truncated:
            break

    logger.info("\n评估结束，正在生成 PPO 算法的表现图表...")
    # 用我们之前写的函数画出 PPO 的单次表现折线图
    plot_simulation_results(history)


if __name__ == "__main__":
    train_real_world_pareto()
    # train_pareto_models()
    # train_scalability_models()
    # train_and_evaluate_ppo()

