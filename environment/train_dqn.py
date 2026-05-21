# train_dqn.py
import os
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from utils.logger import logger
from pathlib import Path
from tqdm import tqdm

from environment.rl_env import DTEngineEnv
from utils.data_generator import load_dataset

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])


class ConvergenceLoggerCallback(BaseCallback):
    def __init__(self, model_name, verbose=0):
        super().__init__(verbose)
        self.model_name = model_name
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True

    def _on_training_end(self) -> None:
        log_dir = os.path.join(PROJECT_ROOT, "results", "training_logs")
        os.makedirs(log_dir, exist_ok=True)
        df = pd.DataFrame({
            'Episode': range(1, len(self.episode_rewards) + 1),
            'Cumulative_Reward': self.episode_rewards
        })
        csv_path = os.path.join(log_dir, f"{self.model_name}_convergence.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"模型 {self.model_name} 的收敛数据已保存至 {csv_path}")


def _make_env_fn(network, users, task_chains, total_reqs, alpha, beta):
    """返回一个无参 lambda，避免 for 循环中的闭包延迟绑定问题。"""
    return lambda: DTEngineEnv(
        network=network,
        users=users,
        task_chains=task_chains,
        request_stream=None,
        total_reqs=total_reqs,
        seed=None,
        alpha=alpha,
        beta=beta
    )


def train_real_world_dqn():
    """在真实数据集上训练一组帕累托权重的 DQN 模型。"""
    logger.info("开始基于真实数据分布训练 DQN 模型...")

    dataset_path = os.path.join(PROJECT_ROOT, "data", "dataset_real_30.pkl")
    if not os.path.exists(dataset_path):
        logger.error("找不到真实数据集文件，请先运行 generate_real_datasets.py")
        return

    dataset = load_dataset(dataset_path)

    TOTAL_TIMESTEPS = 50000
    TRAIN_REQS_PER_EPISODE = 500
    pareto_weights = [(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]

    model_dir = os.path.join(PROJECT_ROOT, "environment", "models", "pareto_real")
    os.makedirs(model_dir, exist_ok=True)

    for alpha, beta in tqdm(pareto_weights, desc="DQN 模型训练进度", colour="blue"):
        model_name = f"dqn_real_a{alpha}_b{beta}"
        logger.info(f"\n>>> 正在训练: {model_name} (α={alpha}, β={beta}) <<<")

        vec_env = make_vec_env(
            _make_env_fn(dataset['network'], dataset['users'], dataset['task_chains'],
                         TRAIN_REQS_PER_EPISODE, alpha, beta),
            n_envs=1
        )

        model = DQN(
            "MlpPolicy", vec_env,
            learning_rate=1e-4,
            buffer_size=20000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.1,
            target_update_interval=1000,
            verbose=0,
            tensorboard_log=os.path.join(PROJECT_ROOT, "ppo_dt_tensorboard")
        )

        callback = ConvergenceLoggerCallback(model_name=model_name)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

        save_path = os.path.join(model_dir, model_name)
        model.save(save_path)
        logger.info(f"模型已保存至: {save_path}.zip")

    logger.info("\n真实数据 DQN 模型训练完毕！")


def train_and_evaluate_dqn(alpha=0.5, beta=0.5):
    """训练单个 DQN 模型（默认均衡权重 α=β=0.5）。"""
    logger.info("\n" + "=" * 50)
    logger.info(f"启动 DQN 训练 (α={alpha}, β={beta})")
    logger.info("=" * 50)

    dataset_path = os.path.join(PROJECT_ROOT, "data", "dataset_real_30.pkl")
    if not os.path.exists(dataset_path):
        logger.error("找不到数据集，请先运行 generate_real_datasets.py")
        return

    dataset = load_dataset(dataset_path)
    TRAIN_REQS_PER_EPISODE = 500
    TOTAL_TIMESTEPS = 20000
    model_name = f"dqn_real_a{alpha}_b{beta}"

    vec_env = make_vec_env(
        _make_env_fn(dataset['network'], dataset['users'], dataset['task_chains'],
                     TRAIN_REQS_PER_EPISODE, alpha, beta),
        n_envs=1
    )

    model = DQN(
        "MlpPolicy", vec_env,
        learning_rate=1e-4,
        buffer_size=20000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.1,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=os.path.join(PROJECT_ROOT, "ppo_dt_tensorboard")
    )

    callback = ConvergenceLoggerCallback(model_name=model_name)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    model_dir = os.path.join(PROJECT_ROOT, "environment", "models", "pareto_real")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, model_name)
    model.save(save_path)
    logger.info(f"DQN 模型已保存至 {save_path}.zip")


if __name__ == "__main__":
    train_real_world_dqn()
