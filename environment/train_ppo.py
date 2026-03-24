# train_ppo.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from utils.logger import logger
from utils.visualization import plot_simulation_results

from environment.rl_env import DTEngineEnv
from main import setup_clean_environment  # 暂用建图
from utils.data_generator import load_dataset


def train_and_evaluate_ppo():
    logger.info("\n" + "=" * 50)
    logger.info("1：初始化 DRL 训练环境 (PPO)")
    logger.info("=" * 50)

    # 1. 准备物理网络环境
    # net, users, task_chains = setup_clean_environment()
    dataset = load_dataset("../data/dataset_small.pkl")
    net = dataset['network']
    users = dataset['users']
    task_chains = dataset['task_chains']


    # 训练时的请求序列可以长一点
    TRAIN_REQS_PER_EPISODE = 200

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
    dataset = load_dataset("../data/dataset_small.pkl")
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
    train_and_evaluate_ppo()
