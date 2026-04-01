# scripts/run_experiments.py
import pandas as pd
import numpy as np
import os
from utils.real_dataset_builder import build_real_dataset
from environment.rl_env import DTEngineEnv
from algorithms.greedy import GreedyAgent
from algorithms.static import StaticAgent
from algorithms.random_agent import RandomAgent
# 假设你封装了 PPO 训练和测试的类
from algorithms.ppo_agent import PPOAgent


def run_load_sensitivity_experiment(seeds=[42, 100, 2024]):
    """实验二：负载敏感性测试 (AoI vs 负载)"""
    arrival_rates = [0.2, 0.4, 0.6, 0.8, 1.0]
    algorithms = ['PPO', 'Greedy', 'Static', 'Random']
    results = []

    print("=== 开始运行负载敏感性实验 ===")

    for rate in arrival_rates:
        for seed in seeds:
            print(f"\n[配置] 负载率: {rate}, 随机种子: {seed}")

            # 1. 生成统一的测试数据集 (确保所有算法面对的请求流完全一致)
            dataset = build_real_dataset(
                telecom_path="data/telecom.xlsx",
                alibaba_path="data/alibaba.csv",
                num_edge_nodes=20,
                arrival_rate=rate,  # 控制负载
                seed=seed
            )

            env = DTEngineEnv(**dataset)

            # 2. 循环测试各个基准算法
            for algo_name in algorithms:
                env.reset(seed=seed)

                if algo_name == 'PPO':
                    # 提示：这里应该加载已经提前训练好的 PPO 模型权重，而不是从头训练
                    agent = PPOAgent(env, load_path="models/ppo_best.zip")
                elif algo_name == 'Greedy':
                    agent = GreedyAgent(env)
                elif algo_name == 'Static':
                    agent = StaticAgent(env)
                elif algo_name == 'Random':
                    agent = RandomAgent(env)

                # 运行评估
                total_aoi = 0
                total_cost = 0
                migrations = 0

                obs, _ = env.reset()
                done = False
                while not done:
                    action = agent.predict(obs)  # 各个算法输出自己的动作
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    total_aoi += info['aoi']
                    total_cost += info['cost']
                    if info['migrated']:
                        migrations += 1

                avg_aoi = total_aoi / dataset['config']['total_requests']

                # 3. 记录结果
                results.append({
                    'Algorithm': algo_name,
                    'ArrivalRate': rate,
                    'Seed': seed,
                    'AvgAoI': avg_aoi,
                    'TotalCost': total_cost,
                    'Migrations': migrations
                })
                print(f"  -> {algo_name} 完成 | 平均AoI: {avg_aoi:.2f} | 迁移次数: {migrations}")

    # 4. 保存为 CSV，供后续画图使用
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/load_sensitivity.csv", index=False)
    print("=== 实验数据已保存至 results/load_sensitivity.csv ===")


if __name__ == "__main__":
    # 执行流水线
    run_load_sensitivity_experiment()
