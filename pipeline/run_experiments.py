# run_experiments.py
import os
import time
import math
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environment.rl_env import DTEngineEnv
from utils.logger import logger
from utils.data_generator import load_dataset
# 导入你的 Agent 类
from algorithms.random_agent import RandomAgent
from algorithms.greedy_agent import GreedyAgent
from algorithms.dp_agent import DPAgent


# ==========================================
# 核心评估函数：支持计时和多维度指标提取
# ==========================================
def evaluate_agent_full(algo_name, agent, env, total_reqs):
    """
    运行算法并提取：成功率、平均AoI、平均Cost、总迁移、单步平均耗时
    """
    obs, _ = env.reset()
    done = False
    history = []
    latencies = []  # 记录每一步决策的耗时

    while not done:
        # 1. 测量决策耗时 (秒)
        start_time = time.perf_counter()
        action, _ = agent.predict(obs, deterministic=True)
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

        # 2. 执行环境步进
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history.append(info)

    # 3. 数据清洗与统计
    successful_reqs = [h for h in history if not math.isinf(h['aoi'])]
    success_count = len(successful_reqs)

    metrics = {
        'Algorithm': algo_name,
        'Success_Rate': (success_count / total_reqs) * 100,
        'Avg_AoI': sum(h['aoi'] for h in successful_reqs) / success_count if success_count > 0 else float('inf'),
        'Avg_Cost': sum(h['cost'] for h in successful_reqs) / success_count if success_count > 0 else float('inf'),
        'Total_Mig': sum(1 for h in successful_reqs if h.get('migrated', False)),
        'Avg_Latency_ms': np.mean(latencies) * 1000  # 转为毫秒
    }
    return metrics


# ==========================================
# 实验 1 & 2: 负载敏感性测试 (AoI & Cost)
# ==========================================
def run_exp_load_sensitivity(dataset):
    logger.info("\n>>> 正在进行实验 1&2: 负载敏感性测试...")
    load_levels = [10, 20, 30, 40, 50]  # 不同请求数量级
    results = []

    for load in load_levels:
        env = DTEngineEnv(
            network=dataset['network'], users=dataset['users'],
            task_chains=dataset['task_chains'], request_stream=dataset['request_stream'][:load],
            total_reqs=load, alpha=0.5, beta=0.5  # 使用平衡权重
        )

        # 注册 Agent
        agents = {
            "Random": RandomAgent(env),
            "Greedy": GreedyAgent(env),
            "DP Real": DPAgent(env),
            "PPO (Balanced)": PPO.load("../environment/models/pareto/ppo_a0.5_b0.5_debug", env=env)
        }

        for name, agent in agents.items():
            m = evaluate_agent_full(name, agent, env, load)
            m['Load'] = load
            results.append(m)

    pd.DataFrame(results).to_csv("results/exp_load_sensitivity.csv", index=False)
    logger.info("实验 1&2 完成，结果已保存。")


# ==========================================
# 实验 3: 可扩展性与执行时间
# ==========================================
def run_exp_scalability(dataset):
    logger.info("\n>>> 正在进行实验 3: 算法可扩展性测试...")
    # 注意：这里需要你有一个能生成不同节点规模的 dataset_builder
    # 假设我们测试节点规模 [10, 20, 30, 40, 50]
    node_scales = [10, 20, 30, 40, 50]
    results = []

    # 我们固定跑 50 个请求来测试时间即可
    TEST_LOAD = 50

    for n in node_scales:
        logger.info(f"  测试规模: {n} Nodes")
        # ⚠️ 未来注意：这里你应该调用 dataset_builder 动态生成包含 n 个节点的网络！
        # 比如：current_dataset = build_real_dataset(num_nodes=n, ...)
        current_dataset = dataset  # 目前暂用传进来的静态 dataset 演示

        env = DTEngineEnv(
            network=current_dataset['network'],
            users=current_dataset['users'],
            task_chains=current_dataset['task_chains'],
            request_stream=current_dataset['request_stream'][:TEST_LOAD],  # 【核心修复】：显式传入请求流！
            total_reqs=TEST_LOAD
        )

        agents = {"PPO (Balanced)": PPO.load("../environment/models/pareto/ppo_a0.5_b0.5_debug", env=env)}

        if n <= 30:  # 超过30个节点 DP 可能会极慢
            try:
                agents["DP Real"] = DPAgent(env)
            except Exception as e:
                logger.error(f"规模 {n} 下 DP 初始化失败: {e}")

        agents["Greedy"] = GreedyAgent(env)

        for name, agent in agents.items():
            m = evaluate_agent_full(name, agent, env, TEST_LOAD)
            m['Node_Scale'] = n
            results.append(m)

    pd.DataFrame(results).to_csv("results/exp_scalability.csv", index=False)
    logger.info("实验 3 完成。")


# ==========================================
# 实验 4: 帕累托权衡 (Trade-off)
# ==========================================
def run_exp_pareto_tradeoff(dataset):
    logger.info("\n>>> 正在进行实验 4: 帕累托权衡测试...")
    results = []

    # 固定的测试负载
    TEST_LOAD = 100
    env = DTEngineEnv(dataset['network'], dataset['users'], dataset['task_chains'], total_reqs=TEST_LOAD)

    # 1. 测试不同权重的 PPO
    weights = [(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]
    for a, b in weights:
        model_path = f"../environment/models/pareto/ppo_a{a}_b{b}_debug"
        if os.path.exists(model_path + ".zip"):
            agent = PPO.load(model_path, env=env)
            m = evaluate_agent_full(f"PPO_a{a}", agent, env, TEST_LOAD)
            m['Weight_Alpha'] = a
            results.append(m)

    # 2. 作为对比，测试不同权重的 Greedy
    for a, b in weights:
        env.alpha, env.beta = a, b  # 动态修改环境权重
        agent = GreedyAgent(env)
        m = evaluate_agent_full(f"Greedy_a{a}", agent, env, TEST_LOAD)
        m['Weight_Alpha'] = a
        results.append(m)

    pd.DataFrame(results).to_csv("results/exp_pareto_tradeoff.csv", index=False)
    logger.info("实验 4 完成。")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    base_dataset = load_dataset("../data/dataset_debug.pkl")

    run_exp_load_sensitivity(base_dataset)
    run_exp_scalability(base_dataset)
    run_exp_pareto_tradeoff(base_dataset)
    logger.info("\n恭喜！所有实验数据已全部采出，准备画图。")