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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        'Avg_Latency_ms': float(np.mean(latencies) * 1000)
    }
    return metrics


# ==========================================
# 实验 1 & 2: 负载敏感性测试 (AoI & Cost)
# ==========================================
def run_exp_load_sensitivity(dataset):
    logger.info("\n>>> 正在进行实验 1&2: 负载敏感性测试...")
    load_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 不同请求数量级
    results = []

    for load in load_levels:
        env = DTEngineEnv(
            network=dataset['network'], users=dataset['users'],
            task_chains=dataset['task_chains'], request_stream=dataset['request_stream'][:load],
            total_reqs=load, alpha=0.5, beta=0.5  # 使用平衡权重
        )

        ppo_path = os.path.join(PROJECT_ROOT, "environment", "models", "pareto", "ppo_a0.5_b0.5_large")

        # 注册 Agent
        agents = {
            "Random": RandomAgent(env),
            "Greedy": GreedyAgent(env),
            "DP Real": DPAgent(env),
            "PPO (Balanced)": PPO.load(ppo_path, env=env)
        }

        for name, agent in agents.items():
            m = evaluate_agent_full(name, agent, env, load)
            m['Load'] = load
            results.append(m)

    save_path = os.path.join(PROJECT_ROOT, "results", "exp_load_sensitivity.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    logger.info(f"实验 1&2 完成，结果已保存至 {save_path}。")


# ==========================================
# 实验 3: 可扩展性与执行时间
# ==========================================
def run_exp_scalability():
    logger.info("\n>>> 正在进行实验 3: 算法可扩展性测试...")
    node_scales = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    results = []

    # 我们固定跑 50 个请求来测试时间即可
    TEST_LOAD = 50

    for n in node_scales:
        logger.info(f"  正在加载规模: {n} Nodes 的物理网络...")

        dataset_path = os.path.join(PROJECT_ROOT, "data", f"dataset_{n}_nodes.pkl")
        current_dataset = load_dataset(dataset_path)

        env = DTEngineEnv(
            network=current_dataset['network'],
            users=current_dataset['users'],
            task_chains=current_dataset['task_chains'],
            request_stream=current_dataset['request_stream'][:TEST_LOAD],
            total_reqs=TEST_LOAD
        )

        agents = {"Greedy": GreedyAgent(env)}

        model_path = os.path.join(PROJECT_ROOT, "environment", "models", "scalability", f"ppo_scale_{n}.zip")
        if os.path.exists(model_path):
            agents["PPO"] = PPO.load(model_path[:-4], env=env)
        else:
            logger.warning(f"缺失 {n} 节点的 PPO 模型，跳过 PPO 测试。")

        dp_init_time = 0.0
        if n <= 50:  # 超过50个节点 DP 可能会极慢
            try:
                # 记录 DP 开始求解的时间
                dp_start = time.perf_counter()
                dp_agent = DPAgent(env)
                dp_end = time.perf_counter()

                # 记录总求解时间
                dp_init_time = dp_end - dp_start
                agents["DP Real"] = dp_agent
            except Exception as e:
                logger.error(f"规模 {n} 下 DP 初始化失败: {e}")

        for name, agent in agents.items():
            m = evaluate_agent_full(name, agent, env, TEST_LOAD)
            m['Node_Scale'] = n
            if "DP" in name:
                m['Avg_Latency_ms'] = (dp_init_time / TEST_LOAD) * 1000

            results.append(m)

    save_path = os.path.join(PROJECT_ROOT, "results", "exp_scalability.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    logger.info(f"实验 3 完成，结果已保存至 {save_path}。")


# ==========================================
# 实验 4: 帕累托权衡 (Trade-off)
# ==========================================
def run_exp_pareto_tradeoff(dataset):
    logger.info("\n>>> 正在进行实验 4: 帕累托权衡测试...")
    results = []

    TEST_LOAD = 100
    env = DTEngineEnv(dataset['network'], dataset['users'], dataset['task_chains'], total_reqs=TEST_LOAD)

    weights = [(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]

    # 1. 测试不同权重的 PPO
    logger.info("--- 开始测试 PPO 帕累托模型族 ---")
    ppo_success_count = 0
    for a, b in weights:
        env.alpha, env.beta = a, b

        # 基础路径 (不含后缀)
        model_base_path = os.path.join(PROJECT_ROOT, "environment", "models", "pareto", f"ppo_a{a}_b{b}_large")

        if os.path.exists(model_base_path + ".zip"):
            # logger.info(f"找到模型: ppo_a{a}_b{b}_large.zip，正在评估...")

            # PPO.load 传 base_path 或加了 .zip 的路径都可以，它内部能处理
            agent = PPO.load(model_base_path, env=env)

            m = evaluate_agent_full(f"PPO_a{a}", agent, env, TEST_LOAD)
            m['Weight_Alpha'] = a
            results.append(m)
            logger.info(f"   -> 评估完成 | AoI: {m['Avg_AoI']:.2f}, Cost: {m['Avg_Cost']:.2f}")
            ppo_success_count += 1
        else:
            # 报错信息里也加上 .zip，方便你核对
            logger.warning(f"找不到模型: {model_base_path}.zip！跳过。")

    # if ppo_success_count < 2:
    #     logger.error(f"⚠警告：只成功评估了 {ppo_success_count} 个 PPO 模型！少于 2 个点无法在图表中连成折线！")

    # 2. 作为对比，测试不同权重的 Greedy
    logger.info("--- 开始测试 Greedy 对比基准 ---")
    for a, b in weights:
        env.alpha, env.beta = a, b
        agent = GreedyAgent(env)
        m = evaluate_agent_full(f"Greedy_a{a}", agent, env, TEST_LOAD)
        m['Weight_Alpha'] = a
        results.append(m)
        logger.info(f"   -> Greedy(a={a}) 评估完成 | AoI: {m['Avg_AoI']:.2f}, Cost: {m['Avg_Cost']:.2f}")

    save_path = os.path.join(PROJECT_ROOT, "results", "exp_pareto_tradeoff.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    logger.info(f"实验 4 完成，结果已保存至 {save_path}。")


if __name__ == "__main__":
    # 确保根目录下有 results 文件夹
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    base_dataset_path = os.path.join(PROJECT_ROOT, "data", "dataset_large.pkl")
    base_dataset = load_dataset(base_dataset_path)

    run_exp_load_sensitivity(base_dataset)
    run_exp_scalability()
    run_exp_pareto_tradeoff(base_dataset)

    logger.info("\n恭喜！所有实验数据已全部采出，准备画图。")
