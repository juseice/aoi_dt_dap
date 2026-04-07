# main_pipeline.py
import os
from stable_baselines3 import PPO
from environment.rl_env import DTEngineEnv
from utils.data_generator import load_dataset
from utils.analyzer import save_simulation_results, generate_summary_report
from utils.visualization import plot_comparative_results
from utils.logger import logger

# 导入我们封装好的各类 Agent
from algorithms.random_agent import RandomAgent
from algorithms.greedy_agent import GreedyAgent
from algorithms.dp_agent import DPAgent


def evaluate_agent(algo_name, agent, env, seed):
    """
    统一的评估函数：任何算法来到这里，都必须遵守 Gym 的规则！
    """
    logger.info(f"\n========== 开始评测算法: {algo_name} ==========")
    history = []

    # 1. 重置环境，保证每次考卷完全一样
    obs, _ = env.reset(seed=seed)
    done = False

    # 2. 标准 MDP 交互循环
    while not done:
        # Agent 唯一的作用就是根据当前观测，给出一个动作索引
        # deterministic=True 保证 PPO 在测试时不乱探索
        action, _ = agent.predict(obs, deterministic=True)

        # 环境负责执行动作、推演物理时间、计算真实 AoI 和 Cost
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 记录环境算出来的权威数据
        history.append({
            'req': info['req_id'],
            'aoi': info['aoi'],
            'cost': info['cost'],
            'migrated': info.get('migrated', False)
        })

        logger.info(
            f"[{algo_name}] 请求 {info['req_id']} -> 动作 Node {action} | AoI={info['aoi']:.2f}, Cost={info['cost']:.2f}")

    return history


def main():
    logger.info("正在加载测试数据集...")
    dataset = load_dataset("data/dataset_debug.pkl")
    COMMON_SEED = dataset['config']['seed']
    TOTAL_REQUESTS = len(dataset['request_stream'])

    # 1. 实例化唯一的环境 (Single Source of Truth)
    # 所有算法都将在同一个沙盒里运行
    env = DTEngineEnv(
        network=dataset['network'],
        users=dataset['users'],
        task_chains=dataset['task_chains'],
        request_stream=dataset['request_stream'],
        total_reqs=TOTAL_REQUESTS,
        seed=COMMON_SEED
    )

    # 2. 像插拔 U 盘一样，注册你要跑的算法
    agents = {
        "Random Baseline": RandomAgent(env),
        "Greedy Best": GreedyAgent(env),
        "DP Real": DPAgent(env)  # 将 DP 封装为 Agent
    }

    # 如果有训练好的 PPO 模型，加载它
    ppo_model_path = "environment/models/ppo_dt_deployment.zip"
    if os.path.exists(ppo_model_path):
        agents["PPO (DRL)"] = PPO.load(ppo_model_path, env=env)
    else:
        logger.warning(f"未找到 PPO 模型 {ppo_model_path}，跳过 PPO 测试。")

    # 3. 核心流水线：遍历并评估所有算法
    all_histories = {}

    logger.info("初始化 DP 智能体...")
    try:
        dp_agent = DPAgent(env)
        # 路线 1：DP Ideal (Oracle) 存入 all_histories
        all_histories["DP Ideal"] = dp_agent.get_oracle_history()

        # 路线 2：DP Real 存入 agents 接受毒打
        agents["DP Real"] = dp_agent
    except Exception as e:
        logger.error(f"DP Agent 初始化失败: {e}")

    for algo_name, agent in agents.items():
        history = evaluate_agent(algo_name, agent, env, COMMON_SEED)
        all_histories[algo_name] = history

    # 4. 数据落盘与可视化 (和你之前一样)
    save_simulation_results(all_histories, filename="results/latest_simulation.json")
    generate_summary_report(all_histories, TOTAL_REQUESTS)
    plot_comparative_results(all_histories)


if __name__ == "__main__":
    main()
