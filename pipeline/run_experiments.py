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
from collections import defaultdict

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
def run_exp_load_sensitivity():
    """
    运行负载敏感性测试：
    遍历读取 data/ 目录下预先生成的 20 个不同到达率(Lambda)的数据集，
    评估不同算法在不同拥塞压力下的抗压能力。
    """
    logger.info("\n>>> 正在进行实验 1&2: 负载敏感性测试 (Arrival Rate Lambda)...")

    # 这里的 np.linspace 必须和生成数据集时的参数完全一致
    arrival_rates = np.linspace(5, 100, 20)
    results = []

    # PPO 模型路径 (确保模型已经训练好)
    ppo_path = os.path.join(PROJECT_ROOT, "environment", "models", "load_test")

    for rate in arrival_rates:
        rate_int = int(rate)
        filename = f"load_dataset_lambda_{rate_int}.pkl"
        filepath = os.path.join(PROJECT_ROOT, "data", filename)

        # 1. 检查并加载对应 Lambda 的数据集
        if not os.path.exists(filepath):
            logger.warning(f"[跳过] 找不到数据集文件: {filepath}")
            continue

        logger.info(f"==> 正在评测负载压力 Lambda = {rate_int} req/s")
        dataset = load_dataset(filepath)

        total_reqs = len(dataset['request_stream'])  # 应该是 500

        # 2. 初始化环境 (投入所有的 500 个请求)
        env = DTEngineEnv(
            network=dataset['network'],
            users=dataset['users'],
            task_chains=dataset['task_chains'],
            request_stream=dataset['request_stream'],
            total_reqs=total_reqs,
            alpha=0.5, beta=0.5  # 使用平衡权重进行测试
        )

        # 3. 注册所有基准 Agent
        agents = {
            "Random": RandomAgent(env),
            "Greedy": GreedyAgent(env),
            "DP Real": DPAgent(env),
            "PA-PPO": PPO.load(ppo_path, env=env)  # 统一改用论文里的名字
        }

        # 4. 依次评测并记录结果
        for name, agent in agents.items():
            try:
                # 假设 evaluate_agent_full 返回一个包含 Success_Rate, Avg_AoI, Avg_Cost 等的字典
                m = evaluate_agent_full(name, agent, env, total_reqs)

                # 重点：记录的自变量是 Lambda (到达率)，而不是请求总数！
                m['Lambda'] = rate_int
                results.append(m)
            except Exception as e:
                logger.error(f"算法 {name} 在 Lambda={rate_int} 时发生错误: {e}")

    # 5. 保存最终 CSV
    save_path = os.path.join(PROJECT_ROOT, "results", "exp_load_sensitivity.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(results).to_csv(save_path, index=False)

    logger.info(f"\n实验完成！负载抗压结果已成功保存至: {save_path}")


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


def run_real_world_pareto_analysis(dataset):
    logger.info("\n>>> 正在进行实验：真实场景帕累托前沿分析 <<<")
    results = []

    # 1. 统一测试负载 (使用真实请求流中的前 500 个)
    TEST_LOAD = 500
    # 确保环境使用真实数据集
    env = DTEngineEnv(
        network=dataset['network'],
        users=dataset['users'],
        task_chains=dataset['task_chains'],
        total_reqs=TEST_LOAD
    )

    # 2. 定义要测试的模型权重组合
    weights = [(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]
    # weights.append((0.45, 0.55), (0.4, 0.6), (0.35, 0.65))
    # weights = [(0.45, 0.55), (0.4, 0.6), (0.35, 0.65)]
    # weights = [(0.55, 0.45), (0.6, 0.4), (0.65, 0.35)]

    # 3. 逐一加载并评估 PPO 模型
    logger.info("--- 正在评估 PPO 模型族 ---")
    for a, b in weights:
        # 模型存储的基础路径
        model_path = os.path.join(PROJECT_ROOT, "environment", "models", "pareto_real", f"ppo_real_a{a}_b{b}")

        if os.path.exists(model_path + ".zip"):
            logger.info(f"正在评估权重 α={a} (偏向成本) 的模型...")
            # 动态调整环境权重以匹配模型偏好
            env.alpha, env.beta = a, b
            agent = PPO.load(model_path, env=env)

            # 使用之前的评估函数获取全量指标
            m = evaluate_agent_full(f"PPO_a{a}", agent, env, TEST_LOAD)
            m['Weight_Alpha'] = a
            m['Method'] = 'PA-PPO'
            results.append(m)
        else:
            logger.warning(f"跳过未找到的模型: {model_path}.zip")

    # 4. 加入 Greedy 算法作为对比基准 (同样测试不同权重)
    logger.info("--- 正在评估 Greedy 基准 ---")
    for a, b in weights:
        env.alpha, env.beta = a, b
        agent = GreedyAgent(env)
        m = evaluate_agent_full(f"Greedy_a{a}", agent, env, TEST_LOAD)
        m['Weight_Alpha'] = a
        m['Method'] = 'Greedy'
        results.append(m)

    # 5. 保存数据，为论文画图做准备
    save_path = os.path.join(PROJECT_ROOT, "results", "exp_real_pareto_results.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    logger.info(f"实验完成！结果已存至 {save_path}")

    return results


def run_baseline_comparison(dataset):
    logger.info("\n" + "=" * 50)
    logger.info(">>> 正在进行实验 2: 多算法基准对比 (全方位降维打击) <<<")
    logger.info("=" * 50)

    results = []
    TEST_LOAD = 1000  # 使用 1000 个真实请求作为压力测试

    # 这保证了无论哪个算法上场，用户的出现顺序、请求的类型都分毫不差，绝对公平。
    TEST_SEED = 2026

    # 初始化评估环境
    env = DTEngineEnv(
        network=dataset['network'],
        users=dataset['users'],
        task_chains=dataset['task_chains'],
        total_reqs=TEST_LOAD,
        seed=TEST_SEED
    )

    # ==========================================
    # 选手 1: PA-PPO (均衡型主将)
    # ==========================================
    model_path = os.path.join(PROJECT_ROOT, "environment", "models", "pareto_real", "ppo_real_a0.5_b0.5")
    if os.path.exists(model_path + ".zip"):
        logger.info("\n[1/4] 正在评估 PA-PPO (a=0.5, b=0.5)...")
        env.alpha, env.beta = 0.5, 0.5
        agent_ppo = PPO.load(model_path, env=env)

        m_ppo = evaluate_agent_full("PA-PPO", agent_ppo, env, TEST_LOAD)
        m_ppo['Algorithm'] = 'PA-PPO'
        results.append(m_ppo)
    else:
        logger.error(f"找不到 PPO 模型: {model_path}.zip")

    # ==========================================
    # 选手 2: Greedy-Cost
    # ==========================================
    # logger.info("\n[2/4] 正在评估 Greedy-Cost (追求极低成本)...")
    # # 权重设为 0.99，让环境在算分时极度放大 Cost 的影响
    # env.alpha, env.beta = 0.99, 0.01
    # # 确保重置环境，复原初始内存和相同的请求流
    # env.reset(seed=TEST_SEED)
    #
    # agent_greedy_cost = GreedyAgent(env)
    # m_gc = evaluate_agent_full("Greedy-Cost", agent_greedy_cost, env, TEST_LOAD)
    # m_gc['Algorithm'] = 'Greedy-Cost'
    # results.append(m_gc)

    # ==========================================
    # 选手 3: Greedy-AoI
    # ==========================================
    # logger.info("\n[3/4] 正在评估 Greedy-AoI (追求极低延迟)...")
    # # 权重设为 0.01，让环境在算分时极度放大 AoI 的影响
    # env.alpha, env.beta = 0.01, 0.99
    # env.reset(seed=TEST_SEED)
    #
    # agent_greedy_aoi = GreedyAgent(env)
    # m_ga = evaluate_agent_full("Greedy-AoI", agent_greedy_aoi, env, TEST_LOAD)
    # m_ga['Algorithm'] = 'Greedy-AoI'
    # results.append(m_ga)

    # ==========================================
    # 选手 2: Greedy
    # ==========================================
    logger.info("\n[2/4] 正在评估 Greedy...")
    # 权重设为 0.01，让环境在算分时极度放大 AoI 的影响
    env.alpha, env.beta = 0.5, 0.5
    env.reset(seed=TEST_SEED)

    agent_greedy_aoi = GreedyAgent(env)
    m_ga = evaluate_agent_full("Greedy", agent_greedy_aoi, env, TEST_LOAD)
    m_ga['Algorithm'] = 'Greedy'
    results.append(m_ga)


    logger.info("\n[4/4] 正在评估 Random Baseline (性能下界)...")
    env.alpha, env.beta = 0.5, 0.5  # 算分标准和 PPO 保持一致以便对比
    env.reset(seed=TEST_SEED)

    agent_random = RandomAgent(env)
    m_rand = evaluate_agent_full("Random", agent_random, env, TEST_LOAD)
    m_rand['Algorithm'] = 'Random'
    results.append(m_rand)

    save_path = os.path.join(PROJECT_ROOT, "results", "exp_baseline_comparison.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)

    logger.info("\n" + "=" * 50)
    logger.info(f"实验 2 完成！四种算法对比数据已存储至: {save_path}")
    logger.info("=" * 50)

    return results


def run_timeseries_stress_test(dataset):
    logger.info("\n" + "=" * 50)
    logger.info(">>> 正在进行实验 3: 时序抗压与动态响应分析 <<<")
    logger.info("=" * 50)

    # 使用较长的请求流观察波动
    TEST_LOAD = 1000
    TEST_SEED = 2026

    # 我们对比两个核心选手
    algorithms = ["PA-PPO", "Greedy"]
    time_series_data = []

    for algo in algorithms:
        logger.info(f"正在追踪 {algo} 的实时动态响应...")
        env = DTEngineEnv(
            network=dataset['network'],
            users=dataset['users'],
            task_chains=dataset['task_chains'],
            total_reqs=TEST_LOAD,
            seed=TEST_SEED
        )

        # 加载对应的智能体
        if algo == "PA-PPO":
            model_path = os.path.join(PROJECT_ROOT, "environment", "models", "pareto_real", "ppo_real_a0.5_b0.5")
            agent = PPO.load(model_path, env=env)
        else:
            agent = GreedyAgent(env)

        obs, info = env.reset()

        for step in range(TEST_LOAD):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # 核心：记录每一个时间步的瞬时指标
            # 我们记录：请求ID、瞬时AoI、瞬时成本、以及系统的总剩余内存（代表压力）
            total_remaining_mem = sum([node.memory for node in env.network.get_edge_nodes()])

            time_series_data.append({
                'Step': step,
                'Algorithm': algo,
                'Instant_AoI': info.get('aoi', 0) if info.get('cost') != float('inf') else 200,  # 崩溃赋予高惩罚值
                'Instant_Cost': info.get('cost', 0) if info.get('cost') != float('inf') else 50,
                'System_Free_Mem': total_remaining_mem,
                'Is_Success': 1 if info.get('cost') != float('inf') else 0
            })

            if terminated or truncated:
                break

    # 保存时序数据
    save_path = os.path.join(PROJECT_ROOT, "results", "exp_timeseries_stress.csv")
    pd.DataFrame(time_series_data).to_csv(save_path, index=False)
    logger.info(f"动态响应数据已保存至: {save_path}")
    return time_series_data


def generate_spatial_heatmap_data(dataset):
    logger.info("\n" + "=" * 50)
    logger.info(">>> 正在进行实验 4: 收集空间部署热力数据 <<<")

    env = DTEngineEnv(
        network=dataset['network'],
        users=dataset['users'],
        task_chains=dataset['task_chains'],
        total_reqs=500,  # 跑 500 个请求看分布
        seed=2026
    )

    # 找一个优秀的 PPO 模型来展示它的“排兵布阵”
    model_path = os.path.join(PROJECT_ROOT, "environment", "models", "pareto_real", "ppo_real_a0.9_b0.1")
    agent = PPO.load(model_path, env=env)

    obs, info = env.reset()

    # 用一个字典来记录每个基站 (EdgeNode) 接收了多少次 DT 部署
    deployment_counts = defaultdict(int)

    # 获取所有的 EdgeNode 列表，方便通过 action (索引) 找回真实的基站对象
    edge_nodes = [data.get('node') for n, data in env.network.graph.nodes(data=True)
                  if type(data.get('node')).__name__ == 'EdgeNode']

    for step in range(500):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # 只要没有崩溃 (Cost != inf)，就记录这次部署落在了哪个基站上
        if info.get('cost') != float('inf'):
            target_en = edge_nodes[action]
            deployment_counts[target_en.id] += 1

        if terminated or truncated:
            break

    # 将统计数据转化为 DataFrame 并附上经纬度
    heatmap_data = []
    for en in edge_nodes:
        count = deployment_counts.get(en.id, 0)
        heatmap_data.append({
            'Node_ID': en.id,
            'Latitude': en.lat,
            'Longitude': en.lon,
            'Load_Count': count
        })

    df = pd.DataFrame(heatmap_data)
    save_path = os.path.join(PROJECT_ROOT, "results", "exp_heatmap_data.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"热力分布数据已保存至: {save_path}")

    return df


def generate_spatial_heatmap_data_greedy(dataset):
    logger.info("\n" + "=" * 50)
    logger.info(">>> 正在进行实验 4.1: 收集空间部署热力数据 (Greedy Agent) <<<")

    # 1. 像评估 PPO 一样，初始化完全相同的强化学习环境
    env = DTEngineEnv(
        network=dataset['network'],
        users=dataset['users'],
        task_chains=dataset['task_chains'],
        total_reqs=500,  # 保持与 PPO 一致的 500 个请求
        seed=2026
    )

    # 2. 实例化你封装好的 GreedyAgent
    agent = GreedyAgent(env)

    obs, info = env.reset()

    # 用一个字典来记录每个基站 (EdgeNode) 接收了多少次 DT 部署
    deployment_counts = defaultdict(int)

    # 获取所有的 EdgeNode 列表，方便通过 action (索引) 找回真实的基站对象
    edge_nodes = [data.get('node') for n, data in env.network.graph.nodes(data=True)
                  if type(data.get('node')).__name__ == 'EdgeNode']

    # 3. 完美复用与 PPO 完全相同的交互循环
    for step in range(500):
        # 调用 GreedyAgent 的 predict 接口
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # 只要没有因为资源枯竭被 Drop (Cost != inf)，就记录有效部署
        if info.get('cost') != float('inf'):
            target_en = edge_nodes[action]
            deployment_counts[target_en.id] += 1

        if terminated or truncated:
            break

    # 4. 将统计数据转化为 DataFrame 并附上经纬度
    heatmap_data = []
    for en in edge_nodes:
        count = deployment_counts.get(en.id, 0)
        heatmap_data.append({
            'Node_ID': en.id,
            'Latitude': en.lat,
            'Longitude': en.lon,
            'Load_Count': count
        })

    df = pd.DataFrame(heatmap_data)

    # 确保 PROJECT_ROOT 在文件顶部已定义
    save_path = os.path.join(PROJECT_ROOT, "results", "exp_heatmap_data_greedy.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"贪心算法热力分布数据已保存至: {save_path}")

    return df


if __name__ == "__main__":
    # === 随机数据集 ===
    # 确保根目录下有 results 文件夹
    # os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    #
    # base_dataset_path = os.path.join(PROJECT_ROOT, "data", "dataset_large.pkl")
    # base_dataset = load_dataset(base_dataset_path)
    #
    run_exp_load_sensitivity()
    # run_exp_scalability()
    # run_exp_pareto_tradeoff(base_dataset)

    # === 真实数据集 ===
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    real_dataset_path = os.path.join(PROJECT_ROOT, "data", "dataset_real_30.pkl")
    real_dataset = load_dataset(real_dataset_path)

    # run_real_world_pareto_analysis(real_dataset)
    # run_baseline_comparison(real_dataset)
    # run_timeseries_stress_test(real_dataset)
    # generate_spatial_heatmap_data(real_dataset)
    # generate_spatial_heatmap_data_greedy(real_dataset)

    logger.info("\n恭喜！所有实验数据已全部采出，准备画图。")
