# [Main] 推进物理时间 (例如 t = 2.5s)，产生了一个新请求
#   │
#   ├──> [Main] 呼叫 [Solver]: "当前时间 2.5s，任务链来了，给我个最优节点列表。"
#   │      │
#   │      ├──> [Solver] 遍历备选 node_list 组合...
#   │      │      ├──> 问 [Simulator]: "evaluate_step([Node1,Node2], t=2.5s) 指标如何？"
#   │      │      └──> ...
#   │      │
#   │      ├──> [Solver] 算了一遍 Score，发现某组合分数最低（最好）。
#   │      └──> [Solver] 返回给 [Main]: "选 [Node2, Node3]，预计得分为 X。"
#   │
#   ├──> [Main] 收到决策，命令 [Simulator]: "commit_step([Node2,Node3], t=2.5s)！"
#   │      └──> [Simulator] 真实更新内部时间轴和部署位置。
#   │
#   └──> [Main] 打印日志，继续循环，等待下一个请求...

from core import Network, Node, Edge, EdgeNode, Sensor, UserNode, Task, TaskChain
from environment import Simulator
from latency import estimate_aoi
from objective import compute_cost
from optimization.brute_solver import select_best_node
from optimization.random_solver import select_random_node
from optimization.dp_solver import solve_dp_offline
from utils.logger import logger
from utils.visualization import plot_network_topology, plot_simulation_results, plot_comparative_results
from utils.data_generator import load_dataset
from utils.analyzer import save_simulation_results, generate_summary_report
from stable_baselines3 import PPO, DQN
from environment.rl_env import DTEngineEnv
from algorithms.random_agent import RandomAgent
from algorithms.greedy_agent import GreedyAgent
from algorithms.dp_agent import DPAgent


def setup_clean_environment():
    """创建一个全新的样例"""
    net = Network()

    # Nodes
    node1 = EdgeNode(1, compute_power=10, cost=3, memory=5)
    node2 = EdgeNode(2, compute_power=5, cost=2, memory=8)
    node3 = EdgeNode(3, compute_power=8, cost=3, memory=8)

    # Sensors
    sensor1 = Sensor(101, data_size=5, period=10)
    sensor2 = Sensor(102, data_size=8, period=10)

    # Users
    user1 = UserNode(0)
    user2 = UserNode(99)

    for n in [node1, node2, node3, sensor1, sensor2, user1, user2]:
        net.add_node(n)

    # Edges (上行、横向、下行)
    net.add_edge(Edge(1001, sensor1, node1, bandwidth=10))
    net.add_edge(Edge(1002, sensor1, node2, bandwidth=5))
    net.add_edge(Edge(1003, sensor1, node3, bandwidth=1))
    net.add_edge(Edge(1004, sensor2, node2, bandwidth=5))

    net.add_edge(Edge(2001, node1, node2, bandwidth=10))
    net.add_edge(Edge(2002, node2, node1, bandwidth=10))
    net.add_edge(Edge(2003, node2, node3, bandwidth=5))
    net.add_edge(Edge(2003, node3, node2, bandwidth=5))

    net.add_edge(Edge(3001, node1, user1, bandwidth=10))
    net.add_edge(Edge(3002, node2, user1, bandwidth=10))
    net.add_edge(Edge(3003, node3, user1, bandwidth=10))
    net.add_edge(Edge(3004, node2, user2, bandwidth=10))

    # Tasks
    t1 = Task(id=1, workload=10, deployment_cost=5, memory_requirement=3)
    t2 = Task(id=2, workload=15, deployment_cost=5, memory_requirement=2)
    t3 = Task(id=3, workload=10, deployment_cost=10, memory_requirement=2)

    # Task chains
    dt_car = TaskChain(id="DT_CAR_01", tasks=[t1, t2], sensor_id=101)
    dt_factory = TaskChain(id="DT_FAC_01", tasks=[t3], sensor_id=102)

    users = [user1, user2]
    task_chains = [dt_car, dt_factory]

    return net, users, task_chains


def run_rl_evaluation(algo_name, model, dataset, total_reqs, seed):
    """
    使用 Gym 环境统一评估 RL 模型（PPO / DQN）。
    与 run_evaluation 返回相同格式的 history 列表，便于合并对比。
    """
    logger.info(f"\n========== 开始评测算法: {algo_name} ==========")

    env = DTEngineEnv(
        network=dataset['network'],
        users=dataset['users'],
        task_chains=dataset['task_chains'],
        request_stream=None,
        total_reqs=total_reqs,
        seed=seed,
        alpha=1.0,
        beta=1.0
    )
    obs, _ = env.reset(seed=seed)
    history = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        history.append({
            'req':      info['req_id'],
            'aoi':      info['aoi'],
            'cost':     info['cost'],
            'migrated': info.get('migrated', False)
        })
        if not (info['aoi'] == float('inf')):
            logger.info(
                f"[{algo_name}] 请求 {info['req_id']} -> Node {action} | "
                f"AoI={info['aoi']:.2f}, Cost={info['cost']:.4f}"
            )
        else:
            logger.warning(f"[{algo_name}] 请求 {info['req_id']} 部署失败")

    return history


def run_evaluation(algo_name, solver_func, total_reqs=15, seed=42):
    """
    在一个干净的环境中运行指定的算法
    """
    logger.info(f"\n========== 开始评测算法: {algo_name} ==========")

    dataset = load_dataset("data/dataset_real_30.pkl")
    net = dataset['network']
    users = dataset['users']
    task_chains = dataset['task_chains']
    request_stream = dataset['request_stream']

    sim = Simulator(net, dt_ttl=10.0)
    alpha, beta = 1.0, 1.0
    history = []

    # ==========================================
    # DP 离线算法分支
    # ==========================================
    if "DP" in algo_name:
        if "Oracle" in algo_name:
            history = solve_dp_offline(net, request_stream, alpha, beta)
            for step in history:
                logger.info(
                    f"[请求 {step['req']}] DP规划部署 Nodes {step['node_ids']} | "
                    f"AoI={step['aoi']:.2f}, Cost={step['cost']:.2f}"
                )
            return history
        else:
            # DP Real：先拿到理想路线图，再在真实物理世界执行
            dp_plan = solve_dp_offline(net, request_stream, alpha, beta)
            if not dp_plan:
                logger.error("DP 求解失败返回空计划，DP Real 无法执行！")
                return []

            for i, req in enumerate(request_stream):
                sim.cleanup_expired_dts(req.trigger_time)

                planned_node_ids = dp_plan[i]['node_ids']
                node_list = [net.get_node(nid) for nid in planned_node_ids]

                try:
                    real_sense, real_queue, real_comp, real_res, real_mig = sim.commit_step(
                        node_list, req.task_chain, req.user, req.trigger_time
                    )
                    real_aoi = estimate_aoi(real_sense, real_queue, real_comp, real_res)
                    real_cost = compute_cost(node_list, req.task_chain, real_mig)

                    history.append({
                        'req': req.id, 'aoi': real_aoi, 'cost': real_cost, 'migrated': real_mig
                    })
                    logger.info(
                        f"[请求 {req.id}] DP 现实执行 Nodes {planned_node_ids} | "
                        f"真实排队={real_queue:.2f}s, 真实 AoI={real_aoi:.2f}"
                    )
                except RuntimeError as e:
                    logger.error(f"[请求 {req.id}] DP 部署失败：{e} (请求被丢弃)")

            return history

    # ==========================================
    # PPO 强化学习分支（暂时注释）
    # ==========================================
    # if "PPO" in algo_name:
    #     try:
    #         model = PPO.load("environment/models/ppo_dt_deployment")
    #     except FileNotFoundError:
    #         logger.error("找不到 PPO 模型文件！请先运行 train_ppo.py")
    #         return []
    #
    #     env = DTEngineEnv(net, users, task_chains, total_reqs=len(request_stream), seed=seed)
    #     obs, _ = env.reset()
    #
    #     for req in request_stream:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #
    #         history.append({
    #             'req': info['req_id'],
    #             'aoi': info['aoi'],
    #             'cost': info['cost'],
    #             'migrated': info.get('migrated', False)
    #         })
    #         logger.info(
    #             f"[请求 {info['req_id']}] PPO 部署 Action {action} | "
    #             f"AoI={info['aoi']:.2f}, Cost={info['cost']:.2f}"
    #         )
    #         if terminated:
    #             break
    #
    #     return history

    # ==========================================
    # 在线算法通用循环 (Random / Greedy)
    # ==========================================
    for req in request_stream:
        evicted_dts = sim.cleanup_expired_dts(req.trigger_time)
        for dt_id, node_ids, memories in evicted_dts:
            total_mem = sum(memories)
            logger.info(
                f" [内存回收] t={req.trigger_time:.1f}s | {dt_id} 因超时未访问被卸载，"
                f"Nodes {node_ids} 共回收 {total_mem:.1f}G 内存"
            )

        best_node_list, score, estimated_metrics = solver_func(
            sim, net, req.task_chain, req.user, alpha, beta, req.trigger_time
        )

        if best_node_list is not None:
            real_sense, real_queue, real_comp, real_res, real_mig = sim.commit_step(
                best_node_list, req.task_chain, req.user, req.trigger_time
            )
            real_aoi = estimate_aoi(real_sense, real_queue, real_comp, real_res)
            real_cost = compute_cost(best_node_list, req.task_chain, real_mig)

            history.append({
                'req': req.id,
                'aoi': real_aoi,
                'cost': real_cost,
                'migrated': real_mig
            })
            node_ids_str = [n.id for n in best_node_list]
            logger.info(
                f"[请求 {req.id}] 部署 Nodes {node_ids_str} | "
                f"真实 AoI={real_aoi:.2f}, 真实 Cost={real_cost:.2f}"
            )
        else:
            logger.warning(f"[请求 {req.id}] 部署失败：资源枯竭或约束不满足")

    return history


def make_eval_env(dataset, total_reqs, seed, alpha=1.0, beta=1.0):
    """创建用于评估的 DTEngineEnv 实例。"""
    return DTEngineEnv(
        network=dataset['network'],
        users=dataset['users'],
        task_chains=dataset['task_chains'],
        request_stream=None,
        total_reqs=total_reqs,
        seed=seed,
        alpha=alpha,
        beta=beta
    )


def run_agent_evaluation(algo_name, agent, env, seed):
    """
    在指定 Gym 环境中运行任意 agent（SB3 模型或自定义 Agent），返回历史记录。
    agent 必须提供 predict(obs, deterministic=True) -> (action, state) 接口。
    """
    logger.info(f"\n========== 开始评测算法: {algo_name} ==========")
    obs, _ = env.reset(seed=seed)
    history = []
    done = False

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        aoi = info['aoi']
        cost = info['cost']
        history.append({
            'req':      info['req_id'],
            'aoi':      aoi,
            'cost':     cost,
            'migrated': info.get('migrated', False)
        })
        if aoi != float('inf'):
            logger.info(
                f"[{algo_name}] 请求 {info['req_id']} -> Node {action} | "
                f"AoI={aoi:.2f}, Cost={cost:.4f}"
            )
        else:
            logger.warning(f"[{algo_name}] 请求 {info['req_id']} 部署失败")

    return history


def main():
    import os
    logger.info("正在加载测试数据集...")
    dataset = load_dataset("data/dataset_real_30.pkl")

    TOTAL_REQUESTS = 200
    ALPHA, BETA = 1.0, 1.0
    COMMON_SEED = dataset['config']['seed']
    logger.info(f"数据集加载完毕，将评估前 {TOTAL_REQUESTS} 个请求。")

    all_histories = {}

    # ==========================================
    # 1. 随机策略
    # ==========================================
    env = make_eval_env(dataset, TOTAL_REQUESTS, COMMON_SEED, ALPHA, BETA)
    all_histories['Random'] = run_agent_evaluation(
        "Random", RandomAgent(env), env, COMMON_SEED
    )

    # ==========================================
    # 2. 贪心策略（O(|nodes|) 单节点贪心，避免指数级搜索）
    # ==========================================
    env = make_eval_env(dataset, TOTAL_REQUESTS, COMMON_SEED, ALPHA, BETA)
    all_histories['Greedy'] = run_agent_evaluation(
        "Greedy", GreedyAgent(env), env, COMMON_SEED
    )

    # ==========================================
    # 3. DP Oracle（离线最优，理论上界）
    # ==========================================
    # 真实数据不适用dp
    # env = make_eval_env(dataset, TOTAL_REQUESTS, COMMON_SEED, ALPHA, BETA)
    # env.reset(seed=COMMON_SEED)          # 提前 reset 以生成 request_stream
    # dp_oracle = DPAgent(env)
    # if dp_oracle.valid:
    #     raw = dp_oracle.get_oracle_history()
    #     all_histories['DP Oracle'] = [
    #         {'req': s['req'], 'aoi': s['aoi'], 'cost': s['cost'],
    #          'migrated': s.get('migrated', False)}
    #         for s in raw
    #     ]
    #     logger.info(f"[DP Oracle] 获取到 {len(all_histories['DP Oracle'])} 条理论最优记录")
    # else:
    #     logger.warning("[DP Oracle] DP 求解失败，跳过。")

    # ==========================================
    # 4. DP Real（在线执行 DP 锦囊，受真实排队影响）
    # ==========================================
    # env = make_eval_env(dataset, TOTAL_REQUESTS, COMMON_SEED, ALPHA, BETA)
    # env.reset(seed=COMMON_SEED)
    # dp_real = DPAgent(env)
    # if dp_real.valid:
    #     all_histories['DP Real'] = run_agent_evaluation(
    #         "DP Real", dp_real, env, COMMON_SEED
    #     )
    # else:
    #     logger.warning("[DP Real] DP 求解失败，跳过。")

    # ==========================================
    # 5. PPO
    # ==========================================
    ppo_path = "environment/models/pareto_real/ppo_real_a0.5_b0.5"
    if os.path.exists(ppo_path + ".zip"):
        env = make_eval_env(dataset, TOTAL_REQUESTS, COMMON_SEED, ALPHA, BETA)
        ppo_model = PPO.load(ppo_path, env=env)
        all_histories['PPO'] = run_agent_evaluation(
            "PPO", ppo_model, env, COMMON_SEED
        )
    else:
        logger.warning(f"未找到 PPO 模型 {ppo_path}.zip，跳过。")

    # ==========================================
    # 6. DQN（无 Action Mask）
    # ==========================================
    dqn_path = "environment/models/pareto_real/dqn_real_a0.5_b0.5_no_ac_mask"
    if os.path.exists(dqn_path):
        env = make_eval_env(dataset, TOTAL_REQUESTS, COMMON_SEED, ALPHA, BETA)
        dqn_model = DQN.load(dqn_path, env=env)
        all_histories['DQN (no mask)'] = run_agent_evaluation(
            "DQN (no mask)", dqn_model, env, COMMON_SEED
        )
    else:
        logger.warning(f"未找到 DQN 模型 {dqn_path}，跳过。")

    # ==========================================
    # 汇总输出
    # ==========================================
    save_simulation_results(all_histories, filename="results/latest_simulation.json")
    generate_summary_report(all_histories, TOTAL_REQUESTS)

    logger.info("仿真结束，正在生成对比可视化报表...")
    plot_comparative_results(all_histories)


if __name__ == "__main__":
    main()
