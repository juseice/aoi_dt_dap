# [Main] 推进物理时间 (例如 t = 2.5s)，产生了一个新请求
#   │
#   ├──> [Main] 呼叫 [Solver]: "当前时间 2.5s，任务链来了，给我个最优节点。"
#   │      │
#   │      ├──> [Solver] 遍历备选 Node 1, 2, 3...
#   │      │      ├──> 问 [Simulator]: "evaluate_step(Node 1, t=2.5s) 指标如何？"
#   │      │      ├──> 问 [Simulator]: "evaluate_step(Node 2, t=2.5s) 指标如何？"
#   │      │      └──> ...
#   │      │
#   │      ├──> [Solver] 算了一遍 Score，发现 Node 2 分数最低（最好）。
#   │      └──> [Solver] 返回给 [Main]: "选 Node 2，预计得分为 X。"
#   │
#   ├──> [Main] 收到决策，命令 [Simulator]: "commit_step(Node 2, t=2.5s)！"
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
from environment.event import generate_poisson_requests
from environment.rl_env import DTEngineEnv
import random


def setup_clean_environment():
    """创建一个全新、干净的网络环境和实体"""
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


def run_evaluation(algo_name, solver_func, total_reqs=15, seed=42):
    """
    在一个干净的环境中运行指定的算法
    """
    logger.info(f"\n========== 开始评测算法: {algo_name} ==========")

    # 1. 重置物理世界
    logger.info("初始化网络拓扑...")
    net, users, task_chains = setup_clean_environment()

    # plot_network_topology(net)  # 如果你想看图，可以把这行取消注释
    sim = Simulator(net, dt_ttl=25.0)

    # 2. 生成固定的事件流 (只要 seed 相同，请求的时间、用户、任务完全一致)
    request_stream = generate_poisson_requests(
        users=users, task_chains=task_chains,
        arrival_rate=0.5, total_requests=total_reqs, seed=seed
    )

    alpha, beta = 1.0, 1.0
    history = []

    # 如果是全局最优的离线 DP 算法，直接出结果
    if "DP" in algo_name:
        if "Oracle" in algo_name:
            history = solve_dp_offline(net, request_stream, alpha, beta)
            for step in history:
                logger.info(
                    f"[请求 {step['req']}] DP规划部署 Node {step['node_id']} | AoI={step['aoi']:.2f}, Cost={step['cost']}")
            return history
        else:
            # 先拿到理想路线图
            dp_plan = solve_dp_offline(net, request_stream, alpha, beta)

            for i, req in enumerate(request_stream):
                sim.cleanup_expired_dts(req.time)
                # 强行提取 DP 计划中的目标节点
                planned_node_id = dp_plan[i]['node_id']
                target_node = net.get_node(planned_node_id)

                try:
                    # 强行在物理世界执行！(这里会自动产生真实的排队时延)
                    real_sense, real_queue, real_comp, real_res, real_mig = sim.commit_step(
                        target_node, req.task_chain, req.user, req.time
                    )

                    real_aoi = estimate_aoi(real_sense, real_queue, real_comp, real_res)
                    real_cost = compute_cost(target_node, real_comp, req.task_chain, real_mig)

                    history.append({
                        'req': req.id, 'aoi': real_aoi, 'cost': real_cost, 'migrated': real_mig
                    })
                    logger.info(
                        f"[请求 {req.id}] DP 现实执行 Node {target_node.id} | 真实排队={real_queue:.2f}s, 真实 AoI={real_aoi:.2f}")

                except RuntimeError as e:
                    # 捕获内存不足导致的物理崩溃
                    logger.error(f"[请求 {req.id}] DP 部署失败：{e} (请求被丢弃)")

            return history



    # 3. 开始仿真循环
    for req in request_stream:
        evicted_dts = sim.cleanup_expired_dts(req.time)
        for dt_id, node_id, mem_req in evicted_dts:
            logger.info(
                f" [内存回收] t={req.time:.1f}s | {dt_id} 因超时未访问被卸载，Node {node_id} 恢复 {mem_req}G 内存")

        best_node, score, estimated_metrics = solver_func(
            sim, net, req.task_chain, req.user, alpha, beta, req.time
        )

        if best_node is not None:
            real_sense, real_queue, real_comp, real_res, real_mig = sim.commit_step(
                best_node, req.task_chain, req.user, req.time
            )
            real_aoi = estimate_aoi(real_sense, real_queue, real_comp, real_res)
            real_cost = compute_cost(best_node, real_comp, req.task_chain, real_mig)
            history.append({
                'req': req.id,
                'aoi': real_aoi,
                'cost': real_cost,
                'migrated': real_mig
            })
            logger.info(f"[请求 {req.id}] 部署 Node {best_node.id} | 真实 AoI={real_aoi:.2f}, 真实 Cost={real_cost:.2f}")
        else:
            logger.warning(f"[请求 {req.id}] 部署失败：资源枯竭或约束不满足")\

    return history


def test_rl_environment():
    logger.info("\n========== 测试强化学习环境接口 ==========")
    net, users, task_chains = setup_clean_environment()  # 借用你之前的建图函数

    # 实例化环境
    env = DTEngineEnv(net, users, task_chains, total_reqs=10, seed=42)

    # 1. 游戏重置 (获得初始状态)
    obs, info = env.reset()
    logger.info(f"初始状态向量: {obs}")

    total_reward = 0
    done = False

    # 2. 交互循环 (马尔可夫决策过程 MDP)
    while not done:
        # 智能体思考：这里我们先用 env.action_space.sample() 随机抛骰子选一个动作
        # 以后这里就会替换成 model.predict(obs) ！！
        action = env.action_space.sample()

        # 告诉环境执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward

        logger.info(f"执行动作(节点索引): {action} | 获得 Reward: {reward:.2f} | Info: {info}")

    logger.info(f"回合结束！累计 Reward: {total_reward:.2f}")


def main():
    TOTAL_REQUESTS = 20
    COMMON_SEED = 42  # 随便取一个幸运数字

    # 1. 跑 Baseline (随机策略)
    history_random = run_evaluation(
        algo_name="Random Baseline",
        solver_func=select_random_node,
        total_reqs=TOTAL_REQUESTS,
        seed=COMMON_SEED
    )

    # 2. 贪心最优策略
    history_best = run_evaluation(
        algo_name="Greedy Best",
        solver_func=select_best_node,
        total_reqs=TOTAL_REQUESTS,
        seed=COMMON_SEED
    )

    history_dp = run_evaluation(
        "DP Optimal (Oracle)",
        solve_dp_offline,
        TOTAL_REQUESTS,
        COMMON_SEED
    )

    history_dp_real = run_evaluation(
        "DP Real (Simulated)",
        solve_dp_offline,
        TOTAL_REQUESTS,
        COMMON_SEED
    )

    # 3. 合并数据
    all_histories = {
        'Random Baseline': history_random,
        'Greedy Best': history_best,
        'DP Ideal': history_dp,
        'DP Real': history_dp_real
    }

    logger.info("\n仿真结束，正在生成对比可视化报表...")
    plot_comparative_results(all_histories)

    test_rl_environment()

if __name__ == "__main__":
    main()
