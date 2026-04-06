from algorithms.greedy_agent import GreedyAgent
from environment.rl_env import DTEngineEnv
from utils.data_generator import load_dataset


def test_greedy_baseline_before_training(dataset_path):
    dataset = load_dataset(dataset_path)
    # 使用 α=0.5, β=0.5 的均衡权重进行测试
    env = DTEngineEnv(
        network=dataset['network'],
        users=dataset['users'],
        task_chains=dataset['task_chains'],
        total_reqs=len(dataset['request_stream']),
        seed=42
    )

    agent = GreedyAgent(env)
    obs, info = env.reset()

    success_count = 0
    total_count = len(dataset['request_stream'])

    print(f">>> 正在运行贪心基准测试 (验证容量是否充足)...")
    for _ in range(total_count):
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # 只要没有因资源不足而导致部署失败 (cost != inf)
        if info.get('cost') != float('inf'):
            success_count += 1

        if terminated or truncated:
            break

    success_rate = (success_count / total_count) * 100
    print(f"\n--- 测试结果 ---")
    print(f"最终成功率: {success_rate:.2f}%")

    if success_rate < 80:
        print("警告：成功率偏低！建议在训练前进一步扩大节点内存容量。")
    else:
        print("容量充足，可以开始强化学习训练。")

    return success_rate


if __name__ == "__main__":
    test_greedy_baseline_before_training("../data/dataset_real_30.pkl")
