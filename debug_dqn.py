# debug_dqn.py — DQN 调试脚本
# 用途：在正式训练前验证 DQN 与环境的对接是否正常
# 运行：python debug_dqn.py

import os
import math
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from environment.rl_env import DTEngineEnv
from utils.data_generator import load_dataset

# ==========================================
# 配置区：按需修改
# ==========================================
DATASET_PATH = "data/dataset_real_30.pkl"  # 可换成 dataset_debug.pkl 更快
ALPHA = 0.5
BETA  = 0.5
TRAIN_REQS_PER_EPISODE = 100   # episode 稍长，让 agent 看到更多状态转移
SHORT_TRAIN_STEPS      = 10000 # 调试训练步数：够收敛又不太慢（正式是 50000）
EVAL_REQS              = 50    # 评估请求数
EVAL_SEED              = 2026


def load_env(dataset, total_reqs, seed=None):
    return DTEngineEnv(
        network=dataset['network'],
        users=dataset['users'],
        task_chains=dataset['task_chains'],
        request_stream=None,
        total_reqs=total_reqs,
        seed=seed,
        alpha=ALPHA,
        beta=BETA
    )


# ==========================================
# 阶段 1：Gym 环境合规性检查
# ==========================================
def stage1_check_env(dataset):
    print("\n" + "=" * 50)
    print("阶段 1：Gym 合规性检查 (check_env)")
    print("=" * 50)
    env = load_env(dataset, TRAIN_REQS_PER_EPISODE)
    try:
        check_env(env, warn=True)
        print("[OK] 环境通过 Gym 合规性检查")
    except Exception as e:
        print(f"[FAIL] 环境检查失败: {e}")
        raise
    return env


# ==========================================
# 阶段 2：手动 step 连跑一轮，确认 step() 无崩溃
# 同时打印 action_masks() 验证约束过滤逻辑
# ==========================================
def stage2_manual_rollout(dataset):
    print("\n" + "=" * 50)
    print("阶段 2：手动 rollout（随机动作 + 验证 action_masks）")
    print("=" * 50)
    env = load_env(dataset, EVAL_REQS, seed=EVAL_SEED)
    obs, _ = env.reset(seed=EVAL_SEED)

    success_count = 0
    fail_count = 0

    for step in range(EVAL_REQS):
        masks = env.action_masks()
        valid_actions = np.where(masks)[0]

        # 从合法动作中随机选，模拟"完美探索"的上限
        action = int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        aoi  = info['aoi']
        cost = info['cost']
        ok   = not math.isinf(aoi)
        if ok:
            success_count += 1
            print(f"  step {step:3d} | action={action} | valid_nodes={list(valid_actions)} | "
                  f"AoI={aoi:7.2f} ms | Cost={cost:.4f}")
        else:
            fail_count += 1
            print(f"  step {step:3d} | action={action} | [DEPLOY FAILED] reward={reward:.2f}")

        if terminated or truncated:
            break

    print(f"\n[结果] 成功={success_count}，失败={fail_count}，"
          f"成功率={success_count / (success_count + fail_count) * 100:.1f}%")
    print("  ↑ 这是使用合法动作时的理论上限成功率，DQN 训练后应接近此值")


# ==========================================
# 阶段 3：短训练（验证 DQN.learn() 不崩溃）
# ==========================================
def stage3_short_train(dataset):
    print("\n" + "=" * 50)
    print(f"阶段 3：短训练 {SHORT_TRAIN_STEPS} 步（验证 DQN.learn() 正常）")
    print("=" * 50)

    vec_env = make_vec_env(
        lambda: load_env(dataset, TRAIN_REQS_PER_EPISODE),
        n_envs=1
    )

    model = DQN(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,          # 与 PPO 对齐，收敛更快
        buffer_size=10000,           # 与训练步数匹配
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.4,    # 前 40% 步数保持高探索
        exploration_final_eps=0.05,  # 最终仍保留 5% 随机探索
        target_update_interval=500,
        verbose=1,
        learning_starts=100,         # 100 步后开始学习
        policy_kwargs=dict(net_arch=[128, 128])  # 稍大的网络，匹配状态空间复杂度
    )

    try:
        model.learn(total_timesteps=SHORT_TRAIN_STEPS)
        print("[OK] 短训练完成，无异常")
    except Exception as e:
        print(f"[FAIL] 训练崩溃: {e}")
        raise

    return model, vec_env


# ==========================================
# 阶段 4：用训练好的模型跑评估
# ==========================================
def stage4_eval(model, dataset):
    print("\n" + "=" * 50)
    print("阶段 4：模型评估（deterministic=True）")
    print("=" * 50)

    eval_env = load_env(dataset, EVAL_REQS, seed=EVAL_SEED)
    obs, _ = eval_env.reset(seed=EVAL_SEED)

    # 计算 q_vec 在 obs 中的起始索引（用于诊断）
    num_nodes   = eval_env.num_nodes
    num_users   = eval_env.num_users
    num_sensors = eval_env.num_sensors
    q_start = num_users + num_sensors + 4 * num_nodes + 1  # u+s+h+m+aoi+s2n+n2u → q_vec
    print(f"  [诊断] obs_dim={eval_env.obs_dim}, num_nodes={num_nodes}, "
          f"q_vec 起始索引={q_start}, 期望obs_dim={num_users+num_sensors+5*num_nodes+1}")

    aoi_list  = []
    cost_list = []
    success   = 0
    done      = False
    step_count = 0

    while not done:
        masks = eval_env.action_masks()
        action, _ = model.predict(obs, deterministic=True)

        # 前 5 步：打印 q_vec（各节点排队等待归一化值）和 Q 值分布，辅助诊断
        if step_count < 5:
            q_vec = obs[q_start: q_start + num_nodes]
            top3_q = sorted(enumerate(q_vec), key=lambda x: x[1], reverse=True)[:3]
            print(f"  [q_vec诊断 step={step_count}] "
                  f"node28排队={q_vec[28]:.3f} | 最拥塞前3: {[(n,f'{v:.3f}') for n,v in top3_q]}")
            # 打印模型对各 action 的 Q 值
            import torch
            obs_tensor = eval_env.observation_space.sample() * 0  # placeholder
            obs_tensor = obs.copy()
            with torch.no_grad():
                obs_th = torch.as_tensor(obs_tensor, dtype=torch.float32).unsqueeze(0)
                q_vals = model.policy.q_net(obs_th).squeeze().numpy()
            top3_a = sorted(enumerate(q_vals), key=lambda x: x[1], reverse=True)[:3]
            print(f"  [Q值诊断  step={step_count}] "
                  f"Q[28]={q_vals[28]:.3f} | 最高Q前3: {[(a,f'{v:.3f}') for a,v in top3_a]}")

        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        step_count += 1

        aoi  = info['aoi']
        cost = info['cost']
        valid_flag = "✓" if masks[action] else "✗ INVALID"
        if not math.isinf(aoi):
            aoi_list.append(aoi)
            cost_list.append(cost)
            success += 1
            print(f"  req={info['req_id']} | action={action}[{valid_flag}] | AoI={aoi:7.2f} ms | Cost={cost:.4f}")
        else:
            print(f"  req={info['req_id']} | action={action}[{valid_flag}] | [DEPLOY FAILED]")

    total = success + (EVAL_REQS - success)
    print(f"\n[评估结果]")
    print(f"  请求总数   : {EVAL_REQS}")
    print(f"  成功部署   : {success}  ({success / EVAL_REQS * 100:.1f}%)")
    if aoi_list:
        print(f"  平均 AoI   : {np.mean(aoi_list):.2f} ms")
        print(f"  平均 Cost  : {np.mean(cost_list):.4f}")


# ==========================================
# 阶段 5（可选）：保存调试模型
# ==========================================
def stage5_save(model):
    print("\n" + "=" * 50)
    print("阶段 5：保存调试模型")
    print("=" * 50)
    save_dir = "environment/models/debug"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "dqn_debug")
    model.save(path)
    print(f"[OK] 调试模型已保存至 {path}.zip")


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    print("加载数据集...")
    dataset = load_dataset(DATASET_PATH)
    print(f"  网络节点数: {len(dataset['network'].get_edge_nodes())}")
    print(f"  用户数    : {len(dataset['users'])}")
    print(f"  任务链数  : {len(dataset['task_chains'])}")

    stage1_check_env(dataset)
    stage2_manual_rollout(dataset)
    model, vec_env = stage3_short_train(dataset)
    stage4_eval(model, dataset)
    stage5_save(model)

    print("\n[全部阶段完成] DQN 调试通过，可以进行正式训练。")
    print("运行正式训练：python environment/train_dqn.py")
