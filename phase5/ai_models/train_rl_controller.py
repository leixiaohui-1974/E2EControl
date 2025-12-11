"""
强化学习控制策略训练脚本
RL Controller Training Script with Stable-Baselines3

用法:
    python train_rl_controller.py --algo ppo --total_timesteps 100000

支持算法:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- A2C (Advantage Actor-Critic)
"""

import argparse
import os
import sys
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stable-Baselines3
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# 自定义环境
from ai_models.water_canal_env import (
    WaterCanalEnv,
    OneGateTwoPoolsEnv,
    WaterCanalEnvConfig
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 算法配置
# ==============================================================================

ALGO_CONFIG = {
    'ppo': {
        'class': PPO,
        'policy': 'MlpPolicy',
        'hyperparams': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    'sac': {
        'class': SAC,
        'policy': 'MlpPolicy',
        'hyperparams': {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
        }
    },
    'td3': {
        'class': TD3,
        'policy': 'MlpPolicy',
        'hyperparams': {
            'learning_rate': 1e-3,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 100,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': (1, "episode"),
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
        }
    },
    'a2c': {
        'class': A2C,
        'policy': 'MlpPolicy',
        'hyperparams': {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.25,
            'max_grad_norm': 0.5,
            'normalize_advantage': False,
        }
    },
}


# ==============================================================================
# 环境工厂
# ==============================================================================

def make_env(config: WaterCanalEnvConfig, rank: int = 0, seed: int = 0):
    """创建单个环境实例"""
    def _init():
        env = OneGateTwoPoolsEnv(config)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_training_env(config: WaterCanalEnvConfig,
                        n_envs: int = 4,
                        seed: int = 42,
                        use_subprocess: bool = False):
    """创建训练环境"""
    if use_subprocess and n_envs > 1:
        env = SubprocVecEnv([make_env(config, i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(config, i, seed) for i in range(n_envs)])
    return env


# ==============================================================================
# 训练函数
# ==============================================================================

def train(args):
    """主训练函数"""
    logger.info("=" * 70)
    logger.info("强化学习控制器训练")
    logger.info("=" * 70)

    # 设置随机种子
    np.random.seed(args.seed)

    # 创建环境配置
    env_config = WaterCanalEnvConfig(
        episode_duration=args.episode_duration,
        dt=900.0,  # 15分钟
        observation_window=16,
        w_level=1.0,
        w_action=0.1,
        w_safety=5.0,
    )

    # 创建训练环境
    logger.info(f"\n创建训练环境 (n_envs={args.n_envs})...")
    train_env = create_training_env(
        env_config,
        n_envs=args.n_envs,
        seed=args.seed
    )

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(env_config, 0, args.seed + 100)])

    # 获取算法配置
    if args.algo not in ALGO_CONFIG:
        raise ValueError(f"不支持的算法: {args.algo}")

    algo_cfg = ALGO_CONFIG[args.algo]
    AlgoClass = algo_cfg['class']
    policy = algo_cfg['policy']
    hyperparams = algo_cfg['hyperparams'].copy()

    # 覆盖学习率
    if args.learning_rate:
        hyperparams['learning_rate'] = args.learning_rate

    logger.info(f"\n算法: {args.algo.upper()}")
    logger.info(f"超参数: {hyperparams}")

    # 创建模型
    logger.info(f"\n创建模型...")
    model = AlgoClass(
        policy,
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(args.output_dir, 'tensorboard'),
        seed=args.seed,
        **hyperparams
    )

    # 打印模型信息
    logger.info(f"  策略网络: {model.policy}")

    # 创建回调
    os.makedirs(args.output_dir, exist_ok=True)

    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.output_dir, 'best_model'),
        log_path=os.path.join(args.output_dir, 'eval_logs'),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.output_dir, 'checkpoints'),
        name_prefix=f'{args.algo}_watercanal',
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # 训练
    logger.info(f"\n开始训练 (total_timesteps={args.total_timesteps})...")
    logger.info("-" * 70)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # 保存最终模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(args.output_dir, f'{args.algo}_final_{timestamp}.zip')
    model.save(final_model_path)
    logger.info(f"\n最终模型已保存: {final_model_path}")

    # 评估
    logger.info(f"\n最终评估...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )
    logger.info(f"  平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 清理
    train_env.close()
    eval_env.close()

    # 汇总
    logger.info("\n" + "=" * 70)
    logger.info("训练完成!")
    logger.info("=" * 70)
    logger.info(f"  算法: {args.algo.upper()}")
    logger.info(f"  总时间步: {args.total_timesteps}")
    logger.info(f"  最终奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
    logger.info(f"  模型路径: {final_model_path}")

    return final_model_path


# ==============================================================================
# 模型评估
# ==============================================================================

def evaluate(model_path: str, num_episodes: int = 5):
    """评估已训练的模型"""
    logger.info("=" * 70)
    logger.info("模型评估")
    logger.info("=" * 70)

    # 创建环境
    env_config = WaterCanalEnvConfig(
        episode_duration=86400.0,  # 24小时
    )
    env = OneGateTwoPoolsEnv(env_config)

    # 加载模型 (自动检测算法类型)
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    elif 'sac' in model_path.lower():
        model = SAC.load(model_path)
    elif 'td3' in model_path.lower():
        model = TD3.load(model_path)
    elif 'a2c' in model_path.lower():
        model = A2C.load(model_path)
    else:
        model = PPO.load(model_path)  # 默认PPO

    # 评估
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        logger.info(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={step_count}")

    # 统计
    logger.info(f"\n统计:")
    logger.info(f"  平均奖励: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    logger.info(f"  平均步数: {np.mean(episode_lengths):.1f}")

    env.close()


# ==============================================================================
# 命令行参数
# ==============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='强化学习控制器训练')

    # 算法参数
    parser.add_argument('--algo', type=str, default='ppo',
                        choices=['ppo', 'sac', 'td3', 'a2c'],
                        help='RL算法')
    parser.add_argument('--total_timesteps', type=int, default=100000,
                        help='总训练步数')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率 (覆盖默认值)')

    # 环境参数
    parser.add_argument('--n_envs', type=int, default=4,
                        help='并行环境数')
    parser.add_argument('--episode_duration', type=float, default=86400.0,
                        help='单episode时长 [s]')

    # 训练参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='评估频率')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='检查点保存频率')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='models/rl',
                        help='输出目录')

    # 评估模式
    parser.add_argument('--eval', type=str, default=None,
                        help='评估已训练的模型')

    return parser.parse_args()


# ==============================================================================
# 主函数
# ==============================================================================

if __name__ == "__main__":
    args = parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        train(args)
