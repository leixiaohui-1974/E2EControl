"""
持续学习模块 (Continual Learner)
支持在线学习和自主优化

特点:
1. 在线经验回放
2. 增量学习不遗忘
3. 异常检测与适应
4. 模型版本管理
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import logging
import pickle
import os
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """经验样本"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class LearningConfig:
    """学习配置"""
    # 缓冲区
    buffer_size: int = 100000
    batch_size: int = 64
    min_samples: int = 1000

    # 学习参数
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005

    # 更新频率
    update_frequency: int = 100       # 每N步更新一次
    target_update_frequency: int = 1000

    # 持续学习
    ewc_lambda: float = 1000.0        # EWC正则化强度
    replay_ratio: float = 0.5         # 旧经验回放比例

    # 版本管理
    checkpoint_frequency: int = 10000
    max_checkpoints: int = 10


class OnlineLearningBuffer:
    """
    在线学习缓冲区
    支持优先级经验回放
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数

        self.buffer = []
        self.priorities = []
        self.position = 0

        # 统计
        self.total_samples = 0
        self.total_updates = 0

    def push(self, experience: Experience, priority: float = None):
        """添加经验"""
        if priority is None:
            # 新经验默认最高优先级
            priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity
        self.total_samples += 1

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        优先级采样

        Returns:
            experiences: 经验列表
            indices: 索引
            weights: 重要性权重
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # 计算采样概率
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        experiences = [self.buffer[i] for i in indices]

        # 重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-0.4)
        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def __len__(self):
        return len(self.buffer)


class ExperienceReplay:
    """
    经验回放管理器
    支持多种回放策略
    """

    def __init__(self, config: LearningConfig):
        self.config = config

        # 主缓冲区 (最近经验)
        self.recent_buffer = OnlineLearningBuffer(config.buffer_size // 2)

        # 重要经验缓冲区 (高TD误差)
        self.important_buffer = OnlineLearningBuffer(config.buffer_size // 4)

        # 旧经验缓冲区 (防遗忘)
        self.old_buffer = OnlineLearningBuffer(config.buffer_size // 4)

        # 分类计数
        self.category_counts = {
            'recent': 0,
            'important': 0,
            'old': 0
        }

    def add(self, experience: Experience, td_error: float = None):
        """添加经验到适当缓冲区"""
        # 最近经验
        self.recent_buffer.push(experience)
        self.category_counts['recent'] += 1

        # 高TD误差 -> 重要经验
        if td_error is not None and abs(td_error) > 1.0:
            self.important_buffer.push(experience, priority=abs(td_error))
            self.category_counts['important'] += 1

    def sample(self, batch_size: int) -> List[Experience]:
        """混合采样"""
        # 分配采样数量
        recent_size = int(batch_size * (1 - self.config.replay_ratio))
        old_size = batch_size - recent_size

        experiences = []

        # 从最近缓冲区采样
        if len(self.recent_buffer) >= recent_size:
            exp, _, _ = self.recent_buffer.sample(recent_size)
            experiences.extend(exp)

        # 从重要/旧缓冲区采样
        if len(self.important_buffer) >= old_size // 2:
            exp, _, _ = self.important_buffer.sample(old_size // 2)
            experiences.extend(exp)

        if len(self.old_buffer) >= old_size // 2:
            exp, _, _ = self.old_buffer.sample(old_size // 2)
            experiences.extend(exp)

        return experiences

    def archive_old_experiences(self):
        """归档旧经验"""
        # 将部分最近经验移至旧缓冲区
        if len(self.recent_buffer) > self.recent_buffer.capacity * 0.8:
            num_to_archive = len(self.recent_buffer) // 10
            for i in range(num_to_archive):
                if self.recent_buffer.buffer:
                    exp = self.recent_buffer.buffer[i]
                    self.old_buffer.push(exp)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'recent_size': len(self.recent_buffer),
            'important_size': len(self.important_buffer),
            'old_size': len(self.old_buffer),
            'category_counts': self.category_counts.copy()
        }


class EWCRegularizer:
    """
    弹性权重合并 (Elastic Weight Consolidation)
    防止灾难性遗忘
    """

    def __init__(self, model: nn.Module, lambda_: float = 1000.0):
        self.model = model
        self.lambda_ = lambda_

        # Fisher信息矩阵
        self.fisher = {}
        # 旧参数
        self.old_params = {}

        self.is_initialized = False

    def compute_fisher(self, data_loader, num_samples: int = 1000):
        """计算Fisher信息矩阵"""
        self.fisher = {}

        for name, param in self.model.named_parameters():
            self.fisher[name] = torch.zeros_like(param)

        self.model.eval()
        count = 0

        for batch in data_loader:
            if count >= num_samples:
                break

            # 前向传播
            self.model.zero_grad()
            output = self.model(batch['input'])
            loss = output.pow(2).mean()  # 简化的损失
            loss.backward()

            # 累积Fisher
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.pow(2)

            count += len(batch['input'])

        # 归一化
        for name in self.fisher:
            self.fisher[name] /= count

        # 保存旧参数
        for name, param in self.model.named_parameters():
            self.old_params[name] = param.clone().detach()

        self.is_initialized = True

    def penalty(self) -> torch.Tensor:
        """计算EWC惩罚项"""
        if not self.is_initialized:
            return torch.tensor(0.0)

        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.old_params[name]).pow(2)).sum()

        return self.lambda_ * loss


class ContinualLearner:
    """
    持续学习器
    管理在线学习过程
    """

    def __init__(self, model: nn.Module, config: LearningConfig = None):
        self.model = model
        self.config = config or LearningConfig()

        # 经验回放
        self.replay = ExperienceReplay(self.config)

        # EWC正则化
        self.ewc = EWCRegularizer(model, self.config.ewc_lambda)

        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # 目标网络 (如果需要)
        self.target_model = None

        # 学习统计
        self.stats = {
            'total_steps': 0,
            'total_updates': 0,
            'avg_loss': 0.0,
            'avg_td_error': 0.0
        }

        # 模型版本
        self.model_versions = []
        self.current_version = 0

        logger.info("Continual Learner initialized")

    def step(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: float,
             next_state: np.ndarray,
             done: bool,
             info: Dict = None) -> Dict:
        """
        学习步骤

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
            info: 额外信息

        Returns:
            学习统计
        """
        # 创建经验
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
            timestamp=datetime.now().timestamp()
        )

        # 添加到缓冲区
        self.replay.add(experience)
        self.stats['total_steps'] += 1

        # 检查是否更新
        result = {'updated': False}

        if (self.stats['total_steps'] % self.config.update_frequency == 0 and
            len(self.replay.recent_buffer) >= self.config.min_samples):

            loss, td_error = self._update()
            result['updated'] = True
            result['loss'] = loss
            result['td_error'] = td_error

            self.stats['total_updates'] += 1
            self.stats['avg_loss'] = 0.9 * self.stats['avg_loss'] + 0.1 * loss
            self.stats['avg_td_error'] = 0.9 * self.stats['avg_td_error'] + 0.1 * td_error

        # 检查是否保存检查点
        if self.stats['total_steps'] % self.config.checkpoint_frequency == 0:
            self._save_checkpoint()

        return result

    def _update(self) -> Tuple[float, float]:
        """执行一次更新"""
        # 采样经验
        experiences = self.replay.sample(self.config.batch_size)

        if not experiences:
            return 0.0, 0.0

        # 准备批次数据
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.FloatTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor([float(e.done) for e in experiences])

        # 计算损失 (简化版)
        self.model.train()

        # 前向传播
        current_output = self.model(states)

        # TD目标 (如果有目标网络)
        with torch.no_grad():
            if self.target_model is not None:
                next_output = self.target_model(next_states)
            else:
                next_output = self.model(next_states)
            td_target = rewards + self.config.gamma * (1 - dones) * next_output.squeeze()

        # TD误差
        td_error = (current_output.squeeze() - td_target).abs().mean()

        # 主损失
        loss = nn.MSELoss()(current_output.squeeze(), td_target)

        # EWC惩罚
        ewc_loss = self.ewc.penalty()
        total_loss = loss + ewc_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item(), td_error.item()

    def _save_checkpoint(self):
        """保存检查点"""
        version = {
            'version': self.current_version,
            'timestamp': datetime.now().isoformat(),
            'model_state': self.model.state_dict().copy(),
            'optimizer_state': self.optimizer.state_dict().copy(),
            'stats': self.stats.copy()
        }

        self.model_versions.append(version)
        self.current_version += 1

        # 限制版本数量
        if len(self.model_versions) > self.config.max_checkpoints:
            self.model_versions.pop(0)

        logger.info(f"Checkpoint saved: version {self.current_version}")

    def rollback(self, version: int = None):
        """回滚到指定版本"""
        if not self.model_versions:
            logger.warning("No checkpoints available")
            return False

        if version is None:
            # 回滚到上一版本
            version_data = self.model_versions[-1]
        else:
            # 找到指定版本
            version_data = None
            for v in self.model_versions:
                if v['version'] == version:
                    version_data = v
                    break

            if version_data is None:
                logger.warning(f"Version {version} not found")
                return False

        # 恢复模型
        self.model.load_state_dict(version_data['model_state'])
        self.optimizer.load_state_dict(version_data['optimizer_state'])

        logger.info(f"Rolled back to version {version_data['version']}")
        return True

    def consolidate(self, data_loader=None):
        """固化当前知识 (更新EWC)"""
        if data_loader is not None:
            self.ewc.compute_fisher(data_loader)
            logger.info("Knowledge consolidated with EWC")

    def get_learning_stats(self) -> Dict:
        """获取学习统计"""
        return {
            'stats': self.stats.copy(),
            'replay_stats': self.replay.get_stats(),
            'num_versions': len(self.model_versions),
            'current_version': self.current_version
        }

    def save(self, path: str):
        """保存学习器状态"""
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'stats': self.stats,
            'config': self.config,
            'versions': self.model_versions
        }

        torch.save(state, path)
        logger.info(f"Learner saved to {path}")

    def load(self, path: str):
        """加载学习器状态"""
        state = torch.load(path, map_location='cpu', weights_only=False)

        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.stats = state['stats']
        self.model_versions = state.get('versions', [])

        logger.info(f"Learner loaded from {path}")


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Continual Learner Test")
    logger.info("=" * 70)

    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    # 创建学习器
    config = LearningConfig(
        buffer_size=10000,
        batch_size=32,
        min_samples=100,
        update_frequency=10
    )
    learner = ContinualLearner(model, config)

    # 模拟学习过程
    logger.info("\n模拟学习过程...")
    for i in range(500):
        state = np.random.randn(10)
        action = np.random.randn(1)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = np.random.random() < 0.01

        result = learner.step(state, action, reward, next_state, done)

        if result['updated'] and i % 50 == 0:
            logger.info(f"  Step {i}: loss={result['loss']:.4f}, td_error={result['td_error']:.4f}")

    # 获取统计
    stats = learner.get_learning_stats()
    logger.info(f"\n学习统计:")
    logger.info(f"  总步数: {stats['stats']['total_steps']}")
    logger.info(f"  总更新: {stats['stats']['total_updates']}")
    logger.info(f"  平均损失: {stats['stats']['avg_loss']:.4f}")
    logger.info(f"  回放缓冲区: {stats['replay_stats']}")

    logger.info("\n" + "=" * 70)
    logger.info("Test completed!")
    logger.info("=" * 70)
