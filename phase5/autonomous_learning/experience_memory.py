# Phase 5.10: Experience Memory System
# 经验记忆系统 - 累积运行经验用于学习和决策

import logging
import threading
import time
import json
import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque
import random
import math
import numpy as np

logger = logging.getLogger(__name__)


class ExperienceType(Enum):
    """经验类型"""
    NORMAL_OPERATION = "normal"  # 正常运行
    FAULT_HANDLING = "fault"  # 故障处理
    EMERGENCY_RESPONSE = "emergency"  # 应急响应
    OPTIMIZATION = "optimization"  # 优化决策
    RECOVERY = "recovery"  # 恢复操作
    EXPLORATION = "exploration"  # 探索新策略
    HUMAN_OVERRIDE = "human"  # 人工干预


class MemoryPriority(Enum):
    """记忆优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    PERMANENT = 5


class OutcomeQuality(Enum):
    """结果质量"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILURE = "failure"


@dataclass
class Experience:
    """经验记录"""
    experience_id: str
    timestamp: datetime
    experience_type: ExperienceType
    priority: MemoryPriority

    # 状态-动作-奖励
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]] = None

    # 上下文
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 结果评估
    outcome: Optional[OutcomeQuality] = None
    outcome_metrics: Dict[str, float] = field(default_factory=dict)

    # 采样权重
    td_error: float = 0.0  # 时序差分误差
    sampling_weight: float = 1.0

    # 使用统计
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    useful_count: int = 0

    def __post_init__(self):
        if self.outcome is None:
            self.outcome = self._evaluate_outcome()

    def _evaluate_outcome(self) -> OutcomeQuality:
        """评估结果质量"""
        if self.reward >= 0.9:
            return OutcomeQuality.EXCELLENT
        elif self.reward >= 0.7:
            return OutcomeQuality.GOOD
        elif self.reward >= 0.5:
            return OutcomeQuality.ACCEPTABLE
        elif self.reward >= 0.2:
            return OutcomeQuality.POOR
        else:
            return OutcomeQuality.FAILURE

    @property
    def state_hash(self) -> str:
        """计算状态哈希"""
        state_str = json.dumps(self.state, sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()[:16]

    @property
    def action_hash(self) -> str:
        """计算动作哈希"""
        action_str = json.dumps(self.action, sort_keys=True, default=str)
        return hashlib.md5(action_str.encode()).hexdigest()[:16]


@dataclass
class ExperienceCluster:
    """经验聚类"""
    cluster_id: str
    representative: Experience
    experiences: List[str] = field(default_factory=list)  # experience_ids
    centroid: Dict[str, float] = field(default_factory=dict)
    average_reward: float = 0.0
    success_rate: float = 0.0
    count: int = 0


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,  # 优先级指数
        beta: float = 0.4,  # 重要性采样
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self._buffer: deque = deque(maxlen=capacity)
        self._priorities: deque = deque(maxlen=capacity)
        self._max_priority = 1.0

        self._lock = threading.RLock()

    def add(self, experience: Experience, priority: Optional[float] = None):
        """添加经验"""
        with self._lock:
            if priority is None:
                priority = self._max_priority

            self._buffer.append(experience)
            self._priorities.append(priority ** self.alpha)

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], List[float]]:
        """优先级采样"""
        with self._lock:
            if len(self._buffer) == 0:
                return [], [], []

            # 计算采样概率
            priorities = np.array(list(self._priorities))
            probabilities = priorities / priorities.sum()

            # 采样
            indices = np.random.choice(
                len(self._buffer),
                size=min(batch_size, len(self._buffer)),
                replace=False,
                p=probabilities,
            )

            # 计算重要性采样权重
            weights = (len(self._buffer) * probabilities[indices]) ** (-self.beta)
            weights = weights / weights.max()

            # 更新beta
            self.beta = min(1.0, self.beta + self.beta_increment)

            experiences = [self._buffer[i] for i in indices]

            return experiences, indices.tolist(), weights.tolist()

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """更新优先级"""
        with self._lock:
            for idx, td_error in zip(indices, td_errors):
                if 0 <= idx < len(self._priorities):
                    priority = (abs(td_error) + 1e-6) ** self.alpha
                    self._priorities[idx] = priority
                    self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return len(self._buffer)


class ExperienceMemory:
    """经验记忆系统"""

    def __init__(
        self,
        capacity: int = 100000,
        short_term_size: int = 1000,
        long_term_ratio: float = 0.1,
        consolidation_interval: float = 300.0,  # 5 minutes
    ):
        self.capacity = capacity
        self.short_term_size = short_term_size
        self.long_term_ratio = long_term_ratio

        # 短期记忆 (最近的经验)
        self.short_term: deque = deque(maxlen=short_term_size)

        # 长期记忆 (重要的经验)
        self.long_term: Dict[str, Experience] = {}

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=capacity)

        # 经验聚类
        self.clusters: Dict[str, ExperienceCluster] = {}

        # 索引
        self._type_index: Dict[ExperienceType, List[str]] = {t: [] for t in ExperienceType}
        self._priority_index: Dict[MemoryPriority, List[str]] = {p: [] for p in MemoryPriority}
        self._outcome_index: Dict[OutcomeQuality, List[str]] = {o: [] for o in OutcomeQuality}

        # 状态
        self._lock = threading.RLock()
        self._next_experience_id = 1
        self._running = False
        self._consolidation_thread: Optional[threading.Thread] = None
        self._consolidation_interval = consolidation_interval

        # 统计
        self.stats = {
            'experiences_added': 0,
            'experiences_consolidated': 0,
            'experiences_forgotten': 0,
            'queries': 0,
            'successful_recalls': 0,
            'cluster_count': 0,
        }

        logger.info("Experience Memory System initialized")

    def store(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        experience_type: ExperienceType = ExperienceType.NORMAL_OPERATION,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        context: Optional[Dict[str, Any]] = None,
        outcome_metrics: Optional[Dict[str, float]] = None,
    ) -> Experience:
        """存储新经验"""
        with self._lock:
            experience = Experience(
                experience_id=f"exp-{self._next_experience_id:08d}",
                timestamp=datetime.now(),
                experience_type=experience_type,
                priority=priority,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                context=context or {},
                outcome_metrics=outcome_metrics or {},
            )

            self._next_experience_id += 1

            # 添加到短期记忆
            self.short_term.append(experience)

            # 添加到回放缓冲区
            self.replay_buffer.add(experience, priority=abs(reward) + 0.1)

            # 更新索引
            self._type_index[experience_type].append(experience.experience_id)
            self._priority_index[priority].append(experience.experience_id)
            self._outcome_index[experience.outcome].append(experience.experience_id)

            # 关键经验直接进入长期记忆
            if priority in [MemoryPriority.CRITICAL, MemoryPriority.PERMANENT]:
                self.long_term[experience.experience_id] = experience

            self.stats['experiences_added'] += 1

            logger.debug(f"Stored experience: {experience.experience_id}")
            return experience

    def recall(
        self,
        query_state: Optional[Dict[str, Any]] = None,
        experience_type: Optional[ExperienceType] = None,
        min_reward: Optional[float] = None,
        outcome: Optional[OutcomeQuality] = None,
        limit: int = 10,
        include_long_term: bool = True,
        include_short_term: bool = True,
    ) -> List[Experience]:
        """回忆相关经验"""
        with self._lock:
            self.stats['queries'] += 1
            candidates = []

            # 从短期记忆收集
            if include_short_term:
                candidates.extend(list(self.short_term))

            # 从长期记忆收集
            if include_long_term:
                candidates.extend(list(self.long_term.values()))

            # 过滤
            results = []
            for exp in candidates:
                if experience_type and exp.experience_type != experience_type:
                    continue
                if min_reward is not None and exp.reward < min_reward:
                    continue
                if outcome and exp.outcome != outcome:
                    continue
                if query_state:
                    similarity = self._compute_state_similarity(query_state, exp.state)
                    if similarity < 0.5:
                        continue
                results.append(exp)

            # 排序 (按相关性和奖励)
            if query_state:
                results.sort(
                    key=lambda e: (
                        self._compute_state_similarity(query_state, e.state),
                        e.reward,
                    ),
                    reverse=True,
                )
            else:
                results.sort(key=lambda e: e.reward, reverse=True)

            # 更新访问统计
            for exp in results[:limit]:
                exp.access_count += 1
                exp.last_accessed = datetime.now()

            if results:
                self.stats['successful_recalls'] += 1

            return results[:limit]

    def recall_similar(
        self,
        query_state: Dict[str, Any],
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> List[Tuple[Experience, float]]:
        """回忆相似经验"""
        with self._lock:
            self.stats['queries'] += 1
            results = []

            # 搜索所有经验
            all_experiences = list(self.short_term) + list(self.long_term.values())

            for exp in all_experiences:
                similarity = self._compute_state_similarity(query_state, exp.state)
                if similarity >= min_similarity:
                    results.append((exp, similarity))

            # 排序
            results.sort(key=lambda x: x[1], reverse=True)

            if results:
                self.stats['successful_recalls'] += 1

            return results[:top_k]

    def _compute_state_similarity(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
    ) -> float:
        """计算状态相似度"""
        # 获取共同键
        common_keys = set(state1.keys()) & set(state2.keys())
        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            v1, v2 = state1[key], state2[key]

            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # 数值相似度 (归一化差异)
                max_val = max(abs(v1), abs(v2), 1e-6)
                sim = 1.0 - min(abs(v1 - v2) / max_val, 1.0)
            elif v1 == v2:
                sim = 1.0
            else:
                sim = 0.0

            similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def sample_batch(self, batch_size: int) -> List[Experience]:
        """采样一批经验用于训练"""
        experiences, _, _ = self.replay_buffer.sample(batch_size)
        return experiences

    def update_experience(
        self,
        experience_id: str,
        td_error: Optional[float] = None,
        useful: bool = False,
    ):
        """更新经验信息"""
        with self._lock:
            # 查找经验
            exp = None
            for e in self.short_term:
                if e.experience_id == experience_id:
                    exp = e
                    break

            if exp is None:
                exp = self.long_term.get(experience_id)

            if exp is None:
                return

            # 更新TD误差
            if td_error is not None:
                exp.td_error = td_error
                exp.sampling_weight = abs(td_error) + 0.01

            # 更新有用计数
            if useful:
                exp.useful_count += 1

                # 如果经验被认为有用且不在长期记忆中，考虑添加
                if exp.useful_count >= 3 and exp.experience_id not in self.long_term:
                    self.long_term[exp.experience_id] = exp

    def consolidate(self):
        """记忆整合 - 将重要的短期记忆转移到长期记忆"""
        with self._lock:
            # 评估短期记忆中的经验
            for exp in list(self.short_term):
                if exp.experience_id in self.long_term:
                    continue

                # 计算重要性分数
                importance = self._compute_importance(exp)

                # 高重要性的经验进入长期记忆
                if importance >= 0.7:
                    self.long_term[exp.experience_id] = exp
                    self.stats['experiences_consolidated'] += 1

            # 限制长期记忆大小
            max_long_term = int(self.capacity * self.long_term_ratio)
            if len(self.long_term) > max_long_term:
                # 移除最不重要的
                sorted_exps = sorted(
                    self.long_term.values(),
                    key=lambda e: self._compute_importance(e),
                )
                to_remove = len(self.long_term) - max_long_term
                for exp in sorted_exps[:to_remove]:
                    if exp.priority != MemoryPriority.PERMANENT:
                        del self.long_term[exp.experience_id]
                        self.stats['experiences_forgotten'] += 1

            logger.info(f"Memory consolidation: {len(self.long_term)} long-term experiences")

    def _compute_importance(self, experience: Experience) -> float:
        """计算经验重要性"""
        score = 0.0

        # 基于优先级
        priority_weights = {
            MemoryPriority.LOW: 0.2,
            MemoryPriority.NORMAL: 0.4,
            MemoryPriority.HIGH: 0.6,
            MemoryPriority.CRITICAL: 0.8,
            MemoryPriority.PERMANENT: 1.0,
        }
        score += priority_weights.get(experience.priority, 0.4) * 0.3

        # 基于奖励
        score += (experience.reward + 1) / 2 * 0.25

        # 基于结果
        outcome_weights = {
            OutcomeQuality.EXCELLENT: 1.0,
            OutcomeQuality.GOOD: 0.8,
            OutcomeQuality.ACCEPTABLE: 0.5,
            OutcomeQuality.POOR: 0.3,
            OutcomeQuality.FAILURE: 0.4,  # 失败经验也重要
        }
        score += outcome_weights.get(experience.outcome, 0.5) * 0.2

        # 基于访问频率
        if experience.access_count > 0:
            score += min(experience.access_count / 10, 1.0) * 0.15

        # 基于有用计数
        if experience.useful_count > 0:
            score += min(experience.useful_count / 5, 1.0) * 0.1

        return min(score, 1.0)

    def cluster_experiences(self, n_clusters: int = 10):
        """聚类经验"""
        with self._lock:
            all_experiences = list(self.short_term) + list(self.long_term.values())

            if len(all_experiences) < n_clusters:
                return

            # 简单的基于经验类型和结果的聚类
            self.clusters.clear()

            for exp_type in ExperienceType:
                for outcome in OutcomeQuality:
                    cluster_key = f"{exp_type.value}_{outcome.value}"
                    cluster_exps = [
                        e for e in all_experiences
                        if e.experience_type == exp_type and e.outcome == outcome
                    ]

                    if cluster_exps:
                        # 选择代表性经验 (最高奖励)
                        representative = max(cluster_exps, key=lambda e: e.reward)
                        avg_reward = sum(e.reward for e in cluster_exps) / len(cluster_exps)

                        self.clusters[cluster_key] = ExperienceCluster(
                            cluster_id=cluster_key,
                            representative=representative,
                            experiences=[e.experience_id for e in cluster_exps],
                            average_reward=avg_reward,
                            success_rate=len([e for e in cluster_exps if e.reward > 0.5]) / len(cluster_exps),
                            count=len(cluster_exps),
                        )

            self.stats['cluster_count'] = len(self.clusters)
            logger.info(f"Clustered experiences into {len(self.clusters)} clusters")

    def get_best_action_for_state(
        self,
        state: Dict[str, Any],
        experience_type: Optional[ExperienceType] = None,
    ) -> Optional[Dict[str, Any]]:
        """获取给定状态的最佳动作"""
        similar_experiences = self.recall_similar(state, top_k=10, min_similarity=0.4)

        if not similar_experiences:
            return None

        # 过滤经验类型
        if experience_type:
            similar_experiences = [
                (e, s) for e, s in similar_experiences
                if e.experience_type == experience_type
            ]

        if not similar_experiences:
            return None

        # 选择奖励最高的经验的动作
        best_exp = max(similar_experiences, key=lambda x: x[0].reward * x[1])
        return best_exp[0].action

    def start(self):
        """启动记忆整合服务"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._consolidation_thread = threading.Thread(
                target=self._consolidation_loop,
                daemon=True,
            )
            self._consolidation_thread.start()
            logger.info("Experience Memory consolidation service started")

    def stop(self):
        """停止记忆整合服务"""
        with self._lock:
            self._running = False
            if self._consolidation_thread:
                self._consolidation_thread.join(timeout=5.0)
                self._consolidation_thread = None
            logger.info("Experience Memory consolidation service stopped")

    def _consolidation_loop(self):
        """记忆整合循环"""
        while self._running:
            try:
                time.sleep(self._consolidation_interval)
                if self._running:
                    self.consolidate()
                    self.cluster_experiences()
            except Exception as e:
                logger.error(f"Consolidation error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'short_term_size': len(self.short_term),
            'long_term_size': len(self.long_term),
            'replay_buffer_size': len(self.replay_buffer),
            'running': self._running,
        }

    def export_experiences(self, include_long_term: bool = True) -> Dict[str, Any]:
        """导出经验"""
        with self._lock:
            experiences = []

            if include_long_term:
                for exp in self.long_term.values():
                    experiences.append({
                        'experience_id': exp.experience_id,
                        'timestamp': exp.timestamp.isoformat(),
                        'type': exp.experience_type.value,
                        'priority': exp.priority.value,
                        'state': exp.state,
                        'action': exp.action,
                        'reward': exp.reward,
                        'next_state': exp.next_state,
                        'outcome': exp.outcome.value if exp.outcome else None,
                        'context': exp.context,
                    })

            return {
                'experiences': experiences,
                'statistics': self.stats.copy(),
                'exported_at': datetime.now().isoformat(),
            }

    def import_experiences(self, data: Dict[str, Any]):
        """导入经验"""
        with self._lock:
            for exp_data in data.get('experiences', []):
                experience = Experience(
                    experience_id=exp_data['experience_id'],
                    timestamp=datetime.fromisoformat(exp_data['timestamp']),
                    experience_type=ExperienceType(exp_data['type']),
                    priority=MemoryPriority(exp_data['priority']),
                    state=exp_data['state'],
                    action=exp_data['action'],
                    reward=exp_data['reward'],
                    next_state=exp_data.get('next_state'),
                    context=exp_data.get('context', {}),
                )

                # 添加到长期记忆
                self.long_term[experience.experience_id] = experience

                # 更新索引
                self._type_index[experience.experience_type].append(experience.experience_id)
                self._priority_index[experience.priority].append(experience.experience_id)
                self._outcome_index[experience.outcome].append(experience.experience_id)

            logger.info(f"Imported {len(data.get('experiences', []))} experiences")
