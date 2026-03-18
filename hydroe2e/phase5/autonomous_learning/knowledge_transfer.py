# Phase 5.10: Knowledge Transfer Engine
# 知识迁移引擎 - 跨场景、跨设备的知识迁移和泛化

import logging
import threading
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """模式类型"""
    CONTROL_STRATEGY = "control"  # 控制策略模式
    FAULT_RESPONSE = "fault"  # 故障响应模式
    OPTIMIZATION = "optimization"  # 优化模式
    RECOVERY = "recovery"  # 恢复模式
    PREDICTION = "prediction"  # 预测模式
    ANOMALY = "anomaly"  # 异常模式


class TransferStrategy(Enum):
    """迁移策略"""
    DIRECT = "direct"  # 直接迁移
    SCALED = "scaled"  # 缩放迁移
    ADAPTED = "adapted"  # 适应性迁移
    ENSEMBLE = "ensemble"  # 集成迁移
    INCREMENTAL = "incremental"  # 增量迁移


class PatternConfidence(Enum):
    """模式置信度等级"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class Pattern:
    """知识模式"""
    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str = ""

    # 模式内容
    conditions: Dict[str, Any] = field(default_factory=dict)  # 触发条件
    actions: Dict[str, Any] = field(default_factory=dict)  # 推荐动作
    parameters: Dict[str, float] = field(default_factory=dict)  # 相关参数
    constraints: Dict[str, Any] = field(default_factory=dict)  # 约束条件

    # 元数据
    source_context: str = ""  # 来源上下文
    applicable_contexts: List[str] = field(default_factory=list)  # 适用上下文
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

    # 统计
    confidence: PatternConfidence = PatternConfidence.MEDIUM
    success_rate: float = 0.0
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # 迁移信息
    transferable: bool = True
    transfer_adaptations: Dict[str, Any] = field(default_factory=dict)

    def update_statistics(self, success: bool):
        """更新统计信息"""
        self.use_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.success_rate = self.success_count / self.use_count if self.use_count > 0 else 0
        self.last_used = datetime.now()

        # 更新置信度
        if self.use_count >= 10:
            if self.success_rate >= 0.9:
                self.confidence = PatternConfidence.VERY_HIGH
            elif self.success_rate >= 0.7:
                self.confidence = PatternConfidence.HIGH
            elif self.success_rate >= 0.5:
                self.confidence = PatternConfidence.MEDIUM
            elif self.success_rate >= 0.3:
                self.confidence = PatternConfidence.LOW
            else:
                self.confidence = PatternConfidence.VERY_LOW

    @property
    def condition_hash(self) -> str:
        """条件哈希"""
        cond_str = json.dumps(self.conditions, sort_keys=True, default=str)
        return hashlib.md5(cond_str.encode()).hexdigest()[:16]


@dataclass
class TransferResult:
    """迁移结果"""
    transfer_id: str
    source_pattern: str
    target_context: str
    strategy: TransferStrategy
    timestamp: datetime

    # 适应后的模式
    adapted_pattern: Optional[Pattern] = None

    # 评估
    success: bool = False
    similarity_score: float = 0.0
    adaptation_score: float = 0.0
    performance_improvement: float = 0.0

    # 详细信息
    adaptations_made: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class KnowledgeBase:
    """知识库"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.patterns: Dict[str, Pattern] = {}
        self._type_index: Dict[PatternType, List[str]] = {t: [] for t in PatternType}
        self._context_index: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        self._next_pattern_id = 1

    def add_pattern(self, pattern: Pattern) -> str:
        """添加模式"""
        with self._lock:
            if not pattern.pattern_id:
                pattern.pattern_id = f"pat-{self._next_pattern_id:06d}"
                self._next_pattern_id += 1

            self.patterns[pattern.pattern_id] = pattern
            self._type_index[pattern.pattern_type].append(pattern.pattern_id)

            for context in pattern.applicable_contexts:
                self._context_index[context].append(pattern.pattern_id)

            logger.debug(f"Added pattern: {pattern.pattern_id} - {pattern.name}")
            return pattern.pattern_id

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """获取模式"""
        return self.patterns.get(pattern_id)

    def find_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        context: Optional[str] = None,
        min_confidence: PatternConfidence = PatternConfidence.LOW,
        min_success_rate: float = 0.0,
    ) -> List[Pattern]:
        """查找模式"""
        with self._lock:
            results = []

            # 获取候选
            if pattern_type:
                candidates = [self.patterns[pid] for pid in self._type_index.get(pattern_type, [])]
            elif context:
                candidates = [self.patterns[pid] for pid in self._context_index.get(context, [])]
            else:
                candidates = list(self.patterns.values())

            # 过滤
            for pattern in candidates:
                if pattern.confidence.value < min_confidence.value:
                    continue
                if pattern.success_rate < min_success_rate:
                    continue
                if context and context not in pattern.applicable_contexts:
                    # 检查部分匹配
                    if not any(context.startswith(ac) for ac in pattern.applicable_contexts):
                        continue
                results.append(pattern)

            # 排序 (按置信度和成功率)
            results.sort(
                key=lambda p: (p.confidence.value, p.success_rate),
                reverse=True,
            )

            return results

    def match_conditions(
        self,
        current_state: Dict[str, Any],
        pattern_type: Optional[PatternType] = None,
    ) -> List[Tuple[Pattern, float]]:
        """匹配当前状态与模式条件"""
        with self._lock:
            matches = []

            candidates = self.find_patterns(pattern_type=pattern_type)

            for pattern in candidates:
                match_score = self._compute_condition_match(current_state, pattern.conditions)
                if match_score > 0.5:  # 匹配阈值
                    matches.append((pattern, match_score))

            matches.sort(key=lambda x: x[1], reverse=True)
            return matches

    def _compute_condition_match(
        self,
        state: Dict[str, Any],
        conditions: Dict[str, Any],
    ) -> float:
        """计算条件匹配度"""
        if not conditions:
            return 0.5  # 无条件时默认中等匹配

        matched = 0
        total = 0

        for key, condition in conditions.items():
            total += 1
            state_value = state.get(key)

            if state_value is None:
                continue

            if isinstance(condition, dict):
                # 范围条件
                if 'min' in condition and state_value < condition['min']:
                    continue
                if 'max' in condition and state_value > condition['max']:
                    continue
                if 'equals' in condition and state_value != condition['equals']:
                    continue
                matched += 1
            elif callable(condition):
                if condition(state_value):
                    matched += 1
            elif state_value == condition:
                matched += 1

        return matched / total if total > 0 else 0.0

    def export(self) -> Dict[str, Any]:
        """导出知识库"""
        with self._lock:
            return {
                'name': self.name,
                'patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'pattern_type': p.pattern_type.value,
                        'name': p.name,
                        'description': p.description,
                        'conditions': p.conditions,
                        'actions': p.actions,
                        'parameters': p.parameters,
                        'constraints': p.constraints,
                        'source_context': p.source_context,
                        'applicable_contexts': p.applicable_contexts,
                        'confidence': p.confidence.value,
                        'success_rate': p.success_rate,
                        'use_count': p.use_count,
                        'transferable': p.transferable,
                    }
                    for p in self.patterns.values()
                ],
                'exported_at': datetime.now().isoformat(),
            }

    def import_patterns(self, data: Dict[str, Any]):
        """导入模式"""
        with self._lock:
            for p_data in data.get('patterns', []):
                pattern = Pattern(
                    pattern_id=p_data['pattern_id'],
                    pattern_type=PatternType(p_data['pattern_type']),
                    name=p_data['name'],
                    description=p_data.get('description', ''),
                    conditions=p_data.get('conditions', {}),
                    actions=p_data.get('actions', {}),
                    parameters=p_data.get('parameters', {}),
                    constraints=p_data.get('constraints', {}),
                    source_context=p_data.get('source_context', ''),
                    applicable_contexts=p_data.get('applicable_contexts', []),
                    confidence=PatternConfidence(p_data.get('confidence', 3)),
                    success_rate=p_data.get('success_rate', 0.0),
                    use_count=p_data.get('use_count', 0),
                    transferable=p_data.get('transferable', True),
                )
                self.add_pattern(pattern)

            logger.info(f"Imported {len(data.get('patterns', []))} patterns")


class KnowledgeTransferEngine:
    """知识迁移引擎"""

    def __init__(self):
        # 知识库
        self.knowledge_base = KnowledgeBase("main")
        self.domain_bases: Dict[str, KnowledgeBase] = {}

        # 迁移历史
        self._transfer_history: List[TransferResult] = []

        # 上下文映射
        self._context_mappings: Dict[str, Dict[str, str]] = {}

        # 状态
        self._lock = threading.RLock()
        self._next_transfer_id = 1

        # 统计
        self.stats = {
            'patterns_learned': 0,
            'transfers_attempted': 0,
            'transfers_successful': 0,
            'average_transfer_score': 0.0,
        }

        logger.info("Knowledge Transfer Engine initialized")

    def learn_pattern(
        self,
        pattern_type: PatternType,
        name: str,
        conditions: Dict[str, Any],
        actions: Dict[str, Any],
        source_context: str,
        parameters: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> Pattern:
        """从经验中学习模式"""
        with self._lock:
            pattern = Pattern(
                pattern_id="",
                pattern_type=pattern_type,
                name=name,
                description=description,
                conditions=conditions,
                actions=actions,
                parameters=parameters or {},
                constraints=constraints or {},
                source_context=source_context,
                applicable_contexts=[source_context],
            )

            pattern_id = self.knowledge_base.add_pattern(pattern)
            pattern.pattern_id = pattern_id
            self.stats['patterns_learned'] += 1

            logger.info(f"Learned pattern: {pattern_id} - {name}")
            return pattern

    def extract_patterns_from_experiences(
        self,
        experiences: List[Dict[str, Any]],
        min_occurrences: int = 3,
    ) -> List[Pattern]:
        """从经验列表中提取模式"""
        patterns = []

        # 按结果分组经验
        success_experiences = [e for e in experiences if e.get('reward', 0) > 0.7]
        failure_experiences = [e for e in experiences if e.get('reward', 0) < 0.3]

        # 从成功经验中提取模式
        if len(success_experiences) >= min_occurrences:
            # 找出共同的状态特征
            common_conditions = self._extract_common_features(
                [e.get('state', {}) for e in success_experiences]
            )

            # 找出共同的动作
            common_actions = self._extract_common_features(
                [e.get('action', {}) for e in success_experiences]
            )

            if common_conditions and common_actions:
                pattern = self.learn_pattern(
                    pattern_type=PatternType.CONTROL_STRATEGY,
                    name=f"Success Pattern ({len(success_experiences)} samples)",
                    conditions=common_conditions,
                    actions=common_actions,
                    source_context="auto_extracted",
                    description="Automatically extracted from successful experiences",
                )
                patterns.append(pattern)

        # 从失败经验中提取需要避免的模式
        if len(failure_experiences) >= min_occurrences:
            common_conditions = self._extract_common_features(
                [e.get('state', {}) for e in failure_experiences]
            )

            if common_conditions:
                pattern = self.learn_pattern(
                    pattern_type=PatternType.ANOMALY,
                    name=f"Failure Pattern ({len(failure_experiences)} samples)",
                    conditions=common_conditions,
                    actions={'avoid': True},
                    source_context="auto_extracted",
                    description="Conditions to avoid (from failure experiences)",
                )
                patterns.append(pattern)

        return patterns

    def _extract_common_features(
        self,
        data_list: List[Dict[str, Any]],
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """提取共同特征"""
        if not data_list:
            return {}

        # 收集所有键
        all_keys: Set[str] = set()
        for data in data_list:
            all_keys.update(data.keys())

        common = {}

        for key in all_keys:
            values = [d.get(key) for d in data_list if key in d]
            occurrence_rate = len(values) / len(data_list)

            if occurrence_rate < threshold:
                continue

            # 检查值是否一致
            if all(isinstance(v, (int, float)) for v in values):
                # 数值型: 使用范围
                min_val = min(values)
                max_val = max(values)
                if max_val - min_val < 0.1 * (abs(max_val) + abs(min_val) + 1e-6):
                    # 值相近
                    common[key] = {'min': min_val * 0.95, 'max': max_val * 1.05}
            elif len(set(str(v) for v in values)) == 1:
                # 所有值相同
                common[key] = values[0]

        return common

    def transfer(
        self,
        pattern_id: str,
        target_context: str,
        strategy: TransferStrategy = TransferStrategy.ADAPTED,
        adaptations: Optional[Dict[str, Any]] = None,
    ) -> TransferResult:
        """迁移模式到新上下文"""
        with self._lock:
            self.stats['transfers_attempted'] += 1
            transfer_id = f"tfr-{self._next_transfer_id:06d}"
            self._next_transfer_id += 1

            pattern = self.knowledge_base.get_pattern(pattern_id)
            if not pattern:
                return TransferResult(
                    transfer_id=transfer_id,
                    source_pattern=pattern_id,
                    target_context=target_context,
                    strategy=strategy,
                    timestamp=datetime.now(),
                    success=False,
                    warnings=["Pattern not found"],
                )

            if not pattern.transferable:
                return TransferResult(
                    transfer_id=transfer_id,
                    source_pattern=pattern_id,
                    target_context=target_context,
                    strategy=strategy,
                    timestamp=datetime.now(),
                    success=False,
                    warnings=["Pattern is not transferable"],
                )

            # 计算上下文相似度
            similarity = self._compute_context_similarity(
                pattern.source_context,
                target_context,
            )

            # 执行迁移
            if strategy == TransferStrategy.DIRECT:
                adapted_pattern = self._direct_transfer(pattern, target_context)
            elif strategy == TransferStrategy.SCALED:
                adapted_pattern = self._scaled_transfer(pattern, target_context, adaptations)
            elif strategy == TransferStrategy.ADAPTED:
                adapted_pattern = self._adapted_transfer(pattern, target_context, adaptations)
            elif strategy == TransferStrategy.ENSEMBLE:
                adapted_pattern = self._ensemble_transfer(pattern, target_context)
            else:
                adapted_pattern = self._incremental_transfer(pattern, target_context)

            # 添加到知识库
            if adapted_pattern:
                self.knowledge_base.add_pattern(adapted_pattern)
                self.stats['transfers_successful'] += 1

            result = TransferResult(
                transfer_id=transfer_id,
                source_pattern=pattern_id,
                target_context=target_context,
                strategy=strategy,
                timestamp=datetime.now(),
                adapted_pattern=adapted_pattern,
                success=adapted_pattern is not None,
                similarity_score=similarity,
                adaptation_score=0.8 if adapted_pattern else 0.0,
                adaptations_made=adaptations or {},
            )

            self._transfer_history.append(result)

            # 更新统计
            successful = len([t for t in self._transfer_history if t.success])
            self.stats['average_transfer_score'] = successful / len(self._transfer_history)

            logger.info(f"Transfer {transfer_id}: {pattern_id} -> {target_context}, success={result.success}")
            return result

    def _direct_transfer(
        self,
        pattern: Pattern,
        target_context: str,
    ) -> Pattern:
        """直接迁移 - 不修改参数"""
        new_pattern = Pattern(
            pattern_id="",
            pattern_type=pattern.pattern_type,
            name=f"{pattern.name} (transferred)",
            description=f"Transferred from {pattern.source_context}",
            conditions=pattern.conditions.copy(),
            actions=pattern.actions.copy(),
            parameters=pattern.parameters.copy(),
            constraints=pattern.constraints.copy(),
            source_context=pattern.source_context,
            applicable_contexts=[target_context],
            confidence=PatternConfidence.LOW,  # 初始置信度较低
            transferable=True,
            transfer_adaptations={'strategy': 'direct'},
        )
        return new_pattern

    def _scaled_transfer(
        self,
        pattern: Pattern,
        target_context: str,
        adaptations: Optional[Dict[str, Any]] = None,
    ) -> Pattern:
        """缩放迁移 - 按比例调整参数"""
        adaptations = adaptations or {}
        scale_factor = adaptations.get('scale_factor', 1.0)

        new_parameters = {}
        for key, value in pattern.parameters.items():
            new_parameters[key] = value * scale_factor

        new_pattern = Pattern(
            pattern_id="",
            pattern_type=pattern.pattern_type,
            name=f"{pattern.name} (scaled x{scale_factor})",
            description=f"Scaled transfer from {pattern.source_context}",
            conditions=pattern.conditions.copy(),
            actions=pattern.actions.copy(),
            parameters=new_parameters,
            constraints=pattern.constraints.copy(),
            source_context=pattern.source_context,
            applicable_contexts=[target_context],
            confidence=PatternConfidence.LOW,
            transferable=True,
            transfer_adaptations={'strategy': 'scaled', 'scale_factor': scale_factor},
        )
        return new_pattern

    def _adapted_transfer(
        self,
        pattern: Pattern,
        target_context: str,
        adaptations: Optional[Dict[str, Any]] = None,
    ) -> Pattern:
        """适应性迁移 - 根据目标上下文调整"""
        adaptations = adaptations or {}

        # 应用映射
        mapping = self._context_mappings.get(
            f"{pattern.source_context}->{target_context}",
            {},
        )

        # 调整条件
        new_conditions = {}
        for key, value in pattern.conditions.items():
            mapped_key = mapping.get(key, key)
            new_conditions[mapped_key] = value

        # 调整动作
        new_actions = {}
        for key, value in pattern.actions.items():
            mapped_key = mapping.get(key, key)
            new_actions[mapped_key] = value

        # 调整参数
        new_parameters = {}
        for key, value in pattern.parameters.items():
            mapped_key = mapping.get(key, key)
            adaptation = adaptations.get(key, 1.0)
            if isinstance(adaptation, (int, float)):
                new_parameters[mapped_key] = value * adaptation
            else:
                new_parameters[mapped_key] = value

        new_pattern = Pattern(
            pattern_id="",
            pattern_type=pattern.pattern_type,
            name=f"{pattern.name} (adapted)",
            description=f"Adapted transfer from {pattern.source_context}",
            conditions=new_conditions,
            actions=new_actions,
            parameters=new_parameters,
            constraints=pattern.constraints.copy(),
            source_context=pattern.source_context,
            applicable_contexts=[target_context],
            confidence=PatternConfidence.LOW,
            transferable=True,
            transfer_adaptations={'strategy': 'adapted', 'mapping': mapping, 'adaptations': adaptations},
        )
        return new_pattern

    def _ensemble_transfer(
        self,
        pattern: Pattern,
        target_context: str,
    ) -> Pattern:
        """集成迁移 - 结合多个相似模式"""
        # 找到类似的模式
        similar_patterns = self.knowledge_base.find_patterns(
            pattern_type=pattern.pattern_type,
            min_confidence=PatternConfidence.MEDIUM,
        )

        if len(similar_patterns) < 2:
            return self._direct_transfer(pattern, target_context)

        # 集成参数
        ensemble_parameters = {}
        for key in pattern.parameters:
            values = [p.parameters.get(key, pattern.parameters[key]) for p in similar_patterns if key in p.parameters]
            if values:
                ensemble_parameters[key] = sum(values) / len(values)

        new_pattern = Pattern(
            pattern_id="",
            pattern_type=pattern.pattern_type,
            name=f"{pattern.name} (ensemble)",
            description=f"Ensemble transfer from {len(similar_patterns)} patterns",
            conditions=pattern.conditions.copy(),
            actions=pattern.actions.copy(),
            parameters=ensemble_parameters,
            constraints=pattern.constraints.copy(),
            source_context=pattern.source_context,
            applicable_contexts=[target_context],
            confidence=PatternConfidence.MEDIUM,
            transferable=True,
            transfer_adaptations={'strategy': 'ensemble', 'source_count': len(similar_patterns)},
        )
        return new_pattern

    def _incremental_transfer(
        self,
        pattern: Pattern,
        target_context: str,
    ) -> Pattern:
        """增量迁移 - 保守的参数调整"""
        # 小幅调整参数
        new_parameters = {}
        for key, value in pattern.parameters.items():
            # 保守调整: ±10%
            new_parameters[key] = value * (1.0 + np.random.uniform(-0.1, 0.1))

        new_pattern = Pattern(
            pattern_id="",
            pattern_type=pattern.pattern_type,
            name=f"{pattern.name} (incremental)",
            description=f"Incremental transfer from {pattern.source_context}",
            conditions=pattern.conditions.copy(),
            actions=pattern.actions.copy(),
            parameters=new_parameters,
            constraints=pattern.constraints.copy(),
            source_context=pattern.source_context,
            applicable_contexts=[target_context],
            confidence=PatternConfidence.LOW,
            transferable=True,
            transfer_adaptations={'strategy': 'incremental'},
        )
        return new_pattern

    def _compute_context_similarity(
        self,
        context1: str,
        context2: str,
    ) -> float:
        """计算上下文相似度"""
        # 简单的字符串相似度
        if context1 == context2:
            return 1.0

        # 基于前缀匹配
        common_prefix = 0
        for c1, c2 in zip(context1, context2):
            if c1 == c2:
                common_prefix += 1
            else:
                break

        max_len = max(len(context1), len(context2))
        return common_prefix / max_len if max_len > 0 else 0.0

    def register_context_mapping(
        self,
        source_context: str,
        target_context: str,
        mapping: Dict[str, str],
    ):
        """注册上下文映射"""
        key = f"{source_context}->{target_context}"
        self._context_mappings[key] = mapping
        logger.info(f"Registered context mapping: {key}")

    def get_applicable_patterns(
        self,
        current_state: Dict[str, Any],
        context: str,
        pattern_type: Optional[PatternType] = None,
    ) -> List[Tuple[Pattern, float]]:
        """获取适用于当前状态的模式"""
        matches = self.knowledge_base.match_conditions(current_state, pattern_type)

        # 过滤上下文
        context_matches = []
        for pattern, score in matches:
            if context in pattern.applicable_contexts:
                context_matches.append((pattern, score))
            else:
                # 检查是否可以迁移
                for applicable in pattern.applicable_contexts:
                    similarity = self._compute_context_similarity(applicable, context)
                    if similarity > 0.5:
                        context_matches.append((pattern, score * similarity))
                        break

        return context_matches

    def update_pattern_feedback(
        self,
        pattern_id: str,
        success: bool,
        performance_delta: float = 0.0,
    ):
        """更新模式反馈"""
        pattern = self.knowledge_base.get_pattern(pattern_id)
        if pattern:
            pattern.update_statistics(success)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'total_patterns': len(self.knowledge_base.patterns),
            'transfer_history_size': len(self._transfer_history),
            'context_mappings': len(self._context_mappings),
        }

    def create_water_network_patterns(self, num_pools: int) -> List[Pattern]:
        """创建智能水网基础模式"""
        patterns = []

        # 水位控制模式
        patterns.append(self.learn_pattern(
            pattern_type=PatternType.CONTROL_STRATEGY,
            name="Normal Water Level Control",
            conditions={
                'water_level': {'min': 2.0, 'max': 4.0},
                'inflow_rate': {'min': 0, 'max': 20},
            },
            actions={
                'gate_adjustment': 'proportional',
                'response_mode': 'normal',
            },
            parameters={
                'kp': 1.0,
                'ki': 0.1,
                'kd': 0.05,
                'deadband': 0.1,
            },
            source_context='water_network_standard',
            description="Standard water level control strategy",
        ))

        # 洪水响应模式
        patterns.append(self.learn_pattern(
            pattern_type=PatternType.FAULT_RESPONSE,
            name="Flood Response",
            conditions={
                'water_level': {'min': 4.5},
                'rising_rate': {'min': 0.1},
            },
            actions={
                'gate_action': 'open_maximum',
                'alert_level': 'high',
                'coordination': 'downstream_priority',
            },
            parameters={
                'emergency_threshold': 5.0,
                'max_gate_speed': 10.0,
            },
            source_context='water_network_emergency',
            description="Flood condition response pattern",
        ))

        # 干旱响应模式
        patterns.append(self.learn_pattern(
            pattern_type=PatternType.FAULT_RESPONSE,
            name="Drought Response",
            conditions={
                'water_level': {'max': 1.5},
                'inflow_rate': {'max': 2.0},
            },
            actions={
                'gate_action': 'minimize_outflow',
                'alert_level': 'medium',
                'conservation_mode': True,
            },
            parameters={
                'minimum_level': 1.0,
                'conservation_factor': 0.5,
            },
            source_context='water_network_emergency',
            description="Drought condition response pattern",
        ))

        # 恢复模式
        patterns.append(self.learn_pattern(
            pattern_type=PatternType.RECOVERY,
            name="Post-Event Recovery",
            conditions={
                'event_cleared': True,
                'system_stable': True,
            },
            actions={
                'mode': 'gradual_return',
                'target': 'normal_operation',
            },
            parameters={
                'recovery_rate': 0.1,
                'stability_threshold': 0.05,
            },
            source_context='water_network_recovery',
            description="Recovery to normal operation pattern",
        ))

        # 优化模式
        patterns.append(self.learn_pattern(
            pattern_type=PatternType.OPTIMIZATION,
            name="Energy Optimization",
            conditions={
                'system_stable': True,
                'optimization_enabled': True,
            },
            actions={
                'objective': 'minimize_energy',
                'constraint': 'maintain_levels',
            },
            parameters={
                'energy_weight': 0.3,
                'level_weight': 0.7,
            },
            source_context='water_network_optimization',
            description="Energy consumption optimization pattern",
        ))

        logger.info(f"Created {len(patterns)} water network patterns")
        return patterns
