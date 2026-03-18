"""
增强型故障诊断引擎 (Enhanced Diagnosis Engine)

扩展功能:
1. 更多故障类型 (级联故障、间歇性故障、复合故障)
2. 模式学习与匹配
3. 多故障关联分析
4. 置信度动态调整
5. 虚警识别与过滤
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class ExtendedFaultType(Enum):
    """扩展故障类型"""
    # 基础故障
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_NOISE = "sensor_noise"
    SENSOR_BIAS = "sensor_bias"

    ACTUATOR_STUCK = "actuator_stuck"
    ACTUATOR_DELAY = "actuator_delay"
    ACTUATOR_DEGRADATION = "actuator_degradation"
    ACTUATOR_OSCILLATION = "actuator_oscillation"

    COMMUNICATION_LOSS = "communication_loss"
    COMMUNICATION_DELAY = "communication_delay"
    COMMUNICATION_CORRUPT = "communication_corrupt"

    # 高级故障类型
    CASCADE_FAULT = "cascade_fault"          # 级联故障
    INTERMITTENT_FAULT = "intermittent_fault"  # 间歇性故障
    COMPOUND_FAULT = "compound_fault"        # 复合故障
    INCIPIENT_FAULT = "incipient_fault"      # 早期/潜在故障
    PROGRESSIVE_FAULT = "progressive_fault"  # 渐进性故障

    # 系统级故障
    HYDRAULIC_ANOMALY = "hydraulic_anomaly"  # 水力异常
    CONTROL_LOOP_FAULT = "control_loop_fault"  # 控制回路故障
    MODEL_MISMATCH = "model_mismatch"        # 模型失配

    # 安全相关
    CYBER_ATTACK = "cyber_attack"
    FALSE_ALARM = "false_alarm"
    UNKNOWN = "unknown"


class FaultSeverity(Enum):
    """故障严重程度"""
    NEGLIGIBLE = 0   # 可忽略
    LOW = 1          # 低
    MEDIUM = 2       # 中
    HIGH = 3         # 高
    CRITICAL = 4     # 严重
    EMERGENCY = 5    # 紧急


@dataclass
class FaultPattern:
    """故障模式特征"""
    pattern_id: str
    fault_type: ExtendedFaultType
    signature: Dict[str, any]  # 特征签名
    temporal_pattern: List[float]  # 时序模式
    correlation_vars: List[str]  # 相关变量
    confidence_threshold: float = 0.7
    occurrence_count: int = 0
    last_seen: Optional[str] = None

    def match_score(self, observed: Dict) -> float:
        """计算与观测数据的匹配度"""
        score = 0.0
        matches = 0

        for key, expected in self.signature.items():
            if key in observed:
                if isinstance(expected, tuple):  # 范围匹配
                    if expected[0] <= observed[key] <= expected[1]:
                        score += 1.0
                    else:
                        deviation = min(
                            abs(observed[key] - expected[0]),
                            abs(observed[key] - expected[1])
                        ) / (expected[1] - expected[0] + 1e-6)
                        score += max(0, 1.0 - deviation)
                elif isinstance(expected, (int, float)):
                    deviation = abs(observed[key] - expected) / (abs(expected) + 1e-6)
                    score += max(0, 1.0 - deviation)
                else:
                    score += 1.0 if observed[key] == expected else 0.0
                matches += 1

        return score / max(matches, 1)


@dataclass
class CompoundFault:
    """复合故障"""
    fault_id: str
    component_faults: List[str]  # 组成故障ID列表
    interaction_type: str  # 交互类型: additive, multiplicative, masking
    combined_severity: FaultSeverity
    root_cause: Optional[str] = None
    propagation_path: List[str] = field(default_factory=list)


@dataclass
class CascadeFault:
    """级联故障"""
    fault_id: str
    trigger_fault: str  # 触发故障
    affected_components: List[str]  # 受影响组件
    propagation_time: float  # 传播时间 (秒)
    propagation_probability: float  # 传播概率
    current_stage: int = 0  # 当前阶段
    max_stages: int = 3


@dataclass
class DiagnosisConfidence:
    """诊断置信度"""
    base_confidence: float  # 基础置信度
    pattern_match_boost: float = 0.0  # 模式匹配加成
    historical_boost: float = 0.0  # 历史加成
    cross_validation_boost: float = 0.0  # 交叉验证加成
    temporal_consistency: float = 1.0  # 时序一致性
    false_alarm_penalty: float = 0.0  # 虚警惩罚

    @property
    def final_confidence(self) -> float:
        """计算最终置信度"""
        raw = (self.base_confidence +
               self.pattern_match_boost +
               self.historical_boost +
               self.cross_validation_boost)
        adjusted = raw * self.temporal_consistency - self.false_alarm_penalty
        return max(0.0, min(1.0, adjusted))


@dataclass
class EnhancedDiagnosisResult:
    """增强型诊断结果"""
    fault_id: str
    fault_type: ExtendedFaultType
    component: str
    severity: FaultSeverity
    confidence: DiagnosisConfidence
    root_causes: List[str]
    contributing_factors: List[str]
    recommended_actions: List[str]
    is_compound: bool = False
    is_cascade: bool = False
    related_faults: List[str] = field(default_factory=list)
    diagnosis_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


class EnhancedDiagnosisEngine:
    """
    增强型故障诊断引擎

    核心能力:
    1. 多故障类型识别 (15+种故障)
    2. 模式学习与匹配
    3. 级联故障追踪
    4. 复合故障分解
    5. 虚警过滤
    6. 置信度动态调整
    """

    def __init__(self, num_pools: int = 3):
        """初始化增强诊断引擎"""
        self.num_pools = num_pools

        # 故障模式库
        self.pattern_library: Dict[str, FaultPattern] = {}
        self._init_pattern_library()

        # 诊断规则
        self.diagnosis_rules = self._init_diagnosis_rules()

        # 历史记录
        self.diagnosis_history: List[EnhancedDiagnosisResult] = []
        self.false_alarm_history: List[str] = []

        # 活跃故障追踪
        self.active_faults: Dict[str, EnhancedDiagnosisResult] = {}
        self.cascade_tracking: Dict[str, CascadeFault] = {}

        # 统计
        self.stats = {
            'total_diagnoses': 0,
            'confirmed_faults': 0,
            'false_alarms': 0,
            'cascade_detections': 0,
            'compound_detections': 0
        }

        logger.info("[EnhancedDiagnosisEngine] 增强诊断引擎初始化完成")

    def _init_pattern_library(self):
        """初始化故障模式库"""
        # 传感器漂移模式
        self.pattern_library['sensor_drift_slow'] = FaultPattern(
            pattern_id='sensor_drift_slow',
            fault_type=ExtendedFaultType.SENSOR_DRIFT,
            signature={
                'drift_rate': (0.001, 0.01),  # m/h
                'direction': 'monotonic',
                'noise_level': (0, 0.05)
            },
            temporal_pattern=[0.0, 0.1, 0.2, 0.3, 0.4],  # 渐进模式
            correlation_vars=['temperature', 'humidity']
        )

        self.pattern_library['sensor_drift_fast'] = FaultPattern(
            pattern_id='sensor_drift_fast',
            fault_type=ExtendedFaultType.SENSOR_DRIFT,
            signature={
                'drift_rate': (0.01, 0.1),
                'direction': 'monotonic'
            },
            temporal_pattern=[0.0, 0.3, 0.6, 0.9, 1.0],
            correlation_vars=[]
        )

        # 传感器卡死模式
        self.pattern_library['sensor_stuck_value'] = FaultPattern(
            pattern_id='sensor_stuck_value',
            fault_type=ExtendedFaultType.SENSOR_STUCK,
            signature={
                'variance': (0, 0.001),
                'duration': (60, float('inf'))  # 至少60秒
            },
            temporal_pattern=[1.0, 1.0, 1.0, 1.0, 1.0],  # 恒定模式
            correlation_vars=[]
        )

        # 执行器卡死模式
        self.pattern_library['actuator_stuck_position'] = FaultPattern(
            pattern_id='actuator_stuck_position',
            fault_type=ExtendedFaultType.ACTUATOR_STUCK,
            signature={
                'command_response_error': (0.1, 1.0),
                'position_variance': (0, 0.01)
            },
            temporal_pattern=[1.0, 1.0, 1.0, 1.0, 1.0],
            correlation_vars=['hydraulic_pressure', 'motor_current']
        )

        # 执行器退化模式
        self.pattern_library['actuator_degradation'] = FaultPattern(
            pattern_id='actuator_degradation',
            fault_type=ExtendedFaultType.ACTUATOR_DEGRADATION,
            signature={
                'response_time_increase': (0.1, 0.5),  # 10-50%增加
                'efficiency_decrease': (0.05, 0.3)
            },
            temporal_pattern=[0.0, 0.05, 0.1, 0.15, 0.2],  # 缓慢退化
            correlation_vars=['operating_hours', 'cycle_count']
        )

        # 级联故障模式
        self.pattern_library['cascade_upstream'] = FaultPattern(
            pattern_id='cascade_upstream',
            fault_type=ExtendedFaultType.CASCADE_FAULT,
            signature={
                'propagation_direction': 'downstream',
                'delay_per_stage': (30, 300)  # 30-300秒
            },
            temporal_pattern=[1.0, 0.8, 0.6, 0.4, 0.2],  # 衰减模式
            correlation_vars=['upstream_level', 'flow_rate']
        )

        # 间歇性故障模式
        self.pattern_library['intermittent_periodic'] = FaultPattern(
            pattern_id='intermittent_periodic',
            fault_type=ExtendedFaultType.INTERMITTENT_FAULT,
            signature={
                'on_duration': (10, 120),
                'off_duration': (30, 600),
                'periodicity': (0.5, 1.0)  # 周期性强度
            },
            temporal_pattern=[1.0, 0.0, 1.0, 0.0, 1.0],  # 开关模式
            correlation_vars=['temperature', 'vibration']
        )

        # 水力异常模式
        self.pattern_library['hydraulic_blockage'] = FaultPattern(
            pattern_id='hydraulic_blockage',
            fault_type=ExtendedFaultType.HYDRAULIC_ANOMALY,
            signature={
                'flow_reduction': (0.2, 0.8),
                'pressure_increase': (0.1, 0.5),
                'level_accumulation': True
            },
            temporal_pattern=[0.0, 0.2, 0.5, 0.8, 1.0],
            correlation_vars=['upstream_level', 'downstream_level']
        )

    def _init_diagnosis_rules(self) -> Dict:
        """初始化诊断规则"""
        return {
            # 传感器故障规则
            'rule_sensor_drift': {
                'conditions': [
                    ('value_trend', 'monotonic'),
                    ('duration', '>', 300),
                    ('deviation', '>', 0.1)
                ],
                'fault_type': ExtendedFaultType.SENSOR_DRIFT,
                'severity': FaultSeverity.MEDIUM,
                'confidence': 0.8
            },

            'rule_sensor_stuck': {
                'conditions': [
                    ('variance', '<', 0.001),
                    ('duration', '>', 60),
                    ('neighbors_active', True)
                ],
                'fault_type': ExtendedFaultType.SENSOR_STUCK,
                'severity': FaultSeverity.HIGH,
                'confidence': 0.85
            },

            'rule_sensor_noise': {
                'conditions': [
                    ('noise_ratio', '>', 0.3),
                    ('signal_quality', '<', 0.5)
                ],
                'fault_type': ExtendedFaultType.SENSOR_NOISE,
                'severity': FaultSeverity.LOW,
                'confidence': 0.75
            },

            # 执行器故障规则
            'rule_actuator_stuck': {
                'conditions': [
                    ('command_response_error', '>', 0.1),
                    ('duration', '>', 30),
                    ('motor_current', 'abnormal')
                ],
                'fault_type': ExtendedFaultType.ACTUATOR_STUCK,
                'severity': FaultSeverity.CRITICAL,
                'confidence': 0.9
            },

            'rule_actuator_degradation': {
                'conditions': [
                    ('response_time', '>', 'baseline * 1.2'),
                    ('efficiency', '<', 'baseline * 0.9'),
                    ('trend', 'worsening')
                ],
                'fault_type': ExtendedFaultType.ACTUATOR_DEGRADATION,
                'severity': FaultSeverity.MEDIUM,
                'confidence': 0.7
            },

            # 级联故障规则
            'rule_cascade_detection': {
                'conditions': [
                    ('fault_sequence', 'detected'),
                    ('propagation_pattern', 'spatial'),
                    ('time_correlation', '>', 0.7)
                ],
                'fault_type': ExtendedFaultType.CASCADE_FAULT,
                'severity': FaultSeverity.HIGH,
                'confidence': 0.75
            },

            # 间歇性故障规则
            'rule_intermittent': {
                'conditions': [
                    ('fault_recurrence', '>', 2),
                    ('recovery_observed', True),
                    ('pattern', 'periodic_or_random')
                ],
                'fault_type': ExtendedFaultType.INTERMITTENT_FAULT,
                'severity': FaultSeverity.MEDIUM,
                'confidence': 0.7
            },

            # 虚警识别规则
            'rule_false_alarm': {
                'conditions': [
                    ('single_sensor', True),
                    ('cross_validation', 'fail'),
                    ('physical_plausibility', 'low'),
                    ('duration', '<', 30)
                ],
                'fault_type': ExtendedFaultType.FALSE_ALARM,
                'severity': FaultSeverity.NEGLIGIBLE,
                'confidence': 0.6
            }
        }

    def diagnose(self,
                 anomaly_reports: List[Dict],
                 system_state: Dict,
                 historical_context: Dict = None) -> List[EnhancedDiagnosisResult]:
        """
        执行增强型故障诊断

        Args:
            anomaly_reports: 异常报告列表
            system_state: 当前系统状态
            historical_context: 历史上下文

        Returns:
            诊断结果列表
        """
        start_time = time.time()
        results = []

        if not anomaly_reports:
            return results

        # 1. 预处理和分组
        grouped_anomalies = self._group_anomalies(anomaly_reports)

        # 2. 虚警过滤
        filtered_anomalies = self._filter_false_alarms(
            grouped_anomalies, system_state
        )

        # 3. 单故障诊断
        single_faults = []
        for component, anomalies in filtered_anomalies.items():
            fault = self._diagnose_single_fault(
                component, anomalies, system_state
            )
            if fault:
                single_faults.append(fault)

        # 4. 复合故障检测
        compound_faults = self._detect_compound_faults(single_faults)

        # 5. 级联故障追踪
        cascade_faults = self._track_cascade_faults(
            single_faults, system_state
        )

        # 6. 置信度调整
        for fault in single_faults:
            self._adjust_confidence(fault, historical_context)

        # 7. 合并结果
        results.extend(single_faults)
        results.extend(compound_faults)
        results.extend(cascade_faults)

        # 8. 更新历史和统计
        diagnosis_time = time.time() - start_time
        for result in results:
            result.diagnosis_time = diagnosis_time
            self.diagnosis_history.append(result)
            self.active_faults[result.fault_id] = result

        self.stats['total_diagnoses'] += len(results)

        return results

    def _group_anomalies(self, anomaly_reports: List[Dict]) -> Dict[str, List[Dict]]:
        """按组件分组异常"""
        grouped = {}
        for report in anomaly_reports:
            component = report.get('component', 'unknown')
            if component not in grouped:
                grouped[component] = []
            grouped[component].append(report)
        return grouped

    def _filter_false_alarms(self,
                             grouped_anomalies: Dict[str, List[Dict]],
                             system_state: Dict) -> Dict[str, List[Dict]]:
        """过滤虚警"""
        filtered = {}

        for component, anomalies in grouped_anomalies.items():
            valid_anomalies = []

            for anomaly in anomalies:
                # 检查物理合理性
                if not self._check_physical_plausibility(anomaly, system_state):
                    self.false_alarm_history.append(anomaly.get('id', 'unknown'))
                    self.stats['false_alarms'] += 1
                    continue

                # 检查交叉验证
                if not self._cross_validate(anomaly, system_state):
                    # 短时单传感器异常可能是虚警
                    duration = anomaly.get('duration', 0)
                    if duration < 30:
                        self.false_alarm_history.append(anomaly.get('id', 'unknown'))
                        self.stats['false_alarms'] += 1
                        continue

                valid_anomalies.append(anomaly)

            if valid_anomalies:
                filtered[component] = valid_anomalies

        return filtered

    def _check_physical_plausibility(self, anomaly: Dict, system_state: Dict) -> bool:
        """检查物理合理性"""
        anomaly_type = anomaly.get('type', '')
        value = anomaly.get('value', 0)

        # 水位异常检查
        if 'level' in anomaly_type:
            if value < -1.0 or value > 15.0:  # 明显不合理的水位
                return False

        # 流量异常检查
        if 'flow' in anomaly_type:
            if value < -5.0 or value > 100.0:  # 明显不合理的流量
                return False

        # 检查变化率
        change_rate = anomaly.get('change_rate', 0)
        if abs(change_rate) > 1.0:  # 水位变化率超过1m/s不合理
            return False

        return True

    def _cross_validate(self, anomaly: Dict, system_state: Dict) -> bool:
        """交叉验证"""
        component = anomaly.get('component', '')
        value = anomaly.get('value', 0)

        # 获取相邻传感器/执行器数据
        neighbors = self._get_neighbor_readings(component, system_state)

        if not neighbors:
            return True  # 无法验证，默认接受

        # 检查一致性
        for neighbor_value in neighbors:
            if abs(value - neighbor_value) > 2.0:  # 与邻居差异过大
                return False

        return True

    def _get_neighbor_readings(self, component: str, system_state: Dict) -> List[float]:
        """获取相邻读数"""
        neighbors = []

        # 根据组件类型获取相邻数据
        if 'level' in component:
            levels = system_state.get('water_levels', [])
            neighbors = [l for l in levels if l is not None]
        elif 'flow' in component:
            flows = system_state.get('flows', [])
            neighbors = [f for f in flows if f is not None]

        return neighbors

    def _diagnose_single_fault(self,
                               component: str,
                               anomalies: List[Dict],
                               system_state: Dict) -> Optional[EnhancedDiagnosisResult]:
        """诊断单个故障"""
        if not anomalies:
            return None

        # 提取特征
        features = self._extract_features(anomalies)

        # 规则匹配
        best_match = None
        best_confidence = 0.0

        for rule_name, rule in self.diagnosis_rules.items():
            match_score = self._match_rule(features, rule['conditions'])
            if match_score > best_confidence:
                best_confidence = match_score
                best_match = rule

        if best_match is None or best_confidence < 0.5:
            return None

        # 模式匹配增强
        pattern_boost = self._match_patterns(features)

        # 构建诊断结果
        fault_id = f"F_{int(time.time()*1000)}_{component}"

        confidence = DiagnosisConfidence(
            base_confidence=best_confidence * best_match['confidence'],
            pattern_match_boost=pattern_boost
        )

        result = EnhancedDiagnosisResult(
            fault_id=fault_id,
            fault_type=best_match['fault_type'],
            component=component,
            severity=best_match['severity'],
            confidence=confidence,
            root_causes=self._analyze_root_cause(features, best_match),
            contributing_factors=self._find_contributing_factors(features, system_state),
            recommended_actions=self._generate_recommendations(best_match['fault_type']),
            metadata={'features': features, 'rule': rule_name}
        )

        self.stats['confirmed_faults'] += 1
        return result

    def _extract_features(self, anomalies: List[Dict]) -> Dict:
        """提取故障特征"""
        features = {
            'count': len(anomalies),
            'duration': max(a.get('duration', 0) for a in anomalies),
            'max_deviation': max(abs(a.get('deviation', 0)) for a in anomalies),
            'types': list(set(a.get('type', '') for a in anomalies)),
            'values': [a.get('value', 0) for a in anomalies],
            'timestamps': [a.get('timestamp', 0) for a in anomalies]
        }

        # 计算统计特征
        values = features['values']
        if values:
            features['mean'] = np.mean(values)
            features['variance'] = np.var(values)
            features['trend'] = self._calculate_trend(values)

        return features

    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'unknown'

        diff = np.diff(values)
        if np.all(diff > 0):
            return 'increasing'
        elif np.all(diff < 0):
            return 'decreasing'
        elif np.std(diff) < 0.01:
            return 'stable'
        else:
            return 'fluctuating'

    def _match_rule(self, features: Dict, conditions: List) -> float:
        """匹配诊断规则"""
        matches = 0
        total = len(conditions)

        for condition in conditions:
            if len(condition) == 2:  # (feature, expected_value)
                feature, expected = condition
                if features.get(feature) == expected:
                    matches += 1
            elif len(condition) == 3:  # (feature, operator, threshold)
                feature, op, threshold = condition
                value = features.get(feature, 0)

                if op == '>' and value > threshold:
                    matches += 1
                elif op == '<' and value < threshold:
                    matches += 1
                elif op == '==' and value == threshold:
                    matches += 1

        return matches / total if total > 0 else 0.0

    def _match_patterns(self, features: Dict) -> float:
        """模式匹配"""
        best_match = 0.0

        for pattern_id, pattern in self.pattern_library.items():
            score = pattern.match_score(features)
            if score > best_match:
                best_match = score
                pattern.occurrence_count += 1
                pattern.last_seen = datetime.now().isoformat()

        return best_match * 0.2  # 最多增加0.2置信度

    def _detect_compound_faults(self,
                                single_faults: List[EnhancedDiagnosisResult]
                                ) -> List[EnhancedDiagnosisResult]:
        """检测复合故障"""
        if len(single_faults) < 2:
            return []

        compound_results = []

        # 检查故障组合
        for i, fault1 in enumerate(single_faults):
            for fault2 in single_faults[i+1:]:
                # 检查时间相关性
                time_diff = abs(fault1.diagnosis_time - fault2.diagnosis_time)
                if time_diff > 60:  # 超过60秒认为不相关
                    continue

                # 检查空间相关性
                if self._are_spatially_related(fault1.component, fault2.component):
                    compound = self._create_compound_fault(fault1, fault2)
                    if compound:
                        compound_results.append(compound)
                        self.stats['compound_detections'] += 1

        return compound_results

    def _are_spatially_related(self, comp1: str, comp2: str) -> bool:
        """检查空间相关性"""
        # 简单实现：相邻池的组件认为相关
        try:
            pool1 = int(comp1.split('_')[-1]) if '_' in comp1 else 0
            pool2 = int(comp2.split('_')[-1]) if '_' in comp2 else 0
            return abs(pool1 - pool2) <= 1
        except (ValueError, IndexError):
            return False

    def _create_compound_fault(self,
                               fault1: EnhancedDiagnosisResult,
                               fault2: EnhancedDiagnosisResult
                               ) -> Optional[EnhancedDiagnosisResult]:
        """创建复合故障"""
        compound_id = f"CF_{int(time.time()*1000)}"

        # 确定交互类型
        interaction = self._determine_interaction(fault1, fault2)

        # 计算复合严重度
        combined_severity = max(fault1.severity, fault2.severity)
        if interaction == 'multiplicative':
            combined_severity = FaultSeverity(min(
                combined_severity.value + 1,
                FaultSeverity.EMERGENCY.value
            ))

        confidence = DiagnosisConfidence(
            base_confidence=min(
                fault1.confidence.final_confidence,
                fault2.confidence.final_confidence
            ) * 0.9
        )

        return EnhancedDiagnosisResult(
            fault_id=compound_id,
            fault_type=ExtendedFaultType.COMPOUND_FAULT,
            component=f"{fault1.component}+{fault2.component}",
            severity=combined_severity,
            confidence=confidence,
            root_causes=[fault1.fault_id, fault2.fault_id],
            contributing_factors=[],
            recommended_actions=self._merge_recommendations(
                fault1.recommended_actions,
                fault2.recommended_actions
            ),
            is_compound=True,
            related_faults=[fault1.fault_id, fault2.fault_id]
        )

    def _determine_interaction(self,
                               fault1: EnhancedDiagnosisResult,
                               fault2: EnhancedDiagnosisResult) -> str:
        """确定故障交互类型"""
        # 传感器+执行器故障通常是乘性
        types = {fault1.fault_type, fault2.fault_type}

        if (ExtendedFaultType.SENSOR_STUCK in types and
            ExtendedFaultType.ACTUATOR_STUCK in types):
            return 'multiplicative'

        return 'additive'

    def _track_cascade_faults(self,
                              single_faults: List[EnhancedDiagnosisResult],
                              system_state: Dict) -> List[EnhancedDiagnosisResult]:
        """追踪级联故障"""
        cascade_results = []

        # 检查现有级联
        for cascade_id, cascade in list(self.cascade_tracking.items()):
            # 检查是否有新的受影响组件
            for fault in single_faults:
                if fault.component in cascade.affected_components:
                    cascade.current_stage += 1

                    if cascade.current_stage >= cascade.max_stages:
                        # 级联完成，生成结果
                        result = self._finalize_cascade(cascade)
                        cascade_results.append(result)
                        del self.cascade_tracking[cascade_id]
                        self.stats['cascade_detections'] += 1

        # 检测新的级联
        for fault in single_faults:
            if self._could_trigger_cascade(fault, system_state):
                cascade = self._init_cascade_tracking(fault)
                self.cascade_tracking[cascade.fault_id] = cascade

        return cascade_results

    def _could_trigger_cascade(self,
                               fault: EnhancedDiagnosisResult,
                               system_state: Dict) -> bool:
        """检查故障是否可能触发级联"""
        # 高严重度的执行器故障可能触发级联
        if fault.severity.value >= FaultSeverity.HIGH.value:
            if 'gate' in fault.component or 'actuator' in fault.component:
                return True

        return False

    def _init_cascade_tracking(self, trigger_fault: EnhancedDiagnosisResult) -> CascadeFault:
        """初始化级联追踪"""
        cascade_id = f"CASCADE_{int(time.time()*1000)}"

        # 确定可能受影响的组件
        affected = self._predict_cascade_path(trigger_fault.component)

        return CascadeFault(
            fault_id=cascade_id,
            trigger_fault=trigger_fault.fault_id,
            affected_components=affected,
            propagation_time=120.0,  # 预估2分钟
            propagation_probability=0.7,
            max_stages=len(affected)
        )

    def _predict_cascade_path(self, trigger_component: str) -> List[str]:
        """预测级联路径"""
        # 简单实现：假设级联沿下游传播
        path = []
        try:
            pool_id = int(trigger_component.split('_')[-1])
            for i in range(pool_id + 1, self.num_pools):
                path.append(f"pool_{i}")
        except (ValueError, IndexError):
            pass
        return path

    def _finalize_cascade(self, cascade: CascadeFault) -> EnhancedDiagnosisResult:
        """完成级联故障诊断"""
        confidence = DiagnosisConfidence(
            base_confidence=0.85,
            pattern_match_boost=0.1
        )

        return EnhancedDiagnosisResult(
            fault_id=cascade.fault_id,
            fault_type=ExtendedFaultType.CASCADE_FAULT,
            component=','.join(cascade.affected_components),
            severity=FaultSeverity.CRITICAL,
            confidence=confidence,
            root_causes=[cascade.trigger_fault],
            contributing_factors=cascade.affected_components,
            recommended_actions=[
                "立即隔离触发故障源",
                "检查所有下游组件",
                "启动紧急运行模式",
                "准备手动干预"
            ],
            is_cascade=True,
            related_faults=[cascade.trigger_fault],
            metadata={'propagation_stages': cascade.current_stage}
        )

    def _adjust_confidence(self,
                           fault: EnhancedDiagnosisResult,
                           historical_context: Dict = None):
        """调整置信度"""
        if historical_context is None:
            return

        # 历史加成：之前见过类似故障
        similar_count = self._count_similar_historical(fault, historical_context)
        if similar_count > 0:
            fault.confidence.historical_boost = min(0.15, similar_count * 0.05)

        # 时序一致性
        temporal_score = self._check_temporal_consistency(fault, historical_context)
        fault.confidence.temporal_consistency = temporal_score

        # 虚警惩罚
        if fault.component in self.false_alarm_history[-10:]:
            fault.confidence.false_alarm_penalty = 0.1

    def _count_similar_historical(self, fault: EnhancedDiagnosisResult, context: Dict) -> int:
        """统计历史相似故障"""
        count = 0
        history = context.get('fault_history', [])

        for hist_fault in history[-50:]:  # 只看最近50条
            if (hist_fault.get('type') == fault.fault_type.value and
                hist_fault.get('component') == fault.component):
                count += 1

        return count

    def _check_temporal_consistency(self, fault: EnhancedDiagnosisResult, context: Dict) -> float:
        """检查时序一致性"""
        # 简单实现：检查故障是否符合预期的时间模式
        return 1.0  # 默认完全一致

    def _analyze_root_cause(self, features: Dict, rule: Dict) -> List[str]:
        """分析根因"""
        causes = []

        fault_type = rule['fault_type']

        if fault_type == ExtendedFaultType.SENSOR_DRIFT:
            causes.append("传感器老化或校准漂移")
            if features.get('trend') == 'increasing':
                causes.append("可能存在环境温度影响")

        elif fault_type == ExtendedFaultType.SENSOR_STUCK:
            causes.append("传感器硬件故障")
            causes.append("信号线路故障")

        elif fault_type == ExtendedFaultType.ACTUATOR_STUCK:
            causes.append("执行器机械卡死")
            causes.append("液压/电气系统故障")

        elif fault_type == ExtendedFaultType.ACTUATOR_DEGRADATION:
            causes.append("执行器磨损")
            causes.append("润滑不足")

        return causes

    def _find_contributing_factors(self, features: Dict, system_state: Dict) -> List[str]:
        """找出贡献因素"""
        factors = []

        # 检查运行时间
        operating_hours = system_state.get('operating_hours', 0)
        if operating_hours > 10000:
            factors.append("长时间运行累积磨损")

        # 检查环境条件
        temperature = system_state.get('temperature', 20)
        if temperature < 0:
            factors.append("低温环境影响")
        elif temperature > 40:
            factors.append("高温环境影响")

        return factors

    def _generate_recommendations(self, fault_type: ExtendedFaultType) -> List[str]:
        """生成修复建议"""
        recommendations = {
            ExtendedFaultType.SENSOR_DRIFT: [
                "执行传感器校准",
                "检查传感器安装位置",
                "考虑更换传感器"
            ],
            ExtendedFaultType.SENSOR_STUCK: [
                "检查传感器电源和信号线",
                "切换到备用传感器",
                "准备更换故障传感器"
            ],
            ExtendedFaultType.ACTUATOR_STUCK: [
                "检查机械限位",
                "检查液压/电气系统",
                "尝试手动复位",
                "准备人工干预"
            ],
            ExtendedFaultType.ACTUATOR_DEGRADATION: [
                "安排预防性维护",
                "检查润滑系统",
                "准备备件"
            ],
            ExtendedFaultType.CASCADE_FAULT: [
                "立即隔离故障源",
                "检查所有受影响组件",
                "启动紧急运行模式"
            ],
            ExtendedFaultType.INTERMITTENT_FAULT: [
                "增加监测频率",
                "检查接线和连接",
                "分析故障发生条件"
            ]
        }

        return recommendations.get(fault_type, ["联系维护人员", "收集更多信息"])

    def _merge_recommendations(self, rec1: List[str], rec2: List[str]) -> List[str]:
        """合并建议"""
        merged = list(set(rec1 + rec2))
        # 优先级排序
        priority_keywords = ['立即', '紧急', '切换', '隔离']
        merged.sort(key=lambda x: -sum(1 for k in priority_keywords if k in x))
        return merged[:5]  # 最多5条建议

    def get_active_faults(self) -> List[EnhancedDiagnosisResult]:
        """获取活跃故障"""
        return list(self.active_faults.values())

    def clear_fault(self, fault_id: str):
        """清除故障"""
        if fault_id in self.active_faults:
            del self.active_faults[fault_id]

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'active_faults': len(self.active_faults),
            'cascade_tracking': len(self.cascade_tracking),
            'pattern_library_size': len(self.pattern_library),
            'false_alarm_rate': (
                self.stats['false_alarms'] /
                max(self.stats['total_diagnoses'], 1)
            )
        }
