"""
安全边界与人机接管模块 (Safety Boundary & Human Override)
确保自主系统运行安全

特点:
1. 多层安全边界检查
2. 实时异常检测
3. 人机接管机制
4. 安全审计追踪
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """安全等级"""
    NORMAL = auto()           # 正常
    CAUTION = auto()          # 注意
    WARNING = auto()          # 警告
    CRITICAL = auto()         # 危险
    EMERGENCY = auto()        # 紧急


class OverrideReason(Enum):
    """接管原因"""
    MANUAL_REQUEST = "manual_request"              # 人工请求
    SAFETY_VIOLATION = "safety_violation"          # 安全违规
    CONFIDENCE_LOW = "confidence_low"              # 置信度低
    ANOMALY_DETECTED = "anomaly_detected"          # 异常检测
    COMMUNICATION_LOSS = "communication_loss"      # 通信丢失
    SENSOR_FAILURE = "sensor_failure"              # 传感器故障
    SYSTEM_ERROR = "system_error"                  # 系统错误


@dataclass
class SafetyConstraints:
    """安全约束"""
    # 水位约束
    level_min: float = 1.5                         # 最低安全水位 (m)
    level_max: float = 6.0                         # 最高安全水位 (m)
    level_change_max: float = 0.5                  # 最大水位变化率 (m/15min)

    # 流量约束
    flow_min: float = 50.0                         # 最小流量 (m³/s)
    flow_max: float = 500.0                        # 最大流量 (m³/s)
    flow_change_max: float = 50.0                  # 最大流量变化率 (m³/s/15min)

    # 闸门约束
    gate_min: float = 0.05                         # 最小开度
    gate_max: float = 1.0                          # 最大开度
    gate_change_max: float = 0.15                  # 最大开度变化率 (/15min)

    # 运行约束
    min_confidence: float = 0.6                    # 最低置信度
    max_consecutive_anomalies: int = 3             # 最大连续异常次数


@dataclass
class SafetyEvent:
    """安全事件"""
    timestamp: float
    level: SafetyLevel
    category: str
    description: str
    location: Optional[int] = None                 # 渠池/闸门位置
    values: Dict = field(default_factory=dict)
    handled: bool = False
    resolution: str = ""


class SafetyMonitor:
    """
    安全监控器
    实时监控系统状态
    """

    def __init__(self, constraints: SafetyConstraints = None, num_pools: int = 63):
        self.constraints = constraints or SafetyConstraints()
        self.num_pools = num_pools

        # 历史数据
        self.history_length = 100
        self.level_history = deque(maxlen=self.history_length)
        self.flow_history = deque(maxlen=self.history_length)
        self.gate_history = deque(maxlen=self.history_length)

        # 安全事件记录
        self.events = deque(maxlen=1000)
        self.active_warnings = []

        # 异常计数
        self.anomaly_counts = np.zeros(num_pools)

        # 统计
        self.stats = {
            'total_checks': 0,
            'violations': 0,
            'warnings': 0,
            'emergencies': 0
        }

    def check_state(self, state: Dict) -> Tuple[SafetyLevel, List[SafetyEvent]]:
        """
        检查当前状态

        Returns:
            safety_level: 整体安全等级
            events: 安全事件列表
        """
        self.stats['total_checks'] += 1
        events = []
        max_level = SafetyLevel.NORMAL

        levels = np.array(state.get('levels', [4.0] * self.num_pools))
        inflows = np.array(state.get('inflows', [300.0] * self.num_pools))
        outflows = np.array(state.get('outflows', [280.0] * self.num_pools))
        gates = np.array(state.get('gates', [0.8] * (self.num_pools + 1)))

        # 更新历史
        self.level_history.append(levels.copy())
        self.flow_history.append(inflows.copy())
        self.gate_history.append(gates.copy())

        # 1. 水位边界检查
        level_events, level_safety = self._check_level_bounds(levels)
        events.extend(level_events)
        max_level = max(max_level, level_safety, key=lambda x: x.value)

        # 2. 流量检查
        flow_events, flow_safety = self._check_flow_bounds(inflows, outflows)
        events.extend(flow_events)
        max_level = max(max_level, flow_safety, key=lambda x: x.value)

        # 3. 变化率检查
        if len(self.level_history) >= 2:
            rate_events, rate_safety = self._check_change_rates()
            events.extend(rate_events)
            max_level = max(max_level, rate_safety, key=lambda x: x.value)

        # 4. 闸门状态检查
        gate_events, gate_safety = self._check_gate_status(gates)
        events.extend(gate_events)
        max_level = max(max_level, gate_safety, key=lambda x: x.value)

        # 5. 异常模式检测
        anomaly_events = self._detect_anomalies(levels, inflows)
        events.extend(anomaly_events)
        if anomaly_events:
            max_level = max(max_level, SafetyLevel.WARNING, key=lambda x: x.value)

        # 记录事件
        for event in events:
            self.events.append(event)
            if event.level.value >= SafetyLevel.WARNING.value:
                self.stats['warnings'] += 1
            if event.level.value >= SafetyLevel.CRITICAL.value:
                self.stats['violations'] += 1
            if event.level == SafetyLevel.EMERGENCY:
                self.stats['emergencies'] += 1

        return max_level, events

    def _check_level_bounds(self, levels: np.ndarray) -> Tuple[List[SafetyEvent], SafetyLevel]:
        """检查水位边界"""
        events = []
        max_level = SafetyLevel.NORMAL
        c = self.constraints

        for i, level in enumerate(levels):
            if level < c.level_min:
                severity = SafetyLevel.EMERGENCY if level < c.level_min - 0.5 else SafetyLevel.CRITICAL
                events.append(SafetyEvent(
                    timestamp=time.time(),
                    level=severity,
                    category="level_low",
                    description=f"渠池{i}水位过低: {level:.2f}m < {c.level_min}m",
                    location=i,
                    values={'level': level, 'threshold': c.level_min}
                ))
                max_level = max(max_level, severity, key=lambda x: x.value)

            elif level > c.level_max:
                severity = SafetyLevel.EMERGENCY if level > c.level_max + 0.5 else SafetyLevel.CRITICAL
                events.append(SafetyEvent(
                    timestamp=time.time(),
                    level=severity,
                    category="level_high",
                    description=f"渠池{i}水位过高: {level:.2f}m > {c.level_max}m",
                    location=i,
                    values={'level': level, 'threshold': c.level_max}
                ))
                max_level = max(max_level, severity, key=lambda x: x.value)

            elif level < c.level_min + 0.5 or level > c.level_max - 0.5:
                events.append(SafetyEvent(
                    timestamp=time.time(),
                    level=SafetyLevel.CAUTION,
                    category="level_boundary",
                    description=f"渠池{i}水位接近边界: {level:.2f}m",
                    location=i,
                    values={'level': level}
                ))
                max_level = max(max_level, SafetyLevel.CAUTION, key=lambda x: x.value)

        return events, max_level

    def _check_flow_bounds(self, inflows: np.ndarray, outflows: np.ndarray) -> Tuple[List[SafetyEvent], SafetyLevel]:
        """检查流量边界"""
        events = []
        max_level = SafetyLevel.NORMAL
        c = self.constraints

        for i, (q_in, q_out) in enumerate(zip(inflows, outflows)):
            # 入流检查
            if q_in < c.flow_min:
                events.append(SafetyEvent(
                    timestamp=time.time(),
                    level=SafetyLevel.WARNING,
                    category="flow_low",
                    description=f"渠池{i}入流过低: {q_in:.1f} m³/s",
                    location=i,
                    values={'inflow': q_in}
                ))
                max_level = max(max_level, SafetyLevel.WARNING, key=lambda x: x.value)

            if q_in > c.flow_max:
                events.append(SafetyEvent(
                    timestamp=time.time(),
                    level=SafetyLevel.CRITICAL,
                    category="flow_high",
                    description=f"渠池{i}入流过高: {q_in:.1f} m³/s",
                    location=i,
                    values={'inflow': q_in}
                ))
                max_level = max(max_level, SafetyLevel.CRITICAL, key=lambda x: x.value)

            # 流量平衡检查
            imbalance = abs(q_in - q_out)
            if imbalance > 100:
                events.append(SafetyEvent(
                    timestamp=time.time(),
                    level=SafetyLevel.CAUTION,
                    category="flow_imbalance",
                    description=f"渠池{i}流量不平衡: Δ={imbalance:.1f} m³/s",
                    location=i,
                    values={'inflow': q_in, 'outflow': q_out, 'imbalance': imbalance}
                ))

        return events, max_level

    def _check_change_rates(self) -> Tuple[List[SafetyEvent], SafetyLevel]:
        """检查变化率"""
        events = []
        max_level = SafetyLevel.NORMAL
        c = self.constraints

        current_levels = self.level_history[-1]
        prev_levels = self.level_history[-2]
        level_changes = current_levels - prev_levels

        for i, change in enumerate(level_changes):
            if abs(change) > c.level_change_max:
                events.append(SafetyEvent(
                    timestamp=time.time(),
                    level=SafetyLevel.WARNING,
                    category="rapid_level_change",
                    description=f"渠池{i}水位变化过快: Δ={change:.3f}m/step",
                    location=i,
                    values={'change': change, 'threshold': c.level_change_max}
                ))
                max_level = max(max_level, SafetyLevel.WARNING, key=lambda x: x.value)

        return events, max_level

    def _check_gate_status(self, gates: np.ndarray) -> Tuple[List[SafetyEvent], SafetyLevel]:
        """检查闸门状态"""
        events = []
        max_level = SafetyLevel.NORMAL
        c = self.constraints

        for i, gate in enumerate(gates):
            if gate < c.gate_min:
                events.append(SafetyEvent(
                    timestamp=time.time(),
                    level=SafetyLevel.WARNING,
                    category="gate_closed",
                    description=f"闸门{i}开度过小: {gate:.2f}",
                    location=i,
                    values={'opening': gate}
                ))
                max_level = max(max_level, SafetyLevel.WARNING, key=lambda x: x.value)

        return events, max_level

    def _detect_anomalies(self, levels: np.ndarray, inflows: np.ndarray) -> List[SafetyEvent]:
        """异常模式检测"""
        events = []

        # 检测水位异常跳变
        if len(self.level_history) >= 10:
            recent_levels = np.array(list(self.level_history)[-10:])
            mean_levels = recent_levels.mean(axis=0)
            std_levels = recent_levels.std(axis=0) + 1e-6

            z_scores = np.abs((levels - mean_levels) / std_levels)

            for i, z in enumerate(z_scores):
                if z > 3.0:  # 3-sigma异常
                    self.anomaly_counts[i] += 1
                    if self.anomaly_counts[i] >= self.constraints.max_consecutive_anomalies:
                        events.append(SafetyEvent(
                            timestamp=time.time(),
                            level=SafetyLevel.WARNING,
                            category="anomaly",
                            description=f"渠池{i}检测到持续异常 (z-score={z:.2f})",
                            location=i,
                            values={'z_score': z, 'consecutive': int(self.anomaly_counts[i])}
                        ))
                else:
                    self.anomaly_counts[i] = max(0, self.anomaly_counts[i] - 1)

        return events

    def get_safety_report(self) -> Dict:
        """获取安全报告"""
        return {
            'stats': self.stats.copy(),
            'active_warnings': len([e for e in self.events if not e.handled and e.level.value >= SafetyLevel.WARNING.value]),
            'recent_events': [
                {
                    'timestamp': e.timestamp,
                    'level': e.level.name,
                    'category': e.category,
                    'description': e.description,
                    'location': e.location
                }
                for e in list(self.events)[-10:]
            ],
            'anomaly_counts': self.anomaly_counts.tolist()
        }


class HumanOverrideInterface:
    """
    人机接管接口
    管理人工接管流程
    """

    def __init__(self, num_gates: int = 64):
        self.num_gates = num_gates

        # 接管状态
        self.is_override_active = False
        self.override_reason = None
        self.override_start_time = None

        # 人工控制命令
        self.manual_commands = {}

        # 接管历史
        self.override_history = deque(maxlen=100)

        # 回调函数
        self.on_override_start: Optional[Callable] = None
        self.on_override_end: Optional[Callable] = None

        logger.info("Human Override Interface initialized")

    def request_override(self, reason: OverrideReason, details: str = "") -> Dict:
        """
        请求人工接管

        Args:
            reason: 接管原因
            details: 详细说明

        Returns:
            接管状态
        """
        self.is_override_active = True
        self.override_reason = reason
        self.override_start_time = time.time()

        record = {
            'action': 'start',
            'reason': reason.value,
            'details': details,
            'timestamp': self.override_start_time
        }
        self.override_history.append(record)

        logger.warning(f"Human override requested: {reason.value} - {details}")

        # 触发回调
        if self.on_override_start:
            self.on_override_start(record)

        return {
            'status': 'override_active',
            'reason': reason.value,
            'details': details,
            'timestamp': self.override_start_time
        }

    def release_override(self, operator_id: str = "unknown") -> Dict:
        """
        释放人工接管

        Args:
            operator_id: 操作员ID

        Returns:
            释放状态
        """
        if not self.is_override_active:
            return {'status': 'no_active_override'}

        duration = time.time() - self.override_start_time

        record = {
            'action': 'end',
            'reason': self.override_reason.value if self.override_reason else 'unknown',
            'duration': duration,
            'operator': operator_id,
            'timestamp': time.time()
        }
        self.override_history.append(record)

        logger.info(f"Human override released by {operator_id}, duration: {duration:.1f}s")

        # 重置状态
        self.is_override_active = False
        self.override_reason = None
        self.override_start_time = None
        self.manual_commands.clear()

        # 触发回调
        if self.on_override_end:
            self.on_override_end(record)

        return {
            'status': 'override_released',
            'duration': duration,
            'operator': operator_id
        }

    def set_manual_command(self, gate_id: int, opening: float):
        """设置人工控制命令"""
        if not self.is_override_active:
            logger.warning("Cannot set manual command: override not active")
            return False

        self.manual_commands[gate_id] = np.clip(opening, 0.0, 1.0)
        return True

    def set_all_commands(self, openings: np.ndarray):
        """设置所有闸门命令"""
        if not self.is_override_active:
            logger.warning("Cannot set commands: override not active")
            return False

        for i, opening in enumerate(openings):
            self.manual_commands[i] = np.clip(opening, 0.0, 1.0)
        return True

    def get_commands(self) -> Optional[Dict[int, float]]:
        """获取人工控制命令"""
        if not self.is_override_active:
            return None
        return self.manual_commands.copy()

    def get_override_status(self) -> Dict:
        """获取接管状态"""
        status = {
            'is_active': self.is_override_active,
            'reason': self.override_reason.value if self.override_reason else None,
            'duration': time.time() - self.override_start_time if self.override_start_time else 0,
            'num_manual_commands': len(self.manual_commands),
            'history_length': len(self.override_history)
        }
        return status


class SafetyBoundary:
    """
    安全边界模块
    整合安全监控和人机接管
    """

    def __init__(self, num_gates: int = 64, num_pools: int = 63,
                 constraints: SafetyConstraints = None):
        self.num_gates = num_gates
        self.num_pools = num_pools
        self.constraints = constraints or SafetyConstraints()

        # 安全监控器
        self.monitor = SafetyMonitor(self.constraints, num_pools)

        # 人机接管接口
        self.override = HumanOverrideInterface(num_gates)

        # 当前安全等级
        self.current_safety_level = SafetyLevel.NORMAL

        # 自动接管规则
        self.auto_override_enabled = True
        self.auto_override_threshold = SafetyLevel.CRITICAL

        logger.info("Safety Boundary module initialized")

    def check_and_filter_action(self,
                                 proposed_action: np.ndarray,
                                 state: Dict,
                                 confidence: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        检查并过滤动作

        Args:
            proposed_action: 提议的动作
            state: 当前状态
            confidence: 动作置信度

        Returns:
            filtered_action: 过滤后的动作
            info: 附加信息
        """
        info = {
            'original_action': proposed_action.copy(),
            'modified': False,
            'override_active': self.override.is_override_active,
            'safety_level': None,
            'events': []
        }

        # 1. 检查状态安全
        safety_level, events = self.monitor.check_state(state)
        self.current_safety_level = safety_level
        info['safety_level'] = safety_level.name
        info['events'] = [e.description for e in events]

        # 2. 检查是否需要自动接管
        if (self.auto_override_enabled and
            safety_level.value >= self.auto_override_threshold.value and
            not self.override.is_override_active):

            self.override.request_override(
                OverrideReason.SAFETY_VIOLATION,
                f"Safety level: {safety_level.name}"
            )

        # 3. 检查置信度
        if confidence < self.constraints.min_confidence and not self.override.is_override_active:
            self.override.request_override(
                OverrideReason.CONFIDENCE_LOW,
                f"Confidence: {confidence:.2f}"
            )

        # 4. 如果接管激活，使用人工命令
        if self.override.is_override_active:
            manual_commands = self.override.get_commands()
            if manual_commands:
                filtered_action = proposed_action.copy()
                for gate_id, opening in manual_commands.items():
                    if gate_id < len(filtered_action):
                        filtered_action[gate_id] = opening
                info['modified'] = True
                info['modification_reason'] = 'manual_override'
                return filtered_action, info

        # 5. 应用安全约束
        filtered_action = self._apply_safety_constraints(proposed_action, state)

        if not np.allclose(filtered_action, proposed_action):
            info['modified'] = True
            info['modification_reason'] = 'safety_constraint'

        return filtered_action, info

    def _apply_safety_constraints(self,
                                   action: np.ndarray,
                                   state: Dict) -> np.ndarray:
        """应用安全约束"""
        filtered = action.copy()
        c = self.constraints
        levels = np.array(state.get('levels', [4.0] * self.num_pools))
        current_gates = np.array(state.get('gates', [0.8] * self.num_gates))

        # 约束1: 闸门开度范围
        filtered = np.clip(filtered, c.gate_min, c.gate_max)

        # 约束2: 变化率限制
        changes = filtered - current_gates[:len(filtered)]
        changes = np.clip(changes, -c.gate_change_max, c.gate_change_max)
        filtered = current_gates[:len(filtered)] + changes

        # 约束3: 水位安全响应
        for i, level in enumerate(levels):
            gate_idx = min(i + 1, len(filtered) - 1)

            # 水位过低：减少出流
            if level < c.level_min + 0.3:
                filtered[gate_idx] = min(filtered[gate_idx], 0.3)

            # 水位过高：增加出流
            elif level > c.level_max - 0.3:
                filtered[gate_idx] = max(filtered[gate_idx], 0.8)

        return filtered

    def emergency_stop(self, reason: str = "Emergency stop requested"):
        """紧急停止"""
        logger.critical(f"EMERGENCY STOP: {reason}")

        # 激活接管
        self.override.request_override(OverrideReason.SAFETY_VIOLATION, reason)

        # 设置保守命令
        safe_openings = np.ones(self.num_gates) * 0.5  # 中等开度
        self.override.set_all_commands(safe_openings)

        return {
            'status': 'emergency_stop_activated',
            'reason': reason,
            'action': safe_openings
        }

    def get_safety_status(self) -> Dict:
        """获取安全状态"""
        return {
            'current_level': self.current_safety_level.name,
            'override_status': self.override.get_override_status(),
            'monitor_report': self.monitor.get_safety_report(),
            'constraints': self.constraints.__dict__
        }


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Safety Boundary Module Test")
    logger.info("=" * 70)

    # 创建安全边界
    safety = SafetyBoundary(num_gates=11, num_pools=10)

    # 正常状态测试
    logger.info("\n1. 测试正常状态:")
    normal_state = {
        'levels': np.random.uniform(3.5, 4.5, 10).tolist(),
        'inflows': np.random.uniform(250, 350, 10).tolist(),
        'outflows': np.random.uniform(240, 340, 10).tolist(),
        'gates': np.random.uniform(0.7, 0.9, 11).tolist()
    }
    action = np.random.uniform(0.7, 0.9, 11)

    filtered_action, info = safety.check_and_filter_action(action, normal_state)
    logger.info(f"  安全等级: {info['safety_level']}")
    logger.info(f"  动作修改: {info['modified']}")

    # 异常状态测试
    logger.info("\n2. 测试异常状态 (水位过低):")
    abnormal_state = normal_state.copy()
    abnormal_state['levels'] = [1.3, 1.4, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]

    filtered_action, info = safety.check_and_filter_action(action, abnormal_state)
    logger.info(f"  安全等级: {info['safety_level']}")
    logger.info(f"  动作修改: {info['modified']}")
    logger.info(f"  事件: {info['events'][:3]}...")

    # 人工接管测试
    logger.info("\n3. 测试人工接管:")
    safety.override.request_override(OverrideReason.MANUAL_REQUEST, "测试接管")
    safety.override.set_manual_command(0, 0.5)
    safety.override.set_manual_command(1, 0.6)

    filtered_action, info = safety.check_and_filter_action(action, normal_state)
    logger.info(f"  接管激活: {info['override_active']}")
    logger.info(f"  闸门0开度: {filtered_action[0]:.2f}")
    logger.info(f"  闸门1开度: {filtered_action[1]:.2f}")

    # 释放接管
    safety.override.release_override("test_operator")

    # 获取状态
    status = safety.get_safety_status()
    logger.info(f"\n4. 安全状态:")
    logger.info(f"  当前等级: {status['current_level']}")
    logger.info(f"  接管状态: {status['override_status']}")

    logger.info("\n" + "=" * 70)
    logger.info("Test completed!")
    logger.info("=" * 70)
