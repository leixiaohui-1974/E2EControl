"""
故障诊断与容错控制系统
Fault Diagnosis and Fault-Tolerant Control System

功能:
1. 设备故障检测 - 传感器、执行器、通信故障检测
2. 故障诊断 - 根因分析、故障定位
3. 容错控制 - 冗余切换、降级运行、安全停机
4. 应急响应 - 突发事件快速响应与处理
5. 运行规则引擎 - 运行规则编码与自动执行

作者: AI Assistant
日期: 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from enum import Enum, auto
from datetime import datetime, timedelta
import math
import random
from collections import deque


# ============ 故障类型定义 ============

class FaultType(Enum):
    """故障类型"""
    # 传感器故障
    SENSOR_DRIFT = auto()           # 传感器漂移
    SENSOR_STUCK = auto()           # 传感器卡死
    SENSOR_NOISE = auto()           # 传感器噪声过大
    SENSOR_BIAS = auto()            # 传感器偏置
    SENSOR_LOSS = auto()            # 传感器信号丢失

    # 执行器故障
    ACTUATOR_STUCK = auto()         # 执行器卡死
    ACTUATOR_SATURATION = auto()    # 执行器饱和
    ACTUATOR_DELAY = auto()         # 执行器响应延迟
    ACTUATOR_LOSS = auto()          # 执行器失效
    ACTUATOR_PARTIAL = auto()       # 执行器部分失效

    # 通信故障
    COMM_DELAY = auto()             # 通信延迟
    COMM_LOSS = auto()              # 通信中断
    COMM_CORRUPTION = auto()        # 数据损坏

    # 系统故障
    CONTROLLER_FAILURE = auto()     # 控制器故障
    POWER_FAILURE = auto()          # 电源故障
    SOFTWARE_ERROR = auto()         # 软件错误

    # 过程故障
    PIPE_LEAK = auto()              # 管道泄漏
    OVERFLOW = auto()               # 溢流
    UNDERFLOW = auto()              # 断流
    CONTAMINATION = auto()          # 水质污染


class FaultSeverity(Enum):
    """故障严重程度"""
    INFO = 0            # 信息提示
    WARNING = 1         # 警告
    MINOR = 2           # 轻微故障
    MAJOR = 3           # 主要故障
    CRITICAL = 4        # 严重故障
    EMERGENCY = 5       # 紧急故障


class FaultStatus(Enum):
    """故障状态"""
    DETECTED = auto()       # 已检测
    CONFIRMED = auto()      # 已确认
    ISOLATED = auto()       # 已隔离
    MITIGATED = auto()      # 已缓解
    RECOVERED = auto()      # 已恢复
    UNRECOVERABLE = auto()  # 不可恢复


class ControlMode(Enum):
    """控制模式"""
    NORMAL = auto()             # 正常控制
    DEGRADED = auto()           # 降级控制
    BACKUP = auto()             # 备用控制
    MANUAL = auto()             # 手动控制
    EMERGENCY = auto()          # 应急控制
    SAFE_SHUTDOWN = auto()      # 安全停机


@dataclass
class FaultEvent:
    """故障事件"""
    fault_id: str
    fault_type: FaultType
    severity: FaultSeverity
    component_id: str           # 故障组件标识
    component_type: str         # 组件类型 (sensor/actuator/controller)
    location: str               # 位置描述 (如 pool_1, gate_2)
    detected_time: float
    description: str

    # 故障特征
    symptoms: List[str] = field(default_factory=list)
    measurements: Dict[str, float] = field(default_factory=dict)

    # 状态跟踪
    status: FaultStatus = FaultStatus.DETECTED
    confirmed_time: Optional[float] = None
    isolated_time: Optional[float] = None
    recovered_time: Optional[float] = None

    # 处理信息
    handler: Optional[str] = None
    actions_taken: List[str] = field(default_factory=list)
    recovery_plan: Optional[str] = None


@dataclass
class DiagnosisResult:
    """诊断结果"""
    fault_id: str
    root_cause: str
    confidence: float           # 诊断置信度 0-1
    affected_components: List[str]
    propagation_risk: float     # 故障传播风险
    estimated_impact: Dict[str, float]
    recommended_actions: List[str]
    priority: int               # 处理优先级 1-10


@dataclass
class ControlReconfiguration:
    """控制重构配置"""
    mode: ControlMode
    reason: str
    affected_loops: List[str]
    backup_controllers: Dict[str, str]
    parameter_adjustments: Dict[str, float]
    constraints_modified: Dict[str, Tuple[float, float]]
    timestamp: float


# ============ 故障检测引擎 ============

class ResidualGenerator:
    """残差生成器 - 基于模型的故障检测"""

    def __init__(self, pool_id: int):
        self.pool_id = pool_id
        self.model_params = {
            'As': 5000.0,       # 水面面积
            'K_gate': 2.0,      # 闸门系数
            'delay': 60.0,      # 传播延迟
        }

        # 残差历史
        self.level_residuals: deque = deque(maxlen=100)
        self.flow_residuals: deque = deque(maxlen=100)

        # 阈值
        self.level_threshold = 0.1  # m
        self.flow_threshold = 1.0   # m³/s

        # 统计
        self.cusum_level = 0.0
        self.cusum_flow = 0.0
        self.cusum_threshold = 5.0

    def compute_residual(self,
                         measured_level: float,
                         measured_flow: float,
                         predicted_level: float,
                         predicted_flow: float) -> Dict[str, float]:
        """计算残差"""
        level_residual = measured_level - predicted_level
        flow_residual = measured_flow - predicted_flow

        self.level_residuals.append(level_residual)
        self.flow_residuals.append(flow_residual)

        # CUSUM检测
        self.cusum_level = max(0, self.cusum_level + abs(level_residual) - self.level_threshold)
        self.cusum_flow = max(0, self.cusum_flow + abs(flow_residual) - self.flow_threshold)

        return {
            'level_residual': level_residual,
            'flow_residual': flow_residual,
            'cusum_level': self.cusum_level,
            'cusum_flow': self.cusum_flow,
            'level_alarm': self.cusum_level > self.cusum_threshold,
            'flow_alarm': self.cusum_flow > self.cusum_threshold,
        }

    def reset(self):
        """重置CUSUM累积"""
        self.cusum_level = 0.0
        self.cusum_flow = 0.0


class SensorFaultDetector:
    """传感器故障检测器"""

    def __init__(self):
        self.history: Dict[str, deque] = {}
        self.window_size = 20

        # 检测阈值 - 适当放宽以避免误检测
        self.stuck_threshold = 0.0001     # 卡死检测阈值 (方差)
        self.noise_threshold = 1.0        # 噪声检测阈值
        self.drift_rate_threshold = 0.05  # 漂移速率阈值
        self.spike_threshold = 5.0        # 跳变检测倍数

    def add_reading(self, sensor_id: str, value: float, timestamp: float):
        """添加传感器读数"""
        if sensor_id not in self.history:
            self.history[sensor_id] = deque(maxlen=self.window_size)
        self.history[sensor_id].append((timestamp, value))

    def detect_faults(self, sensor_id: str) -> List[FaultType]:
        """检测传感器故障"""
        faults = []

        if sensor_id not in self.history:
            return faults

        readings = list(self.history[sensor_id])
        if len(readings) < 5:
            return faults

        values = [r[1] for r in readings]

        # 卡死检测
        if self._detect_stuck(values):
            faults.append(FaultType.SENSOR_STUCK)

        # 噪声检测
        if self._detect_noise(values):
            faults.append(FaultType.SENSOR_NOISE)

        # 漂移检测
        if self._detect_drift(values):
            faults.append(FaultType.SENSOR_DRIFT)

        # 跳变检测 (可能是偏置)
        if self._detect_spike(values):
            faults.append(FaultType.SENSOR_BIAS)

        return faults

    def _detect_stuck(self, values: List[float]) -> bool:
        """检测卡死"""
        if len(values) < 5:
            return False
        recent = values[-5:]
        variance = sum((v - sum(recent)/len(recent))**2 for v in recent) / len(recent)
        return variance < self.stuck_threshold

    def _detect_noise(self, values: List[float]) -> bool:
        """检测噪声过大"""
        if len(values) < 10:
            return False

        # 计算相邻差分的标准差
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        mean_diff = sum(diffs) / len(diffs)
        std_diff = math.sqrt(sum((d - mean_diff)**2 for d in diffs) / len(diffs))

        return std_diff > self.noise_threshold

    def _detect_drift(self, values: List[float]) -> bool:
        """检测漂移"""
        if len(values) < 10:
            return False

        # 线性回归检测趋势
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean)**2 for i in range(n))

        if denominator == 0:
            return False

        slope = numerator / denominator
        return abs(slope) > self.drift_rate_threshold

    def _detect_spike(self, values: List[float]) -> bool:
        """检测跳变"""
        if len(values) < 3:
            return False

        mean_val = sum(values[:-1]) / (len(values) - 1)
        std_val = math.sqrt(sum((v - mean_val)**2 for v in values[:-1]) / (len(values) - 1))

        if std_val == 0:
            return False

        return abs(values[-1] - mean_val) > self.spike_threshold * std_val


class ActuatorFaultDetector:
    """执行器故障检测器"""

    def __init__(self):
        self.command_history: Dict[str, deque] = {}
        self.position_history: Dict[str, deque] = {}
        self.window_size = 30

        # 检测参数
        self.response_timeout = 10.0      # 响应超时 (秒)
        self.position_tolerance = 0.05    # 位置容差
        self.rate_limit = 0.1             # 正常变化速率

    def add_command(self, actuator_id: str, command: float, timestamp: float):
        """添加执行器指令"""
        if actuator_id not in self.command_history:
            self.command_history[actuator_id] = deque(maxlen=self.window_size)
        self.command_history[actuator_id].append((timestamp, command))

    def add_position(self, actuator_id: str, position: float, timestamp: float):
        """添加执行器位置"""
        if actuator_id not in self.position_history:
            self.position_history[actuator_id] = deque(maxlen=self.window_size)
        self.position_history[actuator_id].append((timestamp, position))

    def detect_faults(self, actuator_id: str, current_time: float) -> List[Tuple[FaultType, float]]:
        """检测执行器故障，返回故障类型和置信度"""
        faults = []

        if actuator_id not in self.command_history or actuator_id not in self.position_history:
            return faults

        commands = list(self.command_history[actuator_id])
        positions = list(self.position_history[actuator_id])

        if len(commands) < 2 or len(positions) < 2:
            return faults

        # 卡死检测
        stuck_conf = self._detect_stuck(positions)
        if stuck_conf > 0.5:
            faults.append((FaultType.ACTUATOR_STUCK, stuck_conf))

        # 响应延迟检测
        delay_conf = self._detect_delay(commands, positions, current_time)
        if delay_conf > 0.5:
            faults.append((FaultType.ACTUATOR_DELAY, delay_conf))

        # 饱和检测
        sat_conf = self._detect_saturation(positions)
        if sat_conf > 0.5:
            faults.append((FaultType.ACTUATOR_SATURATION, sat_conf))

        return faults

    def _detect_stuck(self, positions: List[Tuple[float, float]]) -> float:
        """检测卡死，返回置信度"""
        if len(positions) < 5:
            return 0.0

        recent_positions = [p[1] for p in positions[-10:]]
        mean_pos = sum(recent_positions) / len(recent_positions)
        variance = sum((p - mean_pos)**2 for p in recent_positions) / len(recent_positions)

        # 低方差表示可能卡死
        if variance < 0.0001:
            return 0.9
        elif variance < 0.001:
            return 0.5
        return 0.0

    def _detect_delay(self, commands: List[Tuple[float, float]],
                      positions: List[Tuple[float, float]],
                      current_time: float) -> float:
        """检测响应延迟"""
        if len(commands) < 2:
            return 0.0

        # 检查最近命令的响应
        last_cmd_time, last_cmd_val = commands[-1]

        # 查找命令后的位置变化
        positions_after_cmd = [(t, p) for t, p in positions if t > last_cmd_time]

        if not positions_after_cmd:
            return 0.0

        # 计算延迟
        latest_pos = positions_after_cmd[-1][1]
        expected_change = abs(last_cmd_val - positions[-1][1])

        if expected_change < self.position_tolerance:
            return 0.0

        # 如果命令发出后很长时间位置还没变化
        time_since_cmd = current_time - last_cmd_time
        if time_since_cmd > self.response_timeout:
            return min(1.0, time_since_cmd / (2 * self.response_timeout))

        return 0.0

    def _detect_saturation(self, positions: List[Tuple[float, float]]) -> float:
        """检测饱和"""
        if len(positions) < 5:
            return 0.0

        recent = [p[1] for p in positions[-10:]]

        # 检查是否接近边界
        near_max = sum(1 for p in recent if p > 0.98) / len(recent)
        near_min = sum(1 for p in recent if p < 0.02) / len(recent)

        return max(near_max, near_min)


class FaultDetectionEngine:
    """综合故障检测引擎"""

    def __init__(self, num_pools: int = 5):
        self.num_pools = num_pools

        # 组件检测器
        self.sensor_detector = SensorFaultDetector()
        self.actuator_detector = ActuatorFaultDetector()
        self.residual_generators: Dict[int, ResidualGenerator] = {
            i: ResidualGenerator(i) for i in range(num_pools)
        }

        # 故障事件列表
        self.active_faults: Dict[str, FaultEvent] = {}
        self.fault_history: List[FaultEvent] = []

        # 故障计数器
        self.fault_counter = 0

    def process_sensor_reading(self, sensor_id: str, value: float,
                               sensor_type: str, location: str,
                               timestamp: float) -> List[FaultEvent]:
        """处理传感器读数并检测故障"""
        self.sensor_detector.add_reading(sensor_id, value, timestamp)

        detected_faults = self.sensor_detector.detect_faults(sensor_id)
        events = []

        for fault_type in detected_faults:
            fault_id = self._generate_fault_id(sensor_id, fault_type)

            if fault_id not in self.active_faults:
                event = FaultEvent(
                    fault_id=fault_id,
                    fault_type=fault_type,
                    severity=self._get_sensor_fault_severity(fault_type),
                    component_id=sensor_id,
                    component_type='sensor',
                    location=location,
                    detected_time=timestamp,
                    description=f"Sensor {sensor_id} {fault_type.name.lower()} detected",
                    symptoms=[fault_type.name],
                    measurements={'last_value': value},
                )

                self.active_faults[fault_id] = event
                events.append(event)

        return events

    def process_actuator_status(self, actuator_id: str,
                                command: float, position: float,
                                location: str, timestamp: float) -> List[FaultEvent]:
        """处理执行器状态并检测故障"""
        self.actuator_detector.add_command(actuator_id, command, timestamp)
        self.actuator_detector.add_position(actuator_id, position, timestamp)

        detected_faults = self.actuator_detector.detect_faults(actuator_id, timestamp)
        events = []

        for fault_type, confidence in detected_faults:
            fault_id = self._generate_fault_id(actuator_id, fault_type)

            if fault_id not in self.active_faults:
                event = FaultEvent(
                    fault_id=fault_id,
                    fault_type=fault_type,
                    severity=self._get_actuator_fault_severity(fault_type, confidence),
                    component_id=actuator_id,
                    component_type='actuator',
                    location=location,
                    detected_time=timestamp,
                    description=f"Actuator {actuator_id} {fault_type.name.lower()} (conf: {confidence:.2f})",
                    symptoms=[fault_type.name],
                    measurements={'command': command, 'position': position, 'confidence': confidence},
                )

                self.active_faults[fault_id] = event
                events.append(event)

        return events

    def process_model_residual(self, pool_id: int,
                               measured_level: float, measured_flow: float,
                               predicted_level: float, predicted_flow: float,
                               timestamp: float) -> List[FaultEvent]:
        """处理模型残差并检测故障"""
        if pool_id not in self.residual_generators:
            return []

        residual = self.residual_generators[pool_id].compute_residual(
            measured_level, measured_flow, predicted_level, predicted_flow
        )

        events = []

        if residual['level_alarm']:
            fault_id = f"RESIDUAL_LEVEL_{pool_id}_{int(timestamp)}"
            if fault_id not in self.active_faults:
                event = FaultEvent(
                    fault_id=fault_id,
                    fault_type=FaultType.SENSOR_BIAS,  # 可能是传感器或模型问题
                    severity=FaultSeverity.WARNING,
                    component_id=f"level_sensor_{pool_id}",
                    component_type='sensor',
                    location=f"pool_{pool_id}",
                    detected_time=timestamp,
                    description=f"Pool {pool_id} level residual alarm",
                    symptoms=['level_mismatch'],
                    measurements={
                        'residual': residual['level_residual'],
                        'cusum': residual['cusum_level'],
                    },
                )
                self.active_faults[fault_id] = event
                events.append(event)

        if residual['flow_alarm']:
            fault_id = f"RESIDUAL_FLOW_{pool_id}_{int(timestamp)}"
            if fault_id not in self.active_faults:
                event = FaultEvent(
                    fault_id=fault_id,
                    fault_type=FaultType.SENSOR_BIAS,
                    severity=FaultSeverity.WARNING,
                    component_id=f"flow_sensor_{pool_id}",
                    component_type='sensor',
                    location=f"pool_{pool_id}",
                    detected_time=timestamp,
                    description=f"Pool {pool_id} flow residual alarm",
                    symptoms=['flow_mismatch'],
                    measurements={
                        'residual': residual['flow_residual'],
                        'cusum': residual['cusum_flow'],
                    },
                )
                self.active_faults[fault_id] = event
                events.append(event)

        return events

    def clear_fault(self, fault_id: str, timestamp: float):
        """清除故障"""
        if fault_id in self.active_faults:
            fault = self.active_faults[fault_id]
            fault.status = FaultStatus.RECOVERED
            fault.recovered_time = timestamp
            self.fault_history.append(fault)
            del self.active_faults[fault_id]

    def get_active_faults(self) -> List[FaultEvent]:
        """获取所有活动故障"""
        return list(self.active_faults.values())

    def get_faults_by_location(self, location: str) -> List[FaultEvent]:
        """按位置获取故障"""
        return [f for f in self.active_faults.values() if f.location == location]

    def _generate_fault_id(self, component_id: str, fault_type: FaultType) -> str:
        """生成故障ID"""
        self.fault_counter += 1
        return f"{fault_type.name}_{component_id}_{self.fault_counter}"

    def _get_sensor_fault_severity(self, fault_type: FaultType) -> FaultSeverity:
        """获取传感器故障严重程度"""
        severity_map = {
            FaultType.SENSOR_DRIFT: FaultSeverity.MINOR,
            FaultType.SENSOR_STUCK: FaultSeverity.MAJOR,
            FaultType.SENSOR_NOISE: FaultSeverity.WARNING,
            FaultType.SENSOR_BIAS: FaultSeverity.MINOR,
            FaultType.SENSOR_LOSS: FaultSeverity.CRITICAL,
        }
        return severity_map.get(fault_type, FaultSeverity.WARNING)

    def _get_actuator_fault_severity(self, fault_type: FaultType,
                                     confidence: float) -> FaultSeverity:
        """获取执行器故障严重程度"""
        base_severity = {
            FaultType.ACTUATOR_STUCK: FaultSeverity.CRITICAL,
            FaultType.ACTUATOR_SATURATION: FaultSeverity.WARNING,
            FaultType.ACTUATOR_DELAY: FaultSeverity.MINOR,
            FaultType.ACTUATOR_LOSS: FaultSeverity.EMERGENCY,
            FaultType.ACTUATOR_PARTIAL: FaultSeverity.MAJOR,
        }
        severity = base_severity.get(fault_type, FaultSeverity.WARNING)

        # 高置信度提升严重级别
        if confidence > 0.9 and severity.value < FaultSeverity.CRITICAL.value:
            return FaultSeverity(severity.value + 1)

        return severity


# ============ 故障诊断引擎 ============

class FaultDiagnosisEngine:
    """故障诊断引擎 - 根因分析与故障定位"""

    def __init__(self):
        # 故障传播图
        self.propagation_graph: Dict[str, List[str]] = {}

        # 诊断规则库
        self.diagnosis_rules: List[Callable] = []
        self._setup_default_rules()

        # 历史模式
        self.fault_patterns: Dict[str, List[str]] = {}

    def _setup_default_rules(self):
        """设置默认诊断规则"""

        def rule_sensor_correlation(faults: List[FaultEvent]) -> Optional[DiagnosisResult]:
            """规则: 多传感器同时故障可能是电源或通信问题"""
            sensor_faults = [f for f in faults if f.component_type == 'sensor']
            if len(sensor_faults) >= 3:
                locations = set(f.location for f in sensor_faults)
                if len(locations) == 1:
                    return DiagnosisResult(
                        fault_id=sensor_faults[0].fault_id,
                        root_cause="Common power or communication failure",
                        confidence=0.8,
                        affected_components=[f.component_id for f in sensor_faults],
                        propagation_risk=0.7,
                        estimated_impact={'observability': 0.5, 'controllability': 0.3},
                        recommended_actions=[
                            "Check power supply",
                            "Verify communication links",
                            "Switch to backup sensors",
                        ],
                        priority=2,
                    )
            return None

        def rule_actuator_cascade(faults: List[FaultEvent]) -> Optional[DiagnosisResult]:
            """规则: 上游执行器故障会影响下游"""
            actuator_faults = [f for f in faults if f.component_type == 'actuator']
            for fault in actuator_faults:
                if fault.fault_type == FaultType.ACTUATOR_STUCK:
                    return DiagnosisResult(
                        fault_id=fault.fault_id,
                        root_cause=f"Actuator {fault.component_id} mechanical failure",
                        confidence=0.75,
                        affected_components=[fault.component_id],
                        propagation_risk=0.8,
                        estimated_impact={'flow_control': 0.9, 'level_stability': 0.7},
                        recommended_actions=[
                            "Switch to manual control",
                            "Alert maintenance team",
                            "Prepare backup actuator",
                        ],
                        priority=1,
                    )
            return None

        def rule_level_flow_inconsistency(faults: List[FaultEvent]) -> Optional[DiagnosisResult]:
            """规则: 水位流量不一致可能是泄漏"""
            level_faults = [f for f in faults if 'level' in f.component_id.lower()]
            flow_faults = [f for f in faults if 'flow' in f.component_id.lower()]

            if level_faults and flow_faults:
                same_location = set(f.location for f in level_faults) & set(f.location for f in flow_faults)
                if same_location:
                    return DiagnosisResult(
                        fault_id=level_faults[0].fault_id,
                        root_cause="Possible pipe leak or unmetered withdrawal",
                        confidence=0.6,
                        affected_components=[f.component_id for f in level_faults + flow_faults],
                        propagation_risk=0.5,
                        estimated_impact={'water_loss': 0.6, 'pressure': 0.4},
                        recommended_actions=[
                            "Inspect canal section",
                            "Compare with adjacent sensors",
                            "Check for unauthorized withdrawals",
                        ],
                        priority=3,
                    )
            return None

        self.diagnosis_rules = [
            rule_sensor_correlation,
            rule_actuator_cascade,
            rule_level_flow_inconsistency,
        ]

    def add_rule(self, rule: Callable):
        """添加诊断规则"""
        self.diagnosis_rules.append(rule)

    def diagnose(self, faults: List[FaultEvent]) -> List[DiagnosisResult]:
        """诊断故障"""
        results = []

        for rule in self.diagnosis_rules:
            result = rule(faults)
            if result:
                results.append(result)

        # 按优先级排序
        results.sort(key=lambda x: x.priority)

        return results

    def analyze_propagation(self, fault: FaultEvent) -> List[str]:
        """分析故障传播路径"""
        affected = []
        queue = [fault.location]
        visited = set()

        while queue:
            location = queue.pop(0)
            if location in visited:
                continue
            visited.add(location)

            if location in self.propagation_graph:
                downstream = self.propagation_graph[location]
                affected.extend(downstream)
                queue.extend(downstream)

        return affected

    def update_propagation_graph(self, graph: Dict[str, List[str]]):
        """更新故障传播图"""
        self.propagation_graph = graph


# ============ 容错控制器 ============

class FaultTolerantController:
    """容错控制器"""

    def __init__(self, num_pools: int = 5):
        self.num_pools = num_pools

        # 控制模式
        self.current_mode = ControlMode.NORMAL
        self.mode_history: List[Tuple[float, ControlMode]] = []

        # 冗余配置
        self.backup_sensors: Dict[str, str] = {}
        self.backup_actuators: Dict[str, str] = {}
        self.backup_controllers: Dict[str, Any] = {}

        # 降级策略
        self.degradation_levels: Dict[str, int] = {}

        # 安全限制
        self.safety_limits = {
            'max_level': 4.0,       # m
            'min_level': 1.0,       # m
            'max_flow': 100.0,      # m³/s
            'max_gate_speed': 0.1,  # 开度/秒
        }

        # 控制增益调整
        self.gain_adjustments: Dict[str, float] = {}

    def reconfigure(self, faults: List[FaultEvent],
                    diagnosis: List[DiagnosisResult],
                    timestamp: float) -> ControlReconfiguration:
        """重构控制系统"""

        # 评估故障严重程度
        max_severity = max((f.severity.value for f in faults), default=0)

        # 决定控制模式
        new_mode = self._determine_mode(max_severity, faults)

        if new_mode != self.current_mode:
            self.mode_history.append((timestamp, self.current_mode))
            self.current_mode = new_mode

        # 生成重构配置
        config = ControlReconfiguration(
            mode=new_mode,
            reason=self._get_mode_reason(faults),
            affected_loops=[],
            backup_controllers={},
            parameter_adjustments={},
            constraints_modified={},
            timestamp=timestamp,
        )

        # 根据模式配置
        if new_mode == ControlMode.DEGRADED:
            config = self._configure_degraded_mode(config, faults)
        elif new_mode == ControlMode.BACKUP:
            config = self._configure_backup_mode(config, faults)
        elif new_mode == ControlMode.EMERGENCY:
            config = self._configure_emergency_mode(config, faults)
        elif new_mode == ControlMode.SAFE_SHUTDOWN:
            config = self._configure_safe_shutdown(config, faults)

        return config

    def _determine_mode(self, max_severity: int,
                        faults: List[FaultEvent]) -> ControlMode:
        """确定控制模式"""
        if max_severity >= FaultSeverity.EMERGENCY.value:
            return ControlMode.SAFE_SHUTDOWN
        elif max_severity >= FaultSeverity.CRITICAL.value:
            return ControlMode.EMERGENCY
        elif max_severity >= FaultSeverity.MAJOR.value:
            # 检查是否有备份可用
            if self._has_backup_available(faults):
                return ControlMode.BACKUP
            return ControlMode.DEGRADED
        elif max_severity >= FaultSeverity.MINOR.value:
            return ControlMode.DEGRADED
        return ControlMode.NORMAL

    def _has_backup_available(self, faults: List[FaultEvent]) -> bool:
        """检查是否有备份可用"""
        for fault in faults:
            if fault.component_type == 'sensor':
                if fault.component_id in self.backup_sensors:
                    return True
            elif fault.component_type == 'actuator':
                if fault.component_id in self.backup_actuators:
                    return True
        return False

    def _get_mode_reason(self, faults: List[FaultEvent]) -> str:
        """获取模式切换原因"""
        if not faults:
            return "No faults detected"

        severe_faults = [f for f in faults if f.severity.value >= FaultSeverity.MAJOR.value]
        if severe_faults:
            return f"Critical fault: {severe_faults[0].description}"
        return f"Multiple minor faults detected ({len(faults)} total)"

    def _configure_degraded_mode(self, config: ControlReconfiguration,
                                 faults: List[FaultEvent]) -> ControlReconfiguration:
        """配置降级模式"""
        affected_pools = set()

        for fault in faults:
            if 'pool_' in fault.location:
                pool_id = int(fault.location.split('_')[1])
                affected_pools.add(pool_id)

                # 降低控制增益
                config.parameter_adjustments[f'pool_{pool_id}_gain'] = 0.7

                # 收紧约束
                config.constraints_modified[f'pool_{pool_id}_level'] = (
                    self.safety_limits['min_level'] + 0.2,
                    self.safety_limits['max_level'] - 0.2,
                )

        config.affected_loops = [f'pool_{p}' for p in affected_pools]
        return config

    def _configure_backup_mode(self, config: ControlReconfiguration,
                               faults: List[FaultEvent]) -> ControlReconfiguration:
        """配置备用模式"""
        for fault in faults:
            if fault.component_type == 'sensor' and fault.component_id in self.backup_sensors:
                backup_id = self.backup_sensors[fault.component_id]
                config.backup_controllers[fault.component_id] = backup_id
            elif fault.component_type == 'actuator' and fault.component_id in self.backup_actuators:
                backup_id = self.backup_actuators[fault.component_id]
                config.backup_controllers[fault.component_id] = backup_id

        return config

    def _configure_emergency_mode(self, config: ControlReconfiguration,
                                  faults: List[FaultEvent]) -> ControlReconfiguration:
        """配置应急模式"""
        # 所有池使用保守控制
        for i in range(self.num_pools):
            config.parameter_adjustments[f'pool_{i}_gain'] = 0.3
            config.constraints_modified[f'pool_{i}_level'] = (
                self.safety_limits['min_level'] + 0.5,
                self.safety_limits['max_level'] - 0.5,
            )
            config.constraints_modified[f'pool_{i}_gate_speed'] = (
                -self.safety_limits['max_gate_speed'] / 2,
                self.safety_limits['max_gate_speed'] / 2,
            )

        config.affected_loops = [f'pool_{i}' for i in range(self.num_pools)]
        return config

    def _configure_safe_shutdown(self, config: ControlReconfiguration,
                                 faults: List[FaultEvent]) -> ControlReconfiguration:
        """配置安全停机"""
        # 所有闸门缓慢关闭到安全位置
        for i in range(self.num_pools + 1):
            config.parameter_adjustments[f'gate_{i}_target'] = 0.5  # 50%开度
            config.constraints_modified[f'gate_{i}_speed'] = (
                -0.01, 0.01  # 非常缓慢移动
            )

        config.affected_loops = [f'pool_{i}' for i in range(self.num_pools)]
        return config

    def register_backup(self, primary_id: str, backup_id: str,
                        component_type: str):
        """注册备用组件"""
        if component_type == 'sensor':
            self.backup_sensors[primary_id] = backup_id
        elif component_type == 'actuator':
            self.backup_actuators[primary_id] = backup_id

    def get_safe_control(self, pool_id: int,
                        current_level: float,
                        target_level: float) -> float:
        """获取安全控制指令"""
        # 简单PI控制
        error = target_level - current_level

        # 应用模式相关增益
        gain = self.gain_adjustments.get(f'pool_{pool_id}_gain', 1.0)

        # 限幅
        control = gain * 0.5 * error  # 简化的比例控制
        control = max(-self.safety_limits['max_gate_speed'],
                     min(control, self.safety_limits['max_gate_speed']))

        return control


# ============ 应急响应系统 ============

class EmergencyType(Enum):
    """应急事件类型"""
    POLLUTION = auto()          # 水质污染
    FLOOD = auto()              # 洪水
    DROUGHT = auto()            # 干旱
    EQUIPMENT_FAILURE = auto()  # 设备故障
    POWER_OUTAGE = auto()       # 电源中断
    EARTHQUAKE = auto()         # 地震
    ICE_JAM = auto()            # 冰塞


@dataclass
class EmergencyEvent:
    """应急事件"""
    event_id: str
    event_type: EmergencyType
    severity: FaultSeverity
    location: str
    start_time: float
    description: str

    # 事件属性
    affected_area: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0

    # 响应状态
    response_started: bool = False
    response_start_time: Optional[float] = None
    response_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_time: Optional[float] = None


@dataclass
class EmergencyResponse:
    """应急响应"""
    event_id: str
    response_level: int         # 1-4级响应
    lead_agency: str
    actions: List[str]
    resources_required: Dict[str, int]
    communication_plan: List[str]
    evacuation_needed: bool = False


class EmergencyResponseSystem:
    """应急响应系统"""

    def __init__(self):
        # 应急预案库
        self.response_plans: Dict[EmergencyType, List[str]] = {
            EmergencyType.POLLUTION: [
                "关闭上游闸门",
                "隔离污染段",
                "启动退水闸",
                "通知下游取水口",
                "调度应急水源",
            ],
            EmergencyType.FLOOD: [
                "开大下游闸门",
                "预腾库容",
                "启动分洪闸",
                "通知沿线防汛部门",
                "加强巡查",
            ],
            EmergencyType.EQUIPMENT_FAILURE: [
                "切换到备用设备",
                "启动手动控制",
                "通知维修团队",
                "评估影响范围",
                "调整上下游控制",
            ],
            EmergencyType.POWER_OUTAGE: [
                "启动备用电源",
                "切换到手动模式",
                "保持当前闸门位置",
                "通知调度中心",
                "准备应急发电车",
            ],
        }

        # 活动应急事件
        self.active_emergencies: Dict[str, EmergencyEvent] = {}

        # 响应历史
        self.response_history: List[EmergencyResponse] = []

        # 事件计数器
        self.event_counter = 0

    def report_emergency(self, event_type: EmergencyType,
                        location: str, description: str,
                        severity: FaultSeverity,
                        timestamp: float) -> EmergencyEvent:
        """报告应急事件"""
        self.event_counter += 1
        event_id = f"EMG_{event_type.name}_{self.event_counter}"

        event = EmergencyEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            location=location,
            start_time=timestamp,
            description=description,
        )

        # 确定影响范围
        event.affected_area = self._estimate_affected_area(event_type, location)

        # 估计持续时间
        event.estimated_duration = self._estimate_duration(event_type, severity)

        self.active_emergencies[event_id] = event
        return event

    def generate_response(self, event: EmergencyEvent) -> EmergencyResponse:
        """生成应急响应"""
        # 确定响应级别
        response_level = self._determine_response_level(event)

        # 获取应急措施
        actions = self.response_plans.get(event.event_type, [
            "评估现场情况",
            "通知上级部门",
            "启动应急预案",
        ])

        # 确定资源需求
        resources = self._estimate_resources(event)

        # 通信计划
        comm_plan = self._generate_comm_plan(event, response_level)

        response = EmergencyResponse(
            event_id=event.event_id,
            response_level=response_level,
            lead_agency=self._get_lead_agency(response_level),
            actions=actions,
            resources_required=resources,
            communication_plan=comm_plan,
            evacuation_needed=event.severity.value >= FaultSeverity.CRITICAL.value,
        )

        # 更新事件状态
        event.response_started = True
        event.response_actions = actions

        self.response_history.append(response)
        return response

    def resolve_emergency(self, event_id: str, timestamp: float):
        """解除应急事件"""
        if event_id in self.active_emergencies:
            event = self.active_emergencies[event_id]
            event.resolved = True
            event.resolved_time = timestamp
            del self.active_emergencies[event_id]

    def get_active_emergencies(self) -> List[EmergencyEvent]:
        """获取活动应急事件"""
        return list(self.active_emergencies.values())

    def _estimate_affected_area(self, event_type: EmergencyType,
                                location: str) -> List[str]:
        """估计影响范围"""
        # 简化实现
        if event_type == EmergencyType.POLLUTION:
            # 污染影响下游
            if 'pool_' in location:
                pool_id = int(location.split('_')[1])
                return [f'pool_{i}' for i in range(pool_id, pool_id + 5)]
        elif event_type == EmergencyType.FLOOD:
            return [f'pool_{i}' for i in range(10)]  # 全线影响
        return [location]

    def _estimate_duration(self, event_type: EmergencyType,
                          severity: FaultSeverity) -> float:
        """估计持续时间（小时）"""
        base_duration = {
            EmergencyType.POLLUTION: 24.0,
            EmergencyType.FLOOD: 48.0,
            EmergencyType.EQUIPMENT_FAILURE: 4.0,
            EmergencyType.POWER_OUTAGE: 2.0,
            EmergencyType.ICE_JAM: 72.0,
        }

        duration = base_duration.get(event_type, 12.0)

        # 严重程度调整
        duration *= (1 + severity.value * 0.5)

        return duration

    def _determine_response_level(self, event: EmergencyEvent) -> int:
        """确定响应级别 (1最高, 4最低)"""
        if event.severity.value >= FaultSeverity.EMERGENCY.value:
            return 1
        elif event.severity.value >= FaultSeverity.CRITICAL.value:
            return 2
        elif event.severity.value >= FaultSeverity.MAJOR.value:
            return 3
        return 4

    def _estimate_resources(self, event: EmergencyEvent) -> Dict[str, int]:
        """估计资源需求"""
        base_resources = {
            'personnel': 5,
            'vehicles': 2,
            'equipment_sets': 1,
        }

        # 根据严重程度调整
        multiplier = 1 + event.severity.value

        return {k: v * multiplier for k, v in base_resources.items()}

    def _get_lead_agency(self, response_level: int) -> str:
        """获取牵头单位"""
        agencies = {
            1: "水利部应急指挥中心",
            2: "流域管理局",
            3: "管理处",
            4: "现场管理所",
        }
        return agencies.get(response_level, "现场管理所")

    def _generate_comm_plan(self, event: EmergencyEvent,
                           response_level: int) -> List[str]:
        """生成通信计划"""
        plan = ["通知现场人员"]

        if response_level <= 3:
            plan.append("通知管理处值班室")
        if response_level <= 2:
            plan.append("通知流域管理局")
        if response_level <= 1:
            plan.append("通知水利部应急办")
            plan.append("启动新闻发布预案")

        return plan


# ============ 运行规则引擎 ============

@dataclass
class OperatingRule:
    """运行规则"""
    rule_id: str
    name: str
    priority: int               # 优先级 1-10 (1最高)
    condition: str              # 条件描述
    action: str                 # 动作描述

    # 规则函数
    condition_func: Optional[Callable] = None
    action_func: Optional[Callable] = None

    # 元数据
    category: str = "general"
    enabled: bool = True
    last_triggered: Optional[float] = None
    trigger_count: int = 0


class RuleCondition:
    """规则条件"""

    @staticmethod
    def level_above(pool_id: int, threshold: float):
        """水位高于阈值"""
        def check(state: Dict) -> bool:
            level = state.get(f'pool_{pool_id}_level', 0)
            return level > threshold
        return check

    @staticmethod
    def level_below(pool_id: int, threshold: float):
        """水位低于阈值"""
        def check(state: Dict) -> bool:
            level = state.get(f'pool_{pool_id}_level', 0)
            return level < threshold
        return check

    @staticmethod
    def flow_above(location: str, threshold: float):
        """流量高于阈值"""
        def check(state: Dict) -> bool:
            flow = state.get(f'{location}_flow', 0)
            return flow > threshold
        return check

    @staticmethod
    def gate_position(gate_id: int, position: float, tolerance: float = 0.05):
        """闸门位置接近某值"""
        def check(state: Dict) -> bool:
            pos = state.get(f'gate_{gate_id}_position', 0)
            return abs(pos - position) < tolerance
        return check

    @staticmethod
    def time_in_range(start_hour: int, end_hour: int):
        """时间在范围内"""
        def check(state: Dict) -> bool:
            hour = state.get('current_hour', 0)
            if start_hour <= end_hour:
                return start_hour <= hour < end_hour
            else:  # 跨午夜
                return hour >= start_hour or hour < end_hour
        return check

    @staticmethod
    def fault_active(fault_type: FaultType):
        """存在特定故障"""
        def check(state: Dict) -> bool:
            faults = state.get('active_faults', [])
            return any(f.fault_type == fault_type for f in faults)
        return check


class RuleAction:
    """规则动作"""

    @staticmethod
    def set_gate_position(gate_id: int, position: float):
        """设置闸门位置"""
        def execute(state: Dict, controller: Any) -> Dict:
            return {
                'action': 'set_gate',
                'gate_id': gate_id,
                'position': position,
            }
        return execute

    @staticmethod
    def adjust_gate(gate_id: int, delta: float):
        """调整闸门开度"""
        def execute(state: Dict, controller: Any) -> Dict:
            current = state.get(f'gate_{gate_id}_position', 0.5)
            new_pos = max(0, min(1, current + delta))
            return {
                'action': 'set_gate',
                'gate_id': gate_id,
                'position': new_pos,
            }
        return execute

    @staticmethod
    def send_alert(message: str, level: str = 'warning'):
        """发送告警"""
        def execute(state: Dict, controller: Any) -> Dict:
            return {
                'action': 'alert',
                'message': message,
                'level': level,
            }
        return execute

    @staticmethod
    def switch_control_mode(mode: ControlMode):
        """切换控制模式"""
        def execute(state: Dict, controller: Any) -> Dict:
            return {
                'action': 'switch_mode',
                'mode': mode,
            }
        return execute

    @staticmethod
    def emergency_shutdown():
        """紧急停机"""
        def execute(state: Dict, controller: Any) -> Dict:
            return {
                'action': 'emergency_shutdown',
            }
        return execute


class OperatingRuleEngine:
    """运行规则引擎"""

    def __init__(self):
        self.rules: Dict[str, OperatingRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}

        # 规则执行历史
        self.execution_history: List[Tuple[float, str, Dict]] = []

        # 设置默认规则
        self._setup_default_rules()

    def _setup_default_rules(self):
        """设置默认运行规则"""

        # 规则1: 高水位报警
        self.add_rule(OperatingRule(
            rule_id="R001",
            name="高水位报警",
            priority=1,
            condition="任一渠池水位超过3.8m",
            action="发送高水位告警",
            condition_func=lambda s: any(
                s.get(f'pool_{i}_level', 0) > 3.8 for i in range(10)
            ),
            action_func=RuleAction.send_alert("High water level detected", "critical"),
            category="safety",
        ))

        # 规则2: 低水位报警
        self.add_rule(OperatingRule(
            rule_id="R002",
            name="低水位报警",
            priority=1,
            condition="任一渠池水位低于1.2m",
            action="发送低水位告警",
            condition_func=lambda s: any(
                s.get(f'pool_{i}_level', 0) < 1.2 for i in range(10)
            ),
            action_func=RuleAction.send_alert("Low water level detected", "warning"),
            category="safety",
        ))

        # 规则3: 夜间节能
        self.add_rule(OperatingRule(
            rule_id="R003",
            name="夜间节能模式",
            priority=5,
            condition="时间在22:00-06:00之间",
            action="降低控制频率",
            condition_func=RuleCondition.time_in_range(22, 6),
            action_func=lambda s, c: {'action': 'reduce_frequency', 'factor': 0.5},
            category="optimization",
        ))

        # 规则4: 闸门卡死应急
        self.add_rule(OperatingRule(
            rule_id="R004",
            name="闸门卡死应急",
            priority=1,
            condition="检测到执行器卡死故障",
            action="切换到应急模式",
            condition_func=RuleCondition.fault_active(FaultType.ACTUATOR_STUCK),
            action_func=RuleAction.switch_control_mode(ControlMode.EMERGENCY),
            category="fault_response",
        ))

    def add_rule(self, rule: OperatingRule):
        """添加规则"""
        self.rules[rule.rule_id] = rule

        # 添加到分组
        if rule.category not in self.rule_groups:
            self.rule_groups[rule.category] = []
        self.rule_groups[rule.category].append(rule.rule_id)

    def remove_rule(self, rule_id: str):
        """删除规则"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            if rule.category in self.rule_groups:
                self.rule_groups[rule.category].remove(rule_id)
            del self.rules[rule_id]

    def enable_rule(self, rule_id: str):
        """启用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str):
        """禁用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False

    def evaluate(self, state: Dict, controller: Any = None,
                timestamp: float = 0.0) -> List[Dict]:
        """评估所有规则并执行"""
        actions = []

        # 按优先级排序
        sorted_rules = sorted(
            [r for r in self.rules.values() if r.enabled],
            key=lambda r: r.priority
        )

        for rule in sorted_rules:
            try:
                if rule.condition_func and rule.condition_func(state):
                    # 规则触发
                    rule.last_triggered = timestamp
                    rule.trigger_count += 1

                    # 执行动作
                    if rule.action_func:
                        action = rule.action_func(state, controller)
                        action['rule_id'] = rule.rule_id
                        action['rule_name'] = rule.name
                        actions.append(action)

                        self.execution_history.append((timestamp, rule.rule_id, action))
            except Exception as e:
                # 规则执行失败，记录但不中断
                actions.append({
                    'rule_id': rule.rule_id,
                    'action': 'error',
                    'error': str(e),
                })

        return actions

    def get_rules_by_category(self, category: str) -> List[OperatingRule]:
        """按分类获取规则"""
        rule_ids = self.rule_groups.get(category, [])
        return [self.rules[rid] for rid in rule_ids if rid in self.rules]

    def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计"""
        return {
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for r in self.rules.values() if r.enabled),
            'categories': list(self.rule_groups.keys()),
            'total_executions': len(self.execution_history),
            'top_triggered': sorted(
                [(r.rule_id, r.trigger_count) for r in self.rules.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }


# ============ 综合故障诊断与容错系统 ============

class FaultTolerantSystem:
    """综合故障诊断与容错控制系统"""

    def __init__(self, num_pools: int = 5):
        self.num_pools = num_pools

        # 子系统
        self.detection_engine = FaultDetectionEngine(num_pools)
        self.diagnosis_engine = FaultDiagnosisEngine()
        self.ftc_controller = FaultTolerantController(num_pools)
        self.emergency_system = EmergencyResponseSystem()
        self.rule_engine = OperatingRuleEngine()

        # 系统状态
        self.system_health = 1.0  # 系统健康度 0-1
        self.current_control_mode = ControlMode.NORMAL

        # 事件回调
        self.on_fault_detected: Optional[Callable] = None
        self.on_mode_change: Optional[Callable] = None
        self.on_emergency: Optional[Callable] = None

    def process_step(self, state: Dict, timestamp: float) -> Dict[str, Any]:
        """处理一步"""
        result = {
            'timestamp': timestamp,
            'faults_detected': [],
            'diagnosis': [],
            'reconfiguration': None,
            'emergency_response': None,
            'rule_actions': [],
            'system_health': self.system_health,
        }

        # 1. 故障检测
        faults = []

        # 传感器故障检测
        for i in range(self.num_pools):
            level = state.get(f'pool_{i}_level', 0)
            flow = state.get(f'pool_{i}_flow', 0)

            level_faults = self.detection_engine.process_sensor_reading(
                f'level_sensor_{i}', level, 'level', f'pool_{i}', timestamp
            )
            flow_faults = self.detection_engine.process_sensor_reading(
                f'flow_sensor_{i}', flow, 'flow', f'pool_{i}', timestamp
            )

            faults.extend(level_faults)
            faults.extend(flow_faults)

        # 执行器故障检测
        for i in range(self.num_pools + 1):
            command = state.get(f'gate_{i}_command', 0.5)
            position = state.get(f'gate_{i}_position', 0.5)

            actuator_faults = self.detection_engine.process_actuator_status(
                f'gate_{i}', command, position, f'gate_{i}', timestamp
            )
            faults.extend(actuator_faults)

        result['faults_detected'] = faults

        # 2. 故障诊断
        if faults:
            diagnosis = self.diagnosis_engine.diagnose(faults)
            result['diagnosis'] = diagnosis

            # 触发回调
            if self.on_fault_detected:
                self.on_fault_detected(faults)

        # 3. 容错控制重构
        all_faults = self.detection_engine.get_active_faults()
        if all_faults:
            reconfig = self.ftc_controller.reconfigure(
                all_faults, result['diagnosis'], timestamp
            )
            result['reconfiguration'] = reconfig

            if reconfig.mode != self.current_control_mode:
                self.current_control_mode = reconfig.mode
                if self.on_mode_change:
                    self.on_mode_change(reconfig)

        # 4. 应急响应
        critical_faults = [f for f in all_faults
                         if f.severity.value >= FaultSeverity.CRITICAL.value]
        if critical_faults:
            for fault in critical_faults:
                # 判断是否需要触发应急
                if fault.fault_type in [FaultType.CONTAMINATION, FaultType.PIPE_LEAK,
                                       FaultType.OVERFLOW]:
                    emergency_type = {
                        FaultType.CONTAMINATION: EmergencyType.POLLUTION,
                        FaultType.PIPE_LEAK: EmergencyType.EQUIPMENT_FAILURE,
                        FaultType.OVERFLOW: EmergencyType.FLOOD,
                    }.get(fault.fault_type, EmergencyType.EQUIPMENT_FAILURE)

                    event = self.emergency_system.report_emergency(
                        emergency_type, fault.location, fault.description,
                        fault.severity, timestamp
                    )
                    response = self.emergency_system.generate_response(event)
                    result['emergency_response'] = response

                    if self.on_emergency:
                        self.on_emergency(event, response)

        # 5. 规则引擎
        state['active_faults'] = all_faults
        state['current_hour'] = (timestamp / 3600) % 24

        rule_actions = self.rule_engine.evaluate(state, self.ftc_controller, timestamp)
        result['rule_actions'] = rule_actions

        # 6. 更新系统健康度
        self._update_health(all_faults)
        result['system_health'] = self.system_health

        return result

    def _update_health(self, faults: List[FaultEvent]):
        """更新系统健康度"""
        if not faults:
            # 无故障时缓慢恢复
            self.system_health = min(1.0, self.system_health + 0.01)
        else:
            # 根据故障严重程度降低健康度
            max_severity = max(f.severity.value for f in faults)
            degradation = max_severity * 0.05
            self.system_health = max(0.0, self.system_health - degradation)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'health': self.system_health,
            'control_mode': self.current_control_mode.name,
            'active_faults': len(self.detection_engine.get_active_faults()),
            'active_emergencies': len(self.emergency_system.get_active_emergencies()),
            'rules_enabled': sum(1 for r in self.rule_engine.rules.values() if r.enabled),
        }

    def register_backup_sensor(self, primary_id: str, backup_id: str):
        """注册备用传感器"""
        self.ftc_controller.register_backup(primary_id, backup_id, 'sensor')

    def register_backup_actuator(self, primary_id: str, backup_id: str):
        """注册备用执行器"""
        self.ftc_controller.register_backup(primary_id, backup_id, 'actuator')

    def add_operating_rule(self, rule: OperatingRule):
        """添加运行规则"""
        self.rule_engine.add_rule(rule)


# ============ 导出 ============

__all__ = [
    # 枚举
    'FaultType',
    'FaultSeverity',
    'FaultStatus',
    'ControlMode',
    'EmergencyType',

    # 数据类
    'FaultEvent',
    'DiagnosisResult',
    'ControlReconfiguration',
    'EmergencyEvent',
    'EmergencyResponse',
    'OperatingRule',

    # 检测器
    'ResidualGenerator',
    'SensorFaultDetector',
    'ActuatorFaultDetector',
    'FaultDetectionEngine',

    # 诊断器
    'FaultDiagnosisEngine',

    # 控制器
    'FaultTolerantController',

    # 应急系统
    'EmergencyResponseSystem',

    # 规则引擎
    'RuleCondition',
    'RuleAction',
    'OperatingRuleEngine',

    # 综合系统
    'FaultTolerantSystem',
]
