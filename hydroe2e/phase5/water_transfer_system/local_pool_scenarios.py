"""
L1层渠池全场景系统 - 分钟级本地控制场景
L1 Layer Local Pool Scenario System - Minute-Level Local Control Scenarios

核心功能:
1. 污染追踪溯源 (Pollution Tracking & Tracing)
2. 边坡衬砌板漂浮检测与响应 (Slope Lining Panel Floating Detection)
3. 突发污染退水 (Emergency Pollution Discharge)
4. 检修退水 (Maintenance Discharge)
5. 水位异常检测与响应 (Water Level Anomaly Detection)
6. 闸门故障处理 (Gate Failure Handling)
7. 渗漏检测与响应 (Leakage Detection & Response)
8. 冰凌检测与处置 (Ice Detection & Handling)

时间尺度:
- L3: 旬/日级 (240-1440分钟)
- L2: 小时级 (60分钟)
- L1: 分钟级 (1-15分钟)

数学模型:
- 污染传播: C(x,t) = C0 * exp(-k*(x-v*t)) * H(x-v*t)
- 边坡稳定: FS = (c' + (γ_sat - γ_w)*H*cos²β*tanφ') / (γ_sat*H*sinβ*cosβ)
- 退水流量: Q_drain = μ*A*sqrt(2*g*h)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto
import logging
import time
from collections import deque
import math

from .core_types import PoolRole, ScenarioType, ScenarioSeverity

logger = logging.getLogger(__name__)


# ==============================================================================
# L1层场景类型枚举
# ==============================================================================

class L1ScenarioType(Enum):
    """L1层本地场景类型"""
    # 污染类 (Pollution)
    L1_POLLUTION_DETECTED = "污染检测"              # 检测到污染
    L1_POLLUTION_TRACKING = "污染追踪"              # 污染追踪
    L1_POLLUTION_TRACING = "污染溯源"               # 污染溯源
    L1_POLLUTION_DISCHARGE = "污染退水"             # 污染退水
    L1_POLLUTION_ISOLATION = "污染隔离"             # 污染隔离

    # 边坡类 (Slope)
    L1_SLOPE_GROUNDWATER = "地下水位异常"           # 地下水位异常
    L1_SLOPE_RAINFALL = "降雨渗透"                  # 降雨渗透
    L1_SLOPE_PANEL_FLOAT = "衬砌板漂浮"             # 衬砌板漂浮检测
    L1_SLOPE_PANEL_CRACK = "衬砌板裂缝"             # 衬砌板裂缝
    L1_SLOPE_INSTABILITY = "边坡失稳"               # 边坡失稳预警

    # 退水类 (Discharge)
    L1_DISCHARGE_EMERGENCY = "紧急退水"             # 紧急退水
    L1_DISCHARGE_MAINTENANCE = "检修退水"           # 检修退水
    L1_DISCHARGE_POLLUTION = "污染退水"             # 污染退水
    L1_DISCHARGE_FLOOD = "防洪退水"                 # 防洪退水

    # 水位类 (Water Level)
    L1_LEVEL_HIGH = "高水位报警"                    # 高水位报警
    L1_LEVEL_LOW = "低水位报警"                     # 低水位报警
    L1_LEVEL_RAPID_RISE = "水位快速上涨"            # 水位快速上涨
    L1_LEVEL_RAPID_DROP = "水位快速下降"            # 水位快速下降
    L1_LEVEL_OSCILLATION = "水位震荡"               # 水位震荡

    # 闸门类 (Gate)
    L1_GATE_STUCK = "闸门卡阻"                      # 闸门卡阻
    L1_GATE_LEAK = "闸门漏水"                       # 闸门漏水
    L1_GATE_CONTROL_FAIL = "闸门控制失效"           # 闸门控制失效
    L1_GATE_SENSOR_FAIL = "闸门传感器故障"          # 闸门传感器故障

    # 渗漏类 (Leakage)
    L1_LEAKAGE_MINOR = "轻微渗漏"                   # 轻微渗漏
    L1_LEAKAGE_MODERATE = "中度渗漏"                # 中度渗漏
    L1_LEAKAGE_SEVERE = "严重渗漏"                  # 严重渗漏
    L1_LEAKAGE_PIPE_BURST = "管道爆裂"              # 管道爆裂

    # 冰凌类 (Ice)
    L1_ICE_FORMATION = "冰凌形成"                   # 冰凌形成
    L1_ICE_ACCUMULATION = "冰凌堆积"                # 冰凌堆积
    L1_ICE_BLOCKAGE = "冰凌堵塞"                    # 冰凌堵塞
    L1_ICE_DAM = "冰坝形成"                         # 冰坝形成

    # 常规类 (Normal)
    L1_NORMAL = "正常运行"                          # 正常运行
    L1_STANDBY = "待命状态"                         # 待命状态


class L1ActionType(Enum):
    """L1层响应动作类型"""
    # 闸门动作
    GATE_CLOSE = "关闭闸门"
    GATE_OPEN = "开启闸门"
    GATE_ADJUST = "调整闸门"
    GATE_LOCK = "锁定闸门"

    # 退水动作
    DRAIN_START = "启动退水"
    DRAIN_STOP = "停止退水"
    DRAIN_ADJUST = "调整退水流量"

    # 报警动作
    ALARM_TRIGGER = "触发报警"
    ALARM_ESCALATE = "上报上级"
    ALARM_CLEAR = "清除报警"

    # 隔离动作
    ISOLATE_UPSTREAM = "隔离上游"
    ISOLATE_DOWNSTREAM = "隔离下游"
    ISOLATE_BOTH = "双向隔离"

    # 监测动作
    MONITOR_ENHANCE = "加强监测"
    MONITOR_NORMAL = "常规监测"

    # 协调动作
    COORDINATE_UPSTREAM = "协调上游"
    COORDINATE_DOWNSTREAM = "协调下游"


# ==============================================================================
# L1层场景事件
# ==============================================================================

@dataclass
class L1ScenarioEvent:
    """
    L1层场景事件
    """
    event_id: str
    scenario_type: L1ScenarioType
    pool_id: int
    severity: ScenarioSeverity

    # 时间属性 (分钟级)
    timestamp: float = 0.0                  # 发生时间 [s]
    duration: float = 300.0                 # 持续时间 [s] (默认5分钟)

    # 位置属性
    station_km: float = 0.0                 # 桩号 [km]
    position_in_pool: float = 0.5           # 池内位置 [0-1, 0=入口, 1=出口]

    # 测量值
    measured_value: float = 0.0             # 测量值 (如水位、浓度)
    threshold_value: float = 0.0            # 阈值
    deviation: float = 0.0                  # 偏差

    # 环境条件
    groundwater_level: float = 0.0          # 地下水位 [m]
    rainfall_intensity: float = 0.0         # 降雨强度 [mm/h]
    temperature: float = 15.0               # 温度 [°C]

    # 状态
    is_propagating: bool = False            # 是否正在传播
    propagation_velocity: float = 0.0       # 传播速度 [m/s]
    propagation_direction: int = 1          # 传播方向 (1=下游, -1=上游)

    # 元数据
    source_event_id: Optional[str] = None   # 源事件ID (用于追踪)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class L1ActionCommand:
    """
    L1层动作指令
    """
    command_id: str
    action_type: L1ActionType
    pool_id: int

    # 闸门参数
    gate_position: float = 0.0              # 闸门目标开度 [0-1]
    gate_rate: float = 0.1                  # 闸门调整速率 [/min]

    # 流量参数
    target_flow: float = 0.0                # 目标流量 [m³/s]
    max_flow_rate: float = 10.0             # 最大流量变化率 [m³/s/min]

    # 时间参数
    start_time: float = 0.0                 # 开始时间
    duration: float = 60.0                  # 持续时间 [s]
    priority: int = 1                       # 优先级 (1-10)

    # 约束
    level_constraint_min: float = 0.5       # 水位下限约束
    level_constraint_max: float = 8.0       # 水位上限约束

    # 状态
    is_executed: bool = False
    execution_result: str = ""


# ==============================================================================
# 污染追踪溯源模块
# ==============================================================================

@dataclass
class PollutionTrackingState:
    """污染追踪状态"""
    detected_pool: int                      # 检测到污染的池
    detection_time: float                   # 检测时间
    concentration: float                    # 浓度
    pollutant_type: str = "unknown"         # 污染物类型

    # 追踪状态
    tracked_positions: List[Tuple[int, float, float]] = field(default_factory=list)
    # [(池ID, 时间, 浓度), ...]

    # 溯源状态
    estimated_source_pool: int = -1         # 估计的源头池
    estimated_source_time: float = 0.0      # 估计的排放时间
    tracing_confidence: float = 0.0         # 溯源置信度

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


class PollutionTracker:
    """
    污染追踪器

    功能:
    1. 实时追踪污染传播
    2. 反向溯源定位源头
    3. 预测污染到达时间
    """

    def __init__(self, num_pools: int = 60):
        self.num_pools = num_pools

        # 渠池参数 (每个池约24km)
        self.pool_lengths = np.ones(num_pools) * 24000  # [m]
        self.flow_velocities = np.ones(num_pools) * 1.5  # [m/s]

        # 追踪状态
        self.active_trackings: Dict[str, PollutionTrackingState] = {}
        self.history: List[PollutionTrackingState] = []

        # 传感器数据缓存
        self.concentration_history: Dict[int, deque] = {
            i: deque(maxlen=100) for i in range(num_pools)
        }

    def detect_pollution(self,
                         pool_id: int,
                         concentration: float,
                         timestamp: float,
                         threshold: float = 0.1) -> Optional[L1ScenarioEvent]:
        """
        检测污染

        Args:
            pool_id: 渠池ID
            concentration: 浓度
            timestamp: 时间戳
            threshold: 检测阈值

        Returns:
            污染检测事件 (如果检测到)
        """
        # 存储历史
        self.concentration_history[pool_id].append((timestamp, concentration))

        # 检测是否超标
        if concentration > threshold:
            event_id = f"POLL_{pool_id}_{int(timestamp)}"

            # 创建追踪状态
            state = PollutionTrackingState(
                detected_pool=pool_id,
                detection_time=timestamp,
                concentration=concentration,
            )
            state.tracked_positions.append((pool_id, timestamp, concentration))
            self.active_trackings[event_id] = state

            # 确定严重程度
            if concentration > threshold * 10:
                severity = ScenarioSeverity.CRITICAL
            elif concentration > threshold * 5:
                severity = ScenarioSeverity.HIGH
            elif concentration > threshold * 2:
                severity = ScenarioSeverity.MEDIUM
            else:
                severity = ScenarioSeverity.LOW

            return L1ScenarioEvent(
                event_id=event_id,
                scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
                pool_id=pool_id,
                severity=severity,
                timestamp=timestamp,
                measured_value=concentration,
                threshold_value=threshold,
                deviation=concentration - threshold,
            )

        return None

    def track_pollution(self,
                        tracking_id: str,
                        pool_id: int,
                        concentration: float,
                        timestamp: float) -> Optional[L1ScenarioEvent]:
        """
        追踪污染传播

        Args:
            tracking_id: 追踪ID
            pool_id: 当前池ID
            concentration: 当前浓度
            timestamp: 时间戳

        Returns:
            追踪事件 (如果有更新)
        """
        if tracking_id not in self.active_trackings:
            return None

        state = self.active_trackings[tracking_id]
        state.tracked_positions.append((pool_id, timestamp, concentration))

        # 计算传播速度
        if len(state.tracked_positions) >= 2:
            pos1 = state.tracked_positions[-2]
            pos2 = state.tracked_positions[-1]

            if pos2[0] != pos1[0]:  # 不同池
                distance = sum(self.pool_lengths[pos1[0]:pos2[0]])
                time_diff = pos2[1] - pos1[1]
                if time_diff > 0:
                    velocity = distance / time_diff
                    state.metadata['propagation_velocity'] = velocity

        return L1ScenarioEvent(
            event_id=f"{tracking_id}_TRACK_{int(timestamp)}",
            scenario_type=L1ScenarioType.L1_POLLUTION_TRACKING,
            pool_id=pool_id,
            severity=ScenarioSeverity.MEDIUM,
            timestamp=timestamp,
            measured_value=concentration,
            is_propagating=True,
            propagation_velocity=state.metadata.get('propagation_velocity', 1.5),
            source_event_id=tracking_id,
        )

    def trace_source(self, tracking_id: str) -> Optional[L1ScenarioEvent]:
        """
        溯源分析 - 根据传播轨迹反推源头

        Args:
            tracking_id: 追踪ID

        Returns:
            溯源事件
        """
        if tracking_id not in self.active_trackings:
            return None

        state = self.active_trackings[tracking_id]

        if len(state.tracked_positions) < 2:
            return None

        # 简化溯源: 取最早检测到的位置
        positions = sorted(state.tracked_positions, key=lambda x: x[1])
        first_detection = positions[0]

        # 估计源头在上游
        estimated_source = max(0, first_detection[0] - 1)

        # 估计排放时间 (根据流速反推)
        travel_distance = self.pool_lengths[estimated_source] * 0.5
        travel_time = travel_distance / self.flow_velocities[estimated_source]
        estimated_source_time = first_detection[1] - travel_time

        state.estimated_source_pool = estimated_source
        state.estimated_source_time = estimated_source_time
        state.tracing_confidence = min(0.9, 0.3 + 0.1 * len(positions))

        return L1ScenarioEvent(
            event_id=f"{tracking_id}_TRACE",
            scenario_type=L1ScenarioType.L1_POLLUTION_TRACING,
            pool_id=estimated_source,
            severity=ScenarioSeverity.HIGH,
            timestamp=time.time(),
            metadata={
                'estimated_source_time': estimated_source_time,
                'confidence': state.tracing_confidence,
                'tracking_positions': len(positions),
            },
            source_event_id=tracking_id,
        )

    def predict_arrival(self,
                        tracking_id: str,
                        target_pool: int) -> Tuple[float, float]:
        """
        预测污染到达目标池的时间

        Args:
            tracking_id: 追踪ID
            target_pool: 目标池

        Returns:
            (预计到达时间, 预计浓度)
        """
        if tracking_id not in self.active_trackings:
            return -1, 0

        state = self.active_trackings[tracking_id]

        if not state.tracked_positions:
            return -1, 0

        # 获取最新位置
        latest = state.tracked_positions[-1]
        current_pool = latest[0]
        current_time = latest[1]
        current_conc = latest[2]

        if target_pool <= current_pool:
            return current_time, current_conc

        # 计算距离和时间
        distance = sum(self.pool_lengths[current_pool:target_pool])
        velocity = state.metadata.get('propagation_velocity', 1.5)
        travel_time = distance / velocity

        # 浓度衰减 (假设一阶衰减)
        decay_rate = 0.0001  # [1/s]
        predicted_conc = current_conc * np.exp(-decay_rate * travel_time)

        return current_time + travel_time, predicted_conc


# ==============================================================================
# 边坡衬砌板漂浮检测模块
# ==============================================================================

@dataclass
class SlopeCondition:
    """边坡状态"""
    pool_id: int
    groundwater_level: float        # 地下水位 [m]
    canal_level: float              # 渠道水位 [m]
    rainfall_24h: float             # 24小时累计降雨 [mm]
    panel_uplift_pressure: float    # 衬砌板上托压力 [kPa]

    # 安全系数
    safety_factor: float = 1.5
    is_stable: bool = True

    # 裂缝状态
    crack_width: float = 0.0        # 裂缝宽度 [mm]
    crack_length: float = 0.0       # 裂缝长度 [m]


class SlopePanelMonitor:
    """
    边坡衬砌板监测器

    监测内容:
    1. 地下水位与渠道水位差
    2. 降雨渗透影响
    3. 上托压力计算
    4. 稳定性分析
    """

    # 设计参数
    PANEL_WEIGHT = 5.0              # 衬砌板自重 [kPa]
    CRITICAL_UPLIFT = 4.0           # 临界上托压力 [kPa]
    MIN_SAFETY_FACTOR = 1.2         # 最小安全系数

    # 阈值
    GROUNDWATER_ALERT = 0.5         # 地下水位报警差值 [m]
    RAINFALL_ALERT_24H = 50.0       # 24小时降雨报警值 [mm]

    def __init__(self, num_pools: int = 60):
        self.num_pools = num_pools

        # 状态存储
        self.conditions: Dict[int, SlopeCondition] = {}

        # 历史记录
        self.history: Dict[int, deque] = {
            i: deque(maxlen=288) for i in range(num_pools)  # 24小时, 5分钟间隔
        }

    def update_condition(self,
                         pool_id: int,
                         groundwater_level: float,
                         canal_level: float,
                         rainfall_1h: float = 0.0) -> Optional[L1ScenarioEvent]:
        """
        更新边坡状态

        Args:
            pool_id: 渠池ID
            groundwater_level: 地下水位 [m]
            canal_level: 渠道水位 [m]
            rainfall_1h: 最近1小时降雨 [mm]

        Returns:
            事件 (如果检测到异常)
        """
        # 计算24小时累计降雨
        self.history[pool_id].append(rainfall_1h)
        rainfall_24h = sum(self.history[pool_id])

        # 计算上托压力
        # P_uplift = γ_w * (H_gw - H_canal) when H_gw > H_canal
        gamma_w = 10.0  # 水的容重 [kN/m³]
        level_diff = groundwater_level - canal_level

        if level_diff > 0:
            uplift_pressure = gamma_w * level_diff  # [kPa]
        else:
            uplift_pressure = 0

        # 降雨影响 (增加地下水位)
        rainfall_effect = rainfall_24h * 0.01  # 假设10%渗透

        # 总上托压力
        total_uplift = uplift_pressure + rainfall_effect * gamma_w

        # 安全系数
        safety_factor = self.PANEL_WEIGHT / max(0.1, total_uplift)

        # 判断稳定性
        is_stable = safety_factor >= self.MIN_SAFETY_FACTOR

        # 更新状态
        condition = SlopeCondition(
            pool_id=pool_id,
            groundwater_level=groundwater_level,
            canal_level=canal_level,
            rainfall_24h=rainfall_24h,
            panel_uplift_pressure=total_uplift,
            safety_factor=safety_factor,
            is_stable=is_stable,
        )
        self.conditions[pool_id] = condition

        # 生成事件
        events = []

        # 地下水位异常
        if level_diff > self.GROUNDWATER_ALERT:
            events.append(L1ScenarioEvent(
                event_id=f"GW_{pool_id}_{int(time.time())}",
                scenario_type=L1ScenarioType.L1_SLOPE_GROUNDWATER,
                pool_id=pool_id,
                severity=self._calculate_severity(level_diff / self.GROUNDWATER_ALERT),
                groundwater_level=groundwater_level,
                measured_value=level_diff,
                threshold_value=self.GROUNDWATER_ALERT,
            ))

        # 降雨报警
        if rainfall_24h > self.RAINFALL_ALERT_24H:
            events.append(L1ScenarioEvent(
                event_id=f"RAIN_{pool_id}_{int(time.time())}",
                scenario_type=L1ScenarioType.L1_SLOPE_RAINFALL,
                pool_id=pool_id,
                severity=self._calculate_severity(rainfall_24h / self.RAINFALL_ALERT_24H),
                rainfall_intensity=rainfall_24h / 24,
                measured_value=rainfall_24h,
                threshold_value=self.RAINFALL_ALERT_24H,
            ))

        # 衬砌板漂浮预警
        if not is_stable:
            events.append(L1ScenarioEvent(
                event_id=f"PANEL_{pool_id}_{int(time.time())}",
                scenario_type=L1ScenarioType.L1_SLOPE_PANEL_FLOAT,
                pool_id=pool_id,
                severity=ScenarioSeverity.CRITICAL if safety_factor < 1.0 else ScenarioSeverity.HIGH,
                measured_value=safety_factor,
                threshold_value=self.MIN_SAFETY_FACTOR,
                metadata={
                    'uplift_pressure': total_uplift,
                    'panel_weight': self.PANEL_WEIGHT,
                },
            ))

        return events[0] if events else None

    def _calculate_severity(self, ratio: float) -> ScenarioSeverity:
        """根据比值计算严重程度"""
        if ratio > 3.0:
            return ScenarioSeverity.CRITICAL
        elif ratio > 2.0:
            return ScenarioSeverity.HIGH
        elif ratio > 1.5:
            return ScenarioSeverity.MEDIUM
        else:
            return ScenarioSeverity.LOW

    def recommend_action(self, pool_id: int) -> Optional[L1ActionCommand]:
        """
        推荐响应动作

        Args:
            pool_id: 渠池ID

        Returns:
            动作指令
        """
        if pool_id not in self.conditions:
            return None

        condition = self.conditions[pool_id]

        if not condition.is_stable:
            # 衬砌板漂浮风险 -> 建议降低渠道水位
            target_level = condition.groundwater_level - 0.3  # 比地下水位低0.3m

            return L1ActionCommand(
                command_id=f"SLOPE_ACTION_{pool_id}",
                action_type=L1ActionType.GATE_OPEN,
                pool_id=pool_id,
                gate_position=0.8,  # 开大闸门
                priority=8,
                level_constraint_max=target_level,
                duration=1800,  # 30分钟
            )

        return None


# ==============================================================================
# 退水管理模块
# ==============================================================================

class DischargeType(Enum):
    """退水类型"""
    EMERGENCY_POLLUTION = "紧急污染退水"
    PLANNED_MAINTENANCE = "计划检修退水"
    FLOOD_CONTROL = "防洪退水"
    PANEL_PROTECTION = "衬砌板保护退水"
    ICE_REMOVAL = "冰凌清除退水"


@dataclass
class DischargeRequest:
    """退水请求"""
    request_id: str
    discharge_type: DischargeType
    pool_id: int
    priority: int

    target_level: float             # 目标水位 [m]
    max_rate: float                 # 最大退水速率 [m³/s]
    max_level_drop_rate: float      # 最大水位下降速率 [m/h]

    start_time: float = 0.0
    estimated_duration: float = 0.0

    # 约束
    downstream_capacity: float = 0.0  # 下游承接能力


class DischargeManager:
    """
    退水管理器

    功能:
    1. 退水请求处理
    2. 退水流量计算
    3. 下游协调
    4. 进度监控
    """

    # 退水参数
    MAX_LEVEL_DROP_RATE = 0.5       # 最大水位下降速率 [m/h]
    MIN_RESIDUAL_LEVEL = 0.3       # 最小残余水位 [m]

    def __init__(self, num_pools: int = 60):
        self.num_pools = num_pools

        # 活动退水
        self.active_discharges: Dict[str, DischargeRequest] = {}

        # 退水闸参数
        self.discharge_gate_capacity = np.ones(num_pools) * 50.0  # [m³/s]

    def request_discharge(self,
                          pool_id: int,
                          discharge_type: DischargeType,
                          target_level: float,
                          current_level: float,
                          priority: int = 5) -> Tuple[DischargeRequest, L1ScenarioEvent]:
        """
        请求退水

        Args:
            pool_id: 渠池ID
            discharge_type: 退水类型
            target_level: 目标水位
            current_level: 当前水位
            priority: 优先级

        Returns:
            (退水请求, 场景事件)
        """
        request_id = f"DISCHARGE_{pool_id}_{int(time.time())}"

        # 计算退水量和时间
        pool_area = 100000  # 假设水面面积 [m²]
        volume_to_discharge = pool_area * (current_level - target_level)

        # 最大退水流量
        max_rate = min(
            self.discharge_gate_capacity[pool_id],
            pool_area * self.MAX_LEVEL_DROP_RATE / 3600  # 转换为m³/s
        )

        # 预计时间
        estimated_duration = volume_to_discharge / max_rate if max_rate > 0 else 0

        request = DischargeRequest(
            request_id=request_id,
            discharge_type=discharge_type,
            pool_id=pool_id,
            priority=priority,
            target_level=max(target_level, self.MIN_RESIDUAL_LEVEL),
            max_rate=max_rate,
            max_level_drop_rate=self.MAX_LEVEL_DROP_RATE,
            start_time=time.time(),
            estimated_duration=estimated_duration,
        )

        self.active_discharges[request_id] = request

        # 确定场景类型
        scenario_type_map = {
            DischargeType.EMERGENCY_POLLUTION: L1ScenarioType.L1_DISCHARGE_POLLUTION,
            DischargeType.PLANNED_MAINTENANCE: L1ScenarioType.L1_DISCHARGE_MAINTENANCE,
            DischargeType.FLOOD_CONTROL: L1ScenarioType.L1_DISCHARGE_FLOOD,
            DischargeType.PANEL_PROTECTION: L1ScenarioType.L1_DISCHARGE_EMERGENCY,
            DischargeType.ICE_REMOVAL: L1ScenarioType.L1_DISCHARGE_EMERGENCY,
        }

        event = L1ScenarioEvent(
            event_id=request_id,
            scenario_type=scenario_type_map.get(discharge_type, L1ScenarioType.L1_DISCHARGE_EMERGENCY),
            pool_id=pool_id,
            severity=ScenarioSeverity.HIGH if priority > 7 else ScenarioSeverity.MEDIUM,
            duration=estimated_duration,
            metadata={
                'discharge_type': discharge_type.value,
                'target_level': target_level,
                'max_rate': max_rate,
                'estimated_duration': estimated_duration,
            },
        )

        return request, event

    def generate_discharge_commands(self, request_id: str) -> List[L1ActionCommand]:
        """
        生成退水动作指令

        Args:
            request_id: 请求ID

        Returns:
            动作指令列表
        """
        if request_id not in self.active_discharges:
            return []

        request = self.active_discharges[request_id]
        commands = []

        # 主退水指令
        commands.append(L1ActionCommand(
            command_id=f"{request_id}_DRAIN",
            action_type=L1ActionType.DRAIN_START,
            pool_id=request.pool_id,
            target_flow=request.max_rate,
            duration=request.estimated_duration,
            priority=request.priority,
            level_constraint_min=request.target_level,
        ))

        # 上游协调指令 (减少入流)
        if request.pool_id > 0:
            commands.append(L1ActionCommand(
                command_id=f"{request_id}_UPSTREAM",
                action_type=L1ActionType.COORDINATE_UPSTREAM,
                pool_id=request.pool_id - 1,
                gate_position=0.3,  # 减小上游闸门开度
                priority=request.priority - 1,
            ))

        # 下游协调指令 (增加出流能力)
        if request.pool_id < self.num_pools - 1:
            commands.append(L1ActionCommand(
                command_id=f"{request_id}_DOWNSTREAM",
                action_type=L1ActionType.COORDINATE_DOWNSTREAM,
                pool_id=request.pool_id + 1,
                gate_position=0.8,
                priority=request.priority - 1,
            ))

        return commands


# ==============================================================================
# L1场景生成器
# ==============================================================================

class L1ScenarioGenerator:
    """
    L1层场景生成器

    生成分钟级本地场景:
    - 污染场景
    - 边坡场景
    - 退水场景
    - 水位异常场景
    - 闸门故障场景
    - 渗漏场景
    - 冰凌场景
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.generated_count = 0

    def generate_pollution_event(self,
                                  pool_id: int = None,
                                  severity: ScenarioSeverity = None) -> L1ScenarioEvent:
        """生成污染事件"""
        if pool_id is None:
            pool_id = np.random.randint(0, 60)
        if severity is None:
            severity = np.random.choice(list(ScenarioSeverity))

        self.generated_count += 1

        # 根据严重程度设置浓度
        concentration_map = {
            ScenarioSeverity.LOW: 0.15,
            ScenarioSeverity.MEDIUM: 0.3,
            ScenarioSeverity.HIGH: 0.6,
            ScenarioSeverity.CRITICAL: 1.0,
        }

        return L1ScenarioEvent(
            event_id=f"L1_POLL_{self.generated_count:06d}",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=pool_id,
            severity=severity,
            measured_value=concentration_map.get(severity, 0.3),
            threshold_value=0.1,
            position_in_pool=np.random.uniform(0, 1),
        )

    def generate_slope_event(self,
                              pool_id: int = None,
                              groundwater_level: float = None) -> L1ScenarioEvent:
        """生成边坡事件"""
        if pool_id is None:
            pool_id = np.random.randint(0, 60)
        if groundwater_level is None:
            groundwater_level = np.random.uniform(3.0, 5.0)

        self.generated_count += 1

        # 假设渠道水位
        canal_level = 4.0

        # 判断类型
        if groundwater_level > canal_level + 0.5:
            scenario_type = L1ScenarioType.L1_SLOPE_PANEL_FLOAT
            severity = ScenarioSeverity.HIGH
        elif groundwater_level > canal_level:
            scenario_type = L1ScenarioType.L1_SLOPE_GROUNDWATER
            severity = ScenarioSeverity.MEDIUM
        else:
            scenario_type = L1ScenarioType.L1_NORMAL
            severity = ScenarioSeverity.LOW

        return L1ScenarioEvent(
            event_id=f"L1_SLOPE_{self.generated_count:06d}",
            scenario_type=scenario_type,
            pool_id=pool_id,
            severity=severity,
            groundwater_level=groundwater_level,
            measured_value=groundwater_level - canal_level,
        )

    def generate_discharge_event(self,
                                  pool_id: int = None,
                                  discharge_type: DischargeType = None) -> L1ScenarioEvent:
        """生成退水事件"""
        if pool_id is None:
            pool_id = np.random.randint(0, 60)
        if discharge_type is None:
            discharge_type = np.random.choice(list(DischargeType))

        self.generated_count += 1

        scenario_type_map = {
            DischargeType.EMERGENCY_POLLUTION: L1ScenarioType.L1_DISCHARGE_POLLUTION,
            DischargeType.PLANNED_MAINTENANCE: L1ScenarioType.L1_DISCHARGE_MAINTENANCE,
            DischargeType.FLOOD_CONTROL: L1ScenarioType.L1_DISCHARGE_FLOOD,
        }

        return L1ScenarioEvent(
            event_id=f"L1_DRAIN_{self.generated_count:06d}",
            scenario_type=scenario_type_map.get(discharge_type, L1ScenarioType.L1_DISCHARGE_EMERGENCY),
            pool_id=pool_id,
            severity=ScenarioSeverity.MEDIUM,
            metadata={'discharge_type': discharge_type.value},
        )

    def generate_level_event(self,
                              pool_id: int = None,
                              level_type: str = None) -> L1ScenarioEvent:
        """生成水位事件"""
        if pool_id is None:
            pool_id = np.random.randint(0, 60)
        if level_type is None:
            level_type = np.random.choice(['high', 'low', 'rise', 'drop', 'oscillation'])

        self.generated_count += 1

        type_map = {
            'high': L1ScenarioType.L1_LEVEL_HIGH,
            'low': L1ScenarioType.L1_LEVEL_LOW,
            'rise': L1ScenarioType.L1_LEVEL_RAPID_RISE,
            'drop': L1ScenarioType.L1_LEVEL_RAPID_DROP,
            'oscillation': L1ScenarioType.L1_LEVEL_OSCILLATION,
        }

        return L1ScenarioEvent(
            event_id=f"L1_LEVEL_{self.generated_count:06d}",
            scenario_type=type_map.get(level_type, L1ScenarioType.L1_LEVEL_HIGH),
            pool_id=pool_id,
            severity=ScenarioSeverity.MEDIUM,
        )

    def generate_gate_event(self, pool_id: int = None) -> L1ScenarioEvent:
        """生成闸门事件"""
        if pool_id is None:
            pool_id = np.random.randint(0, 60)

        self.generated_count += 1

        gate_types = [
            L1ScenarioType.L1_GATE_STUCK,
            L1ScenarioType.L1_GATE_LEAK,
            L1ScenarioType.L1_GATE_CONTROL_FAIL,
            L1ScenarioType.L1_GATE_SENSOR_FAIL,
        ]

        return L1ScenarioEvent(
            event_id=f"L1_GATE_{self.generated_count:06d}",
            scenario_type=np.random.choice(gate_types),
            pool_id=pool_id,
            severity=np.random.choice([ScenarioSeverity.MEDIUM, ScenarioSeverity.HIGH]),
        )

    def generate_batch(self, count: int) -> List[L1ScenarioEvent]:
        """批量生成L1场景"""
        events = []

        # 分配比例
        pollution_count = int(count * 0.25)
        slope_count = int(count * 0.20)
        discharge_count = int(count * 0.15)
        level_count = int(count * 0.20)
        gate_count = int(count * 0.10)
        other_count = count - pollution_count - slope_count - discharge_count - level_count - gate_count

        for _ in range(pollution_count):
            events.append(self.generate_pollution_event())

        for _ in range(slope_count):
            events.append(self.generate_slope_event())

        for _ in range(discharge_count):
            events.append(self.generate_discharge_event())

        for _ in range(level_count):
            events.append(self.generate_level_event())

        for _ in range(gate_count):
            events.append(self.generate_gate_event())

        # 其他随机类型
        for _ in range(other_count):
            gen_func = np.random.choice([
                self.generate_pollution_event,
                self.generate_slope_event,
                self.generate_level_event,
            ])
            events.append(gen_func())

        np.random.shuffle(events)

        return events

    def count_possible_scenarios(self) -> Dict[str, int]:
        """计算可能的场景数"""
        n_types = len(L1ScenarioType)
        n_pools = 60
        n_severities = len(ScenarioSeverity)

        return {
            'l1_types': n_types,  # 29种L1场景类型
            'single_basic': n_types * n_pools * n_severities,  # ~7000
            'with_position': n_types * n_pools * n_severities * 10,  # ~70000 (位置离散化)
            'with_environment': n_types * n_pools * n_severities * 10 * 5 * 5,  # ~1.7M
        }


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info(" " * 15 + "L1层渠池全场景系统测试")
    logger.info("=" * 70)

    # 测试场景生成
    logger.info(f"\n{'=' * 70}")
    logger.info("测试1: L1场景生成")
    logger.info('=' * 70)

    generator = L1ScenarioGenerator(seed=42)
    events = generator.generate_batch(100)

    logger.info(f"生成场景数: {len(events)}")

    # 统计类型分布
    type_counts = {}
    for e in events:
        t = e.scenario_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    logger.info("\n类型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {t}: {c}")

    # 测试污染追踪
    logger.info(f"\n{'=' * 70}")
    logger.info("测试2: 污染追踪溯源")
    logger.info('=' * 70)

    tracker = PollutionTracker(num_pools=60)

    # 模拟污染检测
    event = tracker.detect_pollution(pool_id=30, concentration=0.5, timestamp=0)
    if event:
        logger.info(f"检测到污染: {event.event_id}")
        logger.info(f"  位置: 池{event.pool_id}")
        logger.info(f"  浓度: {event.measured_value}")
        logger.info(f"  严重程度: {event.severity.value}")

    # 模拟追踪
    tracking_id = event.event_id if event else "TEST"
    tracker.track_pollution(tracking_id, pool_id=31, concentration=0.4, timestamp=1800)
    tracker.track_pollution(tracking_id, pool_id=32, concentration=0.35, timestamp=3600)

    # 溯源
    trace_event = tracker.trace_source(tracking_id)
    if trace_event:
        logger.info(f"\n溯源结果: {trace_event.event_id}")
        logger.info(f"  估计源头: 池{trace_event.pool_id}")
        logger.info(f"  置信度: {trace_event.metadata.get('confidence', 0):.2f}")

    # 测试边坡监测
    logger.info(f"\n{'=' * 70}")
    logger.info("测试3: 边坡衬砌板监测")
    logger.info('=' * 70)

    monitor = SlopePanelMonitor(num_pools=60)

    # 模拟高地下水位
    event = monitor.update_condition(
        pool_id=25,
        groundwater_level=5.0,  # 高于渠道水位
        canal_level=4.0,
        rainfall_1h=10.0,
    )

    if event:
        logger.info(f"检测到边坡异常: {event.scenario_type.value}")
        logger.info(f"  地下水位: {event.groundwater_level}m")
        logger.info(f"  严重程度: {event.severity.value}")

    condition = monitor.conditions.get(25)
    if condition:
        logger.info(f"  安全系数: {condition.safety_factor:.2f}")
        logger.info(f"  稳定性: {'稳定' if condition.is_stable else '不稳定'}")

    # 推荐动作
    action = monitor.recommend_action(25)
    if action:
        logger.info(f"\n推荐动作: {action.action_type.value}")
        logger.info(f"  闸门开度: {action.gate_position}")
        logger.info(f"  优先级: {action.priority}")

    # 测试退水管理
    logger.info(f"\n{'=' * 70}")
    logger.info("测试4: 退水管理")
    logger.info('=' * 70)

    discharge_mgr = DischargeManager(num_pools=60)

    request, event = discharge_mgr.request_discharge(
        pool_id=30,
        discharge_type=DischargeType.EMERGENCY_POLLUTION,
        target_level=2.0,
        current_level=4.0,
        priority=9,
    )

    logger.info(f"退水请求: {request.request_id}")
    logger.info(f"  类型: {request.discharge_type.value}")
    logger.info(f"  目标水位: {request.target_level}m")
    logger.info(f"  最大流量: {request.max_rate:.1f}m³/s")
    logger.info(f"  预计时间: {request.estimated_duration/60:.0f}分钟")

    commands = discharge_mgr.generate_discharge_commands(request.request_id)
    logger.info(f"\n生成动作指令: {len(commands)}个")
    for cmd in commands:
        logger.info(f"  - {cmd.action_type.value} @ 池{cmd.pool_id}")

    # 统计可能场景数
    logger.info(f"\n{'=' * 70}")
    logger.info("L1层场景统计")
    logger.info('=' * 70)

    counts = generator.count_possible_scenarios()
    for key, value in counts.items():
        logger.info(f"  {key}: {value:,}")

    logger.info("\n" + "=" * 70)
    logger.info("测试完成!")
    logger.info("=" * 70)
