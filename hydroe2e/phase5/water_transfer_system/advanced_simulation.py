"""
高级仿真与数据融合模块
Advanced Simulation and Data Fusion Module

核心功能:
1. 传感器仿真 (噪声、延迟、故障模拟)
2. 执行器仿真 (响应延迟、饱和、故障)
3. 数据治理 (异常检测、数据质量评估)
4. 数据同化 (融合观测与模型预测)
5. IDZ模型参数动态更新
6. 渠池状态实时评价
7. 渠池状态实时预测

架构:
┌─────────────────────────────────────────────────────────────┐
│                  AdvancedSimulationLayer                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │SensorModel  │  │ActuatorModel│  │DataGovernance       │ │
│  │ 噪声/延迟   │  │ 延迟/饱和   │  │ 质量评估/异常检测   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              DataAssimilator (数据同化)                 ││
│  │   卡尔曼滤波器 / 粒子滤波器 / 集合调整                  ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │              IDZParameterEstimator (参数估计)           ││
│  │   递归最小二乘 / 在线辨识 / 自适应更新                  ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌──────────────────────┐  ┌──────────────────────────────┐│
│  │StateEvaluator        │  │StatePredictor               ││
│  │ 实时偏差评价         │  │ 多步预测                    ││
│  └──────────────────────┘  └──────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import logging
import time

from .hydraulic_simulator import (
    FullLineHydraulicSimulator, PoolState,
    PoolPhysicalParams, IDZDynamicModel,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 传感器仿真
# ==============================================================================

class SensorType(Enum):
    """传感器类型"""
    LEVEL = "水位计"
    FLOW = "流量计"
    GATE = "开度计"
    QUALITY = "水质仪"
    TEMPERATURE = "温度计"


class SensorStatus(Enum):
    """传感器状态"""
    NORMAL = "正常"
    NOISY = "噪声大"
    DRIFTING = "漂移"
    STUCK = "卡住"
    FAILED = "故障"


@dataclass
class SensorConfig:
    """传感器配置"""
    sensor_id: str
    sensor_type: SensorType
    pool_id: int

    # 噪声参数
    noise_std: float = 0.01         # 噪声标准差
    bias: float = 0.0               # 固定偏差

    # 延迟参数
    delay_steps: int = 0            # 采样延迟 (步数)

    # 量程
    min_value: float = 0.0
    max_value: float = 10.0

    # 采样
    sample_rate: float = 1.0        # 采样率 [Hz]
    resolution: float = 0.001       # 分辨率

    # 故障概率
    failure_prob: float = 0.0       # 故障概率


@dataclass
class SensorReading:
    """传感器读数"""
    sensor_id: str
    timestamp: float
    raw_value: float                # 原始值
    processed_value: float          # 处理后的值
    quality: float = 1.0            # 数据质量 [0-1]
    status: SensorStatus = SensorStatus.NORMAL


class SensorModel:
    """
    传感器模型

    仿真传感器的各种特性
    """

    def __init__(self, config: SensorConfig):
        self.config = config
        self.status = SensorStatus.NORMAL

        # 延迟缓冲区
        self.delay_buffer: deque = deque(maxlen=max(1, config.delay_steps + 1))

        # 历史数据
        self.reading_history: deque = deque(maxlen=1000)

        # 漂移状态
        self.drift_value: float = 0.0
        self.drift_rate: float = 0.0

        # 卡住值
        self.stuck_value: Optional[float] = None

    def read(self, true_value: float, timestamp: float) -> SensorReading:
        """读取传感器值"""
        # 检查故障
        if np.random.random() < self.config.failure_prob:
            self.status = SensorStatus.FAILED

        if self.status == SensorStatus.FAILED:
            return SensorReading(
                sensor_id=self.config.sensor_id,
                timestamp=timestamp,
                raw_value=np.nan,
                processed_value=np.nan,
                quality=0.0,
                status=SensorStatus.FAILED,
            )

        # 卡住状态
        if self.status == SensorStatus.STUCK:
            if self.stuck_value is None:
                self.stuck_value = true_value
            raw = self.stuck_value
        else:
            # 正常读数 + 噪声
            noise = np.random.normal(0, self.config.noise_std)
            raw = true_value + noise + self.config.bias

        # 漂移
        if self.status == SensorStatus.DRIFTING:
            self.drift_value += self.drift_rate
            raw += self.drift_value

        # 延迟
        self.delay_buffer.append(raw)
        if len(self.delay_buffer) > self.config.delay_steps:
            delayed_value = self.delay_buffer[0]
        else:
            delayed_value = raw

        # 量程限制
        processed = np.clip(delayed_value, self.config.min_value, self.config.max_value)

        # 分辨率量化
        processed = round(processed / self.config.resolution) * self.config.resolution

        # 质量评估
        quality = self._assess_quality(processed, true_value)

        reading = SensorReading(
            sensor_id=self.config.sensor_id,
            timestamp=timestamp,
            raw_value=raw,
            processed_value=processed,
            quality=quality,
            status=self.status,
        )

        self.reading_history.append(reading)
        return reading

    def _assess_quality(self, measured: float, true_value: float) -> float:
        """评估数据质量"""
        if self.status == SensorStatus.FAILED:
            return 0.0

        error = abs(measured - true_value)
        # 基于误差计算质量 (误差越大质量越低)
        quality = max(0, 1.0 - error / (self.config.max_value - self.config.min_value))

        if self.status in [SensorStatus.NOISY, SensorStatus.DRIFTING, SensorStatus.STUCK]:
            quality *= 0.7

        return quality

    def inject_fault(self, fault_type: SensorStatus, **kwargs):
        """注入故障"""
        self.status = fault_type

        if fault_type == SensorStatus.DRIFTING:
            self.drift_rate = kwargs.get('drift_rate', 0.001)
        elif fault_type == SensorStatus.STUCK:
            self.stuck_value = kwargs.get('stuck_value', None)

    def reset(self):
        """重置传感器"""
        self.status = SensorStatus.NORMAL
        self.drift_value = 0.0
        self.stuck_value = None


# ==============================================================================
# 执行器仿真
# ==============================================================================

class ActuatorType(Enum):
    """执行器类型"""
    GATE = "闸门"
    PUMP = "水泵"
    VALVE = "阀门"


class ActuatorStatus(Enum):
    """执行器状态"""
    NORMAL = "正常"
    SLOW = "响应慢"
    STUCK = "卡住"
    FAILED = "故障"


@dataclass
class ActuatorConfig:
    """执行器配置"""
    actuator_id: str
    actuator_type: ActuatorType
    pool_id: int

    # 动态特性
    max_rate: float = 0.01          # 最大变化率 [单位/s]
    response_time: float = 60.0     # 响应时间常数 [s]

    # 量程
    min_position: float = 0.0
    max_position: float = 3.0

    # 死区
    deadband: float = 0.01

    # 故障概率
    failure_prob: float = 0.0


@dataclass
class ActuatorCommand:
    """执行器指令"""
    actuator_id: str
    target_position: float
    timestamp: float


@dataclass
class ActuatorState:
    """执行器状态"""
    actuator_id: str
    timestamp: float
    current_position: float
    target_position: float
    is_moving: bool
    status: ActuatorStatus


class ActuatorModel:
    """
    执行器模型

    仿真执行器的响应特性
    """

    def __init__(self, config: ActuatorConfig):
        self.config = config
        self.status = ActuatorStatus.NORMAL

        self.current_position = (config.min_position + config.max_position) / 2
        self.target_position = self.current_position
        self.is_moving = False

        # 历史
        self.position_history: deque = deque(maxlen=1000)
        self.command_history: List[ActuatorCommand] = []

    def set_target(self, target: float, timestamp: float):
        """设置目标位置"""
        cmd = ActuatorCommand(
            actuator_id=self.config.actuator_id,
            target_position=target,
            timestamp=timestamp,
        )
        self.command_history.append(cmd)

        # 故障检查
        if self.status == ActuatorStatus.FAILED:
            return

        if self.status == ActuatorStatus.STUCK:
            return  # 不响应指令

        # 限制量程
        self.target_position = np.clip(
            target,
            self.config.min_position,
            self.config.max_position
        )

        # 死区检测
        if abs(self.target_position - self.current_position) > self.config.deadband:
            self.is_moving = True

    def step(self, dt: float, timestamp: float) -> ActuatorState:
        """执行一步"""
        if self.status == ActuatorStatus.FAILED:
            return self._get_state(timestamp)

        if not self.is_moving:
            return self._get_state(timestamp)

        # 计算变化率
        rate = self.config.max_rate
        if self.status == ActuatorStatus.SLOW:
            rate *= 0.3  # 响应变慢

        # 计算移动量
        delta = self.target_position - self.current_position
        max_delta = rate * dt

        if abs(delta) <= max_delta:
            self.current_position = self.target_position
            self.is_moving = False
        else:
            direction = 1 if delta > 0 else -1
            self.current_position += direction * max_delta

        # 限制量程
        self.current_position = np.clip(
            self.current_position,
            self.config.min_position,
            self.config.max_position
        )

        state = self._get_state(timestamp)
        self.position_history.append(state)
        return state

    def _get_state(self, timestamp: float) -> ActuatorState:
        """获取当前状态"""
        return ActuatorState(
            actuator_id=self.config.actuator_id,
            timestamp=timestamp,
            current_position=self.current_position,
            target_position=self.target_position,
            is_moving=self.is_moving,
            status=self.status,
        )

    def inject_fault(self, fault_type: ActuatorStatus):
        """注入故障"""
        self.status = fault_type

    def reset(self):
        """重置执行器"""
        self.status = ActuatorStatus.NORMAL


# ==============================================================================
# 数据治理
# ==============================================================================

@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float = 1.0       # 完整性
    accuracy: float = 1.0           # 准确性
    timeliness: float = 1.0         # 及时性
    consistency: float = 1.0        # 一致性
    validity: float = 1.0           # 有效性

    @property
    def overall_quality(self) -> float:
        """总体质量"""
        return (self.completeness + self.accuracy + self.timeliness +
                self.consistency + self.validity) / 5


class DataGovernance:
    """
    数据治理

    数据质量评估与异常检测
    """

    def __init__(self):
        self.data_history: Dict[str, deque] = {}  # sensor_id -> readings
        self.quality_history: Dict[str, deque] = {}

        # 异常检测阈值
        self.spike_threshold: float = 3.0  # 标准差倍数
        self.missing_threshold: float = 5  # 允许的缺失步数

    def process_reading(self, reading: SensorReading) -> Tuple[SensorReading, DataQualityMetrics]:
        """处理传感器读数"""
        sensor_id = reading.sensor_id

        # 初始化历史
        if sensor_id not in self.data_history:
            self.data_history[sensor_id] = deque(maxlen=100)
            self.quality_history[sensor_id] = deque(maxlen=100)

        history = self.data_history[sensor_id]

        # 异常检测
        is_anomaly = False
        if len(history) >= 5:
            values = [r.processed_value for r in history]
            mean = np.mean(values)
            std = np.std(values) or 0.01

            # 尖峰检测
            if abs(reading.processed_value - mean) > self.spike_threshold * std:
                is_anomaly = True

        # 数据质量评估
        metrics = self._assess_quality(reading, history, is_anomaly)

        # 保存历史
        history.append(reading)
        self.quality_history[sensor_id].append(metrics)

        # 如果是异常，标记读数
        if is_anomaly:
            reading = SensorReading(
                sensor_id=reading.sensor_id,
                timestamp=reading.timestamp,
                raw_value=reading.raw_value,
                processed_value=reading.processed_value,
                quality=reading.quality * 0.5,  # 降低质量
                status=SensorStatus.NOISY,
            )

        return reading, metrics

    def _assess_quality(self,
                        reading: SensorReading,
                        history: deque,
                        is_anomaly: bool) -> DataQualityMetrics:
        """评估数据质量"""
        metrics = DataQualityMetrics()

        # 完整性
        if np.isnan(reading.processed_value):
            metrics.completeness = 0.0
        else:
            metrics.completeness = 1.0

        # 准确性 (基于传感器质量)
        metrics.accuracy = reading.quality

        # 及时性 (假设都是实时的)
        metrics.timeliness = 1.0

        # 一致性 (与历史比较)
        if len(history) >= 3 and not is_anomaly:
            metrics.consistency = 1.0
        elif is_anomaly:
            metrics.consistency = 0.5
        else:
            metrics.consistency = 0.8

        # 有效性
        metrics.validity = 1.0 if reading.status == SensorStatus.NORMAL else 0.7

        return metrics

    def get_quality_report(self, sensor_id: str) -> Dict[str, Any]:
        """获取质量报告"""
        if sensor_id not in self.quality_history:
            return {}

        history = list(self.quality_history[sensor_id])
        if not history:
            return {}

        return {
            'sensor_id': sensor_id,
            'samples': len(history),
            'avg_completeness': np.mean([m.completeness for m in history]),
            'avg_accuracy': np.mean([m.accuracy for m in history]),
            'avg_consistency': np.mean([m.consistency for m in history]),
            'avg_overall': np.mean([m.overall_quality for m in history]),
        }


# ==============================================================================
# 数据同化
# ==============================================================================

@dataclass
class AssimilationState:
    """同化状态"""
    timestamp: float
    pool_id: int
    estimated_level: float          # 同化后的水位估计
    uncertainty: float              # 不确定性
    innovation: float               # 新息 (观测-预测)
    kalman_gain: float              # 卡尔曼增益


class DataAssimilator:
    """
    数据同化器

    使用简化卡尔曼滤波融合观测与模型预测
    """

    def __init__(self, num_pools: int):
        self.num_pools = num_pools

        # 状态估计 (每个池的水位)
        self.state_estimate: np.ndarray = np.ones(num_pools) * 3.0
        self.state_covariance: np.ndarray = np.eye(num_pools) * 0.1

        # 过程噪声和观测噪声
        self.process_noise = 0.01   # 过程噪声方差
        self.observation_noise = 0.05  # 观测噪声方差

        # 历史
        self.assimilation_history: Dict[int, List[AssimilationState]] = {
            i: [] for i in range(num_pools)
        }

    def predict(self, model_prediction: np.ndarray, dt: float):
        """预测步"""
        # 简化：状态转移为恒等
        # 增加过程噪声
        self.state_covariance += np.eye(self.num_pools) * self.process_noise * dt

    def update(self, observations: Dict[int, float], timestamp: float) -> Dict[int, AssimilationState]:
        """更新步"""
        results = {}

        for pool_id, observation in observations.items():
            if pool_id >= self.num_pools:
                continue

            # 当前估计
            x = self.state_estimate[pool_id]
            P = self.state_covariance[pool_id, pool_id]

            # 新息
            innovation = observation - x

            # 卡尔曼增益
            R = self.observation_noise
            K = P / (P + R)

            # 更新估计
            x_new = x + K * innovation
            P_new = (1 - K) * P

            self.state_estimate[pool_id] = x_new
            self.state_covariance[pool_id, pool_id] = P_new

            # 记录
            state = AssimilationState(
                timestamp=timestamp,
                pool_id=pool_id,
                estimated_level=x_new,
                uncertainty=np.sqrt(P_new),
                innovation=innovation,
                kalman_gain=K,
            )
            results[pool_id] = state
            self.assimilation_history[pool_id].append(state)

        return results

    def get_estimate(self, pool_id: int) -> Tuple[float, float]:
        """获取估计值和不确定性"""
        if pool_id >= self.num_pools:
            return 0.0, 1.0

        return (
            self.state_estimate[pool_id],
            np.sqrt(self.state_covariance[pool_id, pool_id])
        )

    def get_all_estimates(self) -> Dict[int, Tuple[float, float]]:
        """获取所有池的估计"""
        return {
            i: self.get_estimate(i)
            for i in range(self.num_pools)
        }


# ==============================================================================
# IDZ参数估计
# ==============================================================================

@dataclass
class IDZParameters:
    """IDZ模型参数"""
    pool_id: int
    delay_time: float = 300.0       # 延迟时间 [s]
    integrator_gain: float = 1e-4   # 积分增益
    zero_gain: float = 0.0          # 零点增益


class IDZParameterEstimator:
    """
    IDZ参数估计器

    基于在线数据动态更新IDZ模型参数
    """

    def __init__(self, num_pools: int):
        self.num_pools = num_pools

        # 每个池的参数估计
        self.parameters: Dict[int, IDZParameters] = {
            i: IDZParameters(pool_id=i)
            for i in range(num_pools)
        }

        # 递归最小二乘参数
        self.forgetting_factor = 0.99  # 遗忘因子
        self.covariance: Dict[int, np.ndarray] = {
            i: np.eye(3) * 100  # 初始协方差
            for i in range(num_pools)
        }

        # 数据缓冲
        self.data_buffer: Dict[int, deque] = {
            i: deque(maxlen=200) for i in range(num_pools)
        }

    def add_observation(self,
                        pool_id: int,
                        level: float,
                        inflow: float,
                        outflow: float,
                        timestamp: float):
        """添加观测数据"""
        if pool_id >= self.num_pools:
            return

        self.data_buffer[pool_id].append({
            'timestamp': timestamp,
            'level': level,
            'inflow': inflow,
            'outflow': outflow,
        })

    def update_parameters(self, pool_id: int) -> Optional[IDZParameters]:
        """更新参数估计"""
        if pool_id >= self.num_pools:
            return None

        buffer = self.data_buffer[pool_id]
        if len(buffer) < 10:
            return self.parameters[pool_id]

        # 提取数据
        data = list(buffer)
        levels = np.array([d['level'] for d in data])
        inflows = np.array([d['inflow'] for d in data])
        outflows = np.array([d['outflow'] for d in data])

        # 简化的参数估计：基于水量平衡
        # dZ/dt = K * (inflow - outflow)
        dt = 60.0  # 假设60秒步长
        level_changes = np.diff(levels)
        flow_diff = (inflows[:-1] - outflows[:-1])

        # 最小二乘估计积分增益
        if np.std(flow_diff) > 0.1:
            gain = np.mean(level_changes / (flow_diff * dt + 1e-6))
            gain = np.clip(gain, 1e-6, 1e-2)
            self.parameters[pool_id].integrator_gain = gain

        # 估计延迟时间 (简化：使用互相关)
        if len(levels) > 20:
            # 延迟估计暂时保持默认值
            pass

        return self.parameters[pool_id]

    def get_parameters(self, pool_id: int) -> IDZParameters:
        """获取参数"""
        return self.parameters.get(pool_id, IDZParameters(pool_id=pool_id))


# ==============================================================================
# 状态评价
# ==============================================================================

@dataclass
class PoolEvaluation:
    """渠池状态评价"""
    pool_id: int
    timestamp: float

    # 偏差
    level_deviation: float = 0.0    # 水位偏差 [m]
    flow_deviation: float = 0.0     # 流量偏差 [m³/s]

    # 评价等级
    level_rating: str = "正常"       # 良好/正常/偏离/告警/危险
    overall_score: float = 1.0      # 综合评分 [0-1]

    # 趋势
    trend: str = "稳定"              # 上升/稳定/下降


class StateEvaluator:
    """
    状态评价器

    实时评价各渠池状态与控制目标的偏差
    """

    # 评价阈值
    THRESHOLDS = {
        'excellent': 0.1,   # 优秀: 偏差 < 0.1m
        'good': 0.2,        # 良好: 偏差 < 0.2m
        'normal': 0.3,      # 正常: 偏差 < 0.3m
        'warning': 0.5,     # 告警: 偏差 < 0.5m
        # 危险: 偏差 >= 0.5m
    }

    def __init__(self, num_pools: int):
        self.num_pools = num_pools

        # 目标值
        self.target_levels: Dict[int, float] = {
            i: 3.0 for i in range(num_pools)
        }
        self.target_flows: Dict[int, float] = {
            i: 50.0 for i in range(num_pools)
        }

        # 历史评价
        self.evaluation_history: Dict[int, deque] = {
            i: deque(maxlen=100) for i in range(num_pools)
        }

    def set_targets(self, pool_id: int, level: float, flow: float):
        """设置目标值"""
        self.target_levels[pool_id] = level
        self.target_flows[pool_id] = flow

    def evaluate(self,
                 pool_id: int,
                 current_level: float,
                 current_flow: float,
                 timestamp: float) -> PoolEvaluation:
        """评价渠池状态"""
        if pool_id >= self.num_pools:
            return PoolEvaluation(pool_id=pool_id, timestamp=timestamp)

        target_level = self.target_levels.get(pool_id, 3.0)
        target_flow = self.target_flows.get(pool_id, 50.0)

        # 计算偏差
        level_dev = current_level - target_level
        flow_dev = current_flow - target_flow

        # 评价等级
        abs_level_dev = abs(level_dev)
        if abs_level_dev < self.THRESHOLDS['excellent']:
            rating = "优秀"
            score = 1.0
        elif abs_level_dev < self.THRESHOLDS['good']:
            rating = "良好"
            score = 0.9
        elif abs_level_dev < self.THRESHOLDS['normal']:
            rating = "正常"
            score = 0.8
        elif abs_level_dev < self.THRESHOLDS['warning']:
            rating = "告警"
            score = 0.6
        else:
            rating = "危险"
            score = 0.3

        # 趋势判断
        history = self.evaluation_history[pool_id]
        if len(history) >= 3:
            recent_devs = [e.level_deviation for e in list(history)[-3:]]
            if level_dev > recent_devs[-1] + 0.01:
                trend = "上升"
            elif level_dev < recent_devs[-1] - 0.01:
                trend = "下降"
            else:
                trend = "稳定"
        else:
            trend = "稳定"

        evaluation = PoolEvaluation(
            pool_id=pool_id,
            timestamp=timestamp,
            level_deviation=level_dev,
            flow_deviation=flow_dev,
            level_rating=rating,
            overall_score=score,
            trend=trend,
        )

        self.evaluation_history[pool_id].append(evaluation)
        return evaluation

    def get_summary(self) -> Dict[str, Any]:
        """获取评价摘要"""
        all_scores = []
        rating_counts = {}

        for pool_id in range(self.num_pools):
            history = self.evaluation_history[pool_id]
            if history:
                latest = history[-1]
                all_scores.append(latest.overall_score)
                rating = latest.level_rating
                rating_counts[rating] = rating_counts.get(rating, 0) + 1

        return {
            'total_pools': self.num_pools,
            'avg_score': np.mean(all_scores) if all_scores else 0,
            'min_score': min(all_scores) if all_scores else 0,
            'rating_distribution': rating_counts,
        }


# ==============================================================================
# 状态预测
# ==============================================================================

@dataclass
class PredictionResult:
    """预测结果"""
    pool_id: int
    prediction_time: float          # 预测时刻
    horizon: float                  # 预测时域 [s]

    predicted_levels: List[float] = field(default_factory=list)  # 预测水位序列
    predicted_times: List[float] = field(default_factory=list)   # 预测时间点

    confidence_lower: List[float] = field(default_factory=list)  # 置信下界
    confidence_upper: List[float] = field(default_factory=list)  # 置信上界


class StatePredictor:
    """
    状态预测器

    多步预测渠池状态
    """

    def __init__(self, num_pools: int):
        self.num_pools = num_pools

        # 每个池的简化IDZ模型
        self.pool_params: Dict[int, PoolPhysicalParams] = {
            i: PoolPhysicalParams(pool_id=i)
            for i in range(num_pools)
        }
        self.idz_models: Dict[int, IDZDynamicModel] = {
            i: IDZDynamicModel(self.pool_params[i])
            for i in range(num_pools)
        }

        # 预测不确定性
        self.prediction_uncertainty = 0.01  # 每步增加的不确定性

    def predict(self,
                pool_id: int,
                current_level: float,
                current_inflow: float,
                current_outflow: float,
                horizon: float,
                dt: float = 60.0) -> PredictionResult:
        """预测未来状态"""
        if pool_id >= self.num_pools:
            return PredictionResult(pool_id=pool_id, prediction_time=0, horizon=horizon)

        result = PredictionResult(
            pool_id=pool_id,
            prediction_time=0,
            horizon=horizon,
        )

        # 初始化模型状态
        model = self.idz_models[pool_id]
        model.current_level = current_level
        model.current_inflow = current_inflow
        model.current_outflow = current_outflow

        # 多步预测
        num_steps = int(horizon / dt)
        uncertainty = 0.0

        for step in range(num_steps):
            # 假设流量保持不变
            level = model.step(dt, current_inflow, current_outflow)

            result.predicted_levels.append(level)
            result.predicted_times.append(step * dt)

            # 累积不确定性
            uncertainty += self.prediction_uncertainty * dt
            result.confidence_lower.append(level - 2 * uncertainty)
            result.confidence_upper.append(level + 2 * uncertainty)

        return result

    def predict_all(self,
                    current_states: Dict[int, Tuple[float, float, float]],
                    horizon: float,
                    dt: float = 60.0) -> Dict[int, PredictionResult]:
        """预测所有池的状态"""
        results = {}

        for pool_id, (level, inflow, outflow) in current_states.items():
            if pool_id < self.num_pools:
                results[pool_id] = self.predict(
                    pool_id, level, inflow, outflow, horizon, dt
                )

        return results


# ==============================================================================
# 高级仿真层
# ==============================================================================

class AdvancedSimulationLayer:
    """
    高级仿真层

    整合传感器、执行器、数据治理、同化、参数估计、评价与预测
    """

    def __init__(self, simulator: FullLineHydraulicSimulator):
        self.simulator = simulator
        self.num_pools = simulator.num_pools

        # 传感器模型 (每个池一个水位计)
        self.sensors: Dict[str, SensorModel] = {}
        for i in range(self.num_pools):
            config = SensorConfig(
                sensor_id=f"LEVEL_{i}",
                sensor_type=SensorType.LEVEL,
                pool_id=i,
                noise_std=0.02,
            )
            self.sensors[config.sensor_id] = SensorModel(config)

        # 执行器模型 (每个池一个闸门)
        self.actuators: Dict[str, ActuatorModel] = {}
        for i in range(self.num_pools):
            config = ActuatorConfig(
                actuator_id=f"GATE_{i}",
                actuator_type=ActuatorType.GATE,
                pool_id=i,
            )
            self.actuators[config.actuator_id] = ActuatorModel(config)

        # 数据治理
        self.governance = DataGovernance()

        # 数据同化
        self.assimilator = DataAssimilator(self.num_pools)

        # 参数估计
        self.parameter_estimator = IDZParameterEstimator(self.num_pools)

        # 状态评价
        self.evaluator = StateEvaluator(self.num_pools)

        # 状态预测
        self.predictor = StatePredictor(self.num_pools)

    def process_step(self, dt: float = 60.0) -> Dict[str, Any]:
        """处理一步"""
        timestamp = self.simulator.state.current_time

        results = {
            'timestamp': timestamp,
            'sensor_readings': {},
            'actuator_states': {},
            'quality_metrics': {},
            'assimilation': {},
            'evaluations': {},
        }

        # 1. 传感器读数
        for pool_id, pool_state in self.simulator.state.pool_states.items():
            sensor_id = f"LEVEL_{pool_id}"
            if sensor_id in self.sensors:
                reading = self.sensors[sensor_id].read(
                    pool_state.water_level, timestamp
                )

                # 数据治理
                processed_reading, quality = self.governance.process_reading(reading)
                results['sensor_readings'][pool_id] = processed_reading
                results['quality_metrics'][pool_id] = quality

        # 2. 执行器状态
        for actuator_id, actuator in self.actuators.items():
            state = actuator.step(dt, timestamp)
            pool_id = actuator.config.pool_id
            results['actuator_states'][pool_id] = state

        # 3. 数据同化
        observations = {
            pool_id: reading.processed_value
            for pool_id, reading in results['sensor_readings'].items()
            if not np.isnan(reading.processed_value)
        }

        if observations:
            model_predictions = np.array([
                self.simulator.state.pool_states[i].water_level
                for i in range(self.num_pools)
            ])
            self.assimilator.predict(model_predictions, dt)
            assimilation = self.assimilator.update(observations, timestamp)
            results['assimilation'] = assimilation

        # 4. 参数估计更新
        for pool_id, pool_state in self.simulator.state.pool_states.items():
            self.parameter_estimator.add_observation(
                pool_id,
                pool_state.water_level,
                pool_state.inflow,
                pool_state.outflow,
                timestamp,
            )
            if self.simulator.state.step_count % 10 == 0:  # 每10步更新一次
                self.parameter_estimator.update_parameters(pool_id)

        # 5. 状态评价
        for pool_id, pool_state in self.simulator.state.pool_states.items():
            evaluation = self.evaluator.evaluate(
                pool_id,
                pool_state.water_level,
                pool_state.outflow,
                timestamp,
            )
            results['evaluations'][pool_id] = evaluation

        return results

    def get_predictions(self, horizon: float = 600.0) -> Dict[int, PredictionResult]:
        """获取所有池的预测"""
        current_states = {}
        for pool_id, pool_state in self.simulator.state.pool_states.items():
            current_states[pool_id] = (
                pool_state.water_level,
                pool_state.inflow,
                pool_state.outflow,
            )

        return self.predictor.predict_all(current_states, horizon)

    def inject_sensor_fault(self, pool_id: int, fault_type: SensorStatus, **kwargs):
        """注入传感器故障"""
        sensor_id = f"LEVEL_{pool_id}"
        if sensor_id in self.sensors:
            self.sensors[sensor_id].inject_fault(fault_type, **kwargs)

    def inject_actuator_fault(self, pool_id: int, fault_type: ActuatorStatus):
        """注入执行器故障"""
        actuator_id = f"GATE_{pool_id}"
        if actuator_id in self.actuators:
            self.actuators[actuator_id].inject_fault(fault_type)

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评价摘要"""
        return self.evaluator.get_summary()


# ==============================================================================
# 导出
# ==============================================================================

__all__ = [
    'SensorType',
    'SensorStatus',
    'SensorConfig',
    'SensorReading',
    'SensorModel',
    'ActuatorType',
    'ActuatorStatus',
    'ActuatorConfig',
    'ActuatorCommand',
    'ActuatorState',
    'ActuatorModel',
    'DataQualityMetrics',
    'DataGovernance',
    'AssimilationState',
    'DataAssimilator',
    'IDZParameters',
    'IDZParameterEstimator',
    'PoolEvaluation',
    'StateEvaluator',
    'PredictionResult',
    'StatePredictor',
    'AdvancedSimulationLayer',
]
