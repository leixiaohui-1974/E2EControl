"""
历史数据记录与仿真回放系统
Historical Data Recording and Simulation Replay System

功能:
1. 时序数据存储 - 高效存储仿真过程数据
2. 数据索引 - 支持时间范围、事件类型查询
3. 仿真回放 - 支持变速、暂停、跳转
4. 数据分析 - 统计分析、趋势检测
5. 导入导出 - JSON/CSV格式支持

作者: AI Assistant
日期: 2024
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Iterator, Callable, Union
from enum import Enum, auto
from datetime import datetime
import json
import math
import gzip
import hashlib
from collections import defaultdict
from pathlib import Path


# ============ 数据类型定义 ============

class DataChannel(Enum):
    """数据通道类型"""
    POOL_LEVEL = auto()         # 渠池水位
    POOL_FLOW = auto()          # 渠池流量
    GATE_POSITION = auto()      # 闸门开度
    GATE_COMMAND = auto()       # 闸门指令
    SENSOR_READING = auto()     # 传感器读数
    CONTROL_OUTPUT = auto()     # 控制输出
    FAULT_EVENT = auto()        # 故障事件
    SCENARIO_EVENT = auto()     # 场景事件
    SYSTEM_STATE = auto()       # 系统状态
    PERFORMANCE_METRIC = auto() # 性能指标


class RecordingMode(Enum):
    """记录模式"""
    FULL = auto()           # 完整记录所有数据
    SAMPLED = auto()        # 采样记录
    EVENT_ONLY = auto()     # 仅记录事件
    COMPRESSED = auto()     # 压缩记录（差分编码）


@dataclass
class DataPoint:
    """数据点"""
    timestamp: float            # 时间戳
    channel: DataChannel        # 数据通道
    source_id: str              # 数据源标识
    value: Union[float, Dict, str]  # 数据值
    quality: float = 1.0        # 数据质量 0-1
    metadata: Dict = field(default_factory=dict)


@dataclass
class TimeSeriesSegment:
    """时间序列片段"""
    channel: DataChannel
    source_id: str
    start_time: float
    end_time: float
    sample_rate: float          # 采样率 Hz
    values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def add_point(self, timestamp: float, value: float):
        """添加数据点"""
        self.timestamps.append(timestamp)
        self.values.append(value)
        self.end_time = timestamp

    def get_value_at(self, timestamp: float) -> Optional[float]:
        """获取指定时间的值（线性插值）"""
        if not self.timestamps:
            return None

        if timestamp <= self.timestamps[0]:
            return self.values[0]
        if timestamp >= self.timestamps[-1]:
            return self.values[-1]

        # 二分查找
        left, right = 0, len(self.timestamps) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if self.timestamps[mid] <= timestamp:
                left = mid
            else:
                right = mid

        # 线性插值
        t0, t1 = self.timestamps[left], self.timestamps[right]
        v0, v1 = self.values[left], self.values[right]
        alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0
        return v0 + alpha * (v1 - v0)

    def get_slice(self, start: float, end: float) -> 'TimeSeriesSegment':
        """获取时间切片"""
        indices = [i for i, t in enumerate(self.timestamps)
                   if start <= t <= end]

        segment = TimeSeriesSegment(
            channel=self.channel,
            source_id=self.source_id,
            start_time=start,
            end_time=end,
            sample_rate=self.sample_rate,
        )

        for i in indices:
            segment.timestamps.append(self.timestamps[i])
            segment.values.append(self.values[i])

        return segment


@dataclass
class EventRecord:
    """事件记录"""
    event_id: str
    timestamp: float
    event_type: str
    source: str
    severity: str
    description: str
    data: Dict = field(default_factory=dict)
    duration: float = 0.0       # 事件持续时间
    end_time: Optional[float] = None


@dataclass
class SimulationSnapshot:
    """仿真快照"""
    snapshot_id: str
    timestamp: float
    pool_states: Dict[int, Dict]    # 渠池状态
    gate_states: Dict[int, Dict]    # 闸门状态
    control_states: Dict[str, Any]  # 控制状态
    active_events: List[str]        # 活动事件
    system_metrics: Dict[str, float]  # 系统指标


@dataclass
class RecordingSession:
    """记录会话"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    mode: RecordingMode = RecordingMode.FULL
    description: str = ""
    metadata: Dict = field(default_factory=dict)

    # 数据统计
    total_points: int = 0
    total_events: int = 0
    total_snapshots: int = 0
    channels_recorded: List[str] = field(default_factory=list)


# ============ 数据存储引擎 ============

class TimeSeriesStorage:
    """时序数据存储引擎"""

    def __init__(self, max_memory_points: int = 100000):
        self.max_memory_points = max_memory_points

        # 内存存储
        self.segments: Dict[str, TimeSeriesSegment] = {}  # key: channel_source
        self.events: List[EventRecord] = []
        self.snapshots: List[SimulationSnapshot] = []

        # 索引
        self.time_index: Dict[float, List[str]] = defaultdict(list)  # 时间 -> 数据键列表
        self.event_index: Dict[str, List[int]] = defaultdict(list)   # 事件类型 -> 索引列表

        # 统计
        self.total_points = 0
        self.point_count_by_channel: Dict[str, int] = defaultdict(int)

    def _get_segment_key(self, channel: DataChannel, source_id: str) -> str:
        """获取片段键"""
        return f"{channel.name}_{source_id}"

    def add_data_point(self, point: DataPoint):
        """添加数据点"""
        key = self._get_segment_key(point.channel, point.source_id)

        if key not in self.segments:
            self.segments[key] = TimeSeriesSegment(
                channel=point.channel,
                source_id=point.source_id,
                start_time=point.timestamp,
                end_time=point.timestamp,
                sample_rate=1.0,
            )

        segment = self.segments[key]

        if isinstance(point.value, (int, float)):
            segment.add_point(point.timestamp, float(point.value))
        else:
            # 复杂类型存储为JSON字符串的哈希值作为参考
            segment.add_point(point.timestamp, hash(str(point.value)) % 1000000)

        self.total_points += 1
        self.point_count_by_channel[key] += 1

        # 更新时间索引（稀疏索引，每10秒一个条目）
        time_bucket = int(point.timestamp / 10) * 10
        if key not in self.time_index[time_bucket]:
            self.time_index[time_bucket].append(key)

    def add_event(self, event: EventRecord):
        """添加事件"""
        self.events.append(event)
        self.event_index[event.event_type].append(len(self.events) - 1)

    def add_snapshot(self, snapshot: SimulationSnapshot):
        """添加快照"""
        self.snapshots.append(snapshot)

    def get_data(self, channel: DataChannel, source_id: str,
                 start_time: float, end_time: float) -> Optional[TimeSeriesSegment]:
        """获取数据"""
        key = self._get_segment_key(channel, source_id)
        if key not in self.segments:
            return None

        return self.segments[key].get_slice(start_time, end_time)

    def get_value_at(self, channel: DataChannel, source_id: str,
                     timestamp: float) -> Optional[float]:
        """获取指定时间的值"""
        key = self._get_segment_key(channel, source_id)
        if key not in self.segments:
            return None

        return self.segments[key].get_value_at(timestamp)

    def get_events(self, event_type: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> List[EventRecord]:
        """获取事件"""
        if event_type:
            indices = self.event_index.get(event_type, [])
            events = [self.events[i] for i in indices]
        else:
            events = self.events

        if start_time is not None:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time is not None:
            events = [e for e in events if e.timestamp <= end_time]

        return events

    def get_snapshot_at(self, timestamp: float) -> Optional[SimulationSnapshot]:
        """获取指定时间最近的快照"""
        if not self.snapshots:
            return None

        # 找到不超过timestamp的最近快照
        best = None
        for snapshot in self.snapshots:
            if snapshot.timestamp <= timestamp:
                if best is None or snapshot.timestamp > best.timestamp:
                    best = snapshot

        return best

    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计"""
        time_range = (0.0, 0.0)
        if self.segments:
            all_starts = [s.start_time for s in self.segments.values()]
            all_ends = [s.end_time for s in self.segments.values()]
            time_range = (min(all_starts), max(all_ends))

        return {
            'total_points': self.total_points,
            'total_events': len(self.events),
            'total_snapshots': len(self.snapshots),
            'num_channels': len(self.segments),
            'time_range': time_range,
            'points_by_channel': dict(self.point_count_by_channel),
            'memory_usage_estimate': self._estimate_memory(),
        }

    def _estimate_memory(self) -> int:
        """估计内存使用（字节）"""
        # 粗略估计
        point_size = 16  # timestamp + value
        event_size = 200  # 平均事件大小
        snapshot_size = 2000  # 平均快照大小

        return (self.total_points * point_size +
                len(self.events) * event_size +
                len(self.snapshots) * snapshot_size)

    def clear(self):
        """清空存储"""
        self.segments.clear()
        self.events.clear()
        self.snapshots.clear()
        self.time_index.clear()
        self.event_index.clear()
        self.total_points = 0
        self.point_count_by_channel.clear()


# ============ 数据记录器 ============

class SimulationRecorderV2:
    """仿真数据记录器 V2"""

    def __init__(self, session_id: Optional[str] = None,
                 mode: RecordingMode = RecordingMode.FULL,
                 sample_interval: float = 1.0):
        self.session_id = session_id or self._generate_session_id()
        self.mode = mode
        self.sample_interval = sample_interval

        # 存储
        self.storage = TimeSeriesStorage()

        # 会话信息
        self.session = RecordingSession(
            session_id=self.session_id,
            start_time=0.0,
            mode=mode,
        )

        # 采样控制
        self.last_sample_time: Dict[str, float] = {}

        # 压缩控制（差分编码）
        self.last_values: Dict[str, float] = {}
        self.change_threshold = 0.001  # 变化阈值

        # 回调
        self.on_event: Optional[Callable] = None

        # 状态
        self.is_recording = False

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        return f"SIM_{timestamp}_{random_suffix}"

    def start_recording(self, start_time: float = 0.0, description: str = ""):
        """开始记录"""
        self.session.start_time = start_time
        self.session.description = description
        self.is_recording = True

    def stop_recording(self, end_time: float):
        """停止记录"""
        self.session.end_time = end_time
        self.session.total_points = self.storage.total_points
        self.session.total_events = len(self.storage.events)
        self.session.total_snapshots = len(self.storage.snapshots)
        self.session.channels_recorded = list(self.storage.segments.keys())
        self.is_recording = False

    def record_pool_state(self, pool_id: int, timestamp: float,
                          level: float, flow: float,
                          quality_index: float = 1.0):
        """记录渠池状态"""
        if not self.is_recording:
            return

        # 水位
        self._record_value(
            DataChannel.POOL_LEVEL, f"pool_{pool_id}",
            timestamp, level
        )

        # 流量
        self._record_value(
            DataChannel.POOL_FLOW, f"pool_{pool_id}",
            timestamp, flow
        )

    def record_gate_state(self, gate_id: int, timestamp: float,
                          position: float, command: float):
        """记录闸门状态"""
        if not self.is_recording:
            return

        self._record_value(
            DataChannel.GATE_POSITION, f"gate_{gate_id}",
            timestamp, position
        )

        self._record_value(
            DataChannel.GATE_COMMAND, f"gate_{gate_id}",
            timestamp, command
        )

    def record_control_output(self, controller_id: str, timestamp: float,
                              output: Dict[str, float]):
        """记录控制输出"""
        if not self.is_recording:
            return

        for key, value in output.items():
            self._record_value(
                DataChannel.CONTROL_OUTPUT, f"{controller_id}_{key}",
                timestamp, value
            )

    def record_sensor_reading(self, sensor_id: str, timestamp: float,
                              value: float, quality: float = 1.0):
        """记录传感器读数"""
        if not self.is_recording:
            return

        self._record_value(
            DataChannel.SENSOR_READING, sensor_id,
            timestamp, value, quality
        )

    def record_event(self, event_type: str, source: str,
                     timestamp: float, severity: str,
                     description: str, data: Dict = None):
        """记录事件"""
        if not self.is_recording:
            return

        event = EventRecord(
            event_id=f"EVT_{len(self.storage.events)+1:06d}",
            timestamp=timestamp,
            event_type=event_type,
            source=source,
            severity=severity,
            description=description,
            data=data or {},
        )

        self.storage.add_event(event)

        if self.on_event:
            self.on_event(event)

    def record_snapshot(self, timestamp: float,
                        pool_states: Dict[int, Dict],
                        gate_states: Dict[int, Dict],
                        control_states: Dict = None,
                        active_events: List[str] = None,
                        system_metrics: Dict = None):
        """记录快照"""
        if not self.is_recording:
            return

        snapshot = SimulationSnapshot(
            snapshot_id=f"SNAP_{len(self.storage.snapshots)+1:04d}",
            timestamp=timestamp,
            pool_states=pool_states,
            gate_states=gate_states,
            control_states=control_states or {},
            active_events=active_events or [],
            system_metrics=system_metrics or {},
        )

        self.storage.add_snapshot(snapshot)

    def record_performance_metric(self, metric_name: str, timestamp: float,
                                   value: float):
        """记录性能指标"""
        if not self.is_recording:
            return

        self._record_value(
            DataChannel.PERFORMANCE_METRIC, metric_name,
            timestamp, value
        )

    def _record_value(self, channel: DataChannel, source_id: str,
                      timestamp: float, value: float, quality: float = 1.0):
        """内部记录值方法"""
        key = f"{channel.name}_{source_id}"

        # 采样模式检查
        if self.mode == RecordingMode.SAMPLED:
            last_time = self.last_sample_time.get(key, -999)
            if timestamp - last_time < self.sample_interval:
                return
            self.last_sample_time[key] = timestamp

        # 压缩模式检查
        if self.mode == RecordingMode.COMPRESSED:
            last_value = self.last_values.get(key, None)
            if last_value is not None:
                if abs(value - last_value) < self.change_threshold:
                    return
            self.last_values[key] = value

        # 记录数据点
        point = DataPoint(
            timestamp=timestamp,
            channel=channel,
            source_id=source_id,
            value=value,
            quality=quality,
        )

        self.storage.add_data_point(point)

    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        return {
            'session_id': self.session.session_id,
            'start_time': self.session.start_time,
            'end_time': self.session.end_time,
            'mode': self.session.mode.name,
            'description': self.session.description,
            'total_points': self.storage.total_points,
            'total_events': len(self.storage.events),
            'total_snapshots': len(self.storage.snapshots),
            'statistics': self.storage.get_statistics(),
        }


# ============ 仿真回放引擎 ============

class PlaybackState(Enum):
    """回放状态"""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    SEEKING = auto()


@dataclass
class PlaybackFrame:
    """回放帧"""
    timestamp: float
    pool_states: Dict[int, Dict]
    gate_states: Dict[int, Dict]
    events: List[EventRecord]
    metrics: Dict[str, float]


class SimulationReplayer:
    """仿真回放引擎"""

    def __init__(self, storage: TimeSeriesStorage):
        self.storage = storage

        # 回放状态
        self.state = PlaybackState.STOPPED
        self.current_time = 0.0
        self.playback_speed = 1.0

        # 时间范围
        stats = storage.get_statistics()
        self.start_time = stats['time_range'][0]
        self.end_time = stats['time_range'][1]

        # 回调
        self.on_frame: Optional[Callable[[PlaybackFrame], None]] = None
        self.on_event: Optional[Callable[[EventRecord], None]] = None
        self.on_state_change: Optional[Callable[[PlaybackState], None]] = None

        # 缓存
        self._frame_cache: Dict[float, PlaybackFrame] = {}
        self._processed_events: set = set()

    def play(self):
        """开始播放"""
        if self.state == PlaybackState.STOPPED:
            self.current_time = self.start_time
            self._processed_events.clear()

        self.state = PlaybackState.PLAYING
        if self.on_state_change:
            self.on_state_change(self.state)

    def pause(self):
        """暂停"""
        self.state = PlaybackState.PAUSED
        if self.on_state_change:
            self.on_state_change(self.state)

    def stop(self):
        """停止"""
        self.state = PlaybackState.STOPPED
        self.current_time = self.start_time
        self._processed_events.clear()
        if self.on_state_change:
            self.on_state_change(self.state)

    def seek(self, timestamp: float):
        """跳转到指定时间"""
        self.state = PlaybackState.SEEKING
        self.current_time = max(self.start_time, min(timestamp, self.end_time))

        # 重新计算已处理事件
        self._processed_events = set()
        for event in self.storage.events:
            if event.timestamp <= self.current_time:
                self._processed_events.add(event.event_id)

        self.state = PlaybackState.PAUSED
        if self.on_state_change:
            self.on_state_change(self.state)

    def set_speed(self, speed: float):
        """设置播放速度"""
        self.playback_speed = max(0.1, min(speed, 10.0))

    def step(self, dt: float) -> Optional[PlaybackFrame]:
        """步进一帧"""
        if self.state != PlaybackState.PLAYING:
            return None

        # 更新时间
        self.current_time += dt * self.playback_speed

        # 检查结束
        if self.current_time >= self.end_time:
            self.state = PlaybackState.STOPPED
            if self.on_state_change:
                self.on_state_change(self.state)
            return None

        # 生成帧
        frame = self._generate_frame(self.current_time)

        # 处理新事件
        for event in self.storage.events:
            if (event.event_id not in self._processed_events and
                    event.timestamp <= self.current_time):
                self._processed_events.add(event.event_id)
                if self.on_event:
                    self.on_event(event)

        if self.on_frame:
            self.on_frame(frame)

        return frame

    def get_frame_at(self, timestamp: float) -> PlaybackFrame:
        """获取指定时间的帧"""
        return self._generate_frame(timestamp)

    def _generate_frame(self, timestamp: float) -> PlaybackFrame:
        """生成回放帧"""
        # 尝试从缓存获取
        cache_key = round(timestamp, 1)  # 0.1秒精度缓存
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        # 获取渠池状态
        pool_states = {}
        for key, segment in self.storage.segments.items():
            if segment.channel == DataChannel.POOL_LEVEL:
                pool_id = int(segment.source_id.split('_')[1])
                if pool_id not in pool_states:
                    pool_states[pool_id] = {}
                pool_states[pool_id]['level'] = segment.get_value_at(timestamp)

            elif segment.channel == DataChannel.POOL_FLOW:
                pool_id = int(segment.source_id.split('_')[1])
                if pool_id not in pool_states:
                    pool_states[pool_id] = {}
                pool_states[pool_id]['flow'] = segment.get_value_at(timestamp)

        # 获取闸门状态
        gate_states = {}
        for key, segment in self.storage.segments.items():
            if segment.channel == DataChannel.GATE_POSITION:
                gate_id = int(segment.source_id.split('_')[1])
                if gate_id not in gate_states:
                    gate_states[gate_id] = {}
                gate_states[gate_id]['position'] = segment.get_value_at(timestamp)

            elif segment.channel == DataChannel.GATE_COMMAND:
                gate_id = int(segment.source_id.split('_')[1])
                if gate_id not in gate_states:
                    gate_states[gate_id] = {}
                gate_states[gate_id]['command'] = segment.get_value_at(timestamp)

        # 获取当前时间附近的事件
        events = self.storage.get_events(
            start_time=timestamp - 1.0,
            end_time=timestamp
        )

        # 获取性能指标
        metrics = {}
        for key, segment in self.storage.segments.items():
            if segment.channel == DataChannel.PERFORMANCE_METRIC:
                metric_value = segment.get_value_at(timestamp)
                if metric_value is not None:
                    metrics[segment.source_id] = metric_value

        frame = PlaybackFrame(
            timestamp=timestamp,
            pool_states=pool_states,
            gate_states=gate_states,
            events=events,
            metrics=metrics,
        )

        # 缓存
        if len(self._frame_cache) < 1000:
            self._frame_cache[cache_key] = frame

        return frame

    def get_progress(self) -> float:
        """获取播放进度 0-1"""
        if self.end_time == self.start_time:
            return 0.0
        return (self.current_time - self.start_time) / (self.end_time - self.start_time)

    def get_playback_info(self) -> Dict[str, Any]:
        """获取回放信息"""
        return {
            'state': self.state.name,
            'current_time': self.current_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'progress': self.get_progress(),
            'playback_speed': self.playback_speed,
            'duration': self.end_time - self.start_time,
        }


# ============ 数据分析工具 ============

class DataAnalyzer:
    """数据分析工具"""

    def __init__(self, storage: TimeSeriesStorage):
        self.storage = storage

    def compute_statistics(self, channel: DataChannel, source_id: str,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> Dict[str, float]:
        """计算统计量"""
        segment = self.storage.get_data(
            channel, source_id,
            start_time or 0,
            end_time or float('inf')
        )

        if not segment or not segment.values:
            return {}

        values = segment.values
        n = len(values)

        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / n
        std_dev = math.sqrt(variance)

        sorted_values = sorted(values)
        median = sorted_values[n // 2]

        return {
            'count': n,
            'min': min(values),
            'max': max(values),
            'mean': mean_val,
            'std': std_dev,
            'median': median,
            'range': max(values) - min(values),
            'p5': sorted_values[int(n * 0.05)],
            'p95': sorted_values[int(n * 0.95)],
        }

    def detect_trends(self, channel: DataChannel, source_id: str,
                      window_size: int = 10) -> List[Dict]:
        """检测趋势"""
        segment = self.storage.get_data(channel, source_id, 0, float('inf'))

        if not segment or len(segment.values) < window_size:
            return []

        trends = []
        values = segment.values
        timestamps = segment.timestamps

        for i in range(len(values) - window_size):
            window = values[i:i + window_size]
            window_ts = timestamps[i:i + window_size]

            # 计算趋势斜率
            x_mean = sum(range(window_size)) / window_size
            y_mean = sum(window) / window_size

            numerator = sum((j - x_mean) * (window[j] - y_mean)
                           for j in range(window_size))
            denominator = sum((j - x_mean) ** 2 for j in range(window_size))

            if denominator == 0:
                continue

            slope = numerator / denominator

            # 显著趋势
            if abs(slope) > 0.01:  # 阈值可调
                trend_type = 'rising' if slope > 0 else 'falling'
                trends.append({
                    'start_time': window_ts[0],
                    'end_time': window_ts[-1],
                    'type': trend_type,
                    'slope': slope,
                    'start_value': window[0],
                    'end_value': window[-1],
                })

        # 合并连续趋势
        merged_trends = []
        for trend in trends:
            if merged_trends and merged_trends[-1]['type'] == trend['type']:
                # 扩展上一个趋势
                merged_trends[-1]['end_time'] = trend['end_time']
                merged_trends[-1]['end_value'] = trend['end_value']
            else:
                merged_trends.append(trend)

        return merged_trends

    def detect_anomalies(self, channel: DataChannel, source_id: str,
                         threshold: float = 3.0) -> List[Dict]:
        """检测异常"""
        stats = self.compute_statistics(channel, source_id)
        if not stats:
            return []

        segment = self.storage.get_data(channel, source_id, 0, float('inf'))
        if not segment:
            return []

        mean_val = stats['mean']
        std_val = stats['std']

        anomalies = []
        for timestamp, value in zip(segment.timestamps, segment.values):
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                if z_score > threshold:
                    anomalies.append({
                        'timestamp': timestamp,
                        'value': value,
                        'z_score': z_score,
                        'expected_range': (
                            mean_val - threshold * std_val,
                            mean_val + threshold * std_val
                        ),
                    })

        return anomalies

    def compute_correlation(self, channel1: DataChannel, source1: str,
                           channel2: DataChannel, source2: str) -> float:
        """计算相关性"""
        seg1 = self.storage.get_data(channel1, source1, 0, float('inf'))
        seg2 = self.storage.get_data(channel2, source2, 0, float('inf'))

        if not seg1 or not seg2:
            return 0.0

        # 对齐时间序列
        aligned_values1 = []
        aligned_values2 = []

        for ts in seg1.timestamps:
            v1 = seg1.get_value_at(ts)
            v2 = seg2.get_value_at(ts)
            if v1 is not None and v2 is not None:
                aligned_values1.append(v1)
                aligned_values2.append(v2)

        if len(aligned_values1) < 2:
            return 0.0

        # 计算相关系数
        n = len(aligned_values1)
        mean1 = sum(aligned_values1) / n
        mean2 = sum(aligned_values2) / n

        cov = sum((aligned_values1[i] - mean1) * (aligned_values2[i] - mean2)
                  for i in range(n)) / n

        std1 = math.sqrt(sum((v - mean1) ** 2 for v in aligned_values1) / n)
        std2 = math.sqrt(sum((v - mean2) ** 2 for v in aligned_values2) / n)

        if std1 == 0 or std2 == 0:
            return 0.0

        return cov / (std1 * std2)

    def generate_summary_report(self) -> Dict[str, Any]:
        """生成汇总报告"""
        stats = self.storage.get_statistics()

        # 各通道统计
        channel_summaries = {}
        for key in self.storage.segments:
            parts = key.split('_', 1)
            if len(parts) == 2:
                channel_name, source_id = parts
                try:
                    channel = DataChannel[channel_name]
                    channel_stats = self.compute_statistics(channel, source_id)
                    channel_summaries[key] = channel_stats
                except KeyError:
                    pass

        # 事件统计
        event_summary = defaultdict(int)
        for event in self.storage.events:
            event_summary[event.event_type] += 1

        return {
            'time_range': stats['time_range'],
            'duration': stats['time_range'][1] - stats['time_range'][0],
            'total_data_points': stats['total_points'],
            'total_events': stats['total_events'],
            'total_snapshots': stats['total_snapshots'],
            'channel_statistics': channel_summaries,
            'event_counts': dict(event_summary),
            'memory_usage': stats['memory_usage_estimate'],
        }


# ============ 数据导入导出 ============

class DataExporter:
    """数据导出器"""

    def __init__(self, storage: TimeSeriesStorage):
        self.storage = storage

    def export_to_json(self, filepath: str, compress: bool = False):
        """导出为JSON"""
        data = {
            'metadata': self.storage.get_statistics(),
            'segments': {},
            'events': [],
            'snapshots': [],
        }

        # 导出时间序列
        for key, segment in self.storage.segments.items():
            data['segments'][key] = {
                'channel': segment.channel.name,
                'source_id': segment.source_id,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'timestamps': segment.timestamps,
                'values': segment.values,
            }

        # 导出事件
        for event in self.storage.events:
            data['events'].append({
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'source': event.source,
                'severity': event.severity,
                'description': event.description,
                'data': event.data,
            })

        # 导出快照
        for snapshot in self.storage.snapshots:
            data['snapshots'].append({
                'snapshot_id': snapshot.snapshot_id,
                'timestamp': snapshot.timestamp,
                'pool_states': snapshot.pool_states,
                'gate_states': snapshot.gate_states,
                'control_states': snapshot.control_states,
                'system_metrics': snapshot.system_metrics,
            })

        # 写入文件
        json_str = json.dumps(data, indent=2)

        if compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)

    def export_to_csv(self, filepath: str, channel: DataChannel,
                      source_id: str):
        """导出单个通道为CSV"""
        segment = self.storage.get_data(channel, source_id, 0, float('inf'))
        if not segment:
            return

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('timestamp,value\n')
            for ts, val in zip(segment.timestamps, segment.values):
                f.write(f'{ts},{val}\n')

    def export_events_csv(self, filepath: str):
        """导出事件为CSV"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('event_id,timestamp,type,source,severity,description\n')
            for event in self.storage.events:
                desc = event.description.replace(',', ';').replace('\n', ' ')
                f.write(f'{event.event_id},{event.timestamp},{event.event_type},'
                        f'{event.source},{event.severity},"{desc}"\n')


class DataImporter:
    """数据导入器"""

    def __init__(self, storage: TimeSeriesStorage):
        self.storage = storage

    def import_from_json(self, filepath: str):
        """从JSON导入"""
        # 检测是否压缩
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # 导入时间序列
        for key, seg_data in data.get('segments', {}).items():
            channel = DataChannel[seg_data['channel']]
            for ts, val in zip(seg_data['timestamps'], seg_data['values']):
                point = DataPoint(
                    timestamp=ts,
                    channel=channel,
                    source_id=seg_data['source_id'],
                    value=val,
                )
                self.storage.add_data_point(point)

        # 导入事件
        for evt_data in data.get('events', []):
            event = EventRecord(
                event_id=evt_data['event_id'],
                timestamp=evt_data['timestamp'],
                event_type=evt_data['event_type'],
                source=evt_data['source'],
                severity=evt_data['severity'],
                description=evt_data['description'],
                data=evt_data.get('data', {}),
            )
            self.storage.add_event(event)

        # 导入快照
        for snap_data in data.get('snapshots', []):
            snapshot = SimulationSnapshot(
                snapshot_id=snap_data['snapshot_id'],
                timestamp=snap_data['timestamp'],
                pool_states=snap_data['pool_states'],
                gate_states=snap_data['gate_states'],
                control_states=snap_data.get('control_states', {}),
                active_events=snap_data.get('active_events', []),
                system_metrics=snap_data.get('system_metrics', {}),
            )
            self.storage.add_snapshot(snapshot)


# ============ 综合数据记录系统 ============

class DataRecordingSystem:
    """综合数据记录系统"""

    def __init__(self, mode: RecordingMode = RecordingMode.FULL,
                 snapshot_interval: float = 60.0):
        self.mode = mode
        self.snapshot_interval = snapshot_interval

        # 核心组件
        self.storage = TimeSeriesStorage()
        self.recorder = SimulationRecorderV2(mode=mode)
        self.recorder.storage = self.storage
        self.analyzer = DataAnalyzer(self.storage)
        self.exporter = DataExporter(self.storage)
        self.importer = DataImporter(self.storage)

        # 状态
        self.last_snapshot_time = 0.0
        self.is_active = False

    def start(self, description: str = ""):
        """启动记录"""
        self.recorder.start_recording(0.0, description)
        self.is_active = True

    def stop(self, end_time: float):
        """停止记录"""
        self.recorder.stop_recording(end_time)
        self.is_active = False

    def record_simulation_step(self, timestamp: float,
                               pool_states: Dict[int, Dict],
                               gate_states: Dict[int, Dict],
                               events: List[Dict] = None,
                               metrics: Dict[str, float] = None):
        """记录一步仿真数据"""
        if not self.is_active:
            return

        # 记录渠池状态
        for pool_id, state in pool_states.items():
            self.recorder.record_pool_state(
                pool_id, timestamp,
                state.get('level', 0),
                state.get('flow', 0),
            )

        # 记录闸门状态
        for gate_id, state in gate_states.items():
            self.recorder.record_gate_state(
                gate_id, timestamp,
                state.get('position', 0.5),
                state.get('command', 0.5),
            )

        # 记录事件
        if events:
            for evt in events:
                self.recorder.record_event(
                    evt.get('type', 'unknown'),
                    evt.get('source', 'unknown'),
                    timestamp,
                    evt.get('severity', 'info'),
                    evt.get('description', ''),
                    evt.get('data', {}),
                )

        # 记录性能指标
        if metrics:
            for name, value in metrics.items():
                self.recorder.record_performance_metric(name, timestamp, value)

        # 自动快照
        if timestamp - self.last_snapshot_time >= self.snapshot_interval:
            self.recorder.record_snapshot(
                timestamp, pool_states, gate_states,
                system_metrics=metrics
            )
            self.last_snapshot_time = timestamp

    def create_replayer(self) -> SimulationReplayer:
        """创建回放器"""
        return SimulationReplayer(self.storage)

    def get_analysis_report(self) -> Dict[str, Any]:
        """获取分析报告"""
        return self.analyzer.generate_summary_report()

    def save(self, filepath: str, compress: bool = True):
        """保存数据"""
        if compress and not filepath.endswith('.gz'):
            filepath += '.gz'
        self.exporter.export_to_json(filepath, compress)

    def load(self, filepath: str):
        """加载数据"""
        self.storage.clear()
        self.importer.import_from_json(filepath)


# ============ 导出 ============

__all__ = [
    # 枚举
    'DataChannel',
    'RecordingMode',
    'PlaybackState',

    # 数据类
    'DataPoint',
    'TimeSeriesSegment',
    'EventRecord',
    'SimulationSnapshot',
    'RecordingSession',
    'PlaybackFrame',

    # 存储
    'TimeSeriesStorage',

    # 记录器
    'SimulationRecorderV2',

    # 回放器
    'SimulationReplayer',

    # 分析工具
    'DataAnalyzer',

    # 导入导出
    'DataExporter',
    'DataImporter',

    # 综合系统
    'DataRecordingSystem',
]
