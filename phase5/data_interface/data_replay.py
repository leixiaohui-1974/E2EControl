# Phase 5.9: Data Replay Engine
# 数据回放引擎 - 支持历史数据回放和场景重现

import asyncio
import csv
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Iterator
import re
import sqlite3

logger = logging.getLogger(__name__)

# SQL identifier validation pattern: only alphanumeric and underscores
_SQL_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _validate_sql_identifier(name: str) -> str:
    """Validate and return a safe SQL identifier.

    Raises ValueError if *name* contains characters that are not
    allowed in a plain SQL identifier (letters, digits, underscores).
    """
    if not _SQL_IDENT_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


class TimeScaleMode(Enum):
    """时间缩放模式"""
    REAL_TIME = "real_time"  # 1:1 真实时间
    FAST = "fast"  # 加速回放
    SLOW = "slow"  # 减速回放
    STEP = "step"  # 单步回放
    IMMEDIATE = "immediate"  # 立即执行(无延迟)


class DataSourceType(Enum):
    """数据源类型"""
    CSV = "csv"
    JSON = "json"
    SQLITE = "sqlite"
    PARQUET = "parquet"
    MEMORY = "memory"


class ReplayState(Enum):
    """回放状态"""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class DataSource:
    """数据源定义"""
    name: str
    source_type: DataSourceType
    path: Optional[str] = None
    data: Optional[List[Dict]] = None
    timestamp_column: str = "timestamp"
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    value_columns: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    query: Optional[str] = None  # For SQL sources

    def __post_init__(self):
        if not self.value_columns:
            self.value_columns = []


@dataclass
class ReplayConfig:
    """回放配置"""
    time_scale: TimeScaleMode = TimeScaleMode.REAL_TIME
    speed_factor: float = 1.0  # 速度因子 (time_scale为FAST/SLOW时使用)
    start_offset: float = 0.0  # 起始偏移(秒)
    end_offset: Optional[float] = None  # 结束偏移(秒)
    loop: bool = False  # 是否循环播放
    emit_interval: float = 0.1  # 数据发送间隔(秒)
    buffer_size: int = 10000  # 缓冲区大小
    interpolate: bool = True  # 是否插值
    extrapolate: bool = False  # 是否外推


@dataclass
class ReplayDataPoint:
    """回放数据点"""
    timestamp: datetime
    channel: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplaySession:
    """回放会话"""
    session_id: str
    name: str
    description: str = ""
    sources: List[DataSource] = field(default_factory=list)
    config: ReplayConfig = field(default_factory=ReplayConfig)
    state: ReplayState = ReplayState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    data_points_total: int = 0
    data_points_played: int = 0
    error_message: Optional[str] = None

    @property
    def progress(self) -> float:
        """获取回放进度 (0-100%)"""
        if self.data_points_total == 0:
            return 0.0
        return (self.data_points_played / self.data_points_total) * 100.0

    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """获取已用时间"""
        if self.start_time and self.current_time:
            return self.current_time - self.start_time
        return None

    @property
    def remaining_time(self) -> Optional[timedelta]:
        """获取剩余时间"""
        if self.start_time and self.current_time and self.end_time:
            return self.end_time - self.current_time
        return None


class DataReplayEngine:
    """数据回放引擎"""

    def __init__(self):
        self.sessions: Dict[str, ReplaySession] = {}
        self.active_session: Optional[ReplaySession] = None
        self._data_buffer: Dict[str, List[ReplayDataPoint]] = {}
        self._sorted_timeline: List[ReplayDataPoint] = []

        # Internal state
        self._lock = threading.RLock()
        self._running = False
        self._paused = False
        self._replay_thread: Optional[threading.Thread] = None
        self._next_session_id = 1

        # Callbacks
        self._data_callbacks: List[Callable[[ReplayDataPoint], None]] = []
        self._state_callbacks: List[Callable[[ReplaySession, ReplayState], None]] = []
        self._progress_callbacks: List[Callable[[ReplaySession, float], None]] = []

        # Statistics
        self.stats = {
            'sessions_created': 0,
            'sessions_completed': 0,
            'data_points_played': 0,
            'total_play_time_s': 0.0,
            'errors': 0,
        }

        logger.info("Data Replay Engine initialized")

    def create_session(
        self,
        name: str,
        sources: List[DataSource],
        config: Optional[ReplayConfig] = None,
        description: str = "",
    ) -> ReplaySession:
        """创建回放会话"""
        session_id = f"replay-{self._next_session_id:06d}"
        self._next_session_id += 1

        session = ReplaySession(
            session_id=session_id,
            name=name,
            description=description,
            sources=sources,
            config=config or ReplayConfig(),
        )

        self.sessions[session_id] = session
        self.stats['sessions_created'] += 1

        logger.info(f"Created replay session: {session_id} - {name}")
        return session

    def load_session(self, session_id: str) -> bool:
        """加载会话数据"""
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return False

        try:
            session.state = ReplayState.LOADING
            self._notify_state_change(session)

            # Clear buffers
            self._data_buffer.clear()
            self._sorted_timeline.clear()

            # Load data from all sources
            for source in session.sources:
                data_points = self._load_source(source)
                self._data_buffer[source.name] = data_points
                logger.info(f"Loaded {len(data_points)} points from {source.name}")

            # Merge and sort all data points
            all_points = []
            for points in self._data_buffer.values():
                all_points.extend(points)

            self._sorted_timeline = sorted(all_points, key=lambda p: p.timestamp)

            # Apply time offsets
            if self._sorted_timeline:
                session.start_time = self._sorted_timeline[0].timestamp
                session.end_time = self._sorted_timeline[-1].timestamp

                # Apply start/end offset
                if session.config.start_offset > 0:
                    offset = timedelta(seconds=session.config.start_offset)
                    session.start_time = session.start_time + offset
                    self._sorted_timeline = [
                        p for p in self._sorted_timeline
                        if p.timestamp >= session.start_time
                    ]

                if session.config.end_offset:
                    end_offset = timedelta(seconds=session.config.end_offset)
                    new_end_time = session.start_time + end_offset
                    self._sorted_timeline = [
                        p for p in self._sorted_timeline
                        if p.timestamp <= new_end_time
                    ]
                    session.end_time = new_end_time

            session.data_points_total = len(self._sorted_timeline)
            session.data_points_played = 0
            session.current_time = session.start_time
            session.state = ReplayState.READY

            # Set as active session for step/seek operations
            self.active_session = session

            self._notify_state_change(session)
            logger.info(f"Session {session_id} loaded: {session.data_points_total} data points")
            return True

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            session.state = ReplayState.ERROR
            session.error_message = str(e)
            self._notify_state_change(session)
            self.stats['errors'] += 1
            return False

    def _load_source(self, source: DataSource) -> List[ReplayDataPoint]:
        """从数据源加载数据"""
        if source.source_type == DataSourceType.CSV:
            return self._load_csv(source)
        elif source.source_type == DataSourceType.JSON:
            return self._load_json(source)
        elif source.source_type == DataSourceType.SQLITE:
            return self._load_sqlite(source)
        elif source.source_type == DataSourceType.MEMORY:
            return self._load_memory(source)
        else:
            raise ValueError(f"Unsupported source type: {source.source_type}")

    def _load_csv(self, source: DataSource) -> List[ReplayDataPoint]:
        """加载CSV数据"""
        data_points = []

        if not source.path or not os.path.exists(source.path):
            raise FileNotFoundError(f"CSV file not found: {source.path}")

        with open(source.path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Parse timestamp
                ts_str = row.get(source.timestamp_column)
                if not ts_str:
                    continue

                try:
                    timestamp = datetime.strptime(ts_str, source.timestamp_format)
                except ValueError:
                    # Try ISO format
                    timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

                # Extract values
                value_columns = source.value_columns or [
                    k for k in row.keys()
                    if k != source.timestamp_column
                ]

                for col in value_columns:
                    if col in row:
                        try:
                            value = float(row[col])
                        except (ValueError, TypeError):
                            value = row[col]

                        data_points.append(ReplayDataPoint(
                            timestamp=timestamp,
                            channel=f"{source.name}.{col}",
                            value=value,
                            metadata={'source': source.name, 'column': col},
                        ))

        return data_points

    def _load_json(self, source: DataSource) -> List[ReplayDataPoint]:
        """加载JSON数据"""
        data_points = []

        if source.path and os.path.exists(source.path):
            with open(source.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif source.data:
            data = source.data
        else:
            raise ValueError("No JSON data provided")

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and 'data' in data:
            records = data['data']
        else:
            records = [data]

        for record in records:
            # Parse timestamp
            ts_str = record.get(source.timestamp_column)
            if not ts_str:
                continue

            try:
                if isinstance(ts_str, datetime):
                    timestamp = ts_str
                else:
                    timestamp = datetime.strptime(ts_str, source.timestamp_format)
            except ValueError:
                try:
                    timestamp = datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
                except ValueError:
                    continue

            # Extract values
            value_columns = source.value_columns or [
                k for k in record.keys()
                if k != source.timestamp_column
            ]

            for col in value_columns:
                if col in record:
                    data_points.append(ReplayDataPoint(
                        timestamp=timestamp,
                        channel=f"{source.name}.{col}",
                        value=record[col],
                        metadata={'source': source.name, 'column': col},
                    ))

        return data_points

    def _load_sqlite(self, source: DataSource) -> List[ReplayDataPoint]:
        """加载SQLite数据"""
        data_points = []

        if not source.path or not os.path.exists(source.path):
            raise FileNotFoundError(f"SQLite database not found: {source.path}")

        try:
            conn = sqlite3.connect(source.path, timeout=10.0)
        except sqlite3.OperationalError as exc:
            raise OSError(f"Cannot open database: {source.path}") from exc
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()

            if source.query:
                cursor.execute(source.query)
            else:
                # Build default query with validated identifiers
                columns = [source.timestamp_column] + source.value_columns
                safe_cols = [_validate_sql_identifier(c) for c in columns]
                safe_table = _validate_sql_identifier(
                    source.name.replace('.', '_')
                )
                safe_order = _validate_sql_identifier(source.timestamp_column)
                query = (
                    f"SELECT {', '.join(safe_cols)} "
                    f"FROM {safe_table} ORDER BY {safe_order}"
                )
                cursor.execute(query)

            for row in cursor.fetchall():
                row_dict = dict(row)

                # Parse timestamp
                ts_value = row_dict.get(source.timestamp_column)
                if not ts_value:
                    continue

                if isinstance(ts_value, (int, float)):
                    timestamp = datetime.fromtimestamp(ts_value)
                else:
                    try:
                        timestamp = datetime.strptime(str(ts_value), source.timestamp_format)
                    except ValueError:
                        timestamp = datetime.fromisoformat(str(ts_value))

                # Extract values
                for col in source.value_columns:
                    if col in row_dict:
                        data_points.append(ReplayDataPoint(
                            timestamp=timestamp,
                            channel=f"{source.name}.{col}",
                            value=row_dict[col],
                            metadata={'source': source.name, 'column': col},
                        ))

        finally:
            conn.close()

        return data_points

    def _load_memory(self, source: DataSource) -> List[ReplayDataPoint]:
        """加载内存数据"""
        data_points = []

        if not source.data:
            return data_points

        for record in source.data:
            # Parse timestamp
            ts_value = record.get(source.timestamp_column)
            if not ts_value:
                continue

            if isinstance(ts_value, datetime):
                timestamp = ts_value
            elif isinstance(ts_value, (int, float)):
                timestamp = datetime.fromtimestamp(ts_value)
            else:
                try:
                    timestamp = datetime.strptime(str(ts_value), source.timestamp_format)
                except ValueError:
                    timestamp = datetime.fromisoformat(str(ts_value))

            # Extract values
            value_columns = source.value_columns or [
                k for k in record.keys()
                if k != source.timestamp_column
            ]

            for col in value_columns:
                if col in record:
                    data_points.append(ReplayDataPoint(
                        timestamp=timestamp,
                        channel=f"{source.name}.{col}",
                        value=record[col],
                        metadata={'source': source.name, 'column': col},
                    ))

        return data_points

    def play(self, session_id: Optional[str] = None) -> bool:
        """开始回放"""
        with self._lock:
            if session_id:
                session = self.sessions.get(session_id)
                if not session:
                    logger.error(f"Session not found: {session_id}")
                    return False
            elif self.active_session:
                session = self.active_session
            else:
                logger.error("No active session")
                return False

            if session.state not in [ReplayState.READY, ReplayState.PAUSED]:
                logger.error(f"Cannot play session in state: {session.state}")
                return False

            self.active_session = session
            self._paused = False
            session.state = ReplayState.PLAYING
            self._notify_state_change(session)

            # Start replay thread if not running
            if not self._running:
                self._running = True
                self._replay_thread = threading.Thread(target=self._replay_loop, daemon=True)
                self._replay_thread.start()

            logger.info(f"Started replay: {session.session_id}")
            return True

    def pause(self) -> bool:
        """暂停回放"""
        with self._lock:
            if not self.active_session:
                return False

            if self.active_session.state != ReplayState.PLAYING:
                return False

            self._paused = True
            self.active_session.state = ReplayState.PAUSED
            self._notify_state_change(self.active_session)

            logger.info(f"Paused replay: {self.active_session.session_id}")
            return True

    def stop(self) -> bool:
        """停止回放"""
        with self._lock:
            if not self.active_session:
                return False

            self._running = False
            self._paused = False

            if self._replay_thread:
                self._replay_thread.join(timeout=5.0)
                self._replay_thread = None

            self.active_session.state = ReplayState.STOPPED
            self._notify_state_change(self.active_session)

            logger.info(f"Stopped replay: {self.active_session.session_id}")
            return True

    def seek(self, timestamp: Union[datetime, float]) -> bool:
        """跳转到指定时间"""
        with self._lock:
            if not self.active_session:
                return False

            if isinstance(timestamp, (int, float)):
                # Convert seconds offset to datetime
                timestamp = self.active_session.start_time + timedelta(seconds=timestamp)

            if timestamp < self.active_session.start_time:
                timestamp = self.active_session.start_time
            if timestamp > self.active_session.end_time:
                timestamp = self.active_session.end_time

            self.active_session.current_time = timestamp

            # Find data point index
            for i, point in enumerate(self._sorted_timeline):
                if point.timestamp >= timestamp:
                    self.active_session.data_points_played = i
                    break

            self._notify_progress(self.active_session)
            logger.info(f"Seeked to {timestamp}")
            return True

    def step(self, count: int = 1) -> List[ReplayDataPoint]:
        """单步回放指定数量的数据点"""
        with self._lock:
            if not self.active_session:
                return []

            if self.active_session.state not in [ReplayState.READY, ReplayState.PAUSED]:
                return []

            played_points = []
            start_idx = self.active_session.data_points_played

            for i in range(count):
                idx = start_idx + i
                if idx >= len(self._sorted_timeline):
                    break

                point = self._sorted_timeline[idx]
                self._emit_data_point(point)
                played_points.append(point)
                self.active_session.data_points_played = idx + 1
                self.active_session.current_time = point.timestamp

            self._notify_progress(self.active_session)
            return played_points

    def _replay_loop(self):
        """回放主循环"""
        play_start_time = time.time()
        timeline_idx = self.active_session.data_points_played if self.active_session else 0

        while self._running and self.active_session:
            if self._paused:
                time.sleep(0.1)
                continue

            session = self.active_session
            config = session.config

            # Check if we've reached the end
            if timeline_idx >= len(self._sorted_timeline):
                if config.loop:
                    timeline_idx = 0
                    session.data_points_played = 0
                    session.current_time = session.start_time
                    play_start_time = time.time()
                    logger.info("Replay loop restarted")
                else:
                    session.state = ReplayState.COMPLETED
                    self._notify_state_change(session)
                    self.stats['sessions_completed'] += 1
                    logger.info(f"Replay completed: {session.session_id}")
                    break

            # Get next data point
            point = self._sorted_timeline[timeline_idx]

            # Calculate delay based on time scale mode
            if config.time_scale == TimeScaleMode.IMMEDIATE:
                delay = 0
            elif config.time_scale == TimeScaleMode.STEP:
                # Step mode - wait for manual step
                time.sleep(0.1)
                continue
            else:
                # Calculate time since session start
                elapsed_real = time.time() - play_start_time
                point_offset = (point.timestamp - session.start_time).total_seconds()

                if config.time_scale == TimeScaleMode.FAST:
                    point_offset /= config.speed_factor
                elif config.time_scale == TimeScaleMode.SLOW:
                    point_offset *= config.speed_factor

                delay = point_offset - elapsed_real
                if delay > 0:
                    time.sleep(min(delay, config.emit_interval))
                    continue

            # Emit data point
            self._emit_data_point(point)

            # Update session state
            timeline_idx += 1
            session.data_points_played = timeline_idx
            session.current_time = point.timestamp
            self.stats['data_points_played'] += 1

            # Notify progress periodically
            if timeline_idx % 100 == 0:
                self._notify_progress(session)

            time.sleep(config.emit_interval)

        self.stats['total_play_time_s'] += time.time() - play_start_time
        self._running = False

    def _emit_data_point(self, point: ReplayDataPoint):
        """发送数据点到回调"""
        for callback in self._data_callbacks:
            try:
                callback(point)
            except Exception as e:
                logger.error(f"Data callback error: {e}")
                self.stats['errors'] += 1

    def _notify_state_change(self, session: ReplaySession):
        """通知状态变化"""
        for callback in self._state_callbacks:
            try:
                callback(session, session.state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def _notify_progress(self, session: ReplaySession):
        """通知进度更新"""
        for callback in self._progress_callbacks:
            try:
                callback(session, session.progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def on_data(self, callback: Callable[[ReplayDataPoint], None]):
        """注册数据回调"""
        self._data_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[ReplaySession, ReplayState], None]):
        """注册状态变化回调"""
        self._state_callbacks.append(callback)

    def on_progress(self, callback: Callable[[ReplaySession, float], None]):
        """注册进度回调"""
        self._progress_callbacks.append(callback)

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话状态"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        return {
            'session_id': session.session_id,
            'name': session.name,
            'state': session.state.value,
            'progress': session.progress,
            'data_points_total': session.data_points_total,
            'data_points_played': session.data_points_played,
            'start_time': session.start_time.isoformat() if session.start_time else None,
            'end_time': session.end_time.isoformat() if session.end_time else None,
            'current_time': session.current_time.isoformat() if session.current_time else None,
            'elapsed': str(session.elapsed_time) if session.elapsed_time else None,
            'remaining': str(session.remaining_time) if session.remaining_time else None,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取回放引擎统计信息"""
        return {
            'active_session': self.active_session.session_id if self.active_session else None,
            'total_sessions': len(self.sessions),
            **self.stats,
        }

    def create_water_network_replay(
        self,
        name: str,
        num_pools: int,
        duration_hours: float = 24.0,
        sample_interval_s: float = 60.0,
    ) -> ReplaySession:
        """创建智能水网模拟回放会话"""
        import random
        import math

        # Generate simulated data
        data = []
        start_time = datetime.now() - timedelta(hours=duration_hours)
        num_samples = int(duration_hours * 3600 / sample_interval_s)

        for i in range(num_samples):
            timestamp = start_time + timedelta(seconds=i * sample_interval_s)
            hour_of_day = timestamp.hour + timestamp.minute / 60.0

            # Daily pattern
            daily_factor = 1.0 + 0.3 * math.sin(2 * math.pi * (hour_of_day - 6) / 24)

            record = {'timestamp': timestamp}

            for pool_id in range(num_pools):
                # Water level with daily pattern and noise
                base_level = 3.0 + pool_id * 0.3
                record[f'pool_{pool_id}_water_level'] = base_level + 0.5 * daily_factor + random.gauss(0, 0.1)

                # Flow rate
                record[f'pool_{pool_id}_flow_rate'] = 5.0 * daily_factor + random.gauss(0, 0.5)

                # Gate position
                record[f'pool_{pool_id}_gate_position'] = 50.0 + 20.0 * daily_factor + random.gauss(0, 5)

                # Temperature
                temp_factor = 1.0 + 0.15 * math.sin(2 * math.pi * (hour_of_day - 14) / 24)
                record[f'pool_{pool_id}_temperature'] = 18.0 + 5.0 * temp_factor + random.gauss(0, 0.5)

            data.append(record)

        # Create memory data source
        value_columns = []
        for pool_id in range(num_pools):
            value_columns.extend([
                f'pool_{pool_id}_water_level',
                f'pool_{pool_id}_flow_rate',
                f'pool_{pool_id}_gate_position',
                f'pool_{pool_id}_temperature',
            ])

        source = DataSource(
            name='water_network_sim',
            source_type=DataSourceType.MEMORY,
            data=data,
            timestamp_column='timestamp',
            value_columns=value_columns,
        )

        # Create session
        session = self.create_session(
            name=name,
            sources=[source],
            config=ReplayConfig(
                time_scale=TimeScaleMode.FAST,
                speed_factor=60.0,  # 1 minute real = 1 hour replay
            ),
            description=f"Simulated water network data for {num_pools} pools over {duration_hours} hours",
        )

        # Load the session
        self.load_session(session.session_id)

        logger.info(f"Created water network replay session with {len(data)} data points")
        return session

    def export_session_data(
        self,
        session_id: str,
        output_path: str,
        format: str = 'csv',
    ) -> bool:
        """导出会话数据"""
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return False

        try:
            if format == 'csv':
                return self._export_csv(output_path)
            elif format == 'json':
                return self._export_json(output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def _export_csv(self, output_path: str) -> bool:
        """导出为CSV"""
        if not self._sorted_timeline:
            return False

        # Group by timestamp
        rows = {}
        channels = set()

        for point in self._sorted_timeline:
            ts_str = point.timestamp.isoformat()
            if ts_str not in rows:
                rows[ts_str] = {'timestamp': ts_str}
            rows[ts_str][point.channel] = point.value
            channels.add(point.channel)

        # Write CSV
        fieldnames = ['timestamp'] + sorted(list(channels))

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ts_str in sorted(rows.keys()):
                writer.writerow(rows[ts_str])

        logger.info(f"Exported {len(rows)} rows to {output_path}")
        return True

    def _export_json(self, output_path: str) -> bool:
        """导出为JSON"""
        if not self._sorted_timeline:
            return False

        data = [
            {
                'timestamp': point.timestamp.isoformat(),
                'channel': point.channel,
                'value': point.value,
                'metadata': point.metadata,
            }
            for point in self._sorted_timeline
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'data': data, 'count': len(data)}, f, indent=2)

        logger.info(f"Exported {len(data)} points to {output_path}")
        return True
