# Phase 5.9: SCADA Integration Interface
# SCADA集成接口 - 监控与数据采集系统集成

import asyncio
import logging
import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict
import queue

logger = logging.getLogger(__name__)


class SCADAConnectionStatus(Enum):
    """SCADA连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class SCADATagQuality(Enum):
    """标签数据质量"""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    NOT_CONNECTED = "not_connected"
    STALE = "stale"
    SUBSTITUTE = "substitute"


class SCADATagType(Enum):
    """标签数据类型"""
    ANALOG = "analog"
    DIGITAL = "digital"
    STRING = "string"
    DATETIME = "datetime"
    ARRAY = "array"


class SCADAAlarmPriority(Enum):
    """报警优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class SCADAAlarmState(Enum):
    """报警状态"""
    NORMAL = "normal"
    ALARM = "alarm"
    ACKNOWLEDGED = "acknowledged"
    CLEARED = "cleared"
    DISABLED = "disabled"


class SCADACommandState(Enum):
    """命令状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


@dataclass
class SCADATag:
    """SCADA标签定义"""
    tag_name: str
    tag_type: SCADATagType = SCADATagType.ANALOG
    description: str = ""
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    deadband: float = 0.0
    scan_rate: float = 1.0  # seconds
    writable: bool = False

    # Engineering units conversion
    raw_min: float = 0.0
    raw_max: float = 65535.0
    eng_min: float = 0.0
    eng_max: float = 100.0

    # Alarm settings
    alarm_enabled: bool = True
    alarm_hi_hi: Optional[float] = None
    alarm_hi: Optional[float] = None
    alarm_lo: Optional[float] = None
    alarm_lo_lo: Optional[float] = None
    alarm_priority: SCADAAlarmPriority = SCADAAlarmPriority.MEDIUM

    # Water network metadata
    pool_id: Optional[int] = None
    measurement_type: Optional[str] = None
    area: Optional[str] = None

    # Current state (updated at runtime)
    value: Any = None
    quality: SCADATagQuality = SCADATagQuality.NOT_CONNECTED
    timestamp: Optional[datetime] = None
    alarm_state: SCADAAlarmState = SCADAAlarmState.NORMAL

    def to_engineering(self, raw_value: float) -> float:
        """原始值转换为工程值"""
        if self.raw_max == self.raw_min:
            return self.eng_min
        ratio = (raw_value - self.raw_min) / (self.raw_max - self.raw_min)
        return self.eng_min + ratio * (self.eng_max - self.eng_min)

    def to_raw(self, eng_value: float) -> float:
        """工程值转换为原始值"""
        if self.eng_max == self.eng_min:
            return self.raw_min
        ratio = (eng_value - self.eng_min) / (self.eng_max - self.eng_min)
        return self.raw_min + ratio * (self.raw_max - self.raw_min)

    def check_alarm(self, value: float) -> SCADAAlarmState:
        """检查报警状态"""
        if not self.alarm_enabled:
            return SCADAAlarmState.NORMAL

        if self.alarm_hi_hi is not None and value >= self.alarm_hi_hi:
            return SCADAAlarmState.ALARM
        if self.alarm_lo_lo is not None and value <= self.alarm_lo_lo:
            return SCADAAlarmState.ALARM
        if self.alarm_hi is not None and value >= self.alarm_hi:
            return SCADAAlarmState.ALARM
        if self.alarm_lo is not None and value <= self.alarm_lo:
            return SCADAAlarmState.ALARM

        return SCADAAlarmState.NORMAL


@dataclass
class SCADAAlarm:
    """SCADA报警"""
    alarm_id: str
    tag: SCADATag
    priority: SCADAAlarmPriority
    state: SCADAAlarmState
    message: str
    timestamp: datetime
    value: Any
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_time: Optional[datetime] = None
    cleared_time: Optional[datetime] = None
    comment: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.state in [SCADAAlarmState.ALARM, SCADAAlarmState.ACKNOWLEDGED]


@dataclass
class SCADACommand:
    """SCADA命令"""
    command_id: str
    tag_name: str
    value: Any
    requested_by: str
    requested_time: datetime
    state: SCADACommandState = SCADACommandState.PENDING
    executed_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    timeout: float = 30.0  # seconds


class SCADAInterface:
    """SCADA集成接口"""

    def __init__(
        self,
        name: str = "E2EControl-SCADA",
        server_url: Optional[str] = None,
    ):
        self.name = name
        self.server_url = server_url
        self.status = SCADAConnectionStatus.DISCONNECTED

        # Tag management
        self.tags: Dict[str, SCADATag] = {}
        self.tag_groups: Dict[str, List[str]] = defaultdict(list)

        # Alarm management
        self.alarms: Dict[str, SCADAAlarm] = {}
        self.alarm_history: List[SCADAAlarm] = []
        self.max_alarm_history = 10000

        # Command management
        self.commands: Dict[str, SCADACommand] = {}
        self.command_history: List[SCADACommand] = []
        self.max_command_history = 1000

        # Data buffering
        self._data_buffer: queue.Queue = queue.Queue(maxsize=50000)
        self._history_buffer: Dict[str, List[tuple]] = defaultdict(list)
        self._max_history_points = 10000

        # Internal state
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._next_command_id = 1
        self._next_alarm_id = 1

        # Callbacks
        self._tag_change_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._alarm_callbacks: List[Callable] = []
        self._connection_callbacks: List[Callable] = []

        # Statistics
        self.stats = {
            'tag_updates': 0,
            'commands_sent': 0,
            'commands_completed': 0,
            'commands_failed': 0,
            'alarms_generated': 0,
            'alarms_acknowledged': 0,
            'errors': 0,
            'uptime_start': None,
            'last_update': None,
        }

        logger.info(f"SCADA interface '{name}' created")

    def connect(self) -> bool:
        """连接到SCADA系统"""
        with self._lock:
            if self.status == SCADAConnectionStatus.CONNECTED:
                logger.warning("Already connected to SCADA system")
                return True

            try:
                self.status = SCADAConnectionStatus.CONNECTING
                logger.info(f"Connecting to SCADA system: {self.server_url}")

                # Simulate connection
                time.sleep(0.1)

                self.status = SCADAConnectionStatus.CONNECTED
                self.stats['uptime_start'] = datetime.now()

                # Start worker thread
                self._start_worker()

                # Notify callbacks
                for callback in self._connection_callbacks:
                    try:
                        callback(True, None)
                    except Exception as e:
                        logger.error(f"Connection callback error: {e}")

                logger.info("Connected to SCADA system")
                return True

            except Exception as e:
                logger.error(f"Failed to connect to SCADA system: {e}")
                self.status = SCADAConnectionStatus.ERROR
                self.stats['errors'] += 1
                return False

    def disconnect(self):
        """断开SCADA连接"""
        with self._lock:
            if self.status == SCADAConnectionStatus.DISCONNECTED:
                return

            logger.info("Disconnecting from SCADA system")

            self._stop_worker()
            self.status = SCADAConnectionStatus.DISCONNECTED

            # Notify callbacks
            for callback in self._connection_callbacks:
                try:
                    callback(False, None)
                except Exception as e:
                    logger.error(f"Disconnection callback error: {e}")

            logger.info("Disconnected from SCADA system")

    def _start_worker(self):
        """启动工作线程"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _stop_worker(self):
        """停止工作线程"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None

    def _worker_loop(self):
        """工作循环"""
        last_scan_times: Dict[str, float] = {}

        while self._running:
            try:
                current_time = time.time()

                # Scan tags based on scan rate
                for tag_name, tag in self.tags.items():
                    last_scan = last_scan_times.get(tag_name, 0)
                    if current_time - last_scan >= tag.scan_rate:
                        self._scan_tag(tag)
                        last_scan_times[tag_name] = current_time

                # Process command queue
                self._process_commands()

                # Check alarm states
                self._check_alarms()

                time.sleep(0.1)  # 100ms tick

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                self.stats['errors'] += 1

    def _scan_tag(self, tag: SCADATag):
        """扫描标签获取新值"""
        import random

        # Simulate tag value based on measurement type
        if tag.measurement_type == 'water_level':
            value = random.uniform(1.0, 5.0)
        elif tag.measurement_type == 'flow_rate':
            value = random.uniform(0.0, 10.0)
        elif tag.measurement_type == 'gate_position':
            value = random.uniform(0.0, 100.0)
        elif tag.measurement_type == 'pressure':
            value = random.uniform(50.0, 200.0)
        elif tag.measurement_type == 'temperature':
            value = random.uniform(15.0, 25.0)
        elif tag.tag_type == SCADATagType.DIGITAL:
            value = random.choice([0, 1])
        else:
            value = random.uniform(tag.eng_min, tag.eng_max)

        self._update_tag_value(tag.tag_name, value)

    def _process_commands(self):
        """处理待执行命令"""
        for cmd_id, command in list(self.commands.items()):
            if command.state == SCADACommandState.PENDING:
                self._execute_command(command)

    def _execute_command(self, command: SCADACommand):
        """执行命令"""
        try:
            command.state = SCADACommandState.EXECUTING
            command.executed_time = datetime.now()

            # Simulate command execution
            time.sleep(0.05)

            # Update tag value
            tag = self.tags.get(command.tag_name)
            if tag:
                self._update_tag_value(command.tag_name, command.value)
                command.state = SCADACommandState.COMPLETED
                command.result = "Success"
                self.stats['commands_completed'] += 1
            else:
                command.state = SCADACommandState.FAILED
                command.error = "Tag not found"
                self.stats['commands_failed'] += 1

            command.completed_time = datetime.now()

        except Exception as e:
            command.state = SCADACommandState.FAILED
            command.error = str(e)
            command.completed_time = datetime.now()
            self.stats['commands_failed'] += 1
            logger.error(f"Command execution failed: {e}")

    def _check_alarms(self):
        """检查报警状态"""
        for tag_name, tag in self.tags.items():
            if tag.value is None or tag.tag_type != SCADATagType.ANALOG:
                continue

            new_alarm_state = tag.check_alarm(tag.value)

            if new_alarm_state == SCADAAlarmState.ALARM and tag.alarm_state == SCADAAlarmState.NORMAL:
                # Generate new alarm
                self._generate_alarm(tag)
            elif new_alarm_state == SCADAAlarmState.NORMAL and tag.alarm_state == SCADAAlarmState.ALARM:
                # Clear alarm
                self._clear_alarm_for_tag(tag)

            tag.alarm_state = new_alarm_state

    def _generate_alarm(self, tag: SCADATag):
        """生成报警"""
        alarm_id = f"ALM-{self._next_alarm_id:06d}"
        self._next_alarm_id += 1

        # Determine alarm message
        message = f"Alarm on {tag.tag_name}"
        if tag.alarm_hi_hi and tag.value >= tag.alarm_hi_hi:
            message = f"High-High alarm: {tag.value:.2f} >= {tag.alarm_hi_hi}"
        elif tag.alarm_lo_lo and tag.value <= tag.alarm_lo_lo:
            message = f"Low-Low alarm: {tag.value:.2f} <= {tag.alarm_lo_lo}"
        elif tag.alarm_hi and tag.value >= tag.alarm_hi:
            message = f"High alarm: {tag.value:.2f} >= {tag.alarm_hi}"
        elif tag.alarm_lo and tag.value <= tag.alarm_lo:
            message = f"Low alarm: {tag.value:.2f} <= {tag.alarm_lo}"

        alarm = SCADAAlarm(
            alarm_id=alarm_id,
            tag=tag,
            priority=tag.alarm_priority,
            state=SCADAAlarmState.ALARM,
            message=message,
            timestamp=datetime.now(),
            value=tag.value,
        )

        self.alarms[alarm_id] = alarm
        self.stats['alarms_generated'] += 1

        # Notify callbacks
        for callback in self._alarm_callbacks:
            try:
                callback(alarm, 'generated')
            except Exception as e:
                logger.error(f"Alarm callback error: {e}")

        logger.warning(f"Alarm generated: {alarm_id} - {message}")

    def _clear_alarm_for_tag(self, tag: SCADATag):
        """清除标签相关报警"""
        for alarm_id, alarm in list(self.alarms.items()):
            if alarm.tag.tag_name == tag.tag_name and alarm.is_active:
                alarm.state = SCADAAlarmState.CLEARED
                alarm.cleared_time = datetime.now()

                # Move to history
                self.alarm_history.append(alarm)
                if len(self.alarm_history) > self.max_alarm_history:
                    self.alarm_history.pop(0)

                del self.alarms[alarm_id]

                # Notify callbacks
                for callback in self._alarm_callbacks:
                    try:
                        callback(alarm, 'cleared')
                    except Exception as e:
                        logger.error(f"Alarm callback error: {e}")

                logger.info(f"Alarm cleared: {alarm_id}")

    def register_tag(self, tag: SCADATag):
        """注册SCADA标签"""
        with self._lock:
            self.tags[tag.tag_name] = tag
            logger.debug(f"Registered tag: {tag.tag_name}")

    def register_tags(self, tags: List[SCADATag]):
        """批量注册标签"""
        for tag in tags:
            self.register_tag(tag)

    def add_tag_to_group(self, tag_name: str, group_name: str):
        """将标签添加到组"""
        with self._lock:
            if tag_name not in self.tag_groups[group_name]:
                self.tag_groups[group_name].append(tag_name)

    def _update_tag_value(
        self,
        tag_name: str,
        value: Any,
        quality: SCADATagQuality = SCADATagQuality.GOOD,
    ):
        """更新标签值"""
        tag = self.tags.get(tag_name)
        if not tag:
            return

        old_value = tag.value
        tag.value = value
        tag.quality = quality
        tag.timestamp = datetime.now()

        # Add to history
        self._history_buffer[tag_name].append((tag.timestamp, value, quality))
        if len(self._history_buffer[tag_name]) > self._max_history_points:
            self._history_buffer[tag_name].pop(0)

        # Update stats
        self.stats['tag_updates'] += 1
        self.stats['last_update'] = datetime.now()

        # Check deadband for callbacks
        if old_value is None or abs(value - old_value) >= tag.deadband:
            # Notify callbacks
            for callback in self._tag_change_callbacks.get(tag_name, []):
                try:
                    callback(tag)
                except Exception as e:
                    logger.error(f"Tag change callback error: {e}")

    def read_tag(self, tag_name: str) -> Optional[SCADATag]:
        """读取标签当前值"""
        return self.tags.get(tag_name)

    def read_tags(self, tag_names: List[str]) -> Dict[str, SCADATag]:
        """批量读取标签"""
        return {name: self.tags[name] for name in tag_names if name in self.tags}

    def read_group(self, group_name: str) -> Dict[str, SCADATag]:
        """读取标签组"""
        tag_names = self.tag_groups.get(group_name, [])
        return self.read_tags(tag_names)

    def write_tag(self, tag_name: str, value: Any, user: str = "system") -> SCADACommand:
        """写入标签值"""
        command_id = f"CMD-{self._next_command_id:06d}"
        self._next_command_id += 1

        command = SCADACommand(
            command_id=command_id,
            tag_name=tag_name,
            value=value,
            requested_by=user,
            requested_time=datetime.now(),
        )

        self.commands[command_id] = command
        self.stats['commands_sent'] += 1

        logger.debug(f"Command queued: {command_id} - Write {value} to {tag_name}")
        return command

    def acknowledge_alarm(
        self,
        alarm_id: str,
        user: str,
        comment: Optional[str] = None,
    ) -> bool:
        """确认报警"""
        alarm = self.alarms.get(alarm_id)
        if not alarm:
            logger.warning(f"Alarm not found: {alarm_id}")
            return False

        if alarm.acknowledged:
            logger.warning(f"Alarm already acknowledged: {alarm_id}")
            return False

        alarm.acknowledged = True
        alarm.acknowledged_by = user
        alarm.acknowledged_time = datetime.now()
        alarm.comment = comment
        alarm.state = SCADAAlarmState.ACKNOWLEDGED

        self.stats['alarms_acknowledged'] += 1

        # Notify callbacks
        for callback in self._alarm_callbacks:
            try:
                callback(alarm, 'acknowledged')
            except Exception as e:
                logger.error(f"Alarm callback error: {e}")

        logger.info(f"Alarm acknowledged: {alarm_id} by {user}")
        return True

    def subscribe_tag(self, tag_name: str, callback: Callable[[SCADATag], None]):
        """订阅标签变化"""
        self._tag_change_callbacks[tag_name].append(callback)

    def subscribe_alarms(self, callback: Callable[[SCADAAlarm, str], None]):
        """订阅报警事件"""
        self._alarm_callbacks.append(callback)

    def on_connect(self, callback: Callable[[bool, Optional[str]], None]):
        """注册连接状态回调"""
        self._connection_callbacks.append(callback)

    def get_tag_history(
        self,
        tag_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[tuple]:
        """获取标签历史数据"""
        history = self._history_buffer.get(tag_name, [])

        if start_time is None and end_time is None:
            return history.copy()

        result = []
        for ts, value, quality in history:
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            result.append((ts, value, quality))

        return result

    def get_active_alarms(self) -> List[SCADAAlarm]:
        """获取活动报警列表"""
        return [a for a in self.alarms.values() if a.is_active]

    def get_alarm_history(
        self,
        limit: int = 100,
        priority: Optional[SCADAAlarmPriority] = None,
    ) -> List[SCADAAlarm]:
        """获取报警历史"""
        history = self.alarm_history[-limit:]
        if priority:
            history = [a for a in history if a.priority == priority]
        return history

    def get_status(self) -> Dict[str, Any]:
        """获取SCADA接口状态"""
        return {
            'name': self.name,
            'status': self.status.value,
            'server_url': self.server_url,
            'tag_count': len(self.tags),
            'group_count': len(self.tag_groups),
            'active_alarms': len([a for a in self.alarms.values() if a.is_active]),
            'pending_commands': len([c for c in self.commands.values() if c.state == SCADACommandState.PENDING]),
            'statistics': self.stats.copy(),
        }

    def create_water_network_tags(
        self,
        num_pools: int,
        area_name: str = "WaterNetwork",
    ) -> List[SCADATag]:
        """创建智能水网标准SCADA标签"""
        tags = []

        for pool_id in range(num_pools):
            tag_prefix = f"{area_name}.Pool{pool_id}"

            # Water level
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.WaterLevel",
                tag_type=SCADATagType.ANALOG,
                description=f"Pool {pool_id} Water Level",
                unit='m',
                eng_min=0.0,
                eng_max=10.0,
                measurement_type='water_level',
                pool_id=pool_id,
                area=area_name,
                alarm_hi_hi=9.0,
                alarm_hi=8.0,
                alarm_lo=1.0,
                alarm_lo_lo=0.5,
                alarm_priority=SCADAAlarmPriority.HIGH,
            ))

            # Inflow rate
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.InflowRate",
                tag_type=SCADATagType.ANALOG,
                description=f"Pool {pool_id} Inflow Rate",
                unit='m³/s',
                eng_min=0.0,
                eng_max=50.0,
                measurement_type='flow_rate',
                pool_id=pool_id,
                area=area_name,
            ))

            # Outflow rate
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.OutflowRate",
                tag_type=SCADATagType.ANALOG,
                description=f"Pool {pool_id} Outflow Rate",
                unit='m³/s',
                eng_min=0.0,
                eng_max=50.0,
                measurement_type='flow_rate',
                pool_id=pool_id,
                area=area_name,
            ))

            # Gate position
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.GatePosition",
                tag_type=SCADATagType.ANALOG,
                description=f"Pool {pool_id} Gate Position",
                unit='%',
                eng_min=0.0,
                eng_max=100.0,
                measurement_type='gate_position',
                pool_id=pool_id,
                area=area_name,
                writable=True,
            ))

            # Gate setpoint
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.GateSetpoint",
                tag_type=SCADATagType.ANALOG,
                description=f"Pool {pool_id} Gate Setpoint",
                unit='%',
                eng_min=0.0,
                eng_max=100.0,
                measurement_type='gate_setpoint',
                pool_id=pool_id,
                area=area_name,
                writable=True,
            ))

            # Temperature
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.Temperature",
                tag_type=SCADATagType.ANALOG,
                description=f"Pool {pool_id} Temperature",
                unit='°C',
                eng_min=-10.0,
                eng_max=50.0,
                measurement_type='temperature',
                pool_id=pool_id,
                area=area_name,
                alarm_hi=35.0,
                alarm_lo=0.0,
                alarm_priority=SCADAAlarmPriority.MEDIUM,
            ))

            # Pressure
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.Pressure",
                tag_type=SCADATagType.ANALOG,
                description=f"Pool {pool_id} Pressure",
                unit='kPa',
                eng_min=0.0,
                eng_max=500.0,
                measurement_type='pressure',
                pool_id=pool_id,
                area=area_name,
            ))

            # Pump running status
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.PumpRunning",
                tag_type=SCADATagType.DIGITAL,
                description=f"Pool {pool_id} Pump Running",
                measurement_type='pump_status',
                pool_id=pool_id,
                area=area_name,
            ))

            # Pump command
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.PumpCommand",
                tag_type=SCADATagType.DIGITAL,
                description=f"Pool {pool_id} Pump Command",
                measurement_type='pump_command',
                pool_id=pool_id,
                area=area_name,
                writable=True,
            ))

            # Valve status
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.ValveOpen",
                tag_type=SCADATagType.DIGITAL,
                description=f"Pool {pool_id} Valve Open",
                measurement_type='valve_status',
                pool_id=pool_id,
                area=area_name,
            ))

            # Alarm active
            tags.append(SCADATag(
                tag_name=f"{tag_prefix}.AlarmActive",
                tag_type=SCADATagType.DIGITAL,
                description=f"Pool {pool_id} Alarm Active",
                measurement_type='alarm_status',
                pool_id=pool_id,
                area=area_name,
                alarm_enabled=False,
            ))

        # System-level tags
        tags.append(SCADATag(
            tag_name=f"{area_name}.SystemStatus",
            tag_type=SCADATagType.ANALOG,
            description="System Status Code",
            area=area_name,
        ))

        tags.append(SCADATag(
            tag_name=f"{area_name}.SystemMode",
            tag_type=SCADATagType.ANALOG,
            description="System Operation Mode",
            area=area_name,
            writable=True,
        ))

        tags.append(SCADATag(
            tag_name=f"{area_name}.EmergencyStop",
            tag_type=SCADATagType.DIGITAL,
            description="Emergency Stop",
            area=area_name,
            writable=True,
            alarm_priority=SCADAAlarmPriority.CRITICAL,
        ))

        tags.append(SCADATag(
            tag_name=f"{area_name}.TotalFlow",
            tag_type=SCADATagType.ANALOG,
            description="Total Network Flow",
            unit='m³/s',
            eng_min=0.0,
            eng_max=500.0,
            measurement_type='flow_rate',
            area=area_name,
        ))

        tags.append(SCADATag(
            tag_name=f"{area_name}.ControllerHeartbeat",
            tag_type=SCADATagType.ANALOG,
            description="Controller Heartbeat Counter",
            area=area_name,
        ))

        # Register all tags
        self.register_tags(tags)

        # Create standard groups
        for pool_id in range(num_pools):
            pool_tags = [t.tag_name for t in tags if t.pool_id == pool_id]
            for tag_name in pool_tags:
                self.add_tag_to_group(tag_name, f"Pool{pool_id}")
                self.add_tag_to_group(tag_name, "AllPools")

        logger.info(f"Created {len(tags)} water network SCADA tags for {num_pools} pools")
        return tags

    def export_configuration(self) -> Dict[str, Any]:
        """导出SCADA配置"""
        return {
            'name': self.name,
            'server_url': self.server_url,
            'tags': [
                {
                    'tag_name': t.tag_name,
                    'tag_type': t.tag_type.value,
                    'description': t.description,
                    'unit': t.unit,
                    'eng_min': t.eng_min,
                    'eng_max': t.eng_max,
                    'writable': t.writable,
                    'alarm_enabled': t.alarm_enabled,
                    'alarm_hi_hi': t.alarm_hi_hi,
                    'alarm_hi': t.alarm_hi,
                    'alarm_lo': t.alarm_lo,
                    'alarm_lo_lo': t.alarm_lo_lo,
                    'alarm_priority': t.alarm_priority.value if t.alarm_priority else None,
                    'pool_id': t.pool_id,
                    'measurement_type': t.measurement_type,
                    'area': t.area,
                }
                for t in self.tags.values()
            ],
            'groups': dict(self.tag_groups),
        }

    def import_configuration(self, config: Dict[str, Any]):
        """导入SCADA配置"""
        self.name = config.get('name', self.name)
        self.server_url = config.get('server_url', self.server_url)

        # Clear existing
        self.tags.clear()
        self.tag_groups.clear()

        # Import tags
        for tag_config in config.get('tags', []):
            tag = SCADATag(
                tag_name=tag_config['tag_name'],
                tag_type=SCADATagType(tag_config.get('tag_type', 'analog')),
                description=tag_config.get('description', ''),
                unit=tag_config.get('unit'),
                eng_min=tag_config.get('eng_min', 0.0),
                eng_max=tag_config.get('eng_max', 100.0),
                writable=tag_config.get('writable', False),
                alarm_enabled=tag_config.get('alarm_enabled', True),
                alarm_hi_hi=tag_config.get('alarm_hi_hi'),
                alarm_hi=tag_config.get('alarm_hi'),
                alarm_lo=tag_config.get('alarm_lo'),
                alarm_lo_lo=tag_config.get('alarm_lo_lo'),
                alarm_priority=SCADAAlarmPriority(tag_config['alarm_priority']) if tag_config.get('alarm_priority') else SCADAAlarmPriority.MEDIUM,
                pool_id=tag_config.get('pool_id'),
                measurement_type=tag_config.get('measurement_type'),
                area=tag_config.get('area'),
            )
            self.register_tag(tag)

        # Import groups
        for group_name, tag_names in config.get('groups', {}).items():
            for tag_name in tag_names:
                self.add_tag_to_group(tag_name, group_name)

        logger.info(f"Imported SCADA configuration: {len(self.tags)} tags, {len(self.tag_groups)} groups")
