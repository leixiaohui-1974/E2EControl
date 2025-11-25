"""
Unified Control System Integration

Integrates all Phase 5 components into a unified control system:
- Hierarchical MPC (L3 + L2)
- Self-Healing System (L4)
- Performance Monitoring
- Macro Cognitive Layer
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import numpy as np
import threading
import time
import logging

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System operating modes"""
    INITIALIZING = "initializing"
    NORMAL = "normal"
    FLOOD_CONTROL = "flood_control"
    DROUGHT = "drought"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class EventType(Enum):
    """System event types"""
    MODE_CHANGE = "mode_change"
    FAULT_DETECTED = "fault_detected"
    FAULT_RESOLVED = "fault_resolved"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    ALERT_RAISED = "alert_raised"
    ALERT_CLEARED = "alert_cleared"
    COMMAND_RECEIVED = "command_received"
    CONTROL_UPDATE = "control_update"
    HEALTH_UPDATE = "health_update"


@dataclass
class SystemEvent:
    """System event record"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"


@dataclass
class ControlCommand:
    """Control command from higher layer"""
    command_id: str
    command_type: str  # set_mode, set_reference, override, etc.
    target: str  # Component or 'system'
    parameters: Dict[str, Any]
    timestamp: datetime
    priority: int = 5  # 1=highest
    ttl: Optional[timedelta] = None  # Time to live
    source: str = "user"


@dataclass
class SystemStatus:
    """Current system status"""
    mode: SystemMode
    health: float  # 0-1
    active_faults: int
    active_alerts: int
    control_active: bool
    components_online: int
    components_total: int
    uptime: timedelta
    last_update: datetime
    current_levels: List[float]
    current_flows: List[float]


class UnifiedControlSystem:
    """
    Unified Control System integrating all Phase 5 components.

    Architecture:
    - L3: Centralized Scheduler with Macro Cognitive Layer
    - L2: Parameterized Distributed MPC
    - L4: Self-Healing System
    - Performance Monitoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.num_pools = self.config.get('num_pools', 3)

        # System state
        self.mode = SystemMode.INITIALIZING
        self.start_time = datetime.now()
        self.is_running = False

        # Component references (lazy initialization)
        self._hierarchical_controller = None
        self._self_healing_system = None
        self._macro_cognitive = None
        self._performance_monitor = None

        # State tracking
        self.current_levels = [3.0] * self.num_pools
        self.current_flows = [10.0] * self.num_pools
        self.reference_levels = [3.0] * self.num_pools

        # Event management
        self.event_history: List[SystemEvent] = []
        self.event_callbacks: Dict[EventType, List[Callable]] = {
            et: [] for et in EventType
        }

        # Command queue
        self.command_queue: List[ControlCommand] = []
        self._command_lock = threading.Lock()

        # Control loop settings
        self.l3_interval = self.config.get('l3_interval', 3600.0)  # 1 hour
        self.l2_interval = self.config.get('l2_interval', 900.0)   # 15 min
        self.monitoring_interval = self.config.get('monitoring_interval', 60.0)

        # Timing tracking
        self._last_l3_update = None
        self._last_l2_update = None
        self._last_monitoring_update = None

        logger.info(f"[UnifiedControlSystem] 统一控制系统初始化 ({self.num_pools} 渠池)")

    def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            self._emit_event(
                EventType.MODE_CHANGE,
                "system",
                "System initialization started",
                {'from_mode': None, 'to_mode': SystemMode.INITIALIZING.value}
            )

            # Initialize performance monitor
            from .performance_monitor import PerformanceMonitor
            self._performance_monitor = PerformanceMonitor(
                self.config.get('monitoring', {})
            )
            self._performance_monitor.register_component(
                'unified_system', 'control_system',
                []
            )

            # Try to initialize hierarchical controller
            try:
                from ..controllers.hierarchical_mpc import (
                    HierarchicalMPCController, HierarchicalConfig
                )
                hier_config = HierarchicalConfig(
                    l3_planning_horizon=self.config.get('l3_horizon', 24),
                    l3_dt=self.l3_interval,
                    l2_control_horizon=self.config.get('l2_horizon', 5),
                    l2_dt=self.l2_interval
                )
                self._hierarchical_controller = HierarchicalMPCController(
                    num_pools=self.num_pools,
                    config=hier_config
                )
                self._performance_monitor.register_component(
                    'hierarchical_mpc', 'controller',
                    []
                )
                logger.info("[UnifiedControlSystem] 层级MPC控制器初始化成功")
            except ImportError as e:
                logger.warning(f"[UnifiedControlSystem] 层级MPC不可用: {e}")
                self._hierarchical_controller = None

            # Try to initialize self-healing system
            try:
                from ..self_healing import (
                    EnhancedDiagnosisEngine,
                    OptimizedIsolationStrategy,
                    PredictiveRecoveryEngine,
                    FaultLearningEngine,
                    MacroCognitiveLayer
                )
                self._self_healing_system = {
                    'diagnosis': EnhancedDiagnosisEngine(self.num_pools),
                    'isolation': OptimizedIsolationStrategy(self.num_pools),
                    'recovery': PredictiveRecoveryEngine(),
                    'learning': FaultLearningEngine(),
                }
                self._macro_cognitive = MacroCognitiveLayer()
                self._performance_monitor.register_component(
                    'self_healing', 'fault_management',
                    []
                )
                logger.info("[UnifiedControlSystem] 自愈系统初始化成功")
            except ImportError as e:
                logger.warning(f"[UnifiedControlSystem] 自愈系统不可用: {e}")
                self._self_healing_system = None
                self._macro_cognitive = None

            # Set system mode
            self.mode = SystemMode.NORMAL
            self._emit_event(
                EventType.MODE_CHANGE,
                "system",
                "System initialization completed",
                {'from_mode': SystemMode.INITIALIZING.value, 'to_mode': SystemMode.NORMAL.value}
            )

            return True

        except Exception as e:
            logger.error(f"[UnifiedControlSystem] 初始化失败: {e}")
            self.mode = SystemMode.SHUTDOWN
            return False

    def step(
        self,
        measurements: Dict[str, Any],
        dt: float = 60.0
    ) -> Dict[str, Any]:
        """
        Execute one control step.

        Args:
            measurements: Current system measurements
            dt: Time step in seconds

        Returns:
            Control actions and status
        """
        now = datetime.now()
        result = {
            'timestamp': now.isoformat(),
            'mode': self.mode.value,
            'control_actions': {},
            'events': [],
            'health': 1.0
        }

        # Update current state from measurements
        if 'levels' in measurements:
            self.current_levels = measurements['levels']
        if 'flows' in measurements:
            self.current_flows = measurements['flows']

        # Process pending commands
        self._process_commands()

        # Check if L3 update needed
        if self._should_update_l3(now):
            l3_result = self._update_l3(measurements)
            result['l3_update'] = l3_result

        # L2 control update
        if self._should_update_l2(now):
            l2_result = self._update_l2(measurements)
            result['control_actions'] = l2_result.get('actions', {})
            result['l2_update'] = l2_result

        # Self-healing check
        if self._self_healing_system:
            healing_result = self._check_self_healing(measurements)
            result['self_healing'] = healing_result
            if healing_result.get('fault_detected'):
                result['events'].append({
                    'type': 'fault_detected',
                    'data': healing_result
                })

        # Performance monitoring
        if self._should_update_monitoring(now):
            self._update_monitoring(measurements, result)

        # Update health
        if self._performance_monitor:
            health = self._performance_monitor.get_system_health()
            result['health'] = health.overall_health

        return result

    def _should_update_l3(self, now: datetime) -> bool:
        """Check if L3 scheduler needs update"""
        if self._last_l3_update is None:
            return True
        elapsed = (now - self._last_l3_update).total_seconds()
        return elapsed >= self.l3_interval

    def _should_update_l2(self, now: datetime) -> bool:
        """Check if L2 controller needs update"""
        if self._last_l2_update is None:
            return True
        elapsed = (now - self._last_l2_update).total_seconds()
        return elapsed >= self.l2_interval

    def _should_update_monitoring(self, now: datetime) -> bool:
        """Check if monitoring needs update"""
        if self._last_monitoring_update is None:
            return True
        elapsed = (now - self._last_monitoring_update).total_seconds()
        return elapsed >= self.monitoring_interval

    def _update_l3(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """Update L3 centralized scheduler"""
        now = datetime.now()
        self._last_l3_update = now
        result = {'success': False, 'solve_time': 0}

        if not self._hierarchical_controller:
            # Fallback: simple reference generation
            self.reference_levels = [3.0] * self.num_pools
            result['success'] = True
            result['reference_levels'] = self.reference_levels
            return result

        try:
            start_time = time.time()

            # Get macro cognitive advice if available
            cognitive_state = None
            if self._macro_cognitive:
                cognitive_state = self._macro_cognitive.assess(
                    system_state={'equipment_health': {
                        f'pool_{i}': 0.9 for i in range(self.num_pools)
                    }},
                    forecast_data=measurements.get('forecast', {}),
                    fault_reports=[],
                    external_factors=measurements.get('external', {})
                )

            # Update hierarchical controller
            l3_result = self._hierarchical_controller.update_l3(
                current_volumes=[l * 10000 for l in self.current_levels],  # Convert to volume
                forecast=measurements.get('forecast', {}),
                external_factors=measurements.get('external', {})
            )

            solve_time = time.time() - start_time
            result['solve_time'] = solve_time
            result['success'] = l3_result.get('success', True)
            result['mode'] = l3_result.get('mode', 'NORMAL')

            # Update system mode based on L3 result
            new_mode = self._map_scheduling_mode(l3_result.get('mode', 'NORMAL'))
            if new_mode != self.mode:
                self._change_mode(new_mode)

            # Store reference levels
            if 'reference_levels' in l3_result:
                self.reference_levels = l3_result['reference_levels'][:, 0].tolist()
            result['reference_levels'] = self.reference_levels

            # Record performance
            if self._performance_monitor:
                self._performance_monitor.record_solve_time(
                    'l3_scheduler', solve_time, success=result['success']
                )

        except Exception as e:
            logger.error(f"[UnifiedControlSystem] L3更新失败: {e}")
            result['error'] = str(e)

        return result

    def _update_l2(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """Update L2 distributed MPC"""
        now = datetime.now()
        self._last_l2_update = now
        result = {'success': False, 'actions': {}, 'solve_time': 0}

        if not self._hierarchical_controller:
            # Fallback: simple proportional control
            actions = []
            for i, (level, ref) in enumerate(zip(self.current_levels, self.reference_levels)):
                error = ref - level
                q_in = max(0, 10.0 + error * 5)  # Simple P control
                q_out = max(0, 8.0 - error * 3)
                actions.append({'q_in': q_in, 'q_out': q_out})
            result['success'] = True
            result['actions'] = actions
            return result

        try:
            start_time = time.time()

            # Get scenario from mode
            scenario = self._get_scenario_from_mode()

            # Update L2 distributed MPC
            l2_result = self._hierarchical_controller.update_l2(
                current_levels=self.current_levels,
                q_in_prevs=self.current_flows,
                scenario=scenario
            )

            solve_time = time.time() - start_time
            result['solve_time'] = solve_time
            result['success'] = l2_result.get('success', True)
            result['actions'] = l2_result.get('control_actions', [])
            result['converged'] = l2_result.get('converged', True)

            # Record performance
            if self._performance_monitor:
                self._performance_monitor.record_solve_time(
                    'l2_mpc', solve_time,
                    iterations=l2_result.get('iterations', 0),
                    success=result['success']
                )

                # Record tracking errors
                for i, (level, ref) in enumerate(zip(self.current_levels, self.reference_levels)):
                    error = abs(level - ref)
                    self._performance_monitor.record_tracking_error(
                        f'pool_{i}', error, ref, level
                    )

        except Exception as e:
            logger.error(f"[UnifiedControlSystem] L2更新失败: {e}")
            result['error'] = str(e)

        return result

    def _check_self_healing(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """Check and execute self-healing if needed"""
        result = {'fault_detected': False, 'recovery_active': False}

        if not self._self_healing_system:
            return result

        try:
            diagnosis_engine = self._self_healing_system['diagnosis']
            isolation_strategy = self._self_healing_system['isolation']
            recovery_engine = self._self_healing_system['recovery']

            # Check for anomalies in measurements
            anomalies = measurements.get('anomalies', [])

            if anomalies:
                # Diagnose faults
                diagnosis_results = diagnosis_engine.diagnose(
                    anomalies, measurements, {}
                )

                if diagnosis_results:
                    result['fault_detected'] = True
                    result['diagnoses'] = [
                        {
                            'fault_id': d.fault_id,
                            'fault_type': d.fault_type.value if hasattr(d.fault_type, 'value') else str(d.fault_type),
                            'component': d.component,
                            'severity': d.severity.value if hasattr(d.severity, 'value') else str(d.severity)
                        }
                        for d in diagnosis_results
                    ]

                    # Generate recovery plan for most severe fault
                    most_severe = diagnosis_results[0]
                    recovery_plan = recovery_engine.generate_recovery_plan(
                        fault_id=most_severe.fault_id,
                        fault_type=str(most_severe.fault_type),
                        affected_component=most_severe.component,
                        severity=0.7,
                        system_state=measurements
                    )

                    result['recovery_plan'] = {
                        'plan_id': recovery_plan.plan_id,
                        'strategy': recovery_plan.strategy.value,
                        'actions_count': len(recovery_plan.actions)
                    }
                    result['recovery_active'] = True

                    self._emit_event(
                        EventType.FAULT_DETECTED,
                        most_severe.component,
                        f"Fault detected: {most_severe.fault_type}",
                        {'fault_id': most_severe.fault_id}
                    )

        except Exception as e:
            logger.error(f"[UnifiedControlSystem] 自愈检查失败: {e}")
            result['error'] = str(e)

        return result

    def _update_monitoring(
        self,
        measurements: Dict[str, Any],
        step_result: Dict[str, Any]
    ):
        """Update performance monitoring"""
        self._last_monitoring_update = datetime.now()

        if not self._performance_monitor:
            return

        # Record system metrics
        if 'cpu_usage' in measurements:
            from .performance_monitor import MetricType
            self._performance_monitor.record_metric(
                MetricType.CPU_USAGE,
                measurements['cpu_usage'],
                'system',
                'percent'
            )

        if 'memory_usage' in measurements:
            from .performance_monitor import MetricType
            self._performance_monitor.record_metric(
                MetricType.MEMORY_USAGE,
                measurements['memory_usage'],
                'system',
                'MB'
            )

    def _map_scheduling_mode(self, mode_str: str) -> SystemMode:
        """Map scheduling mode string to system mode"""
        mode_map = {
            'NORMAL': SystemMode.NORMAL,
            'FLOOD_CONTROL': SystemMode.FLOOD_CONTROL,
            'DROUGHT': SystemMode.DROUGHT,
            'PRE_RELEASE': SystemMode.FLOOD_CONTROL,
            'EMERGENCY': SystemMode.EMERGENCY
        }
        return mode_map.get(mode_str, SystemMode.NORMAL)

    def _get_scenario_from_mode(self) -> str:
        """Get physical scenario from current system mode"""
        scenario_map = {
            SystemMode.NORMAL: 'NORMAL',
            SystemMode.FLOOD_CONTROL: 'FLOOD',
            SystemMode.DROUGHT: 'DROUGHT',
            SystemMode.EMERGENCY: 'FLOOD',
            SystemMode.DEGRADED: 'NORMAL',
            SystemMode.MAINTENANCE: 'NORMAL'
        }
        return scenario_map.get(self.mode, 'NORMAL')

    def _change_mode(self, new_mode: SystemMode):
        """Change system operating mode"""
        old_mode = self.mode
        self.mode = new_mode

        self._emit_event(
            EventType.MODE_CHANGE,
            "system",
            f"Mode changed from {old_mode.value} to {new_mode.value}",
            {'from_mode': old_mode.value, 'to_mode': new_mode.value}
        )

        logger.info(f"[UnifiedControlSystem] 模式切换: {old_mode.value} -> {new_mode.value}")

    def _process_commands(self):
        """Process pending commands"""
        with self._command_lock:
            if not self.command_queue:
                return

            # Sort by priority
            self.command_queue.sort(key=lambda c: c.priority)

            # Process each command
            processed = []
            for cmd in self.command_queue:
                try:
                    self._execute_command(cmd)
                    processed.append(cmd)
                except Exception as e:
                    logger.error(f"Command execution failed: {e}")

            # Remove processed commands
            for cmd in processed:
                self.command_queue.remove(cmd)

    def _execute_command(self, cmd: ControlCommand):
        """Execute a control command"""
        self._emit_event(
            EventType.COMMAND_RECEIVED,
            cmd.source,
            f"Command received: {cmd.command_type}",
            {'command_id': cmd.command_id, 'parameters': cmd.parameters}
        )

        if cmd.command_type == 'set_mode':
            mode_str = cmd.parameters.get('mode', 'NORMAL')
            new_mode = SystemMode(mode_str.lower())
            self._change_mode(new_mode)

        elif cmd.command_type == 'set_reference':
            if 'levels' in cmd.parameters:
                self.reference_levels = cmd.parameters['levels']

        elif cmd.command_type == 'emergency_stop':
            self._change_mode(SystemMode.EMERGENCY)

        elif cmd.command_type == 'reset':
            self.mode = SystemMode.NORMAL
            self.reference_levels = [3.0] * self.num_pools

    def submit_command(self, cmd: ControlCommand):
        """Submit a command for processing"""
        with self._command_lock:
            self.command_queue.append(cmd)

    def _emit_event(
        self,
        event_type: EventType,
        source: str,
        description: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Emit a system event"""
        event = SystemEvent(
            event_id=f"evt_{datetime.now().timestamp()}",
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            description=description,
            data=data or {}
        )

        self.event_history.append(event)

        # Keep history bounded
        if len(self.event_history) > 10000:
            self.event_history = self.event_history[-5000:]

        # Trigger callbacks
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")

    def register_event_callback(
        self,
        event_type: EventType,
        callback: Callable[[SystemEvent], None]
    ):
        """Register callback for event type"""
        self.event_callbacks[event_type].append(callback)

    def get_status(self) -> SystemStatus:
        """Get current system status"""
        uptime = datetime.now() - self.start_time

        health = 1.0
        active_alerts = 0
        if self._performance_monitor:
            health_metrics = self._performance_monitor.get_system_health()
            health = health_metrics.overall_health
            active_alerts = health_metrics.active_alerts

        return SystemStatus(
            mode=self.mode,
            health=health,
            active_faults=0,  # Would track from self-healing
            active_alerts=active_alerts,
            control_active=self._hierarchical_controller is not None,
            components_online=len(self._performance_monitor.registered_components) if self._performance_monitor else 0,
            components_total=3,  # hierarchical_mpc, self_healing, monitoring
            uptime=uptime,
            last_update=datetime.now(),
            current_levels=self.current_levels,
            current_flows=self.current_flows
        )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        status = self.get_status()

        data = {
            'system': {
                'mode': status.mode.value,
                'health': status.health,
                'uptime_hours': status.uptime.total_seconds() / 3600,
                'control_active': status.control_active
            },
            'state': {
                'levels': status.current_levels,
                'flows': status.current_flows,
                'references': self.reference_levels
            },
            'alerts': {
                'active': status.active_alerts,
                'faults': status.active_faults
            },
            'events': [
                {
                    'type': e.event_type.value,
                    'source': e.source,
                    'description': e.description,
                    'timestamp': e.timestamp.isoformat()
                }
                for e in self.event_history[-20:]
            ]
        }

        if self._performance_monitor:
            data['performance'] = self._performance_monitor.get_dashboard_data()

        return data

    def shutdown(self):
        """Shutdown the system"""
        self._change_mode(SystemMode.SHUTDOWN)
        self.is_running = False

        if self._performance_monitor:
            report = self._performance_monitor.generate_report()
            logger.info(f"[UnifiedControlSystem] 最终性能报告: {report.report_id}")

        logger.info("[UnifiedControlSystem] 系统已关闭")
