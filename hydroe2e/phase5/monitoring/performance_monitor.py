"""
Performance Monitor for Integrated Water Network Control System

Provides comprehensive performance monitoring including:
- Computational performance (solve times, memory usage)
- Control performance (tracking errors, stability)
- System health (module status, resource utilization)
- Real-time alerting and reporting
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import threading
import time
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    # Computational metrics
    SOLVE_TIME = "solve_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ITERATION_COUNT = "iteration_count"
    COMMUNICATION_LATENCY = "communication_latency"

    # Control metrics
    TRACKING_ERROR = "tracking_error"
    CONTROL_EFFORT = "control_effort"
    SETTLING_TIME = "settling_time"
    OVERSHOOT = "overshoot"
    STABILITY_MARGIN = "stability_margin"

    # System health
    MODULE_STATUS = "module_status"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"
    UPTIME = "uptime"
    THROUGHPUT = "throughput"


class MetricLevel(Enum):
    """Severity levels for metrics"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Single performance metric measurement"""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    component: str
    level: MetricLevel = MetricLevel.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert when thresholds exceeded"""
    alert_id: str
    metric_type: MetricType
    level: MetricLevel
    message: str
    current_value: float
    threshold: float
    component: str
    timestamp: datetime
    acknowledged: bool = False
    resolution: Optional[str] = None


@dataclass
class ComputationalMetrics:
    """Aggregated computational performance metrics"""
    avg_solve_time: float
    max_solve_time: float
    min_solve_time: float
    solve_time_std: float
    avg_iterations: float
    memory_usage_mb: float
    cpu_utilization: float
    communication_latency_ms: float
    timestamp: datetime


@dataclass
class ControlMetrics:
    """Aggregated control performance metrics"""
    avg_tracking_error: float
    max_tracking_error: float
    rms_error: float
    avg_control_effort: float
    max_control_effort: float
    settling_time: float
    overshoot_percent: float
    stability_index: float
    constraint_violations: int
    timestamp: datetime


@dataclass
class SystemHealthMetrics:
    """System health metrics"""
    overall_health: float  # 0-1
    active_modules: int
    total_modules: int
    active_alerts: int
    error_count: int
    uptime_hours: float
    throughput_per_sec: float
    queue_utilization: float
    timestamp: datetime


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    report_id: str
    period_start: datetime
    period_end: datetime
    computational: ComputationalMetrics
    control: ControlMetrics
    health: SystemHealthMetrics
    alerts: List[PerformanceAlert]
    recommendations: List[str]


class MetricBuffer:
    """Circular buffer for storing metric history"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, metric: PerformanceMetric):
        """Add metric to buffer"""
        with self._lock:
            self.buffer.append(metric)

    def get_recent(self, count: int = 100) -> List[PerformanceMetric]:
        """Get most recent metrics"""
        with self._lock:
            return list(self.buffer)[-count:]

    def get_by_type(self, metric_type: MetricType) -> List[PerformanceMetric]:
        """Get metrics of specific type"""
        with self._lock:
            return [m for m in self.buffer if m.metric_type == metric_type]

    def get_by_component(self, component: str) -> List[PerformanceMetric]:
        """Get metrics for specific component"""
        with self._lock:
            return [m for m in self.buffer if m.component == component]

    def get_statistics(self, metric_type: MetricType) -> Dict[str, float]:
        """Calculate statistics for a metric type"""
        metrics = self.get_by_type(metric_type)
        if not metrics:
            return {'count': 0}

        values = [m.value for m in metrics]
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }

    def clear(self):
        """Clear buffer"""
        with self._lock:
            self.buffer.clear()


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.

    Features:
    - Real-time metric collection
    - Threshold-based alerting
    - Historical analysis
    - Performance reporting
    - Trend detection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Metric buffers
        self.buffer_size = self.config.get('buffer_size', 10000)
        self.metric_buffers: Dict[str, MetricBuffer] = {}

        # Thresholds for alerting
        self.thresholds: Dict[MetricType, Dict[str, float]] = {
            MetricType.SOLVE_TIME: {'warning': 1.0, 'critical': 5.0},  # seconds
            MetricType.MEMORY_USAGE: {'warning': 500, 'critical': 1000},  # MB
            MetricType.CPU_USAGE: {'warning': 70, 'critical': 90},  # percent
            MetricType.TRACKING_ERROR: {'warning': 0.1, 'critical': 0.3},  # meters
            MetricType.COMMUNICATION_LATENCY: {'warning': 100, 'critical': 500},  # ms
            MetricType.ERROR_RATE: {'warning': 0.01, 'critical': 0.05},  # ratio
        }

        # Update custom thresholds
        if 'thresholds' in self.config:
            for metric, values in self.config['thresholds'].items():
                if isinstance(metric, str):
                    metric = MetricType(metric)
                self.thresholds[metric] = values

        # Alert management
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []

        # Monitoring state
        self.start_time = datetime.now()
        self.is_running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Component tracking
        self.registered_components: Dict[str, Dict[str, Any]] = {}

        # Statistics cache
        self._stats_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=5)

        logger.info("[PerformanceMonitor] 性能监控器初始化完成")

    def register_component(
        self,
        component_id: str,
        component_type: str,
        metrics: List[MetricType]
    ):
        """Register a component for monitoring"""
        self.registered_components[component_id] = {
            'type': component_type,
            'metrics': metrics,
            'registered_at': datetime.now(),
            'status': 'active'
        }

        # Create buffer for component
        self.metric_buffers[component_id] = MetricBuffer(self.buffer_size)
        logger.info(f"[PerformanceMonitor] 注册组件: {component_id} ({component_type})")

    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        component: str,
        unit: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric"""
        timestamp = datetime.now()

        # Determine level based on thresholds
        level = MetricLevel.NORMAL
        if metric_type in self.thresholds:
            thresholds = self.thresholds[metric_type]
            if value >= thresholds.get('critical', float('inf')):
                level = MetricLevel.CRITICAL
            elif value >= thresholds.get('warning', float('inf')):
                level = MetricLevel.WARNING

        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=timestamp,
            component=component,
            level=level,
            metadata=metadata or {}
        )

        # Store in buffer
        if component in self.metric_buffers:
            self.metric_buffers[component].add(metric)

        # Create global buffer if not exists
        if 'global' not in self.metric_buffers:
            self.metric_buffers['global'] = MetricBuffer(self.buffer_size)
        self.metric_buffers['global'].add(metric)

        # Check for alerts
        if level != MetricLevel.NORMAL:
            self._create_alert(metric)

        return metric

    def record_solve_time(
        self,
        component: str,
        solve_time: float,
        iterations: Optional[int] = None,
        success: bool = True
    ):
        """Convenience method to record MPC solve time"""
        metadata = {'success': success}
        if iterations is not None:
            metadata['iterations'] = iterations
            self.record_metric(
                MetricType.ITERATION_COUNT,
                iterations,
                component,
                'iterations',
                metadata
            )

        return self.record_metric(
            MetricType.SOLVE_TIME,
            solve_time,
            component,
            'seconds',
            metadata
        )

    def record_tracking_error(
        self,
        component: str,
        error: float,
        reference: float,
        actual: float
    ):
        """Record control tracking error"""
        return self.record_metric(
            MetricType.TRACKING_ERROR,
            error,
            component,
            'meters',
            {'reference': reference, 'actual': actual}
        )

    def record_control_effort(
        self,
        component: str,
        effort: float,
        control_type: str = "flow"
    ):
        """Record control effort"""
        return self.record_metric(
            MetricType.CONTROL_EFFORT,
            effort,
            component,
            'm3/s' if control_type == 'flow' else '',
            {'type': control_type}
        )

    def _create_alert(self, metric: PerformanceMetric):
        """Create alert from metric exceeding threshold"""
        alert_id = f"alert_{metric.component}_{metric.metric_type.value}_{metric.timestamp.timestamp()}"

        threshold = self.thresholds.get(metric.metric_type, {}).get(
            'warning' if metric.level == MetricLevel.WARNING else 'critical',
            0
        )

        alert = PerformanceAlert(
            alert_id=alert_id,
            metric_type=metric.metric_type,
            level=metric.level,
            message=f"{metric.metric_type.value} exceeded threshold for {metric.component}",
            current_value=metric.value,
            threshold=threshold,
            component=metric.component,
            timestamp=metric.timestamp
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(
            f"[PerformanceMonitor] 告警: {alert.message} "
            f"(当前值: {alert.current_value:.3f}, 阈值: {alert.threshold:.3f})"
        )

    def acknowledge_alert(self, alert_id: str, resolution: Optional[str] = None):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.resolution = resolution
            del self.active_alerts[alert_id]
            return True
        return False

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    def get_computational_metrics(
        self,
        component: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> ComputationalMetrics:
        """Get aggregated computational metrics"""
        buffer = self.metric_buffers.get(component or 'global')
        if not buffer:
            return ComputationalMetrics(
                avg_solve_time=0, max_solve_time=0, min_solve_time=0,
                solve_time_std=0, avg_iterations=0, memory_usage_mb=0,
                cpu_utilization=0, communication_latency_ms=0,
                timestamp=datetime.now()
            )

        # Get solve time stats
        solve_stats = buffer.get_statistics(MetricType.SOLVE_TIME)
        iter_stats = buffer.get_statistics(MetricType.ITERATION_COUNT)
        mem_stats = buffer.get_statistics(MetricType.MEMORY_USAGE)
        cpu_stats = buffer.get_statistics(MetricType.CPU_USAGE)
        latency_stats = buffer.get_statistics(MetricType.COMMUNICATION_LATENCY)

        return ComputationalMetrics(
            avg_solve_time=solve_stats.get('mean', 0),
            max_solve_time=solve_stats.get('max', 0),
            min_solve_time=solve_stats.get('min', 0),
            solve_time_std=solve_stats.get('std', 0),
            avg_iterations=iter_stats.get('mean', 0),
            memory_usage_mb=mem_stats.get('mean', 0),
            cpu_utilization=cpu_stats.get('mean', 0),
            communication_latency_ms=latency_stats.get('mean', 0),
            timestamp=datetime.now()
        )

    def get_control_metrics(
        self,
        component: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> ControlMetrics:
        """Get aggregated control metrics"""
        buffer = self.metric_buffers.get(component or 'global')
        if not buffer:
            return ControlMetrics(
                avg_tracking_error=0, max_tracking_error=0, rms_error=0,
                avg_control_effort=0, max_control_effort=0, settling_time=0,
                overshoot_percent=0, stability_index=1.0, constraint_violations=0,
                timestamp=datetime.now()
            )

        # Get tracking error stats
        error_stats = buffer.get_statistics(MetricType.TRACKING_ERROR)
        effort_stats = buffer.get_statistics(MetricType.CONTROL_EFFORT)

        # Calculate RMS error
        error_metrics = buffer.get_by_type(MetricType.TRACKING_ERROR)
        errors = [m.value for m in error_metrics]
        rms_error = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else 0

        return ControlMetrics(
            avg_tracking_error=error_stats.get('mean', 0),
            max_tracking_error=error_stats.get('max', 0),
            rms_error=rms_error,
            avg_control_effort=effort_stats.get('mean', 0),
            max_control_effort=effort_stats.get('max', 0),
            settling_time=0,  # Requires special calculation
            overshoot_percent=0,  # Requires special calculation
            stability_index=1.0,  # Placeholder
            constraint_violations=0,  # Would need to track
            timestamp=datetime.now()
        )

    def get_system_health(self) -> SystemHealthMetrics:
        """Get system health metrics"""
        now = datetime.now()
        uptime = (now - self.start_time).total_seconds() / 3600

        # Count active and total modules
        active_modules = sum(
            1 for c in self.registered_components.values()
            if c.get('status') == 'active'
        )
        total_modules = len(self.registered_components)

        # Count errors
        error_metrics = self.metric_buffers.get('global')
        error_count = 0
        if error_metrics:
            error_stats = error_metrics.get_statistics(MetricType.ERROR_RATE)
            error_count = int(error_stats.get('count', 0) * error_stats.get('mean', 0))

        # Calculate overall health
        health_factors = []
        if total_modules > 0:
            health_factors.append(active_modules / total_modules)
        if self.active_alerts:
            critical_alerts = sum(
                1 for a in self.active_alerts.values()
                if a.level == MetricLevel.CRITICAL
            )
            warning_alerts = sum(
                1 for a in self.active_alerts.values()
                if a.level == MetricLevel.WARNING
            )
            health_factors.append(max(0, 1 - critical_alerts * 0.3 - warning_alerts * 0.1))
        else:
            health_factors.append(1.0)

        overall_health = np.mean(health_factors) if health_factors else 1.0

        return SystemHealthMetrics(
            overall_health=overall_health,
            active_modules=active_modules,
            total_modules=total_modules,
            active_alerts=len(self.active_alerts),
            error_count=error_count,
            uptime_hours=uptime,
            throughput_per_sec=0,  # Would need to track
            queue_utilization=0,  # Would need to track
            timestamp=now
        )

    def generate_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> PerformanceReport:
        """Generate comprehensive performance report"""
        now = datetime.now()
        period_start = period_start or (now - timedelta(hours=1))
        period_end = period_end or now

        # Get metrics
        computational = self.get_computational_metrics()
        control = self.get_control_metrics()
        health = self.get_system_health()

        # Get alerts in period
        alerts = [
            a for a in self.alert_history
            if period_start <= a.timestamp <= period_end
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            computational, control, health, alerts
        )

        report_id = f"report_{now.strftime('%Y%m%d_%H%M%S')}"

        return PerformanceReport(
            report_id=report_id,
            period_start=period_start,
            period_end=period_end,
            computational=computational,
            control=control,
            health=health,
            alerts=alerts,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        computational: ComputationalMetrics,
        control: ControlMetrics,
        health: SystemHealthMetrics,
        alerts: List[PerformanceAlert]
    ) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []

        # Computational recommendations
        if computational.avg_solve_time > 1.0:
            recommendations.append(
                "考虑减少MPC预测时域或增加求解器容差以降低求解时间"
            )

        if computational.memory_usage_mb > 500:
            recommendations.append(
                "内存使用较高,建议检查是否存在内存泄漏或优化数据结构"
            )

        # Control recommendations
        if control.avg_tracking_error > 0.1:
            recommendations.append(
                "平均跟踪误差较大,建议调整MPC权重矩阵或减小采样周期"
            )

        if control.max_tracking_error > 0.3:
            recommendations.append(
                "峰值跟踪误差过大,检查是否存在扰动或模型失配"
            )

        # Health recommendations
        if health.overall_health < 0.8:
            recommendations.append(
                "系统健康度较低,请检查告警并处理异常情况"
            )

        if health.active_alerts > 5:
            recommendations.append(
                f"存在{health.active_alerts}个活跃告警,建议优先处理CRITICAL级别告警"
            )

        # Alert-based recommendations
        solve_time_alerts = [
            a for a in alerts if a.metric_type == MetricType.SOLVE_TIME
        ]
        if len(solve_time_alerts) > 10:
            recommendations.append(
                "频繁出现求解时间告警,考虑优化MPC参数或升级硬件"
            )

        return recommendations

    def get_metric_trend(
        self,
        metric_type: MetricType,
        component: Optional[str] = None,
        window_size: int = 100
    ) -> Dict[str, Any]:
        """Analyze trend for a metric"""
        buffer = self.metric_buffers.get(component or 'global')
        if not buffer:
            return {'trend': 'unknown', 'slope': 0, 'r_squared': 0}

        metrics = buffer.get_by_type(metric_type)[-window_size:]
        if len(metrics) < 10:
            return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}

        # Simple linear regression
        x = np.arange(len(metrics))
        y = np.array([m.value for m in metrics])

        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x ** 2) - np.sum(x) ** 2 + 1e-10
        )

        # Calculate R-squared
        y_pred = slope * x + (np.mean(y) - slope * np.mean(x))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
        r_squared = 1 - ss_res / ss_tot

        # Determine trend
        if abs(slope) < 0.001:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_squared,
            'recent_mean': np.mean(y[-10:]),
            'overall_mean': np.mean(y)
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        computational = self.get_computational_metrics()
        control = self.get_control_metrics()
        health = self.get_system_health()

        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': health.uptime_hours,
            'overall_health': health.overall_health,
            'computational': {
                'avg_solve_time_ms': computational.avg_solve_time * 1000,
                'max_solve_time_ms': computational.max_solve_time * 1000,
                'avg_iterations': computational.avg_iterations,
                'memory_mb': computational.memory_usage_mb,
                'cpu_percent': computational.cpu_utilization
            },
            'control': {
                'avg_tracking_error': control.avg_tracking_error,
                'max_tracking_error': control.max_tracking_error,
                'rms_error': control.rms_error,
                'avg_control_effort': control.avg_control_effort,
                'stability_index': control.stability_index
            },
            'health': {
                'active_modules': health.active_modules,
                'total_modules': health.total_modules,
                'active_alerts': health.active_alerts,
                'error_count': health.error_count
            },
            'alerts': [
                {
                    'id': a.alert_id,
                    'level': a.level.value,
                    'message': a.message,
                    'component': a.component,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in list(self.active_alerts.values())[:10]  # Latest 10
            ],
            'components': {
                cid: {
                    'type': info['type'],
                    'status': info['status']
                }
                for cid, info in self.registered_components.items()
            }
        }

    def export_metrics(
        self,
        format: str = 'json',
        metric_types: Optional[List[MetricType]] = None
    ) -> Any:
        """Export metrics for external analysis"""
        buffer = self.metric_buffers.get('global')
        if not buffer:
            return [] if format == 'json' else ""

        metrics = buffer.get_recent(1000)

        if metric_types:
            metrics = [m for m in metrics if m.metric_type in metric_types]

        if format == 'json':
            return [
                {
                    'type': m.metric_type.value,
                    'value': m.value,
                    'unit': m.unit,
                    'component': m.component,
                    'level': m.level.value,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in metrics
            ]
        elif format == 'csv':
            lines = ['type,value,unit,component,level,timestamp']
            for m in metrics:
                lines.append(
                    f"{m.metric_type.value},{m.value},{m.unit},"
                    f"{m.component},{m.level.value},{m.timestamp.isoformat()}"
                )
            return '\n'.join(lines)

        return metrics

    def reset(self):
        """Reset all metrics and alerts"""
        for buffer in self.metric_buffers.values():
            buffer.clear()
        self.active_alerts.clear()
        self.alert_history.clear()
        self.start_time = datetime.now()
        logger.info("[PerformanceMonitor] 性能监控器已重置")
