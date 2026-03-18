"""
API Interface for Unified Control System

Provides a clean API for external systems to interact with the control system:
- Status queries
- Command submission
- Configuration updates
- Data export
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """API command types"""
    # System commands
    GET_STATUS = "get_status"
    GET_DASHBOARD = "get_dashboard"
    SET_MODE = "set_mode"
    RESET = "reset"
    SHUTDOWN = "shutdown"

    # Control commands
    SET_REFERENCE = "set_reference"
    SET_PARAMETERS = "set_parameters"
    OVERRIDE_CONTROL = "override_control"
    EMERGENCY_STOP = "emergency_stop"

    # Data commands
    GET_HISTORY = "get_history"
    GET_METRICS = "get_metrics"
    EXPORT_DATA = "export_data"

    # Configuration commands
    GET_CONFIG = "get_config"
    SET_CONFIG = "set_config"
    GET_THRESHOLDS = "get_thresholds"
    SET_THRESHOLDS = "set_thresholds"

    # Alert commands
    GET_ALERTS = "get_alerts"
    ACKNOWLEDGE_ALERT = "acknowledge_alert"
    CLEAR_ALERTS = "clear_alerts"


class ResponseStatus(Enum):
    """API response status"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"
    INVALID = "invalid"


@dataclass
class APIRequest:
    """API request structure"""
    request_id: str
    command: CommandType
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "api"
    auth_token: Optional[str] = None


@dataclass
class APIResponse:
    """API response structure"""
    request_id: str
    status: ResponseStatus
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0


class SystemAPI:
    """
    API interface for the unified control system.

    Provides methods for:
    - Querying system status
    - Submitting control commands
    - Managing alerts
    - Exporting data
    """

    def __init__(self, system=None):
        """
        Initialize API interface.

        Args:
            system: UnifiedControlSystem instance (optional, can be set later)
        """
        self._system = system
        self._request_log: List[Dict[str, Any]] = []
        self._max_log_size = 1000

        logger.info("[SystemAPI] API接口初始化完成")

    def set_system(self, system):
        """Set the system instance"""
        self._system = system

    def process_request(self, request: APIRequest) -> APIResponse:
        """
        Process an API request.

        Args:
            request: API request to process

        Returns:
            API response
        """
        start_time = datetime.now()

        # Log request
        self._log_request(request)

        # Validate request
        if not self._validate_request(request):
            return APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.INVALID,
                message="Invalid request"
            )

        # Process by command type
        try:
            response = self._dispatch_command(request)
        except Exception as e:
            logger.error(f"[SystemAPI] 请求处理失败: {e}")
            response = APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.ERROR,
                message=str(e)
            )

        # Calculate execution time
        response.execution_time_ms = (
            datetime.now() - start_time
        ).total_seconds() * 1000

        return response

    def _validate_request(self, request: APIRequest) -> bool:
        """Validate API request"""
        if not request.request_id:
            return False
        if not isinstance(request.command, CommandType):
            return False
        return True

    def _dispatch_command(self, request: APIRequest) -> APIResponse:
        """Dispatch command to appropriate handler"""
        handlers = {
            # System commands
            CommandType.GET_STATUS: self._handle_get_status,
            CommandType.GET_DASHBOARD: self._handle_get_dashboard,
            CommandType.SET_MODE: self._handle_set_mode,
            CommandType.RESET: self._handle_reset,
            CommandType.SHUTDOWN: self._handle_shutdown,

            # Control commands
            CommandType.SET_REFERENCE: self._handle_set_reference,
            CommandType.SET_PARAMETERS: self._handle_set_parameters,
            CommandType.OVERRIDE_CONTROL: self._handle_override_control,
            CommandType.EMERGENCY_STOP: self._handle_emergency_stop,

            # Data commands
            CommandType.GET_HISTORY: self._handle_get_history,
            CommandType.GET_METRICS: self._handle_get_metrics,
            CommandType.EXPORT_DATA: self._handle_export_data,

            # Configuration commands
            CommandType.GET_CONFIG: self._handle_get_config,
            CommandType.SET_CONFIG: self._handle_set_config,
            CommandType.GET_THRESHOLDS: self._handle_get_thresholds,
            CommandType.SET_THRESHOLDS: self._handle_set_thresholds,

            # Alert commands
            CommandType.GET_ALERTS: self._handle_get_alerts,
            CommandType.ACKNOWLEDGE_ALERT: self._handle_acknowledge_alert,
            CommandType.CLEAR_ALERTS: self._handle_clear_alerts,
        }

        handler = handlers.get(request.command)
        if not handler:
            return APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.NOT_FOUND,
                message=f"Unknown command: {request.command}"
            )

        return handler(request)

    # System command handlers

    def _handle_get_status(self, request: APIRequest) -> APIResponse:
        """Handle get status request"""
        if not self._system:
            return self._no_system_response(request)

        status = self._system.get_status()

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={
                'mode': status.mode.value,
                'health': status.health,
                'active_faults': status.active_faults,
                'active_alerts': status.active_alerts,
                'control_active': status.control_active,
                'uptime_seconds': status.uptime.total_seconds(),
                'current_levels': status.current_levels,
                'current_flows': status.current_flows
            }
        )

    def _handle_get_dashboard(self, request: APIRequest) -> APIResponse:
        """Handle get dashboard request"""
        if not self._system:
            return self._no_system_response(request)

        dashboard_data = self._system.get_dashboard_data()

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data=dashboard_data
        )

    def _handle_set_mode(self, request: APIRequest) -> APIResponse:
        """Handle set mode request"""
        if not self._system:
            return self._no_system_response(request)

        mode = request.parameters.get('mode')
        if not mode:
            return APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.INVALID,
                message="Missing 'mode' parameter"
            )

        from .unified_system import ControlCommand
        cmd = ControlCommand(
            command_id=f"cmd_{request.request_id}",
            command_type='set_mode',
            target='system',
            parameters={'mode': mode},
            timestamp=datetime.now(),
            source=request.source
        )
        self._system.submit_command(cmd)

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={'mode': mode},
            message=f"Mode change to {mode} queued"
        )

    def _handle_reset(self, request: APIRequest) -> APIResponse:
        """Handle reset request"""
        if not self._system:
            return self._no_system_response(request)

        from .unified_system import ControlCommand
        cmd = ControlCommand(
            command_id=f"cmd_{request.request_id}",
            command_type='reset',
            target='system',
            parameters={},
            timestamp=datetime.now(),
            source=request.source,
            priority=1
        )
        self._system.submit_command(cmd)

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            message="System reset queued"
        )

    def _handle_shutdown(self, request: APIRequest) -> APIResponse:
        """Handle shutdown request"""
        if not self._system:
            return self._no_system_response(request)

        self._system.shutdown()

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            message="System shutdown initiated"
        )

    # Control command handlers

    def _handle_set_reference(self, request: APIRequest) -> APIResponse:
        """Handle set reference request"""
        if not self._system:
            return self._no_system_response(request)

        levels = request.parameters.get('levels')
        if not levels:
            return APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.INVALID,
                message="Missing 'levels' parameter"
            )

        from .unified_system import ControlCommand
        cmd = ControlCommand(
            command_id=f"cmd_{request.request_id}",
            command_type='set_reference',
            target='controller',
            parameters={'levels': levels},
            timestamp=datetime.now(),
            source=request.source
        )
        self._system.submit_command(cmd)

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={'levels': levels},
            message="Reference levels update queued"
        )

    def _handle_set_parameters(self, request: APIRequest) -> APIResponse:
        """Handle set parameters request"""
        if not self._system:
            return self._no_system_response(request)

        params = request.parameters
        # Would update controller parameters

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={'parameters': params},
            message="Parameters updated"
        )

    def _handle_override_control(self, request: APIRequest) -> APIResponse:
        """Handle override control request"""
        if not self._system:
            return self._no_system_response(request)

        override = request.parameters.get('override', {})

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={'override': override},
            message="Control override applied"
        )

    def _handle_emergency_stop(self, request: APIRequest) -> APIResponse:
        """Handle emergency stop request"""
        if not self._system:
            return self._no_system_response(request)

        from .unified_system import ControlCommand
        cmd = ControlCommand(
            command_id=f"cmd_{request.request_id}",
            command_type='emergency_stop',
            target='system',
            parameters={},
            timestamp=datetime.now(),
            source=request.source,
            priority=1  # Highest priority
        )
        self._system.submit_command(cmd)

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            message="Emergency stop initiated"
        )

    # Data command handlers

    def _handle_get_history(self, request: APIRequest) -> APIResponse:
        """Handle get history request"""
        if not self._system:
            return self._no_system_response(request)

        count = request.parameters.get('count', 100)
        event_type = request.parameters.get('type')

        events = self._system.event_history[-count:]

        if event_type:
            from .unified_system import EventType
            try:
                filter_type = EventType(event_type)
                events = [e for e in events if e.event_type == filter_type]
            except ValueError:
                pass

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={
                'events': [
                    {
                        'id': e.event_id,
                        'type': e.event_type.value,
                        'source': e.source,
                        'description': e.description,
                        'timestamp': e.timestamp.isoformat()
                    }
                    for e in events
                ],
                'count': len(events)
            }
        )

    def _handle_get_metrics(self, request: APIRequest) -> APIResponse:
        """Handle get metrics request"""
        if not self._system:
            return self._no_system_response(request)

        if not self._system._performance_monitor:
            return APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.ERROR,
                message="Performance monitor not available"
            )

        metric_type = request.parameters.get('type')
        component = request.parameters.get('component')

        if metric_type == 'computational':
            metrics = self._system._performance_monitor.get_computational_metrics(component)
            data = {
                'avg_solve_time': metrics.avg_solve_time,
                'max_solve_time': metrics.max_solve_time,
                'avg_iterations': metrics.avg_iterations,
                'memory_mb': metrics.memory_usage_mb,
                'cpu_percent': metrics.cpu_utilization
            }
        elif metric_type == 'control':
            metrics = self._system._performance_monitor.get_control_metrics(component)
            data = {
                'avg_tracking_error': metrics.avg_tracking_error,
                'max_tracking_error': metrics.max_tracking_error,
                'rms_error': metrics.rms_error,
                'avg_control_effort': metrics.avg_control_effort
            }
        else:
            # Return dashboard data
            data = self._system._performance_monitor.get_dashboard_data()

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data=data
        )

    def _handle_export_data(self, request: APIRequest) -> APIResponse:
        """Handle export data request"""
        if not self._system:
            return self._no_system_response(request)

        if not self._system._performance_monitor:
            return APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.ERROR,
                message="Performance monitor not available"
            )

        format = request.parameters.get('format', 'json')
        exported = self._system._performance_monitor.export_metrics(format)

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={'metrics': exported, 'format': format}
        )

    # Configuration command handlers

    def _handle_get_config(self, request: APIRequest) -> APIResponse:
        """Handle get configuration request"""
        if not self._system:
            return self._no_system_response(request)

        config = {
            'num_pools': self._system.num_pools,
            'l3_interval': self._system.l3_interval,
            'l2_interval': self._system.l2_interval,
            'monitoring_interval': self._system.monitoring_interval
        }

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={'config': config}
        )

    def _handle_set_config(self, request: APIRequest) -> APIResponse:
        """Handle set configuration request"""
        if not self._system:
            return self._no_system_response(request)

        # Would update configuration
        # For now, just return success
        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            message="Configuration update not yet implemented"
        )

    def _handle_get_thresholds(self, request: APIRequest) -> APIResponse:
        """Handle get thresholds request"""
        if not self._system or not self._system._performance_monitor:
            return self._no_system_response(request)

        thresholds = {
            metric_type.value: thresholds
            for metric_type, thresholds in self._system._performance_monitor.thresholds.items()
        }

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={'thresholds': thresholds}
        )

    def _handle_set_thresholds(self, request: APIRequest) -> APIResponse:
        """Handle set thresholds request"""
        if not self._system or not self._system._performance_monitor:
            return self._no_system_response(request)

        new_thresholds = request.parameters.get('thresholds', {})

        for metric_name, values in new_thresholds.items():
            try:
                from .performance_monitor import MetricType
                metric_type = MetricType(metric_name)
                self._system._performance_monitor.thresholds[metric_type] = values
            except ValueError:
                pass

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            message="Thresholds updated"
        )

    # Alert command handlers

    def _handle_get_alerts(self, request: APIRequest) -> APIResponse:
        """Handle get alerts request"""
        if not self._system or not self._system._performance_monitor:
            return self._no_system_response(request)

        active_only = request.parameters.get('active_only', True)

        if active_only:
            alerts = list(self._system._performance_monitor.active_alerts.values())
        else:
            alerts = self._system._performance_monitor.alert_history[-100:]

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={
                'alerts': [
                    {
                        'id': a.alert_id,
                        'type': a.metric_type.value,
                        'level': a.level.value,
                        'message': a.message,
                        'component': a.component,
                        'value': a.current_value,
                        'threshold': a.threshold,
                        'timestamp': a.timestamp.isoformat(),
                        'acknowledged': a.acknowledged
                    }
                    for a in alerts
                ],
                'count': len(alerts)
            }
        )

    def _handle_acknowledge_alert(self, request: APIRequest) -> APIResponse:
        """Handle acknowledge alert request"""
        if not self._system or not self._system._performance_monitor:
            return self._no_system_response(request)

        alert_id = request.parameters.get('alert_id')
        resolution = request.parameters.get('resolution')

        if not alert_id:
            return APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.INVALID,
                message="Missing 'alert_id' parameter"
            )

        success = self._system._performance_monitor.acknowledge_alert(
            alert_id, resolution
        )

        if success:
            return APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.SUCCESS,
                message=f"Alert {alert_id} acknowledged"
            )
        else:
            return APIResponse(
                request_id=request.request_id,
                status=ResponseStatus.NOT_FOUND,
                message=f"Alert {alert_id} not found"
            )

    def _handle_clear_alerts(self, request: APIRequest) -> APIResponse:
        """Handle clear alerts request"""
        if not self._system or not self._system._performance_monitor:
            return self._no_system_response(request)

        count = len(self._system._performance_monitor.active_alerts)
        self._system._performance_monitor.active_alerts.clear()

        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.SUCCESS,
            data={'cleared_count': count},
            message=f"Cleared {count} alerts"
        )

    # Helper methods

    def _no_system_response(self, request: APIRequest) -> APIResponse:
        """Response when no system is connected"""
        return APIResponse(
            request_id=request.request_id,
            status=ResponseStatus.ERROR,
            message="No system connected"
        )

    def _log_request(self, request: APIRequest):
        """Log API request"""
        self._request_log.append({
            'request_id': request.request_id,
            'command': request.command.value,
            'source': request.source,
            'timestamp': request.timestamp.isoformat()
        })

        # Trim log
        if len(self._request_log) > self._max_log_size:
            self._request_log = self._request_log[-self._max_log_size // 2:]

    def get_request_log(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent API request log"""
        return self._request_log[-count:]

    # Convenience methods for common operations

    def get_status(self) -> Dict[str, Any]:
        """Convenience method to get system status"""
        request = APIRequest(
            request_id=f"api_{datetime.now().timestamp()}",
            command=CommandType.GET_STATUS
        )
        response = self.process_request(request)
        return response.data if response.status == ResponseStatus.SUCCESS else {}

    def get_dashboard(self) -> Dict[str, Any]:
        """Convenience method to get dashboard data"""
        request = APIRequest(
            request_id=f"api_{datetime.now().timestamp()}",
            command=CommandType.GET_DASHBOARD
        )
        response = self.process_request(request)
        return response.data if response.status == ResponseStatus.SUCCESS else {}

    def set_mode(self, mode: str) -> bool:
        """Convenience method to set system mode"""
        request = APIRequest(
            request_id=f"api_{datetime.now().timestamp()}",
            command=CommandType.SET_MODE,
            parameters={'mode': mode}
        )
        response = self.process_request(request)
        return response.status == ResponseStatus.SUCCESS

    def emergency_stop(self) -> bool:
        """Convenience method for emergency stop"""
        request = APIRequest(
            request_id=f"api_{datetime.now().timestamp()}",
            command=CommandType.EMERGENCY_STOP
        )
        response = self.process_request(request)
        return response.status == ResponseStatus.SUCCESS
