"""
Tests for Phase 5.5 Performance Monitoring and System Integration

Tests:
1. Performance Monitor
2. Unified Control System
3. API Interface
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time


class TestPerformanceMonitor:
    """Tests for performance monitoring"""

    def test_monitor_initialization(self):
        """Test monitor initializes correctly"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        assert monitor is not None
        assert monitor.buffer_size == 10000

    def test_metric_types(self):
        """Test metric type enumeration"""
        from phase5.monitoring.performance_monitor import MetricType

        assert hasattr(MetricType, 'SOLVE_TIME')
        assert hasattr(MetricType, 'TRACKING_ERROR')
        assert hasattr(MetricType, 'MEMORY_USAGE')
        assert hasattr(MetricType, 'CPU_USAGE')

    def test_metric_levels(self):
        """Test metric level enumeration"""
        from phase5.monitoring.performance_monitor import MetricLevel

        assert hasattr(MetricLevel, 'NORMAL')
        assert hasattr(MetricLevel, 'WARNING')
        assert hasattr(MetricLevel, 'CRITICAL')

    def test_register_component(self):
        """Test component registration"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('test_component', 'controller', [MetricType.SOLVE_TIME])

        assert 'test_component' in monitor.registered_components
        assert 'test_component' in monitor.metric_buffers

    def test_record_metric(self):
        """Test recording metrics"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        metric = monitor.record_metric(
            MetricType.SOLVE_TIME,
            0.5,
            'test',
            'seconds'
        )

        assert metric is not None
        assert metric.value == 0.5
        assert metric.component == 'test'

    def test_record_solve_time(self):
        """Test solve time recording"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        monitor.register_component('mpc', 'controller', [])

        metric = monitor.record_solve_time('mpc', 0.123, iterations=10, success=True)

        assert metric.value == 0.123
        assert metric.metadata.get('iterations') == 10

    def test_record_tracking_error(self):
        """Test tracking error recording"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        monitor.register_component('pool_0', 'pool', [])

        metric = monitor.record_tracking_error('pool_0', 0.05, reference=3.0, actual=2.95)

        assert metric.value == 0.05
        assert metric.metadata.get('reference') == 3.0

    def test_threshold_alerting(self):
        """Test alert generation when threshold exceeded"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType, MetricLevel

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        # Record metric above warning threshold (default: 1.0 for solve_time)
        monitor.record_metric(MetricType.SOLVE_TIME, 1.5, 'test', 'seconds')

        assert len(monitor.active_alerts) > 0

    def test_alert_callback(self):
        """Test alert callback notification"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        callback_called = [False]
        def alert_callback(alert):
            callback_called[0] = True

        monitor.add_alert_callback(alert_callback)

        # Trigger alert
        monitor.record_metric(MetricType.SOLVE_TIME, 10.0, 'test', 'seconds')

        assert callback_called[0]

    def test_acknowledge_alert(self):
        """Test alert acknowledgment"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        # Generate alert
        monitor.record_metric(MetricType.SOLVE_TIME, 10.0, 'test', 'seconds')
        alert_id = list(monitor.active_alerts.keys())[0]

        # Acknowledge
        success = monitor.acknowledge_alert(alert_id, "Fixed")
        assert success
        assert alert_id not in monitor.active_alerts

    def test_computational_metrics(self):
        """Test computational metrics aggregation"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('mpc', 'controller', [])

        # Record multiple solve times
        for i in range(10):
            monitor.record_metric(MetricType.SOLVE_TIME, 0.1 * (i + 1), 'mpc', 'seconds')

        metrics = monitor.get_computational_metrics('mpc')

        assert metrics.avg_solve_time > 0
        assert metrics.max_solve_time == 1.0
        assert metrics.min_solve_time == 0.1

    def test_control_metrics(self):
        """Test control metrics aggregation"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('pool_0', 'pool', [])

        # Record tracking errors
        for i in range(10):
            monitor.record_metric(MetricType.TRACKING_ERROR, 0.01 * (i + 1), 'pool_0', 'meters')

        metrics = monitor.get_control_metrics('pool_0')

        assert metrics.avg_tracking_error > 0
        assert metrics.rms_error > 0

    def test_system_health(self):
        """Test system health calculation"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        monitor.register_component('comp1', 'test', [])
        monitor.register_component('comp2', 'test', [])

        health = monitor.get_system_health()

        assert health.overall_health >= 0
        assert health.overall_health <= 1
        assert health.total_modules == 2

    def test_performance_report(self):
        """Test performance report generation"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        # Add some data
        for i in range(10):
            monitor.record_metric(MetricType.SOLVE_TIME, 0.1, 'test', 'seconds')
            monitor.record_metric(MetricType.TRACKING_ERROR, 0.01, 'test', 'meters')

        report = monitor.generate_report()

        assert report.report_id is not None
        assert report.computational is not None
        assert report.control is not None
        assert report.health is not None

    def test_metric_trend(self):
        """Test metric trend analysis"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        # Add increasing values
        for i in range(20):
            monitor.record_metric(MetricType.SOLVE_TIME, 0.1 + i * 0.01, 'test', 'seconds')

        trend = monitor.get_metric_trend(MetricType.SOLVE_TIME)

        assert trend['trend'] == 'increasing'
        assert trend['slope'] > 0

    def test_dashboard_data(self):
        """Test dashboard data generation"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        data = monitor.get_dashboard_data()

        assert 'timestamp' in data
        assert 'overall_health' in data
        assert 'computational' in data
        assert 'health' in data

    def test_export_metrics_json(self):
        """Test metrics export to JSON"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        for i in range(5):
            monitor.record_metric(MetricType.SOLVE_TIME, 0.1, 'test', 'seconds')

        exported = monitor.export_metrics(format='json')

        assert isinstance(exported, list)
        assert len(exported) > 0

    def test_export_metrics_csv(self):
        """Test metrics export to CSV"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        for i in range(5):
            monitor.record_metric(MetricType.SOLVE_TIME, 0.1, 'test', 'seconds')

        exported = monitor.export_metrics(format='csv')

        assert isinstance(exported, str)
        assert 'solve_time' in exported

    def test_reset(self):
        """Test monitor reset"""
        from phase5.monitoring.performance_monitor import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        monitor.record_metric(MetricType.SOLVE_TIME, 10.0, 'test', 'seconds')
        assert len(monitor.active_alerts) > 0

        monitor.reset()

        assert len(monitor.active_alerts) == 0


class TestUnifiedControlSystem:
    """Tests for unified control system"""

    def test_system_initialization(self):
        """Test system initializes correctly"""
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        assert system is not None
        assert system.num_pools == 3

    def test_system_modes(self):
        """Test system mode enumeration"""
        from phase5.monitoring.unified_system import SystemMode

        assert hasattr(SystemMode, 'NORMAL')
        assert hasattr(SystemMode, 'FLOOD_CONTROL')
        assert hasattr(SystemMode, 'EMERGENCY')
        assert hasattr(SystemMode, 'SHUTDOWN')

    def test_event_types(self):
        """Test event type enumeration"""
        from phase5.monitoring.unified_system import EventType

        assert hasattr(EventType, 'MODE_CHANGE')
        assert hasattr(EventType, 'FAULT_DETECTED')
        assert hasattr(EventType, 'CONTROL_UPDATE')

    def test_system_initialize(self):
        """Test system component initialization"""
        from phase5.monitoring.unified_system import UnifiedControlSystem, SystemMode

        system = UnifiedControlSystem({'num_pools': 3})
        success = system.initialize()

        assert success
        assert system.mode == SystemMode.NORMAL

    def test_system_step(self):
        """Test system control step"""
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        measurements = {
            'levels': [3.0, 2.8, 2.5],
            'flows': [10.0, 8.0, 6.0]
        }

        result = system.step(measurements, dt=60.0)

        assert 'mode' in result
        assert 'control_actions' in result
        assert 'health' in result

    def test_command_submission(self):
        """Test command submission"""
        from phase5.monitoring.unified_system import (
            UnifiedControlSystem, ControlCommand
        )

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        cmd = ControlCommand(
            command_id='cmd_001',
            command_type='set_reference',
            target='controller',
            parameters={'levels': [3.5, 3.5, 3.5]},
            timestamp=datetime.now()
        )

        system.submit_command(cmd)
        assert len(system.command_queue) == 1

    def test_mode_change(self):
        """Test mode change"""
        from phase5.monitoring.unified_system import (
            UnifiedControlSystem, SystemMode, EventType
        )

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        events_received = []
        def event_callback(event):
            events_received.append(event)

        system.register_event_callback(EventType.MODE_CHANGE, event_callback)

        system._change_mode(SystemMode.FLOOD_CONTROL)

        assert system.mode == SystemMode.FLOOD_CONTROL
        assert len(events_received) == 1

    def test_get_status(self):
        """Test status retrieval"""
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        status = system.get_status()

        assert hasattr(status, 'mode')
        assert hasattr(status, 'health')
        assert hasattr(status, 'current_levels')
        assert len(status.current_levels) == 3

    def test_get_dashboard_data(self):
        """Test dashboard data retrieval"""
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        data = system.get_dashboard_data()

        assert 'system' in data
        assert 'state' in data
        assert 'events' in data

    def test_shutdown(self):
        """Test system shutdown"""
        from phase5.monitoring.unified_system import UnifiedControlSystem, SystemMode

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        system.shutdown()

        assert system.mode == SystemMode.SHUTDOWN


class TestAPIInterface:
    """Tests for API interface"""

    def test_api_initialization(self):
        """Test API initializes correctly"""
        from phase5.monitoring.api_interface import SystemAPI

        api = SystemAPI()
        assert api is not None

    def test_command_types(self):
        """Test command type enumeration"""
        from phase5.monitoring.api_interface import CommandType

        assert hasattr(CommandType, 'GET_STATUS')
        assert hasattr(CommandType, 'SET_MODE')
        assert hasattr(CommandType, 'EMERGENCY_STOP')
        assert hasattr(CommandType, 'GET_METRICS')

    def test_response_status(self):
        """Test response status enumeration"""
        from phase5.monitoring.api_interface import ResponseStatus

        assert hasattr(ResponseStatus, 'SUCCESS')
        assert hasattr(ResponseStatus, 'ERROR')
        assert hasattr(ResponseStatus, 'INVALID')

    def test_api_request_structure(self):
        """Test API request structure"""
        from phase5.monitoring.api_interface import APIRequest, CommandType

        request = APIRequest(
            request_id='req_001',
            command=CommandType.GET_STATUS,
            parameters={'include_metrics': True}
        )

        assert request.request_id == 'req_001'
        assert request.command == CommandType.GET_STATUS

    def test_process_request_no_system(self):
        """Test request processing without system"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType, ResponseStatus
        )

        api = SystemAPI()

        request = APIRequest(
            request_id='req_001',
            command=CommandType.GET_STATUS
        )

        response = api.process_request(request)

        assert response.status == ResponseStatus.ERROR
        assert 'No system' in response.message

    def test_process_get_status(self):
        """Test GET_STATUS request"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType, ResponseStatus
        )
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        request = APIRequest(
            request_id='req_001',
            command=CommandType.GET_STATUS
        )

        response = api.process_request(request)

        assert response.status == ResponseStatus.SUCCESS
        assert 'mode' in response.data
        assert 'health' in response.data

    def test_process_get_dashboard(self):
        """Test GET_DASHBOARD request"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType, ResponseStatus
        )
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        request = APIRequest(
            request_id='req_001',
            command=CommandType.GET_DASHBOARD
        )

        response = api.process_request(request)

        assert response.status == ResponseStatus.SUCCESS
        assert 'system' in response.data

    def test_process_set_mode(self):
        """Test SET_MODE request"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType, ResponseStatus
        )
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        request = APIRequest(
            request_id='req_001',
            command=CommandType.SET_MODE,
            parameters={'mode': 'flood_control'}
        )

        response = api.process_request(request)

        assert response.status == ResponseStatus.SUCCESS
        assert len(system.command_queue) == 1

    def test_process_emergency_stop(self):
        """Test EMERGENCY_STOP request"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType, ResponseStatus
        )
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        request = APIRequest(
            request_id='req_001',
            command=CommandType.EMERGENCY_STOP
        )

        response = api.process_request(request)

        assert response.status == ResponseStatus.SUCCESS

    def test_process_set_reference(self):
        """Test SET_REFERENCE request"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType, ResponseStatus
        )
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        request = APIRequest(
            request_id='req_001',
            command=CommandType.SET_REFERENCE,
            parameters={'levels': [3.5, 3.5, 3.5]}
        )

        response = api.process_request(request)

        assert response.status == ResponseStatus.SUCCESS

    def test_process_get_history(self):
        """Test GET_HISTORY request"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType, ResponseStatus
        )
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        request = APIRequest(
            request_id='req_001',
            command=CommandType.GET_HISTORY,
            parameters={'count': 50}
        )

        response = api.process_request(request)

        assert response.status == ResponseStatus.SUCCESS
        assert 'events' in response.data

    def test_process_get_metrics(self):
        """Test GET_METRICS request"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType, ResponseStatus
        )
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        request = APIRequest(
            request_id='req_001',
            command=CommandType.GET_METRICS,
            parameters={'type': 'computational'}
        )

        response = api.process_request(request)

        assert response.status == ResponseStatus.SUCCESS

    def test_process_get_alerts(self):
        """Test GET_ALERTS request"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType, ResponseStatus
        )
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        request = APIRequest(
            request_id='req_001',
            command=CommandType.GET_ALERTS
        )

        response = api.process_request(request)

        assert response.status == ResponseStatus.SUCCESS
        assert 'alerts' in response.data

    def test_convenience_methods(self):
        """Test convenience methods"""
        from phase5.monitoring.api_interface import SystemAPI
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        # Test get_status
        status = api.get_status()
        assert 'mode' in status

        # Test get_dashboard
        dashboard = api.get_dashboard()
        assert 'system' in dashboard

        # Test set_mode
        success = api.set_mode('normal')
        assert success

    def test_request_logging(self):
        """Test request logging"""
        from phase5.monitoring.api_interface import (
            SystemAPI, APIRequest, CommandType
        )
        from phase5.monitoring.unified_system import UnifiedControlSystem

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        api = SystemAPI(system)

        # Make some requests
        for i in range(5):
            request = APIRequest(
                request_id=f'req_{i}',
                command=CommandType.GET_STATUS
            )
            api.process_request(request)

        log = api.get_request_log()
        assert len(log) == 5


class TestIntegration:
    """Integration tests for monitoring system"""

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        from phase5.monitoring import (
            PerformanceMonitor, UnifiedControlSystem, SystemAPI,
            MetricType, CommandType, APIRequest
        )

        # Create system
        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        # Create API
        api = SystemAPI(system)

        # Simulate control steps
        for i in range(10):
            measurements = {
                'levels': [3.0 + np.random.randn() * 0.1 for _ in range(3)],
                'flows': [10.0 + np.random.randn() * 1.0 for _ in range(3)]
            }
            system.step(measurements)

        # Get status via API
        response = api.process_request(APIRequest(
            request_id='test',
            command=CommandType.GET_STATUS
        ))
        assert response.status.value == 'success'

        # Get metrics via API
        response = api.process_request(APIRequest(
            request_id='test2',
            command=CommandType.GET_METRICS
        ))
        assert response.status.value == 'success'

    def test_alert_flow(self):
        """Test alert generation and acknowledgment flow"""
        from phase5.monitoring.performance_monitor import (
            PerformanceMonitor, MetricType
        )

        monitor = PerformanceMonitor()
        monitor.register_component('test', 'test', [])

        # Generate alert
        monitor.record_metric(MetricType.SOLVE_TIME, 10.0, 'test', 'seconds')

        # Verify alert
        assert len(monitor.active_alerts) > 0
        alert_id = list(monitor.active_alerts.keys())[0]

        # Acknowledge
        monitor.acknowledge_alert(alert_id, "Resolved")

        # Verify cleared
        assert alert_id not in monitor.active_alerts

    def test_mode_change_and_recovery(self):
        """Test mode change and recovery"""
        from phase5.monitoring.unified_system import (
            UnifiedControlSystem, SystemMode
        )

        system = UnifiedControlSystem({'num_pools': 3})
        system.initialize()

        # Change to emergency mode
        system._change_mode(SystemMode.EMERGENCY)
        assert system.mode == SystemMode.EMERGENCY

        # Recover to normal
        system._change_mode(SystemMode.NORMAL)
        assert system.mode == SystemMode.NORMAL

        # Check events recorded
        mode_events = [
            e for e in system.event_history
            if 'mode' in e.description.lower()
        ]
        assert len(mode_events) >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
