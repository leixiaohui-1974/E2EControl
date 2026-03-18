"""
Tests for monitor.py (MonitoringSystem, Alert, AlertLevel).

Covers:
- Initialization with mocked config/logger
- Water level checks (warning and critical, high and low)
- Flow rate checks
- Deviation from target level
- Alert callback registration and invocation
- Alert filtering and limiting
- Statistics tracking
- Report generation
- Edge cases and boundary values
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from hydroe2e.monitor import MonitoringSystem, Alert, AlertLevel


def _create_monitor(**overrides):
    """Create a MonitoringSystem with mocked dependencies.

    By default the monitoring thresholds are:
        level_warning_high=8.0, level_warning_low=1.0,
        level_critical_high=9.5, level_critical_low=0.5,
        flow_rate_max=25.0
    """
    mon_defaults = {
        'level_warning_high': 8.0,
        'level_warning_low': 1.0,
        'level_critical_high': 9.5,
        'level_critical_low': 0.5,
        'flow_rate_max': 25.0,
    }
    mon_defaults.update(overrides)

    mock_config = MagicMock()
    mock_config.get_section.return_value = mon_defaults

    mock_logger = MagicMock()

    with patch('hydroe2e.monitor.get_config', return_value=mock_config), \
         patch('hydroe2e.monitor.get_logger', return_value=mock_logger):
        monitor = MonitoringSystem()

    return monitor


class TestMonitorInit(unittest.TestCase):
    """Tests for MonitoringSystem initialization."""

    def test_default_thresholds(self):
        monitor = _create_monitor()
        self.assertEqual(monitor.level_warning_high, 8.0)
        self.assertEqual(monitor.level_warning_low, 1.0)
        self.assertEqual(monitor.level_critical_high, 9.5)
        self.assertEqual(monitor.level_critical_low, 0.5)
        self.assertEqual(monitor.flow_rate_max, 25.0)

    def test_custom_thresholds(self):
        monitor = _create_monitor(level_warning_high=7.0, flow_rate_max=30.0)
        self.assertEqual(monitor.level_warning_high, 7.0)
        self.assertEqual(monitor.flow_rate_max, 30.0)

    def test_empty_alerts_on_init(self):
        monitor = _create_monitor()
        self.assertEqual(len(monitor.alerts), 0)

    def test_initial_stats(self):
        monitor = _create_monitor()
        stats = monitor.get_statistics()
        self.assertEqual(stats['total_alerts'], 0)
        self.assertEqual(stats['warning_count'], 0)
        self.assertEqual(stats['critical_count'], 0)
        self.assertIsNone(stats['last_check_time'])


class TestCheckStateNormal(unittest.TestCase):
    """Tests for check_state with normal (no-alert) values."""

    def setUp(self):
        self.monitor = _create_monitor()
        self.config = {'Z_ref': 3.0}

    def test_normal_state_no_alerts(self):
        alerts = self.monitor.check_state(0, 3.0, 5.0, 5.0, self.config)
        self.assertEqual(len(alerts), 0)

    def test_level_just_below_warning_high(self):
        alerts = self.monitor.check_state(0, 7.99, 5.0, 5.0, self.config)
        # No alert for level, but deviation from Z_ref=3.0 is 4.99 > 2.0
        level_alerts = [a for a in alerts if '水位偏高' in a.alert_type]
        self.assertEqual(len(level_alerts), 0)

    def test_level_just_above_warning_low(self):
        alerts = self.monitor.check_state(0, 1.01, 5.0, 5.0, self.config)
        level_alerts = [a for a in alerts if '水位偏低' in a.alert_type]
        self.assertEqual(len(level_alerts), 0)


class TestCheckStateLevelHigh(unittest.TestCase):
    """Tests for high water level alerts."""

    def setUp(self):
        self.monitor = _create_monitor()
        self.config = {'Z_ref': 3.0}

    def test_warning_high_level(self):
        alerts = self.monitor.check_state(10, 8.5, 5.0, 5.0, self.config)
        warning_alerts = [a for a in alerts if a.alert_type == '水位偏高']
        self.assertEqual(len(warning_alerts), 1)
        self.assertEqual(warning_alerts[0].level, AlertLevel.WARNING)

    def test_critical_high_level(self):
        alerts = self.monitor.check_state(10, 9.8, 5.0, 5.0, self.config)
        critical_alerts = [a for a in alerts if a.alert_type == '水位严重偏高']
        self.assertEqual(len(critical_alerts), 1)
        self.assertEqual(critical_alerts[0].level, AlertLevel.CRITICAL)

    def test_critical_high_suppresses_warning(self):
        """When level >= critical_high, only critical alert is raised (not warning too)."""
        alerts = self.monitor.check_state(10, 9.8, 5.0, 5.0, self.config)
        high_alerts = [a for a in alerts if '水位偏高' == a.alert_type]
        self.assertEqual(len(high_alerts), 0)

    def test_exactly_at_warning_threshold(self):
        alerts = self.monitor.check_state(10, 8.0, 5.0, 5.0, self.config)
        warning_alerts = [a for a in alerts if a.alert_type == '水位偏高']
        self.assertEqual(len(warning_alerts), 1)

    def test_exactly_at_critical_threshold(self):
        alerts = self.monitor.check_state(10, 9.5, 5.0, 5.0, self.config)
        critical_alerts = [a for a in alerts if a.alert_type == '水位严重偏高']
        self.assertEqual(len(critical_alerts), 1)


class TestCheckStateLevelLow(unittest.TestCase):
    """Tests for low water level alerts."""

    def setUp(self):
        self.monitor = _create_monitor()
        self.config = {'Z_ref': 3.0}

    def test_warning_low_level(self):
        alerts = self.monitor.check_state(20, 0.8, 5.0, 5.0, self.config)
        warning_alerts = [a for a in alerts if a.alert_type == '水位偏低']
        self.assertEqual(len(warning_alerts), 1)
        self.assertEqual(warning_alerts[0].level, AlertLevel.WARNING)

    def test_critical_low_level(self):
        alerts = self.monitor.check_state(20, 0.3, 5.0, 5.0, self.config)
        critical_alerts = [a for a in alerts if a.alert_type == '水位严重偏低']
        self.assertEqual(len(critical_alerts), 1)
        self.assertEqual(critical_alerts[0].level, AlertLevel.CRITICAL)

    def test_critical_low_suppresses_warning(self):
        alerts = self.monitor.check_state(20, 0.3, 5.0, 5.0, self.config)
        low_warnings = [a for a in alerts if a.alert_type == '水位偏低']
        self.assertEqual(len(low_warnings), 0)

    def test_exactly_at_low_warning(self):
        alerts = self.monitor.check_state(20, 1.0, 5.0, 5.0, self.config)
        warning_alerts = [a for a in alerts if a.alert_type == '水位偏低']
        self.assertEqual(len(warning_alerts), 1)

    def test_exactly_at_low_critical(self):
        alerts = self.monitor.check_state(20, 0.5, 5.0, 5.0, self.config)
        critical_alerts = [a for a in alerts if a.alert_type == '水位严重偏低']
        self.assertEqual(len(critical_alerts), 1)


class TestCheckStateFlowRate(unittest.TestCase):
    """Tests for flow rate alerts."""

    def setUp(self):
        self.monitor = _create_monitor()
        self.config = {'Z_ref': 3.0}

    def test_excessive_inflow(self):
        alerts = self.monitor.check_state(30, 3.0, 30.0, 5.0, self.config)
        flow_alerts = [a for a in alerts if a.alert_type == '入流过大']
        self.assertEqual(len(flow_alerts), 1)
        self.assertEqual(flow_alerts[0].level, AlertLevel.WARNING)

    def test_inflow_at_max_no_alert(self):
        alerts = self.monitor.check_state(30, 3.0, 25.0, 5.0, self.config)
        flow_alerts = [a for a in alerts if a.alert_type == '入流过大']
        self.assertEqual(len(flow_alerts), 0)

    def test_inflow_just_above_max(self):
        alerts = self.monitor.check_state(30, 3.0, 25.01, 5.0, self.config)
        flow_alerts = [a for a in alerts if a.alert_type == '入流过大']
        self.assertEqual(len(flow_alerts), 1)


class TestCheckStateDeviation(unittest.TestCase):
    """Tests for target level deviation alerts."""

    def setUp(self):
        self.monitor = _create_monitor()

    def test_large_deviation_triggers_alert(self):
        config = {'Z_ref': 3.0}
        # level=6.0, deviation=3.0 > 2.0
        alerts = self.monitor.check_state(40, 6.0, 5.0, 5.0, config)
        dev_alerts = [a for a in alerts if a.alert_type == '水位偏差过大']
        self.assertEqual(len(dev_alerts), 1)

    def test_small_deviation_no_alert(self):
        config = {'Z_ref': 3.0}
        # level=4.0, deviation=1.0 < 2.0
        alerts = self.monitor.check_state(40, 4.0, 5.0, 5.0, config)
        dev_alerts = [a for a in alerts if a.alert_type == '水位偏差过大']
        self.assertEqual(len(dev_alerts), 0)

    def test_default_z_ref_used_when_missing(self):
        config = {}  # no Z_ref -> defaults to 3.0
        alerts = self.monitor.check_state(40, 3.0, 5.0, 5.0, config)
        dev_alerts = [a for a in alerts if a.alert_type == '水位偏差过大']
        self.assertEqual(len(dev_alerts), 0)


class TestMultipleAlerts(unittest.TestCase):
    """Tests that multiple alerts can fire in a single check."""

    def setUp(self):
        self.monitor = _create_monitor()

    def test_high_level_and_high_flow_and_deviation(self):
        config = {'Z_ref': 3.0}
        # level=9.8 (critical high) + q_in=30 (flow too high) + deviation=6.8 > 2.0
        alerts = self.monitor.check_state(50, 9.8, 30.0, 5.0, config)
        types = {a.alert_type for a in alerts}
        self.assertIn('水位严重偏高', types)
        self.assertIn('入流过大', types)
        self.assertIn('水位偏差过大', types)
        self.assertGreaterEqual(len(alerts), 3)


class TestAlertCallbacks(unittest.TestCase):
    """Tests for callback registration and invocation."""

    def setUp(self):
        self.monitor = _create_monitor()

    def test_register_callback(self):
        cb = MagicMock(__name__='my_callback')
        self.monitor.register_callback(cb)
        self.assertIn(cb, self.monitor.alert_callbacks)

    def test_callback_invoked_on_alert(self):
        cb = MagicMock(__name__='my_callback')
        self.monitor.register_callback(cb)
        self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        self.assertTrue(cb.called)

    def test_multiple_callbacks_invoked(self):
        cb1 = MagicMock(__name__='cb1')
        cb2 = MagicMock(__name__='cb2')
        self.monitor.register_callback(cb1)
        self.monitor.register_callback(cb2)
        self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        self.assertTrue(cb1.called)
        self.assertTrue(cb2.called)

    def test_callback_exception_does_not_crash(self):
        def bad_callback(alert):
            raise RuntimeError("callback failed")
        bad_callback.__name__ = 'bad_callback'

        self.monitor.register_callback(bad_callback)
        # Should not raise
        alerts = self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        self.assertGreater(len(alerts), 0)

    def test_no_callback_when_no_alert(self):
        cb = MagicMock(__name__='my_callback')
        self.monitor.register_callback(cb)
        self.monitor.check_state(0, 3.0, 5.0, 5.0, {'Z_ref': 3.0})
        cb.assert_not_called()


class TestGetAlerts(unittest.TestCase):
    """Tests for get_alerts with filtering."""

    def setUp(self):
        self.monitor = _create_monitor()
        # Trigger a mix of warnings and criticals
        self.monitor.check_state(0, 8.5, 5.0, 5.0, {'Z_ref': 3.0})  # WARNING high
        self.monitor.check_state(1, 9.8, 5.0, 5.0, {'Z_ref': 3.0})  # CRITICAL high

    def test_get_all_alerts(self):
        alerts = self.monitor.get_alerts()
        self.assertGreater(len(alerts), 0)

    def test_filter_by_warning(self):
        alerts = self.monitor.get_alerts(level=AlertLevel.WARNING)
        for a in alerts:
            self.assertEqual(a.level, AlertLevel.WARNING)

    def test_filter_by_critical(self):
        alerts = self.monitor.get_alerts(level=AlertLevel.CRITICAL)
        for a in alerts:
            self.assertEqual(a.level, AlertLevel.CRITICAL)

    def test_limit_alerts(self):
        # Add more alerts
        for _ in range(5):
            self.monitor.check_state(0, 8.5, 5.0, 5.0, {'Z_ref': 3.0})
        alerts = self.monitor.get_alerts(limit=2)
        self.assertEqual(len(alerts), 2)

    def test_limit_returns_most_recent(self):
        self.monitor.check_state(99, 0.3, 5.0, 5.0, {'Z_ref': 3.0})
        alerts = self.monitor.get_alerts(limit=1)
        self.assertEqual(len(alerts), 1)
        # The last alert should have time_step=99
        self.assertEqual(alerts[0].data.get('time_step'), 99)


class TestStatistics(unittest.TestCase):
    """Tests for statistics tracking."""

    def setUp(self):
        self.monitor = _create_monitor()

    def test_stats_update_after_warning(self):
        self.monitor.check_state(0, 8.5, 5.0, 5.0, {'Z_ref': 8.0})
        stats = self.monitor.get_statistics()
        self.assertGreaterEqual(stats['warning_count'], 1)
        self.assertGreaterEqual(stats['total_alerts'], 1)

    def test_stats_update_after_critical(self):
        self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 9.8})
        stats = self.monitor.get_statistics()
        self.assertGreaterEqual(stats['critical_count'], 1)

    def test_last_check_time_updated(self):
        self.monitor.check_state(0, 3.0, 5.0, 5.0, {'Z_ref': 3.0})
        stats = self.monitor.get_statistics()
        self.assertIsNotNone(stats['last_check_time'])
        self.assertIsInstance(stats['last_check_time'], datetime)

    def test_stats_is_copy(self):
        stats = self.monitor.get_statistics()
        stats['total_alerts'] = 9999
        self.assertEqual(self.monitor.stats['total_alerts'], 0)


class TestClearAlerts(unittest.TestCase):
    """Tests for clearing alerts."""

    def setUp(self):
        self.monitor = _create_monitor()

    def test_clear_alerts(self):
        self.monitor.check_state(0, 9.8, 30.0, 5.0, {'Z_ref': 3.0})
        self.assertGreater(len(self.monitor.alerts), 0)
        self.monitor.clear_alerts()
        self.assertEqual(len(self.monitor.alerts), 0)

    def test_clear_does_not_reset_stats(self):
        self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        total_before = self.monitor.stats['total_alerts']
        self.monitor.clear_alerts()
        self.assertEqual(self.monitor.stats['total_alerts'], total_before)


class TestGenerateReport(unittest.TestCase):
    """Tests for generate_report."""

    def setUp(self):
        self.monitor = _create_monitor()

    def test_report_contains_header(self):
        report = self.monitor.generate_report()
        self.assertIn("监控系统报告", report)

    def test_report_contains_stats(self):
        self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        report = self.monitor.generate_report()
        self.assertIn("总告警数", report)
        self.assertIn("警告数", report)
        self.assertIn("严重告警数", report)

    def test_report_with_no_alerts(self):
        report = self.monitor.generate_report()
        self.assertIn("总告警数: 0", report)

    def test_report_includes_recent_alerts(self):
        self.monitor.check_state(5, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        report = self.monitor.generate_report()
        self.assertIn("最近告警", report)
        self.assertIn("严重", report)


class TestAlertDataclass(unittest.TestCase):
    """Tests for the Alert dataclass."""

    def test_alert_str(self):
        alert = Alert(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            level=AlertLevel.WARNING,
            alert_type="水位偏高",
            message="水位超标",
            data={'time_step': 10}
        )
        s = str(alert)
        self.assertIn("警告", s)
        self.assertIn("水位偏高", s)
        self.assertIn("水位超标", s)

    def test_alert_level_enum_values(self):
        self.assertEqual(AlertLevel.INFO.value, "信息")
        self.assertEqual(AlertLevel.WARNING.value, "警告")
        self.assertEqual(AlertLevel.CRITICAL.value, "严重")


if __name__ == '__main__':
    unittest.main()
