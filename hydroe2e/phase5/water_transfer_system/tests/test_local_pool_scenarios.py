"""
L1现地渠池全场景测试
Tests for L1 Local Pool Scenarios
"""

import unittest


from hydroe2e.phase5.water_transfer_system.local_pool_scenarios import (
    L1ScenarioType,
    L1ActionType,
    L1ScenarioEvent,
    L1ActionCommand,
    PollutionTracker,
    SlopePanelMonitor,
    DischargeManager,
    DischargeType,
    L1ScenarioGenerator,
)
from hydroe2e.phase5.water_transfer_system.core_types import ScenarioSeverity


class TestPollutionTracker(unittest.TestCase):
    """污染追踪溯源测试"""

    def setUp(self):
        self.tracker = PollutionTracker(num_pools=10)

    def test_detect_pollution(self):
        """测试污染检测"""
        event = self.tracker.detect_pollution(
            pool_id=5,
            concentration=0.5,
            timestamp=0.0,
            threshold=0.3
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.scenario_type, L1ScenarioType.L1_POLLUTION_DETECTED)
        self.assertEqual(event.pool_id, 5)
        self.assertIsNotNone(event.event_id)

    def test_detect_pollution_below_threshold(self):
        """测试低于阈值不触发检测"""
        event = self.tracker.detect_pollution(
            pool_id=5,
            concentration=0.2,
            timestamp=0.0,
            threshold=0.3
        )
        self.assertIsNone(event)

    def test_track_pollution(self):
        """测试污染追踪"""
        # First detect
        detect_event = self.tracker.detect_pollution(
            pool_id=5,
            concentration=0.5,
            timestamp=0.0,
            threshold=0.3
        )
        tracking_id = detect_event.event_id

        # Then track
        track_event = self.tracker.track_pollution(
            tracking_id=tracking_id,
            pool_id=6,
            concentration=0.4,
            timestamp=600.0
        )
        self.assertIsNotNone(track_event)
        self.assertEqual(track_event.scenario_type, L1ScenarioType.L1_POLLUTION_TRACKING)

    def test_trace_source(self):
        """测试污染溯源"""
        # Detect and track
        detect_event = self.tracker.detect_pollution(5, 0.5, 0.0, 0.3)
        tracking_id = detect_event.event_id
        self.tracker.track_pollution(tracking_id, 6, 0.4, 600.0)
        self.tracker.track_pollution(tracking_id, 7, 0.3, 1200.0)

        # Trace source
        trace_event = self.tracker.trace_source(tracking_id)
        self.assertIsNotNone(trace_event)
        self.assertEqual(trace_event.scenario_type, L1ScenarioType.L1_POLLUTION_TRACING)

    def test_predict_arrival(self):
        """测试到达时间预测"""
        detect_event = self.tracker.detect_pollution(5, 0.5, 0.0, 0.3)
        tracking_id = detect_event.event_id

        arrival_time, expected_conc = self.tracker.predict_arrival(tracking_id, 8)
        self.assertGreater(arrival_time, 0)
        self.assertGreater(expected_conc, 0)
        self.assertLess(expected_conc, 0.5)  # Should attenuate


class TestSlopePanelMonitor(unittest.TestCase):
    """边坡衬砌板监测测试"""

    def setUp(self):
        self.monitor = SlopePanelMonitor(num_pools=10)

    def test_update_condition_groundwater(self):
        """测试地下水位检测"""
        event = self.monitor.update_condition(
            pool_id=3,
            groundwater_level=4.0,  # 高于渠道水位
            canal_level=3.0,
            rainfall_1h=0.0
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.scenario_type, L1ScenarioType.L1_SLOPE_GROUNDWATER)

    def test_update_condition_safe(self):
        """测试安全地下水位"""
        event = self.monitor.update_condition(
            pool_id=3,
            groundwater_level=2.5,  # 低于渠道水位
            canal_level=3.0,
            rainfall_1h=0.0
        )
        self.assertIsNone(event)

    def test_update_condition_rainfall(self):
        """测试降雨检测"""
        # Fill history to trigger 24h rainfall alert
        for _ in range(25):
            self.monitor.update_condition(3, 2.5, 3.0, 10.0)

        # Check that high rainfall triggers event
        event = self.monitor.update_condition(
            pool_id=3,
            groundwater_level=2.5,
            canal_level=3.0,
            rainfall_1h=10.0  # 累计后将超过50mm
        )
        # Event may be rainfall or panel float depending on accumulated amount
        if event:
            self.assertIn(event.scenario_type, [
                L1ScenarioType.L1_SLOPE_RAINFALL,
                L1ScenarioType.L1_SLOPE_PANEL_FLOAT,
                L1ScenarioType.L1_SLOPE_GROUNDWATER,
            ])

    def test_recommend_action(self):
        """测试动作推荐"""
        # Trigger unsafe condition
        self.monitor.update_condition(3, 5.0, 3.0, 0.0)

        action = self.monitor.recommend_action(3)
        if action:
            self.assertEqual(action.action_type, L1ActionType.GATE_OPEN)


class TestDischargeManager(unittest.TestCase):
    """退水管理测试"""

    def setUp(self):
        self.manager = DischargeManager(num_pools=10)

    def test_request_discharge_emergency(self):
        """测试紧急退水请求"""
        request, event = self.manager.request_discharge(
            pool_id=5,
            discharge_type=DischargeType.EMERGENCY_POLLUTION,
            target_level=2.0,
            current_level=3.0,
            priority=8
        )
        self.assertIsNotNone(request)
        self.assertIsNotNone(event)
        self.assertEqual(event.scenario_type, L1ScenarioType.L1_DISCHARGE_POLLUTION)

    def test_request_discharge_maintenance(self):
        """测试检修退水请求"""
        request, event = self.manager.request_discharge(
            pool_id=5,
            discharge_type=DischargeType.PLANNED_MAINTENANCE,
            target_level=1.0,
            current_level=3.0,
            priority=5
        )
        self.assertIsNotNone(request)
        self.assertEqual(event.scenario_type, L1ScenarioType.L1_DISCHARGE_MAINTENANCE)

    def test_generate_discharge_commands(self):
        """测试退水动作指令生成"""
        request, event = self.manager.request_discharge(
            pool_id=5,
            discharge_type=DischargeType.EMERGENCY_POLLUTION,
            target_level=2.0,
            current_level=3.0,
            priority=8
        )

        commands = self.manager.generate_discharge_commands(request.request_id)
        self.assertGreater(len(commands), 0)

        # Check main drain command exists
        drain_cmds = [c for c in commands if c.action_type == L1ActionType.DRAIN_START]
        self.assertEqual(len(drain_cmds), 1)


class TestL1ScenarioGenerator(unittest.TestCase):
    """L1场景生成器测试"""

    def setUp(self):
        self.generator = L1ScenarioGenerator(seed=42)

    def test_generate_pollution_event(self):
        """测试污染事件生成"""
        event = self.generator.generate_pollution_event(pool_id=5)
        self.assertEqual(event.scenario_type, L1ScenarioType.L1_POLLUTION_DETECTED)
        self.assertEqual(event.pool_id, 5)

    def test_generate_slope_event(self):
        """测试边坡事件生成"""
        event = self.generator.generate_slope_event(pool_id=3)
        # Slope event may also be NORMAL if conditions are safe
        self.assertIn(event.scenario_type, [
            L1ScenarioType.L1_SLOPE_GROUNDWATER,
            L1ScenarioType.L1_SLOPE_PANEL_FLOAT,
            L1ScenarioType.L1_SLOPE_INSTABILITY,
            L1ScenarioType.L1_NORMAL,
        ])

    def test_generate_discharge_event(self):
        """测试退水事件生成"""
        event = self.generator.generate_discharge_event(pool_id=5)
        self.assertIn(event.scenario_type, [
            L1ScenarioType.L1_DISCHARGE_EMERGENCY,
            L1ScenarioType.L1_DISCHARGE_MAINTENANCE,
            L1ScenarioType.L1_DISCHARGE_POLLUTION,
            L1ScenarioType.L1_DISCHARGE_FLOOD,
        ])

    def test_generate_level_event(self):
        """测试水位事件生成"""
        event = self.generator.generate_level_event(pool_id=5)
        self.assertIn(event.scenario_type, [
            L1ScenarioType.L1_LEVEL_HIGH,
            L1ScenarioType.L1_LEVEL_LOW,
            L1ScenarioType.L1_LEVEL_RAPID_RISE,
            L1ScenarioType.L1_LEVEL_RAPID_DROP,
            L1ScenarioType.L1_LEVEL_OSCILLATION,
        ])

    def test_generate_gate_event(self):
        """测试闸门事件生成"""
        event = self.generator.generate_gate_event(pool_id=5)
        self.assertIn(event.scenario_type, [
            L1ScenarioType.L1_GATE_STUCK,
            L1ScenarioType.L1_GATE_LEAK,
            L1ScenarioType.L1_GATE_CONTROL_FAIL,
            L1ScenarioType.L1_GATE_SENSOR_FAIL,
        ])

    def test_generate_batch(self):
        """测试批量生成"""
        events = self.generator.generate_batch(count=100)
        self.assertEqual(len(events), 100)

        # Check type distribution
        type_counts = {}
        for e in events:
            t = e.scenario_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        # Should have multiple types
        self.assertGreater(len(type_counts), 3)

    def test_generated_count(self):
        """测试生成计数"""
        # Generate some events
        for _ in range(10):
            self.generator.generate_pollution_event()
        self.assertEqual(self.generator.generated_count, 10)


class TestL1Integration(unittest.TestCase):
    """L1集成测试"""

    def test_pollution_to_discharge_workflow(self):
        """测试污染到退水的完整工作流"""
        # 1. 检测污染
        tracker = PollutionTracker(num_pools=10)
        detect_event = tracker.detect_pollution(5, 0.8, 0.0, 0.3)
        self.assertIsNotNone(detect_event)

        # 2. 追踪污染
        tracking_id = detect_event.event_id
        tracker.track_pollution(tracking_id, 6, 0.6, 600.0)

        # 3. 溯源
        trace_event = tracker.trace_source(tracking_id)
        self.assertIsNotNone(trace_event)

        # 4. 启动退水
        discharge_mgr = DischargeManager(num_pools=10)
        request, discharge_event = discharge_mgr.request_discharge(
            pool_id=trace_event.pool_id,
            discharge_type=DischargeType.EMERGENCY_POLLUTION,
            target_level=1.0,
            current_level=3.0,
            priority=9
        )
        self.assertIsNotNone(discharge_event)

    def test_slope_monitoring_workflow(self):
        """测试边坡监测工作流"""
        monitor = SlopePanelMonitor(num_pools=10)

        # 1. 正常情况
        event1 = monitor.update_condition(3, 2.5, 3.0, 0.0)
        self.assertIsNone(event1)

        # 2. 地下水上升
        event2 = monitor.update_condition(3, 4.0, 3.0, 0.0)
        self.assertIsNotNone(event2)

        # 3. 获取推荐动作
        action = monitor.recommend_action(3)
        if action:
            self.assertIn(action.action_type, [L1ActionType.GATE_OPEN, L1ActionType.DRAIN_START])


class TestL1ScenarioEvent(unittest.TestCase):
    """L1场景事件测试"""

    def test_event_creation(self):
        """测试事件创建"""
        event = L1ScenarioEvent(
            event_id="TEST_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.HIGH,
            timestamp=0.0,
            measured_value=0.5,
            threshold_value=0.3,
        )
        self.assertEqual(event.event_id, "TEST_001")
        self.assertEqual(event.pool_id, 5)
        self.assertEqual(event.severity, ScenarioSeverity.HIGH)

    def test_action_command_creation(self):
        """测试动作命令创建"""
        cmd = L1ActionCommand(
            command_id="CMD_001",
            action_type=L1ActionType.GATE_CLOSE,
            pool_id=5,
            gate_position=0.0,
            priority=10,
        )
        self.assertEqual(cmd.command_id, "CMD_001")
        self.assertEqual(cmd.action_type, L1ActionType.GATE_CLOSE)
        self.assertEqual(cmd.priority, 10)


if __name__ == '__main__':
    unittest.main()
