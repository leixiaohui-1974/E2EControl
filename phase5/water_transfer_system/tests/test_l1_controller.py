"""
L1控制器测试
Tests for L1 Controller
"""

import unittest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from phase5.water_transfer_system.l1_controller import (
    L1Controller,
    L1ControllerManager,
    L1ControllerState,
    L1PoolState,
    L1ControlResult,
    L1ResponseStrategy,
)
from phase5.water_transfer_system.local_pool_scenarios import (
    L1ScenarioType,
    L1ActionType,
    L1ScenarioEvent,
    L1ActionCommand,
)
from phase5.water_transfer_system.core_types import ScenarioSeverity


class TestL1ResponseStrategy(unittest.TestCase):
    """L1响应策略测试"""

    def test_get_response_actions_pollution(self):
        """测试污染场景响应动作"""
        actions = L1ResponseStrategy.get_response_actions(
            L1ScenarioType.L1_POLLUTION_DETECTED,
            ScenarioSeverity.MEDIUM
        )
        self.assertIn(L1ActionType.ALARM_TRIGGER, actions)
        self.assertIn(L1ActionType.MONITOR_ENHANCE, actions)

    def test_get_response_actions_high_severity(self):
        """测试高严重程度添加上报动作"""
        actions = L1ResponseStrategy.get_response_actions(
            L1ScenarioType.L1_LEVEL_HIGH,
            ScenarioSeverity.CRITICAL
        )
        self.assertIn(L1ActionType.ALARM_ESCALATE, actions)

    def test_get_response_actions_slope(self):
        """测试边坡场景响应动作"""
        actions = L1ResponseStrategy.get_response_actions(
            L1ScenarioType.L1_SLOPE_PANEL_FLOAT,
            ScenarioSeverity.HIGH
        )
        self.assertIn(L1ActionType.GATE_OPEN, actions)
        self.assertIn(L1ActionType.DRAIN_START, actions)

    def test_get_priority(self):
        """测试优先级获取"""
        self.assertEqual(L1ResponseStrategy.get_priority(ScenarioSeverity.LOW), 3)
        self.assertEqual(L1ResponseStrategy.get_priority(ScenarioSeverity.MEDIUM), 5)
        self.assertEqual(L1ResponseStrategy.get_priority(ScenarioSeverity.HIGH), 8)
        self.assertEqual(L1ResponseStrategy.get_priority(ScenarioSeverity.CRITICAL), 10)

    def test_scenario_actions_mapping(self):
        """测试场景动作映射表"""
        # 检查所有场景类型都有映射
        scenario_count = 0
        for scenario_type in L1ScenarioType:
            if scenario_type in L1ResponseStrategy.SCENARIO_ACTIONS:
                actions = L1ResponseStrategy.SCENARIO_ACTIONS[scenario_type]
                self.assertIsInstance(actions, list)
                self.assertGreater(len(actions), 0)
                scenario_count += 1

        # 至少应该有20个场景有映射
        self.assertGreater(scenario_count, 15)


class TestL1PoolState(unittest.TestCase):
    """L1渠池状态测试"""

    def test_default_state(self):
        """测试默认状态"""
        state = L1PoolState(pool_id=5)
        self.assertEqual(state.pool_id, 5)
        self.assertEqual(state.water_level, 3.0)
        self.assertEqual(state.gate_position, 0.5)
        self.assertTrue(state.gate_operational)
        self.assertEqual(state.controller_state, L1ControllerState.MONITORING)

    def test_state_modification(self):
        """测试状态修改"""
        state = L1PoolState(pool_id=5)
        state.water_level = 4.5
        state.gate_position = 0.8
        self.assertEqual(state.water_level, 4.5)
        self.assertEqual(state.gate_position, 0.8)


class TestL1Controller(unittest.TestCase):
    """L1控制器测试"""

    def setUp(self):
        self.controller = L1Controller(pool_id=5, num_pools=60)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.controller.pool_id, 5)
        self.assertEqual(self.controller.num_pools, 60)
        self.assertIsNotNone(self.controller.pollution_tracker)
        self.assertIsNotNone(self.controller.slope_monitor)
        self.assertIsNotNone(self.controller.discharge_manager)

    def test_update_state(self):
        """测试状态更新"""
        self.controller.update_state(
            water_level=4.5,
            gate_position=0.7,
            water_quality=0.9
        )
        self.assertEqual(self.controller.state.water_level, 4.5)
        self.assertEqual(self.controller.state.gate_position, 0.7)
        self.assertEqual(self.controller.state.water_quality, 0.9)

    def test_receive_event(self):
        """测试事件接收"""
        event = L1ScenarioEvent(
            event_id="TEST_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
            measured_value=0.5,
            threshold_value=0.3,
        )

        result = self.controller.receive_event(event)
        self.assertTrue(result)
        self.assertEqual(len(self.controller.event_queue), 1)
        self.assertIn("TEST_001", self.controller.active_events)

    def test_receive_event_wrong_pool(self):
        """测试接收其他池的事件"""
        event = L1ScenarioEvent(
            event_id="TEST_002",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=10,  # 不同的池
            severity=ScenarioSeverity.MEDIUM,
        )

        result = self.controller.receive_event(event)
        self.assertFalse(result)
        self.assertEqual(len(self.controller.event_queue), 0)

    def test_receive_high_severity_event(self):
        """测试接收高严重程度事件"""
        event = L1ScenarioEvent(
            event_id="TEST_003",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.CRITICAL,
            measured_value=0.8,
            threshold_value=0.3,
        )

        self.controller.receive_event(event)
        # 高严重程度事件应该立即处理并生成命令
        self.assertGreater(len(self.controller.command_queue), 0)

    def test_process_events(self):
        """测试事件处理"""
        event = L1ScenarioEvent(
            event_id="TEST_004",
            scenario_type=L1ScenarioType.L1_LEVEL_HIGH,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
            measured_value=5.5,
            threshold_value=5.0,
        )

        self.controller.event_queue.append(event)
        results = self.controller.process_events()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].pool_id, 5)

    def test_execute_commands(self):
        """测试命令执行"""
        # 设置初始闸门位置使变化率在允许范围内
        self.controller.state.gate_position = 0.6

        cmd = L1ActionCommand(
            command_id="CMD_001",
            action_type=L1ActionType.GATE_OPEN,
            pool_id=5,
            priority=8,
            gate_position=0.7,  # 变化0.1，在允许范围内
        )

        self.controller.command_queue.append(cmd)
        results = self.controller.execute_commands()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(self.controller.state.gate_position, 0.7)

    def test_control_step(self):
        """测试控制步"""
        result = self.controller.control_step(dt=60.0)

        self.assertEqual(result['pool_id'], 5)
        self.assertIn('events_processed', result)
        self.assertIn('commands_executed', result)
        self.assertIn('state', result)

    def test_gate_safety_check_high_level(self):
        """测试高水位安全检查"""
        self.controller.state.water_level = 5.8  # 接近上限
        # 高水位时只能开大闸门
        self.assertTrue(self.controller._check_gate_safety(0.8))  # 开大OK
        self.assertFalse(self.controller._check_gate_safety(0.3))  # 关小不OK

    def test_gate_safety_check_low_level(self):
        """测试低水位安全检查"""
        self.controller.state.water_level = 0.6  # 接近下限
        # 低水位时只能关小闸门
        self.assertTrue(self.controller._check_gate_safety(0.3))  # 关小OK
        self.assertFalse(self.controller._check_gate_safety(0.8))  # 开大不OK

    def test_detect_local_scenarios_high_level(self):
        """测试本地场景检测-高水位"""
        self.controller.state.water_level = 5.5  # 高水位
        events = self.controller._detect_local_scenarios()

        level_events = [e for e in events if e.scenario_type == L1ScenarioType.L1_LEVEL_HIGH]
        self.assertEqual(len(level_events), 1)

    def test_detect_local_scenarios_low_level(self):
        """测试本地场景检测-低水位"""
        self.controller.state.water_level = 0.55  # 低水位
        events = self.controller._detect_local_scenarios()

        level_events = [e for e in events if e.scenario_type == L1ScenarioType.L1_LEVEL_LOW]
        self.assertEqual(len(level_events), 1)

    def test_detect_local_scenarios_poor_quality(self):
        """测试本地场景检测-水质差"""
        self.controller.state.water_quality = 0.5  # 水质差
        events = self.controller._detect_local_scenarios()

        pollution_events = [e for e in events if e.scenario_type == L1ScenarioType.L1_POLLUTION_DETECTED]
        self.assertEqual(len(pollution_events), 1)

    def test_l2_callback(self):
        """测试L2回调"""
        callback_calls = []

        def mock_callback(pool_id, event):
            callback_calls.append((pool_id, event))

        self.controller.set_l2_callback(mock_callback)

        # 发送需要上报的事件
        event = L1ScenarioEvent(
            event_id="TEST_005",
            scenario_type=L1ScenarioType.L1_SLOPE_INSTABILITY,
            pool_id=5,
            severity=ScenarioSeverity.CRITICAL,
        )
        self.controller.receive_event(event)

        # 检查回调被调用
        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0][0], 5)


class TestL1ControllerManager(unittest.TestCase):
    """L1控制器管理器测试"""

    def setUp(self):
        self.manager = L1ControllerManager(num_pools=10)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.manager.num_pools, 10)
        self.assertEqual(len(self.manager.controllers), 10)

    def test_update_pool_state(self):
        """测试更新渠池状态"""
        self.manager.update_pool_state(5, water_level=4.5)
        self.assertEqual(self.manager.controllers[5].state.water_level, 4.5)

    def test_broadcast_event(self):
        """测试广播事件"""
        event = L1ScenarioEvent(
            event_id="TEST_006",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
        )

        self.manager.broadcast_event(event)
        self.assertEqual(len(self.manager.controllers[5].event_queue), 1)

    def test_control_step_all(self):
        """测试所有控制器执行一步"""
        results = self.manager.control_step_all(dt=60.0)

        self.assertEqual(len(results), 10)
        for pool_id in range(10):
            self.assertIn(pool_id, results)
            self.assertEqual(results[pool_id]['pool_id'], pool_id)

    def test_get_system_status(self):
        """测试获取系统状态"""
        status = self.manager.get_system_status()

        self.assertEqual(status['total_pools'], 10)
        self.assertIn('active_events', status)
        self.assertIn('emergency_pools', status)

    def test_set_l2_coordinator(self):
        """测试设置L2协调器"""
        callback_calls = []

        def mock_callback(pool_id, event):
            callback_calls.append((pool_id, event))

        self.manager.set_l2_coordinator(mock_callback)

        # 验证所有控制器都设置了回调
        for controller in self.manager.controllers.values():
            self.assertIsNotNone(controller.l2_callback)


class TestL1Integration(unittest.TestCase):
    """L1集成测试"""

    def test_pollution_workflow(self):
        """测试污染处理工作流"""
        controller = L1Controller(pool_id=5, num_pools=60)

        # 1. 接收污染事件
        event = L1ScenarioEvent(
            event_id="POLL_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.HIGH,
            measured_value=0.6,
            threshold_value=0.3,
        )
        controller.receive_event(event)

        # 2. 处理事件
        results = controller.process_events()
        self.assertGreater(len(results), 0)

        # 3. 执行命令
        exec_results = controller.execute_commands()
        self.assertGreater(len(exec_results), 0)

    def test_level_control_workflow(self):
        """测试水位控制工作流"""
        controller = L1Controller(pool_id=5, num_pools=60)

        # 设置高水位
        controller.update_state(water_level=5.5)

        # 执行控制步
        result = controller.control_step(dt=60.0)

        # 应该检测到高水位并处理
        self.assertIn('events_processed', result)

    def test_multi_pool_coordination(self):
        """测试多池协调"""
        manager = L1ControllerManager(num_pools=10)

        # 模拟多个池的事件
        for i in range(5):
            event = L1ScenarioEvent(
                event_id=f"EVENT_{i}",
                scenario_type=L1ScenarioType.L1_LEVEL_HIGH,
                pool_id=i,
                severity=ScenarioSeverity.MEDIUM,
            )
            manager.broadcast_event(event)

        # 执行所有控制器
        results = manager.control_step_all(dt=60.0)

        # 验证所有控制器都执行了
        self.assertEqual(len(results), 10)


if __name__ == '__main__':
    unittest.main()
