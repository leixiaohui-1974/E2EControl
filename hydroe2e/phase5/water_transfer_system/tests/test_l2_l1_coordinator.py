"""
L2-L1协调器测试
Tests for L2-L1 Coordinator
"""

import unittest
import time


from hydroe2e.phase5.water_transfer_system.l2_l1_coordinator import (
    L2L1Coordinator,
    FullLineCoordinatorManager,
    CoordinationType,
    CoordinationRequest,
    CoordinationResponse,
)
from hydroe2e.phase5.water_transfer_system.l1_controller import (
    L1Controller,
    L1ControllerManager,
    L1ControllerState,
)
from hydroe2e.phase5.water_transfer_system.local_pool_scenarios import (
    L1ScenarioType,
    L1ActionType,
    L1ScenarioEvent,
    L1ActionCommand,
)
from hydroe2e.phase5.water_transfer_system.core_types import ScenarioSeverity


class TestCoordinationType(unittest.TestCase):
    """协调类型测试"""

    def test_all_types_defined(self):
        """测试所有协调类型已定义"""
        expected_types = [
            'POLLUTION_SPREAD',
            'EMERGENCY_DISCHARGE',
            'MAINTENANCE_DISCHARGE',
            'FLOOD_CONTROL',
            'ICE_CONTROL',
            'GATE_FAILURE',
            'LEVEL_BALANCE',
            'FLOW_ADJUSTMENT',
        ]
        for type_name in expected_types:
            self.assertTrue(hasattr(CoordinationType, type_name))


class TestCoordinationRequest(unittest.TestCase):
    """协调请求测试"""

    def test_request_creation(self):
        """测试请求创建"""
        request = CoordinationRequest(
            request_id="REQ_001",
            coordination_type=CoordinationType.POLLUTION_SPREAD,
            source_pool=5,
            priority=8,
            affected_pools=[5, 6, 7, 8],
        )
        self.assertEqual(request.request_id, "REQ_001")
        self.assertEqual(request.coordination_type, CoordinationType.POLLUTION_SPREAD)
        self.assertEqual(request.source_pool, 5)
        self.assertEqual(len(request.affected_pools), 4)
        self.assertFalse(request.is_approved)
        self.assertFalse(request.is_completed)


class TestL2L1Coordinator(unittest.TestCase):
    """L2-L1协调器测试"""

    def setUp(self):
        self.coordinator = L2L1Coordinator(
            region_id=0,
            pool_range=(0, 10),
            num_pools=60
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.coordinator.region_id, 0)
        self.assertEqual(self.coordinator.pool_start, 0)
        self.assertEqual(self.coordinator.pool_end, 10)
        self.assertIsNotNone(self.coordinator.l1_manager)

    def test_map_event_to_coordination_pollution(self):
        """测试事件映射-污染"""
        event = L1ScenarioEvent(
            event_id="TEST_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.HIGH,
        )
        coord_type = self.coordinator._map_event_to_coordination(event)
        self.assertEqual(coord_type, CoordinationType.POLLUTION_SPREAD)

    def test_map_event_to_coordination_discharge(self):
        """测试事件映射-退水"""
        event = L1ScenarioEvent(
            event_id="TEST_002",
            scenario_type=L1ScenarioType.L1_DISCHARGE_EMERGENCY,
            pool_id=5,
            severity=ScenarioSeverity.CRITICAL,
        )
        coord_type = self.coordinator._map_event_to_coordination(event)
        self.assertEqual(coord_type, CoordinationType.EMERGENCY_DISCHARGE)

    def test_map_event_to_coordination_ice(self):
        """测试事件映射-冰凌"""
        event = L1ScenarioEvent(
            event_id="TEST_003",
            scenario_type=L1ScenarioType.L1_ICE_BLOCKAGE,
            pool_id=5,
            severity=ScenarioSeverity.HIGH,
        )
        coord_type = self.coordinator._map_event_to_coordination(event)
        self.assertEqual(coord_type, CoordinationType.ICE_CONTROL)

    def test_determine_affected_pools_pollution(self):
        """测试影响池确定-污染"""
        affected = self.coordinator._determine_affected_pools(
            source_pool=5,
            coord_type=CoordinationType.POLLUTION_SPREAD
        )
        # 污染影响下游池
        self.assertIn(5, affected)
        self.assertIn(6, affected)
        self.assertIn(7, affected)

    def test_determine_affected_pools_discharge(self):
        """测试影响池确定-退水"""
        affected = self.coordinator._determine_affected_pools(
            source_pool=5,
            coord_type=CoordinationType.EMERGENCY_DISCHARGE
        )
        # 退水影响上下游
        self.assertIn(3, affected)  # 上游
        self.assertIn(5, affected)  # 源池
        self.assertIn(7, affected)  # 下游

    def test_determine_affected_pools_ice(self):
        """测试影响池确定-冰凌"""
        affected = self.coordinator._determine_affected_pools(
            source_pool=5,
            coord_type=CoordinationType.ICE_CONTROL
        )
        # 冰凌影响上下游
        self.assertIn(2, affected)
        self.assertIn(5, affected)
        self.assertIn(7, affected)

    def test_calculate_priority(self):
        """测试优先级计算"""
        event = L1ScenarioEvent(
            event_id="TEST_004",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.CRITICAL,
        )

        priority = self.coordinator._calculate_priority(
            event, CoordinationType.POLLUTION_SPREAD
        )
        # 污染基础9 + CRITICAL加2 = 11 -> 限制到10
        self.assertEqual(priority, 10)

    def test_create_coordination_request(self):
        """测试创建协调请求"""
        event = L1ScenarioEvent(
            event_id="TEST_005",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.HIGH,
        )

        request = self.coordinator._create_coordination_request(
            source_pool=5,
            event=event,
            coord_type=CoordinationType.POLLUTION_SPREAD
        )

        self.assertIsNotNone(request.request_id)
        self.assertEqual(request.coordination_type, CoordinationType.POLLUTION_SPREAD)
        self.assertEqual(request.source_pool, 5)
        self.assertGreater(len(request.affected_pools), 0)

    def test_coordinate_pollution(self):
        """测试污染协调"""
        request = CoordinationRequest(
            request_id="POLL_REQ_001",
            coordination_type=CoordinationType.POLLUTION_SPREAD,
            source_pool=5,
            priority=9,
            affected_pools=[5, 6, 7],
        )

        commands = self.coordinator._coordinate_pollution(request)

        self.assertEqual(len(commands), 3)
        # 源池应该有隔离命令
        self.assertTrue(any(
            c.action_type == L1ActionType.ISOLATE_UPSTREAM
            for c in commands.get(5, [])
        ))

    def test_coordinate_emergency_discharge(self):
        """测试紧急退水协调"""
        request = CoordinationRequest(
            request_id="DRAIN_REQ_001",
            coordination_type=CoordinationType.EMERGENCY_DISCHARGE,
            source_pool=5,
            priority=10,
            affected_pools=[3, 4, 5, 6, 7],
        )

        commands = self.coordinator._coordinate_emergency_discharge(request)

        self.assertEqual(len(commands), 5)
        # 源池应该有退水命令
        self.assertTrue(any(
            c.action_type == L1ActionType.DRAIN_START
            for c in commands.get(5, [])
        ))

    def test_coordinate_flood_control(self):
        """测试防洪协调"""
        request = CoordinationRequest(
            request_id="FLOOD_REQ_001",
            coordination_type=CoordinationType.FLOOD_CONTROL,
            source_pool=5,
            priority=9,
            affected_pools=[5, 6, 7, 8],
        )

        commands = self.coordinator._coordinate_flood_control(request)

        self.assertEqual(len(commands), 4)
        # 应该有开闸命令
        for pool_id in [5, 6, 7, 8]:
            self.assertTrue(any(
                c.action_type == L1ActionType.GATE_OPEN
                for c in commands.get(pool_id, [])
            ))

    def test_coordinate_ice_control(self):
        """测试冰凌协调"""
        request = CoordinationRequest(
            request_id="ICE_REQ_001",
            coordination_type=CoordinationType.ICE_CONTROL,
            source_pool=5,
            priority=7,
            affected_pools=[4, 5, 6],
        )

        commands = self.coordinator._coordinate_ice_control(request)

        self.assertEqual(len(commands), 3)
        # 应该有闸门调整和监测加强命令
        for pool_id in [4, 5, 6]:
            pool_cmds = commands.get(pool_id, [])
            self.assertTrue(any(
                c.action_type == L1ActionType.GATE_ADJUST
                for c in pool_cmds
            ))

    def test_coordinate_gate_failure(self):
        """测试闸门故障协调"""
        request = CoordinationRequest(
            request_id="GATE_REQ_001",
            coordination_type=CoordinationType.GATE_FAILURE,
            source_pool=5,
            priority=8,
            affected_pools=[4, 5, 6],
        )

        commands = self.coordinator._coordinate_gate_failure(request)

        self.assertEqual(len(commands), 3)
        # 故障池应该锁定
        self.assertTrue(any(
            c.action_type == L1ActionType.GATE_LOCK
            for c in commands.get(5, [])
        ))

    def test_process_pending_requests(self):
        """测试处理待定请求"""
        request = CoordinationRequest(
            request_id="REQ_001",
            coordination_type=CoordinationType.POLLUTION_SPREAD,
            source_pool=5,
            priority=8,
            affected_pools=[5, 6, 7],
            request_time=time.time(),
            deadline=time.time() + 300,
        )

        self.coordinator.pending_requests.append(request)
        responses = self.coordinator.process_pending_requests()

        self.assertEqual(len(responses), 1)
        self.assertTrue(responses[0].success)

    def test_coordination_step(self):
        """测试协调步"""
        result = self.coordinator.coordination_step(dt=60.0)

        self.assertEqual(result['region_id'], 0)
        self.assertIn('responses', result)
        self.assertIn('active_coordinations', result)
        self.assertIn('l1_results', result)
        self.assertIn('region_state', result)

    def test_handle_l1_escalation(self):
        """测试处理L1上报"""
        event = L1ScenarioEvent(
            event_id="ESC_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.CRITICAL,
        )

        self.coordinator._handle_l1_escalation(5, event)

        # 应该创建协调请求
        self.assertEqual(len(self.coordinator.pending_requests), 1)


class TestFullLineCoordinatorManager(unittest.TestCase):
    """全线协调管理器测试"""

    def setUp(self):
        self.manager = FullLineCoordinatorManager(num_pools=60)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.manager.num_pools, 60)
        self.assertEqual(len(self.manager.coordinators), 6)  # 6个区域

    def test_regions_defined(self):
        """测试区域定义"""
        self.assertEqual(len(self.manager.REGIONS), 6)
        # 检查区域覆盖所有池
        all_pools = set()
        for region in self.manager.REGIONS:
            for pool_id in range(region['pools'][0], region['pools'][1]):
                all_pools.add(pool_id)
        self.assertEqual(len(all_pools), 60)

    def test_get_region_for_pool(self):
        """测试获取渠池所属区域"""
        # 区域0: 池0-9
        self.assertEqual(self.manager.get_region_for_pool(5), 0)
        # 区域1: 池10-19
        self.assertEqual(self.manager.get_region_for_pool(15), 1)
        # 区域5: 池50-59
        self.assertEqual(self.manager.get_region_for_pool(55), 5)

    def test_broadcast_event(self):
        """测试广播事件"""
        event = L1ScenarioEvent(
            event_id="BROAD_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
        )

        self.manager.broadcast_event(event)

        # 事件应该被区域0的协调器接收
        region_0 = self.manager.coordinators[0]
        self.assertEqual(
            len(region_0.l1_manager.controllers[5].event_queue), 1
        )

    def test_coordination_step_all(self):
        """测试所有协调器执行一步"""
        results = self.manager.coordination_step_all(dt=60.0)

        self.assertEqual(len(results), 6)
        for region_id in range(6):
            self.assertIn(region_id, results)
            self.assertEqual(results[region_id]['region_id'], region_id)

    def test_get_system_status(self):
        """测试获取系统状态"""
        status = self.manager.get_system_status()

        self.assertEqual(status['total_regions'], 6)
        self.assertIn('total_events', status)
        self.assertIn('total_coordinations', status)
        self.assertIn('emergency_regions', status)

    def test_handle_cross_region_coordination(self):
        """测试跨区域协调"""
        self.manager.handle_cross_region_coordination(
            source_region=0,
            coord_type=CoordinationType.POLLUTION_SPREAD,
            affected_regions=[0, 1]
        )

        self.assertEqual(len(self.manager.cross_region_events), 1)


class TestL2L1Integration(unittest.TestCase):
    """L2-L1集成测试"""

    def test_pollution_coordination_workflow(self):
        """测试污染协调工作流"""
        manager = FullLineCoordinatorManager(num_pools=60)

        # 1. 在池5检测到污染
        event = L1ScenarioEvent(
            event_id="INT_POLL_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.CRITICAL,
            measured_value=0.8,
            threshold_value=0.3,
        )

        # 2. 广播事件
        manager.broadcast_event(event)

        # 3. 执行协调步
        results = manager.coordination_step_all(dt=60.0)

        # 4. 验证区域0有活动
        region_0_result = results[0]
        self.assertIn('l1_results', region_0_result)

    def test_emergency_discharge_workflow(self):
        """测试紧急退水工作流"""
        coordinator = L2L1Coordinator(
            region_id=0,
            pool_range=(0, 10),
            num_pools=60
        )

        # 1. L1上报紧急退水需求
        event = L1ScenarioEvent(
            event_id="INT_DRAIN_001",
            scenario_type=L1ScenarioType.L1_DISCHARGE_EMERGENCY,
            pool_id=5,
            severity=ScenarioSeverity.CRITICAL,
        )

        coordinator._handle_l1_escalation(5, event)

        # 2. 处理请求
        responses = coordinator.process_pending_requests()

        # 3. 验证协调成功
        self.assertEqual(len(responses), 1)
        self.assertTrue(responses[0].success)

        # 4. 验证命令下发
        self.assertGreater(len(responses[0].pool_commands), 0)

    def test_multi_region_coordination(self):
        """测试多区域协调"""
        manager = FullLineCoordinatorManager(num_pools=60)

        # 在不同区域触发事件
        events = [
            L1ScenarioEvent(
                event_id=f"MULTI_{i}",
                scenario_type=L1ScenarioType.L1_LEVEL_HIGH,
                pool_id=i * 10 + 5,
                severity=ScenarioSeverity.HIGH,
            )
            for i in range(3)  # 区域0, 1, 2
        ]

        for event in events:
            manager.broadcast_event(event)

        # 执行协调步
        results = manager.coordination_step_all(dt=60.0)

        # 验证所有区域都有活动
        for region_id in range(3):
            self.assertIn(region_id, results)


if __name__ == '__main__':
    unittest.main()
