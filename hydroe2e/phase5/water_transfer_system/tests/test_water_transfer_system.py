"""
南水北调中线全线全场景自主运行系统 - 综合集成测试
Water Transfer Autonomous System - Comprehensive Integration Tests

测试覆盖:
1. 核心数据结构测试
2. 物理模型测试
3. 系统辨识测试
4. 全局编排器测试
5. 区域协调器测试
6. 增强MPC测试
7. 全场景端到端测试
"""

import unittest
import numpy as np
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

# 导入测试模块
from hydroe2e.phase5.water_transfer_system.core_types import (
    PoolRole, ScenarioType, ScenarioSeverity, ScenarioPhase,
    ControlDirective, ControlPlan, ScenarioEvent,
    PoolTopology, CanalPoolConfig, RegionConfig, SpecialStructure, StructureType,
)
from hydroe2e.phase5.water_transfer_system.physics_model import (
    SNWDMiddleRouteModel, IDZModel, IDZParameters, CanalPool, SpecialNode,
)
from hydroe2e.phase5.water_transfer_system.system_identification import (
    SystemIdentifier, CrossCorrelationAnalyzer, RecursiveLeastSquares,
)
from hydroe2e.phase5.water_transfer_system.orchestrator import (
    GlobalOrchestrator, ScenarioRoleMatrix,
)
from hydroe2e.phase5.water_transfer_system.regional_coordinator import (
    RegionalCoordinator, FeedforwardDecoupler, GlobalRegionalManager,
)
from hydroe2e.phase5.water_transfer_system.enhanced_mpc import (
    EnhancedParameterizedMPC, HotReconfigurableMPC, RoleParameterMapper,
    MPCWeights, MPCConstraints,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ==============================================================================
# 测试辅助函数
# ==============================================================================

@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    message: str
    metrics: Dict[str, Any] = None


def run_scenario_simulation(model: SNWDMiddleRouteModel,
                            orchestrator: GlobalOrchestrator,
                            event: ScenarioEvent,
                            duration_steps: int = 100) -> Dict[str, Any]:
    """运行场景仿真"""
    model.reset()

    # 处理事件
    plan = orchestrator.process_event(event)

    # 收集指标
    levels_history = []
    flows_history = []

    for step in range(duration_steps):
        # 获取控制命令
        gate_commands = {}
        for pool_id, directive in plan.directives.items():
            if directive.role == PoolRole.ISOLATE:
                gate_commands[pool_id] = 0.0
            elif directive.role == PoolRole.DRAIN:
                gate_commands[pool_id] = 1.0
            elif directive.role == PoolRole.BUFFER:
                gate_commands[pool_id] = 0.3

        # 执行仿真
        state = model.step(gate_commands=gate_commands)

        levels_history.append(model.get_all_levels().copy())
        inflows, outflows = model.get_all_flows()
        flows_history.append(outflows.copy())

    return {
        'levels': np.array(levels_history),
        'flows': np.array(flows_history),
        'plan': plan,
        'final_summary': model.get_summary(),
    }


# ==============================================================================
# 测试类: 核心数据结构
# ==============================================================================

class TestCoreTypes(unittest.TestCase):
    """核心数据结构测试"""

    def test_pool_role_enum(self):
        """测试PoolRole枚举"""
        # 基本角色
        self.assertEqual(PoolRole.TRANSMIT.value, "transmit")
        self.assertEqual(PoolRole.ISOLATE.value, "isolate")
        self.assertEqual(PoolRole.BUFFER.value, "buffer")
        self.assertEqual(PoolRole.DRAIN.value, "drain")

        # 所有角色数量
        self.assertGreaterEqual(len(PoolRole), 6)

    def test_scenario_type_enum(self):
        """测试ScenarioType枚举"""
        # 8大场景
        scenarios = [
            ScenarioType.S1_NORMAL_PLAN,
            ScenarioType.S2_SURGE_DEMAND,
            ScenarioType.S3_POLLUTION,
            ScenarioType.S4_FLOOD_CONTROL,
            ScenarioType.S5_ICE_PERIOD,
            ScenarioType.S6_PUMP_FAILURE,
            ScenarioType.S7_PLANNED_MAINT,
            ScenarioType.S8_EMERGENCY_REPAIR,
        ]
        self.assertEqual(len(scenarios), 8)

    def test_control_directive(self):
        """测试ControlDirective"""
        directive = ControlDirective(
            pool_id=30,
            role=PoolRole.ISOLATE,
            target_bias=-0.5,
            weight_multipliers={'W_Q': 1e9, 'W_Z': 0},
            hard_constraints={'Q_out_max': 0},
            priority=10,
        )

        self.assertEqual(directive.pool_id, 30)
        self.assertEqual(directive.role, PoolRole.ISOLATE)
        self.assertEqual(directive.target_bias, -0.5)
        self.assertEqual(directive.get_q_weight(), 1e9)
        self.assertEqual(directive.get_z_weight(), 0)
        self.assertEqual(directive.get_constraint('Q_out_max'), 0)

        # 测试有效性检查
        self.assertTrue(directive.is_active(100))

        directive.effective_time = 200
        self.assertFalse(directive.is_active(100))
        self.assertTrue(directive.is_active(300))

    def test_pool_topology(self):
        """测试PoolTopology"""
        topology = PoolTopology.create_snwd_middle_route()

        # 基本属性
        self.assertEqual(topology.num_pools, 60)
        self.assertEqual(topology.total_length, 1432)
        self.assertEqual(len(topology.regions), 5)
        self.assertGreater(len(topology.special_structures), 0)

        # 区域
        for region in topology.regions:
            self.assertGreater(len(region.pool_ids), 0)
            self.assertGreater(region.length, 0)

        # 渠池配置
        pool = topology.get_pool(30)
        self.assertIsNotNone(pool)
        self.assertGreater(pool.bottom_width, 0)
        self.assertGreater(pool.design_flow, 0)

        # 连接关系
        self.assertEqual(topology.get_upstream(30), [29])
        self.assertEqual(topology.get_downstream(30), [31])

    def test_canal_pool_config_hydraulics(self):
        """测试渠池配置水力计算"""
        config = CanalPoolConfig(
            pool_id=0,
            length=20,
            bottom_width=15,
            side_slope=2.5,
            manning_n=0.014,
        )

        # 断面面积
        area = config.cross_section_area(4.0)
        self.assertGreater(area, 0)
        expected_area = (15 + 2.5 * 4) * 4  # (B + m*y) * y
        self.assertAlmostEqual(area, expected_area, places=2)

        # 曼宁流量
        flow = config.manning_flow(4.0)
        self.assertGreater(flow, 0)

        # 正常水深
        depth = config.normal_depth(200)
        self.assertGreater(depth, 0)
        self.assertLess(depth, config.max_depth)


# ==============================================================================
# 测试类: 物理模型
# ==============================================================================

class TestPhysicsModel(unittest.TestCase):
    """物理模型测试"""

    def test_idz_parameters(self):
        """测试IDZ参数"""
        params = IDZParameters(tau=14400, A_s=100000)

        self.assertEqual(params.tau, 14400)
        self.assertEqual(params.A_s, 100000)
        self.assertEqual(params.integrator_gain, 1e-5)

        # 根据长度计算tau
        tau_calc = params.compute_tau_from_length(20, velocity=1.2)
        self.assertAlmostEqual(tau_calc, 20000/1.2, places=0)

    def test_idz_model(self):
        """测试IDZ模型"""
        params = IDZParameters(tau=3600, A_s=50000)  # 1小时滞后
        idz = IDZModel(params, dt=900)

        idz.reset(initial_level=4.0, initial_inflow=100)

        # 稳态测试: 入流=出流，水位不变
        for _ in range(10):
            level = idz.step(100, 100)
        self.assertAlmostEqual(level, 4.0, places=1)

        # 阶跃响应: 入流增加
        idz.reset(4.0, 100)
        for _ in range(20):
            level = idz.step(150, 100)

        # 水位应上升
        self.assertGreater(level, 4.0)

    def test_canal_pool(self):
        """测试渠池模型"""
        config = CanalPoolConfig(
            pool_id=0,
            length=20,
            bottom_width=15,
            design_flow=200,
        )
        pool = CanalPool(config, dt=900)

        pool.reset(initial_level=4.0, initial_flow=200)

        # 运行几步
        for _ in range(10):
            level, outflow = pool.step(200, gate_opening=0.8)

        self.assertGreater(level, 0)
        self.assertLess(level, config.max_depth)
        self.assertGreaterEqual(outflow, 0)

    def test_snwd_model(self):
        """测试全线模型"""
        model = SNWDMiddleRouteModel()
        model.reset(initial_level=4.0, initial_flow=300)

        # 运行24步 (6小时)
        for _ in range(24):
            state = model.step(source_inflow=300)

        # 检查结果
        summary = model.get_summary()
        self.assertEqual(summary['num_pools'], 60)
        self.assertGreater(summary['total_volume'], 0)

        levels = model.get_all_levels()
        self.assertEqual(len(levels), 60)
        self.assertTrue(np.all(levels > 0))
        self.assertTrue(np.all(levels < 10))

    def test_special_node(self):
        """测试特殊节点"""
        structure = SpecialStructure(
            structure_id="test",
            structure_type=StructureType.INVERTED_SIPHON,
            chainage=700,
            max_flow=350,
            head_loss_coefficient=0.5,
            delay_time=1800,
        )

        node = SpecialNode(structure, dt=900)
        node.reset(initial_flow=300)

        # 处理流量
        q_out, head_loss = node.process_flow(300)

        self.assertLessEqual(q_out, 350)
        self.assertGreaterEqual(head_loss, 0)

    def test_seasonal_conditions(self):
        """测试季节性条件"""
        model = SNWDMiddleRouteModel()
        model.reset()

        # 设置冰期
        model.set_seasonal_conditions('ice')

        # 检查糙率是否增加
        pool = model.pools[30]
        self.assertGreater(pool.config.manning_n, 0.014)


# ==============================================================================
# 测试类: 系统辨识
# ==============================================================================

class TestSystemIdentification(unittest.TestCase):
    """系统辨识测试"""

    def test_cross_correlation(self):
        """测试互相关分析"""
        analyzer = CrossCorrelationAnalyzer(max_lag=50)

        # 生成测试信号
        n = 200
        true_delay = 10
        signal1 = np.sin(np.linspace(0, 4*np.pi, n))
        signal2 = np.roll(signal1, true_delay) + np.random.randn(n) * 0.1

        # 计算延迟
        delay, corr = analyzer.find_delay(signal1, signal2, dt=1.0)

        # 延迟应接近真实值
        self.assertLess(abs(delay - true_delay), 3)
        self.assertGreater(corr, 0.5)

    def test_rls_estimator(self):
        """测试递推最小二乘"""
        rls = RecursiveLeastSquares(num_params=1, forgetting_factor=0.98)

        # 生成数据: y = 2 * x + noise
        true_theta = 2.0
        for i in range(100):
            x = np.random.randn()
            y = true_theta * x + np.random.randn() * 0.1

            rls.update(y, np.array([x]))

        # 估计值应接近真实值
        estimate = rls.get_estimate()[0]
        self.assertLess(abs(estimate - true_theta), 0.3)

    def test_system_identifier(self):
        """测试系统辨识器"""
        identifier = SystemIdentifier(dt=900)

        # 生成模拟数据
        np.random.seed(42)
        n = 100
        true_tau = 10800  # 3小时
        true_A_s = 80000

        q_in = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, n))
        q_out = np.roll(q_in, int(true_tau / 900)) * 0.98
        z = np.zeros(n)
        z[0] = 4.0

        for i in range(1, n):
            dz = (q_in[i-1] - q_out[i]) * 900 / true_A_s
            z[i] = z[i-1] + dz

        # 辨识
        for i in range(n):
            identifier.add_sample(0, q_in[i], q_out[i], z[i], i * 900)

        result = identifier.identify_idz_parameters(0)

        self.assertIsNotNone(result)
        self.assertGreater(result.confidence, 0)


# ==============================================================================
# 测试类: 全局编排器
# ==============================================================================

class TestOrchestrator(unittest.TestCase):
    """全局编排器测试"""

    def setUp(self):
        self.orchestrator = GlobalOrchestrator()

    def test_scenario_role_matrix(self):
        """测试场景-角色矩阵"""
        # 污染场景
        mapping = ScenarioRoleMatrix.get_mapping(ScenarioType.S3_POLLUTION)
        self.assertEqual(mapping.center_role, PoolRole.ISOLATE)
        self.assertEqual(mapping.upstream_role, PoolRole.BUFFER)
        self.assertEqual(mapping.downstream_role, PoolRole.DRAIN)

        # 冰期场景
        mapping = ScenarioRoleMatrix.get_mapping(ScenarioType.S5_ICE_PERIOD)
        self.assertEqual(mapping.center_role, PoolRole.STABLE)

    def test_normal_event_processing(self):
        """测试常规事件处理"""
        event = ScenarioEvent(
            event_id="E001",
            scenario_type=ScenarioType.S1_NORMAL_PLAN,
            location=30,
            severity=ScenarioSeverity.LOW,
        )

        plan = self.orchestrator.process_event(event)

        self.assertIsNotNone(plan)
        self.assertGreater(plan.num_directives, 0)
        self.assertEqual(plan.scenario, ScenarioType.S1_NORMAL_PLAN)

    def test_pollution_event_processing(self):
        """测试污染事件处理"""
        event = ScenarioEvent(
            event_id="E002",
            scenario_type=ScenarioType.S3_POLLUTION,
            location=30,
            severity=ScenarioSeverity.CRITICAL,
        )

        plan = self.orchestrator.process_event(event)

        # 检查中心指令
        center_directive = plan.get_directive(30)
        self.assertIsNotNone(center_directive)
        self.assertEqual(center_directive.role, PoolRole.ISOLATE)

        # 检查上游指令
        upstream_directive = plan.get_directive(29)
        self.assertIsNotNone(upstream_directive)
        self.assertEqual(upstream_directive.role, PoolRole.BUFFER)

        # 检查下游指令
        downstream_directive = plan.get_directive(31)
        self.assertIsNotNone(downstream_directive)
        self.assertEqual(downstream_directive.role, PoolRole.DRAIN)

    def test_flood_event_processing(self):
        """测试防洪事件处理"""
        event = ScenarioEvent(
            event_id="E003",
            scenario_type=ScenarioType.S4_FLOOD_CONTROL,
            location=20,
            severity=ScenarioSeverity.HIGH,
        )

        plan = self.orchestrator.process_event(event)

        # 全线应有指令
        self.assertEqual(plan.num_directives, 60)

        # 目标偏移应为负 (预泄)
        directive = plan.get_directive(20)
        self.assertLess(directive.target_bias, 0)

    def test_multiple_events(self):
        """测试多事件处理"""
        events = [
            ScenarioEvent("E1", ScenarioType.S1_NORMAL_PLAN, 30, ScenarioSeverity.LOW),
            ScenarioEvent("E2", ScenarioType.S3_POLLUTION, 30, ScenarioSeverity.CRITICAL),
        ]

        for event in events:
            plan = self.orchestrator.process_event(event)

        stats = self.orchestrator.get_statistics()
        self.assertEqual(stats['events_processed'], 2)
        self.assertEqual(stats['plans_generated'], 2)


# ==============================================================================
# 测试类: 区域协调器
# ==============================================================================

class TestRegionalCoordinator(unittest.TestCase):
    """区域协调器测试"""

    def test_feedforward_decoupler(self):
        """测试前馈解耦器"""
        decoupler = FeedforwardDecoupler(num_pools=60, dt=900)

        # 设置IDZ参数
        for i in range(60):
            decoupler.set_idz_params(i, IDZParameters(tau=7200, A_s=50000))

        # 模拟上游流量变化
        for t in range(20):
            flow = 100 + 20 * (t / 20)  # 递增
            decoupler.update_upstream_flow(30, flow, t * 900)

        # 计算前馈
        ff = decoupler.compute_feedforward(30)

        # 应该有非零前馈 (因为流量在变化)
        self.assertNotEqual(ff, 0)

    def test_regional_coordinator(self):
        """测试区域协调器"""
        topology = PoolTopology.create_snwd_middle_route()
        region = topology.regions[2]  # 河南段北

        coord = RegionalCoordinator(region, topology, dt=900)

        n = len(region.pool_ids)
        levels = np.ones(n) * 4.0
        inflows = np.ones(n) * 100
        outflows = np.ones(n) * 100

        coord.update_state(levels, inflows, outflows)

        gates, outflows_cmd = coord.compute_control()

        self.assertEqual(len(gates), n)
        self.assertEqual(len(outflows_cmd), n)
        self.assertTrue(np.all(gates >= 0))
        self.assertTrue(np.all(gates <= 1))

    def test_global_regional_manager(self):
        """测试全线区域管理器"""
        manager = GlobalRegionalManager()

        # 更新状态
        levels = np.ones(60) * 4.0
        inflows = np.ones(60) * 100
        outflows = np.ones(60) * 100

        manager.update_all_states(levels, inflows, outflows)

        # 计算控制
        all_gates, all_outflows = manager.get_all_commands()

        self.assertEqual(len(all_gates), 60)
        self.assertEqual(len(all_outflows), 60)

        summary = manager.get_summary()
        self.assertEqual(summary['num_regions'], 5)


# ==============================================================================
# 测试类: 增强MPC
# ==============================================================================

class TestEnhancedMPC(unittest.TestCase):
    """增强MPC测试"""

    def test_role_parameter_mapper(self):
        """测试角色参数映射"""
        base_weights = MPCWeights()
        base_constraints = MPCConstraints()

        # ISOLATE角色
        weights = RoleParameterMapper.get_weights_for_role(PoolRole.ISOLATE, base_weights)
        constraints = RoleParameterMapper.get_constraints_for_role(PoolRole.ISOLATE, base_constraints)

        self.assertEqual(weights.W_level, 0)  # 放弃水位控制
        self.assertGreater(weights.W_flow, 1000)  # 高流量权重
        self.assertEqual(constraints.Q_max, 0)  # 关闸

        # STABLE角色
        weights = RoleParameterMapper.get_weights_for_role(PoolRole.STABLE, base_weights)
        self.assertGreater(weights.W_smooth, base_weights.W_smooth)  # 更平滑

    def test_enhanced_mpc_basic(self):
        """测试基本MPC求解"""
        mpc = EnhancedParameterizedMPC(pool_id=30, horizon=10, dt=900)

        q_in, q_out, success = mpc.solve(current_level=4.0, q_in_prev=100)

        self.assertTrue(success or mpc.last_solve_status == "simple")
        self.assertGreaterEqual(q_out, 0)

    def test_role_switching(self):
        """测试角色切换"""
        mpc = EnhancedParameterizedMPC(pool_id=30, horizon=10)

        # 初始为TRANSMIT
        self.assertEqual(mpc.current_role, PoolRole.TRANSMIT)

        # 切换到ISOLATE
        mpc.apply_role(PoolRole.ISOLATE)
        self.assertEqual(mpc.current_role, PoolRole.ISOLATE)

        # 求解
        q_in, q_out, _ = mpc.solve(4.0, 100)
        self.assertEqual(q_out, 0)  # ISOLATE应关闸

    def test_directive_application(self):
        """测试指令应用"""
        mpc = EnhancedParameterizedMPC(pool_id=30, horizon=10)

        directive = ControlDirective(
            pool_id=30,
            role=PoolRole.BUFFER,
            target_bias=0.5,
            weight_multipliers={'W_Q': 0.1},
        )

        mpc.apply_directive(directive)

        self.assertEqual(mpc.current_role, PoolRole.BUFFER)
        self.assertEqual(mpc.Z_ref, 4.5)  # 4.0 + 0.5

    def test_hot_reconfigurable_mpc(self):
        """测试热重构MPC管理器"""
        manager = HotReconfigurableMPC(num_pools=60, horizon=10)

        # 重构为污染场景
        directives = {
            30: ControlDirective(pool_id=30, role=PoolRole.ISOLATE),
        }

        manager.reconfigure_for_scenario(
            scenario=ScenarioType.S3_POLLUTION,
            center_pool=30,
            directives=directives
        )

        # 求解
        levels = np.ones(60) * 4.0
        flows = np.ones(60) * 100

        Q_in, Q_out = manager.solve_all(levels, flows)

        # 池30应为0出流
        self.assertEqual(Q_out[30], 0)

        summary = manager.get_summary()
        self.assertEqual(summary['current_scenario'], 'S3_pollution')


# ==============================================================================
# 测试类: 端到端集成测试
# ==============================================================================

class TestEndToEndIntegration(unittest.TestCase):
    """端到端集成测试"""

    def setUp(self):
        self.model = SNWDMiddleRouteModel()
        self.orchestrator = GlobalOrchestrator()
        self.mpc_manager = HotReconfigurableMPC(num_pools=60, horizon=10)

    def test_normal_operation(self):
        """测试常规运行"""
        self.model.reset(initial_level=4.0, initial_flow=300)

        # 运行10步进行快速测试
        for step in range(10):
            levels = self.model.get_all_levels()
            _, flows = self.model.get_all_flows()

            # 构建简单的闸门命令 (使用简单PI控制而非完整MPC)
            gate_commands = {}
            for i in range(60):
                # 简单控制: 如果水位高于目标，开大闸门；否则关小
                level_error = levels[i] - 4.0
                gate_commands[i] = min(1.0, max(0.0, 0.75 + level_error * 0.1))

            state = self.model.step(source_inflow=300, gate_commands=gate_commands)

        summary = self.model.get_summary()

        # 验证系统正常运行，水位在合理范围内
        self.assertGreater(summary['avg_level'], 0.0)
        self.assertLess(summary['avg_level'], 10.0)

    def test_pollution_scenario(self):
        """测试污染场景"""
        self.model.reset()

        # 触发污染事件
        event = ScenarioEvent(
            event_id="P001",
            scenario_type=ScenarioType.S3_POLLUTION,
            location=30,
            severity=ScenarioSeverity.CRITICAL,
            timestamp=0,
        )

        plan = self.orchestrator.process_event(event)

        # 验证污染计划生成正确
        self.assertIsNotNone(plan)
        self.assertEqual(plan.scenario, ScenarioType.S3_POLLUTION)

        # 检查池30应被分配为ISOLATE角色
        directive_30 = plan.get_directive(30)
        self.assertIsNotNone(directive_30)
        self.assertEqual(directive_30.role, PoolRole.ISOLATE)

        # 应用到MPC
        self.mpc_manager.apply_plan(plan)

        # 验证MPC控制器已应用正确角色
        controller_30 = self.mpc_manager.controllers[30]
        self.assertEqual(controller_30.current_role, PoolRole.ISOLATE)

        # 测试MPC求解 - ISOLATE角色应输出0
        levels = self.model.get_all_levels()
        _, flows = self.model.get_all_flows()
        _, Q_out = self.mpc_manager.solve_all(levels, flows)

        # 池30的出流应为0 (隔离) - 使用近似比较处理浮点误差
        self.assertAlmostEqual(Q_out[30], 0, places=5)

    def test_flood_scenario(self):
        """测试防洪场景"""
        self.model.reset(initial_level=4.5, initial_flow=350)

        # 触发防洪事件
        event = ScenarioEvent(
            event_id="F001",
            scenario_type=ScenarioType.S4_FLOOD_CONTROL,
            location=20,
            severity=ScenarioSeverity.HIGH,
        )

        plan = self.orchestrator.process_event(event)

        # 验证防洪计划生成正确
        self.assertIsNotNone(plan)
        self.assertEqual(plan.scenario, ScenarioType.S4_FLOOD_CONTROL)

        # 检查池应该收到DRAIN角色
        drain_count = sum(1 for d in plan.directives.values() if d.role == PoolRole.DRAIN)
        self.assertGreater(drain_count, 0)

        # 检查target_bias应该是负的（降低目标水位）
        negative_bias_count = sum(1 for d in plan.directives.values() if d.target_bias < 0)
        self.assertGreater(negative_bias_count, 0)

        # 快速验证: 运行少量步骤，确保计划可以被执行
        for step in range(5):
            gate_commands = {}
            for pool_id, directive in plan.directives.items():
                if directive.target_bias < 0:  # 预泄
                    gate_commands[pool_id] = 1.0
            self.model.step(source_inflow=200, gate_commands=gate_commands)

        # 验证模型仍在正常运行
        summary = self.model.get_summary()
        self.assertGreater(summary['avg_level'], 0.0)

    def test_ice_period_scenario(self):
        """测试冰期场景"""
        self.model.reset()

        # 设置冰期条件
        self.model.set_seasonal_conditions('ice')

        # 触发冰期事件
        event = ScenarioEvent(
            event_id="I001",
            scenario_type=ScenarioType.S5_ICE_PERIOD,
            location=0,
            severity=ScenarioSeverity.MEDIUM,
        )

        plan = self.orchestrator.process_event(event)

        # 所有池应为STABLE角色
        for pool_id in range(60):
            directive = plan.get_directive(pool_id)
            self.assertEqual(directive.role, PoolRole.STABLE)

    def test_multi_region_coordination(self):
        """测试多区域协调"""
        manager = GlobalRegionalManager()
        self.model.reset()

        # 运行少量步骤进行快速测试
        for step in range(5):
            levels = self.model.get_all_levels()
            inflows, outflows = self.model.get_all_flows()

            manager.update_all_states(levels, inflows, outflows)
            all_gates, all_outflows = manager.get_all_commands()

            gate_commands = {i: all_gates[i] for i in range(60)}
            self.model.step(gate_commands=gate_commands)

        summary = manager.get_summary()
        self.assertEqual(summary['num_regions'], 5)


# ==============================================================================
# 测试类: 性能测试
# ==============================================================================

class TestPerformance(unittest.TestCase):
    """性能测试"""

    def test_model_simulation_speed(self):
        """测试模型仿真速度"""
        import time

        model = SNWDMiddleRouteModel()
        model.reset()

        start = time.time()
        for _ in range(1000):
            model.step(source_inflow=300)
        elapsed = time.time() - start

        # 1000步应在10秒内完成
        self.assertLess(elapsed, 10.0)
        logger.info(f"1000步仿真耗时: {elapsed:.2f}s")

    def test_mpc_solve_speed(self):
        """测试MPC求解速度"""
        import time

        mpc = EnhancedParameterizedMPC(pool_id=0, horizon=10)

        start = time.time()
        for _ in range(100):
            mpc.solve(4.0, 100)
        elapsed = time.time() - start

        # 100次求解应在30秒内完成（根据环境调整阈值）
        self.assertLess(elapsed, 30.0)
        logger.info(f"100次MPC求解耗时: {elapsed:.2f}s")


# ==============================================================================
# 测试运行器
# ==============================================================================

def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestCoreTypes,
        TestPhysicsModel,
        TestSystemIdentification,
        TestOrchestrator,
        TestRegionalCoordinator,
        TestEnhancedMPC,
        TestEndToEndIntegration,
        TestPerformance,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出摘要
    print("\n" + "="*70)
    print("测试摘要")
    print("="*70)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    return result


if __name__ == "__main__":
    run_all_tests()
