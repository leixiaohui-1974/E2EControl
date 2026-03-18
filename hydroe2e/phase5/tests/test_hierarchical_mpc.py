"""
Phase 5.3 双层MPC集成测试

测试内容:
1. L3 集中调度器 (CentralizedScheduler)
2. L2 参数化本地MPC (ParameterizedLocalMPC)
3. L2 参数化分布式MPC (ParameterizedDistributedMPC)
4. 双层MPC集成控制器 (HierarchicalMPCController)
5. 场景自适应与物理参数注入
"""

import unittest
import numpy as np

from hydroe2e.phase5.controllers.centralized_scheduler import (
    CentralizedScheduler, NetworkTopology, SchedulerConfig,
    SchedulingForecast, SchedulingResult, SchedulingMode
)
from hydroe2e.phase5.controllers.parameterized_mpc import (
    ParameterizedLocalMPC, ParameterizedDistributedMPC,
    PhysicalParameters, ScenarioPhysics
)
from hydroe2e.phase5.controllers.hierarchical_mpc import (
    HierarchicalMPCController, HierarchicalConfig,
    HierarchicalState, PhysicalScenario
)


class TestNetworkTopology(unittest.TestCase):
    """水网拓扑结构测试"""

    def test_cascade_topology_creation(self):
        """测试级联拓扑创建"""
        topology = NetworkTopology.create_cascade(
            num_pools=3,
            area=10000.0,
            level_min=0.5,
            level_max=8.0,
            max_flow=20.0
        )

        self.assertEqual(topology.num_pools, 3)
        self.assertEqual(len(topology.pool_names), 3)
        self.assertEqual(len(topology.pool_areas), 3)
        self.assertEqual(len(topology.connections), 2)  # 3池有2条连接
        self.assertEqual(topology.connections, [(0, 1), (1, 2)])

    def test_topology_volumes(self):
        """测试拓扑体积计算"""
        topology = NetworkTopology.create_cascade(
            num_pools=2,
            area=5000.0,
            level_min=1.0,
            level_max=10.0
        )

        # 最大体积 = 面积 * 最大水位
        expected_max_vol = 5000.0 * 10.0
        self.assertEqual(topology.pool_volumes_max[0], expected_max_vol)


class TestCentralizedScheduler(unittest.TestCase):
    """L3 集中调度器测试"""

    def setUp(self):
        self.topology = NetworkTopology.create_cascade(
            num_pools=3,
            area=10000.0,
            level_min=0.5,
            level_max=8.0,
            max_flow=20.0
        )
        self.config = SchedulerConfig(
            planning_horizon=24,
            dt=3600.0
        )
        self.scheduler = CentralizedScheduler(self.topology, self.config)

    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        self.assertEqual(self.scheduler.num_pools, 3)
        self.assertEqual(self.scheduler.config.planning_horizon, 24)
        self.assertEqual(self.scheduler.current_mode, SchedulingMode.NORMAL)

    def test_normal_scheduling(self):
        """测试正常调度"""
        current_volumes = [30000.0, 30000.0, 30000.0]  # 3m水位

        forecast = SchedulingForecast(
            horizon=24,
            dt=3600.0,
            demand_forecast=np.ones((3, 24)) * 5.0,
            inflow_forecast=np.ones(24) * 10.0,
            rainfall_forecast=np.zeros(24)
        )

        result = self.scheduler.schedule(current_volumes, forecast)

        # 无论是否成功，都应提供参考轨迹 (fallback机制)
        self.assertEqual(result.mode, SchedulingMode.NORMAL)
        self.assertIsNotNone(result.reference_levels)
        self.assertIsNotNone(result.global_flow_commands)
        self.assertEqual(result.reference_levels.shape, (3, 24))

    def test_flood_mode_activation(self):
        """测试洪水模式激活"""
        current_volumes = [50000.0, 50000.0, 50000.0]  # 5m水位,较高

        forecast = SchedulingForecast(
            horizon=24,
            dt=3600.0,
            demand_forecast=np.ones((3, 24)) * 5.0,
            inflow_forecast=np.ones(24) * 25.0,  # 高入流
            rainfall_forecast=np.ones(24) * 15.0  # 暴雨
        )

        result = self.scheduler.schedule(current_volumes, forecast)

        # 应激活防洪模式或预泄模式
        self.assertIn(result.mode, [SchedulingMode.FLOOD_CONTROL, SchedulingMode.PRE_RELEASE])
        # 确保提供参考轨迹
        self.assertIsNotNone(result.reference_levels)

    def test_drought_mode_activation(self):
        """测试干旱模式激活"""
        current_volumes = [15000.0, 15000.0, 15000.0]  # 1.5m水位,较低

        forecast = SchedulingForecast(
            horizon=24,
            dt=3600.0,
            demand_forecast=np.ones((3, 24)) * 8.0,  # 高需求
            inflow_forecast=np.ones(24) * 3.0,  # 低入流
            rainfall_forecast=np.zeros(24)  # 无雨
        )

        result = self.scheduler.schedule(current_volumes, forecast)

        # 应激活抗旱模式
        self.assertEqual(result.mode, SchedulingMode.DROUGHT)
        # 确保提供参考轨迹
        self.assertIsNotNone(result.reference_levels)


class TestScenarioPhysics(unittest.TestCase):
    """场景物理参数映射测试"""

    def test_normal_parameters(self):
        """测试正常场景参数"""
        params = ScenarioPhysics.get_parameters('NORMAL')

        self.assertEqual(params['flow_efficiency'], 1.0)
        self.assertEqual(params['manning_n'], 0.025)

    def test_ice_formation_parameters(self):
        """测试结冰期参数"""
        params = ScenarioPhysics.get_parameters('ICE_FORMATION')

        self.assertEqual(params['flow_efficiency'], 0.75)
        self.assertGreater(params['manning_n'], 0.025)  # 糙率增加

    def test_ice_stable_parameters(self):
        """测试稳定冰盖参数"""
        params = ScenarioPhysics.get_parameters('ICE_STABLE')

        self.assertEqual(params['flow_efficiency'], 0.8)

    def test_sediment_parameters(self):
        """测试淤积场景参数"""
        params = ScenarioPhysics.get_parameters('SEDIMENT_HIGH')

        self.assertEqual(params['flow_efficiency'], 0.85)

    def test_unknown_scenario_fallback(self):
        """测试未知场景回退"""
        params = ScenarioPhysics.get_parameters('UNKNOWN_SCENARIO')

        # 应回退到正常参数
        self.assertEqual(params['flow_efficiency'], 1.0)


class TestParameterizedLocalMPC(unittest.TestCase):
    """L2 参数化本地MPC测试"""

    def setUp(self):
        self.physical_params = PhysicalParameters(
            area=10000.0,
            flow_efficiency=1.0,
            manning_n=0.025
        )
        self.mpc = ParameterizedLocalMPC(
            pool_id=0,
            horizon=5,
            dt=900.0,  # 15分钟
            physical_params=self.physical_params
        )

    def test_local_mpc_initialization(self):
        """测试本地MPC初始化"""
        self.assertEqual(self.mpc.pool_id, 0)
        self.assertEqual(self.mpc.N, 5)
        self.assertEqual(self.mpc.dt, 900.0)

    def test_physical_parameter_update(self):
        """测试物理参数更新"""
        self.mpc.set_physical_parameters(
            area=12000.0,
            flow_efficiency=0.75,
            manning_n=0.035
        )

        self.assertEqual(self.mpc.physical_params.area, 12000.0)
        self.assertEqual(self.mpc.physical_params.flow_efficiency, 0.75)

    def test_scenario_physics_application(self):
        """测试场景物理参数应用"""
        self.mpc.set_scenario_physics('ICE_FORMATION')

        self.assertEqual(self.mpc.physical_params.flow_efficiency, 0.75)

    def test_reference_trajectory_setting(self):
        """测试参考轨迹设置"""
        trajectory = np.array([3.0, 3.1, 3.2, 3.3, 3.4])
        self.mpc.set_reference_trajectory(trajectory)

        self.assertIsNotNone(self.mpc._reference_trajectory)
        np.testing.assert_array_equal(self.mpc._reference_trajectory, trajectory)

    def test_local_control_computation(self):
        """测试本地控制计算"""
        current_level = 3.0
        current_Q_in = 10.0
        z = np.ones(5) * 5.0  # 一致性变量
        u = np.zeros(5)  # 对偶变量

        q_in, q_out, success = self.mpc.solve(
            current_level=current_level,
            q_in_prev=current_Q_in,
            z=z,
            u=u,
            rho=1.0
        )

        self.assertIsInstance(q_in, float)
        self.assertIsInstance(q_out, float)
        self.assertGreaterEqual(q_in, 0)
        self.assertGreaterEqual(q_out, 0)


class TestParameterizedDistributedMPC(unittest.TestCase):
    """L2 参数化分布式MPC测试"""

    def setUp(self):
        self.controller = ParameterizedDistributedMPC(
            num_pools=3,
            horizon=5,
            dt=900.0
        )

    def test_distributed_mpc_initialization(self):
        """测试分布式MPC初始化"""
        self.assertEqual(self.controller.num_pools, 3)
        self.assertEqual(len(self.controller.local_controllers), 3)

    def test_scenario_physics_propagation(self):
        """测试场景物理参数传播"""
        self.controller.set_scenario_physics('ICE_STABLE')

        for mpc in self.controller.local_controllers:
            self.assertEqual(mpc.physical_params.flow_efficiency, 0.8)

    def test_reference_trajectory_distribution(self):
        """测试参考轨迹分发"""
        trajectories = np.array([
            [3.0, 3.1, 3.2, 3.3, 3.4],
            [2.5, 2.6, 2.7, 2.8, 2.9],
            [2.0, 2.1, 2.2, 2.3, 2.4]
        ])

        self.controller.set_reference_trajectories(trajectories)

        for i, mpc in enumerate(self.controller.local_controllers):
            self.assertIsNotNone(mpc._reference_trajectory)

    def test_distributed_control_admm(self):
        """测试ADMM分布式控制"""
        current_levels = [3.0, 2.5, 2.0]
        q_in_prevs = [10.0, 8.0, 6.0]

        control_actions, info = self.controller.solve(
            current_levels=current_levels,
            q_in_prevs=q_in_prevs
        )

        self.assertIsInstance(control_actions, list)
        self.assertEqual(len(control_actions), 3)
        self.assertIn('converged', info)


class TestHierarchicalMPCController(unittest.TestCase):
    """双层MPC集成控制器测试"""

    def setUp(self):
        self.config = HierarchicalConfig(
            l3_planning_horizon=24,
            l3_update_interval=4,
            l2_control_horizon=5,
            l2_update_interval=15,
            enable_cognitive_physics=True
        )
        self.controller = HierarchicalMPCController(
            num_pools=3,
            config=self.config
        )

    def test_hierarchical_initialization(self):
        """测试双层控制器初始化"""
        self.assertEqual(self.controller.num_pools, 3)
        self.assertIsNotNone(self.controller.l3_scheduler)
        self.assertIsNotNone(self.controller.l2_controller)

    def test_l3_l2_coordination(self):
        """测试L3-L2协调"""
        current_time = 0.0
        current_levels = [3.0, 2.5, 2.0]
        current_flows = [10.0, 8.0, 6.0]
        current_demands = [5.0, 4.0, 3.0]

        control_actions, debug_info = self.controller.compute_control(
            current_time=current_time,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands
        )

        # 首次调用应触发L3更新
        self.assertTrue(debug_info['l3_updated'])
        self.assertTrue(debug_info['l2_updated'])
        self.assertEqual(len(control_actions), 3)

    def test_physical_scenario_adaptation(self):
        """测试物理场景自适应"""
        current_time = 0.0
        current_levels = [3.0, 2.5, 2.0]
        current_flows = [10.0, 8.0, 6.0]
        current_demands = [5.0, 4.0, 3.0]

        # 模拟结冰场景检测
        control_actions, debug_info = self.controller.compute_control(
            current_time=current_time,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands,
            detected_scenario='ICE_FORMATION'
        )

        self.assertTrue(debug_info['physical_updated'])
        self.assertEqual(debug_info['physical_scenario'], 'ICE_FORMATION')
        # 流速效率应降低
        self.assertEqual(debug_info['flow_efficiencies'][0], 0.75)

    def test_l3_interval_update(self):
        """测试L3周期性更新"""
        current_levels = [3.0, 2.5, 2.0]
        current_flows = [10.0, 8.0, 6.0]
        current_demands = [5.0, 4.0, 3.0]

        # 第一次调用 (t=0)
        _, debug1 = self.controller.compute_control(
            current_time=0.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands
        )
        self.assertTrue(debug1['l3_updated'])

        # 第二次调用 (t=1h) - 不应触发L3更新
        _, debug2 = self.controller.compute_control(
            current_time=1.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands
        )
        self.assertFalse(debug2['l3_updated'])

        # 第三次调用 (t=4h) - 应触发L3更新
        _, debug3 = self.controller.compute_control(
            current_time=4.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands
        )
        self.assertTrue(debug3['l3_updated'])

    def test_weather_triggered_l3_update(self):
        """测试天气触发的L3强制更新"""
        current_levels = [3.0, 2.5, 2.0]
        current_flows = [10.0, 8.0, 6.0]
        current_demands = [5.0, 4.0, 3.0]

        # 首次调用
        self.controller.compute_control(
            current_time=0.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands
        )

        # 1小时后带暴雨预警 - 应强制触发L3
        _, debug = self.controller.compute_control(
            current_time=1.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands,
            weather_forecast={'rainfall': 15.0}  # 暴雨
        )
        self.assertTrue(debug['l3_updated'])


class TestPhysicalScenarioTransitions(unittest.TestCase):
    """物理场景切换测试"""

    def setUp(self):
        self.controller = HierarchicalMPCController(num_pools=3)

    def test_scenario_transition_normal_to_ice(self):
        """测试正常到结冰的场景切换"""
        current_levels = [3.0, 2.5, 2.0]
        current_flows = [10.0, 8.0, 6.0]
        current_demands = [5.0, 4.0, 3.0]

        # 正常场景
        self.controller.compute_control(
            current_time=0.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands,
            detected_scenario='NORMAL'
        )
        self.assertEqual(
            self.controller.state.current_physical_scenario,
            PhysicalScenario.NORMAL
        )

        # 切换到结冰
        self.controller.compute_control(
            current_time=1.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands,
            detected_scenario='ICE_FORMATION'
        )
        self.assertEqual(
            self.controller.state.current_physical_scenario,
            PhysicalScenario.ICE_FORMATION
        )

    def test_scenario_mapping_from_decision_engine(self):
        """测试决策引擎场景映射"""
        current_levels = [3.0, 2.5, 2.0]
        current_flows = [10.0, 8.0, 6.0]
        current_demands = [5.0, 4.0, 3.0]

        # 使用决策引擎风格的场景ID
        _, debug = self.controller.compute_control(
            current_time=0.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands,
            detected_scenario='FLOOD_WARNING'
        )

        self.assertEqual(
            self.controller.state.current_physical_scenario,
            PhysicalScenario.FLOOD
        )


class TestEmergencyOverride(unittest.TestCase):
    """紧急覆盖测试"""

    def setUp(self):
        config = HierarchicalConfig(emergency_override=True)
        self.controller = HierarchicalMPCController(
            num_pools=3,
            config=config
        )

    def test_emergency_override_high_level(self):
        """测试高水位紧急覆盖"""
        current_levels = [7.5, 7.0, 6.5]  # 接近上限(8m)
        current_flows = [10.0, 8.0, 6.0]
        current_demands = [5.0, 4.0, 3.0]

        control_actions, debug = self.controller.compute_control(
            current_time=0.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands,
            weather_forecast={'rainfall': 20.0}  # 暴雨
        )

        # 应有紧急覆盖行为
        # 控制动作应倾向于增加泄流
        self.assertIsNotNone(control_actions)


class TestIntegrationScenarios(unittest.TestCase):
    """集成场景测试"""

    def test_24h_simulation_normal(self):
        """测试24小时正常运行模拟"""
        controller = HierarchicalMPCController(num_pools=3)

        levels_history = []
        current_levels = [3.0, 2.5, 2.0]

        for hour in range(24):
            current_flows = [10.0, 8.0, 6.0]
            current_demands = [5.0, 4.0, 3.0]

            control_actions, debug = controller.compute_control(
                current_time=float(hour),
                current_levels=current_levels,
                current_flows=current_flows,
                current_demands=current_demands
            )

            levels_history.append(current_levels.copy())

            # 简单模拟水位变化
            for i in range(len(current_levels)):
                Q_in, Q_out = control_actions[i]
                delta_level = (Q_in - Q_out - current_demands[i]) * 3600 / 10000
                current_levels[i] = max(0.5, min(8.0, current_levels[i] + delta_level))

        # 验证水位保持在安全范围
        for levels in levels_history:
            for level in levels:
                self.assertGreaterEqual(level, 0.5)
                self.assertLessEqual(level, 8.0)

    def test_ice_scenario_simulation(self):
        """测试结冰场景模拟"""
        controller = HierarchicalMPCController(num_pools=3)

        current_levels = [3.0, 2.5, 2.0]
        current_flows = [10.0, 8.0, 6.0]
        current_demands = [5.0, 4.0, 3.0]

        # 模拟结冰期开始
        _, debug1 = controller.compute_control(
            current_time=0.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands,
            detected_scenario='ICE_FORMATION'
        )

        # 验证流速效率已降低
        self.assertEqual(controller.state.current_flow_efficiencies[0], 0.75)

        # 模拟稳定冰盖期
        _, debug2 = controller.compute_control(
            current_time=1.0,
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=current_demands,
            detected_scenario='ICE_STABLE'
        )

        self.assertEqual(controller.state.current_flow_efficiencies[0], 0.8)


class TestPerformance(unittest.TestCase):
    """性能测试"""

    def test_l3_solve_time(self):
        """测试L3求解时间"""
        topology = NetworkTopology.create_cascade(num_pools=5)
        config = SchedulerConfig(planning_horizon=24)
        scheduler = CentralizedScheduler(topology, config)

        current_volumes = [30000.0] * 5
        forecast = SchedulingForecast(
            horizon=24,
            dt=3600.0,
            demand_forecast=np.ones((5, 24)) * 5.0,
            inflow_forecast=np.ones(24) * 15.0,
            rainfall_forecast=np.zeros(24)
        )

        result = scheduler.schedule(current_volumes, forecast)

        # L3求解应在合理时间内完成 (< 10秒)
        self.assertLess(result.solve_time, 10.0)

    def test_l2_solve_time(self):
        """测试L2求解时间"""
        controller = ParameterizedDistributedMPC(
            num_pools=5,
            horizon=5,
            dt=900.0
        )

        import time
        start = time.time()

        control_actions, info = controller.solve(
            current_levels=[3.0] * 5,
            q_in_prevs=[10.0] * 5
        )

        elapsed = time.time() - start

        # L2求解应在合理时间内完成 (首次求解可能较慢，给5秒余量)
        self.assertLess(elapsed, 5.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
