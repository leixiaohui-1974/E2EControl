"""
Phase 2 集成测试
测试所有新功能的集成
"""

import sys
sys.path.append('..')

import unittest
import numpy as np
from models.cascaded_system import CascadedCanalSystem
from controllers.distributed_mpc import DistributedMPCController
from controllers.improved_admm import ImprovedDistributedMPC, ADMMParameters
from controllers.multi_objective_mpc import MultiObjectiveMPC, OptimizationWeights
from controllers.feedforward_control import FeedforwardController, IntegratedFeedforwardMPC
from topology.network_topology import (
    WaterNetworkTopology, NetworkNode, NetworkEdge, 
    NodeType, EdgeType, create_simple_cascade, create_complex_network
)


class TestCascadedSystem(unittest.TestCase):
    """测试级联系统"""
    
    def setUp(self):
        self.system = CascadedCanalSystem(num_pools=3, dt=3600.0)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.system.num_pools, 3)
        self.assertEqual(len(self.system.pools), 3)
        self.assertEqual(len(self.system.gates), 4)  # n+1个闸门
    
    def test_step_simulation(self):
        """测试仿真步进"""
        control_actions = [0.5] * 4
        state = self.system.step(control_actions, demand=5.0)
        
        self.assertIn('time', state)
        self.assertIn('pools', state)
        self.assertEqual(len(state['pools']), 3)
    
    def test_state_vector(self):
        """测试状态向量"""
        state_vector = self.system.get_state_vector()
        self.assertEqual(len(state_vector), 7)  # 3个水位 + 4个流量


class TestNetworkTopology(unittest.TestCase):
    """测试网络拓扑"""
    
    def setUp(self):
        self.topology = create_simple_cascade(num_pools=3)
    
    def test_node_count(self):
        """测试节点数量"""
        pools = self.topology.get_pools()
        gates = self.topology.get_gates()
        
        self.assertEqual(len(pools), 3)
        self.assertEqual(len(gates), 4)
    
    def test_upstream_downstream(self):
        """测试上下游关系"""
        pools = self.topology.get_pools()
        
        for i, pool in enumerate(pools):
            upstream = self.topology.get_upstream_nodes(pool)
            downstream = self.topology.get_downstream_nodes(pool)
            
            self.assertGreater(len(upstream), 0, f"池{i}应该有上游节点")
            if i < len(pools) - 1:
                self.assertGreater(len(downstream), 0, f"池{i}应该有下游节点")
    
    def test_validation(self):
        """测试拓扑验证"""
        valid, errors = self.topology.validate()
        # 简单级联可能有警告但结构应该有效
        self.assertIsInstance(valid, bool)
        self.assertIsInstance(errors, list)
    
    def test_control_sequence(self):
        """测试控制顺序"""
        sequence = self.topology.get_control_sequence()
        self.assertGreater(len(sequence), 0)
    
    def test_complex_network(self):
        """测试复杂网络"""
        complex_topo = create_complex_network()
        
        pools = complex_topo.get_pools()
        self.assertGreater(len(pools), 3)
        
        # 检查是否有泵站
        pump_nodes = [nid for nid, node in complex_topo.nodes.items()
                     if node.node_type == NodeType.PUMP]
        self.assertGreater(len(pump_nodes), 0)


class TestDistributedMPC(unittest.TestCase):
    """测试分布式MPC"""
    
    def setUp(self):
        self.controller = DistributedMPCController(num_pools=3, horizon=5)
    
    def test_solve(self):
        """测试求解"""
        current_levels = [3.0, 3.0, 3.0]
        q_in_prevs = [5.0, 5.0, 5.0]
        q_out_forecasts = [[5.0]*5] * 3
        
        solutions = self.controller.solve(current_levels, q_in_prevs, q_out_forecasts)
        
        self.assertEqual(len(solutions), 3)
        for q_in, q_out in solutions:
            self.assertGreater(q_in, 0)
            self.assertGreater(q_out, 0)


class TestImprovedADMM(unittest.TestCase):
    """测试改进的ADMM"""
    
    def setUp(self):
        params = ADMMParameters(rho=1.0, adaptive_rho=True, max_iterations=10)
        self.controller = ImprovedDistributedMPC(num_pools=3, horizon=5, params=params)
    
    def test_solve_with_info(self):
        """测试求解并返回信息"""
        current_levels = [3.1, 2.9, 3.0]
        q_in_prevs = [5.0, 5.0, 5.0]
        q_out_forecasts = [[5.0]*5] * 3
        
        solutions, info = self.controller.solve(current_levels, q_in_prevs, q_out_forecasts)
        
        self.assertEqual(len(solutions), 3)
        self.assertIn('converged', info)
        self.assertIn('iterations', info)
        self.assertIn('solve_time', info)
        self.assertGreater(info['solve_time'], 0)
    
    def test_convergence(self):
        """测试收敛性"""
        current_levels = [3.0, 3.0, 3.0]
        q_in_prevs = [5.0, 5.0, 5.0]
        q_out_forecasts = [[5.0]*5] * 3
        
        solutions, info = self.controller.solve(current_levels, q_in_prevs, q_out_forecasts)
        
        # 在简单场景下应该收敛
        self.assertTrue(info['converged'], "ADMM应该收敛")
        self.assertLess(info['iterations'], 15, "迭代次数应该合理")


class TestMultiObjectiveMPC(unittest.TestCase):
    """测试多目标MPC"""
    
    def setUp(self):
        self.controller = MultiObjectiveMPC(pool_id=0, horizon=10)
    
    def test_weighted_sum(self):
        """测试加权求和"""
        q_in, q_out, costs = self.controller.solve(
            current_level=3.2,
            q_in_prev=5.0,
            demand_forecast=[5.0]*10,
            mode='weighted_sum'
        )
        
        self.assertGreater(q_in, 0)
        self.assertGreater(q_out, 0)
        self.assertIsInstance(costs, dict)
    
    def test_adaptive_weights(self):
        """测试自适应权重"""
        for _ in range(5):
            q_in, q_out, costs = self.controller.solve(
                current_level=3.0,
                q_in_prev=5.0,
                demand_forecast=[5.0]*10,
                mode='adaptive'
            )
        
        # 权重应该被调整
        metrics = self.controller.get_performance_metrics()
        self.assertGreater(len(metrics), 0)


class TestFeedforwardControl(unittest.TestCase):
    """测试前馈控制"""
    
    def setUp(self):
        self.ff_controller = FeedforwardController(pool_id=0)
        self.ff_mpc = IntegratedFeedforwardMPC(pool_id=0, horizon=10)
    
    def test_feedforward_computation(self):
        """测试前馈计算"""
        from controllers.feedforward_control import DisturbanceForecast
        
        disturbance = DisturbanceForecast(
            upstream_flow=[5.5]*10,
            demand=[5.0]*10,
            rainfall=[0.0]*10,
            evaporation=[0.5]*10,
            horizon=10
        )
        
        current_state = {'q_in': 5.0, 'q_out': 5.0}
        feedforward = self.ff_controller.compute_feedforward(disturbance, current_state)
        
        self.assertEqual(len(feedforward), 10)
        self.assertIsInstance(feedforward, list)
    
    def test_integrated_control(self):
        """测试集成控制"""
        current_state = {'level': 3.0, 'q_in': 5.0, 'q_out': 5.0}
        feedback = 5.0
        
        total_control, debug = self.ff_mpc.compute_control(
            current_state, feedback, current_time=0
        )
        
        self.assertIsInstance(total_control, (int, float))
        self.assertIn('feedback', debug)
        self.assertIn('feedforward', debug)


class TestIntegrationScenarios(unittest.TestCase):
    """测试集成场景"""
    
    def test_three_pool_integration(self):
        """测试3池集成"""
        # 创建系统
        system = CascadedCanalSystem(num_pools=3, dt=3600.0)
        controller = ImprovedDistributedMPC(num_pools=3, horizon=5,
                                           params=ADMMParameters(max_iterations=10))
        
        # 运行仿真
        for t in range(10):
            current_levels = [pool.level for pool in system.pools]
            q_in_prevs = [pool.inflow_history[-1] for pool in system.pools]
            q_out_forecasts = [[5.0]*5] * 3
            
            solutions, info = controller.solve(current_levels, q_in_prevs, q_out_forecasts)
            
            control_actions = [q_in / 20.0 for q_in, _ in solutions] + [0.25]
            state = system.step(control_actions, demand=5.0)
            
            # 验证状态合理性
            for pool_state in state['pools']:
                self.assertGreater(pool_state['level'], 0)
                self.assertLess(pool_state['level'], 10)
    
    def test_performance_requirements(self):
        """测试性能要求"""
        import time
        
        controller = ImprovedDistributedMPC(num_pools=5, horizon=10,
                                           params=ADMMParameters(max_iterations=20))
        
        current_levels = [3.0] * 5
        q_in_prevs = [5.0] * 5
        q_out_forecasts = [[5.0]*10] * 5
        
        start = time.time()
        solutions, info = controller.solve(current_levels, q_in_prevs, q_out_forecasts)
        solve_time = time.time() - start
        
        # 应该在500ms内完成
        self.assertLess(solve_time, 0.5, 
                       f"求解时间{solve_time*1000:.0f}ms超过500ms要求")


def run_tests():
    """运行所有测试"""
    print("="*70)
    print(" "*20 + "Phase 2 集成测试")
    print("="*70)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestCascadedSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkTopology))
    suite.addTests(loader.loadTestsFromTestCase(TestDistributedMPC))
    suite.addTests(loader.loadTestsFromTestCase(TestImprovedADMM))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiObjectiveMPC))
    suite.addTests(loader.loadTestsFromTestCase(TestFeedforwardControl))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 统计结果
    print("\n" + "="*70)
    print("测试结果统计")
    print("="*70)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
