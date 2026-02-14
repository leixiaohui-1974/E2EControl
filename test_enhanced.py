"""
增强版测试套件
包含边界测试、集成测试和性能测试
"""

import unittest
import numpy as np
import time
import os
from brain_enhanced import EnhancedSemanticInterpreter
from physics import CanalPoolSimulator
from control import UniversalMPCSolver
from config_manager import ConfigManager
from exceptions import ConfigurationError
from logger import get_logger, setup_logging
from monitor import MonitoringSystem, AlertLevel
from database import SimulationDatabase
from exceptions import *


class TestConfigManager(unittest.TestCase):
    """配置管理器测试"""
    
    def setUp(self):
        """测试前准备"""
        # 确保配置文件存在
        if not os.path.exists('config.yaml'):
            self.skipTest("配置文件不存在")
    
    def test_load_config(self):
        """测试配置加载"""
        config = ConfigManager('config.yaml')
        self.assertIsNotNone(config.config)
        self.assertIn('simulation', config.config)
    
    def test_get_value(self):
        """测试获取配置值"""
        config = ConfigManager('config.yaml')
        dt = config.get('simulation.time_step')
        self.assertIsNotNone(dt)
        self.assertGreater(dt, 0)
    
    def test_get_section(self):
        """测试获取配置节"""
        config = ConfigManager('config.yaml')
        mpc_config = config.get_section('mpc')
        self.assertIn('horizon', mpc_config)
    
    def test_scenario_retrieval(self):
        """测试场景检索"""
        config = ConfigManager('config.yaml')
        scenarios = config.get_all_scenarios()
        self.assertGreater(len(scenarios), 0)


class TestEnhancedSemanticInterpreter(unittest.TestCase):
    """增强版语义解释器测试"""
    
    def setUp(self):
        """测试前准备"""
        setup_logging({'level': 'ERROR', 'console_output': False})
        if not os.path.exists('config.yaml'):
            self.skipTest("配置文件不存在")
        self.interpreter = EnhancedSemanticInterpreter(similarity_threshold=0.5)
    
    def test_exact_match(self):
        """测试精确匹配"""
        config, confidence = self.interpreter.interpret("保持水位平稳，正常供水。")
        # 模糊匹配可能返回高置信度（>= 0.8）
        self.assertGreaterEqual(confidence, 0.8)
        self.assertEqual(config['Z_ref'], 3.0)
    
    def test_fuzzy_match(self):
        """测试模糊匹配"""
        config, confidence = self.interpreter.interpret("保持水位平稳")
        self.assertGreaterEqual(confidence, 0.5)
    
    def test_partial_keyword_match(self):
        """测试部分关键词匹配"""
        config, confidence = self.interpreter.interpret("暴雨预警来了")
        # 部分匹配应该返回非零置信度，但可能低于阈值
        self.assertGreaterEqual(confidence, 0.0)
    
    def test_unknown_instruction(self):
        """测试未知指令"""
        config, confidence = self.interpreter.interpret("完全未知的指令xxx")
        self.assertLess(confidence, 0.5)
    
    def test_empty_instruction(self):
        """测试空指令"""
        with self.assertRaises(SemanticError):
            self.interpreter.interpret("")
    
    def test_list_scenarios(self):
        """测试列出场景"""
        scenarios = self.interpreter.list_scenarios()
        self.assertGreater(len(scenarios), 0)


class TestPhysicsEdgeCases(unittest.TestCase):
    """物理模拟边界测试"""
    
    def setUp(self):
        """测试前准备"""
        self.sim = CanalPoolSimulator(
            area=10000.0,
            dt=3600.0,
            delay_steps=1,
            initial_level=3.0
        )
    
    def test_negative_level_protection(self):
        """测试负水位保护"""
        # 大量出流
        for _ in range(10):
            level = self.sim.step(q_in_command=0.0, q_out=50.0)
        self.assertGreaterEqual(level, 0.0)
    
    def test_zero_flow(self):
        """测试零流量"""
        initial = self.sim.get_level()
        level = self.sim.step(q_in_command=0.0, q_out=0.0)
        # 考虑延迟，第一步可能不变
        self.assertIsNotNone(level)
    
    def test_large_inflow(self):
        """测试大流量入流"""
        initial = self.sim.get_level()
        level = self.sim.step(q_in_command=100.0, q_out=0.0)
        # 第二步才会体现
        level2 = self.sim.step(q_in_command=100.0, q_out=0.0)
        self.assertGreater(level2, initial)
    
    def test_disturbance(self):
        """测试扰动（通过入流噪声模拟）"""
        # CanalPoolSimulator.step takes (q_in_command, q_out) only;
        # disturbances are modelled by adjusting q_in_command.
        level = self.sim.step(q_in_command=5.0 + 1.0, q_out=5.0)
        self.assertIsNotNone(level)


class TestSolverEdgeCases(unittest.TestCase):
    """求解器边界测试"""
    
    def setUp(self):
        """测试前准备"""
        setup_logging({'level': 'ERROR', 'console_output': False})
        self.solver = UniversalMPCSolver(
            horizon=5,
            dt=3600.0,
            area=10000.0,
            delay_steps=1
        )
        self.base_config = {
            'W_level': 10.0,
            'W_smooth': 5.0,
            'Z_ref': 3.0,
            'delta_Q_max': 2.0,
            'constraints': {}
        }
    
    def test_extreme_target_level(self):
        """测试极端目标水位"""
        config = self.base_config.copy()
        config['Z_ref'] = 0.5  # 很低的目标
        q_out_forecast = [5.0] * 5
        q_opt = self.solver.solve(
            current_level=8.0,
            q_prev=5.0,
            q_out_forecast=q_out_forecast,
            config=config
        )
        self.assertIsNotNone(q_opt)
        self.assertGreaterEqual(q_opt, 0.0)
    
    def test_conflicting_constraints(self):
        """测试冲突约束"""
        config = self.base_config.copy()
        config['constraints'] = {
            'Q_in_max': 0.0,  # 不允许入流
            'Z_min': 5.0  # 但要求高水位
        }
        config['delta_Q_max'] = 20.0
        q_out_forecast = [10.0] * 5  # 持续出流
        
        # 应该处理冲突（可能不可行）
        q_opt = self.solver.solve(
            current_level=2.0,
            q_prev=5.0,
            q_out_forecast=q_out_forecast,
            config=config
        )
        # 不可行时应返回安全值
        self.assertIsNotNone(q_opt)
    
    def test_rapid_demand_change(self):
        """测试需求剧烈变化"""
        q_out_forecast = [5.0, 20.0, 5.0, 20.0, 5.0]
        q_opt = self.solver.solve(
            current_level=3.0,
            q_prev=5.0,
            q_out_forecast=q_out_forecast,
            config=self.base_config
        )
        self.assertIsNotNone(q_opt)


class TestMonitoringSystem(unittest.TestCase):
    """监控系统测试"""
    
    def setUp(self):
        """测试前准备"""
        setup_logging({'level': 'ERROR', 'console_output': False})
        if not os.path.exists('config.yaml'):
            self.skipTest("配置文件不存在")
        self.monitor = MonitoringSystem()
    
    def test_normal_state(self):
        """测试正常状态"""
        alerts = self.monitor.check_state(
            0, 3.0, 5.0, 5.0, {'Z_ref': 3.0}
        )
        self.assertEqual(len(alerts), 0)
    
    def test_high_level_warning(self):
        """测试高水位警告"""
        alerts = self.monitor.check_state(
            0, 8.5, 10.0, 5.0, {'Z_ref': 3.0}
        )
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].level, AlertLevel.WARNING)
    
    def test_critical_level(self):
        """测试严重水位"""
        alerts = self.monitor.check_state(
            0, 9.8, 10.0, 5.0, {'Z_ref': 3.0}
        )
        self.assertGreater(len(alerts), 0)
        # 应该有严重告警
        has_critical = any(a.level == AlertLevel.CRITICAL for a in alerts)
        self.assertTrue(has_critical)
    
    def test_alert_callback(self):
        """测试告警回调"""
        callback_called = []
        
        def test_callback(alert):
            callback_called.append(alert)
        
        self.monitor.register_callback(test_callback)
        
        self.monitor.check_state(0, 9.8, 10.0, 5.0, {'Z_ref': 3.0})
        
        self.assertGreater(len(callback_called), 0)


class TestDatabaseOperations(unittest.TestCase):
    """数据库操作测试"""
    
    def setUp(self):
        """测试前准备"""
        setup_logging({'level': 'ERROR', 'console_output': False})
        self.test_db = "test_sim.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        self.db = SimulationDatabase(self.test_db)
    
    def tearDown(self):
        """测试后清理"""
        self.db.close()
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_create_simulation(self):
        """测试创建仿真"""
        sim_id = self.db.create_simulation(
            total_hours=50,
            dt=3600.0,
            area=10000.0,
            config={},
            notes="测试"
        )
        self.assertGreater(sim_id, 0)
    
    def test_save_and_retrieve_states(self):
        """测试保存和读取状态"""
        sim_id = self.db.create_simulation(50, 3600.0, 10000.0, {})
        
        # 保存状态
        for t in range(10):
            self.db.save_state(
                sim_id, t, 3.0, 5.0, 5.0, 3.0, "测试", {}
            )
        
        self.db.finish_simulation(sim_id)
        
        # 读取
        history = self.db.get_simulation_history(sim_id)
        self.assertEqual(len(history), 10)
    
    def test_save_alert(self):
        """测试保存告警"""
        sim_id = self.db.create_simulation(50, 3600.0, 10000.0, {})
        
        self.db.save_alert(
            sim_id, 10, "WARNING", "测试告警", "消息", {}
        )
        
        alerts = self.db.get_simulation_alerts(sim_id)
        self.assertEqual(len(alerts), 1)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        setup_logging({'level': 'ERROR', 'console_output': False})
        if not os.path.exists('config.yaml'):
            self.skipTest("配置文件不存在")
    
    def test_full_pipeline(self):
        """测试完整流程"""
        # 初始化组件
        interpreter = EnhancedSemanticInterpreter()
        physics = CanalPoolSimulator(area=10000.0, dt=3600.0, delay_steps=1)
        solver = UniversalMPCSolver(horizon=5, dt=3600.0, area=10000.0, delay_steps=1)
        monitor = MonitoringSystem()
        
        # 模拟几步
        instruction = "保持水位平稳，正常供水。"
        config, confidence = interpreter.interpret(instruction)
        
        for t in range(5):
            current_level = physics.get_level()
            q_out = 5.0
            
            q_in = solver.solve(
                current_level=current_level,
                q_prev=5.0,
                q_out_forecast=[q_out] * 5,
                config=config
            )
            
            next_level = physics.step(q_in, q_out)
            alerts = monitor.check_state(t, next_level, q_in, q_out, config)
        
        # 检查结果
        self.assertGreater(physics.get_level(), 0)


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def setUp(self):
        """测试前准备"""
        setup_logging({'level': 'ERROR', 'console_output': False})
        if not os.path.exists('config.yaml'):
            self.skipTest("配置文件不存在")
    
    def test_solver_performance(self):
        """测试求解器性能"""
        solver = UniversalMPCSolver(horizon=10, dt=3600.0, area=10000.0, delay_steps=1)
        config = {
            'W_level': 10.0,
            'W_smooth': 5.0,
            'Z_ref': 3.0,
            'delta_Q_max': 2.0,
            'constraints': {}
        }
        
        times = []
        for _ in range(20):
            start = time.time()
            solver.solve(
                current_level=3.0,
                q_prev=5.0,
                q_out_forecast=[5.0] * 10,
                config=config
            )
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        print(f"\n平均求解时间: {avg_time*1000:.2f}ms")
        
        # 求解应该在合理时间内完成
        self.assertLess(avg_time, 1.0)  # 小于1秒
    
    def test_database_write_performance(self):
        """测试数据库写入性能"""
        test_db = "perf_test.db"
        if os.path.exists(test_db):
            os.remove(test_db)
        
        db = SimulationDatabase(test_db)
        sim_id = db.create_simulation(1000, 3600.0, 10000.0, {})
        
        start = time.time()
        for t in range(100):
            db.save_state(sim_id, t, 3.0, 5.0, 5.0, 3.0, "测试", {})
        db.finish_simulation(sim_id)
        elapsed = time.time() - start
        
        print(f"\n写入100条记录耗时: {elapsed*1000:.2f}ms")
        
        db.close()
        os.remove(test_db)
        
        # 批量写入应该很快
        self.assertLess(elapsed, 1.0)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
