"""
Comprehensive tests for core modules:
- brain.py (SemanticInterpreter)
- config_manager.py (ConfigManager)
- simulation_manager.py (SimulationManager)
- monitor.py (MonitoringSystem)
- database.py (SimulationDatabase)
- physics/base.py (CanalPoolSimulator, CascadedCanalSystem)
- exceptions.py
"""

import os
import sys
import json
import tempfile
import unittest

import numpy as np
import yaml

from hydroe2e.brain import SemanticInterpreter, DEFAULT_W_LEVEL, DEFAULT_W_SMOOTH, DEFAULT_Z_REF
from hydroe2e.config_manager import ConfigManager, get_config, _StubConfigManager
from hydroe2e.physics.base import CanalPoolSimulator, CascadedCanalSystem
from hydroe2e.exceptions import (
    SmartPoolException, ConfigurationError, OptimizationError,
    PhysicsError, SemanticError, ValidationError,
)


# =========================================================================
# SemanticInterpreter Tests
# =========================================================================

class TestSemanticInterpreter(unittest.TestCase):
    """Tests for brain.py SemanticInterpreter."""

    def setUp(self):
        self.brain = SemanticInterpreter()

    # --- Default behaviour ---

    def test_default_config_returned_for_neutral_instruction(self):
        """Neutral text should return near-default values."""
        config = self.brain.interpret("测试指令")
        self.assertEqual(config['Z_ref'], DEFAULT_Z_REF)
        self.assertEqual(config['constraints'], {})

    def test_interpret_returns_all_keys(self):
        """Result dict must always contain the five standard keys."""
        config = self.brain.interpret("任意文本")
        for key in ('W_level', 'W_smooth', 'Z_ref', 'delta_Q_max', 'constraints'):
            self.assertIn(key, config)

    def test_interpret_does_not_mutate_defaults(self):
        """Calling interpret twice must not pollute the default config."""
        self.brain.interpret("紧急提升水位")
        config2 = self.brain.interpret("测试")
        self.assertEqual(config2['Z_ref'], DEFAULT_Z_REF)
        self.assertEqual(config2['constraints'], {})

    # --- Keyword effects ---

    def test_keyword_emergency_multiplies_w_level(self):
        config = self.brain.interpret("紧急调度")
        self.assertAlmostEqual(config['W_level'], DEFAULT_W_LEVEL * 10.0)

    def test_keyword_storm_multiplies_w_level(self):
        config = self.brain.interpret("暴雨来了")
        self.assertAlmostEqual(config['W_level'], DEFAULT_W_LEVEL * 10.0)

    def test_keyword_safety_first(self):
        config = self.brain.interpret("安全第一")
        self.assertAlmostEqual(config['W_level'], DEFAULT_W_LEVEL * 5.0)

    def test_keyword_steady_affects_both(self):
        """'平稳' should multiply W_smooth by 8 AND W_level by 1.5."""
        config = self.brain.interpret("保持水位平稳")
        self.assertAlmostEqual(config['W_smooth'], DEFAULT_W_SMOOTH * 8.0)
        self.assertAlmostEqual(config['W_level'], DEFAULT_W_LEVEL * 1.5)

    def test_keyword_reduce_level_sets_z_ref(self):
        config = self.brain.interpret("降低水位")
        self.assertEqual(config['Z_ref'], 2.0)

    def test_keyword_raise_level_sets_z_ref(self):
        config = self.brain.interpret("提升水位")
        self.assertEqual(config['Z_ref'], 4.0)

    def test_keyword_pollution_sets_constraints(self):
        config = self.brain.interpret("检测到污染")
        self.assertEqual(config['constraints']['Q_in_max'], 0.0)

    def test_keyword_immediately_increases_delta_q(self):
        config = self.brain.interpret("立刻执行")
        self.assertAlmostEqual(config['delta_Q_max'], 2.0 * 2.5)

    def test_keyword_slowly_decreases_delta_q(self):
        config = self.brain.interpret("缓慢调整")
        self.assertAlmostEqual(config['delta_Q_max'], 2.0 * 0.5)

    # --- Negation ---

    def test_negation_inverts_multiplier(self):
        config = self.brain.interpret("不要紧急")
        self.assertAlmostEqual(config['W_level'], DEFAULT_W_LEVEL * (1 / 10.0))

    def test_negation_skips_set_actions(self):
        config = self.brain.interpret("避免降低水位")
        self.assertEqual(config['Z_ref'], DEFAULT_Z_REF)  # not changed

    # --- Special keywords ---

    def test_strict_no_disturbance(self):
        """'严禁扰动' should multiply W_smooth by 100."""
        config = self.brain.interpret("严禁扰动冰盖")
        self.assertAlmostEqual(config['W_smooth'], DEFAULT_W_SMOOTH * 100.0)

    # --- Numerical extraction ---

    def test_extract_level_chinese(self):
        config = self.brain.interpret("设定水位5.5")
        self.assertAlmostEqual(config['Z_ref'], 5.5)

    def test_extract_level_english(self):
        config = self.brain.interpret("set level 4.2")
        self.assertAlmostEqual(config['Z_ref'], 4.2)

    # --- Combined scenarios ---

    def test_combined_storm_and_reduce(self):
        config = self.brain.interpret("暴雨预警，降低水位")
        self.assertAlmostEqual(config['W_level'], DEFAULT_W_LEVEL * 10.0)
        self.assertEqual(config['Z_ref'], 2.0)

    def test_full_scenario_steady_supply(self):
        """Full-text scenario: steady supply."""
        config = self.brain.interpret("保持水位平稳，正常供水。")
        self.assertAlmostEqual(config['W_level'], DEFAULT_W_LEVEL * 1.5)
        self.assertAlmostEqual(config['W_smooth'], DEFAULT_W_SMOOTH * 8.0)


# =========================================================================
# ConfigManager Tests
# =========================================================================

class TestConfigManager(unittest.TestCase):
    """Tests for config_manager.py."""

    @classmethod
    def setUpClass(cls):
        cls._valid_config = {
            'simulation': {'total_hours': 10, 'time_step': 3600.0, 'seed': 42},
            'physical_system': {'area': 5000.0, 'initial_level': 2.0},
            'scenario_script': [
                {'time': 0, 'instruction': '正常供水'},
            ],
            'demand_profile': {'base_demand': 5.0, 'noise_std_dev': 0.5},
            'monitoring': {'level_warning_high': 8.0},
            'scenarios': [
                {'name': 'test', 'keywords': ['test'], 'config': {'Z_ref': 3.0}},
            ],
        }

    def _write_config(self, data):
        """Helper: write config to a temp YAML file, return path."""
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, encoding='utf-8',
        )
        yaml.dump(data, f, allow_unicode=True)
        f.close()
        return f.name

    def test_load_valid_config(self):
        path = self._write_config(self._valid_config)
        try:
            cm = ConfigManager(path)
            self.assertEqual(cm.get_simulation_params()['total_hours'], 10)
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ConfigManager('/tmp/nonexistent_e2e_config_xyz.yaml')

    def test_missing_section_raises(self):
        bad = dict(self._valid_config)
        del bad['simulation']
        path = self._write_config(bad)
        try:
            with self.assertRaises(ValueError):
                ConfigManager(path)
        finally:
            os.unlink(path)

    def test_dot_notation_get(self):
        path = self._write_config(self._valid_config)
        try:
            cm = ConfigManager(path)
            self.assertEqual(cm.get('simulation.seed'), 42)
            self.assertIsNone(cm.get('nonexistent.key'))
            self.assertEqual(cm.get('nonexistent', 'fallback'), 'fallback')
        finally:
            os.unlink(path)

    def test_get_section(self):
        path = self._write_config(self._valid_config)
        try:
            cm = ConfigManager(path)
            mon = cm.get_section('monitoring')
            self.assertEqual(mon['level_warning_high'], 8.0)
            self.assertEqual(cm.get_section('nonexistent'), {})
        finally:
            os.unlink(path)

    def test_get_all_scenarios(self):
        path = self._write_config(self._valid_config)
        try:
            cm = ConfigManager(path)
            scenarios = cm.get_all_scenarios()
            self.assertEqual(len(scenarios), 1)
            self.assertEqual(scenarios[0]['name'], 'test')
        finally:
            os.unlink(path)

    def test_get_scenario_script_as_tuples(self):
        path = self._write_config(self._valid_config)
        try:
            cm = ConfigManager(path)
            script = cm.get_scenario_script()
            self.assertEqual(script, [(0, '正常供水')])
        finally:
            os.unlink(path)

    def test_env_override(self):
        path = self._write_config(self._valid_config)
        os.environ['E2E_SIM_SEED'] = '99'
        try:
            cm = ConfigManager(path)
            self.assertEqual(cm.get('simulation.seed'), 99)
        finally:
            del os.environ['E2E_SIM_SEED']
            os.unlink(path)


class TestStubConfigManager(unittest.TestCase):
    """Tests for _StubConfigManager fallback."""

    def setUp(self):
        self.stub = _StubConfigManager()

    def test_get_returns_default(self):
        self.assertIsNone(self.stub.get('anything'))
        self.assertEqual(self.stub.get('k', 42), 42)

    def test_get_section_returns_defaults(self):
        mon = self.stub.get_section('monitoring')
        self.assertIn('level_warning_high', mon)

    def test_get_all_scenarios_not_empty(self):
        scenarios = self.stub.get_all_scenarios()
        self.assertGreater(len(scenarios), 0)

    def test_simulation_params(self):
        params = self.stub.get_simulation_params()
        self.assertIn('total_hours', params)
        self.assertIn('time_step', params)

    def test_physical_params(self):
        params = self.stub.get_physical_params()
        self.assertIn('area', params)

    def test_demand_profile_params(self):
        params = self.stub.get_demand_profile_params()
        self.assertIn('base_demand', params)


# =========================================================================
# CanalPoolSimulator Tests
# =========================================================================

class TestCanalPoolSimulator(unittest.TestCase):
    """Tests for physics/base.py CanalPoolSimulator."""

    def test_initial_state(self):
        pool = CanalPoolSimulator(initial_level=5.0, area=1000.0)
        self.assertEqual(pool.get_level(), 5.0)
        self.assertEqual(pool.get_volume(), 5000.0)

    def test_steady_state_no_change(self):
        """Equal inflow and outflow should keep level constant."""
        pool = CanalPoolSimulator(area=1000.0, dt=1.0, delay_steps=0,
                                  initial_level=3.0, initial_flow=5.0)
        level = pool.step(5.0, 5.0)
        self.assertAlmostEqual(level, 3.0, places=5)

    def test_positive_net_inflow_raises_level(self):
        pool = CanalPoolSimulator(area=1000.0, dt=1.0, delay_steps=0,
                                  initial_level=3.0, initial_flow=0.0)
        level = pool.step(10.0, 5.0)
        self.assertGreater(level, 3.0)

    def test_negative_net_inflow_lowers_level(self):
        pool = CanalPoolSimulator(area=1000.0, dt=1.0, delay_steps=0,
                                  initial_level=3.0, initial_flow=0.0)
        level = pool.step(2.0, 5.0)
        self.assertLess(level, 3.0)

    def test_level_never_goes_negative(self):
        pool = CanalPoolSimulator(area=100.0, dt=1.0, delay_steps=0,
                                  initial_level=0.01, initial_flow=0.0)
        level = pool.step(0.0, 1000.0)
        self.assertEqual(level, 0.0)

    def test_transport_delay(self):
        """With delay_steps=2, inflow should take 2 steps to arrive."""
        pool = CanalPoolSimulator(area=1000.0, dt=1.0, delay_steps=2,
                                  initial_level=3.0, initial_flow=0.0)
        # Step 1: command 10 (delayed by 2, so actual inflow = 0)
        pool.step(10.0, 0.0)
        # Step 2: still delayed
        pool.step(10.0, 0.0)
        level_before = pool.get_level()
        # Step 3: the first command of 10 should now arrive
        pool.step(10.0, 0.0)
        self.assertGreater(pool.get_level(), level_before)


class TestCascadedCanalSystem(unittest.TestCase):
    """Tests for physics/base.py CascadedCanalSystem."""

    def test_correct_number_of_pools(self):
        system = CascadedCanalSystem(num_pools=5)
        self.assertEqual(system.num_pools, 5)
        self.assertEqual(len(system.pools), 5)

    def test_get_levels_returns_correct_count(self):
        system = CascadedCanalSystem(num_pools=3)
        levels = system.get_levels()
        self.assertEqual(len(levels), 3)

    def test_step_returns_all_levels(self):
        system = CascadedCanalSystem(num_pools=3)
        levels = system.step([5.0, 5.0, 5.0], 5.0)
        self.assertEqual(len(levels), 3)
        for lvl in levels:
            self.assertIsInstance(lvl, float)


# =========================================================================
# MonitoringSystem Tests
# =========================================================================

class TestMonitoringSystem(unittest.TestCase):
    """Tests for monitor.py MonitoringSystem."""

    def setUp(self):
        from hydroe2e.monitor import MonitoringSystem
        self.monitor = MonitoringSystem()

    def test_normal_state_no_alerts(self):
        alerts = self.monitor.check_state(0, 3.0, 5.0, 5.0, {'Z_ref': 3.0})
        self.assertEqual(len(alerts), 0)

    def test_high_level_warning(self):
        alerts = self.monitor.check_state(1, 8.5, 5.0, 5.0, {'Z_ref': 3.0})
        types = [a.alert_type for a in alerts]
        self.assertIn('水位偏高', types)

    def test_critical_high_level(self):
        from hydroe2e.monitor import AlertLevel
        alerts = self.monitor.check_state(2, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        self.assertGreater(len(critical), 0)

    def test_low_level_warning(self):
        alerts = self.monitor.check_state(3, 0.3, 5.0, 5.0, {'Z_ref': 3.0})
        types = [a.alert_type for a in alerts]
        self.assertIn('水位严重偏低', types)

    def test_high_flow_warning(self):
        alerts = self.monitor.check_state(4, 3.0, 30.0, 5.0, {'Z_ref': 3.0})
        types = [a.alert_type for a in alerts]
        self.assertIn('入流过大', types)

    def test_deviation_warning(self):
        alerts = self.monitor.check_state(5, 8.0, 5.0, 5.0, {'Z_ref': 3.0})
        types = [a.alert_type for a in alerts]
        self.assertIn('水位偏差过大', types)

    def test_statistics_count(self):
        self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        stats = self.monitor.get_statistics()
        self.assertGreater(stats['total_alerts'], 0)

    def test_get_alerts_with_limit(self):
        for i in range(5):
            self.monitor.check_state(i, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        recent = self.monitor.get_alerts(limit=2)
        self.assertEqual(len(recent), 2)

    def test_clear_alerts(self):
        self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        self.monitor.clear_alerts()
        self.assertEqual(len(self.monitor.alerts), 0)

    def test_register_callback(self):
        callback_fired = []
        self.monitor.register_callback(lambda a: callback_fired.append(a))
        self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        self.assertGreater(len(callback_fired), 0)

    def test_generate_report(self):
        self.monitor.check_state(0, 9.8, 5.0, 5.0, {'Z_ref': 3.0})
        report = self.monitor.generate_report()
        self.assertIn('监控系统报告', report)


# =========================================================================
# SimulationDatabase Tests
# =========================================================================

class TestSimulationDatabase(unittest.TestCase):
    """Tests for database.py SimulationDatabase."""

    def setUp(self):
        from hydroe2e.database import SimulationDatabase
        self.db_file = tempfile.mktemp(suffix='.db')
        self.db = SimulationDatabase(self.db_file)

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_file):
            os.unlink(self.db_file)

    def test_create_simulation(self):
        sid = self.db.create_simulation(10, 3600.0, 10000.0, {'test': True})
        self.assertIsInstance(sid, int)
        self.assertGreater(sid, 0)

    def test_save_and_retrieve_states(self):
        sid = self.db.create_simulation(10, 3600.0, 10000.0, {})
        for t in range(5):
            self.db.save_state(sid, t, 3.0, 5.0, 5.0, 3.0, 'test', {})
        self.db.conn.commit()
        history = self.db.get_simulation_history(sid)
        self.assertEqual(len(history), 5)

    def test_save_and_retrieve_alerts(self):
        sid = self.db.create_simulation(10, 3600.0, 10000.0, {})
        self.db.save_alert(sid, 1, 'WARNING', '测试', '消息', {'key': 'val'})
        alerts = self.db.get_simulation_alerts(sid)
        self.assertEqual(len(alerts), 1)

    def test_save_metric(self):
        sid = self.db.create_simulation(10, 3600.0, 10000.0, {})
        self.db.save_metric(sid, 'RMSE', 0.15, 'm')

    def test_finish_simulation(self):
        sid = self.db.create_simulation(10, 3600.0, 10000.0, {})
        self.db.finish_simulation(sid)
        sims = self.db.get_recent_simulations(1)
        self.assertEqual(len(sims), 1)
        self.assertIsNotNone(sims[0]['end_time'])

    def test_get_recent_simulations_limit(self):
        for _ in range(5):
            self.db.create_simulation(10, 3600.0, 10000.0, {})
        sims = self.db.get_recent_simulations(3)
        self.assertEqual(len(sims), 3)


# =========================================================================
# Exception Hierarchy Tests
# =========================================================================

class TestExceptionHierarchy(unittest.TestCase):
    """Tests for exceptions.py."""

    def test_all_inherit_from_base(self):
        for cls in (ConfigurationError, OptimizationError,
                    PhysicsError, SemanticError, ValidationError):
            self.assertTrue(issubclass(cls, SmartPoolException))

    def test_optimization_error_attrs(self):
        e = OptimizationError("fail", status="infeasible", details={'x': 1})
        self.assertEqual(str(e), "fail")
        self.assertEqual(e.status, "infeasible")
        self.assertEqual(e.details, {'x': 1})

    def test_physics_error_attrs(self):
        e = PhysicsError("bad", state={'level': -1})
        self.assertEqual(e.state, {'level': -1})

    def test_semantic_error_attrs(self):
        e = SemanticError("parse fail", instruction="bad input")
        self.assertEqual(e.instruction, "bad input")

    def test_validation_error_attrs(self):
        e = ValidationError("invalid", field="area", value=-100)
        self.assertEqual(e.field, "area")
        self.assertEqual(e.value, -100)

    def test_base_exception_catchable(self):
        with self.assertRaises(SmartPoolException):
            raise ConfigurationError("test")


# =========================================================================
# SimulationManager Tests
# =========================================================================

class TestSimulationManager(unittest.TestCase):
    """Tests for simulation_manager.py."""

    def test_run_short_simulation(self):
        from hydroe2e.simulation_manager import SimulationManager
        demands = np.full(30, 5.0)
        script = [(0, '保持水位平稳，正常供水。')]
        sm = SimulationManager(
            total_hours=5, dt=3600.0, area=10000.0,
            initial_level=3.0, script=script, demands=demands,
        )
        history = sm.run_simulation()
        self.assertEqual(len(history['time']), 5)
        self.assertEqual(len(history['level']), 5)
        self.assertEqual(len(history['q_in']), 5)

    def test_scenario_switch(self):
        from hydroe2e.simulation_manager import SimulationManager
        demands = np.full(30, 5.0)
        script = [
            (0, '正常供水'),
            (3, '紧急降低水位'),
        ]
        sm = SimulationManager(
            total_hours=5, dt=3600.0, area=10000.0,
            initial_level=3.0, script=script, demands=demands,
        )
        history = sm.run_simulation()
        # After step 3, instruction should change
        self.assertEqual(history['instruction'][0], '正常供水')
        self.assertEqual(history['instruction'][3], '紧急降低水位')


if __name__ == '__main__':
    unittest.main(verbosity=2)
