import unittest
import numpy as np
from brain import SemanticInterpreter
from physics import CanalPoolSimulator
from control import UniversalMPCSolver

class TestBrain(unittest.TestCase):
    def setUp(self):
        self.brain = SemanticInterpreter()

    def test_default_scenario(self):
        config = self.brain.interpret("保持水位平稳，正常供水。")
        self.assertEqual(config['Z_ref'], 3.0)
        self.assertEqual(config['W_level'], 10.0)

    def test_flood_warning(self):
        config = self.brain.interpret("收到暴雨预警，立刻降低水位腾出库容！安全第一！")
        self.assertEqual(config['Z_ref'], 2.0)
        self.assertEqual(config['delta_Q_max'], 5.0)

    def test_pollution_emergency(self):
        config = self.brain.interpret("下游检测到污染，紧急切断出流！")
        self.assertIn('Q_in_max', config['constraints'])
        self.assertEqual(config['constraints']['Q_in_max'], 0.0)

    def test_unknown_instruction(self):
        config = self.brain.interpret("Unknown command 123")
        # Should return default
        self.assertEqual(config['Z_ref'], 3.0)

class TestPhysics(unittest.TestCase):
    def setUp(self):
        # Area=10000, dt=3600, delay=1
        self.sim = CanalPoolSimulator(area=10000.0, dt=3600.0, delay_steps=1, initial_level=3.0)

    def test_initial_state(self):
        self.assertEqual(self.sim.get_level(), 3.0)
        self.assertEqual(self.sim.get_volume(), 30000.0)

    def test_step_logic_with_delay(self):
        # Initial: Level 3.0. History Q_in = [0, 0]
        # Step 1: Cmd=10, Q_out=0.
        # Delayed Input for Step 1 comes from History[0] which is 0.
        # So Z shouldn't change yet.
        z1 = self.sim.step(q_in_command=10.0, q_out=0.0)
        self.assertEqual(z1, 3.0)

        # Step 2: Cmd=10, Q_out=0.
        # Delayed Input for Step 2 comes from History[0] (which was updated to 10 in step 1? No.)
        # Let's trace deque.
        # Init: [0, 0]
        # Step 1: append 10. Deque: [0, 10]. q_in_delayed = history[0] = 0.
        # Delta V = (0 - 0) * dt = 0.
        # Step 2: append 10. Deque: [10, 10]. q_in_delayed = history[0] = 10.
        # Delta V = (10 - 0) * 3600 = 36000.
        # New Volume = 30000 + 36000 = 66000.
        # New Level = 6.6.
        z2 = self.sim.step(q_in_command=10.0, q_out=0.0)
        self.assertAlmostEqual(z2, 6.6)

class TestSolver(unittest.TestCase):
    def setUp(self):
        self.solver = UniversalMPCSolver(horizon=5, dt=3600.0, area=10000.0, delay_steps=1)
        self.base_config = {
            'W_level': 10.0, 'W_smooth': 5.0, 'Z_ref': 3.0, 'delta_Q_max': 2.0, 'constraints': {}
        }

    def test_solve_steady_state(self):
        # If already at target, should output Q_in = Q_out
        q_out_forecast = [5.0] * 5
        q_opt = self.solver.solve(current_level=3.0, q_prev=5.0, q_out_forecast=q_out_forecast, config=self.base_config)
        # Should be close to 5.0
        self.assertAlmostEqual(q_opt, 5.0, delta=0.1)

    def test_solve_level_adjustment(self):
        # Level is low (2.0), target is 3.0. Should increase flow.
        q_out_forecast = [5.0] * 5
        q_opt = self.solver.solve(current_level=2.0, q_prev=5.0, q_out_forecast=q_out_forecast, config=self.base_config)
        self.assertGreater(q_opt, 5.0)

    def test_constraints_handling(self):
        # Constraint Q_in_max = 0
        config = self.base_config.copy()
        config['constraints'] = {'Q_in_max': 0.0, 'Z_min': -10.0}
        config['delta_Q_max'] = 20.0  # Allow immediate shutoff
        q_out_forecast = [5.0] * 5
        q_opt = self.solver.solve(current_level=3.0, q_prev=5.0, q_out_forecast=q_out_forecast, config=config)
        self.assertAlmostEqual(q_opt, 0.0, delta=1e-3)

if __name__ == '__main__':
    unittest.main()
