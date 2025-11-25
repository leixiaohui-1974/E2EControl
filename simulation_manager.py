import numpy as np
from brain import SemanticInterpreter
from physics.base import CanalPoolSimulator, CascadedCanalSystem
from control.base import UniversalMPCSolver

class SimulationManager:
    def __init__(self, system_type='single', num_pools=3, **kwargs):
        self.system_type = system_type
        self.num_pools = num_pools
        self.kwargs = kwargs

        self.brain = SemanticInterpreter()
        
        if self.system_type == 'single':
            self.physics = CanalPoolSimulator(**kwargs)
            self.initial_level = kwargs.get('initial_level', 3.0)
            self.last_control_action = kwargs.get('initial_flow', 5.0)
        else:
            self.physics = CascadedCanalSystem(num_pools=num_pools, **kwargs)
            self.initial_level = [kwargs.get('initial_level', 3.0)] * num_pools
            self.last_control_action = [kwargs.get('initial_flow', 5.0)] * num_pools
            
        self.solver = UniversalMPCSolver(horizon=10, dt=kwargs.get('dt', 3600.0), area=kwargs.get('area', 10000.0))

        self.history = self._initialize_history()

    def _initialize_history(self):
        history = {
            'time': [],
            'instruction': [],
            'config': [],
            'q_out': []
        }
        if self.system_type == 'single':
            history.update({'level': [], 'q_in': [], 'target_level': []})
        else:
            for i in range(self.num_pools):
                history[f'level_{i}'] = []
                history[f'q_in_{i}'] = []
                history[f'target_level_{i}'] = []
        return history

    def run_simulation(self, total_hours, script, demands):
        print(f"Starting {self.system_type} simulation...")
        
        current_instruction = script[0][1]

        for t in range(total_hours):
            if any(t == s[0] for s in script):
                current_instruction = next(s[1] for s in script if t == s[0])
                print(f"[Time {t}h] New Instruction: {current_instruction}")

            config = self.brain.interpret(current_instruction)
            q_out_forecast = demands[t : t + self.solver.N]
            
            current_level = self.physics.get_levels() if self.system_type == 'cascaded' else self.physics.get_level()

            q_in_cmd = self.solver.solve(
                current_level=current_level,
                q_prev=self.last_control_action,
                q_out_forecast=q_out_forecast,
                config=config
            )

            q_out_actual = demands[t]
            
            if self.system_type == 'cascaded':
                next_levels = self.physics.step(q_in_cmd, q_out_actual)
            else:
                next_levels = self.physics.step(q_in_cmd, q_out_actual)

            self._log_data(t, current_level, q_in_cmd, q_out_actual, config, current_instruction)
            self.last_control_action = q_in_cmd
        
        print("Simulation Complete.")
        return self.history

    def _log_data(self, t, current_level, q_in_cmd, q_out_actual, config, instruction):
        self.history['time'].append(t)
        self.history['instruction'].append(instruction)
        self.history['config'].append(config)
        self.history['q_out'].append(q_out_actual)

        if self.system_type == 'single':
            self.history['level'].append(current_level)
            self.history['q_in'].append(q_in_cmd)
            self.history['target_level'].append(config['Z_ref'])
        else:
            for i in range(self.num_pools):
                self.history[f'level_{i}'].append(current_level[i])
                self.history[f'q_in_{i}'].append(q_in_cmd[i])
                self.history[f'target_level_{i}'].append(config['Z_ref'])
