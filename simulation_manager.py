import numpy as np
from brain import SemanticInterpreter
from physics import CanalPoolSimulator
from control import UniversalMPCSolver

class SimulationManager:
    def __init__(self, total_hours, dt, area, initial_level, script, demands):
        self.total_hours = total_hours
        self.dt = dt
        self.area = area
        self.initial_level = initial_level
        self.script = script
        self.demands = demands
        self.last_control_action = 5.0  # Assume steady state start

        self.brain = SemanticInterpreter()
        self.physics = CanalPoolSimulator(
            area=self.area,
            dt=self.dt,
            delay_steps=1,
            initial_level=self.initial_level,
            initial_flow=self.last_control_action
        )
        self.solver = UniversalMPCSolver(horizon=10, dt=self.dt, area=self.area, delay_steps=1)

        self.history = {
            'time': [],
            'level': [],
            'q_in': [],
            'q_out': [],
            'target_level': [],
            'instruction': [],
            'config': []
        }

    def run_simulation(self):
        print("Starting Simulation...")
        current_instruction = self.script[0][1]
        last_control_action = 5.0  # Assume steady state start

        for t in range(self.total_hours):
            # A. Check Script for new instruction
            for start_time, instruction in self.script:
                if t == start_time:
                    current_instruction = instruction
                    print(f"[Time {t}h] New Instruction: {current_instruction}")
                    break

            # B. Brain: Interpret Instruction
            config = self.brain.interpret(current_instruction)

            # C. Solver: Calculate Optimal Control
            q_out_forecast = self.demands[t : t + self.solver.N]
            current_level = self.physics.get_level()

            q_in_cmd = self.solver.solve(
                current_level=current_level,
                q_prev=last_control_action,
                q_out_forecast=q_out_forecast,
                config=config
            )


            # D. Physics: Execute Step
            q_out_actual = self.demands[t]
            self.physics.step(q_in_command=q_in_cmd, q_out=q_out_actual)

            # E. Log Data
            self.log_data(t, current_level, q_in_cmd, q_out_actual, config, current_instruction)

            last_control_action = q_in_cmd
        
        print("Simulation Complete.")
        return self.history

    def log_data(self, t, current_level, q_in_cmd, q_out_actual, config, current_instruction):
        self.history['time'].append(t)
        self.history['level'].append(current_level)
        self.history['q_in'].append(q_in_cmd)
        self.history['q_out'].append(q_out_actual)
        self.history['target_level'].append(config['Z_ref'])
        self.history['instruction'].append(current_instruction)
        self.history['config'].append(config)
