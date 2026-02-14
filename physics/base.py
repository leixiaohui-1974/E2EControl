import numpy as np
from collections import deque

class CanalPoolSimulator:
    """Simulates a single canal pool."""
    def __init__(self, area=10000.0, dt=3600.0, delay_steps=1, initial_level=3.0, initial_flow=0.0):
        self.area = area
        self.dt = dt
        self.delay_steps = delay_steps
        self.current_level = initial_level
        self.q_in_history = deque([initial_flow] * (delay_steps + 1), maxlen=delay_steps + 1)

    def step(self, q_in_command, q_out):
        self.q_in_history.append(q_in_command)
        q_in_delayed = self.q_in_history[0]

        delta_v = (q_in_delayed - q_out) * self.dt
        self.current_level += delta_v / self.area

        if self.current_level < 0:
            self.current_level = 0

        return self.current_level

    def get_level(self):
        return self.current_level

    def get_volume(self):
        """Return the current water volume (level * area)."""
        return self.current_level * self.area

class CascadedCanalSystem:
    """Simulates a series of cascaded canal pools."""
    def __init__(self, num_pools=3, area=10000.0, dt=3600.0, initial_level=3.0, initial_flow=5.0):
        self.num_pools = num_pools
        self.pools = [
            CanalPoolSimulator(area=area, dt=dt, initial_level=initial_level, initial_flow=initial_flow)
            for _ in range(num_pools)
        ]

    def step(self, q_in_commands, q_out_final):
        """
        Advances the entire cascaded system by one time step.
        q_in_commands is a list/array of gate flows into each pool.
        """
        levels = []
        q_out = q_in_commands[0] # The outflow of pool 0 is the inflow command for gate 1

        for i in range(self.num_pools):
            q_in = q_in_commands[i]

            # The outflow of the current pool is the inflow to the next
            # For the last pool, the outflow is the final system demand
            q_out = q_in_commands[i+1] if i < self.num_pools - 1 else q_out_final

            level = self.pools[i].step(q_in, q_out)
            levels.append(level)

        return levels

    def get_levels(self):
        return [p.get_level() for p in self.pools]
