import numpy as np
from collections import deque

class CanalPoolSimulator:
    """
    Physical Twin Layer: Canal Pool Simulator.

    Implements the Integrator Delay Model:
    V(k+1) = V(k) + [Q_in(k-tau) - Q_out(k) + d(k)] * dt

    Assumes a simple linear relationship between Volume (V) and Water Level (Z):
    V = Z * SurfaceArea
    """
    def __init__(self, area=10000.0, dt=3600.0, delay_steps=1, initial_level=3.0, initial_flow=0.0):
        """
        Initialize the simulator.

        Args:
            area (float): Surface area of the pool (m^2).
            dt (float): Time step (seconds).
            delay_steps (int): Time delay tau in number of steps.
            initial_level (float): Initial water level (m).
            initial_flow (float): Initial inflow rate (m^3/s) for history.
        """
        self.area = area
        self.dt = dt
        self.delay_steps = delay_steps

        # State variables
        self.current_volume = initial_level * area
        self.current_level = initial_level

        # History for delay handling
        # We need to store enough past Q_in to retrieve Q_in(k-tau)
        # Initialize with initial_flow.
        self.q_in_history = deque([initial_flow] * (delay_steps + 1), maxlen=delay_steps + 1)

    def step(self, q_in_command, q_out, disturbance=0.0):
        """
        Advance the simulation by one step.

        Args:
            q_in_command (float): The commanded inflow Q_in at current time k.
                                  Note: This will affect the system after tau steps.
            q_out (float): The outflow Q_out at current time k (demand).
            disturbance (float): Random disturbance d(k).

        Returns:
            float: The new water level Z(k+1).
        """
        # Store the current command
        self.q_in_history.append(q_in_command)

        # Retrieve delayed inflow Q_in(k-tau)
        # if delay_steps is 1, we need the value from 1 step ago (index -2 in a length 2 deque?)
        # If delay_steps = 0, we use the current one (index -1).
        # Generally, with a deque of length delay_steps + 1, the value at index 0 is the one from delay_steps ago.
        q_in_delayed = self.q_in_history[0]

        # Calculate Volume change
        # V(k+1) = V(k) + (Q_in(k-tau) - Q_out(k) + d(k)) * dt
        delta_v = (q_in_delayed - q_out + disturbance) * self.dt

        self.current_volume += delta_v

        # Update Level
        self.current_level = self.current_volume / self.area

        # For physical realism, level shouldn't be negative
        if self.current_level < 0:
            self.current_level = 0
            self.current_volume = 0

        return self.current_level

    def get_level(self):
        return self.current_level

    def get_volume(self):
        return self.current_volume
