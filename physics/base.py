"""
Canal pool physics simulation module.

Provides single-pool and cascaded canal system simulators
with realistic hydraulic delay modeling.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import List, Optional, Sequence, Union

# Physical defaults
DEFAULT_POOL_AREA: float = 10_000.0       # m^2
DEFAULT_TIME_STEP: float = 3_600.0        # seconds (1 hour)
DEFAULT_INITIAL_LEVEL: float = 3.0        # m
DEFAULT_INITIAL_FLOW: float = 0.0         # m^3/s
DEFAULT_DELAY_STEPS: int = 1


class CanalPoolSimulator:
    """Simulates a single canal pool with transport delay.

    The pool is modelled as a flat-bottomed reservoir whose water level
    changes according to the net inflow minus outflow over each time step.

    Attributes:
        area: Cross-sectional area of the pool (m^2).
        dt: Simulation time step (seconds).
        delay_steps: Number of steps of transport delay on inflow.
        current_level: Current water level (m).
    """

    def __init__(
        self,
        area: float = DEFAULT_POOL_AREA,
        dt: float = DEFAULT_TIME_STEP,
        delay_steps: int = DEFAULT_DELAY_STEPS,
        initial_level: float = DEFAULT_INITIAL_LEVEL,
        initial_flow: float = DEFAULT_INITIAL_FLOW,
    ) -> None:
        self.area = area
        self.dt = dt
        self.delay_steps = delay_steps
        self.current_level = initial_level
        self.q_in_history: deque[float] = deque(
            [initial_flow] * (delay_steps + 1), maxlen=delay_steps + 1
        )

    def step(self, q_in_command: float, q_out: float) -> float:
        """Advance the pool by one time step.

        Args:
            q_in_command: Commanded inflow rate (m^3/s).
            q_out: Actual outflow rate (m^3/s).

        Returns:
            Updated water level (m).
        """
        self.q_in_history.append(q_in_command)
        q_in_delayed = self.q_in_history[0]

        delta_v = (q_in_delayed - q_out) * self.dt
        self.current_level += delta_v / self.area

        if self.current_level < 0:
            self.current_level = 0

        return self.current_level

    def get_level(self) -> float:
        """Return the current water level (m)."""
        return self.current_level

    def get_volume(self) -> float:
        """Return the current water volume (level * area) in m^3."""
        return self.current_level * self.area


class CascadedCanalSystem:
    """Simulates a series of cascaded canal pools.

    Each pool's outflow feeds into the next pool as inflow.
    The last pool's outflow is determined by the downstream demand.

    Attributes:
        num_pools: Number of pools in the cascade.
        pools: List of individual pool simulators.
    """

    def __init__(
        self,
        num_pools: int = 3,
        area: float = DEFAULT_POOL_AREA,
        dt: float = DEFAULT_TIME_STEP,
        initial_level: float = DEFAULT_INITIAL_LEVEL,
        initial_flow: float = 5.0,
    ) -> None:
        self.num_pools = num_pools
        self.pools: List[CanalPoolSimulator] = [
            CanalPoolSimulator(
                area=area, dt=dt,
                initial_level=initial_level, initial_flow=initial_flow,
            )
            for _ in range(num_pools)
        ]

    def step(
        self, q_in_commands: Sequence[float], q_out_final: float
    ) -> List[float]:
        """Advance the entire cascaded system by one time step.

        Args:
            q_in_commands: Gate flows into each pool (length == num_pools).
            q_out_final: Outflow from the last pool (downstream demand).

        Returns:
            List of updated water levels for each pool.
        """
        levels: List[float] = []

        for i in range(self.num_pools):
            q_in = q_in_commands[i]
            q_out = (
                q_in_commands[i + 1] if i < self.num_pools - 1 else q_out_final
            )
            level = self.pools[i].step(q_in, q_out)
            levels.append(level)

        return levels

    def get_levels(self) -> List[float]:
        """Return current water levels for all pools."""
        return [p.get_level() for p in self.pools]
