"""
Simulation orchestration module.

Manages the end-to-end simulation loop: brain interpretation,
MPC optimization, physics stepping, and data logging.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from brain import SemanticInterpreter
from physics import CanalPoolSimulator
from control import UniversalMPCSolver

# Defaults
DEFAULT_INITIAL_FLOW: float = 5.0  # m^3/s - assumed steady-state start


class SimulationManager:
    """Orchestrates a full simulation run.

    Attributes:
        total_hours: Number of hours to simulate.
        dt: Time step in seconds.
        area: Pool area in m^2.
        initial_level: Starting water level in m.
        script: List of (time, instruction) tuples.
        demands: Array of outflow demands per hour.
    """

    def __init__(
        self,
        total_hours: int,
        dt: float,
        area: float,
        initial_level: float,
        script: Sequence[Tuple[int, str]],
        demands: np.ndarray,
    ) -> None:
        self.total_hours = total_hours
        self.dt = dt
        self.area = area
        self.initial_level = initial_level
        self.script = script
        self.demands = demands

        self.brain = SemanticInterpreter()
        self.physics = CanalPoolSimulator(
            area=self.area,
            dt=self.dt,
            delay_steps=1,
            initial_level=self.initial_level,
            initial_flow=DEFAULT_INITIAL_FLOW,
        )
        self.solver = UniversalMPCSolver(
            horizon=10, dt=self.dt, area=self.area, delay_steps=1,
        )

        self.history: Dict[str, List[Any]] = {
            'time': [],
            'level': [],
            'q_in': [],
            'q_out': [],
            'target_level': [],
            'instruction': [],
            'config': [],
        }

    def run_simulation(self) -> Dict[str, List[Any]]:
        """Execute the simulation loop.

        Returns:
            Dictionary of recorded time-series data.
        """
        logger.info("Starting simulation (%d hours)", self.total_hours)
        current_instruction = self.script[0][1]
        last_control_action = DEFAULT_INITIAL_FLOW

        for t in range(self.total_hours):
            # A. Check script for new instruction
            for start_time, instruction in self.script:
                if t == start_time:
                    current_instruction = instruction
                    logger.info("[T=%dh] New instruction: %s", t, current_instruction)
                    break

            # B. Brain: interpret instruction
            config = self.brain.interpret(current_instruction)

            # C. Solver: calculate optimal control
            q_out_forecast = self.demands[t: t + self.solver.N]
            current_level = self.physics.get_level()

            q_in_cmd = self.solver.solve(
                current_level=current_level,
                q_prev=last_control_action,
                q_out_forecast=q_out_forecast,
                config=config,
            )

            # D. Physics: execute step
            q_out_actual = self.demands[t]
            self.physics.step(q_in_command=q_in_cmd, q_out=q_out_actual)

            # E. Log data
            self._log_data(
                t, current_level, q_in_cmd, q_out_actual,
                config, current_instruction,
            )

            last_control_action = q_in_cmd

        logger.info("Simulation complete")
        return self.history

    def _log_data(
        self,
        t: int,
        current_level: float,
        q_in_cmd: float,
        q_out_actual: float,
        config: Dict[str, Any],
        current_instruction: str,
    ) -> None:
        """Append a single time step to the history."""
        self.history['time'].append(t)
        self.history['level'].append(current_level)
        self.history['q_in'].append(q_in_cmd)
        self.history['q_out'].append(q_out_actual)
        self.history['target_level'].append(config['Z_ref'])
        self.history['instruction'].append(current_instruction)
        self.history['config'].append(config)
