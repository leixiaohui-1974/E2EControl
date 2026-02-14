"""
Decision Layer: Universal MPC Solver.

Provides convex optimization-based Model Predictive Control for
single-pool and multi-pool canal systems using CVXPY.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence, Union

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)

# Controller defaults
DEFAULT_HORIZON: int = 10             # prediction horizon (steps)
DEFAULT_TIME_STEP: float = 3_600.0    # seconds
DEFAULT_POOL_AREA: float = 10_000.0   # m^2
DEFAULT_DELAY_STEPS: int = 1
MAX_FLOW_CAPACITY: float = 20.0       # m^3/s
MIN_WATER_LEVEL: float = 0.0          # m
MAX_WATER_LEVEL: float = 10.0         # m


class UniversalMPCSolver:
    """MPC solver for single or multi-pool canal systems.

    Uses CVXPY to solve a quadratic program that minimises a weighted
    combination of level-tracking error and control-action smoothness
    subject to physical and operational constraints.

    Attributes:
        N: Prediction horizon (number of steps).
        dt: Time step (seconds).
        area: Pool cross-sectional area (m^2).
        tau: Transport delay (steps).
        Q_cap: Maximum flow capacity (m^3/s).
        Z_min: Minimum allowable water level (m).
        Z_max: Maximum allowable water level (m).
    """

    def __init__(
        self,
        horizon: int = DEFAULT_HORIZON,
        dt: float = DEFAULT_TIME_STEP,
        area: float = DEFAULT_POOL_AREA,
        delay_steps: int = DEFAULT_DELAY_STEPS,
    ) -> None:
        self.N = horizon
        self.dt = dt
        self.area = area
        self.tau = delay_steps
        self.Q_cap = MAX_FLOW_CAPACITY
        self.Z_min = MIN_WATER_LEVEL
        self.Z_max = MAX_WATER_LEVEL

    def solve(
        self,
        current_level: Union[float, np.ndarray],
        q_prev: Union[float, np.ndarray],
        q_out_forecast: Sequence[float],
        config: Dict[str, Any],
    ) -> Union[float, np.ndarray]:
        """Solve the MPC optimization for the next control action(s).

        Handles both single-pool (scalar inputs) and multi-pool
        (vector inputs) cases transparently.

        Args:
            current_level: Current water level(s) (m).
            q_prev: Previous control action(s) (m^3/s).
            q_out_forecast: Forecast of outflow over the horizon.
            config: MPC configuration from SemanticInterpreter.

        Returns:
            Optimal inflow command(s) for the next step.
        """
        current_level = np.atleast_1d(current_level)
        q_prev = np.atleast_1d(q_prev)
        num_pools = len(current_level)

        W_level = config.get('W_level', 10.0)
        W_smooth = config.get('W_smooth', 5.0)
        Z_ref = np.full(num_pools, config.get('Z_ref', 3.0))
        delta_Q_max = config.get('delta_Q_max', 2.0)
        custom_constraints = config.get('constraints', {})
        Q_in_max_dyn = custom_constraints.get('Q_in_max', self.Q_cap)

        # CVXPY Variables
        Q = cp.Variable((self.N, num_pools))
        Z = cp.Variable((self.N, num_pools))

        cost = 0
        constraints = []

        for k in range(self.N):
            cost += W_level * cp.sum_squares(Z[k, :] - Z_ref)

            if k == 0:
                cost += W_smooth * cp.sum_squares(Q[k, :] - q_prev)
            else:
                cost += W_smooth * cp.sum_squares(Q[k, :] - Q[k - 1, :])

            for i in range(num_pools):
                q_out_k = (
                    q_out_forecast[k]
                    if k < len(q_out_forecast)
                    else q_out_forecast[-1]
                )
                outflow_i = (
                    Q[k, i + 1] if i < num_pools - 1 else q_out_k
                )

                if k == 0:
                    constraints.append(
                        Z[k, i] == current_level[i]
                        + (q_prev[i] - outflow_i) * self.dt / self.area
                    )
                else:
                    constraints.append(
                        Z[k, i] == Z[k - 1, i]
                        + (Q[k - 1, i] - outflow_i) * self.dt / self.area
                    )

                z_min_dyn = custom_constraints.get('Z_min', self.Z_min)
                constraints.append(Z[k, i] >= z_min_dyn)
                constraints.append(Z[k, i] <= self.Z_max)
                constraints.append(Q[k, i] >= 0)
                constraints.append(Q[k, i] <= Q_in_max_dyn)

                if k == 0:
                    constraints.append(
                        cp.abs(Q[k, i] - q_prev[i]) <= delta_Q_max
                    )
                else:
                    constraints.append(
                        cp.abs(Q[k, i] - Q[k - 1, i]) <= delta_Q_max
                    )

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if prob.status in ("infeasible", "unbounded"):
            logger.warning(
                "Optimization status %s. Maintaining previous flow.",
                prob.status,
            )
            return q_prev

        optimal_action = Q.value[0, :]
        return optimal_action if num_pools > 1 else float(optimal_action[0])
