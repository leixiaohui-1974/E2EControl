import cvxpy as cp
import numpy as np

class UniversalMPCSolver:
    """
    Decision Layer: Universal MPC Solver for single or multi-pool systems.
    """
    def __init__(self, horizon=10, dt=3600.0, area=10000.0, delay_steps=1):
        self.N = horizon
        self.dt = dt
        self.area = area
        self.tau = delay_steps
        self.Q_cap = 20.0
        self.Z_min = 0.0
        self.Z_max = 10.0

    def solve(self, current_level, q_prev, q_out_forecast, config):
        """
        Solves the MPC optimization problem for the next control action(s).
        Can handle both single-pool (scalar inputs) and multi-pool (vector inputs) cases.
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
        Q = cp.Variable((self.N, num_pools)) # Control actions for each pool
        Z = cp.Variable((self.N, num_pools)) # Water levels for each pool

        cost = 0
        constraints = []

        for k in range(self.N):
            # Cost Function
            cost += W_level * cp.sum_squares(Z[k, :] - Z_ref)

            if k == 0:
                cost += W_smooth * cp.sum_squares(Q[k, :] - q_prev)
            else:
                cost += W_smooth * cp.sum_squares(Q[k, :] - Q[k-1, :])

            # Constraints for each pool
            for i in range(num_pools):
                q_out_k = q_out_forecast[k] if k < len(q_out_forecast) else q_out_forecast[-1]

                # System Dynamics (Cascaded)
                # Outflow of pool i is the inflow to pool i+1
                outflow_i = Q[k, i+1] if i < num_pools - 1 else q_out_k

                if k == 0:
                    constraints.append(Z[k, i] == current_level[i] + (q_prev[i] - outflow_i) * self.dt / self.area)
                else:
                    constraints.append(Z[k, i] == Z[k-1, i] + (Q[k-1, i] - outflow_i) * self.dt / self.area)

                # Physical Constraints
                z_min_dyn = custom_constraints.get('Z_min', self.Z_min)
                constraints.append(Z[k, i] >= z_min_dyn)
                constraints.append(Z[k, i] <= self.Z_max)

                constraints.append(Q[k, i] >= 0)
                constraints.append(Q[k, i] <= Q_in_max_dyn)

                if k == 0:
                    constraints.append(cp.abs(Q[k, i] - q_prev[i]) <= delta_Q_max)
                else:
                    constraints.append(cp.abs(Q[k, i] - Q[k-1, i]) <= delta_Q_max)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            print(f"[Solver] Warning: Optimization status {prob.status}. Maintaining previous flow.")
            return q_prev

        # Return the first optimal action vector
        optimal_action = Q.value[0, :]
        return optimal_action if num_pools > 1 else float(optimal_action[0])
