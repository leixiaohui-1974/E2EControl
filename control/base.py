import cvxpy as cp
import numpy as np

class UniversalMPCSolver:
    """
    Decision Layer: Universal MPC Solver.

    A generic Model Predictive Control solver using CVXPY.
    The structure remains constant; optimization objectives and constraints
    are dynamically updated based on the configuration provided by the Brain layer.
    """

    def __init__(self, horizon=10, dt=3600.0, area=10000.0, delay_steps=1):
        """
        Initialize the MPC Solver.

        Args:
            horizon (int): Prediction horizon (N).
            dt (float): Time step in seconds.
            area (float): Surface area of the pool.
            delay_steps (int): System delay (tau).
        """
        self.N = horizon
        self.dt = dt
        self.area = area
        self.tau = delay_steps

        # Physical constraints (Hardware limits, static)
        self.Q_cap = 20.0  # Max flow capacity
        self.Z_min = 0.0   # Min water level
        self.Z_max = 10.0  # Max water level

    def solve(self, current_level, q_prev, q_out_forecast, config):
        """
        Solve the optimization problem for the next control action.

        Args:
            current_level (float): Current water level Z(k).
            q_prev (float): The previous control action Q_in(k-1).
                            Used for smoothness cost and delay handling if needed.
                            (Strictly speaking, for delay=1, we might need more history,
                             but here we assume we just need q_prev for the delta constraint
                             and we might approximate the delay or state propagation).
            q_out_forecast (list/array): Forecasted outflow demand for the next N steps.
            config (dict): Configuration from the Semantic Interpreter (Brain).

        Returns:
            float: Optimal Q_in(k).
        """

        # Unpack configuration
        W_level = config.get('W_level', 10.0)
        W_smooth = config.get('W_smooth', 5.0)
        Z_ref = config.get('Z_ref', 3.0)
        delta_Q_max = config.get('delta_Q_max', 2.0)

        # dynamic constraints
        custom_constraints = config.get('constraints', {})
        Q_in_max_dyn = custom_constraints.get('Q_in_max', self.Q_cap)

        # CVXPY Variables
        # Q_in for k, k+1, ..., k+N-1
        Q = cp.Variable(self.N)
        # Z for k+1, ..., k+N
        Z = cp.Variable(self.N)

        cost = 0
        constraints = []

        # Initial State
        Z_curr = current_level
        Q_prev_val = q_prev

        # Build the problem over the horizon
        for k in range(self.N):
            # 1. State Update (Dynamics)
            # V(k+1) = V(k) + (Q_in_effective - Q_out) * dt
            # Level Z = V / Area
            # Z(k+1) = Z(k) + (Q_in_effective - Q_out) * dt / Area

            # Handling Delay:
            # If tau=1, the Q applied at step k affects Z at step k+2.
            # Z(1) depends on Q_applied_at_k_minus_1 (which is q_prev).
            # Z(2) depends on Q(0) (which is Q[0]).

            # For this PoC, let's strictly model the delay if possible, or simplifying.
            # V(k+1) = V(k) + [Q_in(k-tau) - Q_out(k)] * dt

            # Forecast Q_out
            q_out_k = q_out_forecast[k] if k < len(q_out_forecast) else q_out_forecast[-1]

            if k < self.tau:
                # For the first 'tau' steps, the inflow is determined by history.
                # Since we only pass q_prev (Q_in(k-1)), we assume tau=1.
                # So for k=0 (predicting Z(1)), input is Q_in(-1) i.e., q_prev.

                # If tau > 1, we would need more history. Assuming tau=1 here as per physics default.
                if self.tau == 1 and k == 0:
                    effective_Qin = Q_prev_val
                else:
                    # Fallback or if tau=0
                    effective_Qin = Q[k - self.tau] # This index would be negative if not careful
                    # But we are in the 'k < tau' block.
                    # This block is tricky without full history.
                    # Let's assume tau=1.
                    pass
            else:
                # k >= tau.
                # e.g., if tau=1, k=1. effective_Qin = Q[1-1] = Q[0].
                effective_Qin = Q[k - self.tau]

            # Re-evaluating dynamics construction for CVXPY to be clean.
            # Let's define the sequence of inflows affecting the tank:
            # Inputs: Q[0], Q[1], ...
            # History: q_prev
            # The inflows entering the tank at steps 0, 1, 2... are:
            # Step 0 (affects Z1): q_prev (if tau=1)
            # Step 1 (affects Z2): Q[0]
            # Step 2 (affects Z3): Q[1]
            # ...

            if k == 0:
                # Z[0] is Z(k+1).
                # inflow is q_prev (delayed).
                constraints.append(Z[k] == Z_curr + (q_prev - q_out_k) * self.dt / self.area)
            else:
                # Z[k] is Z(k+1+k_index) -> Z_next
                # Z[k-1] is Z_current for this step
                # Inflow is Q[k-1] because of delay tau=1.
                constraints.append(Z[k] == Z[k-1] + (Q[k-1] - q_out_k) * self.dt / self.area)

            # 2. Cost Function
            # Minimize deviations from reference
            cost += W_level * cp.square(Z[k] - Z_ref)

            # Minimize action smoothness
            # (Q(k) - Q(k-1))^2
            if k == 0:
                cost += W_smooth * cp.square(Q[k] - Q_prev_val)
            else:
                cost += W_smooth * cp.square(Q[k] - Q[k-1])

            # 3. Constraints
            # Level constraints
            z_min_dyn = custom_constraints.get('Z_min', self.Z_min)
            z_max_dyn = custom_constraints.get('Z_max', self.Z_max)

            constraints.append(Z[k] >= z_min_dyn)
            constraints.append(Z[k] <= z_max_dyn)

            # Flow capacity constraints
            constraints.append(Q[k] >= 0)
            constraints.append(Q[k] <= Q_in_max_dyn) # Dynamic upper bound

            # Rate of Change constraints (Delta Q)
            if k == 0:
                constraints.append(cp.abs(Q[k] - Q_prev_val) <= delta_Q_max)
            else:
                constraints.append(cp.abs(Q[k] - Q[k-1]) <= delta_Q_max)

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # Use a robust solver if available, else default
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            # Fallback strategy: keep previous flow or zero?
            # Or return None and handle in main.
            # For PoC, let's print and return q_prev (safe mode).
            print(f"[Solver] Warning: Optimization status {prob.status}. Maintaining previous flow.")
            return q_prev

        # Return the first optimal action Q(k) = Q[0].
        # Note: Even though Q[0] only affects Z(2) due to delay, we must commit to it now at step k.
        return Q.value[0]
