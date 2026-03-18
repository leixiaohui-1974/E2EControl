import numpy as np
from typing import List, Dict
from .base import UniversalMPCSolver

class DistributedMPC:
    """
    Control Layer: Distributed MPC for Cascaded System.
    
    Manages multiple local MPC controllers to coordinate control actions.
    Currently implements a decentralized approach where each controller 
    optimizes its own pool, assuming downstream demand is known.
    """
    
    def __init__(self, num_pools: int, horizon: int = 10, dt: float = 3600.0, area: float = 10000.0):
        self.num_pools = num_pools
        self.controllers: List[UniversalMPCSolver] = []
        
        for _ in range(num_pools):
            self.controllers.append(UniversalMPCSolver(
                horizon=horizon,
                dt=dt,
                area=area,
                delay_steps=1 # Assuming uniform delay for now
            ))
            
    def solve(self, current_levels: List[float], prev_flows: List[float], 
              demand_forecasts: List[List[float]], config: Dict) -> List[float]:
        """
        Calculate optimal gate flows for the entire system.
        
        Args:
            current_levels: List of current water levels [Z0, Z1, ...]
            prev_flows: List of previous gate flows [Q0(k-1), Q1(k-1), ...]
            demand_forecasts: List of demand forecasts for each pool. 
                              Shape: [num_pools, horizon]
            config: Control configuration.
            
        Returns:
            List[float]: Optimal flows for all gates [Q0, Q1, ..., QN].
                         Note: The last gate QN is usually determined by downstream demand 
                         or boundary condition. Here we treat it as a control variable 
                         for the last pool's outflow, or fixed.
        """
        
        # Strategy: Solve from downstream to upstream (Reverse Order)
        # This allows upstream pools to know the flow request from downstream pools.
        # Pool N-1 needs to deliver Q_N (outflow) + Demand. 
        # Q_N might be fixed or optimized. Let's assume Q_N (tail gate) is fixed 
        # to meet some downstream requirement or just free flow.
        # For simplicity in this phase:
        # We solve independently but pass the "inflow request" of Pool i+1 
        # as the "outflow demand" for Pool i.
        
        # However, UniversalMPCSolver takes "q_out_forecast" as a parameter.
        # For Pool i, q_out = Gate i+1 flow + Lateral Demand i.
        # Gate i+1 flow is the INFLOW for Pool i+1.
        
        optimal_flows = [0.0] * (self.num_pools + 1)
        
        # Assume tail gate (Gate N) flow is 0 or fixed base demand for now
        # In a real scenario, this is a boundary condition.
        optimal_flows[-1] = 0.0 # Default tail flow (Closed/No downstream demand)
        
        # Iterate from last pool (N-1) to first pool (0)
        for i in range(self.num_pools - 1, -1, -1):
            # Forecasted outflow for Pool i = (Gate i+1 Flow) + (Lateral Demand i)
            # We need a horizon forecast for Gate i+1. 
            # Since we compute one step at a time here, we might need a full plan.
            # UniversalMPCSolver returns only the NEXT step Q.
            
            # Limitation of current UniversalMPCSolver: it doesn't return the full plan Q[0...N].
            # To do proper coupling, we need the full trajectory.
            # For this Phase 2 implementation, we will use a simplified coupling:
            # We assume Gate i+1 flow is constant over the horizon (equal to its calculated next step),
            # plus the dynamic lateral demand.
            
            # 1. Get Lateral Demand Forecast for Pool i
            lat_demand = demand_forecasts[i]
            
            # 2. Get Gate i+1 Flow (which is Inflow for Pool i+1)
            # For the last pool, it's the tail gate.
            # For others, it's the optimal inflow calculated for the downstream pool.
            next_gate_flow = optimal_flows[i+1]
            
            # Combine to get Total Outflow Forecast for Pool i
            # q_out_total[k] = next_gate_flow + lat_demand[k]
            # (Simplification: assuming next_gate_flow is constant for the horizon)
            total_out_forecast = [d + next_gate_flow for d in lat_demand]
            
            # 3. Solve for Pool i
            # We need Q_in (Gate i)
            q_prev = prev_flows[i]
            
            q_cmd = self.controllers[i].solve(
                current_level=current_levels[i],
                q_prev=q_prev,
                q_out_forecast=total_out_forecast,
                config=config
            )
            
            optimal_flows[i] = q_cmd
            
        return optimal_flows
