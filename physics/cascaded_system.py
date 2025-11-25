import numpy as np
from typing import List, Dict
from physics import CanalPoolSimulator

class CascadedCanalSystem:
    """
    Physical Twin Layer: Cascaded Canal System.
    
    Simulates a series of connected canal pools.
    Structure: [Source] -> [Gate 0] -> [Pool 0] -> [Gate 1] -> [Pool 1] -> ... -> [Gate N] -> [Downstream]
    """
    
    def __init__(self, num_pools: int = 3, config: Dict = None, initial_flows: List[float] = None):
        """
        Initialize the cascaded system.
        
        Args:
            num_pools: Number of pools in series.
            config: Configuration dictionary containing pool parameters.
            initial_flows: List of initial inflow rates for each pool [Q_in_0, Q_in_1, ...].
                           If None, defaults to 0.0.
        """
        self.num_pools = num_pools
        self.pools: List[CanalPoolSimulator] = []
        
        # Default parameters if config is missing
        dt = config.get('dt', 3600.0) if config else 3600.0
        area = config.get('area', 10000.0) if config else 10000.0
        delay = config.get('delay_steps', 1) if config else 1
        init_level = config.get('initial_level', 3.0) if config else 3.0
        
        if initial_flows is None:
            initial_flows = [0.0] * num_pools
            
        if len(initial_flows) < num_pools:
            # Pad with last value or 0
            initial_flows.extend([0.0] * (num_pools - len(initial_flows)))
        
        # Initialize pools
        for i in range(num_pools):
            # Allow per-pool configuration in future
            self.pools.append(CanalPoolSimulator(
                area=area,
                dt=dt,
                delay_steps=delay,
                initial_level=init_level,
                initial_flow=initial_flows[i]
            ))
            
        # State tracking
        self.current_levels = [init_level] * num_pools
        self.gate_openings = list(initial_flows) + [0.0] # Approximate gate flows
        
    def step(self, gate_flows: List[float], demands: List[float], disturbances: List[float] = None):
        """
        Advance simulation by one step.
        
        Args:
            gate_flows: List of flows at each gate [Q0, Q1, ..., QN]. 
                        Q0 is inflow to Pool 0. Q1 is outflow from Pool 0 / inflow to Pool 1.
            demands: List of lateral demands (off-takes) for each pool [D0, D1, ...].
            disturbances: List of random disturbances for each pool.
        
        Returns:
            List[float]: New water levels for all pools.
        """
        if len(gate_flows) != self.num_pools + 1:
            raise ValueError(f"Expected {self.num_pools + 1} gate flows, got {len(gate_flows)}")
            
        if disturbances is None:
            disturbances = [0.0] * self.num_pools
            
        new_levels = []
        
        for i in range(self.num_pools):
            # For Pool i:
            # Inflow = Gate i flow (controlled)
            # Outflow = Gate i+1 flow (controlled) + Demand i (uncontrolled)
            
            q_in = gate_flows[i]
            q_out_gate = gate_flows[i+1]
            demand = demands[i]
            
            # Total outflow for mass balance
            total_q_out = q_out_gate + demand
            
            # Update pool physics
            level = self.pools[i].step(
                q_in_command=q_in,
                q_out=total_q_out,
                disturbance=disturbances[i]
            )
            new_levels.append(level)
            
        self.current_levels = new_levels
        return new_levels
        
    def get_levels(self):
        return self.current_levels
