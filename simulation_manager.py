import threading
import time
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np

from main_enhanced import SmartPoolSimulation
from database import SimulationDatabase
from config_manager import get_config
from physics.cascaded_system import CascadedCanalSystem
from control.distributed_mpc import DistributedMPC
from intelligence import ScenarioRecognitionEngine

class SimulationManager:
    """
    Manages the lifecycle of simulations.
    Handles creation, execution (async), status tracking, and history retrieval.
    """
    def __init__(self):
        self.running_simulations: Dict[int, Dict] = {}
        self.lock = threading.Lock()
        self.config = get_config()

    def run_simulation(self, script: Optional[List] = None, is_async: bool = False, system_type: str = 'single') -> Dict:
        """
        Runs a simulation.
        
        Args:
            script: The simulation script (list of time/instruction tuples).
            is_async: Whether to run asynchronously.
            system_type: 'single' or 'cascaded'.
            
        Returns:
            Dict containing simulation ID and status/results.
        """
        
        if system_type == 'cascaded':
            return self._run_cascaded_simulation(script, is_async)
        
        # Default Single Pool Logic
        sim = SmartPoolSimulation()
        
        if is_async:
            # Generate a temporary ID for tracking
            with self.lock:
                sim_id = int(time.time() * 1000) 
                
                self.running_simulations[sim_id] = {
                    'status': 'running',
                    'start_time': datetime.now().isoformat(),
                    'simulation': sim,
                    'type': 'single'
                }

            def _run_thread():
                try:
                    if is_async:
                        sim.interactive = True
                    sim.run(script=script)
                    with self.lock:
                        if sim_id in self.running_simulations:
                            self.running_simulations[sim_id]['status'] = 'completed'
                            self.running_simulations[sim_id]['end_time'] = datetime.now().isoformat()
                            if sim.simulation_id:
                                self.running_simulations[sim_id]['db_id'] = sim.simulation_id
                except Exception as e:
                    with self.lock:
                        if sim_id in self.running_simulations:
                            self.running_simulations[sim_id]['status'] = 'failed'
                            self.running_simulations[sim_id]['error'] = str(e)
                finally:
                    if sim.db:
                        sim.db.close()

            thread = threading.Thread(target=_run_thread)
            thread.start()

            return {
                'success': True,
                'simulation_id': sim_id,
                'status': 'running',
                'message': 'Single pool simulation started asynchronously'
            }
        else:
            # Synchronous
            try:
                sim.run(script=script)
                sim_id = sim.simulation_id
                
                results = {
                    'total_hours': sim.total_hours,
                    'final_level': sim.history['level'][-1] if sim.history['level'] else None,
                    'alerts_count': len(sim.monitor.alerts)
                }
                
                if sim.db:
                    sim.db.close()
                    
                return {
                    'success': True,
                    'simulation_id': sim_id,
                    'status': 'completed',
                    'results': results
                }
            except Exception as e:
                raise e

    def _run_cascaded_simulation(self, script: Optional[List], is_async: bool) -> Dict:
        """Helper to run cascaded simulation."""
        
        sim_id = int(time.time() * 1000)
        
        # Initialize shared state for this simulation
        with self.lock:
            self.running_simulations[sim_id] = {
                'status': 'initializing',
                'start_time': datetime.now().isoformat(),
                'type': 'cascaded',
                'events': [], # Queue for one-time events
                'overrides': {} # Persistent parameter overrides
            }

        def _simulation_logic():
            # Initialize
            num_pools = 3
            
            # Initial state
            # Calculate steady state flows for equilibrium
            # For 3 pools with base demand 5.0 each:
            # Pool 2 (last): In=5, Out=0 (gate)+5(demand) -> Net 0
            # Pool 1: In=10, Out=5(gate)+5(demand) -> Net 0
            # Pool 0: In=15, Out=10(gate)+5(demand) -> Net 0
            # So gates should be [15, 10, 5, 0]
            
            initial_flows = []
            cumulative_flow = 0.0
            base_demand = 5.0
            # Work backwards from last pool
            for _ in range(num_pools):
                cumulative_flow += base_demand
                initial_flows.insert(0, cumulative_flow)
            
            # initial_flows is now [15.0, 10.0, 5.0] for inflow to pools 0, 1, 2
            
            physics = CascadedCanalSystem(num_pools=num_pools, initial_flows=initial_flows)
            controller = DistributedMPC(num_pools=num_pools)
            intelligence = ScenarioRecognitionEngine()
            
            # Config
            total_hours = 50
            history = {
                'time': [],
                'levels': [], # List of lists
                'flows': [],   # List of lists
                'scenarios': [], # List of scenario dicts
                'events': [] # Log of events triggered
            }
            
            initial_levels = [3.0] * num_pools 
            physics.current_levels = list(initial_levels) # Ensure levels are set
            
            # Controller needs prev_flows (gate flows)
            # Gate 0 -> Pool 0 (15)
            # Gate 1 -> Pool 1 (10)
            # Gate 2 -> Pool 2 (5)
            # Gate 3 -> Downstream (0)
            prev_flows = initial_flows + [0.0]
            # Gate 0 -> Pool 0 (15)
            # Gate 1 -> Pool 1 (10)
            # Gate 2 -> Pool 2 (5)
            # Gate 3 -> Downstream (0)
            prev_flows = initial_flows + [0.0]
            
            # Loop
            for t in range(total_hours):
                # Slow down slightly for HITL if async (optional, but good for demo)
                if is_async:
                    time.sleep(1.0) # 1 second per hour for demo interaction

                # 0. Check for Stop Request
                with self.lock:
                    if sim_id in self.running_simulations:
                        if self.running_simulations[sim_id].get('status') == 'stopping':
                            break

                # 0.5 Process External Events & Overrides
                current_events = []
                current_overrides = {}
                
                with self.lock:
                    if sim_id in self.running_simulations:
                        # Pop all pending events
                        while self.running_simulations[sim_id]['events']:
                            evt = self.running_simulations[sim_id]['events'].pop(0)
                            current_events.append(evt)
                            history['events'].append({'time': t, 'event': evt})
                        
                        # Get current overrides
                        current_overrides = self.running_simulations[sim_id]['overrides'].copy()

                # Apply Events (e.g., Sudden Inflow)
                extra_inflow = 0.0
                for evt in current_events:
                    if evt['type'] == 'flood':
                        extra_inflow += evt.get('magnitude', 10.0)
                    elif evt['type'] == 'drought':
                        extra_inflow -= evt.get('magnitude', 5.0)

                # 1. Forecast Demands
                demands = []
                for i in range(num_pools):
                    base = 5.0
                    noise = np.random.normal(0, 0.5, 10)
                    d = base + noise
                    
                    # Apply demand overrides if any
                    # e.g., overrides = {'demand_0': 10.0}
                    if f'demand_{i}' in current_overrides:
                        d[:] = current_overrides[f'demand_{i}']
                        
                    demands.append(d)
                
                current_demands = [d[0] for d in demands]
                
                # 2. Intelligence: Recognize Scenario
                current_levels = physics.get_levels()
                scenario_result = intelligence.recognize(
                    current_levels=current_levels,
                    flows=prev_flows
                )
                
                # Override scenario if forced
                if 'force_scenario' in current_overrides:
                    scenario_result['name'] = current_overrides['force_scenario']
                    scenario_result['description'] = "Manually Forced"
                    scenario_result['confidence'] = 1.0
                
                # 3. Control
                config = scenario_result['recommended_config']
                
                gate_flows = controller.solve(
                    current_levels=current_levels,
                    prev_flows=prev_flows,
                    demand_forecasts=demands,
                    config=config
                )
                
                # Apply Manual Gate Overrides
                # e.g., overrides = {'gate_0': 5.0}
                for i in range(len(gate_flows)):
                    if f'gate_{i}' in current_overrides:
                        gate_flows[i] = float(current_overrides[f'gate_{i}'])

                # 4. Physics Step
                if extra_inflow > 0:
                    gate_flows[0] += extra_inflow
                
                new_levels = physics.step(
                    gate_flows=gate_flows,
                    demands=current_demands
                )
                
                # 5. Log
                history['time'].append(t)
                history['levels'].append(new_levels)
                history['flows'].append(gate_flows) # Ensure this is added
                history['scenarios'].append(scenario_result)
                
                prev_flows = gate_flows
                
                # Update live history in memory for polling
                if is_async:
                    with self.lock:
                        if sim_id in self.running_simulations:
                            self.running_simulations[sim_id]['history'] = history
                
            return history

        if is_async:
            with self.lock:
                # Status already set to initializing, update to running
                self.running_simulations[sim_id]['status'] = 'running'
            
            def _run_thread():
                try:
                    # Initialize Phase 5 Integrated System
                    from phase5.integrated_system import IntegratedWaterNetworkSystem
                    integrated_system = IntegratedWaterNetworkSystem(
                        num_pools=3,
                        enable_digital_twin=False,
                        enable_self_healing=True,
                        enable_anomaly_detection=True
                    )
                    
                    # Store the system instance for interactive control
                    with self.lock:
                         if sim_id in self.running_simulations:
                             self.running_simulations[sim_id]['system_instance'] = integrated_system

                    # Parse script
                    scenario_script = []
                    if script:
                         for item in script:
                             scenario_script.append((int(item.get('time', 0)), item.get('instruction', '')))
                    
                    scenario_dict = {t: instruction for t, instruction in scenario_script}
                    
                    # Run loop
                    t = 0
                    while t < 100: # Default max steps
                        # Check for stop signal
                        with self.lock:
                            if sim_id not in self.running_simulations or \
                               self.running_simulations[sim_id].get('status') == 'stopping':
                                break
                        
                        # Step
                        instruction = scenario_dict.get(t)
                        integrated_system.step(t, instruction)
                        
                        # Update history in SimulationManager for API access
                        # Note: integrated_system.history grows, so we can just reference it or copy latest
                        # For simplicity, we'll rely on integrated_system.history being the source of truth
                        # But we need to sync it to the dict if get_history reads from dict
                        
                        # Actually, get_history reads from self.running_simulations[sim_id]['history']
                        # So we should update it periodically or at the end.
                        # For real-time, we update it every step.
                        with self.lock:
                             if sim_id in self.running_simulations:
                                 self.running_simulations[sim_id]['history'] = integrated_system.history
                        
                        t += 1
                        time.sleep(1) # Real-time simulation speed
                    
                    with self.lock:
                        if sim_id in self.running_simulations:
                            self.running_simulations[sim_id]['status'] = 'completed'
                            self.running_simulations[sim_id]['end_time'] = datetime.now().isoformat()
                            self.running_simulations[sim_id]['history'] = integrated_system.history
                            
                except Exception as e:
                    with self.lock:
                        if sim_id in self.running_simulations:
                            self.running_simulations[sim_id]['status'] = 'failed'
                            self.running_simulations[sim_id]['error'] = str(e)
                            print(f"Simulation failed: {e}")
                            import traceback
                            traceback.print_exc()

            thread = threading.Thread(target=_run_thread)
            thread.start()
            
            return {
                'success': True,
                'simulation_id': sim_id,
                'status': 'running',
                'message': 'Cascaded simulation started asynchronously (Phase 5 Integrated)'
            }
        else:
            # Sync mode (simplified, mostly for testing)
            try:
                from phase5.integrated_system import IntegratedWaterNetworkSystem
                integrated_system = IntegratedWaterNetworkSystem(
                    num_pools=3,
                    enable_digital_twin=False,
                    enable_self_healing=True,
                    enable_anomaly_detection=True
                )
                
                scenario_script = []
                if script:
                        for item in script:
                            scenario_script.append((int(item.get('time', 0)), item.get('instruction', '')))

                hist = integrated_system.run_simulation(scenario_script, total_steps=50)
                
                return {
                    'success': True,
                    'simulation_id': sim_id,
                    'status': 'completed',
                    'results': {
                        'total_hours': 50,
                        'final_levels': hist['levels'][-1] if hist['levels'] else []
                    }
                }
            except Exception as e:
                raise e

    def inject_event(self, sim_id: int, event_type: str, event_data: Dict) -> bool:
        """Injects an event into a running simulation."""
        with self.lock:
            if sim_id in self.running_simulations:
                sim_data = self.running_simulations[sim_id]
                
                # Phase 5 Integration: Call inject_fault on the system instance
                if 'system_instance' in sim_data:
                    return sim_data['system_instance'].inject_fault(event_type, event_data)
                
                # Legacy / Phase 1-4 Support
                event = {'type': event_type, **event_data}
                sim_data['events'].append(event)
                return True
        return False

    def update_parameters(self, sim_id: int, params: Dict) -> bool:
        """Updates parameters (overrides) for a running simulation."""
        with self.lock:
            if sim_id in self.running_simulations:
                self.running_simulations[sim_id]['overrides'].update(params)
                return True
        return False

    def clear_overrides(self, sim_id: int) -> bool:
        """Clears all parameter overrides for a running simulation."""
        with self.lock:
            if sim_id in self.running_simulations:
                self.running_simulations[sim_id]['overrides'] = {}
                return True
        return False

    def stop_simulation(self, sim_id: int) -> bool:
        """Stops a running simulation."""
        with self.lock:
            if sim_id in self.running_simulations:
                self.running_simulations[sim_id]['status'] = 'stopping'
                return True
        return False

    def get_status(self, sim_id: int) -> Dict:
        """Gets the status of a simulation."""
        # 1. Check memory (running or recently finished async)
        with self.lock:
            if sim_id in self.running_simulations:
                info = self.running_simulations[sim_id]
                return {
                    'success': True,
                    'simulation_id': sim_id,
                    'status': info['status'],
                    'start_time': info.get('start_time'),
                    'end_time': info.get('end_time'),
                    'error': info.get('error'),
                    'db_id': info.get('db_id')
                }

        # 2. Check database
        if self.config.get('database.enabled', False):
            try:
                db = SimulationDatabase(self.config.get('database.path'))
                # We need a method to get a single simulation by ID
                # The existing API used get_recent_simulations and filtered.
                # Let's stick to that for now or add a method if needed.
                sims = db.get_recent_simulations(limit=100) # Potential optimization needed here
                db.close()
                
                for sim in sims:
                    if sim['id'] == sim_id:
                        return {
                            'success': True,
                            'simulation_id': sim_id,
                            'status': 'completed',
                            'start_time': sim['start_time'],
                            'end_time': sim['end_time'],
                            'total_hours': sim['total_hours']
                        }
            except Exception as e:
                return {'success': False, 'error': str(e)}

        return {'success': False, 'error': f'Simulation ID {sim_id} not found'}

    def get_history(self, sim_id: int) -> Dict:
        """Gets history for a simulation."""
        # Check memory first for live updates? 
        # Current implementation only supports DB history for simplicity in this refactor,
        # unless we want to expose live data from the 'sim' object in running_simulations.
        
        with self.lock:
            if sim_id in self.running_simulations:
                info = self.running_simulations[sim_id]
                if 'history' in info:
                    # Cascaded simulation history stored directly
                    return {
                        'success': True,
                        'simulation_id': sim_id,
                        'count': len(info['history']['time']),
                        'history': info['history'],
                        'status': info['status']
                    }
                elif 'simulation' in info:
                    # Single pool simulation object
                    sim = info['simulation']
                    return {
                        'success': True,
                        'simulation_id': sim_id,
                        'count': len(sim.history['time']),
                        'history': sim.history,
                        'status': info['status']
                    }

        if not self.config.get('database.enabled', False):
             return {'success': False, 'error': 'Database not enabled'}

        try:
            db = SimulationDatabase(self.config.get('database.path'))
            history = db.get_simulation_history(sim_id)
            db.close()
            
            if not history:
                return {'success': False, 'error': f'No history for ID {sim_id}'}
                
            return {
                'success': True,
                'simulation_id': sim_id,
                'count': len(history),
                'history': history
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def list_simulations(self) -> List[Dict]:
        """Lists all simulations (memory + db)."""
        result = []
        
        # Memory
        with self.lock:
            for sim_id, info in self.running_simulations.items():
                result.append({
                    'id': sim_id,
                    'status': info['status'],
                    'start_time': info.get('start_time'),
                    'end_time': info.get('end_time'),
                    'source': 'memory'
                })
        
        # Database
        if self.config.get('database.enabled', False):
            try:
                db = SimulationDatabase(self.config.get('database.path'))
                sims = db.get_recent_simulations(limit=20)
                db.close()
                
                for sim in sims:
                    # Avoid duplicates if they are in both (unlikely with current ID logic but possible)
                    if not any(r['id'] == sim['id'] for r in result):
                        result.append({
                            'id': sim['id'],
                            'status': 'completed',
                            'start_time': sim['start_time'],
                            'end_time': sim['end_time'],
                            'total_hours': sim['total_hours'],
                            'source': 'database'
                        })
            except Exception:
                pass # Ignore DB errors for listing
                
        return result
