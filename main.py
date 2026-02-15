import logging
import numpy as np
from config_manager import ConfigManager
from simulation_manager import SimulationManager
from simulation_report import ReportGenerator

logger = logging.getLogger(__name__)

def main():
    # --- 1. Load Configuration ---
    try:
        config_manager = ConfigManager()
        sim_params = config_manager.get_simulation_params()
        physical_params = config_manager.get_physical_params()
        script = config_manager.get_scenario_script()
        demand_params = config_manager.get_demand_profile_params()
    except (FileNotFoundError, ValueError) as e:
        logger.error("Error loading configuration: %s", e)
        return

    # --- 2. Prepare Simulation Inputs ---
    np.random.seed(sim_params['seed'])
    total_hours = sim_params['total_hours']
    demands = (demand_params['base_demand'] +
               np.random.normal(0, demand_params['noise_std_dev'], total_hours + 20))

    # --- 3. Run Simulation ---
    sim_manager = SimulationManager(
        total_hours=total_hours,
        dt=sim_params['time_step'],
        area=physical_params['area'],
        initial_level=physical_params['initial_level'],
        script=script,
        demands=demands
    )
    history = sim_manager.run_simulation()

    # --- 4. Generate Report & Visualization ---
    if history:
        report_generator = ReportGenerator(history, script)
        report_generator.generate_all_artifacts()

if __name__ == "__main__":
    main()
