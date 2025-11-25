import yaml

class ConfigManager:
    """Manages loading and accessing the simulation configuration from a YAML file."""
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self):
        """Loads the YAML configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def _validate_config(self):
        """Validates that the essential configuration sections and keys are present."""
        required_sections = {
            'simulation': ['total_hours', 'time_step', 'seed'],
            'physical_system': ['area', 'initial_level'],
            'scenario_script': None, # Must be a list
            'demand_profile': ['base_demand', 'noise_std_dev']
        }

        for section, keys in required_sections.items():
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: '{section}'")
            if keys:
                for key in keys:
                    if key not in self.config[section]:
                        raise ValueError(f"Missing required key '{key}' in section '{section}'")
        
        if not isinstance(self.config['scenario_script'], list):
            raise ValueError("'scenario_script' must be a list of events.")

    def get_simulation_params(self):
        """Returns the main simulation parameters."""
        return self.config['simulation']

    def get_physical_params(self):
        """Returns the physical system parameters."""
        return self.config['physical_system']

    def get_scenario_script(self):
        """Returns the scenario script as a list of tuples (time, instruction)."""
        script = self.config['scenario_script']
        return [(item['time'], item['instruction']) for item in script]

    def get_demand_profile_params(self):
        """Returns the demand profile parameters."""
        return self.config['demand_profile']
