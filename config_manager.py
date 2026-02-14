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

    def get_section(self, section: str) -> dict:
        """Return an entire config section as a dict.

        Args:
            section: Top-level key name (e.g. 'monitoring', 'mpc').

        Returns:
            The section dict, or an empty dict if it doesn't exist.
        """
        return dict(self.config.get(section, {}))

    def get_all_scenarios(self) -> list:
        """Return the list of predefined scenarios from config.

        Returns:
            List of scenario dicts, each with 'name', 'keywords', 'config'.
        """
        return list(self.config.get('scenarios', []))

    def get(self, key: str, default=None):
        """Get a config value using dot-notation key (e.g. 'simulation.dt').

        Args:
            key: Dot-separated path into the config dict.
            default: Value to return if the key is not found.

        Returns:
            The config value, or *default* if the key doesn't exist.
        """
        parts = key.split('.')
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_default_instance = None


def get_config(config_path: str = 'config.yaml'):
    """Return (and cache) a default ConfigManager instance.

    Used by modules like ``monitor.py`` and ``brain_enhanced.py`` that
    import ``get_config`` at module level.
    """
    global _default_instance
    if _default_instance is None:
        try:
            _default_instance = ConfigManager(config_path)
        except (FileNotFoundError, ValueError):
            _default_instance = _StubConfigManager()
    return _default_instance


class _StubConfigManager:
    """Minimal stand-in when config.yaml is not available."""

    def __init__(self):
        self.config = {}

    def get(self, key, default=None):
        return default

    def get_section(self, section):
        _defaults = {
            'monitoring': {
                'level_warning_high': 8.0, 'level_warning_low': 1.0,
                'level_critical_high': 9.5, 'level_critical_low': 0.5,
                'flow_rate_max': 25.0,
            },
            'default_control': {
                'W_level': 10.0, 'W_smooth': 5.0, 'Z_ref': 3.0,
                'delta_Q_max': 2.0, 'constraints': {},
            },
            'mpc': {
                'horizon': 10, 'dt': 3600.0, 'area': 10000.0, 'delay_steps': 1,
            },
        }
        return dict(_defaults.get(section, {}))

    def get_all_scenarios(self):
        return [
            {'name': '正常供水', 'keywords': ['保持水位平稳，正常供水。', '正常', '供水', '平稳'],
             'config': {'Z_ref': 3.0, 'W_level': 10.0, 'W_smooth': 5.0, 'delta_Q_max': 2.0}},
            {'name': '暴雨预警', 'keywords': ['暴雨预警', '暴雨', '降低水位', '腾出库容'],
             'config': {'Z_ref': 2.0, 'W_level': 100.0, 'W_smooth': 5.0, 'delta_Q_max': 5.0}},
        ]

    def get_simulation_params(self):
        return {'total_hours': 50, 'time_step': 3600.0, 'seed': 42}

    def get_physical_params(self):
        return {'area': 10000.0, 'initial_level': 3.0}

    def get_scenario_script(self):
        return [(0, '保持水位平稳，正常供水。')]

    def get_demand_profile_params(self):
        return {'base_demand': 5.0, 'noise_std_dev': 0.5}
