"""
Configuration management module.

Loads, validates, and provides access to the simulation configuration
from a YAML file.  Supports dot-notation key lookups, environment
variable overrides, and a stubbed fallback when the config file is
unavailable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ConfigManager:
    """Manages loading and accessing simulation configuration from YAML.

    Attributes:
        config_path: Path to the YAML configuration file.
        config: The loaded configuration dictionary.
    """

    def __init__(self, config_path: str = 'config.yaml') -> None:
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config()
        self._apply_env_overrides()
        self._validate_config()

    # ------------------------------------------------------------------
    # Loading & validation
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found at: {self.config_path}"
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def _apply_env_overrides(self) -> None:
        """Override config values from environment variables.

        Supported environment variables::

            E2E_SIM_TOTAL_HOURS  -> simulation.total_hours
            E2E_SIM_TIME_STEP    -> simulation.time_step
            E2E_SIM_SEED         -> simulation.seed
            E2E_POOL_AREA        -> physical_system.area
            E2E_POOL_LEVEL       -> physical_system.initial_level
        """
        env_map: Dict[str, Tuple[str, str, type]] = {
            'E2E_SIM_TOTAL_HOURS': ('simulation', 'total_hours', int),
            'E2E_SIM_TIME_STEP': ('simulation', 'time_step', float),
            'E2E_SIM_SEED': ('simulation', 'seed', int),
            'E2E_POOL_AREA': ('physical_system', 'area', float),
            'E2E_POOL_LEVEL': ('physical_system', 'initial_level', float),
        }
        for env_var, (section, key, type_fn) in env_map.items():
            value = os.environ.get(env_var)
            if value is not None:
                self.config.setdefault(section, {})[key] = type_fn(value)

    def _validate_config(self) -> None:
        """Validate that essential configuration sections exist."""
        required_sections: Dict[str, Optional[List[str]]] = {
            'simulation': ['total_hours', 'time_step', 'seed'],
            'physical_system': ['area', 'initial_level'],
            'scenario_script': None,  # must be a list
            'demand_profile': ['base_demand', 'noise_std_dev'],
        }

        for section, keys in required_sections.items():
            if section not in self.config:
                raise ValueError(
                    f"Missing required configuration section: '{section}'"
                )
            if keys:
                for key in keys:
                    if key not in self.config[section]:
                        raise ValueError(
                            f"Missing required key '{key}' in section "
                            f"'{section}'"
                        )

        if not isinstance(self.config['scenario_script'], list):
            raise ValueError("'scenario_script' must be a list of events.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_simulation_params(self) -> Dict[str, Any]:
        """Return the main simulation parameters."""
        return self.config['simulation']

    def get_physical_params(self) -> Dict[str, Any]:
        """Return the physical system parameters."""
        return self.config['physical_system']

    def get_scenario_script(self) -> List[Tuple[int, str]]:
        """Return the scenario script as ``[(time, instruction), ...]``."""
        script = self.config['scenario_script']
        return [(item['time'], item['instruction']) for item in script]

    def get_demand_profile_params(self) -> Dict[str, Any]:
        """Return the demand profile parameters."""
        return self.config['demand_profile']

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return an entire config section as a dict.

        Args:
            section: Top-level key name (e.g. ``'monitoring'``, ``'mpc'``).

        Returns:
            The section dict, or an empty dict if it doesn't exist.
        """
        return dict(self.config.get(section, {}))

    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """Return the list of predefined scenarios from config.

        Returns:
            List of scenario dicts, each with ``'name'``, ``'keywords'``,
            ``'config'``.
        """
        return list(self.config.get('scenarios', []))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot-notation key.

        Example::

            config.get('simulation.time_step')

        Args:
            key: Dot-separated path into the config dict.
            default: Value to return if the key is not found.

        Returns:
            The config value, or *default* if the key doesn't exist.
        """
        parts = key.split('.')
        value: Any = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_default_instance: Optional[ConfigManager] = None


def get_config(config_path: str = 'config.yaml') -> ConfigManager:
    """Return (and cache) a default ConfigManager instance.

    Used by modules like ``monitor.py`` and ``brain_enhanced.py`` that
    import ``get_config`` at module level.
    """
    global _default_instance
    if _default_instance is None:
        try:
            _default_instance = ConfigManager(config_path)
        except (FileNotFoundError, ValueError):
            _default_instance = _StubConfigManager()  # type: ignore[assignment]
    return _default_instance


class _StubConfigManager:
    """Minimal stand-in when config.yaml is not available.

    Provides the same interface as :class:`ConfigManager` so that modules
    relying on ``get_config()`` do not crash at import time.
    """

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return default

    def get_section(self, section: str) -> Dict[str, Any]:
        _defaults: Dict[str, Dict[str, Any]] = {
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
                'horizon': 10, 'dt': 3600.0, 'area': 10000.0,
                'delay_steps': 1,
            },
        }
        return dict(_defaults.get(section, {}))

    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': '正常供水',
                'keywords': ['保持水位平稳，正常供水。', '正常', '供水', '平稳'],
                'config': {
                    'Z_ref': 3.0, 'W_level': 10.0,
                    'W_smooth': 5.0, 'delta_Q_max': 2.0,
                },
            },
            {
                'name': '暴雨预警',
                'keywords': ['暴雨预警', '暴雨', '降低水位', '腾出库容'],
                'config': {
                    'Z_ref': 2.0, 'W_level': 100.0,
                    'W_smooth': 5.0, 'delta_Q_max': 5.0,
                },
            },
        ]

    def get_simulation_params(self) -> Dict[str, Any]:
        return {'total_hours': 50, 'time_step': 3600.0, 'seed': 42}

    def get_physical_params(self) -> Dict[str, Any]:
        return {'area': 10000.0, 'initial_level': 3.0}

    def get_scenario_script(self) -> List[Tuple[int, str]]:
        return [(0, '保持水位平稳，正常供水。')]

    def get_demand_profile_params(self) -> Dict[str, Any]:
        return {'base_demand': 5.0, 'noise_std_dev': 0.5}
