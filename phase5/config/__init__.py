"""
E2EControl 统一配置管理模块

提供集中式配置管理，支持:
- YAML配置文件加载
- 环境变量覆盖
- 多环境支持(dev/test/prod)
- 类型安全的配置访问
"""

from .config_loader import (
    ConfigLoader,
    get_config,
    load_config,
    get_water_canal_env_config,
    get_neural_physics_config,
    get_scenario_encoder_config,
    get_scenario_vae_config,
    get_e2e_controller_config,
    get_world_model_config,
    get_data_generator_config,
)

__all__ = [
    'ConfigLoader',
    'get_config',
    'load_config',
    'get_water_canal_env_config',
    'get_neural_physics_config',
    'get_scenario_encoder_config',
    'get_scenario_vae_config',
    'get_e2e_controller_config',
    'get_world_model_config',
    'get_data_generator_config',
]
