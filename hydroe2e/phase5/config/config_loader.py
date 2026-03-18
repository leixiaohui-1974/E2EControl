"""
统一配置加载器

提供从YAML文件加载配置并转换为各模块dataclass的功能。
支持环境变量覆盖和多环境配置合并。
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Type
from dataclasses import dataclass, field, fields, is_dataclass
import yaml

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class ConfigLoader:
    """
    统一配置加载器

    支持:
    - YAML配置文件加载
    - 环境变量覆盖 (E2E_前缀)
    - 多环境配置合并
    - 类型安全的配置访问
    """

    config_path: Optional[Path] = None
    environment: str = "development"
    _config: Dict[str, Any] = field(default_factory=dict, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.config_path is None:
            self.config_path = DEFAULT_CONFIG_PATH
        elif isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)

    def load(self, reload: bool = False) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            reload: 是否强制重新加载

        Returns:
            配置字典
        """
        if self._loaded and not reload:
            return self._config

        # 加载主配置文件
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"配置已加载: {self.config_path}")
        else:
            logger.warning(f"配置文件不存在: {self.config_path}, 使用默认配置")
            self._config = {}

        # 加载环境特定配置
        env_config_path = self.config_path.parent / f"config.{self.environment}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r', encoding='utf-8') as f:
                env_config = yaml.safe_load(f) or {}
            self._config = self._deep_merge(self._config, env_config)
            logger.info(f"环境配置已合并: {env_config_path}")

        # 应用环境变量覆盖
        self._apply_env_overrides()

        self._loaded = True
        return self._config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并两个字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_env_overrides(self):
        """
        应用环境变量覆盖

        环境变量格式: E2E_SECTION_SUBSECTION_KEY=value
        例如: E2E_NEURAL_PHYSICS_HIDDEN_DIM=256
        """
        prefix = "E2E_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # 解析路径: E2E_SECTION_SUBSECTION_KEY -> section.subsection.key
            parts = key[len(prefix):].lower().split('_')

            # 尝试解析值类型
            parsed_value = self._parse_env_value(value)

            # 设置配置值
            self._set_nested(self._config, parts, parsed_value)
            logger.debug(f"环境变量覆盖: {key}={parsed_value}")

    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值为适当的类型"""
        # 布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # 数值
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # 列表 (逗号分隔)
        if ',' in value:
            return [self._parse_env_value(v.strip()) for v in value.split(',')]

        return value

    def _set_nested(self, d: Dict, keys: list, value: Any):
        """设置嵌套字典值"""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def get(self, path: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            path: 配置路径，用点分隔，如 'neural_physics.architecture.hidden_dim'
            default: 默认值

        Returns:
            配置值
        """
        if not self._loaded:
            self.load()

        keys = path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置节"""
        return self.get(section, {})

    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置"""
        if not self._loaded:
            self.load()
        return self._config


# 全局配置实例
_global_config: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None,
               environment: Optional[str] = None) -> ConfigLoader:
    """
    获取全局配置实例

    Args:
        config_path: 可选的配置文件路径
        environment: 可选的环境名称

    Returns:
        ConfigLoader实例
    """
    global _global_config

    if _global_config is None or config_path is not None:
        env = environment or os.environ.get('E2E_ENV', 'development')
        _global_config = ConfigLoader(
            config_path=Path(config_path) if config_path else None,
            environment=env
        )
        _global_config.load()

    return _global_config


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置并返回字典"""
    return get_config(config_path).config


# =============================================================================
# 各模块配置转换函数
# =============================================================================

def get_water_canal_env_config(config: Optional[ConfigLoader] = None):
    """
    获取WaterCanalEnvConfig

    将统一配置转换为water_canal_env模块所需的配置格式
    """
    from hydroe2e.phase5.ai_models.water_canal_env import WaterCanalEnvConfig

    cfg = config or get_config()
    env_cfg = cfg.get_section('environment')
    timing_cfg = cfg.get_section('timing')
    water_cfg = cfg.get_section('water_system')
    target_cfg = env_cfg.get('target', {})
    reward_cfg = env_cfg.get('reward', {})
    pool_cfg = water_cfg.get('default_pool', {})
    gate_cfg = water_cfg.get('default_gate', {})
    boundary_cfg = water_cfg.get('boundary', {})

    return WaterCanalEnvConfig(
        dt=timing_cfg.get('dt', 900.0),
        episode_duration=timing_cfg.get('episode_duration', 86400.0),
        observation_window=env_cfg.get('observation', {}).get('window', 16),
        num_features=env_cfg.get('observation', {}).get('features', 5),
        target_level_upstream=target_cfg.get('upstream', 4.0),
        target_level_downstream=target_cfg.get('downstream', 3.8),
        min_level=pool_cfg.get('min_level', 1.5),
        max_level=pool_cfg.get('max_level', 5.5),
        w_level=reward_cfg.get('level_weight', 1.0),
        w_action=reward_cfg.get('action_weight', 0.1),
        w_safety=reward_cfg.get('safety_weight', 10.0),
        upstream_pool_area=pool_cfg.get('area', 80000.0),
        downstream_pool_area=pool_cfg.get('area', 100000.0) * 1.25,
        inflow_noise_std=boundary_cfg.get('inflow_noise_std', 5.0),
        demand_noise_std=boundary_cfg.get('demand_noise_std', 3.0),
    )


def get_neural_physics_config(config: Optional[ConfigLoader] = None):
    """
    获取NeuralPhysicsConfig
    """
    from hydroe2e.phase5.ai_models.neural_physics_engine import NeuralPhysicsConfig

    cfg = config or get_config()
    np_cfg = cfg.get_section('neural_physics')
    arch_cfg = np_cfg.get('architecture', {})
    physics_cfg = np_cfg.get('physics_loss', {})
    train_cfg = np_cfg.get('training', {})

    return NeuralPhysicsConfig(
        input_dim=arch_cfg.get('input_dim', 5),
        hidden_dim=arch_cfg.get('hidden_dim', 128),
        num_layers=arch_cfg.get('num_layers', 2),
        output_dim=arch_cfg.get('output_dim', 1),
        dropout=arch_cfg.get('dropout', 0.1),
        sequence_length=arch_cfg.get('sequence_length', 16),
        use_physics_loss=physics_cfg.get('enabled', True),
        physics_loss_weight=physics_cfg.get('weight', 0.1),
        learning_rate=train_cfg.get('learning_rate', 0.001),
        batch_size=train_cfg.get('batch_size', 64),
        num_epochs=train_cfg.get('num_epochs', 100),
    )


def get_scenario_encoder_config(config: Optional[ConfigLoader] = None):
    """
    获取DeepEncoderConfig
    """
    from hydroe2e.phase5.ai_models.deep_scenario_encoder import DeepEncoderConfig

    cfg = config or get_config()
    enc_cfg = cfg.get_section('scenario_encoder')
    arch_cfg = enc_cfg.get('architecture', {})
    contrast_cfg = enc_cfg.get('contrastive', {})

    return DeepEncoderConfig(
        sequence_length=arch_cfg.get('sequence_length', 96),
        input_channels=arch_cfg.get('input_channels', 5),
        encoder_type=arch_cfg.get('type', 'cnn'),
        hidden_dim=arch_cfg.get('hidden_dim', 128),
        embedding_dim=arch_cfg.get('embedding_dim', 64),
        num_layers=arch_cfg.get('num_layers', 3),
        temperature=contrast_cfg.get('temperature', 0.1),
        similarity_threshold=contrast_cfg.get('similarity_threshold', 0.7),
    )


def get_scenario_vae_config(config: Optional[ConfigLoader] = None):
    """
    获取ScenarioVAEConfig
    """
    from hydroe2e.phase5.ai_models.scenario_vae import ScenarioVAEConfig

    cfg = config or get_config()
    vae_cfg = cfg.get_section('scenario_vae')
    arch_cfg = vae_cfg.get('architecture', {})
    train_cfg = vae_cfg.get('training', {})

    return ScenarioVAEConfig(
        sequence_length=arch_cfg.get('sequence_length', 96),
        num_channels=arch_cfg.get('num_channels', 4),
        encoder_hidden_dim=arch_cfg.get('encoder_hidden_dim', 128),
        latent_dim=arch_cfg.get('latent_dim', 32),
        decoder_hidden_dim=arch_cfg.get('decoder_hidden_dim', 128),
        learning_rate=train_cfg.get('learning_rate', 0.001),
        kl_weight=train_cfg.get('kl_weight', 0.001),
        temperature=train_cfg.get('temperature', 1.0),
    )


def get_e2e_controller_config(config: Optional[ConfigLoader] = None):
    """
    获取AutonomousConfig
    """
    from hydroe2e.phase5.ai_models.l4_autonomous.e2e_controller import AutonomousConfig

    cfg = config or get_config()
    ctrl_cfg = cfg.get_section('autonomous_controller')
    water_cfg = cfg.get_section('water_system')
    timing_cfg = cfg.get_section('timing')
    arch_cfg = ctrl_cfg.get('architecture', {})
    autonomy_cfg = ctrl_cfg.get('autonomy', {})
    loss_cfg = ctrl_cfg.get('loss_weights', {})

    return AutonomousConfig(
        num_pools=water_cfg.get('num_pools', 63),
        num_gates=water_cfg.get('num_gates', 64),
        state_dim=arch_cfg.get('state_dim', 128),
        action_dim=arch_cfg.get('action_dim', 64),
        hidden_dim=arch_cfg.get('hidden_dim', 256),
        num_heads=arch_cfg.get('num_attention_heads', 8),
        history_length=timing_cfg.get('history_length', 96),
        prediction_horizon=timing_cfg.get('prediction_horizon', 48),
        confidence_threshold=autonomy_cfg.get('confidence_threshold', 0.3),
        level_weight=loss_cfg.get('level', 1.0),
        safety_weight=loss_cfg.get('safety', 10.0),
    )


def get_world_model_config(config: Optional[ConfigLoader] = None):
    """
    获取WorldModelConfig
    """
    from hydroe2e.phase5.ai_models.l4_autonomous.full_line_world_model import WorldModelConfig

    cfg = config or get_config()
    wm_cfg = cfg.get_section('world_model')
    water_cfg = cfg.get_section('water_system')
    timing_cfg = cfg.get_section('timing')
    arch_cfg = wm_cfg.get('architecture', {})

    return WorldModelConfig(
        num_pools=water_cfg.get('num_pools', 63),
        num_gates=water_cfg.get('num_gates', 64),
        total_length=water_cfg.get('total_length', 1432.0),
        node_dim=arch_cfg.get('node_dim', 64),
        hidden_dim=arch_cfg.get('hidden_dim', 256),
        history_length=timing_cfg.get('history_length', 48),
        avg_velocity=water_cfg.get('avg_velocity', 1.5),
        prediction_horizons=wm_cfg.get('prediction_horizons', [4, 12, 48]),
        min_delay=timing_cfg.get('min_delay', 1800.0),
        max_delay=timing_cfg.get('max_delay', 86400.0),
    )


def get_data_generator_config(config: Optional[ConfigLoader] = None):
    """
    获取DataGeneratorConfig
    """
    from hydroe2e.phase5.ai_models.data_generator import DataGeneratorConfig

    cfg = config or get_config()
    dg_cfg = cfg.get_section('data_generator')
    timing_cfg = cfg.get_section('timing')
    ratios = dg_cfg.get('scenario_ratios', {})
    ranges = dg_cfg.get('ranges', {})

    return DataGeneratorConfig(
        num_samples=dg_cfg.get('num_samples', 100000),
        sequence_length=dg_cfg.get('sequence_length', 16),
        dt=timing_cfg.get('dt', 900.0),
        steady_ratio=ratios.get('steady', 0.2),
        step_ratio=ratios.get('step', 0.2),
        random_ratio=ratios.get('random', 0.3),
        periodic_ratio=ratios.get('periodic', 0.2),
        extreme_ratio=ratios.get('extreme', 0.1),
        inflow_range=tuple(ranges.get('inflow', [50.0, 400.0])),
        level_range=tuple(ranges.get('level', [1.0, 6.0])),
        gate_range=tuple(ranges.get('gate', [0.1, 1.0])),
    )


# =============================================================================
# 便捷访问函数
# =============================================================================

def get_system_config() -> Dict[str, Any]:
    """获取系统配置"""
    return get_config().get_section('system')


def get_water_system_config() -> Dict[str, Any]:
    """获取水利系统配置"""
    return get_config().get_section('water_system')


def get_timing_config() -> Dict[str, Any]:
    """获取时间参数配置"""
    return get_config().get_section('timing')


def get_training_config() -> Dict[str, Any]:
    """获取训练配置"""
    return get_config().get_section('training')


def get_logging_config() -> Dict[str, Any]:
    """获取日志配置"""
    return get_config().get_section('logging')
