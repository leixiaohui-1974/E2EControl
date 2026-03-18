"""
配置加载器单元测试

测试统一配置系统的加载、合并、覆盖功能。
"""

import os
import pytest
import tempfile
from pathlib import Path

import yaml


class TestConfigLoader:
    """测试ConfigLoader类"""

    def test_load_default_config(self):
        """测试加载默认配置"""
        from config.config_loader import ConfigLoader, DEFAULT_CONFIG_PATH

        loader = ConfigLoader()
        config = loader.load()

        assert isinstance(config, dict)
        assert 'system' in config
        assert 'water_system' in config
        assert 'neural_physics' in config

    def test_get_config_value(self):
        """测试获取配置值(开发环境)"""
        from config.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader.load()

        # 开发环境使用较小的hidden_dim
        hidden_dim = loader.get('neural_physics.architecture.hidden_dim')
        assert hidden_dim == 64  # 开发环境值

        # 测试默认值
        missing = loader.get('nonexistent.path', default='default')
        assert missing == 'default'

    def test_get_section(self):
        """测试获取配置节"""
        from config.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader.load()

        water_system = loader.get_section('water_system')
        assert isinstance(water_system, dict)
        assert 'num_pools' in water_system
        # 开发环境使用5个渠池
        assert water_system['num_pools'] == 5

    def test_environment_override(self):
        """测试环境配置覆盖"""
        from config.config_loader import ConfigLoader, DEFAULT_CONFIG_PATH

        # 测试开发环境
        loader = ConfigLoader(
            config_path=DEFAULT_CONFIG_PATH,
            environment='development'
        )
        loader.load()

        # 开发环境应该有较小的num_pools
        num_pools = loader.get('water_system.num_pools')
        assert num_pools == 5  # 开发环境配置

    def test_base_config_values(self):
        """测试基础配置值(不合并环境配置)"""
        from config.config_loader import ConfigLoader, DEFAULT_CONFIG_PATH

        # 使用不存在的环境名，避免合并
        loader = ConfigLoader(
            config_path=DEFAULT_CONFIG_PATH,
            environment='nonexistent'
        )
        loader.load()

        # 应该获取基础配置值
        num_pools = loader.get('water_system.num_pools')
        hidden_dim = loader.get('neural_physics.architecture.hidden_dim')

        assert num_pools == 63  # 基础配置
        assert hidden_dim == 128  # 基础配置

    def test_env_variable_override(self):
        """测试环境变量覆盖"""
        from config.config_loader import ConfigLoader, DEFAULT_CONFIG_PATH

        # 设置简单的环境变量(单词路径无下划线)
        os.environ['E2E_SYSTEM_NAME'] = 'TestSystem'

        try:
            # 创建新的loader实例
            loader = ConfigLoader(
                config_path=DEFAULT_CONFIG_PATH,
                environment='nonexistent'
            )
            # load()时会应用环境变量
            loader.load()

            system_name = loader.get('system.name')
            assert system_name == 'TestSystem'
        finally:
            # 清理环境变量
            del os.environ['E2E_SYSTEM_NAME']

    def test_parse_env_value_types(self):
        """测试环境变量值类型解析"""
        from config.config_loader import ConfigLoader

        loader = ConfigLoader()

        # 布尔值
        assert loader._parse_env_value('true') is True
        assert loader._parse_env_value('false') is False
        assert loader._parse_env_value('yes') is True
        assert loader._parse_env_value('no') is False

        # 数值
        assert loader._parse_env_value('42') == 42
        assert loader._parse_env_value('3.14') == 3.14

        # 列表
        assert loader._parse_env_value('1,2,3') == [1, 2, 3]
        assert loader._parse_env_value('a,b,c') == ['a', 'b', 'c']

        # 字符串
        assert loader._parse_env_value('hello') == 'hello'


class TestGlobalConfig:
    """测试全局配置函数"""

    def test_get_config_singleton(self):
        """测试get_config返回单例"""
        from config.config_loader import get_config, _global_config

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_load_config_dict(self):
        """测试load_config返回字典"""
        from config.config_loader import load_config

        config = load_config()

        assert isinstance(config, dict)
        assert 'system' in config


class TestConfigConversion:
    """测试配置转换函数"""

    def test_get_water_canal_env_config(self):
        """测试获取WaterCanalEnvConfig"""
        from config.config_loader import get_water_canal_env_config
        from hydroe2e.phase5.ai_models.water_canal_env import WaterCanalEnvConfig

        config = get_water_canal_env_config()

        assert isinstance(config, WaterCanalEnvConfig)
        assert config.dt > 0
        assert config.observation_window > 0

    def test_get_neural_physics_config(self):
        """测试获取NeuralPhysicsConfig"""
        from config.config_loader import get_neural_physics_config
        from hydroe2e.phase5.ai_models.neural_physics_engine import NeuralPhysicsConfig

        config = get_neural_physics_config()

        assert isinstance(config, NeuralPhysicsConfig)
        assert config.hidden_dim > 0
        assert config.num_layers > 0

    def test_get_scenario_encoder_config(self):
        """测试获取DeepEncoderConfig"""
        from config.config_loader import get_scenario_encoder_config
        from hydroe2e.phase5.ai_models.deep_scenario_encoder import DeepEncoderConfig

        config = get_scenario_encoder_config()

        assert isinstance(config, DeepEncoderConfig)
        assert config.embedding_dim > 0
        assert config.encoder_type in ['cnn', 'transformer']

    def test_get_scenario_vae_config(self):
        """测试获取ScenarioVAEConfig"""
        from config.config_loader import get_scenario_vae_config
        from hydroe2e.phase5.ai_models.scenario_vae import ScenarioVAEConfig

        config = get_scenario_vae_config()

        assert isinstance(config, ScenarioVAEConfig)
        assert config.latent_dim > 0
        assert config.sequence_length > 0

    def test_get_e2e_controller_config(self):
        """测试获取AutonomousConfig"""
        from config.config_loader import get_e2e_controller_config
        from hydroe2e.phase5.ai_models.l4_autonomous.e2e_controller import AutonomousConfig

        config = get_e2e_controller_config()

        assert isinstance(config, AutonomousConfig)
        assert config.num_pools > 0
        assert config.num_gates > 0

    def test_get_world_model_config(self):
        """测试获取WorldModelConfig"""
        from config.config_loader import get_world_model_config
        from hydroe2e.phase5.ai_models.l4_autonomous.full_line_world_model import WorldModelConfig

        config = get_world_model_config()

        assert isinstance(config, WorldModelConfig)
        assert config.total_length > 0
        assert len(config.prediction_horizons) > 0

    def test_get_data_generator_config(self):
        """测试获取DataGeneratorConfig"""
        from config.config_loader import get_data_generator_config
        from hydroe2e.phase5.ai_models.data_generator import DataGeneratorConfig

        config = get_data_generator_config()

        assert isinstance(config, DataGeneratorConfig)
        assert config.num_samples > 0


class TestCustomConfigFile:
    """测试自定义配置文件"""

    def test_load_custom_config(self):
        """测试加载自定义配置文件"""
        from config.config_loader import ConfigLoader

        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'system': {'name': 'CustomTest'},
                'custom_section': {'key': 'value'}
            }, f)
            temp_path = f.name

        try:
            loader = ConfigLoader(config_path=temp_path)
            loader.load()

            assert loader.get('system.name') == 'CustomTest'
            assert loader.get('custom_section.key') == 'value'
        finally:
            os.unlink(temp_path)


class TestDeepMerge:
    """测试深度合并功能"""

    def test_deep_merge_simple(self):
        """测试简单字典合并"""
        from config.config_loader import ConfigLoader

        loader = ConfigLoader()

        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}

        result = loader._deep_merge(base, override)

        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_deep_merge_nested(self):
        """测试嵌套字典合并"""
        from config.config_loader import ConfigLoader

        loader = ConfigLoader()

        base = {
            'section': {
                'a': 1,
                'b': {'x': 10, 'y': 20}
            }
        }
        override = {
            'section': {
                'b': {'y': 30, 'z': 40},
                'c': 3
            }
        }

        result = loader._deep_merge(base, override)

        assert result['section']['a'] == 1
        assert result['section']['b'] == {'x': 10, 'y': 30, 'z': 40}
        assert result['section']['c'] == 3


class TestConfigValidation:
    """测试配置验证"""

    def test_required_sections_exist(self):
        """测试必需的配置节存在"""
        from config.config_loader import load_config

        config = load_config()

        required_sections = [
            'system',
            'water_system',
            'timing',
            'environment',
            'neural_physics',
            'scenario_encoder',
            'scenario_vae',
            'autonomous_controller',
        ]

        for section in required_sections:
            assert section in config, f"Missing required section: {section}"

    def test_water_system_params_valid(self):
        """测试水利系统参数有效"""
        from config.config_loader import ConfigLoader, DEFAULT_CONFIG_PATH

        # 测试基础配置
        cfg = ConfigLoader(config_path=DEFAULT_CONFIG_PATH, environment='nonexistent')
        cfg.load()

        num_pools = cfg.get('water_system.num_pools')
        num_gates = cfg.get('water_system.num_gates')
        total_length = cfg.get('water_system.total_length')

        assert num_pools > 0
        assert num_gates > 0
        assert total_length > 0
        # 闸门数应该等于渠池数+1
        assert num_gates == num_pools + 1

    def test_timing_params_consistent(self):
        """测试时间参数一致性"""
        from config.config_loader import ConfigLoader, DEFAULT_CONFIG_PATH

        # 测试基础配置(不含环境覆盖)
        cfg = ConfigLoader(config_path=DEFAULT_CONFIG_PATH, environment='nonexistent')
        cfg.load()

        dt = cfg.get('timing.dt')
        episode_duration = cfg.get('timing.episode_duration')
        history_length = cfg.get('timing.history_length')

        assert dt > 0
        assert episode_duration > dt
        assert history_length > 0
        # 历史窗口应该小于一个episode
        assert history_length * dt <= episode_duration
