"""
Pytest配置和Fixtures
Pytest Configuration and Common Fixtures

提供测试所需的通用fixtures和配置
"""

import pytest
import numpy as np
import torch


# ==============================================================================
# 通用Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def device():
    """获取计算设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def random_seed():
    """固定随机种子"""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


# ==============================================================================
# 神经物理引擎Fixtures
# ==============================================================================

@pytest.fixture
def neural_physics_config():
    """神经物理引擎配置"""
    from hydroe2e.phase5.ai_models.neural_physics_engine import NeuralPhysicsConfig
    return NeuralPhysicsConfig(
        input_dim=5,
        hidden_dim=64,  # 使用较小的维度加速测试
        num_layers=1,
        sequence_length=8,
        batch_size=4,
        num_epochs=2,
    )


@pytest.fixture
def neural_physics_engine(neural_physics_config):
    """创建神经物理引擎实例"""
    from hydroe2e.phase5.ai_models.neural_physics_engine import NeuralPhysicsEngine
    return NeuralPhysicsEngine(neural_physics_config)


@pytest.fixture
def sample_sequence_data():
    """生成样本序列数据"""
    batch_size = 4
    seq_len = 8
    input_dim = 5
    return torch.randn(batch_size, seq_len, input_dim)


# ==============================================================================
# 场景编码器Fixtures
# ==============================================================================

@pytest.fixture
def encoder_config():
    """深度场景编码器配置"""
    from hydroe2e.phase5.ai_models.deep_scenario_encoder import DeepEncoderConfig
    return DeepEncoderConfig(
        sequence_length=48,
        input_channels=5,
        hidden_dim=64,
        embedding_dim=32,
        encoder_type='cnn',
    )


@pytest.fixture
def scenario_encoder(encoder_config):
    """创建场景编码器实例"""
    from hydroe2e.phase5.ai_models.deep_scenario_encoder import DeepScenarioEncoder
    return DeepScenarioEncoder(encoder_config)


@pytest.fixture
def sample_scenario_data():
    """生成样本场景数据"""
    seq_len = 48
    num_features = 5
    return np.random.randn(seq_len, num_features).astype(np.float32)


# ==============================================================================
# 场景VAE Fixtures
# ==============================================================================

@pytest.fixture
def vae_config():
    """场景VAE配置"""
    from hydroe2e.phase5.ai_models.scenario_vae import ScenarioVAEConfig
    return ScenarioVAEConfig(
        sequence_length=48,
        input_channels=5,
        hidden_dim=64,
        latent_dim=16,
    )


@pytest.fixture
def scenario_vae(vae_config):
    """创建场景VAE实例"""
    from hydroe2e.phase5.ai_models.scenario_vae import ScenarioVAE
    return ScenarioVAE(vae_config)


# ==============================================================================
# 水渠环境Fixtures
# ==============================================================================

@pytest.fixture
def env_config():
    """水渠环境配置"""
    from hydroe2e.phase5.ai_models.water_canal_env import WaterCanalEnvConfig
    return WaterCanalEnvConfig(
        episode_duration=3600.0,  # 1小时
        dt=300.0,  # 5分钟
        observation_window=8,
    )


@pytest.fixture
def water_canal_env(env_config):
    """创建水渠环境实例"""
    from hydroe2e.phase5.ai_models.water_canal_env import OneGateTwoPoolsEnv
    env = OneGateTwoPoolsEnv(env_config)
    return env


# ==============================================================================
# E2E控制器Fixtures
# ==============================================================================

@pytest.fixture
def e2e_config():
    """E2E控制器配置"""
    from hydroe2e.phase5.ai_models.l4_autonomous.e2e_controller import E2EControllerConfig
    return E2EControllerConfig(
        num_pools=5,
        num_gates=6,
        observation_window=8,
        prediction_horizon=4,
        hidden_dim=64,
        confidence_threshold=0.3,
    )


@pytest.fixture
def e2e_controller(e2e_config):
    """创建E2E控制器实例"""
    from hydroe2e.phase5.ai_models.l4_autonomous.e2e_controller import E2EAutonomousController
    return E2EAutonomousController(e2e_config)


# ==============================================================================
# 测试数据生成工具
# ==============================================================================

@pytest.fixture
def generate_water_state():
    """生成水位状态数据的工厂函数"""
    def _generate(num_pools=10, num_gates=11, seq_len=16):
        return {
            'levels': np.random.uniform(3.5, 4.5, (seq_len, num_pools)),
            'flows': np.random.uniform(200, 300, (seq_len, num_pools)),
            'gates': np.random.uniform(0.3, 0.9, (seq_len, num_gates)),
            'demands': np.random.uniform(0, 10, (seq_len, num_pools)),
            'weather': np.random.uniform(0, 1, (seq_len,)),
        }
    return _generate


# ==============================================================================
# pytest配置
# ==============================================================================

def pytest_configure(config):
    """pytest配置钩子"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """跳过GPU测试如果没有CUDA"""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
