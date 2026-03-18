"""
E2E自主控制器单元测试
Unit Tests for E2E Autonomous Controller

测试覆盖:
- AutonomousConfig 配置
- SpatioTemporalAttention 时空注意力
- E2EAutonomousController 基本功能
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
import os

from hydroe2e.phase5.ai_models.l4_autonomous.e2e_controller import (
    AutonomyLevel,
    AutonomousConfig,
    SpatioTemporalAttention,
    E2EAutonomousController,
)


class TestAutonomyLevel:
    """测试自主等级枚举"""

    def test_level_values(self):
        """测试等级值"""
        assert AutonomyLevel.L0_MANUAL.value == 0
        assert AutonomyLevel.L1_ASSISTED.value == 1
        assert AutonomyLevel.L2_PARTIAL.value == 2
        assert AutonomyLevel.L3_CONDITIONAL.value == 3
        assert AutonomyLevel.L4_HIGH.value == 4
        assert AutonomyLevel.L5_FULL.value == 5

    def test_level_ordering(self):
        """测试等级排序"""
        levels = list(AutonomyLevel)
        for i in range(len(levels) - 1):
            assert levels[i].value < levels[i + 1].value


class TestAutonomousConfig:
    """测试AutonomousConfig配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = AutonomousConfig()
        assert config.num_pools == 63
        assert config.num_gates == 64
        assert config.state_dim == 128
        assert config.history_length == 96
        assert config.prediction_horizon == 48

    def test_custom_config(self):
        """测试自定义配置"""
        config = AutonomousConfig(
            num_pools=10,
            num_gates=11,
            state_dim=64,
            hidden_dim=128,
        )
        assert config.num_pools == 10
        assert config.num_gates == 11
        assert config.state_dim == 64
        assert config.hidden_dim == 128

    def test_decision_parameters(self):
        """测试决策参数"""
        config = AutonomousConfig(
            confidence_threshold=0.5,
            safety_margin=0.3,
        )
        assert config.confidence_threshold == 0.5
        assert config.safety_margin == 0.3


class TestSpatioTemporalAttention:
    """测试时空注意力机制"""

    @pytest.fixture
    def attention(self):
        """创建注意力模块"""
        config = AutonomousConfig(
            num_pools=5,
            state_dim=32,
            num_heads=4,
            history_length=8,
        )
        return SpatioTemporalAttention(config)

    def test_attention_creation(self, attention):
        """测试注意力模块创建"""
        assert attention is not None
        assert attention.spatial_attention is not None
        assert attention.temporal_attention is not None

    def test_forward_pass(self, attention):
        """测试前向传播"""
        batch_size = 2
        time_steps = 8
        num_pools = 5
        state_dim = 32

        x = torch.randn(batch_size, time_steps, num_pools, state_dim)
        output = attention(x)

        assert output.shape == (batch_size, num_pools, state_dim)

    def test_no_nan_output(self, attention):
        """测试输出无NaN"""
        x = torch.randn(2, 8, 5, 32)
        output = attention(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestE2EAutonomousController:
    """测试E2E自主控制器"""

    @pytest.fixture
    def controller(self):
        """创建控制器"""
        config = AutonomousConfig(
            num_pools=5,
            num_gates=6,
            state_dim=32,
            hidden_dim=64,
            action_dim=32,
            num_heads=4,
            num_layers=2,
            history_length=8,
            prediction_horizon=4,
            confidence_threshold=0.3,
        )
        return E2EAutonomousController(config)

    def test_controller_creation(self, controller):
        """测试控制器创建"""
        assert controller is not None
        assert controller.perception is not None
        assert controller.prediction is not None
        assert controller.decision is not None

    def test_autonomy_level(self, controller):
        """测试自主等级"""
        assert controller.autonomy_level == AutonomyLevel.L4_HIGH

    def test_controller_modules(self, controller):
        """测试控制器子模块"""
        # 检查主要组件存在
        assert hasattr(controller, 'perception')
        assert hasattr(controller, 'prediction')
        assert hasattr(controller, 'decision')
        assert hasattr(controller, 'config')

    def test_controller_parameters(self, controller):
        """测试控制器参数"""
        params = list(controller.parameters())
        assert len(params) > 0

        # 参数应该是tensor
        for param in params:
            assert isinstance(param, torch.Tensor)

    def test_controller_state_dict(self, controller):
        """测试状态字典"""
        state_dict = controller.state_dict()
        assert len(state_dict) > 0

        # 可以加载状态
        new_controller = E2EAutonomousController(controller.config)
        new_controller.load_state_dict(state_dict)

    def test_controller_train_eval_mode(self, controller):
        """测试训练/评估模式切换"""
        controller.train()
        assert controller.training

        controller.eval()
        assert not controller.training


class TestControllerConfig:
    """测试控制器配置"""

    def test_small_config(self):
        """测试小配置"""
        config = AutonomousConfig(
            num_pools=3,
            num_gates=4,
            state_dim=16,
            hidden_dim=32,
        )
        controller = E2EAutonomousController(config)
        assert controller.config.num_pools == 3

    def test_large_config(self):
        """测试大配置"""
        config = AutonomousConfig(
            num_pools=63,
            num_gates=64,
            state_dim=256,
            hidden_dim=512,
        )
        controller = E2EAutonomousController(config)
        assert controller.config.num_pools == 63


class TestEdgeCases:
    """测试边界情况"""

    def test_minimal_config(self):
        """测试最小配置"""
        config = AutonomousConfig(
            num_pools=1,
            num_gates=2,
            state_dim=16,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
        )
        controller = E2EAutonomousController(config)
        assert controller is not None

    def test_default_config_controller(self):
        """测试默认配置控制器"""
        controller = E2EAutonomousController()
        assert controller.config.num_pools == 63


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
