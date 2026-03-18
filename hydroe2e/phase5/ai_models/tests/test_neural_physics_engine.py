"""
神经物理引擎单元测试
Unit Tests for Neural Physics Engine

测试覆盖:
- NeuralPhysicsConfig 配置
- LSTMSurrogateModel 模型
- GRUSurrogateModel 模型
- PINNLoss 损失函数
- NeuralPhysicsEngine 主接口
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
import os

from hydroe2e.phase5.ai_models.neural_physics_engine import (
    NeuralPhysicsConfig,
    LSTMSurrogateModel,
    GRUSurrogateModel,
    PINNLoss,
    NeuralPhysicsEngine,
)


class TestNeuralPhysicsConfig:
    """测试NeuralPhysicsConfig配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = NeuralPhysicsConfig()
        assert config.input_dim == 5
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.output_dim == 1
        assert config.sequence_length == 16
        assert config.use_physics_loss is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = NeuralPhysicsConfig(
            input_dim=10,
            hidden_dim=256,
            num_layers=3,
            dropout=0.2,
        )
        assert config.input_dim == 10
        assert config.hidden_dim == 256
        assert config.num_layers == 3
        assert config.dropout == 0.2


class TestLSTMSurrogateModel:
    """测试LSTM代理模型"""

    @pytest.fixture
    def lstm_model(self):
        """创建LSTM模型"""
        config = NeuralPhysicsConfig(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
            sequence_length=8,
        )
        return LSTMSurrogateModel(config)

    def test_model_creation(self, lstm_model):
        """测试模型创建"""
        assert lstm_model is not None
        assert isinstance(lstm_model, nn.Module)

    def test_forward_pass(self, lstm_model):
        """测试前向传播"""
        batch_size = 4
        seq_len = 8
        input_dim = 5

        x = torch.randn(batch_size, seq_len, input_dim)
        output = lstm_model(x)

        assert output.shape == (batch_size, 1)

    def test_forward_batch_independence(self, lstm_model):
        """测试批次独立性"""
        x1 = torch.randn(2, 8, 5)
        x2 = torch.randn(4, 8, 5)

        out1 = lstm_model(x1)
        out2 = lstm_model(x2)

        assert out1.shape[0] == 2
        assert out2.shape[0] == 4

    def test_gradient_flow(self, lstm_model):
        """测试梯度流动"""
        x = torch.randn(2, 8, 5, requires_grad=True)
        output = lstm_model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestGRUSurrogateModel:
    """测试GRU代理模型"""

    @pytest.fixture
    def gru_model(self):
        """创建GRU模型"""
        config = NeuralPhysicsConfig(
            input_dim=5,
            hidden_dim=64,
            num_layers=1,
            sequence_length=8,
        )
        return GRUSurrogateModel(config)

    def test_model_creation(self, gru_model):
        """测试模型创建"""
        assert gru_model is not None
        assert isinstance(gru_model, nn.Module)

    def test_forward_pass(self, gru_model):
        """测试前向传播"""
        batch_size = 4
        seq_len = 8
        input_dim = 5

        x = torch.randn(batch_size, seq_len, input_dim)
        output = gru_model(x)

        assert output.shape == (batch_size, 1)

    def test_parameter_count_less_than_lstm(self):
        """测试GRU参数少于LSTM"""
        config = NeuralPhysicsConfig(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
        )
        lstm = LSTMSurrogateModel(config)
        gru = GRUSurrogateModel(config)

        lstm_params = sum(p.numel() for p in lstm.parameters())
        gru_params = sum(p.numel() for p in gru.parameters())

        # GRU参数应该少于LSTM (因为没有cell state)
        assert gru_params < lstm_params


class TestPINNLoss:
    """测试物理信息损失函数"""

    @pytest.fixture
    def pinn_loss(self):
        """创建PINN损失"""
        return PINNLoss(
            mass_balance_weight=0.1,
            level_constraint_weight=0.05,
            smoothness_weight=0.01,
        )

    def test_loss_creation(self, pinn_loss):
        """测试损失函数创建"""
        assert pinn_loss is not None
        assert pinn_loss.mass_balance_weight == 0.1

    def test_basic_loss_computation(self, pinn_loss):
        """测试基本损失计算"""
        batch_size = 4
        pred = torch.randn(batch_size, 1)
        target = torch.randn(batch_size, 1)
        q_in = torch.abs(torch.randn(batch_size, 1)) * 100
        q_out = torch.abs(torch.randn(batch_size, 1)) * 100

        losses = pinn_loss(pred, target, q_in, q_out)

        assert 'mse_loss' in losses
        assert 'total_loss' in losses
        assert losses['mse_loss'] >= 0

    def test_loss_with_prev_level(self, pinn_loss):
        """测试带前一时刻水位的损失"""
        batch_size = 4
        pred = torch.randn(batch_size, 1) + 4.0
        target = torch.randn(batch_size, 1) + 4.0
        q_in = torch.abs(torch.randn(batch_size, 1)) * 100
        q_out = torch.abs(torch.randn(batch_size, 1)) * 100
        prev_level = torch.randn(batch_size, 1) + 4.0

        losses = pinn_loss(pred, target, q_in, q_out, prev_level)

        assert 'mass_balance_loss' in losses
        assert 'smoothness_loss' in losses

    def test_negative_level_penalty(self, pinn_loss):
        """测试负水位惩罚"""
        batch_size = 4
        pred = torch.tensor([[-1.0], [2.0], [-0.5], [3.0]])  # 包含负值
        target = torch.tensor([[1.0], [2.0], [0.5], [3.0]])
        q_in = torch.ones(batch_size, 1) * 100
        q_out = torch.ones(batch_size, 1) * 100

        losses = pinn_loss(pred, target, q_in, q_out)

        assert losses['level_constraint_loss'] > 0  # 应该有惩罚


class TestNeuralPhysicsEngine:
    """测试神经物理引擎主接口"""

    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        config = NeuralPhysicsConfig(
            input_dim=5,
            hidden_dim=64,
            num_layers=1,
            sequence_length=8,
        )
        return NeuralPhysicsEngine(config, device='cpu')

    def test_engine_creation(self, engine):
        """测试引擎创建"""
        assert engine is not None
        assert engine.device.type == 'cpu'
        assert engine.model is not None

    def test_reset(self, engine):
        """测试重置功能"""
        engine.reset(initial_level=3.5, initial_inflow=150.0)

        assert engine.current_level == 3.5
        assert engine.current_inflow == 150.0

    def test_step(self, engine):
        """测试单步模拟"""
        engine.reset(initial_level=4.0, initial_inflow=100.0)

        # 填充历史数据
        for _ in range(10):
            engine.step(
                q_in=100.0,
                gate_opening=0.5,
                diversion=5.0
            )

        result = engine.step(
            q_in=100.0,
            gate_opening=0.5,
            diversion=5.0
        )

        # step返回水位值 (numpy float或python float)
        assert isinstance(result, (float, np.floating))

    def test_predict_multi_step(self, engine):
        """测试多步预测"""
        engine.reset(initial_level=4.0, initial_inflow=100.0)

        # 填充历史数据
        for _ in range(10):
            engine.step(q_in=100.0, gate_opening=0.5, diversion=5.0)

        predictions = engine.predict(
            steps=4,
            future_inflows=np.array([100.0] * 4),
        )

        assert len(predictions) == 4

    def test_model_save_load(self, engine):
        """测试模型保存加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pt')

            # 保存
            engine.save_model(model_path)
            assert os.path.exists(model_path)

            # 加载
            new_engine = NeuralPhysicsEngine(engine.config, device='cpu')
            new_engine.load_model(model_path)

            # 验证参数一致
            for p1, p2 in zip(engine.model.parameters(), new_engine.model.parameters()):
                assert torch.allclose(p1, p2)

    def test_train_mode_toggle(self, engine):
        """测试训练模式切换"""
        engine.model.train()
        assert engine.model.training

        engine.model.eval()
        assert not engine.model.training


class TestNeuralPhysicsEngineTraining:
    """测试神经物理引擎训练功能"""

    @pytest.fixture
    def engine_with_data(self):
        """创建带训练数据的引擎"""
        config = NeuralPhysicsConfig(
            input_dim=5,
            hidden_dim=32,
            num_layers=1,
            sequence_length=8,
            batch_size=4,
            num_epochs=2,
        )
        return NeuralPhysicsEngine(config, device='cpu')

    @pytest.mark.slow
    def test_training_loop(self, engine_with_data):
        """测试训练循环"""
        # 生成简单训练数据
        num_samples = 20
        seq_len = 8

        train_data = []
        for _ in range(num_samples):
            x = np.random.randn(seq_len, 5).astype(np.float32)
            y = np.random.randn(1).astype(np.float32)
            train_data.append((x, y))

        # 执行简单训练
        engine_with_data.model.train()
        optimizer = torch.optim.Adam(engine_with_data.model.parameters(), lr=1e-3)

        initial_loss = None
        for epoch in range(2):
            epoch_loss = 0.0
            for x, y in train_data:
                x_tensor = torch.from_numpy(x).unsqueeze(0)
                y_tensor = torch.from_numpy(y).unsqueeze(0)

                optimizer.zero_grad()
                pred = engine_with_data.model(x_tensor)
                loss = nn.MSELoss()(pred, y_tensor)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if initial_loss is None:
                initial_loss = epoch_loss

        # 验证损失下降或保持稳定
        assert epoch_loss <= initial_loss * 1.5  # 允许一些波动


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_history_step(self):
        """测试空历史数据时的step"""
        config = NeuralPhysicsConfig(sequence_length=8)
        engine = NeuralPhysicsEngine(config, device='cpu')

        # 应该能处理空历史
        result = engine.step(q_in=100.0, gate_opening=0.5, diversion=5.0)
        assert isinstance(result, (float, np.floating))

    def test_extreme_values(self):
        """测试极端值"""
        config = NeuralPhysicsConfig(sequence_length=8)
        engine = NeuralPhysicsEngine(config, device='cpu')
        engine.reset(initial_level=4.0)

        # 极端闸门开度
        result1 = engine.step(q_in=100.0, gate_opening=1.0, diversion=0.0)
        assert np.isfinite(result1)

        result2 = engine.step(q_in=100.0, gate_opening=0.0, diversion=100.0)
        assert np.isfinite(result2)

    def test_batch_size_variations(self):
        """测试不同批大小"""
        config = NeuralPhysicsConfig(hidden_dim=32, num_layers=1)
        model = LSTMSurrogateModel(config)

        for batch_size in [1, 2, 8, 16, 32]:
            x = torch.randn(batch_size, 16, 5)
            output = model(x)
            assert output.shape == (batch_size, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
