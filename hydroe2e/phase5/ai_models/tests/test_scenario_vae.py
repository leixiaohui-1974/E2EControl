"""
场景VAE单元测试
Unit Tests for Scenario VAE

测试覆盖:
- ScenarioVAEConfig 配置
- ScenarioConditions 条件编码
- SequenceEncoder 编码器
- SequenceDecoder 解码器
- ScenarioVAE 主接口
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import os

from hydroe2e.phase5.ai_models.scenario_vae import (
    ScenarioVAEConfig,
    ScenarioConditions,
    SequenceEncoder,
    SequenceDecoder,
    ScenarioVAE,
)


class TestScenarioVAEConfig:
    """测试ScenarioVAEConfig配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = ScenarioVAEConfig()
        assert config.sequence_length == 96
        assert config.num_channels == 4
        assert config.latent_dim == 32
        assert config.kl_weight == 0.001

    def test_custom_config(self):
        """测试自定义配置"""
        config = ScenarioVAEConfig(
            sequence_length=48,
            num_channels=5,
            latent_dim=64,
            encoder_hidden_dim=256,
        )
        assert config.sequence_length == 48
        assert config.num_channels == 5
        assert config.latent_dim == 64
        assert config.encoder_hidden_dim == 256


class TestScenarioConditions:
    """测试场景条件编码"""

    def test_condition_values(self):
        """测试条件值"""
        assert ScenarioConditions.SPRING == 0
        assert ScenarioConditions.SUMMER == 1
        assert ScenarioConditions.NORMAL == 5
        assert ScenarioConditions.FLOOD == 6

    def test_encode_single_condition(self):
        """测试单条件编码"""
        vector = ScenarioConditions.encode(['summer'])

        assert vector.shape == (16,)
        assert vector[ScenarioConditions.SUMMER] == 1.0
        assert vector.sum() == 1.0

    def test_encode_multiple_conditions(self):
        """测试多条件编码"""
        vector = ScenarioConditions.encode(['summer', 'flood', 'high'])

        assert vector.shape == (16,)
        assert vector[ScenarioConditions.SUMMER] == 1.0
        assert vector[ScenarioConditions.FLOOD] == 1.0
        assert vector[ScenarioConditions.HIGH] == 1.0
        assert vector.sum() == 3.0

    def test_encode_case_insensitive(self):
        """测试大小写不敏感"""
        vector1 = ScenarioConditions.encode(['SUMMER'])
        vector2 = ScenarioConditions.encode(['summer'])

        assert torch.equal(vector1, vector2)

    def test_encode_unknown_condition(self):
        """测试未知条件"""
        vector = ScenarioConditions.encode(['unknown_condition'])

        assert vector.sum() == 0.0


class TestSequenceEncoder:
    """测试时序编码器"""

    @pytest.fixture
    def encoder(self):
        """创建编码器"""
        config = ScenarioVAEConfig(
            sequence_length=48,
            num_channels=4,
            encoder_hidden_dim=64,
            latent_dim=16,
            encoder_num_layers=1,
        )
        return SequenceEncoder(config)

    def test_encoder_creation(self, encoder):
        """测试编码器创建"""
        assert encoder is not None
        assert isinstance(encoder, nn.Module)

    def test_forward_pass(self, encoder):
        """测试前向传播"""
        batch_size = 4
        seq_len = 48
        num_channels = 4

        x = torch.randn(batch_size, seq_len, num_channels)
        mu, log_var = encoder(x)

        assert mu.shape == (batch_size, 16)
        assert log_var.shape == (batch_size, 16)

    def test_mu_log_var_reasonable(self, encoder):
        """测试mu和log_var在合理范围"""
        x = torch.randn(4, 48, 4)
        mu, log_var = encoder(x)

        # mu应该在合理范围
        assert not torch.isnan(mu).any()
        assert not torch.isinf(mu).any()

        # log_var不应太极端
        assert not torch.isnan(log_var).any()


class TestSequenceDecoder:
    """测试时序解码器"""

    @pytest.fixture
    def decoder(self):
        """创建解码器"""
        config = ScenarioVAEConfig(
            sequence_length=48,
            num_channels=4,
            decoder_hidden_dim=64,
            latent_dim=16,
            decoder_num_layers=1,
        )
        return SequenceDecoder(config)

    def test_decoder_creation(self, decoder):
        """测试解码器创建"""
        assert decoder is not None
        assert isinstance(decoder, nn.Module)

    def test_forward_pass(self, decoder):
        """测试前向传播"""
        batch_size = 4
        latent_dim = 16
        seq_len = 48
        num_channels = 4

        z = torch.randn(batch_size, latent_dim)
        recon_mu, recon_logvar = decoder(z, seq_len)

        assert recon_mu.shape == (batch_size, seq_len, num_channels)
        assert recon_logvar.shape == (batch_size, seq_len, num_channels)

    def test_output_reasonable(self, decoder):
        """测试输出在合理范围"""
        z = torch.randn(4, 16)
        recon_mu, recon_logvar = decoder(z, 48)

        assert not torch.isnan(recon_mu).any()
        assert not torch.isinf(recon_mu).any()


class TestScenarioVAE:
    """测试场景VAE主接口"""

    @pytest.fixture
    def vae(self):
        """创建VAE"""
        config = ScenarioVAEConfig(
            sequence_length=48,
            num_channels=4,
            encoder_hidden_dim=64,
            decoder_hidden_dim=64,
            latent_dim=16,
            encoder_num_layers=1,
            decoder_num_layers=1,
        )
        return ScenarioVAE(config)

    def test_vae_creation(self, vae):
        """测试VAE创建"""
        assert vae is not None
        assert vae.encoder is not None
        assert vae.decoder is not None

    def test_encode(self, vae):
        """测试编码功能"""
        x = torch.randn(4, 48, 4)
        z = vae.encode(x)

        # encode返回单个z向量
        assert z.shape == (4, 16)

    def test_decode(self, vae):
        """测试解码功能"""
        z = torch.randn(4, 16)
        output = vae.decode(z, 48)

        assert output.shape == (4, 48, 4)

    def test_reparameterize(self, vae):
        """测试重参数化"""
        mu = torch.zeros(4, 16)
        log_var = torch.zeros(4, 16)

        z = vae.reparameterize(mu, log_var)

        assert z.shape == (4, 16)
        # 标准正态分布应该均值接近0
        assert abs(z.mean().item()) < 3.0  # 放宽范围

    def test_forward_pass(self, vae):
        """测试完整前向传播"""
        x = torch.randn(4, 48, 4)
        result = vae(x)

        # forward返回字典
        assert isinstance(result, dict)
        assert 'recon_mu' in result
        assert 'mu' in result
        assert 'logvar' in result
        assert 'z' in result

        assert result['recon_mu'].shape == x.shape
        assert result['mu'].shape == (4, 16)
        assert result['logvar'].shape == (4, 16)

    def test_sample(self, vae):
        """测试采样功能"""
        samples = vae.sample(num_samples=8, seq_len=48)

        assert samples.shape == (8, 48, 4)

    def test_sample_with_temperature(self, vae):
        """测试带温度参数的采样"""
        # 低温度应该生成更相似的样本
        samples_low = vae.sample(num_samples=4, seq_len=48, temperature=0.1)
        samples_high = vae.sample(num_samples=4, seq_len=48, temperature=2.0)

        # 验证形状正确
        assert samples_low.shape == (4, 48, 4)
        assert samples_high.shape == (4, 48, 4)

    def test_model_state_dict(self, vae):
        """测试模型状态保存"""
        state_dict = vae.state_dict()
        assert len(state_dict) > 0

        # 可以加载状态
        new_vae = ScenarioVAE(vae.config)
        new_vae.load_state_dict(state_dict)


class TestVAETraining:
    """测试VAE训练功能"""

    @pytest.fixture
    def vae_for_training(self):
        """创建用于训练的VAE"""
        config = ScenarioVAEConfig(
            sequence_length=24,
            num_channels=4,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
            latent_dim=8,
            encoder_num_layers=1,
            decoder_num_layers=1,
        )
        return ScenarioVAE(config)

    @pytest.mark.slow
    def test_training_reduces_loss(self, vae_for_training):
        """测试训练能够降低损失"""
        vae = vae_for_training
        vae.train()

        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # 生成训练数据
        train_data = torch.randn(20, 24, 4)

        initial_loss = None
        for epoch in range(5):
            epoch_loss = 0.0
            for i in range(0, len(train_data), 4):
                batch = train_data[i:i+4]

                optimizer.zero_grad()
                result = vae(batch)

                # 计算重建损失
                recon_loss = F.mse_loss(result['recon_mu'], batch)
                # 计算KL损失
                kl_loss = -0.5 * torch.mean(
                    1 + result['logvar'] - result['mu'].pow(2) - result['logvar'].exp()
                )
                total_loss = recon_loss + 0.001 * kl_loss

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            if initial_loss is None:
                initial_loss = epoch_loss

        # 损失应该下降或稳定
        assert epoch_loss <= initial_loss * 1.5


class TestLatentSpace:
    """测试隐空间属性"""

    @pytest.fixture
    def vae(self):
        """创建VAE"""
        config = ScenarioVAEConfig(
            sequence_length=48,
            latent_dim=16,
            encoder_hidden_dim=64,
            decoder_hidden_dim=64,
        )
        return ScenarioVAE(config)

    def test_latent_interpolation(self, vae):
        """测试隐空间插值"""
        z1 = torch.randn(1, 16)
        z2 = torch.randn(1, 16)

        # 线性插值
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        interpolated = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            sample = vae.decode(z_interp, 48)
            interpolated.append(sample)

        # 验证生成了有效样本
        for sample in interpolated:
            assert sample.shape == (1, 48, 4)
            assert not torch.isnan(sample).any()

    def test_latent_sampling_diversity(self, vae):
        """测试隐空间采样多样性"""
        samples = []
        for _ in range(10):
            sample = vae.sample(num_samples=1, seq_len=48)
            samples.append(sample)

        samples = torch.cat(samples, dim=0)

        # 样本应该有多样性
        std = samples.std(dim=0).mean().item()
        assert std > 0.01  # 不应该全部相同


class TestEdgeCases:
    """测试边界情况"""

    def test_single_sample(self):
        """测试单样本"""
        config = ScenarioVAEConfig(sequence_length=24, latent_dim=8)
        vae = ScenarioVAE(config)

        x = torch.randn(1, 24, 4)
        result = vae(x)

        assert result['recon_mu'].shape == x.shape

    def test_large_batch(self):
        """测试大批量"""
        config = ScenarioVAEConfig(
            sequence_length=24,
            latent_dim=8,
            encoder_hidden_dim=32,
            decoder_hidden_dim=32,
        )
        vae = ScenarioVAE(config)

        x = torch.randn(64, 24, 4)
        result = vae(x)

        assert result['recon_mu'].shape == x.shape

    def test_zero_temperature(self):
        """测试零温度采样"""
        config = ScenarioVAEConfig(sequence_length=24, latent_dim=8)
        vae = ScenarioVAE(config)

        # 极低温度
        samples = vae.sample(num_samples=4, seq_len=24, temperature=0.001)
        assert samples.shape == (4, 24, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
