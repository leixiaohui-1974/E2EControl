"""
深度场景编码器单元测试
Unit Tests for Deep Scenario Encoder

测试覆盖:
- DeepEncoderConfig 配置
- CNN1DEncoder 编码器
- TransformerEncoder 编码器
- DeepScenarioEncoder 主接口
- ScenarioVectorDB 向量数据库
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
import os

from hydroe2e.phase5.ai_models.deep_scenario_encoder import (
    DeepEncoderConfig,
    CNN1DEncoder,
    TransformerEncoder,
    DeepScenarioEncoder,
    ScenarioVectorDB,
)


class TestDeepEncoderConfig:
    """测试DeepEncoderConfig配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = DeepEncoderConfig()
        assert config.sequence_length == 96
        assert config.input_channels == 5
        assert config.hidden_dim == 128
        assert config.embedding_dim == 64
        assert config.encoder_type == 'cnn'

    def test_custom_config(self):
        """测试自定义配置"""
        config = DeepEncoderConfig(
            sequence_length=48,
            input_channels=10,
            hidden_dim=256,
            encoder_type='transformer',
        )
        assert config.sequence_length == 48
        assert config.input_channels == 10
        assert config.encoder_type == 'transformer'


class TestCNN1DEncoder:
    """测试1D CNN编码器"""

    @pytest.fixture
    def cnn_encoder(self):
        """创建CNN编码器"""
        config = DeepEncoderConfig(
            sequence_length=48,
            input_channels=5,
            hidden_dim=64,
            embedding_dim=32,
            num_layers=2,
        )
        return CNN1DEncoder(config)

    def test_encoder_creation(self, cnn_encoder):
        """测试编码器创建"""
        assert cnn_encoder is not None
        assert isinstance(cnn_encoder, nn.Module)

    def test_forward_pass(self, cnn_encoder):
        """测试前向传播"""
        batch_size = 4
        seq_len = 48
        input_channels = 5

        x = torch.randn(batch_size, seq_len, input_channels)
        embedding = cnn_encoder(x)

        assert embedding.shape == (batch_size, 32)

    def test_embedding_normalized(self, cnn_encoder):
        """测试嵌入向量已归一化"""
        x = torch.randn(4, 48, 5)
        embedding = cnn_encoder(x)

        # L2范数应该接近1
        norms = torch.norm(embedding, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_different_sequence_lengths(self):
        """测试不同序列长度"""
        for seq_len in [24, 48, 96, 192]:
            config = DeepEncoderConfig(
                sequence_length=seq_len,
                hidden_dim=32,
                embedding_dim=16,
                num_layers=2,
            )
            encoder = CNN1DEncoder(config)
            x = torch.randn(2, seq_len, 5)
            embedding = encoder(x)
            assert embedding.shape == (2, 16)


class TestTransformerEncoder:
    """测试Transformer编码器"""

    @pytest.fixture
    def transformer_encoder(self):
        """创建Transformer编码器"""
        config = DeepEncoderConfig(
            sequence_length=48,
            input_channels=5,
            hidden_dim=64,
            embedding_dim=32,
            num_layers=2,
            num_heads=4,
            encoder_type='transformer',
        )
        return TransformerEncoder(config)

    def test_encoder_creation(self, transformer_encoder):
        """测试编码器创建"""
        assert transformer_encoder is not None

    def test_forward_pass(self, transformer_encoder):
        """测试前向传播"""
        batch_size = 4
        seq_len = 48
        input_channels = 5

        x = torch.randn(batch_size, seq_len, input_channels)
        embedding = transformer_encoder(x)

        assert embedding.shape == (batch_size, 32)

    def test_attention_mechanism(self, transformer_encoder):
        """测试注意力机制"""
        x = torch.randn(2, 48, 5)
        embedding = transformer_encoder(x)

        # 验证输出稳定
        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()


class TestDeepScenarioEncoder:
    """测试深度场景编码器主接口"""

    @pytest.fixture
    def scenario_encoder(self):
        """创建场景编码器"""
        config = DeepEncoderConfig(
            sequence_length=48,
            input_channels=5,
            hidden_dim=64,
            embedding_dim=32,
            encoder_type='cnn',
        )
        return DeepScenarioEncoder(config)

    def test_encoder_creation(self, scenario_encoder):
        """测试编码器创建"""
        assert scenario_encoder is not None
        assert scenario_encoder.config.embedding_dim == 32

    def test_encode_numpy(self, scenario_encoder):
        """测试NumPy数组编码"""
        data = np.random.randn(48, 5).astype(np.float32)
        embedding = scenario_encoder.encode(data)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (32,)

    def test_encode_torch(self, scenario_encoder):
        """测试PyTorch张量编码"""
        data = torch.randn(48, 5)
        embedding = scenario_encoder.encode(data)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (32,)

    def test_model_save_load(self, scenario_encoder):
        """测试模型保存加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'encoder.pt')

            # 保存
            scenario_encoder.save_model(model_path)
            assert os.path.exists(model_path)

            # 加载
            new_encoder = DeepScenarioEncoder(scenario_encoder.config)
            new_encoder.load_model(model_path)

            # 验证编码一致
            data = np.random.randn(48, 5).astype(np.float32)
            emb1 = scenario_encoder.encode(data)
            emb2 = new_encoder.encode(data)

            assert np.allclose(emb1, emb2, atol=1e-5)


class TestScenarioVectorDB:
    """测试场景向量数据库"""

    @pytest.fixture
    def vector_db(self):
        """创建向量数据库"""
        return ScenarioVectorDB(embedding_dim=32)

    def test_db_creation(self, vector_db):
        """测试数据库创建"""
        assert vector_db is not None
        assert vector_db.embedding_dim == 32
        assert len(vector_db.vectors) == 0

    def test_add_vector(self, vector_db):
        """测试添加向量"""
        embedding = np.random.randn(32).astype(np.float32)
        vector_db.add(embedding, 'scenario_001', {'category': 'normal'})

        assert len(vector_db.vectors) == 1

    def test_add_batch(self, vector_db):
        """测试批量添加"""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        scenario_ids = [f'scenario_{i:03d}' for i in range(10)]
        metadata_list = [{'idx': i} for i in range(10)]

        vector_db.add_batch(embeddings, scenario_ids, metadata_list)

        assert len(vector_db.vectors) == 10

    def test_search(self, vector_db):
        """测试向量检索"""
        # 添加一些向量
        for i in range(20):
            emb = np.random.randn(32).astype(np.float32)
            emb = emb / np.linalg.norm(emb)  # 归一化
            vector_db.add(emb, f'scenario_{i:03d}', {'idx': i})

        # 搜索
        query = np.random.randn(32).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = vector_db.search(query, top_k=5)

        assert len(results) == 5
        # 结果应该是 (scenario_id, score, metadata) 三元组
        for scenario_id, score, metadata in results:
            assert isinstance(scenario_id, str)
            assert isinstance(score, float)
            assert isinstance(metadata, dict)

    def test_search_similarity_scores(self, vector_db):
        """测试相似度分数正确性"""
        # 添加一个已知向量
        known_emb = np.array([1.0] + [0.0] * 31, dtype=np.float32)
        vector_db.add(known_emb, 'known', {})

        # 添加更多随机向量
        for i in range(10):
            emb = np.random.randn(32).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            vector_db.add(emb, f'random_{i}', {})

        # 用相同向量查询应该得到高分
        results = vector_db.search(known_emb, top_k=1)
        scenario_id, score, metadata = results[0]

        assert score > 0.99  # 应该非常接近1

    def test_save_load(self, vector_db):
        """测试数据库保存加载"""
        # 添加数据
        for i in range(10):
            emb = np.random.randn(32).astype(np.float32)
            vector_db.add(emb, f'scenario_{i:03d}', {'idx': i})

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'vectors.npz')

            # 保存 (使用npz格式)
            vector_db.save(db_path)
            assert os.path.exists(db_path)

            # 加载
            new_db = ScenarioVectorDB(embedding_dim=32)
            new_db.load(db_path)

            assert len(new_db.vectors) == len(vector_db.vectors)

    def test_empty_search(self, vector_db):
        """测试空数据库搜索"""
        query = np.random.randn(32).astype(np.float32)
        results = vector_db.search(query, top_k=5)

        assert len(results) == 0


class TestContrastiveLearning:
    """测试对比学习功能"""

    @pytest.fixture
    def encoder(self):
        """创建编码器"""
        config = DeepEncoderConfig(
            sequence_length=48,
            hidden_dim=64,
            embedding_dim=32,
        )
        return DeepScenarioEncoder(config)

    def test_similar_inputs_close_embeddings(self, encoder):
        """测试相似输入产生相近嵌入"""
        # 生成基础数据
        base = np.random.randn(48, 5).astype(np.float32)

        # 添加小噪声
        similar = base + np.random.randn(48, 5).astype(np.float32) * 0.01

        emb1 = encoder.encode(base)
        emb2 = encoder.encode(similar)

        # 计算余弦相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # 相似输入的嵌入应该相近 (但未经训练所以放宽标准)
        assert similarity > 0.5

    def test_different_inputs_different_embeddings(self, encoder):
        """测试不同输入产生不同嵌入"""
        data1 = np.random.randn(48, 5).astype(np.float32)
        data2 = np.random.randn(48, 5).astype(np.float32) * 10  # 明显不同

        emb1 = encoder.encode(data1)
        emb2 = encoder.encode(data2)

        # 不应该完全相同
        assert not np.allclose(emb1, emb2, atol=0.1)


class TestEdgeCases:
    """测试边界情况"""

    def test_single_sample_encoding(self):
        """测试单样本编码"""
        config = DeepEncoderConfig(sequence_length=48, embedding_dim=32)
        encoder = DeepScenarioEncoder(config)

        data = np.random.randn(48, 5).astype(np.float32)
        embedding = encoder.encode(data)

        assert embedding.shape == (32,)

    def test_vector_db_dimension_mismatch(self):
        """测试向量维度不匹配"""
        db = ScenarioVectorDB(embedding_dim=32)

        # 正确维度
        db.add(np.random.randn(32), 'good', {})

        # 错误维度应该引发异常
        with pytest.raises(AssertionError):
            db.add(np.random.randn(64), 'bad', {})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
