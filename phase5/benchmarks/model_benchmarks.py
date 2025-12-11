"""
AI模型性能基准测试

测试各个AI模型组件的推理性能、内存使用和吞吐量。
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
import numpy as np

from .benchmark_runner import BenchmarkRunner, BenchmarkResult

logger = logging.getLogger(__name__)


# =============================================================================
# 神经物理引擎基准测试
# =============================================================================

def benchmark_neural_physics(
    runner: Optional[BenchmarkRunner] = None,
    batch_sizes: List[int] = [1, 8, 32, 64],
    sequence_length: int = 16,
) -> List[BenchmarkResult]:
    """
    神经物理引擎基准测试

    测试LSTM/GRU代理模型的推理性能。
    """
    from ai_models.neural_physics_engine import (
        NeuralPhysicsConfig,
        LSTMSurrogateModel,
        GRUSurrogateModel,
    )

    runner = runner or BenchmarkRunner()
    results = []

    # 测试配置
    config = NeuralPhysicsConfig(
        input_dim=5,
        hidden_dim=128,
        num_layers=2,
        sequence_length=sequence_length,
    )

    # LSTM模型测试
    lstm_model = LSTMSurrogateModel(config).to(runner.device)
    lstm_model.eval()

    for batch_size in batch_sizes:
        def lstm_input_gen():
            return torch.randn(batch_size, sequence_length, config.input_dim).to(runner.device)

        result = runner.benchmark(
            name=f"NeuralPhysics_LSTM_b{batch_size}",
            func=lstm_model,
            input_generator=lstm_input_gen,
            batch_size=batch_size,
            metadata={
                'model_type': 'LSTM',
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'sequence_length': sequence_length,
                'params': sum(p.numel() for p in lstm_model.parameters()),
            }
        )
        results.append(result)

    # GRU模型测试
    gru_model = GRUSurrogateModel(config).to(runner.device)
    gru_model.eval()

    for batch_size in batch_sizes:
        def gru_input_gen():
            return torch.randn(batch_size, sequence_length, config.input_dim).to(runner.device)

        result = runner.benchmark(
            name=f"NeuralPhysics_GRU_b{batch_size}",
            func=gru_model,
            input_generator=gru_input_gen,
            batch_size=batch_size,
            metadata={
                'model_type': 'GRU',
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'sequence_length': sequence_length,
                'params': sum(p.numel() for p in gru_model.parameters()),
            }
        )
        results.append(result)

    return results


# =============================================================================
# 场景编码器基准测试
# =============================================================================

def benchmark_scenario_encoder(
    runner: Optional[BenchmarkRunner] = None,
    batch_sizes: List[int] = [1, 8, 32],
    sequence_length: int = 96,
) -> List[BenchmarkResult]:
    """
    场景编码器基准测试

    测试CNN和Transformer编码器的性能。
    """
    from ai_models.deep_scenario_encoder import (
        DeepEncoderConfig,
        CNN1DEncoder,
        TransformerEncoder,
    )

    runner = runner or BenchmarkRunner()
    results = []

    # CNN编码器测试
    cnn_config = DeepEncoderConfig(
        sequence_length=sequence_length,
        input_channels=5,
        encoder_type='cnn',
        hidden_dim=128,
        embedding_dim=64,
    )

    cnn_encoder = CNN1DEncoder(cnn_config).to(runner.device)
    cnn_encoder.eval()

    for batch_size in batch_sizes:
        def cnn_input_gen():
            return torch.randn(batch_size, sequence_length, cnn_config.input_channels).to(runner.device)

        result = runner.benchmark(
            name=f"ScenarioEncoder_CNN_b{batch_size}",
            func=cnn_encoder,
            input_generator=cnn_input_gen,
            batch_size=batch_size,
            metadata={
                'encoder_type': 'CNN',
                'hidden_dim': cnn_config.hidden_dim,
                'embedding_dim': cnn_config.embedding_dim,
                'sequence_length': sequence_length,
                'params': sum(p.numel() for p in cnn_encoder.parameters()),
            }
        )
        results.append(result)

    # Transformer编码器测试
    tf_config = DeepEncoderConfig(
        sequence_length=sequence_length,
        input_channels=5,
        encoder_type='transformer',
        hidden_dim=128,
        embedding_dim=64,
        num_layers=3,
    )

    tf_encoder = TransformerEncoder(tf_config).to(runner.device)
    tf_encoder.eval()

    for batch_size in batch_sizes:
        def tf_input_gen():
            return torch.randn(batch_size, sequence_length, tf_config.input_channels).to(runner.device)

        result = runner.benchmark(
            name=f"ScenarioEncoder_Transformer_b{batch_size}",
            func=tf_encoder,
            input_generator=tf_input_gen,
            batch_size=batch_size,
            metadata={
                'encoder_type': 'Transformer',
                'hidden_dim': tf_config.hidden_dim,
                'embedding_dim': tf_config.embedding_dim,
                'num_layers': tf_config.num_layers,
                'sequence_length': sequence_length,
                'params': sum(p.numel() for p in tf_encoder.parameters()),
            }
        )
        results.append(result)

    return results


# =============================================================================
# 场景VAE基准测试
# =============================================================================

def benchmark_scenario_vae(
    runner: Optional[BenchmarkRunner] = None,
    batch_sizes: List[int] = [1, 8, 32],
    sequence_length: int = 96,
) -> List[BenchmarkResult]:
    """
    场景VAE基准测试

    测试VAE编码、解码和采样性能。
    """
    from ai_models.scenario_vae import ScenarioVAEConfig, ScenarioVAE

    runner = runner or BenchmarkRunner()
    results = []

    config = ScenarioVAEConfig(
        sequence_length=sequence_length,
        num_channels=4,
        latent_dim=32,
        encoder_hidden_dim=128,
        decoder_hidden_dim=128,
    )

    vae = ScenarioVAE(config).to(runner.device)
    vae.eval()

    # 前向传播测试
    for batch_size in batch_sizes:
        def forward_input_gen():
            return torch.randn(batch_size, sequence_length, config.num_channels).to(runner.device)

        result = runner.benchmark(
            name=f"ScenarioVAE_Forward_b{batch_size}",
            func=vae,
            input_generator=forward_input_gen,
            batch_size=batch_size,
            metadata={
                'operation': 'forward',
                'latent_dim': config.latent_dim,
                'sequence_length': sequence_length,
                'params': sum(p.numel() for p in vae.parameters()),
            }
        )
        results.append(result)

    # 编码测试
    for batch_size in batch_sizes:
        def encode_input_gen():
            return torch.randn(batch_size, sequence_length, config.num_channels).to(runner.device)

        result = runner.benchmark(
            name=f"ScenarioVAE_Encode_b{batch_size}",
            func=vae.encode,
            input_generator=encode_input_gen,
            batch_size=batch_size,
            metadata={'operation': 'encode'},
        )
        results.append(result)

    # 采样测试
    for batch_size in batch_sizes:
        def sample_input_gen():
            return batch_size

        result = runner.benchmark(
            name=f"ScenarioVAE_Sample_b{batch_size}",
            func=vae.sample,
            input_generator=sample_input_gen,
            batch_size=batch_size,
            metadata={'operation': 'sample'},
        )
        results.append(result)

    return results


# =============================================================================
# E2E控制器基准测试
# =============================================================================

def benchmark_e2e_controller(
    runner: Optional[BenchmarkRunner] = None,
    batch_sizes: List[int] = [1, 4, 8],
    num_pools: int = 5,
) -> List[BenchmarkResult]:
    """
    E2E自主控制器基准测试

    测试时空注意力和控制器前向传播性能。
    """
    from ai_models.l4_autonomous.e2e_controller import (
        AutonomousConfig,
        SpatioTemporalAttention,
        E2EAutonomousController,
    )

    runner = runner or BenchmarkRunner()
    results = []

    config = AutonomousConfig(
        num_pools=num_pools,
        num_gates=num_pools + 1,
        state_dim=64,
        action_dim=32,
        hidden_dim=128,
        history_length=48,
        prediction_horizon=24,
    )

    # 时空注意力测试 - 输入需要是4D: [batch, time, pools, state_dim]
    attention = SpatioTemporalAttention(config).to(runner.device)
    attention.eval()

    for batch_size in batch_sizes:
        def attention_input_gen():
            # SpatioTemporalAttention需要4D输入
            return torch.randn(
                batch_size, config.history_length, config.num_pools, config.state_dim
            ).to(runner.device)

        result = runner.benchmark(
            name=f"E2E_SpatioTemporalAttention_b{batch_size}",
            func=attention,
            input_generator=attention_input_gen,
            batch_size=batch_size,
            metadata={
                'component': 'attention',
                'num_pools': num_pools,
                'state_dim': config.state_dim,
                'num_heads': config.num_heads,
                'history_length': config.history_length,
                'params': sum(p.numel() for p in attention.parameters()),
            }
        )
        results.append(result)

    # 完整控制器测试 - 需要匹配obs_encoder的input_dim
    controller = E2EAutonomousController(config).to(runner.device)
    controller.eval()

    # obs_encoder固定接受6维输入: [水位, 入流, 出流, 闸门开度, 分水流量, 传感器状态]
    obs_input_dim = 6

    for batch_size in batch_sizes:
        def controller_input_gen():
            # observations: [B, T, P, obs_dim]
            obs = torch.randn(batch_size, config.history_length, config.num_pools, obs_input_dim).to(runner.device)
            target_levels = torch.randn(batch_size, config.num_pools).to(runner.device)
            return {'observations': obs, 'target_levels': target_levels}

        def controller_forward(inputs):
            return controller(inputs['observations'], inputs['target_levels'])

        result = runner.benchmark(
            name=f"E2E_Controller_b{batch_size}",
            func=controller_forward,
            input_generator=controller_input_gen,
            batch_size=batch_size,
            metadata={
                'component': 'full_controller',
                'num_pools': num_pools,
                'history_length': config.history_length,
                'params': sum(p.numel() for p in controller.parameters()),
            }
        )
        results.append(result)

    return results


# =============================================================================
# 世界模型基准测试
# =============================================================================

def benchmark_world_model(
    runner: Optional[BenchmarkRunner] = None,
    batch_sizes: List[int] = [1, 4, 8],
    num_pools: int = 5,
) -> List[BenchmarkResult]:
    """
    全线世界模型基准测试

    注: 世界模型需要复杂的图结构输入，此处测试简化的MLP组件。
    """
    import torch.nn as nn

    runner = runner or BenchmarkRunner()
    results = []

    # 测试简化的预测器MLP (模拟世界模型核心计算)
    class SimplifiedPredictor(nn.Module):
        def __init__(self, num_pools, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(num_pools * 8, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_pools * 4),  # 预测4个时间步
            )

        def forward(self, x):
            B = x.shape[0]
            x = x.view(B, -1)
            return self.net(x)

    predictor = SimplifiedPredictor(num_pools).to(runner.device)
    predictor.eval()

    for batch_size in batch_sizes:
        def predictor_input_gen():
            return torch.randn(batch_size, num_pools, 8).to(runner.device)

        result = runner.benchmark(
            name=f"WorldModel_Predictor_b{batch_size}",
            func=predictor,
            input_generator=predictor_input_gen,
            batch_size=batch_size,
            metadata={
                'num_pools': num_pools,
                'component': 'simplified_predictor',
                'params': sum(p.numel() for p in predictor.parameters()),
            }
        )
        results.append(result)

    return results


# =============================================================================
# 环境基准测试
# =============================================================================

def benchmark_environment(
    runner: Optional[BenchmarkRunner] = None,
    num_steps: int = 1000,
) -> List[BenchmarkResult]:
    """
    强化学习环境基准测试

    测试环境step和reset性能。
    """
    from ai_models.water_canal_env import WaterCanalEnvConfig, OneGateTwoPoolsEnv

    runner = runner or BenchmarkRunner()
    results = []

    config = WaterCanalEnvConfig()
    env = OneGateTwoPoolsEnv(config)

    # Reset性能测试
    def reset_func(_):
        return env.reset()

    result = runner.benchmark(
        name="Environment_Reset",
        func=reset_func,
        input_generator=lambda: None,
        batch_size=1,
        metadata={'operation': 'reset'},
    )
    results.append(result)

    # Step性能测试
    env.reset()

    def step_func(action):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
        return obs

    result = runner.benchmark(
        name="Environment_Step",
        func=step_func,
        input_generator=lambda: env.action_space.sample(),
        batch_size=1,
        metadata={'operation': 'step'},
    )
    results.append(result)

    return results


# =============================================================================
# 完整系统基准测试
# =============================================================================

def run_full_benchmark(
    output_path: Optional[str] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    运行完整系统基准测试

    Args:
        output_path: 输出报告路径
        quick: 是否使用快速模式(减少迭代次数)

    Returns:
        Dict: 完整性能报告
    """
    logger.info("开始E2EControl完整性能基准测试...")

    # 配置
    if quick:
        runner = BenchmarkRunner(warmup_iterations=3, benchmark_iterations=20)
        batch_sizes = [1, 8]
    else:
        runner = BenchmarkRunner(warmup_iterations=10, benchmark_iterations=100)
        batch_sizes = [1, 8, 32]

    # 运行各模块测试
    logger.info("测试神经物理引擎...")
    benchmark_neural_physics(runner, batch_sizes=batch_sizes)

    logger.info("测试场景编码器...")
    benchmark_scenario_encoder(runner, batch_sizes=batch_sizes)

    logger.info("测试场景VAE...")
    benchmark_scenario_vae(runner, batch_sizes=batch_sizes)

    logger.info("测试E2E控制器...")
    benchmark_e2e_controller(runner, batch_sizes=[1, 4])

    logger.info("测试世界模型...")
    benchmark_world_model(runner, batch_sizes=[1, 4])

    logger.info("测试RL环境...")
    benchmark_environment(runner)

    # 生成报告
    runner.print_summary()
    report = runner.generate_report(output_path)

    logger.info("基准测试完成!")
    return report


# =============================================================================
# CLI入口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2EControl性能基准测试")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    parser.add_argument("--output", "-o", type=str, help="输出报告路径")
    parser.add_argument("--module", "-m", type=str, help="指定测试模块")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.module:
        runner = BenchmarkRunner()
        if args.module == "neural_physics":
            benchmark_neural_physics(runner)
        elif args.module == "encoder":
            benchmark_scenario_encoder(runner)
        elif args.module == "vae":
            benchmark_scenario_vae(runner)
        elif args.module == "controller":
            benchmark_e2e_controller(runner)
        elif args.module == "world_model":
            benchmark_world_model(runner)
        elif args.module == "env":
            benchmark_environment(runner)
        runner.print_summary()
    else:
        run_full_benchmark(output_path=args.output, quick=args.quick)
