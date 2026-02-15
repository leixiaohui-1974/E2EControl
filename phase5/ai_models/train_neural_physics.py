"""
神经物理模型训练脚本
Training Script for Neural Physics Engine

用法:
    python train_neural_physics.py --num_samples 100000 --epochs 100

功能:
1. 使用传统物理模型生成训练数据
2. 训练LSTM神经代理模型
3. 验证模型性能
4. 保存训练好的模型
"""

import argparse
import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.neural_physics_engine import (
    NeuralPhysicsEngine,
    NeuralPhysicsConfig,
    LSTMSurrogateModel,
    NeuralPhysicsTrainer,
    PINNLoss,
)
from ai_models.data_generator import (
    PhysicsDataGenerator,
    DataGeneratorConfig,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练神经物理模型')

    # 数据参数
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='生成的样本数量')
    parser.add_argument('--seq_length', type=int, default=16,
                        help='输入序列长度')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM隐层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--use_physics_loss', action='store_true',
                        help='使用物理约束损失')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='neural_physics',
                        help='模型名称')

    # 设备
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备 (cpu/cuda/mps)')

    return parser.parse_args()


def train(args):
    """主训练函数"""
    logger.info("=" * 70)
    logger.info("神经物理模型训练")
    logger.info("=" * 70)

    # 设置设备
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"使用设备: {device}")

    # 创建配置
    data_config = DataGeneratorConfig(
        num_samples=args.num_samples,
        sequence_length=args.seq_length,
    )

    model_config = NeuralPhysicsConfig(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=1,
        dropout=args.dropout,
        sequence_length=args.seq_length,
        use_physics_loss=args.use_physics_loss,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    # 生成数据
    logger.info("\n生成训练数据...")
    generator = PhysicsDataGenerator(data_config)
    train_dataset, val_dataset, test_dataset = generator.create_dataset()

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"  训练批次数: {len(train_loader)}")
    logger.info(f"  验证批次数: {len(val_loader)}")

    # 创建模型
    logger.info("\n创建模型...")
    model = LSTMSurrogateModel(model_config).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")

    # 创建训练器
    trainer = NeuralPhysicsTrainer(model, model_config, device)

    # 训练
    logger.info("\n开始训练...")
    logger.info("-" * 70)

    history = trainer.train(train_loader, val_loader, args.epochs)

    # 测试
    logger.info("\n评估测试集...")
    test_loss = trainer.validate(test_loader)
    logger.info(f"  测试损失: {test_loss:.6f}")

    # 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"{args.model_name}_{timestamp}.pt")

    # 获取归一化参数
    metadata = train_dataset.metadata

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'input_mean': metadata['input_mean'],
        'input_std': metadata['input_std'],
        'output_mean': metadata['output_mean'],
        'output_std': metadata['output_std'],
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1],
        'test_loss': test_loss,
    }

    torch.save(checkpoint, model_path)
    logger.info(f"\n模型已保存: {model_path}")

    # 保存训练历史
    history_path = os.path.join(args.output_dir, f"{args.model_name}_{timestamp}_history.npy")
    np.save(history_path, history)

    # 验证保存的模型
    logger.info("\n验证保存的模型...")
    engine = NeuralPhysicsEngine(model_config, device=str(device))
    engine.load_model(model_path)
    engine.reset(initial_level=4.0, initial_inflow=100.0)

    # 快速测试
    test_levels = []
    for i in range(50):
        q_in = 100 + 20 * np.sin(2 * np.pi * i / 25)
        level = engine.step(q_in)
        test_levels.append(level)

    logger.info(f"  验证测试通过!")
    logger.info(f"  水位范围: [{min(test_levels):.2f}, {max(test_levels):.2f}]m")

    # 汇总
    logger.info("\n" + "=" * 70)
    logger.info("训练完成!")
    logger.info("=" * 70)
    logger.info(f"  最终训练损失: {history['train_loss'][-1]:.6f}")
    logger.info(f"  最终验证损失: {history['val_loss'][-1]:.6f}")
    logger.info(f"  测试损失: {test_loss:.6f}")
    logger.info(f"  模型路径: {model_path}")

    return model_path


def evaluate_model(model_path: str, device: str = None):
    """评估已保存的模型"""
    logger.info("=" * 70)
    logger.info("模型评估")
    logger.info("=" * 70)

    # 加载模型
    engine = NeuralPhysicsEngine(device=device)
    engine.load_model(model_path)

    # 测试场景
    test_cases = [
        ("稳态", lambda t: 200.0),
        ("阶跃", lambda t: 200.0 if t < 25 else 280.0),
        ("正弦", lambda t: 200.0 + 50.0 * np.sin(2 * np.pi * t / 50)),
        ("脉冲", lambda t: 200.0 + 100.0 * np.exp(-((t - 25) ** 2) / 10)),
    ]

    for name, inflow_func in test_cases:
        engine.reset(initial_level=4.0, initial_inflow=inflow_func(0))

        levels = []
        inflows = []
        for t in range(100):
            q_in = inflow_func(t)
            level = engine.step(q_in)
            levels.append(level)
            inflows.append(q_in)

        logger.info(f"\n{name}场景:")
        logger.info(f"  入流范围: [{min(inflows):.1f}, {max(inflows):.1f}] m³/s")
        logger.info(f"  水位范围: [{min(levels):.3f}, {max(levels):.3f}] m")
        logger.info(f"  水位均值: {np.mean(levels):.3f} m")
        logger.info(f"  水位标准差: {np.std(levels):.3f} m")


if __name__ == "__main__":
    args = parse_args()
    model_path = train(args)

    # 评估
    logger.info("\n")
    evaluate_model(model_path, args.device)
