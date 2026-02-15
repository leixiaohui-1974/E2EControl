"""
数据生成器 (Data Generator)
利用现有物理模型生成神经网络训练数据

Teacher-Student 模式:
- Teacher: 传统物理模型 (physics_model.py)
- Student: 神经代理模型 (neural_physics_engine.py)

生成的数据类型:
1. 平稳流数据 - 恒定入流工况
2. 脉冲流数据 - 阶跃变化工况
3. 随机流数据 - 随机波动工况
4. 周期流数据 - 日周期变化工况
5. 极端工况数据 - 边界条件测试
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_transfer_system.physics_model import (
    SNWDMiddleRouteModel,
    CanalPool,
    IDZModel,
    IDZParameters
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class DataGeneratorConfig:
    """数据生成配置"""
    # 数据规模
    num_samples: int = 100000        # 总样本数
    sequence_length: int = 16        # 输入序列长度

    # 工况比例
    steady_ratio: float = 0.2        # 平稳流
    step_ratio: float = 0.2          # 脉冲流
    random_ratio: float = 0.3        # 随机流
    periodic_ratio: float = 0.2      # 周期流
    extreme_ratio: float = 0.1       # 极端工况

    # 物理参数范围
    inflow_range: Tuple[float, float] = (50.0, 400.0)   # 入流范围 [m³/s]
    level_range: Tuple[float, float] = (1.0, 6.0)       # 水位范围 [m]
    gate_range: Tuple[float, float] = (0.1, 1.0)        # 闸门开度范围

    # 时间步长
    dt: float = 900.0                # 时间步长 [s] (15分钟)

    # 输出路径
    output_dir: str = "training_data"


# ==============================================================================
# 数据集类
# ==============================================================================

class WaterPhysicsDataset(Dataset):
    """水力学物理数据集"""

    def __init__(self,
                 sequences: np.ndarray,
                 targets: np.ndarray,
                 metadata: Dict = None):
        """
        Args:
            sequences: 输入序列 [N, seq_len, features]
            targets: 目标值 [N, output_dim]
            metadata: 元数据 (归一化参数等)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.metadata = metadata or {}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input': self.sequences[idx],
            'target': self.targets[idx],
            'current_level': self.sequences[idx, -1, 1:2],  # 当前水位
            'prev_level': self.sequences[idx, -2, 1:2] if self.sequences.shape[1] > 1 else self.sequences[idx, -1, 1:2],
            'q_in': self.sequences[idx, -1, 0:1],
            'q_out': self.targets[idx],  # 简化: 用目标水位近似
        }


# ==============================================================================
# 数据生成器
# ==============================================================================

class PhysicsDataGenerator:
    """
    物理数据生成器

    利用传统物理模型生成训练数据
    """

    def __init__(self, config: DataGeneratorConfig = None):
        self.config = config or DataGeneratorConfig()

        # 创建物理模型
        self.physics_model = self._create_physics_model()

        # 数据存储
        self.data: Dict[str, List] = {
            'sequences': [],
            'targets': [],
            'scenario_types': [],
        }

    def _create_physics_model(self) -> IDZModel:
        """创建单池物理模型"""
        # 使用标准IDZ参数
        params = IDZParameters(
            tau=14400.0,      # 4小时滞后
            A_s=100000.0,     # 10万平方米
            c_in=1.0,
            c_out=1.0
        )
        model = IDZModel(params, dt=self.config.dt)
        return model

    def generate_all(self) -> Dict[str, np.ndarray]:
        """生成所有类型的数据"""
        config = self.config

        # 计算各类型样本数
        n_steady = int(config.num_samples * config.steady_ratio)
        n_step = int(config.num_samples * config.step_ratio)
        n_random = int(config.num_samples * config.random_ratio)
        n_periodic = int(config.num_samples * config.periodic_ratio)
        n_extreme = int(config.num_samples * config.extreme_ratio)

        logger.info(f"开始生成训练数据...")
        logger.info(f"  平稳流: {n_steady}")
        logger.info(f"  脉冲流: {n_step}")
        logger.info(f"  随机流: {n_random}")
        logger.info(f"  周期流: {n_periodic}")
        logger.info(f"  极端工况: {n_extreme}")

        # 生成各类数据
        self._generate_steady_flow(n_steady)
        self._generate_step_flow(n_step)
        self._generate_random_flow(n_random)
        self._generate_periodic_flow(n_periodic)
        self._generate_extreme_scenarios(n_extreme)

        # 转换为numpy数组
        sequences = np.array(self.data['sequences'])
        targets = np.array(self.data['targets'])

        logger.info(f"数据生成完成!")
        logger.info(f"  序列形状: {sequences.shape}")
        logger.info(f"  目标形状: {targets.shape}")

        return {
            'sequences': sequences,
            'targets': targets,
            'scenario_types': np.array(self.data['scenario_types'])
        }

    def _generate_steady_flow(self, num_samples: int):
        """生成平稳流数据"""
        config = self.config
        seq_len = config.sequence_length

        for _ in range(num_samples):
            # 随机选择稳态参数
            q_in = np.random.uniform(*config.inflow_range)
            gate = np.random.uniform(*config.gate_range)
            initial_level = np.random.uniform(*config.level_range)

            # 重置模型
            self.physics_model.reset(initial_level, q_in)

            # 运行模拟
            sequence = []
            for t in range(seq_len + 1):
                # 小扰动
                q_in_t = q_in + np.random.normal(0, q_in * 0.01)
                q_out_t = q_in_t * gate

                level = self.physics_model.step(q_in_t, q_out_t)

                # 构建特征向量 [入流, 水位, 闸门开度, 分水扰动, 时间特征]
                feature = np.array([
                    q_in_t,
                    level,
                    gate,
                    0.0,  # 无分水
                    t / seq_len  # 归一化时间
                ])
                sequence.append(feature)

            # 存储
            self.data['sequences'].append(sequence[:-1])
            self.data['targets'].append([sequence[-1][1]])  # 预测下一时刻水位
            self.data['scenario_types'].append('steady')

    def _generate_step_flow(self, num_samples: int):
        """生成脉冲流/阶跃变化数据"""
        config = self.config
        seq_len = config.sequence_length

        for _ in range(num_samples):
            # 阶跃参数
            q_in_before = np.random.uniform(100, 250)
            q_in_after = np.random.uniform(150, 350)
            step_time = np.random.randint(seq_len // 4, 3 * seq_len // 4)
            gate = np.random.uniform(*config.gate_range)
            initial_level = np.random.uniform(*config.level_range)

            # 重置模型
            self.physics_model.reset(initial_level, q_in_before)

            # 运行模拟
            sequence = []
            for t in range(seq_len + 1):
                # 阶跃变化
                q_in_t = q_in_after if t >= step_time else q_in_before
                q_out_t = q_in_t * gate

                level = self.physics_model.step(q_in_t, q_out_t)

                feature = np.array([
                    q_in_t,
                    level,
                    gate,
                    0.0,
                    t / seq_len
                ])
                sequence.append(feature)

            self.data['sequences'].append(sequence[:-1])
            self.data['targets'].append([sequence[-1][1]])
            self.data['scenario_types'].append('step')

    def _generate_random_flow(self, num_samples: int):
        """生成随机流数据"""
        config = self.config
        seq_len = config.sequence_length

        for _ in range(num_samples):
            # 随机参数
            q_in_base = np.random.uniform(100, 300)
            noise_std = q_in_base * np.random.uniform(0.05, 0.2)
            gate = np.random.uniform(*config.gate_range)
            initial_level = np.random.uniform(*config.level_range)

            # 重置模型
            self.physics_model.reset(initial_level, q_in_base)

            # 运行模拟
            sequence = []
            for t in range(seq_len + 1):
                # 随机波动
                q_in_t = q_in_base + np.random.normal(0, noise_std)
                q_in_t = np.clip(q_in_t, *config.inflow_range)
                q_out_t = q_in_t * gate

                level = self.physics_model.step(q_in_t, q_out_t)

                feature = np.array([
                    q_in_t,
                    level,
                    gate,
                    np.random.uniform(0, 5),  # 随机分水
                    t / seq_len
                ])
                sequence.append(feature)

            self.data['sequences'].append(sequence[:-1])
            self.data['targets'].append([sequence[-1][1]])
            self.data['scenario_types'].append('random')

    def _generate_periodic_flow(self, num_samples: int):
        """生成周期流数据 (日变化模式)"""
        config = self.config
        seq_len = config.sequence_length

        for _ in range(num_samples):
            # 周期参数
            q_in_base = np.random.uniform(150, 280)
            amplitude = q_in_base * np.random.uniform(0.1, 0.3)
            period = np.random.uniform(seq_len * 2, seq_len * 4)
            phase = np.random.uniform(0, 2 * np.pi)
            gate = np.random.uniform(*config.gate_range)
            initial_level = np.random.uniform(*config.level_range)

            # 重置模型
            self.physics_model.reset(initial_level, q_in_base)

            # 运行模拟
            sequence = []
            for t in range(seq_len + 1):
                # 周期变化
                q_in_t = q_in_base + amplitude * np.sin(2 * np.pi * t / period + phase)
                q_out_t = q_in_t * gate

                level = self.physics_model.step(q_in_t, q_out_t)

                feature = np.array([
                    q_in_t,
                    level,
                    gate,
                    0.0,
                    t / seq_len
                ])
                sequence.append(feature)

            self.data['sequences'].append(sequence[:-1])
            self.data['targets'].append([sequence[-1][1]])
            self.data['scenario_types'].append('periodic')

    def _generate_extreme_scenarios(self, num_samples: int):
        """生成极端工况数据"""
        config = self.config
        seq_len = config.sequence_length

        extreme_types = ['flood', 'drought', 'gate_failure', 'sudden_demand']

        for _ in range(num_samples):
            extreme_type = np.random.choice(extreme_types)

            if extreme_type == 'flood':
                # 洪水工况: 高流量、快速上升
                q_in_start = 200
                q_in_peak = np.random.uniform(350, 400)
                rise_rate = np.random.uniform(5, 15)

            elif extreme_type == 'drought':
                # 干旱工况: 低流量、持续下降
                q_in_start = 150
                q_in_end = np.random.uniform(50, 80)
                rise_rate = (q_in_end - q_in_start) / seq_len
                q_in_peak = q_in_end

            elif extreme_type == 'gate_failure':
                # 闸门故障: 突然卡死
                q_in_start = 200
                q_in_peak = 200
                rise_rate = 0

            else:  # sudden_demand
                # 突发需水: 大量分水
                q_in_start = 250
                q_in_peak = 250
                rise_rate = 0

            gate = np.random.uniform(0.3, 0.9)
            initial_level = np.random.uniform(3.0, 5.0)

            # 重置模型
            self.physics_model.reset(initial_level, q_in_start)

            # 运行模拟
            sequence = []
            for t in range(seq_len + 1):
                if extreme_type == 'flood':
                    # 洪水曲线
                    q_in_t = q_in_start + (q_in_peak - q_in_start) * (1 - np.exp(-rise_rate * t / seq_len))
                elif extreme_type == 'drought':
                    q_in_t = q_in_start + rise_rate * t
                else:
                    q_in_t = q_in_peak

                q_in_t = np.clip(q_in_t, *config.inflow_range)

                # 闸门故障: 开度固定
                if extreme_type == 'gate_failure' and t > seq_len // 3:
                    gate_t = 0.5  # 卡死在50%
                else:
                    gate_t = gate

                # 突发需水: 大量分水
                diversion = 30.0 if extreme_type == 'sudden_demand' and t > seq_len // 2 else 0.0

                q_out_t = q_in_t * gate_t - diversion
                q_out_t = max(0, q_out_t)

                level = self.physics_model.step(q_in_t, q_out_t)

                feature = np.array([
                    q_in_t,
                    level,
                    gate_t,
                    diversion,
                    t / seq_len
                ])
                sequence.append(feature)

            self.data['sequences'].append(sequence[:-1])
            self.data['targets'].append([sequence[-1][1]])
            self.data['scenario_types'].append(extreme_type)

    def create_dataset(self,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1) -> Tuple[WaterPhysicsDataset, WaterPhysicsDataset, WaterPhysicsDataset]:
        """
        创建训练/验证/测试数据集

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        # 生成数据
        data = self.generate_all()
        sequences = data['sequences']
        targets = data['targets']

        # 计算归一化参数
        input_mean = np.mean(sequences, axis=(0, 1))
        input_std = np.std(sequences, axis=(0, 1)) + 1e-8
        output_mean = np.mean(targets)
        output_std = np.std(targets) + 1e-8

        # 归一化
        sequences_norm = (sequences - input_mean) / input_std
        targets_norm = (targets - output_mean) / output_std

        # 元数据
        metadata = {
            'input_mean': input_mean,
            'input_std': input_std,
            'output_mean': output_mean,
            'output_std': output_std,
        }

        # 划分数据集
        n = len(sequences_norm)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # 随机打乱
        indices = np.random.permutation(n)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_dataset = WaterPhysicsDataset(
            sequences_norm[train_idx],
            targets_norm[train_idx],
            metadata
        )

        val_dataset = WaterPhysicsDataset(
            sequences_norm[val_idx],
            targets_norm[val_idx],
            metadata
        )

        test_dataset = WaterPhysicsDataset(
            sequences_norm[test_idx],
            targets_norm[test_idx],
            metadata
        )

        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {len(train_dataset)}")
        logger.info(f"  验证集: {len(val_dataset)}")
        logger.info(f"  测试集: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def save_data(self, output_dir: str = None):
        """保存生成的数据"""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        data = self.generate_all()

        np.save(os.path.join(output_dir, 'sequences.npy'), data['sequences'])
        np.save(os.path.join(output_dir, 'targets.npy'), data['targets'])
        np.save(os.path.join(output_dir, 'scenario_types.npy'), data['scenario_types'])

        logger.info(f"数据已保存到 {output_dir}")


# ==============================================================================
# 全线模型数据生成器
# ==============================================================================

class FullLineDataGenerator:
    """
    全线模型数据生成器

    利用 SNWDMiddleRouteModel 生成更复杂的多池数据
    """

    def __init__(self, config: DataGeneratorConfig = None):
        self.config = config or DataGeneratorConfig()
        self.physics_model = SNWDMiddleRouteModel()

    def generate_multi_pool_data(self,
                                  num_samples: int = 10000,
                                  pool_ids: List[int] = None) -> Dict[str, np.ndarray]:
        """
        生成多池协同数据

        Args:
            num_samples: 样本数
            pool_ids: 要生成数据的渠池ID列表

        Returns:
            包含各池数据的字典
        """
        pool_ids = pool_ids or list(range(10))  # 默认前10个池
        config = self.config
        seq_len = config.sequence_length

        data = {pid: {'sequences': [], 'targets': []} for pid in pool_ids}

        for _ in range(num_samples):
            # 随机初始条件
            initial_level = np.random.uniform(3.5, 4.5)
            initial_flow = np.random.uniform(250, 350)

            self.physics_model.reset(initial_level, initial_flow)

            # 生成随机控制序列
            source_inflows = []
            gate_commands_list = []

            for t in range(seq_len + 1):
                # 源头流量随机变化
                if t == 0:
                    source_flow = initial_flow
                else:
                    source_flow = source_inflows[-1] + np.random.normal(0, 5)
                    source_flow = np.clip(source_flow, 200, 400)
                source_inflows.append(source_flow)

                # 随机闸门调整
                gate_commands = {}
                for pid in pool_ids:
                    if np.random.random() < 0.1:  # 10%概率调整
                        gate_commands[pid] = np.random.uniform(0.7, 1.0)
                gate_commands_list.append(gate_commands)

            # 运行模拟
            pool_histories = {pid: [] for pid in pool_ids}

            for t in range(seq_len + 1):
                state = self.physics_model.step(
                    source_inflow=source_inflows[t],
                    gate_commands=gate_commands_list[t]
                )

                for pool_state in state['pools']:
                    pid = pool_state['pool_id']
                    if pid in pool_ids:
                        feature = np.array([
                            pool_state['inflow'],
                            pool_state['level'],
                            pool_state['gate_opening'],
                            pool_state['diversion'],
                            t / seq_len
                        ])
                        pool_histories[pid].append(feature)

            # 存储
            for pid in pool_ids:
                data[pid]['sequences'].append(pool_histories[pid][:-1])
                data[pid]['targets'].append([pool_histories[pid][-1][1]])

        # 转换为numpy
        for pid in pool_ids:
            data[pid]['sequences'] = np.array(data[pid]['sequences'])
            data[pid]['targets'] = np.array(data[pid]['targets'])

        return data


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info(" " * 15 + "物理数据生成器测试")
    logger.info("=" * 70)

    # 配置
    config = DataGeneratorConfig(
        num_samples=1000,  # 测试用小规模数据
        sequence_length=16,
    )

    # 创建生成器
    generator = PhysicsDataGenerator(config)

    # 生成数据集
    train_dataset, val_dataset, test_dataset = generator.create_dataset()

    logger.info(f"\n数据集信息:")
    logger.info(f"  训练集大小: {len(train_dataset)}")
    logger.info(f"  验证集大小: {len(val_dataset)}")
    logger.info(f"  测试集大小: {len(test_dataset)}")

    # 检查数据形状
    sample = train_dataset[0]
    logger.info(f"\n样本形状:")
    logger.info(f"  输入: {sample['input'].shape}")
    logger.info(f"  目标: {sample['target'].shape}")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for batch in train_loader:
        logger.info(f"\n批次形状:")
        logger.info(f"  输入: {batch['input'].shape}")
        logger.info(f"  目标: {batch['target'].shape}")
        break

    # 测试全线数据生成
    logger.info(f"\n{'=' * 70}")
    logger.info("全线模型数据生成测试")
    logger.info('=' * 70)

    full_generator = FullLineDataGenerator(config)
    full_data = full_generator.generate_multi_pool_data(
        num_samples=100,
        pool_ids=[0, 1, 2, 3, 4]
    )

    for pid, pool_data in full_data.items():
        logger.info(f"  池{pid}: 序列{pool_data['sequences'].shape}, 目标{pool_data['targets'].shape}")

    logger.info("\n" + "=" * 70)
    logger.info("测试完成!")
    logger.info("=" * 70)
