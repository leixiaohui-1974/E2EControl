"""
场景向量数据库构建脚本
Scenario Vector Database Builder with DeepScenarioEncoder

用法:
    python build_scenario_db.py --output_dir models/scenario_db

功能:
1. 从L4场景定义中加载历史场景
2. 使用DeepScenarioEncoder编码场景
3. 构建向量数据库用于快速检索
4. 支持相似场景查询和异常检测
"""

import argparse
import os
import sys
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.deep_scenario_encoder import (
    DeepScenarioEncoder,
    DeepEncoderConfig,
    ScenarioVectorDB as VectorDatabase,
)
from ai_models.l4_autonomous.scenarios import (
    COMPLETE_SCENARIO_MATRIX,
    ScenarioCategory,
    ScenarioSeverity,
    ScenarioLibrary,
)
from ai_models.l4_autonomous.scenarios.scenario_generator import (
    ScenarioGenerator,
    SimulationState,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 场景数据生成
# ==============================================================================

def generate_scenario_data(scenario_id: str,
                           num_samples: int = 10,
                           sequence_length: int = 96,
                           num_pools: int = 10) -> List[Dict]:
    """
    为指定场景生成模拟数据

    Args:
        scenario_id: 场景ID
        num_samples: 生成的样本数
        sequence_length: 序列长度
        num_pools: 渠池数量

    Returns:
        场景数据列表
    """
    scenario = COMPLETE_SCENARIO_MATRIX.get(scenario_id)
    if scenario is None:
        logger.warning(f"场景不存在: {scenario_id}")
        return []

    generator = ScenarioGenerator(num_pools=num_pools, num_gates=num_pools+1)
    samples = []

    for i in range(num_samples):
        # 生成场景数据
        try:
            states = generator.generate(scenario, num_steps=sequence_length)
            # 转换SimulationState列表为数组
            levels = np.array([s.levels for s in states])
            flows = np.array([s.inflows for s in states])
            gates = np.array([s.gate_openings for s in states])
            demands = np.array([s.diversions for s in states])
        except Exception as e:
            # 如果生成失败，使用随机数据
            logger.debug(f"场景生成失败，使用随机数据: {e}")
            levels = np.random.uniform(3.5, 4.5, (sequence_length, num_pools))
            flows = np.random.uniform(200, 300, (sequence_length, num_pools))
            gates = np.random.uniform(0.5, 0.9, (sequence_length, num_pools + 1))
            demands = np.random.uniform(0, 10, (sequence_length, num_pools))

        # 转换为标准格式
        sample = {
            'scenario_id': scenario_id,
            'scenario_name': scenario.name,
            'category': scenario.category.value,
            'severity': scenario.severity.value,
            'data': {
                'levels': levels,
                'flows': flows,
                'gates': gates,
                'demands': demands,
            },
            'metadata': {
                'sample_idx': i,
                'generated_at': datetime.now().isoformat(),
                'required_level': scenario.required_level.value,
                'max_response_time_s': scenario.max_response_time_s,
            }
        }
        samples.append(sample)

    return samples


def generate_all_scenario_data(num_samples_per_scenario: int = 10,
                               sequence_length: int = 96) -> Dict[str, List[Dict]]:
    """
    为所有场景生成数据

    Args:
        num_samples_per_scenario: 每个场景的样本数
        sequence_length: 序列长度

    Returns:
        场景ID到数据列表的映射
    """
    all_data = {}

    for scenario_id in COMPLETE_SCENARIO_MATRIX:
        logger.info(f"生成场景数据: {scenario_id}")
        data = generate_scenario_data(
            scenario_id,
            num_samples=num_samples_per_scenario,
            sequence_length=sequence_length
        )
        if data:
            all_data[scenario_id] = data

    return all_data


# ==============================================================================
# 向量数据库构建
# ==============================================================================

def build_vector_database(encoder: DeepScenarioEncoder,
                          scenario_data: Dict[str, List[Dict]],
                          output_dir: str) -> VectorDatabase:
    """
    使用编码器构建向量数据库

    Args:
        encoder: 场景编码器
        scenario_data: 场景数据
        output_dir: 输出目录

    Returns:
        构建好的向量数据库
    """
    logger.info("构建向量数据库...")

    # 创建向量数据库
    vector_db = VectorDatabase(
        embedding_dim=encoder.config.embedding_dim
    )

    total_samples = 0

    for scenario_id, samples in scenario_data.items():
        for sample in samples:
            # 准备输入数据
            data = sample['data']

            # 堆叠所有特征 [T, C]
            input_tensor = np.stack([
                data['levels'].mean(axis=1) if len(data['levels'].shape) > 1 else data['levels'],
                data['flows'].mean(axis=1) if len(data['flows'].shape) > 1 else data['flows'],
                data['gates'].mean(axis=1) if len(data['gates'].shape) > 1 else data['gates'],
                data['demands'].mean(axis=1) if len(data['demands'].shape) > 1 else data['demands'],
                np.zeros(len(data['levels']))  # 天气占位符
            ], axis=1)  # [T, 5]

            # 编码
            embedding = encoder.encode(input_tensor)

            # 添加到数据库
            metadata = {
                'scenario_id': scenario_id,
                'scenario_name': sample['scenario_name'],
                'category': sample['category'],
                'severity': sample['severity'],
                **sample['metadata']
            }
            vector_db.add(embedding, f"{scenario_id}_{sample['metadata']['sample_idx']}", metadata)

            total_samples += 1

    logger.info(f"向量数据库构建完成，共 {total_samples} 个向量")

    # 保存
    db_path = os.path.join(output_dir, 'scenario_vectors.db')
    vector_db.save(db_path)
    logger.info(f"数据库已保存: {db_path}")

    return vector_db


# ==============================================================================
# 主函数
# ==============================================================================

def build_database(args):
    """构建场景向量数据库"""
    logger.info("=" * 70)
    logger.info("场景向量数据库构建")
    logger.info("=" * 70)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建编码器配置
    encoder_config = DeepEncoderConfig(
        sequence_length=args.sequence_length,
        input_channels=5,
        encoder_type='cnn',
        hidden_dim=128,
        embedding_dim=64,
    )

    # 创建编码器
    logger.info("\n创建场景编码器...")
    encoder = DeepScenarioEncoder(encoder_config)

    # 如果有预训练模型，加载
    if args.encoder_model and os.path.exists(args.encoder_model):
        encoder.load_model(args.encoder_model)
        logger.info(f"加载预训练编码器: {args.encoder_model}")

    # 生成场景数据
    logger.info(f"\n生成场景数据 ({args.num_samples}样本/场景)...")
    scenario_data = generate_all_scenario_data(
        num_samples_per_scenario=args.num_samples,
        sequence_length=args.sequence_length
    )

    total_scenarios = len(scenario_data)
    total_samples = sum(len(v) for v in scenario_data.values())
    logger.info(f"  场景总数: {total_scenarios}")
    logger.info(f"  样本总数: {total_samples}")

    # 构建向量数据库
    vector_db = build_vector_database(encoder, scenario_data, args.output_dir)

    # 测试检索
    logger.info("\n测试向量检索...")
    test_scenario_id = list(scenario_data.keys())[0]
    test_sample = scenario_data[test_scenario_id][0]

    # 准备查询数据
    data = test_sample['data']
    query_tensor = np.stack([
        data['levels'].mean(axis=1) if len(data['levels'].shape) > 1 else data['levels'],
        data['flows'].mean(axis=1) if len(data['flows'].shape) > 1 else data['flows'],
        data['gates'].mean(axis=1) if len(data['gates'].shape) > 1 else data['gates'],
        data['demands'].mean(axis=1) if len(data['demands'].shape) > 1 else data['demands'],
        np.zeros(len(data['levels']))
    ], axis=1)

    query_embedding = encoder.encode(query_tensor)
    results = vector_db.search(query_embedding, top_k=5)

    logger.info(f"查询场景: {test_scenario_id}")
    logger.info("Top-5 检索结果:")
    for i, (idx, score, metadata) in enumerate(results):
        logger.info(f"  {i+1}. {metadata['scenario_id']} (相似度: {score:.3f})")

    # 保存编码器
    encoder_path = os.path.join(args.output_dir, 'scenario_encoder.pt')
    encoder.save_model(encoder_path)
    logger.info(f"\n编码器已保存: {encoder_path}")

    # 保存统计信息
    stats = {
        'total_scenarios': total_scenarios,
        'total_samples': total_samples,
        'embedding_dim': encoder_config.embedding_dim,
        'sequence_length': args.sequence_length,
        'created_at': datetime.now().isoformat(),
        'scenario_ids': list(scenario_data.keys()),
    }

    import json
    stats_path = os.path.join(args.output_dir, 'database_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"统计信息已保存: {stats_path}")

    # 汇总
    logger.info("\n" + "=" * 70)
    logger.info("数据库构建完成!")
    logger.info("=" * 70)
    logger.info(f"  场景数: {total_scenarios}")
    logger.info(f"  向量数: {total_samples}")
    logger.info(f"  向量维度: {encoder_config.embedding_dim}")
    logger.info(f"  数据库路径: {args.output_dir}")

    return args.output_dir


def query_database(args):
    """查询场景数据库"""
    logger.info("=" * 70)
    logger.info("场景向量查询")
    logger.info("=" * 70)

    # 加载编码器
    encoder_config = DeepEncoderConfig(
        sequence_length=args.sequence_length,
        embedding_dim=64,
    )
    encoder = DeepScenarioEncoder(encoder_config)

    encoder_path = os.path.join(args.db_path, 'scenario_encoder.pt')
    if os.path.exists(encoder_path):
        encoder.load_model(encoder_path)

    # 加载数据库
    db_path = os.path.join(args.db_path, 'scenario_vectors.db')
    vector_db = VectorDatabase(embedding_dim=64)
    vector_db.load(db_path)

    logger.info(f"数据库加载完成，共 {len(vector_db)} 个向量")

    # 生成随机查询
    query_data = np.random.randn(args.sequence_length, 5).astype(np.float32)
    query_embedding = encoder.encode(query_data)

    # 检索
    results = vector_db.search(query_embedding, top_k=args.top_k)

    logger.info(f"\nTop-{args.top_k} 检索结果:")
    for i, (idx, score, metadata) in enumerate(results):
        logger.info(f"  {i+1}. {metadata['scenario_id']}")
        logger.info(f"      名称: {metadata['scenario_name']}")
        logger.info(f"      类别: {metadata['category']}")
        logger.info(f"      相似度: {score:.3f}")


# ==============================================================================
# 命令行参数
# ==============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='场景向量数据库构建')

    # 模式
    parser.add_argument('--mode', type=str, default='build',
                        choices=['build', 'query'],
                        help='运行模式')

    # 构建参数
    parser.add_argument('--output_dir', type=str, default='models/scenario_db',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='每个场景的样本数')
    parser.add_argument('--sequence_length', type=int, default=96,
                        help='序列长度')
    parser.add_argument('--encoder_model', type=str, default=None,
                        help='预训练编码器路径')

    # 查询参数
    parser.add_argument('--db_path', type=str, default='models/scenario_db',
                        help='数据库路径')
    parser.add_argument('--top_k', type=int, default=5,
                        help='返回Top-K结果')

    return parser.parse_args()


# ==============================================================================
# 主函数
# ==============================================================================

if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'build':
        build_database(args)
    elif args.mode == 'query':
        query_database(args)
