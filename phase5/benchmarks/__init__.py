"""
E2EControl 性能基准测试模块

提供系统各组件的性能测试和瓶颈分析。
"""

from .benchmark_runner import BenchmarkRunner, BenchmarkResult
from .model_benchmarks import (
    benchmark_neural_physics,
    benchmark_scenario_encoder,
    benchmark_scenario_vae,
    benchmark_e2e_controller,
    benchmark_world_model,
    benchmark_environment,
)

__all__ = [
    'BenchmarkRunner',
    'BenchmarkResult',
    'benchmark_neural_physics',
    'benchmark_scenario_encoder',
    'benchmark_scenario_vae',
    'benchmark_e2e_controller',
    'benchmark_world_model',
    'benchmark_environment',
]
