"""
性能基准测试运行器

提供统一的基准测试框架，支持:
- 推理时间测量
- 内存使用监控
- 吞吐量计算
- 性能分析报告生成
"""

import gc
import time
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    # 时间指标 (毫秒)
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float

    # 内存指标 (MB)
    peak_memory_mb: float
    memory_allocated_mb: float

    # 吞吐量指标
    throughput_per_sec: float
    batch_size: int
    num_iterations: int

    # 设备信息
    device: str
    dtype: str

    # 额外元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'timing': {
                'mean_ms': round(self.mean_time_ms, 3),
                'std_ms': round(self.std_time_ms, 3),
                'min_ms': round(self.min_time_ms, 3),
                'max_ms': round(self.max_time_ms, 3),
                'median_ms': round(self.median_time_ms, 3),
                'p95_ms': round(self.p95_time_ms, 3),
                'p99_ms': round(self.p99_time_ms, 3),
            },
            'memory': {
                'peak_mb': round(self.peak_memory_mb, 2),
                'allocated_mb': round(self.memory_allocated_mb, 2),
            },
            'throughput': {
                'samples_per_sec': round(self.throughput_per_sec, 2),
                'batch_size': self.batch_size,
                'iterations': self.num_iterations,
            },
            'device': self.device,
            'dtype': self.dtype,
            'metadata': self.metadata,
        }

    def summary(self) -> str:
        """生成摘要字符串"""
        return (
            f"{self.name}:\n"
            f"  时间: {self.mean_time_ms:.2f}±{self.std_time_ms:.2f}ms "
            f"(p95={self.p95_time_ms:.2f}ms)\n"
            f"  内存: {self.peak_memory_mb:.1f}MB (峰值)\n"
            f"  吞吐: {self.throughput_per_sec:.1f} samples/s"
        )


class BenchmarkRunner:
    """
    基准测试运行器

    提供统一的性能测试接口。
    """

    def __init__(
        self,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        device: str = 'auto',
    ):
        """
        初始化基准测试运行器

        Args:
            warmup_iterations: 预热迭代次数
            benchmark_iterations: 基准测试迭代次数
            device: 运行设备
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

        # 自动选择设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.results: List[BenchmarkResult] = []

    @contextmanager
    def _timer(self):
        """计时上下文管理器"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end))[2]
        else:
            start = time.perf_counter()
            yield lambda: (time.perf_counter() - start) * 1000  # 转换为毫秒

    def _get_memory_stats(self) -> Tuple[float, float]:
        """获取内存统计"""
        if self.device == 'cuda':
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            return peak, allocated
        elif HAS_PSUTIL:
            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss / 1024 / 1024, mem_info.rss / 1024 / 1024
        return 0.0, 0.0

    def _reset_memory_stats(self):
        """重置内存统计"""
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()

    def benchmark(
        self,
        name: str,
        func: Callable,
        input_generator: Callable[[], Any],
        batch_size: int = 1,
        dtype: str = 'float32',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """
        运行基准测试

        Args:
            name: 测试名称
            func: 要测试的函数
            input_generator: 输入数据生成器
            batch_size: 批大小
            dtype: 数据类型
            metadata: 额外元数据

        Returns:
            BenchmarkResult: 测试结果
        """
        logger.info(f"开始基准测试: {name}")

        # 重置内存统计
        self._reset_memory_stats()

        # 预热阶段
        logger.debug(f"预热 {self.warmup_iterations} 次迭代...")
        for _ in range(self.warmup_iterations):
            inputs = input_generator()
            with torch.no_grad():
                _ = func(inputs)

        # 基准测试阶段
        times = []
        self._reset_memory_stats()

        logger.debug(f"运行 {self.benchmark_iterations} 次基准测试...")
        for _ in range(self.benchmark_iterations):
            inputs = input_generator()

            with self._timer() as get_time:
                with torch.no_grad():
                    _ = func(inputs)
            times.append(get_time())

        # 计算统计指标
        times_sorted = sorted(times)
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        p95_idx = int(len(times_sorted) * 0.95)
        p99_idx = int(len(times_sorted) * 0.99)
        p95_time = times_sorted[p95_idx] if p95_idx < len(times_sorted) else max_time
        p99_time = times_sorted[p99_idx] if p99_idx < len(times_sorted) else max_time

        # 内存统计
        peak_memory, allocated_memory = self._get_memory_stats()

        # 吞吐量计算
        throughput = batch_size * 1000 / mean_time  # samples per second

        result = BenchmarkResult(
            name=name,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            median_time_ms=median_time,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time,
            peak_memory_mb=peak_memory,
            memory_allocated_mb=allocated_memory,
            throughput_per_sec=throughput,
            batch_size=batch_size,
            num_iterations=self.benchmark_iterations,
            device=self.device,
            dtype=dtype,
            metadata=metadata or {},
        )

        self.results.append(result)
        logger.info(f"完成: {result.summary()}")

        return result

    def benchmark_scaling(
        self,
        name: str,
        func: Callable,
        input_generator_factory: Callable[[int], Callable[[], Any]],
        batch_sizes: List[int],
        dtype: str = 'float32',
    ) -> List[BenchmarkResult]:
        """
        测试不同批大小下的性能扩展

        Args:
            name: 测试名称
            func: 要测试的函数
            input_generator_factory: 接受batch_size返回input_generator的工厂函数
            batch_sizes: 要测试的批大小列表
            dtype: 数据类型

        Returns:
            List[BenchmarkResult]: 各批大小的测试结果
        """
        results = []
        for batch_size in batch_sizes:
            input_gen = input_generator_factory(batch_size)
            result = self.benchmark(
                name=f"{name}_batch{batch_size}",
                func=func,
                input_generator=input_gen,
                batch_size=batch_size,
                dtype=dtype,
                metadata={'scaling_test': True, 'batch_sizes': batch_sizes},
            )
            results.append(result)
        return results

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成性能分析报告

        Args:
            output_path: 可选的输出文件路径

        Returns:
            Dict: 报告内容
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'config': {
                'warmup_iterations': self.warmup_iterations,
                'benchmark_iterations': self.benchmark_iterations,
            },
            'results': [r.to_dict() for r in self.results],
            'summary': self._generate_summary(),
        }

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"报告已保存: {output_path}")

        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """生成摘要统计"""
        if not self.results:
            return {}

        # 找出最慢和最快的组件
        sorted_by_time = sorted(self.results, key=lambda r: r.mean_time_ms)
        sorted_by_memory = sorted(self.results, key=lambda r: r.peak_memory_mb, reverse=True)

        # 识别瓶颈
        bottlenecks = []
        total_time = sum(r.mean_time_ms for r in self.results)

        for r in self.results:
            time_ratio = r.mean_time_ms / total_time if total_time > 0 else 0
            if time_ratio > 0.3:  # 占用超过30%时间
                bottlenecks.append({
                    'name': r.name,
                    'time_ratio': round(time_ratio, 3),
                    'mean_time_ms': round(r.mean_time_ms, 2),
                    'suggestion': self._get_optimization_suggestion(r),
                })

        return {
            'total_tests': len(self.results),
            'fastest': {
                'name': sorted_by_time[0].name,
                'time_ms': round(sorted_by_time[0].mean_time_ms, 2),
            },
            'slowest': {
                'name': sorted_by_time[-1].name,
                'time_ms': round(sorted_by_time[-1].mean_time_ms, 2),
            },
            'highest_memory': {
                'name': sorted_by_memory[0].name,
                'memory_mb': round(sorted_by_memory[0].peak_memory_mb, 2),
            },
            'bottlenecks': bottlenecks,
            'total_time_ms': round(total_time, 2),
        }

    def _get_optimization_suggestion(self, result: BenchmarkResult) -> str:
        """根据结果生成优化建议"""
        suggestions = []

        # 时间优化建议
        if result.mean_time_ms > 100:
            suggestions.append("考虑使用更小的模型或减少层数")
        if result.std_time_ms / result.mean_time_ms > 0.2:
            suggestions.append("延迟波动大，考虑固定batch size或使用JIT编译")

        # 内存优化建议
        if result.peak_memory_mb > 1000:
            suggestions.append("内存使用高，考虑使用混合精度或梯度检查点")

        # 吞吐量建议
        if result.throughput_per_sec < 10:
            suggestions.append("吞吐量低，考虑增大batch size或使用数据并行")

        return "; ".join(suggestions) if suggestions else "性能良好"

    def print_summary(self):
        """打印性能摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("E2EControl 性能基准测试报告")
        logger.info("=" * 60)
        logger.info(f"设备: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"测试数量: {len(self.results)}")
        logger.info("-" * 60)

        for result in self.results:
            logger.info(result.summary())
            logger.info("-" * 60)

        summary = self._generate_summary()
        if summary.get('bottlenecks'):
            logger.info("\n⚠️  性能瓶颈:")
            for b in summary['bottlenecks']:
                logger.info(f"  - {b['name']}: 占用 {b['time_ratio']*100:.1f}% 时间")
                logger.info(f"    建议: {b['suggestion']}")

        logger.info("\n" + "=" * 60)


def run_quick_benchmark(func: Callable, inputs: Any, name: str = "quick_test") -> BenchmarkResult:
    """快速基准测试"""
    runner = BenchmarkRunner(warmup_iterations=5, benchmark_iterations=50)
    return runner.benchmark(
        name=name,
        func=func,
        input_generator=lambda: inputs,
        batch_size=inputs.shape[0] if hasattr(inputs, 'shape') else 1,
    )
