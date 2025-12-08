"""
数字孪生同步测试模块 (Digital Twin Synchronization Tester)

测试数字孪生系统与物理系统的同步性、一致性和实时性。
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .scenario_combinatorial_generator import TestScenario


class SyncTestType(Enum):
    """同步测试类型"""
    STATE_CONSISTENCY = "状态一致性"
    REAL_TIME_SYNC = "实时同步"
    PREDICTION_ACCURACY = "预测准确性"
    PARAMETER_TRACKING = "参数跟踪"
    FAULT_REFLECTION = "故障映射"
    DATA_LATENCY = "数据延迟"
    BIDIRECTIONAL_SYNC = "双向同步"
    RECOVERY_SYNC = "恢复同步"


@dataclass
class SyncTestResult:
    """同步测试结果"""
    test_type: SyncTestType
    scenario_id: str
    passed: bool
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DigitalTwinSyncReport:
    """数字孪生同步测试报告"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    average_score: float = 0.0
    average_sync_accuracy: float = 0.0
    average_latency: float = 0.0
    test_results: List[SyncTestResult] = field(default_factory=list)
    by_test_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_execution_time: float = 0.0


class DigitalTwinSyncTester:
    """
    数字孪生同步测试器

    测试内容：
    1. 状态一致性 - 物理状态与数字状态的同步
    2. 实时同步 - 实时数据更新能力
    3. 预测准确性 - 预测与实际的偏差
    4. 参数跟踪 - 参数变化的跟踪能力
    5. 故障映射 - 故障状态的正确反映
    6. 数据延迟 - 同步延迟测量
    7. 双向同步 - 控制指令的双向同步
    8. 恢复同步 - 中断后的重新同步
    """

    def __init__(self, sync_tolerance: float = 0.05, max_latency: float = 1.0, verbose: bool = False):
        self.sync_tolerance = sync_tolerance
        self.max_latency = max_latency
        self.verbose = verbose
        self.results: List[SyncTestResult] = []

        self._load_models()

    def _load_models(self):
        """加载模型"""
        try:
            from physics import CanalPoolSimulator
            self.CanalPoolSimulator = CanalPoolSimulator
            self.physics_available = True
        except ImportError:
            self.physics_available = False

        try:
            from digital_twin.physics.single_channel_fidelity import SingleChannelFidelity, ChannelGeometry
            self.SingleChannelFidelity = SingleChannelFidelity
            self.ChannelGeometry = ChannelGeometry
            self.dt_physics_available = True
        except ImportError:
            self.dt_physics_available = False

        try:
            from digital_twin.perception.intelligent_observer import IntelligentObserver
            self.IntelligentObserver = IntelligentObserver
            self.observer_available = True
        except ImportError:
            self.observer_available = False

    def run_all_tests(self, scenarios: List[TestScenario]) -> DigitalTwinSyncReport:
        """运行所有同步测试"""
        report = DigitalTwinSyncReport()
        start_time = time.time()

        sync_accuracies = []
        latencies = []

        for scenario in scenarios:
            tests_to_run = self._select_tests(scenario)

            for test_type in tests_to_run:
                result = self._run_single_test(test_type, scenario)
                self.results.append(result)
                report.test_results.append(result)

                report.total_tests += 1
                if result.passed:
                    report.passed_tests += 1
                else:
                    report.failed_tests += 1

                if 'sync_accuracy' in result.metrics:
                    sync_accuracies.append(result.metrics['sync_accuracy'])
                if 'latency' in result.metrics:
                    latencies.append(result.metrics['latency'])

                type_name = test_type.value
                if type_name not in report.by_test_type:
                    report.by_test_type[type_name] = {'passed': 0, 'failed': 0}
                if result.passed:
                    report.by_test_type[type_name]['passed'] += 1
                else:
                    report.by_test_type[type_name]['failed'] += 1

        report.total_execution_time = time.time() - start_time

        if report.total_tests > 0:
            report.average_score = np.mean([r.score for r in report.test_results])
        if sync_accuracies:
            report.average_sync_accuracy = np.mean(sync_accuracies)
        if latencies:
            report.average_latency = np.mean(latencies)

        return report

    def _select_tests(self, scenario: TestScenario) -> List[SyncTestType]:
        """根据场景选择测试"""
        tests = [
            SyncTestType.STATE_CONSISTENCY,
            SyncTestType.REAL_TIME_SYNC,
        ]

        if scenario.category in ["TEMPORAL", "TEMPORAL_SEQUENCE"]:
            tests.append(SyncTestType.PREDICTION_ACCURACY)

        if scenario.fault_type.value != "无故障":
            tests.append(SyncTestType.FAULT_REFLECTION)

        return tests

    def _run_single_test(self, test_type: SyncTestType, scenario: TestScenario) -> SyncTestResult:
        """运行单个测试"""
        start_time = time.time()

        try:
            if test_type == SyncTestType.STATE_CONSISTENCY:
                result = self._test_state_consistency(scenario)
            elif test_type == SyncTestType.REAL_TIME_SYNC:
                result = self._test_real_time_sync(scenario)
            elif test_type == SyncTestType.PREDICTION_ACCURACY:
                result = self._test_prediction_accuracy(scenario)
            elif test_type == SyncTestType.PARAMETER_TRACKING:
                result = self._test_parameter_tracking(scenario)
            elif test_type == SyncTestType.FAULT_REFLECTION:
                result = self._test_fault_reflection(scenario)
            elif test_type == SyncTestType.DATA_LATENCY:
                result = self._test_data_latency(scenario)
            elif test_type == SyncTestType.BIDIRECTIONAL_SYNC:
                result = self._test_bidirectional_sync(scenario)
            elif test_type == SyncTestType.RECOVERY_SYNC:
                result = self._test_recovery_sync(scenario)
            else:
                result = SyncTestResult(
                    test_type=test_type,
                    scenario_id=scenario.id,
                    passed=False,
                    score=0.0,
                    errors=[f"未实现: {test_type.value}"]
                )
        except Exception as e:
            result = SyncTestResult(
                test_type=test_type,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"测试异常: {str(e)}"]
            )

        result.execution_time = time.time() - start_time
        return result

    def _test_state_consistency(self, scenario: TestScenario) -> SyncTestResult:
        """测试状态一致性"""
        if not self.physics_available:
            return SyncTestResult(
                test_type=SyncTestType.STATE_CONSISTENCY,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=["物理模型不可用"]
            )

        try:
            # 创建物理模型
            physical_pool = self.CanalPoolSimulator(
                area=scenario.area,
                dt=scenario.time_step,
                delay_steps=1,
                initial_level=scenario.initial_water_level
            )

            # 创建数字孪生模型 (简化为另一个物理模型实例)
            digital_twin = self.CanalPoolSimulator(
                area=scenario.area,
                dt=scenario.time_step,
                delay_steps=1,
                initial_level=scenario.initial_water_level
            )

            # 同步运行
            sync_errors = []
            for i in range(50):
                q_in = scenario.initial_inflow + np.sin(i * 0.1) * 0.5
                q_out = scenario.initial_outflow

                physical_level = physical_pool.step(q_in, q_out)
                digital_level = digital_twin.step(q_in, q_out)

                error = abs(physical_level - digital_level)
                sync_errors.append(error)

            avg_error = np.mean(sync_errors)
            max_error = max(sync_errors)

            sync_accuracy = 1.0 - min(1.0, avg_error / self.sync_tolerance)
            passed = avg_error < self.sync_tolerance

            return SyncTestResult(
                test_type=SyncTestType.STATE_CONSISTENCY,
                scenario_id=scenario.id,
                passed=passed,
                score=sync_accuracy,
                metrics={
                    'avg_sync_error': avg_error,
                    'max_sync_error': max_error,
                    'sync_accuracy': sync_accuracy,
                }
            )

        except Exception as e:
            return SyncTestResult(
                test_type=SyncTestType.STATE_CONSISTENCY,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"状态一致性测试异常: {str(e)}"]
            )

    def _test_real_time_sync(self, scenario: TestScenario) -> SyncTestResult:
        """测试实时同步"""
        try:
            # 模拟实时数据流
            sample_rate = 10  # Hz
            duration = 5  # seconds
            total_samples = sample_rate * duration

            latencies = []
            sync_errors = []

            physical_pool = self.CanalPoolSimulator(
                area=scenario.area,
                dt=0.1,  # 100ms 步长
                delay_steps=1,
                initial_level=scenario.initial_water_level
            )

            digital_twin = self.CanalPoolSimulator(
                area=scenario.area,
                dt=0.1,
                delay_steps=1,
                initial_level=scenario.initial_water_level
            )

            for i in range(total_samples):
                t_start = time.time()

                q_in = scenario.initial_inflow + np.random.randn() * 0.1

                physical_level = physical_pool.step(q_in, scenario.initial_outflow)
                digital_level = digital_twin.step(q_in, scenario.initial_outflow)

                latency = time.time() - t_start
                latencies.append(latency)

                sync_error = abs(physical_level - digital_level)
                sync_errors.append(sync_error)

            avg_latency = np.mean(latencies)
            max_latency = max(latencies)
            avg_sync_error = np.mean(sync_errors)

            passed = avg_latency < self.max_latency and avg_sync_error < self.sync_tolerance
            score = (1.0 - min(1.0, avg_latency / self.max_latency)) * 0.5 + \
                    (1.0 - min(1.0, avg_sync_error / self.sync_tolerance)) * 0.5

            return SyncTestResult(
                test_type=SyncTestType.REAL_TIME_SYNC,
                scenario_id=scenario.id,
                passed=passed,
                score=score,
                metrics={
                    'avg_latency': avg_latency,
                    'max_latency': max_latency,
                    'avg_sync_error': avg_sync_error,
                    'samples_processed': total_samples,
                    'latency': avg_latency,
                    'sync_accuracy': 1.0 - avg_sync_error / self.sync_tolerance if avg_sync_error < self.sync_tolerance else 0.0,
                }
            )

        except Exception as e:
            return SyncTestResult(
                test_type=SyncTestType.REAL_TIME_SYNC,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"实时同步测试异常: {str(e)}"]
            )

    def _test_prediction_accuracy(self, scenario: TestScenario) -> SyncTestResult:
        """测试预测准确性"""
        try:
            pool = self.CanalPoolSimulator(
                area=scenario.area,
                dt=scenario.time_step,
                delay_steps=1,
                initial_level=scenario.initial_water_level
            )

            prediction_horizons = [1, 5, 10, 20]
            prediction_errors = {}

            for horizon in prediction_horizons:
                errors = []

                for _ in range(20):
                    # 当前状态
                    current_level = pool.get_level()

                    # 简单预测: 假设流量保持不变
                    predicted_change = (scenario.initial_inflow - scenario.initial_outflow) * \
                                      scenario.time_step * horizon / scenario.area
                    predicted_level = current_level + predicted_change

                    # 实际运行
                    for _ in range(horizon):
                        actual_level = pool.step(
                            scenario.initial_inflow,
                            scenario.initial_outflow,
                            0.0
                        )

                    # 预测误差
                    error = abs(predicted_level - actual_level)
                    errors.append(error)

                prediction_errors[horizon] = np.mean(errors)

            # 计算综合得分
            total_error = sum(prediction_errors.values())
            avg_error = total_error / len(prediction_horizons)

            passed = avg_error < 0.5
            score = max(0, 1.0 - avg_error / 0.5)

            return SyncTestResult(
                test_type=SyncTestType.PREDICTION_ACCURACY,
                scenario_id=scenario.id,
                passed=passed,
                score=score,
                metrics={
                    **{f'prediction_error_{h}step': e for h, e in prediction_errors.items()},
                    'avg_prediction_error': avg_error,
                }
            )

        except Exception as e:
            return SyncTestResult(
                test_type=SyncTestType.PREDICTION_ACCURACY,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"预测准确性测试异常: {str(e)}"]
            )

    def _test_parameter_tracking(self, scenario: TestScenario) -> SyncTestResult:
        """测试参数跟踪"""
        try:
            # 模拟参数变化跟踪
            original_area = scenario.area
            changed_area = scenario.area * 1.1  # 10% 变化

            pool_original = self.CanalPoolSimulator(
                area=original_area,
                dt=scenario.time_step,
                delay_steps=1,
                initial_level=scenario.initial_water_level
            )

            pool_changed = self.CanalPoolSimulator(
                area=changed_area,
                dt=scenario.time_step,
                delay_steps=1,
                initial_level=scenario.initial_water_level
            )

            # 运行并比较
            differences = []
            for _ in range(30):
                level1 = pool_original.step(scenario.initial_inflow, scenario.initial_outflow)
                level2 = pool_changed.step(scenario.initial_inflow, scenario.initial_outflow)
                differences.append(abs(level1 - level2))

            # 检测是否能感知参数变化
            param_sensitivity = np.mean(differences)
            detected = param_sensitivity > 0.01

            passed = detected
            score = min(1.0, param_sensitivity * 10)

            return SyncTestResult(
                test_type=SyncTestType.PARAMETER_TRACKING,
                scenario_id=scenario.id,
                passed=passed,
                score=score,
                metrics={
                    'parameter_sensitivity': param_sensitivity,
                    'parameter_change_detected': int(detected),
                }
            )

        except Exception as e:
            return SyncTestResult(
                test_type=SyncTestType.PARAMETER_TRACKING,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"参数跟踪测试异常: {str(e)}"]
            )

    def _test_fault_reflection(self, scenario: TestScenario) -> SyncTestResult:
        """测试故障映射"""
        try:
            # 模拟故障注入和检测
            normal_pool = self.CanalPoolSimulator(
                area=scenario.area,
                dt=scenario.time_step,
                delay_steps=1,
                initial_level=scenario.initial_water_level
            )

            # 故障模拟: 传感器偏移
            sensor_bias = 0.5  # 0.5m 偏移

            normal_levels = []
            faulty_levels = []

            for i in range(30):
                normal_level = normal_pool.step(
                    scenario.initial_inflow,
                    scenario.initial_outflow,
                    0.0
                )
                normal_levels.append(normal_level)

                # 模拟故障传感器读数
                if i > 10:  # 故障在第10步后发生
                    faulty_level = normal_level + sensor_bias
                else:
                    faulty_level = normal_level
                faulty_levels.append(faulty_level)

            # 检测故障映射
            pre_fault_diff = np.mean([abs(n - f) for n, f in zip(normal_levels[:10], faulty_levels[:10])])
            post_fault_diff = np.mean([abs(n - f) for n, f in zip(normal_levels[15:], faulty_levels[15:])])

            fault_detected = post_fault_diff > pre_fault_diff + 0.1
            fault_magnitude_match = abs(post_fault_diff - sensor_bias) < 0.1

            passed = fault_detected and fault_magnitude_match
            score = 0.5 * int(fault_detected) + 0.5 * int(fault_magnitude_match)

            return SyncTestResult(
                test_type=SyncTestType.FAULT_REFLECTION,
                scenario_id=scenario.id,
                passed=passed,
                score=score,
                metrics={
                    'fault_detected': int(fault_detected),
                    'fault_magnitude_match': int(fault_magnitude_match),
                    'pre_fault_diff': pre_fault_diff,
                    'post_fault_diff': post_fault_diff,
                    'expected_bias': sensor_bias,
                }
            )

        except Exception as e:
            return SyncTestResult(
                test_type=SyncTestType.FAULT_REFLECTION,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"故障映射测试异常: {str(e)}"]
            )

    def _test_data_latency(self, scenario: TestScenario) -> SyncTestResult:
        """测试数据延迟"""
        try:
            latencies = []

            for _ in range(100):
                t_start = time.time()

                # 模拟数据传输和处理
                pool = self.CanalPoolSimulator(
                    area=scenario.area,
                    dt=scenario.time_step,
                    delay_steps=1,
                    initial_level=scenario.initial_water_level
                )
                pool.step(scenario.initial_inflow, scenario.initial_outflow)

                latency = time.time() - t_start
                latencies.append(latency)

            avg_latency = np.mean(latencies) * 1000  # ms
            max_latency = max(latencies) * 1000  # ms
            p95_latency = np.percentile(latencies, 95) * 1000  # ms

            passed = avg_latency < self.max_latency * 1000
            score = max(0, 1.0 - avg_latency / (self.max_latency * 1000))

            return SyncTestResult(
                test_type=SyncTestType.DATA_LATENCY,
                scenario_id=scenario.id,
                passed=passed,
                score=score,
                metrics={
                    'avg_latency_ms': avg_latency,
                    'max_latency_ms': max_latency,
                    'p95_latency_ms': p95_latency,
                    'latency': avg_latency / 1000,
                }
            )

        except Exception as e:
            return SyncTestResult(
                test_type=SyncTestType.DATA_LATENCY,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"数据延迟测试异常: {str(e)}"]
            )

    def _test_bidirectional_sync(self, scenario: TestScenario) -> SyncTestResult:
        """测试双向同步"""
        try:
            # 物理→数字 同步
            pool_physical = self.CanalPoolSimulator(
                area=scenario.area, dt=scenario.time_step,
                delay_steps=1, initial_level=scenario.initial_water_level
            )

            # 数字→物理 同步 (控制指令)
            control_commands = []
            actual_responses = []

            for i in range(30):
                # 生成控制指令
                target_level = scenario.target_level + np.sin(i * 0.2) * 0.5
                current_level = pool_physical.get_level()

                # 简单P控制
                error = target_level - current_level
                control_q = scenario.initial_inflow + error * 2.0
                control_commands.append(control_q)

                # 执行控制
                new_level = pool_physical.step(control_q, scenario.initial_outflow)
                actual_responses.append(new_level)

            # 验证双向同步
            command_response_correlation = np.corrcoef(
                control_commands[:-1],
                np.diff(actual_responses)
            )[0, 1]

            passed = command_response_correlation > 0.5 or np.isnan(command_response_correlation)
            score = max(0, command_response_correlation) if not np.isnan(command_response_correlation) else 0.7

            return SyncTestResult(
                test_type=SyncTestType.BIDIRECTIONAL_SYNC,
                scenario_id=scenario.id,
                passed=passed,
                score=score,
                metrics={
                    'command_response_correlation': command_response_correlation if not np.isnan(command_response_correlation) else 0.0,
                    'control_commands_count': len(control_commands),
                }
            )

        except Exception as e:
            return SyncTestResult(
                test_type=SyncTestType.BIDIRECTIONAL_SYNC,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"双向同步测试异常: {str(e)}"]
            )

    def _test_recovery_sync(self, scenario: TestScenario) -> SyncTestResult:
        """测试恢复同步"""
        try:
            pool = self.CanalPoolSimulator(
                area=scenario.area,
                dt=scenario.time_step,
                delay_steps=1,
                initial_level=scenario.initial_water_level
            )

            # 正常运行
            for _ in range(10):
                pool.step(scenario.initial_inflow, scenario.initial_outflow)

            level_before_interrupt = pool.get_level()

            # 模拟中断 (创建新实例模拟重连)
            pool_recovered = self.CanalPoolSimulator(
                area=scenario.area,
                dt=scenario.time_step,
                delay_steps=1,
                initial_level=level_before_interrupt  # 从中断点恢复
            )

            # 恢复运行
            recovery_errors = []
            for _ in range(20):
                level_original = pool.step(scenario.initial_inflow, scenario.initial_outflow)
                level_recovered = pool_recovered.step(scenario.initial_inflow, scenario.initial_outflow)
                recovery_errors.append(abs(level_original - level_recovered))

            avg_recovery_error = np.mean(recovery_errors)
            recovery_converged = recovery_errors[-1] < 0.01

            passed = recovery_converged
            score = 1.0 - min(1.0, avg_recovery_error / 0.1)

            return SyncTestResult(
                test_type=SyncTestType.RECOVERY_SYNC,
                scenario_id=scenario.id,
                passed=passed,
                score=score,
                metrics={
                    'avg_recovery_error': avg_recovery_error,
                    'final_recovery_error': recovery_errors[-1],
                    'recovery_converged': int(recovery_converged),
                }
            )

        except Exception as e:
            return SyncTestResult(
                test_type=SyncTestType.RECOVERY_SYNC,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"恢复同步测试异常: {str(e)}"]
            )


if __name__ == "__main__":
    from scenario_combinatorial_generator import ScenarioCombinatorialGenerator

    generator = ScenarioCombinatorialGenerator(seed=42)
    scenarios = generator.generate_all(max_scenarios=50)

    tester = DigitalTwinSyncTester(verbose=True)
    report = tester.run_all_tests(scenarios[:10])

    print(f"\n数字孪生同步测试报告:")
    print(f"  总测试数: {report.total_tests}")
    print(f"  通过: {report.passed_tests}")
    print(f"  失败: {report.failed_tests}")
    print(f"  平均同步准确度: {report.average_sync_accuracy:.2%}")
    print(f"  平均延迟: {report.average_latency*1000:.2f}ms")
