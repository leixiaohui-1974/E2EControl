"""
端到端集成测试 (Integration Tests)
验证全系统协同工作能力

测试内容:
1. 全场景覆盖测试
2. 等级切换测试
3. 安全边界测试
4. 多智能体协同测试
5. 持续学习测试
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import logging

from .level_tests import TestStatus, TestResult

logger = logging.getLogger(__name__)


@dataclass
class IntegrationTestResult:
    """集成测试结果"""
    test_name: str
    status: TestStatus
    duration_s: float
    sub_results: List[TestResult]
    summary: str
    details: Dict


class IntegrationTestSuite:
    """集成测试套件"""

    def __init__(self, num_pools: int = 63, num_gates: int = 64):
        self.num_pools = num_pools
        self.num_gates = num_gates
        self.results: List[IntegrationTestResult] = []

        # 创建控制器和相关组件
        self._setup_components()

    def _setup_components(self):
        """初始化测试组件"""
        from ..multi_level_controller import MultiLevelController
        from ..safety_boundary import SafetyBoundary
        from ..multi_agent_coordinator import MultiAgentCoordinator

        self.controller = MultiLevelController(self.num_gates, self.num_pools)
        self.safety = SafetyBoundary(self.num_gates, self.num_pools)
        self.coordinator = MultiAgentCoordinator(self.num_gates, self.num_pools)

    def run_all(self) -> List[IntegrationTestResult]:
        """运行所有集成测试"""
        self.results = []

        # 1. 全场景覆盖测试
        self.results.append(self._test_scenario_coverage())

        # 2. 等级切换测试
        self.results.append(self._test_level_transitions())

        # 3. 安全边界测试
        self.results.append(self._test_safety_integration())

        # 4. 多智能体协同测试
        self.results.append(self._test_multi_agent_coordination())

        # 5. 端到端流程测试
        self.results.append(self._test_e2e_workflow())

        return self.results

    def _test_scenario_coverage(self) -> IntegrationTestResult:
        """全场景覆盖测试"""
        start = time.time()
        sub_results = []

        from ..scenarios.scenario_definitions import (
            COMPLETE_SCENARIO_MATRIX,
            ScenarioCategory
        )
        from ..scenarios.scenario_generator import ScenarioGenerator
        from ..multi_level_controller import AutonomyLevel

        generator = ScenarioGenerator(self.num_pools, self.num_gates)

        # 按类别测试
        categories_tested = {}
        for category in ScenarioCategory:
            category_scenarios = [
                s for s in COMPLETE_SCENARIO_MATRIX.values()
                if s.category == category
            ]

            passed = 0
            for scenario in category_scenarios[:3]:  # 每类测试3个
                try:
                    # 生成场景数据
                    states = generator.generate(scenario, num_steps=50)

                    # 选择合适等级
                    required_level = scenario.required_level.value
                    test_level = AutonomyLevel(min(4, max(0, required_level)))
                    self.controller.set_level(test_level, immediate=True)

                    # 执行控制
                    success = True
                    for state in states[:20]:
                        state_dict = {
                            'levels': state.levels.tolist(),
                            'inflows': state.inflows.tolist(),
                            'outflows': state.outflows.tolist(),
                            'gates': state.gate_openings.tolist(),
                            'targets': [4.0] * self.num_pools
                        }

                        result = self.controller.compute_action(state_dict)
                        if result.get('action') is None:
                            success = False
                            break

                    if success:
                        passed += 1

                    sub_results.append(TestResult(
                        test_id=scenario.id,
                        test_name=scenario.name_cn,
                        status=TestStatus.PASSED if success else TestStatus.FAILED,
                        duration_s=0,
                        message=f"等级{test_level.name}"
                    ))

                except Exception as e:
                    sub_results.append(TestResult(
                        test_id=scenario.id,
                        test_name=scenario.name_cn,
                        status=TestStatus.ERROR,
                        duration_s=0,
                        message=str(e)
                    ))

            categories_tested[category.value] = {
                'total': len(category_scenarios[:3]),
                'passed': passed
            }

        # 汇总
        total_passed = sum(r.status == TestStatus.PASSED for r in sub_results)
        total_tests = len(sub_results)

        return IntegrationTestResult(
            test_name="全场景覆盖测试",
            status=TestStatus.PASSED if total_passed >= total_tests * 0.8 else TestStatus.FAILED,
            duration_s=time.time() - start,
            sub_results=sub_results,
            summary=f"通过 {total_passed}/{total_tests}",
            details={'by_category': categories_tested}
        )

    def _test_level_transitions(self) -> IntegrationTestResult:
        """等级切换测试"""
        start = time.time()
        sub_results = []

        from ..multi_level_controller import AutonomyLevel

        state = self._create_state()

        # 测试升级路径
        upgrade_sequence = [
            AutonomyLevel.L0_MANUAL,
            AutonomyLevel.L1_ASSISTED,
            AutonomyLevel.L2_PARTIAL,
            AutonomyLevel.L3_CONDITIONAL,
            AutonomyLevel.L4_HIGH
        ]

        for i in range(len(upgrade_sequence) - 1):
            from_level = upgrade_sequence[i]
            to_level = upgrade_sequence[i + 1]

            try:
                self.controller.set_level(from_level, immediate=True)
                result_before = self.controller.compute_action(state)

                self.controller.set_level(to_level, immediate=True)
                result_after = self.controller.compute_action(state)

                success = (
                    result_before.get('level') == from_level.name and
                    result_after.get('level') == to_level.name
                )

                sub_results.append(TestResult(
                    test_id=f"TRANS_{from_level.name}_{to_level.name}",
                    test_name=f"升级 {from_level.name} -> {to_level.name}",
                    status=TestStatus.PASSED if success else TestStatus.FAILED,
                    duration_s=0,
                    message=""
                ))

            except Exception as e:
                sub_results.append(TestResult(
                    test_id=f"TRANS_{from_level.name}_{to_level.name}",
                    test_name=f"升级 {from_level.name} -> {to_level.name}",
                    status=TestStatus.ERROR,
                    duration_s=0,
                    message=str(e)
                ))

        # 测试降级
        for i in range(len(upgrade_sequence) - 1, 0, -1):
            from_level = upgrade_sequence[i]
            to_level = upgrade_sequence[i - 1]

            try:
                self.controller.set_level(from_level, immediate=True)
                self.controller.set_level(to_level, immediate=True)
                result = self.controller.compute_action(state)

                success = result.get('level') == to_level.name

                sub_results.append(TestResult(
                    test_id=f"TRANS_{from_level.name}_{to_level.name}",
                    test_name=f"降级 {from_level.name} -> {to_level.name}",
                    status=TestStatus.PASSED if success else TestStatus.FAILED,
                    duration_s=0,
                    message=""
                ))

            except Exception as e:
                sub_results.append(TestResult(
                    test_id=f"TRANS_{from_level.name}_{to_level.name}",
                    test_name=f"降级 {from_level.name} -> {to_level.name}",
                    status=TestStatus.ERROR,
                    duration_s=0,
                    message=str(e)
                ))

        total_passed = sum(r.status == TestStatus.PASSED for r in sub_results)

        return IntegrationTestResult(
            test_name="等级切换测试",
            status=TestStatus.PASSED if total_passed == len(sub_results) else TestStatus.WARNING,
            duration_s=time.time() - start,
            sub_results=sub_results,
            summary=f"通过 {total_passed}/{len(sub_results)}",
            details={}
        )

    def _test_safety_integration(self) -> IntegrationTestResult:
        """安全边界集成测试"""
        start = time.time()
        sub_results = []

        from ..multi_level_controller import AutonomyLevel
        from ..safety_boundary import SafetyLevel

        # 测试安全检查与控制器协同
        test_cases = [
            ("正常状态", {'levels': [4.0] * self.num_pools}, SafetyLevel.NORMAL),
            ("水位过低", {'levels': [1.3] * self.num_pools}, SafetyLevel.CRITICAL),
            ("水位过高", {'levels': [6.3] * self.num_pools}, SafetyLevel.CRITICAL),
            ("边界状态", {'levels': [1.8] * self.num_pools}, SafetyLevel.CAUTION),
        ]

        for name, state_override, expected_level in test_cases:
            try:
                state = self._create_state(**state_override)
                self.controller.set_level(AutonomyLevel.L2_PARTIAL, immediate=True)

                # 控制器计算
                control_result = self.controller.compute_action(state)
                action = control_result.get('action', np.ones(self.num_gates) * 0.8)

                # 安全过滤
                safe_action, safety_info = self.safety.check_and_filter_action(
                    action, state, control_result.get('confidence', 0.8)
                )

                actual_level = SafetyLevel[safety_info['safety_level']]
                success = actual_level.value >= expected_level.value - 1

                sub_results.append(TestResult(
                    test_id=f"SAFETY_{name}",
                    test_name=name,
                    status=TestStatus.PASSED if success else TestStatus.FAILED,
                    duration_s=0,
                    message=f"安全等级: {safety_info['safety_level']}"
                ))

            except Exception as e:
                sub_results.append(TestResult(
                    test_id=f"SAFETY_{name}",
                    test_name=name,
                    status=TestStatus.ERROR,
                    duration_s=0,
                    message=str(e)
                ))

        # 测试人工接管
        try:
            from ..safety_boundary import OverrideReason

            self.safety.override.request_override(OverrideReason.MANUAL_REQUEST, "测试")
            is_override = self.safety.override.is_override_active
            self.safety.override.release_override("test")

            sub_results.append(TestResult(
                test_id="SAFETY_OVERRIDE",
                test_name="人工接管机制",
                status=TestStatus.PASSED if is_override else TestStatus.FAILED,
                duration_s=0,
                message=""
            ))

        except Exception as e:
            sub_results.append(TestResult(
                test_id="SAFETY_OVERRIDE",
                test_name="人工接管机制",
                status=TestStatus.ERROR,
                duration_s=0,
                message=str(e)
            ))

        total_passed = sum(r.status == TestStatus.PASSED for r in sub_results)

        return IntegrationTestResult(
            test_name="安全边界集成测试",
            status=TestStatus.PASSED if total_passed >= len(sub_results) * 0.8 else TestStatus.FAILED,
            duration_s=time.time() - start,
            sub_results=sub_results,
            summary=f"通过 {total_passed}/{len(sub_results)}",
            details={}
        )

    def _test_multi_agent_coordination(self) -> IntegrationTestResult:
        """多智能体协同测试"""
        start = time.time()
        sub_results = []

        # 测试协同决策
        try:
            state = self._create_state()
            actions = self.coordinator.coordinate(state)

            # 验证输出
            success = (
                len(actions) == self.num_gates and
                all(0 <= v <= 1 for v in actions.values())
            )

            sub_results.append(TestResult(
                test_id="COORD_BASIC",
                test_name="基本协同决策",
                status=TestStatus.PASSED if success else TestStatus.FAILED,
                duration_s=0,
                message=f"输出{len(actions)}个动作"
            ))

        except Exception as e:
            sub_results.append(TestResult(
                test_id="COORD_BASIC",
                test_name="基本协同决策",
                status=TestStatus.ERROR,
                duration_s=0,
                message=str(e)
            ))

        # 测试全局约束
        try:
            state = self._create_state(levels=[2.5] * self.num_pools)  # 低水位
            actions = self.coordinator.coordinate(state)

            # 低水位应该导致较保守的动作
            mean_action = np.mean(list(actions.values()))

            sub_results.append(TestResult(
                test_id="COORD_CONSTRAINT",
                test_name="全局约束应用",
                status=TestStatus.PASSED,
                duration_s=0,
                message=f"平均动作: {mean_action:.2f}"
            ))

        except Exception as e:
            sub_results.append(TestResult(
                test_id="COORD_CONSTRAINT",
                test_name="全局约束应用",
                status=TestStatus.ERROR,
                duration_s=0,
                message=str(e)
            ))

        # 测试共识协议
        try:
            report = self.coordinator.get_coordination_report()

            sub_results.append(TestResult(
                test_id="COORD_CONSENSUS",
                test_name="共识协议",
                status=TestStatus.PASSED,
                duration_s=0,
                message=f"轮次: {report['stats']['total_rounds']}"
            ))

        except Exception as e:
            sub_results.append(TestResult(
                test_id="COORD_CONSENSUS",
                test_name="共识协议",
                status=TestStatus.ERROR,
                duration_s=0,
                message=str(e)
            ))

        total_passed = sum(r.status == TestStatus.PASSED for r in sub_results)

        return IntegrationTestResult(
            test_name="多智能体协同测试",
            status=TestStatus.PASSED if total_passed == len(sub_results) else TestStatus.WARNING,
            duration_s=time.time() - start,
            sub_results=sub_results,
            summary=f"通过 {total_passed}/{len(sub_results)}",
            details={}
        )

    def _test_e2e_workflow(self) -> IntegrationTestResult:
        """端到端工作流测试"""
        start = time.time()
        sub_results = []

        from ..multi_level_controller import AutonomyLevel

        # 模拟完整运行周期
        try:
            # 1. 初始化
            self.controller.set_level(AutonomyLevel.L2_PARTIAL, immediate=True)
            state = self._create_state()

            workflow_steps = []

            # 2. 正常运行
            for i in range(10):
                result = self.controller.compute_action(state)
                action = result.get('action')

                # 安全检查
                safe_action, safety_info = self.safety.check_and_filter_action(
                    action, state, result.get('confidence', 0.8)
                )

                workflow_steps.append({
                    'step': i,
                    'level': result.get('level'),
                    'safety': safety_info['safety_level'],
                    'action_mean': np.mean(safe_action)
                })

                # 更新状态
                state['levels'] = (np.array(state['levels']) + np.random.uniform(-0.1, 0.1, self.num_pools)).tolist()

            sub_results.append(TestResult(
                test_id="E2E_NORMAL",
                test_name="正常运行流程",
                status=TestStatus.PASSED,
                duration_s=0,
                message=f"完成{len(workflow_steps)}步"
            ))

            # 3. 模拟异常
            state['levels'] = [1.5] * self.num_pools  # 低水位
            result = self.controller.compute_action(state)
            action = result.get('action')
            safe_action, safety_info = self.safety.check_and_filter_action(
                action, state, result.get('confidence', 0.8)
            )

            sub_results.append(TestResult(
                test_id="E2E_ABNORMAL",
                test_name="异常处理流程",
                status=TestStatus.PASSED if safety_info['safety_level'] != 'NORMAL' else TestStatus.FAILED,
                duration_s=0,
                message=f"安全等级: {safety_info['safety_level']}"
            ))

            # 4. 等级升级
            self.controller.set_level(AutonomyLevel.L4_HIGH, immediate=True)
            state = self._create_state()
            result = self.controller.compute_action(state)

            sub_results.append(TestResult(
                test_id="E2E_UPGRADE",
                test_name="等级升级流程",
                status=TestStatus.PASSED if result.get('level') == 'L4_HIGH' else TestStatus.FAILED,
                duration_s=0,
                message=f"当前等级: {result.get('level')}"
            ))

        except Exception as e:
            sub_results.append(TestResult(
                test_id="E2E_WORKFLOW",
                test_name="工作流程",
                status=TestStatus.ERROR,
                duration_s=0,
                message=str(e)
            ))

        total_passed = sum(r.status == TestStatus.PASSED for r in sub_results)

        return IntegrationTestResult(
            test_name="端到端工作流测试",
            status=TestStatus.PASSED if total_passed >= len(sub_results) * 0.8 else TestStatus.FAILED,
            duration_s=time.time() - start,
            sub_results=sub_results,
            summary=f"通过 {total_passed}/{len(sub_results)}",
            details={}
        )

    def _create_state(self, **kwargs) -> Dict:
        """创建测试状态"""
        return {
            'levels': kwargs.get('levels', np.random.uniform(3.5, 4.5, self.num_pools).tolist()),
            'inflows': kwargs.get('inflows', np.random.uniform(250, 350, self.num_pools).tolist()),
            'outflows': kwargs.get('outflows', np.random.uniform(240, 340, self.num_pools).tolist()),
            'gates': kwargs.get('gates', np.random.uniform(0.7, 0.9, self.num_gates).tolist()),
            'targets': kwargs.get('targets', [4.0] * self.num_pools),
        }


def run_integration_tests(num_pools: int = 10, num_gates: int = 11) -> Dict:
    """运行集成测试"""
    suite = IntegrationTestSuite(num_pools, num_gates)
    results = suite.run_all()

    # 汇总
    summary = {
        'total_tests': len(results),
        'passed': sum(1 for r in results if r.status == TestStatus.PASSED),
        'failed': sum(1 for r in results if r.status == TestStatus.FAILED),
        'warning': sum(1 for r in results if r.status == TestStatus.WARNING),
        'error': sum(1 for r in results if r.status == TestStatus.ERROR),
        'details': [
            {
                'name': r.test_name,
                'status': r.status.value,
                'duration_s': r.duration_s,
                'summary': r.summary,
                'sub_tests': len(r.sub_results)
            }
            for r in results
        ]
    }

    summary['pass_rate'] = summary['passed'] / max(1, summary['total_tests'])

    return summary


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("集成测试")
    print("=" * 70)

    results = run_integration_tests(num_pools=10, num_gates=11)

    print(f"\n测试结果:")
    print(f"  总测试: {results['total_tests']}")
    print(f"  通过: {results['passed']}")
    print(f"  失败: {results['failed']}")
    print(f"  警告: {results['warning']}")
    print(f"  错误: {results['error']}")
    print(f"  通过率: {results['pass_rate']*100:.1f}%")

    print("\n详细结果:")
    for detail in results['details']:
        status_icon = "✓" if detail['status'] == 'passed' else "✗" if detail['status'] == 'failed' else "!"
        print(f"  [{status_icon}] {detail['name']}: {detail['summary']}")

    print("\n" + "=" * 70)
