"""
千级场景测试脚本
Run 1000+ scenario tests
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from phase5.water_transfer_system.scenario_generator import ScenarioGenerator
from phase5.water_transfer_system.batch_testing import (
    BatchTestExecutor, TestSuite, ReportGenerator,
)

logging.basicConfig(level=logging.WARNING)


def run_thousand_test():
    """执行1000场景测试"""
    print("=" * 70)
    print(" " * 15 + "南水北调中线全场景自主运行系统")
    print(" " * 18 + "千级场景测试")
    print("=" * 70)

    # 1. 生成测试套件
    print("\n[1/4] 生成测试场景...")
    start_time = time.time()

    suite = TestSuite("thousand_test")

    # 配置场景分布
    single_count = 600       # 60% 单事件
    dual_count = 250         # 25% 双事件
    stress_count = 100       # 10% 压力测试
    regression_count = 50    # 5% 回归测试

    suite.generate_single_event_suite(single_count)
    print(f"  - 生成单事件场景: {single_count}")

    suite.generate_multi_event_suite(dual_count, max_events=3)
    print(f"  - 生成多事件场景: {dual_count}")

    suite.generate_stress_suite(stress_count)
    print(f"  - 生成压力测试场景: {stress_count}")

    suite.generate_regression_suite()
    print(f"  - 生成回归测试场景")

    total_scenarios = len(suite.scenarios)
    gen_time = time.time() - start_time
    print(f"\n  总场景数: {total_scenarios}")
    print(f"  生成耗时: {gen_time:.2f}s")

    # 2. 验证场景
    print("\n[2/4] 验证场景...")
    valid, errors = suite.validate_all()
    print(f"  有效场景: {valid}/{total_scenarios}")
    if errors:
        print(f"  错误数: {len(errors)}")
        for err in errors[:5]:
            print(f"    - {err}")

    # 3. 执行测试
    print("\n[3/4] 执行测试...")
    executor = BatchTestExecutor(
        simulation_steps=15,      # 每场景15步
        timeout_per_test=60,      # 60秒超时
    )

    tested = 0
    def progress(current, total):
        nonlocal tested
        tested = current
        if current % 100 == 0 or current == total:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            eta = (total - current) / rate if rate > 0 else 0
            print(f"  进度: {current}/{total} ({current/total*100:.1f}%) "
                  f"- 速率: {rate:.1f}/s - 预计剩余: {eta:.0f}s")

    test_start = time.time()
    result = executor.execute_batch(suite.scenarios, progress_callback=progress)
    test_time = time.time() - test_start

    # 4. 生成报告
    print("\n[4/4] 生成报告...")
    summary = ReportGenerator.generate_summary(result)
    print(summary)

    # 详细统计
    print("\n" + "=" * 70)
    print(" " * 25 + "详细统计")
    print("=" * 70)

    print(f"\n测试时间: {test_time:.2f}s")
    print(f"平均每场景: {test_time/total_scenarios*1000:.1f}ms")
    print(f"通过率: {result.passed/total_scenarios*100:.1f}%")

    print("\n按场景类型:")
    for type_name, stats in sorted(result.by_scenario_type.items()):
        rate = stats['passed'] / max(1, stats['total']) * 100
        print(f"  {type_name}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    print("\n按复杂度:")
    for complexity, stats in sorted(result.by_complexity.items()):
        rate = stats['passed'] / max(1, stats['total']) * 100
        print(f"  复杂度{complexity}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    # 保存JSON报告
    json_report = ReportGenerator.generate_json_report(result)
    report_path = "/home/user/E2EControl/phase5/water_transfer_system/test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(json_report)
    print(f"\nJSON报告已保存: {report_path}")

    print("\n" + "=" * 70)
    print(" " * 20 + "测试完成!")
    print("=" * 70)

    return result


if __name__ == "__main__":
    run_thousand_test()
