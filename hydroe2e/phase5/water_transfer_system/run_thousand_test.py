"""
千级场景测试脚本
Run 1000+ scenario tests
"""

import time
import logging
from pathlib import Path

from hydroe2e.phase5.water_transfer_system.scenario_generator import ScenarioGenerator
from hydroe2e.phase5.water_transfer_system.batch_testing import (
    BatchTestExecutor, TestSuite, ReportGenerator,
)

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)


def run_thousand_test():
    """执行1000场景测试"""
    logger.info("=" * 70)
    logger.info("南水北调中线全场景自主运行系统 - 千级场景测试")
    logger.info("=" * 70)

    # 1. 生成测试套件
    logger.info("[1/4] 生成测试场景...")
    start_time = time.time()

    suite = TestSuite("thousand_test")

    # 配置场景分布
    single_count = 600       # 60% 单事件
    dual_count = 250         # 25% 双事件
    stress_count = 100       # 10% 压力测试
    regression_count = 50    # 5% 回归测试

    suite.generate_single_event_suite(single_count)
    logger.info("  - 生成单事件场景: %d", single_count)

    suite.generate_multi_event_suite(dual_count, max_events=3)
    logger.info("  - 生成多事件场景: %d", dual_count)

    suite.generate_stress_suite(stress_count)
    logger.info("  - 生成压力测试场景: %d", stress_count)

    suite.generate_regression_suite()
    logger.info("  - 生成回归测试场景")

    total_scenarios = len(suite.scenarios)
    gen_time = time.time() - start_time
    logger.info("  总场景数: %d", total_scenarios)
    logger.info("  生成耗时: %.2fs", gen_time)

    # 2. 验证场景
    logger.info("[2/4] 验证场景...")
    valid, errors = suite.validate_all()
    logger.info("  有效场景: %d/%d", valid, total_scenarios)
    if errors:
        logger.warning("  错误数: %d", len(errors))
        for err in errors[:5]:
            logger.warning("    - %s", err)

    # 3. 执行测试
    logger.info("[3/4] 执行测试...")
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
            logger.info(
                "  进度: %d/%d (%.1f%%) - 速率: %.1f/s - 预计剩余: %.0fs",
                current, total, current / total * 100, rate, eta,
            )

    test_start = time.time()
    result = executor.execute_batch(suite.scenarios, progress_callback=progress)
    test_time = time.time() - test_start

    # 4. 生成报告
    logger.info("[4/4] 生成报告...")
    summary = ReportGenerator.generate_summary(result)
    logger.info(summary)

    logger.info("=" * 70)
    logger.info("详细统计")
    logger.info("=" * 70)

    logger.info("测试时间: %.2fs", test_time)
    logger.info("平均每场景: %.1fms", test_time / total_scenarios * 1000)
    logger.info("通过率: %.1f%%", result.passed / total_scenarios * 100)

    logger.info("按场景类型:")
    for type_name, stats in sorted(result.by_scenario_type.items()):
        rate = stats['passed'] / max(1, stats['total']) * 100
        logger.info("  %s: %d/%d (%.1f%%)", type_name, stats['passed'], stats['total'], rate)

    logger.info("按复杂度:")
    for complexity, stats in sorted(result.by_complexity.items()):
        rate = stats['passed'] / max(1, stats['total']) * 100
        logger.info("  复杂度%s: %d/%d (%.1f%%)", complexity, stats['passed'], stats['total'], rate)

    # 保存JSON报告
    json_report = ReportGenerator.generate_json_report(result)
    report_path = Path(__file__).parent / "test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(json_report)
    logger.info("JSON报告已保存: %s", report_path)

    logger.info("=" * 70)
    logger.info("测试完成!")
    logger.info("=" * 70)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_thousand_test()
