#!/usr/bin/env python3
"""
全场景在环测试主运行脚本 (Comprehensive HIL Test Runner)

运行成千上万种场景的全功能在环测试：
- 本体仿真 (Physics Simulation)
- 同步孪生 (Digital Twin Synchronization)
- 预测功能 (Prediction)
- 调度优化 (Scheduling Optimization)
- 控制功能 (Control)
- 异常检测与自愈 (Anomaly Detection & Self-Healing)

用法:
    python run_comprehensive_hil_test.py [--scenarios N] [--parallel] [--output PATH]
"""

import sys
import os
import argparse
import time
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.comprehensive_hil_testing.hil_coordinator import HILTestCoordinator, TestModule
from tests.comprehensive_hil_testing.report_generator import ComprehensiveReportGenerator


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗██████╗ ███████╗ ██████╗ ██████╗ ███╗   ██╗████████╗██████╗  ██████╗ ║
║   ██╔════╝╚════██╗██╔════╝██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗║
║   █████╗   █████╔╝█████╗  ██║     ██║   ██║██╔██╗ ██║   ██║   ██████╔╝██║   ██║║
║   ██╔══╝  ██╔═══╝ ██╔══╝  ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══██╗██║   ██║║
║   ███████╗███████╗███████╗╚██████╗╚██████╔╝██║ ╚████║   ██║   ██║  ██║╚██████╔╝║
║   ╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ║
║                                                                              ║
║              全场景在环测试框架 v1.0 - Comprehensive HIL Testing              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='智能水网控制系统全场景在环测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 运行1000场景的完整测试
  python run_comprehensive_hil_test.py --scenarios 1000

  # 并行运行10000场景测试
  python run_comprehensive_hil_test.py --scenarios 10000 --parallel

  # 指定输出路径
  python run_comprehensive_hil_test.py --scenarios 5000 --output ./reports/test_report
        '''
    )

    parser.add_argument(
        '--scenarios', '-n',
        type=int,
        default=1000,
        help='测试场景数量 (默认: 1000)'
    )

    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='启用并行测试'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='并行工作线程数 (默认: 4)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='hil_test_report',
        help='输出报告路径 (不含扩展名)'
    )

    parser.add_argument(
        '--modules', '-m',
        type=str,
        nargs='+',
        choices=['physics', 'digital_twin', 'prediction', 'scheduling', 'control', 'anomaly'],
        help='指定要运行的测试模块'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='快速测试模式 (100场景)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='详细输出模式'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    return parser.parse_args()


def get_modules_from_args(module_names):
    """从参数获取测试模块"""
    if not module_names:
        return list(TestModule)

    module_map = {
        'physics': TestModule.PHYSICS,
        'digital_twin': TestModule.DIGITAL_TWIN,
        'prediction': TestModule.PREDICTION,
        'scheduling': TestModule.SCHEDULING,
        'control': TestModule.CONTROL,
        'anomaly': TestModule.ANOMALY_HEALING,
    }

    return [module_map[name] for name in module_names if name in module_map]


def run_comprehensive_test(args):
    """运行全面测试"""
    print_banner()

    # 处理快速模式
    if args.quick:
        args.scenarios = 100

    print(f"测试配置:")
    print(f"  场景数: {args.scenarios}")
    print(f"  并行模式: {'是' if args.parallel else '否'}")
    print(f"  工作线程: {args.workers}")
    print(f"  随机种子: {args.seed}")
    print(f"  输出路径: {args.output}")
    print()

    # 创建协调器
    coordinator = HILTestCoordinator(
        max_scenarios=args.scenarios,
        parallel=args.parallel,
        max_workers=args.workers,
        seed=args.seed,
        verbose=args.verbose
    )

    # 获取测试模块
    modules = get_modules_from_args(args.modules)

    print(f"测试模块: {[m.value for m in modules]}")
    print()

    # 运行测试
    start_time = time.time()
    report = coordinator.run_all_tests(modules=modules)
    total_time = time.time() - start_time

    # 生成报告
    print(f"\n生成测试报告...")

    # 准备报告数据
    report_data = {
        'test_id': report.test_id,
        'test_start_time': report.test_start_time,
        'test_end_time': report.test_end_time,
        'total_execution_time': report.total_execution_time,
        'scenarios': {
            'total': report.total_scenarios,
            'by_category': report.scenarios_by_category,
            'by_difficulty': report.scenarios_by_difficulty,
        },
        'results': {
            'total_tests': report.total_tests,
            'passed': report.passed_tests,
            'failed': report.failed_tests,
            'pass_rate': report.overall_pass_rate,
            'score': report.overall_score,
        },
        'modules': {
            name: {
                'total': s.total_tests,
                'passed': s.passed_tests,
                'failed': s.failed_tests,
                'pass_rate': s.pass_rate,
                'score': s.average_score,
                'time': s.execution_time,
                'metrics': s.key_metrics,
            }
            for name, s in report.module_summaries.items()
        },
        'certification': {
            'level': report.certification_level,
            'valid': report.certification_valid,
        },
        'issues': report.critical_issues,
        'warnings': report.warnings,
        'recommendations': report.recommendations,
    }

    # 生成报告文件
    generator = ComprehensiveReportGenerator(report_data)

    # 导出所有格式
    output_base = args.output
    generator.export_json(f"{output_base}.json")
    generator.export_markdown(f"{output_base}.md")
    generator.export_html(f"{output_base}.html")

    # 打印最终摘要
    print(f"\n{'='*80}")
    print(f"  测试完成!")
    print(f"{'='*80}")
    print(f"\n最终结果:")
    print(f"  总场景数: {report.total_scenarios}")
    print(f"  总测试数: {report.total_tests}")
    print(f"  通过数: {report.passed_tests}")
    print(f"  失败数: {report.failed_tests}")
    print(f"  通过率: {report.overall_pass_rate:.1%}")
    print(f"  综合得分: {report.overall_score:.2f}")
    print(f"  认证等级: {report.certification_level}")
    print(f"  认证状态: {'✓ 有效' if report.certification_valid else '✗ 未达标'}")
    print(f"  总耗时: {total_time:.1f}s")

    print(f"\n报告已生成:")
    print(f"  - {output_base}.json")
    print(f"  - {output_base}.md")
    print(f"  - {output_base}.html")

    print(f"\n{'='*80}")

    # 返回状态码
    return 0 if report.certification_valid else 1


def main():
    """主函数"""
    args = parse_args()

    try:
        return run_comprehensive_test(args)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return 130
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
