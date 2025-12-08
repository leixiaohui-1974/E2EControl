"""
可视化与报告生成模块
Visualization and Report Generation Module

核心功能:
1. 仿真结果可视化 (文本/ASCII格式)
2. 性能报告生成
3. 场景分析报告
4. 控制效果评估报告
5. 系统状态仪表板
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time
from datetime import datetime
import json

from .hydraulic_simulator import (
    FullLineHydraulicSimulator, PoolState, SimulationRecord,
    PerformanceAnalyzer, PerformanceMetrics,
)
from .integrated_simulation import (
    ClosedLoopSimulation, SimulationConfig, RealTimeMetrics,
)
from .cascade_control import ControlEffectiveness, EscalationReason


# ==============================================================================
# 报告类型
# ==============================================================================

class ReportType(Enum):
    """报告类型"""
    SIMULATION_SUMMARY = "仿真摘要"
    PERFORMANCE_ANALYSIS = "性能分析"
    SCENARIO_ANALYSIS = "场景分析"
    CONTROL_EFFECTIVENESS = "控制效果"
    SYSTEM_DASHBOARD = "系统仪表板"


# ==============================================================================
# 文本可视化工具
# ==============================================================================

class TextVisualizer:
    """
    文本可视化器

    生成ASCII格式的可视化图表
    """

    @staticmethod
    def progress_bar(value: float, max_value: float,
                     width: int = 40, fill: str = "█") -> str:
        """生成进度条"""
        if max_value <= 0:
            return "[" + " " * width + "]"

        ratio = min(1.0, value / max_value)
        filled = int(width * ratio)
        empty = width - filled

        return f"[{fill * filled}{' ' * empty}] {ratio*100:.1f}%"

    @staticmethod
    def bar_chart(data: Dict[str, float], width: int = 40) -> str:
        """生成水平条形图"""
        if not data:
            return "No data"

        max_val = max(abs(v) for v in data.values()) or 1
        max_label = max(len(k) for k in data.keys())

        lines = []
        for label, value in data.items():
            bar_len = int(abs(value) / max_val * width)
            bar = "█" * bar_len
            lines.append(f"{label:>{max_label}}: {bar} {value:.2f}")

        return "\n".join(lines)

    @staticmethod
    def sparkline(values: List[float], width: int = 20) -> str:
        """生成迷你折线图 (使用Unicode字符)"""
        if not values:
            return ""

        # 下采样
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values

        min_val = min(sampled)
        max_val = max(sampled)
        val_range = max_val - min_val or 1

        # Unicode块字符
        blocks = " ▁▂▃▄▅▆▇█"

        line = ""
        for v in sampled:
            idx = int((v - min_val) / val_range * (len(blocks) - 1))
            line += blocks[idx]

        return line

    @staticmethod
    def table(headers: List[str], rows: List[List[Any]],
              col_widths: Optional[List[int]] = None) -> str:
        """生成表格"""
        if not headers or not rows:
            return "Empty table"

        # 计算列宽
        if col_widths is None:
            col_widths = [
                max(len(str(h)), max(len(str(row[i])) for row in rows))
                for i, h in enumerate(headers)
            ]

        # 生成分隔线
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        # 生成表头
        header_line = "|" + "|".join(
            f" {h:^{col_widths[i]}} " for i, h in enumerate(headers)
        ) + "|"

        # 生成数据行
        data_lines = []
        for row in rows:
            line = "|" + "|".join(
                f" {str(row[i]):>{col_widths[i]}} " for i in range(len(headers))
            ) + "|"
            data_lines.append(line)

        return "\n".join([sep, header_line, sep] + data_lines + [sep])

    @staticmethod
    def level_indicator(level: float, min_level: float = 0.5,
                        max_level: float = 6.0, target: float = 3.0) -> str:
        """生成水位指示器"""
        if level < min_level:
            status = "⚠️ 低水位"
        elif level > max_level:
            status = "⚠️ 高水位"
        elif abs(level - target) > 0.5:
            status = "⚡ 偏离目标"
        else:
            status = "✓ 正常"

        return f"{level:.2f}m {status}"


# ==============================================================================
# 仿真报告生成器
# ==============================================================================

@dataclass
class SimulationReport:
    """仿真报告"""
    report_id: str
    report_type: ReportType
    generated_at: str
    content: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    details: List[str] = field(default_factory=list)


class ReportGenerator:
    """
    报告生成器

    生成各类仿真报告
    """

    def __init__(self, visualizer: Optional[TextVisualizer] = None):
        self.visualizer = visualizer or TextVisualizer()

    def generate_simulation_summary(self,
                                     result: Dict[str, Any]) -> SimulationReport:
        """生成仿真摘要报告"""
        report = SimulationReport(
            report_id=f"SIM_SUM_{int(time.time())}",
            report_type=ReportType.SIMULATION_SUMMARY,
            generated_at=datetime.now().isoformat(),
        )

        # 基本信息
        config = result.get('config', {})
        execution = result.get('execution', {})
        control = result.get('control', {})
        scenarios = result.get('scenarios', {})
        performance = result.get('performance', {})

        report.content = {
            'configuration': config,
            'execution': execution,
            'control': control,
            'scenarios': scenarios,
            'performance': performance,
        }

        # 生成摘要
        lines = [
            "=" * 60,
            "              仿真摘要报告",
            "=" * 60,
            "",
            "【配置信息】",
            f"  渠池数量: {config.get('num_pools', 'N/A')}",
            f"  仿真时长: {config.get('duration', 0)/3600:.2f} 小时",
            f"  时间步长: {config.get('dt', 60)} 秒",
            "",
            "【执行情况】",
            f"  总步数: {execution.get('total_steps', 0)}",
            f"  耗时: {execution.get('elapsed_seconds', 0):.2f} 秒",
            f"  速度: {execution.get('steps_per_second', 0):.1f} 步/秒",
            "",
            "【控制统计】",
            f"  上报次数: {control.get('total_escalations', 0)}",
            f"  干预次数: {control.get('total_interventions', 0)}",
            "",
            "【场景统计】",
            f"  注入场景: {scenarios.get('injected', 0)}",
            f"  结束时活跃: {scenarios.get('active_at_end', 0)}",
            "",
            "【性能指标】",
            f"  平均RMSE: {performance.get('avg_rmse', 0):.4f} m",
            f"  最大RMSE: {performance.get('max_rmse', 0):.4f} m",
            f"  平均MAE: {performance.get('avg_mae', 0):.4f} m",
            "",
            "=" * 60,
        ]

        report.summary = "\n".join(lines)
        return report

    def generate_performance_report(self,
                                     simulator: FullLineHydraulicSimulator) -> SimulationReport:
        """生成性能分析报告"""
        report = SimulationReport(
            report_id=f"PERF_{int(time.time())}",
            report_type=ReportType.PERFORMANCE_ANALYSIS,
            generated_at=datetime.now().isoformat(),
        )

        analyzer = PerformanceAnalyzer(simulator)
        analysis = analyzer.generate_report()

        report.content = analysis

        # 按RMSE排序池
        pool_metrics = analysis.get('pool_metrics', {})
        sorted_pools = sorted(
            pool_metrics.items(),
            key=lambda x: x[1].get('rmse', 0),
            reverse=True
        )

        # 生成报告
        lines = [
            "=" * 60,
            "              性能分析报告",
            "=" * 60,
            "",
            "【总体性能】",
        ]

        summary = analysis.get('summary', {})
        lines.extend([
            f"  平均RMSE: {summary.get('avg_rmse', 0):.4f} m",
            f"  最大RMSE: {summary.get('max_rmse', 0):.4f} m",
            f"  平均MAE: {summary.get('avg_mae', 0):.4f} m",
            f"  最大MAE: {summary.get('max_mae', 0):.4f} m",
            "",
            "【RMSE最高的5个池】",
        ])

        # 表格
        headers = ["池ID", "RMSE", "MAE", "超调量", "调节时间"]
        rows = []
        for pool_id, metrics in sorted_pools[:5]:
            rows.append([
                pool_id,
                f"{metrics.get('rmse', 0):.4f}",
                f"{metrics.get('mae', 0):.4f}",
                f"{metrics.get('overshoot', 0):.3f}",
                str(metrics.get('settling_time', 'N/A')),
            ])

        lines.append(self.visualizer.table(headers, rows))

        # 水位趋势
        lines.extend([
            "",
            "【水位历史趋势】",
        ])

        for pool_id in [0, simulator.num_pools // 2, simulator.num_pools - 1]:
            history = list(simulator.state.level_history.get(pool_id, []))
            if history:
                sparkline = self.visualizer.sparkline(history)
                lines.append(f"  池{pool_id:2d}: {sparkline}")

        lines.append("")
        lines.append("=" * 60)

        report.summary = "\n".join(lines)
        return report

    def generate_scenario_report(self,
                                  simulation: ClosedLoopSimulation) -> SimulationReport:
        """生成场景分析报告"""
        report = SimulationReport(
            report_id=f"SCEN_{int(time.time())}",
            report_type=ReportType.SCENARIO_ANALYSIS,
            generated_at=datetime.now().isoformat(),
        )

        injected = simulation.scenario_injector.injected_scenarios
        active = simulation.simulator.active_scenarios

        report.content = {
            'injected_count': len(injected),
            'active_count': len(active),
            'injected_scenarios': injected,
        }

        # 生成报告
        lines = [
            "=" * 60,
            "              场景分析报告",
            "=" * 60,
            "",
            "【场景统计】",
            f"  注入场景总数: {len(injected)}",
            f"  当前活跃场景: {len(active)}",
            "",
        ]

        if injected:
            lines.append("【注入场景列表】")

            headers = ["时间", "池ID", "场景类型", "严重度"]
            rows = []
            for sc in injected[:10]:  # 最多显示10个
                scenario = sc.get('scenario')
                rows.append([
                    f"{sc.get('time', 0):.0f}s",
                    sc.get('pool_id', 'N/A'),
                    scenario.scenario_type.value if scenario else 'N/A',
                    scenario.severity.value if scenario else 'N/A',
                ])

            lines.append(self.visualizer.table(headers, rows))

            if len(injected) > 10:
                lines.append(f"  ... 还有 {len(injected) - 10} 个场景")

        lines.append("")
        lines.append("=" * 60)

        report.summary = "\n".join(lines)
        return report

    def generate_control_report(self,
                                 simulation: ClosedLoopSimulation) -> SimulationReport:
        """生成控制效果报告"""
        report = SimulationReport(
            report_id=f"CTRL_{int(time.time())}",
            report_type=ReportType.CONTROL_EFFECTIVENESS,
            generated_at=datetime.now().isoformat(),
        )

        report.content = {
            'total_escalations': simulation.total_escalations,
            'total_interventions': simulation.total_interventions,
            'executed_commands': simulation.control_interface.executed_commands,
        }

        # 生成报告
        lines = [
            "=" * 60,
            "              控制效果报告",
            "=" * 60,
            "",
            "【控制统计】",
            f"  上报总次数: {simulation.total_escalations}",
            f"  干预总次数: {simulation.total_interventions}",
            f"  执行指令数: {len(simulation.control_interface.executed_commands)}",
            "",
        ]

        # 指令分析
        commands = simulation.control_interface.executed_commands
        if commands:
            lines.append("【执行指令统计】")

            # 按类型统计
            cmd_types = {}
            for cmd in commands:
                action = cmd.get('action', 'unknown')
                cmd_types[action] = cmd_types.get(action, 0) + 1

            for action, count in cmd_types.items():
                lines.append(f"  {action}: {count} 次")

            # 最近5条指令
            lines.extend(["", "【最近执行指令】"])
            headers = ["时间", "池ID", "动作", "值"]
            rows = []
            for cmd in commands[-5:]:
                rows.append([
                    f"{cmd.get('time', 0):.0f}s",
                    cmd.get('pool_id', 'N/A'),
                    cmd.get('action', 'N/A'),
                    f"{cmd.get('value', 0):.2f}",
                ])

            lines.append(self.visualizer.table(headers, rows))

        lines.append("")
        lines.append("=" * 60)

        report.summary = "\n".join(lines)
        return report

    def generate_dashboard(self,
                           simulation: ClosedLoopSimulation) -> SimulationReport:
        """生成系统仪表板"""
        report = SimulationReport(
            report_id=f"DASH_{int(time.time())}",
            report_type=ReportType.SYSTEM_DASHBOARD,
            generated_at=datetime.now().isoformat(),
        )

        # 收集状态
        sim = simulation.simulator
        current_time = sim.state.current_time
        levels = sim.get_all_levels()
        flows = sim.get_all_flows()

        # 监控摘要
        monitor_summary = simulation.monitor.get_summary()

        report.content = {
            'current_time': current_time,
            'levels': levels,
            'flows': flows,
            'monitor_summary': monitor_summary,
        }

        # 生成仪表板
        lines = [
            "╔" + "═" * 58 + "╗",
            "║" + "系统状态仪表板".center(54) + "║",
            "╠" + "═" * 58 + "╣",
            "",
            f"  当前时间: {current_time:.0f}s ({current_time/3600:.2f}小时)",
            f"  仿真步数: {sim.state.step_count}",
            "",
            "  【水位概览】",
        ]

        # 水位统计
        level_vals = list(levels.values())
        if level_vals:
            lines.extend([
                f"    最低: {min(level_vals):.2f}m  "
                f"平均: {sum(level_vals)/len(level_vals):.2f}m  "
                f"最高: {max(level_vals):.2f}m",
            ])

        # 水位分布图
        lines.append("")
        lines.append("  【水位分布 (从上游到下游)】")
        sparkline = self.visualizer.sparkline(level_vals, width=50)
        lines.append(f"    {sparkline}")

        # 告警状态
        lines.extend(["", "  【告警状态】"])
        alarm_count = sum(1 for l in level_vals if abs(l - 3.0) > 0.5)
        if alarm_count == 0:
            lines.append("    ✓ 所有水位正常")
        else:
            lines.append(f"    ⚠️ {alarm_count} 个池水位偏离目标")

        # 活跃场景
        active_scenarios = len(sim.active_scenarios)
        lines.extend(["", "  【活跃场景】"])
        if active_scenarios == 0:
            lines.append("    ✓ 无活跃场景")
        else:
            lines.append(f"    ⚡ {active_scenarios} 个场景进行中")
            for pool_id, sc in list(sim.active_scenarios.items())[:3]:
                lines.append(f"      - 池{pool_id}: {sc.scenario_type.value}")

        # 控制状态
        lines.extend([
            "",
            "  【控制状态】",
            f"    累计上报: {simulation.total_escalations}",
            f"    累计干预: {simulation.total_interventions}",
        ])

        lines.extend([
            "",
            "╚" + "═" * 58 + "╝",
        ])

        report.summary = "\n".join(lines)
        return report


# ==============================================================================
# 综合报告生成器
# ==============================================================================

class ComprehensiveReportGenerator:
    """
    综合报告生成器

    生成包含所有分析的完整报告
    """

    def __init__(self):
        self.report_gen = ReportGenerator()

    def generate_full_report(self,
                              simulation: ClosedLoopSimulation,
                              result: Dict[str, Any]) -> str:
        """生成完整报告"""
        reports = []

        # 1. 仿真摘要
        reports.append(self.report_gen.generate_simulation_summary(result))

        # 2. 性能分析
        reports.append(self.report_gen.generate_performance_report(
            simulation.simulator
        ))

        # 3. 场景分析
        reports.append(self.report_gen.generate_scenario_report(simulation))

        # 4. 控制效果
        reports.append(self.report_gen.generate_control_report(simulation))

        # 5. 系统仪表板
        reports.append(self.report_gen.generate_dashboard(simulation))

        # 合并报告
        full_report = "\n\n".join(r.summary for r in reports)

        # 添加页眉页脚
        header = [
            "╔" + "═" * 78 + "╗",
            "║" + " " * 78 + "║",
            "║" + "南水北调中线全线全场景自主运行系统".center(66) + "║",
            "║" + "综合仿真报告".center(70) + "║",
            "║" + " " * 78 + "║",
            f"║  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 49 + "║",
            "╚" + "═" * 78 + "╝",
            "",
        ]

        footer = [
            "",
            "─" * 80,
            "报告结束",
            "─" * 80,
        ]

        return "\n".join(header) + "\n" + full_report + "\n" + "\n".join(footer)

    def generate_json_report(self,
                              simulation: ClosedLoopSimulation,
                              result: Dict[str, Any]) -> str:
        """生成JSON格式报告"""
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive',
            },
            'simulation_result': result,
            'active_scenarios': len(simulation.simulator.active_scenarios),
            'control_stats': {
                'escalations': simulation.total_escalations,
                'interventions': simulation.total_interventions,
                'commands_executed': len(simulation.control_interface.executed_commands),
            },
            'monitoring': simulation.monitor.get_summary(),
        }

        return json.dumps(report_data, indent=2, ensure_ascii=False)


# ==============================================================================
# 快速报告函数
# ==============================================================================

def print_simulation_summary(result: Dict[str, Any]):
    """打印仿真摘要"""
    gen = ReportGenerator()
    report = gen.generate_simulation_summary(result)
    print(report.summary)


def print_dashboard(simulation: ClosedLoopSimulation):
    """打印系统仪表板"""
    gen = ReportGenerator()
    report = gen.generate_dashboard(simulation)
    print(report.summary)


def print_full_report(simulation: ClosedLoopSimulation, result: Dict[str, Any]):
    """打印完整报告"""
    gen = ComprehensiveReportGenerator()
    print(gen.generate_full_report(simulation, result))


# ==============================================================================
# 导出
# ==============================================================================

__all__ = [
    'ReportType',
    'TextVisualizer',
    'SimulationReport',
    'ReportGenerator',
    'ComprehensiveReportGenerator',
    'print_simulation_summary',
    'print_dashboard',
    'print_full_report',
]
