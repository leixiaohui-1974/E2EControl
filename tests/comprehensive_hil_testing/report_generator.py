"""
综合测试报告生成器 (Comprehensive Report Generator)

生成详细的测试报告，支持多种格式。
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str
    subsections: List['ReportSection'] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    charts: List[Dict] = field(default_factory=list)


class ComprehensiveReportGenerator:
    """
    综合测试报告生成器

    支持格式:
    - JSON
    - Markdown
    - HTML
    - PDF (需要额外依赖)
    """

    def __init__(self, report_data: Dict[str, Any] = None):
        self.report_data = report_data or {}
        self.sections: List[ReportSection] = []

    def set_report_data(self, data: Dict[str, Any]):
        """设置报告数据"""
        self.report_data = data

    def generate_executive_summary(self) -> ReportSection:
        """生成执行摘要"""
        data = self.report_data

        pass_rate = data.get('results', {}).get('pass_rate', 0)
        score = data.get('results', {}).get('score', 0)
        cert_level = data.get('certification', {}).get('level', 'L0')

        content = f"""
## 执行摘要

本报告总结了智能水网控制系统的全场景在环测试结果。

### 测试概况
- **测试日期**: {data.get('test_start_time', 'N/A')[:10]}
- **测试场景数**: {data.get('scenarios', {}).get('total', 0)}
- **总测试用例**: {data.get('results', {}).get('total_tests', 0)}
- **执行时间**: {data.get('total_execution_time', 0):.1f} 秒

### 关键指标
- **通过率**: {pass_rate:.1%}
- **综合得分**: {score:.2f}/1.00
- **认证等级**: {cert_level}

### 结论
{'系统达到生产就绪状态，可以进行实际部署。' if cert_level in ['L4', 'L5'] else '系统需要进一步优化以达到认证标准。'}
"""
        return ReportSection(title="执行摘要", content=content)

    def generate_scenario_analysis(self) -> ReportSection:
        """生成场景分析"""
        scenarios = self.report_data.get('scenarios', {})

        content = "## 场景分析\n\n"
        content += f"本次测试共覆盖 **{scenarios.get('total', 0)}** 个测试场景。\n\n"

        # 按类别统计
        by_category = scenarios.get('by_category', {})
        if by_category:
            content += "### 场景类别分布\n\n"
            content += "| 类别 | 数量 | 占比 |\n"
            content += "|------|------|------|\n"
            total = sum(by_category.values())
            for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                content += f"| {cat} | {count} | {pct:.1f}% |\n"

        # 按难度统计
        by_difficulty = scenarios.get('by_difficulty', {})
        if by_difficulty:
            content += "\n### 场景难度分布\n\n"
            content += "| 难度 | 数量 | 占比 |\n"
            content += "|------|------|------|\n"
            total = sum(by_difficulty.values())
            for diff in sorted(by_difficulty.keys()):
                count = by_difficulty[diff]
                pct = count / total * 100 if total > 0 else 0
                stars = "★" * int(diff) + "☆" * (5 - int(diff))
                content += f"| {stars} | {count} | {pct:.1f}% |\n"

        return ReportSection(title="场景分析", content=content)

    def generate_module_results(self) -> ReportSection:
        """生成模块测试结果"""
        modules = self.report_data.get('modules', {})

        content = "## 模块测试结果\n\n"

        for module_name, module_data in modules.items():
            content += f"### {module_name}\n\n"
            content += f"| 指标 | 值 |\n"
            content += f"|------|------|\n"
            content += f"| 测试数 | {module_data.get('total', 0)} |\n"
            content += f"| 通过数 | {module_data.get('passed', 0)} |\n"
            content += f"| 失败数 | {module_data.get('failed', 0)} |\n"
            content += f"| 通过率 | {module_data.get('pass_rate', 0):.1%} |\n"
            content += f"| 得分 | {module_data.get('score', 0):.2f} |\n"
            content += f"| 执行时间 | {module_data.get('time', 0):.2f}s |\n"

            metrics = module_data.get('metrics', {})
            if metrics:
                content += f"\n**关键指标**:\n"
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        content += f"- {metric_name}: {metric_value:.4f}\n"
                    else:
                        content += f"- {metric_name}: {metric_value}\n"

            content += "\n"

        return ReportSection(title="模块测试结果", content=content)

    def generate_certification_section(self) -> ReportSection:
        """生成认证章节"""
        cert = self.report_data.get('certification', {})
        level = cert.get('level', 'L0')
        valid = cert.get('valid', False)

        # 等级说明
        level_descriptions = {
            'L0': '完全人工操作，系统仅提供数据展示',
            'L1': '辅助决策，系统提供操作建议',
            'L2': '部分自动，特定场景下自动控制',
            'L3': '条件自动，大多数场景自动控制，需人工监督',
            'L4': '高度自动，几乎所有场景自动控制',
            'L5': '完全自主，无需人工干预',
        }

        level_requirements = {
            'L1': {'pass_rate': 0.60, 'score': 0.50},
            'L2': {'pass_rate': 0.75, 'score': 0.70},
            'L3': {'pass_rate': 0.85, 'score': 0.80},
            'L4': {'pass_rate': 0.95, 'score': 0.90},
            'L5': {'pass_rate': 0.99, 'score': 0.95},
        }

        content = f"""## 认证结果

### 达成等级: {level}

**定义**: {level_descriptions.get(level, '未知')}

**认证状态**: {'✅ 认证有效' if valid else '❌ 未达标'}

### 等级要求

| 等级 | 通过率要求 | 得分要求 | 状态 |
|------|-----------|---------|------|
"""
        current_pass_rate = self.report_data.get('results', {}).get('pass_rate', 0)
        current_score = self.report_data.get('results', {}).get('score', 0)

        for l in ['L1', 'L2', 'L3', 'L4', 'L5']:
            req = level_requirements[l]
            meets_pass = current_pass_rate >= req['pass_rate']
            meets_score = current_score >= req['score']
            status = '✅' if (meets_pass and meets_score) else '❌'
            content += f"| {l} | ≥{req['pass_rate']:.0%} | ≥{req['score']:.2f} | {status} |\n"

        return ReportSection(title="认证结果", content=content)

    def generate_issues_recommendations(self) -> ReportSection:
        """生成问题与建议"""
        issues = self.report_data.get('issues', [])
        warnings = self.report_data.get('warnings', [])
        recommendations = self.report_data.get('recommendations', [])

        content = "## 问题与建议\n\n"

        if issues:
            content += "### 严重问题\n\n"
            for issue in issues:
                content += f"- ❌ {issue}\n"
            content += "\n"

        if warnings:
            content += "### 警告\n\n"
            for warning in warnings:
                content += f"- ⚠️ {warning}\n"
            content += "\n"

        if recommendations:
            content += "### 改进建议\n\n"
            for i, rec in enumerate(recommendations, 1):
                content += f"{i}. {rec}\n"
            content += "\n"

        if not issues and not warnings and not recommendations:
            content += "无需关注的问题或建议。系统运行良好。\n"

        return ReportSection(title="问题与建议", content=content)

    def generate_full_report(self) -> str:
        """生成完整报告"""
        sections = [
            self.generate_executive_summary(),
            self.generate_scenario_analysis(),
            self.generate_module_results(),
            self.generate_certification_section(),
            self.generate_issues_recommendations(),
        ]

        self.sections = sections

        # 组合报告
        report_content = f"""# 智能水网控制系统全场景在环测试报告

**报告ID**: {self.report_data.get('test_id', 'N/A')}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
        for section in sections:
            report_content += section.content + "\n---\n\n"

        report_content += """
## 附录

### A. 测试环境
- 操作系统: Linux
- Python版本: 3.8+
- 测试框架: E2EControl HIL Testing Framework v1.0

### B. 术语表
- **HIL**: Hardware-in-the-Loop，在环测试
- **MPC**: Model Predictive Control，模型预测控制
- **ADMM**: Alternating Direction Method of Multipliers，交替方向乘子法
- **FDIA**: False Data Injection Attack，虚假数据注入攻击

---

*本报告由 E2EControl 测试框架自动生成*
"""
        return report_content

    def export_json(self, filepath: str):
        """导出JSON格式报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)

    def export_markdown(self, filepath: str):
        """导出Markdown格式报告"""
        content = self.generate_full_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def export_html(self, filepath: str):
        """导出HTML格式报告"""
        md_content = self.generate_full_report()

        # 简单的Markdown到HTML转换
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能水网控制系统测试报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
            background: #f8f9fa;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #00d9ff;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #16213e;
            margin-top: 40px;
            border-left: 4px solid #00d9ff;
            padding-left: 15px;
        }}
        h3 {{
            color: #0f3460;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #f1f3f5;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        code {{
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        hr {{
            border: none;
            border-top: 1px solid #dee2e6;
            margin: 30px 0;
        }}
        .metric-card {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            margin: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <pre style="white-space: pre-wrap; font-family: inherit;">{md_content}</pre>
    </div>
</body>
</html>"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def export_all(self, base_path: str):
        """导出所有格式"""
        base = Path(base_path).stem
        dir_path = Path(base_path).parent

        self.export_json(str(dir_path / f"{base}.json"))
        self.export_markdown(str(dir_path / f"{base}.md"))
        self.export_html(str(dir_path / f"{base}.html"))

        print(f"报告已导出:")
        print(f"  - {dir_path / f'{base}.json'}")
        print(f"  - {dir_path / f'{base}.md'}")
        print(f"  - {dir_path / f'{base}.html'}")


if __name__ == "__main__":
    # 测试报告生成器
    sample_data = {
        'test_id': 'HIL_TEST_20251208',
        'test_start_time': '2025-12-08T10:00:00',
        'total_execution_time': 3600.0,
        'scenarios': {
            'total': 10000,
            'by_category': {
                'S1_NORMAL': 2000,
                'S2_FLOOD': 1500,
                'S3_DROUGHT': 1000,
                'BOUNDARY': 1500,
                'FAULT': 2000,
                'COMPOUND': 2000,
            },
            'by_difficulty': {
                '1': 2000,
                '2': 2500,
                '3': 2500,
                '4': 2000,
                '5': 1000,
            }
        },
        'results': {
            'total_tests': 50000,
            'passed': 47500,
            'failed': 2500,
            'pass_rate': 0.95,
            'score': 0.92,
        },
        'modules': {
            '本体仿真': {'total': 8000, 'passed': 7800, 'failed': 200, 'pass_rate': 0.975, 'score': 0.95, 'time': 500.0, 'metrics': {}},
            '同步孪生': {'total': 6000, 'passed': 5700, 'failed': 300, 'pass_rate': 0.95, 'score': 0.93, 'time': 400.0, 'metrics': {'sync_accuracy': 0.98}},
            '预测功能': {'total': 10000, 'passed': 9300, 'failed': 700, 'pass_rate': 0.93, 'score': 0.90, 'time': 600.0, 'metrics': {'mae': 0.05}},
            '调度优化': {'total': 8000, 'passed': 7600, 'failed': 400, 'pass_rate': 0.95, 'score': 0.92, 'time': 450.0, 'metrics': {}},
            '控制功能': {'total': 10000, 'passed': 9600, 'failed': 400, 'pass_rate': 0.96, 'score': 0.94, 'time': 550.0, 'metrics': {'tracking_error': 0.02}},
            '异常自愈': {'total': 8000, 'passed': 7500, 'failed': 500, 'pass_rate': 0.9375, 'score': 0.91, 'time': 500.0, 'metrics': {'detection_accuracy': 0.95, 'recovery_rate': 0.88}},
        },
        'certification': {
            'level': 'L4',
            'valid': True,
        },
        'issues': [],
        'warnings': ['预测功能在高难度场景下性能略低'],
        'recommendations': ['建议增加更多边界场景测试', '优化异常自愈恢复策略'],
    }

    generator = ComprehensiveReportGenerator(sample_data)
    generator.export_all('test_report')
