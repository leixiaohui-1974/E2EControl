"""
HIL Test Report Generator - 测试报告生成器

生成HTML、Markdown和JSON格式的测试报告，包含统计数据、图表和详细结果
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from hydroe2e.phase5.hil_testing.test_runner import TestSuite, TestCase, TestStatus
from hydroe2e.phase5.hil_testing.scenario_generator import AutonomousLevel, ScenarioCategory, DifficultyLevel
import logging

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """报告格式"""
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


@dataclass
class ReportConfig:
    """报告配置"""
    title: str = "HIL测试报告"
    author: str = "E2EControl自动化测试系统"
    include_charts: bool = True
    include_details: bool = True
    include_logs: bool = False
    language: str = "zh"  # zh or en


class ReportGenerator:
    """测试报告生成器"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), 'reports'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.config = ReportConfig()

    def generate(self, suites: List[TestSuite], format: ReportFormat = ReportFormat.HTML,
                 config: Optional[ReportConfig] = None) -> str:
        """生成测试报告"""
        if config:
            self.config = config

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format == ReportFormat.HTML:
            return self._generate_html(suites, timestamp)
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown(suites, timestamp)
        elif format == ReportFormat.JSON:
            return self._generate_json(suites, timestamp)
        else:
            return self._generate_text(suites, timestamp)

    def _generate_html(self, suites: List[TestSuite], timestamp: str) -> str:
        """生成HTML报告"""
        filepath = os.path.join(self.output_dir, f"report_{timestamp}.html")

        # 统计数据
        stats = self._calculate_statistics(suites)

        html = f"""<!DOCTYPE html>
<html lang="{self.config.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        :root {{
            --primary-color: #1a73e8;
            --success-color: #34a853;
            --warning-color: #fbbc04;
            --error-color: #ea4335;
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #202124;
            --text-secondary: #5f6368;
            --border-color: #dadce0;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary-color), #1557b0);
            color: white;
            padding: 40px 20px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}

        .header h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}

        .header .meta {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .card h3 {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        .card .value {{
            font-size: 2rem;
            font-weight: 600;
        }}

        .card.success .value {{ color: var(--success-color); }}
        .card.error .value {{ color: var(--error-color); }}
        .card.warning .value {{ color: var(--warning-color); }}
        .card.info .value {{ color: var(--primary-color); }}

        .progress-bar {{
            height: 10px;
            background: var(--border-color);
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }}

        .progress-bar .fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}

        .progress-bar .fill.success {{ background: var(--success-color); }}
        .progress-bar .fill.error {{ background: var(--error-color); }}

        .section {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .section h2 {{
            font-size: 1.3rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: var(--bg-color);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        tr:hover {{
            background: var(--bg-color);
        }}

        .status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        .status.passed {{
            background: #e6f4ea;
            color: var(--success-color);
        }}

        .status.failed {{
            background: #fce8e6;
            color: var(--error-color);
        }}

        .status.error {{
            background: #fce8e6;
            color: var(--error-color);
        }}

        .status.skipped {{
            background: #fef7e0;
            color: var(--warning-color);
        }}

        .level-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 5px;
        }}

        .level-L1 {{ background: #e3f2fd; color: #1976d2; }}
        .level-L2 {{ background: #e8f5e9; color: #388e3c; }}
        .level-L3 {{ background: #fff3e0; color: #f57c00; }}
        .level-L4 {{ background: #fce4ec; color: #c2185b; }}
        .level-L5 {{ background: #f3e5f5; color: #7b1fa2; }}

        .chart-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .chart {{
            background: var(--bg-color);
            border-radius: 8px;
            padding: 20px;
        }}

        .chart h4 {{
            margin-bottom: 15px;
            color: var(--text-secondary);
        }}

        .bar-chart {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .bar-row {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .bar-label {{
            width: 80px;
            font-size: 0.85rem;
        }}

        .bar {{
            flex: 1;
            height: 24px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        .bar-fill.passed {{ background: var(--success-color); }}
        .bar-fill.failed {{ background: var(--error-color); }}

        .pie-chart {{
            width: 200px;
            height: 200px;
            margin: 0 auto;
        }}

        .details {{
            margin-top: 15px;
            padding: 15px;
            background: var(--bg-color);
            border-radius: 8px;
            font-size: 0.9rem;
        }}

        .details h5 {{
            margin-bottom: 10px;
        }}

        .issue-list {{
            list-style: none;
        }}

        .issue-list li {{
            padding: 5px 0;
            border-bottom: 1px solid var(--border-color);
        }}

        .issue-list li:last-child {{
            border-bottom: none;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}

            .header {{
                padding: 20px;
            }}

            .header h1 {{
                font-size: 1.5rem;
            }}

            .card .value {{
                font-size: 1.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.config.title}</h1>
            <div class="meta">
                <div>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div>作者: {self.config.author}</div>
            </div>
        </div>

        <!-- 概览卡片 -->
        <div class="summary-cards">
            <div class="card info">
                <h3>总测试数</h3>
                <div class="value">{stats['total']}</div>
            </div>
            <div class="card success">
                <h3>通过</h3>
                <div class="value">{stats['passed']}</div>
            </div>
            <div class="card error">
                <h3>失败</h3>
                <div class="value">{stats['failed']}</div>
            </div>
            <div class="card warning">
                <h3>跳过</h3>
                <div class="value">{stats['skipped']}</div>
            </div>
            <div class="card {'success' if stats['pass_rate'] >= 0.9 else 'warning' if stats['pass_rate'] >= 0.7 else 'error'}">
                <h3>通过率</h3>
                <div class="value">{stats['pass_rate']:.1%}</div>
                <div class="progress-bar">
                    <div class="fill success" style="width: {stats['pass_rate']*100:.1f}%"></div>
                </div>
            </div>
        </div>

        <!-- 按自主等级统计 -->
        <div class="section">
            <h2>自主等级测试结果</h2>
            <div class="chart-container">
                <div class="chart">
                    <h4>各级别通过率</h4>
                    <div class="bar-chart">
                        {self._generate_level_bars(stats['by_level'])}
                    </div>
                </div>
                <div class="chart">
                    <h4>各级别测试分布</h4>
                    <div class="bar-chart">
                        {self._generate_level_distribution(stats['by_level'])}
                    </div>
                </div>
            </div>
        </div>

        <!-- 按场景类别统计 -->
        <div class="section">
            <h2>场景类别测试结果</h2>
            <table>
                <thead>
                    <tr>
                        <th>场景类别</th>
                        <th>测试数</th>
                        <th>通过</th>
                        <th>失败</th>
                        <th>通过率</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_category_rows(stats['by_category'])}
                </tbody>
            </table>
        </div>

        <!-- 测试套件详情 -->
        {self._generate_suite_sections(suites)}

        <div class="footer">
            <p>由 E2EControl HIL测试框架自动生成</p>
            <p>报告ID: {timestamp}</p>
        </div>
    </div>
</body>
</html>"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        return filepath

    def _generate_level_bars(self, by_level: Dict) -> str:
        """生成等级条形图"""
        bars = []
        for level, data in sorted(by_level.items()):
            pass_rate = data['pass_rate'] * 100
            bars.append(f"""
                <div class="bar-row">
                    <span class="bar-label">{level}</span>
                    <div class="bar">
                        <div class="bar-fill passed" style="width: {pass_rate:.1f}%">
                            {pass_rate:.1f}%
                        </div>
                    </div>
                </div>
            """)
        return '\n'.join(bars) if bars else '<div>暂无数据</div>'

    def _generate_level_distribution(self, by_level: Dict) -> str:
        """生成等级分布图"""
        total = sum(d['total'] for d in by_level.values())
        bars = []
        for level, data in sorted(by_level.items()):
            pct = (data['total'] / total * 100) if total > 0 else 0
            bars.append(f"""
                <div class="bar-row">
                    <span class="bar-label">{level}</span>
                    <div class="bar">
                        <div class="bar-fill passed" style="width: {pct:.1f}%; background: var(--primary-color);">
                            {data['total']}个
                        </div>
                    </div>
                </div>
            """)
        return '\n'.join(bars) if bars else '<div>暂无数据</div>'

    def _generate_category_rows(self, by_category: Dict) -> str:
        """生成类别表格行"""
        rows = []
        for category, data in sorted(by_category.items()):
            rows.append(f"""
                <tr>
                    <td>{category}</td>
                    <td>{data['total']}</td>
                    <td>{data['passed']}</td>
                    <td>{data['failed']}</td>
                    <td>
                        <span class="status {'passed' if data['pass_rate'] >= 0.9 else 'failed'}">
                            {data['pass_rate']:.1%}
                        </span>
                    </td>
                </tr>
            """)
        return '\n'.join(rows) if rows else '<tr><td colspan="5">暂无数据</td></tr>'

    def _generate_suite_sections(self, suites: List[TestSuite]) -> str:
        """生成测试套件详情部分"""
        sections = []

        for suite in suites:
            cases_html = []
            for tc in suite.test_cases:
                status_class = tc.status.value
                level = tc.scenario.autonomous_level.value if tc.scenario.autonomous_level else 'N/A'

                result_html = ""
                if tc.result and self.config.include_details:
                    issues = tc.result.issues[:3] if tc.result.issues else []
                    issues_html = '\n'.join(f'<li>{issue}</li>' for issue in issues)

                    result_html = f"""
                        <div class="details">
                            <h5>评估结果</h5>
                            <p>得分: {tc.result.score:.2f}</p>
                            {'<h5>问题</h5><ul class="issue-list">' + issues_html + '</ul>' if issues_html else ''}
                        </div>
                    """ if tc.result.issues else ""

                score_str = f"{tc.result.score:.2f}" if tc.result else 'N/A'
                cases_html.append(f"""
                    <tr>
                        <td>
                            <span class="level-badge level-{level}">{level}</span>
                            {tc.scenario.name}
                        </td>
                        <td>{tc.scenario.category.value if tc.scenario.category else 'N/A'}</td>
                        <td>{'★' * tc.scenario.difficulty.value if tc.scenario.difficulty else 'N/A'}</td>
                        <td><span class="status {status_class}">{status_class.upper()}</span></td>
                        <td>{score_str}</td>
                    </tr>
                """)

            sections.append(f"""
                <div class="section">
                    <h2>{suite.name}</h2>
                    <p style="color: var(--text-secondary); margin-bottom: 15px;">{suite.description}</p>
                    <div class="summary-cards" style="margin-bottom: 20px;">
                        <div class="card" style="padding: 15px;">
                            <h3>测试数</h3>
                            <div class="value" style="font-size: 1.5rem;">{suite.total_count}</div>
                        </div>
                        <div class="card success" style="padding: 15px;">
                            <h3>通过率</h3>
                            <div class="value" style="font-size: 1.5rem;">{suite.pass_rate:.1%}</div>
                        </div>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>场景名称</th>
                                <th>类别</th>
                                <th>难度</th>
                                <th>状态</th>
                                <th>得分</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(cases_html)}
                        </tbody>
                    </table>
                </div>
            """)

        return '\n'.join(sections)

    def _generate_markdown(self, suites: List[TestSuite], timestamp: str) -> str:
        """生成Markdown报告"""
        filepath = os.path.join(self.output_dir, f"report_{timestamp}.md")
        stats = self._calculate_statistics(suites)

        md = f"""# {self.config.title}

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**作者**: {self.config.author}

## 测试概览

| 指标 | 值 |
|------|-----|
| 总测试数 | {stats['total']} |
| 通过 | {stats['passed']} |
| 失败 | {stats['failed']} |
| 错误 | {stats['errors']} |
| 跳过 | {stats['skipped']} |
| **通过率** | **{stats['pass_rate']:.1%}** |

## 自主等级统计

| 等级 | 测试数 | 通过 | 失败 | 通过率 |
|------|--------|------|------|--------|
"""

        for level, data in sorted(stats['by_level'].items()):
            md += f"| {level} | {data['total']} | {data['passed']} | {data['failed']} | {data['pass_rate']:.1%} |\n"

        md += "\n## 场景类别统计\n\n"
        md += "| 类别 | 测试数 | 通过 | 失败 | 通过率 |\n"
        md += "|------|--------|------|------|--------|\n"

        for category, data in sorted(stats['by_category'].items()):
            md += f"| {category} | {data['total']} | {data['passed']} | {data['failed']} | {data['pass_rate']:.1%} |\n"

        md += "\n## 测试套件详情\n"

        for suite in suites:
            md += f"\n### {suite.name}\n\n"
            md += f"{suite.description}\n\n"
            md += f"- 总测试数: {suite.total_count}\n"
            md += f"- 通过: {suite.passed_count}\n"
            md += f"- 失败: {suite.failed_count}\n"
            md += f"- 通过率: {suite.pass_rate:.1%}\n\n"

            md += "| 场景 | 等级 | 类别 | 难度 | 状态 | 得分 |\n"
            md += "|------|------|------|------|------|------|\n"

            for tc in suite.test_cases:
                level = tc.scenario.autonomous_level.value if tc.scenario.autonomous_level else 'N/A'
                category = tc.scenario.category.value if tc.scenario.category else 'N/A'
                difficulty = '★' * tc.scenario.difficulty.value if tc.scenario.difficulty else 'N/A'
                status = tc.status.value.upper()
                score = f"{tc.result.score:.2f}" if tc.result else 'N/A'

                md += f"| {tc.scenario.name} | {level} | {category} | {difficulty} | {status} | {score} |\n"

        md += f"\n---\n\n*由 E2EControl HIL测试框架自动生成*\n*报告ID: {timestamp}*\n"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)

        return filepath

    def _generate_json(self, suites: List[TestSuite], timestamp: str) -> str:
        """生成JSON报告"""
        filepath = os.path.join(self.output_dir, f"report_{timestamp}.json")
        stats = self._calculate_statistics(suites)

        report = {
            'metadata': {
                'title': self.config.title,
                'author': self.config.author,
                'generated_at': datetime.now().isoformat(),
                'report_id': timestamp
            },
            'summary': stats,
            'suites': []
        }

        for suite in suites:
            suite_data = {
                'id': suite.id,
                'name': suite.name,
                'description': suite.description,
                'statistics': {
                    'total': suite.total_count,
                    'passed': suite.passed_count,
                    'failed': suite.failed_count,
                    'errors': suite.error_count,
                    'skipped': suite.skipped_count,
                    'pass_rate': suite.pass_rate
                },
                'test_cases': []
            }

            for tc in suite.test_cases:
                tc_data = {
                    'id': tc.id,
                    'scenario': {
                        'id': tc.scenario.id,
                        'name': tc.scenario.name,
                        'category': tc.scenario.category.value if tc.scenario.category else None,
                        'difficulty': tc.scenario.difficulty.value if tc.scenario.difficulty else None,
                        'autonomous_level': tc.scenario.autonomous_level.value if tc.scenario.autonomous_level else None
                    },
                    'status': tc.status.value,
                    'start_time': tc.start_time.isoformat() if tc.start_time else None,
                    'end_time': tc.end_time.isoformat() if tc.end_time else None,
                    'error_message': tc.error_message
                }

                if tc.result:
                    tc_data['result'] = {
                        'passed': tc.result.passed,
                        'score': tc.result.score,
                        'issues': tc.result.issues,
                        'recommendations': tc.result.recommendations
                    }

                suite_data['test_cases'].append(tc_data)

            report['suites'].append(suite_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return filepath

    def _generate_text(self, suites: List[TestSuite], timestamp: str) -> str:
        """生成纯文本报告"""
        filepath = os.path.join(self.output_dir, f"report_{timestamp}.txt")
        stats = self._calculate_statistics(suites)

        lines = [
            "=" * 60,
            f"  {self.config.title}",
            "=" * 60,
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"作者: {self.config.author}",
            "",
            "-" * 40,
            "  测试概览",
            "-" * 40,
            f"  总测试数:   {stats['total']}",
            f"  通过:       {stats['passed']}",
            f"  失败:       {stats['failed']}",
            f"  错误:       {stats['errors']}",
            f"  跳过:       {stats['skipped']}",
            f"  通过率:     {stats['pass_rate']:.1%}",
            "",
            "-" * 40,
            "  自主等级统计",
            "-" * 40,
        ]

        for level, data in sorted(stats['by_level'].items()):
            lines.append(f"  {level}: {data['total']}个测试, 通过率 {data['pass_rate']:.1%}")

        lines.extend([
            "",
            "-" * 40,
            "  场景类别统计",
            "-" * 40,
        ])

        for category, data in sorted(stats['by_category'].items()):
            lines.append(f"  {category}: {data['total']}个测试, 通过率 {data['pass_rate']:.1%}")

        lines.extend([
            "",
            "-" * 40,
            "  测试套件详情",
            "-" * 40,
        ])

        for suite in suites:
            lines.extend([
                "",
                f"  [{suite.name}]",
                f"  {suite.description}",
                f"  测试: {suite.total_count} | 通过: {suite.passed_count} | 失败: {suite.failed_count}",
                ""
            ])

            for tc in suite.test_cases:
                status = '✓' if tc.status == TestStatus.PASSED else '✗' if tc.status in [TestStatus.FAILED, TestStatus.ERROR] else '○'
                level = tc.scenario.autonomous_level.value if tc.scenario.autonomous_level else 'N/A'
                score = f"{tc.result.score:.2f}" if tc.result else 'N/A'
                lines.append(f"    {status} [{level}] {tc.scenario.name} (得分: {score})")

        lines.extend([
            "",
            "=" * 60,
            f"  报告ID: {timestamp}",
            "=" * 60,
        ])

        text = '\n'.join(lines)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

        return filepath

    def _calculate_statistics(self, suites: List[TestSuite]) -> Dict:
        """计算统计数据"""
        total = sum(s.total_count for s in suites)
        passed = sum(s.passed_count for s in suites)
        failed = sum(s.failed_count for s in suites)
        errors = sum(s.error_count for s in suites)
        skipped = sum(s.skipped_count for s in suites)

        # 按等级统计
        by_level = {}
        for suite in suites:
            for tc in suite.test_cases:
                level = tc.scenario.autonomous_level.value if tc.scenario.autonomous_level else 'UNKNOWN'
                if level not in by_level:
                    by_level[level] = {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0}
                by_level[level]['total'] += 1
                if tc.status == TestStatus.PASSED:
                    by_level[level]['passed'] += 1
                elif tc.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    by_level[level]['failed'] += 1

        for data in by_level.values():
            executed = data['passed'] + data['failed']
            data['pass_rate'] = data['passed'] / executed if executed > 0 else 0

        # 按类别统计
        by_category = {}
        for suite in suites:
            for tc in suite.test_cases:
                category = tc.scenario.category.value if tc.scenario.category else 'UNKNOWN'
                if category not in by_category:
                    by_category[category] = {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0}
                by_category[category]['total'] += 1
                if tc.status == TestStatus.PASSED:
                    by_category[category]['passed'] += 1
                elif tc.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    by_category[category]['failed'] += 1

        for data in by_category.values():
            executed = data['passed'] + data['failed']
            data['pass_rate'] = data['passed'] / executed if executed > 0 else 0

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'pass_rate': passed / (passed + failed) if (passed + failed) > 0 else 0,
            'by_level': by_level,
            'by_category': by_category
        }


def main():
    """演示报告生成"""
    from hydroe2e.phase5.hil_testing.test_runner import HILTestRunner

    logger.info("=" * 60)
    logger.info("HIL测试报告生成器 - 演示")
    logger.info("=" * 60)

    # 运行一些测试
    runner = HILTestRunner()

    # 创建测试场景
    scenarios = []
    for i, (cat, level, diff) in enumerate([
        ("S1_NORMAL", "L1", 1),
        ("S1_NORMAL", "L2", 2),
        ("S2_FLOOD", "L2", 2),
        ("S2_FLOOD", "L3", 3),
        ("S3_DROUGHT", "L2", 2),
    ]):
        s = runner.scenario_generator.create_scenario(
            id=f"TEST_{cat}_{i:03d}",
            name=f"测试场景 {i+1}",
            category=cat,
            difficulty=diff,
            autonomous_level=level,
            duration=60.0
        )
        scenarios.append(s)

    # 创建并运行测试套件
    suite = runner.create_test_suite(
        name="演示测试套件",
        scenarios=scenarios,
        description="用于演示报告生成的测试套件"
    )

    runner.run_suite(suite)

    # 生成报告
    report_gen = ReportGenerator()

    logger.info("\n生成测试报告...")

    # HTML报告
    html_path = report_gen.generate([suite], ReportFormat.HTML)
    logger.info(f"  HTML报告: {html_path}")

    # Markdown报告
    md_path = report_gen.generate([suite], ReportFormat.MARKDOWN)
    logger.info(f"  Markdown报告: {md_path}")

    # JSON报告
    json_path = report_gen.generate([suite], ReportFormat.JSON)
    logger.info(f"  JSON报告: {json_path}")

    # 文本报告
    txt_path = report_gen.generate([suite], ReportFormat.TEXT)
    logger.info(f"  文本报告: {txt_path}")

    logger.info("\n报告生成完成！")


if __name__ == '__main__':
    main()
