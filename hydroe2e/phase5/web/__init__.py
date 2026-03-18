"""
Phase 5.6 Web Dashboard - Web仪表板

提供简易Web界面用于:
1. 系统状态实时监控
2. 性能指标可视化
3. 告警管理
4. 命令提交
"""

from .web_dashboard import (
    WebDashboard,
    DashboardConfig
)

__all__ = [
    'WebDashboard',
    'DashboardConfig'
]
