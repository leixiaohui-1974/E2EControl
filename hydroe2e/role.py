"""E2E控制器角色 - HydroMind 生态角色注册入口。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class E2ERoleModule:
    """HydroE2E 控制器角色，通过 entry_points 注册到 HydroMind。

    能力清单：
    - mpc_control: MPC/DMPC 最优控制
    - semantic_interpretation: 自然语言指令解析
    - digital_twin: 数字孪生仿真
    - anomaly_detection: 异常检测（15+算法）
    - fault_diagnosis: 故障诊断（5类17规则）
    - self_healing: 自愈控制
    - distributed_mpc: 分布式ADMM协调
    - scenario_recognition: 18类场景识别
    - scenario_report: 场景仿真报告（HydroClaw集成）
    """

    name: str = "e2e_controller"
    version: str = "1.1.0"
    description: str = "端到端智能水网控制系统"
    capabilities: list[str] = field(default_factory=lambda: [
        "mpc_control",
        "semantic_interpretation",
        "digital_twin",
        "anomaly_detection",
        "fault_diagnosis",
        "self_healing",
        "distributed_mpc",
        "scenario_recognition",
        "scenario_report",
    ])

    _initialized: bool = field(default=False, repr=False)
    _config: dict[str, Any] = field(default_factory=dict, repr=False)

    def get_tools(self) -> list[dict[str, Any]]:
        """返回本角色提供的 MCP 工具清单。"""
        return [
            {"name": "run_mpc_simulation", "description": "运行MPC控制仿真"},
            {"name": "interpret_instruction", "description": "解析自然语言控制指令"},
            {"name": "detect_anomaly", "description": "检测水位/流量异常"},
            {"name": "diagnose_fault", "description": "诊断故障类型与根因"},
            {"name": "run_digital_twin", "description": "运行数字孪生仿真"},
            {"name": "get_system_status", "description": "获取系统状态与指标"},
            {"name": "generate_scenario_report", "description": "运行场景仿真并通过HydroClaw生成交互式报告"},
        ]

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        """初始化角色模块。"""
        if config:
            self._config.update(config)
        self._initialized = True

    def shutdown(self) -> None:
        """关闭角色模块，释放资源。"""
        self._initialized = False
        self._config.clear()
