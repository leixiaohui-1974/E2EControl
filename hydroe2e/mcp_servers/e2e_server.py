"""HydroE2E MCP Server - 端到端控制工具服务。

通过 HydroClaw MCP Gateway 或独立运行，暴露以下工具：
- run_mpc_simulation: MPC控制仿真
- interpret_instruction: 自然语言指令解析
- detect_anomaly: 异常检测
- diagnose_fault: 故障诊断
- run_digital_twin: 数字孪生
- generate_scenario_report: 场景仿真报告（HydroClaw集成）
- get_system_status: 系统状态
"""
from __future__ import annotations
import json
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    FastMCP = None


def _create_server() -> Any:
    """创建 MCP 服务实例。"""
    if not HAS_MCP:
        raise ImportError(
            "MCP SDK 未安装。请运行: pip install mcp"
        )

    server = FastMCP("HydroE2E", description="端到端智能水网控制")

    @server.tool()
    def run_mpc_simulation(
        instruction: str,
        total_hours: int = 50,
        time_step: float = 3600.0,
    ) -> dict[str, Any]:
        """运行MPC控制仿真。

        Args:
            instruction: 控制指令（自然语言）
            total_hours: 仿真总时长（小时）
            time_step: 时间步长（秒）
        """
        from hydroe2e.config_manager import get_config
        from hydroe2e.simulation_manager import SimulationManager

        config = get_config()
        config["simulation"]["total_hours"] = total_hours
        config["simulation"]["time_step"] = time_step

        manager = SimulationManager(config)
        result = manager.run(instruction)
        return {"status": "success", "result": result}

    @server.tool()
    def interpret_instruction(instruction: str) -> dict[str, Any]:
        """解析自然语言控制指令为结构化参数。

        Args:
            instruction: 中文控制指令
        """
        from hydroe2e.brain_enhanced import EnhancedSemanticInterpreter

        interpreter = EnhancedSemanticInterpreter()
        result = interpreter.interpret(instruction)
        return {"status": "success", "interpretation": result}

    @server.tool()
    def detect_anomaly(
        water_levels: list[float],
        flow_rates: list[float] | None = None,
        method: str = "ensemble",
    ) -> dict[str, Any]:
        """检测水位/流量数据中的异常。

        Args:
            water_levels: 水位时间序列
            flow_rates: 流量时间序列（可选）
            method: 检测方法 (ensemble|statistical|ml)
        """
        import numpy as np
        try:
            from hydroe2e.phase4.anomaly_detection import EnsembleDetector
            detector = EnsembleDetector()
            data = np.array(water_levels)
            results = detector.detect(data)
            return {"status": "success", "anomalies": results}
        except ImportError:
            return {"status": "error", "message": "异常检测模块未就绪"}

    @server.tool()
    def diagnose_fault(
        anomaly_type: str,
        sensor_data: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """诊断故障类型与根因。

        Args:
            anomaly_type: 异常类型标识
            sensor_data: 传感器数据字典
        """
        try:
            from hydroe2e.phase4.fault_diagnosis import DiagnosisEngine
            engine = DiagnosisEngine()
            result = engine.diagnose({
                "anomaly_type": anomaly_type,
                "sensor_data": sensor_data or {},
            })
            return {"status": "success", "diagnosis": result}
        except ImportError:
            return {"status": "error", "message": "故障诊断模块未就绪"}

    @server.tool()
    def generate_scenario_report(
        scenario_id: str = "S01",
    ) -> dict[str, Any]:
        """运行场景仿真并通过HydroClaw生成交互式报告。

        Args:
            scenario_id: 场景编号 (S01-S06)
        """
        from scripts.generate_scenario_report import SCENARIOS, run_scenario

        # 1. 查找场景定义
        sc = None
        for s in SCENARIOS:
            if s.id == scenario_id:
                sc = s
                break
        if sc is None:
            return {
                "status": "error",
                "message": f"未知场景编号: {scenario_id}，可用: S01-S06",
            }

        # 2. 运行本地仿真
        result = run_scenario(sc)

        # 3. 构建报告数据
        result_data = {
            "scenario_id": sc.id,
            "scenario_name": sc.name,
            "category": sc.category,
            "metrics": result["metrics"],
            "history": result["history"],
            "config": result["history"]["config"][0] if result["history"].get("config") else {},
            "elapsed_ms": result["elapsed"] * 1000,
        }

        # 4. 调用 HydroClaw 报告服务（不可用时自动回退本地生成）
        from hydroe2e.report_client import generate_scenario_report as gen_report

        report_path = gen_report(result_data)

        return {
            "status": "success",
            "scenario_id": sc.id,
            "scenario_name": sc.name,
            "metrics": result["metrics"],
            "report_path": report_path,
            "elapsed_ms": round(result["elapsed"] * 1000, 1),
        }

    @server.tool()
    def get_system_status() -> dict[str, Any]:
        """获取 HydroE2E 系统状态与版本信息。"""
        from hydroe2e import __version__
        return {
            "name": "HydroE2E",
            "version": __version__,
            "status": "running",
            "capabilities": [
                "mpc_control", "semantic_interpretation",
                "digital_twin", "anomaly_detection",
                "fault_diagnosis", "self_healing",
            ],
        }

    return server


# Module-level server instance for entry_point discovery
def main():
    """启动 MCP 服务（独立运行模式）。"""
    server = _create_server()
    server.run()


if __name__ == "__main__":
    main()
