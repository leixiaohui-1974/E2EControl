"""
分布式SIL框架 (Distributed SIL Framework)

将所有模块整合为完整的SIL测试系统

架构:
┌─────────────────────────────────────────────────────────────┐
│                    场景生成器 (Scenario Generator)            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ 真值边界    │  │ 噪声/偏置   │  │ 参数集合    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              双层模型架构 (Two-Level Model)                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          全线降阶引擎 (IDZ - 长时域稳定)              │    │
│  └─────────────────────────────────────────────────────┘    │
│                      ↑↓ 接口同化 ↑↓                          │
│  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐    │
│  │ HF_1  │  │ HF_2  │  │ HF_3  │  │  ...  │  │ HF_N  │    │
│  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘    │
│       局部高保真模型 (Saint-Venant - 物理真实性)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               控制器编排器 (Controller Orchestrator)          │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                        │
│  │ L1  │  │ L2  │  │ L3  │  │ L4  │                        │
│  └─────┘  └─────┘  └─────┘  └─────┘                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  评估器 (SIL Evaluator)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ 水量平衡    │  │ 接口残差    │  │ 漂移率      │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

from .scenario_generator import ScenarioGenerator, ScenarioType
from .controller_orchestrator import ControllerOrchestrator, ControlLevel
from ..models.reduced_order_engine import ReducedOrderEngine
from ..models.segmented_high_fidelity import SegmentedHighFidelityModel
from ..interfaces.boundary_assimilator import BoundaryAssimilator, FusionConfig, FusionMode
from ..interfaces.data_types import (
    SegmentState, BoundaryCondition, ScenarioConfig,
    SimulationResult, KPIMetrics
)
from ..evaluators.sil_evaluator import SILEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """仿真模式"""
    REDUCED_ORDER_ONLY = "reduced_order"     # 仅降阶
    HIGH_FIDELITY_ONLY = "high_fidelity"     # 仅高保真
    HYBRID = "hybrid"                         # 混合模式 (推荐)
    ENSEMBLE = "ensemble"                     # 集合仿真


@dataclass
class SILConfig:
    """SIL配置"""
    # 系统参数
    num_segments: int = 63
    num_gates: int = 64
    total_length: float = 1432000.0  # m

    # 时间参数
    dt_reduced: float = 900.0        # s (降阶模型时间步)
    dt_high_fidelity: float = 60.0   # s (高保真时间步)
    sync_interval: float = 300.0     # s (同步间隔)

    # 模式
    simulation_mode: SimulationMode = SimulationMode.HYBRID

    # 高保真模型位置 (关键闸站)
    high_fidelity_segments: List[int] = field(default_factory=lambda: [0, 10, 20, 30, 40, 50, 62])

    # 融合配置
    fusion_mode: FusionMode = FusionMode.ADAPTIVE

    # 控制层级
    active_control_levels: List[ControlLevel] = field(
        default_factory=lambda: [ControlLevel.L1_LOCAL, ControlLevel.L2_COORDINATED]
    )


class DistributedSILFramework:
    """
    分布式SIL框架

    整合场景生成、双层模型、控制器、评估器的完整SIL系统
    """

    def __init__(
        self,
        config: Optional[SILConfig] = None,
    ):
        """
        初始化SIL框架

        Args:
            config: SIL配置
        """
        self.config = config or SILConfig()

        # 初始化各模块
        self._init_modules()

        # 仿真状态
        self.current_time = 0.0
        self.step_count = 0
        self.is_running = False

        # 结果
        self.results: Optional[SimulationResult] = None

        logger.info(f"DistributedSILFramework initialized: mode={self.config.simulation_mode.value}")

    def _init_modules(self):
        """初始化所有模块"""
        # 场景生成器
        self.scenario_generator = ScenarioGenerator(
            num_segments=self.config.num_segments,
            dt=self.config.dt_reduced,
        )

        # 全线降阶引擎
        self.reduced_order_engine = ReducedOrderEngine(
            num_pools=self.config.num_segments,
            dt=self.config.dt_reduced,
            total_length=self.config.total_length,
        )

        # 局部高保真模型
        self.high_fidelity_models: Dict[str, SegmentedHighFidelityModel] = {}
        for seg_idx in self.config.high_fidelity_segments:
            segment_id = f"SEG_{seg_idx:03d}"
            self.high_fidelity_models[segment_id] = SegmentedHighFidelityModel(
                segment_id=segment_id,
                config=None,  # 使用默认配置
            )

        # 接口同化器
        fusion_config = FusionConfig(mode=self.config.fusion_mode)
        self.boundary_assimilator = BoundaryAssimilator(
            num_interfaces=self.config.num_segments - 1,
            config=fusion_config,
            dt=self.config.dt_reduced,
        )

        # 控制器编排器
        self.controller_orchestrator = ControllerOrchestrator(
            num_gates=self.config.num_gates,
            default_control_interval=self.config.dt_reduced,
        )
        self.controller_orchestrator.create_default_controllers()
        self.controller_orchestrator.activate_controllers(
            self.config.active_control_levels
        )

        # 评估器
        self.evaluator = SILEvaluator(
            num_segments=self.config.num_segments,
            evaluation_interval=self.config.sync_interval,
        )

        logger.info(
            f"Modules initialized: {len(self.high_fidelity_models)} high-fidelity models"
        )

    def load_scenario(
        self,
        scenario_type: ScenarioType = ScenarioType.STEADY_STATE,
        duration: float = 86400.0,
        **kwargs
    ) -> ScenarioConfig:
        """
        加载测试场景

        Args:
            scenario_type: 场景类型
            duration: 持续时间 (s)
            **kwargs: 场景参数

        Returns:
            scenario: 场景配置
        """
        scenario = self.scenario_generator.create_scenario(
            scenario_type=scenario_type,
            duration=duration,
            **kwargs
        )

        # 设置控制目标
        self.evaluator.set_targets(
            level_targets={f"SEG_{i:03d}": 4.0 - i * 0.01 for i in range(self.config.num_segments)}
        )

        logger.info(f"Scenario loaded: {scenario.scenario_id}")
        return scenario

    def reset(
        self,
        initial_level: float = 4.0,
        initial_flow: float = 300.0
    ):
        """
        重置系统状态

        Args:
            initial_level: 初始水位 (m)
            initial_flow: 初始流量 (m³/s)
        """
        self.current_time = 0.0
        self.step_count = 0

        # 重置各模块
        self.reduced_order_engine.reset(initial_level, initial_flow)

        for model in self.high_fidelity_models.values():
            model.reset(initial_level, initial_flow)

        self.boundary_assimilator.reset()
        self.controller_orchestrator.reset()
        self.evaluator.reset()

        self.results = None

        logger.info("Framework reset")

    def step(
        self,
        upstream_flow: float,
        lateral_flows: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        执行一个仿真步

        Args:
            upstream_flow: 上游入流 (m³/s)
            lateral_flows: 侧向流量 [num_segments]

        Returns:
            result: 步进结果
        """
        dt = self.config.dt_reduced

        # 1. 获取当前闸门开度
        gate_openings = self.controller_orchestrator.get_gate_openings()

        # 2. 运行降阶模型
        ro_result = self.reduced_order_engine.step(
            upstream_flow=upstream_flow,
            gate_openings=gate_openings,
            lateral_flows=lateral_flows,
        )

        # 3. 获取降阶模型状态
        ro_states = self.reduced_order_engine.to_segment_states()
        ro_boundaries = self.reduced_order_engine.to_boundary_conditions()

        # 4. 运行高保真模型 (如果是混合模式)
        hf_states = {}
        if self.config.simulation_mode == SimulationMode.HYBRID:
            hf_states = self._run_high_fidelity_models(
                ro_states, ro_boundaries, upstream_flow
            )

        # 5. 接口同化 (融合降阶与高保真)
        fused_boundaries = {}
        for bnd_id, ro_bnd in ro_boundaries.items():
            # 获取相邻高保真模型状态
            upstream_seg = ro_bnd.upstream_segment_id
            downstream_seg = ro_bnd.downstream_segment_id

            hf_state = None
            if upstream_seg in hf_states:
                hf_state = {
                    "level": hf_states[upstream_seg].downstream_level,
                    "flow": hf_states[upstream_seg].downstream_flow,
                }
            elif downstream_seg in hf_states:
                hf_state = {
                    "level": hf_states[downstream_seg].upstream_level,
                    "flow": hf_states[downstream_seg].upstream_flow,
                }

            # 同化
            assimilated = self.boundary_assimilator.assimilate(
                boundary_id=bnd_id,
                idz_state={
                    "level": ro_bnd.water_level,
                    "flow": ro_bnd.flow_rate,
                },
                fine_state=hf_state,
                gate_action=ro_bnd.flow_rate,  # gate action derived from boundary flow
            )

            fused_boundaries[bnd_id] = self.boundary_assimilator.get_fused_boundary(bnd_id)

        # 6. 合并状态
        final_states = ro_states.copy()
        final_states.update(hf_states)

        # 7. 计算控制动作
        gate_openings_dict = self.controller_orchestrator.step(
            states=final_states,
            boundaries=fused_boundaries,
            dt=dt,
        )

        # 8. 记录状态并评估
        self.evaluator.record_state(final_states, fused_boundaries, self.current_time)

        # 周期性评估
        kpi = None
        eval_every = max(1, int(self.config.sync_interval / dt)) if dt > 0 else 1
        if self.step_count % eval_every == 0:
            kpi = self.evaluator.evaluate(
                final_states, fused_boundaries, self.current_time
            )

        # 9. 更新时间
        self.current_time += dt
        self.step_count += 1

        return {
            "time": self.current_time,
            "step": self.step_count,
            "states": final_states,
            "boundaries": fused_boundaries,
            "gate_openings": gate_openings_dict,
            "kpi": kpi,
            "mass_balance": ro_result.get("mass_balance", {}),
        }

    def _run_high_fidelity_models(
        self,
        ro_states: Dict[str, SegmentState],
        ro_boundaries: Dict[str, BoundaryCondition],
        upstream_flow: float,
    ) -> Dict[str, SegmentState]:
        """运行高保真模型"""
        hf_states = {}

        # 计算高保真模型需要的子步数
        num_sub_steps = int(self.config.dt_reduced / self.config.dt_high_fidelity)

        for segment_id, model in self.high_fidelity_models.items():
            seg_idx = int(segment_id.split("_")[1])

            # 设置边界条件
            if seg_idx == 0:
                model.set_boundary_conditions(
                    upstream={"type": "dirichlet_flow", "value": upstream_flow}
                )
            else:
                prev_bnd_id = f"BND_{seg_idx-1:03d}_{seg_idx:03d}"
                if prev_bnd_id in ro_boundaries:
                    model.set_boundary_conditions(
                        upstream={
                            "type": "dirichlet_flow",
                            "value": ro_boundaries[prev_bnd_id].flow_rate
                        }
                    )

            # 运行多个子步
            for _ in range(num_sub_steps):
                model.step()

            # 获取状态
            hf_states[segment_id] = model.get_state()

        return hf_states

    def run_simulation(
        self,
        scenario: ScenarioConfig,
        progress_callback: Optional[callable] = None,
    ) -> SimulationResult:
        """
        运行完整仿真

        Args:
            scenario: 场景配置
            progress_callback: 进度回调函数

        Returns:
            result: 仿真结果
        """
        logger.info(f"Starting simulation: {scenario.scenario_id}")

        # 重置
        self.reset()
        self.is_running = True

        # 计算步数
        num_steps = int(scenario.duration / self.config.dt_reduced)

        # 状态历史
        state_history = []
        boundary_history = []
        kpi_history = []

        try:
            for step in range(num_steps):
                # 获取当前时间的边界流量
                if scenario.upstream_boundary_profile is not None:
                    upstream_flow = scenario.upstream_boundary_profile[
                        min(step, len(scenario.upstream_boundary_profile) - 1)
                    ]
                else:
                    upstream_flow = 300.0

                # 获取侧向流量
                lateral_flows = None
                if scenario.lateral_inflow_profiles:
                    lateral_flows = np.zeros(self.config.num_segments)
                    for seg_id, profile in scenario.lateral_inflow_profiles.items():
                        seg_idx = int(seg_id.split("_")[1])
                        lateral_flows[seg_idx] = profile[min(step, len(profile) - 1)]

                # 执行步进
                result = self.step(upstream_flow, lateral_flows)

                # 记录历史 (降采样)
                if step % 10 == 0:
                    state_history.append(result["states"])
                    boundary_history.append(result["boundaries"])
                    if result["kpi"]:
                        kpi_history.append(result["kpi"])

                # 进度回调
                if progress_callback and step % 100 == 0:
                    progress = (step + 1) / num_steps
                    progress_callback(progress, result)

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
        finally:
            self.is_running = False

        # 生成最终KPI
        final_kpi = self.evaluator.to_kpi_metrics()

        # 判定通过/失败
        is_pass, violations = final_kpi.check_thresholds() if final_kpi else (False, ["No KPI"])

        # 构建结果
        self.results = SimulationResult(
            scenario_id=scenario.scenario_id,
            start_time=datetime.now() - timedelta(seconds=scenario.duration),
            end_time=datetime.now(),
            state_history=state_history,
            boundary_history=boundary_history,
            kpi_history=kpi_history,
            final_kpi=final_kpi,
            passed=is_pass,
            failure_reasons=violations,
        )

        logger.info(
            f"Simulation completed: passed={is_pass}, "
            f"score={final_kpi.overall_score if final_kpi else 0:.3f}"
        )

        return self.results

    def run_ensemble_simulation(
        self,
        base_scenario: ScenarioConfig,
        ensemble_size: int = 10,
    ) -> List[SimulationResult]:
        """
        运行集合仿真 (参数不确定性测试)

        Args:
            base_scenario: 基础场景
            ensemble_size: 集合大小

        Returns:
            results: 仿真结果列表
        """
        logger.info(f"Starting ensemble simulation: size={ensemble_size}")

        # 生成参数集合
        scenarios = self.scenario_generator.create_ensemble_scenarios(
            base_scenario, ensemble_size
        )

        results = []
        for i, scenario in enumerate(scenarios):
            logger.info(f"Running ensemble member {i+1}/{ensemble_size}")
            result = self.run_simulation(scenario)
            results.append(result)

        # 统计
        pass_count = sum(1 for r in results if r.passed)
        logger.info(f"Ensemble completed: {pass_count}/{ensemble_size} passed")

        return results

    def generate_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        # 评估器报告
        eval_report = self.evaluator.generate_report()

        # 同化器统计
        assim_stats = self.boundary_assimilator.get_summary_statistics()

        # 控制器统计
        ctrl_stats = self.controller_orchestrator.get_command_statistics()

        # 降阶模型性能
        ro_perf = self.reduced_order_engine.get_performance_metrics()

        # 高保真模型性能
        hf_perfs = {}
        for seg_id, model in self.high_fidelity_models.items():
            hf_perfs[seg_id] = model.get_performance_metrics()

        return {
            "framework_config": {
                "simulation_mode": self.config.simulation_mode.value,
                "num_segments": self.config.num_segments,
                "num_high_fidelity": len(self.high_fidelity_models),
                "fusion_mode": self.config.fusion_mode.value,
            },
            "simulation_summary": {
                "total_time": self.current_time,
                "total_steps": self.step_count,
                "passed": self.results.passed if self.results else None,
            },
            "evaluation_report": eval_report,
            "assimilation_statistics": assim_stats,
            "controller_statistics": ctrl_stats,
            "reduced_order_performance": ro_perf,
            "high_fidelity_performance": hf_perfs,
        }

    def export_results(self, filepath: str):
        """导出结果到JSON文件"""
        report = self.generate_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Results exported to {filepath}")
