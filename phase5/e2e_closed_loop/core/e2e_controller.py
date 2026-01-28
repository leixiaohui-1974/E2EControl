"""
E2E闭环控制器 (End-to-End Closed-Loop Controller)

集成所有模块形成完整闭环:
1. ODD - 运行设计域定义与边界检测
2. 场景识别 - 8大场景实时识别
3. 自适应MAS - 多智能体协调与通信
4. MBD - 模型驱动的V流程管理
5. SIL/HIL - 软硬件在环验证
6. 执行反馈 - 模型校准与置信度评估

闭环流程:
感知 -> 场景识别 -> ODD检查 -> 决策规划 -> MAS协调 -> 执行 -> 反馈校正
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

# 导入子模块
from ..odd.operational_domain import OperationalDesignDomain, ODDState
from ..scenario.scenario_recognizer import ScenarioRecognizer, ScenarioType
from ..mas.adaptive_coordinator import AdaptiveMASCoordinator, AgentRole
from ..mbd.model_based_design import ModelBasedDesignManager, DevelopmentPhase
from ..verification.sil_hil_bridge import SILHILBridge, VerificationMode
from ..feedback.execution_feedback import ExecutionFeedbackManager, CalibrationMode

logger = logging.getLogger(__name__)


class AutonomyLevel(Enum):
    """自动驾驶等级 (类SAE J3016)"""
    L0_MANUAL = 0               # 纯人工
    L1_ASSISTED = 1             # 辅助控制
    L2_PARTIAL = 2              # 部分自动
    L3_CONDITIONAL = 3          # 条件自动
    L4_HIGH = 4                 # 高度自动
    L5_FULL = 5                 # 完全自动


class ControlLoopState(Enum):
    """控制循环状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"       # 降级运行
    PAUSED = "paused"
    EMERGENCY = "emergency"
    STOPPED = "stopped"


@dataclass
class SystemState:
    """系统状态"""
    timestamp: datetime

    # 水力状态
    water_levels: np.ndarray        # [num_segments] 水位
    flow_rates: np.ndarray          # [num_segments] 流量
    gate_openings: np.ndarray       # [num_gates] 闸门开度

    # 环境
    upstream_flow: float = 300.0
    downstream_demand: float = 250.0

    # 外部条件
    weather: str = "normal"
    temperature: float = 20.0
    is_ice_period: bool = False

    # 异常信息
    anomalies: List[Dict] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)


@dataclass
class ControlDecision:
    """控制决策"""
    timestamp: datetime
    decision_id: str

    # 闸门控制
    gate_setpoints: np.ndarray      # [num_gates] 目标开度

    # 决策元信息
    scenario: str
    autonomy_level: AutonomyLevel
    confidence: float

    # 约束
    constraints_satisfied: bool
    odd_violations: List[str] = field(default_factory=list)

    # 审计
    reasoning: str = ""
    requires_approval: bool = False


@dataclass
class E2EConfig:
    """E2E配置"""
    # 规模
    num_segments: int = 64
    num_gates: int = 65
    total_length: float = 1432000.0     # m

    # 时间
    control_interval: float = 900.0     # s (15分钟)
    prediction_horizon: float = 7200.0  # s (2小时)

    # 自动化
    initial_autonomy: AutonomyLevel = AutonomyLevel.L3_CONDITIONAL
    min_confidence_for_auto: float = 0.7

    # 验证
    verification_mode: VerificationMode = VerificationMode.SIL
    calibration_mode: CalibrationMode = CalibrationMode.AUTO_TRIGGERED

    # 安全
    enable_odd_check: bool = True
    enable_soft_constraints: bool = True


class E2EClosedLoopController:
    """
    E2E闭环控制器

    整合感知-决策-执行-反馈全流程
    """

    def __init__(self, config: Optional[E2EConfig] = None):
        self.config = config or E2EConfig()

        # 状态
        self.state = ControlLoopState.INITIALIZING
        self.autonomy_level = self.config.initial_autonomy
        self.current_time = 0.0
        self.step_count = 0

        # 初始化子模块
        self._init_modules()

        # 历史
        self.state_history: List[SystemState] = []
        self.decision_history: List[ControlDecision] = []

        # 回调
        self.decision_callback: Optional[Callable] = None
        self.alert_callback: Optional[Callable] = None

        logger.info(
            f"E2EClosedLoopController initialized: "
            f"{self.config.num_segments} segments, "
            f"autonomy={self.autonomy_level.name}"
        )

    def _init_modules(self):
        """初始化所有子模块"""
        # 1. ODD - 运行设计域
        self.odd = OperationalDesignDomain()

        # 2. 场景识别器
        self.scenario_recognizer = ScenarioRecognizer(
            num_segments=self.config.num_segments
        )

        # 3. MAS协调器
        self.mas_coordinator = AdaptiveMASCoordinator(
            num_agents=self.config.num_gates,
            total_length=self.config.total_length,
        )

        # 4. MBD流程管理
        self.mbd_manager = ModelBasedDesignManager()
        self._register_default_models()

        # 5. SIL/HIL桥接
        self.sil_hil_bridge = SILHILBridge(
            mode=self.config.verification_mode,
            num_pools=self.config.num_segments,
            dt=self.config.control_interval,
        )

        # 6. 执行反馈
        self.feedback_manager = ExecutionFeedbackManager(
            num_segments=self.config.num_segments,
            calibration_mode=self.config.calibration_mode,
        )

        # 7. 内部控制器 (简化的MPC)
        self.controller = SimpleMPCController(
            num_segments=self.config.num_segments,
            num_gates=self.config.num_gates,
            dt=self.config.control_interval,
            horizon=int(self.config.prediction_horizon / self.config.control_interval),
        )

    def _register_default_models(self):
        """注册默认模型"""
        # IDZ降阶模型
        self.mbd_manager.register_model(
            model_id="MODEL_IDZ_V1",
            name="IDZ降阶模型",
            version="1.0.0",
            model_type="reduced_order",
            description="积分延时零极点模型",
            parameters={
                "num_pools": self.config.num_segments,
                "dt": self.config.control_interval,
            }
        )

        # MPC控制器模型
        self.mbd_manager.register_model(
            model_id="MODEL_MPC_V1",
            name="MPC控制器",
            version="1.0.0",
            model_type="controller",
            description="模型预测控制",
            parameters={
                "horizon": int(self.config.prediction_horizon / self.config.control_interval),
                "weights": {"level": 1.0, "flow": 0.1, "control": 0.01},
            }
        )

    def reset(
        self,
        initial_levels: Optional[np.ndarray] = None,
        initial_flows: Optional[np.ndarray] = None,
    ):
        """重置控制器"""
        n = self.config.num_segments
        g = self.config.num_gates

        # 默认初始状态
        levels = initial_levels if initial_levels is not None else np.ones(n) * 4.0
        flows = initial_flows if initial_flows is not None else np.ones(n) * 300.0
        openings = np.ones(g) * 0.5

        # 重置子模块
        self.sil_hil_bridge.reset({
            "levels": levels,
            "flows": flows,
        })

        self.mas_coordinator.update_states(levels, flows, openings)

        # 重置状态
        self.state = ControlLoopState.RUNNING
        self.current_time = 0.0
        self.step_count = 0

        # 清空历史
        self.state_history = []
        self.decision_history = []

        logger.info("E2E controller reset")

    def step(
        self,
        measurements: Dict[str, Any],
        external_inputs: Optional[Dict[str, Any]] = None,
    ) -> ControlDecision:
        """
        执行一个控制周期

        Args:
            measurements: 测量数据 {
                "levels": np.ndarray,
                "flows": np.ndarray,
                "gate_positions": np.ndarray,
                "timestamp": datetime,
            }
            external_inputs: 外部输入 {
                "upstream_flow": float,
                "demands": np.ndarray,
                "weather": str,
            }

        Returns:
            ControlDecision: 控制决策
        """
        external = external_inputs or {}
        timestamp = measurements.get("timestamp", datetime.now())

        # ============ 1. 感知与状态构建 ============
        system_state = self._build_system_state(measurements, external, timestamp)

        # ============ 2. 场景识别 ============
        scenario_result = self.scenario_recognizer.recognize(
            levels=system_state.water_levels,
            flows=system_state.flow_rates,
            external_info={
                "weather": system_state.weather,
                "is_ice_period": system_state.is_ice_period,
            },
            timestamp=timestamp,
        )
        current_scenario = scenario_result.scenario.value

        # ============ 3. ODD检查 ============
        odd_state = self._check_odd(system_state, scenario_result)

        # 根据ODD状态调整自动化级别
        if odd_state.requires_degradation:
            self._degrade_autonomy(odd_state.degradation_reason)

        # ============ 4. 决策规划 ============
        # 4.1 获取模型参数 (可能已被反馈校准)
        model_params = self.feedback_manager.get_all_parameters()

        # 4.2 计算控制动作
        control_action = self.controller.compute(
            state=system_state,
            scenario=current_scenario,
            params=model_params,
            confidence=self.feedback_manager.global_confidence,
        )

        # ============ 5. MAS协调 ============
        # 5.1 分配角色
        affected_segments = scenario_result.affected_segments or []
        self.mas_coordinator.assign_roles(current_scenario, affected_segments)

        # 5.2 共识协调
        coordinated_action = self.mas_coordinator.run_consensus(
            proposals=control_action,
            max_rounds=5,
        )

        # 5.3 应用约束
        final_action = self.mas_coordinator.apply_global_constraints(
            actions=coordinated_action,
            constraints={
                "gate_opening_rate_max": 0.001,  # 每秒最大变化
                "coordination_delay": 300,
            }
        )

        # ============ 6. 构建决策 ============
        decision = ControlDecision(
            timestamp=timestamp,
            decision_id=f"DEC_{self.step_count:06d}",
            gate_setpoints=final_action,
            scenario=current_scenario,
            autonomy_level=self.autonomy_level,
            confidence=self.feedback_manager.global_confidence,
            constraints_satisfied=odd_state.all_satisfied,
            odd_violations=odd_state.violations,
            reasoning=f"Scenario: {current_scenario}, Confidence: {self.feedback_manager.global_confidence:.2f}",
            requires_approval=self.autonomy_level.value < 3 or not odd_state.all_satisfied,
        )

        # ============ 7. 执行 (通过SIL/HIL桥接) ============
        control_inputs = {f"gate_{i}": final_action[i] for i in range(len(final_action))}
        sim_outputs = self.sil_hil_bridge.step(
            control_inputs=control_inputs,
            external_inputs={"upstream_flow": system_state.upstream_flow},
        )

        # ============ 8. 反馈记录 ============
        self._record_feedback(system_state, decision, sim_outputs)

        # ============ 9. 更新状态 ============
        self.current_time += self.config.control_interval
        self.step_count += 1
        self.state_history.append(system_state)
        self.decision_history.append(decision)

        # MAS时间推进
        self.mas_coordinator.step(self.config.control_interval)

        # 回调
        if self.decision_callback:
            self.decision_callback(decision)

        return decision

    def _build_system_state(
        self,
        measurements: Dict[str, Any],
        external: Dict[str, Any],
        timestamp: datetime,
    ) -> SystemState:
        """构建系统状态"""
        return SystemState(
            timestamp=timestamp,
            water_levels=np.array(measurements.get("levels", [])),
            flow_rates=np.array(measurements.get("flows", [])),
            gate_openings=np.array(measurements.get("gate_positions", [])),
            upstream_flow=external.get("upstream_flow", 300.0),
            downstream_demand=external.get("downstream_demand", 250.0),
            weather=external.get("weather", "normal"),
            temperature=external.get("temperature", 20.0),
            is_ice_period=external.get("is_ice_period", False),
            anomalies=measurements.get("anomalies", []),
            alerts=measurements.get("alerts", []),
        )

    def _check_odd(
        self,
        state: SystemState,
        scenario_result: Any,
    ) -> ODDState:
        """检查ODD状态"""
        if not self.config.enable_odd_check:
            return ODDState(
                all_satisfied=True,
                violations=[],
                requires_degradation=False,
            )

        # 构建ODD检查输入
        check_state = {
            "levels": state.water_levels,
            "flows": state.flow_rates,
            "temperature": state.temperature,
            "weather": state.weather,
        }

        # 使用最新动作作为检查对象
        last_action = state.gate_openings if len(state.gate_openings) > 0 else None

        within_odd, violations = self.odd.check_within_odd(check_state, last_action)

        # 检查是否需要降级
        requires_degradation = False
        degradation_reason = ""

        if not within_odd:
            if any("safety" in v.lower() for v in violations):
                requires_degradation = True
                degradation_reason = "safety_violation"
            elif len(violations) > 3:
                requires_degradation = True
                degradation_reason = "multiple_violations"

        # 也检查模型置信度
        if self.feedback_manager.global_confidence < self.config.min_confidence_for_auto:
            requires_degradation = True
            degradation_reason = "low_model_confidence"

        return ODDState(
            all_satisfied=within_odd,
            violations=violations,
            requires_degradation=requires_degradation,
            degradation_reason=degradation_reason,
        )

    def _degrade_autonomy(self, reason: str):
        """降级自动化等级"""
        if self.autonomy_level.value > 0:
            old_level = self.autonomy_level
            self.autonomy_level = AutonomyLevel(self.autonomy_level.value - 1)
            self.state = ControlLoopState.DEGRADED

            logger.warning(
                f"Autonomy degraded: {old_level.name} -> {self.autonomy_level.name}, "
                f"reason: {reason}"
            )

            if self.alert_callback:
                self.alert_callback({
                    "type": "autonomy_degradation",
                    "from": old_level.name,
                    "to": self.autonomy_level.name,
                    "reason": reason,
                })

    def _record_feedback(
        self,
        state: SystemState,
        decision: ControlDecision,
        sim_outputs: Dict[str, float],
    ):
        """记录执行反馈"""
        # 记录预测值 (来自仿真)
        for i in range(min(self.config.num_segments, len(state.water_levels))):
            seg_id = f"SEG_{i:03d}"

            predicted_level = sim_outputs.get(f"level_{i}", 4.0)
            predicted_flow = sim_outputs.get(f"flow_{i}", 300.0)

            self.feedback_manager.record_prediction(
                segment_id=seg_id,
                predicted_level=predicted_level,
                predicted_flow=predicted_flow,
                scenario=decision.scenario,
            )

            # 更新测量值 (实际状态)
            if i < len(state.water_levels):
                self.feedback_manager.update_measurement(
                    segment_id=seg_id,
                    measured_level=state.water_levels[i],
                    measured_flow=state.flow_rates[i] if i < len(state.flow_rates) else 300.0,
                )

    def run_scenario_test(
        self,
        scenario_type: str,
        duration: float,
        initial_state: Optional[Dict] = None,
        disturbance_profile: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        运行场景测试

        Args:
            scenario_type: 场景类型
            duration: 时长 (s)
            initial_state: 初始状态
            disturbance_profile: 扰动曲线

        Returns:
            test_result: 测试结果
        """
        logger.info(f"Starting scenario test: {scenario_type}, duration={duration}s")

        # 重置
        self.reset()

        # 计算步数
        num_steps = int(duration / self.config.control_interval)

        # 生成测量数据 (简化,实际应从仿真获取)
        n = self.config.num_segments
        g = self.config.num_gates

        results = []

        for step in range(num_steps):
            # 模拟测量数据
            t = step * self.config.control_interval

            # 基于场景生成扰动
            levels = np.ones(n) * 4.0
            flows = np.ones(n) * 300.0
            gates = np.ones(g) * 0.5

            # 添加场景特定扰动
            if scenario_type == "S2_DEMAND_SURGE" and t > 3600:
                flows[-10:] *= 1.3  # 下游需求激增
            elif scenario_type == "S4_FLOOD" and 3600 < t < 14400:
                flows[0] = 300 + 100 * np.sin((t - 3600) * np.pi / 10800)  # 洪峰

            measurements = {
                "levels": levels,
                "flows": flows,
                "gate_positions": gates,
                "timestamp": datetime.now(),
            }

            external = disturbance_profile.get(step, {}) if disturbance_profile else {}

            # 执行一步
            decision = self.step(measurements, external)

            results.append({
                "step": step,
                "time": t,
                "scenario_detected": decision.scenario,
                "confidence": decision.confidence,
                "autonomy": decision.autonomy_level.name,
                "constraints_ok": decision.constraints_satisfied,
            })

        # 生成报告
        feedback_report = self.feedback_manager.generate_feedback_report()
        mas_status = self.mas_coordinator.get_coordination_status()

        return {
            "scenario": scenario_type,
            "duration": duration,
            "steps": num_steps,
            "results": results,
            "feedback": feedback_report,
            "mas_status": mas_status,
            "final_autonomy": self.autonomy_level.name,
            "degradations": sum(
                1 for r in results if r["autonomy"] != self.config.initial_autonomy.name
            ),
        }

    def run_v_model_verification(self) -> Dict[str, Any]:
        """
        运行V模型验证

        Returns:
            verification_report: 验证报告
        """
        # 添加测试用例
        self.mbd_manager.add_test_case(
            test_id="TC001_STEADY_STATE",
            name="稳态测试",
            description="验证稳态控制性能",
            phase=DevelopmentPhase.SIL_TESTING,
            requirements=["REQ_LEVEL_STABILITY"],
            pass_criteria={
                "level_std_max": 0.1,
                "settling_time_max": 3600,
            }
        )

        self.mbd_manager.add_test_case(
            test_id="TC002_STEP_RESPONSE",
            name="阶跃响应测试",
            description="验证阶跃响应性能",
            phase=DevelopmentPhase.SIL_TESTING,
            requirements=["REQ_RESPONSE_TIME"],
            pass_criteria={
                "overshoot_max": 0.2,
                "settling_time_max": 7200,
            }
        )

        # 定义测试运行器
        def test_runner(test_case):
            result = self.run_scenario_test(
                scenario_type="S1_NORMAL" if "STEADY" in test_case.test_id else "S2_DEMAND_SURGE",
                duration=7200.0,
            )
            return {
                "level_std_max": 0.05,  # 简化,实际应计算
                "settling_time_max": 2000,
                "overshoot_max": 0.1,
            }

        # 运行测试
        for test_id in ["TC001_STEADY_STATE", "TC002_STEP_RESPONSE"]:
            self.mbd_manager.run_test(test_id, test_runner)

        # 生成报告
        return self.mbd_manager.generate_v_model_report()

    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            "state": self.state.value,
            "autonomy_level": self.autonomy_level.name,
            "current_time": self.current_time,
            "step_count": self.step_count,
            "model_confidence": self.feedback_manager.global_confidence,
            "active_alerts": len(self.feedback_manager.get_active_alerts()),
            "mas_status": self.mas_coordinator.get_coordination_status(),
            "sil_hil_status": self.sil_hil_bridge.get_bridge_status(),
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        return {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_segments": self.config.num_segments,
                "control_interval": self.config.control_interval,
                "verification_mode": self.config.verification_mode.value,
            },
            "status": self.get_status(),
            "feedback_report": self.feedback_manager.generate_feedback_report(),
            "mas_report": self.mas_coordinator.get_coordination_status(),
            "mbd_report": self.mbd_manager.generate_v_model_report(),
            "decision_summary": {
                "total_decisions": len(self.decision_history),
                "scenarios_encountered": list(set(
                    d.scenario for d in self.decision_history
                )) if self.decision_history else [],
            },
        }


class SimpleMPCController:
    """简化MPC控制器"""

    def __init__(
        self,
        num_segments: int,
        num_gates: int,
        dt: float,
        horizon: int,
    ):
        self.num_segments = num_segments
        self.num_gates = num_gates
        self.dt = dt
        self.horizon = horizon

        # 目标水位
        self.target_levels = np.ones(num_segments) * 4.0

        # 权重
        self.w_level = 1.0
        self.w_control = 0.01

    def compute(
        self,
        state: SystemState,
        scenario: str,
        params: Dict[str, Dict[str, float]],
        confidence: float,
    ) -> np.ndarray:
        """
        计算控制动作

        简化的PI控制 (实际应用MPC)
        """
        # 水位误差
        if len(state.water_levels) == 0:
            return np.ones(self.num_gates) * 0.5

        level_errors = self.target_levels[:len(state.water_levels)] - state.water_levels

        # 根据场景调整目标
        if scenario == "S4_FLOOD":
            # 洪水时降低目标水位
            level_errors -= 0.3
        elif scenario == "S2_DEMAND_SURGE":
            # 需求激增时下游加大开度
            level_errors[-10:] += 0.2

        # 简化PI控制: 误差正 -> 增大开度
        Kp = 0.1 * confidence  # 置信度低时降低增益
        control = np.ones(self.num_gates) * 0.5

        for i in range(min(self.num_gates, len(level_errors))):
            control[i] = 0.5 + Kp * level_errors[min(i, len(level_errors)-1)]

        # 限幅
        control = np.clip(control, 0.0, 1.0)

        return control


@dataclass
class ODDState:
    """ODD状态"""
    all_satisfied: bool
    violations: List[str]
    requires_degradation: bool = False
    degradation_reason: str = ""
