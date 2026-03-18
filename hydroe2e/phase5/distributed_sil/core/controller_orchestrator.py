"""
控制器编排器 (Controller Orchestrator)

功能:
1. 多层级控制器调度 - L1/L2/L3/L4协调
2. 控制指令融合 - 多源指令优先级处理
3. 时钟同步 - 不同时间尺度协调
4. 闭环测试支持 - 与SIL植物模型交互

设计原则:
- 分层分布式控制架构
- 控制器与模型解耦
- 支持多种控制算法接入
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod

from ..interfaces.data_types import (
    SegmentState, BoundaryCondition, ControlCommand,
    GateState, GateType
)

logger = logging.getLogger(__name__)


class ControlLevel(Enum):
    """控制层级"""
    L1_LOCAL = "L1"        # 本地控制 (PID)
    L2_COORDINATED = "L2"  # 协调控制 (分布式MPC)
    L3_SUPERVISORY = "L3"  # 监督控制 (规则/优化)
    L4_STRATEGIC = "L4"    # 战略控制 (AI决策)


class ControlMode(Enum):
    """控制模式"""
    MANUAL = "manual"              # 手动
    AUTO_LOCAL = "auto_local"      # 本地自动
    AUTO_COORDINATED = "auto_coordinated"  # 协调自动
    EMERGENCY = "emergency"        # 应急


@dataclass
class ControllerConfig:
    """控制器配置"""
    controller_id: str
    level: ControlLevel
    gate_ids: List[str]            # 管辖的闸门

    # 时间参数
    control_interval: float = 900.0  # s (控制周期)
    prediction_horizon: int = 12     # 预测步数

    # 约束参数
    max_gate_rate: float = 0.001     # 最大开度变化率 (/s)
    min_level: float = 1.5           # m
    max_level: float = 5.5           # m

    # 目标参数
    target_levels: Dict[str, float] = field(default_factory=dict)
    target_flows: Dict[str, float] = field(default_factory=dict)

    # 权重
    level_weight: float = 1.0
    flow_weight: float = 0.5
    action_weight: float = 0.1


class BaseController(ABC):
    """控制器基类"""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.is_active = False
        self.last_action_time = None

    @abstractmethod
    def compute_action(
        self,
        states: Dict[str, SegmentState],
        boundaries: Dict[str, BoundaryCondition],
        current_time: float,
    ) -> List[ControlCommand]:
        """计算控制动作"""
        pass

    def reset(self):
        """重置控制器"""
        self.is_active = False
        self.last_action_time = None


class PIDController(BaseController):
    """PID本地控制器 (L1)"""

    def __init__(self, config: ControllerConfig):
        super().__init__(config)
        self.kp = 0.5
        self.ki = 0.01
        self.kd = 0.1

        self.integral_error: Dict[str, float] = {}
        self.last_error: Dict[str, float] = {}

    def compute_action(
        self,
        states: Dict[str, SegmentState],
        boundaries: Dict[str, BoundaryCondition],
        current_time: float,
    ) -> List[ControlCommand]:
        commands = []

        for gate_id in self.config.gate_ids:
            # 获取对应渠段状态
            segment_id = self._gate_to_segment(gate_id)
            if segment_id not in states:
                continue

            state = states[segment_id]
            target_level = self.config.target_levels.get(segment_id, 4.0)

            # 计算误差
            error = target_level - state.mean_level

            # 积分项
            if segment_id not in self.integral_error:
                self.integral_error[segment_id] = 0.0
            self.integral_error[segment_id] += error * self.config.control_interval
            self.integral_error[segment_id] = np.clip(
                self.integral_error[segment_id], -10, 10
            )

            # 微分项
            if segment_id not in self.last_error:
                self.last_error[segment_id] = error
            d_error = (error - self.last_error[segment_id]) / self.config.control_interval
            self.last_error[segment_id] = error

            # PID输出
            output = (
                self.kp * error +
                self.ki * self.integral_error[segment_id] +
                self.kd * d_error
            )

            # 转换为闸门开度变化
            delta_opening = np.clip(
                output * 0.01,
                -self.config.max_gate_rate * self.config.control_interval,
                self.config.max_gate_rate * self.config.control_interval
            )

            commands.append(ControlCommand(
                command_id=f"CMD_{gate_id}_{int(current_time)}",
                gate_id=gate_id,
                timestamp=datetime.now(),
                target_opening=None,  # 增量控制
                control_mode="opening",
                source_controller=self.config.controller_id,
                source_level=self.config.level.value,
            ))

        return commands

    def _gate_to_segment(self, gate_id: str) -> str:
        """闸门ID转渠段ID"""
        if gate_id.startswith("GATE_"):
            idx = gate_id.split("_")[1]
            return f"SEG_{idx}"
        return gate_id


class MPCController(BaseController):
    """MPC协调控制器 (L2)"""

    def __init__(self, config: ControllerConfig):
        super().__init__(config)
        self.horizon = config.prediction_horizon
        self.Q = np.eye(len(config.gate_ids)) * config.level_weight
        self.R = np.eye(len(config.gate_ids)) * config.action_weight

    def compute_action(
        self,
        states: Dict[str, SegmentState],
        boundaries: Dict[str, BoundaryCondition],
        current_time: float,
    ) -> List[ControlCommand]:
        """
        MPC优化求解

        简化实现: 使用二次规划求解最优控制序列
        """
        commands = []
        n_gates = len(self.config.gate_ids)

        # 构建当前状态向量
        x0 = np.zeros(n_gates)
        x_target = np.zeros(n_gates)

        for i, gate_id in enumerate(self.config.gate_ids):
            segment_id = self._gate_to_segment(gate_id)
            if segment_id in states:
                x0[i] = states[segment_id].mean_level
                x_target[i] = self.config.target_levels.get(segment_id, 4.0)

        # 简化MPC: 单步最优
        # u* = argmin ||x - x_target||_Q + ||u||_R
        # 简单比例控制近似
        error = x_target - x0
        u_opt = 0.1 * error  # 简化增益

        # 生成控制指令
        for i, gate_id in enumerate(self.config.gate_ids):
            if abs(u_opt[i]) > 0.001:
                commands.append(ControlCommand(
                    command_id=f"MPC_{gate_id}_{int(current_time)}",
                    gate_id=gate_id,
                    timestamp=datetime.now(),
                    target_opening=None,
                    target_level=x_target[i],
                    control_mode="level",
                    source_controller=self.config.controller_id,
                    source_level=self.config.level.value,
                ))

        return commands

    def _gate_to_segment(self, gate_id: str) -> str:
        if gate_id.startswith("GATE_"):
            idx = gate_id.split("_")[1]
            return f"SEG_{idx}"
        return gate_id


class ControllerOrchestrator:
    """
    控制器编排器

    负责:
    1. 多层级控制器的注册和调度
    2. 控制指令的融合和冲突解决
    3. 与SIL植物模型的闭环交互
    """

    def __init__(
        self,
        num_gates: int = 64,
        default_control_interval: float = 900.0,
    ):
        """
        初始化编排器

        Args:
            num_gates: 闸门数量
            default_control_interval: 默认控制周期 (s)
        """
        self.num_gates = num_gates
        self.default_control_interval = default_control_interval

        # 控制器注册表
        self.controllers: Dict[str, BaseController] = {}

        # 当前闸门状态
        self.gate_states: Dict[str, GateState] = {}

        # 控制指令队列
        self.pending_commands: List[ControlCommand] = []

        # 执行历史
        self.command_history: List[ControlCommand] = []

        # 时钟
        self.current_time = 0.0

        # 控制模式
        self.mode = ControlMode.AUTO_COORDINATED

        # 初始化闸门状态
        self._init_gates()

        logger.info(f"ControllerOrchestrator initialized: {num_gates} gates")

    def _init_gates(self):
        """初始化闸门状态"""
        for i in range(self.num_gates):
            gate_id = f"GATE_{i:03d}"
            self.gate_states[gate_id] = GateState(
                gate_id=gate_id,
                gate_type=GateType.CONTROL_GATE,
                opening=0.5,
                flow_rate=300.0,
                upstream_level=4.0,
                downstream_level=3.9,
            )

    def register_controller(
        self,
        controller_id: str,
        controller: BaseController
    ):
        """注册控制器"""
        self.controllers[controller_id] = controller
        controller.config.controller_id = controller_id
        logger.info(f"Registered controller: {controller_id}")

    def create_default_controllers(self):
        """创建默认控制器配置"""
        # L1: 每个闸门一个本地PID
        for i in range(self.num_gates):
            gate_id = f"GATE_{i:03d}"
            config = ControllerConfig(
                controller_id=f"L1_PID_{i:03d}",
                level=ControlLevel.L1_LOCAL,
                gate_ids=[gate_id],
                control_interval=60.0,  # L1较快
            )
            self.register_controller(config.controller_id, PIDController(config))

        # L2: 分区MPC控制器 (每10个闸门一组)
        num_zones = (self.num_gates + 9) // 10
        for z in range(num_zones):
            start_idx = z * 10
            end_idx = min((z + 1) * 10, self.num_gates)
            gate_ids = [f"GATE_{i:03d}" for i in range(start_idx, end_idx)]

            config = ControllerConfig(
                controller_id=f"L2_MPC_ZONE_{z:02d}",
                level=ControlLevel.L2_COORDINATED,
                gate_ids=gate_ids,
                control_interval=900.0,  # L2较慢
                prediction_horizon=12,
            )
            self.register_controller(config.controller_id, MPCController(config))

        logger.info(f"Created default controllers: {len(self.controllers)}")

    def step(
        self,
        states: Dict[str, SegmentState],
        boundaries: Dict[str, BoundaryCondition],
        dt: float,
    ) -> Dict[str, float]:
        """
        执行一个控制步

        Args:
            states: 当前渠段状态
            boundaries: 当前边界条件
            dt: 时间步长

        Returns:
            gate_openings: 各闸门开度
        """
        self.current_time += dt

        # 收集所有控制器的指令
        all_commands: List[ControlCommand] = []

        for controller_id, controller in self.controllers.items():
            if not controller.is_active:
                continue

            # 检查是否到达控制周期
            if controller.last_action_time is not None:
                elapsed = self.current_time - controller.last_action_time
                if elapsed < controller.config.control_interval:
                    continue

            # 计算控制动作
            commands = controller.compute_action(
                states, boundaries, self.current_time
            )
            all_commands.extend(commands)
            controller.last_action_time = self.current_time

        # 融合控制指令 (优先级处理)
        fused_commands = self._fuse_commands(all_commands)

        # 执行指令
        gate_openings = self._execute_commands(fused_commands)

        # 记录历史
        self.command_history.extend(fused_commands)

        return gate_openings

    def _fuse_commands(
        self,
        commands: List[ControlCommand]
    ) -> List[ControlCommand]:
        """
        融合控制指令

        优先级: L4 > L3 > L2 > L1
        同级别: 最新的优先
        """
        # 按闸门分组
        gate_commands: Dict[str, List[ControlCommand]] = {}
        for cmd in commands:
            if cmd.gate_id not in gate_commands:
                gate_commands[cmd.gate_id] = []
            gate_commands[cmd.gate_id].append(cmd)

        # 每个闸门选择最高优先级的指令
        fused = []
        priority_order = {
            "L4": 4, "L3": 3, "L2": 2, "L1": 1
        }

        for gate_id, cmds in gate_commands.items():
            if not cmds:
                continue

            # 按优先级排序
            cmds.sort(
                key=lambda c: (
                    priority_order.get(c.source_level, 0),
                    c.timestamp
                ),
                reverse=True
            )
            fused.append(cmds[0])

        return fused

    def _execute_commands(
        self,
        commands: List[ControlCommand]
    ) -> Dict[str, float]:
        """执行控制指令,返回更新后的闸门开度"""
        for cmd in commands:
            if cmd.gate_id not in self.gate_states:
                continue

            gate = self.gate_states[cmd.gate_id]

            if cmd.target_opening is not None:
                # 直接设置开度
                new_opening = np.clip(cmd.target_opening, 0.0, 1.0)
            elif cmd.target_flow is not None:
                # 流量控制 (简化: 线性映射)
                new_opening = np.clip(cmd.target_flow / 600.0, 0.0, 1.0)
            elif cmd.target_level is not None:
                # 水位控制 (需要根据当前状态计算)
                error = cmd.target_level - gate.upstream_level
                delta = error * 0.1
                new_opening = np.clip(gate.opening + delta, 0.0, 1.0)
            else:
                continue

            # 限制变化率
            max_delta = cmd.max_rate * self.default_control_interval
            delta = new_opening - gate.opening
            delta = np.clip(delta, -max_delta, max_delta)
            gate.opening = gate.opening + delta

        return {gid: g.opening for gid, g in self.gate_states.items()}

    def set_targets(
        self,
        level_targets: Optional[Dict[str, float]] = None,
        flow_targets: Optional[Dict[str, float]] = None
    ):
        """设置控制目标"""
        for controller in self.controllers.values():
            if level_targets:
                controller.config.target_levels.update(level_targets)
            if flow_targets:
                controller.config.target_flows.update(flow_targets)

    def activate_controllers(self, levels: List[ControlLevel]):
        """激活指定层级的控制器"""
        for controller in self.controllers.values():
            controller.is_active = controller.config.level in levels
            if controller.is_active:
                logger.info(f"Activated controller: {controller.config.controller_id}")

    def deactivate_all(self):
        """停用所有控制器"""
        for controller in self.controllers.values():
            controller.is_active = False

    def get_gate_openings(self) -> np.ndarray:
        """获取所有闸门开度"""
        openings = np.zeros(self.num_gates)
        for i in range(self.num_gates):
            gate_id = f"GATE_{i:03d}"
            if gate_id in self.gate_states:
                openings[i] = self.gate_states[gate_id].opening
        return openings

    def set_gate_opening(self, gate_id: str, opening: float):
        """手动设置闸门开度"""
        if gate_id in self.gate_states:
            self.gate_states[gate_id].opening = np.clip(opening, 0.0, 1.0)

    def get_command_statistics(self) -> Dict[str, Any]:
        """获取控制指令统计"""
        level_counts = {level.value: 0 for level in ControlLevel}
        for cmd in self.command_history:
            if cmd.source_level in level_counts:
                level_counts[cmd.source_level] += 1

        return {
            "total_commands": len(self.command_history),
            "commands_by_level": level_counts,
            "active_controllers": sum(
                1 for c in self.controllers.values() if c.is_active
            ),
            "current_time": self.current_time,
        }

    def reset(self):
        """重置编排器"""
        self.current_time = 0.0
        self.pending_commands = []
        self.command_history = []
        self._init_gates()

        for controller in self.controllers.values():
            controller.reset()

        logger.info("ControllerOrchestrator reset")
