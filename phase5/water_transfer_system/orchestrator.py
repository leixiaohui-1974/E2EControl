"""
全局编排器 (Global Orchestrator)
L3 层 - 全局调度与场景响应

核心功能:
1. 场景识别与分类 (8大场景)
2. 动态角色分配 (6种角色)
3. 全场景策略矩阵
4. 控制计划生成
5. 区域协调指令下发

场景-角色矩阵:
| 场景 | 事故点(k) | 上游(<k) | 下游(>k) | MPC策略 |
|------|-----------|----------|----------|---------|
| S1 常规计划 | TRANSMIT | TRANSMIT | TRANSMIT | 追踪Q_plan |
| S2 突发增供 | SOURCE | FEED | WAIT | 反向传播 |
| S3 突发污染 | ISOLATE | BUFFER | DRAIN | 截-蓄-排 |
| S4 暴雨防洪 | DRAIN | HOLD | PASS | 全线预泄 |
| S5 冰期输水 | STABLE | STABLE | STABLE | 稳流 |
| S6 泵站掉电 | PASSIVE | BRAKE | COMPENSATE | 平滑过渡 |
| S7 计划检修 | MAINT | STORE | CONSUME | 蓄能-隔离 |
| S8 临时抢修 | FIX | BYPASS | ISLAND | 降级运行 |
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import time
import uuid

from .core_types import (
    PoolRole, ScenarioType, ScenarioSeverity, ScenarioPhase,
    ControlDirective, ControlPlan, ScenarioEvent,
    PoolTopology, CanalPoolConfig, RegionConfig,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 场景-角色矩阵定义
# ==============================================================================

@dataclass
class ScenarioRoleMapping:
    """场景角色映射"""
    center_role: PoolRole          # 事故点角色
    upstream_role: PoolRole        # 上游角色
    downstream_role: PoolRole      # 下游角色

    # MPC权重调整
    center_weights: Dict[str, float] = field(default_factory=dict)
    upstream_weights: Dict[str, float] = field(default_factory=dict)
    downstream_weights: Dict[str, float] = field(default_factory=dict)

    # 硬约束
    center_constraints: Dict[str, Any] = field(default_factory=dict)
    upstream_constraints: Dict[str, Any] = field(default_factory=dict)
    downstream_constraints: Dict[str, Any] = field(default_factory=dict)

    # 目标偏移
    center_target_bias: float = 0.0
    upstream_target_bias: float = 0.0
    downstream_target_bias: float = 0.0


class ScenarioRoleMatrix:
    """
    场景-角色矩阵

    定义8大场景下的角色分配和MPC参数调整策略
    """

    # 矩阵定义
    MATRIX: Dict[ScenarioType, ScenarioRoleMapping] = {
        # S1: 常规计划 - 所有池正常传输
        ScenarioType.S1_NORMAL_PLAN: ScenarioRoleMapping(
            center_role=PoolRole.TRANSMIT,
            upstream_role=PoolRole.TRANSMIT,
            downstream_role=PoolRole.TRANSMIT,
            center_weights={'W_Q': 1.0, 'W_Z': 1.0},
            upstream_weights={'W_Q': 1.0, 'W_Z': 1.0},
            downstream_weights={'W_Q': 1.0, 'W_Z': 1.0},
        ),

        # S2: 突发增供 - 下游需求增加，反向传播
        ScenarioType.S2_SURGE_DEMAND: ScenarioRoleMapping(
            center_role=PoolRole.SOURCE,
            upstream_role=PoolRole.FEED,
            downstream_role=PoolRole.WAIT,
            center_weights={'W_Q': 2.0, 'W_Z': 0.5},  # 优先流量
            upstream_weights={'W_Q': 1.5, 'W_Z': 0.8},
            downstream_weights={'W_Q': 0.5, 'W_Z': 1.5},  # 等待来水
            downstream_target_bias=-0.3,  # 目标水位略降，为来水留空间
        ),

        # S3: 突发污染 - 截-蓄-排
        ScenarioType.S3_POLLUTION: ScenarioRoleMapping(
            center_role=PoolRole.ISOLATE,
            upstream_role=PoolRole.BUFFER,
            downstream_role=PoolRole.DRAIN,
            center_weights={'W_Q': 1e9, 'W_Z': 0.0},  # 必须断流
            upstream_weights={'W_Q': 0.1, 'W_Z': 0.5},  # 蓄水
            downstream_weights={'W_Q': 0.1, 'W_Z': 2.0},  # 排水
            center_constraints={'Q_out_max': 0, 'Q_drain_min': 50},
            upstream_target_bias=0.5,  # 蓄水，水位升高
            downstream_target_bias=-0.5,  # 排水，水位降低
        ),

        # S4: 暴雨防洪 - 全线预泄
        ScenarioType.S4_FLOOD_CONTROL: ScenarioRoleMapping(
            center_role=PoolRole.DRAIN,
            upstream_role=PoolRole.HOLD,
            downstream_role=PoolRole.PASS,
            center_weights={'W_Q': 1.5, 'W_Z': 2.0},  # 优先控制水位
            upstream_weights={'W_Q': 0.5, 'W_Z': 2.0},
            downstream_weights={'W_Q': 2.0, 'W_Z': 1.0},
            center_target_bias=-0.5,  # 预泄
            upstream_target_bias=-0.3,
            downstream_target_bias=-0.2,
            center_constraints={'Z_max': 'bank_level - 0.5'},
        ),

        # S5: 冰期输水 - 稳流
        ScenarioType.S5_ICE_PERIOD: ScenarioRoleMapping(
            center_role=PoolRole.STABLE,
            upstream_role=PoolRole.STABLE,
            downstream_role=PoolRole.STABLE,
            center_weights={'W_Q': 0.5, 'W_Z': 1.5, 'W_smooth': 5.0},  # 高平滑权重
            upstream_weights={'W_Q': 0.5, 'W_Z': 1.5, 'W_smooth': 5.0},
            downstream_weights={'W_Q': 0.5, 'W_Z': 1.5, 'W_smooth': 5.0},
            center_constraints={'delta_Q_max': 0.1},  # 限制流量变化率
        ),

        # S6: 泵站掉电 - 平滑过渡
        ScenarioType.S6_PUMP_FAILURE: ScenarioRoleMapping(
            center_role=PoolRole.PASSIVE,
            upstream_role=PoolRole.BRAKE,
            downstream_role=PoolRole.COMPENSATE,
            center_weights={'W_Q': 0.1, 'W_Z': 1.0},
            upstream_weights={'W_Q': 2.0, 'W_Z': 1.5},  # 上游减流
            downstream_weights={'W_Q': 1.5, 'W_Z': 0.8},  # 下游补偿
            upstream_constraints={'Q_out_max': 'current * 0.7'},
            downstream_target_bias=-0.2,
        ),

        # S7: 计划检修 - 蓄能-隔离
        ScenarioType.S7_PLANNED_MAINT: ScenarioRoleMapping(
            center_role=PoolRole.MAINT,
            upstream_role=PoolRole.STORE,
            downstream_role=PoolRole.CONSUME,
            center_weights={'W_Q': 1e6, 'W_Z': 0.5},
            upstream_weights={'W_Q': 0.8, 'W_Z': 1.2},
            downstream_weights={'W_Q': 1.2, 'W_Z': 0.8},
            center_constraints={'Q_out': 0, 'Q_in': 0},
            upstream_target_bias=0.8,  # 提前蓄水
            downstream_target_bias=-0.3,  # 消耗库存
        ),

        # S8: 临时抢修 - 降级运行
        ScenarioType.S8_EMERGENCY_REPAIR: ScenarioRoleMapping(
            center_role=PoolRole.FIX,
            upstream_role=PoolRole.BYPASS,
            downstream_role=PoolRole.ISLAND,
            center_weights={'W_Q': 1e9, 'W_Z': 0.0},
            upstream_weights={'W_Q': 0.5, 'W_Z': 1.0},
            downstream_weights={'W_Q': 0.5, 'W_Z': 1.5},
            center_constraints={'Q_out': 0},
            upstream_constraints={'use_bypass': True},
        ),
    }

    @classmethod
    def get_mapping(cls, scenario: ScenarioType) -> ScenarioRoleMapping:
        """获取场景角色映射"""
        return cls.MATRIX.get(scenario, cls.MATRIX[ScenarioType.S1_NORMAL_PLAN])

    @classmethod
    def get_role_for_position(cls,
                              scenario: ScenarioType,
                              pool_id: int,
                              center_id: int,
                              num_pools: int) -> Tuple[PoolRole, Dict, Dict, float]:
        """
        获取特定位置的角色和参数

        Args:
            scenario: 场景类型
            pool_id: 目标池ID
            center_id: 事故点池ID
            num_pools: 总池数

        Returns:
            (角色, 权重, 约束, 目标偏移)
        """
        mapping = cls.get_mapping(scenario)

        if pool_id == center_id:
            return (mapping.center_role,
                   mapping.center_weights,
                   mapping.center_constraints,
                   mapping.center_target_bias)
        elif pool_id < center_id:
            return (mapping.upstream_role,
                   mapping.upstream_weights,
                   mapping.upstream_constraints,
                   mapping.upstream_target_bias)
        else:
            return (mapping.downstream_role,
                   mapping.downstream_weights,
                   mapping.downstream_constraints,
                   mapping.downstream_target_bias)


# ==============================================================================
# 全局编排器
# ==============================================================================

class GlobalOrchestrator:
    """
    全局编排器 (L3层)

    职责:
    1. 接收场景事件
    2. 分析事件并确定响应策略
    3. 生成控制计划
    4. 下发指令给区域协调器
    """

    def __init__(self, topology: PoolTopology = None):
        """
        初始化全局编排器

        Args:
            topology: 渠池拓扑
        """
        self.topology = topology or PoolTopology.create_snwd_middle_route()

        # 当前状态
        self.current_scenario: ScenarioType = ScenarioType.S1_NORMAL_PLAN
        self.current_events: List[ScenarioEvent] = []
        self.active_plans: Dict[str, ControlPlan] = {}

        # 历史记录
        self.event_history: List[ScenarioEvent] = []
        self.plan_history: List[ControlPlan] = []

        # 回调
        self._plan_callbacks: List[Callable[[ControlPlan], None]] = []

        # 统计
        self.stats = {
            'events_processed': 0,
            'plans_generated': 0,
            'active_scenarios': 0,
        }

        logger.info(f"全局编排器初始化: {self.topology.num_pools} 渠池")

    def process_event(self, event: ScenarioEvent) -> ControlPlan:
        """
        处理场景事件

        Args:
            event: 场景事件

        Returns:
            生成的控制计划
        """
        logger.info(f"处理事件: {event.scenario_type.value} @ 池{event.location}, "
                   f"严重度={event.severity.value}")

        # 记录事件
        self.current_events.append(event)
        self.event_history.append(event)
        self.stats['events_processed'] += 1

        # 更新当前场景
        self.current_scenario = event.scenario_type

        # 生成控制计划
        plan = self.generate_plan(event)

        # 激活计划
        self.active_plans[plan.plan_id] = plan
        self.plan_history.append(plan)
        self.stats['plans_generated'] += 1

        # 通知回调
        for callback in self._plan_callbacks:
            try:
                callback(plan)
            except Exception as e:
                logger.error(f"计划回调错误: {e}")

        return plan

    def generate_plan(self, event: ScenarioEvent) -> ControlPlan:
        """
        生成控制计划

        Args:
            event: 场景事件

        Returns:
            控制计划
        """
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        center = event.location

        plan = ControlPlan(
            plan_id=plan_id,
            scenario=event.scenario_type,
            timestamp=event.timestamp,
            duration=self._estimate_duration(event),
            priority=event.severity.value,
        )

        # 根据场景生成指令
        if event.scenario_type == ScenarioType.S3_POLLUTION:
            plan = self._generate_pollution_plan(plan, event)
        elif event.scenario_type == ScenarioType.S4_FLOOD_CONTROL:
            plan = self._generate_flood_plan(plan, event)
        elif event.scenario_type == ScenarioType.S5_ICE_PERIOD:
            plan = self._generate_ice_plan(plan, event)
        elif event.scenario_type == ScenarioType.S6_PUMP_FAILURE:
            plan = self._generate_pump_failure_plan(plan, event)
        elif event.scenario_type == ScenarioType.S7_PLANNED_MAINT:
            plan = self._generate_maintenance_plan(plan, event)
        elif event.scenario_type == ScenarioType.S8_EMERGENCY_REPAIR:
            plan = self._generate_emergency_plan(plan, event)
        elif event.scenario_type == ScenarioType.S2_SURGE_DEMAND:
            plan = self._generate_surge_demand_plan(plan, event)
        else:
            # 常规计划
            plan = self._generate_normal_plan(plan, event)

        logger.info(f"生成计划: {plan_id}, {plan.num_directives} 条指令")
        return plan

    def _generate_pollution_plan(self, plan: ControlPlan, event: ScenarioEvent) -> ControlPlan:
        """生成污染响应计划 (截-蓄-排)"""
        center = event.location
        mapping = ScenarioRoleMatrix.get_mapping(ScenarioType.S3_POLLUTION)

        # 事故点: ISOLATE
        plan.add_directive(ControlDirective(
            pool_id=center,
            role=PoolRole.ISOLATE,
            weight_multipliers=mapping.center_weights.copy(),
            hard_constraints={
                'Q_out_max': 0,          # 关闸
                'Q_drain_min': event.details.get('drain_capacity', 50),  # 开退水
            },
            priority=10,
            source_scenario=event.scenario_type,
            remarks="污染隔离-关闸开退水",
        ))

        # 上游: BUFFER (递归)
        upstream_pools = self._get_upstream_chain(center)
        for i, pool_id in enumerate(upstream_pools):
            # 距离越远，缓冲任务越轻
            distance_factor = max(0.2, 1.0 - i * 0.15)

            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.BUFFER,
                target_bias=mapping.upstream_target_bias * distance_factor,
                weight_multipliers={
                    'W_Q': mapping.upstream_weights.get('W_Q', 0.1) / distance_factor,
                    'W_Z': mapping.upstream_weights.get('W_Z', 0.5) * distance_factor,
                },
                hard_constraints={
                    'Z_max': 'bank_level - 0.2',  # 放宽约束以蓄水
                },
                priority=8 - i,
                source_scenario=event.scenario_type,
                remarks=f"上游蓄滞-缓冲{i+1}级",
            ))

        # 下游: DRAIN (递归)
        downstream_pools = self._get_downstream_chain(center)
        for i, pool_id in enumerate(downstream_pools):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.DRAIN,
                target_bias=mapping.downstream_target_bias * (1 - i * 0.1),
                weight_multipliers={
                    'W_Q': mapping.downstream_weights.get('W_Q', 0.1),
                    'W_Z': mapping.downstream_weights.get('W_Z', 2.0),
                },
                hard_constraints={
                    'Q_user': 0,  # 切断分水口
                },
                priority=7 - i,
                source_scenario=event.scenario_type,
                remarks=f"下游排空-断供{i+1}级",
            ))

        return plan

    def _generate_flood_plan(self, plan: ControlPlan, event: ScenarioEvent) -> ControlPlan:
        """生成防洪计划 (全线预泄)"""
        mapping = ScenarioRoleMatrix.get_mapping(ScenarioType.S4_FLOOD_CONTROL)

        # 全线预泄
        for pool_id in range(self.topology.num_pools):
            pool_config = self.topology.get_pool(pool_id)

            # 计算预泄目标
            target_bias = mapping.center_target_bias
            if pool_id < event.location:
                target_bias = mapping.upstream_target_bias
            elif pool_id > event.location:
                target_bias = mapping.downstream_target_bias

            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.DRAIN if pool_id <= event.location else PoolRole.PASS,
                target_bias=target_bias,
                weight_multipliers={
                    'W_Q': 1.5,
                    'W_Z': 2.5,  # 高水位权重
                },
                hard_constraints={
                    'Z_max': pool_config.max_depth - 0.5 if pool_config else 7.5,
                },
                priority=9,
                source_scenario=event.scenario_type,
                remarks="防洪预泄",
            ))

        return plan

    def _generate_ice_plan(self, plan: ControlPlan, event: ScenarioEvent) -> ControlPlan:
        """生成冰期计划 (稳流)"""
        mapping = ScenarioRoleMatrix.get_mapping(ScenarioType.S5_ICE_PERIOD)

        for pool_id in range(self.topology.num_pools):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.STABLE,
                weight_multipliers={
                    'W_Q': 0.5,
                    'W_Z': 1.5,
                    'W_smooth': 5.0,  # 高平滑权重
                },
                hard_constraints={
                    'delta_Q_max': 0.1,  # 限制日变化率10%
                    'flow_efficiency': 0.75,  # 冰盖影响
                },
                priority=7,
                source_scenario=event.scenario_type,
                remarks="冰期稳流运行",
            ))

        return plan

    def _generate_pump_failure_plan(self, plan: ControlPlan, event: ScenarioEvent) -> ControlPlan:
        """生成泵站故障计划 (平滑过渡)"""
        center = event.location
        mapping = ScenarioRoleMatrix.get_mapping(ScenarioType.S6_PUMP_FAILURE)

        # 故障点
        plan.add_directive(ControlDirective(
            pool_id=center,
            role=PoolRole.PASSIVE,
            weight_multipliers=mapping.center_weights,
            priority=8,
            source_scenario=event.scenario_type,
            remarks="泵站故障-被动响应",
        ))

        # 上游减流
        for pool_id in self._get_upstream_chain(center, max_depth=5):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.BRAKE,
                weight_multipliers=mapping.upstream_weights,
                hard_constraints={
                    'Q_out_reduction': 0.3,  # 减流30%
                },
                priority=7,
                source_scenario=event.scenario_type,
                remarks="上游刹车减流",
            ))

        # 下游补偿
        for pool_id in self._get_downstream_chain(center, max_depth=5):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.COMPENSATE,
                target_bias=-0.2,
                weight_multipliers=mapping.downstream_weights,
                priority=6,
                source_scenario=event.scenario_type,
                remarks="下游补偿供水",
            ))

        return plan

    def _generate_maintenance_plan(self, plan: ControlPlan, event: ScenarioEvent) -> ControlPlan:
        """生成计划检修计划 (蓄能-隔离)"""
        center = event.location
        mapping = ScenarioRoleMatrix.get_mapping(ScenarioType.S7_PLANNED_MAINT)

        # 检修点
        plan.add_directive(ControlDirective(
            pool_id=center,
            role=PoolRole.MAINT,
            weight_multipliers=mapping.center_weights,
            hard_constraints={
                'Q_out': 0,
                'Q_in': 0,
            },
            effective_time=event.details.get('start_time'),
            expiry_time=event.details.get('end_time'),
            priority=6,
            source_scenario=event.scenario_type,
            remarks="检修隔离",
        ))

        # 上游提前蓄水
        for pool_id in self._get_upstream_chain(center, max_depth=3):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.STORE,
                target_bias=0.8,  # 提前24h蓄水
                weight_multipliers=mapping.upstream_weights,
                priority=5,
                source_scenario=event.scenario_type,
                remarks="上游蓄能",
            ))

        # 下游消耗库存
        for pool_id in self._get_downstream_chain(center, max_depth=3):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.CONSUME,
                target_bias=-0.3,
                weight_multipliers=mapping.downstream_weights,
                priority=5,
                source_scenario=event.scenario_type,
                remarks="下游消耗库存",
            ))

        return plan

    def _generate_emergency_plan(self, plan: ControlPlan, event: ScenarioEvent) -> ControlPlan:
        """生成紧急抢修计划 (降级运行)"""
        center = event.location
        mapping = ScenarioRoleMatrix.get_mapping(ScenarioType.S8_EMERGENCY_REPAIR)

        # 抢修点
        plan.add_directive(ControlDirective(
            pool_id=center,
            role=PoolRole.FIX,
            weight_multipliers=mapping.center_weights,
            hard_constraints={'Q_out': 0},
            priority=10,
            source_scenario=event.scenario_type,
            remarks="紧急抢修",
        ))

        # 尝试旁通
        for pool_id in self._get_upstream_chain(center, max_depth=2):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.BYPASS,
                weight_multipliers=mapping.upstream_weights,
                hard_constraints={'use_bypass': True},
                priority=9,
                source_scenario=event.scenario_type,
                remarks="上游旁通",
            ))

        # 下游孤岛运行
        for pool_id in self._get_downstream_chain(center, max_depth=5):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.ISLAND,
                target_bias=-0.2,
                weight_multipliers=mapping.downstream_weights,
                priority=8,
                source_scenario=event.scenario_type,
                remarks="下游孤岛供水",
            ))

        return plan

    def _generate_surge_demand_plan(self, plan: ControlPlan, event: ScenarioEvent) -> ControlPlan:
        """生成突发增供计划 (反向传播)"""
        center = event.location
        mapping = ScenarioRoleMatrix.get_mapping(ScenarioType.S2_SURGE_DEMAND)

        # 反向传播: 下游先开，上游后开
        # 需求点
        plan.add_directive(ControlDirective(
            pool_id=center,
            role=PoolRole.SOURCE,
            weight_multipliers=mapping.center_weights,
            priority=8,
            source_scenario=event.scenario_type,
            remarks="需求点-发起供水",
        ))

        # 下游等待
        for i, pool_id in enumerate(self._get_downstream_chain(center)):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.WAIT,
                target_bias=mapping.downstream_target_bias,
                weight_multipliers=mapping.downstream_weights,
                effective_time=event.timestamp + (i + 1) * 1800,  # 延迟生效
                priority=6,
                source_scenario=event.scenario_type,
                remarks=f"下游等待来水-{i+1}级",
            ))

        # 上游补给
        upstream_pools = self._get_upstream_chain(center)
        for i, pool_id in enumerate(upstream_pools):
            # 反向: 近的先响应，远的后响应
            delay = (len(upstream_pools) - i) * 900  # 15分钟间隔

            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.FEED,
                weight_multipliers=mapping.upstream_weights,
                effective_time=event.timestamp + delay,
                priority=7 - i,
                source_scenario=event.scenario_type,
                remarks=f"上游补给-{i+1}级",
            ))

        return plan

    def _generate_normal_plan(self, plan: ControlPlan, event: ScenarioEvent) -> ControlPlan:
        """生成常规计划"""
        for pool_id in range(self.topology.num_pools):
            plan.add_directive(ControlDirective(
                pool_id=pool_id,
                role=PoolRole.TRANSMIT,
                weight_multipliers={'W_Q': 1.0, 'W_Z': 1.0},
                priority=1,
                source_scenario=event.scenario_type,
                remarks="常规传输",
            ))
        return plan

    def _get_upstream_chain(self, pool_id: int, max_depth: int = None) -> List[int]:
        """获取上游池链"""
        chain = []
        current = pool_id - 1
        depth = 0
        max_depth = max_depth or self.topology.num_pools

        while current >= 0 and depth < max_depth:
            chain.append(current)
            current -= 1
            depth += 1

        return chain

    def _get_downstream_chain(self, pool_id: int, max_depth: int = None) -> List[int]:
        """获取下游池链"""
        chain = []
        current = pool_id + 1
        depth = 0
        max_depth = max_depth or self.topology.num_pools

        while current < self.topology.num_pools and depth < max_depth:
            chain.append(current)
            current += 1
            depth += 1

        return chain

    def _estimate_duration(self, event: ScenarioEvent) -> float:
        """估计事件持续时间 [s]"""
        duration_map = {
            ScenarioType.S1_NORMAL_PLAN: 86400,     # 24小时
            ScenarioType.S2_SURGE_DEMAND: 14400,    # 4小时
            ScenarioType.S3_POLLUTION: 86400,       # 24小时
            ScenarioType.S4_FLOOD_CONTROL: 43200,   # 12小时
            ScenarioType.S5_ICE_PERIOD: 2592000,    # 30天
            ScenarioType.S6_PUMP_FAILURE: 7200,     # 2小时
            ScenarioType.S7_PLANNED_MAINT: 28800,   # 8小时
            ScenarioType.S8_EMERGENCY_REPAIR: 14400,  # 4小时
        }
        return duration_map.get(event.scenario_type, 3600)

    def get_active_plan(self, plan_id: str) -> Optional[ControlPlan]:
        """获取活动计划"""
        return self.active_plans.get(plan_id)

    def cancel_plan(self, plan_id: str) -> bool:
        """取消计划"""
        if plan_id in self.active_plans:
            plan = self.active_plans.pop(plan_id)
            plan.is_active = False
            logger.info(f"取消计划: {plan_id}")
            return True
        return False

    def get_directives_for_pool(self, pool_id: int) -> List[ControlDirective]:
        """获取指定池的所有活动指令"""
        directives = []
        for plan in self.active_plans.values():
            if plan.is_active:
                d = plan.get_directive(pool_id)
                if d:
                    directives.append(d)
        # 按优先级排序
        return sorted(directives, key=lambda x: x.priority, reverse=True)

    def on_plan_generated(self, callback: Callable[[ControlPlan], None]):
        """注册计划生成回调"""
        self._plan_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'current_scenario': self.current_scenario.value,
            'active_plans': len(self.active_plans),
            'total_events': len(self.event_history),
            'total_plans': len(self.plan_history),
        }


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" " * 15 + "全局编排器测试")
    print("="*70)

    # 创建编排器
    orchestrator = GlobalOrchestrator()

    print(f"\n✓ 全局编排器初始化")
    print(f"  渠池数量: {orchestrator.topology.num_pools}")

    # 测试1: 常规场景
    print(f"\n{'='*70}")
    print("测试1: 常规运行场景 (S1)")
    print('='*70)

    event = ScenarioEvent(
        event_id="E001",
        scenario_type=ScenarioType.S1_NORMAL_PLAN,
        location=30,
        severity=ScenarioSeverity.LOW,
        timestamp=0.0,
    )

    plan = orchestrator.process_event(event)
    print(f"  计划ID: {plan.plan_id}")
    print(f"  指令数: {plan.num_directives}")
    print(f"  示例指令: 池30 -> {plan.get_directive(30).role.value}")

    # 测试2: 污染场景
    print(f"\n{'='*70}")
    print("测试2: 突发污染场景 (S3)")
    print('='*70)

    event = ScenarioEvent(
        event_id="E002",
        scenario_type=ScenarioType.S3_POLLUTION,
        location=30,
        severity=ScenarioSeverity.CRITICAL,
        timestamp=100.0,
        details={'drain_capacity': 80},
    )

    plan = orchestrator.process_event(event)
    print(f"  计划ID: {plan.plan_id}")
    print(f"  指令数: {plan.num_directives}")

    # 显示关键指令
    for pool_id in [28, 29, 30, 31, 32]:
        d = plan.get_directive(pool_id)
        if d:
            print(f"  池{pool_id}: {d.role.value}, 偏移={d.target_bias:.2f}, "
                  f"优先级={d.priority}")

    # 测试3: 洪水场景
    print(f"\n{'='*70}")
    print("测试3: 暴雨防洪场景 (S4)")
    print('='*70)

    event = ScenarioEvent(
        event_id="E003",
        scenario_type=ScenarioType.S4_FLOOD_CONTROL,
        location=20,
        severity=ScenarioSeverity.HIGH,
        timestamp=200.0,
    )

    plan = orchestrator.process_event(event)
    print(f"  计划ID: {plan.plan_id}")
    print(f"  指令数: {plan.num_directives}")

    d = plan.get_directive(20)
    print(f"  池20: {d.role.value}, 目标偏移={d.target_bias:.2f}")

    # 测试4: 场景-角色矩阵
    print(f"\n{'='*70}")
    print("测试4: 场景-角色矩阵")
    print('='*70)

    for scenario in ScenarioType:
        mapping = ScenarioRoleMatrix.get_mapping(scenario)
        print(f"  {scenario.value}:")
        print(f"    中心: {mapping.center_role.value}")
        print(f"    上游: {mapping.upstream_role.value}")
        print(f"    下游: {mapping.downstream_role.value}")

    # 统计
    print(f"\n{'='*70}")
    print("统计信息")
    print('='*70)
    stats = orchestrator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
