"""
区域协调器 (Regional Coordinator)
L2 层 - 区域协调与前馈解耦

核心功能:
1. 前馈解耦控制 - 消除上下游耦合延迟
2. 分布式MPC协调 - 区域内多池优化
3. 边界条件处理 - 区域间接口协调
4. 参考轨迹跟踪 - 执行L3下发的计划

前馈解耦原理:
当上游渠池改变出流时，通过通讯让下游提前响应，
消除因水流传播延迟导致的水位波动。

公式:
Q_ff[k] = Q_up[k-τ_comm] - Q_up[k-τ_comm-1]
其中 τ_comm << τ_hydraulic (通讯延迟远小于水力延迟)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
import time

from .core_types import (
    PoolRole, ControlDirective, PoolTopology, RegionConfig,
)
from .physics_model import IDZParameters

logger = logging.getLogger(__name__)


# ==============================================================================
# 前馈解耦器
# ==============================================================================

@dataclass
class FeedforwardState:
    """前馈状态"""
    pool_id: int
    upstream_flow_history: deque = field(default_factory=lambda: deque(maxlen=20))
    feedforward_command: float = 0.0
    last_update_time: float = 0.0


class FeedforwardDecoupler:
    """
    前馈解耦器

    通过预测上游流量变化并提前调整下游控制，
    消除串联渠池的耦合效应。

    解耦策略:
    1. 监听上游池的出流变化
    2. 基于IDZ模型预测传播到本池的影响
    3. 生成前馈补偿信号
    4. 与反馈控制器叠加
    """

    def __init__(self,
                 num_pools: int,
                 dt: float = 900.0,
                 comm_delay: float = 60.0):
        """
        初始化前馈解耦器

        Args:
            num_pools: 渠池数量
            dt: 控制周期 [s]
            comm_delay: 通讯延迟 [s]
        """
        self.num_pools = num_pools
        self.dt = dt
        self.comm_delay = comm_delay

        # 状态
        self.states: Dict[int, FeedforwardState] = {
            i: FeedforwardState(pool_id=i)
            for i in range(num_pools)
        }

        # IDZ参数 (每个池)
        self.idz_params: Dict[int, IDZParameters] = {}

        # 前馈增益
        self.ff_gain = 1.0

        # 预测时域
        self.prediction_steps = 5

    def set_idz_params(self, pool_id: int, params: IDZParameters):
        """设置池的IDZ参数"""
        self.idz_params[pool_id] = params

    def update_upstream_flow(self, pool_id: int, upstream_flow: float, timestamp: float):
        """
        更新上游流量信息

        Args:
            pool_id: 池ID
            upstream_flow: 上游出流 [m³/s]
            timestamp: 时间戳 [s]
        """
        if pool_id in self.states:
            self.states[pool_id].upstream_flow_history.append(upstream_flow)
            self.states[pool_id].last_update_time = timestamp

    def compute_feedforward(self, pool_id: int) -> float:
        """
        计算前馈补偿量

        Args:
            pool_id: 池ID

        Returns:
            前馈流量补偿 [m³/s]
        """
        if pool_id not in self.states or pool_id == 0:
            return 0.0

        state = self.states[pool_id]
        history = list(state.upstream_flow_history)

        if len(history) < 2:
            return 0.0

        # 计算流量变化率
        delta_q = history[-1] - history[-2]

        # 获取IDZ参数
        params = self.idz_params.get(pool_id, IDZParameters())

        # 前馈补偿: 基于流量变化预测水位影响
        # dZ = delta_q * dt / A_s
        # 需要补偿的出流变化
        ff_command = self.ff_gain * delta_q

        state.feedforward_command = ff_command
        return ff_command

    def compute_all_feedforward(self) -> Dict[int, float]:
        """计算所有池的前馈补偿"""
        return {
            pool_id: self.compute_feedforward(pool_id)
            for pool_id in range(1, self.num_pools)
        }

    def predict_impact(self, pool_id: int, horizon: int = 5) -> np.ndarray:
        """
        预测上游变化对本池的影响

        Args:
            pool_id: 池ID
            horizon: 预测时域 [步]

        Returns:
            预测的水位影响 [m]
        """
        if pool_id not in self.states:
            return np.zeros(horizon)

        state = self.states[pool_id]
        history = list(state.upstream_flow_history)

        if len(history) < 3:
            return np.zeros(horizon)

        params = self.idz_params.get(pool_id, IDZParameters())
        delay_steps = int(params.tau / self.dt)

        # 预测未来水位变化
        impact = np.zeros(horizon)

        for k in range(horizon):
            # 延迟后的流量变化
            if len(history) > delay_steps + k:
                delta_q = history[-(delay_steps + k + 1)] - history[-(delay_steps + k + 2)]
            else:
                delta_q = 0

            impact[k] = delta_q * self.dt / params.A_s

        return impact


# ==============================================================================
# 区域MPC协调器
# ==============================================================================

@dataclass
class RegionalMPCConfig:
    """区域MPC配置"""
    horizon: int = 10                  # 预测时域
    dt: float = 900.0                  # 时间步长 [s]

    # 权重
    W_level: float = 10.0              # 水位跟踪权重
    W_flow: float = 5.0                # 流量跟踪权重
    W_smooth: float = 2.0              # 平滑权重
    W_boundary: float = 20.0           # 边界协调权重

    # 约束
    Z_min: float = 0.5
    Z_max: float = 8.0
    Q_min: float = 0.0
    Q_max: float = 400.0
    delta_Q_max: float = 50.0          # 最大流量变化率


class RegionalMPCCoordinator:
    """
    区域MPC协调器

    在L2层协调区域内多个渠池的优化控制
    """

    def __init__(self,
                 region_config: RegionConfig,
                 mpc_config: RegionalMPCConfig = None):
        """
        初始化区域MPC协调器

        Args:
            region_config: 区域配置
            mpc_config: MPC配置
        """
        self.region = region_config
        self.config = mpc_config or RegionalMPCConfig()

        self.pool_ids = region_config.pool_ids
        self.num_pools = len(self.pool_ids)

        # 状态
        self.current_levels: Dict[int, float] = {}
        self.current_flows: Dict[int, float] = {}
        self.reference_levels: Dict[int, float] = {}
        self.reference_flows: Dict[int, float] = {}

        # 边界条件
        self.upstream_boundary_flow: float = 0.0
        self.downstream_boundary_level: float = 4.0

        # 指令
        self.active_directives: Dict[int, ControlDirective] = {}

        logger.info(f"创建区域协调器: {region_config.name}, {self.num_pools}池")

    def set_state(self,
                  levels: Dict[int, float],
                  flows: Dict[int, float]):
        """设置当前状态"""
        self.current_levels.update(levels)
        self.current_flows.update(flows)

    def set_references(self,
                       levels: Dict[int, float] = None,
                       flows: Dict[int, float] = None):
        """设置参考值"""
        if levels:
            self.reference_levels.update(levels)
        if flows:
            self.reference_flows.update(flows)

    def set_boundary_conditions(self,
                                upstream_flow: float = None,
                                downstream_level: float = None):
        """设置边界条件"""
        if upstream_flow is not None:
            self.upstream_boundary_flow = upstream_flow
        if downstream_level is not None:
            self.downstream_boundary_level = downstream_level

    def apply_directives(self, directives: List[ControlDirective]):
        """应用控制指令"""
        for d in directives:
            if d.pool_id in self.pool_ids:
                self.active_directives[d.pool_id] = d

    def solve(self) -> Dict[int, Tuple[float, float]]:
        """
        求解区域MPC

        Returns:
            {pool_id: (gate_opening, target_outflow)}
        """
        # 简化求解: 基于指令和参考轨迹计算控制
        results = {}

        for i, pool_id in enumerate(self.pool_ids):
            # 获取指令
            directive = self.active_directives.get(pool_id)

            # 获取状态
            current_level = self.current_levels.get(pool_id, 4.0)
            current_flow = self.current_flows.get(pool_id, 100.0)

            # 获取参考
            ref_level = self.reference_levels.get(pool_id, 4.0)
            ref_flow = self.reference_flows.get(pool_id, current_flow)

            # 基于指令调整
            if directive:
                ref_level += directive.target_bias

                # 根据角色调整
                if directive.role == PoolRole.ISOLATE:
                    # 关闸
                    results[pool_id] = (0.0, 0.0)
                    continue
                elif directive.role == PoolRole.DRAIN:
                    # 增大出流
                    ref_flow *= 1.2
                elif directive.role == PoolRole.BUFFER:
                    # 减小出流
                    ref_flow *= 0.8

            # 简单PI控制计算闸门开度
            level_error = current_level - ref_level
            gate_adjustment = -0.1 * level_error  # P控制

            gate_opening = np.clip(0.8 + gate_adjustment, 0.0, 1.0)
            target_outflow = ref_flow * gate_opening

            results[pool_id] = (gate_opening, target_outflow)

        return results


# ==============================================================================
# 区域协调器 (完整)
# ==============================================================================

class RegionalCoordinator:
    """
    区域协调器 (L2层完整实现)

    集成:
    1. 前馈解耦
    2. 区域MPC
    3. 指令执行
    4. 边界协调
    """

    def __init__(self,
                 region_config: RegionConfig,
                 topology: PoolTopology = None,
                 dt: float = 900.0):
        """
        初始化区域协调器

        Args:
            region_config: 区域配置
            topology: 全线拓扑
            dt: 控制周期 [s]
        """
        self.region = region_config
        self.topology = topology or PoolTopology.create_snwd_middle_route()
        self.dt = dt

        self.pool_ids = region_config.pool_ids
        self.num_pools = len(self.pool_ids)

        # 子模块
        self.decoupler = FeedforwardDecoupler(
            num_pools=self.topology.num_pools,
            dt=dt
        )
        self.mpc = RegionalMPCCoordinator(region_config)

        # 状态
        self.current_levels: np.ndarray = np.ones(self.num_pools) * 4.0
        self.current_inflows: np.ndarray = np.ones(self.num_pools) * 100.0
        self.current_outflows: np.ndarray = np.ones(self.num_pools) * 100.0

        # 参考轨迹
        self.reference_trajectory: Optional[np.ndarray] = None

        # 控制输出
        self.gate_commands: np.ndarray = np.ones(self.num_pools)
        self.outflow_commands: np.ndarray = np.ones(self.num_pools) * 100.0

        # 活动指令
        self.active_directives: Dict[int, ControlDirective] = {}

        # 邻居协调器
        self.upstream_coordinator: Optional['RegionalCoordinator'] = None
        self.downstream_coordinator: Optional['RegionalCoordinator'] = None

        # 统计
        self.stats = {
            'control_cycles': 0,
            'feedforward_active': 0,
            'directives_applied': 0,
        }

        logger.info(f"区域协调器初始化: {region_config.name}")

    def update_state(self,
                     levels: np.ndarray,
                     inflows: np.ndarray,
                     outflows: np.ndarray):
        """
        更新系统状态

        Args:
            levels: 水位数组 [num_pools]
            inflows: 入流数组 [num_pools]
            outflows: 出流数组 [num_pools]
        """
        self.current_levels = levels.copy()
        self.current_inflows = inflows.copy()
        self.current_outflows = outflows.copy()

        # 更新前馈解耦器的上游流量
        for i, pool_id in enumerate(self.pool_ids):
            if i > 0:
                upstream_flow = outflows[i - 1]
                self.decoupler.update_upstream_flow(
                    pool_id, upstream_flow, time.time()
                )

        # 更新MPC状态
        self.mpc.set_state(
            levels={self.pool_ids[i]: levels[i] for i in range(self.num_pools)},
            flows={self.pool_ids[i]: outflows[i] for i in range(self.num_pools)}
        )

    def set_reference_trajectory(self, trajectory: np.ndarray):
        """
        设置参考轨迹 (从L3接收)

        Args:
            trajectory: [num_pools x horizon] 参考水位
        """
        self.reference_trajectory = trajectory.copy()

        # 设置MPC参考
        if trajectory.shape[0] >= self.num_pools:
            self.mpc.set_references(
                levels={self.pool_ids[i]: trajectory[i, 0]
                       for i in range(self.num_pools)}
            )

    def apply_directives(self, directives: List[ControlDirective]):
        """
        应用L3下发的控制指令

        Args:
            directives: 控制指令列表
        """
        for d in directives:
            if d.pool_id in self.pool_ids:
                self.active_directives[d.pool_id] = d
                self.stats['directives_applied'] += 1

        # 传递给MPC
        self.mpc.apply_directives(directives)

    def coordinate_with_neighbors(self):
        """与相邻区域协调边界条件"""
        # 上游边界
        if self.upstream_coordinator:
            # 获取上游区域的出流作为本区域入流
            upstream_outflow = self.upstream_coordinator.get_boundary_outflow()
            self.mpc.set_boundary_conditions(upstream_flow=upstream_outflow)

        # 下游边界
        if self.downstream_coordinator:
            # 获取下游区域的水位作为约束
            downstream_level = self.downstream_coordinator.get_boundary_level()
            self.mpc.set_boundary_conditions(downstream_level=downstream_level)

    def compute_control(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算控制动作

        Returns:
            (gate_commands, outflow_commands)
        """
        self.stats['control_cycles'] += 1

        # 1. 与邻居协调
        self.coordinate_with_neighbors()

        # 2. 计算前馈补偿
        feedforward = np.zeros(self.num_pools)
        for i, pool_id in enumerate(self.pool_ids):
            ff = self.decoupler.compute_feedforward(pool_id)
            feedforward[i] = ff
            if abs(ff) > 1.0:
                self.stats['feedforward_active'] += 1

        # 3. 求解区域MPC
        mpc_results = self.mpc.solve()

        # 4. 组合控制
        for i, pool_id in enumerate(self.pool_ids):
            if pool_id in mpc_results:
                gate, outflow = mpc_results[pool_id]
                self.gate_commands[i] = gate
                self.outflow_commands[i] = outflow + feedforward[i]

        # 5. 安全约束
        self.outflow_commands = np.clip(
            self.outflow_commands, 0, 400
        )
        self.gate_commands = np.clip(
            self.gate_commands, 0, 1
        )

        return self.gate_commands.copy(), self.outflow_commands.copy()

    def get_boundary_outflow(self) -> float:
        """获取区域出口流量 (供下游协调)"""
        return self.current_outflows[-1] if len(self.current_outflows) > 0 else 100.0

    def get_boundary_level(self) -> float:
        """获取区域入口水位 (供上游协调)"""
        return self.current_levels[0] if len(self.current_levels) > 0 else 4.0

    def get_control_summary(self) -> Dict[str, Any]:
        """获取控制摘要"""
        return {
            'region_id': self.region.region_id,
            'region_name': self.region.name,
            'num_pools': self.num_pools,
            'avg_level': float(np.mean(self.current_levels)),
            'total_inflow': float(self.current_inflows[0]) if len(self.current_inflows) > 0 else 0,
            'total_outflow': float(self.current_outflows[-1]) if len(self.current_outflows) > 0 else 0,
            'avg_gate': float(np.mean(self.gate_commands)),
            'active_directives': len(self.active_directives),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'region_id': self.region.region_id,
            'region_name': self.region.name,
        }


# ==============================================================================
# 全线区域协调管理器
# ==============================================================================

class GlobalRegionalManager:
    """
    全线区域协调管理器

    管理所有区域协调器，处理区域间协调
    """

    def __init__(self, topology: PoolTopology = None, dt: float = 900.0):
        """
        初始化

        Args:
            topology: 全线拓扑
            dt: 控制周期 [s]
        """
        self.topology = topology or PoolTopology.create_snwd_middle_route()
        self.dt = dt

        # 创建区域协调器
        self.coordinators: Dict[int, RegionalCoordinator] = {}

        for region in self.topology.regions:
            coord = RegionalCoordinator(region, self.topology, dt)
            self.coordinators[region.region_id] = coord

        # 建立邻居关系
        self._setup_neighbors()

        logger.info(f"全线区域管理器初始化: {len(self.coordinators)} 个区域")

    def _setup_neighbors(self):
        """建立区域间邻居关系"""
        region_ids = sorted(self.coordinators.keys())

        for i, rid in enumerate(region_ids):
            coord = self.coordinators[rid]

            if i > 0:
                coord.upstream_coordinator = self.coordinators[region_ids[i-1]]
            if i < len(region_ids) - 1:
                coord.downstream_coordinator = self.coordinators[region_ids[i+1]]

    def update_all_states(self,
                          all_levels: np.ndarray,
                          all_inflows: np.ndarray,
                          all_outflows: np.ndarray):
        """
        更新所有区域状态

        Args:
            all_levels: 全线水位 [num_pools]
            all_inflows: 全线入流 [num_pools]
            all_outflows: 全线出流 [num_pools]
        """
        for region in self.topology.regions:
            coord = self.coordinators[region.region_id]

            # 提取区域数据
            indices = [self.topology.pools.index(
                self.topology.get_pool(pid)
            ) for pid in region.pool_ids if self.topology.get_pool(pid)]

            if not indices:
                continue

            levels = all_levels[indices]
            inflows = all_inflows[indices]
            outflows = all_outflows[indices]

            coord.update_state(levels, inflows, outflows)

    def apply_plan(self, plan: Any):
        """
        应用控制计划

        Args:
            plan: ControlPlan对象
        """
        for region_id, coord in self.coordinators.items():
            region_directives = [
                d for d in plan.directives.values()
                if d.pool_id in coord.pool_ids
            ]
            coord.apply_directives(region_directives)

    def compute_all_controls(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        计算所有区域控制

        Returns:
            {region_id: (gate_commands, outflow_commands)}
        """
        results = {}

        # 按顺序计算 (上游优先)
        for region_id in sorted(self.coordinators.keys()):
            coord = self.coordinators[region_id]
            gates, outflows = coord.compute_control()
            results[region_id] = (gates, outflows)

        return results

    def get_all_commands(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取全线控制命令

        Returns:
            (all_gates, all_outflows)
        """
        results = self.compute_all_controls()

        all_gates = np.ones(self.topology.num_pools)
        all_outflows = np.ones(self.topology.num_pools) * 100

        for region in self.topology.regions:
            coord = self.coordinators[region.region_id]
            gates, outflows = results[region.region_id]

            for i, pool_id in enumerate(region.pool_ids):
                if i < len(gates):
                    all_gates[pool_id] = gates[i]
                    all_outflows[pool_id] = outflows[i]

        return all_gates, all_outflows

    def get_summary(self) -> Dict[str, Any]:
        """获取全线摘要"""
        summaries = []
        for coord in self.coordinators.values():
            summaries.append(coord.get_control_summary())

        return {
            'num_regions': len(self.coordinators),
            'regions': summaries,
        }


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" " * 15 + "区域协调器测试")
    logger.info("="*70)

    # 创建拓扑
    topology = PoolTopology.create_snwd_middle_route()

    logger.info(f"\n✓ 创建南水北调中线拓扑")
    logger.info(f"  渠池数量: {topology.num_pools}")
    logger.info(f"  区域数量: {len(topology.regions)}")

    # 测试1: 前馈解耦器
    logger.info(f"\n{'='*70}")
    logger.info("测试1: 前馈解耦器")
    logger.info('='*70)

    decoupler = FeedforwardDecoupler(num_pools=60, dt=900.0)

    # 设置IDZ参数
    for i in range(60):
        decoupler.set_idz_params(i, IDZParameters(tau=14400, A_s=100000))

    # 模拟上游流量变化
    for t in range(10):
        upstream_flow = 100 + 10 * np.sin(t * 0.5)
        decoupler.update_upstream_flow(30, upstream_flow, t * 900)

    ff = decoupler.compute_feedforward(30)
    logger.info(f"  池30前馈补偿: {ff:.2f} m³/s")

    impact = decoupler.predict_impact(30, horizon=5)
    logger.info(f"  预测水位影响: {impact}")

    # 测试2: 区域MPC
    logger.info(f"\n{'='*70}")
    logger.info("测试2: 区域MPC协调器")
    logger.info('='*70)

    region = topology.regions[2]  # 河南段北
    mpc = RegionalMPCCoordinator(region)

    # 设置状态
    levels = {pid: 4.0 + 0.1 * np.random.randn() for pid in region.pool_ids}
    flows = {pid: 100 + 5 * np.random.randn() for pid in region.pool_ids}
    mpc.set_state(levels, flows)

    # 设置参考
    ref_levels = {pid: 4.0 for pid in region.pool_ids}
    mpc.set_references(levels=ref_levels)

    # 求解
    results = mpc.solve()
    logger.info(f"  区域: {region.name}")
    logger.info(f"  池数: {len(region.pool_ids)}")
    for pid in list(region.pool_ids)[:3]:
        gate, outflow = results[pid]
        logger.info(f"  池{pid}: 闸门={gate:.2f}, 目标出流={outflow:.1f}m³/s")

    # 测试3: 完整区域协调器
    logger.info(f"\n{'='*70}")
    logger.info("测试3: 完整区域协调器")
    logger.info('='*70)

    coord = RegionalCoordinator(region, topology, dt=900.0)

    # 更新状态
    n = len(region.pool_ids)
    coord.update_state(
        levels=np.ones(n) * 4.0,
        inflows=np.ones(n) * 100,
        outflows=np.ones(n) * 100
    )

    # 计算控制
    gates, outflows = coord.compute_control()
    logger.info(f"  控制计算完成")
    logger.info(f"  平均闸门: {np.mean(gates):.2f}")
    logger.info(f"  平均出流: {np.mean(outflows):.1f} m³/s")

    summary = coord.get_control_summary()
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    # 测试4: 全线协调
    logger.info(f"\n{'='*70}")
    logger.info("测试4: 全线区域协调")
    logger.info('='*70)

    manager = GlobalRegionalManager(topology, dt=900.0)

    # 更新全线状态
    all_levels = np.ones(60) * 4.0 + np.random.randn(60) * 0.2
    all_inflows = np.ones(60) * 100 + np.random.randn(60) * 5
    all_outflows = np.ones(60) * 100 + np.random.randn(60) * 5

    manager.update_all_states(all_levels, all_inflows, all_outflows)

    # 计算控制
    all_gates, all_outflows_cmd = manager.get_all_commands()

    logger.info(f"  全线闸门范围: [{all_gates.min():.2f}, {all_gates.max():.2f}]")
    logger.info(f"  全线出流范围: [{all_outflows_cmd.min():.1f}, {all_outflows_cmd.max():.1f}]")

    summary = manager.get_summary()
    logger.info(f"  区域数: {summary['num_regions']}")
    for r in summary['regions'][:3]:
        logger.info(f"    {r['region_name']}: 平均水位={r['avg_level']:.2f}m")

    logger.info("\n" + "="*70)
    logger.info("测试完成!")
    logger.info("="*70)
