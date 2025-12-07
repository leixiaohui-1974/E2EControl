"""
南水北调中线物理模型
SNWD Middle Route Physics Model

基于真实物理特性的高保真模型:
1. IDZ模型 (Integrator Delay Zero) - 渠池水力学模型
2. 全线60+渠池串联模型
3. 特殊节点 (倒虹吸、渡槽) 模型

数学模型:
- 体积平衡: dV/dt = Q_in - Q_out
- IDZ传递函数: G(s) = c_in * exp(-τ*s) / (A_s * s)
- 曼宁公式: Q = (1/n) * A * R^(2/3) * S^(1/2)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

from .core_types import (
    CanalPoolConfig, PoolTopology, SpecialStructure, StructureType
)

logger = logging.getLogger(__name__)


# ==============================================================================
# IDZ模型参数
# ==============================================================================

@dataclass
class IDZParameters:
    """
    IDZ模型参数 (Integrator Delay Zero)

    传递函数: G(s) = c_in * exp(-τ*s) / (A_s * s)

    参数:
    - tau: 滞后时间 [s]，水流从上游到下游的传播时间
    - A_s: 蓄水面积 [m²]，反映水位变化的缓冲能力
    - c_in: 入流增益，通常为1
    - c_out: 出流增益，通常为1
    """
    tau: float = 14400.0        # 滞后时间 [s] (默认4小时)
    A_s: float = 100000.0       # 蓄水面积 [m²]
    c_in: float = 1.0           # 入流增益
    c_out: float = 1.0          # 出流增益

    # 附加参数
    dead_time_fraction: float = 0.8  # 纯滞后占总滞后比例
    wave_celerity: float = 1.5       # 波速 [m/s]

    @property
    def integrator_gain(self) -> float:
        """积分增益 (1/A_s)"""
        return 1.0 / self.A_s if self.A_s > 0 else 0

    def compute_tau_from_length(self, length_km: float, velocity: float = 1.2) -> float:
        """根据渠池长度计算滞后时间"""
        return (length_km * 1000) / velocity

    def compute_A_s_from_geometry(self, length_km: float, avg_width: float) -> float:
        """根据几何参数计算蓄水面积"""
        return length_km * 1000 * avg_width


# ==============================================================================
# IDZ模型
# ==============================================================================

class IDZModel:
    """
    IDZ模型 (Integrator Delay Zero)

    用于描述渠池的水位-流量动态关系:
    - 积分特性: 入流-出流差值累积为水位变化
    - 延迟特性: 上游流量变化需要时间传播到下游

    离散化状态空间:
    Z[k+1] = Z[k] + (Q_in[k-d] - Q_out[k]) * dt / A_s

    其中d为延迟步数，d = tau / dt
    """

    def __init__(self, params: IDZParameters = None, dt: float = 900.0):
        """
        初始化IDZ模型

        Args:
            params: IDZ参数
            dt: 时间步长 [s]
        """
        self.params = params or IDZParameters()
        self.dt = dt

        # 计算延迟步数
        self.delay_steps = int(np.ceil(self.params.tau / dt))

        # 历史入流缓冲 (用于延迟)
        self._inflow_buffer: List[float] = [0.0] * (self.delay_steps + 1)
        self._buffer_index: int = 0

        # 状态
        self.current_level: float = 4.0  # 当前水位 [m]

    def reset(self, initial_level: float = 4.0, initial_inflow: float = 100.0):
        """重置模型状态"""
        self.current_level = initial_level
        self._inflow_buffer = [initial_inflow] * (self.delay_steps + 1)
        self._buffer_index = 0

    def step(self, q_in: float, q_out: float) -> float:
        """
        执行一步模拟

        Args:
            q_in: 入流 [m³/s]
            q_out: 出流 [m³/s]

        Returns:
            新的水位 [m]
        """
        # 存储当前入流
        self._inflow_buffer[self._buffer_index] = q_in

        # 获取延迟后的入流
        delayed_index = (self._buffer_index - self.delay_steps) % (self.delay_steps + 1)
        q_in_delayed = self._inflow_buffer[delayed_index]

        # 更新水位 (积分特性)
        dZ = (self.params.c_in * q_in_delayed - self.params.c_out * q_out) * self.dt / self.params.A_s
        self.current_level += dZ

        # 更新缓冲区索引
        self._buffer_index = (self._buffer_index + 1) % (self.delay_steps + 1)

        return self.current_level

    def get_delayed_inflow(self) -> float:
        """获取延迟后的入流"""
        delayed_index = (self._buffer_index - self.delay_steps) % (self.delay_steps + 1)
        return self._inflow_buffer[delayed_index]

    def get_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        获取状态空间矩阵 (用于MPC)

        返回:
            A, B_in, B_out, C 矩阵

        状态: [Z, Q_in[k-d], ..., Q_in[k-1]]
        """
        n_states = 1 + self.delay_steps

        A = np.zeros((n_states, n_states))
        A[0, 0] = 1.0  # Z[k+1] = Z[k] + ...
        if self.delay_steps > 0:
            A[0, self.delay_steps] = self.params.c_in * self.dt / self.params.A_s
            for i in range(1, self.delay_steps):
                A[i, i+1] = 1.0  # 移位

        B_in = np.zeros((n_states, 1))
        B_in[self.delay_steps if self.delay_steps > 0 else 0, 0] = 1.0

        B_out = np.zeros((n_states, 1))
        B_out[0, 0] = -self.params.c_out * self.dt / self.params.A_s

        C = np.zeros((1, n_states))
        C[0, 0] = 1.0

        return A, B_in, B_out, C


# ==============================================================================
# 渠池模型
# ==============================================================================

class CanalPool:
    """
    渠池模型 - 单个渠池的完整物理模型

    包含:
    - IDZ模型 (水位-流量动态)
    - 曼宁公式 (流量-水位关系)
    - 闸门模型 (出流控制)
    - 季节性参数调整
    """

    def __init__(self, config: CanalPoolConfig, dt: float = 900.0):
        """
        初始化渠池模型

        Args:
            config: 渠池配置
            dt: 时间步长 [s]
        """
        self.config = config
        self.dt = dt

        # 计算IDZ参数
        velocity = 1.2  # 平均流速 [m/s]
        tau = (config.length * 1000) / velocity
        A_s = config.surface_area

        self.idz_params = IDZParameters(
            tau=tau,
            A_s=A_s,
            c_in=1.0,
            c_out=1.0,
        )
        self.idz = IDZModel(self.idz_params, dt)

        # 状态变量
        self.current_level: float = 4.0
        self.current_inflow: float = config.design_flow
        self.current_outflow: float = config.design_flow
        self.gate_opening: float = 1.0  # 闸门开度 [0-1]

        # 季节性糙率调整
        self._manning_n_base = config.manning_n
        self._manning_n_factor = 1.0

    def reset(self, initial_level: float = 4.0, initial_flow: float = None):
        """重置模型状态"""
        self.current_level = initial_level
        flow = initial_flow or self.config.design_flow
        self.current_inflow = flow
        self.current_outflow = flow
        self.gate_opening = 1.0
        self.idz.reset(initial_level, flow)

    def step(self, q_in: float, gate_opening: float = None) -> Tuple[float, float]:
        """
        执行一步模拟

        Args:
            q_in: 入流 [m³/s]
            gate_opening: 闸门开度 [0-1]

        Returns:
            (新水位, 出流)
        """
        if gate_opening is not None:
            self.gate_opening = np.clip(gate_opening, 0.0, 1.0)

        # 计算出流 (基于闸门开度和水位)
        self.current_outflow = self._compute_outflow()

        # 更新水位 (IDZ模型)
        self.current_level = self.idz.step(q_in, self.current_outflow)

        # 限制水位范围
        self.current_level = np.clip(
            self.current_level,
            self.config.min_depth,
            self.config.max_depth
        )

        self.current_inflow = q_in

        return self.current_level, self.current_outflow

    def _compute_outflow(self) -> float:
        """计算出流"""
        # 基于曼宁公式和闸门开度
        max_flow = self.config.manning_flow(
            self.current_level,
            self.config.bed_slope
        )

        # 闸门限制
        q_out = max_flow * self.gate_opening

        # 流量约束
        q_out = np.clip(q_out, 0, self.config.max_flow)

        return q_out

    def set_seasonal_factor(self, factor: float):
        """
        设置季节性因子

        Args:
            factor: 糙率因子 (结冰/藻类时>1)
        """
        self._manning_n_factor = factor
        self.config.manning_n = self._manning_n_base * factor

    def get_state(self) -> Dict[str, float]:
        """获取当前状态"""
        return {
            'pool_id': self.config.pool_id,
            'level': self.current_level,
            'inflow': self.current_inflow,
            'outflow': self.current_outflow,
            'gate_opening': self.gate_opening,
            'manning_n': self.config.manning_n,
        }


# ==============================================================================
# 特殊节点模型
# ==============================================================================

class SpecialNode:
    """
    特殊节点模型 - 倒虹吸、渡槽等特殊建筑物

    特点:
    - 倒虹吸: 高阻尼、大滞后、压力流
    - 渡槽: 流量限制、波动放大
    """

    def __init__(self, structure: SpecialStructure, dt: float = 900.0):
        """
        初始化特殊节点

        Args:
            structure: 特殊建筑物配置
            dt: 时间步长
        """
        self.structure = structure
        self.dt = dt

        # 延迟缓冲
        delay_steps = int(np.ceil(structure.delay_time / dt))
        self._flow_buffer: List[float] = [0.0] * (delay_steps + 1)
        self._buffer_index: int = 0

        # 状态
        self.current_flow: float = 0.0
        self.head_loss: float = 0.0

    def reset(self, initial_flow: float = 100.0):
        """重置状态"""
        self.current_flow = initial_flow
        delay_steps = len(self._flow_buffer)
        self._flow_buffer = [initial_flow] * delay_steps
        self._buffer_index = 0
        self.head_loss = self.structure.compute_head_loss(initial_flow)

    def process_flow(self, q_in: float) -> Tuple[float, float]:
        """
        处理流量通过特殊节点

        Args:
            q_in: 入流 [m³/s]

        Returns:
            (出流, 水头损失)
        """
        # 存储入流
        self._flow_buffer[self._buffer_index] = q_in

        # 获取延迟后的流量
        delay_steps = len(self._flow_buffer) - 1
        if delay_steps > 0:
            delayed_index = (self._buffer_index - delay_steps) % len(self._flow_buffer)
            q_delayed = self._flow_buffer[delayed_index]
        else:
            q_delayed = q_in

        # 流量限制
        q_out = min(q_delayed, self.structure.get_effective_flow_capacity())

        # 计算水头损失
        self.head_loss = self.structure.compute_head_loss(q_out)

        # 更新缓冲区
        self._buffer_index = (self._buffer_index + 1) % len(self._flow_buffer)
        self.current_flow = q_out

        return q_out, self.head_loss


# ==============================================================================
# 南水北调中线全线模型
# ==============================================================================

class SNWDMiddleRouteModel:
    """
    南水北调中线全线模型

    特点:
    - 60+渠池串联
    - 特殊建筑物 (穿黄倒虹吸、渡槽等)
    - 分水口
    - 退水闸

    架构:
    - L3: 全局体积平衡
    - L2: 区域协调
    - L1: 现地控制
    """

    def __init__(self, topology: PoolTopology = None, dt: float = 900.0):
        """
        初始化全线模型

        Args:
            topology: 渠池拓扑 (默认创建标准中线拓扑)
            dt: 时间步长 [s]
        """
        self.dt = dt
        self.topology = topology or PoolTopology.create_snwd_middle_route()

        # 创建渠池模型
        self.pools: Dict[int, CanalPool] = {}
        for config in self.topology.pools:
            self.pools[config.pool_id] = CanalPool(config, dt)

        # 创建特殊节点模型
        self.special_nodes: Dict[str, SpecialNode] = {}
        for structure in self.topology.special_structures:
            self.special_nodes[structure.structure_id] = SpecialNode(structure, dt)

        # 源头入流 (丹江口)
        self.source_inflow: float = 300.0  # m³/s

        # 分水口流量
        self.diversion_flows: Dict[int, float] = {}

        # 全局时间
        self.current_time: float = 0.0

        # 历史记录
        self.history: List[Dict] = []

        logger.info(f"创建南水北调中线模型: {self.topology.num_pools} 渠池, "
                   f"{len(self.special_nodes)} 特殊建筑物")

    def reset(self, initial_level: float = 4.0, initial_flow: float = 300.0):
        """
        重置全线模型

        Args:
            initial_level: 初始水位 [m]
            initial_flow: 初始流量 [m³/s]
        """
        self.current_time = 0.0
        self.source_inflow = initial_flow
        self.history.clear()

        # 重置所有渠池
        for pool in self.pools.values():
            pool.reset(initial_level, initial_flow)

        # 重置特殊节点
        for node in self.special_nodes.values():
            node.reset(initial_flow)

        logger.info(f"模型重置: 初始水位={initial_level}m, 初始流量={initial_flow}m³/s")

    def step(self,
             source_inflow: float = None,
             gate_commands: Dict[int, float] = None,
             diversion_flows: Dict[int, float] = None) -> Dict[str, Any]:
        """
        执行一步全线模拟

        Args:
            source_inflow: 源头入流 [m³/s]
            gate_commands: 闸门开度命令 {pool_id: opening}
            diversion_flows: 分水口流量 {pool_id: flow}

        Returns:
            全线状态
        """
        if source_inflow is not None:
            self.source_inflow = source_inflow

        if diversion_flows is not None:
            self.diversion_flows.update(diversion_flows)

        if gate_commands is None:
            gate_commands = {}

        # 从上游到下游逐池模拟
        states = []
        prev_outflow = self.source_inflow

        for pool_id in range(self.topology.num_pools):
            pool = self.pools[pool_id]

            # 检查是否有特殊建筑物
            q_in = prev_outflow
            for structure in self.topology.get_structures_in_pool(pool_id):
                node = self.special_nodes.get(structure.structure_id)
                if node:
                    q_in, _ = node.process_flow(q_in)

            # 扣除分水
            diversion = self.diversion_flows.get(pool_id, 0.0)
            q_in -= diversion

            # 获取闸门命令
            gate = gate_commands.get(pool_id, pool.gate_opening)

            # 模拟渠池
            level, outflow = pool.step(q_in, gate)

            states.append({
                'pool_id': pool_id,
                'level': level,
                'inflow': q_in,
                'outflow': outflow,
                'gate_opening': gate,
                'diversion': diversion,
            })

            prev_outflow = outflow

        # 更新时间
        self.current_time += self.dt

        # 记录历史
        result = {
            'time': self.current_time,
            'source_inflow': self.source_inflow,
            'pools': states,
            'total_diversion': sum(self.diversion_flows.values()),
            'end_outflow': prev_outflow,
        }
        self.history.append(result)

        return result

    def get_all_levels(self) -> np.ndarray:
        """获取所有渠池水位"""
        return np.array([self.pools[i].current_level
                        for i in range(self.topology.num_pools)])

    def get_all_flows(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取所有渠池流量 (入流, 出流)"""
        inflows = np.array([self.pools[i].current_inflow
                          for i in range(self.topology.num_pools)])
        outflows = np.array([self.pools[i].current_outflow
                           for i in range(self.topology.num_pools)])
        return inflows, outflows

    def get_region_state(self, region_id: int) -> Dict[str, Any]:
        """获取区域状态"""
        region = self.topology.get_region(region_id)
        if region is None:
            return {}

        levels = []
        inflows = []
        outflows = []

        for pool_id in region.pool_ids:
            pool = self.pools[pool_id]
            levels.append(pool.current_level)
            inflows.append(pool.current_inflow)
            outflows.append(pool.current_outflow)

        return {
            'region_id': region_id,
            'region_name': region.name,
            'num_pools': region.num_pools,
            'avg_level': np.mean(levels),
            'total_inflow': inflows[0] if inflows else 0,
            'total_outflow': outflows[-1] if outflows else 0,
            'levels': levels,
            'inflows': inflows,
            'outflows': outflows,
        }

    def set_seasonal_conditions(self, season: str):
        """
        设置季节性条件

        Args:
            season: 'normal', 'ice', 'algae', 'flood'
        """
        factor_map = {
            'normal': 1.0,
            'ice': 1.5,      # 结冰增加糙率
            'algae': 1.2,    # 藻类增加糙率
            'flood': 0.95,   # 高水位略减糙率
        }

        factor = factor_map.get(season, 1.0)
        for pool in self.pools.values():
            pool.set_seasonal_factor(factor)

        logger.info(f"设置季节条件: {season}, 糙率因子={factor}")

    def inject_fault(self, pool_id: int, fault_type: str, **kwargs):
        """
        注入故障

        Args:
            pool_id: 故障渠池ID
            fault_type: 故障类型
            kwargs: 故障参数
        """
        if pool_id not in self.pools:
            return

        pool = self.pools[pool_id]

        if fault_type == 'gate_stuck':
            # 闸门卡死
            stuck_position = kwargs.get('position', pool.gate_opening)
            pool.gate_opening = stuck_position
            # 标记为故障状态 (不响应命令)
            pool._gate_stuck = True

        elif fault_type == 'sensor_bias':
            # 传感器偏差
            bias = kwargs.get('bias', 0.5)
            pool._sensor_bias = bias

        elif fault_type == 'leak':
            # 渗漏
            leak_rate = kwargs.get('rate', 10.0)
            pool._leak_rate = leak_rate

        logger.warning(f"注入故障: 池{pool_id}, 类型={fault_type}")

    def compute_total_volume(self) -> float:
        """计算全线总蓄水量 [m³]"""
        total = 0.0
        for pool in self.pools.values():
            # 简化计算: 水位 × 水面面积
            area = pool.config.surface_area
            total += pool.current_level * area
        return total

    def compute_travel_time(self, from_pool: int, to_pool: int) -> float:
        """计算水流传播时间 [s]"""
        if from_pool >= to_pool:
            return 0.0

        total_time = 0.0
        for i in range(from_pool, to_pool):
            pool = self.pools[i]
            total_time += pool.idz_params.tau

        return total_time

    def get_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        levels = self.get_all_levels()
        inflows, outflows = self.get_all_flows()

        return {
            'time': self.current_time,
            'num_pools': self.topology.num_pools,
            'source_inflow': self.source_inflow,
            'end_outflow': outflows[-1] if len(outflows) > 0 else 0,
            'total_volume': self.compute_total_volume(),
            'avg_level': np.mean(levels),
            'min_level': np.min(levels),
            'max_level': np.max(levels),
            'level_std': np.std(levels),
        }


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" " * 15 + "南水北调中线物理模型测试")
    print("="*70)

    # 创建全线模型
    model = SNWDMiddleRouteModel()
    model.reset(initial_level=4.0, initial_flow=300.0)

    print(f"\n✓ 创建南水北调中线模型")
    print(f"  渠池数量: {model.topology.num_pools}")
    print(f"  特殊建筑物: {len(model.special_nodes)}")

    # 运行模拟
    print(f"\n运行24小时模拟 (时间步长=15分钟)...")

    for hour in range(24):
        for step in range(4):  # 每小时4步
            # 源头流量有波动
            source_flow = 300 + 20 * np.sin(2 * np.pi * hour / 24)

            # 模拟
            state = model.step(source_inflow=source_flow)

        if hour % 6 == 0:
            summary = model.get_summary()
            print(f"  {hour:2d}h: 平均水位={summary['avg_level']:.2f}m, "
                  f"入流={summary['source_inflow']:.0f}m³/s, "
                  f"出流={summary['end_outflow']:.0f}m³/s")

    # 最终状态
    print(f"\n最终状态:")
    summary = model.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # 测试IDZ模型
    print(f"\n{'='*70}")
    print("IDZ模型测试")
    print('='*70)

    idz_params = IDZParameters(tau=14400, A_s=100000)
    idz = IDZModel(idz_params, dt=900)
    idz.reset(initial_level=4.0, initial_inflow=100)

    print(f"  滞后时间: {idz_params.tau/3600:.1f} 小时")
    print(f"  延迟步数: {idz.delay_steps}")
    print(f"  积分增益: {idz_params.integrator_gain:.2e} 1/m²")

    # 阶跃响应
    print(f"\n阶跃响应 (入流从100增加到150):")
    levels = []
    for i in range(20):
        q_in = 150 if i >= 5 else 100
        q_out = 100
        level = idz.step(q_in, q_out)
        levels.append(level)
        if i % 4 == 0:
            print(f"    步骤{i:2d}: 入流={q_in}, 水位={level:.3f}m")

    # 测试区域状态
    print(f"\n{'='*70}")
    print("区域状态测试")
    print('='*70)

    for region_id in range(5):
        state = model.get_region_state(region_id)
        print(f"  {state['region_name']}: {state['num_pools']}池, "
              f"平均水位={state['avg_level']:.2f}m")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
