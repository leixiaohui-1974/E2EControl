"""
级联渠道系统模型
Phase 2: 多池级联控制
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class Gate:
    """闸门模型"""
    
    def __init__(self, gate_id: str, max_flow: float = 20.0, response_time: float = 60.0):
        """
        初始化闸门
        
        Args:
            gate_id: 闸门ID
            max_flow: 最大流量 (m³/s)
            response_time: 响应时间 (秒)
        """
        self.gate_id = gate_id
        self.max_flow = max_flow
        self.response_time = response_time
        
        self.current_opening = 0.5  # 当前开度 (0-1)
        self.target_opening = 0.5   # 目标开度
        self.flow_coefficient = max_flow  # 流量系数
    
    def set_opening(self, opening: float):
        """设置闸门开度"""
        self.target_opening = np.clip(opening, 0.0, 1.0)
    
    def step(self, dt: float) -> float:
        """一步仿真
        
        Args:
            dt: 时间步长
            
        Returns:
            实际流量
        """
        # 一阶惯性环节模拟闸门响应
        tau = self.response_time
        alpha = dt / (tau + dt)
        
        self.current_opening += alpha * (self.target_opening - self.current_opening)
        
        # 计算流量
        flow = self.current_opening * self.flow_coefficient
        
        return flow
    
    def get_flow(self) -> float:
        """获取当前流量"""
        return self.current_opening * self.flow_coefficient


class CascadedPool:
    """级联渠池单元"""
    
    def __init__(self, pool_id: str, area: float, dt: float, initial_level: float = 3.0):
        """
        初始化渠池
        
        Args:
            pool_id: 渠池ID
            area: 面积 (m²)
            dt: 时间步长 (秒)
            initial_level: 初始水位 (m)
        """
        self.pool_id = pool_id
        self.area = area
        self.dt = dt
        self.level = initial_level
        self.volume = initial_level * area
        
        # 流量历史（用于延迟模拟）
        self.inflow_history = deque([0.0], maxlen=5)
        self.outflow_history = deque([0.0], maxlen=5)
    
    def step(self, q_in: float, q_out: float) -> float:
        """
        一步演化
        
        Args:
            q_in: 入流 (m³/s)
            q_out: 出流 (m³/s)
            
        Returns:
            新水位 (m)
        """
        # 记录历史
        self.inflow_history.append(q_in)
        self.outflow_history.append(q_out)
        
        # 体积变化
        delta_v = (q_in - q_out) * self.dt
        self.volume += delta_v
        
        # 更新水位
        self.level = self.volume / self.area
        
        # 物理约束
        self.level = max(0.0, self.level)
        self.volume = self.level * self.area
        
        return self.level
    
    def get_state(self) -> Dict:
        """获取状态"""
        return {
            'pool_id': self.pool_id,
            'level': self.level,
            'volume': self.volume,
            'inflow': self.inflow_history[-1],
            'outflow': self.outflow_history[-1]
        }


class CascadedCanalSystem:
    """级联渠道系统
    
    系统结构:
    [上游] → [闸门0] → [渠池0] → [闸门1] → [渠池1] → [闸门2] → [渠池2] → [闸门3] → [下游]
    """
    
    def __init__(self, num_pools: int = 3, pool_area: float = 10000.0, dt: float = 3600.0):
        """
        初始化级联系统
        
        Args:
            num_pools: 渠池数量
            pool_area: 渠池面积 (m²)
            dt: 时间步长 (秒)
        """
        self.num_pools = num_pools
        self.dt = dt
        
        # 创建渠池
        self.pools = [
            CascadedPool(
                pool_id=f"pool_{i}",
                area=pool_area,
                dt=dt,
                initial_level=3.0
            )
            for i in range(num_pools)
        ]
        
        # 创建闸门（比渠池多1个）
        self.gates = [
            Gate(gate_id=f"gate_{i}", max_flow=20.0)
            for i in range(num_pools + 1)
        ]
        
        # 流量传播延迟（简化：每个渠池1步延迟）
        self.propagation_delay = [
            deque([5.0], maxlen=2) for _ in range(num_pools)
        ]
        
        # 系统状态
        self.time = 0
        self.history = []
    
    def step(self, control_actions: List[float], demand: float = 5.0):
        """
        一步仿真
        
        Args:
            control_actions: 各闸门开度 [g0_opening, g1_opening, ...]
            demand: 下游需求 (m³/s)
            
        Returns:
            系统状态
        """
        # 设置闸门开度
        for gate, action in zip(self.gates, control_actions):
            gate.set_opening(action)
        
        # 闸门响应（计算实际流量）
        gate_flows = [gate.step(self.dt) for gate in self.gates]
        
        # 渠池水力演化（从上游到下游）
        for i, pool in enumerate(self.pools):
            q_in = gate_flows[i]
            q_out = gate_flows[i + 1]
            
            # 考虑流量传播延迟
            self.propagation_delay[i].append(q_in)
            delayed_q_in = self.propagation_delay[i][0]
            
            pool.step(delayed_q_in, q_out)
        
        # 更新时间
        self.time += self.dt
        
        # 记录状态
        state = self.get_system_state()
        state['demand'] = demand
        self.history.append(state)
        
        return state
    
    def get_system_state(self) -> Dict:
        """获取系统完整状态"""
        return {
            'time': self.time,
            'pools': [pool.get_state() for pool in self.pools],
            'gates': [
                {
                    'gate_id': gate.gate_id,
                    'opening': gate.current_opening,
                    'flow': gate.get_flow()
                }
                for gate in self.gates
            ]
        }
    
    def get_state_vector(self) -> np.ndarray:
        """获取状态向量（用于MPC）
        
        Returns:
            [level_0, level_1, ..., level_n-1, flow_0, flow_1, ..., flow_n]
        """
        levels = [pool.level for pool in self.pools]
        flows = [gate.get_flow() for gate in self.gates]
        return np.array(levels + flows)
    
    def reset(self):
        """重置系统"""
        for pool in self.pools:
            pool.level = 3.0
            pool.volume = 3.0 * pool.area
        
        for gate in self.gates:
            gate.current_opening = 0.5
            gate.target_opening = 0.5
        
        self.time = 0
        self.history = []


# 示例使用
if __name__ == "__main__":
    # 创建3池级联系统
    system = CascadedCanalSystem(num_pools=3, pool_area=10000.0, dt=3600.0)
    
    logger.info("="*60)
    logger.info("级联渠道系统仿真示例")
    logger.info("="*60)
    logger.info(f"渠池数量: {system.num_pools}")
    logger.info(f"闸门数量: {len(system.gates)}")
    logger.info()
    
    # 运行10步仿真
    logger.info("运行仿真...")
    for t in range(10):
        # 简单控制策略：所有闸门开度0.5
        control_actions = [0.5] * (system.num_pools + 1)
        
        # 仿真一步
        state = system.step(control_actions, demand=5.0)
        
        # 打印状态
        if t % 2 == 0:
            logger.info(f"\n时间步 {t}:")
            for i, pool_state in enumerate(state['pools']):
                logger.info(f"  池{i}: 水位={pool_state['level']:.3f}m, "
                      f"入流={pool_state['inflow']:.2f}m³/s, "
                      f"出流={pool_state['outflow']:.2f}m³/s")
    
    logger.info("\n" + "="*60)
    logger.info("仿真完成！")
    logger.info("="*60)
