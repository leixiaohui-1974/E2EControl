"""
高精度单渠池物理本体 - 分布式参数模拟
Single Channel Fidelity Model with Distributed Parameters
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChannelGeometry:
    """渠道几何参数"""
    length: float = 20000.0  # 总长度 [m]
    N: int = 20  # 空间离散切片数
    width: float = 50.0  # 渠道宽度 [m]
    slope: float = 0.0001  # 底坡
    side_slope: float = 2.0  # 边坡系数 (水平:垂直)
    
    def segment_length(self) -> float:
        """每个切片的长度"""
        return self.length / self.N
    
    def x_positions(self) -> np.ndarray:
        """各切片中心位置"""
        dx = self.segment_length()
        return np.linspace(dx/2, self.length - dx/2, self.N)


@dataclass
class PhysicalState:
    """完整物理状态 (N x 5维)"""
    Z: np.ndarray  # 水位 [m]
    Q: np.ndarray  # 流量 [m³/s]
    C: np.ndarray  # 污染物浓度 [mg/L]
    T_ice: np.ndarray  # 冰层厚度 [m]
    n_roughness: np.ndarray  # 曼宁粗糙度系数


class SingleChannelFidelity:
    """
    高精度单渠池物理模型
    
    核心特性:
    1. 空间离散 (N=20切片)
    2. 空间异质性 (局部水草生长)
    3. 多维状态耦合
    4. 攻击注入能力
    """
    
    def __init__(self, geometry: ChannelGeometry, dt: float = 60.0):
        """
        初始化物理模型
        
        Args:
            geometry: 渠道几何参数
            dt: 时间步长 [s]
        """
        self.geom = geometry
        self.dt = dt
        self.N = geometry.N
        
        # 初始化状态
        self.state = self._initialize_state()
        
        # 底高程分布 (带微小起伏)
        self.z_bed = self._generate_bed_profile()
        
        # 物理参数
        self.g = 9.81  # 重力加速度 [m/s²]
        
        # 侧向入流/出流 (每个切片的净入流 [m³/s])
        self.lateral_inflow = np.zeros(self.N)
        
        # 网络攻击参数
        self.attack_active = False
        self.attack_segment = 9  # 第10切片 (index=9)
        self.attack_bias = 0.0  # 攻击偏差 [m]
        
        # 历史记录
        self.history = {
            'Z': [],
            'Q': [],
            'C': [],
            'n': [],
            'lateral': [],
            'attack': []
        }
        
    def _initialize_state(self) -> PhysicalState:
        """初始化物理状态"""
        N = self.N
        
        # 初始水位 (相对河底) - 接近正常水深
        Z_init = 3.0 + 0.1 * np.random.randn(N)
        Z_init = np.clip(Z_init, 2.5, 3.5)
        
        # 初始流量 - 均匀分布
        Q_init = np.ones(N) * 50.0
        
        # 污染物浓度 - 初始为0
        C_init = np.zeros(N)
        
        # 冰层厚度 - 无冰期
        T_ice_init = np.zeros(N)
        
        # 曼宁粗糙度 - 初始均匀
        n_init = np.ones(N) * 0.025  # 清洁混凝土渠道
        
        return PhysicalState(
            Z=Z_init,
            Q=Q_init,
            C=C_init,
            T_ice=T_ice_init,
            n_roughness=n_init
        )
    
    def _generate_bed_profile(self) -> np.ndarray:
        """生成河底高程分布"""
        x = self.geom.x_positions()
        
        # 基础坡度
        z_bed = 100.0 - self.geom.slope * x
        
        # 添加微小起伏 (模拟自然河床)
        perturbation = 0.05 * np.sin(2 * np.pi * x / 5000.0)
        z_bed += perturbation
        
        return z_bed
    
    def grow_vegetation(self, start_seg: int = 9, end_seg: int = 14, 
                       growth_factor: float = 1.8):
        """
        模拟水草生长 (局部阻力增大)
        
        Args:
            start_seg: 起始切片 (第10个, index=9)
            end_seg: 结束切片 (第15个, index=14)
            growth_factor: 粗糙度增长倍数
        """
        for i in range(start_seg, end_seg + 1):
            self.state.n_roughness[i] = 0.025 * growth_factor
        
        logger.info(f"💚 水草生长：切片 {start_seg+1}-{end_seg+1}，粗糙度增至 {0.025*growth_factor:.4f}")
    
    def inject_attack(self, active: bool = True, bias: float = -0.5):
        """
        注入网络攻击 (虚假数据注入)
        
        Args:
            active: 是否激活攻击
            bias: 水位偏差 [m] (负值表示虚假降低)
        """
        self.attack_active = active
        self.attack_bias = bias
        
        if active:
            logger.info(f"🔴 网络攻击激活：中游水位传感器被篡改 (偏差 {bias:.2f}m)")
        else:
            logger.info(f"🟢 网络攻击解除")
    
    def set_lateral_inflow(self, lateral: np.ndarray):
        """设置侧向入流分布"""
        self.lateral_inflow = lateral.copy()
    
    def get_measured_state(self) -> PhysicalState:
        """
        获取测量状态 (可能被攻击污染)
        
        Returns:
            测量状态 (包含可能的攻击偏差)
        """
        measured = PhysicalState(
            Z=self.state.Z.copy(),
            Q=self.state.Q.copy(),
            C=self.state.C.copy(),
            T_ice=self.state.T_ice.copy(),
            n_roughness=self.state.n_roughness.copy()
        )
        
        # 如果攻击激活，污染中游水位数据
        if self.attack_active:
            measured.Z[self.attack_segment] += self.attack_bias
        
        return measured
    
    def step(self, u_in: float, u_out: float, 
             pollution_source: Optional[Tuple[int, float]] = None):
        """
        物理演化一步 (圣维南方程 + 对流扩散)
        
        Args:
            u_in: 上游控制闸开度 [m³/s]
            u_out: 下游泵站抽水量 [m³/s]
            pollution_source: 污染源 (切片索引, 污染负荷 [mg/L·m³/s])
        """
        Z = self.state.Z
        Q = self.state.Q
        C = self.state.C
        n = self.state.n_roughness
        
        dx = self.geom.segment_length()
        dt = self.dt
        
        # 1. 水位更新 (连续性方程)
        Z_new = Z.copy()
        
        # 上游边界
        A_0 = self._compute_area(Z[0])
        dQ_0 = u_in - Q[0]
        Z_new[0] = Z[0] + dt * (dQ_0 + self.lateral_inflow[0]) / A_0
        
        # 中间段
        for i in range(1, self.N - 1):
            A_i = self._compute_area(Z[i])
            dQ = Q[i-1] - Q[i] + self.lateral_inflow[i]
            Z_new[i] = Z[i] + dt * dQ / A_i
        
        # 下游边界
        A_N = self._compute_area(Z[-1])
        dQ_N = Q[-2] - u_out + self.lateral_inflow[-1]
        Z_new[-1] = Z[-1] + dt * dQ_N / A_N
        
        # 2. 流量更新 (动量方程 - 简化为曼宁公式)
        Q_new = Q.copy()
        
        for i in range(self.N):
            # 水力半径
            A = self._compute_area(Z_new[i])
            P = self._compute_wetted_perimeter(Z_new[i])
            R = A / P if P > 0 else 0
            
            # 水力坡度 (简化)
            if i == 0:
                S_f = self.geom.slope
            elif i == self.N - 1:
                S_f = (Z_new[i-1] - Z_new[i]) / dx
            else:
                S_f = (Z_new[i-1] - Z_new[i+1]) / (2 * dx)
            
            S_f = max(S_f, 1e-6)  # 防止负坡度
            
            # 曼宁公式
            Q_new[i] = (A / n[i]) * (R ** (2/3)) * (S_f ** 0.5)
        
        # 3. 污染物浓度更新 (对流扩散)
        C_new = C.copy()
        
        # 简化为纯对流
        for i in range(1, self.N):
            if Q_new[i] > 0:
                C_new[i] = C[i] + dt * (Q_new[i-1] / A) * (C[i-1] - C[i]) / dx
        
        # 污染源注入
        if pollution_source is not None:
            seg_idx, load = pollution_source
            if 0 <= seg_idx < self.N:
                A = self._compute_area(Z_new[seg_idx])
                C_new[seg_idx] += dt * load / A
        
        # 自然降解 (一阶动力学)
        k_decay = 0.0001  # 降解系数 [1/s]
        C_new = C_new * np.exp(-k_decay * dt)
        
        # 4. 更新状态
        self.state.Z = np.clip(Z_new, 0.5, 6.0)  # 物理约束
        self.state.Q = np.clip(Q_new, 0.0, 200.0)
        self.state.C = np.clip(C_new, 0.0, 1000.0)
        
        # 记录历史
        self.history['Z'].append(self.state.Z.copy())
        self.history['Q'].append(self.state.Q.copy())
        self.history['C'].append(self.state.C.copy())
        self.history['n'].append(self.state.n_roughness.copy())
        self.history['lateral'].append(self.lateral_inflow.copy())
        self.history['attack'].append(self.attack_active)
    
    def _compute_area(self, depth: float) -> float:
        """计算过水断面积 (梯形断面)"""
        if depth <= 0:
            return 0.1  # 防止除零
        
        b = self.geom.width
        m = self.geom.side_slope
        
        return b * depth + m * depth**2
    
    def _compute_wetted_perimeter(self, depth: float) -> float:
        """计算湿周"""
        if depth <= 0:
            return self.geom.width
        
        b = self.geom.width
        m = self.geom.side_slope
        
        return b + 2 * depth * np.sqrt(1 + m**2)
    
    def get_surface_elevation(self) -> np.ndarray:
        """获取水面高程 (绝对高程)"""
        return self.z_bed + self.state.Z
    
    def compute_slope_safety_factor(self) -> np.ndarray:
        """
        计算边坡安全系数
        
        简化公式: Fs = c / (γ * H * tanφ) + (1 - γ_w * H_w / γ * H) * tanφ / tanβ
        """
        Z = self.state.Z
        
        # 简化模型：安全系数与水位变化率相关
        if len(self.history['Z']) < 2:
            return np.ones(self.N) * 3.0  # 初始安全
        
        dZ_dt = (self.state.Z - self.history['Z'][-1]) / self.dt
        
        # 急剧退水降低安全系数
        Fs = 2.5 - 5.0 * np.clip(dZ_dt, -0.05, 0)
        
        return np.clip(Fs, 0.5, 3.0)
    
    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        return {
            'mean_level': np.mean(self.state.Z),
            'max_level': np.max(self.state.Z),
            'min_level': np.min(self.state.Z),
            'mean_flow': np.mean(self.state.Q),
            'max_concentration': np.max(self.state.C),
            'attack_active': self.attack_active,
            'vegetation_roughness': np.max(self.state.n_roughness)
        }


# 演示
if __name__ == "__main__":
    logger.info("="*80)
    logger.info(" "*20 + "单渠池高精度物理模型演示")
    logger.info("="*80)
    
    # 创建物理模型
    geom = ChannelGeometry()
    model = SingleChannelFidelity(geom, dt=60.0)
    
    logger.info(f"\n渠道参数:")
    logger.info(f"  总长度: {geom.length/1000:.1f} km")
    logger.info(f"  切片数: {geom.N}")
    logger.info(f"  切片长度: {geom.segment_length():.0f} m")
    logger.info(f"  渠道宽度: {geom.width:.0f} m")
    
    # 仿真10步
    logger.info(f"\n运行仿真...")
    for t in range(10):
        u_in = 50.0 + 10.0 * np.sin(0.1 * t)
        u_out = 48.0
        
        model.step(u_in, u_out)
        
        if t % 3 == 0:
            stats = model.get_statistics()
            logger.info(f"  t={t:3d}: 平均水位={stats['mean_level']:.2f}m, "
                  f"平均流量={stats['mean_flow']:.1f}m³/s")
    
    logger.info("\n✅ 物理模型演示完成！")
    logger.info("="*80)
