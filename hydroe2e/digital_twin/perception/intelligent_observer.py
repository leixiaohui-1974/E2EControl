"""
智能感知层 - 数字孪生大脑
Intelligent Observer with Adaptive ID & Cyber Defense
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from hydroe2e.digital_twin.physics.single_channel_fidelity import SingleChannelFidelity, PhysicalState, ChannelGeometry


class OperationMode(Enum):
    """运行模式"""
    NORMAL = "正常模式"
    ADAPTIVE_ID = "自适应辨识中"
    CYBER_DEFENSE = "网络防御模式"
    SLOPE_PROTECTION = "边坡保护模式"
    POLLUTION_EMERGENCY = "污染应急模式"


@dataclass
class RiskAssessment:
    """风险评估结果"""
    slope_risk: np.ndarray  # 边坡风险 (N维)
    water_quality_risk: np.ndarray  # 水质风险
    cyber_risk: bool  # 网络攻击风险
    debris_eta: float  # 漂浮物到达时间 [s]
    max_risk_level: str  # 最高风险等级


@dataclass
class DynamicConstraints:
    """动态约束包"""
    max_dZ_dt: float  # 最大水位变化率 [m/s]
    max_flow_change: float  # 最大流量变化 [m³/s]
    emergency_level: int  # 应急等级 [0-3]
    protected_segments: List[int]  # 需要保护的切片


class IntelligentObserver:
    """
    智能感知层
    
    核心功能:
    1. 自适应参数辨识 (在线学习粗糙度)
    2. 网络安全防御 (FDIA检测)
    3. 多维风险扫描
    4. 动态约束生成
    """
    
    def __init__(self, physical_model: SingleChannelFidelity):
        """
        初始化感知层
        
        Args:
            physical_model: 物理模型引用
        """
        self.model = physical_model
        self.N = physical_model.N
        self.geom = physical_model.geom
        
        # 数字孪生内部模型 (用于参数辨识)
        self.twin_n = np.ones(self.N) * 0.025  # 孪生模型的粗糙度估计
        self.twin_Z_pred = np.zeros(self.N)  # 孪生模型预测的水位
        
        # 自适应辨识参数
        self.learning_rate = 0.01
        self.identification_window = 10  # 辨识窗口长度
        
        # 网络安全参数
        self.cyber_defense_active = False
        self.attack_detected_at = -1
        self.cleaned_state = None
        
        # 风险评估历史
        self.risk_history = []
        
        # 当前运行模式
        self.current_mode = OperationMode.NORMAL
        
        # 漂浮物跟踪
        self.debris_position = None  # 当前位置 [m]
        self.debris_velocity = 0.0  # 速度 [m/s]
        
    def observe_and_analyze(self, measured_state: PhysicalState, 
                           time_step: int) -> Tuple[PhysicalState, RiskAssessment, DynamicConstraints]:
        """
        观测-分析主循环
        
        Args:
            measured_state: 测量状态 (可能被污染)
            time_step: 当前时间步
            
        Returns:
            (清洗后状态, 风险评估, 动态约束)
        """
        # 1. 网络安全防御 - 检测并清洗数据
        cleaned_state, cyber_attack_detected = self._cyber_defense(measured_state, time_step)
        
        # 2. 自适应参数辨识 - 更新粗糙度估计
        if not cyber_attack_detected:
            self._adaptive_identification(cleaned_state)
        
        # 3. 多维风险扫描
        risk = self._scan_risks(cleaned_state, time_step)
        
        # 4. 生成动态约束
        constraints = self._generate_dynamic_constraints(risk)
        
        # 5. 更新运行模式
        self._update_operation_mode(cyber_attack_detected, risk)
        
        return cleaned_state, risk, constraints
    
    def _cyber_defense(self, measured_state: PhysicalState, 
                      time_step: int) -> Tuple[PhysicalState, bool]:
        """
        网络安全防御 - 物理一致性探针
        
        检测逻辑:
        - 检查水位-流量的物理一致性
        - 检查连续性方程残差
        - 检查时间连续性
        """
        cleaned_state = PhysicalState(
            Z=measured_state.Z.copy(),
            Q=measured_state.Q.copy(),
            C=measured_state.C.copy(),
            T_ice=measured_state.T_ice.copy(),
            n_roughness=measured_state.n_roughness.copy()
        )
        
        attack_detected = False
        
        # 如果有足够历史数据
        if len(self.model.history['Z']) > 2:
            prev_Z = self.model.history['Z'][-1]
            prev_Q = self.model.history['Q'][-1]
            
            # 检查每个切片的物理一致性
            for i in range(self.N):
                # 1. 时间连续性检查
                dZ = measured_state.Z[i] - prev_Z[i]
                max_reasonable_dZ = 0.5  # 一个时间步最大变化
                
                if abs(dZ) > max_reasonable_dZ:
                    attack_detected = True
                    suspicious_seg = i
                    
                    # 2. 流量积分检查 (简化)
                    if i > 0 and i < self.N - 1:
                        # 期望的水位变化 (基于流量守恒)
                        dQ = prev_Q[i-1] - prev_Q[i]
                        A = self.model._compute_area(prev_Z[i])
                        expected_dZ = self.model.dt * dQ / A
                        
                        # 如果实测dZ与expected_dZ严重不符
                        if abs(dZ - expected_dZ) > 0.3:
                            logger.info(f"    🔴 物理一致性检测: 切片{i+1}数据异常！")
                            logger.info(f"       实测dZ={dZ:.3f}m, 期望dZ={expected_dZ:.3f}m")
                            
                            # 使用孪生模型推演值替代
                            if self.twin_Z_pred is not None and len(self.twin_Z_pred) == self.N:
                                cleaned_state.Z[i] = self.twin_Z_pred[i]
                                logger.info(f"       ✅ 使用孪生模型清洗值: {cleaned_state.Z[i]:.3f}m")
                            else:
                                # 使用邻近插值
                                if i > 0 and i < self.N - 1:
                                    cleaned_state.Z[i] = (prev_Z[i-1] + prev_Z[i+1]) / 2
                                    logger.info(f"       ✅ 使用邻近插值: {cleaned_state.Z[i]:.3f}m")
        
        if attack_detected and not self.cyber_defense_active:
            self.cyber_defense_active = True
            self.attack_detected_at = time_step
            self.current_mode = OperationMode.CYBER_DEFENSE
            logger.info(f"\n🛡️  启动网络防御模式 (t={time_step})")
        
        elif not attack_detected and self.cyber_defense_active:
            # 攻击结束
            self.cyber_defense_active = False
            logger.info(f"\n✅ 网络防御模式解除 (t={time_step})")
        
        self.cleaned_state = cleaned_state
        
        return cleaned_state, attack_detected
    
    def _adaptive_identification(self, state: PhysicalState):
        """
        自适应参数辨识
        
        使用最小二乘法在线估计粗糙度系数
        """
        if len(self.model.history['Z']) < self.identification_window:
            return
        
        # 获取最近的观测数据
        recent_Z = np.array(self.model.history['Z'][-self.identification_window:])
        recent_Q = np.array(self.model.history['Q'][-self.identification_window:])
        
        # 对每个切片进行辨识
        for i in range(self.N):
            # 使用曼宁公式反算粗糙度
            Z_mean = np.mean(recent_Z[:, i])
            Q_mean = np.mean(recent_Q[:, i])
            
            if Q_mean > 1.0 and Z_mean > 0.5:
                A = self.model._compute_area(Z_mean)
                P = self.model._compute_wetted_perimeter(Z_mean)
                R = A / P if P > 0 else 0
                
                # 水力坡度估计
                if i > 0 and i < self.N - 1:
                    S_f = (recent_Z[-1, i-1] - recent_Z[-1, i+1]) / (2 * self.geom.segment_length())
                    S_f = max(S_f, 1e-6)
                else:
                    S_f = self.geom.slope
                
                # 反算粗糙度: n = (A * R^(2/3) * S^(1/2)) / Q
                n_estimated = (A / Q_mean) * (R ** (2/3)) * (S_f ** 0.5)
                n_estimated = np.clip(n_estimated, 0.015, 0.08)
                
                # 在线更新 (指数平滑)
                self.twin_n[i] = (1 - self.learning_rate) * self.twin_n[i] + \
                                 self.learning_rate * n_estimated
        
        # 更新孪生模型预测
        self._update_twin_prediction(state)
    
    def _update_twin_prediction(self, state: PhysicalState):
        """使用孪生模型预测下一步水位"""
        # 简化：直接使用当前状态
        self.twin_Z_pred = state.Z.copy()
    
    def _scan_risks(self, state: PhysicalState, time_step: int) -> RiskAssessment:
        """
        多维风险扫描
        
        Returns:
            风险评估结果
        """
        # 1. 边坡风险扫描
        slope_safety = self.model.compute_slope_safety_factor()
        slope_risk = np.zeros(self.N)
        
        for i in range(self.N):
            if slope_safety[i] < 1.2:
                slope_risk[i] = 3  # 高风险
            elif slope_safety[i] < 1.5:
                slope_risk[i] = 2  # 中风险
            elif slope_safety[i] < 2.0:
                slope_risk[i] = 1  # 低风险
            else:
                slope_risk[i] = 0  # 无风险
        
        # 2. 水质风险扫描
        water_quality_risk = np.zeros(self.N)
        
        for i in range(self.N):
            if state.C[i] > 50.0:
                water_quality_risk[i] = 3  # 严重污染
            elif state.C[i] > 20.0:
                water_quality_risk[i] = 2  # 中度污染
            elif state.C[i] > 5.0:
                water_quality_risk[i] = 1  # 轻度污染
            else:
                water_quality_risk[i] = 0  # 清洁
        
        # 3. 网络攻击风险
        cyber_risk = self.cyber_defense_active
        
        # 4. 漂浮物视觉跟踪 (简化模拟)
        debris_eta = self._track_debris(state)
        
        # 5. 综合风险等级
        max_slope_risk = np.max(slope_risk)
        max_wq_risk = np.max(water_quality_risk)
        
        if max_slope_risk >= 3 or cyber_risk:
            max_risk_level = "危急"
        elif max_slope_risk >= 2 or max_wq_risk >= 3:
            max_risk_level = "严重"
        elif max_slope_risk >= 1 or max_wq_risk >= 2:
            max_risk_level = "中等"
        else:
            max_risk_level = "低"
        
        risk = RiskAssessment(
            slope_risk=slope_risk,
            water_quality_risk=water_quality_risk,
            cyber_risk=cyber_risk,
            debris_eta=debris_eta,
            max_risk_level=max_risk_level
        )
        
        self.risk_history.append(risk)
        
        return risk
    
    def _track_debris(self, state: PhysicalState) -> float:
        """
        漂浮物视觉跟踪
        
        Returns:
            到达下游的剩余时间 [s]
        """
        if self.debris_position is None:
            return -1.0  # 无漂浮物
        
        # 估计平均流速 (Q/A)
        mean_Q = np.mean(state.Q)
        mean_Z = np.mean(state.Z)
        mean_A = self.model._compute_area(mean_Z)
        mean_velocity = mean_Q / mean_A if mean_A > 0 else 1.0
        
        self.debris_velocity = mean_velocity
        
        # 计算剩余距离
        remaining_distance = self.geom.length - self.debris_position
        
        if remaining_distance <= 0:
            return 0.0  # 已到达
        
        # 剩余时间
        eta = remaining_distance / mean_velocity if mean_velocity > 0 else 9999.0
        
        # 更新位置
        self.debris_position += mean_velocity * self.model.dt
        
        return eta
    
    def inject_debris(self, position: float = 0.0):
        """注入漂浮物"""
        self.debris_position = position
        logger.info(f"⚠️  检测到漂浮物！位置: {position/1000:.1f}km")
    
    def _generate_dynamic_constraints(self, risk: RiskAssessment) -> DynamicConstraints:
        """
        生成动态约束包
        
        根据风险评估结果动态调整约束
        """
        # 基础约束
        max_dZ_dt = 0.01  # 正常情况 [m/s]
        max_flow_change = 20.0  # [m³/s]
        emergency_level = 0
        protected_segments = []
        
        # 根据边坡风险调整
        max_slope_risk = np.max(risk.slope_risk)
        
        if max_slope_risk >= 3:
            # 危急：严格限制退水速度
            max_dZ_dt = 0.002
            max_flow_change = 5.0
            emergency_level = 3
            protected_segments = list(np.where(risk.slope_risk >= 2)[0])
            
        elif max_slope_risk >= 2:
            # 严重：限制退水
            max_dZ_dt = 0.005
            max_flow_change = 10.0
            emergency_level = 2
            protected_segments = list(np.where(risk.slope_risk >= 2)[0])
        
        elif max_slope_risk >= 1:
            # 中等：适度限制
            max_dZ_dt = 0.008
            max_flow_change = 15.0
            emergency_level = 1
        
        # 网络攻击时增加保守性
        if risk.cyber_risk:
            max_flow_change *= 0.5
            emergency_level = max(emergency_level, 2)
        
        return DynamicConstraints(
            max_dZ_dt=max_dZ_dt,
            max_flow_change=max_flow_change,
            emergency_level=emergency_level,
            protected_segments=protected_segments
        )
    
    def _update_operation_mode(self, cyber_attack: bool, risk: RiskAssessment):
        """更新运行模式"""
        # 判断是否需要辨识
        n_diff = np.max(np.abs(self.twin_n - 0.025))
        
        if cyber_attack:
            self.current_mode = OperationMode.CYBER_DEFENSE
        elif np.max(risk.slope_risk) >= 2:
            self.current_mode = OperationMode.SLOPE_PROTECTION
        elif np.max(risk.water_quality_risk) >= 3:
            self.current_mode = OperationMode.POLLUTION_EMERGENCY
        elif n_diff > 0.01:
            self.current_mode = OperationMode.ADAPTIVE_ID
        else:
            self.current_mode = OperationMode.NORMAL
    
    def get_identification_result(self) -> Dict:
        """获取参数辨识结果"""
        return {
            'estimated_n': self.twin_n.copy(),
            'true_n': self.model.state.n_roughness.copy(),
            'error': np.abs(self.twin_n - self.model.state.n_roughness)
        }
    
    def get_current_mode(self) -> OperationMode:
        """获取当前运行模式"""
        return self.current_mode


# 演示
if __name__ == "__main__":
    logger.info("="*80)
    logger.info(" "*20 + "智能感知层演示")
    logger.info("="*80)
    
    from hydroe2e.digital_twin.physics.single_channel_fidelity import ChannelGeometry, SingleChannelFidelity
    
    # 创建物理模型
    geom = ChannelGeometry()
    physical_model = SingleChannelFidelity(geom)
    
    # 创建感知层
    observer = IntelligentObserver(physical_model)
    
    logger.info(f"\n初始化完成")
    logger.info(f"  运行模式: {observer.current_mode.value}")
    
    # 仿真几步
    for t in range(5):
        u_in = 50.0
        u_out = 48.0
        
        physical_model.step(u_in, u_out)
        
        measured_state = physical_model.get_measured_state()
        cleaned_state, risk, constraints = observer.observe_and_analyze(measured_state, t)
        
        logger.info(f"\nt={t}:")
        logger.info(f"  风险等级: {risk.max_risk_level}")
        logger.info(f"  运行模式: {observer.current_mode.value}")
        logger.info(f"  约束: 最大dZ/dt={constraints.max_dZ_dt:.4f}m/s")
    
    logger.info("\n✅ 感知层演示完成！")
    logger.info("="*80)
