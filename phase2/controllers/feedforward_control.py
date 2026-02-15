"""
前馈补偿控制器
预测性扰动抑制，提升控制性能
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class DisturbanceForecast:
    """扰动预测"""
    upstream_flow: List[float]      # 上游来流预测
    demand: List[float]              # 需求预测
    rainfall: List[float]            # 降雨预测
    evaporation: List[float]         # 蒸发预测
    horizon: int                     # 预测时域


class FeedforwardController:
    """前馈控制器"""
    
    def __init__(self, pool_id: int, dt: float = 3600.0, area: float = 10000.0):
        """
        初始化前馈控制器
        
        Args:
            pool_id: 渠池ID
            dt: 时间步长
            area: 渠池面积
        """
        self.pool_id = pool_id
        self.dt = dt
        self.area = area
        
        # 扰动历史
        self.disturbance_history = deque(maxlen=100)
        
        # 前馈增益（通过学习自适应调整）
        self.feedforward_gains = {
            'upstream_flow': 0.8,    # 上游流量扰动增益
            'demand': 1.0,           # 需求扰动增益
            'rainfall': 0.3,         # 降雨扰动增益
            'evaporation': 0.1       # 蒸发扰动增益
        }
        
        # 延迟补偿参数
        self.delay = 1  # 时间步数
        self.delay_buffer = deque([0.0] * self.delay, maxlen=self.delay)
    
    def compute_feedforward(self, 
                           disturbance: DisturbanceForecast,
                           current_state: Dict) -> List[float]:
        """
        计算前馈补偿
        
        Args:
            disturbance: 扰动预测
            current_state: 当前状态
            
        Returns:
            前馈补偿序列
        """
        horizon = disturbance.horizon
        feedforward = np.zeros(horizon)
        
        for k in range(horizon):
            # 1. 上游流量扰动补偿
            if k < len(disturbance.upstream_flow):
                # 预测上游流量变化
                upstream_delta = (disturbance.upstream_flow[k] - 
                                 current_state.get('q_in', 5.0))
                feedforward[k] += self.feedforward_gains['upstream_flow'] * upstream_delta
            
            # 2. 需求扰动补偿
            if k < len(disturbance.demand):
                demand_delta = (disturbance.demand[k] - 
                               current_state.get('q_out', 5.0))
                feedforward[k] += self.feedforward_gains['demand'] * demand_delta
            
            # 3. 降雨扰动补偿
            if k < len(disturbance.rainfall):
                # 降雨转化为入流增量
                rainfall_inflow = (disturbance.rainfall[k] * self.area / 
                                  1000.0 / self.dt)  # mm -> m³/s
                feedforward[k] -= self.feedforward_gains['rainfall'] * rainfall_inflow
            
            # 4. 蒸发扰动补偿
            if k < len(disturbance.evaporation):
                # 蒸发转化为水量损失
                evap_loss = (disturbance.evaporation[k] * self.area / 
                            1000.0 / self.dt)  # mm -> m³/s
                feedforward[k] += self.feedforward_gains['evaporation'] * evap_loss
        
        return feedforward.tolist()
    
    def apply_feedforward(self, 
                         feedback_control: float,
                         feedforward_signal: float,
                         alpha: float = 0.7) -> float:
        """
        应用前馈补偿
        
        Args:
            feedback_control: 反馈控制量
            feedforward_signal: 前馈信号
            alpha: 前馈权重（0-1）
            
        Returns:
            组合控制量
        """
        # 延迟补偿
        self.delay_buffer.append(feedforward_signal)
        delayed_ff = self.delay_buffer[0]
        
        # 组合控制
        total_control = (1 - alpha) * feedback_control + alpha * delayed_ff
        
        return total_control
    
    def update_gains(self, 
                    actual_disturbance: Dict,
                    predicted_disturbance: Dict,
                    performance_error: float):
        """
        自适应更新前馈增益
        
        Args:
            actual_disturbance: 实际扰动
            predicted_disturbance: 预测扰动
            performance_error: 性能误差
        """
        learning_rate = 0.01
        
        # 简单的梯度下降更新
        for key in self.feedforward_gains.keys():
            if key in actual_disturbance and key in predicted_disturbance:
                prediction_error = (actual_disturbance[key] - 
                                  predicted_disturbance[key])
                
                # 如果预测准确但性能差，说明增益需要调整
                if abs(prediction_error) < 0.1 and abs(performance_error) > 0.1:
                    # 根据误差符号调整增益
                    self.feedforward_gains[key] += (
                        learning_rate * np.sign(performance_error) * 
                        np.sign(predicted_disturbance[key])
                    )
                    
                    # 限制增益范围
                    self.feedforward_gains[key] = np.clip(
                        self.feedforward_gains[key], 0.0, 2.0
                    )
        
        # 记录历史
        self.disturbance_history.append({
            'actual': actual_disturbance.copy(),
            'predicted': predicted_disturbance.copy(),
            'error': performance_error
        })


class DemandPredictor:
    """需求预测器"""
    
    def __init__(self, horizon: int = 24):
        """
        初始化预测器
        
        Args:
            horizon: 预测时域
        """
        self.horizon = horizon
        self.history = deque(maxlen=168)  # 一周历史
        self.seasonal_pattern = None
    
    def add_observation(self, demand: float, timestamp: int):
        """添加观测值"""
        self.history.append({'demand': demand, 'time': timestamp})
    
    def predict(self, current_time: int) -> List[float]:
        """
        预测未来需求
        
        Args:
            current_time: 当前时间戳
            
        Returns:
            需求预测序列
        """
        if len(self.history) < 24:
            # 历史不足，返回平均值
            avg = np.mean([h['demand'] for h in self.history]) if self.history else 5.0
            return [avg] * self.horizon
        
        # 简单时序模型：周期性 + 趋势
        history_values = np.array([h['demand'] for h in self.history])
        
        # 识别周期模式（24小时）
        if len(history_values) >= 24:
            daily_pattern = np.zeros(24)
            for i in range(24):
                indices = list(range(i, len(history_values), 24))
                if indices:
                    daily_pattern[i] = np.mean(history_values[indices])
            
            # 预测
            forecast = []
            for k in range(self.horizon):
                hour_of_day = (current_time + k) % 24
                base_demand = daily_pattern[hour_of_day]
                
                # 添加趋势（线性回归）
                if len(history_values) > 48:
                    recent = history_values[-48:]
                    trend = (recent[-1] - recent[0]) / len(recent)
                    base_demand += trend * k
                
                forecast.append(max(0, base_demand))
            
            return forecast
        else:
            # 移动平均
            ma = np.mean(history_values[-24:])
            return [ma] * self.horizon
    
    def get_uncertainty(self) -> float:
        """获取预测不确定性"""
        if len(self.history) < 24:
            return 1.0
        
        history_values = np.array([h['demand'] for h in self.history])
        return float(np.std(history_values[-24:]))


class WeatherForecast:
    """天气预报接口"""
    
    def __init__(self):
        """初始化天气预报"""
        self.rainfall_history = deque(maxlen=100)
        self.evaporation_history = deque(maxlen=100)
    
    def get_forecast(self, horizon: int = 24) -> Dict[str, List[float]]:
        """
        获取天气预报
        
        Args:
            horizon: 预测时域
            
        Returns:
            天气预报字典
        """
        # 这里应该调用外部天气API，这里用简单模型模拟
        
        # 降雨预测（mm/h）
        if self.rainfall_history:
            avg_rain = np.mean(self.rainfall_history)
            std_rain = np.std(self.rainfall_history)
        else:
            avg_rain = 0.0
            std_rain = 5.0
        
        rainfall = np.random.normal(avg_rain, std_rain, horizon)
        rainfall = np.maximum(0, rainfall)  # 非负
        
        # 蒸发预测（mm/h）
        if self.evaporation_history:
            avg_evap = np.mean(self.evaporation_history)
            std_evap = np.std(self.evaporation_history)
        else:
            avg_evap = 0.5
            std_evap = 0.2
        
        evaporation = np.random.normal(avg_evap, std_evap, horizon)
        evaporation = np.maximum(0, evaporation)
        
        return {
            'rainfall': rainfall.tolist(),
            'evaporation': evaporation.tolist()
        }
    
    def update(self, rainfall: float, evaporation: float):
        """更新观测值"""
        self.rainfall_history.append(rainfall)
        self.evaporation_history.append(evaporation)


class IntegratedFeedforwardMPC:
    """集成前馈的MPC控制器"""
    
    def __init__(self, pool_id: int, horizon: int = 10):
        """
        初始化集成控制器
        
        Args:
            pool_id: 渠池ID
            horizon: 预测时域
        """
        self.pool_id = pool_id
        self.horizon = horizon
        
        # 子模块
        self.ff_controller = FeedforwardController(pool_id)
        self.demand_predictor = DemandPredictor(horizon)
        self.weather_forecast = WeatherForecast()
    
    def compute_control(self, 
                       current_state: Dict,
                       feedback_control: float,
                       current_time: int) -> Tuple[float, Dict]:
        """
        计算带前馈补偿的控制量
        
        Args:
            current_state: 当前状态
            feedback_control: 反馈控制量
            current_time: 当前时间
            
        Returns:
            (总控制量, 调试信息)
        """
        # 1. 需求预测
        demand_forecast = self.demand_predictor.predict(current_time)
        
        # 2. 天气预报
        weather = self.weather_forecast.get_forecast(self.horizon)
        
        # 3. 构造扰动预测
        disturbance = DisturbanceForecast(
            upstream_flow=[current_state.get('q_in', 5.0)] * self.horizon,
            demand=demand_forecast,
            rainfall=weather['rainfall'],
            evaporation=weather['evaporation'],
            horizon=self.horizon
        )
        
        # 4. 计算前馈补偿
        feedforward_signals = self.ff_controller.compute_feedforward(
            disturbance, current_state
        )
        
        # 5. 应用前馈（当前时刻）
        total_control = self.ff_controller.apply_feedforward(
            feedback_control, 
            feedforward_signals[0],
            alpha=0.7
        )
        
        # 调试信息
        debug_info = {
            'feedback': feedback_control,
            'feedforward': feedforward_signals[0],
            'total': total_control,
            'demand_forecast': demand_forecast,
            'gains': self.ff_controller.feedforward_gains.copy()
        }
        
        return total_control, debug_info


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*20 + "前馈补偿控制器演示")
    logger.info("="*70)
    
    # 创建控制器
    ff_mpc = IntegratedFeedforwardMPC(pool_id=0, horizon=10)
    
    # 模拟场景
    logger.info("\n1. 模拟需求突变场景...")
    
    # 训练需求预测器
    for t in range(72):
        # 模拟日常需求模式
        hour = t % 24
        if 8 <= hour <= 18:
            demand = 8.0 + np.random.normal(0, 0.5)  # 白天高需求
        else:
            demand = 4.0 + np.random.normal(0, 0.3)  # 夜间低需求
        
        ff_mpc.demand_predictor.add_observation(demand, t)
    
    # 测试控制
    current_time = 72
    current_state = {'level': 3.0, 'q_in': 5.0, 'q_out': 5.0}
    
    logger.info("\n2. 前馈补偿效果测试...")
    for step in range(10):
        # 反馈控制（这里用简单P控制模拟）
        level_error = current_state['level'] - 3.0
        feedback = 5.0 - 2.0 * level_error
        
        # 计算带前馈的总控制
        total_control, debug = ff_mpc.compute_control(
            current_state, feedback, current_time + step
        )
        
        logger.info(f"\n   步骤 {step + 1}:")
        logger.info(f"     反馈控制: {debug['feedback']:.2f}")
        logger.info(f"     前馈补偿: {debug['feedforward']:.2f}")
        logger.info(f"     总控制量: {debug['total']:.2f}")
        
        if step == 0:
            logger.info(f"     需求预测: {debug['demand_forecast'][:5]}")
        
        # 更新状态（简化）
        current_state['q_in'] = total_control
        current_state['level'] += (total_control - current_state['q_out']) * 0.0001
    
    logger.info("\n3. 前馈增益...")
    for key, gain in debug['gains'].items():
        logger.info(f"   {key}: {gain:.3f}")
    
    logger.info("\n" + "="*70)
    logger.info("演示完成！")
    logger.info("="*70)
