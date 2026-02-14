"""
系统辨识模块
System Identification Module

从历史数据中辨识IDZ模型参数:
- 滞后时间 (tau): 使用互相关分析
- 蓄水面积 (A_s): 使用递推最小二乘法 (RLS)

方法:
1. Cross-Correlation Analysis: 确定滞后时间
2. Recursive Least Squares (RLS): 在线参数估计
3. Kalman Filter: 状态估计与参数跟踪
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging

from .physics_model import IDZParameters

logger = logging.getLogger(__name__)


# ==============================================================================
# 辨识结果数据结构
# ==============================================================================

@dataclass
class IdentificationResult:
    """辨识结果"""
    pool_id: int
    timestamp: float

    # IDZ参数
    tau: float                 # 滞后时间 [s]
    A_s: float                 # 蓄水面积 [m²]
    c_in: float = 1.0          # 入流增益
    c_out: float = 1.0         # 出流增益

    # 辨识质量
    confidence: float = 0.0    # 置信度 [0-1]
    r_squared: float = 0.0     # 决定系数
    rmse: float = 0.0          # 均方根误差

    # 元数据
    samples_used: int = 0
    method: str = "unknown"

    def to_idz_params(self) -> IDZParameters:
        """转换为IDZ参数"""
        return IDZParameters(
            tau=self.tau,
            A_s=self.A_s,
            c_in=self.c_in,
            c_out=self.c_out,
        )


# ==============================================================================
# 互相关分析 (Cross-Correlation)
# ==============================================================================

class CrossCorrelationAnalyzer:
    """
    互相关分析器 - 用于确定滞后时间

    计算入流Q_in和下游水位Z_down之间的互相关函数，
    峰值对应的滞后即为传播延迟。
    """

    def __init__(self, max_lag: int = 100):
        """
        初始化

        Args:
            max_lag: 最大滞后步数
        """
        self.max_lag = max_lag

    def compute_correlation(self,
                           signal1: np.ndarray,
                           signal2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算互相关函数

        Args:
            signal1: 输入信号 (如Q_in)
            signal2: 输出信号 (如Z_down)

        Returns:
            (滞后数组, 相关系数数组)
        """
        # 标准化
        s1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
        s2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)

        n = len(s1)
        lags = np.arange(-self.max_lag, self.max_lag + 1)
        correlations = np.zeros(len(lags))

        for i, lag in enumerate(lags):
            if lag >= 0:
                corr = np.sum(s1[:n-lag] * s2[lag:]) / (n - lag)
            else:
                corr = np.sum(s1[-lag:] * s2[:n+lag]) / (n + lag)
            correlations[i] = corr

        return lags, correlations

    def find_delay(self,
                  signal1: np.ndarray,
                  signal2: np.ndarray,
                  dt: float = 1.0) -> Tuple[float, float]:
        """
        找到最佳延迟

        Args:
            signal1: 输入信号
            signal2: 输出信号
            dt: 时间步长 [s]

        Returns:
            (延迟时间[s], 相关系数)
        """
        lags, correlations = self.compute_correlation(signal1, signal2)

        # 找正滞后的最大相关
        positive_mask = lags >= 0
        positive_lags = lags[positive_mask]
        positive_corr = correlations[positive_mask]

        best_idx = np.argmax(positive_corr)
        best_lag = positive_lags[best_idx]
        best_corr = positive_corr[best_idx]

        return best_lag * dt, best_corr


# ==============================================================================
# 递推最小二乘法 (RLS)
# ==============================================================================

class RecursiveLeastSquares:
    """
    递推最小二乘法 (RLS) - 在线参数估计

    用于估计IDZ模型的积分增益 (1/A_s)

    模型: Z[k+1] = Z[k] + theta * (Q_in[k-d] - Q_out[k]) * dt
    其中 theta = 1/A_s
    """

    def __init__(self,
                 num_params: int = 1,
                 forgetting_factor: float = 0.98,
                 initial_covariance: float = 1000.0):
        """
        初始化RLS估计器

        Args:
            num_params: 参数数量
            forgetting_factor: 遗忘因子 (0.9-0.99)
            initial_covariance: 初始协方差
        """
        self.num_params = num_params
        self.lambda_ = forgetting_factor

        # 参数估计
        self.theta = np.zeros(num_params)

        # 协方差矩阵
        self.P = np.eye(num_params) * initial_covariance

        # 统计
        self.samples_processed = 0
        self.error_history: List[float] = []

    def update(self,
               y: float,
               phi: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        更新参数估计

        Args:
            y: 实际输出 (Z[k+1] - Z[k])
            phi: 回归向量 [(Q_in[k-d] - Q_out[k]) * dt]

        Returns:
            (更新后的参数, 预测误差)
        """
        phi = np.atleast_1d(phi).flatten().reshape(-1, 1)

        # 预测
        y_hat = (phi.T @ self.theta).item()
        error = y - y_hat

        # 增益计算
        denom = self.lambda_ + (phi.T @ self.P @ phi).item()
        K = self.P @ phi / denom

        # 参数更新
        self.theta = self.theta + K.flatten() * error

        # 协方差更新
        self.P = (self.P - K @ phi.T @ self.P) / self.lambda_

        # 记录
        self.samples_processed += 1
        self.error_history.append(error)

        return self.theta.copy(), error

    def get_estimate(self) -> np.ndarray:
        """获取当前参数估计"""
        return self.theta.copy()

    def get_confidence(self) -> float:
        """计算估计置信度 (基于协方差)"""
        uncertainty = np.sqrt(np.diag(self.P)).sum()
        confidence = 1.0 / (1.0 + uncertainty)
        return confidence


# ==============================================================================
# Kalman滤波器
# ==============================================================================

class KalmanFilter:
    """
    Kalman滤波器 - 状态估计与参数跟踪

    状态: [Z, theta]^T
    其中 theta = 1/A_s
    """

    def __init__(self,
                 process_noise: float = 0.001,
                 measurement_noise: float = 0.1):
        """
        初始化Kalman滤波器

        Args:
            process_noise: 过程噪声协方差
            measurement_noise: 测量噪声协方差
        """
        # 状态: [Z, theta]
        self.x = np.array([4.0, 1e-5])  # 初始估计

        # 协方差
        self.P = np.diag([1.0, 1e-8])

        # 噪声
        self.Q = np.diag([process_noise, process_noise * 0.01])
        self.R = measurement_noise

    def predict(self, u: float, dt: float = 1.0):
        """
        预测步骤

        Args:
            u: 输入 (Q_in - Q_out) * dt
            dt: 时间步长
        """
        # 状态转移: Z[k+1] = Z[k] + theta * u
        # theta 假设缓慢变化
        F = np.array([
            [1, u],
            [0, 1]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_measured: float) -> Tuple[float, float]:
        """
        更新步骤

        Args:
            z_measured: 测量水位

        Returns:
            (估计水位, 估计theta)
        """
        # 观测矩阵
        H = np.array([[1, 0]])

        # Kalman增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S

        # 状态更新
        innovation = z_measured - H @ self.x
        self.x = self.x + K.flatten() * innovation

        # 协方差更新
        self.P = (np.eye(2) - K @ H) @ self.P

        return self.x[0], self.x[1]

    def get_estimates(self) -> Tuple[float, float]:
        """获取估计值"""
        return self.x[0], self.x[1]


# ==============================================================================
# 系统辨识器
# ==============================================================================

class SystemIdentifier:
    """
    系统辨识器 - 综合辨识IDZ模型参数

    功能:
    1. 辨识滞后时间 (互相关分析)
    2. 辨识蓄水面积 (RLS)
    3. 在线参数跟踪 (Kalman)
    4. 参数验证与置信度评估
    """

    def __init__(self,
                 dt: float = 900.0,
                 window_size: int = 100,
                 forgetting_factor: float = 0.98):
        """
        初始化系统辨识器

        Args:
            dt: 时间步长 [s]
            window_size: 数据窗口大小
            forgetting_factor: RLS遗忘因子
        """
        self.dt = dt
        self.window_size = window_size

        # 分析器
        self.cc_analyzer = CrossCorrelationAnalyzer(max_lag=50)

        # RLS估计器 (每个池一个)
        self.rls_estimators: Dict[int, RecursiveLeastSquares] = {}

        # Kalman滤波器 (每个池一个)
        self.kalman_filters: Dict[int, KalmanFilter] = {}

        # 数据缓冲
        self.data_buffers: Dict[int, Dict[str, deque]] = {}

        # 辨识结果
        self.results: Dict[int, IdentificationResult] = {}

        # 配置
        self.forgetting_factor = forgetting_factor

    def initialize_pool(self, pool_id: int):
        """初始化池的辨识器"""
        self.rls_estimators[pool_id] = RecursiveLeastSquares(
            num_params=1,
            forgetting_factor=self.forgetting_factor
        )
        self.kalman_filters[pool_id] = KalmanFilter()
        self.data_buffers[pool_id] = {
            'q_in': deque(maxlen=self.window_size),
            'q_out': deque(maxlen=self.window_size),
            'z': deque(maxlen=self.window_size),
            'time': deque(maxlen=self.window_size),
        }

    def add_sample(self,
                   pool_id: int,
                   q_in: float,
                   q_out: float,
                   z: float,
                   timestamp: float = None):
        """
        添加数据样本

        Args:
            pool_id: 渠池ID
            q_in: 入流 [m³/s]
            q_out: 出流 [m³/s]
            z: 水位 [m]
            timestamp: 时间戳 [s]
        """
        if pool_id not in self.data_buffers:
            self.initialize_pool(pool_id)

        buf = self.data_buffers[pool_id]
        buf['q_in'].append(q_in)
        buf['q_out'].append(q_out)
        buf['z'].append(z)
        buf['time'].append(timestamp or 0)

    def identify_delay(self, pool_id: int) -> Tuple[float, float]:
        """
        辨识滞后时间

        Args:
            pool_id: 渠池ID

        Returns:
            (滞后时间[s], 相关系数)
        """
        if pool_id not in self.data_buffers:
            return 14400.0, 0.0  # 默认4小时

        buf = self.data_buffers[pool_id]
        if len(buf['q_in']) < 30:
            return 14400.0, 0.0

        q_in = np.array(buf['q_in'])
        z = np.array(buf['z'])

        # 使用入流和水位的互相关
        tau, corr = self.cc_analyzer.find_delay(q_in, z, self.dt)

        return tau, corr

    def identify_area(self, pool_id: int, tau: float) -> Tuple[float, float]:
        """
        辨识蓄水面积

        Args:
            pool_id: 渠池ID
            tau: 已知滞后时间 [s]

        Returns:
            (蓄水面积[m²], 置信度)
        """
        if pool_id not in self.rls_estimators:
            self.initialize_pool(pool_id)

        buf = self.data_buffers[pool_id]
        if len(buf['q_in']) < 10:
            return 100000.0, 0.0  # 默认

        # 计算延迟步数
        delay_steps = int(np.ceil(tau / self.dt))

        q_in = list(buf['q_in'])
        q_out = list(buf['q_out'])
        z = list(buf['z'])

        rls = self.rls_estimators[pool_id]

        # 使用最新数据更新
        for i in range(delay_steps + 1, len(q_in)):
            # 回归向量
            net_flow = (q_in[i - delay_steps] - q_out[i]) * self.dt
            phi = np.array([net_flow])

            # 输出 (水位变化)
            dz = z[i] - z[i-1]

            # 更新
            rls.update(dz, phi)

        # 获取估计
        theta = rls.get_estimate()
        if abs(theta[0]) > 1e-10:
            A_s = 1.0 / theta[0]
        else:
            A_s = 100000.0

        # 限制合理范围
        A_s = np.clip(A_s, 10000, 10000000)

        confidence = rls.get_confidence()

        return A_s, confidence

    def identify_idz_parameters(self,
                                pool_id: int,
                                historical_data: Dict[str, np.ndarray] = None) -> IdentificationResult:
        """
        辨识IDZ模型完整参数

        Args:
            pool_id: 渠池ID
            historical_data: 历史数据 {'Q_in': [...], 'Q_out': [...], 'Z': [...]}

        Returns:
            辨识结果
        """
        # 如果提供历史数据，先添加
        if historical_data is not None:
            q_in = historical_data.get('Q_in', [])
            q_out = historical_data.get('Q_out', [])
            z = historical_data.get('Z', [])

            for i in range(len(q_in)):
                self.add_sample(
                    pool_id,
                    q_in[i],
                    q_out[i] if i < len(q_out) else q_in[i],
                    z[i] if i < len(z) else 4.0,
                    i * self.dt
                )

        # 步骤1: 辨识滞后时间
        tau, tau_confidence = self.identify_delay(pool_id)
        logger.info(f"池{pool_id} 滞后时间: {tau/3600:.2f}h (相关系数={tau_confidence:.3f})")

        # 步骤2: 辨识蓄水面积
        A_s, A_s_confidence = self.identify_area(pool_id, tau)
        logger.info(f"池{pool_id} 蓄水面积: {A_s:.0f}m² (置信度={A_s_confidence:.3f})")

        # 计算整体置信度
        overall_confidence = (tau_confidence + A_s_confidence) / 2

        # 计算RMSE
        rmse = self._compute_rmse(pool_id, tau, A_s)

        # 计算R²
        r_squared = self._compute_r_squared(pool_id, tau, A_s)

        result = IdentificationResult(
            pool_id=pool_id,
            timestamp=self.data_buffers.get(pool_id, {}).get('time', deque([0]))[-1],
            tau=tau,
            A_s=A_s,
            c_in=1.0,
            c_out=1.0,
            confidence=overall_confidence,
            r_squared=r_squared,
            rmse=rmse,
            samples_used=len(self.data_buffers.get(pool_id, {}).get('q_in', [])),
            method="CC+RLS"
        )

        self.results[pool_id] = result
        return result

    def _compute_rmse(self, pool_id: int, tau: float, A_s: float) -> float:
        """计算RMSE"""
        if pool_id not in self.data_buffers:
            return 0.0

        buf = self.data_buffers[pool_id]
        if len(buf['z']) < 10:
            return 0.0

        delay_steps = int(np.ceil(tau / self.dt))
        q_in = list(buf['q_in'])
        q_out = list(buf['q_out'])
        z = list(buf['z'])

        errors = []
        for i in range(delay_steps + 1, len(q_in) - 1):
            # 预测
            net_flow = q_in[i - delay_steps] - q_out[i]
            dz_pred = net_flow * self.dt / A_s
            z_pred = z[i] + dz_pred

            # 实际
            z_actual = z[i + 1]

            errors.append((z_pred - z_actual) ** 2)

        if errors:
            return np.sqrt(np.mean(errors))
        return 0.0

    def _compute_r_squared(self, pool_id: int, tau: float, A_s: float) -> float:
        """计算决定系数R²"""
        if pool_id not in self.data_buffers:
            return 0.0

        buf = self.data_buffers[pool_id]
        if len(buf['z']) < 10:
            return 0.0

        delay_steps = int(np.ceil(tau / self.dt))
        q_in = list(buf['q_in'])
        q_out = list(buf['q_out'])
        z = list(buf['z'])

        predictions = []
        actuals = []

        for i in range(delay_steps + 1, len(q_in) - 1):
            net_flow = q_in[i - delay_steps] - q_out[i]
            dz_pred = net_flow * self.dt / A_s
            z_pred = z[i] + dz_pred

            predictions.append(z_pred)
            actuals.append(z[i + 1])

        if not predictions:
            return 0.0

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)

        if ss_tot < 1e-10:
            return 1.0

        return 1.0 - ss_res / ss_tot

    def online_update(self,
                      pool_id: int,
                      q_in: float,
                      q_out: float,
                      z_measured: float) -> Tuple[float, float]:
        """
        在线更新参数估计

        Args:
            pool_id: 渠池ID
            q_in: 入流
            q_out: 出流
            z_measured: 测量水位

        Returns:
            (估计水位, 估计theta=1/A_s)
        """
        if pool_id not in self.kalman_filters:
            self.initialize_pool(pool_id)

        kf = self.kalman_filters[pool_id]

        # 输入
        u = (q_in - q_out) * self.dt

        # 预测
        kf.predict(u, self.dt)

        # 更新
        z_est, theta_est = kf.update(z_measured)

        # 添加到缓冲
        self.add_sample(pool_id, q_in, q_out, z_measured)

        return z_est, theta_est

    def get_result(self, pool_id: int) -> Optional[IdentificationResult]:
        """获取辨识结果"""
        return self.results.get(pool_id)

    def get_all_results(self) -> Dict[int, IdentificationResult]:
        """获取所有辨识结果"""
        return self.results.copy()

    def export_parameters(self) -> Dict[int, Dict[str, float]]:
        """导出所有参数"""
        return {
            pool_id: {
                'tau': result.tau,
                'A_s': result.A_s,
                'c_in': result.c_in,
                'c_out': result.c_out,
                'confidence': result.confidence,
            }
            for pool_id, result in self.results.items()
        }


# ==============================================================================
# 批量辨识器
# ==============================================================================

class BatchSystemIdentifier:
    """
    批量系统辨识器 - 对全线渠池进行辨识
    """

    def __init__(self, num_pools: int, dt: float = 900.0):
        """
        初始化

        Args:
            num_pools: 渠池数量
            dt: 时间步长
        """
        self.num_pools = num_pools
        self.dt = dt
        self.identifier = SystemIdentifier(dt)

    def identify_all(self,
                     all_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, IdentificationResult]:
        """
        辨识所有渠池

        Args:
            all_data: {pool_id: {'Q_in': [...], 'Q_out': [...], 'Z': [...]}}

        Returns:
            {pool_id: IdentificationResult}
        """
        results = {}

        for pool_id in range(self.num_pools):
            data = all_data.get(pool_id)
            if data:
                result = self.identifier.identify_idz_parameters(pool_id, data)
                results[pool_id] = result
                logger.info(f"池{pool_id}: tau={result.tau/3600:.1f}h, "
                           f"A_s={result.A_s:.0f}m², R²={result.r_squared:.3f}")
            else:
                # 使用默认参数
                results[pool_id] = IdentificationResult(
                    pool_id=pool_id,
                    timestamp=0,
                    tau=14400,  # 4小时
                    A_s=100000,  # 10万m²
                    confidence=0.0,
                    method="default"
                )

        return results

    def get_summary(self) -> Dict[str, Any]:
        """获取辨识摘要"""
        results = self.identifier.get_all_results()

        if not results:
            return {}

        taus = [r.tau for r in results.values()]
        areas = [r.A_s for r in results.values()]
        confidences = [r.confidence for r in results.values()]
        r_squareds = [r.r_squared for r in results.values()]

        return {
            'num_identified': len(results),
            'tau_mean': np.mean(taus),
            'tau_std': np.std(taus),
            'A_s_mean': np.mean(areas),
            'A_s_std': np.std(areas),
            'avg_confidence': np.mean(confidences),
            'avg_r_squared': np.mean(r_squareds),
        }


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" " * 15 + "系统辨识模块测试")
    print("="*70)

    # 生成模拟数据
    np.random.seed(42)
    dt = 900.0  # 15分钟
    n_samples = 200

    # 真实参数
    true_tau = 14400  # 4小时
    true_A_s = 150000  # 15万m²
    delay_steps = int(true_tau / dt)

    print(f"\n真实参数:")
    print(f"  滞后时间: {true_tau/3600:.1f} 小时 ({delay_steps} 步)")
    print(f"  蓄水面积: {true_A_s} m²")

    # 生成数据
    q_in = 100 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 48)  # 12小时周期
    q_in += np.random.randn(n_samples) * 2  # 噪声

    q_out = np.zeros(n_samples)
    z = np.zeros(n_samples)
    z[0] = 4.0

    for i in range(1, n_samples):
        # 延迟入流
        q_in_delayed = q_in[i - delay_steps] if i >= delay_steps else q_in[0]
        q_out[i] = q_in_delayed * 0.98  # 略小于入流

        # 水位变化
        dz = (q_in_delayed - q_out[i]) * dt / true_A_s
        z[i] = z[i-1] + dz + np.random.randn() * 0.01

    # 创建辨识器
    identifier = SystemIdentifier(dt=dt)

    print(f"\n{'='*70}")
    print("测试1: 互相关分析 (滞后时间辨识)")
    print('='*70)

    for i in range(n_samples):
        identifier.add_sample(0, q_in[i], q_out[i], z[i], i * dt)

    tau_est, corr = identifier.identify_delay(0)
    print(f"  估计滞后: {tau_est/3600:.2f} 小时 (真实: {true_tau/3600:.1f}h)")
    print(f"  相关系数: {corr:.3f}")
    print(f"  误差: {abs(tau_est - true_tau)/true_tau * 100:.1f}%")

    print(f"\n{'='*70}")
    print("测试2: RLS (蓄水面积辨识)")
    print('='*70)

    A_s_est, conf = identifier.identify_area(0, tau_est)
    print(f"  估计面积: {A_s_est:.0f} m² (真实: {true_A_s}m²)")
    print(f"  置信度: {conf:.3f}")
    print(f"  误差: {abs(A_s_est - true_A_s)/true_A_s * 100:.1f}%")

    print(f"\n{'='*70}")
    print("测试3: 完整辨识")
    print('='*70)

    result = identifier.identify_idz_parameters(0)
    print(f"  滞后时间: {result.tau/3600:.2f} h")
    print(f"  蓄水面积: {result.A_s:.0f} m²")
    print(f"  置信度: {result.confidence:.3f}")
    print(f"  R²: {result.r_squared:.4f}")
    print(f"  RMSE: {result.rmse:.4f} m")
    print(f"  使用样本: {result.samples_used}")
    print(f"  方法: {result.method}")

    print(f"\n{'='*70}")
    print("测试4: 在线更新")
    print('='*70)

    # 在线更新几步
    for i in range(5):
        new_q_in = 100 + np.random.randn() * 5
        new_q_out = 100 + np.random.randn() * 3
        new_z = 4.0 + np.random.randn() * 0.1

        z_est, theta_est = identifier.online_update(0, new_q_in, new_q_out, new_z)
        A_s_online = 1.0 / theta_est if abs(theta_est) > 1e-10 else float('inf')
        print(f"  步骤{i}: Z_est={z_est:.3f}m, A_s_est={A_s_online:.0f}m²")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
