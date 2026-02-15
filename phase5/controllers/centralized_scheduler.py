"""
L3 上层集中调度器 (Centralized Scheduler)

基于线性体积平衡模型 (Linear Volume Balance Model) 的全局水量规划
数学模型: V_{k+1} = V_k + (Q_in,k - Q_out,k - Q_demand,k + Q_rain,k) * dt

功能:
1. 全局水量平衡规划 (24h视野)
2. 生成下层参考轨迹
3. 协调多池蓄水与泄洪
4. 整合天气预报和需求预测
"""

import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class SchedulingMode(Enum):
    """调度模式"""
    NORMAL = "normal"              # 正常供水
    FLOOD_CONTROL = "flood_control"  # 防洪调度
    DROUGHT = "drought"            # 抗旱调度
    PRE_RELEASE = "pre_release"    # 预泄腾库容
    EMERGENCY = "emergency"        # 紧急调度


@dataclass
class NetworkTopology:
    """水网拓扑结构"""
    num_pools: int
    pool_names: List[str] = field(default_factory=list)
    pool_areas: List[float] = field(default_factory=list)  # m²
    pool_volumes_max: List[float] = field(default_factory=list)  # m³
    pool_volumes_min: List[float] = field(default_factory=list)  # m³
    pool_levels_max: List[float] = field(default_factory=list)  # m
    pool_levels_min: List[float] = field(default_factory=list)  # m
    connections: List[Tuple[int, int]] = field(default_factory=list)  # (from, to)
    max_flow_rates: List[float] = field(default_factory=list)  # m³/s

    @classmethod
    def create_cascade(cls, num_pools: int,
                       area: float = 10000.0,
                       level_min: float = 0.5,
                       level_max: float = 8.0,
                       max_flow: float = 20.0) -> 'NetworkTopology':
        """创建级联水网拓扑"""
        return cls(
            num_pools=num_pools,
            pool_names=[f"Pool_{i}" for i in range(num_pools)],
            pool_areas=[area] * num_pools,
            pool_volumes_max=[area * level_max] * num_pools,
            pool_volumes_min=[area * level_min] * num_pools,
            pool_levels_max=[level_max] * num_pools,
            pool_levels_min=[level_min] * num_pools,
            connections=[(i, i+1) for i in range(num_pools - 1)],
            max_flow_rates=[max_flow] * (num_pools - 1)
        )


@dataclass
class SchedulingForecast:
    """调度预测输入"""
    horizon: int  # 预测步数
    dt: float  # 时间步长 (s)

    # 需求预测 [num_pools x horizon]
    demand_forecast: np.ndarray = None

    # 天气预测 [horizon]
    rainfall_forecast: np.ndarray = None  # mm/h
    evaporation_forecast: np.ndarray = None  # mm/h

    # 上游来水预测 (第一个池的入流)
    inflow_forecast: np.ndarray = None  # m³/s

    # 预测不确定性 (可选)
    demand_uncertainty: np.ndarray = None
    inflow_uncertainty: np.ndarray = None


@dataclass
class SchedulingResult:
    """调度结果"""
    success: bool
    mode: SchedulingMode

    # 参考轨迹 [num_pools x horizon]
    reference_levels: np.ndarray = None  # 水位参考轨迹
    reference_volumes: np.ndarray = None  # 体积参考轨迹

    # 全局流量指令 [num_connections x horizon]
    global_flow_commands: np.ndarray = None

    # 上游入流建议
    suggested_inflow: np.ndarray = None

    # 预泄/蓄水指令
    pre_release_volumes: np.ndarray = None  # 预泄量
    storage_targets: np.ndarray = None  # 蓄水目标

    # 求解信息
    solve_time: float = 0.0
    objective_value: float = 0.0
    solver_status: str = ""

    # 调度建议
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SchedulerConfig:
    """调度器配置"""
    # 时间参数
    planning_horizon: int = 24  # 规划时域 (步数)
    dt: float = 3600.0  # 时间步长 (秒)

    # 权重
    W_level_tracking: float = 10.0  # 水位跟踪权重
    W_volume_balance: float = 5.0  # 体积平衡权重
    W_flow_smooth: float = 2.0  # 流量平滑权重
    W_demand_satisfaction: float = 8.0  # 需求满足权重
    W_energy: float = 0.5  # 能耗权重
    W_safety_margin: float = 15.0  # 安全裕度权重

    # 约束参数
    safety_margin_flood: float = 1.0  # 防洪安全裕度 (m)
    safety_margin_drought: float = 0.5  # 抗旱安全裕度 (m)
    max_level_change_rate: float = 0.3  # 最大水位变化率 (m/h)
    max_flow_change_rate: float = 3.0  # 最大流量变化率 (m³/s/h)

    # 预泄参数
    pre_release_lead_time: int = 6  # 预泄提前量 (小时)
    flood_threshold_rainfall: float = 10.0  # 触发预泄的降雨阈值 (mm/h)


class CentralizedScheduler:
    """
    L3 集中调度器

    职责:
    1. 全局优化: 考虑全网拓扑和24h预测的全局最优
    2. 参考生成: 为L2下层MPC生成参考轨迹
    3. 模式切换: 根据预报切换调度模式(正常/防洪/抗旱)
    4. 预泄决策: 暴雨预警时决定预泄策略
    """

    def __init__(self,
                 topology: NetworkTopology,
                 config: SchedulerConfig = None):
        """
        初始化集中调度器

        Args:
            topology: 水网拓扑
            config: 调度器配置
        """
        self.topology = topology
        self.config = config or SchedulerConfig()
        self.num_pools = topology.num_pools

        # 当前模式
        self.current_mode = SchedulingMode.NORMAL

        # 上一次调度结果 (用于平滑过渡)
        self.last_result: Optional[SchedulingResult] = None

        # 统计
        self.stats = {
            'total_schedules': 0,
            'mode_changes': 0,
            'pre_release_triggers': 0,
            'solve_failures': 0
        }

        # 预构建优化问题
        self._build_optimization_problem()

    def _build_optimization_problem(self):
        """构建CVXPY优化问题框架"""
        N = self.config.planning_horizon
        M = self.num_pools
        dt = self.config.dt

        # 决策变量
        self.V_var = cp.Variable((M, N))  # 各池体积轨迹
        self.Q_flow_var = cp.Variable((M-1, N))  # 池间流量
        self.Q_out_var = cp.Variable((M, N))  # 各池出流(含需求)

        # 参数 (每次求解时更新)
        self.V_init_param = cp.Parameter(M)  # 初始体积
        self.V_target_param = cp.Parameter((M, N))  # 目标体积
        self.Q_in_param = cp.Parameter(N)  # 上游入流
        self.Q_demand_param = cp.Parameter((M, N))  # 需求
        self.Q_rain_param = cp.Parameter((M, N))  # 降雨补给
        self.Q_evap_param = cp.Parameter((M, N))  # 蒸发损失

        # 模式相关参数
        self.V_max_param = cp.Parameter(M)  # 最大体积(可调)
        self.V_min_param = cp.Parameter(M)  # 最小体积(可调)
        self.Q_max_param = cp.Parameter(M-1)  # 最大流量(可调)

        # 权重参数 (便于动态调整)
        self.W_level_param = cp.Parameter(nonneg=True)
        self.W_demand_param = cp.Parameter(nonneg=True)
        self.W_smooth_param = cp.Parameter(nonneg=True)
        self.W_safety_param = cp.Parameter(nonneg=True)

    def schedule(self,
                 current_volumes: List[float],
                 forecast: SchedulingForecast,
                 mode_override: SchedulingMode = None) -> SchedulingResult:
        """
        执行调度计算

        Args:
            current_volumes: 各池当前蓄水量 (m³)
            forecast: 预测数据
            mode_override: 强制模式 (可选)

        Returns:
            SchedulingResult: 调度结果
        """
        start_time = time.time()

        # 1. 确定调度模式
        mode = mode_override or self._determine_mode(forecast)
        if mode != self.current_mode:
            self.stats['mode_changes'] += 1
            self.current_mode = mode

        # 2. 准备参数
        params = self._prepare_parameters(current_volumes, forecast, mode)

        # 3. 构建并求解优化问题
        try:
            result = self._solve_optimization(params, mode)
        except Exception as e:
            self.stats['solve_failures'] += 1
            result = self._fallback_schedule(current_volumes, forecast, mode, str(e))

        # 4. 后处理
        result.mode = mode
        result.solve_time = time.time() - start_time
        result.recommendations = self._generate_recommendations(result, forecast, mode)

        self.stats['total_schedules'] += 1
        self.last_result = result

        return result

    def _determine_mode(self, forecast: SchedulingForecast) -> SchedulingMode:
        """根据预报确定调度模式"""
        # 检查暴雨预警
        if forecast.rainfall_forecast is not None:
            max_rainfall = np.max(forecast.rainfall_forecast)
            if max_rainfall > self.config.flood_threshold_rainfall:
                # 计算暴雨到来时间
                heavy_rain_indices = np.where(
                    forecast.rainfall_forecast > self.config.flood_threshold_rainfall
                )[0]
                if len(heavy_rain_indices) > 0:
                    hours_to_rain = heavy_rain_indices[0]
                    if hours_to_rain <= self.config.pre_release_lead_time:
                        self.stats['pre_release_triggers'] += 1
                        return SchedulingMode.PRE_RELEASE
                    else:
                        return SchedulingMode.FLOOD_CONTROL

        # 检查入流不足 (干旱)
        if forecast.inflow_forecast is not None:
            avg_inflow = np.mean(forecast.inflow_forecast)
            avg_demand = np.mean(forecast.demand_forecast) if forecast.demand_forecast is not None else 5.0
            if avg_inflow < 0.7 * avg_demand:
                return SchedulingMode.DROUGHT

        return SchedulingMode.NORMAL

    def _prepare_parameters(self,
                           current_volumes: List[float],
                           forecast: SchedulingForecast,
                           mode: SchedulingMode) -> Dict:
        """准备优化参数"""
        N = self.config.planning_horizon
        M = self.num_pools
        dt = self.config.dt

        params = {}

        # 初始体积
        params['V_init'] = np.array(current_volumes)

        # 上游入流
        if forecast.inflow_forecast is not None:
            params['Q_in'] = forecast.inflow_forecast[:N]
        else:
            params['Q_in'] = np.ones(N) * 10.0  # 默认10 m³/s

        # 需求
        if forecast.demand_forecast is not None:
            params['Q_demand'] = forecast.demand_forecast[:, :N]
        else:
            params['Q_demand'] = np.ones((M, N)) * 5.0  # 默认5 m³/s

        # 降雨补给 (mm/h -> m³/s)
        if forecast.rainfall_forecast is not None:
            rain_m3s = forecast.rainfall_forecast[:N] / 1000 / 3600 * np.array(
                self.topology.pool_areas
            ).reshape(-1, 1)
            params['Q_rain'] = rain_m3s
        else:
            params['Q_rain'] = np.zeros((M, N))

        # 蒸发损失
        if forecast.evaporation_forecast is not None:
            evap_m3s = forecast.evaporation_forecast[:N] / 1000 / 3600 * np.array(
                self.topology.pool_areas
            ).reshape(-1, 1)
            params['Q_evap'] = evap_m3s
        else:
            params['Q_evap'] = np.zeros((M, N))

        # 根据模式调整约束
        if mode == SchedulingMode.PRE_RELEASE:
            # 预泄模式: 降低目标水位
            safety_margin = self.config.safety_margin_flood
            params['V_max'] = np.array([
                area * (level_max - safety_margin)
                for area, level_max in zip(
                    self.topology.pool_areas,
                    self.topology.pool_levels_max
                )
            ])
            params['V_target'] = params['V_max'][:, np.newaxis] * 0.6  # 目标60%库容

        elif mode == SchedulingMode.FLOOD_CONTROL:
            # 防洪模式: 严格上限约束
            params['V_max'] = np.array(self.topology.pool_volumes_max)
            params['V_target'] = params['V_max'][:, np.newaxis] * 0.7

        elif mode == SchedulingMode.DROUGHT:
            # 抗旱模式: 保证最低蓄水
            safety_margin = self.config.safety_margin_drought
            params['V_min'] = np.array([
                area * (level_min + safety_margin)
                for area, level_min in zip(
                    self.topology.pool_areas,
                    self.topology.pool_levels_min
                )
            ])
            params['V_target'] = params['V_min'][:, np.newaxis] * 1.5
        else:
            # 正常模式
            params['V_max'] = np.array(self.topology.pool_volumes_max)
            params['V_min'] = np.array(self.topology.pool_volumes_min)
            # 目标: 50%库容
            params['V_target'] = (params['V_max'] + params['V_min'])[:, np.newaxis] / 2 * np.ones((1, N))

        # 默认边界
        if 'V_max' not in params:
            params['V_max'] = np.array(self.topology.pool_volumes_max)
        if 'V_min' not in params:
            params['V_min'] = np.array(self.topology.pool_volumes_min)
        if 'V_target' not in params:
            params['V_target'] = (params['V_max'] + params['V_min'])[:, np.newaxis] / 2 * np.ones((1, N))

        params['Q_max'] = np.array(self.topology.max_flow_rates)
        params['mode'] = mode

        return params

    def _solve_optimization(self, params: Dict, mode: SchedulingMode) -> SchedulingResult:
        """求解优化问题"""
        N = self.config.planning_horizon
        M = self.num_pools
        dt = self.config.dt

        # 决策变量
        V = cp.Variable((M, N), nonneg=True)  # 体积
        Q_flow = cp.Variable((M-1, N), nonneg=True)  # 池间流量
        Q_release = cp.Variable((M, N), nonneg=True)  # 各池泄流

        # 约束列表
        constraints = []

        # 1. 体积平衡约束 (核心物理模型)
        # V_{k+1} = V_k + (Q_in - Q_out - Q_demand + Q_rain - Q_evap) * dt
        for k in range(N):
            for i in range(M):
                if k == 0:
                    V_prev = params['V_init'][i]
                else:
                    V_prev = V[i, k-1]

                # 计算净入流
                if i == 0:
                    # 第一个池: 来自上游
                    Q_in_i = params['Q_in'][k]
                else:
                    # 其他池: 来自上游池的流量
                    Q_in_i = Q_flow[i-1, k]

                # 出流
                if i < M - 1:
                    Q_out_i = Q_flow[i, k]
                else:
                    Q_out_i = Q_release[i, k]

                # 体积平衡
                constraints.append(
                    V[i, k] == V_prev + (
                        Q_in_i - Q_out_i - params['Q_demand'][i, k] +
                        params['Q_rain'][i, k] - params['Q_evap'][i, k]
                    ) * dt
                )

        # 2. 体积约束
        for i in range(M):
            constraints.append(V[i, :] >= params['V_min'][i])
            constraints.append(V[i, :] <= params['V_max'][i])

        # 3. 流量约束
        for i in range(M-1):
            constraints.append(Q_flow[i, :] >= 0)
            constraints.append(Q_flow[i, :] <= params['Q_max'][i])

        # 4. 流量变化率约束
        for i in range(M-1):
            for k in range(1, N):
                constraints.append(
                    cp.abs(Q_flow[i, k] - Q_flow[i, k-1]) <=
                    self.config.max_flow_change_rate * (dt / 3600)
                )

        # 5. 需求满足约束 (软约束通过目标函数)
        # Q_release 需要满足需求
        for i in range(M):
            constraints.append(Q_release[i, :] >= 0)

        # 构建目标函数
        cost = 0

        # 目标1: 体积/水位跟踪
        for i in range(M):
            cost += self.config.W_level_tracking * cp.sum_squares(
                V[i, :] - params['V_target'][i, :]
            )

        # 目标2: 需求满足
        for i in range(M):
            cost += self.config.W_demand_satisfaction * cp.sum_squares(
                Q_release[i, :] - params['Q_demand'][i, :]
            )

        # 目标3: 流量平滑
        for i in range(M-1):
            for k in range(1, N):
                cost += self.config.W_flow_smooth * cp.square(
                    Q_flow[i, k] - Q_flow[i, k-1]
                )

        # 目标4: 安全裕度 (远离边界)
        for i in range(M):
            # 惩罚接近上限
            upper_slack = params['V_max'][i] - V[i, :]
            cost += self.config.W_safety_margin * cp.sum(cp.inv_pos(upper_slack + 1000))

            # 惩罚接近下限
            lower_slack = V[i, :] - params['V_min'][i]
            cost += self.config.W_safety_margin * cp.sum(cp.inv_pos(lower_slack + 1000))

        # 模式特定目标
        if mode == SchedulingMode.PRE_RELEASE:
            # 预泄模式: 鼓励快速降低水位
            for i in range(M):
                cost -= 2.0 * cp.sum(Q_flow[i, :]) if i < M-1 else 0

        # 求解
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception as exc:
            logger.debug("ECOS solver failed, falling back to SCS: %s", exc)
            problem.solve(solver=cp.SCS, verbose=False)

        # 提取结果
        if problem.status in ['optimal', 'optimal_inaccurate']:
            # 将体积转换为水位
            areas = np.array(self.topology.pool_areas)
            ref_levels = V.value / areas[:, np.newaxis]

            return SchedulingResult(
                success=True,
                mode=mode,
                reference_levels=ref_levels,
                reference_volumes=V.value,
                global_flow_commands=Q_flow.value if M > 1 else None,
                suggested_inflow=params['Q_in'],
                storage_targets=params['V_target'],
                objective_value=problem.value,
                solver_status=problem.status
            )
        else:
            raise RuntimeError(f"优化求解失败: {problem.status}")

    def _fallback_schedule(self,
                          current_volumes: List[float],
                          forecast: SchedulingForecast,
                          mode: SchedulingMode,
                          error_msg: str) -> SchedulingResult:
        """回退调度方案"""
        N = self.config.planning_horizon
        M = self.num_pools

        # 简单保守策略: 维持当前状态
        areas = np.array(self.topology.pool_areas)
        current_levels = np.array(current_volumes) / areas

        ref_levels = np.tile(current_levels[:, np.newaxis], (1, N))
        ref_volumes = np.tile(np.array(current_volumes)[:, np.newaxis], (1, N))

        # 流量 = 入流
        if forecast.inflow_forecast is not None:
            flow_commands = np.tile(forecast.inflow_forecast[:N], (M-1, 1))
        else:
            flow_commands = np.ones((M-1, N)) * 5.0

        return SchedulingResult(
            success=False,
            mode=mode,
            reference_levels=ref_levels,
            reference_volumes=ref_volumes,
            global_flow_commands=flow_commands,
            solver_status=f"fallback: {error_msg}",
            recommendations=[f"警告: 优化求解失败 ({error_msg}), 使用保守策略"]
        )

    def _generate_recommendations(self,
                                  result: SchedulingResult,
                                  forecast: SchedulingForecast,
                                  mode: SchedulingMode) -> List[str]:
        """生成调度建议"""
        recommendations = []

        if mode == SchedulingMode.PRE_RELEASE:
            recommendations.append("建议: 启动预泄程序，腾出防洪库容")
            if forecast.rainfall_forecast is not None:
                peak_rain = np.max(forecast.rainfall_forecast)
                recommendations.append(f"预计最大降雨强度: {peak_rain:.1f} mm/h")

        elif mode == SchedulingMode.FLOOD_CONTROL:
            recommendations.append("建议: 进入防洪调度模式")
            recommendations.append("注意: 下游泄流需协调")

        elif mode == SchedulingMode.DROUGHT:
            recommendations.append("建议: 进入抗旱调度模式")
            recommendations.append("建议: 限制非必要用水")

        if not result.success:
            recommendations.append("警告: 调度优化未收敛，请人工复核")

        return recommendations

    def get_reference_trajectory(self,
                                 pool_id: int,
                                 horizon: int = None) -> np.ndarray:
        """
        获取指定池的参考轨迹 (供L2使用)

        Args:
            pool_id: 池ID
            horizon: 需要的时域长度 (默认全部)

        Returns:
            参考水位轨迹
        """
        if self.last_result is None or self.last_result.reference_levels is None:
            return None

        ref = self.last_result.reference_levels[pool_id, :]

        if horizon is not None and horizon < len(ref):
            return ref[:horizon]

        return ref

    def get_flow_command(self,
                        connection_id: int,
                        step: int = 0) -> float:
        """
        获取池间流量指令 (供L2使用)

        Args:
            connection_id: 连接ID (0表示池0到池1)
            step: 时间步

        Returns:
            流量指令 (m³/s)
        """
        if self.last_result is None or self.last_result.global_flow_commands is None:
            return None

        return float(self.last_result.global_flow_commands[connection_id, step])

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'current_mode': self.current_mode.value
        }


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*15 + "L3 集中调度器演示")
    logger.info("="*70)

    # 创建拓扑
    topology = NetworkTopology.create_cascade(
        num_pools=3,
        area=10000.0,
        level_min=0.5,
        level_max=8.0,
        max_flow=20.0
    )

    # 创建调度器
    config = SchedulerConfig(
        planning_horizon=24,
        dt=3600.0
    )
    scheduler = CentralizedScheduler(topology, config)

    logger.info(f"\n✓ 调度器已初始化")
    logger.info(f"  池数量: {topology.num_pools}")
    logger.info(f"  规划时域: {config.planning_horizon} 小时")

    # 测试正常模式
    logger.info(f"\n{'='*70}")
    logger.info("测试1: 正常运行模式")
    logger.info('='*70)

    current_volumes = [30000, 30000, 30000]  # m³
    forecast = SchedulingForecast(
        horizon=24,
        dt=3600.0,
        demand_forecast=np.ones((3, 24)) * 5.0,
        inflow_forecast=np.ones(24) * 10.0
    )

    result = scheduler.schedule(current_volumes, forecast)

    logger.info(f"\n结果:")
    logger.info(f"  成功: {result.success}")
    logger.info(f"  模式: {result.mode.value}")
    logger.info(f"  求解时间: {result.solve_time*1000:.1f} ms")
    logger.info(f"  目标值: {result.objective_value:.2f}")
    logger.info(f"\n参考水位轨迹 (前6小时):")
    for i in range(3):
        levels = result.reference_levels[i, :6]
        logger.info(f"  池{i}: {', '.join([f'{l:.2f}' for l in levels])} m")

    # 测试预泄模式
    logger.info(f"\n{'='*70}")
    logger.info("测试2: 暴雨预警 -> 预泄模式")
    logger.info('='*70)

    forecast_rain = SchedulingForecast(
        horizon=24,
        dt=3600.0,
        demand_forecast=np.ones((3, 24)) * 5.0,
        inflow_forecast=np.ones(24) * 10.0,
        rainfall_forecast=np.concatenate([
            np.zeros(4),  # 前4小时无雨
            np.ones(4) * 15.0,  # 4-8小时暴雨
            np.ones(16) * 5.0  # 之后小雨
        ])
    )

    result_rain = scheduler.schedule(current_volumes, forecast_rain)

    logger.info(f"\n结果:")
    logger.info(f"  成功: {result_rain.success}")
    logger.info(f"  模式: {result_rain.mode.value}")
    logger.info(f"\n建议:")
    for rec in result_rain.recommendations:
        logger.info(f"  - {rec}")
    logger.info(f"\n预泄后目标水位 (6小时后):")
    for i in range(3):
        logger.info(f"  池{i}: {result_rain.reference_levels[i, 5]:.2f} m")

    # 统计
    logger.info(f"\n{'='*70}")
    logger.info("统计信息")
    logger.info('='*70)
    stats = scheduler.get_statistics()
    logger.info(f"  总调度次数: {stats['total_schedules']}")
    logger.info(f"  模式切换: {stats['mode_changes']}")
    logger.info(f"  预泄触发: {stats['pre_release_triggers']}")

    logger.info("\n" + "="*70)
    logger.info("演示完成!")
    logger.info("="*70)
