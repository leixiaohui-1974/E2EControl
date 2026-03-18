"""
双层MPC控制器 (Hierarchical MPC Controller)

架构:
┌─────────────────────────────────────────────────────────────┐
│                   L3 集中调度层 (慢循环)                      │
│  - 执行周期: 1-4小时                                         │
│  - 功能: 全局水量平衡, 预泄决策, 参考轨迹生成                  │
│  - 输出: 24h参考水位轨迹, 全局流量指令                        │
├─────────────────────────────────────────────────────────────┤
│                   L2 分布式控制层 (快循环)                    │
│  - 执行周期: 10-15分钟                                       │
│  - 功能: 轨迹跟踪, 实时波动处理, ADMM协调                     │
│  - 输入: L3参考轨迹, 场景物理参数                             │
├─────────────────────────────────────────────────────────────┤
│                   场景感知与物理参数注入                       │
│  - 场景识别 -> 物理参数映射 (结冰/淤积/正常)                   │
│  - 认知驱动的在线模型调整                                     │
└─────────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)

# 导入L3和L2组件
from hydroe2e.phase5.controllers.centralized_scheduler import (
    CentralizedScheduler, NetworkTopology, SchedulerConfig,
    SchedulingForecast, SchedulingResult, SchedulingMode
)
from hydroe2e.phase5.controllers.parameterized_mpc import (
    ParameterizedDistributedMPC, PhysicalParameters, ScenarioPhysics
)

# Phase 3 决策引擎 (如果可用)
try:
    from hydroe2e.phase3.scenario_recognition.rule_engine import SystemState
    from hydroe2e.phase3.decision.decision_engine import DecisionEngine
    DECISION_ENGINE_AVAILABLE = True
except ImportError:
    DECISION_ENGINE_AVAILABLE = False


class PhysicalScenario(Enum):
    """物理场景类型 (影响流速效率)"""
    NORMAL = "NORMAL"
    ICE_FORMATION = "ICE_FORMATION"
    ICE_STABLE = "ICE_STABLE"
    ICE_BREAKUP = "ICE_BREAKUP"
    SEDIMENT_HIGH = "SEDIMENT_HIGH"
    VEGETATION_GROWTH = "VEGETATION_GROWTH"
    FLOOD = "FLOOD"
    DROUGHT = "DROUGHT"


@dataclass
class HierarchicalConfig:
    """双层MPC配置"""
    # L3 慢循环配置
    l3_planning_horizon: int = 24      # L3规划时域 (小时)
    l3_update_interval: int = 4        # L3更新间隔 (小时)
    l3_dt: float = 3600.0              # L3时间步长 (秒)

    # L2 快循环配置
    l2_control_horizon: int = 5        # L2控制时域 (步)
    l2_update_interval: int = 15       # L2更新间隔 (分钟)
    l2_dt: float = 900.0               # L2时间步长 (秒, 15分钟)

    # 认知-物理耦合配置
    enable_cognitive_physics: bool = True    # 启用认知驱动的物理参数调整
    enable_scenario_adaptation: bool = True  # 启用场景自适应

    # 物理参数映射
    ice_flow_efficiency: float = 0.75        # 结冰时流速效率
    sediment_flow_efficiency: float = 0.85   # 淤积时流速效率
    flood_flow_efficiency: float = 1.1       # 洪水时流速效率 (可能略高)

    # 安全配置
    emergency_override: bool = True          # 紧急情况覆盖


@dataclass
class HierarchicalState:
    """双层控制器状态"""
    # L3状态
    l3_last_update_time: float = 0.0
    l3_last_result: Optional[SchedulingResult] = None
    l3_current_mode: SchedulingMode = SchedulingMode.NORMAL

    # L2状态
    l2_last_update_time: float = 0.0
    l2_current_trajectory: Optional[np.ndarray] = None

    # 物理场景状态
    current_physical_scenario: PhysicalScenario = PhysicalScenario.NORMAL
    current_flow_efficiencies: List[float] = field(default_factory=list)

    # 统计
    l3_updates: int = 0
    l2_updates: int = 0
    physical_changes: int = 0


class HierarchicalMPCController:
    """
    双层MPC控制器

    职责:
    1. 协调L3慢循环和L2快循环
    2. 将L3参考轨迹下发给L2
    3. 根据场景识别结果调整L2物理参数
    4. 处理紧急情况覆盖
    """

    def __init__(self,
                 num_pools: int = 3,
                 config: HierarchicalConfig = None,
                 topology: NetworkTopology = None):
        """
        初始化双层MPC控制器

        Args:
            num_pools: 池数量
            config: 双层配置
            topology: 水网拓扑
        """
        self.num_pools = num_pools
        self.config = config or HierarchicalConfig()

        # 创建拓扑
        if topology is None:
            topology = NetworkTopology.create_cascade(
                num_pools=num_pools,
                area=10000.0,
                level_min=0.5,
                level_max=8.0,
                max_flow=20.0
            )
        self.topology = topology

        # L3: 集中调度器
        l3_config = SchedulerConfig(
            planning_horizon=self.config.l3_planning_horizon,
            dt=self.config.l3_dt
        )
        self.l3_scheduler = CentralizedScheduler(topology, l3_config)

        # L2: 参数化分布式MPC
        self.l2_controller = ParameterizedDistributedMPC(
            num_pools=num_pools,
            horizon=self.config.l2_control_horizon,
            dt=self.config.l2_dt
        )

        # 决策引擎 (如果可用)
        self.decision_engine = DecisionEngine() if DECISION_ENGINE_AVAILABLE else None

        # 状态
        self.state = HierarchicalState(
            current_flow_efficiencies=[1.0] * num_pools
        )

        # 历史记录
        self.history = {
            'l3_schedules': [],
            'l2_controls': [],
            'physical_scenarios': [],
            'timestamps': []
        }

    def compute_control(self,
                        current_time: float,
                        current_levels: List[float],
                        current_flows: List[float],
                        current_demands: List[float],
                        weather_forecast: Dict = None,
                        demand_forecast: np.ndarray = None,
                        inflow_forecast: np.ndarray = None,
                        detected_scenario: str = None) -> Tuple[List[Tuple[float, float]], Dict]:
        """
        计算控制动作 (双层协调)

        Args:
            current_time: 当前时间 (小时)
            current_levels: 各池当前水位 [m]
            current_flows: 各池当前流量 [m³/s]
            current_demands: 各池当前需求 [m³/s]
            weather_forecast: 天气预报 (降雨等)
            demand_forecast: 需求预测 [num_pools x horizon]
            inflow_forecast: 入流预测 [horizon]
            detected_scenario: 检测到的物理场景 (如 'ICE_FORMATION')

        Returns:
            (控制动作列表, 调试信息)
        """
        debug_info = {
            'current_time': current_time,
            'l3_updated': False,
            'l2_updated': True,
            'physical_updated': False
        }

        # 1. 物理场景适应 (认知驱动)
        if self.config.enable_cognitive_physics and detected_scenario:
            physical_changed = self._update_physical_scenario(detected_scenario)
            debug_info['physical_updated'] = physical_changed
            debug_info['physical_scenario'] = self.state.current_physical_scenario.value
            debug_info['flow_efficiencies'] = self.state.current_flow_efficiencies.copy()

        # 2. L3慢循环检查
        time_since_l3 = current_time - self.state.l3_last_update_time
        l3_interval_hours = self.config.l3_update_interval

        if (time_since_l3 >= l3_interval_hours or
            self.state.l3_last_result is None or
            self._should_force_l3_update(weather_forecast)):

            # 执行L3调度
            l3_result = self._run_l3_scheduling(
                current_time, current_levels,
                weather_forecast, demand_forecast, inflow_forecast
            )

            debug_info['l3_updated'] = True
            debug_info['l3_result'] = {
                'success': l3_result.success,
                'mode': l3_result.mode.value,
                'solve_time': l3_result.solve_time,
                'recommendations': l3_result.recommendations
            }

            # 更新L2参考轨迹
            self._update_l2_trajectory(l3_result)

        # 3. L2快循环 (每次都执行)
        l2_result = self._run_l2_control(
            current_levels, current_flows, current_demands
        )

        debug_info['l2_result'] = {
            'converged': l2_result['info']['converged'],
            'iterations': l2_result['info']['iterations'],
            'solve_time': l2_result['info']['solve_time']
        }

        # 4. 紧急覆盖检查
        if self.config.emergency_override:
            l2_result = self._check_emergency_override(
                l2_result, current_levels, weather_forecast
            )
            debug_info['emergency_override'] = l2_result.get('emergency_applied', False)

        # 记录历史
        self._update_history(current_time, l2_result, debug_info)

        return l2_result['control_actions'], debug_info

    def _update_physical_scenario(self, detected_scenario: str) -> bool:
        """
        更新物理场景和参数

        Args:
            detected_scenario: 检测到的场景ID

        Returns:
            是否发生变化
        """
        try:
            new_scenario = PhysicalScenario(detected_scenario)
        except ValueError:
            # 尝试映射
            scenario_mapping = {
                'NORMAL_OPERATION': PhysicalScenario.NORMAL,
                'FLOOD_WARNING': PhysicalScenario.FLOOD,
                'DROUGHT_ALERT': PhysicalScenario.DROUGHT,
                'ICE_PERIOD': PhysicalScenario.ICE_STABLE
            }
            new_scenario = scenario_mapping.get(detected_scenario, PhysicalScenario.NORMAL)

        if new_scenario == self.state.current_physical_scenario:
            return False

        # 更新状态
        old_scenario = self.state.current_physical_scenario
        self.state.current_physical_scenario = new_scenario
        self.state.physical_changes += 1

        # 获取物理参数
        physics_params = ScenarioPhysics.get_parameters(new_scenario.value)
        new_efficiency = physics_params.get('flow_efficiency', 1.0)

        # 应用到所有池
        self.state.current_flow_efficiencies = [new_efficiency] * self.num_pools

        # 更新L2物理参数
        self.l2_controller.set_scenario_physics(new_scenario.value)

        logger.info(f"[物理场景切换] {old_scenario.value} -> {new_scenario.value}")
        logger.info(f"  流速效率: {new_efficiency:.2f}")

        return True

    def _should_force_l3_update(self, weather_forecast: Dict) -> bool:
        """检查是否需要强制更新L3"""
        if weather_forecast is None:
            return False

        # 暴雨预警触发强制更新
        rainfall = weather_forecast.get('rainfall', 0)
        if rainfall > 10.0:  # mm/h
            return True

        # 模式变化触发
        if self.l3_scheduler.current_mode != self.state.l3_current_mode:
            return True

        return False

    def _run_l3_scheduling(self,
                          current_time: float,
                          current_levels: List[float],
                          weather_forecast: Dict,
                          demand_forecast: np.ndarray,
                          inflow_forecast: np.ndarray) -> SchedulingResult:
        """执行L3调度"""
        # 转换为体积
        areas = self.topology.pool_areas
        current_volumes = [level * area for level, area in zip(current_levels, areas)]

        # 构建预测
        horizon = self.config.l3_planning_horizon

        # 默认预测
        if demand_forecast is None:
            demand_forecast = np.ones((self.num_pools, horizon)) * 5.0
        if inflow_forecast is None:
            inflow_forecast = np.ones(horizon) * 10.0

        # 天气预测
        rainfall = np.zeros(horizon)
        if weather_forecast and 'rainfall_forecast' in weather_forecast:
            rf = weather_forecast['rainfall_forecast']
            rainfall[:len(rf)] = rf[:horizon]
        elif weather_forecast and 'rainfall' in weather_forecast:
            rainfall[:] = weather_forecast['rainfall']

        forecast = SchedulingForecast(
            horizon=horizon,
            dt=self.config.l3_dt,
            demand_forecast=demand_forecast,
            inflow_forecast=inflow_forecast,
            rainfall_forecast=rainfall
        )

        # 调度
        result = self.l3_scheduler.schedule(current_volumes, forecast)

        # 更新状态
        self.state.l3_last_update_time = current_time
        self.state.l3_last_result = result
        self.state.l3_current_mode = result.mode
        self.state.l3_updates += 1

        # 记录
        self.history['l3_schedules'].append({
            'time': current_time,
            'mode': result.mode.value,
            'success': result.success
        })

        return result

    def _update_l2_trajectory(self, l3_result: SchedulingResult):
        """将L3结果转换为L2参考轨迹"""
        if l3_result.reference_levels is None:
            return

        # L3轨迹时间分辨率: 1小时
        # L2控制时域: 通常5步 x 15分钟 = 1.25小时
        # 需要插值

        l2_horizon = self.config.l2_control_horizon
        l3_dt = self.config.l3_dt / 3600  # 小时
        l2_dt = self.config.l2_dt / 3600  # 小时

        # 简化: 直接截取前几个点
        l3_ref = l3_result.reference_levels  # [num_pools x L3_horizon]

        # 线性插值到L2时间点
        l2_times = np.arange(l2_horizon) * l2_dt
        l3_times = np.arange(l3_ref.shape[1]) * l3_dt

        l2_trajectory = np.zeros((self.num_pools, l2_horizon))
        for i in range(self.num_pools):
            l2_trajectory[i, :] = np.interp(l2_times, l3_times, l3_ref[i, :])

        # 设置到L2
        self.l2_controller.set_reference_trajectories(l2_trajectory)
        self.state.l2_current_trajectory = l2_trajectory

    def _run_l2_control(self,
                       current_levels: List[float],
                       current_flows: List[float],
                       current_demands: List[float]) -> Dict:
        """执行L2控制"""
        # 求解
        control_actions, info = self.l2_controller.solve(
            current_levels=current_levels,
            q_in_prevs=current_flows[:self.num_pools]
        )

        self.state.l2_updates += 1

        return {
            'control_actions': control_actions,
            'info': info
        }

    def _check_emergency_override(self,
                                  l2_result: Dict,
                                  current_levels: List[float],
                                  weather_forecast: Dict) -> Dict:
        """紧急情况覆盖"""
        emergency_applied = False

        # 检查水位超限
        for i, level in enumerate(current_levels):
            if level > self.topology.pool_levels_max[i] - 0.5:
                # 接近上限, 强制增大出流
                q_in, q_out = l2_result['control_actions'][i]
                q_out_emergency = min(self.topology.max_flow_rates[i] if i < len(self.topology.max_flow_rates) else 20.0,
                                     q_out * 1.5)
                l2_result['control_actions'][i] = (q_in, q_out_emergency)
                emergency_applied = True
                logger.info(f"[紧急覆盖] 池{i}水位{level:.2f}m接近上限, 增大出流至{q_out_emergency:.2f}")

            elif level < self.topology.pool_levels_min[i] + 0.3:
                # 接近下限, 强制减小出流
                q_in, q_out = l2_result['control_actions'][i]
                q_out_emergency = max(0, q_out * 0.5)
                l2_result['control_actions'][i] = (q_in, q_out_emergency)
                emergency_applied = True
                logger.info(f"[紧急覆盖] 池{i}水位{level:.2f}m接近下限, 减小出流至{q_out_emergency:.2f}")

        l2_result['emergency_applied'] = emergency_applied
        return l2_result

    def _update_history(self, current_time: float, l2_result: Dict, debug_info: Dict):
        """更新历史记录"""
        self.history['timestamps'].append(current_time)
        self.history['l2_controls'].append({
            'time': current_time,
            'actions': l2_result['control_actions'],
            'converged': l2_result['info']['converged']
        })
        self.history['physical_scenarios'].append(self.state.current_physical_scenario.value)

    def get_current_trajectory(self) -> Optional[np.ndarray]:
        """获取当前参考轨迹"""
        return self.state.l2_current_trajectory

    def get_current_mode(self) -> str:
        """获取当前调度模式"""
        return self.state.l3_current_mode.value

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'l3_updates': self.state.l3_updates,
            'l2_updates': self.state.l2_updates,
            'physical_changes': self.state.physical_changes,
            'current_mode': self.state.l3_current_mode.value,
            'current_physical_scenario': self.state.current_physical_scenario.value,
            'current_flow_efficiencies': self.state.current_flow_efficiencies
        }

    def set_flow_efficiency_directly(self, pool_id: int, efficiency: float):
        """直接设置流速效率 (外部调用)"""
        if 0 <= pool_id < self.num_pools:
            self.state.current_flow_efficiencies[pool_id] = efficiency
            self.l2_controller.set_flow_efficiency(pool_id, efficiency)


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*15 + "双层MPC控制器演示")
    logger.info("="*70)

    # 创建双层控制器
    config = HierarchicalConfig(
        l3_planning_horizon=24,
        l3_update_interval=4,
        l2_control_horizon=5,
        l2_update_interval=15,
        enable_cognitive_physics=True
    )

    controller = HierarchicalMPCController(num_pools=3, config=config)

    logger.info(f"\n✓ 双层MPC控制器已初始化")
    logger.info(f"  L3规划时域: {config.l3_planning_horizon} 小时")
    logger.info(f"  L3更新间隔: {config.l3_update_interval} 小时")
    logger.info(f"  L2控制时域: {config.l2_control_horizon} 步")

    # 测试1: 正常运行
    logger.info(f"\n{'='*70}")
    logger.info("测试1: 正常运行模式")
    logger.info('='*70)

    current_time = 0.0
    current_levels = [3.0, 3.0, 3.0]
    current_flows = [5.0, 5.0, 5.0]
    current_demands = [5.0, 5.0, 5.0]

    control_actions, debug_info = controller.compute_control(
        current_time=current_time,
        current_levels=current_levels,
        current_flows=current_flows,
        current_demands=current_demands
    )

    logger.info(f"\n调试信息:")
    logger.info(f"  L3更新: {debug_info['l3_updated']}")
    if debug_info['l3_updated']:
        logger.info(f"  L3模式: {debug_info['l3_result']['mode']}")
        logger.info(f"  L3求解时间: {debug_info['l3_result']['solve_time']*1000:.1f} ms")
    logger.info(f"  L2收敛: {debug_info['l2_result']['converged']}")
    logger.info(f"  L2迭代: {debug_info['l2_result']['iterations']}")

    logger.info(f"\n控制动作:")
    for i, (q_in, q_out) in enumerate(control_actions):
        logger.info(f"  池{i}: q_in={q_in:.2f}, q_out={q_out:.2f} m³/s")

    # 测试2: 结冰场景 + 物理参数变化
    logger.info(f"\n{'='*70}")
    logger.info("测试2: 结冰场景 (认知驱动物理参数调整)")
    logger.info('='*70)

    current_time = 0.5  # 不触发L3更新

    control_actions, debug_info = controller.compute_control(
        current_time=current_time,
        current_levels=current_levels,
        current_flows=current_flows,
        current_demands=current_demands,
        detected_scenario='ICE_FORMATION'
    )

    logger.info(f"\n调试信息:")
    logger.info(f"  物理参数更新: {debug_info['physical_updated']}")
    logger.info(f"  当前物理场景: {debug_info.get('physical_scenario', 'N/A')}")
    logger.info(f"  流速效率: {debug_info.get('flow_efficiencies', [])}")

    # 测试3: 暴雨预警 -> L3强制更新
    logger.info(f"\n{'='*70}")
    logger.info("测试3: 暴雨预警 (L3强制更新 + 预泄模式)")
    logger.info('='*70)

    current_time = 1.0

    weather_forecast = {
        'rainfall': 15.0,  # mm/h 暴雨
        'rainfall_forecast': [0, 0, 5, 15, 20, 15, 10, 5, 2, 1] + [0]*14
    }

    # 恢复正常物理参数
    control_actions, debug_info = controller.compute_control(
        current_time=current_time,
        current_levels=current_levels,
        current_flows=current_flows,
        current_demands=current_demands,
        weather_forecast=weather_forecast,
        detected_scenario='NORMAL'
    )

    logger.info(f"\n调试信息:")
    logger.info(f"  L3强制更新: {debug_info['l3_updated']}")
    if debug_info['l3_updated']:
        logger.info(f"  L3模式: {debug_info['l3_result']['mode']}")
        for rec in debug_info['l3_result'].get('recommendations', []):
            logger.info(f"    - {rec}")

    # 显示参考轨迹
    trajectory = controller.get_current_trajectory()
    if trajectory is not None:
        logger.info(f"\n当前参考轨迹 (前5步):")
        for i in range(min(3, trajectory.shape[0])):
            levels = trajectory[i, :5]
            logger.info(f"  池{i}: {', '.join([f'{l:.2f}' for l in levels])} m")

    # 统计
    logger.info(f"\n{'='*70}")
    logger.info("统计信息")
    logger.info('='*70)
    stats = controller.get_statistics()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "="*70)
    logger.info("演示完成!")
    logger.info("="*70)
