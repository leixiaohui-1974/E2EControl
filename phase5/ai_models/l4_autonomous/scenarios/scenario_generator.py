"""
场景生成器 (Scenario Generator)
根据场景定义生成仿真数据

功能:
1. 基于场景参数生成时序数据
2. 支持动态组合场景
3. 支持随机扰动注入
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from .scenario_definitions import (
    Scenario,
    ScenarioParameters,
    ScenarioCategory,
    COMPLETE_SCENARIO_MATRIX,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """仿真状态"""
    time: float                              # 当前时间 (秒)
    levels: np.ndarray                       # 各渠池水位
    inflows: np.ndarray                      # 各渠池入流
    outflows: np.ndarray                     # 各渠池出流
    gate_openings: np.ndarray                # 各闸门开度
    diversions: np.ndarray                   # 各分水口流量
    sensor_status: np.ndarray                # 传感器状态
    gate_status: np.ndarray                  # 闸门状态


class ScenarioGenerator:
    """场景生成器"""

    def __init__(self, num_pools: int = 63, num_gates: int = 64, dt: float = 900.0):
        self.num_pools = num_pools
        self.num_gates = num_gates
        self.dt = dt  # 时间步长 (秒)

        # 物理参数
        self.pool_areas = np.ones(num_pools) * 100000.0  # 渠池面积 m²
        self.pool_lengths = np.ones(num_pools) * 22.7 * 1000  # 渠池长度 m (1432km/63)

    def generate(self, scenario: Scenario, num_steps: int = None) -> List[SimulationState]:
        """
        生成场景数据

        Args:
            scenario: 场景定义
            num_steps: 仿真步数

        Returns:
            状态序列
        """
        params = scenario.parameters

        # 计算步数
        if num_steps is None:
            num_steps = int(params.duration_hours * 3600 / self.dt)

        # 初始化状态
        state = self._initialize_state(params)

        # 生成序列
        states = [state]

        for step in range(1, num_steps):
            time = step * self.dt

            # 根据场景类型更新状态
            state = self._update_state(state, scenario, time)

            # 添加噪声
            state = self._add_noise(state, params.noise_level)

            states.append(state)

        return states

    def _initialize_state(self, params: ScenarioParameters) -> SimulationState:
        """初始化状态"""
        # 水位初始化
        level_mean = np.mean(params.level_range)
        levels = np.ones(self.num_pools) * level_mean

        # 流量初始化
        inflow_mean = np.mean(params.inflow_range)
        inflows = np.ones(self.num_pools) * inflow_mean
        outflows = inflows.copy()

        # 闸门初始化
        gate_openings = np.ones(self.num_gates) * 0.8

        # 分水初始化
        diversions = np.zeros(self.num_pools)
        if params.affected_pools:
            for pool_id in params.affected_pools:
                if pool_id < self.num_pools:
                    diversions[pool_id] = np.mean(params.demand_range)

        return SimulationState(
            time=0.0,
            levels=levels,
            inflows=inflows,
            outflows=outflows,
            gate_openings=gate_openings,
            diversions=diversions,
            sensor_status=np.ones(self.num_pools),
            gate_status=np.ones(self.num_gates)
        )

    def _update_state(self, prev_state: SimulationState,
                      scenario: Scenario, time: float) -> SimulationState:
        """更新状态"""
        params = scenario.parameters
        category = scenario.category

        # 复制前一状态
        levels = prev_state.levels.copy()
        inflows = prev_state.inflows.copy()
        outflows = prev_state.outflows.copy()
        gate_openings = prev_state.gate_openings.copy()
        diversions = prev_state.diversions.copy()
        sensor_status = prev_state.sensor_status.copy()
        gate_status = prev_state.gate_status.copy()

        # 计算时间进度
        duration_s = params.duration_hours * 3600
        ramp_s = params.ramp_time_hours * 3600
        progress = min(1.0, time / duration_s) if duration_s > 0 else 1.0
        ramp_progress = min(1.0, time / ramp_s) if ramp_s > 0 else 1.0

        # 根据场景类型更新
        if category == ScenarioCategory.NORMAL_OPERATION:
            inflows, outflows = self._update_normal(params, time, ramp_progress, inflows, outflows)

        elif category == ScenarioCategory.WATER_DEMAND:
            diversions = self._update_demand(params, time, ramp_progress, diversions)

        elif category == ScenarioCategory.EXTREME_WEATHER:
            inflows = self._update_weather(params, time, ramp_progress, inflows)

        elif category == ScenarioCategory.EQUIPMENT_FAILURE:
            sensor_status, gate_status = self._update_failure(params, time, sensor_status, gate_status)

        elif category == ScenarioCategory.EMERGENCY:
            inflows, gate_status = self._update_emergency(params, time, inflows, gate_status)

        elif category == ScenarioCategory.MAINTENANCE:
            gate_status = self._update_maintenance(params, time, gate_status)

        # 物理更新: 水位变化
        for i in range(self.num_pools):
            dQ = inflows[i] - outflows[i] - diversions[i]
            dZ = dQ * self.dt / self.pool_areas[i]
            levels[i] = np.clip(levels[i] + dZ, 0.5, 7.0)

        # 级联更新: 出流成为下一池入流
        for i in range(self.num_pools - 1):
            outflows[i] = inflows[i] * gate_openings[i + 1] - diversions[i]
            outflows[i] = max(0, outflows[i])
            inflows[i + 1] = outflows[i]

        return SimulationState(
            time=time,
            levels=levels,
            inflows=inflows,
            outflows=outflows,
            gate_openings=gate_openings,
            diversions=diversions,
            sensor_status=sensor_status,
            gate_status=gate_status
        )

    def _update_normal(self, params: ScenarioParameters, time: float,
                       ramp: float, inflows: np.ndarray, outflows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """更新正常运行场景"""
        # 日周期变化
        hour = (time / 3600) % 24
        daily_factor = 1.0 + 0.1 * np.sin(2 * np.pi * hour / 24)

        base_flow = np.mean(params.inflow_range)
        variation = (params.inflow_range[1] - params.inflow_range[0]) / 2

        inflows[0] = base_flow + variation * daily_factor * ramp
        for i in range(1, len(inflows)):
            inflows[i] = inflows[i - 1] * 0.995  # 沿程损失

        return inflows, outflows

    def _update_demand(self, params: ScenarioParameters, time: float,
                       ramp: float, diversions: np.ndarray) -> np.ndarray:
        """更新需水场景"""
        demand_min, demand_max = params.demand_range

        for pool_id in params.affected_pools:
            if pool_id < len(diversions):
                target_demand = demand_min + (demand_max - demand_min) * ramp
                diversions[pool_id] = target_demand

        return diversions

    def _update_weather(self, params: ScenarioParameters, time: float,
                        ramp: float, inflows: np.ndarray) -> np.ndarray:
        """更新天气场景"""
        base_flow = np.mean(params.inflow_range)
        peak_flow = params.inflow_range[1]

        # 雨量曲线
        weather_factor = ramp * params.disturbance_magnitude

        inflows[0] = base_flow + (peak_flow - base_flow) * weather_factor
        for i in range(1, len(inflows)):
            # 延迟传播
            delay_factor = max(0, ramp - i * 0.02)
            inflows[i] = base_flow + (peak_flow - base_flow) * delay_factor * weather_factor

        return inflows

    def _update_failure(self, params: ScenarioParameters, time: float,
                        sensor_status: np.ndarray, gate_status: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """更新故障场景"""
        for pool_id in params.affected_pools:
            if pool_id < len(sensor_status):
                sensor_status[pool_id] = 0  # 传感器故障
            if pool_id < len(gate_status):
                gate_status[pool_id] = 0  # 闸门故障

        return sensor_status, gate_status

    def _update_emergency(self, params: ScenarioParameters, time: float,
                          inflows: np.ndarray, gate_status: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """更新应急场景"""
        # 突发大流量
        if params.inflow_range[1] > 400:
            inflows[0] = params.inflow_range[1]

        return inflows, gate_status

    def _update_maintenance(self, params: ScenarioParameters, time: float,
                            gate_status: np.ndarray) -> np.ndarray:
        """更新维护场景"""
        for pool_id in params.affected_pools:
            if pool_id < len(gate_status):
                gate_status[pool_id] = 0.5  # 维护中

        return gate_status

    def _add_noise(self, state: SimulationState, noise_level: float) -> SimulationState:
        """添加噪声"""
        if noise_level <= 0:
            return state

        state.levels += np.random.normal(0, noise_level * 0.1, self.num_pools)
        state.inflows += np.random.normal(0, noise_level * 10, self.num_pools)
        state.outflows += np.random.normal(0, noise_level * 10, self.num_pools)

        # 确保非负
        state.levels = np.clip(state.levels, 0.5, 7.0)
        state.inflows = np.maximum(state.inflows, 0)
        state.outflows = np.maximum(state.outflows, 0)

        return state


class DynamicScenarioBuilder:
    """动态场景构建器"""

    def __init__(self):
        self.scenarios = []
        self.transitions = []

    def add_scenario(self, scenario: Scenario, start_time: float, end_time: float):
        """添加场景段"""
        self.scenarios.append({
            'scenario': scenario,
            'start': start_time,
            'end': end_time
        })

    def add_transition(self, from_scenario: str, to_scenario: str,
                       transition_time: float):
        """添加场景转换"""
        self.transitions.append({
            'from': from_scenario,
            'to': to_scenario,
            'time': transition_time
        })

    def build(self, total_duration: float, dt: float = 900.0) -> List[Dict]:
        """构建复合场景"""
        num_steps = int(total_duration / dt)
        timeline = []

        for step in range(num_steps):
            time = step * dt

            # 确定当前激活的场景
            active_scenarios = []
            for item in self.scenarios:
                if item['start'] <= time < item['end']:
                    active_scenarios.append(item['scenario'])

            timeline.append({
                'time': time,
                'scenarios': active_scenarios
            })

        return timeline

    def create_stress_test(self, duration_hours: float = 24.0) -> List[Dict]:
        """创建压力测试场景"""
        # 组合多个困难场景
        self.add_scenario(
            COMPLETE_SCENARIO_MATRIX["WEATHER_001"],
            0, duration_hours * 3600 * 0.3
        )
        self.add_scenario(
            COMPLETE_SCENARIO_MATRIX["DEMAND_002"],
            duration_hours * 3600 * 0.2, duration_hours * 3600 * 0.5
        )
        self.add_scenario(
            COMPLETE_SCENARIO_MATRIX["FAILURE_001"],
            duration_hours * 3600 * 0.4, duration_hours * 3600 * 0.6
        )
        self.add_scenario(
            COMPLETE_SCENARIO_MATRIX["EMERGENCY_001"],
            duration_hours * 3600 * 0.7, duration_hours * 3600 * 0.8
        )

        return self.build(duration_hours * 3600)


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("场景生成器测试")
    logger.info("=" * 70)

    generator = ScenarioGenerator(num_pools=10, num_gates=11)

    # 测试单个场景
    scenario = COMPLETE_SCENARIO_MATRIX["NORMAL_001"]
    states = generator.generate(scenario, num_steps=100)

    logger.info(f"\n场景: {scenario.name_cn}")
    logger.info(f"生成步数: {len(states)}")
    logger.info(f"初始水位: {states[0].levels[:5]}")
    logger.info(f"最终水位: {states[-1].levels[:5]}")

    # 测试动态场景
    builder = DynamicScenarioBuilder()
    timeline = builder.create_stress_test(24.0)
    logger.info(f"\n压力测试场景段数: {len(timeline)}")

    logger.info("\n" + "=" * 70)
