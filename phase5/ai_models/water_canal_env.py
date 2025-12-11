"""
水渠强化学习环境 (Water Canal RL Environment)
Gymnasium API Compatible Environment for RL Training

基于"一闸两渠池"拓扑的标准RL环境:
- Agent控制中间闸门
- 观测上游渠池水位和下游渠池水位
- 复合奖励函数 (水位控制 + 动作平滑 + 安全约束)

技术特点:
1. Gymnasium API兼容 - 支持标准RL算法库 (Stable-Baselines3, RLlib等)
2. 生成式边界条件 - 使用ScenarioVAE生成来水/需水
3. 神经物理引擎 - 使用NeuralPhysicsEngine进行推演
4. 可配置奖励 - 支持多目标优化
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging

# Gymnasium导入 (兼容gym和gymnasium)
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    import gym
    from gym import spaces
    GYMNASIUM_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==============================================================================
# 配置数据类
# ==============================================================================

@dataclass
class WaterCanalEnvConfig:
    """水渠环境配置"""
    # 时间参数
    dt: float = 900.0                   # 时间步长 [s] (15分钟)
    episode_duration: float = 86400.0   # 单episode时长 [s] (24小时)

    # 观测空间
    observation_window: int = 16        # 历史观测窗口
    num_features: int = 5               # 特征数 [水位_上, 水位_下, 流量, 闸门开度, reward]

    # 动作空间
    action_low: float = -0.1            # 闸门开度增量下限
    action_high: float = 0.1            # 闸门开度增量上限

    # 物理参数
    upstream_pool_area: float = 80000.0   # 上游渠池面积 [m²]
    downstream_pool_area: float = 100000.0  # 下游渠池面积 [m²]
    gate_delay: int = 4                   # 闸门响应延迟 [步]

    # 目标参数
    target_level_upstream: float = 4.0   # 上游目标水位 [m]
    target_level_downstream: float = 3.8  # 下游目标水位 [m]
    level_tolerance: float = 0.2          # 水位容差 [m]

    # 约束参数
    min_level: float = 1.5               # 最低安全水位 [m]
    max_level: float = 5.5               # 最高安全水位 [m]
    min_gate: float = 0.1                # 最小闸门开度
    max_gate: float = 1.0                # 最大闸门开度

    # 奖励权重
    w_level: float = 1.0                 # 水位偏差权重
    w_action: float = 0.1                # 动作平滑权重
    w_safety: float = 10.0               # 安全约束权重

    # 随机性
    inflow_noise_std: float = 5.0        # 入流噪声标准差
    demand_noise_std: float = 3.0        # 需水噪声标准差


# ==============================================================================
# 简化物理模型 (用于环境内部)
# ==============================================================================

class SimplePoolDynamics:
    """简化的单池动态模型"""

    def __init__(self,
                 area: float,
                 delay_steps: int = 4,
                 dt: float = 900.0):
        self.area = area
        self.delay_steps = delay_steps
        self.dt = dt

        # 入流历史缓冲
        self.inflow_buffer = []
        self.current_level = 4.0

    def reset(self, initial_level: float = 4.0, initial_inflow: float = 100.0):
        """重置状态"""
        self.current_level = initial_level
        self.inflow_buffer = [initial_inflow] * (self.delay_steps + 1)

    def step(self, q_in: float, q_out: float) -> float:
        """
        单步更新

        Args:
            q_in: 入流
            q_out: 出流

        Returns:
            新水位
        """
        # 存储入流
        self.inflow_buffer.append(q_in)
        if len(self.inflow_buffer) > self.delay_steps + 1:
            self.inflow_buffer.pop(0)

        # 延迟入流
        q_in_delayed = self.inflow_buffer[0]

        # 水位更新
        dZ = (q_in_delayed - q_out) * self.dt / self.area
        self.current_level += dZ

        return self.current_level


# ==============================================================================
# 一闸两渠池环境
# ==============================================================================

class OneGateTwoPoolsEnv(gym.Env):
    """
    一闸两渠池强化学习环境

    拓扑:
        [上游渠池] ---> [闸门(Agent控制)] ---> [下游渠池]
              ↑                                    ↓
         (上游来水)                            (下游需水)

    观测:
        - 上游水位历史
        - 下游水位历史
        - 当前流量
        - 当前闸门开度
        - 上一步奖励

    动作:
        - 闸门开度增量 Δu ∈ [-0.1, 0.1]

    奖励:
        R = -w1*|Z_up - Z_target_up|² - w1*|Z_down - Z_target_down|²
            - w2*|Δu|²
            + R_safety
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 config: WaterCanalEnvConfig = None,
                 render_mode: str = None):
        super().__init__()

        self.config = config or WaterCanalEnvConfig()
        self.render_mode = render_mode

        # 定义观测空间
        obs_shape = (self.config.observation_window, self.config.num_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # 定义动作空间 (闸门开度增量)
        self.action_space = spaces.Box(
            low=np.array([self.config.action_low]),
            high=np.array([self.config.action_high]),
            dtype=np.float32
        )

        # 初始化物理模型
        self.upstream_pool = SimplePoolDynamics(
            area=self.config.upstream_pool_area,
            delay_steps=self.config.gate_delay,
            dt=self.config.dt
        )
        self.downstream_pool = SimplePoolDynamics(
            area=self.config.downstream_pool_area,
            delay_steps=self.config.gate_delay,
            dt=self.config.dt
        )

        # 状态变量
        self.gate_opening = 0.5
        self.current_step = 0
        self.max_steps = int(self.config.episode_duration / self.config.dt)

        # 历史缓冲
        self.obs_history = []
        self.last_reward = 0.0

        # 边界条件生成器 (可以替换为ScenarioVAE)
        self._inflow_generator = None
        self._demand_generator = None

        logger.info(f"OneGateTwoPoolsEnv 初始化完成")
        logger.info(f"  观测空间: {self.observation_space.shape}")
        logger.info(f"  动作空间: {self.action_space.shape}")

    def reset(self,
              seed: int = None,
              options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境

        Args:
            seed: 随机种子
            options: 可选参数

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        # 随机初始化
        initial_level_up = np.random.uniform(3.5, 4.5)
        initial_level_down = np.random.uniform(3.3, 4.3)
        initial_flow = np.random.uniform(150, 250)

        self.upstream_pool.reset(initial_level_up, initial_flow)
        self.downstream_pool.reset(initial_level_down, initial_flow)

        self.gate_opening = np.random.uniform(0.4, 0.6)
        self.current_step = 0
        self.last_reward = 0.0

        # 初始化观测历史
        initial_obs = self._get_current_obs()
        self.obs_history = [initial_obs.copy() for _ in range(self.config.observation_window)]

        # 生成边界条件
        self._generate_boundary_conditions()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步

        Args:
            action: 闸门开度增量 [Δu]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # 解析动作
        delta_gate = float(action[0])

        # 更新闸门开度
        self.gate_opening = np.clip(
            self.gate_opening + delta_gate,
            self.config.min_gate,
            self.config.max_gate
        )

        # 获取边界条件
        upstream_inflow = self._get_upstream_inflow()
        downstream_demand = self._get_downstream_demand()

        # 计算闸门流量 (简化: 基于上游水位和闸门开度)
        gate_flow = self._compute_gate_flow()

        # 更新物理状态
        level_up = self.upstream_pool.step(upstream_inflow, gate_flow)
        level_down = self.downstream_pool.step(gate_flow, downstream_demand)

        # 限制水位范围
        level_up = np.clip(level_up, self.config.min_level, self.config.max_level)
        level_down = np.clip(level_down, self.config.min_level, self.config.max_level)
        self.upstream_pool.current_level = level_up
        self.downstream_pool.current_level = level_down

        # 计算奖励
        reward = self._compute_reward(level_up, level_down, delta_gate)
        self.last_reward = reward

        # 更新观测历史
        current_obs = self._get_current_obs()
        self.obs_history.append(current_obs)
        if len(self.obs_history) > self.config.observation_window:
            self.obs_history.pop(0)

        # 更新步数
        self.current_step += 1

        # 检查终止条件
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_current_obs(self) -> np.ndarray:
        """获取当前时刻的观测向量"""
        return np.array([
            self.upstream_pool.current_level,
            self.downstream_pool.current_level,
            self._compute_gate_flow(),
            self.gate_opening,
            self.last_reward
        ], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """获取完整观测 (历史窗口)"""
        obs = np.array(self.obs_history, dtype=np.float32)
        return obs

    def _compute_gate_flow(self) -> float:
        """计算闸门流量"""
        # 简化模型: Q = Cd * A * sqrt(2 * g * h)
        # 这里用更简化的线性关系
        h = self.upstream_pool.current_level
        max_flow = 300.0  # 最大流量
        flow = max_flow * self.gate_opening * np.sqrt(max(0.1, h) / 4.0)
        return flow

    def _compute_reward(self,
                        level_up: float,
                        level_down: float,
                        delta_gate: float) -> float:
        """
        计算复合奖励

        R = -w1*|Z_up - Z_target|² - w1*|Z_down - Z_target|²
            - w2*|Δu|²
            + R_safety
        """
        config = self.config

        # 水位偏差惩罚
        level_error_up = (level_up - config.target_level_upstream) ** 2
        level_error_down = (level_down - config.target_level_downstream) ** 2
        level_penalty = config.w_level * (level_error_up + level_error_down)

        # 动作平滑惩罚
        action_penalty = config.w_action * (delta_gate ** 2)

        # 安全约束奖励/惩罚
        safety_reward = 0.0

        # 水位越界惩罚
        if level_up < config.min_level + 0.5 or level_up > config.max_level - 0.5:
            safety_reward -= config.w_safety
        if level_down < config.min_level + 0.5 or level_down > config.max_level - 0.5:
            safety_reward -= config.w_safety

        # 水位在目标范围内奖励
        if abs(level_up - config.target_level_upstream) < config.level_tolerance:
            safety_reward += 1.0
        if abs(level_down - config.target_level_downstream) < config.level_tolerance:
            safety_reward += 1.0

        # 总奖励
        reward = -level_penalty - action_penalty + safety_reward

        return reward

    def _check_terminated(self) -> bool:
        """检查是否触发终止条件"""
        config = self.config

        # 水位严重越界
        if (self.upstream_pool.current_level < config.min_level or
            self.upstream_pool.current_level > config.max_level or
            self.downstream_pool.current_level < config.min_level or
            self.downstream_pool.current_level > config.max_level):
            return True

        return False

    def _get_info(self) -> Dict:
        """获取附加信息"""
        return {
            'step': self.current_step,
            'level_upstream': self.upstream_pool.current_level,
            'level_downstream': self.downstream_pool.current_level,
            'gate_opening': self.gate_opening,
            'gate_flow': self._compute_gate_flow(),
        }

    def _generate_boundary_conditions(self):
        """生成边界条件 (来水和需水曲线)"""
        # 简单实现: 使用正弦波 + 噪声模拟日变化
        t = np.arange(self.max_steps)

        # 上游来水: 基础流量 + 日变化 + 噪声
        base_inflow = 200.0
        daily_variation = 30.0 * np.sin(2 * np.pi * t / (self.max_steps))
        noise = np.random.normal(0, self.config.inflow_noise_std, self.max_steps)
        self._inflow_curve = base_inflow + daily_variation + noise

        # 下游需水: 基础需求 + 日变化 + 噪声
        base_demand = 180.0
        demand_variation = 20.0 * np.sin(2 * np.pi * t / (self.max_steps) + np.pi/4)
        demand_noise = np.random.normal(0, self.config.demand_noise_std, self.max_steps)
        self._demand_curve = base_demand + demand_variation + demand_noise

    def _get_upstream_inflow(self) -> float:
        """获取当前时刻的上游来水"""
        if hasattr(self, '_inflow_curve') and self.current_step < len(self._inflow_curve):
            return self._inflow_curve[self.current_step]
        return 200.0

    def _get_downstream_demand(self) -> float:
        """获取当前时刻的下游需水"""
        if hasattr(self, '_demand_curve') and self.current_step < len(self._demand_curve):
            return self._demand_curve[self.current_step]
        return 180.0

    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            print(f"Step {self.current_step}: "
                  f"Level_up={self.upstream_pool.current_level:.2f}m, "
                  f"Level_down={self.downstream_pool.current_level:.2f}m, "
                  f"Gate={self.gate_opening:.2f}")
        return None

    def close(self):
        """关闭环境"""
        pass


# ==============================================================================
# 带神经物理引擎的环境
# ==============================================================================

class WaterCanalEnv(OneGateTwoPoolsEnv):
    """
    带神经物理引擎的水渠环境

    扩展OneGateTwoPoolsEnv:
    - 使用NeuralPhysicsEngine进行推演
    - 使用ScenarioVAE生成边界条件
    """

    def __init__(self,
                 config: WaterCanalEnvConfig = None,
                 neural_engine=None,
                 scenario_generator=None,
                 render_mode: str = None):
        super().__init__(config, render_mode)

        self.neural_engine = neural_engine
        self.scenario_generator = scenario_generator

    def set_neural_engine(self, engine):
        """设置神经物理引擎"""
        self.neural_engine = engine

    def set_scenario_generator(self, generator):
        """设置场景生成器"""
        self.scenario_generator = generator

    def _generate_boundary_conditions(self):
        """使用AI生成边界条件"""
        if self.scenario_generator is not None:
            # 随机选择条件
            conditions = np.random.choice(
                ['normal', 'flood', 'drought'],
                p=[0.7, 0.15, 0.15]
            )
            seasons = ['spring', 'summer', 'autumn', 'winter']
            season = np.random.choice(seasons)

            try:
                scenarios = self.scenario_generator.generate_from_ai(
                    conditions=[season, conditions],
                    num_scenarios=1,
                    duration_hours=self.config.episode_duration / 3600
                )
                self._inflow_curve = scenarios['upstream_flow'][0]
                self._demand_curve = scenarios['downstream_demand'][0]
                return
            except Exception as e:
                logger.warning(f"AI场景生成失败: {e}, 使用默认生成器")

        # 回退到父类实现
        super()._generate_boundary_conditions()


# ==============================================================================
# 多智能体环境 (PettingZoo风格)
# ==============================================================================

class MultiGateEnv:
    """
    多闸门协同控制环境

    支持多个Agent分别控制不同闸门
    (简化实现，完整版本需要继承PettingZoo的ParallelEnv)
    """

    def __init__(self,
                 num_gates: int = 3,
                 config: WaterCanalEnvConfig = None):
        self.num_gates = num_gates
        self.config = config or WaterCanalEnvConfig()

        # 为每个闸门创建观测和动作空间
        self.agents = [f"gate_{i}" for i in range(num_gates)]

        obs_shape = (self.config.observation_window, self.config.num_features)
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                low=np.array([self.config.action_low]),
                high=np.array([self.config.action_high]),
                dtype=np.float32
            )
            for agent in self.agents
        }

        # 渠池模型
        self.pools = [
            SimplePoolDynamics(
                area=self.config.downstream_pool_area,
                delay_steps=self.config.gate_delay
            )
            for _ in range(num_gates + 1)
        ]

        self.gate_openings = [0.5] * num_gates

    def reset(self, seed=None):
        """重置环境"""
        for pool in self.pools:
            pool.reset(
                initial_level=np.random.uniform(3.5, 4.5),
                initial_inflow=np.random.uniform(150, 250)
            )
        self.gate_openings = [0.5] * self.num_gates

        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        """执行一步"""
        # 更新所有闸门
        for i, agent in enumerate(self.agents):
            if agent in actions:
                delta = float(actions[agent][0])
                self.gate_openings[i] = np.clip(
                    self.gate_openings[i] + delta,
                    self.config.min_gate,
                    self.config.max_gate
                )

        # 更新物理状态 (从上游到下游)
        flow = 200.0  # 源头流量
        for i in range(self.num_gates + 1):
            if i < self.num_gates:
                gate_flow = flow * self.gate_openings[i]
            else:
                gate_flow = flow * 0.9  # 末端需水

            level = self.pools[i].step(flow, gate_flow)
            flow = gate_flow

        # 计算奖励
        rewards = {}
        for i, agent in enumerate(self.agents):
            rewards[agent] = self._compute_agent_reward(i)

        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent_idx: int) -> np.ndarray:
        """获取指定Agent的观测"""
        obs = np.zeros((self.config.observation_window, self.config.num_features), dtype=np.float32)
        # 简化: 只填充最后一个时刻
        obs[-1] = [
            self.pools[agent_idx].current_level,
            self.pools[agent_idx + 1].current_level,
            200.0 * self.gate_openings[agent_idx],
            self.gate_openings[agent_idx],
            0.0
        ]
        return obs

    def _compute_agent_reward(self, agent_idx: int) -> float:
        """计算单个Agent的奖励"""
        level_up = self.pools[agent_idx].current_level
        level_down = self.pools[agent_idx + 1].current_level

        error = (level_up - 4.0) ** 2 + (level_down - 4.0) ** 2
        return -error


# ==============================================================================
# Random Agent测试
# ==============================================================================

def test_random_agent(env, num_episodes: int = 3, max_steps: int = 100):
    """使用Random Agent测试环境"""
    print("\n" + "=" * 70)
    print("Random Agent 测试")
    print("=" * 70)

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0

        print(f"\nEpisode {episode + 1}")
        print("-" * 50)

        for step in range(max_steps):
            # 随机动作
            action = env.action_space.sample()

            # 执行
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 20 == 0:
                print(f"  Step {step:3d}: "
                      f"Level_up={info['level_upstream']:.2f}m, "
                      f"Level_down={info['level_downstream']:.2f}m, "
                      f"Gate={info['gate_opening']:.2f}, "
                      f"Reward={reward:.2f}")

            if terminated or truncated:
                break

        print(f"\n  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step + 1}")

    return True


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "水渠RL环境测试")
    print("=" * 70)

    # 创建配置
    config = WaterCanalEnvConfig(
        episode_duration=14400.0,  # 4小时测试
        w_level=1.0,
        w_action=0.1,
        w_safety=5.0,
    )

    # 创建环境
    env = OneGateTwoPoolsEnv(config)

    print(f"\n环境信息:")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  每episode最大步数: {env.max_steps}")

    # 测试reset
    print(f"\n测试 reset()...")
    obs, info = env.reset()
    print(f"  观测形状: {obs.shape}")
    print(f"  初始信息: {info}")

    # 测试step
    print(f"\n测试 step()...")
    action = np.array([0.05])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  动作: {action}")
    print(f"  奖励: {reward:.4f}")
    print(f"  终止: {terminated}")
    print(f"  截断: {truncated}")

    # Random Agent测试
    test_random_agent(env, num_episodes=2, max_steps=50)

    # 测试多闸门环境
    print("\n" + "=" * 70)
    print("多闸门环境测试")
    print("=" * 70)

    multi_env = MultiGateEnv(num_gates=3, config=config)
    observations, infos = multi_env.reset()

    print(f"\n  Agent数量: {len(multi_env.agents)}")
    print(f"  Agents: {multi_env.agents}")

    # 执行一步
    actions = {agent: np.array([0.01]) for agent in multi_env.agents}
    observations, rewards, terms, truncs, infos = multi_env.step(actions)

    print(f"\n  奖励: {rewards}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
