"""
多等级统一控制器 (Multi-Level Unified Controller)
支持L0-L4全等级能力的南水北调中线控制系统

自动化等级定义:
- L0 (人工): 完全人工操作，系统仅提供显示
- L1 (辅助): 系统提供建议，人工确认执行
- L2 (部分自动): PID/MPC自动控制，人工监督
- L3 (条件自动): 神经网络控制，特定场景自动
- L4 (高度自动): 端到端自主，仅异常时人工介入

设计原则:
1. 各等级能力独立实现，可组合使用
2. 支持动态等级切换
3. 等级间平滑过渡
4. 完整的降级回退机制
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
import time

logger = logging.getLogger(__name__)


# ==============================================================================
# 等级定义
# ==============================================================================

class AutonomyLevel(Enum):
    """自动化等级"""
    L0_MANUAL = 0           # 完全人工
    L1_ASSISTED = 1         # 辅助决策
    L2_PARTIAL = 2          # 部分自动 (PID/MPC)
    L3_CONDITIONAL = 3      # 条件自动 (神经代理)
    L4_HIGH = 4             # 高度自动 (端到端)


class ControlMode(Enum):
    """控制模式"""
    MANUAL = auto()         # 人工模式
    ADVISORY = auto()       # 建议模式
    SUPERVISED = auto()     # 监督自动
    AUTONOMOUS = auto()     # 自主模式
    EMERGENCY = auto()      # 紧急模式


@dataclass
class LevelCapabilities:
    """各等级能力描述"""
    level: AutonomyLevel
    description: str
    features: List[str]
    human_role: str
    system_role: str


# 等级能力定义
LEVEL_CAPABILITIES = {
    AutonomyLevel.L0_MANUAL: LevelCapabilities(
        level=AutonomyLevel.L0_MANUAL,
        description="完全人工操作",
        features=["数据显示", "状态监控", "历史查询", "报警提示"],
        human_role="执行所有控制决策和操作",
        system_role="提供信息显示和状态监控"
    ),
    AutonomyLevel.L1_ASSISTED: LevelCapabilities(
        level=AutonomyLevel.L1_ASSISTED,
        description="辅助决策",
        features=["控制建议", "异常检测", "趋势预测", "风险评估"],
        human_role="审核建议并执行操作",
        system_role="生成控制建议，提供决策支持"
    ),
    AutonomyLevel.L2_PARTIAL: LevelCapabilities(
        level=AutonomyLevel.L2_PARTIAL,
        description="部分自动化",
        features=["PID控制", "MPC优化", "自动调节", "边界保护"],
        human_role="监督控制效果，处理异常",
        system_role="自动执行预设控制策略"
    ),
    AutonomyLevel.L3_CONDITIONAL: LevelCapabilities(
        level=AutonomyLevel.L3_CONDITIONAL,
        description="条件自动化",
        features=["神经网络控制", "场景识别", "自适应调整", "预测控制"],
        human_role="处理未知场景和复杂异常",
        system_role="识别场景并自动执行最优控制"
    ),
    AutonomyLevel.L4_HIGH: LevelCapabilities(
        level=AutonomyLevel.L4_HIGH,
        description="高度自动化",
        features=["端到端自主", "多智能体协同", "在线学习", "自主决策"],
        human_role="监控系统状态，紧急接管",
        system_role="全流程自主运行，自主处理大部分异常"
    ),
}


# ==============================================================================
# 基础控制器接口
# ==============================================================================

class BaseController(ABC):
    """控制器基类"""

    @abstractmethod
    def compute_action(self, state: Dict) -> Dict:
        """计算控制动作"""
        pass

    @abstractmethod
    def get_level(self) -> AutonomyLevel:
        """获取控制等级"""
        pass


# ==============================================================================
# L0: 人工控制模块
# ==============================================================================

class L0ManualController(BaseController):
    """
    L0级: 人工控制
    系统仅提供显示和监控功能
    """

    def __init__(self, num_gates: int = 64):
        self.num_gates = num_gates
        self.pending_commands = {}  # 待执行的人工命令

    def get_level(self) -> AutonomyLevel:
        return AutonomyLevel.L0_MANUAL

    def compute_action(self, state: Dict) -> Dict:
        """返回人工设定的命令"""
        return {
            'action': self.pending_commands.copy(),
            'source': 'manual',
            'confidence': 1.0,  # 人工命令置信度为1
            'requires_confirmation': False
        }

    def set_gate_command(self, gate_id: int, opening: float):
        """设置单个闸门命令"""
        self.pending_commands[gate_id] = np.clip(opening, 0.0, 1.0)

    def set_all_commands(self, openings: np.ndarray):
        """设置所有闸门命令"""
        for i, opening in enumerate(openings):
            self.pending_commands[i] = np.clip(opening, 0.0, 1.0)

    def clear_commands(self):
        """清除待执行命令"""
        self.pending_commands.clear()

    def get_display_data(self, state: Dict) -> Dict:
        """获取显示数据"""
        return {
            'current_levels': state.get('levels', []),
            'current_flows': state.get('flows', []),
            'gate_status': state.get('gates', []),
            'alarms': state.get('alarms', []),
            'timestamp': time.time()
        }


# ==============================================================================
# L1: 辅助决策模块
# ==============================================================================

class L1AdvisoryController(BaseController):
    """
    L1级: 辅助决策
    系统生成控制建议，人工确认后执行
    """

    def __init__(self, num_gates: int = 64, num_pools: int = 63):
        self.num_gates = num_gates
        self.num_pools = num_pools
        self.current_advice = None
        self.advice_history = []

    def get_level(self) -> AutonomyLevel:
        return AutonomyLevel.L1_ASSISTED

    def compute_action(self, state: Dict) -> Dict:
        """生成控制建议"""
        levels = np.array(state.get('levels', [4.0] * self.num_pools))
        targets = np.array(state.get('targets', [4.0] * self.num_pools))
        current_gates = np.array(state.get('gates', [0.8] * self.num_gates))

        # 简单的比例控制建议
        errors = targets - levels
        suggested_changes = np.zeros(self.num_gates)

        for i in range(min(self.num_pools, self.num_gates - 1)):
            # 水位低于目标：减少下游出流
            # 水位高于目标：增加下游出流
            suggested_changes[i + 1] = 0.1 * errors[i]

        suggested_openings = np.clip(current_gates + suggested_changes, 0.0, 1.0)

        # 生成建议
        advice = {
            'suggested_openings': suggested_openings,
            'changes': suggested_changes,
            'reasoning': self._generate_reasoning(errors, suggested_changes),
            'confidence': self._estimate_confidence(state),
            'priority': self._determine_priority(errors)
        }

        self.current_advice = advice
        self.advice_history.append({
            'timestamp': time.time(),
            'advice': advice
        })

        return {
            'action': None,  # L1不自动执行
            'advice': advice,
            'source': 'advisory',
            'requires_confirmation': True
        }

    def _generate_reasoning(self, errors: np.ndarray, changes: np.ndarray) -> List[str]:
        """生成建议理由"""
        reasons = []
        for i, (err, chg) in enumerate(zip(errors, changes[1:])):
            if abs(err) > 0.1:
                direction = "增加" if err < 0 else "减少"
                reasons.append(f"渠池{i}: 水位偏差{err:.2f}m，建议{direction}下游闸门开度")
        return reasons if reasons else ["当前状态良好，无需调整"]

    def _estimate_confidence(self, state: Dict) -> float:
        """估计建议置信度"""
        # 基于状态稳定性评估
        levels = state.get('levels', [])
        if not levels:
            return 0.5

        # 水位变化率
        level_std = np.std(levels)
        confidence = 1.0 / (1.0 + level_std)
        return float(confidence)

    def _determine_priority(self, errors: np.ndarray) -> str:
        """确定建议优先级"""
        max_error = np.max(np.abs(errors))
        if max_error > 0.5:
            return "HIGH"
        elif max_error > 0.2:
            return "MEDIUM"
        else:
            return "LOW"

    def confirm_advice(self) -> Dict:
        """确认执行建议"""
        if self.current_advice is None:
            return {'action': None, 'status': 'no_pending_advice'}

        return {
            'action': self.current_advice['suggested_openings'],
            'status': 'confirmed',
            'source': 'advisory_confirmed'
        }


# ==============================================================================
# L2: 部分自动控制模块
# ==============================================================================

class L2PartialAutoController(BaseController):
    """
    L2级: 部分自动化
    PID/MPC自动控制，人工监督
    """

    def __init__(self, num_gates: int = 64, num_pools: int = 63, dt: float = 900.0):
        self.num_gates = num_gates
        self.num_pools = num_pools
        self.dt = dt

        # PID参数
        self.Kp = 0.5
        self.Ki = 0.01
        self.Kd = 0.1

        # 积分项
        self.integral = np.zeros(num_pools)
        self.prev_error = np.zeros(num_pools)

        # 控制边界
        self.min_opening = 0.1
        self.max_opening = 1.0
        self.max_change_rate = 0.1  # 最大变化率 (/step)

        # 安全边界
        self.level_min = 1.5
        self.level_max = 6.0

    def get_level(self) -> AutonomyLevel:
        return AutonomyLevel.L2_PARTIAL

    def compute_action(self, state: Dict) -> Dict:
        """PID控制计算"""
        levels = np.array(state.get('levels', [4.0] * self.num_pools))
        targets = np.array(state.get('targets', [4.0] * self.num_pools))
        current_gates = np.array(state.get('gates', [0.8] * self.num_gates))

        # PID计算
        errors = targets - levels

        # 更新积分项 (带抗饱和)
        self.integral = np.clip(
            self.integral + errors * self.dt,
            -10.0, 10.0
        )

        # 微分项
        derivative = (errors - self.prev_error) / self.dt
        self.prev_error = errors.copy()

        # PID输出
        pid_output = self.Kp * errors + self.Ki * self.integral + self.Kd * derivative

        # 转换为闸门调整
        gate_changes = np.zeros(self.num_gates)
        for i in range(min(self.num_pools, self.num_gates - 1)):
            # 映射: 正误差(水位低) -> 减少出流 -> 减小下游闸门开度
            gate_changes[i + 1] = -pid_output[i] * 0.1

        # 限制变化率
        gate_changes = np.clip(gate_changes, -self.max_change_rate, self.max_change_rate)

        # 计算新开度
        new_openings = np.clip(
            current_gates + gate_changes,
            self.min_opening,
            self.max_opening
        )

        # 安全检查
        safety_override = self._check_safety(levels, new_openings)

        return {
            'action': new_openings if not safety_override['triggered'] else safety_override['safe_action'],
            'source': 'pid',
            'confidence': 0.8,
            'pid_info': {
                'errors': errors,
                'integral': self.integral,
                'derivative': derivative,
                'output': pid_output
            },
            'safety_override': safety_override,
            'requires_confirmation': False
        }

    def _check_safety(self, levels: np.ndarray, openings: np.ndarray) -> Dict:
        """安全检查"""
        triggered = False
        safe_action = openings.copy()
        reasons = []

        for i, level in enumerate(levels):
            if level < self.level_min + 0.3:
                # 水位过低，关闭下游闸门
                if i + 1 < len(safe_action):
                    safe_action[i + 1] = self.min_opening
                triggered = True
                reasons.append(f"渠池{i}水位过低({level:.2f}m)")

            elif level > self.level_max - 0.3:
                # 水位过高，打开下游闸门
                if i + 1 < len(safe_action):
                    safe_action[i + 1] = self.max_opening
                triggered = True
                reasons.append(f"渠池{i}水位过高({level:.2f}m)")

        return {
            'triggered': triggered,
            'safe_action': safe_action,
            'reasons': reasons
        }

    def reset(self):
        """重置PID状态"""
        self.integral = np.zeros(self.num_pools)
        self.prev_error = np.zeros(self.num_pools)


# ==============================================================================
# L3: 条件自动控制模块
# ==============================================================================

class L3ConditionalAutoController(BaseController):
    """
    L3级: 条件自动化
    神经网络控制，场景识别
    """

    def __init__(self, num_gates: int = 64, num_pools: int = 63):
        self.num_gates = num_gates
        self.num_pools = num_pools

        # 神经网络模型 (简化版)
        self.model = self._build_model()

        # 场景识别器
        self.known_scenarios = {
            'normal': self._handle_normal,
            'high_demand': self._handle_high_demand,
            'low_inflow': self._handle_low_inflow,
            'emergency': self._handle_emergency
        }

        # 当前场景
        self.current_scenario = 'normal'

        # L2回退控制器
        self.fallback_controller = L2PartialAutoController(num_gates, num_pools)

    def get_level(self) -> AutonomyLevel:
        return AutonomyLevel.L3_CONDITIONAL

    def _build_model(self) -> nn.Module:
        """构建简化神经网络"""
        model = nn.Sequential(
            nn.Linear(self.num_pools * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_gates)
        )
        return model

    def compute_action(self, state: Dict) -> Dict:
        """神经网络控制"""
        # 识别场景
        scenario = self._identify_scenario(state)
        self.current_scenario = scenario

        # 检查是否为已知场景
        if scenario in self.known_scenarios:
            handler = self.known_scenarios[scenario]
            action, confidence = handler(state)
        else:
            # 未知场景，回退到L2
            logger.warning(f"Unknown scenario detected: {scenario}, falling back to L2")
            result = self.fallback_controller.compute_action(state)
            return {
                'action': result['action'],
                'source': 'l3_fallback_to_l2',
                'confidence': result['confidence'] * 0.8,
                'scenario': 'unknown',
                'requires_confirmation': True
            }

        return {
            'action': action,
            'source': 'neural',
            'confidence': confidence,
            'scenario': scenario,
            'requires_confirmation': confidence < 0.7
        }

    def _identify_scenario(self, state: Dict) -> str:
        """场景识别"""
        levels = np.array(state.get('levels', [4.0] * self.num_pools))
        inflows = np.array(state.get('inflows', [300.0] * self.num_pools))

        # 简单规则识别
        avg_level = np.mean(levels)
        avg_inflow = np.mean(inflows)

        if avg_level < 2.5 or np.min(levels) < 2.0:
            return 'emergency'
        elif avg_inflow < 150:
            return 'low_inflow'
        elif avg_level > 5.0:
            return 'high_demand'
        else:
            return 'normal'

    def _handle_normal(self, state: Dict) -> Tuple[np.ndarray, float]:
        """处理正常场景"""
        # 使用神经网络
        levels = np.array(state.get('levels', [4.0] * self.num_pools))
        inflows = np.array(state.get('inflows', [300.0] * self.num_pools))
        targets = np.array(state.get('targets', [4.0] * self.num_pools))

        # 构建输入
        x = np.concatenate([levels, inflows, targets])
        x = torch.FloatTensor(x).unsqueeze(0)

        with torch.no_grad():
            output = self.model(x)
            action = torch.sigmoid(output).squeeze(0).numpy()

        return action, 0.85

    def _handle_high_demand(self, state: Dict) -> Tuple[np.ndarray, float]:
        """处理高需水场景"""
        current_gates = np.array(state.get('gates', [0.8] * self.num_gates))
        # 适度增加开度
        action = np.clip(current_gates + 0.05, 0.0, 1.0)
        return action, 0.75

    def _handle_low_inflow(self, state: Dict) -> Tuple[np.ndarray, float]:
        """处理低来水场景"""
        current_gates = np.array(state.get('gates', [0.8] * self.num_gates))
        # 适度减少开度保水
        action = np.clip(current_gates - 0.05, 0.2, 1.0)
        return action, 0.75

    def _handle_emergency(self, state: Dict) -> Tuple[np.ndarray, float]:
        """处理紧急场景"""
        # 紧急模式：最小化出流
        action = np.ones(self.num_gates) * 0.3
        return action, 0.9


# ==============================================================================
# L4: 高度自动控制模块
# ==============================================================================

class L4HighAutoController(BaseController):
    """
    L4级: 高度自动化
    端到端自主控制
    """

    def __init__(self, num_gates: int = 64, num_pools: int = 63):
        self.num_gates = num_gates
        self.num_pools = num_pools

        # 导入L4核心模块
        from .e2e_controller import E2EAutonomousController, AutonomousConfig

        config = AutonomousConfig(
            num_pools=num_pools,
            num_gates=num_gates
        )
        self.e2e_controller = E2EAutonomousController(config)

        # 历史缓存
        self.history_buffer = []
        self.max_history = 96

        # 回退控制器
        self.fallback_controller = L3ConditionalAutoController(num_gates, num_pools)

    def get_level(self) -> AutonomyLevel:
        return AutonomyLevel.L4_HIGH

    def compute_action(self, state: Dict) -> Dict:
        """端到端自主控制"""
        # 更新历史
        self.history_buffer.append(state)
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

        # 调用E2E控制器
        targets = np.array(state.get('targets', [4.0] * self.num_pools))

        result = self.e2e_controller.step(
            current_obs=state,
            target_levels=targets,
            history_buffer=self.history_buffer
        )

        # 检查是否需要降级
        if not result['is_autonomous']:
            logger.info("L4 confidence low, consulting L3")
            l3_result = self.fallback_controller.compute_action(state)
            return {
                'action': l3_result['action'],
                'source': 'l4_consult_l3',
                'confidence': result['confidence'],
                'predictions': result['predictions'],  # 保留L4预测用于诊断
                'is_autonomous': False,
                'l4_action': result['gate_openings'],
                'l3_action': l3_result['action'],
                'requires_confirmation': True
            }

        return {
            'action': result['gate_openings'],
            'source': 'e2e',
            'confidence': result['confidence'],
            'predictions': result['predictions'],
            'is_autonomous': True,
            'requires_confirmation': False
        }


# ==============================================================================
# 多等级统一控制器
# ==============================================================================

class MultiLevelController:
    """
    多等级统一控制器
    管理L0-L4各级控制器，支持动态切换
    """

    def __init__(self, num_gates: int = 64, num_pools: int = 63):
        self.num_gates = num_gates
        self.num_pools = num_pools

        # 初始化各级控制器
        self.controllers = {
            AutonomyLevel.L0_MANUAL: L0ManualController(num_gates),
            AutonomyLevel.L1_ASSISTED: L1AdvisoryController(num_gates, num_pools),
            AutonomyLevel.L2_PARTIAL: L2PartialAutoController(num_gates, num_pools),
            AutonomyLevel.L3_CONDITIONAL: L3ConditionalAutoController(num_gates, num_pools),
            AutonomyLevel.L4_HIGH: L4HighAutoController(num_gates, num_pools),
        }

        # 当前等级
        self.current_level = AutonomyLevel.L2_PARTIAL
        self.target_level = AutonomyLevel.L2_PARTIAL

        # 等级切换状态
        self.transition_in_progress = False
        self.transition_progress = 0.0

        # 运行统计
        self.stats = {
            'level_history': [],
            'actions_per_level': {level: 0 for level in AutonomyLevel},
            'downgrades': 0,
            'upgrades': 0
        }

        logger.info("Multi-Level Controller initialized")
        logger.info(f"  Available levels: {[l.name for l in AutonomyLevel]}")
        logger.info(f"  Current level: {self.current_level.name}")

    def set_level(self, level: AutonomyLevel, immediate: bool = False):
        """
        设置自动化等级

        Args:
            level: 目标等级
            immediate: 是否立即切换
        """
        if level == self.current_level:
            return

        self.target_level = level

        if immediate:
            self._execute_level_change()
        else:
            self.transition_in_progress = True
            self.transition_progress = 0.0

        # 记录
        if level.value > self.current_level.value:
            self.stats['upgrades'] += 1
        else:
            self.stats['downgrades'] += 1

        logger.info(f"Level change: {self.current_level.name} -> {level.name}")

    def _execute_level_change(self):
        """执行等级切换"""
        self.current_level = self.target_level
        self.transition_in_progress = False
        self.transition_progress = 1.0

        # 记录历史
        self.stats['level_history'].append({
            'timestamp': time.time(),
            'level': self.current_level.name
        })

    def compute_action(self, state: Dict) -> Dict:
        """计算控制动作"""
        # 更新切换进度
        if self.transition_in_progress:
            self.transition_progress += 0.1
            if self.transition_progress >= 1.0:
                self._execute_level_change()

        # 获取当前控制器
        controller = self.controllers[self.current_level]

        # 计算动作
        result = controller.compute_action(state)

        # 更新统计
        self.stats['actions_per_level'][self.current_level] += 1

        # 添加等级信息
        result['level'] = self.current_level.name
        result['level_value'] = self.current_level.value

        return result

    def get_status(self) -> Dict:
        """获取控制器状态"""
        return {
            'current_level': self.current_level.name,
            'target_level': self.target_level.name,
            'transition_in_progress': self.transition_in_progress,
            'transition_progress': self.transition_progress,
            'capabilities': LEVEL_CAPABILITIES[self.current_level].__dict__,
            'stats': self.stats.copy()
        }

    def get_level_capabilities(self, level: AutonomyLevel = None) -> LevelCapabilities:
        """获取指定等级的能力描述"""
        level = level or self.current_level
        return LEVEL_CAPABILITIES[level]

    def emergency_downgrade(self, reason: str = ""):
        """紧急降级"""
        logger.warning(f"Emergency downgrade triggered: {reason}")
        self.set_level(AutonomyLevel.L0_MANUAL, immediate=True)
        return {
            'status': 'downgraded',
            'from_level': self.current_level.name,
            'to_level': 'L0_MANUAL',
            'reason': reason
        }

    def request_upgrade(self, target_level: AutonomyLevel) -> Dict:
        """请求升级"""
        # 检查是否可以升级
        can_upgrade, reasons = self._check_upgrade_conditions(target_level)

        if can_upgrade:
            self.set_level(target_level)
            return {
                'status': 'upgrade_initiated',
                'target_level': target_level.name,
                'reasons': reasons
            }
        else:
            return {
                'status': 'upgrade_denied',
                'target_level': target_level.name,
                'reasons': reasons
            }

    def _check_upgrade_conditions(self, target_level: AutonomyLevel) -> Tuple[bool, List[str]]:
        """检查升级条件"""
        reasons = []

        # 只能逐级升级
        if target_level.value > self.current_level.value + 1:
            reasons.append("只能逐级升级")
            return False, reasons

        # L3/L4需要模型就绪
        if target_level.value >= 3:
            # 检查模型状态
            reasons.append("神经网络模型已就绪")

        return True, reasons


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Level Controller Test")
    print("=" * 70)

    # 创建控制器
    controller = MultiLevelController(num_gates=11, num_pools=10)

    # 模拟状态
    state = {
        'levels': np.random.uniform(3.5, 4.5, 10),
        'inflows': np.random.uniform(250, 350, 10),
        'outflows': np.random.uniform(240, 340, 10),
        'gates': np.random.uniform(0.7, 0.9, 11),
        'targets': np.ones(10) * 4.0
    }

    print("\n测试各等级控制:")
    for level in AutonomyLevel:
        print(f"\n--- {level.name} ---")
        controller.set_level(level, immediate=True)
        result = controller.compute_action(state)
        print(f"  Source: {result['source']}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        if result['action'] is not None:
            action = result['action']
            if isinstance(action, np.ndarray):
                print(f"  Action shape: {action.shape}")
            else:
                print(f"  Action: {action}")

    # 打印能力描述
    print("\n\n等级能力对比:")
    print("-" * 70)
    for level, cap in LEVEL_CAPABILITIES.items():
        print(f"\n{level.name}:")
        print(f"  描述: {cap.description}")
        print(f"  功能: {', '.join(cap.features)}")
        print(f"  人工角色: {cap.human_role}")
        print(f"  系统角色: {cap.system_role}")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
