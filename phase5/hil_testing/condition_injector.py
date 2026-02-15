"""
工况注入器 (Condition Injector)
向仿真系统注入各种扰动和故障
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InjectionType(Enum):
    """注入类型"""
    STEP = "step"              # 阶跃变化
    RAMP = "ramp"              # 斜坡变化
    PULSE = "pulse"            # 脉冲
    NOISE = "noise"            # 噪声
    SINUSOID = "sinusoid"      # 正弦波
    DRIFT = "drift"            # 漂移
    STUCK = "stuck"            # 卡死
    BIAS = "bias"              # 偏置
    INTERMITTENT = "intermittent"  # 间歇性故障
    FDIA = "fdia"              # 虚假数据注入攻击


class InjectionTarget(Enum):
    """注入目标"""
    INFLOW = "inflow"              # 入流
    OUTFLOW = "outflow"            # 出流
    WATER_LEVEL = "water_level"    # 水位
    SENSOR_LEVEL = "sensor_level"  # 水位传感器
    SENSOR_FLOW = "sensor_flow"    # 流量传感器
    ACTUATOR_GATE = "actuator_gate"  # 闸门执行器
    CONTROLLER = "controller"      # 控制器
    COMMUNICATION = "communication"  # 通信


@dataclass
class InjectionEvent:
    """注入事件"""
    time: float                # 触发时间
    target: InjectionTarget    # 目标
    injection_type: InjectionType  # 类型
    value: float               # 注入值
    original_value: float = 0.0  # 原始值
    is_active: bool = True     # 是否激活


class _SubInjector:
    """子注入器 - 用于分类管理注入"""

    def __init__(self, parent, category: str):
        self.parent = parent
        self.category = category
        self.injections: List[Dict] = []

    def add_injection(self, target: str, injection_type: InjectionType,
                     start_time: float, magnitude: float = 0,
                     end_time: float = None, parameters: Dict = None):
        """添加注入配置"""
        self.injections.append({
            'target': target,
            'type': injection_type,
            'start_time': start_time,
            'end_time': end_time or float('inf'),
            'magnitude': magnitude,
            'parameters': parameters or {}
        })

    def get_value(self, target: str, time: float, base_value: float) -> float:
        """获取目标在指定时间的注入值"""
        result = base_value

        for inj in self.injections:
            if inj['target'] == target:
                if inj['start_time'] <= time < inj['end_time']:
                    inj_type = inj['type']
                    magnitude = inj['magnitude']

                    if inj_type == InjectionType.STEP:
                        result = base_value + magnitude
                    elif inj_type == InjectionType.PULSE:
                        result = base_value + magnitude
                    elif inj_type == InjectionType.RAMP:
                        elapsed = time - inj['start_time']
                        duration = inj['end_time'] - inj['start_time']
                        if duration > 0:
                            result = base_value + magnitude * (elapsed / duration)
                    elif inj_type == InjectionType.BIAS:
                        result = base_value + magnitude
                    elif inj_type == InjectionType.NOISE:
                        import numpy as np
                        result = base_value + np.random.normal(0, magnitude)

        return result


class ConditionInjector:
    """
    工况注入器

    功能：
    1. 注入各种扰动信号
    2. 模拟设备故障
    3. 模拟网络攻击
    4. 管理注入事件队列
    """

    def __init__(self):
        """初始化注入器"""
        self.active_injections: Dict[str, Dict] = {}
        self.injection_history: List[InjectionEvent] = []
        self.current_time: float = 0.0
        self._injection_counter = 0

        # 子注入器（用于兼容性）
        self.flow_injector = _SubInjector(self, 'flow')
        self.sensor_injector = _SubInjector(self, 'sensor')
        self.actuator_injector = _SubInjector(self, 'actuator')
        self.security_injector = _SubInjector(self, 'security')

        # 注入函数映射
        self._injection_functions = {
            InjectionType.STEP: self._step_injection,
            InjectionType.RAMP: self._ramp_injection,
            InjectionType.PULSE: self._pulse_injection,
            InjectionType.NOISE: self._noise_injection,
            InjectionType.SINUSOID: self._sinusoid_injection,
            InjectionType.DRIFT: self._drift_injection,
            InjectionType.STUCK: self._stuck_injection,
            InjectionType.BIAS: self._bias_injection,
            InjectionType.INTERMITTENT: self._intermittent_injection,
            InjectionType.FDIA: self._fdia_injection,
        }

    def reset(self):
        """重置注入器状态"""
        self.active_injections.clear()
        self.injection_history.clear()
        self.current_time = 0.0
        self._injection_counter = 0
        self.flow_injector.injections.clear()
        self.sensor_injector.injections.clear()
        self.actuator_injector.injections.clear()
        self.security_injector.injections.clear()

    def clear_all(self):
        """清除所有注入"""
        self.reset()

    def add_flow_injection(self, target: str, injection_type: str, start_time: float,
                          magnitude: float = 0, end_time: float = None, parameters: Dict = None):
        """添加流量注入"""
        self._injection_counter += 1
        inj_id = f"flow_{self._injection_counter}"
        duration = (end_time - start_time) if end_time else 0

        self.flow_injector.add_injection(
            target=target,
            injection_type=InjectionType(injection_type.lower()) if isinstance(injection_type, str) else injection_type,
            start_time=start_time,
            end_time=end_time,
            magnitude=magnitude,
            parameters=parameters or {}
        )

        self.add_injection(
            inj_id, injection_type, 'inflow', start_time, magnitude, duration, **(parameters or {})
        )

    def add_sensor_injection(self, target: str, injection_type: str, start_time: float,
                            magnitude: float = 0, end_time: float = None, parameters: Dict = None):
        """添加传感器注入"""
        self._injection_counter += 1
        inj_id = f"sensor_{self._injection_counter}"
        duration = (end_time - start_time) if end_time else 0

        self.sensor_injector.add_injection(
            target=target,
            injection_type=InjectionType(injection_type.lower()) if isinstance(injection_type, str) else injection_type,
            start_time=start_time,
            end_time=end_time,
            magnitude=magnitude,
            parameters=parameters or {}
        )

        self.add_injection(
            inj_id, injection_type, 'sensor_level', start_time, magnitude, duration, **(parameters or {})
        )

    def add_actuator_injection(self, target: str, injection_type: str, start_time: float,
                              magnitude: float = 0, end_time: float = None, parameters: Dict = None):
        """添加执行器注入"""
        self._injection_counter += 1
        inj_id = f"actuator_{self._injection_counter}"
        duration = (end_time - start_time) if end_time else 0

        self.actuator_injector.add_injection(
            target=target,
            injection_type=InjectionType(injection_type.lower()) if isinstance(injection_type, str) else injection_type,
            start_time=start_time,
            end_time=end_time,
            magnitude=magnitude,
            parameters=parameters or {}
        )

        self.add_injection(
            inj_id, injection_type, 'actuator_gate', start_time, magnitude, duration, **(parameters or {})
        )

    def add_security_injection(self, target: str, injection_type: str, start_time: float,
                              magnitude: float = 0, end_time: float = None, parameters: Dict = None):
        """添加安全攻击注入"""
        self._injection_counter += 1
        inj_id = f"security_{self._injection_counter}"
        duration = (end_time - start_time) if end_time else 0

        self.security_injector.add_injection(
            target=target,
            injection_type=InjectionType(injection_type.lower()) if isinstance(injection_type, str) else injection_type,
            start_time=start_time,
            end_time=end_time,
            magnitude=magnitude,
            parameters=parameters or {}
        )

        self.add_injection(
            inj_id, injection_type, 'sensor_level', start_time, magnitude, duration, **(parameters or {})
        )

    def get_disturbances(self, time: float) -> Dict:
        """获取当前时刻的扰动"""
        self.update(time)
        disturbances = {
            'flow': {},
            'sensor': {},
            'actuator': {},
            'security': {}
        }

        for inj_id, config in self.active_injections.items():
            if config['is_active']:
                category = inj_id.split('_')[0]
                target = config.get('target', InjectionTarget.INFLOW)
                target_name = target.value if hasattr(target, 'value') else str(target)
                disturbances[category][target_name] = config['magnitude']

        return disturbances

    def add_injection(self,
                     injection_id: str,
                     injection_type: str,
                     target: str,
                     start_time: float,
                     magnitude: float,
                     duration: float = 0.0,
                     **kwargs):
        """
        添加注入配置

        Args:
            injection_id: 注入ID
            injection_type: 注入类型
            target: 目标
            start_time: 开始时间
            magnitude: 幅度
            duration: 持续时间
            **kwargs: 其他参数
        """
        self.active_injections[injection_id] = {
            'type': InjectionType(injection_type) if isinstance(injection_type, str) else injection_type,
            'target': InjectionTarget(target) if isinstance(target, str) else target,
            'start_time': start_time,
            'magnitude': magnitude,
            'duration': duration,
            'end_time': start_time + duration if duration > 0 else float('inf'),
            'parameters': kwargs,
            'is_active': False,
            'original_value': None
        }

    def update(self, time: float) -> List[str]:
        """
        更新注入器状态

        Args:
            time: 当前时间

        Returns:
            激活的注入ID列表
        """
        self.current_time = time
        activated = []

        for inj_id, config in self.active_injections.items():
            # 检查是否应该激活
            if config['start_time'] <= time < config['end_time']:
                if not config['is_active']:
                    config['is_active'] = True
                    activated.append(inj_id)
            else:
                config['is_active'] = False

        return activated

    def apply_injection(self, value: float, target: str, time: float = None) -> float:
        """
        应用注入到值

        Args:
            value: 原始值
            target: 目标变量
            time: 当前时间

        Returns:
            注入后的值
        """
        if time is not None:
            self.current_time = time

        result = value
        target_enum = InjectionTarget(target) if isinstance(target, str) else target

        for inj_id, config in self.active_injections.items():
            if config['target'] == target_enum and config['is_active']:
                # 保存原始值
                if config['original_value'] is None:
                    config['original_value'] = value

                # 应用注入
                injection_func = self._injection_functions.get(config['type'])
                if injection_func:
                    result = injection_func(result, config)

                # 记录事件
                self.injection_history.append(InjectionEvent(
                    time=self.current_time,
                    target=target_enum,
                    injection_type=config['type'],
                    value=result,
                    original_value=value
                ))

        return result

    def get_active_injections(self) -> List[str]:
        """获取当前激活的注入"""
        return [inj_id for inj_id, config in self.active_injections.items()
                if config['is_active']]

    def is_injection_active(self, injection_id: str) -> bool:
        """检查指定注入是否激活"""
        return self.active_injections.get(injection_id, {}).get('is_active', False)

    # ==================== 注入函数 ====================

    def _step_injection(self, value: float, config: Dict) -> float:
        """阶跃注入"""
        return config['magnitude']

    def _ramp_injection(self, value: float, config: Dict) -> float:
        """斜坡注入"""
        elapsed = self.current_time - config['start_time']
        rate = config['parameters'].get('rate', config['magnitude'] / max(config['duration'], 1))
        return value + rate * elapsed

    def _pulse_injection(self, value: float, config: Dict) -> float:
        """脉冲注入"""
        pulse_width = config['parameters'].get('pulse_width', 10.0)
        elapsed = self.current_time - config['start_time']
        if elapsed < pulse_width:
            return config['magnitude']
        return value

    def _noise_injection(self, value: float, config: Dict) -> float:
        """噪声注入"""
        std = config['parameters'].get('std', config['magnitude'])
        return value + np.random.normal(0, std)

    def _sinusoid_injection(self, value: float, config: Dict) -> float:
        """正弦波注入"""
        frequency = config['parameters'].get('frequency', 0.01)
        elapsed = self.current_time - config['start_time']
        return value + config['magnitude'] * np.sin(2 * np.pi * frequency * elapsed)

    def _drift_injection(self, value: float, config: Dict) -> float:
        """漂移注入"""
        drift_rate = config['parameters'].get('drift_rate', 0.001)
        elapsed = self.current_time - config['start_time']
        return value + drift_rate * elapsed

    def _stuck_injection(self, value: float, config: Dict) -> float:
        """卡死注入"""
        stuck_value = config['parameters'].get('stuck_value', config['original_value'])
        return stuck_value if stuck_value is not None else value

    def _bias_injection(self, value: float, config: Dict) -> float:
        """偏置注入"""
        return value + config['magnitude']

    def _intermittent_injection(self, value: float, config: Dict) -> float:
        """间歇性故障注入"""
        failure_prob = config['parameters'].get('failure_prob', 0.3)
        if np.random.random() < failure_prob:
            return config['magnitude']
        return value

    def _fdia_injection(self, value: float, config: Dict) -> float:
        """虚假数据注入攻击"""
        attack_mode = config['parameters'].get('mode', 'bias')
        if attack_mode == 'bias':
            return value + config['magnitude']
        elif attack_mode == 'scale':
            return value * config['magnitude']
        elif attack_mode == 'replay':
            # 重放攻击，返回固定值
            return config['parameters'].get('replay_value', value)
        elif attack_mode == 'random':
            return value + np.random.uniform(-config['magnitude'], config['magnitude'])
        return value

    def get_injection_summary(self) -> Dict:
        """获取注入摘要"""
        return {
            'total_injections': len(self.active_injections),
            'active_count': len(self.get_active_injections()),
            'history_count': len(self.injection_history),
            'injections': {
                inj_id: {
                    'type': config['type'].value,
                    'target': config['target'].value,
                    'is_active': config['is_active']
                }
                for inj_id, config in self.active_injections.items()
            }
        }


class FlowInjector(ConditionInjector):
    """流量注入器 - 专门用于入流/出流扰动"""

    def inject_flood(self, start_time: float, peak_flow: float, duration: float):
        """注入洪水事件"""
        self.add_injection(
            f"flood_{start_time}",
            InjectionType.RAMP,
            InjectionTarget.INFLOW,
            start_time,
            peak_flow,
            duration / 2,
            rate=peak_flow / (duration / 2)
        )

    def inject_drought(self, start_time: float, min_flow: float, duration: float):
        """注入干旱事件"""
        self.add_injection(
            f"drought_{start_time}",
            InjectionType.RAMP,
            InjectionTarget.INFLOW,
            start_time,
            min_flow,
            duration,
            rate=-min_flow / duration
        )


class SensorInjector(ConditionInjector):
    """传感器注入器 - 专门用于传感器故障"""

    def inject_sensor_drift(self, sensor_id: str, start_time: float, drift_rate: float, duration: float):
        """注入传感器漂移"""
        self.add_injection(
            f"drift_{sensor_id}_{start_time}",
            InjectionType.DRIFT,
            InjectionTarget.SENSOR_LEVEL,
            start_time,
            0,
            duration,
            drift_rate=drift_rate
        )

    def inject_sensor_failure(self, sensor_id: str, start_time: float, stuck_value: float, duration: float):
        """注入传感器失效"""
        self.add_injection(
            f"failure_{sensor_id}_{start_time}",
            InjectionType.STUCK,
            InjectionTarget.SENSOR_LEVEL,
            start_time,
            stuck_value,
            duration,
            stuck_value=stuck_value
        )


class ActuatorInjector(ConditionInjector):
    """执行器注入器 - 专门用于闸门故障"""

    def inject_gate_stuck(self, gate_id: str, start_time: float, stuck_position: float, duration: float):
        """注入闸门卡死"""
        self.add_injection(
            f"stuck_{gate_id}_{start_time}",
            InjectionType.STUCK,
            InjectionTarget.ACTUATOR_GATE,
            start_time,
            stuck_position,
            duration,
            stuck_value=stuck_position
        )

    def inject_gate_delay(self, gate_id: str, delay_time: float):
        """注入闸门响应延迟"""
        self.add_injection(
            f"delay_{gate_id}",
            InjectionType.BIAS,
            InjectionTarget.ACTUATOR_GATE,
            0,
            0,
            float('inf'),
            delay=delay_time
        )


class SecurityInjector(ConditionInjector):
    """安全注入器 - 专门用于网络攻击模拟"""

    def inject_fdia_attack(self, target: str, start_time: float, bias: float, duration: float):
        """注入FDIA攻击"""
        self.add_injection(
            f"fdia_{target}_{start_time}",
            InjectionType.FDIA,
            InjectionTarget(target),
            start_time,
            bias,
            duration,
            mode='bias'
        )

    def inject_dos_attack(self, start_time: float, duration: float):
        """注入DoS攻击 (通信中断)"""
        self.add_injection(
            f"dos_{start_time}",
            InjectionType.STUCK,
            InjectionTarget.COMMUNICATION,
            start_time,
            0,
            duration
        )


# 示例使用
if __name__ == "__main__":
    # 测试基本注入器
    injector = ConditionInjector()

    # 添加阶跃注入
    injector.add_injection("step_1", "step", "inflow", 100, 150)

    # 添加噪声注入
    injector.add_injection("noise_1", "noise", "sensor_level", 0, 0.1, 1000, std=0.1)

    # 模拟时间推进
    for t in range(200):
        injector.update(t)

        if t == 50:
            value = injector.apply_injection(50.0, "inflow", t)
            logger.info(f"T={t}: inflow = {value} (before injection)")

        if t == 120:
            value = injector.apply_injection(50.0, "inflow", t)
            logger.info(f"T={t}: inflow = {value} (after step injection)")

    logger.info("\n注入摘要:")
    logger.info(injector.get_injection_summary())
