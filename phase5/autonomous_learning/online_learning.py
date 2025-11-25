# Phase 5.10: Online Learning Engine
# 在线学习引擎 - 实时模型参数自适应优化

import logging
import threading
import time
import math
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class LearningAlgorithm(Enum):
    """学习算法类型"""
    SGD = "sgd"  # Stochastic Gradient Descent
    ADAM = "adam"  # Adaptive Moment Estimation
    RMSPROP = "rmsprop"  # Root Mean Square Propagation
    ADAGRAD = "adagrad"  # Adaptive Gradient
    RECURSIVE_LEAST_SQUARES = "rls"  # 递推最小二乘
    KALMAN_FILTER = "kalman"  # 卡尔曼滤波
    REINFORCEMENT = "rl"  # 强化学习


class LearningRate(Enum):
    """学习率策略"""
    CONSTANT = "constant"
    DECAY = "decay"
    ADAPTIVE = "adaptive"
    CYCLIC = "cyclic"
    WARMUP = "warmup"


class UpdateTrigger(Enum):
    """更新触发条件"""
    TIME_BASED = "time"  # 固定时间间隔
    ERROR_THRESHOLD = "error"  # 误差超过阈值
    SAMPLE_COUNT = "samples"  # 样本数量达到
    PERFORMANCE_DROP = "performance"  # 性能下降


@dataclass
class ModelUpdate:
    """模型更新记录"""
    update_id: str
    timestamp: datetime
    algorithm: LearningAlgorithm
    parameter_name: str
    old_value: float
    new_value: float
    gradient: float
    learning_rate: float
    loss_before: float
    loss_after: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        """计算改进比例"""
        if self.loss_before == 0:
            return 0.0
        return (self.loss_before - self.loss_after) / self.loss_before


@dataclass
class LearningConfig:
    """学习配置"""
    algorithm: LearningAlgorithm = LearningAlgorithm.ADAM
    learning_rate: float = 0.001
    learning_rate_strategy: LearningRate = LearningRate.ADAPTIVE
    momentum: float = 0.9
    beta1: float = 0.9  # Adam参数
    beta2: float = 0.999  # Adam参数
    epsilon: float = 1e-8
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 0.1
    warmup_steps: int = 100
    decay_rate: float = 0.99
    update_interval: float = 60.0  # seconds
    min_samples: int = 10
    batch_size: int = 32
    max_iterations: int = 100


@dataclass
class ParameterState:
    """参数状态"""
    name: str
    value: float
    gradient: float = 0.0
    momentum: float = 0.0
    velocity: float = 0.0  # Adam v
    squared_grad: float = 0.0  # Adam s / RMSprop
    step_count: int = 0
    last_update: Optional[datetime] = None
    bounds: Tuple[float, float] = (-float('inf'), float('inf'))


class AdaptiveOptimizer:
    """自适应优化器"""

    def __init__(self, config: LearningConfig):
        self.config = config
        self.parameters: Dict[str, ParameterState] = {}
        self.global_step = 0
        self.current_lr = config.learning_rate

    def register_parameter(
        self,
        name: str,
        initial_value: float,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        """注册可学习参数"""
        self.parameters[name] = ParameterState(
            name=name,
            value=initial_value,
            bounds=bounds or (-float('inf'), float('inf')),
        )

    def compute_gradient(
        self,
        param_name: str,
        loss_fn: Callable[[float], float],
        epsilon: float = 1e-5,
    ) -> float:
        """计算数值梯度"""
        param = self.parameters.get(param_name)
        if not param:
            return 0.0

        # 中心差分法
        loss_plus = loss_fn(param.value + epsilon)
        loss_minus = loss_fn(param.value - epsilon)
        gradient = (loss_plus - loss_minus) / (2 * epsilon)

        return gradient

    def update_learning_rate(self):
        """更新学习率"""
        strategy = self.config.learning_rate_strategy
        base_lr = self.config.learning_rate

        if strategy == LearningRate.CONSTANT:
            self.current_lr = base_lr

        elif strategy == LearningRate.DECAY:
            self.current_lr = base_lr * (self.config.decay_rate ** self.global_step)
            self.current_lr = max(self.current_lr, self.config.min_learning_rate)

        elif strategy == LearningRate.WARMUP:
            if self.global_step < self.config.warmup_steps:
                self.current_lr = base_lr * (self.global_step / self.config.warmup_steps)
            else:
                decay_steps = self.global_step - self.config.warmup_steps
                self.current_lr = base_lr * (self.config.decay_rate ** decay_steps)
            self.current_lr = max(self.current_lr, self.config.min_learning_rate)

        elif strategy == LearningRate.CYCLIC:
            cycle = math.floor(1 + self.global_step / (2 * self.config.warmup_steps))
            x = abs(self.global_step / self.config.warmup_steps - 2 * cycle + 1)
            self.current_lr = self.config.min_learning_rate + (
                self.config.max_learning_rate - self.config.min_learning_rate
            ) * max(0, (1 - x))

        elif strategy == LearningRate.ADAPTIVE:
            # 基于损失变化自适应调整
            self.current_lr = base_lr

    def step(
        self,
        param_name: str,
        gradient: float,
    ) -> Tuple[float, float]:
        """执行一步优化"""
        param = self.parameters.get(param_name)
        if not param:
            return 0.0, 0.0

        old_value = param.value
        param.step_count += 1
        self.global_step += 1

        # 更新学习率
        self.update_learning_rate()
        lr = self.current_lr

        # 梯度裁剪
        if abs(gradient) > self.config.gradient_clip:
            gradient = self.config.gradient_clip * np.sign(gradient)

        # 根据算法更新参数
        if self.config.algorithm == LearningAlgorithm.SGD:
            # SGD with momentum
            param.momentum = self.config.momentum * param.momentum + gradient
            delta = -lr * param.momentum

        elif self.config.algorithm == LearningAlgorithm.ADAM:
            # Adam optimizer
            param.momentum = self.config.beta1 * param.momentum + (1 - self.config.beta1) * gradient
            param.squared_grad = self.config.beta2 * param.squared_grad + (1 - self.config.beta2) * (gradient ** 2)

            # Bias correction
            m_hat = param.momentum / (1 - self.config.beta1 ** param.step_count)
            v_hat = param.squared_grad / (1 - self.config.beta2 ** param.step_count)

            delta = -lr * m_hat / (math.sqrt(v_hat) + self.config.epsilon)

        elif self.config.algorithm == LearningAlgorithm.RMSPROP:
            param.squared_grad = self.config.beta2 * param.squared_grad + (1 - self.config.beta2) * (gradient ** 2)
            delta = -lr * gradient / (math.sqrt(param.squared_grad) + self.config.epsilon)

        elif self.config.algorithm == LearningAlgorithm.ADAGRAD:
            param.squared_grad += gradient ** 2
            delta = -lr * gradient / (math.sqrt(param.squared_grad) + self.config.epsilon)

        else:
            delta = -lr * gradient

        # 应用权重衰减
        if self.config.weight_decay > 0:
            delta -= self.config.weight_decay * lr * param.value

        # 更新参数值
        new_value = param.value + delta

        # 应用边界约束
        new_value = max(param.bounds[0], min(param.bounds[1], new_value))

        param.value = new_value
        param.gradient = gradient
        param.last_update = datetime.now()

        return old_value, new_value


class OnlineLearningEngine:
    """在线学习引擎"""

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self.optimizer = AdaptiveOptimizer(self.config)

        # 数据缓冲区
        self._sample_buffer: deque = deque(maxlen=10000)
        self._loss_history: deque = deque(maxlen=1000)
        self._update_history: List[ModelUpdate] = []

        # 模型参数
        self._model_parameters: Dict[str, float] = {}
        self._parameter_importance: Dict[str, float] = {}

        # 状态
        self._lock = threading.RLock()
        self._running = False
        self._learning_thread: Optional[threading.Thread] = None
        self._next_update_id = 1

        # 回调
        self._update_callbacks: List[Callable[[ModelUpdate], None]] = []
        self._loss_callbacks: List[Callable[[float], None]] = []

        # 统计
        self.stats = {
            'samples_processed': 0,
            'updates_performed': 0,
            'total_improvement': 0.0,
            'average_loss': 0.0,
            'learning_rate': self.config.learning_rate,
            'start_time': None,
            'last_update_time': None,
        }

        logger.info("Online Learning Engine initialized")

    def register_parameter(
        self,
        name: str,
        initial_value: float,
        bounds: Optional[Tuple[float, float]] = None,
        importance: float = 1.0,
    ):
        """注册模型参数"""
        with self._lock:
            self._model_parameters[name] = initial_value
            self._parameter_importance[name] = importance
            self.optimizer.register_parameter(name, initial_value, bounds)
            logger.info(f"Registered parameter: {name} = {initial_value}")

    def add_sample(
        self,
        inputs: Dict[str, float],
        target: float,
        prediction: Optional[float] = None,
        weight: float = 1.0,
    ):
        """添加训练样本"""
        sample = {
            'timestamp': datetime.now(),
            'inputs': inputs,
            'target': target,
            'prediction': prediction,
            'weight': weight,
            'error': (target - prediction) if prediction is not None else None,
        }

        with self._lock:
            self._sample_buffer.append(sample)
            self.stats['samples_processed'] += 1

    def compute_loss(
        self,
        predictions: List[float],
        targets: List[float],
        weights: Optional[List[float]] = None,
    ) -> float:
        """计算损失函数 (MSE)"""
        if not predictions or not targets:
            return 0.0

        if weights is None:
            weights = [1.0] * len(predictions)

        total_loss = 0.0
        total_weight = 0.0

        for pred, target, weight in zip(predictions, targets, weights):
            total_loss += weight * (pred - target) ** 2
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_loss / total_weight

    def _create_loss_function(
        self,
        param_name: str,
        samples: List[Dict],
        base_prediction_fn: Callable[[Dict[str, float], float], float],
    ) -> Callable[[float], float]:
        """创建参数的损失函数"""
        def loss_fn(param_value: float) -> float:
            total_loss = 0.0
            for sample in samples:
                pred = base_prediction_fn(sample['inputs'], param_value)
                target = sample['target']
                weight = sample.get('weight', 1.0)
                total_loss += weight * (pred - target) ** 2
            return total_loss / len(samples) if samples else 0.0
        return loss_fn

    def learn_step(
        self,
        param_name: str,
        gradient: Optional[float] = None,
        loss_fn: Optional[Callable[[float], float]] = None,
    ) -> Optional[ModelUpdate]:
        """执行一步学习"""
        with self._lock:
            if param_name not in self._model_parameters:
                logger.warning(f"Parameter not registered: {param_name}")
                return None

            # 计算当前损失
            if loss_fn:
                current_value = self._model_parameters[param_name]
                loss_before = loss_fn(current_value)

                # 计算梯度
                if gradient is None:
                    gradient = self.optimizer.compute_gradient(param_name, loss_fn)
            else:
                loss_before = self.stats['average_loss']
                if gradient is None:
                    gradient = 0.0

            # 执行优化步骤
            old_value, new_value = self.optimizer.step(param_name, gradient)

            # 计算更新后损失
            if loss_fn:
                loss_after = loss_fn(new_value)
            else:
                loss_after = loss_before

            # 更新模型参数
            self._model_parameters[param_name] = new_value

            # 计算置信度
            improvement = loss_before - loss_after if loss_before > 0 else 0
            confidence = min(1.0, max(0.0, improvement / (loss_before + 1e-8)))

            # 记录更新
            update = ModelUpdate(
                update_id=f"upd-{self._next_update_id:06d}",
                timestamp=datetime.now(),
                algorithm=self.config.algorithm,
                parameter_name=param_name,
                old_value=old_value,
                new_value=new_value,
                gradient=gradient,
                learning_rate=self.optimizer.current_lr,
                loss_before=loss_before,
                loss_after=loss_after,
                confidence=confidence,
            )

            self._next_update_id += 1
            self._update_history.append(update)
            self._loss_history.append(loss_after)

            # 更新统计
            self.stats['updates_performed'] += 1
            self.stats['total_improvement'] += improvement
            self.stats['average_loss'] = sum(self._loss_history) / len(self._loss_history)
            self.stats['learning_rate'] = self.optimizer.current_lr
            self.stats['last_update_time'] = datetime.now()

            # 通知回调
            for callback in self._update_callbacks:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Update callback error: {e}")

            logger.debug(f"Updated {param_name}: {old_value:.6f} -> {new_value:.6f}, loss: {loss_before:.6f} -> {loss_after:.6f}")

            return update

    def batch_learn(
        self,
        prediction_fn: Callable[[Dict[str, float], Dict[str, float]], float],
        max_iterations: Optional[int] = None,
    ) -> List[ModelUpdate]:
        """批量学习"""
        updates = []
        iterations = max_iterations or self.config.max_iterations

        with self._lock:
            samples = list(self._sample_buffer)[-self.config.batch_size:]

            if len(samples) < self.config.min_samples:
                logger.warning(f"Not enough samples for learning: {len(samples)}")
                return updates

            for iteration in range(iterations):
                for param_name in self._model_parameters:
                    # 创建当前参数的损失函数
                    def loss_fn(param_value: float) -> float:
                        total_loss = 0.0
                        params = self._model_parameters.copy()
                        params[param_name] = param_value

                        for sample in samples:
                            pred = prediction_fn(sample['inputs'], params)
                            target = sample['target']
                            weight = sample.get('weight', 1.0)
                            total_loss += weight * (pred - target) ** 2

                        return total_loss / len(samples)

                    update = self.learn_step(param_name, loss_fn=loss_fn)
                    if update:
                        updates.append(update)

                # 检查收敛
                if len(self._loss_history) >= 10:
                    recent_losses = list(self._loss_history)[-10:]
                    loss_std = np.std(recent_losses)
                    if loss_std < 1e-6:
                        logger.info(f"Converged after {iteration + 1} iterations")
                        break

        return updates

    def start(self, update_interval: Optional[float] = None):
        """启动在线学习"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self.stats['start_time'] = datetime.now()

            interval = update_interval or self.config.update_interval
            self._learning_thread = threading.Thread(
                target=self._learning_loop,
                args=(interval,),
                daemon=True,
            )
            self._learning_thread.start()

            logger.info("Online Learning Engine started")

    def stop(self):
        """停止在线学习"""
        with self._lock:
            self._running = False
            if self._learning_thread:
                self._learning_thread.join(timeout=5.0)
                self._learning_thread = None

            logger.info("Online Learning Engine stopped")

    def _learning_loop(self, interval: float):
        """学习主循环"""
        while self._running:
            try:
                # 检查是否有足够的新样本
                if len(self._sample_buffer) >= self.config.min_samples:
                    # 执行学习
                    for param_name in list(self._model_parameters.keys()):
                        # 简单的基于误差的梯度估计
                        recent_samples = list(self._sample_buffer)[-self.config.batch_size:]
                        errors = [s.get('error', 0) or 0 for s in recent_samples]
                        avg_error = np.mean(errors) if errors else 0

                        # 使用误差作为梯度信号
                        gradient = avg_error * self._parameter_importance.get(param_name, 1.0)
                        self.learn_step(param_name, gradient=gradient)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                time.sleep(1.0)

    def on_update(self, callback: Callable[[ModelUpdate], None]):
        """注册更新回调"""
        self._update_callbacks.append(callback)

    def on_loss(self, callback: Callable[[float], None]):
        """注册损失回调"""
        self._loss_callbacks.append(callback)

    def get_parameter(self, name: str) -> Optional[float]:
        """获取参数当前值"""
        return self._model_parameters.get(name)

    def get_all_parameters(self) -> Dict[str, float]:
        """获取所有参数"""
        return self._model_parameters.copy()

    def set_parameter(self, name: str, value: float):
        """手动设置参数值"""
        with self._lock:
            if name in self._model_parameters:
                self._model_parameters[name] = value
                self.optimizer.parameters[name].value = value

    def get_update_history(
        self,
        param_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[ModelUpdate]:
        """获取更新历史"""
        history = self._update_history[-limit:]
        if param_name:
            history = [u for u in history if u.parameter_name == param_name]
        return history

    def get_statistics(self) -> Dict[str, Any]:
        """获取学习统计"""
        return {
            **self.stats,
            'parameter_count': len(self._model_parameters),
            'sample_buffer_size': len(self._sample_buffer),
            'loss_history_size': len(self._loss_history),
            'update_history_size': len(self._update_history),
            'running': self._running,
        }

    def export_model(self) -> Dict[str, Any]:
        """导出模型状态"""
        return {
            'parameters': self._model_parameters.copy(),
            'parameter_importance': self._parameter_importance.copy(),
            'optimizer_state': {
                name: {
                    'value': state.value,
                    'momentum': state.momentum,
                    'squared_grad': state.squared_grad,
                    'step_count': state.step_count,
                }
                for name, state in self.optimizer.parameters.items()
            },
            'statistics': self.stats.copy(),
            'config': {
                'algorithm': self.config.algorithm.value,
                'learning_rate': self.config.learning_rate,
                'learning_rate_strategy': self.config.learning_rate_strategy.value,
            },
            'exported_at': datetime.now().isoformat(),
        }

    def import_model(self, model_state: Dict[str, Any]):
        """导入模型状态"""
        with self._lock:
            # 恢复参数
            for name, value in model_state.get('parameters', {}).items():
                self._model_parameters[name] = value

            for name, importance in model_state.get('parameter_importance', {}).items():
                self._parameter_importance[name] = importance

            # 恢复优化器状态
            optimizer_state = model_state.get('optimizer_state', {})
            for name, state in optimizer_state.items():
                if name in self.optimizer.parameters:
                    param = self.optimizer.parameters[name]
                    param.value = state.get('value', param.value)
                    param.momentum = state.get('momentum', 0.0)
                    param.squared_grad = state.get('squared_grad', 0.0)
                    param.step_count = state.get('step_count', 0)

            logger.info(f"Imported model with {len(self._model_parameters)} parameters")

    def create_water_network_parameters(self, num_pools: int) -> Dict[str, float]:
        """创建智能水网学习参数"""
        parameters = {}

        # 全局控制参数
        self.register_parameter('global_safety_margin', 0.1, bounds=(0.01, 0.5))
        self.register_parameter('global_efficiency_weight', 0.5, bounds=(0.0, 1.0))
        self.register_parameter('response_speed_factor', 1.0, bounds=(0.5, 2.0))

        for pool_id in range(num_pools):
            # 水位控制参数
            self.register_parameter(
                f'pool_{pool_id}_level_kp',
                1.0, bounds=(0.1, 10.0), importance=1.5
            )
            self.register_parameter(
                f'pool_{pool_id}_level_ki',
                0.1, bounds=(0.0, 1.0), importance=1.0
            )
            self.register_parameter(
                f'pool_{pool_id}_level_kd',
                0.05, bounds=(0.0, 0.5), importance=0.8
            )

            # 流量预测参数
            self.register_parameter(
                f'pool_{pool_id}_flow_alpha',
                0.3, bounds=(0.1, 0.9), importance=1.2
            )
            self.register_parameter(
                f'pool_{pool_id}_flow_beta',
                0.5, bounds=(0.1, 0.9), importance=1.0
            )

            # 闸门控制参数
            self.register_parameter(
                f'pool_{pool_id}_gate_deadband',
                2.0, bounds=(0.5, 10.0), importance=0.7
            )
            self.register_parameter(
                f'pool_{pool_id}_gate_rate_limit',
                5.0, bounds=(1.0, 20.0), importance=0.9
            )

        logger.info(f"Created {len(self._model_parameters)} water network parameters for {num_pools} pools")
        return self._model_parameters.copy()
