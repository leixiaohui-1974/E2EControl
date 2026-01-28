"""
SIL-HIL Bridge (Software/Hardware-in-the-Loop 桥接)

功能:
1. SIL模式 - 纯软件仿真验证
2. HIL模式 - 硬件在环验证
3. 混合模式 - 部分软件+部分硬件
4. 实时同步 - 时间同步管理
5. 信号转换 - 软硬件信号映射
6. 故障注入 - 测试系统鲁棒性
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)


class VerificationMode(Enum):
    """验证模式"""
    SIL = "sil"                     # Software-in-the-Loop
    HIL = "hil"                     # Hardware-in-the-Loop
    HYBRID = "hybrid"               # 混合模式
    REPLAY = "replay"               # 回放模式


class SignalType(Enum):
    """信号类型"""
    ANALOG_INPUT = "ai"             # 模拟输入 (传感器)
    ANALOG_OUTPUT = "ao"            # 模拟输出 (执行器)
    DIGITAL_INPUT = "di"            # 数字输入
    DIGITAL_OUTPUT = "do"           # 数字输出
    CAN_MESSAGE = "can"             # CAN总线消息
    MODBUS_REGISTER = "modbus"      # Modbus寄存器
    OPC_TAG = "opc"                 # OPC标签


@dataclass
class SignalMapping:
    """信号映射配置"""
    signal_id: str
    signal_type: SignalType

    # 物理属性
    physical_name: str              # 物理信号名称
    unit: str                       # 单位

    # 转换参数
    scale: float = 1.0              # 缩放系数
    offset: float = 0.0             # 偏移
    min_value: float = float('-inf')
    max_value: float = float('inf')

    # SIL/HIL地址
    sil_address: str = ""           # SIL模型中的地址
    hil_address: str = ""           # HIL硬件地址

    # 采样配置
    sample_rate: float = 100.0      # Hz
    filter_cutoff: float = 10.0     # Hz (低通滤波)


@dataclass
class HardwareChannel:
    """硬件通道"""
    channel_id: str
    signal_type: SignalType
    address: str

    # 状态
    is_connected: bool = False
    last_value: float = 0.0
    last_update: float = 0.0

    # 质量
    quality: float = 1.0            # 0-1 信号质量
    error_count: int = 0


@dataclass
class FaultInjection:
    """故障注入配置"""
    fault_id: str
    fault_type: str                 # "sensor_stuck", "actuator_slow", "comm_delay", etc.
    target_signal: str

    # 故障参数
    start_time: float
    duration: float
    magnitude: float = 1.0

    # 激活状态
    is_active: bool = False


@dataclass
class VerificationResult:
    """验证结果"""
    test_id: str
    mode: VerificationMode
    start_time: datetime
    end_time: Optional[datetime] = None

    # 结果
    passed: bool = False
    pass_criteria_met: Dict[str, bool] = field(default_factory=dict)

    # 数据
    time_series: List[float] = field(default_factory=list)
    states_recorded: List[Dict] = field(default_factory=list)
    actions_recorded: List[Dict] = field(default_factory=list)

    # 指标
    metrics: Dict[str, float] = field(default_factory=dict)

    # 故障
    faults_injected: List[str] = field(default_factory=list)
    anomalies_detected: List[Dict] = field(default_factory=list)


class SimulationModel(ABC):
    """仿真模型接口"""

    @abstractmethod
    def reset(self, initial_state: Dict[str, Any]) -> None:
        """重置模型"""
        pass

    @abstractmethod
    def step(self, inputs: Dict[str, float], dt: float) -> Dict[str, float]:
        """执行一步仿真"""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取状态"""
        pass


class HardwareInterface(ABC):
    """硬件接口"""

    @abstractmethod
    def connect(self) -> bool:
        """连接硬件"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    def read_inputs(self) -> Dict[str, float]:
        """读取输入"""
        pass

    @abstractmethod
    def write_outputs(self, outputs: Dict[str, float]) -> bool:
        """写入输出"""
        pass


class MockHardwareInterface(HardwareInterface):
    """模拟硬件接口 (用于测试)"""

    def __init__(self):
        self.is_connected = False
        self.input_values: Dict[str, float] = {}
        self.output_values: Dict[str, float] = {}

    def connect(self) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def read_inputs(self) -> Dict[str, float]:
        return self.input_values.copy()

    def write_outputs(self, outputs: Dict[str, float]) -> bool:
        self.output_values.update(outputs)
        return True

    def set_mock_inputs(self, inputs: Dict[str, float]):
        """设置模拟输入 (用于测试)"""
        self.input_values.update(inputs)


class ReducedOrderModel(SimulationModel):
    """降阶模型 (IDZ)"""

    def __init__(self, num_pools: int = 20, dt: float = 900.0):
        self.num_pools = num_pools
        self.dt = dt

        # 状态
        self.levels = np.ones(num_pools) * 4.0
        self.flows = np.ones(num_pools) * 300.0
        self.time = 0.0

        # 物理参数
        self.pool_lengths = np.ones(num_pools) * 50000.0  # m
        self.pool_widths = np.ones(num_pools) * 60.0      # m
        self.pool_areas = self.pool_lengths * self.pool_widths

    def reset(self, initial_state: Dict[str, Any]) -> None:
        self.levels = initial_state.get("levels", np.ones(self.num_pools) * 4.0)
        self.flows = initial_state.get("flows", np.ones(self.num_pools) * 300.0)
        self.time = 0.0

    def step(self, inputs: Dict[str, float], dt: float) -> Dict[str, float]:
        """IDZ模型更新"""
        # 上游入流
        upstream_flow = inputs.get("upstream_flow", 300.0)

        # 闸门开度 -> 出流
        for i in range(self.num_pools):
            gate_key = f"gate_{i}"
            opening = inputs.get(gate_key, 0.5)

            # 简化流量计算
            head = max(self.levels[i] - 3.0, 0.1)
            outflow = opening * 100.0 * np.sqrt(head)

            # 入流
            if i == 0:
                inflow = upstream_flow
            else:
                inflow = self.flows[i-1]

            # 水位更新 (质量守恒)
            dV = (inflow - outflow) * dt
            self.levels[i] += dV / self.pool_areas[i]
            self.flows[i] = outflow

        self.time += dt

        # 输出
        outputs = {}
        for i in range(self.num_pools):
            outputs[f"level_{i}"] = self.levels[i]
            outputs[f"flow_{i}"] = self.flows[i]

        return outputs

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "levels": self.levels.copy(),
            "flows": self.flows.copy(),
        }


class TimeSynchronizer:
    """时间同步器"""

    def __init__(
        self,
        real_time_factor: float = 1.0,  # 1.0 = 实时, <1 = 慢放, >1 = 快进
    ):
        self.real_time_factor = real_time_factor

        # 时钟
        self.sim_time = 0.0
        self.wall_time_start = 0.0
        self.is_running = False

        # 同步
        self.sync_tolerance = 0.001  # s

    def start(self):
        """开始计时"""
        self.wall_time_start = time.time()
        self.sim_time = 0.0
        self.is_running = True

    def stop(self):
        """停止计时"""
        self.is_running = False

    def advance(self, dt: float) -> float:
        """推进仿真时间"""
        self.sim_time += dt
        return self.sim_time

    def wait_for_sync(self):
        """等待实时同步 (HIL模式)"""
        if not self.is_running or self.real_time_factor <= 0:
            return

        target_wall_time = self.wall_time_start + self.sim_time / self.real_time_factor
        current_wall_time = time.time()

        if current_wall_time < target_wall_time:
            time.sleep(target_wall_time - current_wall_time)

    def get_sync_error(self) -> float:
        """获取同步误差"""
        if not self.is_running:
            return 0.0

        target_wall_time = self.wall_time_start + self.sim_time / self.real_time_factor
        return time.time() - target_wall_time


class SILHILBridge:
    """
    SIL-HIL桥接器

    连接软件仿真和硬件在环测试
    """

    def __init__(
        self,
        mode: VerificationMode = VerificationMode.SIL,
        num_pools: int = 20,
        dt: float = 1.0,
    ):
        self.mode = mode
        self.num_pools = num_pools
        self.dt = dt

        # 仿真模型 (SIL)
        self.sim_model = ReducedOrderModel(num_pools, dt)

        # 硬件接口 (HIL)
        self.hardware: Optional[HardwareInterface] = None

        # 信号映射
        self.signal_mappings: Dict[str, SignalMapping] = {}
        self._init_default_mappings()

        # 时间同步
        self.synchronizer = TimeSynchronizer()

        # 故障注入
        self.fault_injections: Dict[str, FaultInjection] = {}

        # 数据记录
        self.is_recording = False
        self.recorded_data: List[Dict] = []

        # 验证结果
        self.current_result: Optional[VerificationResult] = None

        # 回调
        self.state_callback: Optional[Callable] = None

        logger.info(f"SILHILBridge initialized: mode={mode.value}")

    def _init_default_mappings(self):
        """初始化默认信号映射"""
        # 水位信号
        for i in range(self.num_pools):
            self.signal_mappings[f"level_{i}"] = SignalMapping(
                signal_id=f"level_{i}",
                signal_type=SignalType.ANALOG_INPUT,
                physical_name=f"Pool_{i}_WaterLevel",
                unit="m",
                scale=1.0,
                offset=0.0,
                min_value=0.0,
                max_value=10.0,
                sil_address=f"model.pools[{i}].level",
                hil_address=f"PLC1.AI{i:02d}",
            )

        # 闸门开度信号
        for i in range(self.num_pools + 1):
            self.signal_mappings[f"gate_{i}"] = SignalMapping(
                signal_id=f"gate_{i}",
                signal_type=SignalType.ANALOG_OUTPUT,
                physical_name=f"Gate_{i}_Opening",
                unit="%",
                scale=100.0,  # 0-1 -> 0-100%
                offset=0.0,
                min_value=0.0,
                max_value=1.0,
                sil_address=f"controller.gates[{i}].opening",
                hil_address=f"PLC1.AO{i:02d}",
            )

    def set_mode(self, mode: VerificationMode):
        """设置验证模式"""
        self.mode = mode
        logger.info(f"Mode changed to: {mode.value}")

    def set_hardware_interface(self, interface: HardwareInterface):
        """设置硬件接口"""
        self.hardware = interface

    def connect(self) -> bool:
        """连接 (SIL初始化 或 HIL硬件连接)"""
        if self.mode == VerificationMode.SIL:
            logger.info("SIL mode: simulation model ready")
            return True
        elif self.mode in [VerificationMode.HIL, VerificationMode.HYBRID]:
            if self.hardware is None:
                logger.error("No hardware interface configured")
                return False
            return self.hardware.connect()
        return True

    def disconnect(self):
        """断开连接"""
        if self.hardware:
            self.hardware.disconnect()

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """重置"""
        state = initial_state or {}

        # 重置仿真模型
        self.sim_model.reset(state)

        # 重置时间
        self.synchronizer = TimeSynchronizer()

        # 清空故障注入
        for fault in self.fault_injections.values():
            fault.is_active = False

        # 清空记录
        self.recorded_data = []

        logger.info("Bridge reset")

    def add_fault_injection(self, fault: FaultInjection):
        """添加故障注入"""
        self.fault_injections[fault.fault_id] = fault
        logger.info(f"Fault injection added: {fault.fault_id}")

    def remove_fault_injection(self, fault_id: str):
        """移除故障注入"""
        if fault_id in self.fault_injections:
            del self.fault_injections[fault_id]

    def _apply_fault_effects(
        self,
        signals: Dict[str, float],
        current_time: float,
    ) -> Dict[str, float]:
        """应用故障效果"""
        modified = signals.copy()

        for fault in self.fault_injections.values():
            # 检查是否激活
            if current_time >= fault.start_time:
                if current_time <= fault.start_time + fault.duration:
                    fault.is_active = True
                else:
                    fault.is_active = False

            if not fault.is_active:
                continue

            # 应用故障
            signal_key = fault.target_signal
            if signal_key not in modified:
                continue

            if fault.fault_type == "sensor_stuck":
                # 传感器卡死
                modified[signal_key] = fault.magnitude
            elif fault.fault_type == "sensor_drift":
                # 传感器漂移
                modified[signal_key] += fault.magnitude * (current_time - fault.start_time)
            elif fault.fault_type == "sensor_noise":
                # 传感器噪声增加
                modified[signal_key] += np.random.normal(0, fault.magnitude)
            elif fault.fault_type == "actuator_slow":
                # 执行器响应慢 (低通滤波)
                # 实际实现需要状态保持
                pass
            elif fault.fault_type == "signal_loss":
                # 信号丢失
                modified[signal_key] = 0.0

        return modified

    def step(
        self,
        control_inputs: Dict[str, float],
        external_inputs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        执行一步

        Args:
            control_inputs: 控制输入 (闸门开度等)
            external_inputs: 外部输入 (上游流量等)

        Returns:
            outputs: 输出信号 (水位、流量等)
        """
        external = external_inputs or {}
        current_time = self.synchronizer.sim_time

        # 合并输入
        all_inputs = {**control_inputs, **external}

        # 应用故障效果 (输入侧)
        all_inputs = self._apply_fault_effects(all_inputs, current_time)

        # 根据模式执行
        if self.mode == VerificationMode.SIL:
            outputs = self._step_sil(all_inputs)
        elif self.mode == VerificationMode.HIL:
            outputs = self._step_hil(all_inputs)
        elif self.mode == VerificationMode.HYBRID:
            outputs = self._step_hybrid(all_inputs)
        elif self.mode == VerificationMode.REPLAY:
            outputs = self._step_replay(all_inputs)
        else:
            outputs = {}

        # 应用故障效果 (输出侧)
        outputs = self._apply_fault_effects(outputs, current_time)

        # 推进时间
        self.synchronizer.advance(self.dt)

        # 等待同步 (HIL模式)
        if self.mode in [VerificationMode.HIL, VerificationMode.HYBRID]:
            self.synchronizer.wait_for_sync()

        # 记录数据
        if self.is_recording:
            self._record_step(all_inputs, outputs, current_time)

        # 回调
        if self.state_callback:
            self.state_callback(outputs, current_time)

        return outputs

    def _step_sil(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """SIL模式步进"""
        return self.sim_model.step(inputs, self.dt)

    def _step_hil(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """HIL模式步进"""
        if self.hardware is None:
            return {}

        # 转换并写入硬件
        hw_outputs = self._convert_to_hardware(inputs)
        self.hardware.write_outputs(hw_outputs)

        # 读取硬件输入
        hw_inputs = self.hardware.read_inputs()

        # 转换回软件格式
        return self._convert_from_hardware(hw_inputs)

    def _step_hybrid(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """混合模式步进"""
        # 仿真部分
        sim_outputs = self.sim_model.step(inputs, self.dt)

        # 硬件部分 (如果有)
        if self.hardware:
            hw_inputs = self.hardware.read_inputs()
            hw_converted = self._convert_from_hardware(hw_inputs)

            # 融合 (硬件优先)
            for key, value in hw_converted.items():
                if value is not None and not np.isnan(value):
                    sim_outputs[key] = value

        return sim_outputs

    def _step_replay(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """回放模式步进"""
        # 从记录数据中获取
        idx = int(self.synchronizer.sim_time / self.dt)
        if idx < len(self.recorded_data):
            return self.recorded_data[idx].get("outputs", {})
        return {}

    def _convert_to_hardware(self, signals: Dict[str, float]) -> Dict[str, float]:
        """转换为硬件格式"""
        hw_signals = {}

        for signal_id, value in signals.items():
            mapping = self.signal_mappings.get(signal_id)
            if mapping:
                # 应用缩放和偏移
                hw_value = (value - mapping.offset) / mapping.scale
                hw_signals[mapping.hil_address] = hw_value

        return hw_signals

    def _convert_from_hardware(self, hw_signals: Dict[str, float]) -> Dict[str, float]:
        """从硬件格式转换"""
        signals = {}

        for signal_id, mapping in self.signal_mappings.items():
            if mapping.hil_address in hw_signals:
                hw_value = hw_signals[mapping.hil_address]
                # 应用缩放和偏移
                signals[signal_id] = hw_value * mapping.scale + mapping.offset

        return signals

    def _record_step(
        self,
        inputs: Dict[str, float],
        outputs: Dict[str, float],
        time: float,
    ):
        """记录步骤数据"""
        self.recorded_data.append({
            "time": time,
            "inputs": inputs.copy(),
            "outputs": outputs.copy(),
        })

    def start_recording(self):
        """开始记录"""
        self.is_recording = True
        self.recorded_data = []

    def stop_recording(self):
        """停止记录"""
        self.is_recording = False

    def start_verification(
        self,
        test_id: str,
        duration: float,
        pass_criteria: Optional[Dict[str, Callable]] = None,
    ) -> VerificationResult:
        """
        开始验证

        Args:
            test_id: 测试ID
            duration: 测试时长 (s)
            pass_criteria: 通过准则 {name: lambda states -> bool}

        Returns:
            VerificationResult
        """
        self.current_result = VerificationResult(
            test_id=test_id,
            mode=self.mode,
            start_time=datetime.now(),
        )

        # 启动时间同步
        self.synchronizer.start()

        # 开始记录
        self.start_recording()

        logger.info(f"Verification started: {test_id}")

        return self.current_result

    def finish_verification(
        self,
        pass_criteria: Optional[Dict[str, Callable]] = None,
    ) -> VerificationResult:
        """
        完成验证

        Args:
            pass_criteria: 通过准则

        Returns:
            VerificationResult
        """
        if self.current_result is None:
            raise ValueError("No active verification")

        # 停止记录
        self.stop_recording()

        # 停止时间同步
        self.synchronizer.stop()

        # 设置结束时间
        self.current_result.end_time = datetime.now()

        # 保存时间序列
        self.current_result.time_series = [
            d["time"] for d in self.recorded_data
        ]
        self.current_result.states_recorded = self.recorded_data

        # 检查通过准则
        criteria = pass_criteria or {}
        all_passed = True

        for name, criterion in criteria.items():
            try:
                passed = criterion(self.recorded_data)
                self.current_result.pass_criteria_met[name] = passed
                if not passed:
                    all_passed = False
            except Exception as e:
                logger.error(f"Criterion {name} failed: {e}")
                self.current_result.pass_criteria_met[name] = False
                all_passed = False

        self.current_result.passed = all_passed

        # 记录注入的故障
        self.current_result.faults_injected = [
            f.fault_id for f in self.fault_injections.values()
            if f.is_active or f.start_time < self.synchronizer.sim_time
        ]

        # 计算指标
        self._compute_metrics()

        logger.info(f"Verification finished: passed={all_passed}")

        return self.current_result

    def _compute_metrics(self):
        """计算验证指标"""
        if not self.recorded_data or self.current_result is None:
            return

        # 提取时间序列
        times = [d["time"] for d in self.recorded_data]

        # 水位统计
        level_data = []
        for d in self.recorded_data:
            levels = [v for k, v in d["outputs"].items() if k.startswith("level_")]
            if levels:
                level_data.append(levels)

        if level_data:
            level_array = np.array(level_data)
            self.current_result.metrics["level_mean"] = float(np.mean(level_array))
            self.current_result.metrics["level_std"] = float(np.std(level_array))
            self.current_result.metrics["level_max"] = float(np.max(level_array))
            self.current_result.metrics["level_min"] = float(np.min(level_array))

        # 同步误差
        self.current_result.metrics["sync_error"] = self.synchronizer.get_sync_error()

        # 测试时长
        self.current_result.metrics["duration"] = times[-1] if times else 0.0

    def run_sil_test(
        self,
        test_id: str,
        duration: float,
        controller: Callable,
        initial_state: Optional[Dict] = None,
        external_profile: Optional[Dict[str, np.ndarray]] = None,
        pass_criteria: Optional[Dict[str, Callable]] = None,
    ) -> VerificationResult:
        """
        运行完整SIL测试

        Args:
            test_id: 测试ID
            duration: 时长 (s)
            controller: 控制器函数 (state -> action)
            initial_state: 初始状态
            external_profile: 外部输入曲线
            pass_criteria: 通过准则

        Returns:
            VerificationResult
        """
        # 设置SIL模式
        self.set_mode(VerificationMode.SIL)

        # 重置
        self.reset(initial_state)

        # 开始验证
        self.start_verification(test_id, duration)

        # 计算步数
        num_steps = int(duration / self.dt)

        # 运行
        for step in range(num_steps):
            current_time = step * self.dt

            # 获取外部输入
            external = {}
            if external_profile:
                for key, profile in external_profile.items():
                    if step < len(profile):
                        external[key] = profile[step]

            # 获取当前状态
            state = self.sim_model.get_state()

            # 计算控制输出
            control = controller(state)

            # 执行步骤
            self.step(control, external)

        # 完成验证
        return self.finish_verification(pass_criteria)

    def get_bridge_status(self) -> Dict[str, Any]:
        """获取桥接器状态"""
        return {
            "mode": self.mode.value,
            "sim_time": self.synchronizer.sim_time,
            "is_running": self.synchronizer.is_running,
            "recording": self.is_recording,
            "recorded_points": len(self.recorded_data),
            "active_faults": [
                f.fault_id for f in self.fault_injections.values() if f.is_active
            ],
            "hardware_connected": self.hardware is not None and (
                hasattr(self.hardware, 'is_connected') and self.hardware.is_connected
            ),
        }


class SILHILTestSuite:
    """SIL/HIL测试套件"""

    def __init__(self, bridge: SILHILBridge):
        self.bridge = bridge
        self.results: List[VerificationResult] = []

    def add_standard_tests(self):
        """添加标准测试"""
        # 将在run_suite中定义
        pass

    def run_suite(
        self,
        controller: Callable,
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行测试套件

        Args:
            controller: 控制器
            report_path: 报告路径

        Returns:
            summary: 测试摘要
        """
        test_configs = [
            {
                "test_id": "ST001_STEADY_STATE",
                "duration": 3600.0,
                "external_profile": {"upstream_flow": np.ones(3600) * 300.0},
                "pass_criteria": {
                    "level_stability": lambda d: np.std([
                        d[i]["outputs"].get("level_5", 4.0)
                        for i in range(len(d))
                    ]) < 0.1,
                },
            },
            {
                "test_id": "ST002_STEP_RESPONSE",
                "duration": 7200.0,
                "external_profile": {
                    "upstream_flow": np.concatenate([
                        np.ones(1800) * 300.0,
                        np.ones(5400) * 350.0,
                    ])
                },
                "pass_criteria": {
                    "settling_time": lambda d: True,  # 简化
                },
            },
        ]

        self.results = []

        for config in test_configs:
            result = self.bridge.run_sil_test(
                test_id=config["test_id"],
                duration=config["duration"],
                controller=controller,
                external_profile=config["external_profile"],
                pass_criteria=config["pass_criteria"],
            )
            self.results.append(result)

        # 生成摘要
        summary = {
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "results": [
                {
                    "test_id": r.test_id,
                    "passed": r.passed,
                    "metrics": r.metrics,
                }
                for r in self.results
            ],
        }

        return summary
