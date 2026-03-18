# Phase 5.9: Modbus Protocol Adapter
# Modbus协议适配器 - 工业控制标准协议

import asyncio
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModbusDataType(Enum):
    """Modbus数据类型"""
    COIL = "coil"  # 1-bit, read/write
    DISCRETE_INPUT = "discrete_input"  # 1-bit, read-only
    HOLDING_REGISTER = "holding_register"  # 16-bit, read/write
    INPUT_REGISTER = "input_register"  # 16-bit, read-only


class ModbusRegisterFormat(Enum):
    """寄存器数据格式"""
    UINT16 = "uint16"
    INT16 = "int16"
    UINT32 = "uint32"  # 2 registers
    INT32 = "int32"  # 2 registers
    FLOAT32 = "float32"  # 2 registers
    UINT64 = "uint64"  # 4 registers
    INT64 = "int64"  # 4 registers
    FLOAT64 = "float64"  # 4 registers
    STRING = "string"  # Variable length


class ModbusByteOrder(Enum):
    """字节序"""
    BIG_ENDIAN = "big"
    LITTLE_ENDIAN = "little"
    BIG_ENDIAN_SWAP = "big_swap"  # CDAB
    LITTLE_ENDIAN_SWAP = "little_swap"  # BADC


class ModbusFunctionCode(Enum):
    """Modbus功能码"""
    READ_COILS = 0x01
    READ_DISCRETE_INPUTS = 0x02
    READ_HOLDING_REGISTERS = 0x03
    READ_INPUT_REGISTERS = 0x04
    WRITE_SINGLE_COIL = 0x05
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_COILS = 0x0F
    WRITE_MULTIPLE_REGISTERS = 0x10
    READ_WRITE_MULTIPLE_REGISTERS = 0x17


class ModbusExceptionCode(Enum):
    """Modbus异常码"""
    ILLEGAL_FUNCTION = 0x01
    ILLEGAL_DATA_ADDRESS = 0x02
    ILLEGAL_DATA_VALUE = 0x03
    SLAVE_DEVICE_FAILURE = 0x04
    ACKNOWLEDGE = 0x05
    SLAVE_DEVICE_BUSY = 0x06
    MEMORY_PARITY_ERROR = 0x08
    GATEWAY_PATH_UNAVAILABLE = 0x0A
    GATEWAY_TARGET_DEVICE_FAILED = 0x0B


@dataclass
class ModbusDeviceConfig:
    """Modbus设备配置"""
    # Connection settings
    host: str = "localhost"
    port: int = 502
    unit_id: int = 1
    timeout: float = 3.0

    # Protocol settings
    protocol: str = "tcp"  # tcp, rtu, ascii
    serial_port: Optional[str] = None  # For RTU/ASCII
    baudrate: int = 9600
    parity: str = "N"
    stopbits: int = 1
    bytesize: int = 8

    # Advanced settings
    byte_order: ModbusByteOrder = ModbusByteOrder.BIG_ENDIAN
    word_order: ModbusByteOrder = ModbusByteOrder.BIG_ENDIAN
    retries: int = 3
    retry_delay: float = 0.5
    inter_frame_delay: float = 0.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10


@dataclass
class ModbusRegister:
    """Modbus寄存器定义"""
    address: int
    name: str
    data_type: ModbusDataType = ModbusDataType.HOLDING_REGISTER
    format: ModbusRegisterFormat = ModbusRegisterFormat.UINT16
    count: int = 1  # Number of registers
    scale: float = 1.0
    offset: float = 0.0
    unit: Optional[str] = None
    description: str = ""
    writable: bool = False

    # Water network metadata
    pool_id: Optional[int] = None
    measurement_type: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        # Calculate register count based on format
        if self.format in [ModbusRegisterFormat.UINT32, ModbusRegisterFormat.INT32,
                           ModbusRegisterFormat.FLOAT32]:
            self.count = max(self.count, 2)
        elif self.format in [ModbusRegisterFormat.UINT64, ModbusRegisterFormat.INT64,
                             ModbusRegisterFormat.FLOAT64]:
            self.count = max(self.count, 4)


@dataclass
class ModbusReadResult:
    """Modbus读取结果"""
    register: ModbusRegister
    raw_value: Any
    scaled_value: Any
    timestamp: datetime
    success: bool
    error: Optional[str] = None

    @property
    def quality(self) -> str:
        return "good" if self.success else "bad"


class ModbusAdapter:
    """Modbus协议适配器"""

    def __init__(self, config: ModbusDeviceConfig):
        self.config = config
        self.connected = False
        self.registers: Dict[str, ModbusRegister] = {}
        self.register_by_address: Dict[int, ModbusRegister] = {}
        self.data_cache: Dict[str, ModbusReadResult] = {}

        # Internal state
        self._lock = threading.RLock()
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._reconnect_count = 0

        # Simulated register values (for testing)
        self._simulated_registers: Dict[int, int] = {}
        self._simulated_coils: Dict[int, bool] = {}

        # Polling configuration
        self._poll_groups: Dict[str, List[ModbusRegister]] = defaultdict(list)
        self._poll_intervals: Dict[str, float] = {}
        self._poll_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Callbacks
        self._connection_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []

        # Statistics
        self.stats = {
            'reads': 0,
            'writes': 0,
            'errors': 0,
            'timeouts': 0,
            'reconnects': 0,
            'last_read_time': None,
            'last_write_time': None,
            'avg_response_time_ms': 0.0,
        }

        logger.info(f"Modbus adapter created for {config.host}:{config.port}")

    def connect(self) -> bool:
        """连接到Modbus设备"""
        with self._lock:
            if self.connected:
                logger.warning("Already connected to Modbus device")
                return True

            try:
                logger.info(f"Connecting to Modbus device: {self.config.host}:{self.config.port}")

                if self.config.protocol == "tcp":
                    self._connect_tcp()
                elif self.config.protocol == "rtu":
                    self._connect_rtu()
                else:
                    raise ValueError(f"Unsupported protocol: {self.config.protocol}")

                self.connected = True
                self._reconnect_count = 0

                # Start polling thread
                self._start_polling()

                # Notify callbacks
                for callback in self._connection_callbacks:
                    try:
                        callback(True, None)
                    except Exception as e:
                        logger.error(f"Connection callback error: {e}")

                logger.info("Connected to Modbus device")
                return True

            except Exception as e:
                logger.error(f"Failed to connect to Modbus device: {e}")
                self.stats['errors'] += 1

                for callback in self._error_callbacks:
                    try:
                        callback(e)
                    except Exception as ce:
                        logger.error(f"Error callback failed: {ce}")

                return False

    def _connect_tcp(self):
        """TCP连接"""
        # Simulate TCP connection
        time.sleep(0.05)
        logger.debug(f"TCP connection established to {self.config.host}:{self.config.port}")

    def _connect_rtu(self):
        """RTU串口连接"""
        # Simulate serial connection
        time.sleep(0.05)
        logger.debug(f"RTU connection established on {self.config.serial_port}")

    def disconnect(self):
        """断开Modbus连接"""
        with self._lock:
            if not self.connected:
                return

            logger.info("Disconnecting from Modbus device")

            # Stop polling
            self._stop_polling()

            self.connected = False

            # Notify callbacks
            for callback in self._connection_callbacks:
                try:
                    callback(False, None)
                except Exception as e:
                    logger.error(f"Disconnection callback error: {e}")

            logger.info("Disconnected from Modbus device")

    def _start_polling(self):
        """启动轮询线程"""
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _stop_polling(self):
        """停止轮询线程"""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None

    def _poll_loop(self):
        """轮询循环"""
        last_poll_times: Dict[str, float] = {}

        while self._running:
            try:
                current_time = time.time()

                for group_name, registers in self._poll_groups.items():
                    interval = self._poll_intervals.get(group_name, 1.0)
                    last_time = last_poll_times.get(group_name, 0)

                    if current_time - last_time >= interval:
                        # Poll registers
                        results = self._poll_registers(registers)

                        # Invoke callbacks
                        for callback in self._poll_callbacks.get(group_name, []):
                            try:
                                callback(results)
                            except Exception as e:
                                logger.error(f"Poll callback error: {e}")

                        last_poll_times[group_name] = current_time

                time.sleep(0.1)  # 100ms tick

            except Exception as e:
                logger.error(f"Poll loop error: {e}")
                self.stats['errors'] += 1

    def _poll_registers(self, registers: List[ModbusRegister]) -> Dict[str, ModbusReadResult]:
        """轮询一组寄存器"""
        results = {}
        for register in registers:
            result = self.read_register(register.name)
            if result:
                results[register.name] = result
        return results

    def register_register(self, register: ModbusRegister):
        """注册Modbus寄存器"""
        with self._lock:
            self.registers[register.name] = register
            self.register_by_address[register.address] = register
            logger.debug(f"Registered register: {register.name} at address {register.address}")

    def register_registers(self, registers: List[ModbusRegister]):
        """批量注册寄存器"""
        for register in registers:
            self.register_register(register)

    def add_poll_group(
        self,
        group_name: str,
        registers: List[ModbusRegister],
        interval: float = 1.0,
        callback: Optional[Callable] = None,
    ):
        """添加轮询组"""
        with self._lock:
            # Register all registers
            for register in registers:
                self.register_register(register)
                self._poll_groups[group_name].append(register)

            self._poll_intervals[group_name] = interval

            if callback:
                self._poll_callbacks[group_name].append(callback)

            logger.info(f"Added poll group '{group_name}' with {len(registers)} registers")

    def read_register(self, name: str) -> Optional[ModbusReadResult]:
        """读取单个寄存器"""
        register = self.registers.get(name)
        if not register:
            logger.warning(f"Register not found: {name}")
            return None

        return self._read_register(register)

    def read_registers(self, names: List[str]) -> Dict[str, ModbusReadResult]:
        """批量读取寄存器"""
        results = {}
        for name in names:
            result = self.read_register(name)
            if result:
                results[name] = result
        return results

    def _read_register(self, register: ModbusRegister) -> ModbusReadResult:
        """内部读取寄存器"""
        start_time = time.time()

        try:
            if not self.connected:
                raise Exception("Not connected to Modbus device")

            # Read based on data type
            if register.data_type == ModbusDataType.COIL:
                raw_value = self._read_coils(register.address, 1)[0]
            elif register.data_type == ModbusDataType.DISCRETE_INPUT:
                raw_value = self._read_discrete_inputs(register.address, 1)[0]
            elif register.data_type == ModbusDataType.HOLDING_REGISTER:
                raw_value = self._read_holding_registers(register.address, register.count)
            elif register.data_type == ModbusDataType.INPUT_REGISTER:
                raw_value = self._read_input_registers(register.address, register.count)
            else:
                raise ValueError(f"Unknown data type: {register.data_type}")

            # Convert to proper format
            converted_value = self._convert_from_registers(raw_value, register)

            # Apply scaling
            if isinstance(converted_value, (int, float)):
                scaled_value = converted_value * register.scale + register.offset
            else:
                scaled_value = converted_value

            result = ModbusReadResult(
                register=register,
                raw_value=raw_value,
                scaled_value=scaled_value,
                timestamp=datetime.now(),
                success=True,
            )

            # Update cache
            self.data_cache[register.name] = result

            # Update statistics
            self.stats['reads'] += 1
            self.stats['last_read_time'] = datetime.now()
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats['avg_response_time_ms'] = (
                self.stats['avg_response_time_ms'] * 0.9 + elapsed_ms * 0.1
            )

            return result

        except Exception as e:
            logger.error(f"Failed to read register {register.name}: {e}")
            self.stats['errors'] += 1

            return ModbusReadResult(
                register=register,
                raw_value=None,
                scaled_value=None,
                timestamp=datetime.now(),
                success=False,
                error=str(e),
            )

    def _read_coils(self, address: int, count: int) -> List[bool]:
        """读取线圈 (FC 01)"""
        result = []
        for i in range(count):
            addr = address + i
            value = self._simulated_coils.get(addr, False)
            result.append(value)
        return result

    def _read_discrete_inputs(self, address: int, count: int) -> List[bool]:
        """读取离散输入 (FC 02)"""
        import random
        return [random.choice([True, False]) for _ in range(count)]

    def _read_holding_registers(self, address: int, count: int) -> List[int]:
        """读取保持寄存器 (FC 03)"""
        result = []
        for i in range(count):
            addr = address + i
            if addr not in self._simulated_registers:
                # Generate simulated value
                import random
                self._simulated_registers[addr] = random.randint(0, 65535)
            result.append(self._simulated_registers[addr])
        return result

    def _read_input_registers(self, address: int, count: int) -> List[int]:
        """读取输入寄存器 (FC 04)"""
        import random
        return [random.randint(0, 65535) for _ in range(count)]

    def _convert_from_registers(
        self,
        raw_value: Union[List[int], List[bool], bool],
        register: ModbusRegister,
    ) -> Any:
        """将原始寄存器值转换为目标格式"""
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, list) and len(raw_value) > 0 and isinstance(raw_value[0], bool):
            return raw_value[0] if len(raw_value) == 1 else raw_value

        # Handle register list
        if not isinstance(raw_value, list):
            raw_value = [raw_value]

        format_type = register.format
        byte_order = self.config.byte_order
        word_order = self.config.word_order

        if format_type == ModbusRegisterFormat.UINT16:
            return raw_value[0]
        elif format_type == ModbusRegisterFormat.INT16:
            val = raw_value[0]
            if val >= 32768:
                val -= 65536
            return val
        elif format_type == ModbusRegisterFormat.UINT32:
            if word_order == ModbusByteOrder.BIG_ENDIAN:
                return (raw_value[0] << 16) | raw_value[1]
            else:
                return (raw_value[1] << 16) | raw_value[0]
        elif format_type == ModbusRegisterFormat.INT32:
            val = self._convert_from_registers(raw_value, ModbusRegister(
                address=register.address, name=register.name, format=ModbusRegisterFormat.UINT32
            ))
            if val >= 2147483648:
                val -= 4294967296
            return val
        elif format_type == ModbusRegisterFormat.FLOAT32:
            if word_order == ModbusByteOrder.BIG_ENDIAN:
                combined = (raw_value[0] << 16) | raw_value[1]
            else:
                combined = (raw_value[1] << 16) | raw_value[0]
            return struct.unpack('>f', struct.pack('>I', combined))[0]
        elif format_type == ModbusRegisterFormat.FLOAT64:
            if len(raw_value) < 4:
                return 0.0
            if word_order == ModbusByteOrder.BIG_ENDIAN:
                combined = (raw_value[0] << 48) | (raw_value[1] << 32) | (raw_value[2] << 16) | raw_value[3]
            else:
                combined = (raw_value[3] << 48) | (raw_value[2] << 32) | (raw_value[1] << 16) | raw_value[0]
            return struct.unpack('>d', struct.pack('>Q', combined))[0]
        else:
            return raw_value[0] if len(raw_value) == 1 else raw_value

    def write_register(self, name: str, value: Any) -> bool:
        """写入单个寄存器"""
        register = self.registers.get(name)
        if not register:
            logger.warning(f"Register not found: {name}")
            return False

        return self._write_register(register, value)

    def write_registers(self, values: Dict[str, Any]) -> Dict[str, bool]:
        """批量写入寄存器"""
        results = {}
        for name, value in values.items():
            results[name] = self.write_register(name, value)
        return results

    def _write_register(self, register: ModbusRegister, value: Any) -> bool:
        """内部写入寄存器"""
        try:
            if not self.connected:
                raise Exception("Not connected to Modbus device")

            if not register.writable:
                raise ValueError(f"Register is not writable: {register.name}")

            # Remove scaling
            if isinstance(value, (int, float)):
                raw_value = (value - register.offset) / register.scale
            else:
                raw_value = value

            # Convert to register format
            if register.data_type == ModbusDataType.COIL:
                self._write_coil(register.address, bool(raw_value))
            elif register.data_type == ModbusDataType.HOLDING_REGISTER:
                register_values = self._convert_to_registers(raw_value, register)
                self._write_holding_registers(register.address, register_values)
            else:
                raise ValueError(f"Cannot write to data type: {register.data_type}")

            # Update statistics
            self.stats['writes'] += 1
            self.stats['last_write_time'] = datetime.now()

            logger.debug(f"Written {value} to register {register.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to write register {register.name}: {e}")
            self.stats['errors'] += 1
            return False

    def _write_coil(self, address: int, value: bool):
        """写入线圈 (FC 05)"""
        self._simulated_coils[address] = value

    def _write_holding_registers(self, address: int, values: List[int]):
        """写入保持寄存器 (FC 06/16)"""
        for i, value in enumerate(values):
            self._simulated_registers[address + i] = value & 0xFFFF

    def _convert_to_registers(self, value: Any, register: ModbusRegister) -> List[int]:
        """将值转换为寄存器格式"""
        format_type = register.format
        word_order = self.config.word_order

        if format_type == ModbusRegisterFormat.UINT16:
            return [int(value) & 0xFFFF]
        elif format_type == ModbusRegisterFormat.INT16:
            val = int(value)
            if val < 0:
                val += 65536
            return [val & 0xFFFF]
        elif format_type == ModbusRegisterFormat.UINT32:
            val = int(value) & 0xFFFFFFFF
            if word_order == ModbusByteOrder.BIG_ENDIAN:
                return [(val >> 16) & 0xFFFF, val & 0xFFFF]
            else:
                return [val & 0xFFFF, (val >> 16) & 0xFFFF]
        elif format_type == ModbusRegisterFormat.FLOAT32:
            packed = struct.pack('>f', float(value))
            combined = struct.unpack('>I', packed)[0]
            if word_order == ModbusByteOrder.BIG_ENDIAN:
                return [(combined >> 16) & 0xFFFF, combined & 0xFFFF]
            else:
                return [combined & 0xFFFF, (combined >> 16) & 0xFFFF]
        else:
            return [int(value) & 0xFFFF]

    def on_connect(self, callback: Callable[[bool, Optional[str]], None]):
        """注册连接状态回调"""
        self._connection_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]):
        """注册错误回调"""
        self._error_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """获取适配器状态"""
        return {
            'connected': self.connected,
            'host': self.config.host,
            'port': self.config.port,
            'unit_id': self.config.unit_id,
            'protocol': self.config.protocol,
            'register_count': len(self.registers),
            'poll_group_count': len(self._poll_groups),
            'statistics': self.stats.copy(),
        }

    def create_water_network_registers(
        self,
        num_pools: int,
        base_address: int = 0,
    ) -> List[ModbusRegister]:
        """创建智能水网标准寄存器映射"""
        registers = []
        addr = base_address

        for pool_id in range(num_pools):
            pool_base = addr + pool_id * 100  # Each pool uses 100 addresses

            # Water level (float32, 2 registers)
            registers.append(ModbusRegister(
                address=pool_base,
                name=f"pool_{pool_id}_water_level",
                data_type=ModbusDataType.HOLDING_REGISTER,
                format=ModbusRegisterFormat.FLOAT32,
                scale=0.001,  # mm to m
                unit='m',
                measurement_type='water_level',
                pool_id=pool_id,
                min_value=0.0,
                max_value=10.0,
            ))

            # Inflow rate (float32, 2 registers)
            registers.append(ModbusRegister(
                address=pool_base + 2,
                name=f"pool_{pool_id}_inflow_rate",
                data_type=ModbusDataType.HOLDING_REGISTER,
                format=ModbusRegisterFormat.FLOAT32,
                scale=0.001,  # L/s to m³/s
                unit='m³/s',
                measurement_type='flow_rate',
                pool_id=pool_id,
            ))

            # Outflow rate (float32, 2 registers)
            registers.append(ModbusRegister(
                address=pool_base + 4,
                name=f"pool_{pool_id}_outflow_rate",
                data_type=ModbusDataType.HOLDING_REGISTER,
                format=ModbusRegisterFormat.FLOAT32,
                scale=0.001,
                unit='m³/s',
                measurement_type='flow_rate',
                pool_id=pool_id,
            ))

            # Gate position (uint16)
            registers.append(ModbusRegister(
                address=pool_base + 6,
                name=f"pool_{pool_id}_gate_position",
                data_type=ModbusDataType.HOLDING_REGISTER,
                format=ModbusRegisterFormat.UINT16,
                scale=0.01,  # 0-10000 -> 0-100%
                unit='%',
                measurement_type='gate_position',
                pool_id=pool_id,
                writable=True,
            ))

            # Gate setpoint (uint16) - writable
            registers.append(ModbusRegister(
                address=pool_base + 7,
                name=f"pool_{pool_id}_gate_setpoint",
                data_type=ModbusDataType.HOLDING_REGISTER,
                format=ModbusRegisterFormat.UINT16,
                scale=0.01,
                unit='%',
                measurement_type='gate_setpoint',
                pool_id=pool_id,
                writable=True,
            ))

            # Temperature (int16)
            registers.append(ModbusRegister(
                address=pool_base + 8,
                name=f"pool_{pool_id}_temperature",
                data_type=ModbusDataType.HOLDING_REGISTER,
                format=ModbusRegisterFormat.INT16,
                scale=0.1,  # 0.1°C resolution
                unit='°C',
                measurement_type='temperature',
                pool_id=pool_id,
            ))

            # Pressure (uint16)
            registers.append(ModbusRegister(
                address=pool_base + 9,
                name=f"pool_{pool_id}_pressure",
                data_type=ModbusDataType.HOLDING_REGISTER,
                format=ModbusRegisterFormat.UINT16,
                scale=0.1,  # 0.1 kPa resolution
                unit='kPa',
                measurement_type='pressure',
                pool_id=pool_id,
            ))

            # Status word (uint16)
            registers.append(ModbusRegister(
                address=pool_base + 10,
                name=f"pool_{pool_id}_status",
                data_type=ModbusDataType.HOLDING_REGISTER,
                format=ModbusRegisterFormat.UINT16,
                pool_id=pool_id,
            ))

            # Alarm word (uint16)
            registers.append(ModbusRegister(
                address=pool_base + 11,
                name=f"pool_{pool_id}_alarm",
                data_type=ModbusDataType.HOLDING_REGISTER,
                format=ModbusRegisterFormat.UINT16,
                pool_id=pool_id,
            ))

        # System registers
        system_base = base_address + num_pools * 100

        # System status
        registers.append(ModbusRegister(
            address=system_base,
            name="system_status",
            data_type=ModbusDataType.HOLDING_REGISTER,
            format=ModbusRegisterFormat.UINT16,
        ))

        # System mode
        registers.append(ModbusRegister(
            address=system_base + 1,
            name="system_mode",
            data_type=ModbusDataType.HOLDING_REGISTER,
            format=ModbusRegisterFormat.UINT16,
            writable=True,
        ))

        # Emergency stop (coil)
        registers.append(ModbusRegister(
            address=0,
            name="emergency_stop",
            data_type=ModbusDataType.COIL,
            writable=True,
        ))

        # Register all
        self.register_registers(registers)

        logger.info(f"Created {len(registers)} water network Modbus registers for {num_pools} pools")
        return registers
