# Phase 5.9: Real-time Data Interface
# 实时数据接口模块 - 工业协议适配器和数据回放

from .opc_ua_adapter import (
    OPCUAAdapter,
    OPCUANode,
    OPCUASubscription,
    OPCUADataPoint,
    OPCUAConnectionConfig,
)

from .modbus_adapter import (
    ModbusAdapter,
    ModbusRegister,
    ModbusDeviceConfig,
    ModbusDataType,
    ModbusReadResult,
)

from .scada_interface import (
    SCADAInterface,
    SCADATag,
    SCADAAlarm,
    SCADACommand,
    SCADAConnectionStatus,
)

from .data_replay import (
    DataReplayEngine,
    ReplaySession,
    ReplayConfig,
    TimeScaleMode,
    DataSource,
)

__all__ = [
    # OPC-UA
    'OPCUAAdapter',
    'OPCUANode',
    'OPCUASubscription',
    'OPCUADataPoint',
    'OPCUAConnectionConfig',
    # Modbus
    'ModbusAdapter',
    'ModbusRegister',
    'ModbusDeviceConfig',
    'ModbusDataType',
    'ModbusReadResult',
    # SCADA
    'SCADAInterface',
    'SCADATag',
    'SCADAAlarm',
    'SCADACommand',
    'SCADAConnectionStatus',
    # Data Replay
    'DataReplayEngine',
    'ReplaySession',
    'ReplayConfig',
    'TimeScaleMode',
    'DataSource',
]
