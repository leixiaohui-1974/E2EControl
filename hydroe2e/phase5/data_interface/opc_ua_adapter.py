# Phase 5.9: OPC-UA Protocol Adapter
# OPC-UA协议适配器 - 工业自动化标准协议

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict
import threading
import queue
import time
import struct

logger = logging.getLogger(__name__)


class OPCUAStatusCode(Enum):
    """OPC-UA状态码"""
    GOOD = 0x00000000
    UNCERTAIN = 0x40000000
    BAD = 0x80000000
    BAD_NOT_CONNECTED = 0x80080000
    BAD_TIMEOUT = 0x800A0000
    BAD_NODE_ID_UNKNOWN = 0x80340000
    BAD_ATTRIBUTE_ID_INVALID = 0x80350000
    BAD_DATA_TYPE_ID_UNKNOWN = 0x80120000
    BAD_OUT_OF_MEMORY = 0x80030000
    BAD_SERVER_NOT_CONNECTED = 0x800D0000
    BAD_SUBSCRIPTION_ID_INVALID = 0x80280000


class OPCUADataType(Enum):
    """OPC-UA数据类型"""
    BOOLEAN = "Boolean"
    SBYTE = "SByte"
    BYTE = "Byte"
    INT16 = "Int16"
    UINT16 = "UInt16"
    INT32 = "Int32"
    UINT32 = "UInt32"
    INT64 = "Int64"
    UINT64 = "UInt64"
    FLOAT = "Float"
    DOUBLE = "Double"
    STRING = "String"
    DATETIME = "DateTime"
    BYTE_STRING = "ByteString"


class OPCUANodeClass(Enum):
    """OPC-UA节点类型"""
    OBJECT = 1
    VARIABLE = 2
    METHOD = 4
    OBJECT_TYPE = 8
    VARIABLE_TYPE = 16
    REFERENCE_TYPE = 32
    DATA_TYPE = 64
    VIEW = 128


class OPCUASecurityMode(Enum):
    """OPC-UA安全模式"""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class OPCUASecurityPolicy(Enum):
    """OPC-UA安全策略"""
    NONE = "http://opcfoundation.org/UA/SecurityPolicy#None"
    BASIC128RSA15 = "http://opcfoundation.org/UA/SecurityPolicy#Basic128Rsa15"
    BASIC256 = "http://opcfoundation.org/UA/SecurityPolicy#Basic256"
    BASIC256SHA256 = "http://opcfoundation.org/UA/SecurityPolicy#Basic256Sha256"
    AES128SHA256RSAOAEP = "http://opcfoundation.org/UA/SecurityPolicy#Aes128_Sha256_RsaOaep"
    AES256SHA256RSAPSS = "http://opcfoundation.org/UA/SecurityPolicy#Aes256_Sha256_RsaPss"


@dataclass
class OPCUAConnectionConfig:
    """OPC-UA连接配置"""
    endpoint_url: str
    security_mode: OPCUASecurityMode = OPCUASecurityMode.NONE
    security_policy: OPCUASecurityPolicy = OPCUASecurityPolicy.NONE
    username: Optional[str] = None
    password: Optional[str] = None
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    application_name: str = "E2EControl-OPCUAClient"
    application_uri: str = "urn:e2econtrol:opcua:client"
    session_timeout: float = 3600.0  # seconds
    secure_channel_lifetime: float = 3600.0  # seconds
    request_timeout: float = 10.0  # seconds
    keepalive_interval: float = 30.0  # seconds
    reconnect_delay: float = 5.0  # seconds
    max_reconnect_attempts: int = 10


@dataclass
class OPCUANode:
    """OPC-UA节点定义"""
    node_id: str
    namespace: int = 2
    browse_name: str = ""
    display_name: str = ""
    node_class: OPCUANodeClass = OPCUANodeClass.VARIABLE
    data_type: OPCUADataType = OPCUADataType.DOUBLE
    description: str = ""
    writable: bool = False

    # Water network specific metadata
    pool_id: Optional[int] = None
    measurement_type: Optional[str] = None  # water_level, flow_rate, gate_position, etc.
    unit: Optional[str] = None  # m, m³/s, %, etc.
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        if not self.browse_name:
            self.browse_name = self.node_id.split('.')[-1] if '.' in self.node_id else self.node_id
        if not self.display_name:
            self.display_name = self.browse_name

    @property
    def full_node_id(self) -> str:
        """获取完整的节点ID字符串"""
        return f"ns={self.namespace};s={self.node_id}"


@dataclass
class OPCUADataPoint:
    """OPC-UA数据点"""
    node: OPCUANode
    value: Any
    status_code: OPCUAStatusCode
    source_timestamp: datetime
    server_timestamp: datetime

    @property
    def is_good(self) -> bool:
        """检查数据质量是否良好"""
        return self.status_code == OPCUAStatusCode.GOOD

    @property
    def quality(self) -> str:
        """获取数据质量描述"""
        if self.status_code == OPCUAStatusCode.GOOD:
            return "good"
        elif self.status_code.value & 0x40000000:
            return "uncertain"
        else:
            return "bad"


@dataclass
class OPCUASubscription:
    """OPC-UA订阅"""
    subscription_id: int
    publishing_interval: float = 1000.0  # milliseconds
    lifetime_count: int = 10000
    max_keepalive_count: int = 10
    max_notifications_per_publish: int = 10000
    priority: int = 0
    nodes: List[OPCUANode] = field(default_factory=list)
    callbacks: Dict[str, List[Callable]] = field(default_factory=lambda: defaultdict(list))
    enabled: bool = True

    def add_node(self, node: OPCUANode):
        """添加监控节点"""
        if node not in self.nodes:
            self.nodes.append(node)

    def add_callback(self, node_id: str, callback: Callable):
        """添加数据变化回调"""
        self.callbacks[node_id].append(callback)

    def remove_callback(self, node_id: str, callback: Callable):
        """移除数据变化回调"""
        if callback in self.callbacks[node_id]:
            self.callbacks[node_id].remove(callback)


class OPCUAAdapter:
    """OPC-UA协议适配器"""

    def __init__(self, config: OPCUAConnectionConfig):
        self.config = config
        self.connected = False
        self.session_id: Optional[str] = None
        self.subscriptions: Dict[int, OPCUASubscription] = {}
        self.node_cache: Dict[str, OPCUANode] = {}
        self.data_cache: Dict[str, OPCUADataPoint] = {}

        # Internal state
        self._next_subscription_id = 1
        self._lock = threading.RLock()
        self._data_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._reconnect_count = 0

        # Callbacks
        self._connection_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []

        # Statistics
        self.stats = {
            'reads': 0,
            'writes': 0,
            'notifications': 0,
            'errors': 0,
            'reconnects': 0,
            'last_read_time': None,
            'last_write_time': None,
            'avg_read_latency_ms': 0.0,
        }

        logger.info(f"OPC-UA adapter created for endpoint: {config.endpoint_url}")

    def connect(self) -> bool:
        """连接到OPC-UA服务器"""
        with self._lock:
            if self.connected:
                logger.warning("Already connected to OPC-UA server")
                return True

            try:
                logger.info(f"Connecting to OPC-UA server: {self.config.endpoint_url}")

                # Simulate connection handshake
                self._perform_handshake()

                # Create session
                self.session_id = f"session-{int(time.time() * 1000)}"
                self.connected = True
                self._reconnect_count = 0

                # Start background worker
                self._start_worker()

                # Notify callbacks
                for callback in self._connection_callbacks:
                    try:
                        callback(True, None)
                    except Exception as e:
                        logger.error(f"Connection callback error: {e}")

                logger.info(f"Connected to OPC-UA server, session: {self.session_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to connect to OPC-UA server: {e}")
                self.stats['errors'] += 1

                for callback in self._error_callbacks:
                    try:
                        callback(e)
                    except Exception as ce:
                        logger.error(f"Error callback failed: {ce}")

                return False

    def disconnect(self):
        """断开与OPC-UA服务器的连接"""
        with self._lock:
            if not self.connected:
                return

            logger.info("Disconnecting from OPC-UA server")

            # Stop worker
            self._stop_worker()

            # Clear subscriptions
            self.subscriptions.clear()

            # Clear session
            self.session_id = None
            self.connected = False

            # Notify callbacks
            for callback in self._connection_callbacks:
                try:
                    callback(False, None)
                except Exception as e:
                    logger.error(f"Disconnection callback error: {e}")

            logger.info("Disconnected from OPC-UA server")

    def _perform_handshake(self):
        """执行OPC-UA握手协议"""
        # Hello message
        logger.debug("Sending Hello message")
        time.sleep(0.01)  # Simulate network latency

        # Open secure channel
        logger.debug("Opening secure channel")
        time.sleep(0.01)

        # Create session
        logger.debug("Creating session")
        time.sleep(0.01)

        # Activate session
        logger.debug("Activating session")
        time.sleep(0.01)

    def _start_worker(self):
        """启动后台工作线程"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _stop_worker(self):
        """停止后台工作线程"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None

    def _worker_loop(self):
        """后台工作循环 - 处理订阅和发布"""
        last_keepalive = time.time()

        while self._running:
            try:
                current_time = time.time()

                # Send keepalive
                if current_time - last_keepalive > self.config.keepalive_interval:
                    self._send_keepalive()
                    last_keepalive = current_time

                # Process subscriptions
                self._process_subscriptions()

                # Process data queue
                self._process_data_queue()

                time.sleep(0.1)  # 100ms tick

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                self.stats['errors'] += 1

    def _send_keepalive(self):
        """发送保活消息"""
        if self.connected:
            logger.debug("Sending keepalive")

    def _process_subscriptions(self):
        """处理订阅的数据更新"""
        for sub_id, subscription in self.subscriptions.items():
            if not subscription.enabled:
                continue

            # Simulate data changes
            for node in subscription.nodes:
                # Get simulated data
                data_point = self._simulate_data_read(node)

                # Update cache
                self.data_cache[node.node_id] = data_point

                # Check for callbacks
                if node.node_id in subscription.callbacks:
                    for callback in subscription.callbacks[node.node_id]:
                        try:
                            callback(data_point)
                        except Exception as e:
                            logger.error(f"Subscription callback error: {e}")

                self.stats['notifications'] += 1

    def _process_data_queue(self):
        """处理数据队列"""
        while not self._data_queue.empty():
            try:
                data = self._data_queue.get_nowait()
                # Process queued data
            except queue.Empty:
                break

    def _simulate_data_read(self, node: OPCUANode) -> OPCUADataPoint:
        """模拟读取节点数据 (用于测试)"""
        import random

        now = datetime.now()

        # Generate value based on measurement type
        if node.measurement_type == 'water_level':
            value = random.uniform(1.0, 5.0)
        elif node.measurement_type == 'flow_rate':
            value = random.uniform(0.0, 10.0)
        elif node.measurement_type == 'gate_position':
            value = random.uniform(0.0, 100.0)
        elif node.measurement_type == 'pressure':
            value = random.uniform(0.0, 200.0)
        elif node.measurement_type == 'temperature':
            value = random.uniform(10.0, 30.0)
        elif node.data_type == OPCUADataType.BOOLEAN:
            value = random.choice([True, False])
        elif node.data_type == OPCUADataType.STRING:
            value = f"value_{int(time.time())}"
        else:
            value = random.uniform(0.0, 100.0)

        return OPCUADataPoint(
            node=node,
            value=value,
            status_code=OPCUAStatusCode.GOOD,
            source_timestamp=now,
            server_timestamp=now,
        )

    def register_node(self, node: OPCUANode):
        """注册OPC-UA节点"""
        with self._lock:
            self.node_cache[node.node_id] = node
            logger.debug(f"Registered node: {node.node_id}")

    def register_nodes(self, nodes: List[OPCUANode]):
        """批量注册OPC-UA节点"""
        for node in nodes:
            self.register_node(node)

    def read(self, node_id: str) -> Optional[OPCUADataPoint]:
        """读取单个节点的值"""
        return self.read_multiple([node_id]).get(node_id)

    def read_multiple(self, node_ids: List[str]) -> Dict[str, OPCUADataPoint]:
        """批量读取多个节点的值"""
        if not self.connected:
            logger.error("Not connected to OPC-UA server")
            return {}

        results = {}
        start_time = time.time()

        for node_id in node_ids:
            try:
                node = self.node_cache.get(node_id)
                if not node:
                    logger.warning(f"Node not registered: {node_id}")
                    continue

                # Get data (from cache or simulate)
                if node_id in self.data_cache:
                    data_point = self.data_cache[node_id]
                else:
                    data_point = self._simulate_data_read(node)
                    self.data_cache[node_id] = data_point

                results[node_id] = data_point
                self.stats['reads'] += 1

            except Exception as e:
                logger.error(f"Failed to read node {node_id}: {e}")
                self.stats['errors'] += 1

        # Update statistics
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['last_read_time'] = datetime.now()
        self.stats['avg_read_latency_ms'] = (
            self.stats['avg_read_latency_ms'] * 0.9 + elapsed_ms * 0.1
        )

        return results

    def write(self, node_id: str, value: Any) -> bool:
        """写入单个节点的值"""
        return self.write_multiple({node_id: value}).get(node_id, False)

    def write_multiple(self, values: Dict[str, Any]) -> Dict[str, bool]:
        """批量写入多个节点的值"""
        if not self.connected:
            logger.error("Not connected to OPC-UA server")
            return {k: False for k in values}

        results = {}

        for node_id, value in values.items():
            try:
                node = self.node_cache.get(node_id)
                if not node:
                    logger.warning(f"Node not registered: {node_id}")
                    results[node_id] = False
                    continue

                if not node.writable:
                    logger.warning(f"Node is not writable: {node_id}")
                    results[node_id] = False
                    continue

                # Update cache with new value
                now = datetime.now()
                self.data_cache[node_id] = OPCUADataPoint(
                    node=node,
                    value=value,
                    status_code=OPCUAStatusCode.GOOD,
                    source_timestamp=now,
                    server_timestamp=now,
                )

                results[node_id] = True
                self.stats['writes'] += 1
                self.stats['last_write_time'] = now

                logger.debug(f"Written to node {node_id}: {value}")

            except Exception as e:
                logger.error(f"Failed to write node {node_id}: {e}")
                results[node_id] = False
                self.stats['errors'] += 1

        return results

    def create_subscription(
        self,
        publishing_interval: float = 1000.0,
        nodes: Optional[List[OPCUANode]] = None,
    ) -> OPCUASubscription:
        """创建订阅"""
        with self._lock:
            subscription = OPCUASubscription(
                subscription_id=self._next_subscription_id,
                publishing_interval=publishing_interval,
                nodes=nodes or [],
            )

            self._next_subscription_id += 1
            self.subscriptions[subscription.subscription_id] = subscription

            # Register nodes
            for node in subscription.nodes:
                self.register_node(node)

            logger.info(f"Created subscription {subscription.subscription_id} with {len(subscription.nodes)} nodes")
            return subscription

    def delete_subscription(self, subscription_id: int) -> bool:
        """删除订阅"""
        with self._lock:
            if subscription_id in self.subscriptions:
                del self.subscriptions[subscription_id]
                logger.info(f"Deleted subscription {subscription_id}")
                return True
            return False

    def add_nodes_to_subscription(
        self,
        subscription_id: int,
        nodes: List[OPCUANode],
    ) -> bool:
        """向订阅添加节点"""
        with self._lock:
            subscription = self.subscriptions.get(subscription_id)
            if not subscription:
                logger.error(f"Subscription not found: {subscription_id}")
                return False

            for node in nodes:
                subscription.add_node(node)
                self.register_node(node)

            logger.info(f"Added {len(nodes)} nodes to subscription {subscription_id}")
            return True

    def subscribe_to_changes(
        self,
        node_id: str,
        callback: Callable[[OPCUADataPoint], None],
        subscription_id: Optional[int] = None,
    ):
        """订阅节点数据变化"""
        with self._lock:
            if subscription_id is None:
                # Use first subscription or create new one
                if not self.subscriptions:
                    node = self.node_cache.get(node_id)
                    if node:
                        self.create_subscription(nodes=[node])
                subscription_id = next(iter(self.subscriptions.keys()), None)

            if subscription_id is None:
                logger.error("No subscription available")
                return

            subscription = self.subscriptions.get(subscription_id)
            if subscription:
                subscription.add_callback(node_id, callback)
                logger.debug(f"Added callback for {node_id} in subscription {subscription_id}")

    def on_connect(self, callback: Callable[[bool, Optional[str]], None]):
        """注册连接状态回调"""
        self._connection_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]):
        """注册错误回调"""
        self._error_callbacks.append(callback)

    def browse(self, parent_node_id: Optional[str] = None) -> List[OPCUANode]:
        """浏览服务器节点树"""
        if not self.connected:
            logger.error("Not connected to OPC-UA server")
            return []

        # Return registered nodes as browse result
        if parent_node_id is None:
            return list(self.node_cache.values())

        # Filter by parent
        prefix = f"{parent_node_id}."
        return [
            node for node in self.node_cache.values()
            if node.node_id.startswith(prefix)
        ]

    def get_status(self) -> Dict[str, Any]:
        """获取适配器状态"""
        return {
            'connected': self.connected,
            'endpoint': self.config.endpoint_url,
            'session_id': self.session_id,
            'subscription_count': len(self.subscriptions),
            'node_count': len(self.node_cache),
            'statistics': self.stats.copy(),
        }

    def create_water_network_nodes(
        self,
        num_pools: int,
        namespace: int = 2,
    ) -> List[OPCUANode]:
        """创建智能水网标准节点结构"""
        nodes = []

        for pool_id in range(num_pools):
            pool_prefix = f"WaterNetwork.Pool{pool_id}"

            # Water level sensor
            nodes.append(OPCUANode(
                node_id=f"{pool_prefix}.WaterLevel",
                namespace=namespace,
                display_name=f"Pool {pool_id} Water Level",
                data_type=OPCUADataType.DOUBLE,
                measurement_type='water_level',
                unit='m',
                pool_id=pool_id,
                min_value=0.0,
                max_value=10.0,
            ))

            # Inflow rate sensor
            nodes.append(OPCUANode(
                node_id=f"{pool_prefix}.InflowRate",
                namespace=namespace,
                display_name=f"Pool {pool_id} Inflow Rate",
                data_type=OPCUADataType.DOUBLE,
                measurement_type='flow_rate',
                unit='m³/s',
                pool_id=pool_id,
                min_value=0.0,
                max_value=50.0,
            ))

            # Outflow rate sensor
            nodes.append(OPCUANode(
                node_id=f"{pool_prefix}.OutflowRate",
                namespace=namespace,
                display_name=f"Pool {pool_id} Outflow Rate",
                data_type=OPCUADataType.DOUBLE,
                measurement_type='flow_rate',
                unit='m³/s',
                pool_id=pool_id,
                min_value=0.0,
                max_value=50.0,
            ))

            # Gate position (writable)
            nodes.append(OPCUANode(
                node_id=f"{pool_prefix}.GatePosition",
                namespace=namespace,
                display_name=f"Pool {pool_id} Gate Position",
                data_type=OPCUADataType.DOUBLE,
                measurement_type='gate_position',
                unit='%',
                pool_id=pool_id,
                min_value=0.0,
                max_value=100.0,
                writable=True,
            ))

            # Temperature sensor
            nodes.append(OPCUANode(
                node_id=f"{pool_prefix}.Temperature",
                namespace=namespace,
                display_name=f"Pool {pool_id} Temperature",
                data_type=OPCUADataType.DOUBLE,
                measurement_type='temperature',
                unit='°C',
                pool_id=pool_id,
                min_value=-10.0,
                max_value=50.0,
            ))

            # Pressure sensor
            nodes.append(OPCUANode(
                node_id=f"{pool_prefix}.Pressure",
                namespace=namespace,
                display_name=f"Pool {pool_id} Pressure",
                data_type=OPCUADataType.DOUBLE,
                measurement_type='pressure',
                unit='kPa',
                pool_id=pool_id,
                min_value=0.0,
                max_value=500.0,
            ))

            # Pump status
            nodes.append(OPCUANode(
                node_id=f"{pool_prefix}.PumpStatus",
                namespace=namespace,
                display_name=f"Pool {pool_id} Pump Status",
                data_type=OPCUADataType.BOOLEAN,
                measurement_type='pump_status',
                pool_id=pool_id,
            ))

            # Alarm status
            nodes.append(OPCUANode(
                node_id=f"{pool_prefix}.AlarmActive",
                namespace=namespace,
                display_name=f"Pool {pool_id} Alarm Active",
                data_type=OPCUADataType.BOOLEAN,
                measurement_type='alarm_status',
                pool_id=pool_id,
            ))

        # System-level nodes
        nodes.append(OPCUANode(
            node_id="WaterNetwork.SystemStatus",
            namespace=namespace,
            display_name="System Status",
            data_type=OPCUADataType.STRING,
        ))

        nodes.append(OPCUANode(
            node_id="WaterNetwork.TotalFlow",
            namespace=namespace,
            display_name="Total Network Flow",
            data_type=OPCUADataType.DOUBLE,
            measurement_type='flow_rate',
            unit='m³/s',
        ))

        nodes.append(OPCUANode(
            node_id="WaterNetwork.EmergencyStop",
            namespace=namespace,
            display_name="Emergency Stop",
            data_type=OPCUADataType.BOOLEAN,
            writable=True,
        ))

        # Register all nodes
        self.register_nodes(nodes)

        logger.info(f"Created {len(nodes)} water network OPC-UA nodes for {num_pools} pools")
        return nodes
