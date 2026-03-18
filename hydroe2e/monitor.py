"""
监控告警系统
实时监控系统状态并触发告警
"""

from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hydroe2e.logger import get_logger
from hydroe2e.config_manager import get_config
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "信息"
    WARNING = "警告"
    CRITICAL = "严重"


@dataclass
class Alert:
    """告警记录"""
    timestamp: datetime
    level: AlertLevel
    alert_type: str
    message: str
    data: Dict
    
    def __str__(self):
        return f"[{self.level.value}] {self.timestamp.isoformat()} - {self.alert_type}: {self.message}"


class MonitoringSystem:
    """监控系统"""
    
    def __init__(self):
        """初始化监控系统"""
        self.config = get_config()
        self.logger = get_logger()
        
        # 加载监控配置
        mon_config = self.config.get_section('monitoring')
        self.level_warning_high = mon_config.get('level_warning_high', 8.0)
        self.level_warning_low = mon_config.get('level_warning_low', 1.0)
        self.level_critical_high = mon_config.get('level_critical_high', 9.5)
        self.level_critical_low = mon_config.get('level_critical_low', 0.5)
        self.flow_rate_max = mon_config.get('flow_rate_max', 25.0)
        
        # 告警历史
        self.alerts: List[Alert] = []
        
        # 告警回调
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 状态统计
        self.stats = {
            'total_alerts': 0,
            'warning_count': 0,
            'critical_count': 0,
            'last_check_time': None
        }
        
        self.logger.info("监控系统初始化完成")
    
    def check_state(self, time_step: int, level: float, q_in: float, 
                   q_out: float, config: Dict) -> List[Alert]:
        """
        检查系统状态并生成告警
        
        Args:
            time_step: 时间步
            level: 当前水位
            q_in: 入流
            q_out: 出流
            config: 控制配置
            
        Returns:
            告警列表
        """
        alerts = []
        self.stats['last_check_time'] = datetime.now()
        
        # 1. 水位检查
        if level >= self.level_critical_high:
            alert = self._create_alert(
                AlertLevel.CRITICAL,
                "水位严重偏高",
                f"当前水位 {level:.2f}m 达到严重告警阈值 {self.level_critical_high}m",
                {'time_step': time_step, 'level': level, 'threshold': self.level_critical_high}
            )
            alerts.append(alert)
            
        elif level >= self.level_warning_high:
            alert = self._create_alert(
                AlertLevel.WARNING,
                "水位偏高",
                f"当前水位 {level:.2f}m 超过警戒线 {self.level_warning_high}m",
                {'time_step': time_step, 'level': level, 'threshold': self.level_warning_high}
            )
            alerts.append(alert)
        
        if level <= self.level_critical_low:
            alert = self._create_alert(
                AlertLevel.CRITICAL,
                "水位严重偏低",
                f"当前水位 {level:.2f}m 低于严重告警阈值 {self.level_critical_low}m",
                {'time_step': time_step, 'level': level, 'threshold': self.level_critical_low}
            )
            alerts.append(alert)
            
        elif level <= self.level_warning_low:
            alert = self._create_alert(
                AlertLevel.WARNING,
                "水位偏低",
                f"当前水位 {level:.2f}m 低于警戒线 {self.level_warning_low}m",
                {'time_step': time_step, 'level': level, 'threshold': self.level_warning_low}
            )
            alerts.append(alert)
        
        # 2. 流量检查
        if q_in > self.flow_rate_max:
            alert = self._create_alert(
                AlertLevel.WARNING,
                "入流过大",
                f"入流 {q_in:.2f}m³/s 超过最大流量 {self.flow_rate_max}m³/s",
                {'time_step': time_step, 'q_in': q_in, 'max_flow': self.flow_rate_max}
            )
            alerts.append(alert)
        
        # 3. 水位偏差检查
        target_level = config.get('Z_ref', 3.0)
        deviation = abs(level - target_level)
        max_acceptable_deviation = 2.0  # 可配置
        
        if deviation > max_acceptable_deviation:
            alert = self._create_alert(
                AlertLevel.WARNING,
                "水位偏差过大",
                f"水位偏差 {deviation:.2f}m 超过可接受范围",
                {'time_step': time_step, 'level': level, 'target': target_level, 'deviation': deviation}
            )
            alerts.append(alert)
        
        # 4. 流量剧烈变化检查（需要历史数据）
        # 这里简化处理，实际应该从历史中获取
        
        # 处理告警
        for alert in alerts:
            self._handle_alert(alert)
        
        return alerts
    
    def _create_alert(self, level: AlertLevel, alert_type: str, 
                     message: str, data: Dict) -> Alert:
        """
        创建告警对象
        
        Args:
            level: 告警级别
            alert_type: 告警类型
            message: 告警信息
            data: 附加数据
            
        Returns:
            Alert对象
        """
        return Alert(
            timestamp=datetime.now(),
            level=level,
            alert_type=alert_type,
            message=message,
            data=data
        )
    
    def _handle_alert(self, alert: Alert):
        """
        处理告警
        
        Args:
            alert: 告警对象
        """
        # 记录告警
        self.alerts.append(alert)
        self.stats['total_alerts'] += 1
        
        if alert.level == AlertLevel.WARNING:
            self.stats['warning_count'] += 1
            self.logger.warning("WARNING: %s", alert.message)
        elif alert.level == AlertLevel.CRITICAL:
            self.stats['critical_count'] += 1
            self.logger.error("CRITICAL: %s", alert.message)
        else:
            self.logger.info("INFO: %s", alert.message)
        
        # 调用回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error("告警回调执行失败: %s", e)
    
    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        注册告警回调
        
        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)
        self.logger.info("注册告警回调: %s", callback.__name__)
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                   limit: Optional[int] = None) -> List[Alert]:
        """
        获取告警历史
        
        Args:
            level: 筛选告警级别
            limit: 限制返回数量
            
        Returns:
            告警列表
        """
        alerts = self.alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if limit:
            alerts = alerts[-limit:]
        
        return alerts
    
    def get_statistics(self) -> Dict:
        """
        获取监控统计信息
        
        Returns:
            统计字典
        """
        return self.stats.copy()
    
    def clear_alerts(self) -> None:
        """清除告警历史"""
        self.alerts.clear()
        self.logger.info("告警历史已清除")
    
    def generate_report(self) -> str:
        """
        生成监控报告
        
        Returns:
            报告文本
        """
        report = ["=== 监控系统报告 ===\n"]
        report.append(f"总告警数: {self.stats['total_alerts']}")
        report.append(f"警告数: {self.stats['warning_count']}")
        report.append(f"严重告警数: {self.stats['critical_count']}")
        report.append(f"\n最近告警:")
        
        recent_alerts = self.get_alerts(limit=10)
        for alert in recent_alerts:
            report.append(f"  - {alert}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # 测试监控系统
    from logger import setup_logging
    
    setup_logging({'level': 'INFO', 'console_output': True})
    
    monitor = MonitoringSystem()
    
    # 模拟监控检查
    test_cases = [
        (0, 3.0, 5.0, 5.0, {'Z_ref': 3.0}),  # 正常
        (10, 8.5, 10.0, 5.0, {'Z_ref': 3.0}),  # 水位偏高
        (20, 9.8, 15.0, 5.0, {'Z_ref': 3.0}),  # 水位严重偏高
        (30, 0.3, 2.0, 5.0, {'Z_ref': 3.0}),  # 水位严重偏低
        (40, 3.0, 30.0, 5.0, {'Z_ref': 3.0}),  # 流量过大
    ]
    
    logger.info("\n=== 监控系统测试 ===\n")
    for t, level, q_in, q_out, config in test_cases:
        logger.info("时间步 %dh: 水位=%.1fm, 入流=%.1fm³/s", t, level, q_in)
        alerts = monitor.check_state(t, level, q_in, q_out, config)
        logger.info("触发 %d 个告警", len(alerts))
    
    logger.info(monitor.generate_report())
