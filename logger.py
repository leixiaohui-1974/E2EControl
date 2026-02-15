"""
日志系统模块
提供结构化日志记录功能
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """结构化日志格式器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录
        
        Args:
            record: 日志记录
            
        Returns:
            格式化后的字符串
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # 添加额外字段
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # 控制台输出使用可读格式
        if record.levelno >= logging.WARNING:
            return f"[{log_data['timestamp']}] {log_data['level']} - {log_data['message']}"
        else:
            return f"[{log_data['timestamp']}] {log_data['level']} - {log_data['module']}.{log_data['function']} - {log_data['message']}"


class SmartPoolLogger:
    """智能闸门日志器"""
    
    _instance: Optional['SmartPoolLogger'] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化日志器"""
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = logging.getLogger('SmartPool')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # 清除现有处理器
        self.logger.handlers.clear()
    
    def setup(self, 
              level: str = "INFO",
              log_file: Optional[str] = None,
              console_output: bool = True,
              max_bytes: int = 10485760,
              backup_count: int = 5):
        """
        配置日志系统
        
        Args:
            level: 日志级别
            log_file: 日志文件路径
            console_output: 是否输出到控制台
            max_bytes: 日志文件最大字节数
            backup_count: 备份文件数量
        """
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 设置日志级别
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        formatter = StructuredFormatter()
        
        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, *args, **kwargs):
        """调试日志"""
        self.logger.debug(message, *args, extra={'extra_data': kwargs})

    def info(self, message: str, *args, **kwargs):
        """信息日志"""
        self.logger.info(message, *args, extra={'extra_data': kwargs})

    def warning(self, message: str, *args, **kwargs):
        """警告日志"""
        self.logger.warning(message, *args, extra={'extra_data': kwargs})

    def error(self, message: str, *args, **kwargs):
        """错误日志"""
        self.logger.error(message, *args, extra={'extra_data': kwargs})

    def critical(self, message: str, *args, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, *args, extra={'extra_data': kwargs})
    
    def log_control_action(self, time_step: int, level: float, q_in: float, 
                          q_out: float, config: dict):
        """
        记录控制动作
        
        Args:
            time_step: 时间步
            level: 当前水位
            q_in: 入流
            q_out: 出流
            config: 控制配置
        """
        self.info(
            f"控制动作 [T={time_step}h]: 水位={level:.2f}m, 入流={q_in:.2f}m³/s, 出流={q_out:.2f}m³/s",
            time_step=time_step,
            level=level,
            q_in=q_in,
            q_out=q_out,
            target_level=config.get('Z_ref', 0)
        )
    
    def log_scenario_change(self, time_step: int, instruction: str, config: dict):
        """
        记录场景切换
        
        Args:
            time_step: 时间步
            instruction: 指令文本
            config: 新配置
        """
        self.info(
            f"场景切换 [T={time_step}h]: {instruction}",
            time_step=time_step,
            instruction=instruction,
            W_level=config.get('W_level'),
            Z_ref=config.get('Z_ref'),
            delta_Q_max=config.get('delta_Q_max')
        )
    
    def log_optimization_result(self, status: str, cost: Optional[float] = None,
                               solve_time: Optional[float] = None):
        """
        记录优化结果
        
        Args:
            status: 求解状态
            cost: 目标函数值
            solve_time: 求解时间
        """
        if status == "optimal":
            if cost is not None and solve_time is not None:
                self.debug(
                    f"优化成功: cost={cost:.4f}, time={solve_time:.3f}s",
                    status=status,
                    cost=cost,
                    solve_time=solve_time
                )
            elif solve_time is not None:
                self.debug(
                    f"优化成功: time={solve_time:.3f}s",
                    status=status,
                    solve_time=solve_time
                )
            else:
                self.debug(
                    f"优化成功",
                    status=status
                )
        else:
            self.warning(
                f"优化状态异常: {status}",
                status=status
            )
    
    def log_alert(self, alert_type: str, message: str, level: float = None):
        """
        记录告警
        
        Args:
            alert_type: 告警类型
            message: 告警信息
            level: 相关水位
        """
        self.warning(
            f"⚠️  告警 [{alert_type}]: {message}",
            alert_type=alert_type,
            level=level
        )


# 全局日志器实例
_logger_instance: Optional[SmartPoolLogger] = None


def get_logger() -> SmartPoolLogger:
    """获取全局日志器实例"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SmartPoolLogger()
    return _logger_instance


def setup_logging(config: dict):
    """
    从配置设置日志
    
    Args:
        config: 日志配置字典
    """
    logger = get_logger()
    logger.setup(
        level=config.get('level', 'INFO'),
        log_file=config.get('file'),
        console_output=config.get('console_output', True),
        max_bytes=config.get('max_bytes', 10485760),
        backup_count=config.get('backup_count', 5)
    )


if __name__ == "__main__":
    # 测试日志系统
    logger = get_logger()
    logger.setup(level="DEBUG", console_output=True)
    
    logger.debug("这是调试信息")
    logger.info("系统启动")
    logger.warning("水位接近上限", level=8.5)
    logger.error("优化求解失败", reason="infeasible")
    logger.log_control_action(10, 3.5, 5.2, 4.8, {'Z_ref': 3.0})
    logger.log_scenario_change(20, "进入冰期模式", {'W_smooth': 500.0})
