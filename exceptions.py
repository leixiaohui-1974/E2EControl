"""
自定义异常类
定义系统中使用的所有异常类型
"""


class SmartPoolException(Exception):
    """智能闸门系统基础异常"""
    pass


class ConfigurationError(SmartPoolException):
    """配置错误"""
    pass


class OptimizationError(SmartPoolException):
    """优化求解错误"""
    
    def __init__(self, message: str, status: str = None, details: dict = None):
        """
        初始化优化错误
        
        Args:
            message: 错误信息
            status: 求解状态
            details: 详细信息
        """
        super().__init__(message)
        self.status = status
        self.details = details or {}


class PhysicsError(SmartPoolException):
    """物理模拟错误"""
    
    def __init__(self, message: str, state: dict = None):
        """
        初始化物理错误
        
        Args:
            message: 错误信息
            state: 当前状态
        """
        super().__init__(message)
        self.state = state or {}


class SemanticError(SmartPoolException):
    """语义解释错误"""
    
    def __init__(self, message: str, instruction: str = None):
        """
        初始化语义错误
        
        Args:
            message: 错误信息
            instruction: 原始指令
        """
        super().__init__(message)
        self.instruction = instruction


class ValidationError(SmartPoolException):
    """数据验证错误"""
    
    def __init__(self, message: str, field: str = None, value = None):
        """
        初始化验证错误
        
        Args:
            message: 错误信息
            field: 字段名
            value: 字段值
        """
        super().__init__(message)
        self.field = field
        self.value = value


class DatabaseError(SmartPoolException):
    """数据库操作错误"""
    pass


class MonitoringError(SmartPoolException):
    """监控系统错误"""
    pass
