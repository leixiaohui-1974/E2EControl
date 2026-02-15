"""
降级运行模式管理
Degraded Mode Management
"""

import sys
import logging

logger = logging.getLogger(__name__)
sys.path.append('..')

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


class OperationMode(Enum):
    """运行模式"""
    NORMAL = "正常模式"
    DEGRADED_MINOR = "轻度降级"
    DEGRADED_MODERATE = "中度降级"
    DEGRADED_SEVERE = "重度降级"
    EMERGENCY = "应急模式"
    SAFE_MODE = "安全模式"
    MANUAL = "手动模式"


@dataclass
class ModeConfig:
    """模式配置"""
    mode: OperationMode
    control_precision: float  # 控制精度 [0-1]
    response_time: float  # 响应时间 [s]
    safety_margin: float  # 安全裕度
    max_flow_change: float  # 最大流量变化 [m³/s]
    update_frequency: float  # 更新频率 [Hz]
    enable_optimization: bool  # 是否启用优化
    enable_feedforward: bool  # 是否启用前馈
    enable_coordination: bool  # 是否启用协调控制
    constraints_tightness: float  # 约束紧度 [0-1]
    description: str = ""


class DegradedModeManager:
    """
    降级运行模式管理器
    
    职责：
    1. 根据系统健康状态选择运行模式
    2. 动态调整控制参数
    3. 管理功能降级策略
    4. 确保系统安全运行
    """
    
    def __init__(self):
        """初始化降级模式管理器"""
        self.current_mode = OperationMode.NORMAL
        
        # 定义各模式配置
        self.mode_configs = self._define_mode_configs()
        
        # 系统健康指标
        self.system_health = {
            'sensor_availability': 1.0,  # 传感器可用率
            'actuator_availability': 1.0,  # 执行器可用率
            'controller_availability': 1.0,  # 控制器可用率
            'communication_quality': 1.0,  # 通信质量
            'power_stability': 1.0  # 电源稳定性
        }
        
        # 模式切换历史
        self.mode_history = []
        
    def _define_mode_configs(self) -> Dict[OperationMode, ModeConfig]:
        """定义各运行模式的配置"""
        configs = {}
        
        # 正常模式
        configs[OperationMode.NORMAL] = ModeConfig(
            mode=OperationMode.NORMAL,
            control_precision=1.0,
            response_time=60.0,
            safety_margin=0.5,
            max_flow_change=10.0,
            update_frequency=1/60.0,  # 每分钟
            enable_optimization=True,
            enable_feedforward=True,
            enable_coordination=True,
            constraints_tightness=0.8,
            description="所有功能正常，最优控制"
        )
        
        # 轻度降级
        configs[OperationMode.DEGRADED_MINOR] = ModeConfig(
            mode=OperationMode.DEGRADED_MINOR,
            control_precision=0.9,
            response_time=90.0,
            safety_margin=0.6,
            max_flow_change=8.0,
            update_frequency=1/90.0,
            enable_optimization=True,
            enable_feedforward=True,
            enable_coordination=True,
            constraints_tightness=0.9,
            description="少量传感器/执行器失效，略微降低性能"
        )
        
        # 中度降级
        configs[OperationMode.DEGRADED_MODERATE] = ModeConfig(
            mode=OperationMode.DEGRADED_MODERATE,
            control_precision=0.75,
            response_time=120.0,
            safety_margin=0.8,
            max_flow_change=5.0,
            update_frequency=1/120.0,
            enable_optimization=True,
            enable_feedforward=False,  # 关闭前馈
            enable_coordination=True,
            constraints_tightness=1.0,
            description="多个组件失效，显著降低性能，增强安全"
        )
        
        # 重度降级
        configs[OperationMode.DEGRADED_SEVERE] = ModeConfig(
            mode=OperationMode.DEGRADED_SEVERE,
            control_precision=0.5,
            response_time=180.0,
            safety_margin=1.0,
            max_flow_change=3.0,
            update_frequency=1/180.0,
            enable_optimization=False,  # 关闭优化
            enable_feedforward=False,
            enable_coordination=False,  # 关闭协调
            constraints_tightness=1.2,
            description="严重故障，最小功能运行，优先安全"
        )
        
        # 应急模式
        configs[OperationMode.EMERGENCY] = ModeConfig(
            mode=OperationMode.EMERGENCY,
            control_precision=0.3,
            response_time=300.0,
            safety_margin=1.5,
            max_flow_change=2.0,
            update_frequency=1/300.0,
            enable_optimization=False,
            enable_feedforward=False,
            enable_coordination=False,
            constraints_tightness=1.5,
            description="紧急情况，保持最小流量，防止事故"
        )
        
        # 安全模式
        configs[OperationMode.SAFE_MODE] = ModeConfig(
            mode=OperationMode.SAFE_MODE,
            control_precision=0.1,
            response_time=600.0,
            safety_margin=2.0,
            max_flow_change=1.0,
            update_frequency=1/600.0,
            enable_optimization=False,
            enable_feedforward=False,
            enable_coordination=False,
            constraints_tightness=2.0,
            description="安全模式，锁定关键参数，等待人工干预"
        )
        
        # 手动模式
        configs[OperationMode.MANUAL] = ModeConfig(
            mode=OperationMode.MANUAL,
            control_precision=0.0,
            response_time=float('inf'),
            safety_margin=2.5,
            max_flow_change=0.5,
            update_frequency=0.0,
            enable_optimization=False,
            enable_feedforward=False,
            enable_coordination=False,
            constraints_tightness=2.5,
            description="手动模式，完全由人工控制"
        )
        
        return configs
    
    def update_system_health(self, health_metrics: Dict[str, float]):
        """
        更新系统健康指标
        
        Args:
            health_metrics: 健康指标字典
        """
        for key, value in health_metrics.items():
            if key in self.system_health:
                self.system_health[key] = np.clip(value, 0.0, 1.0)
    
    def calculate_overall_health(self) -> float:
        """计算系统总体健康度"""
        weights = {
            'sensor_availability': 0.25,
            'actuator_availability': 0.30,
            'controller_availability': 0.25,
            'communication_quality': 0.10,
            'power_stability': 0.10
        }
        
        overall_health = sum(
            self.system_health[key] * weights[key]
            for key in weights.keys()
        )
        
        return overall_health
    
    def determine_operation_mode(self, force_mode: Optional[OperationMode] = None) -> OperationMode:
        """
        确定运行模式
        
        Args:
            force_mode: 强制模式（用于手动切换）
            
        Returns:
            推荐的运行模式
        """
        if force_mode:
            return force_mode
        
        # 计算总体健康度
        overall_health = self.calculate_overall_health()
        
        # 根据健康度确定模式
        if overall_health >= 0.95:
            return OperationMode.NORMAL
        elif overall_health >= 0.85:
            return OperationMode.DEGRADED_MINOR
        elif overall_health >= 0.70:
            return OperationMode.DEGRADED_MODERATE
        elif overall_health >= 0.50:
            return OperationMode.DEGRADED_SEVERE
        elif overall_health >= 0.30:
            return OperationMode.EMERGENCY
        else:
            return OperationMode.SAFE_MODE
    
    def switch_mode(self, new_mode: OperationMode, reason: str = "") -> Dict:
        """
        切换运行模式
        
        Args:
            new_mode: 新模式
            reason: 切换原因
            
        Returns:
            切换结果
        """
        old_mode = self.current_mode
        
        if old_mode == new_mode:
            return {
                'success': True,
                'message': f'已处于{new_mode.value}',
                'mode_changed': False
            }
        
        # 记录切换历史
        self.mode_history.append({
            'from': old_mode,
            'to': new_mode,
            'reason': reason,
            'health': self.calculate_overall_health()
        })
        
        # 执行模式切换
        self.current_mode = new_mode
        
        # 获取新模式配置
        config = self.mode_configs[new_mode]
        
        return {
            'success': True,
            'message': f'模式切换: {old_mode.value} → {new_mode.value}',
            'mode_changed': True,
            'old_mode': old_mode,
            'new_mode': new_mode,
            'config': config,
            'reason': reason
        }
    
    def get_current_config(self) -> ModeConfig:
        """获取当前模式配置"""
        return self.mode_configs[self.current_mode]
    
    def get_control_adjustments(self) -> Dict:
        """
        获取控制参数调整建议
        
        Returns:
            参数调整字典
        """
        config = self.get_current_config()
        
        return {
            'horizon': int(10 * config.control_precision),  # 预测时域
            'update_interval': 1.0 / config.update_frequency if config.update_frequency > 0 else 600.0,
            'weight_tracking': 1.0 * config.control_precision,
            'weight_safety': 2.0 * (1.0 - config.control_precision),
            'max_flow_change': config.max_flow_change,
            'safety_margin': config.safety_margin,
            'enable_feedforward': config.enable_feedforward,
            'enable_coordination': config.enable_coordination,
            'constraint_penalty': config.constraints_tightness
        }
    
    def auto_adjust(self) -> Dict:
        """
        自动调整模式
        
        Returns:
            调整结果
        """
        # 确定推荐模式
        recommended_mode = self.determine_operation_mode()
        
        # 如果需要切换
        if recommended_mode != self.current_mode:
            overall_health = self.calculate_overall_health()
            reason = f"系统健康度={overall_health:.2%}, 自动调整"
            
            return self.switch_mode(recommended_mode, reason)
        
        return {
            'success': True,
            'message': '当前模式适合，无需调整',
            'mode_changed': False
        }
    
    def get_status(self) -> Dict:
        """获取状态信息"""
        config = self.get_current_config()
        
        return {
            'current_mode': self.current_mode.value,
            'overall_health': self.calculate_overall_health(),
            'system_health': self.system_health.copy(),
            'control_precision': config.control_precision,
            'safety_margin': config.safety_margin,
            'enabled_features': {
                'optimization': config.enable_optimization,
                'feedforward': config.enable_feedforward,
                'coordination': config.enable_coordination
            },
            'mode_switches': len(self.mode_history)
        }


# 演示
if __name__ == "__main__":
    logger.info("="*80)
    logger.info(" "*20 + "降级运行模式演示")
    logger.info("="*80)
    
    # 创建降级模式管理器
    manager = DegradedModeManager()
    
    logger.info(f"\n初始状态:")
    status = manager.get_status()
    logger.info(f"  当前模式: {status['current_mode']}")
    logger.info(f"  系统健康: {status['overall_health']:.2%}")
    
    # 模拟健康度逐渐下降
    scenarios = [
        (0.90, "传感器1失效"),
        (0.75, "传感器2失效 + 通信质量下降"),
        (0.55, "执行器1故障"),
        (0.35, "控制器部分失效"),
        (0.20, "多重故障"),
    ]
    
    for health, description in scenarios:
        logger.info(f"\n{'='*80}")
        logger.info(f"场景: {description}")
        logger.info('='*80)
        
        # 更新健康指标（简化：统一更新所有指标）
        manager.update_system_health({
            'sensor_availability': health,
            'actuator_availability': health,
            'controller_availability': health + 0.1,
            'communication_quality': health + 0.05,
            'power_stability': 0.95
        })
        
        # 自动调整模式
        result = manager.auto_adjust()
        
        if result['mode_changed']:
            logger.info(f"\n✓ {result['message']}")
            logger.info(f"  原因: {result['reason']}")
            
            config = result['config']
            logger.info(f"\n新模式配置:")
            logger.info(f"  控制精度: {config.control_precision:.2%}")
            logger.info(f"  安全裕度: {config.safety_margin}")
            logger.info(f"  最大流量变化: {config.max_flow_change} m³/s")
            logger.info(f"  优化: {'启用' if config.enable_optimization else '禁用'}")
            logger.info(f"  前馈: {'启用' if config.enable_feedforward else '禁用'}")
            logger.info(f"  协调: {'启用' if config.enable_coordination else '禁用'}")
            logger.info(f"  描述: {config.description}")
        else:
            logger.info(f"\n  {result['message']}")
    
    # 恢复场景
    logger.info(f"\n{'='*80}")
    logger.info("场景: 故障修复，系统恢复")
    logger.info('='*80)
    
    manager.update_system_health({
        'sensor_availability': 0.98,
        'actuator_availability': 0.95,
        'controller_availability': 1.0,
        'communication_quality': 0.95,
        'power_stability': 1.0
    })
    
    result = manager.auto_adjust()
    logger.info(f"\n✓ {result['message']}")
    
    # 最终状态
    logger.info(f"\n{'='*80}")
    logger.info("最终状态")
    logger.info('='*80)
    
    status = manager.get_status()
    logger.info(f"  当前模式: {status['current_mode']}")
    logger.info(f"  系统健康: {status['overall_health']:.2%}")
    logger.info(f"  模式切换次数: {status['mode_switches']}")
    
    logger.info("\n✅ 演示完成！")
    logger.info("="*80)
