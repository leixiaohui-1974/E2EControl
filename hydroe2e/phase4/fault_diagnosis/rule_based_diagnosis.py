"""
基于规则的故障诊断系统
"""

import logging

logger = logging.getLogger(__name__)

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class FaultType(Enum):
    """故障类型"""
    SENSOR_FAULT = "sensor_fault"
    ACTUATOR_FAULT = "actuator_fault"
    CONTROLLER_FAULT = "controller_fault"
    PHYSICAL_FAULT = "physical_fault"
    UNKNOWN = "unknown"


@dataclass
class DiagnosisRule:
    """诊断规则"""
    name: str
    conditions: List[str]  # 症状列表
    fault_type: FaultType
    fault_location: str
    confidence: float
    description: str
    recommendations: List[str]


@dataclass
class DiagnosisReport:
    """诊断报告"""
    fault_type: FaultType
    fault_location: str
    confidence: float
    symptoms: List[str]
    root_cause: str
    recommendations: List[str]
    matched_rules: List[str]


class RuleBasedDiagnosisSystem:
    """基于规则的诊断系统"""
    
    def __init__(self):
        """初始化诊断系统"""
        self.rules: List[DiagnosisRule] = []
        self._build_rules()
    
    def _build_rules(self):
        """构建诊断规则库"""
        
        # 规则1: 出口闸门故障
        self.rules.append(DiagnosisRule(
            name="出口闸门卡死",
            conditions=["水位异常高", "入流正常", "出流异常低"],
            fault_type=FaultType.ACTUATOR_FAULT,
            fault_location="出口闸门",
            confidence=0.90,
            description="出口闸门可能卡死或响应不正常，导致出流受阻",
            recommendations=[
                "检查出口闸门机械状态",
                "检查闸门控制信号",
                "启用备用闸门",
                "降低入流减轻压力"
            ]
        ))
        
        # 规则2: 入口闸门故障
        self.rules.append(DiagnosisRule(
            name="入口闸门故障",
            conditions=["水位异常低", "入流异常低", "出流正常"],
            fault_type=FaultType.ACTUATOR_FAULT,
            fault_location="入口闸门",
            confidence=0.85,
            description="入口闸门可能故障，导致进水不足",
            recommendations=[
                "检查入口闸门状态",
                "检查上游供水",
                "切换备用水源",
                "降低出流保持水位"
            ]
        ))
        
        # 规则3: 水位传感器故障
        self.rules.append(DiagnosisRule(
            name="水位传感器故障",
            conditions=["水位读数异常", "流量平衡正常"],
            fault_type=FaultType.SENSOR_FAULT,
            fault_location="水位传感器",
            confidence=0.80,
            description="水位传感器可能故障，读数不准确",
            recommendations=[
                "切换到备用传感器",
                "进行传感器校准",
                "使用流量推算水位"
            ]
        ))
        
        # 规则4: 流量传感器故障
        self.rules.append(DiagnosisRule(
            name="流量传感器故障",
            conditions=["流量读数异常", "水位变化正常"],
            fault_type=FaultType.SENSOR_FAULT,
            fault_location="流量传感器",
            confidence=0.80,
            description="流量传感器可能故障",
            recommendations=[
                "切换到备用流量计",
                "检查传感器连接",
                "根据水位变化推算流量"
            ]
        ))
        
        # 规则5: 控制器故障
        self.rules.append(DiagnosisRule(
            name="控制器异常",
            conditions=["控制输出异常", "传感器正常", "执行器正常"],
            fault_type=FaultType.CONTROLLER_FAULT,
            fault_location="MPC控制器",
            confidence=0.75,
            description="控制器计算异常或通信故障",
            recommendations=[
                "切换到备用控制器",
                "检查控制算法",
                "使用PID后备控制"
            ]
        ))
        
        # 规则6: 管道泄漏
        self.rules.append(DiagnosisRule(
            name="管道泄漏",
            conditions=["水位持续下降", "入流正常", "出流正常"],
            fault_type=FaultType.PHYSICAL_FAULT,
            fault_location="管道系统",
            confidence=0.85,
            description="可能存在管道泄漏",
            recommendations=[
                "巡检管道查找泄漏点",
                "增大入流补偿",
                "隔离泄漏段",
                "启用备用管道"
            ]
        ))
        
        # 规则7: 上游供水不足
        self.rules.append(DiagnosisRule(
            name="上游供水不足",
            conditions=["水位低", "入流低", "入口闸门全开"],
            fault_type=FaultType.PHYSICAL_FAULT,
            fault_location="上游供水",
            confidence=0.90,
            description="上游来水不足",
            recommendations=[
                "联系上游调度",
                "降低下游需求",
                "启用备用水源"
            ]
        ))
        
        # 规则8: 需求激增
        self.rules.append(DiagnosisRule(
            name="下游需求激增",
            conditions=["水位快速下降", "出流大", "入流正常"],
            fault_type=FaultType.PHYSICAL_FAULT,
            fault_location="下游需求",
            confidence=0.85,
            description="下游需求突然增大",
            recommendations=[
                "增大入流",
                "通知下游用户",
                "启用应急预案"
            ]
        ))
    
    def diagnose(self, symptoms: Dict[str, bool]) -> List[DiagnosisReport]:
        """
        诊断故障
        
        Args:
            symptoms: 症状字典 {symptom_name: is_present}
            
        Returns:
            诊断报告列表，按置信度排序
        """
        reports = []
        
        for rule in self.rules:
            # 检查规则条件是否匹配
            matched_conditions = []
            for condition in rule.conditions:
                if symptoms.get(condition, False):
                    matched_conditions.append(condition)
            
            # 计算匹配度
            match_ratio = len(matched_conditions) / len(rule.conditions)
            
            if match_ratio > 0.6:  # 至少匹配60%的条件
                confidence = rule.confidence * match_ratio
                
                report = DiagnosisReport(
                    fault_type=rule.fault_type,
                    fault_location=rule.fault_location,
                    confidence=confidence,
                    symptoms=matched_conditions,
                    root_cause=rule.description,
                    recommendations=rule.recommendations,
                    matched_rules=[rule.name]
                )
                reports.append(report)
        
        # 按置信度排序
        reports.sort(key=lambda r: r.confidence, reverse=True)
        
        return reports


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*20 + "故障诊断系统演示")
    logger.info("="*70)
    
    # 创建诊断系统
    diagnosis_system = RuleBasedDiagnosisSystem()
    
    logger.info(f"\n规则库: {len(diagnosis_system.rules)}条规则")
    
    # 测试场景
    test_cases = [
        {
            'name': '出口闸门故障',
            'symptoms': {
                '水位异常高': True,
                '入流正常': True,
                '出流异常低': True
            }
        },
        {
            'name': '水位传感器故障',
            'symptoms': {
                '水位读数异常': True,
                '流量平衡正常': True
            }
        },
        {
            'name': '管道泄漏',
            'symptoms': {
                '水位持续下降': True,
                '入流正常': True,
                '出流正常': True
            }
        }
    ]
    
    for test in test_cases:
        logger.info(f"\n{'='*70}")
        logger.info(f"测试场景: {test['name']}")
        logger.info('='*70)
        
        logger.info(f"\n症状:")
        for symptom, present in test['symptoms'].items():
            if present:
                logger.info(f"  • {symptom}")
        
        reports = diagnosis_system.diagnose(test['symptoms'])
        
        if reports:
            logger.info(f"\n诊断结果 (共{len(reports)}个):")
            for i, report in enumerate(reports[:3], 1):
                logger.info(f"\n{i}. {report.matched_rules[0]}")
                logger.info(f"   故障类型: {report.fault_type.value}")
                logger.info(f"   故障位置: {report.fault_location}")
                logger.info(f"   置信度: {report.confidence:.2%}")
                logger.info(f"   根本原因: {report.root_cause}")
                logger.info(f"   建议措施:")
                for rec in report.recommendations[:3]:
                    logger.info(f"     • {rec}")
        else:
            logger.info("\n未能诊断出故障")
    
    logger.info("\n" + "="*70)
    logger.info("演示完成！")
    logger.info("="*70)
