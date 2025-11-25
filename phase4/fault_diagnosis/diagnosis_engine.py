"""
故障诊断引擎 - Diagnosis Engine
基于规则和专家知识的故障诊断系统
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class FaultType(Enum):
    """故障类型"""
    SENSOR_FAULT = "传感器故障"
    ACTUATOR_FAULT = "执行器故障"
    COMMUNICATION_FAULT = "通信故障"
    PHYSICAL_FAULT = "物理故障"
    CYBER_ATTACK = "网络攻击"
    UNKNOWN = "未知故障"


class FaultSeverity(Enum):
    """故障严重程度"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "严重"


@dataclass
class DiagnosisResult:
    """诊断结果"""
    fault_type: str
    fault_component: str
    severity: str
    confidence: float
    root_cause: str
    recommended_actions: List[str]
    diagnosis_time: str
    additional_info: Dict


class DiagnosisEngine:
    """
    故障诊断引擎
    
    功能：
    1. 基于规则的故障识别
    2. 专家知识库
    3. 故障严重度评估
    4. 根因分析
    5. 推荐修复措施
    """
    
    def __init__(self):
        """初始化诊断引擎"""
        self.diagnosis_rules = self._init_diagnosis_rules()
        self.knowledge_base = self._init_knowledge_base()
        self.diagnosis_history = []
        
        print("[DiagnosisEngine] 故障诊断引擎初始化完成")
    
    def _init_diagnosis_rules(self) -> Dict:
        """初始化诊断规则库"""
        return {
            # 传感器故障规则
            'sensor_drift': {
                'patterns': ['value_drift', 'gradual_change'],
                'threshold': 0.1,
                'fault_type': FaultType.SENSOR_FAULT,
                'severity': FaultSeverity.MEDIUM
            },
            'sensor_stuck': {
                'patterns': ['no_change', 'constant_value'],
                'threshold': 0.01,
                'fault_type': FaultType.SENSOR_FAULT,
                'severity': FaultSeverity.HIGH
            },
            'sensor_noise': {
                'patterns': ['high_variance', 'random_noise'],
                'threshold': 0.5,
                'fault_type': FaultType.SENSOR_FAULT,
                'severity': FaultSeverity.LOW
            },
            
            # 执行器故障规则
            'actuator_stuck': {
                'patterns': ['no_response', 'command_ignored'],
                'threshold': 0.05,
                'fault_type': FaultType.ACTUATOR_FAULT,
                'severity': FaultSeverity.CRITICAL
            },
            'actuator_delay': {
                'patterns': ['slow_response', 'time_delay'],
                'threshold': 0.2,
                'fault_type': FaultType.ACTUATOR_FAULT,
                'severity': FaultSeverity.MEDIUM
            },
            
            # 通信故障规则
            'communication_loss': {
                'patterns': ['packet_loss', 'timeout'],
                'threshold': 0.3,
                'fault_type': FaultType.COMMUNICATION_FAULT,
                'severity': FaultSeverity.HIGH
            },
            
            # 网络攻击规则
            'cyber_attack': {
                'patterns': ['abnormal_pattern', 'data_injection'],
                'threshold': 0.8,
                'fault_type': FaultType.CYBER_ATTACK,
                'severity': FaultSeverity.CRITICAL
            }
        }
    
    def _init_knowledge_base(self) -> Dict:
        """初始化专家知识库"""
        return {
            FaultType.SENSOR_FAULT: {
                'common_causes': [
                    '传感器老化',
                    '环境干扰',
                    '线路故障',
                    '校准偏差'
                ],
                'recommended_actions': [
                    '切换到备用传感器',
                    '启用传感器融合',
                    '增加测量频率',
                    '安排维护检查'
                ]
            },
            FaultType.ACTUATOR_FAULT: {
                'common_causes': [
                    '机械磨损',
                    '电源问题',
                    '控制信号异常',
                    '负载过大'
                ],
                'recommended_actions': [
                    '切换到备用执行器',
                    '降低控制频率',
                    '检查电源供应',
                    '立即安排检修'
                ]
            },
            FaultType.COMMUNICATION_FAULT: {
                'common_causes': [
                    '网络拥塞',
                    '设备故障',
                    '配置错误',
                    '干扰信号'
                ],
                'recommended_actions': [
                    '重启通信模块',
                    '切换备用链路',
                    '检查网络配置',
                    '增加重传机制'
                ]
            },
            FaultType.CYBER_ATTACK: {
                'common_causes': [
                    '恶意数据注入',
                    '拒绝服务攻击',
                    '中间人攻击',
                    '权限提升'
                ],
                'recommended_actions': [
                    '隔离受影响系统',
                    '启用安全模式',
                    '验证数据完整性',
                    '通知安全团队'
                ]
            }
        }
    
    def diagnose(self, anomaly: Dict) -> Optional[DiagnosisResult]:
        """
        诊断异常，识别故障类型
        
        Args:
            anomaly: 异常信息字典，包含：
                - detector: 检测器名称
                - value: 异常值
                - threshold: 阈值
                - timestamp: 时间戳
                - consecutive: 连续次数
                
        Returns:
            DiagnosisResult: 诊断结果，如果无法诊断则返回None
        """
        if not anomaly:
            return None
        
        # 提取异常特征
        detector = anomaly.get('detector', 'unknown')
        value = anomaly.get('value', 0)
        threshold = anomaly.get('threshold', 0)
        consecutive = anomaly.get('consecutive', 1)
        
        # 计算异常程度
        if threshold > 0:
            anomaly_degree = abs(value - threshold) / threshold
        else:
            anomaly_degree = abs(value)
        
        # 规则匹配
        matched_rule = None
        max_confidence = 0.0
        
        for rule_name, rule in self.diagnosis_rules.items():
            confidence = self._calculate_confidence(
                anomaly_degree, 
                consecutive,
                rule
            )
            
            if confidence > max_confidence:
                max_confidence = confidence
                matched_rule = (rule_name, rule)
        
        if matched_rule and max_confidence > 0.5:
            rule_name, rule = matched_rule
            fault_type = rule['fault_type']
            severity = rule['severity']
            
            # 生成诊断结果
            result = DiagnosisResult(
                fault_type=fault_type.value,
                fault_component=self._identify_component(detector, anomaly),
                severity=severity.value,
                confidence=max_confidence,
                root_cause=self._analyze_root_cause(fault_type, anomaly),
                recommended_actions=self._get_recommended_actions(fault_type),
                diagnosis_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                additional_info={
                    'detector': detector,
                    'anomaly_degree': anomaly_degree,
                    'consecutive_count': consecutive,
                    'matched_rule': rule_name
                }
            )
            
            # 保存诊断历史
            self.diagnosis_history.append({
                'timestamp': result.diagnosis_time,
                'fault_type': result.fault_type,
                'severity': result.severity,
                'confidence': result.confidence
            })
            
            return result
        
        # 无法确定故障类型
        return DiagnosisResult(
            fault_type=FaultType.UNKNOWN.value,
            fault_component="未知组件",
            severity=FaultSeverity.MEDIUM.value,
            confidence=max_confidence,
            root_cause="无法确定根本原因，需要进一步分析",
            recommended_actions=["增加监控频率", "收集更多数据", "人工检查"],
            diagnosis_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            additional_info={
                'detector': detector,
                'anomaly_degree': anomaly_degree
            }
        )
    
    def _calculate_confidence(self, anomaly_degree: float, consecutive: int, rule: Dict) -> float:
        """计算诊断置信度"""
        # 基础置信度（基于异常程度）
        base_confidence = min(anomaly_degree / rule['threshold'], 1.0)
        
        # 连续性加权
        consecutive_weight = min(consecutive / 5.0, 1.0)
        
        # 综合置信度
        confidence = 0.6 * base_confidence + 0.4 * consecutive_weight
        
        return confidence
    
    def _identify_component(self, detector: str, anomaly: Dict) -> str:
        """识别故障组件"""
        # 基于检测器类型推断组件
        if 'sensor' in detector.lower() or 'level' in detector.lower():
            return "水位传感器"
        elif 'flow' in detector.lower():
            return "流量传感器"
        elif 'gate' in detector.lower() or 'actuator' in detector.lower():
            return "闸门执行器"
        elif 'network' in detector.lower() or 'comm' in detector.lower():
            return "通信模块"
        else:
            return "未知组件"
    
    def _analyze_root_cause(self, fault_type: FaultType, anomaly: Dict) -> str:
        """分析根本原因"""
        kb = self.knowledge_base.get(fault_type, {})
        causes = kb.get('common_causes', ['未知原因'])
        
        # 根据异常特征选择最可能的原因
        if len(causes) > 0:
            return causes[0]  # 简化实现，返回最常见原因
        return "需要进一步调查"
    
    def _get_recommended_actions(self, fault_type: FaultType) -> List[str]:
        """获取推荐措施"""
        kb = self.knowledge_base.get(fault_type, {})
        actions = kb.get('recommended_actions', ['人工检查'])
        return actions
    
    def get_diagnosis_history(self) -> List[Dict]:
        """获取诊断历史"""
        return self.diagnosis_history
    
    def get_statistics(self) -> Dict:
        """获取诊断统计信息"""
        if not self.diagnosis_history:
            return {
                'total_diagnoses': 0,
                'fault_type_distribution': {},
                'severity_distribution': {},
                'average_confidence': 0.0
            }
        
        # 统计故障类型分布
        fault_types = {}
        severities = {}
        total_confidence = 0.0
        
        for record in self.diagnosis_history:
            fault_type = record['fault_type']
            severity = record['severity']
            confidence = record['confidence']
            
            fault_types[fault_type] = fault_types.get(fault_type, 0) + 1
            severities[severity] = severities.get(severity, 0) + 1
            total_confidence += confidence
        
        return {
            'total_diagnoses': len(self.diagnosis_history),
            'fault_type_distribution': fault_types,
            'severity_distribution': severities,
            'average_confidence': total_confidence / len(self.diagnosis_history)
        }
    
    def clear_history(self):
        """清空诊断历史"""
        self.diagnosis_history = []
        print("[DiagnosisEngine] 诊断历史已清空")


# 便捷函数
def quick_diagnose(anomaly: Dict) -> Optional[DiagnosisResult]:
    """快速诊断函数"""
    engine = DiagnosisEngine()
    return engine.diagnose(anomaly)


if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "故障诊断引擎测试")
    print("="*80)
    
    # 创建诊断引擎
    engine = DiagnosisEngine()
    
    # 测试案例1：传感器漂移
    print("\n测试1: 传感器漂移")
    anomaly1 = {
        'detector': '3-sigma',
        'value': 8.0,
        'threshold': 2.0,
        'consecutive': 5
    }
    result1 = engine.diagnose(anomaly1)
    if result1:
        print(f"  故障类型: {result1.fault_type}")
        print(f"  故障组件: {result1.fault_component}")
        print(f"  严重程度: {result1.severity}")
        print(f"  置信度: {result1.confidence:.2f}")
        print(f"  根本原因: {result1.root_cause}")
        print(f"  推荐措施: {', '.join(result1.recommended_actions[:2])}")
    
    # 测试案例2：执行器故障
    print("\n测试2: 执行器响应延迟")
    anomaly2 = {
        'detector': 'actuator_monitor',
        'value': 0.5,
        'threshold': 0.2,
        'consecutive': 3
    }
    result2 = engine.diagnose(anomaly2)
    if result2:
        print(f"  故障类型: {result2.fault_type}")
        print(f"  严重程度: {result2.severity}")
        print(f"  置信度: {result2.confidence:.2f}")
    
    # 统计信息
    print("\n诊断统计:")
    stats = engine.get_statistics()
    print(f"  总诊断次数: {stats['total_diagnoses']}")
    print(f"  平均置信度: {stats['average_confidence']:.2f}")
    
    print("\n" + "="*80)
    print("✅ 故障诊断引擎测试完成")
    print("="*80)
