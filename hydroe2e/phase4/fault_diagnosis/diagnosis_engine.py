"""
故障诊断引擎 - Water Control Domain Diagnosis Engine
基于规则和专家知识的水利控制系统故障诊断引擎

支持五大故障类别:
  1. sensor    - 传感器故障 (水位/流量/水质/冰期传感器)
  2. actuator  - 执行器故障 (闸门/泵站/阀门)
  3. controller - 控制器故障 (MPC/PID控制器异常)
  4. physics   - 物理故障 (管道泄漏/淤积/边坡失稳)
  5. network   - 网络故障 (通信中断/数据注入攻击/延迟)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FaultCategory(Enum):
    """五大故障类别"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CONTROLLER = "controller"
    PHYSICS = "physics"
    NETWORK = "network"
    UNKNOWN = "unknown"


class FaultSeverity(Enum):
    """故障严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    """诊断结果"""
    fault_category: str
    fault_type: str
    fault_component: str
    severity: str
    confidence: float
    root_cause: str
    recommended_actions: List[str]
    diagnosis_time: str
    additional_info: Dict = field(default_factory=dict)


@dataclass
class DiagnosisRule:
    """诊断规则"""
    name: str
    category: FaultCategory
    fault_type: str
    patterns: List[str]
    detector_hints: List[str]
    threshold: float
    severity: FaultSeverity
    component: str
    description: str
    root_causes: List[str]
    actions: List[str]


# ---------------------------------------------------------------------------
# DiagnosisEngine
# ---------------------------------------------------------------------------

class DiagnosisEngine:
    """
    水利控制系统故障诊断引擎

    功能:
    1. 基于规则的故障识别 (覆盖5大故障类别)
    2. 水利领域专家知识库
    3. 故障严重度评估
    4. 根因分析 (结合物理一致性)
    5. 推荐修复/应急措施
    6. 与 phase4 异常检测模块集成
    """

    FAULT_CATEGORIES = [
        FaultCategory.SENSOR,
        FaultCategory.ACTUATOR,
        FaultCategory.CONTROLLER,
        FaultCategory.PHYSICS,
        FaultCategory.NETWORK,
    ]

    def __init__(self):
        """初始化诊断引擎"""
        self.rules: List[DiagnosisRule] = self._build_rules()
        self.knowledge_base: Dict = self._build_knowledge_base()
        self.diagnosis_history: List[Dict] = []

        logger.info(
            "[DiagnosisEngine] 水利故障诊断引擎初始化完成, "
            "规则数=%d, 故障类别=%d",
            len(self.rules), len(self.FAULT_CATEGORIES),
        )

    # ------------------------------------------------------------------
    # Rule construction
    # ------------------------------------------------------------------

    def _build_rules(self) -> List[DiagnosisRule]:
        """构建水利控制领域诊断规则库"""
        rules: List[DiagnosisRule] = []

        # ---- 1. Sensor 传感器故障规则 ----
        rules.append(DiagnosisRule(
            name="水位传感器漂移",
            category=FaultCategory.SENSOR,
            fault_type="sensor_drift",
            patterns=["value_drift", "gradual_change", "bias_increase"],
            detector_hints=["level", "sensor", "3-sigma"],
            threshold=0.1,
            severity=FaultSeverity.MEDIUM,
            component="水位传感器",
            description="水位传感器读数缓慢偏移，可能由淤泥堆积或电路老化导致",
            root_causes=["传感器老化", "泥沙覆盖探头", "温度补偿失效", "校准偏差累积"],
            actions=[
                "切换到备用水位传感器",
                "启用多传感器融合校验",
                "安排现场传感器清洗与校准",
                "用流量守恒方程反推水位作为交叉验证",
            ],
        ))
        rules.append(DiagnosisRule(
            name="水位传感器卡死",
            category=FaultCategory.SENSOR,
            fault_type="sensor_stuck",
            patterns=["no_change", "constant_value", "zero_variance"],
            detector_hints=["level", "sensor"],
            threshold=0.01,
            severity=FaultSeverity.HIGH,
            component="水位传感器",
            description="水位传感器输出恒定值，疑似卡死或信号线断开",
            root_causes=["信号线脱落", "采集板故障", "传感器损坏", "供电中断"],
            actions=[
                "立即切换备用传感器",
                "检查传感器供电与信号线",
                "使用物理模型预测值临时替代",
            ],
        ))
        rules.append(DiagnosisRule(
            name="流量传感器噪声异常",
            category=FaultCategory.SENSOR,
            fault_type="sensor_noise",
            patterns=["high_variance", "random_noise", "spike"],
            detector_hints=["flow", "sensor", "sigma"],
            threshold=0.5,
            severity=FaultSeverity.LOW,
            component="流量传感器",
            description="流量传感器噪声增大，可能受气泡或水草干扰",
            root_causes=["气泡干扰", "水草缠绕探头", "电磁干扰", "安装松动"],
            actions=[
                "增加滤波窗口",
                "清理传感器探头",
                "检查接地与屏蔽",
            ],
        ))
        rules.append(DiagnosisRule(
            name="水质传感器异常",
            category=FaultCategory.SENSOR,
            fault_type="sensor_wq_fault",
            patterns=["value_drift", "spike", "out_of_range"],
            detector_hints=["quality", "turbidity", "concentration"],
            threshold=0.15,
            severity=FaultSeverity.MEDIUM,
            component="水质传感器",
            description="水质传感器读数异常，浊度或浓度探头可能被污染",
            root_causes=["探头生物附着", "试剂耗尽", "光路遮挡"],
            actions=[
                "清洗水质探头",
                "更换试剂",
                "交叉比对上下游传感器",
            ],
        ))

        # ---- 2. Actuator 执行器故障规则 ----
        rules.append(DiagnosisRule(
            name="闸门卡死",
            category=FaultCategory.ACTUATOR,
            fault_type="actuator_stuck",
            patterns=["no_response", "command_ignored", "position_mismatch"],
            detector_hints=["gate", "actuator", "valve"],
            threshold=0.05,
            severity=FaultSeverity.CRITICAL,
            component="闸门执行器",
            description="闸门无法响应控制指令，可能机械卡死或液压系统故障",
            root_causes=["异物卡阻", "液压油泄漏", "电机过载保护跳闸", "限位开关故障"],
            actions=[
                "启用备用闸门或旁通阀",
                "下发闸门复位指令",
                "派遣现场人员检查机械部件",
                "调整上下游流量补偿闸门失效影响",
            ],
        ))
        rules.append(DiagnosisRule(
            name="闸门响应延迟",
            category=FaultCategory.ACTUATOR,
            fault_type="actuator_delay",
            patterns=["slow_response", "time_delay", "overshoot"],
            detector_hints=["gate", "actuator", "response"],
            threshold=0.2,
            severity=FaultSeverity.MEDIUM,
            component="闸门执行器",
            description="闸门响应明显滞后于控制指令，导致水位调节不及时",
            root_causes=["液压系统压力不足", "机械磨损增大摩擦", "控制信号传输延迟"],
            actions=[
                "降低MPC控制频率适配执行器速度",
                "检查液压系统压力",
                "润滑闸门导轨",
            ],
        ))
        rules.append(DiagnosisRule(
            name="泵站故障",
            category=FaultCategory.ACTUATOR,
            fault_type="actuator_pump_fault",
            patterns=["no_response", "vibration_high", "flow_mismatch"],
            detector_hints=["pump", "actuator", "vibration"],
            threshold=0.1,
            severity=FaultSeverity.HIGH,
            component="泵站",
            description="泵站运行异常，流量输出与设定值不符",
            root_causes=["叶轮磨损", "轴承损坏", "电机过热", "进水口堵塞"],
            actions=[
                "切换备用泵组",
                "降低泵站转速运行",
                "检查进水口滤网",
                "安排泵站检修",
            ],
        ))

        # ---- 3. Controller 控制器故障规则 ----
        rules.append(DiagnosisRule(
            name="MPC求解失败",
            category=FaultCategory.CONTROLLER,
            fault_type="controller_solver_failure",
            patterns=["solver_infeasible", "optimization_timeout", "nan_output"],
            detector_hints=["controller", "mpc", "optimizer"],
            threshold=0.1,
            severity=FaultSeverity.HIGH,
            component="MPC控制器",
            description="MPC优化求解器无法收敛或返回异常结果",
            root_causes=[
                "约束冲突导致不可行", "模型参数过期", "预测步长设置不当",
                "状态估计严重偏差",
            ],
            actions=[
                "切换到PID后备控制器",
                "放松约束边界并重新求解",
                "更新物理模型参数",
                "检查状态观测器输出",
            ],
        ))
        rules.append(DiagnosisRule(
            name="控制器输出振荡",
            category=FaultCategory.CONTROLLER,
            fault_type="controller_oscillation",
            patterns=["oscillation", "high_variance", "rapid_switching"],
            detector_hints=["controller", "output", "control"],
            threshold=0.3,
            severity=FaultSeverity.MEDIUM,
            component="控制器",
            description="控制器输出频繁振荡，闸门反复开闭，加速机械磨损",
            root_causes=[
                "控制增益过大", "采样周期与系统时常不匹配",
                "模型失配", "传感器噪声传递",
            ],
            actions=[
                "降低控制增益",
                "增加控制输出平滑滤波",
                "重新辨识物理模型参数",
                "增加死区设置减少频繁动作",
            ],
        ))
        rules.append(DiagnosisRule(
            name="PID后备控制器饱和",
            category=FaultCategory.CONTROLLER,
            fault_type="controller_saturation",
            patterns=["saturation", "integral_windup", "output_limit"],
            detector_hints=["pid", "controller", "backup"],
            threshold=0.2,
            severity=FaultSeverity.MEDIUM,
            component="PID后备控制器",
            description="PID控制器积分饱和，控制输出长期处于极限位置",
            root_causes=["设定值不可达", "执行器能力不足", "积分时间常数过小"],
            actions=[
                "启用抗积分饱和机制",
                "调整设定值至可达范围",
                "增大积分时间常数",
            ],
        ))

        # ---- 4. Physics 物理故障规则 ----
        rules.append(DiagnosisRule(
            name="渠道泄漏",
            category=FaultCategory.PHYSICS,
            fault_type="physics_leakage",
            patterns=["water_loss", "level_drop", "flow_imbalance"],
            detector_hints=["physics", "mass_balance", "leakage"],
            threshold=0.15,
            severity=FaultSeverity.HIGH,
            component="渠道衬砌",
            description="流量守恒方程残差持续为负，疑似渠道渗漏或管道破裂",
            root_causes=["衬砌老化开裂", "地基不均匀沉降", "冻融循环损坏", "根系侵入"],
            actions=[
                "增大上游入流补偿水量损失",
                "巡检渠道查找渗漏段",
                "隔离渗漏段并启用旁通",
                "安排衬砌修复",
            ],
        ))
        rules.append(DiagnosisRule(
            name="渠道淤积",
            category=FaultCategory.PHYSICS,
            fault_type="physics_sedimentation",
            patterns=["flow_decrease", "level_rise", "roughness_increase"],
            detector_hints=["physics", "roughness", "sedimentation"],
            threshold=0.2,
            severity=FaultSeverity.MEDIUM,
            component="渠道断面",
            description="泥沙淤积导致过流断面减小、糙率增大",
            root_causes=["上游来沙量大", "流速过低沉降", "弯道淤积", "闸前淤积"],
            actions=[
                "安排清淤作业",
                "适当增大流速冲刷",
                "调整粗糙度参数适配当前状况",
                "加强上游拦沙措施",
            ],
        ))
        rules.append(DiagnosisRule(
            name="边坡失稳",
            category=FaultCategory.PHYSICS,
            fault_type="physics_slope_instability",
            patterns=["slope_safety_low", "rapid_drawdown", "seepage_high"],
            detector_hints=["slope", "safety", "stability"],
            threshold=0.1,
            severity=FaultSeverity.CRITICAL,
            component="渠道边坡",
            description="边坡安全系数低于警戒值，退水速率过快可能诱发滑坡",
            root_causes=["退水速率过快", "暴雨入渗", "地下水位升高", "边坡土体软化"],
            actions=[
                "立即降低退水速率至安全限值",
                "启用边坡保护模式",
                "加强边坡位移监测",
                "必要时停止退水操作",
            ],
        ))
        rules.append(DiagnosisRule(
            name="冰期堵塞",
            category=FaultCategory.PHYSICS,
            fault_type="physics_ice_blockage",
            patterns=["flow_decrease", "level_rise", "temperature_low"],
            detector_hints=["ice", "temperature", "winter"],
            threshold=0.15,
            severity=FaultSeverity.HIGH,
            component="渠道冰盖",
            description="冰期冰盖或冰塞导致过流能力下降",
            root_causes=["气温骤降", "冰盖生长过快", "闸前冰塞", "冰花堆积"],
            actions=[
                "启用冰期运行模式",
                "降低流量避免冰塞恶化",
                "开启破冰设备",
                "调整粗糙度参数考虑冰盖影响",
            ],
        ))

        # ---- 5. Network 网络故障规则 ----
        rules.append(DiagnosisRule(
            name="SCADA通信中断",
            category=FaultCategory.NETWORK,
            fault_type="network_comm_loss",
            patterns=["packet_loss", "timeout", "no_data"],
            detector_hints=["network", "comm", "scada", "timeout"],
            threshold=0.3,
            severity=FaultSeverity.HIGH,
            component="SCADA通信链路",
            description="与远程站点的SCADA通信中断，无法获取实时数据",
            root_causes=["光纤断裂", "RTU设备故障", "基站掉电", "网络设备宕机"],
            actions=[
                "切换到备用通信链路",
                "启用本地自主控制模式",
                "使用物理模型预测值填补数据空白",
                "派遣人员检查通信设备",
            ],
        ))
        rules.append(DiagnosisRule(
            name="数据注入攻击(FDIA)",
            category=FaultCategory.NETWORK,
            fault_type="network_fdia",
            patterns=["abnormal_pattern", "data_injection", "physics_inconsistency"],
            detector_hints=["cyber", "attack", "fdia", "security"],
            threshold=0.8,
            severity=FaultSeverity.CRITICAL,
            component="数据采集网络",
            description="检测到虚假数据注入攻击，传感器数据与物理模型预测严重不一致",
            root_causes=["恶意数据篡改", "中间人攻击", "RTU被入侵", "传输层数据劫持"],
            actions=[
                "启用网络防御模式",
                "使用数字孪生模型清洗数据",
                "隔离受影响的传感器通道",
                "通知网络安全团队",
                "记录攻击特征用于取证",
            ],
        ))
        rules.append(DiagnosisRule(
            name="网络延迟过大",
            category=FaultCategory.NETWORK,
            fault_type="network_latency",
            patterns=["high_latency", "time_delay", "data_stale"],
            detector_hints=["network", "latency", "delay"],
            threshold=0.25,
            severity=FaultSeverity.MEDIUM,
            component="控制网络",
            description="控制网络延迟超出容忍范围，实时控制性能下降",
            root_causes=["网络拥塞", "路由环路", "带宽不足", "设备队列溢出"],
            actions=[
                "降低控制频率适配网络延迟",
                "启用本地前馈补偿",
                "检查网络拓扑和路由配置",
                "增加带宽或启用QoS优先级",
            ],
        ))

        return rules

    # ------------------------------------------------------------------
    # Knowledge base
    # ------------------------------------------------------------------

    def _build_knowledge_base(self) -> Dict[FaultCategory, Dict]:
        """构建水利控制领域专家知识库"""
        return {
            FaultCategory.SENSOR: {
                "label": "传感器故障",
                "common_causes": [
                    "传感器老化漂移", "探头污染或堵塞", "信号线故障",
                    "供电异常", "环境干扰(电磁/温度)",
                ],
                "general_actions": [
                    "切换备用传感器", "启用多传感器融合",
                    "用物理模型交叉校验", "安排现场检查校准",
                ],
            },
            FaultCategory.ACTUATOR: {
                "label": "执行器故障",
                "common_causes": [
                    "机械磨损/卡阻", "液压系统泄漏", "电机过载",
                    "限位开关故障", "异物阻塞",
                ],
                "general_actions": [
                    "切换备用执行器", "降低控制动作频率",
                    "检查电源与液压系统", "安排紧急检修",
                ],
            },
            FaultCategory.CONTROLLER: {
                "label": "控制器故障",
                "common_causes": [
                    "优化求解不收敛", "模型参数失配", "控制增益不当",
                    "采样周期不匹配", "状态估计偏差大",
                ],
                "general_actions": [
                    "切换到PID后备控制器", "更新模型参数",
                    "调整控制增益", "检查状态观测器",
                ],
            },
            FaultCategory.PHYSICS: {
                "label": "物理故障",
                "common_causes": [
                    "渠道渗漏", "泥沙淤积", "边坡失稳",
                    "冰期堵塞", "水质污染",
                ],
                "general_actions": [
                    "调整运行工况补偿", "安排现场巡检",
                    "启用应急运行模式", "更新物理模型参数",
                ],
            },
            FaultCategory.NETWORK: {
                "label": "网络故障",
                "common_causes": [
                    "通信链路中断", "网络攻击(FDIA)", "设备宕机",
                    "网络拥塞延迟", "配置错误",
                ],
                "general_actions": [
                    "切换备用通信链路", "启用本地自主控制",
                    "使用数字孪生预测值替代", "通知运维团队",
                ],
            },
        }

    # ------------------------------------------------------------------
    # Core diagnosis
    # ------------------------------------------------------------------

    def diagnose(self, anomaly: Dict) -> Optional[DiagnosisResult]:
        """
        诊断异常，识别故障类型

        与 phase4 异常检测模块集成: 接受 anomaly_detection 输出的异常字典。

        Args:
            anomaly: 异常信息字典，包含:
                - detector (str): 检测器名称 (如 '3-sigma', 'isolation_forest')
                - value (float): 异常值/异常分数
                - threshold (float): 检测阈值
                - timestamp (str, optional): 时间戳
                - consecutive (int): 连续异常次数
                - category_hint (str, optional): 故障类别提示
                - component_hint (str, optional): 组件提示

        Returns:
            DiagnosisResult 或 None (输入为空时)
        """
        if not anomaly:
            return None

        detector = anomaly.get("detector", "unknown")
        value = anomaly.get("value", 0)
        threshold = anomaly.get("threshold", 0)
        consecutive = anomaly.get("consecutive", 1)
        category_hint = anomaly.get("category_hint", "")
        component_hint = anomaly.get("component_hint", "")

        # 计算异常程度
        if threshold > 0:
            anomaly_degree = abs(value - threshold) / threshold
        else:
            anomaly_degree = abs(value)

        # 规则匹配: 对所有规则打分
        scored_rules: List[Tuple[DiagnosisRule, float]] = []

        for rule in self.rules:
            score = self._score_rule(
                rule, detector, anomaly_degree, consecutive,
                category_hint, component_hint,
            )
            if score > 0.0:
                scored_rules.append((rule, score))

        # 按分数降序
        scored_rules.sort(key=lambda x: x[1], reverse=True)

        if scored_rules and scored_rules[0][1] > 0.4:
            best_rule, confidence = scored_rules[0]
            confidence = min(confidence, 1.0)

            result = DiagnosisResult(
                fault_category=best_rule.category.value,
                fault_type=best_rule.fault_type,
                fault_component=best_rule.component,
                severity=best_rule.severity.value,
                confidence=round(confidence, 4),
                root_cause=self._select_root_cause(best_rule, anomaly),
                recommended_actions=best_rule.actions,
                diagnosis_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                additional_info={
                    "detector": detector,
                    "anomaly_degree": round(anomaly_degree, 4),
                    "consecutive_count": consecutive,
                    "matched_rule": best_rule.name,
                    "rule_fault_type": best_rule.fault_type,
                    "description": best_rule.description,
                    "candidate_count": len(scored_rules),
                },
            )

            self._record_history(result)
            return result

        # 无法确定 -> 返回 UNKNOWN
        result = DiagnosisResult(
            fault_category=FaultCategory.UNKNOWN.value,
            fault_type="unknown",
            fault_component=component_hint or "未知组件",
            severity=FaultSeverity.MEDIUM.value,
            confidence=round(scored_rules[0][1], 4) if scored_rules else 0.0,
            root_cause="无法确定根本原因，需要进一步分析",
            recommended_actions=["增加监控频率", "收集更多数据", "安排人工巡检"],
            diagnosis_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            additional_info={
                "detector": detector,
                "anomaly_degree": round(anomaly_degree, 4),
            },
        )
        self._record_history(result)
        return result

    def diagnose_batch(self, anomalies: List[Dict]) -> List[DiagnosisResult]:
        """
        批量诊断多个异常

        Args:
            anomalies: 异常字典列表

        Returns:
            诊断结果列表
        """
        results = []
        for anomaly in anomalies:
            result = self.diagnose(anomaly)
            if result is not None:
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # Rule scoring
    # ------------------------------------------------------------------

    def _score_rule(
        self,
        rule: DiagnosisRule,
        detector: str,
        anomaly_degree: float,
        consecutive: int,
        category_hint: str,
        component_hint: str,
    ) -> float:
        """
        对规则打分

        综合考虑:
        - 检测器名称与规则 detector_hints 的匹配度
        - 异常程度与规则阈值的比较
        - 连续异常次数的加权
        - 类别提示与组件提示的额外加分
        """
        score = 0.0

        # 1. 检测器名称匹配 (0 ~ 0.3)
        detector_lower = detector.lower()
        hint_match_count = sum(
            1 for hint in rule.detector_hints if hint in detector_lower
        )
        if hint_match_count > 0:
            score += min(hint_match_count * 0.15, 0.3)

        # 2. 异常程度 vs 阈值 (0 ~ 0.4)
        if rule.threshold > 0:
            degree_ratio = anomaly_degree / rule.threshold
            score += min(degree_ratio * 0.2, 0.4)

        # 3. 连续次数加权 (0 ~ 0.2)
        consecutive_factor = min(consecutive / 5.0, 1.0)
        score += consecutive_factor * 0.2

        # 4. 类别提示匹配 (0 ~ 0.1)
        if category_hint and category_hint.lower() == rule.category.value:
            score += 0.1

        # 5. 组件提示匹配 (0 ~ 0.1)
        if component_hint and component_hint in rule.component:
            score += 0.1

        return score

    # ------------------------------------------------------------------
    # Root cause analysis
    # ------------------------------------------------------------------

    def _select_root_cause(self, rule: DiagnosisRule, anomaly: Dict) -> str:
        """
        根据匹配规则和异常特征选择最可能的根因

        简化实现: 根据异常程度选择不同级别的原因。
        """
        causes = rule.root_causes
        if not causes:
            return "需要进一步调查"

        value = anomaly.get("value", 0)
        threshold = anomaly.get("threshold", 1)
        consecutive = anomaly.get("consecutive", 1)

        # 严重程度越高选择越靠前(更常见)的原因
        if threshold > 0 and abs(value) > 2 * threshold:
            return causes[0]  # 最严重/最常见
        elif consecutive >= 5:
            return causes[min(1, len(causes) - 1)]
        else:
            return causes[min(2, len(causes) - 1)]

    # ------------------------------------------------------------------
    # History & statistics
    # ------------------------------------------------------------------

    def _record_history(self, result: DiagnosisResult) -> None:
        """记录诊断历史"""
        self.diagnosis_history.append({
            "timestamp": result.diagnosis_time,
            "fault_category": result.fault_category,
            "fault_type": result.fault_type,
            "severity": result.severity,
            "confidence": result.confidence,
            "component": result.fault_component,
        })

    def get_diagnosis_history(self) -> List[Dict]:
        """获取诊断历史"""
        return list(self.diagnosis_history)

    def get_statistics(self) -> Dict:
        """获取诊断统计信息"""
        if not self.diagnosis_history:
            return {
                "total_diagnoses": 0,
                "category_distribution": {},
                "severity_distribution": {},
                "average_confidence": 0.0,
            }

        categories: Dict[str, int] = {}
        severities: Dict[str, int] = {}
        total_confidence = 0.0

        for record in self.diagnosis_history:
            cat = record["fault_category"]
            sev = record["severity"]
            categories[cat] = categories.get(cat, 0) + 1
            severities[sev] = severities.get(sev, 0) + 1
            total_confidence += record["confidence"]

        return {
            "total_diagnoses": len(self.diagnosis_history),
            "category_distribution": categories,
            "severity_distribution": severities,
            "average_confidence": round(
                total_confidence / len(self.diagnosis_history), 4
            ),
        }

    def get_category_summary(self) -> Dict[str, int]:
        """按五大类别统计故障数量"""
        summary = {cat.value: 0 for cat in self.FAULT_CATEGORIES}
        for record in self.diagnosis_history:
            cat = record.get("fault_category", "unknown")
            if cat in summary:
                summary[cat] += 1
        return summary

    def clear_history(self) -> None:
        """清空诊断历史"""
        self.diagnosis_history.clear()
        logger.info("[DiagnosisEngine] 诊断历史已清空")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def quick_diagnose(anomaly: Dict) -> Optional[DiagnosisResult]:
    """快速诊断函数 (创建临时引擎实例)"""
    engine = DiagnosisEngine()
    return engine.diagnose(anomaly)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 80)
    logger.info(" " * 15 + "水利控制系统故障诊断引擎测试")
    logger.info("=" * 80)

    engine = DiagnosisEngine()
    logger.info(f"规则总数: {len(engine.rules)}")
    logger.info(f"故障类别: {[c.value for c in engine.FAULT_CATEGORIES]}")

    test_cases = [
        {
            "name": "水位传感器漂移",
            "anomaly": {
                "detector": "level_sensor_3-sigma",
                "value": 8.0,
                "threshold": 2.0,
                "consecutive": 5,
                "category_hint": "sensor",
            },
        },
        {
            "name": "闸门卡死",
            "anomaly": {
                "detector": "gate_actuator_monitor",
                "value": 0.95,
                "threshold": 0.05,
                "consecutive": 10,
                "category_hint": "actuator",
            },
        },
        {
            "name": "MPC求解失败",
            "anomaly": {
                "detector": "mpc_controller_status",
                "value": 1.0,
                "threshold": 0.1,
                "consecutive": 3,
                "category_hint": "controller",
            },
        },
        {
            "name": "渠道泄漏",
            "anomaly": {
                "detector": "mass_balance_physics",
                "value": -5.0,
                "threshold": 1.0,
                "consecutive": 8,
                "category_hint": "physics",
            },
        },
        {
            "name": "FDIA网络攻击",
            "anomaly": {
                "detector": "cyber_security_fdia",
                "value": 0.95,
                "threshold": 0.8,
                "consecutive": 2,
                "category_hint": "network",
            },
        },
    ]

    for tc in test_cases:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"测试: {tc['name']}")
        result = engine.diagnose(tc["anomaly"])
        if result:
            logger.info(f"  故障类别:   {result.fault_category}")
            logger.info(f"  故障类型:   {result.fault_type}")
            logger.info(f"  故障组件:   {result.fault_component}")
            logger.info(f"  严重程度:   {result.severity}")
            logger.info(f"  置信度:     {result.confidence:.2f}")
            logger.info(f"  根本原因:   {result.root_cause}")
            logger.info(f"  推荐措施:   {result.recommended_actions[0]}")

    logger.info(f"\n{'=' * 80}")
    logger.info("诊断统计:")
    stats = engine.get_statistics()
    logger.info(f"  总诊断次数: {stats['total_diagnoses']}")
    logger.info(f"  类别分布:   {stats['category_distribution']}")
    logger.info(f"  平均置信度: {stats['average_confidence']:.2f}")
    logger.info(f"\n类别汇总: {engine.get_category_summary()}")
    logger.info("=" * 80)
