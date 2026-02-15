"""
L3 宏观认知层 (Macro Cognitive Layer)

为集中调度器提供宏观层面的认知能力:
1. 系统健康态势感知
2. 长期设备退化趋势
3. 历史故障模式分析
4. 季节/天气宏观影响
5. 全网风险评估
6. 战略性调度建议

架构位置:
┌─────────────────────────────────────────────────────────────┐
│                   宏观认知层 (MacroCognitive)                 │
│  - 输入: 系统健康度、历史数据、外部预测                         │
│  - 输出: 战略建议、风险评估、约束调整                          │
├─────────────────────────────────────────────────────────────┤
│                   L3 集中调度层 (CentralizedScheduler)        │
│  - 接收宏观认知建议                                           │
│  - 调整调度参数和约束                                         │
└─────────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


class SystemHealthLevel(Enum):
    """系统健康等级"""
    EXCELLENT = "excellent"      # 优秀 (>95%)
    GOOD = "good"                # 良好 (85-95%)
    FAIR = "fair"                # 一般 (70-85%)
    DEGRADED = "degraded"        # 退化 (50-70%)
    CRITICAL = "critical"        # 临界 (<50%)


class SeasonalPattern(Enum):
    """季节模式"""
    SPRING_FLOOD = "spring_flood"    # 春汛
    SUMMER_PEAK = "summer_peak"      # 夏季用水高峰
    AUTUMN_NORMAL = "autumn_normal"  # 秋季正常
    WINTER_ICE = "winter_ice"        # 冬季冰期
    TRANSITION = "transition"        # 过渡期


class RiskLevel(Enum):
    """风险等级"""
    MINIMAL = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    SEVERE = 4
    EXTREME = 5


@dataclass
class EquipmentHealth:
    """设备健康状态"""
    component_id: str
    health_score: float  # 0-1
    degradation_rate: float  # 每月退化率
    remaining_useful_life: float  # 剩余使用寿命 (天)
    last_maintenance: Optional[str] = None
    failure_probability: float = 0.0  # 未来24h故障概率
    trend: str = "stable"  # improving, stable, degrading


@dataclass
class NetworkRiskAssessment:
    """全网风险评估"""
    overall_risk: RiskLevel
    risk_score: float  # 0-100
    top_risks: List[Dict]  # 主要风险因素
    risk_by_pool: Dict[str, float]  # 各池风险
    risk_trend: str  # increasing, stable, decreasing
    confidence: float = 0.8


@dataclass
class StrategicAdvice:
    """战略建议"""
    advice_id: str
    category: str  # scheduling, maintenance, safety, efficiency
    priority: int  # 1-5, 5最高
    description: str
    rationale: str
    recommended_action: str
    constraint_adjustments: Dict = field(default_factory=dict)
    valid_until: Optional[str] = None


@dataclass
class MacroCognitiveState:
    """宏观认知状态"""
    timestamp: str
    system_health: SystemHealthLevel
    health_score: float
    seasonal_pattern: SeasonalPattern
    risk_assessment: NetworkRiskAssessment
    equipment_health: Dict[str, EquipmentHealth]
    active_advisories: List[StrategicAdvice]
    forecast_horizon: int = 24  # 小时


class MacroCognitiveLayer:
    """
    L3 宏观认知层

    职责:
    1. 聚合系统健康信息
    2. 分析长期趋势
    3. 评估全网风险
    4. 生成战略建议
    5. 为L3调度器提供认知支持
    """

    def __init__(self, num_pools: int = 3):
        """初始化宏观认知层"""
        self.num_pools = num_pools

        # 状态
        self.current_state: Optional[MacroCognitiveState] = None
        self.state_history: List[MacroCognitiveState] = []

        # 设备健康追踪
        self.equipment_registry: Dict[str, EquipmentHealth] = {}
        self._init_equipment_registry()

        # 历史模式
        self.fault_history: List[Dict] = []
        self.performance_history: List[Dict] = []

        # 配置
        self.config = {
            'health_update_interval': 3600,  # 1小时
            'risk_threshold_high': 60,
            'risk_threshold_moderate': 40,
            'degradation_alert_days': 30,
            'maintenance_lead_time': 7  # 天
        }

        # 统计
        self.stats = {
            'assessments': 0,
            'advisories_generated': 0,
            'risk_alerts': 0
        }

        logger.info("[MacroCognitiveLayer] 宏观认知层初始化完成")

    def _init_equipment_registry(self):
        """初始化设备注册表"""
        # 水位传感器
        for i in range(self.num_pools):
            self.equipment_registry[f"level_sensor_{i}"] = EquipmentHealth(
                component_id=f"level_sensor_{i}",
                health_score=1.0,
                degradation_rate=0.005,  # 每月0.5%
                remaining_useful_life=365 * 3,  # 3年
            )
            self.equipment_registry[f"level_sensor_{i}_backup"] = EquipmentHealth(
                component_id=f"level_sensor_{i}_backup",
                health_score=1.0,
                degradation_rate=0.003,
                remaining_useful_life=365 * 4,
            )

        # 闸门
        for i in range(self.num_pools):
            self.equipment_registry[f"gate_{i}"] = EquipmentHealth(
                component_id=f"gate_{i}",
                health_score=1.0,
                degradation_rate=0.01,  # 每月1%
                remaining_useful_life=365 * 5,  # 5年
            )

        # 控制器
        self.equipment_registry["mpc_controller"] = EquipmentHealth(
            component_id="mpc_controller",
            health_score=1.0,
            degradation_rate=0.002,
            remaining_useful_life=365 * 7,
        )

    def assess(self,
               system_state: Dict,
               forecast_data: Dict = None,
               fault_reports: List[Dict] = None,
               external_factors: Dict = None) -> MacroCognitiveState:
        """
        执行宏观认知评估

        Args:
            system_state: 当前系统状态
            forecast_data: 预测数据 (天气、需求等)
            fault_reports: 故障报告
            external_factors: 外部因素 (季节、政策等)

        Returns:
            MacroCognitiveState: 宏观认知状态
        """
        start_time = time.time()

        # 1. 更新设备健康
        self._update_equipment_health(system_state, fault_reports)

        # 2. 计算系统健康度
        system_health, health_score = self._assess_system_health()

        # 3. 识别季节模式
        seasonal_pattern = self._identify_seasonal_pattern(external_factors)

        # 4. 风险评估
        risk_assessment = self._assess_network_risk(
            system_state, forecast_data, fault_reports
        )

        # 5. 生成战略建议
        advisories = self._generate_strategic_advice(
            system_health, risk_assessment, seasonal_pattern, forecast_data
        )

        # 6. 构建认知状态
        state = MacroCognitiveState(
            timestamp=datetime.now().isoformat(),
            system_health=system_health,
            health_score=health_score,
            seasonal_pattern=seasonal_pattern,
            risk_assessment=risk_assessment,
            equipment_health=self.equipment_registry.copy(),
            active_advisories=advisories
        )

        # 7. 更新历史
        self.current_state = state
        self.state_history.append(state)
        if len(self.state_history) > 168:  # 保留7天
            self.state_history = self.state_history[-168:]

        self.stats['assessments'] += 1

        return state

    def _update_equipment_health(self,
                                 system_state: Dict,
                                 fault_reports: List[Dict] = None):
        """更新设备健康状态"""
        current_time = datetime.now()

        for comp_id, health in self.equipment_registry.items():
            # 基于时间的自然退化
            days_since_check = 1  # 简化处理
            natural_degradation = health.degradation_rate / 30 * days_since_check

            # 基于故障的加速退化
            fault_degradation = 0.0
            if fault_reports:
                for fault in fault_reports:
                    if fault.get('component', '') == comp_id:
                        severity = fault.get('severity', 1)
                        fault_degradation += severity * 0.02

            # 更新健康分数
            health.health_score = max(
                0.0,
                health.health_score - natural_degradation - fault_degradation
            )

            # 更新剩余寿命
            if health.degradation_rate > 0:
                health.remaining_useful_life = max(
                    0,
                    health.health_score / (health.degradation_rate / 30)
                )

            # 计算故障概率
            health.failure_probability = self._estimate_failure_probability(health)

            # 更新趋势
            health.trend = self._determine_health_trend(comp_id)

    def _estimate_failure_probability(self, health: EquipmentHealth) -> float:
        """估算24小时内故障概率"""
        # 基于健康分数的简单模型
        if health.health_score > 0.9:
            base_prob = 0.001
        elif health.health_score > 0.7:
            base_prob = 0.01
        elif health.health_score > 0.5:
            base_prob = 0.05
        else:
            base_prob = 0.15

        # 退化加速因子
        if health.trend == 'degrading':
            base_prob *= 1.5

        return min(1.0, base_prob)

    def _determine_health_trend(self, component_id: str) -> str:
        """确定健康趋势"""
        # 从历史状态中分析趋势
        if len(self.state_history) < 3:
            return "stable"

        recent_scores = []
        for state in self.state_history[-24:]:  # 最近24小时
            if component_id in state.equipment_health:
                recent_scores.append(
                    state.equipment_health[component_id].health_score
                )

        if len(recent_scores) < 2:
            return "stable"

        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if trend < -0.01:
            return "degrading"
        elif trend > 0.01:
            return "improving"
        else:
            return "stable"

    def _assess_system_health(self) -> Tuple[SystemHealthLevel, float]:
        """评估系统健康度"""
        if not self.equipment_registry:
            return SystemHealthLevel.GOOD, 0.85

        # 计算加权健康分数
        total_score = 0.0
        total_weight = 0.0

        weights = {
            'gate': 3.0,       # 闸门权重高
            'sensor': 2.0,    # 传感器中等
            'controller': 2.5  # 控制器较高
        }

        for comp_id, health in self.equipment_registry.items():
            weight = 1.0
            for key, w in weights.items():
                if key in comp_id:
                    weight = w
                    break

            total_score += health.health_score * weight
            total_weight += weight

        health_score = total_score / total_weight if total_weight > 0 else 0.85

        # 映射到健康等级
        if health_score > 0.95:
            level = SystemHealthLevel.EXCELLENT
        elif health_score > 0.85:
            level = SystemHealthLevel.GOOD
        elif health_score > 0.70:
            level = SystemHealthLevel.FAIR
        elif health_score > 0.50:
            level = SystemHealthLevel.DEGRADED
        else:
            level = SystemHealthLevel.CRITICAL

        return level, health_score

    def _identify_seasonal_pattern(self, external_factors: Dict = None) -> SeasonalPattern:
        """识别季节模式"""
        if external_factors is None:
            external_factors = {}

        # 获取月份
        month = external_factors.get('month', datetime.now().month)
        temperature = external_factors.get('temperature', 20)

        # 基于月份和温度判断
        if month in [3, 4, 5]:
            if external_factors.get('flood_warning', False):
                return SeasonalPattern.SPRING_FLOOD
            return SeasonalPattern.TRANSITION

        elif month in [6, 7, 8]:
            return SeasonalPattern.SUMMER_PEAK

        elif month in [9, 10, 11]:
            return SeasonalPattern.AUTUMN_NORMAL

        else:  # 12, 1, 2
            if temperature < 0:
                return SeasonalPattern.WINTER_ICE
            return SeasonalPattern.TRANSITION

    def _assess_network_risk(self,
                             system_state: Dict,
                             forecast_data: Dict = None,
                             fault_reports: List[Dict] = None) -> NetworkRiskAssessment:
        """评估全网风险"""
        risk_factors = []
        risk_score = 0.0

        # 1. 设备风险
        equipment_risk = self._calculate_equipment_risk()
        risk_score += equipment_risk * 30  # 30%权重
        if equipment_risk > 0.5:
            risk_factors.append({
                'factor': '设备健康度下降',
                'score': equipment_risk * 100,
                'components': self._get_degraded_components()
            })

        # 2. 水位风险
        level_risk = self._calculate_level_risk(system_state)
        risk_score += level_risk * 25  # 25%权重
        if level_risk > 0.3:
            risk_factors.append({
                'factor': '水位异常风险',
                'score': level_risk * 100,
                'details': self._get_level_risk_details(system_state)
            })

        # 3. 天气风险
        weather_risk = self._calculate_weather_risk(forecast_data)
        risk_score += weather_risk * 25  # 25%权重
        if weather_risk > 0.3:
            risk_factors.append({
                'factor': '天气风险',
                'score': weather_risk * 100,
                'forecast': forecast_data.get('weather_warning', '') if forecast_data else ''
            })

        # 4. 故障历史风险
        fault_risk = self._calculate_fault_history_risk(fault_reports)
        risk_score += fault_risk * 20  # 20%权重
        if fault_risk > 0.3:
            risk_factors.append({
                'factor': '近期故障频繁',
                'score': fault_risk * 100,
                'count': len(fault_reports) if fault_reports else 0
            })

        # 确定风险等级
        if risk_score > 80:
            overall_risk = RiskLevel.EXTREME
        elif risk_score > 60:
            overall_risk = RiskLevel.SEVERE
        elif risk_score > 40:
            overall_risk = RiskLevel.HIGH
        elif risk_score > 25:
            overall_risk = RiskLevel.MODERATE
        elif risk_score > 10:
            overall_risk = RiskLevel.LOW
        else:
            overall_risk = RiskLevel.MINIMAL

        # 各池风险
        risk_by_pool = {}
        for i in range(self.num_pools):
            pool_risk = self._calculate_pool_risk(i, system_state)
            risk_by_pool[f"pool_{i}"] = pool_risk

        # 风险趋势
        risk_trend = self._determine_risk_trend()

        if overall_risk.value >= RiskLevel.HIGH.value:
            self.stats['risk_alerts'] += 1

        return NetworkRiskAssessment(
            overall_risk=overall_risk,
            risk_score=risk_score,
            top_risks=sorted(risk_factors, key=lambda x: x['score'], reverse=True)[:3],
            risk_by_pool=risk_by_pool,
            risk_trend=risk_trend
        )

    def _calculate_equipment_risk(self) -> float:
        """计算设备风险"""
        if not self.equipment_registry:
            return 0.0

        # 关键设备的故障概率加权
        total_risk = 0.0
        critical_count = 0

        for comp_id, health in self.equipment_registry.items():
            if 'gate' in comp_id:  # 闸门是关键设备
                total_risk += health.failure_probability * 2
                critical_count += 2
            else:
                total_risk += health.failure_probability
                critical_count += 1

        return total_risk / critical_count if critical_count > 0 else 0.0

    def _get_degraded_components(self) -> List[str]:
        """获取退化组件列表"""
        degraded = []
        for comp_id, health in self.equipment_registry.items():
            if health.health_score < 0.7 or health.trend == 'degrading':
                degraded.append(comp_id)
        return degraded

    def _calculate_level_risk(self, system_state: Dict) -> float:
        """计算水位风险"""
        levels = system_state.get('water_levels', [])
        if not levels:
            return 0.0

        risk = 0.0
        for level in levels:
            if level is None:
                continue
            # 假设正常范围 1-7m，上下限 0.5-8m
            if level < 1.0:
                risk += (1.0 - level) * 0.5  # 低水位风险
            elif level > 7.0:
                risk += (level - 7.0) * 0.5  # 高水位风险

        return min(1.0, risk / len(levels)) if levels else 0.0

    def _get_level_risk_details(self, system_state: Dict) -> Dict:
        """获取水位风险详情"""
        levels = system_state.get('water_levels', [])
        details = {'low_risk_pools': [], 'high_risk_pools': []}

        for i, level in enumerate(levels):
            if level is None:
                continue
            if level < 1.0:
                details['low_risk_pools'].append(f"pool_{i}")
            elif level > 7.0:
                details['high_risk_pools'].append(f"pool_{i}")

        return details

    def _calculate_weather_risk(self, forecast_data: Dict = None) -> float:
        """计算天气风险"""
        if forecast_data is None:
            return 0.0

        risk = 0.0

        # 降雨风险
        rainfall = forecast_data.get('max_rainfall', 0)
        if rainfall > 50:  # mm/h
            risk += 0.8
        elif rainfall > 20:
            risk += 0.4
        elif rainfall > 10:
            risk += 0.2

        # 温度风险
        temp = forecast_data.get('min_temperature', 20)
        if temp < -10:
            risk += 0.5  # 严重冰冻
        elif temp < 0:
            risk += 0.3  # 冰冻

        return min(1.0, risk)

    def _calculate_fault_history_risk(self, fault_reports: List[Dict] = None) -> float:
        """计算故障历史风险"""
        if not fault_reports:
            return 0.0

        # 最近24小时故障数
        recent_faults = len(fault_reports)

        if recent_faults > 10:
            return 0.9
        elif recent_faults > 5:
            return 0.6
        elif recent_faults > 2:
            return 0.3
        else:
            return 0.1

    def _calculate_pool_risk(self, pool_id: int, system_state: Dict) -> float:
        """计算单个池的风险"""
        risk = 0.0

        # 水位风险
        levels = system_state.get('water_levels', [])
        if pool_id < len(levels) and levels[pool_id] is not None:
            level = levels[pool_id]
            if level < 1.0 or level > 7.0:
                risk += 0.3

        # 设备风险
        gate_health = self.equipment_registry.get(f"gate_{pool_id}")
        if gate_health and gate_health.health_score < 0.8:
            risk += 0.3

        sensor_health = self.equipment_registry.get(f"level_sensor_{pool_id}")
        if sensor_health and sensor_health.health_score < 0.8:
            risk += 0.2

        return min(1.0, risk)

    def _determine_risk_trend(self) -> str:
        """确定风险趋势"""
        if len(self.state_history) < 3:
            return "stable"

        recent_risks = [
            state.risk_assessment.risk_score
            for state in self.state_history[-12:]
        ]

        if len(recent_risks) < 2:
            return "stable"

        trend = np.polyfit(range(len(recent_risks)), recent_risks, 1)[0]

        if trend > 2:
            return "increasing"
        elif trend < -2:
            return "decreasing"
        else:
            return "stable"

    def _generate_strategic_advice(self,
                                   system_health: SystemHealthLevel,
                                   risk_assessment: NetworkRiskAssessment,
                                   seasonal_pattern: SeasonalPattern,
                                   forecast_data: Dict = None) -> List[StrategicAdvice]:
        """生成战略建议"""
        advisories = []
        advice_id = 0

        # 1. 基于系统健康度的建议
        if system_health in [SystemHealthLevel.DEGRADED, SystemHealthLevel.CRITICAL]:
            advisories.append(StrategicAdvice(
                advice_id=f"ADV_{advice_id:03d}",
                category="maintenance",
                priority=5,
                description="系统健康度下降，建议安排维护",
                rationale=f"系统健康等级: {system_health.value}",
                recommended_action="安排预防性维护检查",
                constraint_adjustments={
                    'safety_margin': 1.2,  # 增加安全裕度
                    'response_time': 0.8   # 缩短响应时间
                }
            ))
            advice_id += 1

        # 2. 基于风险评估的建议
        if risk_assessment.overall_risk.value >= RiskLevel.HIGH.value:
            advisories.append(StrategicAdvice(
                advice_id=f"ADV_{advice_id:03d}",
                category="safety",
                priority=5,
                description="全网风险等级较高",
                rationale=f"风险分数: {risk_assessment.risk_score:.1f}",
                recommended_action="启动风险缓解措施",
                constraint_adjustments={
                    'max_level_change_rate': 0.2,  # 限制变化率
                    'min_reserve': 0.3  # 增加最小储备
                }
            ))
            advice_id += 1

        # 3. 基于季节模式的建议
        if seasonal_pattern == SeasonalPattern.SPRING_FLOOD:
            advisories.append(StrategicAdvice(
                advice_id=f"ADV_{advice_id:03d}",
                category="scheduling",
                priority=4,
                description="春汛期间调度建议",
                rationale="春季融雪和降雨增多",
                recommended_action="提前腾空库容，准备防洪",
                constraint_adjustments={
                    'target_level_reduction': 0.5,  # 目标水位降低
                    'pre_release_enabled': True
                }
            ))
            advice_id += 1

        elif seasonal_pattern == SeasonalPattern.SUMMER_PEAK:
            advisories.append(StrategicAdvice(
                advice_id=f"ADV_{advice_id:03d}",
                category="efficiency",
                priority=3,
                description="夏季用水高峰调度优化",
                rationale="用水需求增加",
                recommended_action="优化供水调度，确保供需平衡",
                constraint_adjustments={
                    'demand_priority': 1.2  # 提高需求满足权重
                }
            ))
            advice_id += 1

        elif seasonal_pattern == SeasonalPattern.WINTER_ICE:
            advisories.append(StrategicAdvice(
                advice_id=f"ADV_{advice_id:03d}",
                category="scheduling",
                priority=4,
                description="冰期运行调度建议",
                rationale="冰盖影响过流能力",
                recommended_action="降低流速，增加安全裕度",
                constraint_adjustments={
                    'flow_efficiency': 0.75,  # 流速效率降低
                    'max_flow_rate': 0.8,      # 最大流量限制
                    'level_stability_weight': 1.3  # 增加水位稳定权重
                }
            ))
            advice_id += 1

        # 4. 基于设备退化的建议
        degraded = self._get_components_needing_maintenance()
        if degraded:
            advisories.append(StrategicAdvice(
                advice_id=f"ADV_{advice_id:03d}",
                category="maintenance",
                priority=3,
                description=f"建议对{len(degraded)}个组件进行维护",
                rationale="设备健康度下降或接近寿命末期",
                recommended_action=f"安排维护: {', '.join(degraded[:3])}",
                constraint_adjustments={
                    'redundancy_required': True
                }
            ))
            advice_id += 1

        # 5. 基于天气预报的建议
        if forecast_data:
            max_rainfall = forecast_data.get('max_rainfall', 0)
            if max_rainfall > 20:
                advisories.append(StrategicAdvice(
                    advice_id=f"ADV_{advice_id:03d}",
                    category="scheduling",
                    priority=4 if max_rainfall > 50 else 3,
                    description="暴雨预警调度建议",
                    rationale=f"预报最大降雨: {max_rainfall}mm/h",
                    recommended_action="启动预泄程序，腾空库容",
                    constraint_adjustments={
                        'pre_release_volume': max_rainfall * 100,  # 预泄量
                        'flood_control_mode': True
                    }
                ))
                advice_id += 1

        self.stats['advisories_generated'] += len(advisories)

        # 按优先级排序
        advisories.sort(key=lambda x: x.priority, reverse=True)

        return advisories

    def _get_components_needing_maintenance(self) -> List[str]:
        """获取需要维护的组件"""
        components = []

        for comp_id, health in self.equipment_registry.items():
            # 健康度低于70%或剩余寿命少于30天
            if (health.health_score < 0.7 or
                health.remaining_useful_life < self.config['degradation_alert_days']):
                components.append(comp_id)

        return components

    def get_scheduling_constraints(self) -> Dict:
        """
        获取调度约束调整

        Returns:
            Dict: 建议的约束调整
        """
        if not self.current_state:
            return {}

        constraints = {}

        # 合并所有建议的约束调整
        for advisory in self.current_state.active_advisories:
            for key, value in advisory.constraint_adjustments.items():
                if key not in constraints:
                    constraints[key] = value
                else:
                    # 对于数值型，取更保守的值
                    if isinstance(value, (int, float)):
                        if 'max' in key or 'efficiency' in key:
                            constraints[key] = min(constraints[key], value)
                        else:
                            constraints[key] = max(constraints[key], value)

        return constraints

    def get_risk_adjusted_targets(self,
                                   base_targets: Dict,
                                   risk_tolerance: float = 0.5) -> Dict:
        """
        获取风险调整后的目标

        Args:
            base_targets: 基础目标
            risk_tolerance: 风险容忍度 (0-1)

        Returns:
            Dict: 调整后的目标
        """
        if not self.current_state:
            return base_targets

        adjusted = base_targets.copy()
        risk_level = self.current_state.risk_assessment.overall_risk.value

        # 根据风险等级调整
        if risk_level >= RiskLevel.HIGH.value:
            # 高风险：更保守的目标
            if 'target_levels' in adjusted:
                adjusted['target_levels'] = [
                    l * (1 - 0.1 * (1 - risk_tolerance))
                    for l in adjusted['target_levels']
                ]
            if 'max_flow' in adjusted:
                adjusted['max_flow'] *= (0.8 + 0.2 * risk_tolerance)

        return adjusted

    def update_from_fault(self, fault_report: Dict):
        """从故障报告更新认知"""
        self.fault_history.append({
            **fault_report,
            'timestamp': datetime.now().isoformat()
        })

        # 限制历史长度
        if len(self.fault_history) > 1000:
            self.fault_history = self.fault_history[-1000:]

        # 更新相关设备健康度
        component = fault_report.get('component', '')
        if component in self.equipment_registry:
            severity = fault_report.get('severity', 1)
            self.equipment_registry[component].health_score *= (1 - severity * 0.05)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'current_health_level': (
                self.current_state.system_health.value
                if self.current_state else 'unknown'
            ),
            'current_risk_level': (
                self.current_state.risk_assessment.overall_risk.value
                if self.current_state else 0
            ),
            'active_advisories': (
                len(self.current_state.active_advisories)
                if self.current_state else 0
            ),
            'degraded_components': len(self._get_degraded_components()),
            'state_history_length': len(self.state_history)
        }
