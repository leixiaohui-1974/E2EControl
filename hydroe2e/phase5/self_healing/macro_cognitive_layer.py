"""
Macro Cognitive Layer for L3 Centralized Scheduler

Implements system-level cognitive capabilities for strategic decision-making
at the L3 (centralized) level of the hierarchical control architecture.

Key features:
- System health assessment and trending
- Equipment degradation tracking
- Seasonal pattern recognition
- Network-wide risk assessment
- Strategic scheduling advice
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SystemHealthLevel(Enum):
    """Overall system health classification"""
    EXCELLENT = "excellent"      # >95% health
    GOOD = "good"                # 85-95% health
    FAIR = "fair"                # 70-85% health
    DEGRADED = "degraded"        # 50-70% health
    CRITICAL = "critical"        # <50% health


class SeasonalPattern(Enum):
    """Seasonal operational patterns"""
    SPRING_FLOOD = "spring_flood"        # High inflow, flood risk
    SUMMER_PEAK = "summer_peak"          # Peak demand season
    AUTUMN_NORMAL = "autumn_normal"      # Normal operations
    WINTER_ICE = "winter_ice"            # Ice formation risk
    TRANSITION = "transition"            # Between seasons


class RiskLevel(Enum):
    """Risk level classification"""
    MINIMAL = "minimal"      # No significant risk
    LOW = "low"              # Low risk, monitor
    MEDIUM = "medium"        # Moderate risk, prepare
    HIGH = "high"            # High risk, take action
    CRITICAL = "critical"    # Critical risk, emergency


class TrendDirection(Enum):
    """Trend direction for metrics"""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    RAPID_DEGRADING = "rapid_degrading"


@dataclass
class EquipmentHealth:
    """Health status for individual equipment"""
    equipment_id: str
    current_health: float  # 0-1
    degradation_trend: float  # Negative = degrading
    predicted_failure_date: Optional[datetime]
    maintenance_priority: int  # 1=highest
    last_maintenance: Optional[datetime]
    operating_hours: float = 0
    failure_history: List[datetime] = field(default_factory=list)


@dataclass
class DegradationTrend:
    """Degradation trend analysis"""
    component_id: str
    current_value: float
    trend_direction: TrendDirection
    rate_per_day: float
    predicted_threshold_breach: Optional[datetime]
    contributing_factors: List[str]
    confidence: float


@dataclass
class NetworkRiskAssessment:
    """Network-wide risk assessment"""
    assessment_id: str
    overall_risk: RiskLevel
    risk_factors: Dict[str, float]  # Factor -> contribution
    vulnerable_components: List[str]
    mitigation_recommendations: List[str]
    assessment_time: datetime
    valid_until: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=4))


@dataclass
class StrategicAdvice:
    """Strategic advice for scheduling"""
    advice_id: str
    advice_type: str  # maintenance, capacity, configuration, etc.
    priority: int  # 1=highest
    description: str
    affected_components: List[str]
    expected_benefit: float  # 0-1 improvement
    implementation_cost: float
    time_horizon: timedelta
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class MacroCognitiveState:
    """Complete macro cognitive state"""
    timestamp: datetime
    system_health: SystemHealthLevel
    health_score: float
    seasonal_pattern: SeasonalPattern
    overall_risk: RiskLevel
    equipment_health: Dict[str, EquipmentHealth]
    degradation_trends: List[DegradationTrend]
    risk_assessment: NetworkRiskAssessment
    strategic_advice: List[StrategicAdvice]
    scheduling_constraints: Dict[str, Any]
    confidence: float


class MacroCognitiveLayer:
    """
    Macro Cognitive Layer for L3 scheduling decisions.

    Provides system-level awareness and strategic recommendations:
    - Aggregates equipment health across the network
    - Identifies seasonal patterns and adjusts strategies
    - Assesses network-wide risks
    - Generates strategic scheduling constraints
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Health thresholds
        self.health_thresholds = {
            SystemHealthLevel.EXCELLENT: 0.95,
            SystemHealthLevel.GOOD: 0.85,
            SystemHealthLevel.FAIR: 0.70,
            SystemHealthLevel.DEGRADED: 0.50,
            SystemHealthLevel.CRITICAL: 0.0
        }

        # Risk weights
        self.risk_weights = {
            'equipment_failure': 0.3,
            'capacity_shortage': 0.25,
            'external_events': 0.2,
            'maintenance_backlog': 0.15,
            'communication_issues': 0.1
        }

        # State tracking
        self.equipment_registry: Dict[str, EquipmentHealth] = {}
        self.historical_health: List[Tuple[datetime, float]] = []
        self.current_state: Optional[MacroCognitiveState] = None

        # Seasonal detection parameters
        self.season_indicators = {
            SeasonalPattern.SPRING_FLOOD: {'month_range': (3, 5), 'inflow_threshold': 1.5},
            SeasonalPattern.SUMMER_PEAK: {'month_range': (6, 8), 'demand_threshold': 1.3},
            SeasonalPattern.AUTUMN_NORMAL: {'month_range': (9, 11)},
            SeasonalPattern.WINTER_ICE: {'month_range': (12, 2), 'temp_threshold': 0}
        }

    def assess(
        self,
        system_state: Dict[str, Any],
        forecast_data: Dict[str, Any],
        fault_reports: List[Dict[str, Any]],
        external_factors: Dict[str, Any]
    ) -> MacroCognitiveState:
        """
        Perform comprehensive system assessment.

        Args:
            system_state: Current system state including equipment health
            forecast_data: Demand and weather forecasts
            fault_reports: Recent fault reports
            external_factors: External conditions (weather, events, etc.)

        Returns:
            Complete macro cognitive state assessment
        """
        timestamp = datetime.now()

        # 1. Update equipment health registry
        self._update_equipment_health(system_state.get('equipment_health', {}))

        # 2. Calculate overall system health
        health_score = self._calculate_system_health()
        system_health = self._classify_health_level(health_score)

        # 3. Detect seasonal pattern
        seasonal_pattern = self._detect_seasonal_pattern(
            timestamp, forecast_data, external_factors
        )

        # 4. Analyze degradation trends
        degradation_trends = self._analyze_degradation_trends()

        # 5. Perform risk assessment
        risk_assessment = self._assess_network_risk(
            system_state, forecast_data, fault_reports, external_factors
        )

        # 6. Generate strategic advice
        strategic_advice = self._generate_strategic_advice(
            system_health, seasonal_pattern, risk_assessment, degradation_trends
        )

        # 7. Generate scheduling constraints
        scheduling_constraints = self._generate_scheduling_constraints(
            system_health, seasonal_pattern, risk_assessment
        )

        # Calculate confidence
        confidence = self._calculate_assessment_confidence(
            len(self.equipment_registry),
            len(fault_reports),
            forecast_data
        )

        self.current_state = MacroCognitiveState(
            timestamp=timestamp,
            system_health=system_health,
            health_score=health_score,
            seasonal_pattern=seasonal_pattern,
            overall_risk=risk_assessment.overall_risk,
            equipment_health=dict(self.equipment_registry),
            degradation_trends=degradation_trends,
            risk_assessment=risk_assessment,
            strategic_advice=strategic_advice,
            scheduling_constraints=scheduling_constraints,
            confidence=confidence
        )

        # Store historical health
        self.historical_health.append((timestamp, health_score))
        if len(self.historical_health) > 1000:
            self.historical_health = self.historical_health[-500:]

        return self.current_state

    def _update_equipment_health(self, health_data: Dict[str, float]):
        """Update equipment health registry"""
        for equip_id, health_value in health_data.items():
            if equip_id in self.equipment_registry:
                # Update existing
                equip = self.equipment_registry[equip_id]
                old_health = equip.current_health
                equip.current_health = health_value

                # Calculate degradation trend
                if old_health > 0:
                    equip.degradation_trend = (health_value - old_health)
            else:
                # Create new entry
                self.equipment_registry[equip_id] = EquipmentHealth(
                    equipment_id=equip_id,
                    current_health=health_value,
                    degradation_trend=0.0,
                    predicted_failure_date=None,
                    maintenance_priority=5,  # Default low priority
                    last_maintenance=None
                )

    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        if not self.equipment_registry:
            return 1.0

        # Weighted average based on equipment importance
        total_weight = 0
        weighted_health = 0

        for equip_id, equip in self.equipment_registry.items():
            # Higher priority equipment has more weight
            weight = 1.0 / equip.maintenance_priority if equip.maintenance_priority > 0 else 1.0
            weighted_health += equip.current_health * weight
            total_weight += weight

        return weighted_health / total_weight if total_weight > 0 else 1.0

    def _classify_health_level(self, health_score: float) -> SystemHealthLevel:
        """Classify health score into level"""
        for level, threshold in self.health_thresholds.items():
            if health_score >= threshold:
                return level
        return SystemHealthLevel.CRITICAL

    def _detect_seasonal_pattern(
        self,
        timestamp: datetime,
        forecast_data: Dict[str, Any],
        external_factors: Dict[str, Any]
    ) -> SeasonalPattern:
        """Detect current seasonal pattern"""
        month = timestamp.month

        # Check explicit season from external factors
        if 'season' in external_factors:
            season_map = {
                'spring': SeasonalPattern.SPRING_FLOOD,
                'summer': SeasonalPattern.SUMMER_PEAK,
                'autumn': SeasonalPattern.AUTUMN_NORMAL,
                'fall': SeasonalPattern.AUTUMN_NORMAL,
                'winter': SeasonalPattern.WINTER_ICE
            }
            return season_map.get(
                external_factors['season'].lower(),
                SeasonalPattern.TRANSITION
            )

        # Check by month
        if month in [3, 4, 5]:
            return SeasonalPattern.SPRING_FLOOD
        elif month in [6, 7, 8]:
            return SeasonalPattern.SUMMER_PEAK
        elif month in [9, 10, 11]:
            return SeasonalPattern.AUTUMN_NORMAL
        else:
            return SeasonalPattern.WINTER_ICE

    def _analyze_degradation_trends(self) -> List[DegradationTrend]:
        """Analyze equipment degradation trends"""
        trends = []

        for equip_id, equip in self.equipment_registry.items():
            if equip.degradation_trend < -0.001:  # Significant degradation
                # Determine trend direction
                if equip.degradation_trend < -0.01:
                    direction = TrendDirection.RAPID_DEGRADING
                else:
                    direction = TrendDirection.DEGRADING

                # Predict threshold breach
                if equip.current_health > 0.3 and equip.degradation_trend < 0:
                    days_to_threshold = (equip.current_health - 0.3) / abs(equip.degradation_trend)
                    breach_date = datetime.now() + timedelta(days=days_to_threshold)
                else:
                    breach_date = None

                trends.append(DegradationTrend(
                    component_id=equip_id,
                    current_value=equip.current_health,
                    trend_direction=direction,
                    rate_per_day=equip.degradation_trend,
                    predicted_threshold_breach=breach_date,
                    contributing_factors=['aging', 'usage'],
                    confidence=0.8
                ))
            elif abs(equip.degradation_trend) < 0.0001:
                trends.append(DegradationTrend(
                    component_id=equip_id,
                    current_value=equip.current_health,
                    trend_direction=TrendDirection.STABLE,
                    rate_per_day=equip.degradation_trend,
                    predicted_threshold_breach=None,
                    contributing_factors=[],
                    confidence=0.9
                ))

        return trends

    def _assess_network_risk(
        self,
        system_state: Dict[str, Any],
        forecast_data: Dict[str, Any],
        fault_reports: List[Dict[str, Any]],
        external_factors: Dict[str, Any]
    ) -> NetworkRiskAssessment:
        """Assess network-wide risk"""
        risk_factors = {}

        # Equipment failure risk
        degraded_count = sum(
            1 for e in self.equipment_registry.values()
            if e.current_health < 0.7
        )
        risk_factors['equipment_failure'] = min(1.0, degraded_count * 0.2)

        # Capacity risk
        network_load = system_state.get('network_load', 0.5)
        risk_factors['capacity_shortage'] = max(0, (network_load - 0.7) / 0.3)

        # External events risk
        if 'extreme_weather' in external_factors:
            risk_factors['external_events'] = 0.8
        else:
            risk_factors['external_events'] = 0.1

        # Maintenance backlog
        overdue_maintenance = sum(
            1 for e in self.equipment_registry.values()
            if e.last_maintenance and
            (datetime.now() - e.last_maintenance).days > 90
        )
        risk_factors['maintenance_backlog'] = min(1.0, overdue_maintenance * 0.15)

        # Recent fault activity
        risk_factors['fault_activity'] = min(1.0, len(fault_reports) * 0.1)

        # Calculate weighted risk
        total_risk = sum(
            risk_factors.get(factor, 0) * weight
            for factor, weight in self.risk_weights.items()
        )

        # Classify risk level
        if total_risk > 0.8:
            overall_risk = RiskLevel.CRITICAL
        elif total_risk > 0.6:
            overall_risk = RiskLevel.HIGH
        elif total_risk > 0.4:
            overall_risk = RiskLevel.MEDIUM
        elif total_risk > 0.2:
            overall_risk = RiskLevel.LOW
        else:
            overall_risk = RiskLevel.MINIMAL

        # Identify vulnerable components
        vulnerable = [
            e.equipment_id for e in self.equipment_registry.values()
            if e.current_health < 0.6
        ]

        # Generate recommendations
        recommendations = []
        if risk_factors.get('equipment_failure', 0) > 0.3:
            recommendations.append("Schedule preventive maintenance for degraded equipment")
        if risk_factors.get('capacity_shortage', 0) > 0.3:
            recommendations.append("Consider load shedding or backup activation")
        if risk_factors.get('maintenance_backlog', 0) > 0.3:
            recommendations.append("Address overdue maintenance items")

        return NetworkRiskAssessment(
            assessment_id=f"risk_{datetime.now().strftime('%Y%m%d_%H%M')}",
            overall_risk=overall_risk,
            risk_factors=risk_factors,
            vulnerable_components=vulnerable,
            mitigation_recommendations=recommendations,
            assessment_time=datetime.now()
        )

    def _generate_strategic_advice(
        self,
        system_health: SystemHealthLevel,
        seasonal_pattern: SeasonalPattern,
        risk_assessment: NetworkRiskAssessment,
        degradation_trends: List[DegradationTrend]
    ) -> List[StrategicAdvice]:
        """Generate strategic advice based on current state"""
        advice_list = []

        # Health-based advice
        if system_health in [SystemHealthLevel.DEGRADED, SystemHealthLevel.CRITICAL]:
            advice_list.append(StrategicAdvice(
                advice_id="adv_health_001",
                advice_type="maintenance",
                priority=1,
                description="System health is low - prioritize maintenance activities",
                affected_components=[
                    e.equipment_id for e in self.equipment_registry.values()
                    if e.current_health < 0.6
                ],
                expected_benefit=0.3,
                implementation_cost=10000,
                time_horizon=timedelta(days=7)
            ))

        # Seasonal advice
        if seasonal_pattern == SeasonalPattern.SPRING_FLOOD:
            advice_list.append(StrategicAdvice(
                advice_id="adv_season_flood",
                advice_type="capacity",
                priority=2,
                description="Spring flood season - maintain buffer capacity for high inflows",
                affected_components=['all_reservoirs'],
                expected_benefit=0.2,
                implementation_cost=0,
                time_horizon=timedelta(days=60)
            ))
        elif seasonal_pattern == SeasonalPattern.WINTER_ICE:
            advice_list.append(StrategicAdvice(
                advice_id="adv_season_ice",
                advice_type="configuration",
                priority=2,
                description="Winter ice season - adjust flow parameters for ice conditions",
                affected_components=['all_channels'],
                expected_benefit=0.15,
                implementation_cost=500,
                time_horizon=timedelta(days=90)
            ))

        # Risk-based advice
        if risk_assessment.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            advice_list.append(StrategicAdvice(
                advice_id="adv_risk_001",
                advice_type="protection",
                priority=1,
                description="High risk detected - activate protective measures",
                affected_components=risk_assessment.vulnerable_components,
                expected_benefit=0.4,
                implementation_cost=2000,
                time_horizon=timedelta(hours=24)
            ))

        # Degradation-based advice
        rapid_degrading = [
            t for t in degradation_trends
            if t.trend_direction == TrendDirection.RAPID_DEGRADING
        ]
        if rapid_degrading:
            advice_list.append(StrategicAdvice(
                advice_id="adv_degrade_001",
                advice_type="urgent_maintenance",
                priority=1,
                description="Rapid degradation detected - immediate attention required",
                affected_components=[t.component_id for t in rapid_degrading],
                expected_benefit=0.5,
                implementation_cost=5000,
                time_horizon=timedelta(days=3)
            ))

        # Sort by priority
        advice_list.sort(key=lambda a: a.priority)
        return advice_list

    def _generate_scheduling_constraints(
        self,
        system_health: SystemHealthLevel,
        seasonal_pattern: SeasonalPattern,
        risk_assessment: NetworkRiskAssessment
    ) -> Dict[str, Any]:
        """Generate constraints for L3 scheduler"""
        constraints = {}

        # Health-based constraints
        health_factor = {
            SystemHealthLevel.EXCELLENT: 1.0,
            SystemHealthLevel.GOOD: 0.95,
            SystemHealthLevel.FAIR: 0.85,
            SystemHealthLevel.DEGRADED: 0.7,
            SystemHealthLevel.CRITICAL: 0.5
        }.get(system_health, 0.8)

        constraints['capacity_factor'] = health_factor
        constraints['safety_margin'] = 1.0 - health_factor + 0.1

        # Seasonal constraints
        if seasonal_pattern == SeasonalPattern.SPRING_FLOOD:
            constraints['min_buffer_capacity'] = 0.3
            constraints['max_release_rate'] = 0.8
        elif seasonal_pattern == SeasonalPattern.SUMMER_PEAK:
            constraints['min_storage_level'] = 0.4
            constraints['demand_priority'] = 'high'
        elif seasonal_pattern == SeasonalPattern.WINTER_ICE:
            constraints['flow_efficiency_factor'] = 0.85
            constraints['ice_formation_risk'] = True

        # Risk constraints
        if risk_assessment.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            constraints['conservative_mode'] = True
            constraints['backup_required'] = True

        # Equipment constraints
        constraints['unavailable_equipment'] = [
            e.equipment_id for e in self.equipment_registry.values()
            if e.current_health < 0.3
        ]

        constraints['degraded_equipment'] = [
            e.equipment_id for e in self.equipment_registry.values()
            if 0.3 <= e.current_health < 0.7
        ]

        return constraints

    def _calculate_assessment_confidence(
        self,
        equipment_count: int,
        fault_count: int,
        forecast_data: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the assessment"""
        confidence = 0.5  # Base confidence

        # More equipment data = higher confidence
        if equipment_count > 10:
            confidence += 0.2
        elif equipment_count > 5:
            confidence += 0.1

        # Forecast data improves confidence
        if forecast_data.get('demand_forecast'):
            confidence += 0.1
        if forecast_data.get('weather_forecast'):
            confidence += 0.1

        # Recent faults reduce confidence
        if fault_count > 5:
            confidence -= 0.1

        return min(1.0, max(0.3, confidence))

    def get_scheduling_constraints(self) -> Dict[str, Any]:
        """Get current scheduling constraints"""
        if self.current_state:
            return self.current_state.scheduling_constraints
        return {'capacity_factor': 1.0, 'safety_margin': 0.1}

    def get_risk_adjusted_targets(
        self,
        base_targets: Dict[str, float],
        risk_tolerance: float = 0.5
    ) -> Dict[str, float]:
        """
        Adjust targets based on current risk level.

        Args:
            base_targets: Base target values
            risk_tolerance: How much risk to tolerate (0=conservative, 1=aggressive)

        Returns:
            Risk-adjusted targets
        """
        if not self.current_state:
            return base_targets

        # Risk adjustment factor
        risk_factor = {
            RiskLevel.MINIMAL: 1.0,
            RiskLevel.LOW: 0.95,
            RiskLevel.MEDIUM: 0.85,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.5
        }.get(self.current_state.overall_risk, 0.8)

        # Blend with risk tolerance
        adjustment = risk_factor + (1.0 - risk_factor) * risk_tolerance

        adjusted = {}
        for key, value in base_targets.items():
            adjusted[key] = value * adjustment

        return adjusted

    def get_maintenance_priority_list(self) -> List[Tuple[str, int, float]]:
        """
        Get prioritized list of equipment needing maintenance.

        Returns:
            List of (equipment_id, priority, health) tuples
        """
        priority_list = []

        for equip_id, equip in self.equipment_registry.items():
            # Calculate priority based on health and degradation
            base_priority = 5 - int(equip.current_health * 4)

            # Adjust for rapid degradation
            if equip.degradation_trend < -0.01:
                base_priority = min(base_priority + 2, 5)

            priority_list.append((
                equip_id,
                base_priority,
                equip.current_health
            ))

        # Sort by priority (highest first)
        priority_list.sort(key=lambda x: (-x[1], x[2]))
        return priority_list
