"""
Predictive Recovery Module for L4 Self-Healing

Implements predictive recovery strategies that anticipate system needs
and proactively initiate recovery before failures become critical.

Key features:
- Failure trajectory prediction
- Pre-emptive resource allocation
- Recovery timeline optimization
- Multi-scenario recovery planning
- Adaptive recovery speed control
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RecoveryPhase(Enum):
    """Recovery process phases"""
    DETECTION = "detection"           # Fault detected
    ASSESSMENT = "assessment"         # Impact assessment
    PLANNING = "planning"             # Recovery planning
    PREPARATION = "preparation"       # Resource preparation
    EXECUTION = "execution"           # Active recovery
    VERIFICATION = "verification"     # Recovery verification
    STABILIZATION = "stabilization"   # System stabilization
    COMPLETE = "complete"             # Recovery complete


class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    IMMEDIATE = "immediate"           # Instant switchover
    GRADUAL = "gradual"               # Gradual transition
    STAGED = "staged"                 # Multi-stage recovery
    PARALLEL = "parallel"             # Parallel path activation
    ROLLBACK = "rollback"             # Rollback to known state
    ADAPTIVE = "adaptive"             # Dynamically adjusted


class RecoveryPriority(Enum):
    """Recovery priority levels"""
    CRITICAL = 1      # Life safety, immediate
    HIGH = 2          # Major service impact
    MEDIUM = 3        # Moderate impact
    LOW = 4           # Minor impact
    DEFERRED = 5      # Can be delayed


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    HIGH = "high"           # >90% confidence
    MEDIUM = "medium"       # 70-90% confidence
    LOW = "low"             # 50-70% confidence
    UNCERTAIN = "uncertain" # <50% confidence


@dataclass
class FailureTrajectory:
    """Predicted failure trajectory"""
    component_id: str
    current_health: float  # 0-1
    predicted_health: List[Tuple[datetime, float]]  # Timeline
    time_to_failure: Optional[timedelta]
    failure_probability: float
    degradation_rate: float  # Health loss per hour
    confidence: PredictionConfidence
    contributing_factors: List[str]

    def get_health_at(self, time: datetime) -> float:
        """Get predicted health at a specific time"""
        if not self.predicted_health:
            return self.current_health

        for pred_time, health in self.predicted_health:
            if pred_time >= time:
                return health

        return self.predicted_health[-1][1] if self.predicted_health else 0.0


@dataclass
class RecoveryResource:
    """Resource required for recovery"""
    resource_type: str  # spare_part, backup_system, personnel, etc.
    resource_id: str
    quantity_needed: float
    quantity_available: float
    procurement_time: timedelta  # Time to get if not available
    cost: float
    is_critical: bool
    alternatives: List[str] = field(default_factory=list)


@dataclass
class RecoveryAction:
    """Individual recovery action"""
    action_id: str
    action_type: str
    target_component: str
    description: str
    duration: timedelta
    prerequisites: List[str]  # Action IDs that must complete first
    resources_needed: List[RecoveryResource]
    success_probability: float
    rollback_action: Optional[str] = None
    is_reversible: bool = True


@dataclass
class RecoveryTimeline:
    """Timeline for recovery execution"""
    start_time: datetime
    estimated_completion: datetime
    phases: Dict[RecoveryPhase, Tuple[datetime, datetime]]  # Start, end times
    milestones: List[Tuple[datetime, str]]  # Time, description
    critical_path: List[str]  # Action IDs on critical path
    buffer_time: timedelta


@dataclass
class RecoveryPlan:
    """Comprehensive recovery plan"""
    plan_id: str
    fault_id: str
    strategy: RecoveryStrategy
    priority: RecoveryPriority
    actions: List[RecoveryAction]
    timeline: RecoveryTimeline
    resources: List[RecoveryResource]
    expected_outcomes: Dict[str, float]  # Metric -> target value
    risk_assessment: Dict[str, float]  # Risk -> probability
    contingency_plans: List[str]  # Alternative plan IDs
    confidence: float


@dataclass
class RecoveryProgress:
    """Track recovery execution progress"""
    plan_id: str
    current_phase: RecoveryPhase
    completed_actions: List[str]
    active_actions: List[str]
    pending_actions: List[str]
    progress_percentage: float
    actual_vs_planned: float  # >1 means behind schedule
    issues_encountered: List[str]
    adaptations_made: List[str]


@dataclass
class PreemptiveAction:
    """Preemptive action to prevent failure"""
    action_id: str
    target_component: str
    action_type: str  # maintenance, replacement, reconfiguration
    trigger_condition: str
    optimal_timing: datetime
    benefit: float  # Expected improvement
    cost: float
    urgency: RecoveryPriority


class PredictiveRecoveryEngine:
    """
    Engine for predictive recovery planning and execution.

    Features:
    - Failure trajectory prediction using degradation models
    - Pre-emptive recovery planning
    - Resource optimization
    - Timeline optimization with critical path analysis
    - Adaptive recovery execution
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Prediction parameters
        self.prediction_horizon = timedelta(hours=self.config.get('prediction_horizon_hours', 72))
        self.health_threshold_warning = self.config.get('health_threshold_warning', 0.7)
        self.health_threshold_critical = self.config.get('health_threshold_critical', 0.3)

        # Recovery parameters
        self.max_parallel_actions = self.config.get('max_parallel_actions', 5)
        self.min_success_probability = self.config.get('min_success_probability', 0.8)

        # State tracking
        self.component_health_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.degradation_models: Dict[str, Dict[str, float]] = {}
        self.active_plans: Dict[str, RecoveryPlan] = {}
        self.plan_progress: Dict[str, RecoveryProgress] = {}

        # Resource inventory
        self.resource_inventory: Dict[str, float] = defaultdict(float)
        self.resource_reservations: Dict[str, Dict[str, float]] = {}

        # Recovery history for learning
        self.recovery_history: List[Dict[str, Any]] = []

    def predict_failure_trajectory(
        self,
        component_id: str,
        current_health: float,
        historical_data: List[Tuple[datetime, float]],
        external_factors: Optional[Dict[str, float]] = None
    ) -> FailureTrajectory:
        """
        Predict failure trajectory for a component.

        Uses degradation modeling with:
        - Linear degradation estimation
        - Acceleration factors from external conditions
        - Historical pattern matching
        """
        external_factors = external_factors or {}

        # Update health history
        now = datetime.now()
        self.component_health_history[component_id].append((now, current_health))

        # Calculate degradation rate from historical data
        degradation_rate = self._calculate_degradation_rate(
            component_id, historical_data
        )

        # Adjust for external factors
        acceleration_factor = 1.0
        contributing_factors = []

        if external_factors.get('temperature_stress', 0) > 0.5:
            acceleration_factor *= 1.3
            contributing_factors.append("high_temperature_stress")

        if external_factors.get('load_stress', 0) > 0.7:
            acceleration_factor *= 1.5
            contributing_factors.append("high_load_stress")

        if external_factors.get('age_factor', 0) > 0.8:
            acceleration_factor *= 1.2
            contributing_factors.append("equipment_aging")

        if external_factors.get('maintenance_overdue', False):
            acceleration_factor *= 1.4
            contributing_factors.append("overdue_maintenance")

        adjusted_rate = degradation_rate * acceleration_factor

        # Predict future health values
        predicted_health = []
        prediction_steps = 24  # Hourly predictions for 24 hours, then daily

        for i in range(prediction_steps):
            if i < 24:
                future_time = now + timedelta(hours=i+1)
            else:
                future_time = now + timedelta(days=i-23)

            future_health = max(0.0, current_health - adjusted_rate * (
                (future_time - now).total_seconds() / 3600
            ))
            predicted_health.append((future_time, future_health))

        # Calculate time to failure (health < critical threshold)
        time_to_failure = None
        if adjusted_rate > 0 and current_health > self.health_threshold_critical:
            hours_to_failure = (current_health - self.health_threshold_critical) / adjusted_rate
            time_to_failure = timedelta(hours=hours_to_failure)

        # Calculate failure probability within prediction horizon
        horizon_health = max(0.0, current_health - adjusted_rate * (
            self.prediction_horizon.total_seconds() / 3600
        ))
        failure_probability = max(0.0, min(1.0,
            1.0 - (horizon_health / self.health_threshold_critical)
        )) if horizon_health < self.health_threshold_critical else 0.0

        # Determine confidence
        data_points = len(historical_data)
        if data_points > 100 and adjusted_rate > 0:
            confidence = PredictionConfidence.HIGH
        elif data_points > 20:
            confidence = PredictionConfidence.MEDIUM
        elif data_points > 5:
            confidence = PredictionConfidence.LOW
        else:
            confidence = PredictionConfidence.UNCERTAIN

        return FailureTrajectory(
            component_id=component_id,
            current_health=current_health,
            predicted_health=predicted_health,
            time_to_failure=time_to_failure,
            failure_probability=failure_probability,
            degradation_rate=adjusted_rate,
            confidence=confidence,
            contributing_factors=contributing_factors
        )

    def _calculate_degradation_rate(
        self,
        component_id: str,
        historical_data: List[Tuple[datetime, float]]
    ) -> float:
        """Calculate degradation rate from historical health data"""
        if len(historical_data) < 2:
            # Use default rate from model or config
            return self.degradation_models.get(
                component_id, {}
            ).get('default_rate', 0.001)  # 0.1% per hour default

        # Linear regression on health over time
        times = []
        healths = []
        base_time = historical_data[0][0]

        for time, health in historical_data:
            hours = (time - base_time).total_seconds() / 3600
            times.append(hours)
            healths.append(health)

        # Simple linear fit
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(healths)
        sum_xy = sum(t * h for t, h in zip(times, healths))
        sum_x2 = sum(t * t for t in times)

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.001  # Default rate

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Degradation rate is negative slope (health decreasing)
        return max(0.0, -slope)

    def generate_recovery_plan(
        self,
        fault_id: str,
        fault_type: str,
        affected_component: str,
        severity: float,
        system_state: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> RecoveryPlan:
        """
        Generate a comprehensive recovery plan.

        Creates optimized recovery plan considering:
        - Available resources
        - System constraints
        - Recovery time objectives
        - Risk minimization
        """
        constraints = constraints or {}
        now = datetime.now()

        # Determine priority
        priority = self._determine_priority(severity, affected_component, system_state)

        # Select recovery strategy
        strategy = self._select_strategy(fault_type, severity, constraints)

        # Generate recovery actions
        actions = self._generate_actions(
            fault_type, affected_component, strategy, system_state
        )

        # Identify required resources
        resources = self._identify_resources(actions, system_state)

        # Create timeline with critical path analysis
        timeline = self._create_timeline(actions, resources, priority, now)

        # Risk assessment
        risk_assessment = self._assess_risks(actions, resources, system_state)

        # Expected outcomes
        expected_outcomes = {
            'health_restored': 0.95,
            'service_availability': 0.99,
            'performance_level': 0.9
        }

        # Calculate overall confidence
        resource_availability = min(
            (r.quantity_available / r.quantity_needed if r.quantity_needed > 0 else 1.0)
            for r in resources
        ) if resources else 1.0

        action_confidence = np.mean([a.success_probability for a in actions]) if actions else 0.5
        confidence = min(resource_availability, action_confidence)

        plan_id = f"recovery_{fault_id}_{now.strftime('%Y%m%d_%H%M%S')}"

        plan = RecoveryPlan(
            plan_id=plan_id,
            fault_id=fault_id,
            strategy=strategy,
            priority=priority,
            actions=actions,
            timeline=timeline,
            resources=resources,
            expected_outcomes=expected_outcomes,
            risk_assessment=risk_assessment,
            contingency_plans=[],
            confidence=confidence
        )

        # Generate contingency plans for high-priority recoveries
        if priority in [RecoveryPriority.CRITICAL, RecoveryPriority.HIGH]:
            contingency = self._generate_contingency_plan(plan, system_state)
            if contingency:
                plan.contingency_plans.append(contingency.plan_id)
                self.active_plans[contingency.plan_id] = contingency

        self.active_plans[plan_id] = plan
        self.plan_progress[plan_id] = RecoveryProgress(
            plan_id=plan_id,
            current_phase=RecoveryPhase.DETECTION,
            completed_actions=[],
            active_actions=[],
            pending_actions=[a.action_id for a in actions],
            progress_percentage=0.0,
            actual_vs_planned=1.0,
            issues_encountered=[],
            adaptations_made=[]
        )

        return plan

    def _determine_priority(
        self,
        severity: float,
        component: str,
        system_state: Dict[str, Any]
    ) -> RecoveryPriority:
        """Determine recovery priority based on severity and context"""
        critical_components = system_state.get('critical_components', set())

        if component in critical_components or severity > 0.9:
            return RecoveryPriority.CRITICAL
        elif severity > 0.7:
            return RecoveryPriority.HIGH
        elif severity > 0.4:
            return RecoveryPriority.MEDIUM
        elif severity > 0.2:
            return RecoveryPriority.LOW
        else:
            return RecoveryPriority.DEFERRED

    def _select_strategy(
        self,
        fault_type: str,
        severity: float,
        constraints: Dict[str, Any]
    ) -> RecoveryStrategy:
        """Select optimal recovery strategy"""
        max_downtime = constraints.get('max_downtime_minutes', 60)
        has_backup = constraints.get('has_backup', False)
        allow_degraded = constraints.get('allow_degraded_operation', True)

        if severity > 0.9 and max_downtime < 5:
            if has_backup:
                return RecoveryStrategy.IMMEDIATE
            else:
                return RecoveryStrategy.PARALLEL
        elif fault_type in ['progressive_fault', 'degradation']:
            return RecoveryStrategy.GRADUAL
        elif severity > 0.6 and allow_degraded:
            return RecoveryStrategy.STAGED
        elif constraints.get('prefer_rollback', False):
            return RecoveryStrategy.ROLLBACK
        else:
            return RecoveryStrategy.ADAPTIVE

    def _generate_actions(
        self,
        fault_type: str,
        component: str,
        strategy: RecoveryStrategy,
        system_state: Dict[str, Any]
    ) -> List[RecoveryAction]:
        """Generate recovery actions based on fault type and strategy"""
        actions = []

        # Common actions for all recoveries
        actions.append(RecoveryAction(
            action_id=f"isolate_{component}",
            action_type="isolation",
            target_component=component,
            description=f"Isolate faulty component {component}",
            duration=timedelta(minutes=5),
            prerequisites=[],
            resources_needed=[],
            success_probability=0.99,
            is_reversible=True
        ))

        # Strategy-specific actions
        if strategy == RecoveryStrategy.IMMEDIATE:
            actions.append(RecoveryAction(
                action_id=f"switchover_{component}",
                action_type="switchover",
                target_component=component,
                description=f"Activate backup for {component}",
                duration=timedelta(minutes=2),
                prerequisites=[f"isolate_{component}"],
                resources_needed=[RecoveryResource(
                    resource_type="backup_system",
                    resource_id=f"backup_{component}",
                    quantity_needed=1,
                    quantity_available=1,
                    procurement_time=timedelta(0),
                    cost=0,
                    is_critical=True
                )],
                success_probability=0.95
            ))

        elif strategy == RecoveryStrategy.GRADUAL:
            actions.extend([
                RecoveryAction(
                    action_id=f"reduce_load_{component}",
                    action_type="load_reduction",
                    target_component=component,
                    description=f"Reduce load on {component}",
                    duration=timedelta(minutes=10),
                    prerequisites=[f"isolate_{component}"],
                    resources_needed=[],
                    success_probability=0.98
                ),
                RecoveryAction(
                    action_id=f"repair_{component}",
                    action_type="repair",
                    target_component=component,
                    description=f"Perform repair on {component}",
                    duration=timedelta(hours=2),
                    prerequisites=[f"reduce_load_{component}"],
                    resources_needed=[RecoveryResource(
                        resource_type="spare_part",
                        resource_id=f"part_{component}",
                        quantity_needed=1,
                        quantity_available=0.8,
                        procurement_time=timedelta(hours=4),
                        cost=1000,
                        is_critical=True
                    )],
                    success_probability=0.85
                ),
                RecoveryAction(
                    action_id=f"restore_load_{component}",
                    action_type="load_restoration",
                    target_component=component,
                    description=f"Gradually restore load to {component}",
                    duration=timedelta(minutes=30),
                    prerequisites=[f"repair_{component}"],
                    resources_needed=[],
                    success_probability=0.95
                )
            ])

        elif strategy == RecoveryStrategy.STAGED:
            stages = ['stage1', 'stage2', 'stage3']
            prev_action = f"isolate_{component}"

            for i, stage in enumerate(stages):
                action = RecoveryAction(
                    action_id=f"{stage}_{component}",
                    action_type="staged_recovery",
                    target_component=component,
                    description=f"Recovery stage {i+1} for {component}",
                    duration=timedelta(minutes=20),
                    prerequisites=[prev_action],
                    resources_needed=[],
                    success_probability=0.9
                )
                actions.append(action)
                prev_action = action.action_id

        elif strategy == RecoveryStrategy.PARALLEL:
            actions.extend([
                RecoveryAction(
                    action_id=f"activate_alternate_{component}",
                    action_type="parallel_activation",
                    target_component=component,
                    description=f"Activate alternate path for {component}",
                    duration=timedelta(minutes=5),
                    prerequisites=[f"isolate_{component}"],
                    resources_needed=[],
                    success_probability=0.92
                ),
                RecoveryAction(
                    action_id=f"balance_load_{component}",
                    action_type="load_balancing",
                    target_component=component,
                    description=f"Balance load across parallel paths",
                    duration=timedelta(minutes=10),
                    prerequisites=[f"activate_alternate_{component}"],
                    resources_needed=[],
                    success_probability=0.95
                )
            ])

        elif strategy == RecoveryStrategy.ROLLBACK:
            actions.append(RecoveryAction(
                action_id=f"rollback_{component}",
                action_type="rollback",
                target_component=component,
                description=f"Rollback {component} to last known good state",
                duration=timedelta(minutes=15),
                prerequisites=[f"isolate_{component}"],
                resources_needed=[],
                success_probability=0.88,
                is_reversible=False
            ))

        else:  # ADAPTIVE
            actions.append(RecoveryAction(
                action_id=f"adaptive_recovery_{component}",
                action_type="adaptive",
                target_component=component,
                description=f"Adaptive recovery for {component}",
                duration=timedelta(minutes=30),
                prerequisites=[f"isolate_{component}"],
                resources_needed=[],
                success_probability=0.85
            ))

        # Verification action
        actions.append(RecoveryAction(
            action_id=f"verify_{component}",
            action_type="verification",
            target_component=component,
            description=f"Verify {component} recovery",
            duration=timedelta(minutes=10),
            prerequisites=[actions[-1].action_id],  # After last recovery action
            resources_needed=[],
            success_probability=0.99
        ))

        return actions

    def _identify_resources(
        self,
        actions: List[RecoveryAction],
        system_state: Dict[str, Any]
    ) -> List[RecoveryResource]:
        """Identify all resources needed for recovery actions"""
        all_resources = []
        seen_resources = set()

        for action in actions:
            for resource in action.resources_needed:
                if resource.resource_id not in seen_resources:
                    seen_resources.add(resource.resource_id)

                    # Update availability from inventory
                    resource.quantity_available = self.resource_inventory.get(
                        resource.resource_id, resource.quantity_available
                    )
                    all_resources.append(resource)

        return all_resources

    def _create_timeline(
        self,
        actions: List[RecoveryAction],
        resources: List[RecoveryResource],
        priority: RecoveryPriority,
        start_time: datetime
    ) -> RecoveryTimeline:
        """Create optimized recovery timeline with critical path analysis"""
        # Build dependency graph
        action_map = {a.action_id: a for a in actions}

        # Calculate earliest start times
        earliest_start: Dict[str, datetime] = {}
        earliest_end: Dict[str, datetime] = {}

        def calc_earliest(action_id: str) -> datetime:
            if action_id in earliest_end:
                return earliest_end[action_id]

            action = action_map[action_id]

            if not action.prerequisites:
                earliest_start[action_id] = start_time
            else:
                prereq_ends = [calc_earliest(p) for p in action.prerequisites]
                earliest_start[action_id] = max(prereq_ends)

            earliest_end[action_id] = earliest_start[action_id] + action.duration
            return earliest_end[action_id]

        for action in actions:
            calc_earliest(action.action_id)

        # Find critical path (longest path)
        end_times = [(a.action_id, earliest_end[a.action_id]) for a in actions]
        latest_action = max(end_times, key=lambda x: x[1])
        completion_time = latest_action[1]

        # Trace back critical path
        critical_path = []
        current = latest_action[0]
        while current:
            critical_path.insert(0, current)
            action = action_map[current]
            if action.prerequisites:
                # Find prerequisite that ends latest (on critical path)
                current = max(
                    action.prerequisites,
                    key=lambda p: earliest_end[p]
                )
            else:
                current = None

        # Calculate phases
        total_duration = completion_time - start_time
        phases = {
            RecoveryPhase.DETECTION: (start_time, start_time),
            RecoveryPhase.ASSESSMENT: (start_time, start_time + timedelta(minutes=5)),
            RecoveryPhase.PLANNING: (
                start_time + timedelta(minutes=5),
                start_time + timedelta(minutes=10)
            ),
            RecoveryPhase.PREPARATION: (
                start_time + timedelta(minutes=10),
                start_time + timedelta(minutes=15)
            ),
            RecoveryPhase.EXECUTION: (
                start_time + timedelta(minutes=15),
                completion_time - timedelta(minutes=10)
            ),
            RecoveryPhase.VERIFICATION: (
                completion_time - timedelta(minutes=10),
                completion_time
            ),
            RecoveryPhase.STABILIZATION: (
                completion_time,
                completion_time + timedelta(minutes=15)
            ),
            RecoveryPhase.COMPLETE: (
                completion_time + timedelta(minutes=15),
                completion_time + timedelta(minutes=15)
            )
        }

        # Create milestones
        milestones = [
            (start_time, "Recovery initiated"),
            (start_time + timedelta(minutes=10), "Planning complete"),
            (start_time + timedelta(minutes=15), "Execution started"),
            (completion_time, "Recovery actions complete"),
            (completion_time + timedelta(minutes=15), "System stabilized")
        ]

        # Buffer based on priority
        buffer_factors = {
            RecoveryPriority.CRITICAL: 0.1,
            RecoveryPriority.HIGH: 0.2,
            RecoveryPriority.MEDIUM: 0.3,
            RecoveryPriority.LOW: 0.5,
            RecoveryPriority.DEFERRED: 1.0
        }
        buffer_time = total_duration * buffer_factors.get(priority, 0.3)

        return RecoveryTimeline(
            start_time=start_time,
            estimated_completion=completion_time + timedelta(minutes=15),
            phases=phases,
            milestones=milestones,
            critical_path=critical_path,
            buffer_time=buffer_time
        )

    def _assess_risks(
        self,
        actions: List[RecoveryAction],
        resources: List[RecoveryResource],
        system_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess risks in recovery plan"""
        risks = {}

        # Resource shortage risk
        resource_shortage = any(
            r.quantity_available < r.quantity_needed for r in resources
        )
        risks['resource_shortage'] = 0.8 if resource_shortage else 0.1

        # Action failure risk
        min_success = min(a.success_probability for a in actions) if actions else 0.0
        risks['action_failure'] = 1.0 - min_success

        # Cascade failure risk
        if system_state.get('system_stress', 0) > 0.7:
            risks['cascade_failure'] = 0.4
        else:
            risks['cascade_failure'] = 0.1

        # Timeline overrun risk
        risks['timeline_overrun'] = 0.2  # Base risk

        # Side effect risk
        irreversible_actions = sum(1 for a in actions if not a.is_reversible)
        risks['side_effects'] = min(0.5, irreversible_actions * 0.1)

        return risks

    def _generate_contingency_plan(
        self,
        primary_plan: RecoveryPlan,
        system_state: Dict[str, Any]
    ) -> Optional[RecoveryPlan]:
        """Generate contingency plan for primary recovery"""
        # Create alternative plan with different strategy
        alternative_strategies = {
            RecoveryStrategy.IMMEDIATE: RecoveryStrategy.PARALLEL,
            RecoveryStrategy.GRADUAL: RecoveryStrategy.ROLLBACK,
            RecoveryStrategy.STAGED: RecoveryStrategy.ADAPTIVE,
            RecoveryStrategy.PARALLEL: RecoveryStrategy.STAGED,
            RecoveryStrategy.ROLLBACK: RecoveryStrategy.GRADUAL,
            RecoveryStrategy.ADAPTIVE: RecoveryStrategy.STAGED
        }

        alt_strategy = alternative_strategies.get(
            primary_plan.strategy, RecoveryStrategy.ADAPTIVE
        )

        # Generate contingency with alternative strategy
        component = primary_plan.actions[0].target_component if primary_plan.actions else "unknown"

        return self.generate_recovery_plan(
            fault_id=f"{primary_plan.fault_id}_contingency",
            fault_type="contingency",
            affected_component=component,
            severity=0.8,  # Assume high severity for contingency
            system_state=system_state,
            constraints={'strategy_override': alt_strategy}
        )

    def generate_preemptive_actions(
        self,
        trajectories: List[FailureTrajectory],
        system_state: Dict[str, Any]
    ) -> List[PreemptiveAction]:
        """
        Generate preemptive actions to prevent predicted failures.

        Analyzes failure trajectories and recommends proactive measures.
        """
        preemptive_actions = []
        now = datetime.now()

        for trajectory in trajectories:
            if trajectory.failure_probability < 0.3:
                continue  # Low risk, no action needed

            # Determine optimal timing
            if trajectory.time_to_failure:
                # Schedule at 70% of remaining time
                optimal_offset = trajectory.time_to_failure * 0.7
                optimal_timing = now + optimal_offset
            else:
                # Default to 24 hours
                optimal_timing = now + timedelta(hours=24)

            # Determine action type based on degradation rate
            if trajectory.degradation_rate > 0.01:  # Fast degradation
                action_type = "replacement"
                benefit = 0.95
                cost = 5000
                urgency = RecoveryPriority.HIGH
            elif trajectory.degradation_rate > 0.005:  # Moderate degradation
                action_type = "maintenance"
                benefit = 0.7
                cost = 1000
                urgency = RecoveryPriority.MEDIUM
            else:  # Slow degradation
                action_type = "reconfiguration"
                benefit = 0.4
                cost = 200
                urgency = RecoveryPriority.LOW

            # Create preemptive action
            preemptive_actions.append(PreemptiveAction(
                action_id=f"preempt_{trajectory.component_id}_{now.strftime('%H%M%S')}",
                target_component=trajectory.component_id,
                action_type=action_type,
                trigger_condition=f"health < {trajectory.current_health * 0.9:.2f}",
                optimal_timing=optimal_timing,
                benefit=benefit,
                cost=cost,
                urgency=urgency
            ))

        # Sort by urgency and benefit
        preemptive_actions.sort(
            key=lambda a: (a.urgency.value, -a.benefit)
        )

        return preemptive_actions

    def execute_recovery_step(
        self,
        plan_id: str,
        system_state: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute next step in recovery plan.

        Returns:
            Tuple of (success, result_info)
        """
        if plan_id not in self.active_plans:
            return False, {'error': 'Plan not found'}

        plan = self.active_plans[plan_id]
        progress = self.plan_progress[plan_id]

        # Find next action to execute
        ready_actions = []
        for action in plan.actions:
            if action.action_id in progress.completed_actions:
                continue
            if action.action_id in progress.active_actions:
                continue

            # Check prerequisites
            prereqs_met = all(
                p in progress.completed_actions
                for p in action.prerequisites
            )
            if prereqs_met:
                ready_actions.append(action)

        if not ready_actions:
            if not progress.active_actions:
                # All done
                progress.current_phase = RecoveryPhase.COMPLETE
                progress.progress_percentage = 100.0
                return True, {'status': 'complete'}
            else:
                return True, {'status': 'waiting', 'active': progress.active_actions}

        # Execute ready actions (up to max parallel)
        results = []
        for action in ready_actions[:self.max_parallel_actions - len(progress.active_actions)]:
            # Simulate action execution
            success = np.random.random() < action.success_probability

            if success:
                progress.pending_actions.remove(action.action_id)
                progress.completed_actions.append(action.action_id)
                results.append({
                    'action_id': action.action_id,
                    'status': 'completed',
                    'duration': action.duration.total_seconds()
                })
            else:
                progress.issues_encountered.append(
                    f"Action {action.action_id} failed"
                )
                results.append({
                    'action_id': action.action_id,
                    'status': 'failed'
                })

        # Update progress
        total_actions = len(plan.actions)
        completed = len(progress.completed_actions)
        progress.progress_percentage = (completed / total_actions) * 100 if total_actions > 0 else 0

        # Update phase
        if progress.progress_percentage < 10:
            progress.current_phase = RecoveryPhase.PREPARATION
        elif progress.progress_percentage < 90:
            progress.current_phase = RecoveryPhase.EXECUTION
        else:
            progress.current_phase = RecoveryPhase.VERIFICATION

        return True, {'results': results, 'progress': progress.progress_percentage}

    def adapt_recovery_plan(
        self,
        plan_id: str,
        issue: str,
        system_state: Dict[str, Any]
    ) -> Optional[RecoveryPlan]:
        """
        Adapt recovery plan based on encountered issues.

        Modifies plan in response to:
        - Action failures
        - Resource shortages
        - Changed system conditions
        """
        if plan_id not in self.active_plans:
            return None

        plan = self.active_plans[plan_id]
        progress = self.plan_progress[plan_id]

        # Record adaptation
        progress.adaptations_made.append(f"Adapted for: {issue}")

        # Determine adaptation strategy
        if 'resource' in issue.lower():
            # Try alternative resources
            for resource in plan.resources:
                if resource.quantity_available < resource.quantity_needed:
                    if resource.alternatives:
                        # Switch to alternative
                        resource.resource_id = resource.alternatives[0]
                        resource.quantity_available = self.resource_inventory.get(
                            resource.resource_id, 0
                        )

        elif 'action_failed' in issue.lower():
            # Insert retry or alternative action
            failed_action_id = issue.split(':')[-1].strip() if ':' in issue else None
            if failed_action_id and plan.contingency_plans:
                # Activate contingency
                return self.active_plans.get(plan.contingency_plans[0])

        elif 'timeout' in issue.lower():
            # Accelerate remaining actions
            for action in plan.actions:
                if action.action_id not in progress.completed_actions:
                    action.duration = action.duration * 0.8  # 20% faster

        return plan

    def get_recovery_status(self, plan_id: str) -> Optional[RecoveryProgress]:
        """Get current status of recovery plan"""
        return self.plan_progress.get(plan_id)

    def complete_recovery(
        self,
        plan_id: str,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete recovery and record results for learning.
        """
        if plan_id not in self.active_plans:
            return {'error': 'Plan not found'}

        plan = self.active_plans[plan_id]
        progress = self.plan_progress[plan_id]

        # Record history for learning
        history_entry = {
            'plan_id': plan_id,
            'strategy': plan.strategy.value,
            'priority': plan.priority.value,
            'total_actions': len(plan.actions),
            'completed_actions': len(progress.completed_actions),
            'issues': progress.issues_encountered,
            'adaptations': progress.adaptations_made,
            'actual_vs_planned': progress.actual_vs_planned,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        }
        self.recovery_history.append(history_entry)

        # Update phase
        progress.current_phase = RecoveryPhase.COMPLETE
        progress.progress_percentage = 100.0

        # Clean up
        del self.active_plans[plan_id]

        return {
            'status': 'completed',
            'history_recorded': True,
            'total_recoveries': len(self.recovery_history)
        }
