"""
Tests for Phase 5.4 Self-Healing Enhancements

Tests L4-level self-healing capabilities:
1. Enhanced Diagnosis Engine
2. Optimized Isolation Strategy
3. Predictive Recovery Engine
4. Fault Learning Engine
5. Macro Cognitive Layer
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestEnhancedDiagnosis:
    """Tests for enhanced fault diagnosis"""

    def test_diagnosis_engine_initialization(self):
        """Test diagnosis engine creates successfully"""
        from hydroe2e.phase5.self_healing.enhanced_diagnosis import EnhancedDiagnosisEngine

        engine = EnhancedDiagnosisEngine()
        assert engine is not None
        assert hasattr(engine, 'diagnose')

    def test_extended_fault_types(self):
        """Test extended fault type enumeration"""
        from hydroe2e.phase5.self_healing.enhanced_diagnosis import ExtendedFaultType

        # Check all expected fault types exist
        assert hasattr(ExtendedFaultType, 'SENSOR_DRIFT')
        assert hasattr(ExtendedFaultType, 'CASCADE_FAULT')
        assert hasattr(ExtendedFaultType, 'COMPOUND_FAULT')
        assert hasattr(ExtendedFaultType, 'INTERMITTENT_FAULT')
        assert hasattr(ExtendedFaultType, 'CYBER_ATTACK')

    def test_fault_pattern_structure(self):
        """Test fault pattern dataclass"""
        from hydroe2e.phase5.self_healing.enhanced_diagnosis import FaultPattern, ExtendedFaultType

        pattern = FaultPattern(
            pattern_id="test_pattern",
            fault_type=ExtendedFaultType.SENSOR_DRIFT,
            signature={'deviation': (0.1, 0.3)},
            temporal_pattern=[0.1, 0.2, 0.3],
            correlation_vars=['sensor_1', 'sensor_2'],
            confidence_threshold=0.85,
            occurrence_count=10
        )

        assert pattern.pattern_id == "test_pattern"
        assert pattern.confidence_threshold == 0.85

    def test_compound_fault_detection(self):
        """Test compound fault representation"""
        from hydroe2e.phase5.self_healing.enhanced_diagnosis import CompoundFault, FaultSeverity

        compound = CompoundFault(
            fault_id="compound_001",
            component_faults=['fault_1', 'fault_2'],
            interaction_type="additive",
            combined_severity=FaultSeverity.HIGH,
            root_cause='sensor_1'
        )

        assert len(compound.component_faults) == 2
        assert compound.combined_severity == FaultSeverity.HIGH

    def test_cascade_fault_tracking(self):
        """Test cascade fault tracking"""
        from hydroe2e.phase5.self_healing.enhanced_diagnosis import CascadeFault

        cascade = CascadeFault(
            fault_id="cascade_001",
            trigger_fault="pump_fault_001",
            affected_components=["pump_1", "pipe_1", "tank_1"],
            propagation_time=30.0,
            propagation_probability=0.8,
            current_stage=2
        )

        assert cascade.trigger_fault == "pump_fault_001"
        assert len(cascade.affected_components) == 3
        assert cascade.current_stage == 2

    def test_diagnosis_confidence_structure(self):
        """Test diagnosis confidence dataclass"""
        from hydroe2e.phase5.self_healing.enhanced_diagnosis import DiagnosisConfidence

        confidence = DiagnosisConfidence(
            base_confidence=0.8,
            pattern_match_boost=0.1,
            historical_boost=0.05
        )
        assert confidence.base_confidence == 0.8
        assert confidence.final_confidence > 0.8

    def test_diagnosis_result_structure(self):
        """Test enhanced diagnosis result structure"""
        from hydroe2e.phase5.self_healing.enhanced_diagnosis import (
            EnhancedDiagnosisResult, ExtendedFaultType,
            FaultSeverity, DiagnosisConfidence
        )

        result = EnhancedDiagnosisResult(
            fault_id='fault_001',
            fault_type=ExtendedFaultType.SENSOR_DRIFT,
            component='sensor_1',
            severity=FaultSeverity.MEDIUM,
            confidence=DiagnosisConfidence(base_confidence=0.85),
            root_causes=['sensor_degradation'],
            contributing_factors=['temperature', 'age'],
            recommended_actions=['recalibrate']
        )

        assert result.fault_id == 'fault_001'
        assert result.component == 'sensor_1'


class TestOptimizedIsolation:
    """Tests for optimized isolation strategy"""

    def test_isolation_strategy_initialization(self):
        """Test isolation strategy creates successfully"""
        from hydroe2e.phase5.self_healing.optimized_isolation import OptimizedIsolationStrategy

        strategy = OptimizedIsolationStrategy()
        assert strategy is not None

    def test_isolation_actions(self):
        """Test isolation action enumeration"""
        from hydroe2e.phase5.self_healing.optimized_isolation import IsolationAction

        assert hasattr(IsolationAction, 'DISABLE')
        assert hasattr(IsolationAction, 'SWITCH_BACKUP')
        assert hasattr(IsolationAction, 'BYPASS')

    def test_isolation_scope(self):
        """Test isolation scope enumeration"""
        from hydroe2e.phase5.self_healing.optimized_isolation import IsolationScope

        assert hasattr(IsolationScope, 'COMPONENT')
        assert hasattr(IsolationScope, 'SUBSYSTEM')
        assert hasattr(IsolationScope, 'NETWORK')

    def test_impact_analysis(self):
        """Test impact analysis structure"""
        from hydroe2e.phase5.self_healing.optimized_isolation import ImpactAnalysis

        impact = ImpactAnalysis(
            affected_components=['pump_1', 'valve_1'],
            affected_pools=[1, 2],
            service_impact=0.3,
            safety_impact=0.1,
            estimated_duration=7200.0,
            cascading_risk=0.2,
            mitigation_options=['switch_to_backup', 'reduce_load']
        )

        assert len(impact.affected_components) == 2
        assert impact.service_impact == 0.3

    def test_graceful_degradation_path(self):
        """Test graceful degradation path"""
        from hydroe2e.phase5.self_healing.optimized_isolation import GracefulDegradationPath

        path = GracefulDegradationPath(
            path_id="degradation_001",
            stages=[
                {'level': 1, 'action': 'reduce_load_50'},
                {'level': 2, 'action': 'switch_backup'}
            ],
            current_stage=0,
            service_levels=[1.0, 0.7, 0.5],
            transition_times=[30.0, 60.0]
        )

        assert len(path.stages) == 2
        assert len(path.service_levels) == 3

    def test_service_continuity_plan(self):
        """Test service continuity plan"""
        from hydroe2e.phase5.self_healing.optimized_isolation import ServiceContinuityPlan

        plan = ServiceContinuityPlan(
            plan_id="continuity_001",
            primary_path='main_route',
            backup_paths=['backup_route_1', 'backup_route_2'],
            minimum_service_level=0.6,
            max_degradation_time=3600.0,
            checkpoints=[{'time': 300, 'check': 'status'}],
            recovery_triggers=['fault_cleared', 'backup_ready']
        )

        assert plan.plan_id == "continuity_001"
        assert plan.minimum_service_level == 0.6

    def test_generate_isolation_plan(self):
        """Test generating isolation plan"""
        from hydroe2e.phase5.self_healing.optimized_isolation import OptimizedIsolationStrategy

        strategy = OptimizedIsolationStrategy()

        system_state = {
            'components': {'pump_1': {'status': 'faulty'}},
            'dependencies': {'pump_1': ['pipe_1', 'tank_1']}
        }

        plan = strategy.generate_isolation_plan(
            fault_component='pump_1',
            fault_severity=0.7,
            system_state=system_state
        )

        assert plan is not None
        assert plan.target_component == 'pump_1'
        assert plan.isolation_action is not None

    def test_impact_analysis_generation(self):
        """Test impact analysis generation"""
        from hydroe2e.phase5.self_healing.optimized_isolation import OptimizedIsolationStrategy

        strategy = OptimizedIsolationStrategy()

        system_state = {
            'components': {'pump_1': {'connections': ['pipe_1']}},
            'services': {'water_supply': {'components': ['pump_1']}}
        }

        impact = strategy.analyze_impact('pump_1', system_state)
        assert impact is not None


class TestPredictiveRecovery:
    """Tests for predictive recovery engine"""

    def test_recovery_engine_initialization(self):
        """Test recovery engine creates successfully"""
        from hydroe2e.phase5.self_healing.predictive_recovery import PredictiveRecoveryEngine

        engine = PredictiveRecoveryEngine()
        assert engine is not None

    def test_recovery_phases(self):
        """Test recovery phase enumeration"""
        from hydroe2e.phase5.self_healing.predictive_recovery import RecoveryPhase

        assert hasattr(RecoveryPhase, 'DETECTION')
        assert hasattr(RecoveryPhase, 'ASSESSMENT')
        assert hasattr(RecoveryPhase, 'EXECUTION')
        assert hasattr(RecoveryPhase, 'VERIFICATION')
        assert hasattr(RecoveryPhase, 'COMPLETE')

    def test_recovery_strategies(self):
        """Test recovery strategy enumeration"""
        from hydroe2e.phase5.self_healing.predictive_recovery import RecoveryStrategy

        assert hasattr(RecoveryStrategy, 'IMMEDIATE')
        assert hasattr(RecoveryStrategy, 'GRADUAL')
        assert hasattr(RecoveryStrategy, 'STAGED')
        assert hasattr(RecoveryStrategy, 'ADAPTIVE')

    def test_recovery_priority(self):
        """Test recovery priority levels"""
        from hydroe2e.phase5.self_healing.predictive_recovery import RecoveryPriority

        assert RecoveryPriority.CRITICAL.value < RecoveryPriority.HIGH.value
        assert RecoveryPriority.HIGH.value < RecoveryPriority.MEDIUM.value

    def test_failure_trajectory_prediction(self):
        """Test failure trajectory structure"""
        from hydroe2e.phase5.self_healing.predictive_recovery import (
            PredictiveRecoveryEngine, FailureTrajectory, PredictionConfidence
        )

        engine = PredictiveRecoveryEngine()

        historical_data = [
            (datetime.now() - timedelta(hours=i), 1.0 - i * 0.01)
            for i in range(48, 0, -1)
        ]

        trajectory = engine.predict_failure_trajectory(
            component_id='pump_1',
            current_health=0.6,
            historical_data=historical_data,
            external_factors={'load_stress': 0.8}
        )

        assert trajectory is not None
        assert trajectory.component_id == 'pump_1'
        assert trajectory.current_health == 0.6
        assert len(trajectory.predicted_health) > 0

    def test_recovery_plan_generation(self):
        """Test generating recovery plan"""
        from hydroe2e.phase5.self_healing.predictive_recovery import PredictiveRecoveryEngine

        engine = PredictiveRecoveryEngine()

        system_state = {
            'critical_components': {'pump_main'},
            'system_stress': 0.5
        }

        plan = engine.generate_recovery_plan(
            fault_id='fault_001',
            fault_type='sensor_drift',
            affected_component='sensor_1',
            severity=0.7,
            system_state=system_state
        )

        assert plan is not None
        assert plan.fault_id == 'fault_001'
        assert len(plan.actions) > 0

    def test_recovery_timeline_creation(self):
        """Test recovery timeline structure"""
        from hydroe2e.phase5.self_healing.predictive_recovery import RecoveryTimeline, RecoveryPhase

        now = datetime.now()
        timeline = RecoveryTimeline(
            start_time=now,
            estimated_completion=now + timedelta(hours=2),
            phases={
                RecoveryPhase.DETECTION: (now, now + timedelta(minutes=5)),
                RecoveryPhase.EXECUTION: (now + timedelta(minutes=10), now + timedelta(hours=1))
            },
            milestones=[(now + timedelta(minutes=30), "Halfway point")],
            critical_path=['action_1', 'action_2'],
            buffer_time=timedelta(minutes=30)
        )

        assert timeline.start_time == now
        assert len(timeline.critical_path) == 2

    def test_preemptive_action_generation(self):
        """Test preemptive action generation"""
        from hydroe2e.phase5.self_healing.predictive_recovery import (
            PredictiveRecoveryEngine, FailureTrajectory, PredictionConfidence
        )

        engine = PredictiveRecoveryEngine()

        trajectories = [
            FailureTrajectory(
                component_id='pump_1',
                current_health=0.5,
                predicted_health=[(datetime.now() + timedelta(hours=i), 0.5 - i * 0.05) for i in range(10)],
                time_to_failure=timedelta(hours=10),
                failure_probability=0.7,
                degradation_rate=0.005,
                confidence=PredictionConfidence.MEDIUM,
                contributing_factors=['aging']
            )
        ]

        preemptive = engine.generate_preemptive_actions(trajectories, {})
        assert isinstance(preemptive, list)

    def test_recovery_execution_step(self):
        """Test recovery execution step"""
        from hydroe2e.phase5.self_healing.predictive_recovery import PredictiveRecoveryEngine

        engine = PredictiveRecoveryEngine()

        # First generate a plan
        plan = engine.generate_recovery_plan(
            fault_id='fault_002',
            fault_type='actuator_stuck',
            affected_component='valve_1',
            severity=0.6,
            system_state={}
        )

        # Execute a step
        success, result = engine.execute_recovery_step(plan.plan_id, {})
        assert isinstance(success, bool)
        assert isinstance(result, dict)


class TestFaultLearning:
    """Tests for fault learning engine"""

    def test_learning_engine_initialization(self):
        """Test learning engine creates successfully"""
        from hydroe2e.phase5.self_healing.fault_learning import FaultLearningEngine

        engine = FaultLearningEngine()
        assert engine is not None

    def test_learning_modes(self):
        """Test learning mode enumeration"""
        from hydroe2e.phase5.self_healing.fault_learning import LearningMode

        assert hasattr(LearningMode, 'SUPERVISED')
        assert hasattr(LearningMode, 'UNSUPERVISED')
        assert hasattr(LearningMode, 'HYBRID')

    def test_pattern_status(self):
        """Test pattern status enumeration"""
        from hydroe2e.phase5.self_healing.fault_learning import PatternStatus

        assert hasattr(PatternStatus, 'CANDIDATE')
        assert hasattr(PatternStatus, 'VALIDATED')
        assert hasattr(PatternStatus, 'ACTIVE')

    def test_fault_signature_structure(self):
        """Test fault signature dataclass"""
        from hydroe2e.phase5.self_healing.fault_learning import (
            FaultSignature, PatternStatus, ConfidenceLevel
        )

        signature = FaultSignature(
            signature_id='sig_001',
            fault_type='sensor_drift',
            feature_vector=[0.1, 0.2, 0.3],
            feature_names=['f1', 'f2', 'f3'],
            threshold_ranges={'f1': (0.0, 0.2)},
            temporal_pattern=None,
            occurrence_count=10,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=ConfidenceLevel.MEDIUM,
            status=PatternStatus.CANDIDATE
        )

        assert signature.signature_id == 'sig_001'
        assert signature.occurrence_count == 10

    def test_causal_relation(self):
        """Test causal relation structure"""
        from hydroe2e.phase5.self_healing.fault_learning import CausalRelation, ConfidenceLevel

        relation = CausalRelation(
            relation_id='rel_001',
            cause_type='sensor_drift',
            effect_type='control_error',
            cause_component='sensor_1',
            effect_component='controller_1',
            probability=0.8,
            typical_delay=timedelta(minutes=5),
            delay_variance=60.0,
            observation_count=20,
            confidence=ConfidenceLevel.HIGH,
            conditions=['high_load']
        )

        assert relation.probability == 0.8
        assert relation.observation_count == 20

    def test_learn_from_fault(self):
        """Test learning from fault event"""
        from hydroe2e.phase5.self_healing.fault_learning import FaultLearningEngine

        engine = FaultLearningEngine()

        fault_data = {
            'severity': 0.7,
            'confidence': 0.85,
            'affected_components': ['sensor_1'],
            'sensor_values': {'s1': 50, 's2': 30},
            'deviations': {'pressure': 0.15}
        }

        result = engine.learn_from_fault(
            fault_data=fault_data,
            diagnosis_result='sensor_drift',
            ground_truth='sensor_drift'
        )

        assert isinstance(result, dict)
        assert 'accuracy_feedback' in result

    def test_learn_from_recovery(self):
        """Test learning from recovery outcome"""
        from hydroe2e.phase5.self_healing.fault_learning import (
            FaultLearningEngine, RecoveryOutcome
        )

        engine = FaultLearningEngine()

        outcome = RecoveryOutcome(
            recovery_id='rec_001',
            fault_type='sensor_drift',
            strategy='gradual',
            actions_taken=['isolate', 'repair'],
            success=True,
            time_to_recovery=timedelta(hours=1),
            side_effects=[],
            resource_usage={'spare_parts': 1},
            effectiveness_score=0.9,
            timestamp=datetime.now()
        )

        result = engine.learn_from_recovery(outcome)
        assert 'success_rate' in result

    def test_get_best_strategy(self):
        """Test getting best recovery strategy"""
        from hydroe2e.phase5.self_healing.fault_learning import (
            FaultLearningEngine, RecoveryOutcome
        )

        engine = FaultLearningEngine()

        # Add some recovery outcomes
        for i in range(5):
            engine.learn_from_recovery(RecoveryOutcome(
                recovery_id=f'rec_{i}',
                fault_type='sensor_drift',
                strategy='gradual',
                actions_taken=['isolate', 'repair'],
                success=True,
                time_to_recovery=timedelta(hours=1),
                side_effects=[],
                resource_usage={},
                effectiveness_score=0.9,
                timestamp=datetime.now()
            ))

        strategy, confidence = engine.get_best_strategy('sensor_drift', {})
        assert isinstance(strategy, str)
        assert isinstance(confidence, float)

    def test_causal_relation_learning(self):
        """Test learning causal relationships"""
        from hydroe2e.phase5.self_healing.fault_learning import FaultLearningEngine

        engine = FaultLearningEngine()

        cause_event = {
            'fault_type': 'sensor_drift',
            'component': 'sensor_1'
        }

        effect_event = {
            'fault_type': 'control_error',
            'component': 'controller_1'
        }

        relation = engine.learn_causal_relation(
            cause_event=cause_event,
            effect_event=effect_event,
            delay=timedelta(minutes=5)
        )

        assert relation is not None
        assert relation.cause_type == 'sensor_drift'

    def test_export_import_knowledge(self):
        """Test knowledge export and import"""
        from hydroe2e.phase5.self_healing.fault_learning import FaultLearningEngine

        engine = FaultLearningEngine()

        # Add some learning
        engine.learn_from_fault(
            fault_data={'severity': 0.5, 'sensor_values': {'s1': 10}},
            diagnosis_result='drift',
            ground_truth='drift'
        )

        # Export
        exported = engine.export_knowledge()
        assert 'version' in exported
        assert 'signatures' in exported

        # Import into new engine
        new_engine = FaultLearningEngine()
        success = new_engine.import_knowledge(exported)
        assert success

    def test_learning_metrics(self):
        """Test getting learning metrics"""
        from hydroe2e.phase5.self_healing.fault_learning import FaultLearningEngine

        engine = FaultLearningEngine()
        metrics = engine.get_learning_metrics()

        assert hasattr(metrics, 'total_samples')
        assert hasattr(metrics, 'patterns_learned')
        assert hasattr(metrics, 'diagnosis_accuracy')


class TestMacroCognitiveLayer:
    """Tests for L3 macro cognitive layer"""

    def test_macro_cognitive_initialization(self):
        """Test macro cognitive layer creates successfully"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import MacroCognitiveLayer

        layer = MacroCognitiveLayer()
        assert layer is not None

    def test_system_health_levels(self):
        """Test system health level enumeration"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import SystemHealthLevel

        assert hasattr(SystemHealthLevel, 'EXCELLENT')
        assert hasattr(SystemHealthLevel, 'GOOD')
        assert hasattr(SystemHealthLevel, 'CRITICAL')

    def test_seasonal_patterns(self):
        """Test seasonal pattern enumeration"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import SeasonalPattern

        assert hasattr(SeasonalPattern, 'SPRING_FLOOD')
        assert hasattr(SeasonalPattern, 'SUMMER_PEAK')
        assert hasattr(SeasonalPattern, 'WINTER_ICE')

    def test_risk_levels(self):
        """Test risk level enumeration"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import RiskLevel

        assert hasattr(RiskLevel, 'LOW')
        assert hasattr(RiskLevel, 'MEDIUM')
        assert hasattr(RiskLevel, 'HIGH')
        assert hasattr(RiskLevel, 'CRITICAL')

    def test_equipment_health_tracking(self):
        """Test equipment health structure"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import EquipmentHealth

        health = EquipmentHealth(
            equipment_id='pump_1',
            current_health=0.85,
            degradation_trend=-0.001,
            predicted_failure_date=datetime.now() + timedelta(days=90),
            maintenance_priority=2,
            last_maintenance=datetime.now() - timedelta(days=30)
        )

        assert health.equipment_id == 'pump_1'
        assert health.current_health == 0.85

    def test_network_risk_assessment(self):
        """Test network risk assessment structure"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import (
            NetworkRiskAssessment, RiskLevel
        )

        assessment = NetworkRiskAssessment(
            assessment_id='risk_001',
            overall_risk=RiskLevel.MEDIUM,
            risk_factors={
                'equipment_aging': 0.3,
                'capacity_stress': 0.4
            },
            vulnerable_components=['pump_old_1'],
            mitigation_recommendations=['schedule_maintenance'],
            assessment_time=datetime.now()
        )

        assert assessment.overall_risk == RiskLevel.MEDIUM
        assert 'equipment_aging' in assessment.risk_factors

    def test_strategic_advice_generation(self):
        """Test strategic advice structure"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import StrategicAdvice

        advice = StrategicAdvice(
            advice_id='adv_001',
            advice_type='maintenance',
            priority=2,
            description='Schedule preventive maintenance for pump cluster',
            affected_components=['pump_1', 'pump_2'],
            expected_benefit=0.15,
            implementation_cost=5000,
            time_horizon=timedelta(days=30)
        )

        assert advice.priority == 2
        assert len(advice.affected_components) == 2

    def test_macro_cognitive_assessment(self):
        """Test macro cognitive state assessment"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import MacroCognitiveLayer

        layer = MacroCognitiveLayer()

        system_state = {
            'equipment_health': {
                'pump_1': 0.85,
                'pump_2': 0.70,
                'valve_1': 0.95
            },
            'network_load': 0.6,
            'active_faults': []
        }

        forecast_data = {
            'demand_forecast': [100, 110, 105],
            'weather_forecast': {'temperature': 25, 'precipitation': 0}
        }

        state = layer.assess(
            system_state=system_state,
            forecast_data=forecast_data,
            fault_reports=[],
            external_factors={'season': 'summer'}
        )

        assert state is not None
        assert hasattr(state, 'system_health')
        assert hasattr(state, 'strategic_advice')

    def test_scheduling_constraints_generation(self):
        """Test generating scheduling constraints"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import MacroCognitiveLayer

        layer = MacroCognitiveLayer()

        # Perform initial assessment
        layer.assess(
            system_state={'equipment_health': {'pump_1': 0.7}},
            forecast_data={},
            fault_reports=[],
            external_factors={}
        )

        constraints = layer.get_scheduling_constraints()
        assert isinstance(constraints, dict)

    def test_risk_adjusted_targets(self):
        """Test risk-adjusted target generation"""
        from hydroe2e.phase5.self_healing.macro_cognitive_layer import MacroCognitiveLayer

        layer = MacroCognitiveLayer()

        # Perform initial assessment
        layer.assess(
            system_state={'equipment_health': {'pump_1': 0.5}},  # Low health
            forecast_data={},
            fault_reports=[],
            external_factors={}
        )

        base_targets = {
            'throughput': 100,
            'pressure': 50
        }

        adjusted = layer.get_risk_adjusted_targets(base_targets, risk_tolerance=0.5)
        assert isinstance(adjusted, dict)


class TestSelfHealingIntegration:
    """Integration tests for self-healing components"""

    def test_module_imports(self):
        """Test all module imports work correctly"""
        from hydroe2e.phase5.self_healing import (
            EnhancedDiagnosisEngine,
            OptimizedIsolationStrategy,
            PredictiveRecoveryEngine,
            FaultLearningEngine,
            MacroCognitiveLayer
        )

        # All should be importable
        assert EnhancedDiagnosisEngine is not None
        assert OptimizedIsolationStrategy is not None
        assert PredictiveRecoveryEngine is not None
        assert FaultLearningEngine is not None
        assert MacroCognitiveLayer is not None

    def test_isolation_strategy_flow(self):
        """Test isolation strategy generates valid plans"""
        from hydroe2e.phase5.self_healing import OptimizedIsolationStrategy

        isolation_strategy = OptimizedIsolationStrategy()

        # Generate isolation plan
        plan = isolation_strategy.generate_isolation_plan(
            fault_component='pump_1',
            fault_severity=0.7,
            system_state={}
        )

        assert plan is not None
        assert plan.target_component == 'pump_1'

    def test_recovery_planning_flow(self):
        """Test recovery planning workflow"""
        from hydroe2e.phase5.self_healing import PredictiveRecoveryEngine

        recovery_engine = PredictiveRecoveryEngine()

        # Generate recovery plan
        recovery_plan = recovery_engine.generate_recovery_plan(
            fault_id='fault_integration_001',
            fault_type='sensor_drift',
            affected_component='sensor_1',
            severity=0.6,
            system_state={}
        )

        assert recovery_plan is not None
        assert len(recovery_plan.actions) > 0

    def test_recovery_to_learning_flow(self):
        """Test flow from recovery to learning"""
        from hydroe2e.phase5.self_healing import (
            PredictiveRecoveryEngine,
            FaultLearningEngine
        )
        from hydroe2e.phase5.self_healing.fault_learning import RecoveryOutcome

        recovery_engine = PredictiveRecoveryEngine()
        learning_engine = FaultLearningEngine()

        # Generate and execute recovery
        plan = recovery_engine.generate_recovery_plan(
            fault_id='fault_learn_001',
            fault_type='actuator_delay',
            affected_component='valve_1',
            severity=0.5,
            system_state={}
        )

        # Create outcome
        outcome = RecoveryOutcome(
            recovery_id=plan.plan_id,
            fault_type='actuator_delay',
            strategy=plan.strategy.value,
            actions_taken=[a.action_id for a in plan.actions],
            success=True,
            time_to_recovery=timedelta(hours=1),
            side_effects=[],
            resource_usage={},
            effectiveness_score=0.85,
            timestamp=datetime.now()
        )

        # Learn from recovery
        result = learning_engine.learn_from_recovery(outcome)
        assert 'success_rate' in result

    def test_macro_cognitive_to_scheduling_flow(self):
        """Test flow from macro cognitive to scheduling constraints"""
        from hydroe2e.phase5.self_healing import MacroCognitiveLayer

        layer = MacroCognitiveLayer()

        # Assess system
        state = layer.assess(
            system_state={
                'equipment_health': {
                    'pump_1': 0.6,
                    'pump_2': 0.9
                },
                'network_load': 0.75
            },
            forecast_data={
                'demand_forecast': [100, 120, 110]
            },
            fault_reports=[],
            external_factors={'season': 'summer'}
        )

        # Get constraints for scheduler
        constraints = layer.get_scheduling_constraints()
        targets = layer.get_risk_adjusted_targets(
            {'throughput': 100},
            risk_tolerance=0.5
        )

        assert state is not None
        assert isinstance(constraints, dict)
        assert isinstance(targets, dict)

    def test_integrated_self_healing_workflow(self):
        """Test integrated self-healing workflow without diagnosis"""
        from hydroe2e.phase5.self_healing import (
            OptimizedIsolationStrategy,
            PredictiveRecoveryEngine,
            FaultLearningEngine,
            MacroCognitiveLayer
        )
        from hydroe2e.phase5.self_healing.fault_learning import RecoveryOutcome

        # Initialize components
        isolation = OptimizedIsolationStrategy()
        recovery = PredictiveRecoveryEngine()
        learning = FaultLearningEngine()
        cognitive = MacroCognitiveLayer()

        # 1. Macro assessment
        macro_state = cognitive.assess(
            system_state={'equipment_health': {'pump_1': 0.7}},
            forecast_data={},
            fault_reports=[],
            external_factors={}
        )

        # 2. Impact analysis and isolation
        impact = isolation.analyze_impact('pump_1', {})
        isolation_plan = isolation.generate_isolation_plan('pump_1', 0.6, {})

        # 3. Recovery planning
        recovery_plan = recovery.generate_recovery_plan(
            fault_id='cycle_test_001',
            fault_type='degradation',
            affected_component='pump_1',
            severity=0.6,
            system_state={}
        )

        # 4. Execute recovery (simulated)
        recovery.execute_recovery_step(recovery_plan.plan_id, {})

        # 5. Learn from outcome
        outcome = RecoveryOutcome(
            recovery_id=recovery_plan.plan_id,
            fault_type='degradation',
            strategy=recovery_plan.strategy.value,
            actions_taken=['isolate', 'repair'],
            success=True,
            time_to_recovery=timedelta(hours=2),
            side_effects=[],
            resource_usage={},
            effectiveness_score=0.9,
            timestamp=datetime.now()
        )

        learning_result = learning.learn_from_recovery(outcome)

        # Verify complete cycle
        assert macro_state is not None
        assert impact is not None
        assert isolation_plan is not None
        assert recovery_plan is not None
        assert learning_result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
