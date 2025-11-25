"""
Fault Learning Engine for L4 Self-Healing

Implements continuous learning from fault events to improve
diagnosis accuracy and recovery effectiveness over time.

Key features:
- Pattern recognition from historical faults
- Diagnosis model refinement
- Recovery strategy optimization
- Root cause analysis improvement
- Anomaly signature learning
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning mode for the engine"""
    SUPERVISED = "supervised"      # Learn from labeled examples
    SEMI_SUPERVISED = "semi"       # Partial labels
    UNSUPERVISED = "unsupervised"  # Cluster-based learning
    REINFORCEMENT = "reinforcement"  # Reward-based learning
    HYBRID = "hybrid"              # Combination


class PatternStatus(Enum):
    """Status of learned patterns"""
    CANDIDATE = "candidate"    # New pattern, needs validation
    VALIDATED = "validated"    # Validated by experts
    ACTIVE = "active"          # Actively used for detection
    DEPRECATED = "deprecated"  # No longer used
    ARCHIVED = "archived"      # Kept for reference


class ConfidenceLevel(Enum):
    """Confidence level for learned patterns"""
    VERY_HIGH = 5   # >95%
    HIGH = 4        # 85-95%
    MEDIUM = 3      # 70-85%
    LOW = 2         # 50-70%
    VERY_LOW = 1    # <50%


@dataclass
class FaultSignature:
    """Learned fault signature"""
    signature_id: str
    fault_type: str
    feature_vector: List[float]
    feature_names: List[str]
    threshold_ranges: Dict[str, Tuple[float, float]]
    temporal_pattern: Optional[List[float]]  # Time series pattern
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    confidence: ConfidenceLevel
    status: PatternStatus
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelation:
    """Learned causal relationship"""
    relation_id: str
    cause_type: str
    effect_type: str
    cause_component: str
    effect_component: str
    probability: float  # P(effect | cause)
    typical_delay: timedelta
    delay_variance: float  # seconds
    observation_count: int
    confidence: ConfidenceLevel
    conditions: List[str]  # Conditions when relation holds


@dataclass
class RecoveryOutcome:
    """Outcome of a recovery action"""
    recovery_id: str
    fault_type: str
    strategy: str
    actions_taken: List[str]
    success: bool
    time_to_recovery: timedelta
    side_effects: List[str]
    resource_usage: Dict[str, float]
    effectiveness_score: float  # 0-1
    timestamp: datetime


@dataclass
class LearningMetrics:
    """Metrics for learning performance"""
    total_samples: int
    patterns_learned: int
    patterns_validated: int
    false_positive_rate: float
    false_negative_rate: float
    diagnosis_accuracy: float
    recovery_success_rate: float
    avg_time_to_diagnosis: float  # seconds
    avg_time_to_recovery: float   # seconds
    model_version: str
    last_updated: datetime


@dataclass
class DiagnosisRule:
    """Rule for fault diagnosis"""
    rule_id: str
    conditions: List[Dict[str, Any]]  # Feature conditions
    conclusion: str  # Fault type
    confidence: float
    support: int  # Number of supporting examples
    lift: float   # Rule quality metric
    source: str   # How rule was learned


@dataclass
class FeatureImportance:
    """Importance of features for diagnosis"""
    feature_name: str
    importance_score: float
    correlation_with_fault: Dict[str, float]  # Fault type -> correlation
    stability: float  # How stable importance is over time
    last_updated: datetime


class FaultLearningEngine:
    """
    Engine for learning from fault events and improving self-healing.

    Features:
    - Incremental pattern learning
    - Causal relationship discovery
    - Recovery strategy optimization
    - Diagnosis rule refinement
    - Continuous model improvement
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Learning parameters
        self.learning_mode = LearningMode(
            self.config.get('learning_mode', 'hybrid')
        )
        self.min_samples_for_pattern = self.config.get('min_samples', 5)
        self.pattern_validation_threshold = self.config.get('validation_threshold', 0.85)
        self.forgetting_factor = self.config.get('forgetting_factor', 0.95)

        # Learned knowledge base
        self.fault_signatures: Dict[str, FaultSignature] = {}
        self.causal_relations: Dict[str, CausalRelation] = {}
        self.diagnosis_rules: Dict[str, DiagnosisRule] = {}
        self.feature_importance: Dict[str, FeatureImportance] = {}

        # Training data storage
        self.fault_history: List[Dict[str, Any]] = []
        self.recovery_outcomes: List[RecoveryOutcome] = []
        self.pending_validation: List[str] = []  # Signature IDs

        # Clustering for pattern discovery
        self.cluster_centers: Dict[str, np.ndarray] = {}
        self.cluster_labels: Dict[str, List[str]] = defaultdict(list)

        # Recovery strategy statistics
        self.strategy_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'success_count': 0, 'total_count': 0, 'avg_time': 0}
        )

        # Model versioning
        self.model_version = "1.0.0"
        self.last_training_time = datetime.now()

    def learn_from_fault(
        self,
        fault_data: Dict[str, Any],
        diagnosis_result: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Learn from a fault event.

        Args:
            fault_data: Fault event data including features
            diagnosis_result: What the system diagnosed
            ground_truth: Actual fault type (if known)

        Returns:
            Learning result with any new patterns discovered
        """
        timestamp = datetime.now()
        learning_result = {
            'patterns_updated': [],
            'new_patterns': [],
            'rules_updated': [],
            'accuracy_feedback': None
        }

        # Extract features
        features = self._extract_features(fault_data)
        feature_vector = list(features.values())
        feature_names = list(features.keys())

        # Store in history
        self.fault_history.append({
            'timestamp': timestamp.isoformat(),
            'features': features,
            'diagnosis': diagnosis_result,
            'ground_truth': ground_truth,
            'raw_data': fault_data
        })

        # Update diagnosis accuracy if ground truth available
        if ground_truth and diagnosis_result:
            is_correct = diagnosis_result == ground_truth
            learning_result['accuracy_feedback'] = {
                'predicted': diagnosis_result,
                'actual': ground_truth,
                'correct': is_correct
            }

            # Learn from mistake if incorrect
            if not is_correct:
                self._learn_from_misdiagnosis(
                    features, diagnosis_result, ground_truth
                )

        # Try to match existing patterns
        matched_signature = self._match_signature(feature_vector)

        if matched_signature:
            # Update existing pattern
            self._update_signature(matched_signature, features, timestamp)
            learning_result['patterns_updated'].append(matched_signature.signature_id)
        else:
            # Try to create new pattern
            new_signature = self._try_create_signature(
                features, diagnosis_result or ground_truth or 'unknown',
                timestamp
            )
            if new_signature:
                learning_result['new_patterns'].append(new_signature.signature_id)

        # Update feature importance
        self._update_feature_importance(features, ground_truth or diagnosis_result)

        # Update diagnosis rules
        if ground_truth:
            updated_rules = self._update_diagnosis_rules(features, ground_truth)
            learning_result['rules_updated'] = updated_rules

        return learning_result

    def _extract_features(self, fault_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from fault data"""
        features = {}

        # Basic fault properties
        if 'severity' in fault_data:
            features['severity'] = float(fault_data['severity'])

        if 'confidence' in fault_data:
            features['confidence'] = float(fault_data['confidence'])

        if 'affected_components' in fault_data:
            features['component_count'] = float(len(fault_data['affected_components']))

        # Sensor readings
        if 'sensor_values' in fault_data:
            for sensor_id, value in fault_data['sensor_values'].items():
                if isinstance(value, (int, float)):
                    features[f'sensor_{sensor_id}'] = float(value)

        # Deviation metrics
        if 'deviations' in fault_data:
            for metric, value in fault_data['deviations'].items():
                features[f'deviation_{metric}'] = float(value)

        # Temporal features
        if 'duration_seconds' in fault_data:
            features['duration'] = float(fault_data['duration_seconds'])

        if 'occurrence_rate' in fault_data:
            features['occurrence_rate'] = float(fault_data['occurrence_rate'])

        # System state features
        if 'system_load' in fault_data:
            features['system_load'] = float(fault_data['system_load'])

        if 'communication_latency' in fault_data:
            features['comm_latency'] = float(fault_data['communication_latency'])

        return features

    def _match_signature(
        self,
        feature_vector: List[float],
        threshold: float = 0.8
    ) -> Optional[FaultSignature]:
        """Match feature vector to existing signatures"""
        best_match = None
        best_score = threshold

        for signature in self.fault_signatures.values():
            if signature.status in [PatternStatus.DEPRECATED, PatternStatus.ARCHIVED]:
                continue

            # Calculate similarity
            if len(signature.feature_vector) != len(feature_vector):
                continue

            similarity = self._calculate_similarity(
                feature_vector, signature.feature_vector
            )

            if similarity > best_score:
                best_score = similarity
                best_match = signature

        return best_match

    def _calculate_similarity(
        self,
        vector1: List[float],
        vector2: List[float]
    ) -> float:
        """Calculate cosine similarity between feature vectors"""
        if not vector1 or not vector2:
            return 0.0

        v1 = np.array(vector1)
        v2 = np.array(vector2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _update_signature(
        self,
        signature: FaultSignature,
        features: Dict[str, float],
        timestamp: datetime
    ):
        """Update existing signature with new observation"""
        # Incremental update of feature vector
        new_vector = list(features.values())

        if len(new_vector) == len(signature.feature_vector):
            # Exponential moving average
            alpha = 1.0 / (signature.occurrence_count + 1)
            signature.feature_vector = [
                (1 - alpha) * old + alpha * new
                for old, new in zip(signature.feature_vector, new_vector)
            ]

        # Update metadata
        signature.occurrence_count += 1
        signature.last_seen = timestamp

        # Update threshold ranges
        for fname, fvalue in features.items():
            if fname in signature.threshold_ranges:
                low, high = signature.threshold_ranges[fname]
                signature.threshold_ranges[fname] = (
                    min(low, fvalue),
                    max(high, fvalue)
                )
            else:
                margin = abs(fvalue) * 0.1 + 0.01
                signature.threshold_ranges[fname] = (fvalue - margin, fvalue + margin)

        # Update confidence based on occurrences
        if signature.occurrence_count >= 10:
            signature.confidence = ConfidenceLevel.HIGH
        elif signature.occurrence_count >= 5:
            signature.confidence = ConfidenceLevel.MEDIUM

    def _try_create_signature(
        self,
        features: Dict[str, float],
        fault_type: str,
        timestamp: datetime
    ) -> Optional[FaultSignature]:
        """Try to create new signature from observations"""
        # Need minimum samples of same type
        same_type_samples = [
            h for h in self.fault_history[-100:]  # Last 100 samples
            if h.get('ground_truth') == fault_type or h.get('diagnosis') == fault_type
        ]

        if len(same_type_samples) < self.min_samples_for_pattern:
            return None

        # Create signature
        feature_vector = list(features.values())
        feature_names = list(features.keys())

        # Calculate threshold ranges from samples
        threshold_ranges = {}
        for fname in feature_names:
            values = [
                s['features'].get(fname, 0)
                for s in same_type_samples
                if fname in s.get('features', {})
            ]
            if values:
                threshold_ranges[fname] = (min(values), max(values))

        # Generate signature ID
        signature_id = hashlib.md5(
            f"{fault_type}_{timestamp.isoformat()}".encode()
        ).hexdigest()[:12]

        signature = FaultSignature(
            signature_id=signature_id,
            fault_type=fault_type,
            feature_vector=feature_vector,
            feature_names=feature_names,
            threshold_ranges=threshold_ranges,
            temporal_pattern=None,
            occurrence_count=len(same_type_samples),
            first_seen=timestamp,
            last_seen=timestamp,
            confidence=ConfidenceLevel.LOW,
            status=PatternStatus.CANDIDATE
        )

        self.fault_signatures[signature_id] = signature
        self.pending_validation.append(signature_id)

        logger.info(f"Created new fault signature: {signature_id} for {fault_type}")
        return signature

    def _learn_from_misdiagnosis(
        self,
        features: Dict[str, float],
        predicted: str,
        actual: str
    ):
        """Learn from diagnosis mistakes"""
        # Find discriminating features
        predicted_samples = [
            h for h in self.fault_history[-200:]
            if h.get('ground_truth') == predicted
        ]
        actual_samples = [
            h for h in self.fault_history[-200:]
            if h.get('ground_truth') == actual
        ]

        if not predicted_samples or not actual_samples:
            return

        # Calculate feature differences
        for fname, fvalue in features.items():
            pred_values = [
                s['features'].get(fname, 0)
                for s in predicted_samples
                if fname in s.get('features', {})
            ]
            actual_values = [
                s['features'].get(fname, 0)
                for s in actual_samples
                if fname in s.get('features', {})
            ]

            if pred_values and actual_values:
                pred_mean = np.mean(pred_values)
                actual_mean = np.mean(actual_values)

                # This feature helps discriminate
                if abs(pred_mean - actual_mean) > 0.1:
                    # Update feature importance
                    if fname in self.feature_importance:
                        fi = self.feature_importance[fname]
                        fi.importance_score = min(1.0, fi.importance_score + 0.1)
                        fi.correlation_with_fault[actual] = (
                            fi.correlation_with_fault.get(actual, 0) + 0.1
                        )

    def _update_feature_importance(
        self,
        features: Dict[str, float],
        fault_type: Optional[str]
    ):
        """Update feature importance based on observations"""
        timestamp = datetime.now()

        for fname, fvalue in features.items():
            if fname not in self.feature_importance:
                self.feature_importance[fname] = FeatureImportance(
                    feature_name=fname,
                    importance_score=0.5,
                    correlation_with_fault={},
                    stability=0.5,
                    last_updated=timestamp
                )

            fi = self.feature_importance[fname]
            fi.last_updated = timestamp

            if fault_type:
                # Track correlation with fault type
                if fault_type not in fi.correlation_with_fault:
                    fi.correlation_with_fault[fault_type] = 0.0

                # Simple exponential update
                fi.correlation_with_fault[fault_type] = (
                    0.9 * fi.correlation_with_fault[fault_type] + 0.1
                )

    def _update_diagnosis_rules(
        self,
        features: Dict[str, float],
        fault_type: str
    ) -> List[str]:
        """Update diagnosis rules from labeled data"""
        updated_rules = []

        # Find or create rule for this fault type
        rule_key = f"rule_{fault_type}"

        if rule_key not in self.diagnosis_rules:
            # Create new rule
            conditions = []
            for fname, fvalue in features.items():
                margin = abs(fvalue) * 0.2 + 0.01
                conditions.append({
                    'feature': fname,
                    'operator': 'between',
                    'value': (fvalue - margin, fvalue + margin)
                })

            self.diagnosis_rules[rule_key] = DiagnosisRule(
                rule_id=rule_key,
                conditions=conditions[:5],  # Top 5 conditions
                conclusion=fault_type,
                confidence=0.5,
                support=1,
                lift=1.0,
                source='incremental_learning'
            )
            updated_rules.append(rule_key)
        else:
            # Update existing rule
            rule = self.diagnosis_rules[rule_key]
            rule.support += 1

            # Refine conditions based on new data
            for condition in rule.conditions:
                fname = condition['feature']
                if fname in features:
                    fvalue = features[fname]
                    low, high = condition['value']

                    # Expand range if needed
                    if fvalue < low:
                        condition['value'] = (fvalue, high)
                    elif fvalue > high:
                        condition['value'] = (low, fvalue)

            # Update confidence based on support
            rule.confidence = min(0.95, 0.5 + (rule.support / 100))
            updated_rules.append(rule_key)

        return updated_rules

    def learn_from_recovery(
        self,
        recovery_outcome: RecoveryOutcome
    ) -> Dict[str, Any]:
        """
        Learn from recovery outcome to improve strategy selection.

        Args:
            recovery_outcome: Outcome of a recovery action

        Returns:
            Learning result with strategy updates
        """
        self.recovery_outcomes.append(recovery_outcome)

        # Update strategy statistics
        key = f"{recovery_outcome.fault_type}_{recovery_outcome.strategy}"
        stats = self.strategy_stats[key]

        stats['total_count'] += 1
        if recovery_outcome.success:
            stats['success_count'] += 1

        # Update average time (exponential moving average)
        time_seconds = recovery_outcome.time_to_recovery.total_seconds()
        if stats['avg_time'] == 0:
            stats['avg_time'] = time_seconds
        else:
            stats['avg_time'] = 0.9 * stats['avg_time'] + 0.1 * time_seconds

        # Calculate success rate
        success_rate = stats['success_count'] / stats['total_count']

        return {
            'strategy_key': key,
            'success_rate': success_rate,
            'avg_recovery_time': stats['avg_time'],
            'total_samples': stats['total_count']
        }

    def learn_causal_relation(
        self,
        cause_event: Dict[str, Any],
        effect_event: Dict[str, Any],
        delay: timedelta
    ) -> Optional[CausalRelation]:
        """
        Learn causal relationship between fault events.

        Args:
            cause_event: Event believed to be cause
            effect_event: Event believed to be effect
            delay: Time between events

        Returns:
            New or updated causal relation
        """
        cause_type = cause_event.get('fault_type', 'unknown')
        effect_type = effect_event.get('fault_type', 'unknown')
        cause_component = cause_event.get('component', 'unknown')
        effect_component = effect_event.get('component', 'unknown')

        relation_id = f"{cause_type}_{cause_component}_{effect_type}_{effect_component}"

        if relation_id in self.causal_relations:
            # Update existing relation
            relation = self.causal_relations[relation_id]
            relation.observation_count += 1

            # Update delay statistics
            old_delay = relation.typical_delay.total_seconds()
            new_delay = delay.total_seconds()

            # Exponential moving average for delay
            alpha = 1.0 / relation.observation_count
            updated_delay = (1 - alpha) * old_delay + alpha * new_delay
            relation.typical_delay = timedelta(seconds=updated_delay)

            # Update variance
            diff = new_delay - updated_delay
            relation.delay_variance = (1 - alpha) * relation.delay_variance + alpha * (diff ** 2)

            # Update probability estimate
            relation.probability = min(0.95, 0.5 + (relation.observation_count / 50))

            # Update confidence
            if relation.observation_count >= 20:
                relation.confidence = ConfidenceLevel.HIGH
            elif relation.observation_count >= 10:
                relation.confidence = ConfidenceLevel.MEDIUM

            return relation
        else:
            # Create new relation
            relation = CausalRelation(
                relation_id=relation_id,
                cause_type=cause_type,
                effect_type=effect_type,
                cause_component=cause_component,
                effect_component=effect_component,
                probability=0.5,
                typical_delay=delay,
                delay_variance=0.0,
                observation_count=1,
                confidence=ConfidenceLevel.LOW,
                conditions=[]
            )

            self.causal_relations[relation_id] = relation
            return relation

    def get_best_strategy(
        self,
        fault_type: str,
        system_state: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Get best recovery strategy based on learned outcomes.

        Args:
            fault_type: Type of fault to recover from
            system_state: Current system state

        Returns:
            Tuple of (best_strategy, expected_success_rate)
        """
        candidates = []

        for key, stats in self.strategy_stats.items():
            if key.startswith(fault_type):
                strategy = key.split('_', 1)[1] if '_' in key else key
                if stats['total_count'] >= 3:  # Minimum samples
                    success_rate = stats['success_count'] / stats['total_count']
                    candidates.append((strategy, success_rate, stats['avg_time']))

        if not candidates:
            return ('adaptive', 0.5)  # Default strategy

        # Sort by success rate, then by time
        candidates.sort(key=lambda x: (-x[1], x[2]))

        best = candidates[0]
        return (best[0], best[1])

    def validate_signature(
        self,
        signature_id: str,
        is_valid: bool,
        expert_notes: Optional[str] = None
    ) -> bool:
        """
        Validate a candidate signature.

        Args:
            signature_id: ID of signature to validate
            is_valid: Whether signature is valid
            expert_notes: Optional notes from expert

        Returns:
            Success status
        """
        if signature_id not in self.fault_signatures:
            return False

        signature = self.fault_signatures[signature_id]

        if is_valid:
            signature.status = PatternStatus.VALIDATED
            signature.confidence = ConfidenceLevel.HIGH
            if expert_notes:
                signature.metadata['expert_notes'] = expert_notes
        else:
            signature.status = PatternStatus.DEPRECATED
            signature.confidence = ConfidenceLevel.VERY_LOW

        if signature_id in self.pending_validation:
            self.pending_validation.remove(signature_id)

        return True

    def activate_signature(self, signature_id: str) -> bool:
        """Activate a validated signature for use in detection"""
        if signature_id not in self.fault_signatures:
            return False

        signature = self.fault_signatures[signature_id]

        if signature.status == PatternStatus.VALIDATED:
            signature.status = PatternStatus.ACTIVE
            return True

        return False

    def get_diagnosis_rules_for_fault(
        self,
        fault_type: str
    ) -> List[DiagnosisRule]:
        """Get all diagnosis rules for a fault type"""
        return [
            rule for rule in self.diagnosis_rules.values()
            if rule.conclusion == fault_type
        ]

    def get_causal_chain(
        self,
        fault_type: str,
        max_depth: int = 5
    ) -> List[List[CausalRelation]]:
        """
        Get potential causal chains leading to a fault.

        Args:
            fault_type: Target fault type
            max_depth: Maximum chain depth

        Returns:
            List of causal chains
        """
        chains = []

        def find_causes(effect_type: str, current_chain: List[CausalRelation], depth: int):
            if depth >= max_depth:
                return

            for relation in self.causal_relations.values():
                if relation.effect_type == effect_type and relation.probability > 0.3:
                    if relation not in current_chain:  # Avoid cycles
                        new_chain = current_chain + [relation]
                        chains.append(new_chain)
                        find_causes(relation.cause_type, new_chain, depth + 1)

        find_causes(fault_type, [], 0)

        # Sort by chain probability
        def chain_probability(chain):
            return np.prod([r.probability for r in chain])

        chains.sort(key=chain_probability, reverse=True)
        return chains[:10]  # Top 10 chains

    def get_important_features(
        self,
        fault_type: Optional[str] = None,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get most important features for diagnosis.

        Args:
            fault_type: Optional specific fault type
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        features = []

        for fname, fi in self.feature_importance.items():
            if fault_type:
                score = fi.correlation_with_fault.get(fault_type, 0) * fi.importance_score
            else:
                score = fi.importance_score

            features.append((fname, score))

        features.sort(key=lambda x: x[1], reverse=True)
        return features[:top_n]

    def get_learning_metrics(self) -> LearningMetrics:
        """Get current learning performance metrics"""
        # Calculate accuracy from recent history
        recent = self.fault_history[-100:]
        correct = sum(
            1 for h in recent
            if h.get('ground_truth') and h.get('diagnosis') == h.get('ground_truth')
        )
        labeled = sum(1 for h in recent if h.get('ground_truth'))
        accuracy = correct / labeled if labeled > 0 else 0.0

        # Recovery success rate
        recent_recoveries = self.recovery_outcomes[-100:]
        recovery_success = (
            sum(1 for r in recent_recoveries if r.success) / len(recent_recoveries)
            if recent_recoveries else 0.0
        )

        # Average times
        avg_recovery_time = (
            np.mean([r.time_to_recovery.total_seconds() for r in recent_recoveries])
            if recent_recoveries else 0.0
        )

        return LearningMetrics(
            total_samples=len(self.fault_history),
            patterns_learned=len(self.fault_signatures),
            patterns_validated=sum(
                1 for s in self.fault_signatures.values()
                if s.status in [PatternStatus.VALIDATED, PatternStatus.ACTIVE]
            ),
            false_positive_rate=0.0,  # Would need more tracking
            false_negative_rate=0.0,  # Would need more tracking
            diagnosis_accuracy=accuracy,
            recovery_success_rate=recovery_success,
            avg_time_to_diagnosis=0.0,  # Would need timestamp tracking
            avg_time_to_recovery=avg_recovery_time,
            model_version=self.model_version,
            last_updated=datetime.now()
        )

    def export_knowledge(self) -> Dict[str, Any]:
        """Export learned knowledge for persistence"""
        return {
            'version': self.model_version,
            'exported_at': datetime.now().isoformat(),
            'signatures': {
                sig_id: {
                    'fault_type': sig.fault_type,
                    'feature_vector': sig.feature_vector,
                    'feature_names': sig.feature_names,
                    'threshold_ranges': sig.threshold_ranges,
                    'occurrence_count': sig.occurrence_count,
                    'confidence': sig.confidence.value,
                    'status': sig.status.value
                }
                for sig_id, sig in self.fault_signatures.items()
            },
            'causal_relations': {
                rel_id: {
                    'cause_type': rel.cause_type,
                    'effect_type': rel.effect_type,
                    'probability': rel.probability,
                    'typical_delay_seconds': rel.typical_delay.total_seconds(),
                    'observation_count': rel.observation_count
                }
                for rel_id, rel in self.causal_relations.items()
            },
            'diagnosis_rules': {
                rule_id: {
                    'conditions': rule.conditions,
                    'conclusion': rule.conclusion,
                    'confidence': rule.confidence,
                    'support': rule.support
                }
                for rule_id, rule in self.diagnosis_rules.items()
            },
            'strategy_stats': dict(self.strategy_stats),
            'feature_importance': {
                fname: {
                    'score': fi.importance_score,
                    'correlations': fi.correlation_with_fault
                }
                for fname, fi in self.feature_importance.items()
            }
        }

    def import_knowledge(self, data: Dict[str, Any]) -> bool:
        """Import previously exported knowledge"""
        try:
            # Import signatures
            for sig_id, sig_data in data.get('signatures', {}).items():
                self.fault_signatures[sig_id] = FaultSignature(
                    signature_id=sig_id,
                    fault_type=sig_data['fault_type'],
                    feature_vector=sig_data['feature_vector'],
                    feature_names=sig_data['feature_names'],
                    threshold_ranges=sig_data['threshold_ranges'],
                    temporal_pattern=None,
                    occurrence_count=sig_data['occurrence_count'],
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    confidence=ConfidenceLevel(sig_data['confidence']),
                    status=PatternStatus(sig_data['status'])
                )

            # Import causal relations
            for rel_id, rel_data in data.get('causal_relations', {}).items():
                self.causal_relations[rel_id] = CausalRelation(
                    relation_id=rel_id,
                    cause_type=rel_data['cause_type'],
                    effect_type=rel_data['effect_type'],
                    cause_component='',
                    effect_component='',
                    probability=rel_data['probability'],
                    typical_delay=timedelta(seconds=rel_data['typical_delay_seconds']),
                    delay_variance=0.0,
                    observation_count=rel_data['observation_count'],
                    confidence=ConfidenceLevel.MEDIUM,
                    conditions=[]
                )

            # Import diagnosis rules
            for rule_id, rule_data in data.get('diagnosis_rules', {}).items():
                self.diagnosis_rules[rule_id] = DiagnosisRule(
                    rule_id=rule_id,
                    conditions=rule_data['conditions'],
                    conclusion=rule_data['conclusion'],
                    confidence=rule_data['confidence'],
                    support=rule_data['support'],
                    lift=1.0,
                    source='imported'
                )

            # Import strategy stats
            for key, stats in data.get('strategy_stats', {}).items():
                self.strategy_stats[key] = stats

            self.model_version = data.get('version', self.model_version)
            return True

        except Exception as e:
            logger.error(f"Failed to import knowledge: {e}")
            return False

    def decay_old_patterns(self, decay_threshold_days: int = 90):
        """
        Apply time-based decay to old patterns.

        Patterns not seen recently have reduced confidence.
        """
        cutoff = datetime.now() - timedelta(days=decay_threshold_days)

        for signature in self.fault_signatures.values():
            if signature.last_seen < cutoff:
                # Reduce confidence
                current = signature.confidence.value
                if current > 1:
                    signature.confidence = ConfidenceLevel(current - 1)

                # Archive very old patterns
                age_days = (datetime.now() - signature.last_seen).days
                if age_days > decay_threshold_days * 2:
                    signature.status = PatternStatus.ARCHIVED
