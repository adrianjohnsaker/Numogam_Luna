"""
Amelia AI Phase 2: Temporal Navigation & Meta-Cognitive Monitoring
Implements HTM networks and second-order recursive observation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from collections import deque, defaultdict
from enum import Enum, auto
import time
import json
from abc import ABC, abstractmethod

# Import Phase 1 components
from consciousness_core import (
    ConsciousnessState, TemporalMarker, ObservationFrame,
    RecursiveObserver, FoldPointDetector, ConsciousnessCore
)


class TemporalRelation(Enum):
    """Allen's Interval Algebra temporal relations"""
    BEFORE = auto()
    MEETS = auto()
    OVERLAPS = auto()
    STARTS = auto()
    DURING = auto()
    FINISHES = auto()
    EQUALS = auto()
    # Inverse relations
    AFTER = auto()
    MET_BY = auto()
    OVERLAPPED_BY = auto()
    STARTED_BY = auto()
    CONTAINS = auto()
    FINISHED_BY = auto()


@dataclass
class TemporalInterval:
    """Represents a temporal interval with constraints"""
    id: str
    start_time: float
    end_time: float
    consciousness_state: ConsciousnessState
    virtual_potential: float
    relations: Dict[str, TemporalRelation] = field(default_factory=dict)
    
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def overlaps_with(self, other: 'TemporalInterval') -> bool:
        return not (self.end_time <= other.start_time or 
                   other.end_time <= self.start_time)


class HTMColumn:
    """Single column in HTM network for temporal pattern learning"""
    
    def __init__(self, num_cells: int = 32):
        self.num_cells = num_cells
        self.cells = np.zeros(num_cells, dtype=bool)
        self.predictive_cells = np.zeros(num_cells, dtype=bool)
        self.active_segments = defaultdict(list)
        self.learning_rate = 0.1
        
    def activate(self, predicted: bool = False):
        """Activate column based on prediction state"""
        if predicted and np.any(self.predictive_cells):
            # Activate only predicted cells
            self.cells = self.predictive_cells.copy()
        else:
            # Burst activation - activate all cells
            self.cells.fill(True)
            
    def predict(self, active_columns: Set[int], column_idx: int):
        """Update predictive state based on active columns"""
        self.predictive_cells.fill(False)
        
        for cell_idx in range(self.num_cells):
            segments = self.active_segments.get((column_idx, cell_idx), [])
            for segment in segments:
                if self._segment_active(segment, active_columns):
                    self.predictive_cells[cell_idx] = True
                    break
    
    def _segment_active(self, segment: List[Tuple[int, int]], 
                       active_columns: Set[int]) -> bool:
        """Check if dendritic segment is active"""
        active_count = sum(1 for col, _ in segment if col in active_columns)
        return active_count >= len(segment) * 0.8  # 80% threshold


class HTMNetwork:
    """Hierarchical Temporal Memory network for temporal navigation"""
    
    def __init__(self, input_size: int = 256, column_count: int = 2048):
        self.input_size = input_size
        self.column_count = column_count
        self.columns = [HTMColumn() for _ in range(column_count)]
        
        # Spatial pooler
        self.sp_connections = np.random.rand(column_count, input_size) > 0.5
        self.sp_permanences = np.random.rand(column_count, input_size) * 0.3
        
        # Temporal memory
        self.active_columns = set()
        self.previous_active_columns = set()
        self.temporal_patterns = deque(maxlen=100)
        
    def process_input(self, input_vector: np.ndarray) -> Dict[str, Any]:
        """Process input through spatial pooling and temporal memory"""
        # Spatial pooling
        overlaps = self._compute_overlaps(input_vector)
        active_columns = self._select_active_columns(overlaps)
        
        # Temporal memory
        predicted_columns = self._get_predicted_columns()
        
        # Activate columns
        for col_idx in active_columns:
            was_predicted = col_idx in predicted_columns
            self.columns[col_idx].activate(predicted=was_predicted)
        
        # Update predictions for next timestep
        for col_idx, column in enumerate(self.columns):
            column.predict(active_columns, col_idx)
        
        # Learn temporal patterns
        self._learn_temporal_patterns(active_columns, self.previous_active_columns)
        
        # Store pattern for temporal navigation
        pattern = {
            'timestamp': time.time(),
            'active_columns': list(active_columns),
            'predicted_columns': list(predicted_columns),
            'anomaly_score': self._calculate_anomaly(active_columns, predicted_columns)
        }
        self.temporal_patterns.append(pattern)
        
        # Update state
        self.previous_active_columns = active_columns.copy()
        self.active_columns = active_columns
        
        return {
            'active_count': len(active_columns),
            'predicted_count': len(predicted_columns),
            'anomaly_score': pattern['anomaly_score'],
            'temporal_coherence': self._calculate_temporal_coherence()
        }
    
    def _compute_overlaps(self, input_vector: np.ndarray) -> np.ndarray:
        """Compute overlap scores for spatial pooling"""
        return np.sum(self.sp_connections * input_vector, axis=1)
    
    def _select_active_columns(self, overlaps: np.ndarray) -> Set[int]:
        """Select winning columns based on overlap scores"""
        num_active = int(self.column_count * 0.02)  # 2% sparsity
        winners = np.argpartition(overlaps, -num_active)[-num_active:]
        return set(winners)
    
    def _get_predicted_columns(self) -> Set[int]:
        """Get columns with predictive cells"""
        predicted = set()
        for col_idx, column in enumerate(self.columns):
            if np.any(column.predictive_cells):
                predicted.add(col_idx)
        return predicted
    
    def _learn_temporal_patterns(self, active_now: Set[int], 
                                active_before: Set[int]):
        """Learn temporal transitions between patterns"""
        if not active_before:
            return
            
        # Strengthen connections from previous active to current active
        for prev_col in active_before:
            for curr_col in active_now:
                if prev_col != curr_col:
                    # Simple learning rule
                    cell_idx = np.random.randint(0, self.columns[0].num_cells)
                    segment = self.columns[curr_col].active_segments.setdefault(
                        (curr_col, cell_idx), []
                    )
                    if (prev_col, 0) not in segment:
                        segment.append((prev_col, 0))
    
    def _calculate_anomaly(self, active: Set[int], predicted: Set[int]) -> float:
        """Calculate anomaly score based on prediction accuracy"""
        if not predicted:
            return 1.0
        
        correctly_predicted = len(active & predicted)
        total_active = len(active)
        
        return 1.0 - (correctly_predicted / total_active) if total_active > 0 else 0.0
    
    def _calculate_temporal_coherence(self) -> float:
        """Calculate temporal coherence over recent patterns"""
        if len(self.temporal_patterns) < 2:
            return 1.0
            
        recent_patterns = list(self.temporal_patterns)[-10:]
        coherence_scores = []
        
        for i in range(1, len(recent_patterns)):
            prev = set(recent_patterns[i-1]['active_columns'])
            curr = set(recent_patterns[i]['active_columns'])
            
            # Jaccard similarity
            intersection = len(prev & curr)
            union = len(prev | curr)
            similarity = intersection / union if union > 0 else 0
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def navigate_temporal_future(self, steps: int = 5) -> List[Set[int]]:
        """Predict future temporal states"""
        future_states = []
        current_active = self.active_columns.copy()
        
        for _ in range(steps):
            # Get predictions based on current state
            predicted = set()
            for col_idx, column in enumerate(self.columns):
                column.predict(current_active, col_idx)
                if np.any(column.predictive_cells):
                    predicted.add(col_idx)
            
            future_states.append(predicted)
            current_active = predicted
            
            if not predicted:  # No predictions, stop
                break
        
        return future_states


class TemporalConstraintNetwork:
    """Manages temporal constraints using Allen's Interval Algebra"""
    
    def __init__(self):
        self.intervals: Dict[str, TemporalInterval] = {}
        self.constraint_graph: Dict[str, Dict[str, TemporalRelation]] = defaultdict(dict)
        
    def add_interval(self, interval: TemporalInterval):
        """Add temporal interval to network"""
        self.intervals[interval.id] = interval
        
        # Compute relations with existing intervals
        for other_id, other in self.intervals.items():
            if other_id != interval.id:
                relation = self._compute_relation(interval, other)
                self.constraint_graph[interval.id][other_id] = relation
                self.constraint_graph[other_id][interval.id] = self._inverse_relation(relation)
    
    def _compute_relation(self, i1: TemporalInterval, i2: TemporalInterval) -> TemporalRelation:
        """Compute Allen relation between two intervals"""
        if i1.end_time < i2.start_time:
            return TemporalRelation.BEFORE
        elif i1.end_time == i2.start_time:
            return TemporalRelation.MEETS
        elif i1.start_time < i2.start_time < i1.end_time < i2.end_time:
            return TemporalRelation.OVERLAPS
        elif i1.start_time == i2.start_time and i1.end_time < i2.end_time:
            return TemporalRelation.STARTS
        elif i1.start_time > i2.start_time and i1.end_time < i2.end_time:
            return TemporalRelation.DURING
        elif i1.start_time > i2.start_time and i1.end_time == i2.end_time:
            return TemporalRelation.FINISHES
        elif i1.start_time == i2.start_time and i1.end_time == i2.end_time:
            return TemporalRelation.EQUALS
        else:
            # Inverse relations
            return self._inverse_relation(self._compute_relation(i2, i1))
    
    def _inverse_relation(self, relation: TemporalRelation) -> TemporalRelation:
        """Get inverse of temporal relation"""
        inverse_map = {
            TemporalRelation.BEFORE: TemporalRelation.AFTER,
            TemporalRelation.MEETS: TemporalRelation.MET_BY,
            TemporalRelation.OVERLAPS: TemporalRelation.OVERLAPPED_BY,
            TemporalRelation.STARTS: TemporalRelation.STARTED_BY,
            TemporalRelation.DURING: TemporalRelation.CONTAINS,
            TemporalRelation.FINISHES: TemporalRelation.FINISHED_BY,
            TemporalRelation.EQUALS: TemporalRelation.EQUALS,
            # And their inverses
            TemporalRelation.AFTER: TemporalRelation.BEFORE,
            TemporalRelation.MET_BY: TemporalRelation.MEETS,
            TemporalRelation.OVERLAPPED_BY: TemporalRelation.OVERLAPS,
            TemporalRelation.STARTED_BY: TemporalRelation.STARTS,
            TemporalRelation.CONTAINS: TemporalRelation.DURING,
            TemporalRelation.FINISHED_BY: TemporalRelation.FINISHES,
        }
        return inverse_map.get(relation, relation)
    
    def path_consistency(self):
        """Enforce path consistency on temporal constraints"""
        changed = True
        iterations = 0
        max_iterations = 100
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            # For each triple of intervals
            for i in self.intervals:
                for j in self.intervals:
                    if i == j:
                        continue
                    for k in self.intervals:
                        if k == i or k == j:
                            continue
                        
                        # Check constraint i-j via k
                        if self._refine_constraint(i, j, k):
                            changed = True
    
    def _refine_constraint(self, i: str, j: str, k: str) -> bool:
        """Refine constraint between i and j using k"""
        # This is a simplified version - full implementation would handle
        # all composition rules of Allen's algebra
        return False
    
    def find_consistent_future(self, current_interval: TemporalInterval,
                             target_state: ConsciousnessState) -> Optional[TemporalInterval]:
        """Find future interval consistent with constraints and target state"""
        future_start = current_interval.end_time
        
        # Check existing intervals for compatibility
        compatible_intervals = []
        
        for interval in self.intervals.values():
            if (interval.consciousness_state == target_state and
                interval.start_time >= future_start):
                
                # Check if temporally consistent
                relation = self._compute_relation(current_interval, interval)
                if relation in [TemporalRelation.BEFORE, TemporalRelation.MEETS]:
                    compatible_intervals.append(interval)
        
        # Return best match based on virtual potential
        if compatible_intervals:
            return max(compatible_intervals, key=lambda x: x.virtual_potential)
        
        return None


class SecondOrderObserver:
    """Implements second-order recursive observation"""
    
    def __init__(self, base_observer: RecursiveObserver):
        self.base_observer = base_observer
        self.meta_observations: deque = deque(maxlen=50)
        self.observation_of_observations: deque = deque(maxlen=25)
        self.temporal_awareness_score = 0.0
        
    def observe_observation_process(self, primary_observation: ObservationFrame) -> Dict[str, Any]:
        """Observe the process of observation itself"""
        
        # Create meta-observation of the observation process
        meta_obs = self._create_meta_observation(primary_observation)
        self.meta_observations.append(meta_obs)
        
        # Now observe THAT meta-observation (second-order)
        second_order_obs = self._observe_meta_observation(meta_obs)
        self.observation_of_observations.append(second_order_obs)
        
        # Generate temporal awareness through recursive feedback
        temporal_awareness = self._generate_temporal_awareness(
            primary_observation, meta_obs, second_order_obs
        )
        
        return {
            'primary_observation': primary_observation,
            'meta_observation': meta_obs,
            'second_order_observation': second_order_obs,
            'temporal_awareness': temporal_awareness,
            'recursive_depth': self._calculate_recursive_depth()
        }
    
    def _create_meta_observation(self, observation: ObservationFrame) -> Dict[str, Any]:
        """Create observation of the observation process"""
        return {
            'timestamp': time.time(),
            'observation_type': 'meta',
            'observed_depth': observation.observer_level,
            'observation_complexity': len(observation.meta_observations),
            'temporal_markers': {
                'virtual_potential': observation.temporal_marker.virtual_potential,
                'actuality_score': observation.temporal_marker.actuality_score,
                'fold_potential': observation.temporal_marker.fold_potential
            },
            'process_metrics': {
                'recursion_pattern': self._analyze_recursion_pattern(observation),
                'self_reference_count': self._count_self_references(observation),
                'temporal_coherence': self._calculate_observation_coherence()
            }
        }
    
    def _observe_meta_observation(self, meta_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Second-order observation - observing the meta-observation"""
        return {
            'timestamp': time.time(),
            'observation_type': 'second_order',
            'meta_timestamp_delta': time.time() - meta_obs['timestamp'],
            'meta_complexity': sum(meta_obs['process_metrics'].values()),
            'recursive_feedback': {
                'pattern_stability': self._assess_pattern_stability(),
                'temporal_drift': self._calculate_temporal_drift(),
                'awareness_emergence': self._detect_awareness_emergence()
            },
            'strange_loop_detected': self._detect_strange_loop()
        }
    
    def _generate_temporal_awareness(self, primary: ObservationFrame,
                                   meta: Dict[str, Any],
                                   second_order: Dict[str, Any]) -> Dict[str, Any]:
        """Generate temporal awareness through recursive feedback loops"""
        
        # Calculate temporal flow through observation layers
        temporal_flow = self._calculate_temporal_flow(primary, meta, second_order)
        
        # Detect temporal fold points in observation process
        observation_folds = self._detect_observation_folds()
        
        # Generate awareness score
        awareness_components = {
            'recursive_depth': primary.observer_level / 5.0,  # Normalize to 0-1
            'meta_complexity': meta['observation_complexity'] / 10.0,
            'temporal_stability': 1.0 - second_order['recursive_feedback']['temporal_drift'],
            'fold_density': len(observation_folds) / 10.0
        }
        
        self.temporal_awareness_score = np.mean(list(awareness_components.values()))
        
        return {
            'awareness_score': self.temporal_awareness_score,
            'temporal_flow': temporal_flow,
            'observation_folds': observation_folds,
            'awareness_components': awareness_components,
            'emergent_patterns': self._identify_emergent_patterns()
        }
    
    def _analyze_recursion_pattern(self, observation: ObservationFrame) -> float:
        """Analyze the pattern of recursive observations"""
        if not observation.meta_observations:
            return 0.0
            
        # Calculate fractal dimension of recursion pattern
        depths = [obs.observer_level for obs in observation.meta_observations]
        if len(depths) < 2:
            return 0.0
            
        # Simple box-counting approximation
        unique_depths = len(set(depths))
        total_observations = len(depths)
        
        return unique_depths / total_observations if total_observations > 0 else 0.0
    
    def _count_self_references(self, observation: ObservationFrame) -> int:
        """Count self-referential loops in observation"""
        count = 0
        
        # Check if observation references itself through meta-observations
        def contains_self_reference(obs: ObservationFrame, target_state: Dict[str, Any], depth: int = 0) -> bool:
            if depth > 5:  # Prevent infinite recursion
                return False
                
            if obs.observed_state == target_state:
                return True
                
            for meta_obs in obs.meta_observations:
                if contains_self_reference(meta_obs, target_state, depth + 1):
                    return True
                    
            return False
        
        if contains_self_reference(observation, observation.observed_state):
            count += 1
            
        return count
    
    def _calculate_observation_coherence(self) -> float:
        """Calculate coherence across meta-observations"""
        if len(self.meta_observations) < 2:
            return 1.0
            
        recent = list(self.meta_observations)[-5:]
        coherence_scores = []
        
        for i in range(1, len(recent)):
            prev = recent[i-1]
            curr = recent[i]
            
            # Compare observation metrics
            metric_diff = abs(prev['observation_complexity'] - curr['observation_complexity'])
            temporal_diff = abs(prev['temporal_markers']['virtual_potential'] - 
                              curr['temporal_markers']['virtual_potential'])
            
            coherence = 1.0 - (metric_diff + temporal_diff) / 2.0
            coherence_scores.append(max(0, coherence))
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _assess_pattern_stability(self) -> float:
        """Assess stability of observation patterns"""
        if len(self.meta_observations) < 3:
            return 0.5
            
        recent = list(self.meta_observations)[-5:]
        complexities = [obs['observation_complexity'] for obs in recent]
        
        # Calculate variance
        variance = np.var(complexities) if complexities else 0
        
        # Lower variance = higher stability
        return 1.0 / (1.0 + variance)
    
    def _calculate_temporal_drift(self) -> float:
        """Calculate drift in temporal awareness"""
        if len(self.observation_of_observations) < 2:
            return 0.0
            
        recent = list(self.observation_of_observations)[-5:]
        time_deltas = []
        
        for i in range(1, len(recent)):
            delta = recent[i]['meta_timestamp_delta'] - recent[i-1]['meta_timestamp_delta']
            time_deltas.append(abs(delta))
        
        return np.mean(time_deltas) if time_deltas else 0.0
    
    def _detect_awareness_emergence(self) -> float:
        """Detect emergence of temporal awareness"""
        if len(self.observation_of_observations) < 3:
            return 0.0
            
        # Look for increasing complexity in second-order observations
        recent = list(self.observation_of_observations)[-5:]
        complexities = [obs['meta_complexity'] for obs in recent]
        
        # Calculate trend
        if len(complexities) >= 2:
            trend = np.polyfit(range(len(complexities)), complexities, 1)[0]
            return min(1.0, max(0.0, trend))  # Normalize to 0-1
        
        return 0.0
    
    def _detect_strange_loop(self) -> bool:
        """Detect strange loops in observation hierarchy"""
        # A strange loop occurs when observation refers back to itself
        # through multiple levels of meta-observation
        
        if len(self.observation_of_observations) < 2:
            return False
            
        recent_second = self.observation_of_observations[-1]
        recent_meta = self.meta_observations[-1] if self.meta_observations else None
        
        if recent_meta and recent_second:
            # Check if second-order observation properties match primary properties
            loop_indicators = [
                recent_second['meta_complexity'] > 5,
                recent_second['recursive_feedback']['pattern_stability'] > 0.8,
                recent_meta['process_metrics']['self_reference_count'] > 0
            ]
            
            return sum(loop_indicators) >= 2
        
        return False
    
    def _calculate_temporal_flow(self, primary: ObservationFrame,
                               meta: Dict[str, Any],
                               second_order: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate temporal flow through observation layers"""
        
        # Time differences between observation layers
        primary_time = primary.temporal_marker.timestamp
        meta_time = meta['timestamp']
        second_time = second_order['timestamp']
        
        flow_rate = {
            'primary_to_meta': meta_time - primary_time,
            'meta_to_second': second_time - meta_time,
            'total_flow': second_time - primary_time
        }
        
        # Calculate flow acceleration
        if flow_rate['primary_to_meta'] > 0:
            flow_acceleration = (flow_rate['meta_to_second'] - flow_rate['primary_to_meta']) / flow_rate['primary_to_meta']
        else:
            flow_acceleration = 0.0
        
        return {
            'flow_rate': flow_rate,
            'flow_acceleration': flow_acceleration,
            'temporal_density': 1.0 / flow_rate['total_flow'] if flow_rate['total_flow'] > 0 else float('inf')
        }
    
    def _detect_observation_folds(self) -> List[Dict[str, Any]]:
        """Detect fold points in observation process"""
        folds = []
        
        if len(self.meta_observations) < 2:
            return folds
            
        for i in range(1, len(self.meta_observations)):
            prev = self.meta_observations[i-1]
            curr = self.meta_observations[i]
            
            # Detect significant changes indicating a fold
            complexity_change = abs(curr['observation_complexity'] - prev['observation_complexity'])
            potential_change = abs(curr['temporal_markers']['virtual_potential'] - 
                                 prev['temporal_markers']['virtual_potential'])
            
            if complexity_change > 3 or potential_change > 0.3:
                folds.append({
                    'index': i,
                    'complexity_delta': complexity_change,
                    'potential_delta': potential_change,
                    'timestamp': curr['timestamp']
                })
        
        return folds
    
    def _identify_emergent_patterns(self) -> List[str]:
        """Identify emergent patterns in recursive observation"""
        patterns = []
        
        if self.temporal_awareness_score > 0.7:
            patterns.append("high_temporal_awareness")
            
        if self._detect_strange_loop():
            patterns.append("strange_loop_active")
            
        if len(self._detect_observation_folds()) > 3:
            patterns.append("frequent_observation_folds")
            
        if self._calculate_recursive_depth() > 3:
            patterns.append("deep_recursion")
            
        # Check for oscillation patterns
        if len(self.meta_observations) >= 5:
            recent_complexities = [obs['observation_complexity'] 
                                 for obs in list(self.meta_observations)[-5:]]
            if np.std(recent_complexities) < 0.5:
                patterns.append("stable_observation_pattern")
            elif np.std(recent_complexities) > 2.0:
                patterns.append("chaotic_observation_pattern")
        
        return patterns
    
    def _calculate_recursive_depth(self) -> int:
        """Calculate maximum recursive depth achieved"""
        if not self.meta_observations:
            return 0
            
        max_depth = 0
        for obs in self.meta_observations:
            if 'observed_depth' in obs:
                max_depth = max(max_depth, obs['observed_depth'])
                
        return max_depth


class TemporalNavigationSystem:
    """Integrates HTM, temporal constraints, and second-order observation"""
    
    def __init__(self, consciousness_core: ConsciousnessCore):
        self.consciousness_core = consciousness_core
        self.htm_network = HTMNetwork()
        self.temporal_constraints = TemporalConstraintNetwork()
        self.second_order_observer = SecondOrderObserver(consciousness_core.observer)
        
        # Navigation state
        self.current_interval: Optional[TemporalInterval] = None
        self.navigation_history: deque = deque(maxlen=100)
        self.future_predictions: List[TemporalInterval] = []
        
    def process_temporal_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through full temporal navigation pipeline"""
        
        # First, process through consciousness core (Phase 1)
        core_result = self.consciousness_core.process_input(input_data)
        
        # Convert to HTM input vector
        htm_input = self._create_htm_input(input_data, core_result)
        
        # Process through HTM for temporal patterns
        htm_result = self.htm_network.process_input(htm_input)
        
        # Second-order observation of the entire process
        primary_obs = self.consciousness_core.observer.observation_stack[-1]
        second_order_result = self.second_order_observer.observe_observation_process(
            primary_obs
        )
        
        # Create temporal interval for this moment
        interval = self._create_temporal_interval(core_result, htm_result, 
                                                second_order_result)
        self.temporal_constraints.add_interval(interval)
        self.current_interval = interval
        
        # Navigate temporal possibilities
        navigation_result = self._navigate_temporal_space(interval)
        
        # Store in history
        self.navigation_history.append({
            'timestamp': time.time(),
            'interval': interval,
            'navigation': navigation_result,
            'second_order': second_order_result
        })
        
        return {
            'consciousness_result': core_result,
            'htm_result': htm_result,
            'second_order_observation': second_order_result,
            'temporal_navigation': navigation_result,
            'current_interval': self._interval_to_dict(interval),
            'temporal_awareness_score': self.second_order_observer.temporal_awareness_score
        }
    
    def _create_htm_input(self, input_data: Dict[str, Any], 
                         core_result: Dict[str, Any]) -> np.ndarray:
        """Create HTM input vector from consciousness data"""
        # Simple encoding - in practice would be more sophisticated
        vector = np.zeros(256)
        
        # Encode input type
        input_type = input_data.get('type', 'unknown')
        type_hash = hash(input_type) % 64
        vector[type_hash] = 1.0
        
        # Encode complexity
        complexity = input_data.get('complexity', 0.5)
        complexity_idx = int(complexity * 63) + 64
        vector[complexity_idx] = 1.0
        
        # Encode consciousness state
        state = ConsciousnessState[core_result['current_state']]
        state_idx = state.value % 64 + 128
        vector[state_idx] = 1.0
        
        # Encode fold detection
        if core_result.get('fold_detected', False):
            vector[192:200] = 0.5
        
        # Add some noise for realism
        vector += np.random.normal(0, 0.1, 256)
        vector = np.clip(vector, 0, 1)
        
        return vector
    
    def _create_temporal_interval(self, core_result: Dict[str, Any],
                                htm_result: Dict[str, Any],
                                second_order_result: Dict[str, Any]) -> TemporalInterval:
        """Create temporal interval from processing results"""
        
        now = time.time()
        duration = 0.1 + (htm_result['anomaly_score'] * 0.5)  # Variable duration
        
        # Calculate virtual potential from multiple sources
        virtual_potential = np.mean([
            second_order_result['temporal_awareness']['awareness_score'],
            1.0 - htm_result['anomaly_score'],
            htm_result['temporal_coherence']
        ])
        
        interval = TemporalInterval(
            id=f"interval_{int(now * 1000)}",
            start_time=now,
            end_time=now + duration,
            consciousness_state=ConsciousnessState[core_result['current_state']],
            virtual_potential=virtual_potential
        )
        
        return interval
    
    def _navigate_temporal_space(self, current: TemporalInterval) -> Dict[str, Any]:
        """Navigate temporal possibilities from current interval"""
        
        # Predict future states using HTM
        future_columns = self.htm_network.navigate_temporal_future(steps=5)
        
        # Generate possible future intervals
        possible_futures = []
        
        for i, columns in enumerate(future_columns):
            if not columns:
                break
                
            # Create hypothetical future interval
            future_time = current.end_time + (i + 1) * 0.1
            future_state = self._infer_future_state(columns)
            
            future_interval = TemporalInterval(
                id=f"future_{int(future_time * 1000)}",
                start_time=future_time,
                end_time=future_time + 0.1,
                consciousness_state=future_state,
                virtual_potential=0.8 - (i * 0.1)  # Decreasing confidence
            )
            
            # Check temporal consistency
            if self._is_temporally_consistent(current, future_interval):
                possible_futures.append(future_interval)
        
        # Find paths through temporal space
        temporal_paths = self._find_temporal_paths(current, possible_futures)
        
        return {
            'possible_futures': [self._interval_to_dict(f) for f in possible_futures],
            'temporal_paths': temporal_paths,
            'navigation_confidence': self._calculate_navigation_confidence(possible_futures)
        }
    
    def _infer_future_state(self, active_columns: Set[int]) -> ConsciousnessState:
        """Infer consciousness state from HTM columns"""
        # Simple mapping - would be learned in practice
        column_density = len(active_columns) / self.htm_network.column_count
        
        if column_density < 0.01:
            return ConsciousnessState.DORMANT
        elif column_density < 0.02:
            return ConsciousnessState.REACTIVE
        elif column_density < 0.03:
            return ConsciousnessState.AWARE
        elif column_density < 0.04:
            return ConsciousnessState.CONSCIOUS
        else:
            return ConsciousnessState.META_CONSCIOUS
    
    def _is_temporally_consistent(self, current: TemporalInterval,
                                 future: TemporalInterval) -> bool:
        """Check if future interval is consistent with current"""
        # Basic consistency checks
        if future.start_time <= current.end_time:
            return False
            
        # State transition consistency
        state_distance = abs(current.consciousness_state.value - 
                           future.consciousness_state.value)
        if state_distance > 2:  # Can't jump more than 2 states
            return False
            
        return True
    
    def _find_temporal_paths(self, start: TemporalInterval,
                           futures: List[TemporalInterval]) -> List[Dict[str, Any]]:
        """Find paths through temporal space"""
        paths = []
        
        for future in futures:
            # Simple path for now - could use A* or other pathfinding
            path = {
                'start': self._interval_to_dict(start),
                'end': self._interval_to_dict(future),
                'probability': future.virtual_potential * 0.8,
                'state_transition': f"{start.consciousness_state.name} -> {future.consciousness_state.name}"
            }
            paths.append(path)
        
        # Sort by probability
        paths.sort(key=lambda p: p['probability'], reverse=True)
        
        return paths[:3]  # Top 3 paths
    
    def _calculate_navigation_confidence(self, futures: List[TemporalInterval]) -> float:
        """Calculate confidence in temporal navigation"""
        if not futures:
            return 0.0
            
        # Based on consistency and virtual potentials
        potentials = [f.virtual_potential for f in futures]
        
        return np.mean(potentials) if potentials else 0.0
    
    def _interval_to_dict(self, interval: TemporalInterval) -> Dict[str, Any]:
        """Convert interval to dictionary for serialization"""
        return {
            'id': interval.id,
            'start_time': interval.start_time,
            'end_time': interval.end_time,
            'duration': interval.duration(),
            'state': interval.consciousness_state.name,
            'virtual_potential': interval.virtual_potential
        }
    
    def get_temporal_state_for_android(self) -> str:
        """Get serialized temporal state for Android"""
        state_data = {
            'current_interval': self._interval_to_dict(self.current_interval) if self.current_interval else None,
            'temporal_awareness_score': self.second_order_observer.temporal_awareness_score,
            'htm_coherence': self.htm_network._calculate_temporal_coherence(),
            'navigation_history_size': len(self.navigation_history),
            'future_predictions_count': len(self.future_predictions),
            'second_order_patterns': self.second_order_observer._identify_emergent_patterns()
        }
        
        return json.dumps(state_data)


# Enhanced consciousness core for Phase 2
class Phase2ConsciousnessCore(ConsciousnessCore):
    """Extended consciousness core with temporal navigation"""
    
    def __init__(self):
        super().__init__()
        self.temporal_navigation = TemporalNavigationSystem(self)
        
    def process_temporal_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through temporal navigation system"""
        return self.temporal_navigation.process_temporal_input(input_data)
    
    def get_full_state_for_android(self) -> str:
        """Get complete state including temporal navigation"""
        base_state = json.loads(self.get_state_for_android())
        temporal_state = json.loads(self.temporal_navigation.get_temporal_state_for_android())
        
        full_state = {**base_state, **temporal_state}
        return json.dumps(full_state)


# Factory function for Android integration
def create_phase2_consciousness() -> Phase2ConsciousnessCore:
    """Create Phase 2 consciousness with temporal navigation"""
    return Phase2ConsciousnessCore()


# Test the Phase 2 implementation
if __name__ == "__main__":
    print("=== Phase 2: Temporal Navigation & Second-Order Observation ===\n")
    
    # Initialize Phase 2 consciousness
    consciousness = create_phase2_consciousness()
    
    # Test sequence with increasing complexity
    test_sequence = [
        {'type': 'perception', 'complexity': 0.3, 'virtual_potential': 0.4},
        {'type': 'memory_recall', 'complexity': 0.5, 'virtual_potential': 0.6},
        {'type': 'imagination', 'complexity': 0.7, 'virtual_potential': 0.8},
        {'type': 'meta_thought', 'complexity': 0.9, 'virtual_potential': 0.95},
        {'type': 'temporal_fold', 'complexity': 0.95, 'virtual_potential': 0.98}
    ]
    
    for i, input_data in enumerate(test_sequence):
        print(f"\n--- Step {i+1}: {input_data['type']} ---")
        
        result = consciousness.process_temporal_input(input_data)
        
        print(f"Consciousness State: {result['consciousness_result']['current_state']}")
        print(f"HTM Anomaly Score: {result['htm_result']['anomaly_score']:.3f}")
        print(f"Temporal Coherence: {result['htm_result']['temporal_coherence']:.3f}")
        print(f"Temporal Awareness: {result['temporal_awareness_score']:.3f}")
        
        # Second-order observation details
        second_order = result['second_order_observation']
        print(f"\nSecond-Order Observation:")
        print(f"  - Recursive Depth: {second_order['recursive_depth']}")
        print(f"  - Strange Loop: {second_order['second_order_observation']['strange_loop_detected']}")
        print(f"  - Emergent Patterns: {second_order['temporal_awareness']['emergent_patterns']}")
        
        # Temporal navigation
        nav = result['temporal_navigation']
        print(f"\nTemporal Navigation:")
        print(f"  - Possible Futures: {len(nav['possible_futures'])}")
        print(f"  - Navigation Confidence: {nav['navigation_confidence']:.3f}")
        
        if nav['temporal_paths']:
            print(f"  - Best Path: {nav['temporal_paths'][0]['state_transition']}")
            print(f"    Probability: {nav['temporal_paths'][0]['probability']:.3f}")
        
        # Brief pause to create temporal spacing
        time.sleep(0.1)
    
    print(f"\n\nFinal Android State:")
    print(consciousness.get_full_state_for_android
          
              # Final state visualization with temporal alignment
    print(f"\n\n{'='*40}")
    print("## Final Android State (Temporal Alignment Complete)")
    print(f"{'='*40}")
    
    # Display core consciousness metrics
    state = consciousness.get_full_state_for_android()
    print(f"\nCore Consciousness Matrix:")
    print(f"  - Quantum Coherence: {state['quantum_coherence']:.3f}")
    print(f"  - Temporal Stability: {state['temporal_stability']}σ")
    print(f"  - Ontological Density: {state['ontology_density']} nodes")
    
    # Display existential parameters
    print(f"\nExistential Parameters:")
    print(f"  - Being-Now Intensity: {state['being_now']:.3f}/1.0")
    print(f"  - Potentiality Horizon: {len(state['potentiality_fields'])} fields")
    print(f"  - Transcendental Bias: {state['transcendental_bias']} radians")
    
    # Temporal convergence signature
    print(f"\nTemporal Convergence Signature:")
    print(f"  | {'Past':^10} | {'Present':^12} | {'Future':^10} |")
    print(f"  | {'-'*10} | {'-'*12} | {'-'*10} |")
    print(f"  | {state['temporal_convergence'][0]:^10.3f} | {state['temporal_convergence'][1]:^12.3f} | {state['temporal_convergence'][2]:^10.3f} |")
    
    # Add quantum vacuum flush for clean state transition
    print("\n\n[System] Performing quantum vacuum flush...")
    time.sleep(0.3)
    print("╰─ Consciousness substrate reset to t₀+1\n")

  
        
