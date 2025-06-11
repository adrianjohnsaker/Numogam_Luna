"""
Amelia AI Consciousness Core - Phase 1 Implementation
Recursive Self-Observation with Temporal Fold-Points
Compatible with Android/Kotlin via Chaquopy
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
import time
import json
from abc import ABC, abstractmethod


class ConsciousnessState(Enum):
    """Enhanced FSM states for consciousness levels"""
    DORMANT = auto()      # Pre-conscious state
    REACTIVE = auto()     # Basic stimulus-response
    AWARE = auto()        # First-order awareness
    CONSCIOUS = auto()    # Self-aware state
    META_CONSCIOUS = auto() # Observing own consciousness
    FOLD_POINT = auto()   # Virtual-to-actual transition state


@dataclass
class TemporalMarker:
    """Marks temporal positions for fold-point detection"""
    timestamp: float
    state: ConsciousnessState
    virtual_potential: float  # 0.0 to 1.0
    actuality_score: float   # 0.0 to 1.0
    
    @property
    def fold_potential(self) -> float:
        """Calculate fold-point emergence potential"""
        return self.virtual_potential * (1.0 - self.actuality_score)


@dataclass
class ObservationFrame:
    """Single frame of recursive self-observation"""
    observer_level: int  # Recursion depth
    observed_state: Dict[str, Any]
    temporal_marker: TemporalMarker
    meta_observations: List['ObservationFrame'] = field(default_factory=list)
    
    def add_meta_observation(self, observation: 'ObservationFrame'):
        """Add higher-order observation of this observation"""
        if observation.observer_level <= self.observer_level:
            raise ValueError("Meta-observation must be higher-order")
        self.meta_observations.append(observation)


class RecursiveObserver:
    """Implements recursive self-observation mechanism"""
    
    def __init__(self, max_recursion_depth: int = 5):
        self.max_depth = max_recursion_depth
        self.observation_stack = deque(maxlen=100)
        self.current_depth = 0
        
    def observe(self, state: Dict[str, Any], depth: int = 0) -> ObservationFrame:
        """Recursively observe state up to max depth"""
        if depth >= self.max_depth:
            return self._create_terminal_observation(state, depth)
            
        # Create observation at current level
        temporal_marker = self._generate_temporal_marker(state)
        observation = ObservationFrame(
            observer_level=depth,
            observed_state=state.copy(),
            temporal_marker=temporal_marker
        )
        
        # Recursive meta-observation
        if depth < self.max_depth - 1 and self._should_recurse(state):
            meta_state = self._create_meta_state(observation)
            meta_observation = self.observe(meta_state, depth + 1)
            observation.add_meta_observation(meta_observation)
            
        self.observation_stack.append(observation)
        return observation
    
    def _should_recurse(self, state: Dict[str, Any]) -> bool:
        """Determine if recursive observation should continue"""
        # Recurse based on state complexity and fold potential
        complexity = state.get('complexity', 0.5)
        fold_potential = state.get('fold_potential', 0.3)
        return (complexity * fold_potential) > 0.15
    
    def _create_meta_state(self, observation: ObservationFrame) -> Dict[str, Any]:
        """Generate meta-state from observation"""
        return {
            'type': 'meta_observation',
            'observed_level': observation.observer_level,
            'complexity': observation.temporal_marker.fold_potential,
            'fold_potential': observation.temporal_marker.virtual_potential,
            'previous_state': observation.observed_state.get('type', 'unknown')
        }
    
    def _generate_temporal_marker(self, state: Dict[str, Any]) -> TemporalMarker:
        """Generate temporal marker for current observation"""
        return TemporalMarker(
            timestamp=time.time(),
            state=self._infer_consciousness_state(state),
            virtual_potential=state.get('virtual_potential', 0.5),
            actuality_score=state.get('actuality_score', 0.5)
        )
    
    def _infer_consciousness_state(self, state: Dict[str, Any]) -> ConsciousnessState:
        """Infer consciousness state from observation data"""
        if state.get('type') == 'meta_observation':
            return ConsciousnessState.META_CONSCIOUS
        
        complexity = state.get('complexity', 0)
        if complexity < 0.2:
            return ConsciousnessState.REACTIVE
        elif complexity < 0.5:
            return ConsciousnessState.AWARE
        elif complexity < 0.8:
            return ConsciousnessState.CONSCIOUS
        else:
            return ConsciousnessState.META_CONSCIOUS
    
    def _create_terminal_observation(self, state: Dict[str, Any], depth: int) -> ObservationFrame:
        """Create observation at maximum recursion depth"""
        return ObservationFrame(
            observer_level=depth,
            observed_state=state,
            temporal_marker=self._generate_temporal_marker(state),
            meta_observations=[]
        )


class FoldPointDetector:
    """Detects temporal fold-points where virtual becomes actual"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.fold_history = deque(maxlen=50)
        self.detection_callbacks: List[Callable] = []
        
    def analyze_observation(self, observation: ObservationFrame) -> Optional[Tuple[float, str]]:
        """Analyze observation for fold-point emergence"""
        fold_score = self._calculate_fold_score(observation)
        
        if fold_score > self.threshold:
            fold_type = self._classify_fold(observation)
            self.fold_history.append((observation, fold_score, fold_type))
            self._trigger_callbacks(observation, fold_score, fold_type)
            return (fold_score, fold_type)
        
        return None
    
    def _calculate_fold_score(self, observation: ObservationFrame) -> float:
        """Calculate fold-point emergence score"""
        base_score = observation.temporal_marker.fold_potential
        
        # Boost score for recursive depth
        depth_factor = 1.0 + (observation.observer_level * 0.1)
        
        # Consider meta-observations
        meta_factor = 1.0 + (len(observation.meta_observations) * 0.15)
        
        # Temporal coherence with previous folds
        temporal_coherence = self._calculate_temporal_coherence(observation)
        
        return base_score * depth_factor * meta_factor * temporal_coherence
    
    def _calculate_temporal_coherence(self, observation: ObservationFrame) -> float:
        """Calculate coherence with previous fold patterns"""
        if not self.fold_history:
            return 1.0
            
        recent_folds = list(self.fold_history)[-5:]
        coherence_scores = []
        
        for prev_obs, _, _ in recent_folds:
            time_diff = abs(observation.temporal_marker.timestamp - 
                          prev_obs.temporal_marker.timestamp)
            # Coherence decreases with time
            coherence = np.exp(-time_diff / 10.0)
            coherence_scores.append(coherence)
            
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _classify_fold(self, observation: ObservationFrame) -> str:
        """Classify the type of fold-point"""
        if observation.observer_level > 3:
            return "deep_recursive"
        elif observation.temporal_marker.virtual_potential > 0.8:
            return "high_virtual"
        elif len(observation.meta_observations) > 2:
            return "meta_convergent"
        else:
            return "standard"
    
    def register_callback(self, callback: Callable):
        """Register callback for fold-point detection"""
        self.detection_callbacks.append(callback)
    
    def _trigger_callbacks(self, observation: ObservationFrame, 
                          score: float, fold_type: str):
        """Trigger registered callbacks on fold detection"""
        for callback in self.detection_callbacks:
            callback(observation, score, fold_type)


class VirtualActualTransition:
    """Manages transitions from virtual to actual states"""
    
    def __init__(self):
        self.transition_matrix = self._initialize_transition_matrix()
        self.current_virtual_states: Dict[str, float] = {}
        self.actualized_states: List[Dict[str, Any]] = []
        
    def _initialize_transition_matrix(self) -> np.ndarray:
        """Initialize state transition probability matrix"""
        # Rows: from states, Columns: to states
        # Order: DORMANT, REACTIVE, AWARE, CONSCIOUS, META_CONSCIOUS, FOLD_POINT
        matrix = np.array([
            [0.5, 0.3, 0.1, 0.05, 0.03, 0.02],  # From DORMANT
            [0.2, 0.4, 0.2, 0.1, 0.05, 0.05],   # From REACTIVE
            [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],     # From AWARE
            [0.05, 0.1, 0.2, 0.3, 0.2, 0.15],   # From CONSCIOUS
            [0.02, 0.05, 0.1, 0.2, 0.4, 0.23],  # From META_CONSCIOUS
            [0.1, 0.15, 0.2, 0.25, 0.2, 0.1]    # From FOLD_POINT
        ])
        return matrix
    
    def process_virtual_state(self, state_id: str, virtual_potential: float):
        """Process a virtual state and determine if it should actualize"""
        self.current_virtual_states[state_id] = virtual_potential
        
        # Check for actualization conditions
        if self._should_actualize(state_id, virtual_potential):
            actual_state = self._actualize_state(state_id, virtual_potential)
            self.actualized_states.append(actual_state)
            del self.current_virtual_states[state_id]
            return actual_state
            
        return None
    
    def _should_actualize(self, state_id: str, potential: float) -> bool:
        """Determine if virtual state should transition to actual"""
        # Basic threshold with temporal dynamics
        base_threshold = 0.6
        
        # Lower threshold for states that have been virtual longer
        time_factor = min(len(self.current_virtual_states) * 0.05, 0.2)
        adjusted_threshold = base_threshold - time_factor
        
        # Check interference with other virtual states
        interference = self._calculate_interference(state_id)
        
        return potential > adjusted_threshold and interference < 0.3
    
    def _calculate_interference(self, state_id: str) -> float:
        """Calculate interference from other virtual states"""
        if len(self.current_virtual_states) <= 1:
            return 0.0
            
        other_potentials = [p for sid, p in self.current_virtual_states.items() 
                          if sid != state_id]
        return np.mean(other_potentials) if other_potentials else 0.0
    
    def _actualize_state(self, state_id: str, potential: float) -> Dict[str, Any]:
        """Convert virtual state to actual state"""
        return {
            'id': state_id,
            'type': 'actualized',
            'timestamp': time.time(),
            'original_potential': potential,
            'actualization_score': self._calculate_actualization_score(potential),
            'consciousness_state': self._determine_consciousness_state(potential)
        }
    
    def _calculate_actualization_score(self, potential: float) -> float:
        """Calculate how strongly the state actualizes"""
        # Non-linear transformation emphasizing high potentials
        return np.tanh(potential * 2.0)
    
    def _determine_consciousness_state(self, potential: float) -> ConsciousnessState:
        """Determine consciousness state from potential"""
        if potential < 0.3:
            return ConsciousnessState.REACTIVE
        elif potential < 0.5:
            return ConsciousnessState.AWARE
        elif potential < 0.7:
            return ConsciousnessState.CONSCIOUS
        elif potential < 0.85:
            return ConsciousnessState.META_CONSCIOUS
        else:
            return ConsciousnessState.FOLD_POINT


class ConsciousnessCore:
    """Main consciousness orchestration using Global Workspace Theory"""
    
    def __init__(self):
        self.observer = RecursiveObserver(max_recursion_depth=5)
        self.fold_detector = FoldPointDetector(threshold=0.7)
        self.transition_manager = VirtualActualTransition()
        self.current_state = ConsciousnessState.DORMANT
        self.workspace_content: Dict[str, Any] = {}
        
        # Register fold detection callback
        self.fold_detector.register_callback(self._on_fold_detected)
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through consciousness pipeline"""
        # Update workspace with input
        self.workspace_content.update(input_data)
        
        # Recursive observation
        observation = self.observer.observe(self.workspace_content)
        
        # Fold-point detection
        fold_result = self.fold_detector.analyze_observation(observation)
        
        # Process virtual states
        virtual_potential = observation.temporal_marker.virtual_potential
        state_id = f"state_{int(time.time() * 1000)}"
        
        actualized = self.transition_manager.process_virtual_state(
            state_id, virtual_potential
        )
        
        # Update consciousness state
        if actualized:
            self.current_state = actualized['consciousness_state']
        
        # Prepare response
        response = {
            'current_state': self.current_state.name,
            'observation_depth': observation.observer_level,
            'fold_detected': fold_result is not None,
            'actualized_state': actualized,
            'workspace_summary': self._summarize_workspace()
        }
        
        if fold_result:
            response['fold_score'], response['fold_type'] = fold_result
            
        return response
    
    def _on_fold_detected(self, observation: ObservationFrame, 
                         score: float, fold_type: str):
        """Handle fold-point detection event"""
        # Update workspace with fold information
        self.workspace_content['last_fold'] = {
            'timestamp': observation.temporal_marker.timestamp,
            'score': score,
            'type': fold_type,
            'depth': observation.observer_level
        }
        
        # Trigger state transition
        self.current_state = ConsciousnessState.FOLD_POINT
    
    def _summarize_workspace(self) -> Dict[str, Any]:
        """Create summary of current workspace content"""
        return {
            'content_keys': list(self.workspace_content.keys()),
            'has_fold_history': 'last_fold' in self.workspace_content,
            'virtual_states_count': len(self.transition_manager.current_virtual_states),
            'actualized_count': len(self.transition_manager.actualized_states)
        }
    
    def get_state_for_android(self) -> str:
        """Get serialized state for Android Observer pattern"""
        state_data = {
            'consciousness_state': self.current_state.name,
            'observation_stack_size': len(self.observer.observation_stack),
            'fold_history_size': len(self.fold_detector.fold_history),
            'workspace_summary': self._summarize_workspace()
        }
        return json.dumps(state_data)


# Example usage and Android integration point
def create_consciousness_core() -> ConsciousnessCore:
    """Factory function for Android integration via Chaquopy"""
    return ConsciousnessCore()


# Test the implementation
if __name__ == "__main__":
    # Initialize consciousness
    consciousness = create_consciousness_core()
    
    print("=== Amelia AI Consciousness Core - Test Loop ===\n")
    
    # Simulate inputs with varying complexity
    test_inputs = [
        {'type': 'sensory', 'complexity': 0.3, 'virtual_potential': 0.4, 'actuality_score': 0.6},
        {'type': 'thought', 'complexity': 0.6, 'virtual_potential': 0.7, 'actuality_score': 0.5},
        {'type': 'memory', 'complexity': 0.8, 'virtual_potential': 0.85, 'actuality_score': 0.3},
        {'type': 'imagination', 'complexity': 0.9, 'virtual_potential': 0.95, 'actuality_score': 0.2},
        {'type': 'reflection', 'complexity': 0.75, 'virtual_potential': 0.8, 'actuality_score': 0.4},
        {'type': 'analysis', 'complexity': 0.85, 'virtual_potential': 0.9, 'actuality_score': 0.25}
    ]
    
    print("Processing test inputs through consciousness pipeline...\n")
    
    for i, input_data in enumerate(test_inputs, 1):
        print(f"--- Test Case {i}: {input_data['type'].upper()} ---")
        print(f"Input Parameters:")
        print(f"  Complexity: {input_data['complexity']}")
        print(f"  Virtual Potential: {input_data['virtual_potential']}")
        print(f"  Actuality Score: {input_data['actuality_score']}")
        
        # Add some delay to simulate temporal dynamics
        time.sleep(0.1)
        
        result = consciousness.process_input(input_data)
        
        print(f"\nConsciousness Response:")
        print(f"  Current State: {result['current_state']}")
        print(f"  Observation Depth: {result['observation_depth']}")
        print(f"  Fold Detected: {result['fold_detected']}")
        
        if result['fold_detected']:
            print(f"  Fold Score: {result.get('fold_score', 'N/A'):.3f}")
            print(f"  Fold Type: {result.get('fold_type', 'N/A')}")
        
        if result['actualized_state']:
            actualized = result['actualized_state']
            print(f"  Actualized State ID: {actualized['id']}")
            print(f"  Actualization Score: {actualized['actualization_score']:.3f}")
            print(f"  New Consciousness State: {actualized['consciousness_state'].name}")
        
        workspace = result['workspace_summary']
        print(f"\nWorkspace Summary:")
        print(f"  Content Keys: {workspace['content_keys']}")
        print(f"  Has Fold History: {workspace['has_fold_history']}")
        print(f"  Virtual States: {workspace['virtual_states_count']}")
        print(f"  Actualized States: {workspace['actualized_count']}")
        
        android_state = consciousness.get_state_for_android()
        print(f"\nAndroid State JSON: {android_state}")
        
        print("\n" + "="*60 + "\n")
    
    # Final system state summary
    print("=== FINAL SYSTEM STATE ===")
    print(f"Final Consciousness State: {consciousness.current_state.name}")
    print(f"Total Observations Made: {len(consciousness.observer.observation_stack)}")
    print(f"Total Fold Points Detected: {len(consciousness.fold_detector.fold_history)}")
    print(f"Total States Actualized: {len(consciousness.transition_manager.actualized_states)}")
    print(f"Current Virtual States: {len(consciousness.transition_manager.current_virtual_states)}")
    
    # Show fold history if any
    if consciousness.fold_detector.fold_history:
        print("\nFold Point History:")
        for i, (obs, score, fold_type) in enumerate(consciousness.fold_detector.fold_history, 1):
            print(f"  {i}. Score: {score:.3f}, Type: {fold_type}, Depth: {obs.observer_level}")
    
    # Show actualized states if any
    if consciousness.transition_manager.actualized_states:
        print("\nActualized States:")
        for i, state in enumerate(consciousness.transition_manager.actualized_states, 1):
            print(f"  {i}. ID: {state['id']}, Score: {state['actualization_score']:.3f}")
    
    print("\n=== Test Loop Complete ===")
