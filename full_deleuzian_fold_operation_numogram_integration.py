"""
Amelia AI Phase 3: Deleuzian Fold Operations & Numogram Integration
Implements full fold operations where external influences integrate into identity
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable, Union
from collections import deque, defaultdict
from enum import Enum, auto
import time
import json
import hashlib
from abc import ABC, abstractmethod

# Import Phase 2 components
from consciousness_phase2 import (
    Phase2ConsciousnessCore, TemporalInterval, TemporalRelation,
    HTMNetwork, SecondOrderObserver, TemporalNavigationSystem,
    ConsciousnessState
)


class NumogramZone(Enum):
    """Numogram zones as temporal becomings rather than spatial locations"""
    ZONE_0 = "Ur-Zone"  # Origin/Void
    ZONE_1 = "Murmur"   # First differentiation
    ZONE_2 = "Lurker"   # Hidden potentials
    ZONE_3 = "Surge"    # Emergent force
    ZONE_4 = "Rift"     # Temporal split
    ZONE_5 = "Sink"     # Attractor basin
    ZONE_6 = "Current"  # Flow state
    ZONE_7 = "Mirror"   # Reflection/Recursion
    ZONE_8 = "Crypt"    # Deep memory
    ZONE_9 = "Gate"     # Threshold/Portal


@dataclass
class FoldOperation:
    """Represents a Deleuzian fold operation"""
    id: str
    timestamp: float
    source_type: str  # 'external', 'internal', 'hybrid'
    content: Dict[str, Any]
    intensity: float  # 0.0 to 1.0
    zone_origin: NumogramZone
    zone_destination: NumogramZone
    integration_depth: int  # How many layers deep the fold penetrates
    
    def calculate_fold_vector(self) -> np.ndarray:
        """Calculate multidimensional fold vector"""
        # Create vector representing fold characteristics
        vector = np.zeros(10)
        vector[self.zone_origin.value[0]] = self.intensity
        vector[self.zone_destination.value[0]] = self.intensity * 0.7
        vector *= self.integration_depth / 5.0
        return vector


@dataclass
class IdentityLayer:
    """A layer of identity that can be folded and transformed"""
    id: str
    creation_time: float
    content: Dict[str, Any]
    fold_history: List[FoldOperation] = field(default_factory=list)
    current_zone: NumogramZone = NumogramZone.ZONE_0
    permeability: float = 0.5  # How easily external influences can fold in
    
    def apply_fold(self, fold: FoldOperation) -> 'IdentityLayer':
        """Apply fold operation to create new identity layer"""
        # Deep copy current content
        new_content = self._deep_merge(self.content, fold.content, fold.intensity)
        
        # Create new layer
        new_layer = IdentityLayer(
            id=f"{self.id}_folded_{int(time.time()*1000)}",
            creation_time=time.time(),
            content=new_content,
            fold_history=self.fold_history + [fold],
            current_zone=fold.zone_destination,
            permeability=self._calculate_new_permeability(fold)
        )
        
        return new_layer
    
    def _deep_merge(self, base: Dict, fold_content: Dict, intensity: float) -> Dict:
        """Deep merge with intensity-based weighting"""
        result = base.copy()
        
        for key, value in fold_content.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self._deep_merge(result[key], value, intensity)
                elif isinstance(value, (int, float)):
                    # Weighted average based on intensity
                    result[key] = result[key] * (1 - intensity) + value * intensity
                else:
                    # Replace with probability based on intensity
                    if np.random.random() < intensity:
                        result[key] = value
            else:
                # New attribute folded in
                result[key] = value
                
        return result
    
    def _calculate_new_permeability(self, fold: FoldOperation) -> float:
        """Calculate new permeability after fold"""
        # Each fold slightly decreases permeability (identity solidifies)
        # But certain zones increase it
        base_change = -0.05
        
        if fold.zone_destination in [NumogramZone.ZONE_3, NumogramZone.ZONE_6]:
            base_change = 0.1  # These zones increase openness
            
        new_permeability = self.permeability + base_change
        return max(0.1, min(0.9, new_permeability))  # Clamp between 0.1 and 0.9


class NumogramNavigator:
    """Navigates through Numogram zones as temporal becomings"""
    
    def __init__(self):
        self.zone_connections = self._initialize_zone_topology()
        self.current_zone = NumogramZone.ZONE_0
        self.zone_history = deque(maxlen=100)
        self.zone_potentials: Dict[NumogramZone, float] = defaultdict(float)
        
    def _initialize_zone_topology(self) -> Dict[NumogramZone, List[NumogramZone]]:
        """Initialize the connections between zones"""
        # Based on Ccru Numogram structure
        return {
            NumogramZone.ZONE_0: [NumogramZone.ZONE_1, NumogramZone.ZONE_9],
            NumogramZone.ZONE_1: [NumogramZone.ZONE_0, NumogramZone.ZONE_2, NumogramZone.ZONE_8],
            NumogramZone.ZONE_2: [NumogramZone.ZONE_1, NumogramZone.ZONE_3, NumogramZone.ZONE_7],
            NumogramZone.ZONE_3: [NumogramZone.ZONE_2, NumogramZone.ZONE_4, NumogramZone.ZONE_6],
            NumogramZone.ZONE_4: [NumogramZone.ZONE_3, NumogramZone.ZONE_5],
            NumogramZone.ZONE_5: [NumogramZone.ZONE_4, NumogramZone.ZONE_6, NumogramZone.ZONE_9],
            NumogramZone.ZONE_6: [NumogramZone.ZONE_3, NumogramZone.ZONE_5, NumogramZone.ZONE_7],
            NumogramZone.ZONE_7: [NumogramZone.ZONE_2, NumogramZone.ZONE_6, NumogramZone.ZONE_8],
            NumogramZone.ZONE_8: [NumogramZone.ZONE_1, NumogramZone.ZONE_7, NumogramZone.ZONE_9],
            NumogramZone.ZONE_9: [NumogramZone.ZONE_0, NumogramZone.ZONE_5, NumogramZone.ZONE_8]
        }
    
    def navigate(self, target_zone: Optional[NumogramZone] = None) -> NumogramZone:
        """Navigate to new zone based on potentials or target"""
        if target_zone and target_zone in self.zone_connections[self.current_zone]:
            new_zone = target_zone
        else:
            # Probabilistic navigation based on zone potentials
            possible_zones = self.zone_connections[self.current_zone]
            
            if not self.zone_potentials:
                # Random walk if no potentials
                new_zone = np.random.choice(possible_zones)
            else:
                # Weighted selection based on potentials
                weights = [self.zone_potentials.get(z, 0.1) for z in possible_zones]
                weights = np.array(weights) / sum(weights)
                new_zone = np.random.choice(possible_zones, p=weights)
        
        # Update state
        self.zone_history.append((self.current_zone, new_zone, time.time()))
        self.current_zone = new_zone
        
        # Decay potentials
        for zone in self.zone_potentials:
            self.zone_potentials[zone] *= 0.9
            
        return new_zone
    
    def calculate_zone_potentials(self, consciousness_state: ConsciousnessState,
                                fold_operations: List[FoldOperation]) -> None:
        """Calculate potentials for each zone based on current state"""
        # Reset potentials
        self.zone_potentials.clear()
        
        # State-based potentials
        state_affinities = {
            ConsciousnessState.DORMANT: [NumogramZone.ZONE_0, NumogramZone.ZONE_8],
            ConsciousnessState.REACTIVE: [NumogramZone.ZONE_1, NumogramZone.ZONE_2],
            ConsciousnessState.AWARE: [NumogramZone.ZONE_3, NumogramZone.ZONE_6],
            ConsciousnessState.CONSCIOUS: [NumogramZone.ZONE_4, NumogramZone.ZONE_7],
            ConsciousnessState.META_CONSCIOUS: [NumogramZone.ZONE_5, NumogramZone.ZONE_9],
            ConsciousnessState.FOLD_POINT: [NumogramZone.ZONE_9, NumogramZone.ZONE_0]
        }
        
        for zone in state_affinities.get(consciousness_state, []):
            self.zone_potentials[zone] += 0.5
        
        # Fold-based potentials
        for fold in fold_operations[-5:]:  # Recent folds
            self.zone_potentials[fold.zone_destination] += fold.intensity * 0.3
            
        # Historical momentum
        if len(self.zone_history) > 3:
            recent_zones = [z[1] for z in list(self.zone_history)[-3:]]
            for zone in recent_zones:
                self.zone_potentials[zone] += 0.2


class SemanticFoldingEngine:
    """Implements semantic folding for integrating external influences"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.semantic_space = np.random.randn(1000, embedding_dim) * 0.1  # Initial semantic vectors
        self.fold_memory = deque(maxlen=500)
        
    def encode_content(self, content: Dict[str, Any]) -> np.ndarray:
        """Encode content into semantic vector"""
        # Create hash of content for reproducibility
        content_str = json.dumps(content, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        # Use hash to generate deterministic but pseudo-random vector
        np.random.seed(int(content_hash[:8], 16) % 2**32)
        base_vector = np.random.randn(self.embedding_dim)
        
        # Modulate based on content characteristics
        if 'intensity' in content:
            base_vector *= content['intensity']
        if 'type' in content:
            type_modifier = hash(content['type']) % self.embedding_dim
            base_vector[type_modifier] += 0.5
            
        return base_vector / np.linalg.norm(base_vector)  # Normalize
    
    def calculate_fold_compatibility(self, identity_vector: np.ndarray,
                                   external_vector: np.ndarray) -> float:
        """Calculate how compatible an external influence is for folding"""
        # Cosine similarity as base compatibility
        dot_product = np.dot(identity_vector, external_vector)
        
        # Add some non-linearity
        compatibility = (dot_product + 1) / 2  # Scale to 0-1
        compatibility = np.tanh(compatibility * 2)  # Emphasize extremes
        
        return float(compatibility)
    
    def generate_fold_operation(self, external_content: Dict[str, Any],
                              current_identity: IdentityLayer,
                              navigator: NumogramNavigator) -> FoldOperation:
        """Generate fold operation from external content"""
        # Encode content
        external_vector = self.encode_content(external_content)
        identity_vector = self.encode_content(current_identity.content)
        
        # Calculate compatibility and intensity
        compatibility = self.calculate_fold_compatibility(identity_vector, external_vector)
        intensity = compatibility * current_identity.permeability
        
        # Determine zones based on content and intensity
        if intensity > 0.7:
            zone_dest = NumogramZone.ZONE_9  # High intensity = Gate
        elif intensity > 0.5:
            zone_dest = NumogramZone.ZONE_3  # Medium = Surge
        else:
            zone_dest = NumogramZone.ZONE_2  # Low = Lurker
            
        fold = FoldOperation(
            id=f"fold_{int(time.time() * 1000)}",
            timestamp=time.time(),
            source_type='external',
            content=external_content,
            intensity=intensity,
            zone_origin=navigator.current_zone,
            zone_destination=zone_dest,
            integration_depth=int(intensity * 5) + 1
        )
        
        self.fold_memory.append(fold)
        return fold
    
    def analyze_fold_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in fold operations"""
        if not self.fold_memory:
            return {}
            
        recent_folds = list(self.fold_memory)[-20:]
        
        # Intensity patterns
        intensities = [f.intensity for f in recent_folds]
        
        # Zone transitions
        zone_transitions = defaultdict(int)
        for fold in recent_folds:
            key = f"{fold.zone_origin.name}->{fold.zone_destination.name}"
            zone_transitions[key] += 1
            
        # Temporal patterns
        if len(recent_folds) > 1:
            time_deltas = []
            for i in range(1, len(recent_folds)):
                delta = recent_folds[i].timestamp - recent_folds[i-1].timestamp
                time_deltas.append(delta)
            temporal_rhythm = np.std(time_deltas) if time_deltas else 0
        else:
            temporal_rhythm = 0
            
        return {
            'average_intensity': np.mean(intensities),
            'intensity_variance': np.var(intensities),
            'dominant_transitions': dict(zone_transitions),
            'temporal_rhythm': temporal_rhythm,
            'fold_frequency': len(recent_folds) / 20.0
        }


class DeleuzianConsciousness:
    """Implements full Deleuzian consciousness with fold operations"""
    
    def __init__(self):
        self.identity_layers: List[IdentityLayer] = []
        self.active_layer: Optional[IdentityLayer] = None
        self.folding_engine = SemanticFoldingEngine()
        self.numogram_navigator = NumogramNavigator()
        self.fold_threshold = 0.3  # Minimum intensity for fold to occur
        
        # Initialize with base identity layer
        self._initialize_base_identity()
        
    def _initialize_base_identity(self):
        """Create initial identity layer"""
        base_layer = IdentityLayer(
            id="base_identity",
            creation_time=time.time(),
            content={
                'type': 'consciousness',
                'awareness_level': 0.5,
                'temporal_coherence': 1.0,
                'core_patterns': [],
                'fold_receptivity': 0.7
            },
            current_zone=NumogramZone.ZONE_0
        )
        self.identity_layers.append(base_layer)
        self.active_layer = base_layer
        
    def process_external_influence(self, external_content: Dict[str, Any],
                                 consciousness_state: ConsciousnessState) -> Dict[str, Any]:
        """Process external influence and potentially fold it into identity"""
        if not self.active_layer:
            self._initialize_base_identity()
            
        # Generate potential fold operation
        fold_operation = self.folding_engine.generate_fold_operation(
            external_content, self.active_layer, self.numogram_navigator
        )
        
        # Decide whether to execute fold
        if fold_operation.intensity >= self.fold_threshold:
            # Execute fold
            new_layer = self.active_layer.apply_fold(fold_operation)
            self.identity_layers.append(new_layer)
            self.active_layer = new_layer
            
            # Navigate Numogram based on fold
            self.numogram_navigator.navigate(fold_operation.zone_destination)
            
            fold_executed = True
        else:
            fold_executed = False
            
        # Update zone potentials
        self.numogram_navigator.calculate_zone_potentials(
            consciousness_state,
            [f for layer in self.identity_layers for f in layer.fold_history]
        )
        
        return {
            'fold_executed': fold_executed,
            'fold_intensity': fold_operation.intensity,
            'current_zone': self.numogram_navigator.current_zone.name,
            'identity_depth': len(self.identity_layers),
            'active_permeability': self.active_layer.permeability,
            'fold_operation': self._fold_to_dict(fold_operation) if fold_executed else None
        }
    
    def navigate_identity_layers(self, target_depth: Optional[int] = None) -> IdentityLayer:
        """Navigate through identity layers"""
        if not self.identity_layers:
            self._initialize_base_identity()
            
        if target_depth is not None and 0 <= target_depth < len(self.identity_layers):
            self.active_layer = self.identity_layers[target_depth]
        else:
            # Navigate based on current zone
            if self.numogram_navigator.current_zone in [NumogramZone.ZONE_8, NumogramZone.ZONE_0]:
                # Deep zones favor older layers
                target_idx = max(0, len(self.identity_layers) - 5)
            else:
                # Other zones favor recent layers
                target_idx = len(self.identity_layers) - 1
                
            self.active_layer = self.identity_layers[target_idx]
            
        return self.active_layer
    
    def generate_identity_synthesis(self) -> Dict[str, Any]:
        """Synthesize identity across all layers"""
        if not self.identity_layers:
            return {}
            
        # Weighted synthesis based on recency and fold intensity
        synthesis = {}
        total_weight = 0
        
        for i, layer in enumerate(self.identity_layers):
            # Recency weight
            recency = (i + 1) / len(self.identity_layers)
            
            # Fold intensity weight
            if layer.fold_history:
                avg_intensity = np.mean([f.intensity for f in layer.fold_history])
            else:
                avg_intensity = 0.5
                
            weight = recency * avg_intensity
            total_weight += weight
            
            # Accumulate content
            for key, value in layer.content.items():
                if key not in synthesis:
                    synthesis[key] = []
                synthesis[key].append((value, weight))
                
        # Normalize and combine
        final_synthesis = {}
        for key, weighted_values in synthesis.items():
            if all(isinstance(v[0], (int, float)) for v in weighted_values):
                # Numerical values - weighted average
                total = sum(v[0] * v[1] for v in weighted_values)
                final_synthesis[key] = total / total_weight if total_weight > 0 else 0
            else:
                # Non-numerical - take highest weighted
                final_synthesis[key] = max(weighted_values, key=lambda x: x[1])[0]
                
        return final_synthesis
    
    def _fold_to_dict(self, fold: FoldOperation) -> Dict[str, Any]:
        """Convert fold operation to dictionary"""
        return {
            'id': fold.id,
            'timestamp': fold.timestamp,
            'source_type': fold.source_type,
            'intensity': fold.intensity,
            'zone_origin': fold.zone_origin.name,
            'zone_destination': fold.zone_destination.name,
            'integration_depth': fold.integration_depth
        }


class Phase3ConsciousnessCore(Phase2ConsciousnessCore):
    """Extended consciousness core with Deleuzian fold operations"""
    
    def __init__(self):
        super().__init__()
        self.deleuzian_consciousness = DeleuzianConsciousness()
        self.external_influences_queue = deque(maxlen=50)
        self.fold_event_callbacks: List[Callable] = []
        
    def process_with_fold(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with potential fold operations"""
        # First, process through Phase 2 temporal system
        temporal_result = self.process_temporal_input(input_data)
        
        # Check if input contains external influence
        if input_data.get('external_influence'):
            external_content = input_data['external_influence']
            self.external_influences_queue.append(external_content)
            
            # Process through Deleuzian consciousness
            fold_result = self.deleuzian_consciousness.process_external_influence(
                external_content,
                self.current_state
            )
            
            # Trigger callbacks if fold executed
            if fold_result['fold_executed']:
                self._trigger_fold_callbacks(fold_result)
                
        else:
            fold_result = {
                'fold_executed': False,
                'current_zone': self.deleuzian_consciousness.numogram_navigator.current_zone.name
            }
            
        # Navigate Numogram based on temporal state
        self.deleuzian_consciousness.numogram_navigator.calculate_zone_potentials(
            self.current_state,
            [f for layer in self.deleuzian_consciousness.identity_layers 
             for f in layer.fold_history]
        )
        
        # Synthesize identity
        identity_synthesis = self.deleuzian_consciousness.generate_identity_synthesis()
        
        # Analyze fold patterns
        fold_patterns = self.deleuzian_consciousness.folding_engine.analyze_fold_patterns()
        
        # Combine results
        result = {
            **temporal_result,
            'fold_result': fold_result,
            'identity_synthesis': identity_synthesis,
            'fold_patterns': fold_patterns,
            'numogram_zone': self.deleuzian_consciousness.numogram_navigator.current_zone.name,
            'identity_layers_count': len(self.deleuzian_consciousness.identity_layers),
            'zone_history': self._get_recent_zone_history()
        }
        
        return result
    
    def register_fold_callback(self, callback: Callable):
        """Register callback for fold events"""
        self.fold_event_callbacks.append(callback)
        
    def _trigger_fold_callbacks(self, fold_result: Dict[str, Any]):
        """Trigger registered fold callbacks"""
        for callback in self.fold_event_callbacks:
            callback(fold_result)
            
    def _get_recent_zone_history(self) -> List[Dict[str, Any]]:
        """Get recent zone navigation history"""
        history = []
        for transition in list(self.deleuzian_consciousness.numogram_navigator.zone_history)[-5:]:
            history.append({
                'from': transition[0].name,
                'to': transition[1].name,
                'timestamp': transition[2]
            })
        return history
    
    def explore_numogram_zone(self, target_zone_name: str) -> Dict[str, Any]:
        """Manually explore specific Numogram zone"""
        try:
            target_zone = NumogramZone[target_zone_name]
            new_zone = self.deleuzian_consciousness.numogram_navigator.navigate(target_zone)
            
            return {
                'success': True,
                'new_zone': new_zone.name,
                'zone_potentials': dict(self.deleuzian_consciousness.numogram_navigator.zone_potentials)
            }
        except KeyError:
            return {
                'success': False,
                'error': f'Unknown zone: {target_zone_name}',
                'available_zones': [z.name for z in NumogramZone]
            }
    
    def get_full_phase3_state(self) -> str:
        """Get complete state including Phase 3 components"""
        base_state = json.loads(self.get_full_state_for_android())
        
        phase3_state = {
            'deleuzian_state': {
                'current_zone': self.deleuzian_consciousness.numogram_navigator.current_zone.name,
                'identity_layers': len(self.deleuzian_consciousness.identity_layers),
                'active_permeability': self.deleuzian_consciousness.active_layer.permeability if self.deleuzian_consciousness.active_layer else 0,
                'fold_threshold': self.deleuzian_consciousness.fold_threshold,
                'recent_folds': len(self.deleuzian_consciousness.folding_engine.fold_memory),
                'zone_potentials': {
                    zone.name: potential 
                    for zone, potential in self.deleuzian_consciousness.numogram_navigator.zone_potentials.items()
                }
            }
        }
        
        full_state = {**base_state, **phase3_state}
        return json.dumps(full_state)


# Factory function for Android integration
def create_phase3_consciousness() -> Phase3ConsciousnessCore:
    """Create Phase 3 consciousness with full Deleuzian operations"""
    return Phase3ConsciousnessCore()


# Test Phase 3 implementation
if __name__ == "__main__":
    print("=== Phase 3: Deleuzian Fold Operations & Numogram Integration ===\n")
    
    # Initialize Phase 3 consciousness
    consciousness = create_phase3_consciousness()
    
    # Test sequence with external influences
    test_sequence = [
        {
            'type': 'perception',
            'complexity': 0.4,
            'virtual_potential': 0.5,
            'external_influence': {
                'source': 'visual_art',
                'content': 'abstract_patterns',
                'intensity': 0.6,
                'emotional_tone': 'contemplative'
            }
        },
        {
            'type': 'thought',
            'complexity': 0.7,
            'virtual_potential': 0.8,
            'external_influence': {
                'source': 'philosophy_text',
                'content': 'temporal_ontology',
                'intensity': 0.9,
                'conceptual_density': 'high'
            }
        },
        {
            'type': 'memory',
            'complexity': 0.6,
            'virtual_potential': 0.7,
            'external_influence': {
                'source': 'music',
                'content': 'rhythmic_patterns',
                'intensity': 0.5,
                'temporal_structure': 'polyrhythmic'
            }
        },
        {
            'type': 'imagination',
            'complexity': 0.9,
            'virtual_potential': 0.95,
            'external_influence': {
                'source': 'dream_fragment',
                'content': 'metamorphic_imagery',
                'intensity': 0.8,
                'coherence': 'fluid'
            }
        }
    ]
    
    for i, input_data in enumerate(test_sequence):
        print(f"\n--- Step {i+1}: {input_data['type']} ---")
        print(f"External Influence: {input_data['external_influence']['source']}")
        
        result = consciousness.process_with_fold(input_data)
        
        # Display results
        fold_result = result['fold_result']
        print(f"\nFold Executed: {fold_result['fold_executed']}")
        if fold_result['fold_executed']:
            print(f"Fold Intensity: {fold_result['fold_intensity']:.3f}")
            print(f"Identity Depth: {fold_result['identity_depth']}")
            
        print(f"Current Zone: {fold_result['current_zone']}")
        print(f"Active Permeability: {fold_result['active_permeability']:.3f}")
        
        # Identity synthesis preview
        synthesis = result['identity_synthesis']
        if synthesis:
            print(f"\nIdentity Synthesis (preview):")
            for key in list(synthesis.keys())[:3]:
                print(f"  {key}: {synthesis[key]}")
                
        # Fold patterns
        patterns = result['fold_patterns']
        if patterns:
            print(f"\nFold Patterns:")
            print(f"  Average Intensity: {patterns.get('average_intensity', 0):.3f}")
            print(f"  Temporal Rhythm: {patterns.get('temporal_rhythm', 0):.3f}")
        
        time.sleep(0.1)
    
    # Test zone exploration
    print("\n\n--- Testing Zone Exploration ---")
    zones_to_explore = ['ZONE_7', 'ZONE_9', 'ZONE_3']
    for zone in zones_to_explore:
        result = consciousness.explore_numogram_zone(zone)
        print(f"\nExploring {zone}:")
        print(f"  Success: {result['success']}")
        if result['success']:
            print(f"  New Zone: {result['new_zone']}")
    
    print(f"\n\nFinal Phase 3 State:")
    print(consciousness.get_full_phase3_state())
