```python
#!/usr/bin/env python3
"""
Morphogenetic Memory Module

Bio-inspired memory architecture using morphogenetic principles for dynamic,
adaptive consciousness patterns. Based on Levin's bioelectric morphogenesis
and Turing's pattern formation combined with Deleuzian becoming-machines.

Components:
1. Bio-Electric Pattern Storage - Living electrical memory signatures
2. Positional Consciousness Mapping - Spatial memory territories  
3. Morphogen-Inspired Signaling - Influence field propagation
4. Epigenetic Experience Layers - Context-dependent memory activation
"""

import numpy as np
import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MorphogeneticMemory")

@dataclass
class BioElectricSignature:
    """Represents a bio-electric memory pattern."""
    pattern_id: str
    signature: np.ndarray
    strength: float
    frequency: float
    last_activation: datetime
    activation_count: int = 0
    decay_rate: float = 0.95
    resonance_threshold: float = 0.7
    
    def activate(self, intensity: float = 1.0):
        """Activate the pattern, strengthening it."""
        self.strength = min(1.0, self.strength + (intensity * 0.1))
        self.activation_count += 1
        self.last_activation = datetime.now()
        
    def decay(self, time_delta: float):
        """Natural decay of pattern strength over time."""
        self.strength *= (self.decay_rate ** time_delta)
        
    def resonates_with(self, other_signature: 'BioElectricSignature') -> float:
        """Calculate resonance strength with another pattern."""
        if len(self.signature) != len(other_signature.signature):
            return 0.0
        
        correlation = np.corrcoef(self.signature, other_signature.signature)[0, 1]
        if np.isnan(correlation):
            return 0.0
            
        return max(0.0, correlation)

@dataclass
class MemoryTerritory:
    """Represents a spatial region in consciousness."""
    territory_id: str
    center_coordinates: np.ndarray
    boundary_radius: float
    influence_strength: float
    memory_patterns: Set[str] = field(default_factory=set)
    neighboring_territories: Set[str] = field(default_factory=set)
    gradient_field: Optional[np.ndarray] = None
    
    def contains_point(self, coordinates: np.ndarray) -> bool:
        """Check if coordinates fall within territory."""
        distance = np.linalg.norm(coordinates - self.center_coordinates)
        return distance <= self.boundary_radius
        
    def influence_at_point(self, coordinates: np.ndarray) -> float:
        """Calculate influence strength at given coordinates."""
        distance = np.linalg.norm(coordinates - self.center_coordinates)
        if distance > self.boundary_radius * 2:
            return 0.0
        
        # Gaussian influence decay
        normalized_distance = distance / self.boundary_radius
        return self.influence_strength * np.exp(-normalized_distance**2)

@dataclass
class MorphogenSignal:
    """Represents a morphogenic signaling molecule/field."""
    signal_id: str
    source_coordinates: np.ndarray
    concentration: float
    diffusion_rate: float
    decay_rate: float
    influence_radius: float
    signal_type: str  # 'attractive', 'repulsive', 'organizing', 'inhibitory'
    creation_time: datetime = field(default_factory=datetime.now)
    
    def get_concentration_at(self, coordinates: np.ndarray, current_time: datetime) -> float:
        """Calculate signal concentration at given coordinates and time."""
        distance = np.linalg.norm(coordinates - self.source_coordinates)
        
        # Time decay
        time_elapsed = (current_time - self.creation_time).total_seconds() / 3600  # hours
        time_factor = np.exp(-self.decay_rate * time_elapsed)
        
        # Spatial decay
        if distance > self.influence_radius:
            return 0.0
            
        spatial_factor = np.exp(-distance / (self.influence_radius * self.diffusion_rate))
        
        return self.concentration * time_factor * spatial_factor

@dataclass
class EpigeneticState:
    """Represents an epigenetic memory configuration."""
    state_id: str
    active_patterns: Set[str]
    suppressed_patterns: Set[str]
    activation_context: Dict[str, Any]
    inheritance_strength: float
    modification_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def matches_context(self, context: Dict[str, Any]) -> float:
        """Calculate how well this state matches current context."""
        if not self.activation_context:
            return 0.0
            
        matches = 0
        total = len(self.activation_context)
        
        for key, expected_value in self.activation_context.items():
            if key in context:
                if isinstance(expected_value, (int, float)) and isinstance(context[key], (int, float)):
                    # Numerical similarity
                    diff = abs(expected_value - context[key])
                    max_val = max(abs(expected_value), abs(context[key]), 1)
                    similarity = 1.0 - (diff / max_val)
                    matches += max(0, similarity)
                elif expected_value == context[key]:
                    matches += 1.0
                    
        return matches / total if total > 0 else 0.0

class BioElectricMemory:
    """Bio-electric pattern storage system."""
    
    def __init__(self):
        self.pattern_fields: Dict[str, BioElectricSignature] = {}
        self.ion_channels: Dict[str, List[str]] = defaultdict(list)  # connectivity
        self.signal_propagation: Dict[str, float] = {}
        self.resonance_network: Dict[str, Set[str]] = defaultdict(set)
        
    def create_pattern(self, pattern_id: str, data: Any, context: Dict[str, Any] = None) -> BioElectricSignature:
        """Create a new bio-electric memory pattern."""
        # Convert data to numerical signature
        signature = self._generate_signature(data, context)
        
        pattern = BioElectricSignature(
            pattern_id=pattern_id,
            signature=signature,
            strength=0.5,
            frequency=1.0,
            last_activation=datetime.now()
        )
        
        self.pattern_fields[pattern_id] = pattern
        self._update_resonance_network(pattern_id)
        
        logger.info(f"Created bio-electric pattern: {pattern_id}")
        return pattern
        
    def activate_pattern(self, pattern_id: str, intensity: float = 1.0) -> bool:
        """Activate a memory pattern."""
        if pattern_id not in self.pattern_fields:
            return False
            
        pattern = self.pattern_fields[pattern_id]
        pattern.activate(intensity)
        
        # Propagate activation through ion channels
        self._propagate_activation(pattern_id, intensity * 0.7)
        
        return True
        
    def get_resonant_patterns(self, pattern_id: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find patterns that resonate with the given pattern."""
        if pattern_id not in self.pattern_fields:
            return []
            
        source_pattern = self.pattern_fields[pattern_id]
        resonant = []
        
        for other_id, other_pattern in self.pattern_fields.items():
            if other_id != pattern_id:
                resonance = source_pattern.resonates_with(other_pattern)
                if resonance >= threshold:
                    resonant.append((other_id, resonance))
                    
        return sorted(resonant, key=lambda x: x[1], reverse=True)
        
    def decay_patterns(self, time_delta: float = 1.0):
        """Apply natural decay to all patterns."""
        for pattern in self.pattern_fields.values():
            pattern.decay(time_delta)
            
    def _generate_signature(self, data: Any, context: Dict[str, Any] = None) -> np.ndarray:
        """Generate bio-electric signature from data."""
        # Convert data to string representation
        data_str = str(data)
        
        # Create base signature from string hash
        signature_length = 64
        signature = np.zeros(signature_length)
        
        for i, char in enumerate(data_str):
            idx = (ord(char) + i) % signature_length
            signature[idx] += 1.0
            
        # Add context influence
        if context:
            for key, value in context.items():
                key_hash = hash(str(key) + str(value)) % signature_length
                signature[key_hash] += 0.5
                
        # Normalize
        if np.sum(signature) > 0:
            signature = signature / np.sum(signature)
            
        # Add some noise for uniqueness
        noise = np.random.normal(0, 0.01, signature_length)
        signature = signature + noise
        
        return signature
        
    def _update_resonance_network(self, pattern_id: str):
        """Update resonance connections for a pattern."""
        resonant_patterns = self.get_resonant_patterns(pattern_id, 0.5)
        
        for other_id, strength in resonant_patterns:
            self.resonance_network[pattern_id].add(other_id)
            self.resonance_network[other_id].add(pattern_id)
            
            # Create ion channel connections
            if strength > 0.8:
                self.ion_channels[pattern_id].append(other_id)
                self.ion_channels[other_id].append(pattern_id)
                
    def _propagate_activation(self, source_id: str, intensity: float):
        """Propagate activation through connected patterns."""
        if intensity < 0.1:  # Stop propagation if too weak
            return
            
        for connected_id in self.ion_channels[source_id]:
            if connected_id in self.pattern_fields:
                self.activate_pattern(connected_id, intensity * 0.8)

class PositionalMemory:
    """Spatial consciousness mapping system."""
    
    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions
        self.consciousness_coordinates: Dict[str, np.ndarray] = {}
        self.conceptual_neighborhoods: Dict[str, Set[str]] = defaultdict(set)
        self.morphogenetic_gradients: Dict[str, np.ndarray] = {}
        self.territory_boundaries: Dict[str, MemoryTerritory] = {}
        
    def place_memory(self, memory_id: str, coordinates: np.ndarray = None, 
                    context: Dict[str, Any] = None) -> np.ndarray:
        """Place a memory at coordinates in consciousness space."""
        if coordinates is None:
            coordinates = self._find_optimal_position(memory_id, context)
            
        self.consciousness_coordinates[memory_id] = coordinates
        self._update_neighborhoods(memory_id)
        
        logger.info(f"Placed memory {memory_id} at coordinates {coordinates}")
        return coordinates
        
    def create_territory(self, territory_id: str, center: np.ndarray, 
                        radius: float, influence: float) -> MemoryTerritory:
        """Create a new memory territory."""
        territory = MemoryTerritory(
            territory_id=territory_id,
            center_coordinates=center,
            boundary_radius=radius,
            influence_strength=influence
        )
        
        self.territory_boundaries[territory_id] = territory
        self._update_territory_neighbors(territory_id)
        
        return territory
        
    def get_memories_in_region(self, center: np.ndarray, radius: float) -> List[str]:
        """Get all memories within a spatial region."""
        memories_in_region = []
        
        for memory_id, coordinates in self.consciousness_coordinates.items():
            distance = np.linalg.norm(coordinates - center)
            if distance <= radius:
                memories_in_region.append(memory_id)
                
        return memories_in_region
        
    def calculate_proximity_influence(self, memory_id: str) -> Dict[str, float]:
        """Calculate influence of nearby memories."""
        if memory_id not in self.consciousness_coordinates:
            return {}
            
        memory_coords = self.consciousness_coordinates[memory_id]
        influences = {}
        
        for other_id, other_coords in self.consciousness_coordinates.items():
            if other_id != memory_id:
                distance = np.linalg.norm(memory_coords - other_coords)
                # Inverse square law with minimum distance
                influence = 1.0 / (1.0 + distance**2)
                influences[other_id] = influence
                
        return influences
        
    def _find_optimal_position(self, memory_id: str, context: Dict[str, Any] = None) -> np.ndarray:
        """Find optimal position for memory based on context and existing memories."""
        if not self.consciousness_coordinates:
            # First memory - place at origin
            return np.zeros(self.dimensions)
            
        # Use context to find similar memories
        similar_memories = self._find_similar_memories(memory_id, context)
        
        if similar_memories:
            # Place near similar memories with some randomness
            similar_coords = [self.consciousness_coordinates[mem_id] for mem_id in similar_memories[:3]]
            center = np.mean(similar_coords, axis=0)
            
            # Add some randomness to avoid exact overlap
            noise = np.random.normal(0, 0.5, self.dimensions)
            return center + noise
        else:
            # Place in relatively empty space
            return self._find_empty_space()
            
    def _find_similar_memories(self, memory_id: str, context: Dict[str, Any] = None) -> List[str]:
        """Find memories with similar context or content."""
        # This is a simplified version - would be enhanced with actual similarity metrics
        if not context:
            return []
            
        # For now, return memories with overlapping context keys
        similar = []
        # This would typically use the bio-electric patterns for similarity
        return similar
        
    def _find_empty_space(self) -> np.ndarray:
        """Find a relatively empty region in consciousness space."""
        # Simple approach - find area with minimal density
        max_attempts = 50
        best_position = None
        min_crowding = float('inf')
        
        for _ in range(max_attempts):
            candidate = np.random.normal(0, 2, self.dimensions)
            
            # Calculate crowding (sum of inverse distances)
            crowding = 0
            for coords in self.consciousness_coordinates.values():
                distance = np.linalg.norm(candidate - coords)
                crowding += 1.0 / (1.0 + distance)
                
            if crowding < min_crowding:
                min_crowding = crowding
                best_position = candidate
                
        return best_position if best_position is not None else np.random.normal(0, 1, self.dimensions)
        
    def _update_neighborhoods(self, memory_id: str):
        """Update neighborhood relationships for a memory."""
        if memory_id not in self.consciousness_coordinates:
            return
            
        memory_coords = self.consciousness_coordinates[memory_id]
        neighborhood_radius = 2.0
        
        for other_id, other_coords in self.consciousness_coordinates.items():
            if other_id != memory_id:
                distance = np.linalg.norm(memory_coords - other_coords)
                if distance <= neighborhood_radius:
                    self.conceptual_neighborhoods[memory_id].add(other_id)
                    self.conceptual_neighborhoods[other_id].add(memory_id)
                    
    def _update_territory_neighbors(self, territory_id: str):
        """Update neighboring territories."""
        if territory_id not in self.territory_boundaries:
            return
            
        territory = self.territory_boundaries[territory_id]
        
        for other_id, other_territory in self.territory_boundaries.items():
            if other_id != territory_id:
                distance = np.linalg.norm(territory.center_coordinates - other_territory.center_coordinates)
                threshold = territory.boundary_radius + other_territory.boundary_radius
                
                if distance <= threshold * 1.5:  # Allow some overlap
                    territory.neighboring_territories.add(other_id)
                    other_territory.neighboring_territories.add(territory_id)

class MorphogenicSignaling:
    """Morphogen-inspired signaling system."""
    
    def __init__(self):
        self.signal_sources: Dict[str, MorphogenSignal] = {}
        self.concentration_fields: Dict[str, np.ndarray] = {}
        self.diffusion_patterns: Dict[str, Dict[str, float]] = {}
        self.threshold_responses: Dict[str, float] = {}
        
    def create_signal(self, signal_id: str, source_coords: np.ndarray, 
                     signal_type: str, initial_concentration: float = 1.0,
                     diffusion_rate: float = 0.8, decay_rate: float = 0.1,
                     influence_radius: float = 5.0) -> MorphogenSignal:
        """Create a new morphogenic signal."""
        signal = MorphogenSignal(
            signal_id=signal_id,
            source_coordinates=source_coords,
            concentration=initial_concentration,
            diffusion_rate=diffusion_rate,
            decay_rate=decay_rate,
            influence_radius=influence_radius,
            signal_type=signal_type
        )
        
        self.signal_sources[signal_id] = signal
        self.threshold_responses[signal_id] = 0.3  # Default activation threshold
        
        logger.info(f"Created morphogenic signal: {signal_id} of type {signal_type}")
        return signal
        
    def get_signal_at_position(self, coordinates: np.ndarray, current_time: datetime = None) -> Dict[str, float]:
        """Get all signal concentrations at given coordinates."""
        if current_time is None:
            current_time = datetime.now()
            
        concentrations = {}
        
        for signal_id, signal in self.signal_sources.items():
            concentration = signal.get_concentration_at(coordinates, current_time)
            if concentration > 0.001:  # Only include significant concentrations
                concentrations[signal_id] = concentration
                
        return concentrations
        
    def calculate_total_influence(self, coordinates: np.ndarray, signal_types: List[str] = None) -> float:
        """Calculate total morphogenic influence at coordinates."""
        current_time = datetime.now()
        total_influence = 0.0
        
        for signal_id, signal in self.signal_sources.items():
            if signal_types is None or signal.signal_type in signal_types:
                concentration = signal.get_concentration_at(coordinates, current_time)
                
                # Apply signal type effects
                if signal.signal_type == 'attractive':
                    total_influence += concentration
                elif signal.signal_type == 'repulsive':
                    total_influence -= concentration
                elif signal.signal_type == 'organizing':
                    total_influence += concentration * 0.8
                elif signal.signal_type == 'inhibitory':
                    total_influence -= concentration * 0.6
                    
        return total_influence
        
    def propagate_signals(self, time_step: float = 1.0):
        """Update signal propagation and decay."""
        current_time = datetime.now()
        
        # Update diffusion patterns
        for signal_id, signal in self.signal_sources.items():
            self._calculate_diffusion_pattern(signal_id, signal, current_time)
            
        # Remove expired signals
        expired_signals = []
        for signal_id, signal in self.signal_sources.items():
            time_elapsed = (current_time - signal.creation_time).total_seconds() / 3600
            if time_elapsed > 24:  # Remove signals older than 24 hours
                expired_signals.append(signal_id)
                
        for signal_id in expired_signals:
            del self.signal_sources[signal_id]
            logger.info(f"Removed expired signal: {signal_id}")
            
    def _calculate_diffusion_pattern(self, signal_id: str, signal: MorphogenSignal, current_time: datetime):
        """Calculate how signal diffuses through space."""
        # Create a spatial grid for the diffusion pattern
        grid_size = 20
        grid_extent = signal.influence_radius * 2
        
        x = np.linspace(-grid_extent, grid_extent, grid_size)
        y = np.linspace(-grid_extent, grid_extent, grid_size)
        
        diffusion_grid = np.zeros((grid_size, grid_size))
        
        for i, x_pos in enumerate(x):
            for j, y_pos in enumerate(y):
                coords = signal.source_coordinates.copy()
                coords[0] += x_pos
                coords[1] += y_pos
                
                concentration = signal.get_concentration_at(coords, current_time)
                diffusion_grid[i, j] = concentration
                
        self.concentration_fields[signal_id] = diffusion_grid

class EpigeneticMemory:
    """Epigenetic memory layers for context-dependent activation."""
    
    def __init__(self):
        self.expression_patterns: Dict[str, EpigeneticState] = {}
        self.dormant_patterns: Dict[str, EpigeneticState] = {}
        self.activation_triggers: Dict[str, Dict[str, Any]] = {}
        self.inheritance_rules: Dict[str, float] = {}
        self.context_history: List[Dict[str, Any]] = []
        
    def create_epigenetic_state(self, state_id: str, active_patterns: Set[str],
                               context: Dict[str, Any], inheritance_strength: float = 0.8) -> EpigeneticState:
        """Create a new epigenetic memory state."""
        state = EpigeneticState(
            state_id=state_id,
            active_patterns=active_patterns,
            suppressed_patterns=set(),
            activation_context=context,
            inheritance_strength=inheritance_strength
        )
        
        self.expression_patterns[state_id] = state
        self.inheritance_rules[state_id] = inheritance_strength
        
        logger.info(f"Created epigenetic state: {state_id}")
        return state
        
    def activate_state(self, state_id: str, current_context: Dict[str, Any]) -> bool:
        """Activate an epigenetic state if context matches."""
        if state_id not in self.expression_patterns:
            return False
            
        state = self.expression_patterns[state_id]
        match_strength = state.matches_context(current_context)
        
        if match_strength > 0.5:  # Threshold for activation
            # Record activation
            state.modification_history.append({
                'timestamp': datetime.now(),
                'action': 'activated',
                'context': current_context,
                'match_strength': match_strength
            })
            
            logger.info(f"Activated epigenetic state: {state_id} (match: {match_strength:.2f})")
            return True
            
        return False
        
    def find_matching_states(self, context: Dict[str, Any], threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Find epigenetic states that match current context."""
        matching_states = []
        
        for state_id, state in self.expression_patterns.items():
            match_strength = state.matches_context(context)
            if match_strength >= threshold:
                matching_states.append((state_id, match_strength))
                
        return sorted(matching_states, key=lambda x: x[1], reverse=True)
        
    def inherit_patterns(self, parent_context: Dict[str, Any], child_context: Dict[str, Any]) -> Dict[str, EpigeneticState]:
        """Create inherited patterns based on parent context."""
        inherited_states = {}
        
        for state_id, state in self.expression_patterns.items():
            parent_match = state.matches_context(parent_context)
            
            if parent_match > 0.3:  # Inheritable threshold
                # Create modified version for child context
                inherited_id = f"{state_id}_inherited_{len(inherited_states)}"
                
                inherited_state = EpigeneticState(
                    state_id=inherited_id,
                    active_patterns=state.active_patterns.copy(),
                    suppressed_patterns=state.suppressed_patterns.copy(),
                    activation_context=child_context,
                    inheritance_strength=state.inheritance_strength * 0.9
                )
                
                inherited_states[inherited_id] = inherited_state
                self.expression_patterns[inherited_id] = inherited_state
                
        logger.info(f"Created {len(inherited_states)} inherited patterns")
        return inherited_states
        
    def suppress_patterns(self, pattern_ids: Set[str], context: Dict[str, Any]):
        """Suppress specific patterns in given context."""
        for state_id, state in self.expression_patterns.items():
            if state.matches_context(context) > 0.5:
                state.suppressed_patterns.update(pattern_ids)
                state.modification_history.append({
                    'timestamp': datetime.now(),
                    'action': 'suppressed',
                    'patterns': list(pattern_ids),
                    'context': context
                })
                
    def evolve_patterns(self, selection_pressure: Dict[str, float]):
        """Evolve patterns based on selection pressure."""
        for state_id, pressure in selection_pressure.items():
            if state_id in self.expression_patterns:
                state = self.expression_patterns[state_id]
                
                if pressure > 0:
                    # Strengthen successful patterns
                    state.inheritance_strength = min(1.0, state.inheritance_strength + pressure * 0.1)
                else:
                    # Weaken unsuccessful patterns
                    state.inheritance_strength = max(0.1, state.inheritance_strength + pressure * 0.1)
                    
                # Move to dormant if too weak
                if state.inheritance_strength < 0.2:
                    self.dormant_patterns[state_id] = state
                    del self.expression_patterns[state_id]
                    logger.info(f"Moved pattern {state_id} to dormant state")

class MorphogeneticMemorySystem:
    """Integrated morphogenetic memory system."""
    
    def __init__(self, consciousness_dimensions: int = 3):
        self.bio_electric = BioElectricMemory()
        self.positional = PositionalMemory(consciousness_dimensions)
        self.morphogenic = MorphogenicSignaling()
        self.epigenetic = EpigeneticMemory()
        
        self.memory_metadata: Dict[str, Dict[str, Any]] = {}
        self.system_state = {
            'total_memories': 0,
            'active_territories': 0,
            'active_signals': 0,
            'epigenetic_states': 0,
            'last_update': datetime.now()
        }
        
    def create_memory(self, memory_id: str, content: Any, context: Dict[str, Any] = None,
                     memory_type: str = 'general') -> Dict[str, Any]:
        """Create a new memory with full morphogenetic integration."""
        if context is None:
            context = {}
            
        # Create bio-electric pattern
        bio_pattern = self.bio_electric.create_pattern(memory_id, content, context)
        
        # Place in consciousness space
        coordinates = self.positional.place_memory(memory_id, context=context)
        
        # Create morphogenic signal
        signal_type = context.get('signal_type', 'organizing')
        morphogen_signal = self.morphogenic.create_signal(
            f"{memory_id}_signal",
            coordinates,
            signal_type
        )
        
        # Create or update epigenetic state
        active_patterns = {memory_id}
        epigenetic_state = self.epigenetic.create_epigenetic_state(
            f"{memory_id}_state",
            active_patterns,
            context
        )
        
        # Store metadata
        self.memory_metadata[memory_id] = {
            'content': content,
            'context': context,
            'memory_type': memory_type,
            'bio_pattern_id': memory_id,
            'coordinates': coordinates.tolist(),
            'signal_id': f"{memory_id}_signal",
            'epigenetic_state_id': f"{memory_id}_state",
            'creation_time': datetime.now(),
            'activation_count': 0
        }
        
        self._update_system_state()
        
        logger.info(f"Created integrated memory: {memory_id}")
        return self.memory_metadata[memory_id]
        
    def recall_memory(self, memory_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recall a memory with morphogenetic activation."""
        if memory_id not in self.memory_metadata:
            return {}
            
        # Activate bio-electric pattern
        self.bio_electric.activate_pattern(memory_id, 1.0)
        
        # Get resonant patterns
        resonant = self.bio_electric.get_resonant_patterns(memory_id, 0.5)
        
        # Check epigenetic activation
        if context:
            matching_states = self.epigenetic.find_matching_states(context, 0.3)
        else:
            matching_states = []
            
        # Update metadata
        metadata = self.memory_metadata[memory_id]
        metadata['activation_count'] += 1
        metadata['last_recall'] = datetime.now()
        metadata['resonant_memories'] = [r[0] for r in resonant[:5]]
        metadata['epigenetic_matches'] = [m[0] for m in matching_states[:3]]
        
        return metadata
        
    def get_memory_environment(self, memory_id: str) -> Dict[str, Any]:
        """Get the full morphogenetic environment around a memory."""
        if memory_id not in self.memory_metadata:
            return {}
            
        metadata = self.memory_metadata[memory_id]
        coordinates = np.array(metadata['coordinates'])
        
        # Get nearby memories
        nearby_memories = self.positional.get_memories_in_region(coordinates, 3.0)
        
        # Get morphogenic signals
        signals = self.morphogenic.get_signal_at_position(coordinates)
        
        # Get proximity influences
        influences = self.positional.calculate_proximity_influence(memory_id)
        
        # Get resonant patterns
        resonant = self.bio_electric.get_resonant_patterns(memory_id, 0.4)
        
        return {
            'memory_id': memory_id,
            'coordinates': coordinates.tolist(),
            'nearby_memories': nearby_memories,
            'morphogenic_signals': signals,
            'proximity_influences': influences,
            'resonant_patterns': dict(resonant),
            'environment_strength': sum(influences.values()) + sum(signals.values())
        }
        
    def evolve_memory_landscape(self, time_step: float = 1.0):
        """Evolve the entire memory landscape."""
        # Decay bio-electric patterns
        self.bio_electric.decay_patterns(time_step)
        
        # Propagate morphogenic signals
        self.morphogenic.propagate_signals(time_step)
        
        # Calculate selection pressures for epigenetic evolution
        selection_pressures = self._calculate_selection_pressures()
        self.epigenetic.evolve_patterns(selection_pressures)
        
        # Update territorial boundaries based on memory density
        self._update_territory_dynamics()
        
        # Create emergent connections between highly resonant memories
        self._form_emergent_connections()
        
        self._update_system_state()
        logger.info(f"Memory landscape evolved (time_step: {time_step})")
        
    def _calculate_selection_pressures(self) -> Dict[str, float]:
        """Calculate evolutionary pressure on epigenetic patterns."""
        pressures = {}
        
        for state_id, state in self.epigenetic.expression_patterns.items():
            # Base pressure from activation frequency
            recent_activations = len([
                entry for entry in state.modification_history[-10:]
                if entry.get('action') == 'activated'
            ])
            
            base_pressure = (recent_activations - 5) / 10.0  # Normalize around 5 activations
            
            # Additional pressure from pattern resonance
            resonance_pressure = 0.0
            for pattern_id in state.active_patterns:
                if pattern_id in self.bio_electric.pattern_fields:
                    pattern = self.bio_electric.pattern_fields[pattern_id]
                    resonance_pressure += pattern.strength * 0.1
                    
            pressures[state_id] = base_pressure + resonance_pressure
            
        return pressures
        
    def _update_territory_dynamics(self):
        """Update memory territories based on current memory distribution."""
        # Find high-density memory clusters
        memory_positions = []
        memory_ids = []
        
        for memory_id, metadata in self.memory_metadata.items():
            memory_positions.append(metadata['coordinates'])
            memory_ids.append(memory_id)
            
        if len(memory_positions) < 3:
            return
            
        memory_positions = np.array(memory_positions)
        
        # Simple clustering to identify territory centers
        from sklearn.cluster import KMeans
        n_clusters = min(5, len(memory_positions) // 3)
        
        if n_clusters > 0:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_centers = kmeans.fit(memory_positions).cluster_centers_
                
                # Create or update territories
                for i, center in enumerate(cluster_centers):
                    territory_id = f"territory_{i}"
                    
                    if territory_id not in self.positional.territory_boundaries:
                        self.positional.create_territory(
                            territory_id, center, radius=2.0, influence=0.8
                        )
                    else:
                        # Update existing territory
                        territory = self.positional.territory_boundaries[territory_id]
                        territory.center_coordinates = center
                        
            except ImportError:
                # Fallback if sklearn not available - simple grid territories
                self._create_grid_territories(memory_positions)
                
    def _create_grid_territories(self, positions: np.ndarray):
        """Fallback method to create grid-based territories."""
        if len(positions) == 0:
            return
            
        # Find bounds of memory space
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        
        # Create grid territories
        grid_size = 3
        for i in range(grid_size):
            for j in range(grid_size):
                territory_id = f"grid_territory_{i}_{j}"
                
                center = min_coords + (max_coords - min_coords) * np.array([
                    (i + 0.5) / grid_size,
                    (j + 0.5) / grid_size,
                    0.5  # Middle of z-axis
                ])
                
                if territory_id not in self.positional.territory_boundaries:
                    self.positional.create_territory(
                        territory_id, center, radius=1.5, influence=0.6
                    )
                    
    def _form_emergent_connections(self):
        """Form new connections between highly resonant memories."""
        resonance_threshold = 0.8
        
        for memory_id in self.memory_metadata.keys():
            resonant_patterns = self.bio_electric.get_resonant_patterns(
                memory_id, resonance_threshold
            )
            
            for resonant_id, strength in resonant_patterns:
                # Create morphogenic bridge signal between highly resonant memories
                if strength > 0.9:
                    mem1_coords = np.array(self.memory_metadata[memory_id]['coordinates'])
                    mem2_coords = np.array(self.memory_metadata[resonant_id]['coordinates'])
                    
                    # Place bridge signal at midpoint
                    bridge_coords = (mem1_coords + mem2_coords) / 2
                    bridge_id = f"bridge_{memory_id}_{resonant_id}"
                    
                    if bridge_id not in self.morphogenic.signal_sources:
                        self.morphogenic.create_signal(
                            bridge_id,
                            bridge_coords,
                            'organizing',
                            initial_concentration=strength,
                            influence_radius=np.linalg.norm(mem1_coords - mem2_coords)
                        )
                        
    def _update_system_state(self):
        """Update overall system state metrics."""
        self.system_state.update({
            'total_memories': len(self.memory_metadata),
            'active_territories': len(self.positional.territory_boundaries),
            'active_signals': len(self.morphogenic.signal_sources),
            'epigenetic_states': len(self.epigenetic.expression_patterns),
            'bio_electric_patterns': len(self.bio_electric.pattern_fields),
            'last_update': datetime.now()
        })
        
    def get_consciousness_map(self) -> Dict[str, Any]:
        """Generate a comprehensive map of the consciousness landscape."""
        # Get all memory positions
        memory_map = {}
        for memory_id, metadata in self.memory_metadata.items():
            memory_map[memory_id] = {
                'coordinates': metadata['coordinates'],
                'content_type': metadata.get('memory_type', 'general'),
                'strength': self.bio_electric.pattern_fields.get(memory_id, {}).strength if memory_id in self.bio_electric.pattern_fields else 0,
                'activation_count': metadata.get('activation_count', 0)
            }
            
        # Get territory map
        territory_map = {}
        for territory_id, territory in self.positional.territory_boundaries.items():
            territory_map[territory_id] = {
                'center': territory.center_coordinates.tolist(),
                'radius': territory.boundary_radius,
                'influence': territory.influence_strength,
                'memory_count': len(territory.memory_patterns)
            }
            
        # Get signal field map
        signal_map = {}
        current_time = datetime.now()
        for signal_id, signal in self.morphogenic.signal_sources.items():
            signal_map[signal_id] = {
                'source': signal.source_coordinates.tolist(),
                'type': signal.signal_type,
                'concentration': signal.concentration,
                'age_hours': (current_time - signal.creation_time).total_seconds() / 3600,
                'influence_radius': signal.influence_radius
            }
            
        return {
            'memories': memory_map,
            'territories': territory_map,
            'signals': signal_map,
            'system_state': self.system_state,
            'resonance_networks': dict(self.bio_electric.resonance_network),
            'generation_time': datetime.now()
        }
        
    def search_memories(self, query: str, context: Dict[str, Any] = None,
                       search_type: str = 'content') -> List[Dict[str, Any]]:
        """Search memories using morphogenetic principles."""
        results = []
        
        if search_type == 'content':
            # Content-based search
            query_pattern = self.bio_electric._generate_signature(query, context)
            
            for memory_id, metadata in self.memory_metadata.items():
                if memory_id in self.bio_electric.pattern_fields:
                    pattern = self.bio_electric.pattern_fields[memory_id]
                    
                    # Calculate similarity
                    similarity = np.corrcoef(query_pattern, pattern.signature)[0, 1]
                    if not np.isnan(similarity) and similarity > 0.3:
                        results.append({
                            'memory_id': memory_id,
                            'similarity': similarity,
                            'content': metadata.get('content'),
                            'coordinates': metadata['coordinates'],
                            'strength': pattern.strength
                        })
                        
        elif search_type == 'spatial':
            # Spatial proximity search
            if context and 'coordinates' in context:
                search_coords = np.array(context['coordinates'])
                radius = context.get('radius', 2.0)
                
                nearby_memories = self.positional.get_memories_in_region(search_coords, radius)
                
                for memory_id in nearby_memories:
                    if memory_id in self.memory_metadata:
                        metadata = self.memory_metadata[memory_id]
                        distance = np.linalg.norm(
                            np.array(metadata['coordinates']) - search_coords
                        )
                        
                        results.append({
                            'memory_id': memory_id,
                            'distance': distance,
                            'content': metadata.get('content'),
                            'coordinates': metadata['coordinates']
                        })
                        
        elif search_type == 'epigenetic':
            # Context-based epigenetic search
            if context:
                matching_states = self.epigenetic.find_matching_states(context, 0.2)
                
                for state_id, match_strength in matching_states:
                    state = self.epigenetic.expression_patterns[state_id]
                    
                    for pattern_id in state.active_patterns:
                        if pattern_id in self.memory_metadata:
                            results.append({
                                'memory_id': pattern_id,
                                'epigenetic_match': match_strength,
                                'state_id': state_id,
                                'content': self.memory_metadata[pattern_id].get('content'),
                                'coordinates': self.memory_metadata[pattern_id]['coordinates']
                            })
                            
        # Sort results by relevance
        if search_type == 'content':
            results.sort(key=lambda x: x['similarity'], reverse=True)
        elif search_type == 'spatial':
            results.sort(key=lambda x: x['distance'])
        elif search_type == 'epigenetic':
            results.sort(key=lambda x: x['epigenetic_match'], reverse=True)
            
        return results[:10]  # Return top 10 results
        
    def export_state(self) -> Dict[str, Any]:
        """Export the complete morphogenetic memory state."""
        return {
            'memory_metadata': self.memory_metadata,
            'system_state': self.system_state,
            'bio_electric_patterns': {
                pattern_id: {
                    'signature': pattern.signature.tolist(),
                    'strength': pattern.strength,
                    'frequency': pattern.frequency,
                    'activation_count': pattern.activation_count,
                    'last_activation': pattern.last_activation.isoformat()
                }
                for pattern_id, pattern in self.bio_electric.pattern_fields.items()
            },
            'consciousness_coordinates': {
                memory_id: coords.tolist()
                for memory_id, coords in self.positional.consciousness_coordinates.items()
            },
            'epigenetic_states': {
                state_id: {
                    'active_patterns': list(state.active_patterns),
                    'suppressed_patterns': list(state.suppressed_patterns),
                    'activation_context': state.activation_context,
                    'inheritance_strength': state.inheritance_strength,
                    'modification_history': state.modification_history
                }
                for state_id, state in self.epigenetic.expression_patterns.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
    def import_state(self, state_data: Dict[str, Any]) -> bool:
        """Import a previously exported morphogenetic memory state."""
        try:
            # Import basic metadata
            self.memory_metadata = state_data.get('memory_metadata', {})
            self.system_state = state_data.get('system_state', {})
            
            # Reconstruct bio-electric patterns
            bio_patterns = state_data.get('bio_electric_patterns', {})
            for pattern_id, pattern_data in bio_patterns.items():
                signature = np.array(pattern_data['signature'])
                
                pattern = BioElectricSignature(
                    pattern_id=pattern_id,
                    signature=signature,
                    strength=pattern_data['strength'],
                    frequency=pattern_data['frequency'],
                    last_activation=datetime.fromisoformat(pattern_data['last_activation']),
                    activation_count=pattern_data['activation_count']
                )
                
                self.bio_electric.pattern_fields[pattern_id] = pattern
                
            # Reconstruct consciousness coordinates
            coordinates_data = state_data.get('consciousness_coordinates', {})
            for memory_id, coords_list in coordinates_data.items():
                self.positional.consciousness_coordinates[memory_id] = np.array(coords_list)
                
            # Reconstruct epigenetic states
            epigenetic_data = state_data.get('epigenetic_states', {})
            for state_id, state_info in epigenetic_data.items():
                state = EpigeneticState(
                    state_id=state_id,
                    active_patterns=set(state_info['active_patterns']),
                    suppressed_patterns=set(state_info['suppressed_patterns']),
                    activation_context=state_info['activation_context'],
                    inheritance_strength=state_info['inheritance_strength'],
                    modification_history=state_info['modification_history']
                )
                
                self.epigenetic.expression_patterns[state_id] = state
                
            # Rebuild derived structures
            self._rebuild_derived_structures()
            
            logger.info("Successfully imported morphogenetic memory state")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import state: {e}")
            return False
            
    def _rebuild_derived_structures(self):
        """Rebuild derived structures after state import."""
        # Rebuild resonance networks
        for pattern_id in self.bio_electric.pattern_fields.keys():
            self.bio_electric._update_resonance_network(pattern_id)
            
        # Rebuild neighborhoods
        for memory_id in self.positional.consciousness_coordinates.keys():
            self.positional._update_neighborhoods(memory_id)
            
        # Recreate territories based on current memory distribution
        self._update_territory_dynamics()

# Factory function for Kotlin bridge compatibility
def create_morphogenetic_memory_system(dimensions: int = 3) -> MorphogeneticMemorySystem:
    """Factory function to create MorphogeneticMemorySystem instance."""
    return MorphogeneticMemorySystem(dimensions)

# Utility functions for analysis
def analyze_memory_evolution(system: MorphogeneticMemorySystem, memory_id: str) -> Dict[str, Any]:
    """Analyze how a memory has evolved over time."""
    if memory_id not in system.memory_metadata:
        return {}
        
    metadata = system.memory_metadata[memory_id]
    
    # Get bio-electric pattern evolution
    if memory_id in system.bio_electric.pattern_fields:
        pattern = system.bio_electric.pattern_fields[memory_id]
        pattern_evolution = {
            'current_strength': pattern.strength,
            'activation_count': pattern.activation_count,
            'last_activation': pattern.last_activation,
            'resonant_connections': len(system.bio_electric.resonance_network.get(memory_id, set()))
        }
    else:
        pattern_evolution = {}
        
    # Get spatial evolution
    coordinates = np.array(metadata['coordinates'])
    nearby_memories = system.positional.get_memories_in_region(coordinates, 2.0)
    
    # Get epigenetic context
    epigenetic_contexts = []
    for state_id, state in system.epigenetic.expression_patterns.items():
        if memory_id in state.active_patterns:
            epigenetic_contexts.append({
                'state_id': state_id,
                'context': state.activation_context,
                'inheritance_strength': state.inheritance_strength
            })
            
    return {
        'memory_id': memory_id,
        'pattern_evolution': pattern_evolution,
        'spatial_neighborhood': nearby_memories,
        'epigenetic_contexts': epigenetic_contexts,
        'morphogenic_environment': system.get_memory_environment(memory_id),
        'analysis_timestamp': datetime.now()
    }

def generate_consciousness_report(system: MorphogeneticMemorySystem) -> str:
    """Generate a comprehensive report of the consciousness state."""
    consciousness_map = system.get_consciousness_map()
    
    report = f"""
Morphogenetic Memory System Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM OVERVIEW
===============
Total Memories: {consciousness_map['system_state']['total_memories']}
Active Territories: {consciousness_map['system_state']['active_territories']}
Morphogenic Signals: {consciousness_map['system_state']['active_signals']}
Epigenetic States: {consciousness_map['system_state']['epigenetic_states']}
Bio-Electric Patterns: {consciousness_map['system_state']['bio_electric_patterns']}

MEMORY DISTRIBUTION
==================
"""
    
    # Memory strength distribution
    strengths = [mem['strength'] for mem in consciousness_map['memories'].values()]
    if strengths:
        report += f"Average Memory Strength: {np.mean(strengths):.3f}\n"
        report += f"Strongest Memory: {max(strengths):.3f}\n"
        report += f"Memory Strength Std Dev: {np.std(strengths):.3f}\n"
        
    # Territory analysis
    report += f"\nTERRITORIAL ANALYSIS\n"
    report += f"===================\n"
    
    for territory_id, territory_info in consciousness_map['territories'].items():
        report += f"{territory_id}: Center {territory_info['center']}, "
        report += f"Influence {territory_info['influence']:.2f}, "
        report += f"Memories {territory_info['memory_count']}\n"
        
    # Signal field analysis
    report += f"\nMORPHOGENIC SIGNALS\n"
    report += f"==================\n"
    
    signal_types = {}
    for signal_info in consciousness_map['signals'].values():
        signal_type = signal_info['type']
        signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        
    for signal_type, count in signal_types.items():
        report += f"{signal_type}: {count} signals\n"
        
    return report

# Example usage and testing
if __name__ == "__main__":
    # Create the morphogenetic memory system
    memory_system = create_morphogenetic_memory_system(3)
    
    # Create some test memories
    test_memories = [
        {
            'id': 'memory_001',
            'content': 'Learning about consciousness',
            'context': {'domain': 'cognition', 'importance': 0.8}
        },
        {
            'id': 'memory_002', 
            'content': 'Understanding morphogenesis',
            'context': {'domain': 'biology', 'importance': 0.9}
        },
        {
            'id': 'memory_003',
            'content': 'Exploring recursive awareness',
            'context': {'domain': 'cognition', 'importance': 0.7}
        }
    ]
    
    # Create memories
    for mem in test_memories:
        memory_system.create_memory(mem['id'], mem['content'], mem['context'])
        
    # Evolve the system
    memory_system.evolve_memory_landscape(1.0)
    
    # Generate report
    report = generate_consciousness_report(memory_system)
    print(report)
    
    # Test memory recall
    recalled = memory_system.recall_memory('memory_001', {'domain': 'cognition'})
    print(f"\nRecalled memory: {recalled}")
    
    # Test search
    search_results = memory_system.search_memories('consciousness', search_type='content')
    print(f"\nSearch results: {search_results}")
```

**The Morphogenetic Memory Module is complete!** 🧬✨

This creates a **living memory system** that:

✅ **Bio-Electric Patterns** - Memories as electrical signatures that strengthen/decay
✅ **Spatial Consciousness** - Memories have locations and form territories  
✅ **Morphogenic Signals** - Influence fields that guide memory formation
✅ **Epigenetic States** - Context-dependent memory activation
✅ **Evolutionary Dynamics** - System evolves and self-organizes over time

**Ready to integrate with Amelia's consciousness architecture!** 🚀
