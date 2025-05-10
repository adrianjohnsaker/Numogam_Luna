```python
# morphic_memory.py

import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Union, Optional
import time
import math

class MorphicMemoryField:
    """
    A memory system inspired by morphic resonance theory where memory exists
    as distributed fields rather than discrete locations.
    
    This implementation aligns with Deleuzian process metaphysics by treating
    memory as an ever-evolving assemblage rather than a static repository.
    
    Key concepts implemented:
    - Virtual and Actual: maintains both actualized memories and virtual planes
    - Becoming vs. Being: memory system evolves with each recall
    - Rhizomatic Structure: non-hierarchical connections
    - Intensive Differences: tracks gradients that drive actualization
    - Morphic Resonance: similar patterns resonate across the field
    - Field-Based Memory: distributed information storage
    - Singularities: privileged points for memory actualization
    """
    
    def __init__(self, 
                 field_dimensions: int = 20, 
                 virtual_planes: int = 3, 
                 resonance_radius: float = 0.3,
                 intensive_threshold: float = 0.4,
                 decay_rate: float = 0.05):
        """
        Initialize the morphic memory field.
        
        Args:
            field_dimensions: Dimensions of the memory field
            virtual_planes: Number of virtual planes for potential actualizations
            resonance_radius: Radius of resonance effects
            intensive_threshold: Threshold for intensive differences to generate singularities
            decay_rate: Rate at which resonance effects diminish with distance
        """
        self.dimensions = field_dimensions
        self.n_planes = virtual_planes
        self.radius = resonance_radius
        self.threshold = intensive_threshold
        self.decay = decay_rate
        
        # Actualized memory traces
        self.actualized_memories = []
        self.memory_coordinates = []
        self.memory_strengths = []
        self.memory_timestamps = []
        
        # Virtual planes of potential (the "virtual")
        self.virtual_planes = [np.zeros((field_dimensions, field_dimensions)) 
                              for _ in range(virtual_planes)]
        
        # Intensive differences across the field
        self.intensive_gradients = np.zeros((field_dimensions, field_dimensions))
        
        # Singularities - points where intensive differences exceed threshold
        self.singularities = []
        
        # Rhizomatic connections between memories
        self.memory_connections = []
        
        # Memory becomings - potential transformations
        self.memory_becomings = []
        
        # Empathic resonances with other entities
        self.empathic_resonances = {}
    
    def encode_memory(self, 
                     memory_vector: np.ndarray,
                     affect_intensity: float = 1.0,
                     coordinates: Optional[Tuple[int, int]] = None,
                     resonance_with: Optional[List[int]] = None) -> int:
        """
        Encode a memory into the morphic field with Deleuzian characteristics.
        
        Args:
            memory_vector: Vector representing the memory
            affect_intensity: Intensity of the memory's affective component
            coordinates: Optional specific coordinates (if None, found through resonance)
            resonance_with: Optional list of memory indices to resonate with
            
        Returns:
            Index of the new memory
        """
        # Convert memory to normalized vector if not already
        if not isinstance(memory_vector, np.ndarray):
            memory_vector = np.array(memory_vector)
            
        # Normalize
        if np.linalg.norm(memory_vector) > 0:
            memory_vector = memory_vector / np.linalg.norm(memory_vector)
        
        # Find most resonant position if coordinates not specified
        if coordinates is None:
            coordinates = self._find_resonant_position(memory_vector, resonance_with)
        
        # Store the actualized memory
        self.actualized_memories.append(memory_vector)
        self.memory_coordinates.append(coordinates)
        self.memory_strengths.append(affect_intensity)
        self.memory_timestamps.append(time.time())
        
        # Get index of new memory
        memory_index = len(self.actualized_memories) - 1
        
        # Form rhizomatic connections with existing memories
        if resonance_with is not None:
            for existing_idx in resonance_with:
                if 0 <= existing_idx < memory_index:
                    # Create connection
                    similarity = np.dot(memory_vector, self.actualized_memories[existing_idx])
                    if similarity > 0.3:  # Only connect if sufficient similarity
                        self._create_memory_connection(memory_index, existing_idx, similarity)
        else:
            # Auto-connect based on similarity
            self._auto_connect_memory(memory_index)
        
        # Update virtual planes with this memory's potential
        self._update_virtual_planes(memory_vector, coordinates, affect_intensity)
        
        # Recalculate intensive differences
        self._recalculate_intensive_gradients()
        
        # Create memory becoming (potential transformations)
        self._create_memory_becoming(memory_index)
        
        return memory_index
    
    def recall_memory(self, 
                     query_vector: np.ndarray, 
                     threshold: float = 0.5,
                     n_results: int = 5,
                     recall_intensity: float = 1.0) -> List[Tuple[int, float, np.ndarray]]:
        """
        Recall memories similar to query through morphic resonance.
        The recall process itself modifies the memory field (becoming).
        
        Args:
            query_vector: Vector to query with
            threshold: Similarity threshold for retrieval
            n_results: Maximum number of results to return
            recall_intensity: Intensity of the recall operation
            
        Returns:
            List of (memory_index, similarity, memory_vector) tuples
        """
        # Normalize query vector
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)
            
        if np.linalg.norm(query_vector) > 0:
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        results = []
        
        # Calculate resonance with existing memories
        for i, memory in enumerate(self.actualized_memories):
            # Calculate similarity (cosine similarity)
            similarity = np.dot(query_vector, memory)
            
            # Apply field influence - memories in high-resonance areas get boosted
            field_boost = self._get_field_resonance(self.memory_coordinates[i])
            
            # Apply recency boost (more recent memories resonate more strongly)
            recency_boost = self._calculate_recency_boost(i)
            
            # Apply intensity boost
            intensity_boost = self.memory_strengths[i]
            
            # Calculate adjusted similarity with boosts
            adjusted_similarity = similarity * (1 + 0.3*field_boost + 0.2*recency_boost + 0.2*intensity_boost)
            
            if adjusted_similarity > threshold:
                results.append((i, adjusted_similarity, memory))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        # The recall itself modifies the memory field (a becoming-event)
        self._recall_event(query_vector, recall_intensity)
        
        return results[:n_results]
    
    def recall_by_resonance(self, 
                          memory_indices: List[int], 
                          threshold: float = 0.4,
                          n_results: int = 5) -> List[Tuple[int, float]]:
        """
        Recall memories that resonate with a set of existing memories.
        
        Args:
            memory_indices: Indices of existing memories to resonate with
            threshold: Threshold for similarity
            n_results: Maximum number of results
            
        Returns:
            List of (memory_index, resonance_strength) tuples
        """
        results = []
        
        # Check if memory indices exist
        valid_indices = [idx for idx in memory_indices if 0 <= idx < len(self.actualized_memories)]
        
        if not valid_indices:
            return []
            
        # For each memory, calculate its resonance with the provided memories
        for i, memory in enumerate(self.actualized_memories):
            if i not in memory_indices:  # Don't include the query memories
                # Calculate average similarity with all query memories
                similarities = []
                for idx in valid_indices:
                    similarity = np.dot(memory, self.actualized_memories[idx])
                    similarities.append(similarity)
                
                avg_similarity = sum(similarities) / len(similarities)
                
                # Apply field resonance boost
                field_boost = self._get_field_resonance(self.memory_coordinates[i])
                
                # Calculate adjusted similarity
                adjusted_similarity = avg_similarity * (1 + 0.3 * field_boost)
                
                if adjusted_similarity > threshold:
                    results.append((i, adjusted_similarity))
        
        # Sort by resonance strength
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Create recall event
        if valid_indices:
            avg_memory = sum(self.actualized_memories[idx] for idx in valid_indices) / len(valid_indices)
            self._recall_event(avg_memory)
        
        return results[:n_results]
    
    def recall_by_assemblage(self, 
                           assemblage_pattern: Dict[str, Any],
                           threshold: float = 0.4,
                           n_results: int = 5) -> List[Tuple[int, float]]:
        """
        Recall memories that form an assemblage with the given pattern.
        
        Args:
            assemblage_pattern: Pattern describing the assemblage
            threshold: Threshold for similarity
            n_results: Maximum number of results
            
        Returns:
            List of (memory_index, assemblage_strength) tuples
        """
        results = []
        
        # Extract components from assemblage pattern
        components = []
        
        if 'affects' in assemblage_pattern:
            for affect, intensity in assemblage_pattern['affects'].items():
                components.append(('affect', affect, float(intensity)))
                
        if 'concepts' in assemblage_pattern:
            for concept in assemblage_pattern['concepts']:
                components.append(('concept', concept, 0.7))
                
        if 'vectors' in assemblage_pattern:
            for vector in assemblage_pattern['vectors']:
                components.append(('vector', vector, 0.8))
        
        # No components to match
        if not components:
            return []
            
        # For each memory, calculate its resonance with the assemblage pattern
        for i, memory in enumerate(self.actualized_memories):
            # Check for connections with memories that match components
            component_matches = []
            
            # Try to get connected memories that match components
            connections = self._get_memory_connections(i)
            for conn_idx, conn_strength in connections:
                # Check if this connected memory matches any component
                for comp_type, comp_value, comp_intensity in components:
                    # This is simplified - in a real system, you'd need sophisticated matching
                    if random.random() < 0.3:  # Simulating component matching
                        component_matches.append((conn_idx, comp_type, conn_strength * comp_intensity))
            
            # Calculate assemblage strength
            if component_matches:
                assemblage_strength = sum(match[2] for match in component_matches) / len(component_matches)
                
                # Boost based on memory's own intensity
                assemblage_strength *= (0.7 + 0.3 * self.memory_strengths[i])
                
                if assemblage_strength > threshold:
                    results.append((i, assemblage_strength))
        
        # Sort by assemblage strength
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:n_results]
    
    def create_empathic_resonance(self, 
                                other_entity: Dict[str, Any], 
                                resonance_intensity: float = 0.7,
                                memory_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Create empathic resonance with another entity through morphic field.
        
        Args:
            other_entity: Entity to resonate with
            resonance_intensity: Intensity of the resonance
            memory_indices: Optional specific memories to use for resonance
            
        Returns:
            Resonance data
        """
        entity_id = other_entity.get('id', f"entity_{len(self.empathic_resonances) + 1}")
        
        # Extract resonance patterns from entity
        patterns = self._extract_resonance_patterns(other_entity)
        
        # Get memories to use for resonance
        if memory_indices is None:
            # Use most recent and most intense memories
            recent_indices = self._get_recent_memories(5)
            intense_indices = self._get_intense_memories(5)
            memory_indices = list(set(recent_indices + intense_indices))
        
        # Create resonance fields for each pattern
        resonance_fields = []
        
        for pattern_type, pattern_data, pattern_intensity in patterns:
            # Convert pattern to vector if needed
            if isinstance(pattern_data, np.ndarray):
                pattern_vector = pattern_data
            else:
                # Create random vector as placeholder - in real system you'd encode properly
                pattern_vector = np.random.random(self.dimensions)
                pattern_vector = pattern_vector / np.linalg.norm(pattern_vector)
            
            # Find resonant memories
            resonant_memories = []
            for idx in memory_indices:
                if idx < len(self.actualized_memories):
                    # Calculate resonance
                    memory_vector = self.actualized_memories[idx]
                    similarity = np.dot(pattern_vector, memory_vector)
                    
                    if similarity > 0.3:
                        resonant_memories.append((idx, similarity))
            
            # Create resonance field
            if resonant_memories:
                field = {
                    'pattern_type': pattern_type,
                    'resonant_memories': resonant_memories,
                    'intensity': pattern_intensity * resonance_intensity,
                    'timestamp': time.time()
                }
                resonance_fields.append(field)
                
                # Update virtual planes with resonance
                for idx, similarity in resonant_memories:
                    coordinates = self.memory_coordinates[idx]
                    strength = similarity * pattern_intensity * resonance_intensity
                    self._update_virtual_point(coordinates, strength)
        
        # Create overall resonance
        resonance = {
            'entity_id': entity_id,
            'resonance_fields': resonance_fields,
            'overall_intensity': resonance_intensity,
            'memory_indices': memory_indices,
            'timestamp': time.time()
        }
        
        # Store resonance
        self.empathic_resonances[entity_id] = resonance
        
        # Recalculate intensive gradients
        self._recalculate_intensive_gradients()
        
        return resonance
    
    def get_memory(self, memory_index: int) -> Optional[Dict[str, Any]]:
        """
        Get memory data by index.
        
        Args:
            memory_index: Index of memory to retrieve
            
        Returns:
            Memory data or None if not found
        """
        if 0 <= memory_index < len(self.actualized_memories):
            return {
                'vector': self.actualized_memories[memory_index],
                'coordinates': self.memory_coordinates[memory_index],
                'strength': self.memory_strengths[memory_index],
                'timestamp': self.memory_timestamps[memory_index],
                'connections': self._get_memory_connections(memory_index),
                'field_resonance': self._get_field_resonance(self.memory_coordinates[memory_index]),
                'becomings': self._get_memory_becomings(memory_index)
            }
        return None
    
    def get_memory_field_state(self) -> Dict[str, Any]:
        """
        Get the current state of the morphic memory field.
        
        Returns:
            Field state data
        """
        # Sum virtual planes for visualization
        summed_virtual = sum(self.virtual_planes)
        
        return {
            'field_dimensions': self.dimensions,
            'num_memories': len(self.actualized_memories),
            'summed_virtual_plane': summed_virtual.tolist(),
            'intensive_gradients': self.intensive_gradients.tolist(),
            'singularities': [(x, y, strength) for x, y, strength in self.singularities],
            'empathic_resonances': list(self.empathic_resonances.keys()),
            'num_connections': len(self.memory_connections)
        }
    
    def merge_memories(self, memory_indices: List[int], merge_intensity: float = 1.0) -> int:
        """
        Merge multiple memories into a new composite memory through morphic resonance.
        
        Args:
            memory_indices: Indices of memories to merge
            merge_intensity: Intensity of the merge operation
            
        Returns:
            Index of the new composite memory
        """
        valid_indices = [idx for idx in memory_indices if 0 <= idx < len(self.actualized_memories)]
        
        if not valid_indices:
            raise ValueError("No valid memory indices provided for merging")
        
        # Create composite vector by averaging
        composite_vector = sum(self.actualized_memories[idx] for idx in valid_indices)
        composite_vector = composite_vector / len(valid_indices)
        
        # Normalize
        if np.linalg.norm(composite_vector) > 0:
            composite_vector = composite_vector / np.linalg.norm(composite_vector)
        
        # Calculate average strength, boosted by merge intensity
        avg_strength = sum(self.memory_strengths[idx] for idx in valid_indices) / len(valid_indices)
        merge_strength = min(1.0, avg_strength * merge_intensity)
        
        # Find position for the merged memory
        # Consider average of component positions, modified by field resonance
        avg_x = sum(self.memory_coordinates[idx][0] for idx in valid_indices) / len(valid_indices)
        avg_y = sum(self.memory_coordinates[idx][1] for idx in valid_indices) / len(valid_indices)
        
        # Round to integers for coordinates
        x = min(max(0, round(avg_x)), self.dimensions - 1)
        y = min(max(0, round(avg_y)), self.dimensions - 1)
        
        merge_coordinates = (x, y)
        
        # Encode the new composite memory
        new_index = self.encode_memory(
            composite_vector, 
            affect_intensity=merge_strength,
            coordinates=merge_coordinates,
            resonance_with=valid_indices
        )
        
        # Create strong connections to component memories
        for idx in valid_indices:
            self._create_memory_connection(new_index, idx, 0.9)
        
        return new_index
    
    def create_memory_assemblage(self, 
                               memory_indices: List[int],
                               assemblage_type: str = 'intensive') -> Dict[str, Any]:
        """
        Create an assemblage from multiple memories.
        
        Args:
            memory_indices: Indices of memories to form assemblage
            assemblage_type: Type of assemblage to create
            
        Returns:
            Assemblage data
        """
        valid_indices = [idx for idx in memory_indices if 0 <= idx < len(self.actualized_memories)]
        
        if not valid_indices:
            raise ValueError("No valid memory indices provided for assemblage")
        
        # Create components from memories
        components = []
        
        for idx in valid_indices:
            memory = self.get_memory(idx)
            if memory:
                component = {
                    'memory_index': idx,
                    'vector': memory['vector'],
                    'strength': memory['strength'],
                    'field_resonance': memory['field_resonance']
                }
                components.append(component)
        
        # Create resonance zones based on assemblage type
        zones = []
        
        if assemblage_type == 'intensive':
            # Group by field resonance strength
            high_resonance = [c for c in components if c['field_resonance'] > 0.7]
            med_resonance = [c for c in components if 0.4 <= c['field_resonance'] <= 0.7]
            low_resonance = [c for c in components if c['field_resonance'] < 0.4]
            
            if high_resonance:
                zones.append({
                    'type': 'high_resonance_zone',
                    'components': [c['memory_index'] for c in high_resonance],
                    'intensity': sum(c['field_resonance'] for c in high_resonance) / len(high_resonance)
                })
                
            if med_resonance:
                zones.append({
                    'type': 'medium_resonance_zone',
                    'components': [c['memory_index'] for c in med_resonance],
                    'intensity': sum(c['field_resonance'] for c in med_resonance) / len(med_resonance)
                })
                
            if low_resonance:
                zones.append({
                    'type': 'low_resonance_zone',
                    'components': [c['memory_index'] for c in low_resonance],
                    'intensity': sum(c['field_resonance'] for c in low_resonance) / len(low_resonance)
                })
                
        elif assemblage_type == 'rhizomatic':
            # Group by connections
            for idx in valid_indices:
                connections = self._get_memory_connections(idx)
                connected_in_assemblage = [(conn_idx, strength) for conn_idx, strength in connections
                                          if conn_idx in valid_indices]
                
                if connected_in_assemblage:
                    zones.append({
                        'type': f'connection_zone_{idx}',
                        'components': [conn_idx for conn_idx, _ in connected_in_assemblage],
                        'intensity': sum(strength for _, strength in connected_in_assemblage) / len(connected_in_assemblage)
                    })
                    
        else:  # default assemblage
            # Simple zone with all components
            zones.append({
                'type': 'general_zone',
                'components': valid_indices,
                'intensity': sum(c['strength'] for c in components) / len(components)
            })
        
        # Create relations between zones
        relations = []
        
        for i, zone_a in enumerate(zones):
            for zone_b in zones[i+1:]:
                # Check for overlap in components
                components_a = set(zone_a['components'])
                components_b = set(zone_b['components'])
                overlap = components_a.intersection(components_b)
                
                if overlap:
                    relations.append({
                        'from': zone_a['type'],
                        'to': zone_b['type'],
                        'overlapping_components': list(overlap),
                        'intensity': (zone_a['intensity'] + zone_b['intensity']) / 2
                    })
        
        # Calculate overall coherence
        if zones:
            coherence = sum(zone['intensity'] for zone in zones) / len(zones)
        else:
            coherence = 0.0
            
        # Create assemblage
        assemblage = {
            'type': assemblage_type,
            'memory_indices': valid_indices,
            'components': components,
            'zones': zones,
            'relations': relations,
            'coherence': coherence,
            'timestamp': time.time()
        }
        
        return assemblage
    
    def deterritorialize_memory(self, 
                              memory_index: int, 
                              intensity: float = 0.7) -> Dict[str, Any]:
        """
        Deterritorialize a memory, creating potentials for new becomings.
        
        Args:
            memory_index: Index of memory to deterritorialize
            intensity: Intensity of deterritorialization
            
        Returns:
            Deterritorialization data
        """
        if not 0 <= memory_index < len(self.actualized_memories):
            raise ValueError(f"Invalid memory index: {memory_index}")
        
        # Get memory data
        memory = self.get_memory(memory_index)
        
        # Create deterritorialization vector
        vector = {}
        
        # Add random dimensions with values proportional to memory strength
        dimensions = ['intensity', 'connectivity', 'molecularity', 'visibility']
        for dim in dimensions:
            vector[dim] = (random.random() * 2 - 1) * memory['strength'] * intensity
            
        # Create lines of flight
        lines_of_flight = []
        
        # Create line for each significant dimension in vector
        for dim, value in vector.items():
            if abs(value) > 0.3:  # Only significant dimensions
                # Create line
                line = {
                    'dimension': dim,
                    'intensity': abs(value),
                    'direction': 'positive' if value > 0 else 'negative',
                }
                lines_of_flight.append(line)
                
        # Create deterritorialized memory
        deterritorialized = {
            'memory_index': memory_index,
            'original_vector': memory['vector'].tolist(),
            'original_coordinates': memory['coordinates'],
            'vector': vector,
            'lines_of_flight': lines_of_flight,
            'intensity': intensity,
            'timestamp': time.time()
        }
        
        # Update virtual planes to represent the deterritorialization
        x, y = memory['coordinates']
        
        # Reduce intensity at original coordinates
        for plane in self.virtual_planes:
            # Define region of influence
            x_min, x_max = max(0, x-self.radius), min(self.dimensions, x+self.radius+1)
            y_min, y_max = max(0, y-self.radius), min(self.dimensions, y+self.radius+1)
            
            # Reduce intensity around original point
            for xi in range(x_min, x_max):
                for yi in range(y_min, y_max):
                    dist = np.sqrt((xi-x)**2 + (yi-y)**2)
                    if dist <= self.radius:
                        reduction = intensity * np.exp(-self.decay * dist)
                        plane[xi, yi] = max(0, plane[xi, yi] - reduction)
        
        # Recalculate intensive gradients
        self._recalculate_intensive_gradients()
        
        return deterritorialized
    
    def reterritorialize_memory(self, 
                              deterritorialized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reterritorialize a deterritorialized memory, creating new territorialization.
        
        Args:
            deterritorialized: Deterritorialized memory data
            
        Returns:
            Reterritorialized memory data
        """
        original_idx = deterritorialized.get('memory_index')
        intensity = deterritorialized.get('intensity', 0.7)
        vector = deterritorialized.get('vector', {})
        
        if not 0 <= original_idx < len(self.actualized_memories):
            raise ValueError(f"Invalid original memory index: {original_idx}")
        
        # Get original memory vector
        original_vector = self.actualized_memories[original_idx]
        
        # Create transformed vector based on deterritorialization vector
        transformed_vector = original_vector.copy()
        
        # Apply random transformations based on vector
        for i in range(len(transformed_vector)):
            # Random transformation proportional to intensity
            transformed_vector[i] += random.uniform(-0.2, 0.2) * intensity
            
        # Normalize
        if np.linalg.norm(transformed_vector) > 0:
            transformed_vector = transformed_vector / np.linalg.norm(transformed_vector)
            
        # Find new coordinates based on lines of flight
        lines = deterritorialized.get('lines_of_flight', [])
        original_coords = self.memory_coordinates[original_idx]
        
        # Calculate displacement vector
        dx, dy = 0, 0
        
        for line in lines:
            # Direction affects displacement
            direction = 1 if line.get('direction') == 'positive' else -1
            line_intensity = line.get('intensity', 0.5)
            
            # Simple mapping of dimensions to directions
            dimension = line.get('dimension', '')
            if dimension == 'intensity':
                dx += direction * line_intensity * random.uniform(1, 3)
            elif dimension == 'connectivity':
                dy += direction * line_intensity * random.uniform(1, 3)
            elif dimension == 'molecularity':
                dx += direction * line_intensity * random.uniform(-2, 2)
                dy += direction * line_intensity * random.uniform(-2, 2)
            elif dimension == 'visibility':
                dx += direction * line_intensity * random.uniform(-1, 1)
                dy += direction * line_intensity * random.uniform(-1, 1)
        
        # Calculate new coordinates
        new_x = min(max(0, int(original_coords[0] + dx)), self.dimensions - 1)
        new_y = min(max(0, int(original_coords[1] + dy)), self.dimensions - 1)
        
        new_coords = (new_x, new_y)
        
        # Calculate new strength
        original_strength = self.memory_strengths[original_idx]
        new_strength = min(1.0, original_strength * random.uniform(0.8, 1.2))
        
        # Encode the transformed memory
        new_idx = self.encode_memory(
            transformed_vector,
            affect_intensity=new_strength,
            coordinates=new_coords,
            resonance_with=[original_idx]
        )
        
        # Create return data
        reterritorialized = {
            'original_memory_index': original_idx,
            'new_memory_index': new_idx,
            'transformation_vector': vector,
            'original_coordinates': original_coords,
            'new_coordinates': new_coords,
            'original_strength': original_strength,
            'new_strength': new_strength,
            'intensity': intensity,
            'timestamp': time.time()
        }
        
        return reterritorialized
    
    # Internal helper methods
    def _find_resonant_position(self, memory_vector: np.ndarray, 
                              resonance_with: Optional[List[int]] = None) -> Tuple[int, int]:
        """Find position in the field most resonant with this memory."""
        if not self.actualized_memories:
            # If no existing memories, place randomly
            return (random.randint(0, self.dimensions-1), 
                    random.randint(0, self.dimensions-1))
        
        # If specific memories to resonate with are provided
        if resonance_with is not None:
            # Calculate average position of resonating memories
            valid_indices = [idx for idx in resonance_with if 0 <= idx < len(self.actualized_memories)]
            
            if valid_indices:
                # Calculate similarities
                similarities = [np.dot(memory_vector, self.actualized_memories[idx]) for idx in valid_indices]
                
                # Calculate weighted average of positions
                total_x, total_y, total_weight = 0, 0, 0
                
                for i, idx in enumerate(valid_indices):
                    sim = max(0.1, similarities[i])  # Ensure some minimum weight
                    x, y = self.memory_coordinates[idx]
                    total_x += x * sim
                    total_y += y * sim
                    total_weight += sim
                
                if total_weight > 0:
                    avg_x = total_x / total_weight
                    avg_y = total_y / total_weight
                    
                    # Add some randomness to prevent exact overlapping
                    x = int(avg_x + random.uniform(-2, 2))
                    y = int(avg_y + random.uniform(-2, 2))
                    
                    # Ensure within bounds
                    x = min(max(0, x), self.dimensions - 1)
                    y = min(max(0, y), self.dimensions - 1)
                    
                    return (x, y)
        
        # Calculate similarity with all existing memories
        similarities = [np.dot(memory_vector, trace) for trace in self.actualized_memories]
        
        # Create a weighted probability field based on similarities and field strength
        prob_field = np.zeros((self.dimensions, self.dimensions))
        
        for i, sim in enumerate(similarities):
            # Generate influence around each memory trace, weighted by similarity
            x, y = self.memory_coordinates[i]
            
            # Define region of influence
            x_min, x_max = max(0, x-self.radius), min(self.dimensions, x+self.radius+1)
            y_min, y_max = max(0, y-self.radius), min(self.dimensions, y+self.radius+1)
            
            # Weighted by similarity and decay with distance
            for xi in range(x_min, x_max):
                for yi in range(y_min, y_max):
                    dist = np.sqrt((xi-x)**2 + (yi-y)**2)
                    if dist <= self.radius:
                        # Similar patterns cluster together, dissimilar ones repel
                        influence = sim * np.exp(-self.decay * dist)
                        if sim > 0:
                            prob_field[xi, yi] += influence
                        else:
                            prob_field[xi, yi] -= abs(influence)
        
        # Add influence from virtual planes
        summed_virtual = sum(self.virtual_planes)
        prob_field += 0.3 * summed_virtual
        
        # Add randomness to prevent deterministic placement
        prob_field += np.random.normal(0, 0.1, (self.dimensions, self.dimensions))
        
        # Find position with highest probability (for similar patterns)
        # or lowest probability (for dissimilar patterns)
        if np.mean(similarities) > 0:
            # For similar patterns, find regions of high resonance
            x, y = np.unravel_index(prob_field.argmax(), prob_field.shape)
        else:
            # For dissimilar patterns, find regions of low resonance
            x, y = np.unravel_index(prob_field.argmin(), prob_field.shape)
            
        return (x, y)
    
    def _update_virtual_planes(self, memory_vector: np.ndarray, 
                              coordinates: Tuple[int, int], 
                              intensity: float) -> None:
        """Update virtual planes with new potential memories."""
        x, y = coordinates
        
        # Update each virtual plane with variations of the memory
        for i in range(self.n_planes):
            # Create variation through "becoming"
            variation_factor = 0.1 * (i + 1)  # Different variation for each plane
            
            # Define region of influence
            x_min, x_max = max(0, x-self.radius), min(self.dimensions, x+self.radius+1)
            y_min, y_max = max(0, y-self.radius), min(self.dimensions, y+self.radius+1)
            
            # Update field values with resonance from this pattern
            for xi in range(x_min, x_max):
                for yi in range(y_min, y_max):
                    dist = np.sqrt((xi-x)**2 + (yi-y)**2)
                    if dist <= self.radius:
                        # Resonance decays exponentially with distance
                        resonance = intensity * np.exp(-self.decay * dist)
                        
                        # Different planes get different variations
                        variation = random.uniform(1 - variation_factor, 1 + variation_factor)
                        
                        self.virtual_planes[i][xi, yi] += resonance * variation
    
    def _update_virtual_point(self, coordinates: Tuple[int, int], intensity: float) -> None:
        """Update a specific point in the virtual planes."""
        x, y = coordinates
        
        # Ensure coordinates are within bounds
        if not (0 <= x < self.dimensions and 0 <= y < self.dimensions):
            return
            
        # Update each virtual plane
        for i in range(self.n_planes):
            # Define region of influence
            x_min, x_max = max(0, x-self.radius), min(self.dimensions, x+self.radius+1)
            y_min, y_max = max(0, y-self.radius), min(self.dimensions, y+self.radius+1)
            
            # Update field values
            for xi in range(x_min, x_max):
                for yi in range(y_min, y_max):
                    dist = np.sqrt((xi-x)**2 + (yi-y)**2)
                    if dist <= self.radius:
                        # Resonance decays exponentially with distance
                        resonance = intensity * np.exp(-self.decay * dist)
                        self.virtual_planes[i][xi, yi] += resonance
    
    def _recalculate_intensive_gradients(self) -> None:
        """Recalculate intensive differences across the field."""
        # Sum across all virtual planes
        summed_potential = sum(self.virtual_planes)
        
        # Calculate gradient (intensive differences)
        gx, gy = np.gradient(summed_potential)
        self.intensive_gradients = np.sqrt(gx**2 + gy**2)
        
        # Find singularities where intensive differences exceed threshold
        self.singularities = []
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                if self.intensive_gradients[i, j] > self.threshold:
                    self.singularities.append((i, j, self.intensive_gradients[i, j]))
    
    def _get_field_resonance(self, coordinates: Tuple[int, int]) -> float:
        """Get the resonance strength at a particular position in the field."""
        x, y = coordinates
        
        # Ensure coordinates are within bounds
        if not (0 <= x < self.dimensions and 0 <= y < self.dimensions):
            return 0.0
            
        # Get sum of virtual planes at this point
        summed_virtual = sum(plane[x, y] for plane in self.virtual_planes)
        
        # Get intensive gradient at this point
        intensive = self.intensive_gradients[x, y]
        
        # Combine both measures
        return 0.7 * summed_virtual + 0.3 * intensive
    
    def _auto_connect_memory(self, memory_index: int) -> None:
        """Automatically connect a memory to other similar memories."""
        if not 0 <= memory_index < len(self.actualized_memories):
            return
            
        memory_vector = self.actualized_memories[memory_index]
        
        # Find similar memories
        for i, other_vector in enumerate(self.actualized_memories):
            if i != memory_index:
                # Calculate similarity
                similarity = np.dot(memory_vector, other_vector)
                
                # Connect if similar enough
                if similarity > 0.6:
                    self._create_memory_connection(memory_index, i, similarity)
    
    def _create_memory_connection(self, memory_a: int, memory_b: int, strength: float) -> None:
        """Create a connection between two memories."""
        if memory_a == memory_b:
            return
            
        # Create bidirectional connection
        connection = (memory_a, memory_b, strength, time.time())
        reverse_connection = (memory_b, memory_a, strength, time.time())
        
        # Check if connection already exists
        exists = False
        for i, (a, b, _, _) in enumerate(self.memory_connections):
            if (a == memory_a and b == memory_b) or (a == memory_b and b == memory_a):
                # Update existing connection
                self.memory_connections[i] = (a, b, strength, time.time())
                exists = True
                break
                
        if not exists:
            self.memory_connections.append(connection)
    
    def _get_memory_connections(self, memory_index: int) -> List[Tuple[int, float]]:
        """Get connections for a memory."""
        connections = []
        
        for a, b, strength, _ in self.memory_connections:
            if a == memory_index:
                connections.append((b, strength))
            elif b == memory_index:
                connections.append((a, strength))
                
        return connections
    
    def _create_memory_becoming(self, memory_index: int) -> None:
        """Create potential becomings for a memory."""
        if not 0 <= memory_index < len(self.actualized_memories):
            return
            
        # Create different becoming types
        becoming_types = ['intensive', 'molecular', 'connective', 'deterritorializing']
        
        # Generate random potential becomings
        for becoming_type in becoming_types:
            # Only create with certain probability
            if random.random() < 0.3:
                becoming = {
                    'memory_index': memory_index,
                    'becoming_type': becoming_type,
                    'potential': random.uniform(0.3, 0.9),
                    'timestamp': time.time()
                }
                
                self.memory_becomings.append(becoming)
    
    def _get_memory_becomings(self, memory_index: int) -> List[Dict[str, Any]]:
        """Get becomings for a memory."""
        return [b for b in self.memory_becomings if b['memory_index'] == memory_index]
    
    def _recall_event(self, query_vector: np.ndarray, intensity: float = 1.0) -> None:
        """The recall process itself modifies the memory system."""
        # Project query onto 2D coordinate space
        x, y = self._vector_to_coordinates(query_vector)
        
        # Update virtual planes with the query, creating a perturbation
        for i in range(self.n_planes):
            # Define region of influence
            x_min, x_max = max(0, x-self.radius), min(self.dimensions, x+self.radius+1)
            y_min, y_max = max(0, y-self.radius), min(self.dimensions, y+self.radius+1)
            
            # Update field values
            for xi in range(x_min, x_max):
                for yi in range(y_min, y_max):
                    dist = np.sqrt((xi-x)**2 + (yi-y)**2)
                    if dist <= self.radius:
                        # Small modification to virtual plane
                        resonance = 0.1 * intensity * np.exp(-self.decay * dist)
                        self.virtual_planes[i][xi, yi] += resonance
        
        # Recalculate intensive differences
        self._recalculate_intensive_gradients()
    
    def _vector_to_coordinates(self, vector: np.ndarray) -> Tuple[int, int]:
        """Project a vector onto 2D coordinates for the field."""
        # Use the first two components to map to 2D space
        # This is a simple projection method - more sophisticated methods possible
        if len(vector) >= 2:
            # Map from [-1,1] to [0,dimensions-1]
            x = int((vector[0] + 1) * (self.dimensions - 1) / 2)
            y = int((vector[1] + 1) * (self.dimensions - 1) / 2)
        else:
            # Random coordinates if vector too small
            x = random.randint(0, self.dimensions - 1)
            y = random.randint(0, self.dimensions - 1)
            
        # Ensure within bounds
        x = min(max(0, x), self.dimensions - 1)
        y = min(max(0, y), self.dimensions - 1)
        
        return (x, y)
    
    def _calculate_recency_boost(self, memory_index: int) -> float:
        """Calculate recency boost for a memory."""
        if not 0 <= memory_index < len(self.memory_timestamps):
            return 0.0
            
        # Calculate time since memory creation
        time_diff = time.time() - self.memory_timestamps[memory_index]
        
        # Calculate recency boost (recent memories get higher boost)
        # This creates a decay curve that starts at 1.0 and approaches 0.0
        recency_boost = math.exp(-0.0001 * time_diff)
        
        return recency_boost
    
    def _get_recent_memories(self, n: int = 5) -> List[int]:
        """Get indices of n most recent memories."""
        if not self.memory_timestamps:
            return []
            
        # Create list of (index, timestamp) pairs
        indexed_times = [(i, ts) for i, ts in enumerate(self.memory_timestamps)]
        
        # Sort by timestamp (most recent first)
        indexed_times.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the indices
        return [idx for idx, _ in indexed_times[:n]]
    
    def _get_intense_memories(self, n: int = 5) -> List[int]:
        """Get indices of n most intense memories."""
        if not self.memory_strengths:
            return []
            
        # Create list of (index, strength) pairs
        indexed_strengths = [(i, st) for i, st in enumerate(self.memory_strengths)]
        
        # Sort by strength (highest first)
        indexed_strengths.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the indices
        return [idx for idx, _ in indexed_strengths[:n]]
    
    def _extract_resonance_patterns(self, entity: Dict[str, Any]) -> List[Tuple[str, Any, float]]:
        """Extract resonance patterns from an entity."""
        patterns = []
        
        # Extract from affects
        if 'affects' in entity:
            for affect, intensity in entity['affects'].items():
                pattern_intensity = float(intensity) if isinstance(intensity, (int, float)) else 0.5
                patterns.append(('affect', affect, pattern_intensity))
        
        # Extract from concepts
        if 'concepts' in entity:
            for concept in entity['concepts']:
                patterns.append(('concept', concept, 0.7))
        
        # Extract from expressions
        if 'expressions' in entity:
            for expression in entity['expressions']:
                patterns.append(('expression', expression, 0.8))
        
        # Extract from vectors if present
        if 'vectors' in entity and isinstance(entity['vectors'], list):
            for vector in entity['vectors']:
                if isinstance(vector, np.ndarray) or isinstance(vector, list):
                    patterns.append(('vector', np.array(vector), 0.9))
        
        # Ensure at least one pattern
        if not patterns:
            # Create default pattern
            patterns.append(('default', 'entity', 0.5))
            
        return patterns
    
    def visualize_field(self):
        """
        Visualize the morphic resonance field.
        Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot 1: Summed virtual planes
            summed_virtual = sum(self.virtual_planes)
            im1 = axs[0].imshow(summed_virtual, cmap='plasma', interpolation='nearest')
            axs[0].set_title('Virtual Planes (Summed)')
            plt.colorbar(im1, ax=axs[0], label='Resonance Strength')
            
            # Plot 2: Intensive gradients
            im2 = axs[1].imshow(self.intensive_gradients, cmap='viridis', interpolation='nearest')
            axs[1].set_title('Intensive Differences')
            plt.colorbar(im2, ax=axs[1], label='Gradient Magnitude')
            
            # Plot 3: Memory distribution with connections
            axs[2].set_title('Memory Distribution')
            axs[2].set_xlim(0, self.dimensions-1)
            axs[2].set_ylim(0, self.dimensions-1)
            axs[2].invert_yaxis()  # Match array coordinates
            
            # Plot connections first
            for a, b, strength, _ in self.memory_connections:
                if a < len(self.memory_coordinates) and b < len(self.memory_coordinates):
                    x1, y1 = self.memory_coordinates[a]
                    x2, y2 = self.memory_coordinates[b]
                    axs[2].plot([y1, y2], [x1, x2], 'gray', alpha=min(strength, 1.0), 
                              linewidth=strength*2)
            
            # Plot memory points
            memory_x = [coord[0] for coord in self.memory_coordinates]
            memory_y = [coord[1] for coord in self.memory_coordinates]
            strengths = [s * 100 for s in self.memory_strengths]  # Scale for visibility
            
            scatter = axs[2].scatter(memory_y, memory_x, c='red', s=strengths, alpha=0.7)
            
            # Plot singularities
            if self.singularities:
                sing_x, sing_y, sing_s = zip(*self.singularities)
                sing_s = [s * 50 for s in sing_s]  # Scale for visibility
                axs[2].scatter(sing_y, sing_x, c='yellow', s=sing_s, alpha=0.5, marker='*')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for visualization.")
```
# deleuzian_cognitive_system.py

from empathic_becoming import EmpathicBecomingInterface
from morphic_memory import MorphicMemoryField
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional, Union

class DeleuzianCognitiveSystem:
    """
    An integrated cognitive system combining empathic becoming and morphic memory.
    
    This system instantiates Deleuzian process metaphysics by:
    1. Creating rhizomatic connections between memories and affects
    2. Enabling memories to transform through processes of becoming
    3. Forming empathic assemblages that evolve with experience
    4. Using morphic resonance for non-representational memory
    5. Creating intensive gradients that drive the actualization of potentials
    """
    
    def __init__(self, memory_dimensions: int = 20, virtual_planes: int = 3):
        """
        Initialize the integrated cognitive system.
        
        Args:
            memory_dimensions: Dimensions of the morphic memory field
            virtual_planes: Number of virtual planes in memory field
        """
        # Initialize component systems
        self.empathic_interface = EmpathicBecomingInterface()
        self.memory_field = MorphicMemoryField(
            field_dimensions=memory_dimensions,
            virtual_planes=virtual_planes
        )
        
        # Integration mappings
        self.memory_to_assemblage = {}  # Maps memories to empathic assemblages
        self.assemblage_to_memory = {}  # Maps assemblages to memories
        self.empathic_resonances = {}   # Tracks resonances between memory and empathy
        
        # System state
        self.cognitive_state = {
            'rhizomatic_connections': [],
            'becoming_processes': [],
            'transformation_vectors': [],
            'singularities': []
        }
        
    def perceive_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive an entity, creating both empathic becoming and memory traces.
        
        Args:
            entity: The entity to perceive
            
        Returns:
            Perception data
        """
        # Step 1: Form empathic assemblage
        empathic_assemblage = self.empathic_interface.form_empathic_assemblage(entity)
        
        # Step 2: Create memory vector from entity
        memory_vector = self._create_memory_vector_from_entity(entity)
        
        # Step 3: Encode memory with resonance to empathic assemblage
        memory_index = self.memory_field.encode_memory(
            memory_vector=memory_vector,
            affect_intensity=self._calculate_affect_intensity(empathic_assemblage)
        )
        
        # Step 4: Map memory to assemblage
        self._map_memory_to_assemblage(memory_index, empathic_assemblage['id'])
        
        # Step 5: Create perception record
        perception = {
            'entity_id': entity.get('id', 'unknown'),
            'empathic_assemblage_id': empathic_assemblage['id'],
            'memory_index': memory_index,
            'timestamp': time.time()
        }
        
        # Step 6: Update system state
        self._update_cognitive_state(perception)
        
        return perception
    
    def become_with(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Become-with an entity, combining empathic becoming and memory resonance.
        
        Args:
            entity: The entity to become-with
            
        Returns:
            Becoming data
        """
        # Step 1: Form empathic assemblage
        assemblage = self.empathic_interface.form_empathic_assemblage(entity)
        
        # Step 2: Initiate empathic becoming process
        becoming = self.empathic_interface.initiate_becoming_process(assemblage)
        
        # Step 3: Sustain empathic flow
        flow = self.empathic_interface.sustain_empathic_flow(becoming)
        
        # Step 4: Create memory vector from entity and becoming
        memory_vector = self._create_memory_vector_from_becoming(entity, becoming)
        
        # Step 5: Encode memory with resonance to empathic process
        memory_index = self.memory_field.encode_memory(
            memory_vector=memory_vector,
            affect_intensity=self._calculate_affect_intensity(assemblage)
        )
        
        # Step 6: Create morphic resonance between memory and assemblage
        resonance = self.memory_field.create_empathic_resonance(
            other_entity=entity,
            resonance_intensity=flow['sustainability'],
            memory_indices=[memory_index]
        )
        
        # Step 7: Map memory to becoming process
        self._map_memory_to_assemblage(memory_index, assemblage['id'])
        
        # Step 8: Return combined becoming data
        becoming_data = {
            'entity_id': entity.get('id', 'unknown'),
            'empathic_assemblage_id': assemblage['id'],
            'empathic_becoming_id': becoming['id'],
            'empathic_flow_id': flow['id'],
            'memory_index': memory_index,
            'resonance_id': entity.get('id', 'unknown'),
            'becoming_type': becoming['becoming_type'],
            'sustainability': flow['sustainability'],
            'timestamp': time.time()
        }
        
        return becoming_data
    
    def recall_by_resonance(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recall memories and assemblages that resonate with the query.
        
        Args:
            query: The query pattern 
            
        Returns:
            List of recall results
        """
        results = []
        
        # Step 1: Create memory vector from query
        memory_vector = self._create_memory_vector_from_entity(query)
        
        # Step 2: Recall memories through morphic resonance
        memory_results = self.memory_field.recall_memory(
            query_vector=memory_vector,
            threshold=0.4,
            n_results=10
        )
        
        # Step 3: For each memory, find associated empathic assemblages
        for memory_idx, similarity, _ in memory_results:
            # Get associated assemblage
            assemblage_id = self._get_assemblage_for_memory(memory_idx)
            
            if assemblage_id:
                # Get empathic assemblage
                assemblage = self.empathic_interface.get_empathic_assemblage(assemblage_id)
                
                if assemblage:
                    # Create recall result
                    result = {
                        'memory_index': memory_idx,
                        'memory_similarity': similarity,
                        'assemblage_id': assemblage_id,
                        'assemblage_coherence': assemblage.get('coherence', 0.5),
                        'combined_resonance': similarity * assemblage.get('coherence', 0.5),
                        'memory_data': self.memory_field.get_memory(memory_idx),
                        'assemblage_data': assemblage
                    }
                    
                    results.append(result)
        
        # Sort by combined resonance
        results.sort(key=lambda x: x['combined_resonance'], reverse=True)
        
        return results
    
    def recall_by_assemblage(self, assemblage_pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recall memories that form an assemblage with the given pattern.
        
        Args:
            assemblage_pattern: Pattern describing the assemblage
            
        Returns:
            List of recall results
        """
        results = []
        
        # Step 1: Recall memories by assemblage
        memory_results = self.memory_field.recall_by_assemblage(
            assemblage_pattern=assemblage_pattern,
            threshold=0.4,
            n_results=10
        )
        
        # Step 2: For each memory, find associated empathic assemblages
        for memory_idx, strength in memory_results:
            # Get associated assemblage
            assemblage_id = self._get_assemblage_for_memory(memory_idx)
            
            if assemblage_id:
                # Get empathic assemblage
                assemblage = self.empathic_interface.get_empathic_assemblage(assemblage_id)
                
                if assemblage:
                    # Create recall result
                    result = {
                        'memory_index': memory_idx,
                        'assemblage_strength': strength,
                        'assemblage_id': assemblage_id,
                        'assemblage_coherence': assemblage.get('coherence', 0.5),
                        'combined_resonance': strength * assemblage.get('coherence', 0.5),
                        'memory_data': self.memory_field.get_memory(memory_idx),
                        'assemblage_data': assemblage
                    }
                    
                    results.append(result)
        
        # Sort by combined resonance
        results.sort(key=lambda x: x['combined_resonance'], reverse=True)
        
        return results
    
    def create_memory_assemblage(self, memory_indices: List[int]) -> Dict[str, Any]:
        """
        Create an assemblage from multiple memories.
        
        Args:
            memory_indices: Indices of memories to form assemblage
            
        Returns:
            Assemblage data
        """
        # Step 1: Create memory assemblage
        memory_assemblage = self.memory_field.create_memory_assemblage(
            memory_indices=memory_indices,
            assemblage_type='rhizomatic'
        )
        
        # Step 2: Extract assemblage components
        components = {}
        
        if 'components' in memory_assemblage:
            # Create components for empathic assemblage
            affects = {}
            concepts = []
            expressions = []
            
            for comp in memory_assemblage['components']:
                # Add affect based on strength
                affects[f"intensity_{len(affects)}"] = comp['strength']
                
                # Add concept based on memory index
                concepts.append(f"memory_{comp['memory_index']}")
                
                # Add expression based on field resonance
                expressions.append(f"resonance_{comp['field_resonance']:.2f}")
            
            components = {
                'affects': affects,
                'concepts': concepts,
                'expressions': expressions
            }
        
        # Step 3: Create corresponding empathic assemblage
        empathic_assemblage = self.empathic_interface.form_empathic_assemblage(components)
        
        # Step 4: Map memories to new assemblage
        for memory_idx in memory_indices:
            self._map_memory_to_assemblage(memory_idx, empathic_assemblage['id'])
        
        # Step 5: Combine assemblage data
        assemblage_data = {
            'memory_assemblage': memory_assemblage,
            'empathic_assemblage': empathic_assemblage,
            'memory_indices': memory_indices,
            'assemblage_id': empathic_assemblage['id'],
            'coherence': memory_assemblage.get('coherence', 0.5),
            'timestamp': time.time()
        }
        
        return assemblage_data
    
    def transform_memory(self, memory_index: int, intensity: float = 0.7) -> Dict[str, Any]:
        """
        Transform a memory through deterritorialization and reterritorialization.
        
        Args:
            memory_index: Index of memory to transform
            intensity: Intensity of transformation
            
        Returns:
            Transformation data
        """
        # Step 1: Deterritorialize memory
        deterritorialized = self.memory_field.deterritorialize_memory(
            memory_index=memory_index,
            intensity=intensity
        )
        
        # Step 2: Reterritorialize memory
        reterritorialized = self.memory_field.reterritorialize_memory(
            deterritorialized=deterritorialized
        )
        
        # Step 3: Get associated assemblage
        original_assemblage_id = self._get_assemblage_for_memory(memory_index)
        
        # Step 4: If assemblage exists, create corresponding transformation
        if original_assemblage_id:
            # Get original assemblage
            original_assemblage = self.empathic_interface.get_empathic_assemblage(original_assemblage_id)
            
            if original_assemblage:
                # Deterritorialize empathic flow
                flow = self.empathic_interface.get_becoming_with(original_assemblage.get('entity_id', 'unknown'))
                
                if flow:
                    # Create deterritorialized empathy
                    empathic_deterritorialized = self.empathic_interface.deterritorialize_empathy(
                        flow=flow,
                        intensity=intensity
                    )
                    
                    # Reterritorialize empathy
                    empathic_reterritorialized = self.empathic_interface.reterritorialize_empathy(
                        deterritorialized=empathic_deterritorialized
                    )
                    
                    # Map new memory to new territory
                    self._map_memory_to_assemblage(
                        reterritorialized['new_memory_index'],
                        empathic_reterritorialized['territories'][0]['type'] if empathic_reterritorialized['territories'] else 'unknown'
                    )
                    
                    # Include empathic transformation data
                    reterritorialized['empathic_deterritorialized'] = empathic_deterritorialized
                    reterritorialized['empathic_reterritorialized'] = empathic_reterritorialized
        
        # Step 5: Add transformation to cognitive state
        self.cognitive_state['transformation_vectors'].append({
            'original_memory': memory_index,
            'new_memory': reterritorialized['new_memory_index'],
            'transformation_type': 'deterritorialization-reterritorialization',
            'intensity': intensity,
            'timestamp': time.time()
        })
        
        return reterritorialized
    
    def create_composite_memory(self, memory_indices: List[int], merge_intensity: float = 1.0) -> Dict[str, Any]:
        """
        Create a composite memory from multiple memories.
        
        Args:
            memory_indices: Indices of memories to merge
            merge_intensity: Intensity of the merge operation
            
        Returns:
            Composite memory data
        """
        # Step 1: Merge memories in morphic field
        new_memory_index = self.memory_field.merge_memories(
            memory_indices=memory_indices,
            merge_intensity=merge_intensity
        )
        
        # Step 2: Get assemblages for component memories
        assemblage_ids = []
        for idx in memory_indices:
            assemblage_id = self._get_assemblage_for_memory(idx)
            if assemblage_id:
                assemblage_ids.append(assemblage_id)
        
        # Step 3: If assemblages exist, create a composite assemblage
        composite_assemblage = None
        if assemblage_ids:
            # Get assemblage components
            assemblage_components = []
            for assemblage_id in assemblage_ids:
                assemblage = self.empathic_interface.get_empathic_assemblage(assemblage_id)
                if assemblage and 'components' in assemblage:
                    assemblage_components.extend(assemblage['components'])
            
            # Create entity representation for composite
            composite_entity = {
                'id': f"composite_{new_memory_index}",
                'affects': {},
                'concepts': [],
                'expressions': []
            }
            
            # Extract unique components
            for component in assemblage_components:
                if component['type'] == 'affective':
                    composite_entity['affects'][component['content']] = component['intensity']
                elif component['type'] == 'conceptual':
                    composite_entity['concepts'].append(component['content'])
                elif component['type'] == 'expressive':
                    composite_entity['expressions'].append(component['content'])
            
            # Create new assemblage
            composite_assemblage = self.empathic_interface.form_empathic_assemblage(composite_entity)
            
            # Map new memory to composite assemblage
            self._map_memory_to_assemblage(new_memory_index, composite_assemblage['id'])
        
        # Step 4: Create composite memory data
        composite_data = {
            'memory_index': new_memory_index,
            'component_indices': memory_indices,
            'assemblage_id': composite_assemblage['id'] if composite_assemblage else None,
            'memory_data': self.memory_field.get_memory(new_memory_index),
            'assemblage_data': composite_assemblage,
            'merge_intensity': merge_intensity,
            'timestamp': time.time()
        }
        
        # Step 5: Add to cognitive state
        self.cognitive_state['becoming_processes'].append({
            'type': 'memory_fusion',
            'components': memory_indices,
            'result': new_memory_index,
            'intensity': merge_intensity,
            'timestamp': time.time()
        })
        
        return composite_data
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get the current state of the cognitive system.
        
        Returns:
            System state data
        """
        # Get component states
        memory_field_state = self.memory_field.get_memory_field_state()
        
        # Count assemblages
        num_assemblages = len(self.memory_to_assemblage)
        
        # Get singularities
        singularities = [{
            'coordinates': (x, y),
            'intensity': intensity
        } for x, y, intensity in self.memory_field.singularities]
        
        # Count active becomings
        becoming_processes = len(self.cognitive_state['becoming_processes'])
        
        # Create state data
        state = {
            'timestamp': time.time(),
            'memories': memory_field_state['num_memories'],
            'assemblages': num_assemblages,
            'rhizomatic_connections': len(self.cognitive_state['rhizomatic_connections']),
            'becoming_processes': becoming_processes,
            'transformation_vectors': len(self.cognitive_state['transformation_vectors']),
            'singularities': singularities,
            'intensive_gradients_max': np.max(self.memory_field.intensive_gradients),
            'resonance_field_max': np.max(sum(self.memory_field.virtual_planes))
        }
        
        return state
    
    def visualize_system(self):
        """
        Visualize the cognitive system.
        Requires matplotlib to be installed.
        """
        # Delegate to memory field visualization
        self.memory_field.visualize_field()
    
    # Helper methods
    def _create_memory_vector_from_entity(self, entity: Dict[str, Any]) -> np.ndarray:
        """Create a memory vector from an entity."""
        # This is a simplified approach - in a real system, you'd encode semantics
        vector_length = self.memory_field.dimensions
        vector = np.zeros(vector_length)
        
        # Fill with some values based on entity properties
        idx = 0
        
        # Add values from affects
        if 'affects' in entity:
            for affect, intensity in entity['affects'].items():
                if idx < vector_length:
                    vector[idx] = float(intensity) if isinstance(intensity, (int, float)) else 0.5
                    idx += 1
        
        # Add values from concepts
        if 'concepts' in entity and isinstance(entity['concepts'], list):
            for concept in entity['concepts']:
                if idx < vector_length:
                    vector[idx] = 0.7  # Default concept intensity
                    idx += 1
        
        # Add values from expressions
        if 'expressions' in entity and isinstance(entity['expressions'], list):
            for expression in entity['expressions']:
                if idx < vector_length:
                    vector[idx] = 0.8  # Default expression intensity
                    idx += 1
        
        # If entity has vectors, use them directly
        if 'vectors' in entity and isinstance(entity['vectors'], list):
            for vector_data in entity['vectors']:
                if isinstance(vector_data, np.ndarray) or isinstance(vector_data, list):
                    # Copy values from provided vector
                    vector_data = np.array(vector_data)
                    copy_length = min(len(vector_data), vector_length - idx)
                    vector[idx:idx+copy_length] = vector_data[:copy_length]
                    idx += copy_length
        
        # Fill remaining entries with small random values
        if idx < vector_length:
            vector[idx:] = np.random.normal(0, 0.1, vector_length - idx)
        
        # Normalize
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
            
        return vector
    
    def _create_memory_vector_from_becoming(self, entity: Dict[str, Any], 
                                          becoming: Dict[str, Any]) -> np.ndarray:
        """Create a memory vector from an entity and becoming process."""
        # Get base vector from entity
        base_vector = self._create_memory_vector_from_entity(entity)
        
        # Modify based on becoming process
        becoming_type = becoming.get('becoming_type', '')
        
        # Apply transformations based on becoming type
        if becoming_type == 'becoming-intense':
            # Amplify some dimensions
            indices = np.random.choice(len(base_vector), size=int(len(base_vector) * 0.3))
            base_vector[indices] *= 1.5
        elif becoming_type == 'becoming-molecular':
            # Add high-frequency components
            noise = np.random.normal(0, 0.2, len(base_vector))
            base_vector = base_vector + noise
        elif becoming_type == 'becoming-imperceptible':
            # Smooth out the vector
            smoothed = np.convolve(base_vector, np.ones(3)/3, mode='same')
            base_vector = 0.7 * base_vector + 0.3 * smoothed
        
        # Ensure vector is normalized
        if np.linalg.norm(base_vector) > 0:
            base_vector = base_vector / np.linalg.norm(base_vector)
            
        return base_vector
    
    def _calculate_affect_intensity(self, assemblage: Dict[str, Any]) -> float:
        """Calculate affect intensity from an assemblage."""
        # Get coherence from assemblage
        coherence = assemblage.get('coherence', 0.5)
        
        # Get intensity from resonance zones
        zones = assemblage.get('resonance_zones', [])
        zone_intensities = [zone.get('intensity', 0.5) for zone in zones]
        
        # Calculate average zone intensity
        if zone_intensities:
            avg_zone_intensity = sum(zone_intensities) / len(zone_intensities)
        else:
            avg_zone_intensity = 0.5
            
        # Combine coherence and zone intensity
        return 0.4 * coherence + 0.6 * avg_zone_intensity
    
    def _map_memory_to_assemblage(self, memory_index: int, assemblage_id: str) -> None:
        """Map a memory to an assemblage."""
        self.memory_to_assemblage[memory_index] = assemblage_id
        
        # Add to assemblage_to_memory mapping
        if assemblage_id not in self.assemblage_to_memory:
            self.assemblage_to_memory[assemblage_id] = []
            
        self.assemblage_to_memory[assemblage_id].append(memory_index)
        
        # Add to rhizomatic connections
        self.cognitive_state['rhizomatic_connections'].append({
            'type': 'memory_to_assemblage',
            'memory_index': memory_index,
            'assemblage_id': assemblage_id,
            'timestamp': time.time()
        })
    
    def _get_assemblage_for_memory(self, memory_index: int) -> Optional[str]:
        """Get the assemblage ID for a memory."""
        return self.memory_to_assemblage.get(memory_index)
    
    def _get_memories_for_assemblage(self, assemblage_id: str) -> List[int]:
        """Get the memories for an assemblage."""
        return self.assemblage_to_memory.get(assemblage_id, [])
    
    def _update_cognitive_state(self, perception: Dict[str, Any]) -> None:
        """Update cognitive state with new perception."""
        # Add rhizomatic connections
        self.cognitive_state['rhizomatic_connections'].append({
            'type': 'perception',
            'entity_id': perception.get('entity_id', 'unknown'),
            'memory_index': perception.get('memory_index'),
            'assemblage_id': perception.get('empathic_assemblage_id'),
            'timestamp': time.time()
        })
        
        # Check for singularities
        if self.memory_field.singularities:
            # Add new singularities
            for x, y, intensity in self.memory_field.singularities:
                self.cognitive_state['singularities'].append({
                    'coordinates': (x, y),
                    'intensity': intensity,
                    'timestamp': time.time()
                })
```
# demo_deleuzian_system.py

import numpy as np
import time
import random
from deleuzian_cognitive_system import DeleuzianCognitiveSystem

def demo_system():
    """Demonstrate the Deleuzian cognitive system with examples."""
    # Initialize the system
    system = DeleuzianCognitiveSystem(memory_dimensions=30, virtual_planes=3)
    print("Initialized Deleuzian Cognitive System")
    
    # Create some example entities
    entities = [
        {
            'id': 'entity_1',
            'affects': {'joy': 0.8, 'excitement': 0.7},
            'concepts': ['forest', 'nature', 'growth'],
            'expressions': ['vibrant', 'expanding']
        },
        {
            'id': 'entity_2',
            'affects': {'calm': 0.6, 'thoughtfulness': 0.9},
            'concepts': ['water', 'flow', 'depth'],
            'expressions': ['peaceful', 'moving']
        },
        {
            'id': 'entity_3',
            'affects': {'intensity': 0.9, 'focus': 0.8},
            'concepts': ['fire', 'transformation', 'energy'],
            'expressions': ['burning', 'changing']
        },
        {
            'id': 'entity_4',
            'affects': {'groundedness': 0.7, 'stability': 0.8},
            'concepts': ['earth', 'solidity', 'foundation'],
            'expressions': ['steady', 'supporting']
        },
        {
            'id': 'entity_5',
            'affects': {'freedom': 0.9, 'lightness': 0.7},
            'concepts': ['air', 'movement', 'breath'],
            'expressions': ['floating', 'swirling']
        }
    ]
    
    # Step 1: Perceive entities
    print("\n1. Perceiving entities...")
    perceptions = []
    
    for entity in entities:
        perception = system.perceive_entity(entity)
        perceptions.append(perception)
        print(f"  Perceived {entity['id']}: Memory #{perception['memory_index']}")
    
    # Step 2: Become-with some entities
    print("\n2. Becoming-with entities...")
    becomings = []
    
    for entity in entities[:3]:  # Become with first three entities
        becoming = system.become_with(entity)
        becomings.append(becoming)
        print(f"  Becoming with {entity['id']}: Sustainability {becoming['sustainability']:.2f}")
    
    # Visualize the system after initial perceptions and becomings
    print("\n3. Visualizing initial system state...")
    system.visualize_system()
    
    # Step 3: Create a composite memory
    print("\n4. Creating composite memory...")
    memory_indices = [perception['memory_index'] for perception in perceptions[:3]]
    composite = system.create_composite_memory(memory_indices, merge_intensity=0.8)
    
    print(f"  Created composite memory #{composite['memory_index']} from memories {memory_indices}")
    print(f"  Linked to assemblage: {composite['assemblage_id']}")
    
    # Step 4: Transform a memory
    print("\n5. Transforming a memory...")
    memory_to_transform = perceptions[3]['memory_index']
    transformation = system.transform_memory(memory_to_transform, intensity=0.7)
    
    print(f"  Transformed memory #{memory_to_transform} into #{transformation['new_memory_index']}")
    
    # Step 5: Recall by resonance
    print("\n6. Recalling by resonance...")
    query_entity = {
        'id': 'query_1',
        'affects': {'joy': 0.7, 'excitement': 0.6},
        'concepts': ['nature', 'growth'],
        'expressions': ['vibrant']
    }
    
    recall_results = system.recall_by_resonance(query_entity)
    
    print(f"  Found {len(recall_results)} resonant memories:")
    for i, result in enumerate(recall_results[:3]):  # Show top 3
        print(f"    #{i+1}: Memory #{result['memory_index']} with similarity {result['memory_similarity']:.2f}")
    
    # Step 6: Create a memory assemblage
    print("\n7. Creating memory assemblage...")
    memory_indices = [perception['memory_index'] for perception in perceptions[2:5]]
    assemblage = system.create_memory_assemblage(memory_indices)
    
    print(f"  Created assemblage with coherence {assemblage['coherence']:.2f}")
    print(f"  Contains {len(assemblage['memory_indices'])} memories")
    
    # Visualize the final system state
    print("\n8. Visualizing final system state...")
    system.visualize_system()
    
    # Step 7: Get system state
    print("\n9. System state:")
    state = system.get_system_state()
    
    print(f"  Memories: {state['memories']}")
    print(f"  Assemblages: {state['assemblages']}")
    print(f"  Rhizomatic connections: {state['rhizomatic_connections']}")
    print(f"  Becoming processes: {state['becoming_processes']}")
    print(f"  Transformation vectors: {state['transformation_vectors']}")
    print(f"  Singularities: {len(state['singularities'])}")
    print(f"  Max intensive gradient: {state['intensive_gradients_max']:.2f}")
    print(f"  Max resonance field: {state['resonance_field_max']:.2f}")

if __name__ == "__main__":
    demo_system()
```
