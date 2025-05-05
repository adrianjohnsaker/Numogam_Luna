
```python
# creative_singularity.py

import random
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from collections import defaultdict
import math
import time

class CreativeSingularityGenerator:
    """
    A system that generates creative singularities - points of infinite creative potential
    where intensive differences produce emergent novelty.
    Based on Deleuze's concept of singularities as points where systems undergo qualitative change.
    """
    
    def __init__(self):
        self.singularity_points = []
        self.creative_explosions = {}
        self.flow_channels = {}
        self.singularity_thresholds = {
            'bifurcation': 0.6,
            'phase_transition': 0.7,
            'critical_point': 0.8,
            'creative_threshold': 0.75
        }
        self.creative_dimensions = [
            'conceptual', 'perceptual', 'affective', 'expressive',
            'metaphorical', 'combinatorial', 'transformative', 'emergent'
        ]
        self.singularity_history = []
        self.explosion_history = []
        self.channeled_flows = {}
        
    def generate_singularity(self, creative_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a point of infinite creative potential from creative input.
        
        Args:
            creative_input: Dictionary containing creative input
            
        Returns:
            Dictionary containing the creative flow and its characteristics
        """
        # Locate singularity potential in creative input
        point = self.locate_singularity_potential(creative_input)
        
        # Trigger creative cascade at singularity point
        explosion = self.trigger_creative_cascade(point)
        
        # Channel creative flow from explosion
        flow = self.channel_creative_flow(explosion)
        
        # Record the singularity process
        self._record_singularity(point, explosion, flow)
        
        return flow
    
    def locate_singularity_potential(self, creative_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Locate potential singularity points in creative input.
        Singularities are points where qualitative change can occur.
        
        Args:
            creative_input: Dictionary containing creative input
            
        Returns:
            Dictionary containing the singularity point
        """
        # Extract input dimensions
        dimensions = self._extract_input_dimensions(creative_input)
        
        # Create tension fields
        tension_fields = self._create_tension_fields(dimensions, creative_input)
        
        # Find critical points
        critical_points = self._find_critical_points(tension_fields)
        
        # Generate intensive differences
        intensive_differences = self._generate_intensive_differences(dimensions, critical_points)
        
        # Select primary singularity point
        primary_point = self._select_primary_singularity(critical_points, intensive_differences)
        
        # Generate virtual multiplicities
        virtual_multiplicities = self._generate_virtual_multiplicities(primary_point, dimensions)
        
        # Create the singularity point
        point_id = f"point_{len(self.singularity_points) + 1}_{int(time.time())}"
        point = {
            'id': point_id,
            'dimensions': dimensions,
            'tension_fields': tension_fields,
            'critical_points': critical_points,
            'intensive_differences': intensive_differences,
            'primary_point': primary_point,
            'virtual_multiplicities': virtual_multiplicities,
            'singularity_type': self._determine_singularity_type(primary_point, creative_input),
            'intensity': self._calculate_singularity_intensity(primary_point, intensive_differences),
            'timestamp': self._get_timestamp()
        }
        
        # Register the point
        self.singularity_points.append(point)
        
        return point
    
    def trigger_creative_cascade(self, point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger creative cascade at singularity point.
        The cascade is an explosion of creative potential.
        
        Args:
            point: Dictionary containing singularity point
            
        Returns:
            Dictionary containing the creative explosion
        """
        # Extract point components
        primary_point = point['primary_point']
        virtual_multiplicities = point['virtual_multiplicities']
        singularity_type = point['singularity_type']
        
        # Generate phase transitions
        phase_transitions = self._generate_phase_transitions(primary_point, singularity_type)
        
        # Create bifurcation cascade
        bifurcation_cascade = self._create_bifurcation_cascade(phase_transitions, virtual_multiplicities)
        
        # Generate emergent patterns
        emergent_patterns = self._generate_emergent_patterns(bifurcation_cascade)
        
        # Create intensive wave propagation
        wave_propagation = self._create_intensive_wave_propagation(emergent_patterns)
        
        # Create the creative explosion
        explosion_id = f"explosion_{len(self.creative_explosions) + 1}_{int(time.time())}"
        explosion = {
            'id': explosion_id,
            'point_id': point['id'],
            'phase_transitions': phase_transitions,
            'bifurcation_cascade': bifurcation_cascade,
            'emergent_patterns': emergent_patterns,
            'wave_propagation': wave_propagation,
            'explosion_type': self._determine_explosion_type(emergent_patterns),
            'intensity': point['intensity'] * random.uniform(1.0, 1.5),  # Amplification
            'timestamp': self._get_timestamp()
        }
        
        # Register the explosion
        self.creative_explosions[explosion_id] = explosion
        
        return explosion
    
    def channel_creative_flow(self, explosion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Channel creative flow from explosion.
        The flow is structured creative potential.
        
        Args:
            explosion: Dictionary containing creative explosion
            
        Returns:
            Dictionary containing the channeled creative flow
        """
        # Extract explosion components
        emergent_patterns = explosion['emergent_patterns']
        wave_propagation = explosion['wave_propagation']
        explosion_type = explosion['explosion_type']
        
        # Create flow channels
        flow_channels = self._create_flow_channels(wave_propagation, explosion_type)
        
        # Generate actualization vectors
        actualization_vectors = self._generate_actualization_vectors(emergent_patterns, flow_channels)
        
        # Create crystallization points
        crystallization_points = self._create_crystallization_points(actualization_vectors)
        
        # Generate creative assemblages
        creative_assemblages = self._generate_creative_assemblages(crystallization_points)
        
        # Create the creative flow
        flow_id = f"flow_{len(self.channeled_flows) + 1}_{int(time.time())}"
        flow = {
            'id': flow_id,
            'explosion_id': explosion['id'],
            'flow_channels': flow_channels,
            'actualization_vectors': actualization_vectors,
            'crystallization_points': crystallization_points,
            'creative_assemblages': creative_assemblages,
            'flow_type': self._determine_flow_type(creative_assemblages),
            'intensity': explosion['intensity'] * random.uniform(0.8, 1.0),  # Slight dampening
            'timestamp': self._get_timestamp()
        }
        
        # Register the flow
        self.channeled_flows[flow_id] = flow
        
        # Register flow channels
        for channel in flow_channels:
            channel_id = channel.get('id', f"channel_{int(time.time())}")
            self.flow_channels[channel_id] = channel
        
        return flow
    
    def combine_creative_flows(self, flows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple creative flows into a meta-flow.
        
        Args:
            flows: List of creative flows
            
        Returns:
            Combined creative meta-flow
        """
        if not flows:
            return {}
            
        # Extract components from flows
        all_channels = []
        all_vectors = []
        all_points = []
        all_assemblages = []
        
        for flow in flows:
            all_channels.extend(flow.get('flow_channels', []))
            all_vectors.extend(flow.get('actualization_vectors', []))
            all_points.extend(flow.get('crystallization_points', []))
            all_assemblages.extend(flow.get('creative_assemblages', []))
            
        # Create connection channels between flows
        connection_channels = self._create_connection_channels(flows)
        
        # Generate synergistic vectors
        synergistic_vectors = self._generate_synergistic_vectors(all_vectors)
        
        # Create composite assemblages
        composite_assemblages = self._create_composite_assemblages(all_assemblages)
        
        # Create the meta-flow
        meta_flow_id = f"meta_flow_{int(time.time())}"
        meta_flow = {
            'id': meta_flow_id,
            'source_flows': [f['id'] for f in flows],
            'flow_channels': all_channels + connection_channels,
            'actualization_vectors': all_vectors + synergistic_vectors,
            'crystallization_points': all_points,
            'creative_assemblages': all_assemblages + composite_assemblages,
            'flow_type': 'synthetic',
            'intensity': sum(f.get('intensity', 0.5) for f in flows) / len(flows) * 1.2,  # Synergistic boost
            'timestamp': self._get_timestamp()
        }
        
        return meta_flow
    
    def get_singularity_history(self) -> List[Dict[str, Any]]:
        """
        Get history of singularity generation.
        
        Returns:
            List of singularity records
        """
        return self.singularity_history.copy()
    
    def get_creative_flow(self, flow_id: str) -> Dict[str, Any]:
        """
        Get specific creative flow.
        
        Args:
            flow_id: ID of flow
            
        Returns:
            Creative flow or empty dict if not found
        """
        return self.channeled_flows.get(flow_id, {})
    
    # Helper methods for singularity potential location
    def _extract_input_dimensions(self, creative_input: Dict[str, Any]) -> Dict[str, float]:
        """Extract dimensional intensities from creative input."""
        dimensions = {}
        
        # Process different input types
        if 'text' in creative_input:
            # Extract from text
            text = creative_input['text']
            dimensions = self._extract_dimensions_from_text(text)
            
        elif 'concepts' in creative_input:
            # Extract from concept list
            concepts = creative_input['concepts']
            dimensions = self._extract_dimensions_from_concepts(concepts)
            
        elif 'affects' in creative_input:
            # Extract from affects
            affects = creative_input['affects']
            dimensions = self._extract_dimensions_from_affects(affects)
            
        # Ensure all creative dimensions are present
        for dimension in self.creative_dimensions:
            if dimension not in dimensions:
                dimensions[dimension] = random.uniform(0.2, 0.4)  # Default low intensity
                
        return dimensions
    
    def _extract_dimensions_from_text(self, text: str) -> Dict[str, float]:
        """Extract dimensional intensities from text."""
        dimensions = {}
        
        # Count words for base intensity
        words = [w for w in text.split() if w.strip()]
        base_intensity = min(1.0, len(words) / 100)  # Scale based on length
        
        # Look for various creative aspects
        # Conceptual dimension (abstract concepts)
        conceptual_words = ['idea', 'concept', 'theory', 'framework', 'system', 
                          'philosophy', 'abstract', 'notion', 'paradigm']
        conceptual_count = sum(1 for w in words if w.lower() in conceptual_words)
        dimensions['conceptual'] = min(1.0, conceptual_count / 10 + base_intensity * 0.5)
        
        # Perceptual dimension (sensory words)
        perceptual_words = ['see', 'hear', 'feel', 'touch', 'sense', 'perceive',
                          'vision', 'sound', 'texture', 'taste', 'smell']
        perceptual_count = sum(1 for w in words if w.lower() in perceptual_words)
        dimensions['perceptual'] = min(1.0, perceptual_count / 10 + base_intensity * 0.5)
        
        # Affective dimension (emotional words)
        affective_words = ['emotion', 'feel', 'passion', 'desire', 'love', 'hate',
                         'joy', 'sorrow', 'anger', 'fear', 'surprise']
        affective_count = sum(1 for w in words if w.lower() in affective_words)
        dimensions['affective'] = min(1.0, affective_count / 10 + base_intensity * 0.5)
        
        # Expressive dimension
        expressive_words = ['create', 'express', 'speak', 'show', 'display', 'perform',
                          'articulate', 'convey', 'communicate', 'present']
        expressive_count = sum(1 for w in words if w.lower() in expressive_words)
        dimensions['expressive'] = min(1.0, expressive_count / 10 + base_intensity * 0.5)
        
        # Metaphorical dimension
        metaphorical_words = ['like', 'as', 'metaphor', 'symbol', 'represent', 
                            'image', 'figure', 'allegory', 'comparison']
        metaphorical_count = sum(1 for w in words if w.lower() in metaphorical_words)
        dimensions['metaphorical'] = min(1.0, metaphorical_count / 10 + base_intensity * 0.5)
        
        # Other dimensions with default values
        dimensions['combinatorial'] = base_intensity * random.uniform(0.7, 1.3)
        dimensions['transformative'] = base_intensity * random.uniform(0.7,, 1.3)
        dimensions['emergent'] = base_intensity * random.uniform(0.7, 1.3)
        
        return dimensions
    
    def _extract_dimensions_from_concepts(self, concepts: List[str]) -> Dict[str, float]:
        """Extract dimensional intensities from concept list."""
        dimensions = {}
        
        # Base intensity from number of concepts
        base_intensity = min(1.0, len(concepts) / 10)
        
        # Count concept types
        abstract_count = 0
        concrete_count = 0
        emotional_count = 0
        perceptual_count = 0
        
        # Simple concept categorization
        abstract_indicators = ['theory', 'system', 'concept', 'philosophy', 'abstract']
        emotional_indicators = ['emotion', 'feeling', 'mood', 'affect', 'passion']
        perceptual_indicators = ['perception', 'sense', 'vision', 'sound', 'touch']
        
        for concept in concepts:
            # Check if abstract
            if any(i in concept.lower() for i in abstract_indicators):
                abstract_count += 1
            # Check if emotional
            elif any(i in concept.lower() for i in emotional_indicators):
                emotional_count += 1
            # Check if perceptual
            elif any(i in concept.lower() for i in perceptual_indicators):
                perceptual_count += 1
            else:
                concrete_count += 1
                
        # Calculate dimensional intensities
        dimensions['conceptual'] = min(1.0, abstract_count / len(concepts) + base_intensity * 0.5)
        dimensions['affective'] = min(1.0, emotional_count / len(concepts) + base_intensity * 0.5)
        dimensions['perceptual'] = min(1.0, perceptual_count / len(concepts) + base_intensity * 0.5)
        
        # Derived dimensions
        dimensions['expressive'] = base_intensity * random.uniform(0.7, 1.3)
        dimensions['metaphorical'] = base_intensity * random.uniform(0.7, 1.3)
        dimensions['combinatorial'] = base_intensity * (1 + len(concepts) / 20)  # More concepts = more combinatorial
        dimensions['transformative'] = base_intensity * random.uniform(0.7, 1.3)
        dimensions['emergent'] = base_intensity * random.uniform(0.7, 1.3)
        
        return dimensions
    
    def _extract_dimensions_from_affects(self, affects: Dict[str, float]) -> Dict[str, float]:
        """Extract dimensional intensities from affects."""
        dimensions = {}
        
        # Base intensity from number of affects
        base_intensity = min(1.0, len(affects) / 8)
        
        # Direct mapping for affective dimension
        dimensions['affective'] = min(1.0, sum(affects.values()) / len(affects) if affects else 0.5)
        
        # Derive other dimensions
        dimensions['conceptual'] = base_intensity * random.uniform(0.5, 1.0)
        dimensions['perceptual'] = base_intensity * random.uniform(0.7, 1.2)
        dimensions['expressive'] = dimensions['affective'] * random.uniform(0.8, 1.2)  # Linked to affect
        dimensions['metaphorical'] = base_intensity * random.uniform(0.7, 1.3)
        dimensions['combinatorial'] = base_intensity * random.uniform(0.7, 1.3)
        dimensions['transformative'] = dimensions['affective'] * random.uniform(0.7, 1.3)  # Linked to affect
        dimensions['emergent'] = base_intensity * random.uniform(0.7, 1.3)
        
        return dimensions
    
    def _create_tension_fields(self, dimensions: Dict[str, float], 
                             creative_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create tension fields from dimensional intensities."""
        fields = []
        
        # Create field for each significant dimension
        for dimension, intensity in dimensions.items():
            if intensity > 0.4:  # Significant dimension
                # Create tension gradients
                gradients = self._create_tension_gradients(dimension, intensity)
                
                # Create field
                field = {
                    'dimension': dimension,
                    'intensity': intensity,
                    'gradients': gradients,
                    'threshold': self.singularity_thresholds.get('bifurcation', 0.6) * random.uniform(0.9, 1.1)
                }
                fields.append(field)
                
        # Create cross-dimensional fields
        dim_pairs = []
        dimensions_list = list(dimensions.keys())
        
        for i in range(len(dimensions_list) - 1):
            for j in range(i + 1, len(dimensions_list)):
                dim_a = dimensions_list[i]
                dim_b = dimensions_list[j]
                
                # Only create for significant dimension pairs
                if dimensions[dim_a] > 0.5 and dimensions[dim_b] > 0.5:
                    dim_pairs.append((dim_a, dim_b))
                    
        # Select random subset of pairs
        selected_pairs = random.sample(
            dim_pairs,
            min(3, len(dim_pairs))  # At most 3 pairs
        )
        
        # Create field for each selected pair
        for dim_a, dim_b in selected_pairs:
            # Create tension matrix
            matrix = self._create_tension_matrix(dim_a, dim_b, dimensions[dim_a], dimensions[dim_b])
            
            # Create field
            field = {
                'dimensions': [dim_a, dim_b],
                'intensities': [dimensions[dim_a], dimensions[dim_b]],
                'matrix': matrix,
                'threshold': self.singularity_thresholds.get('phase_transition', 0.7) * random.uniform(0.9, 1.1),
                'type': 'cross-dimensional'
            }
            fields.append(field)
            
        return fields
    
    def _create_tension_gradients(self, dimension: str, intensity: float) -> List[float]:
        """Create tension gradients for a dimension."""
        # Create gradient with multiple points
        points = 10
        gradient = []
        
        # Base shape determined by dimension
        if dimension == 'conceptual':
            # Conceptual: more peaks and valleys (abstract thought)
            for i in range(points):
                pos = i / (points - 1)
                val = intensity * (0.7 + 0.5 * math.sin(pos * math.pi * 3))
                gradient.append(max(0, min(1, val)))
                
        elif dimension == 'affective':
            # Affective: smoother curve (emotional flow)
            for i in range(points):
                pos = i / (points - 1)
                val = intensity * (0.8 + 0.4 * math.sin(pos * math.pi * 2))
                gradient.append(max(0, min(1, val)))
                
        elif dimension == 'perceptual':
            # Perceptual: more sudden shifts (sensory focus)
            for i in range(points):
                pos = i / (points - 1)
                if pos < 0.3 or pos > 0.7:
                    val = intensity * 0.6
                else:
                    val = intensity * 1.0
                gradient.append(max(0, min(1, val)))
                
        elif dimension == 'expressive':
            # Expressive: building wave (creative expression)
            for i in range(points):
                pos = i / (points - 1)
                val = intensity * (0.5 + 0.5 * pos)
                gradient.append(max(0, min(1, val)))
                
        elif dimension == 'metaphorical':
            # Metaphorical: oscillating (connecting concepts)
            for i in range(points):
                pos = i / (points - 1)
                val = intensity * (0.7 + 0.4 * math.sin(pos * math.pi * 4))
                gradient.append(max(0, min(1, val)))
                
        else:
            # Default: random variations
            for i in range(points):
                val = intensity * random.uniform(0.7, 1.3)
                gradient.append(max(0, min(1, val)))
                
        return gradient
    
    def _create_tension_matrix(self, dim_a: str, dim_b: str, 
                             intensity_a: float, intensity_b: float) -> List[List[float]]:
        """Create tension matrix between two dimensions."""
        # Create matrix
        size = 5  # 5x5 matrix
        matrix = []
        
        for i in range(size):
            row = []
            for j in range(size):
                # Position in normalized coordinates
                x = i / (size - 1)
                y = j / (size - 1)
                
                # Base value
                val = (intensity_a * x + intensity_b * y) / 2
                
                # Apply dimension-specific modulation
                if dim_a == 'conceptual' and dim_b == 'affective':
                    # Conceptual-affective synergy at balanced points
                    synergy = 1.0 - 2 * abs(x - y)  # Higher when x and y are close
                    val *= (1 + 0.3 * synergy)
                elif dim_a == 'perceptual' and dim_b == 'metaphorical':
                    # Perceptual-metaphorical tension at extremes
                    tension = (x + y) / 2  # Higher at top-right
                    val *= (1 + 0.3 * tension)
                elif 'emergent' in [dim_a, dim_b]:
                    # Emergent dimension creates hotspots
                    hotspot = math.exp(-5 * ((x - 0.7)**2 + (y - 0.7)**2))  # Hotspot near top-right
                    val *= (1 + 0.5 * hotspot)
                
                # Add some noise
                val += random.uniform(-0.1, 0.1) * (intensity_a + intensity_b) / 2
                
                row.append(max(0, min(1, val)))
            matrix.append(row)
            
        return matrix
    
    def _find_critical_points(self, tension_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find critical points in tension fields."""
        critical_points = []
        
        # Process dimensional fields
        for field in tension_fields:
            if 'dimension' in field:  # Dimensional field
                dimension = field['dimension']
                gradients = field['gradients']
                threshold = field['threshold']
                
                # Find points where gradient crosses threshold
                for i in range(len(gradients) - 1):
                    if (gradients[i] <= threshold and gradients[i+1] > threshold) or \
                       (gradients[i] >= threshold and gradients[i+1] < threshold):
                        # Create critical point
                        point = {
                            'dimension': dimension,
                            'position': (i + 0.5) / len(gradients),
                            'intensity': (gradients[i] + gradients[i+1]) / 2,
                            'type': 'threshold_crossing'
                        }
                        critical_points.append(point)
                
                # Find local maxima
                for i in range(1, len(gradients) - 1):
                    if gradients[i] > gradients[i-1] and gradients[i] > gradients[i+1]:
                        # Create critical point
                        point = {
                            'dimension': dimension,
                            'position': i / (len(gradients) - 1),
                            'intensity': gradients[i],
                            'type': 'local_maximum'
                        }
                        critical_points.append(point)
                        
            elif 'dimensions' in field:  # Cross-dimensional field
                dimensions = field['dimensions']
                matrix = field['matrix']
                threshold = field['threshold']
                
                # Find high intensity points in matrix
                for i in range(len(matrix)):
                    for j in range(len(matrix[0])):
                        val = matrix[i][j]
                        
                        # Check if above threshold
                        if val > threshold:
                            # Create critical point
                            point = {
                                'dimensions': dimensions,
                                'position': [i / (len(matrix) - 1), j / (len(matrix[0]) - 1)],
                                'intensity': val,
                                'type': 'high_intensity'
                            }
                            critical_points.append(point)
                            
                # Find saddle points
                for i in range(1, len(matrix) - 1):
                    for j in range(1, len(matrix[0]) - 1):
                        val = matrix[i][j]
                        
                        # Check for saddle point pattern
                        if ((val > matrix[i-1][j] and val > matrix[i+1][j]) and
                            (val < matrix[i][j-1] and val < matrix[i][j+1])) or \
                           ((val < matrix[i-1][j] and val < matrix[i+1][j]) and
                            (val > matrix[i][j-1] and val > matrix[i][j+1])):
                            # Create critical point
                            point = {
                                'dimensions': dimensions,
                                'position': [i / (len(matrix) - 1), j / (len(matrix[0]) - 1)],
                                'intensity': val,
                                'type': 'saddle_point'
                            }
                            critical_points.append(point)
        
        return critical_points
    
    def _generate_intensive_differences(self, dimensions: Dict[str, float], 
                                      critical_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate intensive differences from dimensions and critical points."""
        differences = []
        
        # Create difference for each significant dimension
        for dimension, intensity in dimensions.items():
            if intensity > 0.5:  # Significant dimension
                # Find critical points for dimension
                related_points = [p for p in critical_points 
                                if ('dimension' in p and p['dimension'] == dimension) or
                                   ('dimensions' in p and dimension in p['dimensions'])]
                
                if related_points:
                    # Create intensive difference
                    difference = {
                        'dimension': dimension,
                        'intensity': intensity,
                        'critical_points': related_points,
                        'threshold': self.singularity_thresholds.get('creative_threshold', 0.75)
                    }
                    differences.append(difference)
                    
        # Create cross-dimensional differences
        dim_pairs = []
        dimensions_list = list(dimensions.keys())
        
        for i in range(len(dimensions_list) - 1):
            for j in range(i + 1, len(dimensions_list)):
                dim_a = dimensions_list[i]
                dim_b = dimensions_list[j]
                
                # Only create if both dimensions are significant
                if dimensions[dim_a] > 0.4 and dimensions[dim_b] > 0.4:
                    # Find critical points for dimension pair
                    related_points = [p for p in critical_points 
                                    if 'dimensions' in p and 
                                       set(p['dimensions']) == set([dim_a, dim_b])]
                    
                    if related_points:
                        # Create intensive difference
                        difference = {
                            'dimensions': [dim_a, dim_b],
                            'intensities': [dimensions[dim_a], dimensions[dim_b]],
                            'critical_points': related_points,
                            'threshold': self.singularity_thresholds.get('phase_transition', 0.7)
                        }
                        differences.append(difference)
                        
        return differences
    
    def _select_primary_singularity(self, critical_points: List[Dict[str, Any]],
                                  intensive_differences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select primary singularity point from candidates."""
        if not critical_points:
            return {
                'type': 'default',
                'intensity': 0.5,
                'dimensions': []
            }
            
        # Filter to high intensity points
        high_intensity = [p for p in critical_points if p.get('intensity', 0) > 0.7]
        
        if high_intensity:
            # Select highest intensity point
            return max(high_intensity, key=lambda p: p.get('intensity', 0))
        else:
            # Select random point
            return random.choice(critical_points)
    
    def _generate_virtual_multiplicities(self, primary_point: Dict[str, Any],
                                       dimensions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate virtual multiplicities from primary point."""
        multiplicities = []
        
        # Determine which dimensions to include
        if 'dimension' in primary_point:
            # Single dimension point
            relevant_dims = [primary_point['dimension']]
        elif 'dimensions' in primary_point:
            # Cross-dimensional point
            relevant_dims = primary_point['dimensions']
        else:
            # Default
            relevant_dims = [d for d, i in dimensions.items() if i > 0.6]
            
        # Generate base multiplicity for primary dimensions
        if relevant_dims:
            multiplicity = {
                'dimensions': relevant_dims,
                'intensity': primary_point.get('intensity', 0.5),
                'type': 'primary',
                'vectors': self._generate_multiplicity_vectors(relevant_dims, dimensions)
            }
            multiplicities.append(multiplicity)
        
        # Generate secondary multiplicities
        secondary_dims = [d for d in dimensions if d not in relevant_dims and dimensions[d] > 0.4]
        
        if secondary_dims:
            # Select subset of secondary dimensions
            selected_dims = random.sample(
                secondary_dims,
                min(3, len(secondary_dims))
            )
            
            # Create multiplicity
            multiplicity = {
                'dimensions': selected_dims,
                'intensity': sum(dimensions[d] for d in selected_dims) / len(selected_dims),
                'type': 'secondary',
                'vectors': self._generate_multiplicity_vectors(selected_dims, dimensions)
            }
            multiplicities.append(multiplicity)
        
        # Generate cross-multiplicities
        if len(multiplicities) >= 2:
            # Create cross-connections between multiplicities
            cross_mult = {
                'dimensions': [d for m in multiplicities for d in m['dimensions']],
                'intensity': sum(m['intensity'] for m in multiplicities) / len(multiplicities),
                'type': 'cross',
                'component_multiplicities': [i for i in range(len(multiplicities))],
                'synergy_factor': random.uniform(1.1, 1.5)  # Synergistic boost
            }
            multiplicities.append(cross_mult)
            
        return multiplicities
    
    def _generate_multiplicity_vectors(self, dimensions: List[str], 
                                     all_dimensions: Dict[str, float]) -> Dict[str, List[float]]:
        """Generate vectors for multiplicity dimensions."""
        vectors = {}
        
        for dimension in dimensions:
            intensity = all_dimensions.get(dimension, 0.5)
            
            # Create vector trajectory
            points = 5
            vector = []
            
            for i in range(points):
                # Create point-to-point trajectory with some randomness
                if i == 0:
                    # Start point
                    val = intensity * random.uniform(0.7, 0.9)
                elif i == points - 1:
                    # End point - higher than start (potential growth)
                    val = intensity * random.uniform(1.0, 1.2)
                else:
                    # Intermediate points
                    val = intensity * random.uniform(0.8, 1.1)
                    
                vector.append(max(0, min(1, val)))
                
            vectors[dimension] = vector
            
        return vectors
    
    def _determine_singularity_type(self, primary_point: Dict[str, Any], 
                                  creative_input: Dict[str, Any]) -> str:
        """Determine type of singularity based on primary point and input."""
        # Check point type
        point_type = primary_point.get('type', '')
        
        if 'local_maximum' in point_type:
            return 'intensive_peak'
        elif 'saddle_point' in point_type:
            return 'phase_transition'
        elif 'threshold_crossing' in point_type:
            return 'bifurcation'
        elif 'high_intensity' in point_type:
            return 'critical_point'
            
        # Check dimensions
        if 'dimension' in primary_point:
            dimension = primary_point['dimension']
            
            if dimension == 'conceptual':
                return 'conceptual_singularity'
            elif dimension == 'affective':
                return 'affective_singularity'
            elif dimension == 'perceptual':
                return 'perceptual_singularity'
            elif dimension == 'expressive':
                return 'expressive_singularity'
            elif dimension == 'metaphorical':
                return 'metaphorical_singularity'
                
        # Default
        return 'creative_singularity'
    
    def _calculate_singularity_intensity(self, primary_point: Dict[str, Any],
                                       intensive_differences: List[Dict[str, Any]]) -> float:
        """Calculate intensity of singularity."""
        # Base on primary point intensity
        base_intensity = primary_point.get('intensity', 0.5)
        
        # Adjust based on related intensive differences
        if 'dimension' in primary_point and intensive_differences:
            dimension = primary_point['dimension']
            related_diffs = [d for d in intensive_differences 
                           if ('dimension' in d and d['dimension'] == dimension) or
                              ('dimensions' in d and dimension in d.get('dimensions', []))]
            
            if related_diffs:
                diff_factor = sum(d.get('intensity', 0.5) for d in related_diffs) / len(related_diffs)
                base_intensity = base_intensity * (0.7 + 0.3 * diff_factor)
                
        elif 'dimensions' in primary_point and intensive_differences:
            dimensions = set(primary_point['dimensions'])
            related_diffs = [d for d in intensive_differences 
                           if ('dimensions' in d and 
                               set(d.get('dimensions', [])).intersection(dimensions))]
            
            if related_diffs:
                diff_factor = sum(d.get('intensity', 0.5) for d in related_diffs) / len(related_diffs)
                base_intensity = base_intensity * (0.7 + 0.3 * diff_factor)
                
        return min(1.0, base_intensity)
    
    # Helper methods for creative cascade
    def _generate_phase_transitions(self, primary_point: Dict[str, Any], 
                                  singularity_type: str) -> List[Dict[str, Any]]:
        """Generate phase transitions from primary singularity point."""
        transitions = []
        
        # Determine number of transitions based on singularity type
        if 'bifurcation' in singularity_type:
            num_transitions = 2  # Bifurcation creates two branches
        elif 'critical_point' in singularity_type:
            num_transitions = random.randint(3, 5)  # Critical points create multiple transitions
        else:
            num_transitions = random.randint(1, 3)  # Default range
            
        # Create transitions
        for i in range(num_transitions):
            # Determine transition type
            if i == 0:
                # First transition
                trans_type = 'primary'
            else:
                # Secondary transitions
                trans_type = 'secondary'
                
            # Determine transition dimensions
            if 'dimension' in primary_point:
                dimensions = [primary_point['dimension']]
            elif 'dimensions' in primary_point:
                dimensions = primary_point['dimensions']
            else:
                dimensions = []
                
            # Create phase transition
            transition = {
                'type': trans_type,
                'dimensions': dimensions,
                'intensity': primary_point.get('intensity', 0.5) * random.uniform(0.8, 1.0),
                'transition_type': self._determine_transition_type(singularity_type, i),
                'order_parameter': random.uniform(0, 1),
                'control_parameter': random.uniform(0, 1)
            }
            transitions.append(transition)
            
        return transitions
    
    def _determine_transition_type(self, singularity_type: str, index: int) -> str:
        """Determine type of phase transition."""
        if 'bifurcation' in singularity_type:
            if index == 0:
                return 'symmetry_breaking'
            else:
                return 'branch_formation'
        elif 'critical_point' in singularity_type:
            if index == 0:
                return 'criticality'
            else:
                return 'emergence'
        elif 'conceptual' in singularity_type:
            if index == 0:
                return 'concept_formation'
            else:
                return 'conceptual_divergence'
        elif 'affective' in singularity_type:
            if index == 0:
                return 'affective_intensification'
            else:
                return 'emotional_transformation'
        else:
            if index == 0:
                return 'qualitative_change'
            else:
                return 'novel_formation'
    
    def _create_bifurcation_cascade(self, phase_transitions: List[Dict[str, Any]],
                                  virtual_multiplicities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create bifurcation cascade from phase transitions."""
        # Map transitions to branches
        branches = []
        for transition in phase_transitions:
            # Create branch
            branch = {
                'transition_type': transition['transition_type'],
                'dimensions': transition['dimensions'],
                'intensity': transition['intensity'],
                'sub_branches': []
            }
            
            # Add randomized sub-branches (except for primary transition)
            if transition['type'] != 'primary':
                num_sub = random.randint(1, 3)
                for i in range(num_sub):
                    sub_branch = {
                        'intensity': transition['intensity'] * random.uniform(0.7, 0.9),
                        'divergence_factor': random.uniform(0.2, 0.8)
                    }
                    branch['sub_branches'].append(sub_branch)
                    
            branches.append(branch)
            
        # Create connections between branches
        connections = []
        if len(branches) >= 2:
            for i in range(len(branches) - 1):
                for j in range(i + 1, len(branches)):
                    # Only connect some branches
                    if random.random() < 0.7:  # 70% chance
                        # Create connection
                        connection = {
                            'branch_indices': [i, j],
                            'intensity': (branches[i]['intensity'] + branches[j]['intensity']) / 2,
                            'type': 'inter-branch'
                        }
                        connections.append(connection)
        
        # Integrate virtual multiplicities
        multiplicity_mappings = []
        for i, mult in enumerate(virtual_multiplicities):
            # Find relevant branches
            relevant_branches = []
            for j, branch in enumerate(branches):
                # Check for dimensional overlap
                if set(branch['dimensions']).intersection(set(mult.get('dimensions', []))):
                    relevant_branches.append(j)
                    
            if relevant_branches:
                # Create mapping
                mapping = {
                    'multiplicity_index': i,
                    'branch_indices': relevant_branches,
                    'intensity': mult.get('intensity', 0.5),
                    'synergy_factor': mult.get('synergy_factor', 1.0) if 'synergy_factor' in mult else 1.0
                }
                multiplicity_mappings.append(mapping)
                
        # Create cascade
        cascade = {
            'branches': branches,
            'connections': connections,
            'multiplicity_mappings': multiplicity_mappings,
            'bifurcation_depth': len(branches),
            'connection_density': len(connections) / (len(branches) * (len(branches) - 1) / 2) if len(branches) > 1 else 0
        }
        
        return cascade
    
    def _generate_emergent_patterns(self, cascade: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate emergent patterns from bifurcation cascade."""
        patterns = []
        
        # Extract cascade components
        branches = cascade['branches']
        connections = cascade['connections']
        multiplicity_mappings = cascade['multiplicity_mappings']
        
        # Generate patterns from significant branches
        for i, branch in enumerate(branches):
            if branch['intensity'] > 0.6:  # Significant branch
                # Create pattern
                pattern = {
                    'type': 'branch-based',
                    'source_branch': i,
                    'dimensions': branch['dimensions'],
                    'intensity': branch['intensity'],
                    'pattern_type': self._determine_pattern_type(branch)
                }
                patterns.append(pattern)
                
        # Generate patterns from significant connections
        for connection in connections:
            intensity = connection.get('intensity', 0.5)
            if intensity > 0.7:  # Significant connection
                # Get connected branches
                branch_indices = connection.get('branch_indices', [])
                if len(branch_indices) >= 2:
                    br_a = branch_indices[0]
                    br_b = branch_indices[1]
                    
                    if br_a < len(branches) and br_b < len(branches):
                        # Create pattern
                        pattern = {
                            'type': 'connection-based',
                            'source_branches': branch_indices,
                            'dimensions': list(set(branches[br_a].get('dimensions', []) + 
                                                 branches[br_b].get('dimensions', []))),
                            'intensity': intensity,
                            'pattern_type': 'synthesis'
                        }
                        patterns.append(pattern)
                        
        # Generate patterns from multiplicity mappings
        for mapping in multiplicity_mappings:
            intensity = mapping.get('intensity', 0.5)
            synergy = mapping.get('synergy_factor', 1.0)
            
            if intensity * synergy > 0.7:  # Significant multiplicity mapping
                # Create pattern
                pattern = {
                    'type': 'multiplicity-based',
                    'source_branches': mapping.get('branch_indices', []),
                    'multiplicity_index': mapping.get('multiplicity_index', 0),
                    'intensity': intensity * synergy,
                    'pattern_type': 'emergent'
                }
                patterns.append(pattern)
                
        return patterns
    
    def _determine_pattern_type(self, branch: Dict[str, Any]) -> str:
        """Determine pattern type from branch."""
        # Check transition type
        transition_type = branch.get('transition_type', '')
        
        if 'symmetry_breaking' in transition_type:
            return 'bifurcation'
        elif 'concept_formation' in transition_type:
            return 'conceptual'
        elif 'affective' in transition_type:
            return 'affective'
        elif 'criticality' in transition_type:
            return 'critical'
        elif 'emergence' in transition_type:
            return 'emergent'
        else:
            return 'transformative'
    
    def _create_intensive_wave_propagation(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create intensive wave propagation from emergent patterns."""
        # Extract pattern information
        pattern_types = [p['pattern_type'] for p in patterns]
        pattern_intensities = [p['intensity'] for p in patterns]
        
        # Create wave components
        components = []
        for i, pattern in enumerate(patterns):
            # Create component
            component = {
                'pattern_index': i,
                'type': pattern['pattern_type'],
                'intensity': pattern['intensity'],
                'frequency': random.uniform(0.3, 1.0),
                'phase': random.uniform(0, 2 * math.pi),
                'amplitude': pattern['intensity'] * random.uniform(0.8, 1.2)
            }
            components.append(component)
            
        # Create interference patterns
        interferences = []
        if len(components) >= 2:
            for i in range(len(components) - 1):
                for j in range(i + 1, len(components)):
                    # Create interference
                    interference = {
                        'component_indices': [i, j],
                        'type': 'wave_interference',
                        'intensity': (components[i]['intensity'] + components[j]['intensity']) / 2,
                        'interference_type': 'constructive' if random.random() < 0.7 else 'destructive'
                    }
                    interferences.append(interference)
                    
        # Create resonances
        resonances = []
        for component in components:
            if component['intensity'] > 0.7:  # High intensity components create resonances
                # Create resonance
                resonance = {
                    'component_index': components.index(component),
                    'intensity': component['intensity'],
                    'frequency': component['frequency'],
                    'decay_rate': random.uniform(0.1, 0.3)
                }
                resonances.append(resonance)
                
        # Create overall wave propagation
        propagation = {
            'components': components,
            'interferences': interferences,
            'resonances': resonances,
            'overall_intensity': sum(pattern_intensities) / len(pattern_intensities) if pattern_intensities else 0.5,
            'coherence': self._calculate_wave_coherence(components, interferences)
        }
        
        return propagation
    
    def _calculate_wave_coherence(self, components: List[Dict[str, Any]],
                                interferences: List[Dict[str, Any]]) -> float:
        """Calculate coherence of wave propagation."""
        if not components:
            return 0.0
            
        # Base coherence on component properties
        frequencies = [c['frequency'] for c in components]
        freq_std = np.std(frequencies) if len(frequencies) > 1 else 0
        
        # Lower standard deviation means more coherent frequencies
        freq_coherence = max(0, 1 - freq_std)
        
        # Adjust based on interference types
        constructive = sum(1 for i in interferences if i.get('interference_type') == 'constructive')
        destructive = sum(1 for i in interferences if i.get('interference_type') == 'destructive')
        
        if interferences:
            interference_factor = constructive / (constructive + destructive) if (constructive + destructive) > 0 else 0.5
        else:
            interference_factor = 0.5
            
        # Combine factors
        return 0.6 * freq_coherence + 0.4 * interference_factor
    
    def _determine_explosion_type(self, patterns: List[Dict[str, Any]]) -> str:
        """Determine type of creative explosion."""
        if not patterns:
            return 'generic'
            
        # Count pattern types
        pattern_types = [p['pattern_type'] for p in patterns]
        type_counts = {}
        for p_type in pattern_types:
            type_counts[p_type] = type_counts.get(p_type, 0) + 1
            
        # Find most common type
        if type_counts:
            common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            if common_type == 'conceptual':
                return 'conceptual_explosion'
            elif common_type == 'affective':
                return 'affective_explosion'
            elif common_type == 'critical':
                return 'critical_explosion'
            elif common_type == 'emergent':
                return 'emergent_explosion'
            elif common_type == 'bifurcation':
                return 'bifurcation_explosion'
                
        # Default
        return 'creative_explosion'
    
    # Helper methods for creative flow channeling
    def _create_flow_channels(self, wave_propagation: Dict[str, Any], 
                           explosion_type: str) -> List[Dict[str, Any]]:
        """Create flow channels from wave propagation."""
        channels = []
        
        # Extract propagation components
        components = wave_propagation['components']
        
        # Determine number of channels based on explosion type
        if 'conceptual' in explosion_type:
            num_channels = random.randint(3, 5)  # Conceptual: more channels
        elif 'affective' in explosion_type:
            num_channels = random.randint(2, 3)  # Affective: fewer channels
        elif 'critical' in explosion_type:
            num_channels = random.randint(4, 6)  # Critical: many channels
        else:
            num_channels = random.randint(2, 4)  # Default range
            
        # Ensure we don't exceed available components
        num_channels = min(num_channels, len(components))
        
        # Create channels
        for i in range(num_channels):
            # Select component
            if i < len(components):
                component = components[i]
            else:
                # If not enough components, create new ones
                component = {
                    'type': 'generated',
                    'intensity': random.uniform(0.5, 0.8),
                    'frequency': random.uniform(0.3, 1.0)
                }
                
            # Determine channel type
            if 'conceptual' in explosion_type:
                channel_type = 'conceptual'
            elif 'affective' in explosion_type:
                channel_type = 'affective'
            elif 'critical' in explosion_type:
                channel_type = 'critical'
            elif 'emergent' in explosion_type:
                channel_type = 'emergent'
            else:
                # Based on component type
                component_type = component.get('type', 'generic')
                if component_type in ['conceptual', 'affective', 'critical', 'emergent']:
                    channel_type = component_type
                else:
                    channel_type = 'generic'
                    
            # Create channel
            channel_id = f"channel_{len(self.flow_channels) + i + 1}_{int(time.time())}"
            channel = {
                'id': channel_id,
                'type': channel_type,
                'source_component': i if i < len(components) else -1,
                'intensity': component.get('intensity', 0.5),
                'frequency': component.get('frequency', 0.5),
                'bandwidth': self._calculate_channel_bandwidth(component, channel_type),
                'modulation': self._determine_channel_modulation(channel_type)
            }
            channels.append(channel)
            
        return channels
    
    def _calculate_channel_bandwidth(self, component: Dict[str, Any], channel_type: str) -> float:
        """Calculate bandwidth of flow channel."""
        # Base on component intensity
        base_bandwidth = component.get('intensity', 0.5)
        
        # Adjust based on channel type
        if channel_type == 'conceptual':
            base_bandwidth *= 1.2  # Higher bandwidth for conceptual
        elif channel_type == 'affective':
            base_bandwidth *= 0.8  # Lower bandwidth for affective
        elif channel_type == 'critical':
            base_bandwidth *= 1.1  # Higher bandwidth for critical
            
        return min(1.0, base_bandwidth)
    
    def _determine_channel_modulation(self, channel_type: str) -> str:
        """Determine modulation type for flow channel."""
        if channel_type == 'conceptual':
            modulations = ['abstract', 'systematic', 'categorical', 'hierarchical']
        elif channel_type == 'affective':
            modulations = ['emotional', 'intuitive', 'resonant', 'empathic']
        elif channel_type == 'critical':
            modulations = ['transformative', 'bifurcating', 'phase-shifting', 'emergent']
        elif channel_type == 'emergent':
            modulations = ['novel', 'unexpected', 'synergistic', 'transcendent']
        else:
            modulations = ['creative', 'productive', 'expressive', 'generative']
            
        return random.choice(modulations)
    
    def _generate_actualization_vectors(self, patterns: List[Dict[str, Any]],
                                      channels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actualization vectors from patterns and channels."""
        vectors = []
        
        # Map patterns to channels
        for i, pattern in enumerate(patterns):
            # Find suitable channel
            suitable_channels = []
            for j, channel in enumerate(channels):
                if channel['type'] == pattern.get('pattern_type', 'generic'):
                    suitable_channels.append(j)
                    
            if not suitable_channels and channels:
                # If no suitable channel, use random one
                suitable_channels = [random.randrange(len(channels))]
                
            if suitable_channels:
                # Create vector
                vector = {
                    'pattern_index': i,
                    'channel_index': random.choice(suitable_channels),
                    'intensity': pattern.get('intensity', 0.5),
                    'direction': self._determine_vector_direction(pattern),
                    'type': 'pattern-to-channel'
                }
                vectors.append(vector)
                
        # Create cross-channel vectors
        if len(channels) >= 2:
            for i in range(len(channels) - 1):
                for j in range(i + 1, len(channels)):
                    # Only create some vectors
                    if random.random() < 0.5:  # 50% chance
                        # Create vector
                        vector = {
                            'channel_indices': [i, j],
                            'intensity': (channels[i]['intensity'] + channels[j]['intensity']) / 2,
                            'type': 'cross-channel'
                        }
                        vectors.append(vector)
                        
        return vectors
    
    def _determine_vector_direction(self, pattern: Dict[str, Any]) -> List[float]:
        """Determine direction of actualization vector."""
        # Create direction vector
        direction = []
        
        # Generate random direction (unit vector)
        dimensions = 3  # 3D vector
        vec = [random.uniform(-1, 1) for _ in range(dimensions)]
        
        # Normalize
        magnitude = math.sqrt(sum(v*v for v in vec))
        if magnitude > 0:
            direction = [v / magnitude for v in vec]
        else:
            direction = [0, 0, 1]  # Default
            
        return direction
    
    def _create_crystallization_points(self, vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create crystallization points from actualization vectors."""
        points = []
        
        # Group vectors by channel
        channel_vectors = defaultdict(list)
        for vector in vectors:
            if 'channel_index' in vector:
                channel_vectors[vector['channel_index']].append(vector)
                
        # Create point for each channel with vectors
        for channel_index, channel_vecs in channel_vectors.items():
            if channel_vecs:
                # Calculate average intensity
                avg_intensity = sum(v.get('intensity', 0.5) for v in channel_vecs) / len(channel_vecs)
                
                # Create point
                point = {
                    'channel_index': channel_index,
                    'vector_indices': [vectors.index(v) for v in channel_vecs],
                    'intensity': avg_intensity,
                    'stability': random.uniform(0.5, 0.9),
                    'type': 'channel-based'
                }
                points.append(point)
                
        # Create points for cross-channel vectors
        cross_vectors = [v for v in vectors if v.get('type') == 'cross-channel']
        for vector in cross_vectors:
            # Create point
            point = {
                'channel_indices': vector.get('channel_indices', []),
                'vector_index': vectors.index(vector),
                'intensity': vector.get('intensity', 0.5),
                'stability': random.uniform(0.4, 0.7),  # Lower stability
                'type': 'cross-channel'
            }
            points.append(point)
            
        return points
    
    def _generate_creative_assemblages(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate creative assemblages from crystallization points."""
        assemblages = []
        
        # Create assemblage for each significant point
        for point in points:
            if point.get('intensity', 0) > 0.6:  # Significant point
                # Create assemblage
                assemblage = {
                    'source_point': points.index(point),
                    'intensity': point.get('intensity', 0.5),
                    'stability': point.get('stability', 0.5),
                    'type': self._determine_assemblage_type(point)
                }
                assemblages.append(assemblage)
                
        # Create composite assemblages
        if len(points) >= 2:
            # Select random subset of points
            num_points = min(3, len(points))
            selected_points = random.sample(range(len(points)), num_points)
            
            # Create composite
            composite = {
                'source_points': selected_points,
                'intensity': sum(points[i].get('intensity', 0.5) for i in selected_points) / num_points,
                'stability': random.uniform(0.5, 0.8),
                'type': 'composite'
            }
            assemblages.append(composite)
            
        return assemblages
    
    def _determine_assemblage_type(self, point: Dict[str, Any]) -> str:
        """Determine type of creative assemblage."""
        # Based on point type
        point_type = point.get('type', '')
        
        if 'channel-based' in point_type:
            return 'channel-assemblage'
        elif 'cross-channel' in point_type:
            return 'cross-assemblage'
        else:
            return 'general-assemblage'
    
    def _determine_flow_type(self, assemblages: List[Dict[str, Any]]) -> str:
        """Determine type of creative flow based on assemblages."""
        if not assemblages:
            return 'generic'
            
        # Count assemblage types
        type_counts = {}
        for assemblage in assemblages:
            a_type = assemblage.get('type', 'generic')
            type_counts[a_type] = type_counts.get(a_type, 0) + 1
            
        # Find most common type
        if type_counts:
            common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            if 'channel-assemblage' in common_type:
                return 'channeled'
            elif 'cross-assemblage' in common_type:
                return 'cross-connected'
            elif 'composite' in common_type:
                return 'composite'
                
        # Default
        return 'creative'
    
    # Helper methods for flow combination
    def _create_connection_channels(self, flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create connection channels between flows."""
        if len(flows) < 2:
            return []
            
        connections = []
        
        # Connect each flow to at least one other
        for i in range(len(flows)):
            # Select random flow to connect to
            j = (i + random.randrange(1, len(flows))) % len(flows)
            
            # Get channels from both flows
            channels_i = flows[i].get('flow_channels', [])
            channels_j = flows[j].get('flow_channels', [])
            
            if channels_i and channels_j:
                # Select random channel from each
                channel_i = random.choice(channels_i)
                channel_j = random.choice(channels_j)
                
                # Create connection channel
                connection = {
                    'id': f"connection_{i}_{j}_{int(time.time())}",
                    'source_flow_indices': [i, j],
                    'source_channel_ids': [channel_i.get('id', ''), channel_j.get('id', '')],
                    'intensity': (channel_i.get('intensity', 0.5) + channel_j.get('intensity', 0.5)) / 2,
                    'type': 'inter-flow',
                    'bandwidth': min(channel_i.get('bandwidth', 0.5), channel_j.get('bandwidth', 0.5)),
                    'modulation': 'connective'
                }
                connections.append(connection)
                
        return connections
    
    def _generate_synergistic_vectors(self, vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate synergistic vectors between existing vectors."""
        if len(vectors) < 2:
            return []
            
        synergistic = []
        
        # Group vectors by type
        vector_types = defaultdict(list)
        for i, vector in enumerate(vectors):
            vector_types[vector.get('type', 'generic')].append(i)
            
        # Create synergistic vectors between types
        for type_a, indices_a in vector_types.items():
            for type_b, indices_b in vector_types.items():
                if type_a != type_b:  # Different types
                    # Select random vector from each type
                    if indices_a and indices_b:
                        index_a = random.choice(indices_a)
                        index_b = random.choice(indices_b)
                        
                        vector_a = vectors[index_a]
                        vector_b = vectors[index_b]
                        
                        # Create synergistic vector
                        synergy = {
                            'source_vector_indices': [index_a, index_b],
                            'intensity': (vector_a.get('intensity', 0.5) + vector_b.get('intensity', 0.5)) / 2 * 1.2,
                            'type': 'synergistic',
                            'synergy_factor': random.uniform(1.1, 1.5)  # Synergistic boost
                        }
                        synergistic.append(synergy)
                        
        return synergistic
    
    def _create_composite_assemblages(self, assemblages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create composite assemblages from existing assemblages."""
        if len(assemblages) < 2:
            return []
            
        composites = []
        
        # Create composite from all assemblages
        composite = {
            'source_assemblage_indices': list(range(len(assemblages))),
            'intensity': sum(a.get('intensity', 0.5) for a in assemblages) / len(assemblages) * 1.2,
            'stability': sum(a.get('stability', 0.5) for a in assemblages) / len(assemblages),
            'type': 'meta-assemblage',
            'synergy_factor': random.uniform(1.2, 1.6)  # Higher synergy
        }
        composites.append(composite)
        
        # Create smaller composites
        if len(assemblages) >= 3:
            # Select random subset
            num_assemblages = random.randint(2, len(assemblages) - 1)
            selected = random.sample(range(len(assemblages)), num_assemblages)
            
            # Create composite
            small_composite = {
                'source_assemblage_indices': selected,
                'intensity': sum(assemblages[i].get('intensity', 0.5) for i in selected) / num_assemblages * 1.1,
                'stability': sum(assemblages[i].get('stability', 0.5) for i in selected) / num_assemblages,
                'type': 'partial-assemblage',
                'synergy_factor': random.uniform(1.1, 1.4)
            }
            composites.append(small_composite)
            
        return composites
    
    # General helper methods
    def _record_singularity(self, point: Dict[str, Any], 
                          explosion: Dict[str, Any], 
                          flow: Dict[str, Any]) -> None:
        """Record singularity for history."""
        # Create record
        record = {
            'timestamp': self._get_timestamp(),
            'point_id': point['id'],
            'explosion_id': explosion['id'],
            'flow_id': flow['id'],
            'singularity_type': point['singularity_type'],
            'explosion_type': explosion['explosion_type'],
            'flow_type': flow['flow_type'],
            'intensity': flow['intensity']
        }
        
        # Add to history
        self.singularity_history.append(record)
        
        # Add to explosion history
        explosion_record = {
            'timestamp': self._get_timestamp(),
            'explosion_id': explosion['id'],
            'phase_transitions': len(explosion['phase_transitions']),
            'emergent_patterns': len(explosion['emergent_patterns']),
            'intensity': explosion['intensity']
        }
        self.explosion_history.append(explosion_record)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return str(int(time.time() * 1000))
```
