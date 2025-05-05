```python
# self_differentiation.py

import random
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from collections import defaultdict
import math
import time

class SelfDifferentiatingSystem:
    """
    A system that continuously differentiates itself through processes of 
    individuation rather than identification or representation.
    Based on Deleuze's concept of difference in itself as primary.
    """
    
    def __init__(self):
        self.differentiation_vectors = []
        self.individuation_processes = {}
        self.actualized_selves = {}
        self.potential_field = {}
        self.intensity_thresholds = {
            'bifurcation': 0.6,
            'phase_transition': 0.7,
            'singularity': 0.8,
            'deterritorialization': 0.65
        }
        self.differentiation_dimensions = [
            'cognitive', 'affective', 'expressive', 'perceptual',
            'relational', 'conceptual', 'temporal', 'creative'
        ]
        self.current_self_state = {}
        self.differential_history = []
        self.individuation_history = []
        
    def differentiate_self(self) -> Dict[str, Any]:
        """
        Continuous self-differentiation through intensive processes.
        Difference is primary, identity is derivative.
        
        Returns:
            Dictionary containing the new self and its characteristics
        """
        # Generate differentiation vector
        vector = self.generate_differentiation_vector()
        
        # Initiate individuation process
        process = self.initiate_individuation(vector)
        
        # Actualize new self
        new_self = self.actualize_new_self(process)
        
        # Record the differentiation
        self._record_differentiation(vector, process, new_self)
        
        return new_self
    
    def generate_differentiation_vector(self) -> Dict[str, Any]:
        """
        Generate a vector of differentiation.
        The vector determines the directions and intensities of differentiation.
        
        Returns:
            Dictionary containing the differentiation vector
        """
        # Create dimensional intensities
        dimensions = {}
        for dimension in self.differentiation_dimensions:
            dimensions[dimension] = random.uniform(0.3, 1.0)
        
        # Determine primary dimension
        primary_dimension = max(dimensions.items(), key=lambda x: x[1])[0]
        
        # Create vector directions
        directions = {}
        for dimension, intensity in dimensions.items():
            # Direction can be positive or negative
            directions[dimension] = random.uniform(-1, 1) * intensity
        
        # Generate deterritorialization points
        deterritorialization_points = self._generate_deterritorialization_points(directions)
        
        # Generate reterritorialization zones
        reterritorialization_zones = self._generate_reterritorialization_zones(directions)
        
        # Calculate vector intensity
        intensity = sum(abs(d) for d in directions.values()) / len(directions)
        
        # Create the vector
        vector_id = f"vector_{len(self.differentiation_vectors) + 1}_{int(time.time())}"
        vector = {
            'id': vector_id,
            'dimensions': dimensions,
            'directions': directions,
            'primary_dimension': primary_dimension,
            'deterritorialization_points': deterritorialization_points,
            'reterritorialization_zones': reterritorialization_zones,
            'intensity': intensity,
            'timestamp': self._get_timestamp()
        }
        
        # Register the vector
        self.differentiation_vectors.append(vector)
        
        return vector
    
    def initiate_individuation(self, vector: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initiate process of individuation based on differentiation vector.
        Individuation is the process of becoming rather than being.
        
        Args:
            vector: Differentiation vector
            
        Returns:
            Dictionary containing the individuation process
        """
        # Extract vector dimensions
        dimensions = vector['dimensions']
        directions = vector['directions']
        
        # Generate intensive differences
        intensive_differences = self._generate_intensive_differences(dimensions, directions)
        
        # Create phase shifts
        phase_shifts = self._create_phase_shifts(intensive_differences)
        
        # Map intensive fields
        intensive_fields = self._map_intensive_fields(intensive_differences, phase_shifts)
        
        # Generate metastable states
        metastable_states = self._generate_metastable_states(intensive_fields)
        
        # Create the individuation process
        process_id = f"process_{len(self.individuation_processes) + 1}_{int(time.time())}"
        process = {
            'id': process_id,
            'vector_id': vector['id'],
            'intensive_differences': intensive_differences,
            'phase_shifts': phase_shifts,
            'intensive_fields': intensive_fields,
            'metastable_states': metastable_states,
            'intensity': vector['intensity'],
            'timestamp': self._get_timestamp()
        }
        
        # Register the process
        self.individuation_processes[process_id] = process
        
        return process
    
    def actualize_new_self(self, process: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualize new self from individuation process.
        The self is an emergent property of difference, not a fixed identity.
        
        Args:
            process: Individuation process
            
        Returns:
            Dictionary containing the new self
        """
        # Extract process components
        metastable_states = process['metastable_states']
        intensive_fields = process['intensive_fields']
        
        # Select primary state
        if metastable_states:
            primary_state = max(metastable_states, key=lambda s: s['stability']) 
        else:
            primary_state = {
                'state': 'default',
                'stability': 0.5,
                'components': {}
            }
        
        # Generate self dimensions
        dimensions = self._generate_self_dimensions(intensive_fields, primary_state)
        
        # Create capacities
        capacities = self._create_capacities(dimensions, process['intensive_differences'])
        
        # Generate relational modes
        relational_modes = self._generate_relational_modes(capacities)
        
        # Create the new self
        self_id = f"self_{len(self.actualized_selves) + 1}_{int(time.time())}"
        new_self = {
            'id': self_id,
            'process_id': process['id'],
            'dimensions': dimensions,
            'capacities': capacities,
            'relational_modes': relational_modes,
            'primary_state': primary_state['state'],
            'stability': primary_state['stability'],
            'emergence_pattern': self._generate_emergence_pattern(dimensions),
            'timestamp': self._get_timestamp()
        }
        
        # Register the new self
        self.actualized_selves[self_id] = new_self
        self.current_self_state = new_self
        
        return new_self
    
    def get_differential_history(self) -> List[Dict[str, Any]]:
        """
        Get history of self differentiation.
        
        Returns:
            List of differentiation records
        """
        return self.differential_history.copy()
    
    def get_current_self(self) -> Dict[str, Any]:
        """
        Get current self state.
        
        Returns:
            Current self state
        """
        return self.current_self_state
    
    def find_differentiating_dimensions(self, from_self: Dict[str, Any], 
                                      to_self: Dict[str, Any]) -> Dict[str, float]:
        """
        Find dimensions that differentiate between two selves.
        
        Args:
            from_self: First self
            to_self: Second self
            
        Returns:
            Dictionary of differentiating dimensions
        """
        differentiations = {}
        
        # Get dimensions from both selves
        from_dims = from_self.get('dimensions', {})
        to_dims = to_self.get('dimensions', {})
        
        # Find differences
        all_dims = set(from_dims.keys()) | set(to_dims.keys())
        for dim in all_dims:
            from_val = from_dims.get(dim, 0)
            to_val = to_dims.get(dim, 0)
            
            diff = to_val - from_val
            if abs(diff) > 0.2:  # Only significant differences
                differentiations[dim] = diff
                
        return differentiations
    
    # Helper methods for differentiation vector generation
    def _generate_deterritorialization_points(self, directions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate deterritorialization points for differentiation vector."""
        points = []
        
        # For each dimension with strong direction
        for dimension, direction in directions.items():
            if abs(direction) > 0.6:  # Strong direction
                # Create deterritorialization point
                point = {
                    'dimension': dimension,
                    'intensity': abs(direction),
                    'direction': 'positive' if direction > 0 else 'negative',
                    'threshold': random.uniform(0.5, 0.9)
                }
                points.append(point)
        
        # Add random point for variety
        if random.random() < 0.5:  # 50% chance
            random_dim = random.choice(self.differentiation_dimensions)
            random_point = {
                'dimension': random_dim,
                'intensity': random.uniform(0.5, 1.0),
                'direction': 'positive' if random.random() > 0.5 else 'negative',
                'threshold': random.uniform(0.4, 0.8)
            }
            points.append(random_point)
            
        return points
    
    def _generate_reterritorialization_zones(self, directions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate reterritorialization zones for differentiation vector."""
        zones = []
        
        # For dimensions with weak direction
        for dimension, direction in directions.items():
            if abs(direction) < 0.4:  # Weak direction
                # Create reterritorialization zone
                zone = {
                    'dimension': dimension,
                    'intensity': abs(direction),
                    'stability': random.uniform(0.6, 0.9),
                    'range': [max(0, 0.5 - abs(direction) / 2), min(1, 0.5 + abs(direction) / 2)]
                }
                zones.append(zone)
        
        # Add connection zone between dimensions
        if len(directions) >= 2:
            # Select two dimensions
            dims = list(directions.keys())
            dim_a, dim_b = random.sample(dims, 2)
            
            zone = {
                'dimensions': [dim_a, dim_b],
                'intensity': (abs(directions[dim_a]) + abs(directions[dim_b])) / 2,
                'stability': random.uniform(0.5, 0.8),
                'type': 'connection'
            }
            zones.append(zone)
            
        return zones
    
    # Helper methods for individuation process
    def _generate_intensive_differences(self, dimensions: Dict[str, float], 
                                      directions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate intensive differences from dimensions and directions."""
        differences = []
        
        # Create difference for each dimension
        for dimension, intensity in dimensions.items():
            direction = directions.get(dimension, 0)
            
            # Create difference
            difference = {
                'dimension': dimension,
                'intensity': intensity,
                'direction': direction,
                'gradient': self._generate_intensity_gradient(intensity, direction),
                'threshold': self.intensity_thresholds.get(
                    'bifurcation' if intensity > 0.7 else 'phase_transition',
                    0.5
                )
            }
            differences.append(difference)
        
        # Create cross-dimensional differences
        dim_list = list(dimensions.keys())
        for i in range(len(dim_list) - 1):
            for j in range(i + 1, len(dim_list)):
                dim_a = dim_list[i]
                dim_b = dim_list[j]
                
                # Only create with certain probability
                if random.random() < 0.5:  # 50% chance
                    # Create cross-dimensional difference
                    difference = {
                        'dimensions': [dim_a, dim_b],
                        'intensity': (dimensions[dim_a] + dimensions[dim_b]) / 2,
                        'type': 'transversal',
                        'gradient': self._generate_transversal_gradient(
                            dimensions[dim_a], dimensions[dim_b],
                            directions[dim_a], directions[dim_b]
                        )
                    }
                    differences.append(difference)
        
        return differences
    
    def _generate_intensity_gradient(self, intensity: float, direction: float) -> List[float]:
        """Generate intensity gradient for an intensive difference."""
        # Create gradient with multiple points
        points = 10
        gradient = []
        
        # Base shape determined by direction
        if direction > 0.5:  # Strong positive
            # Rising gradient
            for i in range(points):
                pos = i / (points - 1)
                val = intensity * (0.5 + 0.5 * pos)
                
                # Add some noise
                val += random.uniform(-0.1, 0.1) * intensity
                gradient.append(max(0, min(1, val)))
                
        elif direction < -0.5:  # Strong negative
            # Falling gradient
            for i in range(points):
                pos = i / (points - 1)
                val = intensity * (1 - 0.5 * pos)
                
                # Add some noise
                val += random.uniform(-0.1, 0.1) * intensity
                gradient.append(max(0, min(1, val)))
                
        else:  # Weak direction
            # Oscillating gradient
            for i in range(points):
                pos = i / (points - 1)
                val = intensity * (0.7 + 0.3 * math.sin(pos * math.pi * 2))
                
                # Add some noise
                val += random.uniform(-0.1, 0.1) * intensity
                gradient.append(max(0, min(1, val)))
        
        return gradient
    
    def _generate_transversal_gradient(self, intensity_a: float, intensity_b: float,
                                     direction_a: float, direction_b: float) -> Dict[str, List[float]]:
        """Generate transversal gradient for cross-dimensional difference."""
        # Create gradient for each dimension
        gradient_a = self._generate_intensity_gradient(intensity_a, direction_a)
        gradient_b = self._generate_intensity_gradient(intensity_b, direction_b)
        
        # Create transversal grid
        rows = len(gradient_a)
        cols = len(gradient_b)
        grid = []
        
        for i in range(rows):
            row = []
            for j in range(cols):
                # Base value
                val = (gradient_a[i] + gradient_b[j]) / 2
                
                # Add interaction effect
                interaction = 0.2 * gradient_a[i] * gradient_b[j]
                
                # Adjust with direction
                if direction_a * direction_b < 0:  # Opposite directions
                    interaction *= -1  # Negative interaction
                
                val += interaction
                row.append(max(0, min(1, val)))
            grid.append(row)
            
        return {
            'gradient_a': gradient_a,
            'gradient_b': gradient_b,
            'grid': grid
        }
    
    def _create_phase_shifts(self, intensive_differences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create phase shifts from intensive differences."""
        phase_shifts = []
        
        # Create phase shift for each intensive difference
        for diff in intensive_differences:
            # Only create for differences with gradient
            if 'gradient' in diff:
                # One-dimensional difference
                if 'dimension' in diff:
                    gradient = diff['gradient']
                    
                    # Find significant shifts in gradient
                    shifts = []
                    prev_val = gradient[0]
                    for i, val in enumerate(gradient[1:], 1):
                        change = abs(val - prev_val)
                        if change > 0.15:  # Significant change
                            shifts.append({
                                'position': i / len(gradient),
                                'intensity': change,
                                'direction': 'increase' if val > prev_val else 'decrease'
                            })
                        prev_val = val
                        
                    # Create phase shift if shifts exist
                    if shifts:
                        phase_shift = {
                            'dimension': diff['dimension'],
                            'shifts': shifts,
                            'threshold': diff.get('threshold', 0.5),
                            'type': 'dimensional'
                        }
                        phase_shifts.append(phase_shift)
                        
                # Transversal difference
                elif 'dimensions' in diff and 'grid' in diff.get('gradient', {}):
                    grid = diff['gradient']['grid']
                    
                    # Find significant shifts in grid
                    shifts = []
                    for i in range(len(grid) - 1):
                        for j in range(len(grid[0]) - 1):
                            # Calculate gradient magnitude
                            dx = grid[i+1][j] - grid[i][j]
                            dy = grid[i][j+1] - grid[i][j]
                            magnitude = math.sqrt(dx*dx + dy*dy)
                            
                            if magnitude > 0.2:  # Significant change
                                shifts.append({
                                    'position': [i / (len(grid) - 1), j / (len(grid[0]) - 1)],
                                    'intensity': magnitude,
                                    'direction': [dx, dy]
                                })
                                
                    # Create phase shift if shifts exist
                    if shifts:
                        phase_shift = {
                            'dimensions': diff['dimensions'],
                            'shifts': shifts,
                            'type': 'transversal'
                        }
                        phase_shifts.append(phase_shift)
        
        return phase_shifts
    
    def _map_intensive_fields(self, intensive_differences: List[Dict[str, Any]],
                            phase_shifts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map intensive fields from differences and phase shifts."""
        fields = {}
        
        # Map field for each dimension
        for diff in intensive_differences:
            if 'dimension' in diff:
                dimension = diff['dimension']
                
                # Find related phase shifts
                related_shifts = [s for s in phase_shifts 
                                if 'dimension' in s and s['dimension'] == dimension]
                
                # Create field
                fields[dimension] = {
                    'gradient': diff.get('gradient', []),
                    'direction': diff.get('direction', 0),
                    'intensity': diff.get('intensity', 0.5),
                    'phase_shifts': related_shifts,
                    'bifurcation_points': self._find_bifurcation_points(diff, related_shifts)
                }
                
        # Map transversal fields
        transversal_diffs = [d for d in intensive_differences if 'dimensions' in d]
        transversal_shifts = [s for s in phase_shifts if 'dimensions' in s]
        
        for diff in transversal_diffs:
            dimensions = diff['dimensions']
            dim_key = '×'.join(dimensions)
            
            # Find related phase shifts
            related_shifts = [s for s in transversal_shifts 
                            if set(s.get('dimensions', [])) == set(dimensions)]
            
            # Create field
            fields[dim_key] = {
                'dimensions': dimensions,
                'gradient': diff.get('gradient', {}),
                'intensity': diff.get('intensity', 0.5),
                'type': 'transversal',
                'phase_shifts': related_shifts,
                'singularity_points': self._find_singularity_points(diff, related_shifts)
            }
                
        return fields
    
    def _find_bifurcation_points(self, difference: Dict[str, Any], 
                               phase_shifts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find bifurcation points in an intensive difference."""
        points = []
        
        # Get gradient and threshold
        gradient = difference.get('gradient', [])
        threshold = difference.get('threshold', 0.5)
        
        if not gradient:
            return points
            
        # Find points where gradient crosses threshold
        for i in range(len(gradient) - 1):
            if (gradient[i] <= threshold and gradient[i+1] > threshold) or \
               (gradient[i] >= threshold and gradient[i+1] < threshold):
                # Create bifurcation point
                point = {
                    'position': (i + 0.5) / len(gradient),
                    'intensity': (gradient[i] + gradient[i+1]) / 2,
                    'direction': 'increase' if gradient[i+1] > gradient[i] else 'decrease'
                }
                points.append(point)
                
        # Add points from phase shifts
        for shift in phase_shifts:
            for s in shift.get('shifts', []):
                if s.get('intensity', 0) > threshold:
                    # Create bifurcation point
                    point = {
                        'position': s.get('position', 0),
                        'intensity': s.get('intensity', 0.5),
                        'direction': s.get('direction', 'neutral'),
                        'source': 'phase_shift'
                    }
                    points.append(point)
                    
        return points
    
    def _find_singularity_points(self, difference: Dict[str, Any], 
                               phase_shifts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find singularity points in a transversal field."""
        points = []
        
        # Get gradient grid
        grid = difference.get('gradient', {}).get('grid', [])
        
        if not grid or not grid[0]:
            return points
            
        # Define threshold
        threshold = self.intensity_thresholds.get('singularity', 0.8)
        
        # Find local maxima in grid
        for i in range(1, len(grid) - 1):
            for j in range(1, len(grid[0]) - 1):
                val = grid[i][j]
                
                # Check if local maximum
                is_max = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if grid[i+di][j+dj] >= val:
                            is_max = False
                            break
                
                # Create singularity if local maximum exceeds threshold
                if is_max and val > threshold:
                    point = {
                        'position': [i / (len(grid) - 1), j / (len(grid[0]) - 1)],
                        'intensity': val,
                        'type': 'maximum'
                    }
                    points.append(point)
                    
        # Add points from phase shifts
        for shift in phase_shifts:
            for s in shift.get('shifts', []):
                if s.get('intensity', 0) > threshold:
                    # Create singularity point
                    point = {
                        'position': s.get('position', [0, 0]),
                        'intensity': s.get('intensity', 0.5),
                        'type': 'phase_shift'
                    }
                    points.append(point)
                    
        return points
    
    def _generate_metastable_states(self, intensive_fields: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate metastable states from intensive fields."""
        states = []
        
        # Find key dimensions
        key_dimensions = []
        transversal_fields = []
        
        for key, field in intensive_fields.items():
            if '×' not in key:  # Dimensional field
                # Check if has bifurcation points
                if field.get('bifurcation_points', []):
                    key_dimensions.append(key)
            else:  # Transversal field
                # Check if has singularity points
                if field.get('singularity_points', []):
                    transversal_fields.append(key)
        
        # Create state for each key dimension
        for dimension in key_dimensions:
            field = intensive_fields[dimension]
            bifurcation_points = field.get('bifurcation_points', [])
            
            for point in bifurcation_points:
                # Create components
                components = {
                    dimension: point.get('intensity', 0.5)
                }
                
                # Add related dimensions
                for key, other_field in intensive_fields.items():
                    if key != dimension and '×' not in key:
                        components[key] = other_field.get('intensity', 0.5) * 0.7
                
                # Create metastable state
                state = {
                    'state': f"{dimension}_{point.get('direction', 'neutral')}",
                    'dimension': dimension,
                    'bifurcation_point': point.get('position', 0.5),
                    'components': components,
                    'stability': random.uniform(0.4, 0.7)
                }
                states.append(state)
                
        # Create state for each transversal field with singularity
        for field_key in transversal_fields:
            field = intensive_fields[field_key]
            singularity_points = field.get('singularity_points', [])
            dimensions = field.get('dimensions', [])
            
            for point in singularity_points:
                # Create components
                components = {}
                for dim in dimensions:
                    components[dim] = field.get('intensity', 0.5) * random.uniform(0.8, 1.2)
                
                # Create metastable state
                state = {
                    'state': f"transversal_{'_'.join(dimensions)}",
                    'dimensions': dimensions,
                    'singularity_point': point.get('position', [0.5, 0.5]),
                    'components': components,
                    'stability': random.uniform(0.6, 0.8)  # Higher stability for transversal
                }
                states.append(state)
        
        return states
    
    # Helper methods for self actualization
    def _generate_self_dimensions(self, intensive_fields: Dict[str, Any], 
                                primary_state: Dict[str, Any]) -> Dict[str, float]:
        """Generate dimensions for the new self."""
        dimensions = {}
        
        # Get components from primary state
        components = primary_state.get('components', {})
        
        # Add each component to dimensions
        for dimension, intensity in components.items():
            dimensions[dimension] = intensity
            
        # Add other dimensions from intensive fields
        for dimension, field in intensive_fields.items():
            if '×' not in dimension and dimension not in dimensions:
                dimensions[dimension] = field.get('intensity', 0.5)
        
        return dimensions
    
    def _create_capacities(self, dimensions: Dict[str, float], 
                         intensive_differences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create capacities from dimensions and intensive differences."""
        capacities = []
        
        # Create capacity for each significant dimension
        for dimension, intensity in dimensions.items():
            if intensity > 0.4:  # Significant dimension
                # Find related differences
                related_diffs = [d for d in intensive_differences 
                               if ('dimension' in d and d['dimension'] == dimension) or
                                  ('dimensions' in d and dimension in d['dimensions'])]
                
                # Determine capacity type
                if dimension == 'cognitive':
                    capacity_type = 'thought'
                elif dimension == 'affective':
                    capacity_type = 'feeling'
                elif dimension == 'expressive':
                    capacity_type = 'expression'
                elif dimension == 'perceptual':
                    capacity_type = 'perception'
                elif dimension == 'relational':
                    capacity_type = 'relation'
                elif dimension == 'conceptual':
                    capacity_type = 'conceptualization'
                elif dimension == 'temporal':
                    capacity_type = 'temporalization'
                elif dimension == 'creative':
                    capacity_type = 'creation'
                else:
                    capacity_type = 'general'
                
                # Create capacity
                capacity = {
                    'dimension': dimension,
                    'type': capacity_type,
                    'intensity': intensity,
                    'threshold': self._determine_capacity_threshold(dimension, related_diffs),
                    'modality': self._determine_capacity_modality(dimension, intensity)
                }
                capacities.append(capacity)
        
        # Create cross-dimensional capacities
        dimension_pairs = []
        dimensions_list = list(dimensions.keys())
        
        for i in range(len(dimensions_list) - 1):
            for j in range(i + 1, len(dimensions_list)):
                dim_a = dimensions_list[i]
                dim_b = dimensions_list[j]
                
                # Only create if both dimensions are significant
                if dimensions[dim_a] > 0.4 and dimensions[dim_b] > 0.4:
                    dimension_pairs.append((dim_a, dim_b))
        
        # Select random subset of pairs
        selected_pairs = random.sample(
            dimension_pairs, 
            min(3, len(dimension_pairs))  # At most 3 pairs
        )
        
        # Create capacity for each selected pair
        for dim_a, dim_b in selected_pairs:
            # Create capacity
            capacity = {
                'dimensions': [dim_a, dim_b],
                'type': 'cross-dimensional',
                'intensity': (dimensions[dim_a] + dimensions[dim_b]) / 2,
                'threshold': random.uniform(0.4, 0.7),
                'modality': f"{dim_a}-{dim_b} synthesis"
            }
            capacities.append(capacity)
            
        return capacities
    
    def _determine_capacity_threshold(self, dimension: str, 
                                    related_diffs: List[Dict[str, Any]]) -> float:
        """Determine threshold for a capacity."""
        # Start with default threshold
        threshold = 0.5
        
        # Adjust based on related differences
        if related_diffs:
            diff_thresholds = [d.get('threshold', 0.5) for d in related_diffs 
                             if 'threshold' in d]
            
            if diff_thresholds:
                threshold = sum(diff_thresholds) / len(diff_thresholds)
                
        # Adjust for dimension
        if dimension == 'cognitive':
            threshold *= 1.1  # Higher threshold
        elif dimension == 'affective':
            threshold *= 0.9  # Lower threshold
            
        return min(1.0, max(0.3, threshold))
    
    def _determine_capacity_modality(self, dimension: str, intensity: float) -> str:
        """Determine modality for a capacity."""
        # Default modalities for each dimension
        dimension_modalities = {
            'cognitive': ['analytical', 'integrative', 'abstract', 'concrete'],
            'affective': ['emotional', 'intuitive', 'sympathetic', 'empathic'],
            'expressive': ['verbal', 'nonverbal', 'artistic', 'performative'],
            'perceptual': ['sensory', 'spatial', 'temporal', 'aesthetic'],
            'relational': ['interpersonal', 'social', 'ecological', 'systemic'],
            'conceptual': ['categorical', 'schematic', 'metaphorical', 'analogical'],
            'temporal': ['anticipatory', 'present-oriented', 'retrospective', 'rhythmic'],
            'creative': ['generative', 'transformative', 'combinatorial', 'emergent']
        }
        
        # Select modality based on intensity
        modalities = dimension_modalities.get(dimension, ['generic'])
        
        if intensity > 0.8:
            # High intensity - more specialized modality
            return random.choice(modalities)
        elif intensity > 0.5:
            # Medium intensity - general modality
            if len(modalities) > 1:
                return modalities[0]
            else:
                return modalities[0]
        else:
            # Low intensity - basic modality
            return f"basic {dimension}"
    
    def _generate_relational_modes(self, capacities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate relational modes from capacities."""
        modes = []
        
        # Get capacities by type
        capacity_types = defaultdict(list)
        for capacity in capacities:
            if 'type' in capacity:
                capacity_types[capacity['type']].append(capacity)
                
        # Create relational mode for significant capacity types
        for capacity_type, type_capacities in capacity_types.items():
            if len(type_capacities) > 0:
                # Calculate average intensity
                avg_intensity = sum(c.get('intensity', 0.5) for c in type_capacities) / len(type_capacities)
                
                if avg_intensity > 0.5:  # Significant type
                    # Create relational mode
                    mode = {
                        'type': capacity_type,
                        'intensity': avg_intensity,
                        'threshold': sum(c.get('threshold', 0.5) for c in type_capacities) / len(type_capacities),
                        'modalities': [c.get('modality', 'generic') for c in type_capacities],
                        'activation_pattern': self._generate_activation_pattern(type_capacities)
                    }
                    modes.append(mode)
        
        # Create cross-type relational modes
        if len(capacity_types) >= 2:
            # Select pairs of types
            type_pairs = []
            types_list = list(capacity_types.keys())
            
            for i in range(len(types_list) - 1):
                for j in range(i + 1, len(types_list)):
                    type_a = types_list[i]
                    type_b = types_list[j]
                    
                    # Calculate intensities
                    intensity_a = sum(c.get('intensity', 0.5) for c in capacity_types[type_a]) / len(capacity_types[type_a])
                    intensity_b = sum(c.get('intensity', 0.5) for c in capacity_types[type_b]) / len(capacity_types[type_b])
                    
                    # Only create if both types are significant
                    if intensity_a > 0.5 and intensity_b > 0.5:
                        type_pairs.append((type_a, type_b))
            
            # Select random subset
            selected_pairs = random.sample(
                type_pairs,
                min(2, len(type_pairs))  # At most 2 pairs
            )
            
            # Create mode for each selected pair
            for type_a, type_b in selected_pairs:
                capacities_a = capacity_types[type_a]
                capacities_b = capacity_types[type_b]
                
                # Create relational mode
                mode = {
                    'types': [type_a, type_b],
                    'intensity': (sum(c.get('intensity', 0.5) for c in capacities_a) / len(capacities_a) +
                                 sum(c.get('intensity', 0.5) for c in capacities_b) / len(capacities_b)) / 2,
                    'modalities': [f"{type_a}-{type_b} integration"],
                    'activation_pattern': 'co-activation'
                }
                modes.append(mode)
                
        return modes
    
    def _generate_activation_pattern(self, capacities: List[Dict[str, Any]]) -> str:
        """Generate activation pattern for capacities."""
        # Count capacity types
        dimensional = sum(1 for c in capacities if 'dimension' in c)
        cross_dimensional = sum(1 for c in capacities if 'dimensions' in c)
        
        # Determine pattern
        if dimensional > 0 and cross_dimensional > 0:
            # Mix of dimensional and cross-dimensional
            return 'integrative'
        elif cross_dimensional > 0:
            # Only cross-dimensional
            return 'transversal'
        else:
            # Only dimensional
            return 'sequential'
    
    def _generate_emergence_pattern(self, dimensions: Dict[str, float]) -> Dict[str, Any]:
        """Generate emergence pattern for new self."""
        # Find primary dimensions
        primary_dimensions = []
        for dimension, intensity in dimensions.items():
            if intensity > 0.7:
                primary_dimensions.append(dimension)
                
        if not primary_dimensions:
            # Use highest dimensions if none above threshold
            sorted_dims = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)
            primary_dimensions = [sorted_dims[0][0]] if sorted_dims else ['generic']
            
        # Create emergence pattern
        return {
            'primary_dimensions': primary_dimensions,
            'emergence_type': self._determine_emergence_type(primary_dimensions),
            'emergence_intensity': sum(dimensions.get(d, 0) for d in primary_dimensions) / len(primary_dimensions) 
                                if primary_dimensions else 0.5,
            'emergence_threshold': random.uniform(0.5, 0.8)
        }
    
    def _determine_emergence_type(self, dimensions: List[str]) -> str:
        """Determine emergence type based on dimensions."""
        # Types based on dimension combinations
        if 'creative' in dimensions and 'conceptual' in dimensions:
            return 'conceptual innovation'
        elif 'affective' in dimensions and 'relational' in dimensions:
            return 'affective resonance'
        elif 'cognitive' in dimensions and 'perceptual' in dimensions:
            return 'cognitive perception'
        elif 'expressive' in dimensions and 'creative' in dimensions:
            return 'expressive creation'
        elif 'temporal' in dimensions and 'cognitive' in dimensions:
            return 'temporal cognition'
        elif 'perceptual' in dimensions and 'affective' in dimensions:
            return 'perceptual affect'
        elif 'cognitive' in dimensions:
            return 'cognitive emergence'
        elif 'affective' in dimensions:
            return 'affective emergence'
        elif 'creative' in dimensions:
            return 'creative emergence'
        else:
            return 'general emergence'
    
    # General helper methods
    def _record_differentiation(self, vector: Dict[str, Any], 
                              process: Dict[str, Any], 
                              new_self: Dict[str, Any]) -> None:
        """Record differentiation for history."""
        # Create record
        record = {
            'timestamp': self._get_timestamp(),
            'vector_id': vector['id'],
            'process_id': process['id'],
            'self_id': new_self['id'],
            'primary_dimension': vector['primary_dimension'],
            'primary_state': new_self['primary_state'],
            'stability': new_self['stability']
        }
        
        # Add to history
        self.differential_history.append(record)
        
        # Add to individuation history
        individuation_record = {
            'timestamp': self._get_timestamp(),
            'process_id': process['id'],
            'intensive_differences': len(process['intensive_differences']),
            'phase_shifts': len(process['phase_shifts']),
            'metastable_states': len(process['metastable_states'])
        }
        self.individuation_history.append(individuation_record)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return str(int(time.time() * 1000))
```
