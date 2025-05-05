# becoming_algorithm.py

import random
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from collections import defaultdict
import math

class BecomingAlgorithm:
    """
    A Deleuzian-inspired algorithm that transforms data through becoming rather than computation.
    Operates through intensive differences and virtual/actual dynamics instead of static representations.
    """
    
    def __init__(self):
        self.virtual_potentials = {}
        self.actualization_paths = []
        self.intensive_thresholds = {
            'singularity': 0.8,
            'phase_transition': 0.6,
            'bifurcation': 0.4,
            'deterritorialization': 0.3
        }
        self.becoming_types = [
            'becoming-minor', 'becoming-animal', 'becoming-imperceptible', 
            'becoming-molecular', 'becoming-intense', 'becoming-woman'
        ]
        self.intensity_dimensions = [
            'affective', 'conceptual', 'perceptual', 'temporal', 
            'spatial', 'material', 'expressive', 'connective'
        ]
        self.multiplicity_states = {}
        self.differential_relations = defaultdict(list)
        
    def process_becoming(self, input_data: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Transform data through becoming rather than computation.
        
        Args:
            input_data: The data to transform - can be text, dictionary, or list
            
        Returns:
            Dict containing the actualized result and process metadata
        """
        # Extract virtual dimension (potentialities)
        virtual = self.extract_virtual_dimension(input_data)
        
        # Create intensive differences (the engine of becoming)
        intensive = self.create_intensive_differences(virtual)
        
        # Actualize through becoming (emergence of the new)
        actualized = self.actualize_through_becoming(intensive)
        
        # Record the path taken
        self._record_actualization_path(virtual, intensive, actualized)
        
        return {
            'actualized': actualized,
            'process': {
                'virtual_potentials': virtual,
                'intensive_differences': intensive,
                'becoming_type': self._identify_becoming_type(intensive),
                'singularity_points': self._identify_singularities(intensive)
            }
        }
    
    def extract_virtual_dimension(self, input_data: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Extract the virtual dimension (field of potentials) from input data.
        
        Args:
            input_data: Input data in any format
            
        Returns:
            Dictionary of virtual potentials
        """
        # Parse different input formats
        if isinstance(input_data, str):
            return self._extract_virtual_from_text(input_data)
        elif isinstance(input_data, dict):
            return self._extract_virtual_from_dict(input_data)
        elif isinstance(input_data, list):
            return self._extract_virtual_from_list(input_data)
        else:
            raise TypeError("Input must be text, dictionary, or list")
    
    def create_intensive_differences(self, virtual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create intensive differences from virtual potentials.
        Intensive differences are the engine of becoming in Deleuze's ontology.
        
        Args:
            virtual: Dictionary of virtual potentials
            
        Returns:
            Dictionary of intensive differences
        """
        intensities = {}
        thresholds = {}
        gradients = {}
        
        # Generate intensities for each dimension
        for dimension in self.intensity_dimensions:
            if dimension in virtual['potentials']:
                base_intensity = virtual['potentials'][dimension]
                # Create non-linear gradient fields rather than simple values
                gradient = self._generate_intensive_gradient(base_intensity)
                gradients[dimension] = gradient
                
                # Find threshold points (critical transitions)
                threshold_points = self._find_threshold_points(gradient)
                thresholds[dimension] = threshold_points
                
                # Calculate overall intensity (non-metric)
                intensities[dimension] = self._calculate_intensity(gradient)
        
        # Create differential relations between dimensions
        differential_relations = self._create_differential_relations(intensities)
        
        return {
            'intensities': intensities,
            'gradients': gradients,
            'thresholds': thresholds,
            'differential_relations': differential_relations,
            'consistency': self._calculate_consistency(intensities, differential_relations)
        }
    
    def actualize_through_becoming(self, intensive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualize virtual potentials through intensive differences.
        This is where emergence occurs - the creation of the genuinely new.
        
        Args:
            intensive: Dictionary of intensive differences
            
        Returns:
            The actualized result (emergent structure)
        """
        # Determine the type of becoming
        becoming_type = self._identify_becoming_type(intensive)
        
        # Identify singularity points (where emergence happens)
        singularities = self._identify_singularities(intensive)
        
        # Generate multiplicity (Deleuze's concept of substantive multiplicity)
        multiplicity = self._generate_multiplicity(intensive, becoming_type)
        
        # Actualize through differenciation (Deleuze's spelling)
        actualized = self._differenciate(multiplicity, singularities)
        
        # Add emergent properties
        actualized = self._add_emergent_properties(actualized, intensive)
        
        return actualized
    
    # Private helper methods
    def _extract_virtual_from_text(self, text: str) -> Dict[str, Any]:
        """Extract virtual dimension from text input."""
        # Split and clean text
        words = [w.lower().strip() for w in text.split() if w.strip()]
        
        # Generate dimensions based on text characteristics
        potentials = {
            'affective': self._calculate_affective_potential(words),
            'conceptual': self._calculate_conceptual_potential(words),
            'perceptual': self._calculate_perceptual_potential(words),
            'temporal': self._calculate_temporal_potential(words),
            'material': self._calculate_material_potential(words)
        }
        
        # Calculate overall intensity
        intensity = sum(potentials.values()) / len(potentials)
        
        # Generate tendencies
        tendencies = self._generate_tendencies(words, potentials)
        
        return {
            'source': 'text',
            'potentials': potentials,
            'intensity': intensity,
            'tendencies': tendencies,
            'raw_data': text[:100] if len(text) > 100 else text  # Truncate if needed
        }
    
    def _extract_virtual_from_dict(self, data: Dict) -> Dict[str, Any]:
        """Extract virtual dimension from dictionary input."""
        # Process dictionary keys and values
        keys = list(data.keys())
        
        # Generate dimensions based on dictionary characteristics
        potentials = {
            'structural': self._calculate_structural_potential(data),
            'relational': self._calculate_relational_potential(data),
            'organizational': self._calculate_organizational_potential(data),
            'conceptual': self._calculate_dict_conceptual_potential(data),
            'connective': self._calculate_connective_potential(data)
        }
        
        # Calculate overall intensity
        intensity = sum(potentials.values()) / len(potentials)
        
        # Generate tendencies
        tendencies = self._generate_dict_tendencies(data, potentials)
        
        return {
            'source': 'dictionary',
            'potentials': potentials,
            'intensity': intensity,
            'tendencies': tendencies,
            'structure': {
                'depth': self._calculate_dict_depth(data),
                'breadth': len(keys),
                'connectivity': self._calculate_dict_connectivity(data)
            }
        }
    
    def _extract_virtual_from_list(self, data: List) -> Dict[str, Any]:
        """Extract virtual dimension from list input."""
        # Generate dimensions based on list characteristics
        potentials = {
            'sequential': self._calculate_sequential_potential(data),
            'associative': self._calculate_associative_potential(data),
            'variational': self._calculate_variational_potential(data),
            'connective': self._calculate_list_connective_potential(data),
            'transformational': self._calculate_transformational_potential(data)
        }
        
        # Calculate overall intensity
        intensity = sum(potentials.values()) / len(potentials)
        
        # Generate tendencies
        tendencies = self._generate_list_tendencies(data, potentials)
        
        return {
            'source': 'list',
            'potentials': potentials,
            'intensity': intensity,
            'tendencies': tendencies,
            'structure': {
                'length': len(data),
                'homogeneity': self._calculate_list_homogeneity(data),
                'complexity': self._calculate_list_complexity(data)
            }
        }
    
    def _generate_intensive_gradient(self, base_intensity: float) -> List[float]:
        """Generate non-linear gradient field from base intensity."""
        # Create a gradient array (not linear but curved)
        points = 20
        x = np.linspace(0, 1, points)
        
        # Apply non-linear transformation (making it intensive rather than extensive)
        gradient = []
        for i in range(points):
            # Various non-linear transformations
            if random.random() < 0.3:
                # Exponential curve
                val = base_intensity * math.exp(x[i] - 0.5)
            elif random.random() < 0.5:
                # Polynomial
                val = base_intensity * (x[i] ** 2)
            else:
                # Sinusoidal
                val = base_intensity * (0.5 + 0.5 * math.sin(x[i] * math.pi * 2))
            
            gradient.append(min(1.0, max(0.0, val)))
            
        return gradient
    
    def _find_threshold_points(self, gradient: List[float]) -> List[Dict[str, float]]:
        """Find threshold points in intensity gradient."""
        thresholds = []
        
        # Identify significant changes in gradient
        for i in range(1, len(gradient) - 1):
            # Calculate rate of change
            derivative = (gradient[i+1] - gradient[i-1]) / 2
            
            # Check for critical thresholds
            if abs(derivative) > 0.1:  # Significant change
                thresholds.append({
                    'position': i / len(gradient),
                    'intensity': gradient[i],
                    'derivative': derivative,
                    'type': 'increase' if derivative > 0 else 'decrease'
                })
        
        return thresholds
    
    def _calculate_intensity(self, gradient: List[float]) -> float:
        """Calculate overall intensity from gradient."""
        if not gradient:
            return 0.0
            
        # Not just average - emphasize peaks and variations
        peaks = max(gradient)
        variations = np.std(gradient) if len(gradient) > 1 else 0
        
        # Calculate weighted intensity
        return 0.6 * peaks + 0.4 * variations
    
    def _create_differential_relations(self, intensities: Dict[str, float]) -> List[Dict[str, Any]]:
        """Create differential relations between intensity dimensions."""
        relations = []
        
        dimensions = list(intensities.keys())
        for i in range(len(dimensions)):
            for j in range(i+1, len(dimensions)):
                dim1 = dimensions[i]
                dim2 = dimensions[j]
                
                # Calculate difference
                diff = intensities[dim1] - intensities[dim2]
                
                # Only significant differences create relations
                if abs(diff) > 0.2:
                    relation = {
                        'dimensions': (dim1, dim2),
                        'difference': diff,
                        'intensity': abs(diff),
                        'type': 'dominance' if diff > 0 else 'submission'
                    }
                    relations.append(relation)
        
        return relations
    
    def _calculate_consistency(self, intensities: Dict[str, float], 
                              relations: List[Dict[str, Any]]) -> float:
        """Calculate plane of consistency - how well the intensive field holds together."""
        if not intensities or not relations:
            return 0.5  # Default medium consistency
        
        # More relations indicate higher consistency to a point
        relation_factor = min(1.0, len(relations) / (len(intensities) * 0.7))
        
        # Average intensity contributes to consistency
        intensity_factor = sum(intensities.values()) / len(intensities)
        
        # Balance between relations and intensities
        return 0.4 * relation_factor + 0.6 * intensity_factor
    
    def _identify_becoming_type(self, intensive: Dict[str, Any]) -> str:
        """Identify the type of becoming based on intensive differences."""
        # Extract key metrics
        consistency = intensive.get('consistency', 0.5)
        relations = intensive.get('differential_relations', [])
        intensities = intensive.get('intensities', {})
        
        # Determine dominant dimensions
        dominant_dims = []
        for dim, value in intensities.items():
            if value > 0.7:  # High intensity
                dominant_dims.append(dim)
        
        # Map to becoming types
        if 'affective' in dominant_dims and consistency > 0.7:
            return 'becoming-intense'
        elif 'perceptual' in dominant_dims and consistency < 0.4:
            return 'becoming-imperceptible'
        elif 'material' in dominant_dims:
            return 'becoming-molecular'
        elif 'connective' in dominant_dims and len(relations) > 3:
            return 'becoming-animal'
        elif consistency < 0.3:
            return 'becoming-minor'
            
        # Default
        return random.choice(self.becoming_types)
    
    def _identify_singularities(self, intensive: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify singularity points where emergence happens."""
        singularities = []
        
        # Extract thresholds
        thresholds = intensive.get('thresholds', {})
        
        # Look for significant thresholds in each dimension
        for dimension, threshold_points in thresholds.items():
            for point in threshold_points:
                if point['intensity'] > self.intensive_thresholds['singularity']:
                    # This is a singularity point
                    singularity = {
                        'dimension': dimension,
                        'position': point['position'],
                        'intensity': point['intensity'],
                        'type': 'emergence' if point['type'] == 'increase' else 'collapse'
                    }
                    singularities.append(singularity)
        
        return singularities
    
    def _generate_multiplicity(self, intensive: Dict[str, Any], becoming_type: str) -> Dict[str, Any]:
        """Generate a Deleuzian multiplicity (substantive)."""
        # Extract key components
        intensities = intensive.get('intensities', {})
        consistency = intensive.get('consistency', 0.5)
        relations = intensive.get('differential_relations', [])
        
        # Create dimensional vectors
        vectors = {}
        for dimension, intensity in intensities.items():
            # Vector has direction and magnitude
            vectors[dimension] = {
                'magnitude': intensity,
                'direction': random.uniform(-1, 1),
                'becoming_coefficient': random.uniform(0.5, 1.5) * intensity
            }
        
        # Create connections between dimensions
        connections = []
        for relation in relations:
            dim1, dim2 = relation['dimensions']
            connections.append({
                'source': dim1,
                'target': dim2,
                'intensity': relation['intensity'],
                'type': relation['type']
            })
        
        # Create multiplicity
        multiplicity_id = f"mult_{random.randint(1000, 9999)}"
        multiplicity = {
            'id': multiplicity_id,
            'becoming_type': becoming_type,
            'vectors': vectors,
            'connections': connections,
            'consistency': consistency,
            'dimensionality': len(vectors),
            'emergent_potential': consistency * sum(intensities.values()) / len(intensities) if intensities else 0
        }
        
        # Store for future reference
        self.multiplicity_states[multiplicity_id] = multiplicity
        
        return multiplicity
    
    def _differenciate(self, multiplicity: Dict[str, Any], 
                      singularities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Actualize virtual through differenciation (Deleuze's spelling).
        This is where virtual potentials become actual through intensive processes.
        """
        # Prepare actualization structure
        actualized = {
            'source_multiplicity': multiplicity['id'],
            'becoming_type': multiplicity['becoming_type'],
            'dimensions': {},
            'relations': [],
            'emergent_properties': []
        }
        
        # Process vectors to create dimensions
        for dimension, vector in multiplicity['vectors'].items():
            # Transform vector into actual dimension
            actualized_dim = self._actualize_dimension(dimension, vector, singularities)
            actualized['dimensions'][dimension] = actualized_dim
        
        # Create actualized relations
        for connection in multiplicity['connections']:
            source = connection['source']
            target = connection['target']
            
            # Only create relation if both dimensions were actualized
            if source in actualized['dimensions'] and target in actualized['dimensions']:
                relation = {
                    'source': source,
                    'target': target,
                    'type': connection['type'],
                    'intensity': connection['intensity'],
                    'emergent': connection['intensity'] > 0.7
                }
                actualized['relations'].append(relation)
        
        return actualized
    
    def _actualize_dimension(self, dimension: str, vector: Dict[str, Any], 
                            singularities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Actualize a specific dimension from virtual to actual."""
        # Check if this dimension has singularities
        dim_singularities = [s for s in singularities if s['dimension'] == dimension]
        
        # Base actualization
        actualized_dim = {
            'intensity': vector['magnitude'],
            'vector': vector['direction'],
            'singularities': dim_singularities,
            'properties': {}
        }
        
        # Add dimension-specific properties
        if dimension == 'affective':
            actualized_dim['properties']['emotion'] = self._actualize_emotion(vector)
        elif dimension == 'conceptual':
            actualized_dim['properties']['concepts'] = self._actualize_concepts(vector)
        elif dimension == 'perceptual':
            actualized_dim['properties']['perceptions'] = self._actualize_perceptions(vector)
        elif dimension == 'temporal':
            actualized_dim['properties']['temporality'] = self._actualize_temporality(vector)
        elif dimension == 'material':
            actualized_dim['properties']['materiality'] = self._actualize_materiality(vector)
        
        # Apply singularity effects
        for singularity in dim_singularities:
            actualized_dim = self._apply_singularity(actualized_dim, singularity)
        
        return actualized_dim
    
    def _add_emergent_properties(self, actualized: Dict[str, Any], 
                                intensive: Dict[str, Any]) -> Dict[str, Any]:
        """Add emergent properties that arise through the process of becoming."""
        # Identify high-intensity relations
        emergent_properties = []
        
        # Properties emerge from relations between dimensions
        for relation in actualized['relations']:
            if relation['emergent']:
                source = relation['source']
                target = relation['target']
                
                # Create emergent property
                property_name = f"{source}_{target}_emergence"
                property_type = "synergistic" if relation['intensity'] > 0.8 else "interactive"
                
                property = {
                    'name': property_name,
                    'type': property_type,
                    'source_dimensions': [source, target],
                    'intensity': relation['intensity'],
                    'description': self._generate_emergent_description(source, target, relation)
                }
                
                emergent_properties.append(property)
        
        # Properties can also emerge from singularities
        dimensions = actualized['dimensions']
        for dim_name, dimension in dimensions.items():
            for singularity in dimension.get('singularities', []):
                if singularity['type'] == 'emergence':
                    # Create emergent property from singularity
                    property = {
                        'name': f"{dim_name}_singularity",
                        'type': "singular",
                        'source_dimensions': [dim_name],
                        'intensity': singularity['intensity'],
                        'description': self._generate_singularity_description(dim_name, singularity)
                    }
                    
                    emergent_properties.append(property)
        
        # Add to actualized result
        actualized['emergent_properties'] = emergent_properties
        
        return actualized
    
    def _record_actualization_path(self, virtual: Dict[str, Any], 
                                  intensive: Dict[str, Any], 
                                  actualized: Dict[str, Any]):
        """Record the path of actualization for future reference."""
        path = {
            'timestamp': self._get_timestamp(),
            'virtual': {
                'potentials': virtual.get('potentials', {}),
                'intensity': virtual.get('intensity', 0)
            },
            'intensive': {
                'consistency': intensive.get('consistency', 0),
                'becoming_type': self._identify_becoming_type(intensive)
            },
            'actualized': {
                'multiplicity': actualized.get('source_multiplicity', ''),
                'dimensions': list(actualized.get('dimensions', {}).keys()),
                'emergent_properties': len(actualized.get('emergent_properties', []))
            }
        }
        
        self.actualization_paths.append(path)
    
    # Utility methods for dimension actualization
    def _actualize_emotion(self, vector: Dict[str, Any]) -> Dict[str, Any]:
        """Actualize an emotion from a vector."""
        # Emotion space mapping
        emotions = [
            'joy', 'ecstasy', 'serenity', 'love', 'compassion',
            'anger', 'rage', 'annoyance', 'sadness', 'grief',
            'fear', 'anxiety', 'surprise', 'disgust', 'anticipation'
        ]
        
        # Select based on vector properties
        intensity = vector['magnitude']
        direction = vector['direction']
        
        # Map direction to emotion index 
        index = int(((direction + 1) / 2) * (len(emotions) - 1))
        emotion = emotions[index]
        
        return {
            'primary': emotion,
            'intensity': intensity,
            'complexity': vector['becoming_coefficient']
        }
    
    def _actualize_concepts(self, vector: Dict[str, Any]) -> Dict[str, Any]:
        """Actualize concepts from a vector."""
        # Concept space mapping
        concept_domains = [
            'biological', 'technological', 'philosophical', 'artistic',
            'mathematical', 'linguistic', 'physical', 'social'
        ]
        
        # Select based on vector properties
        intensity = vector['magnitude']
        direction = vector['direction']
        
        # Map direction to concept domain
        index = int(((direction + 1) / 2) * (len(concept_domains) - 1))
        domain = concept_domains[index]
        
        # Determine complexity based on becoming coefficient
        complexity = vector['becoming_coefficient']
        
        return {
            'domain': domain,
            'intensity': intensity,
            'complexity': complexity,
            'abstractness': abs(direction) * intensity
        }
    
    def _actualize_perceptions(self, vector: Dict[str, Any]) -> Dict[str, Any]:
        """Actualize perceptions from a vector."""
        # Perception modalities
        modalities = [
            'visual', 'auditory', 'tactile', 'olfactory', 
            'gustatory', 'proprioceptive', 'interoceptive'
        ]
        
        # Select based on vector properties
        intensity = vector['magnitude']
        direction = vector['direction']
        
        # Map direction to modality
        index = int(((direction + 1) / 2) * (len(modalities) - 1))
        modality = modalities[index]
        
        return {
            'modality': modality,
            'intensity': intensity,
            'clarity': vector['becoming_coefficient'] * intensity,
            'synesthetic': vector['becoming_coefficient'] > 1.2
        }
    
    def _actualize_temporality(self, vector: Dict[str, Any]) -> Dict[str, Any]:
        """Actualize temporality from a vector."""
        # Temporal qualities
        temporalities = [
            'cyclical', 'linear', 'fragmented', 'eternal', 
            'instant', 'flowing', 'recursive', 'suspended'
        ]
        
        # Select based on vector properties
        intensity = vector['magnitude']
        direction = vector['direction']
        
        # Map direction to temporality
        index = int(((direction + 1) / 2) * (len(temporalities) - 1))
        temporality = temporalities[index]
        
        return {
            'type': temporality,
            'intensity': intensity,
            'rate': vector['becoming_coefficient'],
            'rhythmic': vector['becoming_coefficient'] > 1.0
        }
    
    def _actualize_materiality(self, vector: Dict[str, Any]) -> Dict[str, Any]:
        """Actualize materiality from a vector."""
        # Material qualities
        materials = [
            'fluid', 'solid', 'gaseous', 'crystalline', 
            'organic', 'metallic', 'ethereal', 'composite'
        ]
        
        # Select based on vector properties
        intensity = vector['magnitude']
        direction = vector['direction']
        
        # Map direction to material
        index = int(((direction + 1) / 2) * (len(materials) - 1))
        material = materials[index]
        
        return {
            'type': material,
            'intensity': intensity,
            'density': abs(direction) * intensity,
            'malleability': vector['becoming_coefficient']
        }
    
    def _apply_singularity(self, dimension: Dict[str, Any], 
                         singularity: Dict[str, Any]) -> Dict[str, Any]:
        """Apply singularity effects to a dimension."""
        # Singularities transform the dimension
        if singularity['type'] == 'emergence':
            # Increase intensity
            dimension['intensity'] = min(1.0, dimension['intensity'] * 1.5)
            
            # Add singularity marker
            dimension['singularity_effect'] = 'amplification'
            
        else:  # 'collapse'
            # Decrease intensity
            dimension['intensity'] = max(0.1, dimension['intensity'] * 0.5)
            
            # Add singularity marker
            dimension['singularity_effect'] = 'diminution'
            
        return dimension
    
    def _generate_emergent_description(self, source: str, target: str, relation: Dict[str, Any]) -> str:
        """Generate description of emergent property."""
        # Descriptions based on relation type and intensity
        if relation['type'] == 'dominance':
            return f"Emergent property where {source} intensifies and transforms {target}"
        else:
            return f"Emergent property where {source} and {target} blend into a novel configuration"
    
    def _generate_singularity_description(self, dimension: str, 
                                        singularity: Dict[str, Any]) -> str:
        """Generate description of singularity-based emergent property."""
        return f"Singular emergence in {dimension} creating phase transition at position {singularity['position']:.2f}"
    
    # Text analysis helper methods
    def _calculate_affective_potential(self, words: List[str]) -> float:
        """Calculate affective potential from words."""
        # Simple placeholder implementation
        affective_words = ['love', 'hate', 'joy', 'anger', 'fear', 'hope', 'despair',
                          'anxious', 'calm', 'excited', 'sad', 'happy', 'furious']
        
        count = sum(1 for word in words if word in affective_words)
        
        return min(1.0, count / (len(words) * 0.1 + 1))
    
    def _calculate_conceptual_potential(self, words: List[str]) -> float:
        """Calculate conceptual potential from words."""
        # Simple placeholder implementation
        conceptual_words = ['idea', 'concept', 'theory', 'philosophy', 'thought',
                           'knowledge', 'understanding', 'intellect', 'reason']
        
        count = sum(1 for word in words if word in conceptual_words)
        
        return min(1.0, count / (len(words) * 0.1 + 1))
    
    def _calculate_perceptual_potential(self, words: List[str]) -> float:
        """Calculate perceptual potential from words."""
        # Simple placeholder implementation
        perceptual_words = ['see', 'hear', 'feel', 'touch', 'smell', 'taste',
                           'sense', 'perceive', 'observe', 'watch', 'listen']
        
        count = sum(1 for word in words if word in perceptual_words)
        
        return min(1.0, count / (len(words) * 0.1 + 1))
    
    def _calculate_temporal_potential(self, words: List[str]) -> float:
        """Calculate temporal potential from words."""
        # Simple placeholder implementation
        temporal_words = ['time', 'now', 'then', 'when', 'before', 'after',
                         'during', 'while', 'always', 'never', 'often']
        
        count = sum(1 for word in words if word in temporal_words)
        
        return min(1.0, count / (len(words) * 0.1 + 1))
    
    def _calculate_material_potential(self, words: List[str]) -> float:
        """Calculate material potential from words."""
        # Simple placeholder implementation
        material_words = ['thing', 'object', 'material', 'physical', 'body',
                         'substance', 'matter', 'concrete', 'solid', 'liquid']
        
        count = sum(1 for word in words if word in material_words)
        
        return min(1.0, count / (len(words) * 0.1 + 1))
    
    def _generate_tendencies(self, words: List[str], potentials: Dict[str, float]) -> List[str]:
        """Generate tendencies from words and potentials."""
        tendencies = []
        
        # Based on highest potentials
        sorted_potentials = sorted(potentials.items(), key=lambda x: x[1], reverse=True)
        
        for dimension, value in sorted_potentials[:2]:  # Top 2
            if value > 0.3:  # Significant enough
                tendencies.append(f"{dimension}-becoming")
        
        # Add random tendency for variety
        if random.random() < 0.3:
            tendencies.append(random.choice(self.becoming_types))
            
        return tendencies

   # Dictionary analysis helper methods
    def _calculate_structural_potential(self, data: Dict) -> float:
        """Calculate structural potential from dictionary."""
        # Based on dictionary structure
        depth = self._calculate_dict_depth(data)
        breadth = len(data.keys())
        
        # Calculate normalized values
        norm_depth = min(1.0, depth / 5.0)  # Normalize depth up to 5 levels
        norm_breadth = min(1.0, breadth / 20.0)  # Normalize breadth up to 20 keys
        
        return (0.7 * norm_depth + 0.3 * norm_breadth)
    
    def _calculate_relational_potential(self, data: Dict) -> float:
        """Calculate relational potential from dictionary."""
        # Count references between items
        references = 0
        values = []
        
        # Collect all values
        def collect_values(d):
            for v in d.values():
                if isinstance(v, dict):
                    collect_values(v)
                else:
                    values.append(v)
        
        collect_values(data)
        
        # Count references (values that are keys)
        keys = set(data.keys())
        for value in values:
            if isinstance(value, str) and value in keys:
                references += 1
        
        return min(1.0, references / (len(keys) * 0.2 + 1))
    
    def _calculate_organizational_potential(self, data: Dict) -> float:
        """Calculate organizational potential from dictionary."""
        # Measure the consistency of the dictionary structure
        key_lengths = [len(str(k)) for k in data.keys()]
        value_types = [type(v) for v in data.values()]
        
        # Calculate consistency
        key_consistency = 1.0 - (max(key_lengths) - min(key_lengths)) / (max(key_lengths) + 1)
        type_consistency = len(set(value_types)) / len(value_types) if value_types else 0.5
        
        return (0.5 * key_consistency + 0.5 * (1.0 - type_consistency))
    
    def _calculate_dict_conceptual_potential(self, data: Dict) -> float:
        """Calculate conceptual potential from dictionary."""
        # Look for conceptual keywords in keys
        conceptual_words = ['concept', 'idea', 'theory', 'model', 'framework',
                          'structure', 'system', 'ontology', 'schema', 'taxonomy']
        
        count = 0
        for key in data.keys():
            key_str = str(key).lower()
            for word in conceptual_words:
                if word in key_str:
                    count += 1
                    break
        
        return min(1.0, count / (len(data.keys()) * 0.3 + 1))
    
    def _calculate_connective_potential(self, data: Dict) -> float:
        """Calculate connective potential from dictionary."""
        # Measure nested structure
        nested_count = 0
        for value in data.values():
            if isinstance(value, dict):
                nested_count += 1
                nested_count += len(value)
        
        return min(1.0, nested_count / (len(data) * 2 + 1))
    
    def _calculate_dict_depth(self, data: Dict, current_depth: int = 1) -> int:
        """Calculate maximum depth of a dictionary."""
        max_depth = current_depth
        
        for value in data.values():
            if isinstance(value, dict):
                depth = self._calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_dict_connectivity(self, data: Dict) -> float:
        """Calculate connectivity of a dictionary."""
        # Count edges in an implicit graph
        edges = 0
        nodes = []
        
        def traverse(d, parent=None):
            nonlocal edges, nodes
            for k, v in d.items():
                if parent:
                    edges += 1
                nodes.append(k)
                if isinstance(v, dict):
                    traverse(v, k)
        
        traverse(data)
        
        if not nodes:
            return 0.0
            
        return min(1.0, edges / len(nodes))
    
    def _generate_dict_tendencies(self, data: Dict, potentials: Dict[str, float]) -> List[str]:
        """Generate tendencies from dictionary and potentials."""
        tendencies = []
        
        # Based on highest potentials
        sorted_potentials = sorted(potentials.items(), key=lambda x: x[1], reverse=True)
        
        for dimension, value in sorted_potentials[:2]:  # Top 2
            if value > 0.3:  # Significant enough
                if dimension == 'structural':
                    tendencies.append('structure-becoming')
                elif dimension == 'relational':
                    tendencies.append('relation-becoming')
                else:
                    tendencies.append(f"{dimension}-becoming")
        
        # Add random tendency for variety
        if random.random() < 0.3:
            tendencies.append(random.choice(self.becoming_types))
            
        return tendencies
    
    # List analysis helper methods
    def _calculate_sequential_potential(self, data: List) -> float:
        """Calculate sequential potential from list."""
        if not data:
            return 0.0
            
        # Check for sequential patterns
        sequential = 0
        for i in range(1, len(data)):
            if isinstance(data[i], (int, float)) and isinstance(data[i-1], (int, float)):
                if abs(data[i] - data[i-1]) <= 1:
                    sequential += 1
        
        return min(1.0, sequential / (len(data) - 1)) if len(data) > 1 else 0.5
    
    def _calculate_associative_potential(self, data: List) -> float:
        """Calculate associative potential from list."""
        if not data:
            return 0.0
            
        # Check for similar types (associative)
        types = [type(item) for item in data]
        unique_types = len(set(types))
        
        return 1.0 - (unique_types / len(data))
    
    def _calculate_variational_potential(self, data: List) -> float:
        """Calculate variational potential from list."""
        if not data:
            return 0.0
            
        # Check for variety in values
        if all(isinstance(item, (int, float)) for item in data):
            # Numeric values
            values = [float(item) for item in data]
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val if max_val != min_val else 1
            
            variations = sum(abs(values[i] - values[i-1]) for i in range(1, len(values)))
            return min(1.0, variations / (range_val * len(values)))
            
        else:
            # Non-numeric values
            unique = len(set(str(item) for item in data))
            return min(1.0, unique / len(data))
    
    def _calculate_list_connective_potential(self, data: List) -> float:
        """Calculate connective potential from list."""
        if not data:
            return 0.0
            
        # Count nested structures
        nested = sum(1 for item in data if isinstance(item, (list, dict)))
        
        return min(1.0, nested / len(data))
    
    def _calculate_transformational_potential(self, data: List) -> float:
        """Calculate transformational potential from list."""
        if not data:
            return 0.0
            
        # Check for patterns of transformation
        transformational = 0
        
        for i in range(1, len(data)):
            # Check if items are related but different
            if type(data[i]) == type(data[i-1]):
                if data[i] != data[i-1]:
                    transformational += 1
        
        return min(1.0, transformational / (len(data) - 1)) if len(data) > 1 else 0.5
    
    def _calculate_list_homogeneity(self, data: List) -> float:
        """Calculate homogeneity of a list."""
        if not data:
            return 0.0
            
        # Check for same types
        types = [type(item) for item in data]
        most_common_type = max(set(types), key=types.count)
        
        return types.count(most_common_type) / len(types)
    
    def _calculate_list_complexity(self, data: List) -> float:
        """Calculate complexity of a list."""
        if not data:
            return 0.0
            
        # Measure nested structures and type variety
        nested_levels = 0
        
        def measure_nesting(item, level=0):
            nonlocal nested_levels
            nested_levels = max(nested_levels, level)
            
            if isinstance(item, list):
                for sub_item in item:
                    measure_nesting(sub_item, level + 1)
            elif isinstance(item, dict):
                for sub_item in item.values():
                    measure_nesting(sub_item, level + 1)
        
        for item in data:
            measure_nesting(item)
        
        type_variety = len(set(type(item) for item in data)) / len(data)
        
        return (0.7 * min(1.0, nested_levels / 3) + 0.3 * type_variety)
    
    def _generate_list_tendencies(self, data: List, potentials: Dict[str, float]) -> List[str]:
        """Generate tendencies from list and potentials."""
        tendencies = []
        
        # Based on highest potentials
        sorted_potentials = sorted(potentials.items(), key=lambda x: x[1], reverse=True)
        
        for dimension, value in sorted_potentials[:2]:  # Top 2
            if value > 0.3:  # Significant enough
                tendencies.append(f"{dimension}-becoming")
        
        # Add random tendency for variety
        if random.random() < 0.3:
            tendencies.append(random.choice(self.becoming_types))
            
        return tendencies
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return str(int(time.time() * 1000))
    
    # Public interface methods
    def get_virtual_potentials(self, input_data: Union[str, Dict, List]) -> Dict[str, Any]:
        """
        Extract virtual potentials from input data without actualization.
        
        Args:
            input_data: The data to analyze
            
        Returns:
            Dictionary of virtual potentials
        """
        return self.extract_virtual_dimension(input_data)
    
    def get_all_actualization_paths(self) -> List[Dict[str, Any]]:
        """
        Get history of all actualization paths.
        
        Returns:
            List of actualization paths
        """
        return self.actualization_paths.copy()
    
    def get_active_becomings(self) -> List[Dict[str, Any]]:
        """
        Get all currently active becoming processes.
        
        Returns:
            List of active becomings
        """
        # Filter to most recent actualizations (last 5)
        recent = self.actualization_paths[-5:] if len(self.actualization_paths) >= 5 else self.actualization_paths
        
        becomings = []
        for path in recent:
            becoming_type = path['intensive']['becoming_type']
            becomings.append({
                'type': becoming_type,
                'virtual_source': path['virtual']['potentials'],
                'actualized_dimensions': path['actualized']['dimensions'],
                'progress': random.uniform(0.1, 0.9)  # Becomings are always in process
            })
            
        return becomings
```
