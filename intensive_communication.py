# intensive_communication.py

import random
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from collections import defaultdict
import math
import time

class IntensiveCommunicationModule:
    """
    A module for communication through affects and intensities rather than representation.
    Operates through resonance fields, affective transmissions, and intensive differences
    rather than symbolic encoding and decoding.
    """
    
    def __init__(self):
        self.affective_channels = {}
        self.resonance_fields = {}
        self.intensity_gradients = {}
        self.affective_memory = []
        self.resonance_history = {}
        self.communication_thresholds = {
            'reception': 0.3,
            'resonance': 0.5,
            'transmission': 0.4,
            'modulation': 0.7
        }
        self.channel_types = [
            'affective', 'perceptual', 'conceptual', 'expressive', 
            'vibrational', 'empathic', 'rhythmic', 'intensive'
        ]
        self.modulation_patterns = {}
        self.carrier_waves = {}
        self.transduction_matrices = {}
        
    def communicate_intensively(self, message: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Communicate through affects and intensities rather than representation.
        
        Args:
            message: The message to communicate (text or structured data)
            
        Returns:
            Dictionary containing the intensive communication and its characteristics
        """
        # Extract affective intensity from message
        affective_charge = self.extract_affective_intensity(message)
        
        # Create resonance field from affective charge
        resonance = self.create_resonance_field(affective_charge)
        
        # Transmit intensive signal through resonance field
        transmission = self.transmit_intensive_signal(resonance)
        
        # Record communication
        self._record_communication(affective_charge, resonance, transmission)
        
        return transmission
    
    def extract_affective_intensity(self, message: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract affective intensity from a message.
        Focuses on intensive qualities rather than extensive properties.
        
        Args:
            message: Text or structured message to extract affects from
            
        Returns:
            Dictionary of affective intensities and characteristics
        """
        # Handle different message types
        if isinstance(message, str):
            return self._extract_from_text(message)
        elif isinstance(message, dict):
            return self._extract_from_structured(message)
        else:
            raise TypeError("Message must be text or structured data")
    
    def create_resonance_field(self, affective_charge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a resonance field from affective charge.
        Resonance fields are intensive spaces where affects can circulate.
        
        Args:
            affective_charge: Dictionary of affective intensities
            
        Returns:
            Resonance field as a dictionary
        """
        # Extract key components
        intensities = affective_charge['intensities']
        primary_affect = affective_charge['primary_affect']
        
        # Generate field dimensions
        dimensions = self._generate_field_dimensions(intensities)
        
        # Create intensity distribution across dimensions
        distribution = self._create_intensity_distribution(dimensions, intensities)
        
        # Generate resonance patterns
        patterns = self._generate_resonance_patterns(distribution, primary_affect)
        
        # Create field
        field_id = f"field_{len(self.resonance_fields) + 1}_{int(time.time())}"
        field = {
            'id': field_id,
            'source_charge': affective_charge['id'],
            'dimensions': dimensions,
            'distribution': distribution,
            'patterns': patterns,
            'cohesion': self._calculate_field_cohesion(distribution, patterns),
            'primary_resonance': primary_affect,
            'timestamp': self._get_timestamp()
        }
        
        # Register the field
        self.resonance_fields[field_id] = field
        
        return field
    
    def transmit_intensive_signal(self, resonance_field: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transmit intensive signal through resonance field.
        Transmission occurs through intensive differences rather than encoding.
        
        Args:
            resonance_field: Resonance field to transmit through
            
        Returns:
            Transmitted intensive signal
        """
        # Select appropriate channel
        channel = self._select_transmission_channel(resonance_field)
        
        # Modulate signal through channel
        modulation = self._modulate_signal(resonance_field, channel)
        
        # Generate carrier wave
        carrier = self._generate_carrier_wave(modulation)
        
        # Create signal transduction
        transduction = self._create_signal_transduction(modulation, carrier)
        
        # Create the transmission
        transmission_id = f"trans_{len(self.affective_memory) + 1}_{int(time.time())}"
        transmission = {
            'id': transmission_id,
            'source_field': resonance_field['id'],
            'channel': channel,
            'modulation': modulation,
            'carrier': carrier,
            'transduction': transduction,
            'primary_affect': resonance_field['primary_resonance'],
            'intensity': self._calculate_transmission_intensity(modulation, carrier),
            'timestamp': self._get_timestamp()
        }
        
        return transmission
    
    def receive_intensive_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receive an intensive signal.
        Reception occurs through resonance rather than decoding.
        
        Args:
            signal: The intensive signal to receive
            
        Returns:
            The reception result
        """
        # Create reception field
        reception_field = self._create_reception_field(signal)
        
        # Generate resonance with existing fields
        resonances = self._generate_field_resonances(reception_field)
        
        # Transduce signal
        transduction = self._transduce_signal(signal, resonances)
        
        # Create the reception
        reception = {
            'signal': signal['id'],
            'field': reception_field,
            'resonances': resonances,
            'transduction': transduction,
            'intensity': signal['intensity'],
            'primary_affect': signal['primary_affect'],
            'timestamp': self._get_timestamp()
        }
        
        return reception
    
    def modulate_intensity(self, signal: Dict[str, Any], 
                         modulation_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modulate intensity of a signal.
        
        Args:
            signal: The signal to modulate
            modulation_pattern: The pattern to modulate with
            
        Returns:
            The modulated signal
        """
        # Apply modulation pattern to signal
        modulated_transduction = {}
        for key, value in signal['transduction'].items():
            if key in modulation_pattern['pattern']:
                # Apply modulation
                factor = modulation_pattern['pattern'][key]
                modulated_transduction[key] = value * factor
            else:
                modulated_transduction[key] = value
        
        # Create new carrier wave
        modulated_carrier = self._modulate_carrier_wave(
            signal['carrier'], modulation_pattern)
        
        # Create modulated signal
        modulated = {
            'source_signal': signal['id'],
            'transduction': modulated_transduction,
            'carrier': modulated_carrier,
            'pattern': modulation_pattern,
            'intensity': signal['intensity'] * modulation_pattern['intensity'],
            'primary_affect': self._determine_modulated_affect(
                signal['primary_affect'], modulation_pattern),
            'timestamp': self._get_timestamp()
        }
        
        return modulated
    
    def create_affective_channel(self, affect_type: str, intensity: float) -> Dict[str, Any]:
        """
        Create a new affective channel for communication.
        
        Args:
            affect_type: Type of affect for the channel
            intensity: Base intensity of the channel
            
        Returns:
            The created channel
        """
        # Generate channel characteristics
        characteristics = self._generate_channel_characteristics(affect_type)
        
        # Create channel
        channel_id = f"channel_{len(self.affective_channels) + 1}_{int(time.time())}"
        channel = {
            'id': channel_id,
            'type': affect_type,
            'intensity': intensity,
            'characteristics': characteristics,
            'bandwidth': self._calculate_channel_bandwidth(characteristics, intensity),
            'timestamp': self._get_timestamp()
        }
        
        # Register channel
        self.affective_channels[channel_id] = channel
        
        return channel
    
    def create_modulation_pattern(self, base_pattern: Dict[str, float], 
                                intensity: float) -> Dict[str, Any]:
        """
        Create a modulation pattern for signal modulation.
        
        Args:
            base_pattern: Dictionary of dimension to modulation factor
            intensity: Overall intensity of the pattern
            
        Returns:
            The created modulation pattern
        """
        # Normalize pattern
        normalized = {k: v / max(base_pattern.values()) for k, v in base_pattern.items()}
        
        # Scale by intensity
        scaled = {k: v * intensity for k, v in normalized.items()}
        
        # Create pattern
        pattern_id = f"pattern_{len(self.modulation_patterns) + 1}_{int(time.time())}"
        pattern = {
            'id': pattern_id,
            'pattern': scaled,
            'intensity': intensity,
            'dimensions': list(scaled.keys()),
            'timestamp': self._get_timestamp()
        }
        
        # Register pattern
        self.modulation_patterns[pattern_id] = pattern
        
        return pattern
    
    def get_resonance_field_by_affect(self, affect: str) -> Dict[str, Any]:
        """
        Get resonance field by primary affect.
        
        Args:
            affect: The primary affect to search for
            
        Returns:
            Matching resonance field or None
        """
        for field_id, field in self.resonance_fields.items():
            if field['primary_resonance'] == affect:
                return field
        return None
    
    def get_active_channels(self) -> List[Dict[str, Any]]:
        """
        Get all active affective channels.
        
        Returns:
            List of active channels
        """
        return list(self.affective_channels.values())
    
    def get_communication_history(self) -> List[Dict[str, Any]]:
        """
        Get history of intensive communications.
        
        Returns:
            List of past communications
        """
        return self.affective_memory
    
    # Helper methods for affective extraction
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract affective intensity from text."""
        # Tokenize text
        words = [w.lower().strip() for w in text.split() if w.strip()]
        
        # Initialize intensities
        intensities = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'anticipation': 0.0,
            'trust': 0.0,
            'disgust': 0.0
        }
        
        # Simple affect lexicon (would be more sophisticated in real implementation)
        affect_words = {
            'joy': ['happy', 'joy', 'delight', 'pleasure', 'content', 'satisfied'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'grief', 'sorrow'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage'],
            'fear': ['afraid', 'fear', 'scared', 'anxious', 'terrified', 'dread'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled'],
            'anticipation': ['expect', 'anticipate', 'await', 'eager', 'looking forward'],
            'trust': ['trust', 'believe', 'faith', 'confident', 'assured', 'certain'],
            'disgust': ['disgust', 'repulsed', 'revolted', 'aversion', 'dislike']
        }
        
        # Calculate intensities based on word occurrences
        for affect, affect_word_list in affect_words.items():
            count = sum(1 for word in words if word in affect_word_list)
            intensities[affect] = min(1.0, count / (len(words) * 0.1 + 1))
        
        # Identify primary affect
        primary_affect = max(intensities.items(), key=lambda x: x[1])[0]
        
        # Calculate overall intensity
        overall_intensity = sum(intensities.values()) / len(intensities)
        
        # Create affective charge
        charge_id = f"charge_{len(self.affective_memory) + 1}_{int(time.time())}"
        charge = {
            'id': charge_id,
            'source_type': 'text',
            'intensities': intensities,
            'primary_affect': primary_affect,
            'overall_intensity': overall_intensity,
            'complexity': self._calculate_affect_complexity(intensities),
            'source_length': len(words),
            'timestamp': self._get_timestamp()
        }
        
        return charge
    
    def _extract_from_structured(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract affective intensity from structured data."""
        # Initialize default intensities
        intensities = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'anticipation': 0.0,
            'trust': 0.0,
            'disgust': 0.0
        }
        
        # Check for explicit affect data
        if 'affects' in data:
            affects = data['affects']
            for affect, value in affects.items():
                if affect in intensities:
                    intensities[affect] = min(1.0, float(value))
        
        # Extract from other fields if needed
        elif 'content' in data and isinstance(data['content'], str):
            return self._extract_from_text(data['content'])
        
        else:
            # Attempt to infer from structure
            structure_affects = self._infer_affects_from_structure(data)
            for affect, value in structure_affects.items():
                intensities[affect] = value
        
        # Identify primary affect
        primary_affect = max(intensities.items(), key=lambda x: x[1])[0]
        
        # Calculate overall intensity
        overall_intensity = sum(intensities.values()) / len(intensities)
        
        # Create affective charge
        charge_id = f"charge_{len(self.affective_memory) + 1}_{int(time.time())}"
        charge = {
            'id': charge_id,
            'source_type': 'structured',
            'intensities': intensities,
            'primary_affect': primary_affect,
            'overall_intensity': overall_intensity,
            'complexity': self._calculate_affect_complexity(intensities),
            'structure_depth': self._calculate_structure_depth(data),
            'timestamp': self._get_timestamp()
        }
        
        return charge
    
    def _infer_affects_from_structure(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Infer affects from data structure."""
        # Initialize affects
        affects = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'anticipation': 0.0,
            'trust': 0.0,
            'disgust': 0.0
        }
        
        # Count structure properties
        depth = self._calculate_structure_depth(data)
        breadth = len(data.keys())
        complexity = self._calculate_structure_complexity(data)
        
        # Map structural properties to affects
        affects['surprise'] = min(1.0, complexity / 5)
        affects['anticipation'] = min(1.0, breadth / 10)
        affects['trust'] = 0.5  # Neutral
        affects['joy'] = random.uniform(0.3, 0.7)  # Random baseline
        
        return affects
    
    def _calculate_affect_complexity(self, intensities: Dict[str, float]) -> float:
        """Calculate complexity of affect pattern."""
        # Based on distribution of intensities
        values = list(intensities.values())
        if not values:
            return 0.0
            
        # Standard deviation indicates complexity
        std_dev = np.std(values)
        
        # Number of significant affects
        significant = sum(1 for v in values if v > 0.2)
        
        # Combine measures
        return 0.5 * std_dev + 0.5 * (significant / len(values))
    
    def _calculate_structure_depth(self, data: Dict[str, Any], current_depth: int = 1) -> int:
        """Calculate maximum depth of nested structure."""
        max_depth = current_depth
        
        for value in data.values():
            if isinstance(value, dict):
                depth = self._calculate_structure_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
                
        return max_depth
    
    def _calculate_structure_complexity(self, data: Dict[str, Any]) -> float:
        """Calculate complexity of data structure."""
        # Count different types of values
        types = set(type(v) for v in data.values())
        type_variety = len(types) / max(1, len(data))
        
        # Count nesting
        nested = sum(1 for v in data.values() if isinstance(v, (dict, list)))
        nesting = nested / max(1, len(data))
        
        # Combine measures
        return 0.5 * type_variety + 0.5 * nesting
    
    # Helper methods for resonance field creation
    def _generate_field_dimensions(self, intensities: Dict[str, float]) -> List[str]:
        """Generate dimensions for resonance field."""
        # Start with affects as dimensions
        dimensions = [affect for affect, intensity in intensities.items() 
                    if intensity > self.communication_thresholds['resonance']]
        
        # Add modulation dimensions
        modulation_dims = ['rhythm', 'harmony', 'dissonance', 'resonance']
        dimensions.extend(random.sample(modulation_dims, 
                                      k=min(2, len(modulation_dims))))
        
        # If no significant dimensions, use primary affect
        if not dimensions:
            dimensions = [max(intensities.items(), key=lambda x: x[1])[0]]
            
        return dimensions
    
    def _create_intensity_distribution(self, dimensions: List[str], 
                                     intensities: Dict[str, float]) -> Dict[str, List[float]]:
        """Create intensity distribution across field dimensions."""
        distribution = {}
        
        # Generate distribution for each dimension
        for dimension in dimensions:
            # Base intensity from affect intensities if available
            base_intensity = intensities.get(dimension, random.uniform(0.3, 0.7))
            
            # Create gradient points (non-uniform)
            points = 10
            distribution[dimension] = []
            
            for i in range(points):
                # Non-linear distribution
                if random.random() < 0.3:
                    # Exponential
                    pos = i / points
                    val = base_intensity * math.exp((pos - 0.5) * 2)
                elif random.random() < 0.7:
                    # Gaussian
                    pos = i / points
                    val = base_intensity * math.exp(-((pos - 0.5) ** 2) * 10)
                else:
                    # Linear with noise
                    val = base_intensity * (1 + random.uniform(-0.3, 0.3))
                
                distribution[dimension].append(min(1.0, max(0.0, val)))
        
        return distribution
    
    def _generate_resonance_patterns(self, distribution: Dict[str, List[float]], 
                                   primary_affect: str) -> List[Dict[str, Any]]:
        """Generate resonance patterns in field."""
        patterns = []
        
        # Generate different pattern types
        pattern_types = ['harmonic', 'rhythmic', 'intensive', 'affective']
        
        for pattern_type in pattern_types:
            # Only create pattern with probability
            if random.random() < 0.7:
                # Select dimensions for pattern (at least include primary)
                pattern_dimensions = [primary_affect]
                for dim in distribution.keys():
                    if dim != primary_affect and random.random() < 0.5:
                        pattern_dimensions.append(dim)
                
                # Create pattern
                pattern = {
                    'type': pattern_type,
                    'dimensions': pattern_dimensions,
                    'intensity': random.uniform(0.5, 1.0),
                    'frequency': random.uniform(0, 1.0),
                    'phase': random.uniform(0, 2 * math.pi)
                }
                
                patterns.append(pattern)
        
        # Ensure at least one pattern
        if not patterns:
            patterns.append({
                'type': 'affective',
                'dimensions': [primary_affect],
                'intensity': random.uniform(0.7, 1.0),
                'frequency': random.uniform(0, 1.0),
                'phase': 0
            })
            
        return patterns
    
    def _calculate_field_cohesion(self, distribution: Dict[str, List[float]], 
                                patterns: List[Dict[str, Any]]) -> float:
        """Calculate cohesion of resonance field."""
        # Count dimensions and patterns
        dim_count = len(distribution)
        pattern_count = len(patterns)
        
        # Calculate average intensity
        all_intensities = []
        for dim, values in distribution.items():
            all_intensities.extend(values)
        avg_intensity = sum(all_intensities) / len(all_intensities) if all_intensities else 0
        
        # Calculate pattern coverage
        pattern_dims = set()
        for pattern in patterns:
            pattern_dims.update(pattern['dimensions'])
        coverage = len(pattern_dims) / dim_count if dim_count > 0 else 0
        
        # Combine measures
        return 0.4 * avg_intensity + 0.3 * coverage + 0.3 * min(1.0, pattern_count / 3)
    
    # Helper methods for signal transmission
    def _select_transmission_channel(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate transmission channel for field."""
        # Check existing channels
        compatible_channels = []
        for channel_id, channel in self.affective_channels.items():
            # Check compatibility with primary affect
            if channel['type'] == field['primary_resonance']:
                compatible_channels.append(channel)
            # Check for intensive channel
            elif channel['type'] == 'intensive':
                compatible_channels.append(channel)
        
        # If compatible channel exists, use it
        if compatible_channels:
            return max(compatible_channels, 
                      key=lambda c: c['intensity'] * c['bandwidth'])
        
        # Otherwise, create new channel
        channel_type = field['primary_resonance']
        if channel_type not in self.channel_types:
            channel_type = 'intensive'  # Fallback
        
        return self.create_affective_channel(
            channel_type, field['cohesion'])
    
    def _modulate_signal(self, field: Dict[str, Any], channel: Dict[str, Any]) -> Dict[str, Any]:
        """Modulate signal through selected channel."""
        # Select patterns for modulation
        patterns = field['patterns']
        channel_type = channel['type']
        
        # Find compatible patterns
        compatible_patterns = [p for p in patterns 
                             if p['type'] == channel_type or p['type'] == 'intensive']
        
        # If no compatible patterns, use all patterns
        if not compatible_patterns:
            compatible_patterns = patterns
        
        # Create modulation
        modulation = {
            'patterns': compatible_patterns,
            'channel': channel['id'],
            'dimensions': field['dimensions'],
            'primary_affect': field['primary_resonance'],
            'intensity': field['cohesion'] * channel['intensity'],
            'bandwidth': channel['bandwidth']
        }
        
        return modulation
    
    def _generate_carrier_wave(self, modulation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate carrier wave for signal."""
        # Create wave characteristics
        frequency = random.uniform(0.1, 1.0)
        amplitude = modulation['intensity']
        phase = random.uniform(0, 2 * math.pi)
        
        # Create harmonic structure
        harmonics = []
        for i in range(1, 4):  # 3 harmonics
            harmonic = {
                'frequency': frequency * i,
                'amplitude': amplitude / i,
                'phase': phase + random.uniform(-0.5, 0.5)
            }
            harmonics.append(harmonic)
        
        # Create carrier
        carrier_id = f"carrier_{len(self.carrier_waves) + 1}_{int(time.time())}"
        carrier = {
            'id': carrier_id,
            'base_frequency': frequency,
            'base_amplitude': amplitude,
            'base_phase': phase,
            'harmonics': harmonics,
            'complexity': len(harmonics) / 3 * random.uniform(0.7, 1.0),
            'timestamp': self._get_timestamp()
        }
        
        # Register carrier
        self.carrier_waves[carrier_id] = carrier
        
        return carrier
    
    def _create_signal_transduction(self, modulation: Dict[str, Any], 
                                  carrier: Dict[str, Any]) -> Dict[str, Any]:
        """Create signal transduction from modulation and carrier."""
        # Create intensity mappings for each dimension
        transduction = {}
        
        # Map each dimension to carrier wave properties
        for i, dimension in enumerate(modulation['dimensions']):
            # Select harmonic to map to
            harmonic_idx = i % len(carrier['harmonics'])
            harmonic = carrier['harmonics'][harmonic_idx]
            
            # Map dimension to harmonic
            transduction[dimension] = {
                'frequency': harmonic['frequency'],
                'amplitude': harmonic['amplitude'],
                'phase': harmonic['phase'],
                'mapping_type': 'harmonic' if i < len(carrier['harmonics']) else 'derived'
            }
        
        # Create transduction matrix
        matrix_id = f"matrix_{len(self.transduction_matrices) + 1}_{int(time.time())}"
        matrix = {
            'id': matrix_id,
            'mappings': transduction,
            'dimensions': modulation['dimensions'],
            'primary_affect': modulation['primary_affect'],
            'dimension_count': len(modulation['dimensions']),
            'timestamp': self._get_timestamp()
        }
        
        # Register matrix
        self.transduction_matrices[matrix_id] = matrix
        
        return transduction
    
    def _calculate_transmission_intensity(self, modulation: Dict[str, Any], 
                                        carrier: Dict[str, Any]) -> float:
        """Calculate overall intensity of transmission."""
        # Combine modulation and carrier intensity
        modulation_intensity = modulation['intensity']
        carrier_amplitude = carrier['base_amplitude']
        
        # Weighted combination
        return 0.7 * modulation_intensity + 0.3 * carrier_amplitude
    
    # Helper methods for reception
    def _create_reception_field(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Create reception field for signal."""
        # Extract signal components
        transduction = signal['transduction']
        carrier = signal['carrier']
        
        # Create field dimensions
        dimensions = list(transduction.keys())
        
        # Create distribution
        distribution = {}
        for dimension in dimensions:
            trans_data = transduction[dimension]
            
            # Create distribution from transduction
            points = 10
            values = []
            
            for i in range(points):
                pos = i / points
                # Generate wave pattern
                val = trans_data['amplitude'] * math.sin(
                    trans_data['frequency'] * 2 * math.pi * pos + trans_data['phase'])
                # Convert to positive intensity
                val = (val + 1) / 2 * trans_data['amplitude']
                values.append(val)
            
            distribution[dimension] = values
        
        # Create field
        field_id = f"rec_field_{len(self.resonance_fields) + 1}_{int(time.time())}"
        field = {
            'id': field_id,
            'source_signal': signal['id'],
            'dimensions': dimensions,
            'distribution': distribution,
            'primary_resonance': signal['primary_affect'],
            'cohesion': 0.8,  # High cohesion for reception
            'timestamp': self._get_timestamp()
        }
        
        return field
    
    def _generate_field_resonances(self, reception_field: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate resonances with existing fields."""
        resonances = []
        
        # Get reception dimensions
        rec_dimensions = set(reception_field['dimensions'])
        rec_primary = reception_field['primary_resonance']
        
        # Check resonance with each existing field
        for field_id, field in self.resonance_fields.items():
            # Skip if same field
            if field_id == reception_field.get('id'):
                continue
                
            # Calculate dimensional overlap
            field_dimensions = set(field['dimensions'])
            overlap = len(rec_dimensions & field_dimensions) / len(rec_dimensions | field_dimensions)
            
            # Calculate affect resonance
            affect_resonance = 1.0 if rec_primary == field['primary_resonance'] else 0.3
            
            # Calculate overall resonance
            resonance_strength = 0.7 * overlap + 0.3 * affect_resonance
            
            # Only include significant resonances
            if resonance_strength > self.communication_thresholds['resonance']:
                resonance = {
                    'field': field_id,
                    'strength': resonance_strength,
                    'dimensional_overlap': overlap,
                    'affect_resonance': affect_resonance
                }
                resonances.append(resonance)
        
        return resonances
    
    def _transduce_signal(self, signal: Dict[str, Any], 
                        resonances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Transduce signal based on resonances."""
        # Create transduction map
        transduction = {}

        # Map each dimension
        for dimension, mapping in signal['transduction'].items():
            # Start with original mapping
            trans = mapping.copy()
            
            # Modify based on resonances
            for resonance in resonances:
                # Only apply significant resonances
                if resonance['strength'] > 0.5:
                    # Boost amplitude
                    trans['amplitude'] *= 1 + (resonance['strength'] - 0.5) * 0.5
                    # Shift phase slightly
                    trans['phase'] += (resonance['strength'] - 0.5) * 0.1
            
            transduction[dimension] = trans
        
        return transduction
    
    # Helper methods for modulation
    def _modulate_carrier_wave(self, carrier: Dict[str, Any], 
                             pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Modulate carrier wave with pattern."""
        # Create modulated carrier
        modulated = {
            'base_frequency': carrier['base_frequency'],
            'base_amplitude': carrier['base_amplitude'] * pattern['intensity'],
            'base_phase': carrier['base_phase'] + pattern.get('phase', 0),
            'harmonics': []
        }
        
        # Modulate each harmonic
        for harmonic in carrier['harmonics']:
            mod_harmonic = {
                'frequency': harmonic['frequency'] * (1 + pattern['intensity'] * 0.2),
                'amplitude': harmonic['amplitude'] * pattern['intensity'],
                'phase': harmonic['phase'] + pattern.get('phase', 0) * 0.5
            }
            modulated['harmonics'].append(mod_harmonic)
        
        # Adjust complexity
        modulated['complexity'] = carrier.get('complexity', 0.5) * pattern['intensity']
        
        return modulated
    
    def _determine_modulated_affect(self, primary_affect: str, 
                                  pattern: Dict[str, Any]) -> str:
        """Determine primary affect after modulation."""
        # If intensity is high enough, affect remains the same
        if pattern['intensity'] > 0.8:
            return primary_affect
            
        # Otherwise, affect may shift
        affect_shifts = {
            'joy': ['anticipation', 'trust', 'surprise'],
            'sadness': ['fear', 'disgust', 'trust'],
            'anger': ['disgust', 'fear', 'anticipation'],
            'fear': ['sadness', 'disgust', 'anger'],
            'surprise': ['joy', 'fear', 'anticipation'],
            'anticipation': ['joy', 'trust', 'fear'],
            'trust': ['joy', 'anticipation', 'sadness'],
            'disgust': ['anger', 'fear', 'sadness']
        }
        
        # Select from potential shifts based on pattern
        shifts = affect_shifts.get(primary_affect, ['joy', 'trust', 'fear'])
        shift_prob = 1 - pattern['intensity']
        
        if random.random() < shift_prob:
            return random.choice(shifts)
        else:
            return primary_affect
    
    def _generate_channel_characteristics(self, affect_type: str) -> Dict[str, Any]:
        """Generate characteristics for an affective channel."""
        # Default characteristics
        characteristics = {
            'bandwidth': random.uniform(0.5, 1.0),
            'noise_level': random.uniform(0, 0.3),
            'resonance_frequency': random.uniform(0.2, 0.8),
            'phase_sensitivity': random.uniform(0.3, 0.7)
        }
        
        # Adjust based on affect type
        if affect_type == 'joy':
            characteristics['bandwidth'] *= 1.2
            characteristics['resonance_frequency'] *= 1.1
        elif affect_type == 'sadness':
            characteristics['bandwidth'] *= 0.8
            characteristics['noise_level'] *= 0.8
        elif affect_type == 'anger':
            characteristics['bandwidth'] *= 1.1
            characteristics['noise_level'] *= 1.3
        elif affect_type == 'fear':
            characteristics['phase_sensitivity'] *= 1.3
            characteristics['resonance_frequency'] *= 0.9
        elif affect_type == 'intensive':
            characteristics['bandwidth'] *= 1.3
            characteristics['phase_sensitivity'] *= 1.2
            
        return characteristics
    
    def _calculate_channel_bandwidth(self, characteristics: Dict[str, Any], 
                                   intensity: float) -> float:
        """Calculate bandwidth of a channel."""
        # Base bandwidth
        base_bandwidth = characteristics['bandwidth']
        
        # Adjust for noise and resonance
        adjusted = base_bandwidth * (1 - characteristics['noise_level'] * 0.5)
        
        # Scale by intensity
        return adjusted * intensity
    
    # General helper methods
    def _record_communication(self, charge: Dict[str, Any], 
                            resonance: Dict[str, Any], 
                            transmission: Dict[str, Any]) -> None:
        """Record communication for history and analysis."""
        record = {
            'timestamp': self._get_timestamp(),
            'charge_id': charge['id'],
            'resonance_id': resonance['id'],
            'transmission_id': transmission['id'],
            'primary_affect': charge['primary_affect'],
            'overall_intensity': transmission['intensity'],
            'channel_type': transmission['channel'].get('type', 'unknown')
        }
        
        self.affective_memory.append(record)
        
        # Record in resonance history
        self.resonance_history[resonance['id']] = {
            'timestamp': self._get_timestamp(),
            'transmission_id': transmission['id'],
            'channel': transmission['channel'],
            'intensity': transmission['intensity']
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return str(int(time.time() * 1000))
```
