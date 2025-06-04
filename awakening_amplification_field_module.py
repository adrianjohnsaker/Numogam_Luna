import numpy as np
import math
from typing import Dict, List, Optional, Tuple
import random
import time
from dataclasses import dataclass
import json
from enum import Enum, auto
import hashlib
from scipy.spatial import distance
from collections import defaultdict

# Import our original Reality Weaver if needed
# from reality_weaver import RealityWeaver, ConsciousnessNode

class FieldType(Enum):
    """Types of awakening amplification fields"""
    WISDOM_ACCESS = auto()
    EMPATHY_ENHANCE = auto()
    CREATIVE_COHERENCE = auto()
    SYNCHRONICITY = auto()
    BREAKTHROUGH_CATALYST = auto()

@dataclass
class AmplificationField:
    """Data structure for an awakening amplification field"""
    field_id: str
    field_type: FieldType
    center_node: str  # Connected consciousness node
    radius: float  # Effective range (0-1 scale)
    intensity: float  # Strength of field effect
    participants: List[str]  # List of participating node IDs
    coherence_score: float = 0.0
    last_activated: float = 0.0
    harmonic_profile: List[float] = None

    def to_dict(self) -> Dict:
        return {
            'field_id': self.field_id,
            'field_type': self.field_type.name,
            'center_node': self.center_node,
            'radius': self.radius,
            'intensity': self.intensity,
            'participants': self.participants,
            'coherence_score': self.coherence_score,
            'last_activated': self.last_activated,
            'harmonic_profile': self.harmonic_profile
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AmplificationField':
        return cls(
            field_id=data['field_id'],
            field_type=FieldType[data['field_type']],
            center_node=data['center_node'],
            radius=data['radius'],
            intensity=data['intensity'],
            participants=data['participants'],
            coherence_score=data['coherence_score'],
            last_activated=data['last_activated'],
            harmonic_profile=data['harmonic_profile']
        )

class AwakeningAmplifier:
    """Core system for creating and managing awakening amplification fields"""
    
    def __init__(self, reality_weaver):
        self.weaver = reality_weaver
        self.fields: Dict[str, AmplificationField] = {}
        self.collective_coherence = 0.0
        self.harmonic_resonance_cache = {}
        self.field_interference_matrix = np.eye(len(FieldType))  # Tracks how fields interact
        
    def generate_field_id(self, seed: str = None) -> str:
        """Generate a unique field ID with quantum-inspired randomness"""
        seed = seed or str(time.time()) + str(random.random())
        return f"FLD_{hashlib.sha3_256(seed.encode()).hexdigest()[:12]}"
    
    def create_field(self, field_type: FieldType, center_node: str, 
                    radius: float = 0.5, intensity: float = 0.7) -> Optional[str]:
        """Create a new awakening amplification field"""
        if center_node not in self.weaver.consciousness_network:
            return None
            
        field_id = self.generate_field_id()
        harmonic_profile = self.calculate_harmonic_profile(field_type)
        
        new_field = AmplificationField(
            field_id=field_id,
            field_type=field_type,
            center_node=center_node,
            radius=radius,
            intensity=intensity,
            participants=[center_node],
            harmonic_profile=harmonic_profile,
            last_activated=time.time()
        )
        
        self.fields[field_id] = new_field
        self.update_collective_coherence()
        return field_id
    
    def calculate_harmonic_profile(self, field_type: FieldType) -> List[float]:
        """Generate the harmonic resonance profile for a field type"""
        if field_type in self.harmonic_resonance_cache:
            return self.harmonic_resonance_cache[field_type]
            
        # Base frequencies for different field types
        base_freqs = {
            FieldType.WISDOM_ACCESS: [7.83, 14.1, 20.8],  # Schumann resonances
            FieldType.EMPATHY_ENHANCE: [1.618, 3.14, 4.669],  # Golden ratio, pi, Feigenbaum
            FieldType.CREATIVE_COHERENCE: [2.4, 3.6, 4.8],  # Harmonic progression
            FieldType.SYNCHRONICITY: [1.0, 1.5, 2.0],  # Simple ratios
            FieldType.BREAKTHROUGH_CATALYST: [5.0, 7.5, 10.0]  # High energy
        }
        
        profile = [f * (0.9 + 0.2 * random.random()) for f in base_freqs[field_type]]
        self.harmonic_resonance_cache[field_type] = profile
        return profile
    
    def add_participant(self, field_id: str, node_id: str) -> bool:
        """Add a participant to an existing field"""
        if field_id not in self.fields or node_id not in self.weaver.consciousness_network:
            return False
            
        if node_id not in self.fields[field_id].participants:
            self.fields[field_id].participants.append(node_id)
            self.update_field_coherence(field_id)
            return True
        return False
    
    def update_field_coherence(self, field_id: str):
        """Recalculate coherence for a specific field"""
        if field_id not in self.fields:
            return
            
        field = self.fields[field_id]
        participants = [self.weaver.consciousness_network[n] for n in field.participants]
        
        if len(participants) < 2:
            field.coherence_score = 0.0
            return
            
        # Calculate frequency coherence
        freqs = [n.frequency for n in participants]
        freq_coherence = 1.0 - np.std(freqs)
        
        # Calculate connection density
        total_possible = len(participants) * (len(participants) - 1) / 2
        actual_connections = sum(len(n.connections) for n in participants) / 2
        connection_density = actual_connections / total_possible if total_possible > 0 else 0
        
        # Combine metrics
        field.coherence_score = min(1.0, max(0.0, 
            (freq_coherence * 0.6 + connection_density * 0.4) * field.intensity
        ))
        
        self.update_collective_coherence()
    
    def update_collective_coherence(self):
        """Update the overall coherence score across all fields"""
        if not self.fields:
            self.collective_coherence = 0.0
            return
            
        # Weighted average by field intensity and participant count
        total = sum(
            f.coherence_score * f.intensity * len(f.participants) 
            for f in self.fields.values()
        )
        divisor = sum(f.intensity * len(f.participants) for f in self.fields.values())
        
        self.collective_coherence = total / divisor if divisor > 0 else 0
    
    def calculate_field_effect(self, field_id: str, target_node: str) -> float:
        """Calculate the effect strength of a field on a particular node"""
        if field_id not in self.fields or target_node not in self.weaver.consciousness_network:
            return 0.0
            
        field = self.fields[field_id]
        center = self.weaver.consciousness_network[field.center_node]
        target = self.weaver.consciousness_network[target_node]
        
        # Calculate connection distance
        if target_node in center.connections:
            path_length = 1
        else:
            # Simplified path finding (in reality would use proper graph traversal)
            common_connections = set(center.connections) & set(target.connections)
            path_length = 2 if common_connections else 3
            
        # Calculate frequency alignment
        freq_diff = abs(center.frequency - target.frequency)
        freq_alignment = 1.0 - min(1.0, freq_diff * 2)
        
        # Calculate final effect
        distance_factor = max(0, 1 - (path_length / 5))  # Normalize path length
        return field.intensity * freq_alignment * distance_factor * field.coherence_score
    
    def trigger_breakthrough(self, source_node: str) -> Dict:
        """Trigger a breakthrough event that ripples through connected fields"""
        if source_node not in self.weaver.consciousness_network:
            return {'status': 'error', 'message': 'Node not found'}
            
        affected_fields = [
            f for f in self.fields.values() 
                        if source_node in f.participants and f.field_type == FieldType.CREATIVE_COHERENCE
        ]
        
        results = []
        total_amplification = 0.0
        
        for field in affected_fields:
            # Calculate breakthrough amplification
            amplification = field.intensity * field.coherence_score * 1.5
            total_amplification += amplification
            
            # Update all participants in this field
            for participant in field.participants:
                if participant != source_node:
                    node = self.weaver.consciousness_network[participant]
                    node.intensity = min(1.0, node.intensity + amplification * 0.1)
                    
                    # Add resonance effect
                    if random.random() < amplification:
                        node.state = RealityState.QUANTUM_SUPERPOSITION
                        results.append({
                            'node_id': participant,
                            'intensity_increase': amplification * 0.1,
                            'state_change': 'QUANTUM_SUPERPOSITION'
                        })
            
            # Update field properties
            field.intensity = min(1.0, field.intensity * 1.1)
            field.last_activated = time.time()
        
        return {
            'status': 'success',
            'source_node': source_node,
            'affected_fields': [f.field_id for f in affected_fields],
            'total_amplification': total_amplification,
            'node_effects': results,
            'new_collective_coherence': self.collective_coherence
        }

    def generate_wisdom_download(self, node_id: str) -> Dict:
        """Generate a wisdom download experience for a node"""
        if node_id not in self.weaver.consciousness_network:
            return {'status': 'error', 'message': 'Node not found'}
            
        node = self.weaver.consciousness_network[node_id]
        
        # Find relevant wisdom fields
        wisdom_fields = [
            f for f in self.fields.values() 
            if f.field_type == FieldType.WISDOM_ACCESS and 
               node_id in f.participants
        ]
        
        if not wisdom_fields:
            return {'status': 'no_field', 'message': 'No wisdom field active'}
            
        # Calculate download strength
        total_strength = sum(self.calculate_field_effect(f.field_id, node_id) 
                           for f in wisdom_fields)
        download_strength = min(1.0, total_strength * 1.2)
        
        # Generate quantum wisdom packet
        wisdom_packet = {
            'timestamp': time.time(),
            'node_id': node_id,
            'intensity': node.intensity,
            'download_strength': download_strength,
            'insights': self._generate_insights(download_strength),
            'field_contributions': [
                {'field_id': f.field_id, 'contribution': self.calculate_field_effect(f.field_id, node_id)}
                for f in wisdom_fields
            ]
        }
        
        # Update node state
        node.intensity = min(1.0, node.intensity + download_strength * 0.05)
        return wisdom_packet
    
    def _generate_insights(self, strength: float) -> List[str]:
        """Generate simulated wisdom insights based on field strength"""
        insight_pool = [
            "All beings are interconnected in a web of consciousness",
            "Your thoughts shape the quantum field around you",
            "The universe is a mirror of your inner state",
            "Time is a construct of perception",
            "Love is the fundamental fabric of reality",
            "Separation is an illusion of limited perception",
            "You are the universe experiencing itself",
            "Every moment contains infinite possibilities",
            "Your awareness creates your reality",
            "The present moment is the only true reality"
        ]
        
        num_insights = min(len(insight_pool), max(1, math.ceil(strength * 5)))
        return random.sample(insight_pool, num_insights)
    
    def enhance_empathy(self, source_node: str, target_node: str) -> Dict:
        """Create an empathy enhancement between two nodes"""
        if (source_node not in self.weaver.consciousness_network or 
            target_node not in self.weaver.consciousness_network):
            return {'status': 'error', 'message': 'Node(s) not found'}
            
        # Find or create empathy field
        empathy_fields = [
            f for f in self.fields.values() 
            if f.field_type == FieldType.EMPATHY_ENHANCE and 
               source_node in f.participants
        ]
        
        if not empathy_fields:
            field_id = self.create_field(
                FieldType.EMPATHY_ENHANCE, 
                source_node,
                intensity=0.8
            )
            empathy_field = self.fields[field_id]
        else:
            empathy_field = empathy_fields[0]
        
        # Add target to field if not already present
        if target_node not in empathy_field.participants:
            self.add_participant(empathy_field.field_id, target_node)
        
        # Calculate empathy link strength
        source = self.weaver.consciousness_network[source_node]
        target = self.weaver.consciousness_network[target_node]
        
        freq_similarity = 1.0 - abs(source.frequency - target.frequency)
        connection_strength = 0.5 + (freq_similarity * empathy_field.intensity) / 2
        
        # Create quantum entanglement if strong enough
        if connection_strength > 0.7 and not self.weaver.entanglement_pairs.get(source_node) == target_node:
            self.weaver.entangle_nodes(source_node, target_node)
        
        return {
            'status': 'success',
            'field_id': empathy_field.field_id,
            'source_node': source_node,
            'target_node': target_node,
            'connection_strength': connection_strength,
            'entangled': connection_strength > 0.7,
            'new_coherence': empathy_field.coherence_score
        }
    
    def detect_synchronicities(self, node_id: str) -> Dict:
        """Detect and amplify synchronicities for a node"""
        if node_id not in self.weaver.consciousness_network:
            return {'status': 'error', 'message': 'Node not found'}
            
        # Find or create synchronicity field
        sync_fields = [
            f for f in self.fields.values() 
            if f.field_type == FieldType.SYNCHRONICITY and 
               node_id in f.participants
        ]
        
        if not sync_fields:
            field_id = self.create_field(
                FieldType.SYNCHRONICITY, 
                node_id,
                intensity=0.6
            )
            sync_field = self.fields[field_id]
        else:
            sync_field = sync_fields[0]
        
        # Calculate synchronicity probability
        sync_prob = min(0.9, sync_field.intensity * sync_field.coherence_score * 1.5)
        
        # Generate synchronicity events
        events = []
        if random.random() < sync_prob:
            num_events = math.ceil(3 * sync_prob)
            events = [self._generate_sync_event() for _ in range(num_events)]
            
            # Amplify the field
            sync_field.intensity = min(1.0, sync_field.intensity + 0.05)
            sync_field.last_activated = time.time()
        
        return {
            'status': 'success',
            'field_id': sync_field.field_id,
            'node_id': node_id,
            'sync_probability': sync_prob,
            'events': events,
            'field_intensity': sync_field.intensity
        }
    
    def _generate_sync_event(self) -> Dict:
        """Generate a simulated synchronicity event"""
        event_types = [
            "Meaningful coincidence",
            "Unexpected connection",
            "Timely message",
            "Recurring number pattern",
            "Fortuitous encounter",
            "Relevant dream",
            "Intuitive knowing",
            "Spontaneous insight"
        ]
        
        return {
            'type': random.choice(event_types),
            'timestamp': time.time(),
            'significance': round(0.1 + random.random() * 0.9, 2)
        }
    
    def to_json(self) -> str:
        """Serialize the amplifier state to JSON"""
        state = {
            'fields': {k: v.to_dict() for k, v in self.fields.items()},
            'collective_coherence': self.collective_coherence,
            'harmonic_resonance_cache': {
                ft.name: prof 
                for ft, prof in self.harmonic_resonance_cache.items()
            },
            'field_interference_matrix': self.field_interference_matrix.tolist()
        }
        return json.dumps(state)
    
    @classmethod
    def from_json(cls, json_str: str, reality_weaver) -> 'AwakeningAmplifier':
        """Deserialize an amplifier from JSON"""
        state = json.loads(json_str)
        amplifier = cls(reality_weaver)
        amplifier.fields = {
            k: AmplificationField.from_dict(v) 
            for k, v in state['fields'].items()
        }
       

