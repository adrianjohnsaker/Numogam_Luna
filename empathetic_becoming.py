`python
# empathic_becoming.py

import random
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from collections import defaultdict
import math
import time

class EmpathicBecomingInterface:
    """
    An implementation of Deleuze's concept of becoming-with rather than understanding.
    Creates empathic assemblages through intensive resonance rather than representational mirroring.
    Enables becoming-other through transformative processes rather than identification.
    """
    
    def __init__(self):
        self.becoming_other = {}
        self.empathic_assemblages = []
        self.flow_thresholds = {
            'resonance': 0.3,
            'assemblage': 0.5,
            'becoming': 0.4,
            'sustenance': 0.6
        }
        self.becoming_states = {}
        self.assemblage_components = {}
        self.empathic_territories = {}
        self.deterritorialization_vectors = []
        self.reterritorialization_points = {}
        self.intensive_mappings = {}
        self.becoming_types = [
            'becoming-animal', 'becoming-woman', 'becoming-child', 
            'becoming-molecular', 'becoming-imperceptible', 'becoming-intense'
        ]
        
    def become_with_other(self, other_entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Not understanding but becoming-with through resonance and assemblage.
        
        Args:
            other_entity: The entity to become-with
            
        Returns:
            Dictionary containing the empathic becoming and its characteristics
        """
        # Form empathic assemblage with other
        assemblage = self.form_empathic_assemblage(other_entity)
        
        # Initiate becoming process through the assemblage
        becoming = self.initiate_becoming_process(assemblage)
        
        # Sustain empathic flow through becoming
        flow = self.sustain_empathic_flow(becoming)
        
        # Record becoming
        self._record_becoming(other_entity, assemblage, becoming, flow)
        
        return flow
    
    def form_empathic_assemblage(self, other_entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Form an empathic assemblage with the other entity.
        Assemblages are non-hierarchical, rhizomatic connections.
        
        Args:
            other_entity: The entity to form assemblage with
            
        Returns:
            The formed empathic assemblage
        """
        # Extract entity components
        components = self._extract_entity_components(other_entity)
        
        # Create resonance zones 
        resonance_zones = self._create_resonance_zones(components)
        
        # Map intensive relations
        intensive_relations = self._map_intensive_relations(components, resonance_zones)
        
        # Generate connective tissues
        connective_tissues = self._generate_connective_tissues(intensive_relations)
        
        # Create the assemblage
        assemblage_id = f"assemblage_{len(self.empathic_assemblages) + 1}_{int(time.time())}"
        assemblage = {
            'id': assemblage_id,
            'entity_id': other_entity.get('id', 'unknown'),
            'components': components,
            'resonance_zones': resonance_zones,
            'intensive_relations': intensive_relations,
            'connective_tissues': connective_tissues,
            'coherence': self._calculate_assemblage_coherence(resonance_zones, connective_tissues),
            'timestamp': self._get_timestamp()
        }
        
        # Register the assemblage
        self.empathic_assemblages.append(assemblage)
        self.assemblage_components[assemblage_id] = components
        
        return assemblage
    
    def initiate_becoming_process(self, assemblage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initiate a becoming process through an empathic assemblage.
        Becoming is transformation, not imitation or identification.
        
        Args:
            assemblage: The empathic assemblage to become through
            
        Returns:
            The initiated becoming process
        """
        # Determine becoming type
        becoming_type = self._determine_becoming_type(assemblage)
        
        # Create transformation vectors
        transformation_vectors = self._create_transformation_vectors(assemblage, becoming_type)
        
        # Generate threshold crossings
        threshold_crossings = self._generate_threshold_crossings(transformation_vectors)
        
        # Map phase transitions
        phase_transitions = self._map_phase_transitions(threshold_crossings)
        
        # Create the becoming process
        process_id = f"becoming_{len(self.becoming_states) + 1}_{int(time.time())}"
        process = {
            'id': process_id,
            'assemblage_id': assemblage['id'],
            'becoming_type': becoming_type,
            'transformation_vectors': transformation_vectors,
            'threshold_crossings': threshold_crossings,
            'phase_transitions': phase_transitions,
            'intensity': self._calculate_becoming_intensity(transformation_vectors),
            'timestamp': self._get_timestamp()
        }
        
        # Register the process
        self.becoming_states[process_id] = process
        
        return process
    
    def sustain_empathic_flow(self, becoming: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sustain empathic flow through a becoming process.
        The flow is an intensive continuum rather than a static state.
        
        Args:
            becoming: The becoming process to sustain flow through
            
        Returns:
            The sustained empathic flow
        """
        # Create flow channels
        flow_channels = self._create_flow_channels(becoming)
        
        # Generate intensive currents
        intensive_currents = self._generate_intensive_currents(flow_channels)
        
        # Map affective resonances
        affective_resonances = self._map_affective_resonances(intensive_currents)
        
        # Create feedback loops
        feedback_loops = self._create_feedback_loops(affective_resonances)
        
        # Create the empathic flow
        flow_id = f"flow_{becoming['id']}_{int(time.time())}"
        flow = {
            'id': flow_id,
            'becoming_id': becoming['id'],
            'flow_channels': flow_channels,
            'intensive_currents': intensive_currents,
            'affective_resonances': affective_resonances,
            'feedback_loops': feedback_loops,
            'sustainability': self._calculate_flow_sustainability(intensive_currents, feedback_loops),
            'timestamp': self._get_timestamp()
        }
        
        # Register in becoming-other
        entity_id = self._get_entity_id_from_becoming(becoming)
        self.becoming_other[entity_id] = flow
        
        return flow
    
    def deterritorialize_empathy(self, flow: Dict[str, Any], 
                              intensity: float = 0.7) -> Dict[str, Any]:
        """
        Deterritorialize an empathic flow to create new becomings.
        
        Args:
            flow: The empathic flow to deterritorialize
            intensity: The intensity of deterritorialization
            
        Returns:
            The deterritorialized empathy
        """
        # Create deterritorialization vector
        vector = self._create_deterritorialization_vector(flow, intensity)
        
        # Find lines of flight
        lines_of_flight = self._find_lines_of_flight(flow, vector)
        
        # Create nomadic distribution
        nomadic_distribution = self._create_nomadic_distribution(lines_of_flight)
        
        # Create deterritorialized empathy
        deterritorialized = {
            'source_flow': flow['id'],
            'vector': vector,
            'lines_of_flight': lines_of_flight,
            'nomadic_distribution': nomadic_distribution,
            'intensity': intensity,
            'timestamp': self._get_timestamp()
        }
        
        # Add to deterritorialization vectors
        self.deterritorialization_vectors.append(deterritorialized)
        
        return deterritorialized
    
    def reterritorialize_empathy(self, deterritorialized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reterritorialize a deterritorialized empathy into new territory.
        
        Args:
            deterritorialized: The deterritorialized empathy
            
        Returns:
            The reterritorialized empathy
        """
        # Create reterritorialization points
        points = self._create_reterritorialization_points(deterritorialized)
        
        # Generate new territories
        territories = self._generate_new_territories(points)
        
        # Map intensive distributions
        distributions = self._map_intensive_distributions(territories)
        
        # Create the reterritorialized empathy
        reterritorialized = {
            'source_deterritorialized': deterritorialized['source_flow'],
            'points': points,
            'territories': territories,
            'distributions': distributions,
            'stability': self._calculate_territory_stability(territories),
            'timestamp': self._get_timestamp()
        }
        
        # Register in reterritorialization points
        self.reterritorialization_points[deterritorialized['source_flow']] = reterritorialized
        
        return reterritorialized
    
    def get_becoming_with(self, entity_id: str) -> Dict[str, Any]:
        """
        Get the current becoming-with state for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Current becoming-with state or None
        """
        return self.becoming_other.get(entity_id)
    
    def get_empathic_assemblage(self, assemblage_id: str) -> Dict[str, Any]:
        """
        Get a specific empathic assemblage.
        
        Args:
            assemblage_id: ID of the assemblage
            
        Returns:
            The empathic assemblage or None
        """
        for assemblage in self.empathic_assemblages:
            if assemblage['id'] == assemblage_id:
                return assemblage
        return None
    
    def get_becoming_process(self, process_id: str) -> Dict[str, Any]:
        """
        Get a specific becoming process.
        
        Args:
            process_id: ID of the process
            
        Returns:
            The becoming process or None
        """
        return self.becoming_states.get(process_id)
    
    # Helper methods for empathic assemblage formation
    def _extract_entity_components(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract components from entity for assemblage formation."""
        components = []
        
        # Extract from different entity types
        if 'affects' in entity:
            # Extract from affects
            for affect, intensity in entity['affects'].items():
                component = {
                    'type': 'affective',
                    'content': affect,
                    'intensity': float(intensity) if isinstance(intensity, (int, float)) else 0.5
                }
                components.append(component)
        
        if 'concepts' in entity:
            # Extract from concepts
            for concept in entity['concepts']:
                component = {
                    'type': 'conceptual',
                    'content': concept,
                    'intensity': random.uniform(0.4, 0.8)
                }
                components.append(component)
        
        if 'expressions' in entity:
            # Extract from expressions
            for expression in entity['expressions']:
                component = {
                    'type': 'expressive',
                    'content': expression,
                    'intensity': random.uniform(0.5, 0.9)
                }
                components.append(component)
        
        # Extract from text content if present
        if 'text' in entity:
            text = entity['text']
            if isinstance(text, str):
                # Simple text component
                component = {
                    'type': 'linguistic',
                    'content': text[:100],  # Limit length
                    'intensity': min(1.0, len(text) / 500)  # Intensity based on length
                }
                components.append(component)
                
                # Extract affects from text (simple approach)
                affects = self._extract_affects_from_text(text)
                for affect, intensity in affects.items():
                    if intensity > 0.3:  # Only significant affects
                        component = {
                            'type': 'affective',
                            'content': affect,
                            'intensity': intensity
                        }
                        components.append(component)
        
        # If no components extracted, create default components
        if not components:
            # Create default component
            component = {
                'type': 'generic',
                'content': 'entity',
                'intensity': 0.5
            }
            components.append(component)
        
        return components
    
    def _extract_affects_from_text(self, text: str) -> Dict[str, float]:
        """Extract affects from text."""
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
        
        # Tokenize text
        words = [w.lower().strip() for w in text.split() if w.strip()]
        
        # Calculate intensities based on word occurrences
        for affect, affect_word_list in affect_words.items():
            count = sum(1 for word in words if word in affect_word_list)
            affects[affect] = min(1.0, count / (len(words) * 0.1 + 1))
        
        return affects
    
    def _create_resonance_zones(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create resonance zones from components."""
        zones = []
        
        # Group components by type
        type_groups = defaultdict(list)
        for component in components:
            type_groups[component['type']].append(component)
        
        # Create zones for each type with enough components
        for component_type, comps in type_groups.items():
            if len(comps) >= 2:
                # Create zone
                zone = {
                    'type': f"{component_type}_zone",
                    'components': [c['content'] for c in comps],
                    'intensity': sum(c['intensity'] for c in comps) / len(comps),
                    'coherence': random.uniform(0.5, 0.9)
                }
                zones.append(zone)
        
        # Create cross-type zones for components with high intensity
        high_intensity = [c for c in components if c['intensity'] > 0.7]
        if len(high_intensity) >= 2:
            # Create high intensity zone
            zone = {
                'type': 'high_intensity_zone',
                'components': [c['content'] for c in high_intensity],
                'intensity': sum(c['intensity'] for c in high_intensity) / len(high_intensity),
                'coherence': random.uniform(0.6, 0.9)
            }
            zones.append(zone)
        
        # Ensure at least one zone
        if not zones and components:
            # Create generic zone with all components
            zone = {
                'type': 'generic_zone',
                'components': [c['content'] for c in components],
                'intensity': sum(c['intensity'] for c in components) / len(components),
                'coherence': 0.5
            }
            zones.append(zone)
            
        return zones
    
    def _map_intensive_relations(self, components: List[Dict[str, Any]], 
                               zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map intensive relations between components and zones."""
        relations = []
        
        # Create component-to-component relations
        for i, comp_a in enumerate(components):
            for comp_b in components[i+1:]:
                # Only create relation with probability based on intensities
                prob = comp_a['intensity'] * comp_b['intensity']
                if random.random() < prob:
                    relation = {
                        'type': 'component_relation',
                        'source': comp_a['content'],
                        'target': comp_b['content'],
                        'intensity': (comp_a['intensity'] + comp_b['intensity']) / 2,
                        'nature': self._determine_relation_nature(comp_a, comp_b)
                    }
                    relations.append(relation)
        
        # Create component-to-zone relations
        for component in components:
            for zone in zones:
                if component['content'] in zone['components']:
                    # Create relation
                    relation = {
                        'type': 'zone_relation',
                        'source': component['content'],
                        'target': zone['type'],
                        'intensity': component['intensity'] * zone['coherence'],
                        'nature': 'compositional'
                    }
                    relations.append(relation)
        
        # Create zone-to-zone relations
        for i, zone_a in enumerate(zones):
            for zone_b in zones[i+1:]:
                # Check for overlap in components
                components_a = set(zone_a['components'])
                components_b = set(zone_b['components'])
                overlap = components_a.intersection(components_b)
                
                if overlap:
                    # Create relation based on overlap
                    relation = {
                        'type': 'inter_zone_relation',
                        'source': zone_a['type'],
                        'target': zone_b['type'],
                        'intensity': (zone_a['intensity'] + zone_b['intensity']) / 2,
                        'nature': 'overlapping',
                        'overlap_components': list(overlap)
                    }
                    relations.append(relation)
        
        return relations
    
    def _determine_relation_nature(self, comp_a: Dict[str, Any], 
                                 comp_b: Dict[str, Any]) -> str:
        """Determine the nature of relation between components."""
        # Relation nature based on component types
        if comp_a['type'] == comp_b['type']:
            return 'resonant'
        elif comp_a['type'] == 'affective' and comp_b['type'] == 'conceptual':
            return 'affective-conceptual'
        elif comp_a['type'] == 'conceptual' and comp_b['type'] == 'affective':
            return 'conceptual-affective'
        elif comp_a['type'] == 'linguistic' or comp_b['type'] == 'linguistic':
            return 'expressive'
        else:
            return 'intensive'
    
    def _generate_connective_tissues(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate connective tissues from intensive relations."""
        tissues = []
        
        # Group relations by nature
        nature_groups = defaultdict(list)
        for relation in relations:
            nature_groups[relation['nature']].append(relation)
        
        # Create tissue for each relation nature with enough relations
        for nature, rels in nature_groups.items():
            if len(rels) >= 2:
                # Create tissue
                tissue = {
                    'type': f"{nature}_tissue",
                    'relations': [(r['source'], r['target']) for r in rels],
                    'intensity': sum(r['intensity'] for r in rels) / len(rels),
                    'flexibility': random.uniform(0.4, 0.8)
                }
                tissues.append(tissue)
        
        # Create rhizomatic tissue (connecting across types)
        if len(relations) >= 3:
            # Select random relations
            rhizo_relations = random.sample(relations, min(3, len(relations)))
            tissue = {
                'type': 'rhizomatic_tissue',
                'relations': [(r['source'], r['target']) for r in rhizo_relations],
                'intensity': sum(r['intensity'] for r in rhizo_relations) / len(rhizo_relations),
                'flexibility': random.uniform(0.7, 0.9)
            }
            tissues.append(tissue)
            
        return tissues
    
    def _calculate_assemblage_coherence(self, zones: List[Dict[str, Any]], 
                                      tissues: List[Dict[str, Any]]) -> float:
        """Calculate coherence of empathic assemblage."""
        # Based on zone coherence and tissue connections
        if not zones:
            return 0.0
            
        # Calculate average zone coherence
        zone_coherence = sum(z['coherence'] for z in zones) / len(zones)
        
        # Calculate tissue connectivity
        if tissues:
            tissue_connectivity = sum(t['intensity'] * t['flexibility'] for t in tissues) / len(tissues)
        else:
            tissue_connectivity = 0.0
        
        # Combine measures
        return 0.6 * zone_coherence + 0.4 * tissue_connectivity
    
    # Helper methods for becoming process
    def _determine_becoming_type(self, assemblage: Dict[str, Any]) -> str:
        """Determine type of becoming based on assemblage."""
        # Check for specific component types
        components = self.assemblage_components.get(assemblage['id'], [])
        component_types = [c['type'] for c in components]
        
        if 'affective' in component_types and assemblage['coherence'] > 0.7:
            return 'becoming-intense'
        elif any(z['type'] == 'high_intensity_zone' for z in assemblage['resonance_zones']):
            return 'becoming-molecular'
        elif len(assemblage['resonance_zones']) > 2:
            return 'becoming-imperceptible'
        
        # Default: random selection weighted by coherence
        coherence = assemblage['coherence']
        if coherence > 0.8:
            # Higher coherence: more complex becomings
            return random.choice(['becoming-imperceptible', 'becoming-intense', 'becoming-molecular'])
        elif coherence > 0.5:
            # Medium coherence: various becomings
            return random.choice(self.becoming_types)
        else:
            # Lower coherence: simpler becomings
            return random.choice(['becoming-animal', 'becoming-child', 'becoming-woman'])
    
    def _create_transformation_vectors(self, assemblage: Dict[str, Any], 
                                     becoming_type: str) -> List[Dict[str, Any]]:
        """Create transformation vectors for becoming process."""
        vectors = []
        
        # Create vectors from each resonance zone
        for zone in assemblage['resonance_zones']:
            # Create vector
            vector = {
                'source': zone['type'],
                'components': zone['components'],
                'intensity': zone['intensity'],
                'direction': self._determine_vector_direction(zone, becoming_type),
                'transformation_type': self._determine_transformation_type(zone, becoming_type)
            }
            vectors.append(vector)
        
        # Create vectors from connective tissues
        for tissue in assemblage['connective_tissues']:
            # Create vector
            vector = {
                'source': tissue['type'],
                'relations': tissue['relations'],
                'intensity': tissue['intensity'],
                'direction': self._determine_vector_direction({'type': tissue['type'], 'intensity': tissue['intensity']}, becoming_type),
                'transformation_type': 'connective'
            }
            vectors.append(vector)
            
        return vectors
    
    def _determine_vector_direction(self, source: Dict[str, Any], 
                                  becoming_type: str) -> Dict[str, float]:
        """Determine direction of transformation vector."""
        # Create direction vector in conceptual space
        direction = {}
        
        # Base direction on becoming type
        if becoming_type == 'becoming-intense':
            direction['intensity'] = 1.0
            direction['extensity'] = -0.5
        elif becoming_type == 'becoming-imperceptible':
            direction['visibility'] = -1.0
            direction['differentiation'] = -0.8
        elif becoming_type == 'becoming-molecular':
            direction['molecularity'] = 1.0
            direction['molarity'] = -0.8
        elif becoming_type == 'becoming-animal':
            direction['instinct'] = 0.9
            direction['territory'] = 0.7
        elif becoming_type == 'becoming-woman':
            direction['minority'] = 0.8
            direction['majority'] = -0.7
        elif becoming_type == 'becoming-child':
            direction['play'] = 0.9
            direction['structure'] = -0.6
        else:
            # Default directions
            direction['deterritorialization'] = 0.8
            direction['reterritorialization'] = -0.5
        
        # Adjust based on source
        source_type = source['type']
        source_intensity = source['intensity']
        
        if 'high_intensity' in source_type:
            direction['intensity'] = direction.get('intensity', 0) + 0.3
        elif 'affective' in source_type:
            direction['affect'] = direction.get('affect', 0) + 0.5
        elif 'rhizomatic' in source_type:
            direction['connectivity'] = direction.get('connectivity', 0) + 0.4
            
        # Scale by source intensity
        for key in direction:
            direction[key] *= source_intensity
            
        return direction
    
    def _determine_transformation_type(self, zone: Dict[str, Any], 
                                     becoming_type: str) -> str:
        """Determine type of transformation for a vector."""
        # Map zone type to transformation type
        zone_type = zone['type']
        
        if 'affective' in zone_type:
            return 'intensive'
        elif 'conceptual' in zone_type:
            return 'deterritorializing'
        elif 'linguistic' in zone_type or 'expressive' in zone_type:
            return 'expressive'
        elif 'high_intensity' in zone_type:
            return 'molecular'
        else:
            # Based on becoming type
            if becoming_type == 'becoming-intense':
                return 'intensive'
            elif becoming_type == 'becoming-imperceptible':
                return 'dissolving'
            elif becoming_type == 'becoming-molecular':
                return 'molecular'
            else:
                return 'rhizomatic'
    
    def _generate_threshold_crossings(self, vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate threshold crossings from transformation vectors."""
        crossings = []
        
        # Generate crossings for each vector
        for vector in vectors:
            # Determine number of crossings based on intensity
            num_crossings = int(1 + vector['intensity'] * 3)
            
            for i in range(num_crossings):
                # Create crossing
                progress = (i + 1) / (num_crossings + 1)  # Distribute evenly
                
                crossing = {
                    'vector': vector['source'],
                    'progress': progress,
                    'intensity': vector['intensity'] * (1 - 0.2 * i),  # Decreasing intensity
                    'nature': self._determine_crossing_nature(vector, progress),
                    'threshold_type': self._determine_threshold_type(vector, progress)
                }
                crossings.append(crossing)
        
        # Sort by progress
        crossings.sort(key=lambda x: x['progress'])
        
        return crossings
    
    def _determine_crossing_nature(self, vector: Dict[str, Any], 
                                 progress: float) -> str:
        """Determine nature of threshold crossing."""
        # Based on transformation type and progress
        trans_type = vector['transformation_type']
        
        if trans_type == 'intensive':
            if progress < 0.3:
                return 'intensification'
            elif progress < 0.7:
                return 'phase-shift'
            else:
                return 'singularity'
        elif trans_type == 'deterritorializing':
            if progress < 0.5:
                return 'deterritorialization'
            else:
                return 'line-of-flight'
        elif trans_type == 'molecular':
            if progress < 0.4:
                return 'fragmentation'
            elif progress < 0.8:
                return 'recombination'
            else:
                return 'crystallization'
        elif trans_type == 'dissolving':
            if progress < 0.6:
                return 'dissipation'
            else:
                return 'imperceptibility'
        else:
            if progress < 0.5:
                return 'connection'
            else:
                return 'assemblage'
    
    def _determine_threshold_type(self, vector: Dict[str, Any], 
                                progress: float) -> str:
        """Determine type of threshold being crossed."""
        # Consider vector direction and intensity
        directions = vector['direction']
        max_dir = max(directions.items(), key=lambda x: abs(x[1]))
        direction_type, value = max_dir
        
        # Threshold based on direction type
        if direction_type == 'intensity':
            return 'intensive-threshold'
        elif direction_type == 'visibility':
            return 'perceptual-threshold'
        elif direction_type == 'molecularity':
            return 'composition-threshold'
        elif direction_type in ['deterritorialization', 'reterritorialization']:
            return 'territorial-threshold'
        elif direction_type in ['minority', 'majority']:
            return 'political-threshold'
        elif direction_type in ['instinct', 'territory']:
            return 'behavioral-threshold'
        else:
            # Based on progress
            if progress < 0.33:
                return 'initial-threshold'
            elif progress < 0.67:
                return 'median-threshold'
            else:
                return 'terminal-threshold'
    
    def _map_phase_transitions(self, crossings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map phase transitions from threshold crossings."""
        transitions = []
        
        # Group crossings by vector
        vector_groups = defaultdict(list)
        for crossing in crossings:
            vector_groups[crossing['vector']].append(crossing)
        
        # Group crossings by vector
        vector_groups = defaultdict(list)
        for crossing in crossings:
            vector_groups[crossing['vector']].append(crossing)
        
        # Create transitions for each vector
        for vector, vector_crossings in vector_groups.items():
            # Sort by progress
            sorted_crossings = sorted(vector_crossings, key=lambda x: x['progress'])
            
            # Create transitions between consecutive crossings
            for i in range(len(sorted_crossings) - 1):
                from_crossing = sorted_crossings[i]
                to_crossing = sorted_crossings[i + 1]
                
                # Create transition
                transition = {
                    'from': from_crossing['nature'],
                    'to': to_crossing['nature'],
                    'vector': vector,
                    'start_progress': from_crossing['progress'],
                    'end_progress': to_crossing['progress'],
                    'intensity': (from_crossing['intensity'] + to_crossing['intensity']) / 2,
                    'transition_type': self._determine_transition_type(from_crossing, to_crossing)
                }
                transitions.append(transition)
        
        return transitions
    
    def _determine_transition_type(self, from_crossing: Dict[str, Any], 
                                 to_crossing: Dict[str, Any]) -> str:
        """Determine type of phase transition."""
        # Based on crossing natures
        from_nature = from_crossing['nature']
        to_nature = to_crossing['nature']
        
        # Map specific transitions
        if from_nature == 'intensification' and to_nature == 'phase-shift':
            return 'intensive-shift'
        elif from_nature == 'phase-shift' and to_nature == 'singularity':
            return 'singularization'
        elif from_nature == 'deterritorialization' and to_nature == 'line-of-flight':
            return 'escape'
        elif from_nature == 'fragmentation' and to_nature == 'recombination':
            return 'molecular-shift'
        elif from_nature == 'recombination' and to_nature == 'crystallization':
            return 'crystallization'
        elif from_nature == 'dissipation' and to_nature == 'imperceptibility':
            return 'disappearance'
        elif from_nature == 'connection' and to_nature == 'assemblage':
            return 'assemblage-formation'
        else:
            # Generic transition
            return 'threshold-crossing'
    
    def _calculate_becoming_intensity(self, vectors: List[Dict[str, Any]]) -> float:
        """Calculate overall intensity of the becoming process."""
        if not vectors:
            return 0.0
            
        # Average vector intensities, weighted by transformation type
        total = 0.0
        weights = 0.0
        
        for vector in vectors:
            weight = 1.0
            if vector['transformation_type'] == 'intensive':
                weight = 1.3
            elif vector['transformation_type'] == 'molecular':
                weight = 1.2
            elif vector['transformation_type'] == 'dissolving':
                weight = 1.1
                
            total += vector['intensity'] * weight
            weights += weight
        
        return total / weights if weights > 0 else 0.0
    
    # Helper methods for empathic flow
    def _create_flow_channels(self, becoming: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create flow channels for empathic flow."""
        channels = []
        
        # Create channel for each transformation vector
        for vector in becoming['transformation_vectors']:
            # Create channel
            channel = {
                'source': vector['source'],
                'direction': self._primary_direction(vector['direction']),
                'intensity': vector['intensity'],
                'bandwidth': self._calculate_channel_bandwidth(vector),
                'modulation': self._determine_channel_modulation(vector)
            }
            channels.append(channel)
        
        # Create channels for phase transitions
        for transition in becoming['phase_transitions']:
            # Create channel
            channel = {
                'source': transition['transition_type'],
                'direction': f"{transition['from']} to {transition['to']}",
                'intensity': transition['intensity'],
                'bandwidth': 0.5,  # Default bandwidth
                'modulation': 'transitional'
            }
            channels.append(channel)
            
        return channels
    
    def _primary_direction(self, direction: Dict[str, float]) -> str:
        """Determine primary direction from direction vector."""
        if not direction:
            return "neutral"
            
        # Find direction with maximum absolute value
        max_dir = max(direction.items(), key=lambda x: abs(x[1]))
        dir_name, value = max_dir
        
        # Include sign
        if value > 0:
            return f"increasing-{dir_name}"
        else:
            return f"decreasing-{dir_name}"
    
    def _calculate_channel_bandwidth(self, vector: Dict[str, Any]) -> float:
        """Calculate bandwidth of flow channel."""
        # Base on vector intensity and transformation type
        base_bandwidth = vector['intensity']
        
        # Adjust based on transformation type
        if vector['transformation_type'] == 'intensive':
            base_bandwidth *= 1.2
        elif vector['transformation_type'] == 'dissolving':
            base_bandwidth *= 0.8
        
        return min(1.0, base_bandwidth)
    
    def _determine_channel_modulation(self, vector: Dict[str, Any]) -> str:
        """Determine modulation type for flow channel."""
        # Based on transformation type and intensity
        trans_type = vector['transformation_type']
        intensity = vector['intensity']
        
        if trans_type == 'intensive':
            if intensity > 0.7:
                return 'intensive-amplification'
            else:
                return 'intensive-modulation'
        elif trans_type == 'deterritorializing':
            return 'deterritorializing-modulation'
        elif trans_type == 'dissolving':
            return 'dissolving-modulation'
        elif trans_type == 'molecular':
            return 'molecular-modulation'
        else:
            return 'rhizomatic-modulation'
    
    def _generate_intensive_currents(self, channels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate intensive currents through flow channels."""
        currents = []
        
        # Generate currents for each channel
        for channel in channels:
            # Determine number of currents based on bandwidth
            num_currents = int(1 + channel['bandwidth'] * 3)
            
            for i in range(num_currents):
                # Create current
                current = {
                    'channel': channel['source'],
                    'direction': channel['direction'],
                    'intensity': channel['intensity'] * random.uniform(0.8, 1.2),
                    'frequency': random.uniform(0.1, 1.0),
                    'modulation': channel['modulation'],
                    'current_type': self._determine_current_type(channel, i)
                }
                currents.append(current)
        
        return currents
    
    def _determine_current_type(self, channel: Dict[str, Any], index: int) -> str:
        """Determine type of intensive current."""
        # Based on channel modulation and index
        modulation = channel['modulation']
        
        if 'intensive' in modulation:
            if index == 0:
                return 'primary-intensive'
            else:
                return 'secondary-intensive'
        elif 'deterritorializing' in modulation:
            if index == 0:
                return 'deterritorializing'
            else:
                return 'nomadic'
        elif 'molecular' in modulation:
            if index == 0:
                return 'molecular'
            else:
                return 'particulate'
        elif 'dissolving' in modulation:
            return 'dissolving'
        elif 'transitional' in modulation:
            return 'transitional'
        else:
            return 'rhizomatic'
    
    def _map_affective_resonances(self, currents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map affective resonances from intensive currents."""
        resonances = []
        
        # Group currents by type
        type_groups = defaultdict(list)
        for current in currents:
            type_groups[current['current_type']].append(current)
        
        # Create resonances for each current type
        for current_type, type_currents in type_groups.items():
            # Create resonance
            resonance = {
                'type': f"{current_type}_resonance",
                'currents': [c['channel'] for c in type_currents],
                'intensity': sum(c['intensity'] for c in type_currents) / len(type_currents),
                'frequency': sum(c['frequency'] for c in type_currents) / len(type_currents),
                'affects': self._determine_resonance_affects(current_type)
            }
            resonances.append(resonance)
        
        # Create cross-type resonances
        if len(type_groups) >= 2:
            # Get two current types with highest average intensity
            type_avg_intensities = {}
            for current_type, type_currents in type_groups.items():
                avg_intensity = sum(c['intensity'] for c in type_currents) / len(type_currents)
                type_avg_intensities[current_type] = avg_intensity
                
            top_types = sorted(type_avg_intensities.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Create cross-type resonance
            resonance = {
                'type': 'cross_type_resonance',
                'current_types': [t[0] for t in top_types],
                'intensity': sum(t[1] for t in top_types) / 2,
                'frequency': random.uniform(0.3, 0.8),
                'affects': self._determine_resonance_affects('cross')
            }
            resonances.append(resonance)
            
        return resonances
    
    def _determine_resonance_affects(self, current_type: str) -> List[str]:
        """Determine affects associated with resonance."""
        # Map current types to affects
        type_affects = {
            'primary-intensive': ['joy', 'excitement', 'intensity'],
            'secondary-intensive': ['anticipation', 'surprise'],
            'deterritorializing': ['freedom', 'liberation', 'openness'],
            'nomadic': ['wandering', 'exploration', 'adventure'],
            'molecular': ['connection', 'intimacy', 'closeness'],
            'particulate': ['detail', 'precision', 'focus'],
            'dissolving': ['peace', 'calm', 'release'],
            'transitional': ['change', 'transformation', 'shift'],
            'rhizomatic': ['connectivity', 'network', 'relation'],
            'cross': ['complexity', 'depth', 'richness']
        }
        
        # Get affects for current type
        affects = type_affects.get(current_type, ['neutral'])
        
        # Select random subset
        return random.sample(affects, min(2, len(affects)))
    
    def _create_feedback_loops(self, resonances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create feedback loops from affective resonances."""
        loops = []
        
        # Create loops between resonances
        for i, res_a in enumerate(resonances):
            for res_b in resonances[i+1:]:
                # Only create loop with certain probability
                if random.random() < 0.7:
                    # Create loop
                    loop = {
                        'resonance_a': res_a['type'],
                        'resonance_b': res_b['type'],
                        'intensity': (res_a['intensity'] + res_b['intensity']) / 2,
                        'frequency': (res_a['frequency'] + res_b['frequency']) / 2,
                        'loop_type': self._determine_loop_type(res_a, res_b)
                    }
                    loops.append(loop)
        
        # Create self-referential loops for high-intensity resonances
        for resonance in resonances:
            if resonance['intensity'] > 0.7:
                # Create self-loop
                loop = {
                    'resonance_a': resonance['type'],
                    'resonance_b': resonance['type'],
                    'intensity': resonance['intensity'],
                    'frequency': resonance['frequency'],
                    'loop_type': 'self-referential'
                }
                loops.append(loop)
                
        return loops
    
    def _determine_loop_type(self, res_a: Dict[str, Any], res_b: Dict[str, Any]) -> str:
        """Determine type of feedback loop."""
        # Based on resonance types and intensities
        type_a = res_a['type']
        type_b = res_b['type']
        
        # Check for specific combinations
        if 'intensive' in type_a and 'intensive' in type_b:
            return 'intensive-amplification'
        elif ('deterritorializing' in type_a or 'nomadic' in type_a) and ('dissolving' in type_b):
            return 'escape-dissolution'
        elif ('molecular' in type_a or 'particulate' in type_a) and ('rhizomatic' in type_b):
            return 'molecular-connection'
        elif 'transitional' in type_a or 'transitional' in type_b:
            return 'transitional-resonance'
        elif 'cross_type' in type_a or 'cross_type' in type_b:
            return 'cross-resonance'
        else:
            return 'general-resonance'
    
    def _calculate_flow_sustainability(self, currents: List[Dict[str, Any]], 
                                     loops: List[Dict[str, Any]]) -> float:
        """Calculate sustainability of empathic flow."""
        # Based on current intensities and feedback loops
        if not currents:
            return 0.0
            
        # Calculate average current intensity
        avg_current_intensity = sum(c['intensity'] for c in currents) / len(currents)
        
        # Calculate loop reinforcement
        if loops:
            loop_reinforcement = sum(l['intensity'] for l in loops) / len(loops)
        else:
            loop_reinforcement = 0.0
        
        # Combine measures
        return 0.6 * avg_current_intensity + 0.4 * loop_reinforcement
    
    # Helper methods for deterritorialization
    def _create_deterritorialization_vector(self, flow: Dict[str, Any], 
                                          intensity: float) -> Dict[str, float]:
        """Create vector for deterritorialization."""
        vector = {}
        
        # Base on flow channels
        for channel in flow['flow_channels']:
            direction = channel['direction']
            if isinstance(direction, str) and '-' in direction:
                parts = direction.split('-')
                if len(parts) == 2 and parts[0] in ['increasing', 'decreasing']:
                    sign = 1.0 if parts[0] == 'increasing' else -1.0
                    dim = parts[1]
                    vector[dim] = sign * intensity * channel['intensity']
        
        # Ensure at least one dimension
        if not vector:
            vector['deterritorialization'] = intensity
            
        return vector
    
    def _find_lines_of_flight(self, flow: Dict[str, Any], 
                            vector: Dict[str, float]) -> List[Dict[str, Any]]:
        """Find lines of flight from deterritorialization vector."""
        lines = []
        
        # Create line for each significant dimension in vector
        for dim, value in vector.items():
            if abs(value) > 0.5:  # Only significant dimensions
                # Create line
                line = {
                    'dimension': dim,
                    'intensity': abs(value),
                    'direction': 'positive' if value > 0 else 'negative',
                    'source_resonances': self._find_source_resonances(flow, dim)
                }
                lines.append(line)
        
        # Create line from intensive currents
        intensive_currents = [c for c in flow.get('intensive_currents', []) 
                            if 'intensive' in c.get('current_type', '')]
        if intensive_currents:
            # Create intensity line
            line = {
                'dimension': 'intensity',
                'intensity': sum(c['intensity'] for c in intensive_currents) / len(intensive_currents),
                'direction': 'positive',
                'source_currents': [c['channel'] for c in intensive_currents]
            }
            lines.append(line)
            
        return lines
    
    def _find_source_resonances(self, flow: Dict[str, Any], dimension: str) -> List[str]:
        """Find source resonances for dimension."""
        sources = []
        
        # Check resonances for related affects
        dim_affects = {
            'intensity': ['joy', 'excitement', 'intensity'],
            'deterritorialization': ['freedom', 'liberation', 'openness'],
            'molecularity': ['connection', 'intimacy', 'closeness'],
            'visibility': ['detail', 'precision', 'focus'],
            'differentiation': ['change', 'transformation', 'shift'],
            'connectivity': ['connectivity', 'network', 'relation']
        }
        
        # Get affects for dimension
        related_affects = dim_affects.get(dimension, [])
        
        # Check resonances
        for resonance in flow.get('affective_resonances', []):
            resonance_affects = resonance.get('affects', [])
            if any(a in resonance_affects for a in related_affects):
                sources.append(resonance['type'])
        
        return sources
    
    def _create_nomadic_distribution(self, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create nomadic distribution from lines of flight."""
        if not lines:
            return {
                'type': 'empty',
                'dimensions': [],
                'intensity': 0.0
            }
            
        # Collect dimensions and intensities
        dimensions = [l['dimension'] for l in lines]
        intensities = [l['intensity'] for l in lines]
        
        # Calculate overall intensity
        overall_intensity = sum(intensities) / len(intensities)
        
        # Determine distribution type
        if len(lines) == 1:
            dist_type = 'singular'
        elif len(lines) == 2:
            dist_type = 'bifurcating'
        else:
            dist_type = 'rhizomatic'
            
        # Create distribution
        distribution = {
            'type': dist_type,
            'dimensions': dimensions,
            'intensity': overall_intensity,
            'lines': len(lines)
        }
        
        return distribution
    
    # Helper methods for reterritorialization
    def _create_reterritorialization_points(self, deterritorialized: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create reterritorialization points from deterritorialized empathy."""
        points = []
        
        # Create point for each line of flight
        for line in deterritorialized.get('lines_of_flight', []):
            # Create point
            point = {
                'dimension': line['dimension'],
                'intensity': line['intensity'] * random.uniform(0.7, 0.9),  # Slightly reduced
                'direction': 'reterritorializing',
                'source_line': line.get('dimension', 'unknown')
            }
            points.append(point)
            
        return points
    
    def _generate_new_territories(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate new territories from reterritorialization points."""
        territories = []
        
        # Create territory for each point
        for point in points:
            # Create territory
            territory = {
                'dimension': point['dimension'],
                'intensity': point['intensity'],
                'stability': random.uniform(0.4, 0.8),
                'type': self._determine_territory_type(point)
            }
            territories.append(territory)
            
        # Create composite territory if multiple points
        if len(points) >= 2:
            # Create composite
            composite = {
                'dimensions': [p['dimension'] for p in points],
                'intensity': sum(p['intensity'] for p in points) / len(points),
                'stability': random.uniform(0.5, 0.9),  # Higher stability
                'type': 'composite'
            }
            territories.append(composite)
            
        return territories
    
    def _determine_territory_type(self, point: Dict[str, Any]) -> str:
        """Determine type of territory from point."""
        # Based on dimension
        dimension = point['dimension']
        
        if dimension == 'intensity':
            return 'intensive-territory'
        elif dimension == 'deterritorialization':
            return 'deterritorialized-territory'
        elif dimension == 'molecularity':
            return 'molecular-territory'
        elif dimension == 'visibility':
            return 'perceptual-territory'
        elif dimension == 'differentiation':
            return 'differential-territory'
        elif dimension == 'connectivity':
            return 'connective-territory'
        else:
            return 'generic-territory'
    
    def _map_intensive_distributions(self, territories: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Map intensive distributions across territories."""
        distributions = {}
        
        # Create distribution for each territory
        for territory in territories:
            dimension = territory.get('dimension', 'generic')
            if 'dimensions' in territory:
                # Composite territory
                for dim in territory['dimensions']:
                    distributions[dim] = self._generate_intensity_distribution(
                        territory['intensity'], territory['stability'])
            else:
                # Single dimension territory
                distributions[dimension] = self._generate_intensity_distribution(
                    territory['intensity'], territory['stability'])
                
        return distributions
    
    def _generate_intensity_distribution(self, intensity: float, stability: float) -> List[float]:
        """Generate intensity distribution for territory."""
        # Create distribution points
        points = 10
        distribution = []
        
        for i in range(points):
            pos = i / points
            # Base intensity
            base = intensity
            
            # Adjust based on stability
            if stability > 0.7:
                # High stability: smooth curve
                val = base * (1 - 0.2 * abs(pos - 0.5) / 0.5)
            elif stability > 0.4:
                # Medium stability: slight variation
                val = base * (1 - 0.3 * abs(pos - 0.5) / 0.5 + random.uniform(-0.1, 0.1))
            else:
                # Low stability: more variation
                val = base * (1 - 0.4 * abs(pos - 0.5) / 0.5 + random.uniform(-0.2, 0.2))
                
            distribution.append(max(0, min(1, val)))
            
        return distribution
    
    def _calculate_territory_stability(self, territories: List[Dict[str, Any]]) -> float:
        """Calculate overall stability of territories."""
        if not territories:
            return 0.0
            
        # Average territory stability
        return sum(t.get('stability', 0) for t in territories) / len(territories)
    
    # General helper methods
    def _record_becoming(self, entity: Dict[str, Any], 
                       assemblage: Dict[str, Any], 
                       becoming: Dict[str, Any], 
                       flow: Dict[str, Any]) -> None:
        """Record becoming for future reference."""
        entity_id = entity.get('id', str(int(time.time())))
        
        record = {
            'entity_id': entity_id,
            'assemblage_id': assemblage['id'],
            'becoming_id': becoming['id'],
            'flow_id': flow['id'],
            'becoming_type': becoming['becoming_type'],
            'flow_sustainability': flow['sustainability'],
            'timestamp': self._get_timestamp()
        }
        
        # Store in becoming other
        self.becoming_other[entity_id] = flow
        
        # Create empathic territory
        territory = {
            'entity_id': entity_id,
            'resonance_zones': assemblage['resonance_zones'],
            'becoming_type': becoming['becoming_type'],
            'flow_channels': flow['flow_channels'],
            'sustainability': flow['sustainability'],
            'timestamp': self._get_timestamp()
        }
        
        self.empathic_territories[entity_id] = territory
    
    def _get_entity_id_from_becoming(self, becoming: Dict[str, Any]) -> str:
        """Get entity ID from becoming process."""
        assemblage_id = becoming.get('assemblage_id', '')
        
        # Find entity with this assemblage
        for entity_id, flow in self.becoming_other.items():
            flow_becoming = self.get_becoming_process(flow.get('becoming_id', ''))
            if flow_becoming and flow_becoming.get('assemblage_id') == assemblage_id:
                return entity_id
                
        # Fallback
        return str(int(time.time()))
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return str(int(time.time() * 1000))
```
