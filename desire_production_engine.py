# desire_production_engine.py

import random
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from collections import defaultdict
import math
import time

class DesireProductionEngine:
    """
    An implementation of Deleuze and Guattari's concept of desire as productive force.
    Generates desires as positive productions rather than representations of lack.
    Creates desiring-machines that connect and produce flows rather than fixed objects.
    """
    
    def __init__(self):
        self.desiring_machines = []
        self.flows = {}
        self.connections = defaultdict(list)
        self.body_without_organs = {
            'surface': {},
            'intensities': {},
            'inscriptions': []
        }
        self.production_registers = {
            'connective': [],  # recording-of-production
            'disjunctive': [],  # recording-of-distribution
            'conjunctive': []   # recording-of-consumption
        }
        self.schizo_process_states = {}
        self.deterritorialization_vectors = []
        self.reterritorialization_points = {}
        
    def produce_creative_desire(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate productive desires, not lacks.
        
        Args:
            context: The current context for desire production
            
        Returns:
            Dictionary containing the productive desire and its characteristics
        """
        # Assemble a desiring-machine appropriate to the context
        machine = self.assemble_desiring_machine(context)
        
        # Create desire flow from the machine
        flow = self.create_desire_flow(machine)
        
        # Channel productive force
        production = self.channel_productive_force(flow)
        
        # Record the production
        self._record_production(machine, flow, production)
        
        return production
    
    def assemble_desiring_machine(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assemble a desiring-machine from context elements.
        Desiring-machines are assemblages that produce and channel flows.
        
        Args:
            context: Current context for machine assembly
            
        Returns:
            A desiring-machine as a dictionary of components and connections
        """
        # Extract context elements
        elements = self._extract_context_elements(context)
        
        # Create machine components
        components = self._create_machine_components(elements)
        
        # Establish connections between components
        connections = self._create_machine_connections(components)
        
        # Determine machine type
        machine_type = self._determine_machine_type(components, connections)
        
        # Create the machine
        machine_id = f"machine_{len(self.desiring_machines) + 1}_{int(time.time())}"
        machine = {
            'id': machine_id,
            'type': machine_type,
            'components': components,
            'connections': connections,
            'productive_capacity': self._calculate_productive_capacity(components, connections),
            'deterritorialization_potential': self._calculate_deterritorialization_potential(components),
            'creation_timestamp': self._get_timestamp()
        }
        
        # Register the machine
        self.desiring_machines.append(machine)
        
        return machine
    
    def create_desire_flow(self, machine: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create flow of desire from a desiring-machine.
        Flows are intensities that pass between machines.
        
        Args:
            machine: The desiring-machine that generates the flow
            
        Returns:
            A desire flow as a dictionary of characteristics
        """
        # Calculate flow intensity
        intensity = self._calculate_flow_intensity(machine)
        
        # Determine flow direction
        direction = self._determine_flow_direction(machine)
        
        # Create flow vector
        vector = self._create_flow_vector(direction, intensity)
        
        # Determine flow characteristics
        characteristics = self._determine_flow_characteristics(machine, intensity)
        
        # Create the flow
        flow_id = f"flow_{machine['id']}_{int(time.time())}"
        flow = {
            'id': flow_id,
            'source_machine': machine['id'],
            'intensity': intensity,
            'vector': vector,
            'characteristics': characteristics,
            'creation_timestamp': self._get_timestamp()
        }
        
        # Register the flow
        self.flows[flow_id] = flow
        
        # Connect the flow to the body-without-organs
        self._connect_flow_to_bwo(flow)
        
        return flow
    
    def channel_productive_force(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Channel the productive force of desire.
        This is where desire becomes productive rather than representative.
        
        Args:
            flow: The desire flow to channel
            
        Returns:
            The channeled productive desire
        """
        # Record the flow in the connective synthesis
        connective_synthesis = self._perform_connective_synthesis(flow)
        
        # Distribute the flow in the disjunctive synthesis
        disjunctive_synthesis = self._perform_disjunctive_synthesis(flow, connective_synthesis)
        
        # Consume the flow in the conjunctive synthesis
        conjunctive_synthesis = self._perform_conjunctive_synthesis(flow, disjunctive_synthesis)
        
        # Create desire production
        production = {
            'source_flow': flow['id'],
            'type': self._determine_production_type(flow, conjunctive_synthesis),
            'intensity': flow['intensity'],
            'creative_vector': self._create_creative_vector(flow, conjunctive_synthesis),
            'actualizations': self._generate_actualizations(flow, conjunctive_synthesis),
            'deterritorialization': self._calculate_production_deterritorialization(flow),
            'reterritorialization': self._identify_reterritorialization_points(flow),
            'creation_timestamp': self._get_timestamp()
        }
        
        # Add to appropriate register
        self._add_to_production_registers(production)
        
        return production
    
    def deterritorialize_desire(self, production: Dict[str, Any], 
                               intensity: float = 0.7) -> Dict[str, Any]:
        """
        Deterritorialize a desire production to create line of flight.
        
        Args:
            production: The desire production to deterritorialize
            intensity: The intensity of deterritorialization
            
        Returns:
            The deterritorialized desire
        """
        # Create deterritorialization vector
        vector = self._create_deterritorialization_vector(production, intensity)
        
        # Find lines of flight
        lines_of_flight = self._find_lines_of_flight(production, vector)
        
        # Create nomadic distribution
        nomadic_distribution = self._create_nomadic_distribution(lines_of_flight)
        
        # Create deterritorialized desire
        deterritorialized = {
            'source_production': production,
            'vector': vector,
            'lines_of_flight': lines_of_flight,
            'nomadic_distribution': nomadic_distribution,
            'intensity': intensity,
            'creation_timestamp': self._get_timestamp()
        }
        
        # Add to deterritorialization vectors
        self.deterritorialization_vectors.append(deterritorialized)
        
        return deterritorialized
    
    def create_schizo_process(self, productions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a schizo-process from multiple desire productions.
        The schizo-process is the revolutionary process of desire.
        
        Args:
            productions: List of desire productions to combine
            
        Returns:
            The schizo-process
        """
        # Extract production elements
        elements = [self._extract_production_elements(prod) for prod in productions]
        
        # Create decoded flows
        decoded_flows = self._decode_flows(productions)
        
        # Generate breakthrough points
        breakthrough_points = self._generate_breakthrough_points(decoded_flows)
        
        # Create schizo mapping
        mapping = self._create_schizo_mapping(breakthrough_points)
        
        # Create the schizo-process
        process_id = f"schizo_{int(time.time())}"
        process = {
            'id': process_id,
            'productions': [p['source_flow'] for p in productions],
            'decoded_flows': decoded_flows,
            'breakthrough_points': breakthrough_points,
            'mapping': mapping,
            'intensity': sum(p['intensity'] for p in productions) / len(productions),
            'creation_timestamp': self._get_timestamp()
        }
        
        # Register the process
        self.schizo_process_states[process_id] = process
        
        return process
    
    def connect_machines(self, machine_a: Dict[str, Any], 
                        machine_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connect two desiring-machines to form a new assemblage.
        
        Args:
            machine_a: First desiring-machine
            machine_b: Second desiring-machine
            
        Returns:
            The connection between machines
        """
        # Determine connection type
        connection_type = self._determine_connection_type(machine_a, machine_b)
        
        # Calculate connection intensity
        intensity = self._calculate_connection_intensity(machine_a, machine_b)
        
        # Create flow between machines
        flow = self._create_inter_machine_flow(machine_a, machine_b, intensity)
        
        # Create the connection
        connection = {
            'machine_a': machine_a['id'],
            'machine_b': machine_b['id'],
            'type': connection_type,
            'intensity': intensity,
            'flow': flow,
            'creation_timestamp': self._get_timestamp()
        }
        
        # Register the connection
        self.connections[machine_a['id']].append(connection)
        self.connections[machine_b['id']].append(connection)
        
        return connection
    
    def get_active_desiring_machines(self) -> List[Dict[str, Any]]:
        """
        Get all active desiring-machines.
        
        Returns:
            List of active desiring-machines
        """
        # Return most recent machines (last 10)
        recent = sorted(self.desiring_machines, 
                     key=lambda m: m['creation_timestamp'], 
                     reverse=True)[:10]
        
        return recent
    
    def get_desire_flows(self) -> List[Dict[str, Any]]:
        """
        Get all active desire flows.
        
        Returns:
            List of active desire flows
        """
        # Return all flows, sorted by recency
        return sorted(self.flows.values(), 
                    key=lambda f: f['creation_timestamp'],
                    reverse=True)
    
    def get_body_without_organs(self) -> Dict[str, Any]:
        """
        Get current state of the body-without-organs.
        
        Returns:
            The body-without-organs
        """
        return self.body_without_organs
    
    # Helper methods for desiring-machine assembly
    def _extract_context_elements(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract elements from context for machine assembly."""
        elements = []
        
        # Process different context types
        if 'text' in context:
            # Extract from text
            text = context['text']
            words = [w for w in text.split() if w.strip()]
            
            # Create elements from words
            for i, word in enumerate(words):
                if len(word) > 3:  # Ignore very short words
                    element = {
                        'type': 'linguistic',
                        'content': word,
                        'position': i / len(words),
                        'intensity': min(1.0, len(word) / 10)
                    }
                    elements.append(element)
        
        if 'concepts' in context:
            # Extract from concepts
            for concept in context['concepts']:
                element = {
                    'type': 'conceptual',
                    'content': concept,
                    'intensity': random.uniform(0.5, 1.0)
                }
                elements.append(element)
        
        if 'affects' in context:
            # Extract from affects
            for affect, intensity in context['affects'].items():
                element = {
                    'type': 'affective',
                    'content': affect,
                    'intensity': intensity
                }
                elements.append(element)
        
        # If no elements extracted, create default elements
        if not elements:
            # Create default elements based on context keys
            for key in context.keys():
                element = {
                    'type': 'contextual',
                    'content': key,
                    'intensity': random.uniform(0.3, 0.7)
                }
                elements.append(element)
        
        return elements
    
    def _create_machine_components(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create machine components from elements."""
        components = []
        
        # Transform elements into components
        for i, element in enumerate(elements):
            # Component types depend on element types
            if element['type'] == 'linguistic':
                component_type = 'expression-machine'
            elif element['type'] == 'conceptual':
                component_type = 'concept-machine'
            elif element['type'] == 'affective':
                component_type = 'affect-machine'
            else:
                component_type = 'generic-machine'
                
            # Create the component
            component = {
                'id': f"comp_{i}_{int(time.time())}",
                'type': component_type,
                'source_element': element,
                'productive_capacity': random.uniform(0.4, 1.0) * element['intensity'],
                'connectivity': random.uniform(0.3, 0.9)
            }
            
            components.append(component)
        
        return components
    
    def _create_machine_connections(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create connections between machine components."""
        connections = []
        
        # Connect components based on connectivity
        for i, comp_a in enumerate(components):
            for comp_b in components[i+1:]:
                # Calculate connection probability
                prob = comp_a['connectivity'] * comp_b['connectivity']
                
                # Only create connection if probability threshold met
                if random.random() < prob:
                    connection = {
                        'from': comp_a['id'],
                        'to': comp_b['id'],
                        'type': self._determine_component_connection_type(comp_a, comp_b),
                        'intensity': (comp_a['productive_capacity'] + comp_b['productive_capacity']) / 2
                    }
                    connections.append(connection)
        
        return connections
    
    def _determine_component_connection_type(self, comp_a: Dict[str, Any], 
                                           comp_b: Dict[str, Any]) -> str:
        """Determine the type of connection between components."""
        # Connection types based on component types
        if comp_a['type'] == comp_b['type']:
            return 'resonance'
        elif 'affect' in comp_a['type'] and 'concept' in comp_b['type']:
            return 'affective-conceptual'
        elif 'concept' in comp_a['type'] and 'affect' in comp_b['type']:
            return 'conceptual-affective'
        elif 'expression' in comp_a['type'] or 'expression' in comp_b['type']:
            return 'expressive'
        else:
            return 'connective'
    
    def _determine_machine_type(self, components: List[Dict[str, Any]], 
                              connections: List[Dict[str, Any]]) -> str:
        """Determine the type of desiring-machine based on components and connections."""
        # Count component types
        type_counts = defaultdict(int)
        for comp in components:
            type_counts[comp['type']] += 1
        
        # Determine dominant type
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # Map to machine types
        if dominant_type == 'expression-machine':
            return 'expressive-machine'
        elif dominant_type == 'concept-machine':
            return 'conceptual-machine'
        elif dominant_type == 'affect-machine':
            return 'affective-machine'
        else:
            # Consider connection density
            connection_density = len(connections) / (len(components) * (len(components) - 1) / 2)
            
            if connection_density > 0.7:
                return 'rhizomatic-machine'
            elif connection_density < 0.3:
                return 'molecular-machine'
            else:
                return 'assemblage-machine'
    
    def _calculate_productive_capacity(self, components: List[Dict[str, Any]], 
                                     connections: List[Dict[str, Any]]) -> float:
        """Calculate the productive capacity of a desiring-machine."""
        # Base capacity on component capacities
        component_capacity = sum(c['productive_capacity'] for c in components) / len(components)
        
        # Adjust based on connection density
        connection_density = len(connections) / max(1, (len(components) * (len(components) - 1) / 2))
        
        # Final capacity
        return component_capacity * (1 + connection_density)
    
    def _calculate_deterritorialization_potential(self, components: List[Dict[str, Any]]) -> float:
        """Calculate deterritorialization potential of a machine."""
        # Base on component diversity
        types = set(c['type'] for c in components)
        type_diversity = len(types) / len(components) if components else 0
        
        # Calculate intensity variance
        intensities = [c['productive_capacity'] for c in components]
        intensity_variance = np.var(intensities) if intensities else 0
        
        # Final potential
        return 0.4 * type_diversity + 0.6 * intensity_variance
    
    # Helper methods for desire flow creation
    def _calculate_flow_intensity(self, machine: Dict[str, Any]) -> float:
        """Calculate the intensity of a desire flow."""
        # Base on machine productive capacity
        base_intensity = machine['productive_capacity']
        
        # Adjust based on machine type
        if machine['type'] == 'rhizomatic-machine':
            base_intensity *= 1.2
        elif machine['type'] == 'molecular-machine':
            base_intensity *= 0.8
            
        # Ensure within range
        return min(1.0, max(0.1, base_intensity))
    
    def _determine_flow_direction(self, machine: Dict[str, Any]) -> Dict[str, float]:
        """Determine the direction of a desire flow."""
        # Create direction vector in conceptual space
        direction = {}
        
        # Base directions on machine components
        for comp in machine['components']:
            comp_type = comp['type'].split('-')[0]  # Get base type
            direction[comp_type] = direction.get(comp_type, 0) + comp['productive_capacity']
        
        # Normalize
        total = sum(direction.values())
        if total > 0:
            for key in direction:
                direction[key] /= total
        
        return direction
    
    def _create_flow_vector(self, direction: Dict[str, float], intensity: float) -> Dict[str, float]:
        """Create a flow vector from direction and intensity."""
        # Scale direction by intensity
        vector = {dim: val * intensity for dim, val in direction.items()}
        
        # Add magnitude
        vector['magnitude'] = intensity
        
        return vector
    
    def _determine_flow_characteristics(self, machine: Dict[str, Any], 
                                      intensity: float) -> Dict[str, Any]:
        """Determine the characteristics of a desire flow."""
        # Base characteristics on machine type
        machine_type = machine['type']
        
        # Extract component content
        contents = [comp['source_element']['content'] 
                  for comp in machine['components'] 
                  if 'source_element' in comp and 'content' in comp['source_element']]
        
        # Determine primary characteristic based on machine type
        if 'expressive' in machine_type:
            primary = 'expression'
        elif 'conceptual' in machine_type:
            primary = 'concept'
        elif 'affective' in machine_type:
            primary = 'affect'
        elif 'rhizomatic' in machine_type:
            primary = 'connection'
        else:
            primary = 'production'
            
        # Create characteristics
        characteristics = {
            'primary': primary,
            'contents': contents[:5],  # Limit to 5 contents
            'fluidity': random.uniform(0.3, 0.9),
            'consistency': intensity * random.uniform(0.8, 1.2)
        }
        
        return characteristics
    
    def _connect_flow_to_bwo(self, flow: Dict[str, Any]) -> None:
        """Connect a flow to the body-without-organs."""
        # Record flow on BwO surface
        self.body_without_organs['surface'][flow['id']] = {
            'intensity': flow['intensity'],
            'vector': flow['vector'],
            'timestamp': self._get_timestamp()
        }
        
        # Create intensity zone on BwO
        primary = flow['characteristics']['primary']
        self.body_without_organs['intensities'][primary] = (
            self.body_without_organs['intensities'].get(primary, 0) + 
            flow['intensity']
        )
        
        # Create inscription
        inscription = {
            'flow': flow['id'],
            'machine': flow['source_machine'],
            'intensity': flow['intensity'],
            'timestamp': self._get_timestamp()
        }
        self.body_without_organs['inscriptions'].append(inscription)
    
    # Helper methods for productive force channeling
    def _perform_connective_synthesis(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the connective synthesis of production."""
        # Recording of production
        synthesis = {
            'flow': flow['id'],
            'type': 'connective',
            'connections': [],
            'productive_elements': []
        }
        
        # Extract flow vector dimensions
        vector = flow['vector']
        dimensions = [d for d in vector.keys() if d != 'magnitude']
        
        # Create connective elements
        for dimension in dimensions:
            if vector[dimension] > 0.2:  # Only significant dimensions
                element = {
                    'dimension': dimension,
                    'intensity': vector[dimension],
                    'connective_type': self._determine_connective_type(dimension)
                }
                synthesis['productive_elements'].append(element)
        
        # Create connections to other flows
        other_flows = [f for f_id, f in self.flows.items() if f_id != flow['id']]
        for other_flow in other_flows[:3]:  # Limit to 3 connections
            connection = self._create_flow_connection(flow, other_flow)
            if connection:
                synthesis['connections'].append(connection)
        
        # Record synthesis
        self.production_registers['connective'].append(synthesis)
        
        return synthesis
    
    def _determine_connective_type(self, dimension: str) -> str:
        """Determine the connective type for a dimension."""
        # Map dimensions to connective types
        dimension_map = {
            'expression': 'code-connection',
            'concept': 'code-decoding',
            'affect': 'intensity-connection',
            'connection': 'rhizomatic-connection',
            'production': 'production-connection'
        }
        
        return dimension_map.get(dimension, 'generic-connection')
    
    def _create_flow_connection(self, flow_a: Dict[str, Any], 
                              flow_b: Dict[str, Any]) -> Dict[str, Any]:
        """Create a connection between flows."""
        # Calculate connection intensity
        vector_a = flow_a['vector']
        vector_b = flow_b['vector']
        
        # Calculate dot product as similarity
        similarity = sum(vector_a.get(k, 0) * vector_b.get(k, 0) 
                       for k in set(vector_a.keys()) & set(vector_b.keys()) 
                       if k != 'magnitude')
        
        # Only create connection if similarity is significant
        if similarity > 0.1:
            return {
                'flow_a': flow_a['id'],
                'flow_b': flow_b['id'],
                'intensity': similarity,
                'type': 'productive-connection' if similarity > 0.5 else 'partial-connection'
            }
        
        return None
    
    def _perform_disjunctive_synthesis(self, flow: Dict[str, Any], 
                                     connective: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the disjunctive synthesis of recording."""
        # Recording of distribution
        synthesis = {
            'flow': flow['id'],
            'type': 'disjunctive',
            'source_synthesis': connective,
            'disjunctions': [],
            'chains': []
        }
        
        # Create disjunctions from connective elements
        elements = connective['productive_elements']
        for i, element_a in enumerate(elements):
            for element_b in elements[i+1:]:
                # Create disjunction between elements
                disjunction = {
                    'element_a': element_a['dimension'],
                    'element_b': element_b['dimension'],
                    'intensity': (element_a['intensity'] + element_b['intensity']) / 2,
                    'type': 'inclusive' if random.random() < 0.7 else 'exclusive'
                }
                synthesis['disjunctions'].append(disjunction)
        
        # Create signifying chains
        elements = sorted(elements, key=lambda e: e['intensity'], reverse=True)
        if elements:
            chain = {
                'elements': [e['dimension'] for e in elements],
                'intensity': sum(e['intensity'] for e in elements) / len(elements),
                'type': 'decoded' if flow['intensity'] > 0.7 else 'coded'
            }
            synthesis['chains'].append(chain)
        
        # Record synthesis
        self.production_registers['disjunctive'].append(synthesis)
        
        return synthesis
    
    def _perform_conjunctive_synthesis(self, flow: Dict[str, Any], 
                                     disjunctive: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the conjunctive synthesis of consumption."""
        # Recording of consumption
        synthesis = {
            'flow': flow['id'],
            'type': 'conjunctive',
            'source_synthesis': disjunctive,
            'subject_positions': [],
            'consumptions': []
        }
        
        # Create subject positions
        for chain in disjunctive['chains']:
            position = {
                'elements': chain['elements'],
                'intensity': chain['intensity'],
                'type': 'nomadic' if chain['type'] == 'decoded' else 'sedentary'
            }
            synthesis['subject_positions'].append(position)
        
        # Create consumptions
        for disjunction in disjunctive['disjunctions']:
            if disjunction['type'] == 'inclusive':
                consumption = {
                    'elements': [disjunction['element_a'], disjunction['element_b']],
                    'intensity': disjunction['intensity'],
                    'type': 'productive'
                }
            else:
                consumption = {
                    'elements': [disjunction['element_a'], disjunction['element_b']],
                    'intensity': disjunction['intensity'],
                    'type': 'anti-productive'
                }
            synthesis['consumptions'].append(consumption)
        
        # Record synthesis
        self.production_registers['conjunctive'].append(synthesis)
        
        return synthesis
    
    def _determine_production_type(self, flow: Dict[str, Any], 
                                 conjunctive: Dict[str, Any]) -> str:
        """Determine the type of desire production."""
        # Check subject positions
        positions = conjunctive['subject_positions']
        
        # Determine based on subject positions and flow characteristics
        if positions and positions[0]['type'] == 'nomadic':
            return 'schizophrenic-production'
        elif flow['characteristics']['primary'] == 'affect':
            return 'affective-production'
        elif flow['characteristics']['primary'] == 'concept':
            return 'conceptual-production'
        elif flow['characteristics']['primary'] == 'expression':
            return 'expressive-production'
        elif flow['intensity'] > 0.8:
            return 'intensive-production'
        else:
            return 'molecular-production'
    
    def _create_creative_vector(self, flow: Dict[str, Any], 
                              conjunctive: Dict[str, Any]) -> Dict[str, float]:
        """Create a creative vector from flow and conjunctive synthesis."""
        vector = {}
        
        # Base on flow vector
        flow_vector = flow['vector']
        for k, v in flow_vector.items():
            if k != 'magnitude':
                vector[k] = v
        
        # Adjust based on consumptions
        for consumption in conjunctive['consumptions']:
            if consumption['type'] == 'productive':
                for element in consumption['elements']:
                    vector[element] = vector.get(element, 0) + consumption['intensity'] * 0.2
            else:
                for element in consumption['elements']:
                    vector[element] = vector.get(element, 0) - consumption['intensity'] * 0.1
        
        # Ensure values in range
        for k in vector:
            vector[k] = min(1.0, max(0.0, vector[k]))
        
        # Add magnitude
        magnitude = math.sqrt(sum(v*v for v in vector.values()))
        vector['magnitude'] = magnitude
        
        return vector
    
    def _generate_actualizations(self, flow: Dict[str, Any], 
                               conjunctive: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate possible actualizations of the desire production."""
        actualizations = []
        
        # Generate from flow characteristics
        primary = flow['characteristics']['primary']
        contents = flow['characteristics']['contents']
        
        # Create actualizations based on primary characteristic
        if primary == 'expression':
            # Expressive actualizations
            for content in contents[:2]:  # Limit to 2
                actualization = {
                    'type': 'expressive',
                    'content': content,
                    'intensity': flow['intensity'] * random.uniform(0.8, 1.2),
                    'medium': self._select_expressive_medium(flow)
                }
                actualizations.append(actualization)
                
        elif primary == 'concept':
            # Conceptual actualizations
            for position in conjunctive['subject_positions']:
                actualization = {
                    'type': 'conceptual',
                    'elements': position['elements'],
                    'intensity': position['intensity'],
                    'framework': self._select_conceptual_framework(position)
                }
                actualizations.append(actualization)
                
        elif primary == 'affect':
            # Affective actualizations
            actualization = {
                'type': 'affective',
                'content': contents[0] if contents else 'intensity',
                'intensity': flow['intensity'],
                'modulations': self._generate_affective_modulations(flow)
            }
            actualizations.append(actualization)
            
        else:
            # Generic actualizations
            actualization = {
                'type': 'generic',
                'elements': contents[:3],
                'intensity': flow['intensity'],
                'manifestation': self._select_manifestation_type(flow)
            }
            actualizations.append(actualization)
            
        return actualizations
    
    def _calculate_production_deterritorialization(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the deterritorialization aspects of the production."""
        # Calculate deterritorialization
        intensity = flow['intensity'] * random.uniform(0.8, 1.2)
        
        deterritorialization = {
            'intensity': intensity,
            'vectors': {
                'creative': random.uniform(0.5, 1.0) * intensity,
                'social': random.uniform(0.2, 0.8) * intensity,
                'psychic': random.uniform(0.4, 0.9) * intensity
            },
            'threshold': 0.7  # Threshold for line of flight
        }
        
        return deterritorialization
    
    def _identify_reterritorialization_points(self, flow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential reterritorialization points."""
        points = []
        
        # Generate potential reterritorialization points
        num_points = random.randint(1, 3)
        for _ in range(num_points):
            point = {
                'intensity': random.uniform(0.3, 0.7),
                'type': random.choice(['social', 'psychic', 'creative']),
                'stability': random.uniform(0.4, 0.8)
            }
            points.append(point)
        
        return points
    
    def _select_expressive_medium(self, flow: Dict[str, Any]) -> str:
        """Select an expressive medium based on flow characteristics."""
        media = ['text', 'image', 'sound', 'movement', 'code', 'speech']
        weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]  # Default weights
        
        # Adjust weights based on flow
        if flow['intensity'] > 0.8:
            # High intensity favors more dynamic media
            weights = [0.15, 0.25, 0.25, 0.2, 0.05, 0.1]
        
        # Select medium
        return random.choices(media, weights=weights)[0]
    
    def _select_conceptual_framework(self, position: Dict[str, Any]) -> str:
        """Select a conceptual framework based on subject position."""
        frameworks = ['rhizomatic', 'dialectical', 'systems', 'network', 'process', 'algorithmic']
        
        # Select based on position type
        if position['type'] == 'nomadic':
            # Favor rhizomatic and process for nomadic positions
            weights = [0.3, 0.1, 0.1, 0.2, 0.2, 0.1]
        else:
            # Favor more structured frameworks for sedentary
            weights = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]
        
        return random.choices(frameworks, weights=weights)[0]
    
    def _generate_affective_modulations(self, flow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate affective modulations for the actualization."""
        modulations = []
        
        # Generate modulations
        num_modulations = random.randint(1, 3)
        for _ in range(num_modulations):
            modulation = {
                'type': random.choice(['intensification', 'diminution', 'transformation']),
                'intensity': random.uniform(0.3, 0.9),
                'duration': random.uniform(0.2, 1.0)
            }
            modulations.append(modulation)
        
        return modulations
    
    def _select_manifestation_type(self, flow: Dict[str, Any]) -> str:
        """Select a manifestation type for generic actualization."""
        types = ['pattern', 'structure', 'process', 'event', 'entity']
        
        # Select based on flow characteristics
        if 'fluidity' in flow['characteristics'] and flow['characteristics']['fluidity'] > 0.7:
            # High fluidity favors process and event
            weights = [0.1, 0.1, 0.4, 0.3, 0.1]
        else:
            # Otherwise favor more stable manifestations
            weights = [0.2, 0.3, 0.2, 0.1, 0.2]
        
        return random.choices(types, weights=weights)[0]
    
    def _add_to_production_registers(self, production: Dict[str, Any]) -> None:
        """Add production to appropriate registers."""
        # Add to registers based on production type
        if 'schizophrenic' in production['type']:
            # Add all schizophrenic productions to all registers
            for register in self.production_registers.values():
                register.append(production)
        elif 'affective' in production['type']:
            self.production_registers['conjunctive'].append(production)
        elif 'conceptual' in production['type']:
            self.production_registers['disjunctive'].append(production)
        elif 'expressive' in production['type']:
            self.production_registers['connective'].append(production)
        else:
            # Generic and other types
            reg_type = random.choice(list(self.production_registers.keys()))
            self.production_registers[reg_type].append(production)
    
    # Helper methods for deterritorialization
    def _create_deterritorialization_vector(self, production: Dict[str, Any], 
                                          intensity: float) -> Dict[str, float]:
        """Create a deterritorialization vector."""
        # Base vector on production creative vector
        base_vector = production['creative_vector']
        
        # Create deterritorialization direction
        direction = {}
        for k, v in base_vector.items():
            if k != 'magnitude':
                # Randomize direction slightly
                direction[k] = v * random.uniform(0.8, 1.2)
        
        # Scale by intensity
        vector = {k: v * intensity for k, v in direction.items()}
        
        # Add magnitude
        magnitude = math.sqrt(sum(v*v for v in vector.values()))
        vector['magnitude'] = magnitude
        
        return vector
    
    def _find_lines_of_flight(self, production: Dict[str, Any], 
                            vector: Dict[str, float]) -> List[Dict[str, Any]]:
        """Find potential lines of flight for deterritorialization."""
        lines = []
        
        # Generate lines of flight
        num_lines = random.randint(1, 3)
        for i in range(num_lines):
            # Create a line of flight
            line = {
                'id': f"line_{i}_{int(time.time())}",
                'intensity': vector['magnitude'] * random.uniform(0.8, 1.2),
                'direction': {k: v * random.uniform(0.9, 1.1) for k, v in vector.items() if k != 'magnitude'},
                'breakthrough_potential': random.uniform(0.4, 0.9) * vector['magnitude']
            }
            
            lines.append(line)
        
        return lines
    
    def _create_nomadic_distribution(self, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a nomadic distribution for deterritorialized desire."""
        # Total intensity
        total_intensity = sum(line['intensity'] for line in lines)
        
        # Create distribution
        distribution = {
            'type': 'nomadic',
            'intensity': total_intensity,
            'dimensions': {},
            'consistency': random.uniform(0.3, 0.7) * total_intensity
        }
        
        # Generate dimensional distribution
        for line in lines:
            for dim, val in line['direction'].items():
                distribution['dimensions'][dim] = distribution['dimensions'].get(dim, 0) + val
        
        # Normalize dimensions
        total = sum(distribution['dimensions'].values())
        if total > 0:
            for dim in distribution['dimensions']:
                distribution['dimensions'][dim] /= total
        
        return distribution
    
    # Helper methods for schizo-process
    def _extract_production_elements(self, production: Dict[str, Any]) -> List[str]:
        """Extract elements from a production for schizo-process."""
        elements = []
        
        # Extract from creative vector
        for dim, val in production['creative_vector'].items():
            if dim != 'magnitude' and val > 0.3:
                elements.append(dim)
        
        # Extract from actualizations
        for actualization in production['actualizations']:
            if 'content' in actualization:
                elements.append(actualization['content'])
            elif 'elements' in actualization:
                elements.extend(actualization['elements'])
        
        return elements
    
    def _decode_flows(self, productions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Decode flows for schizo-process."""
        decoded_flows = []
        
        # Decode each production
        for production in productions:
            # Extract flow
            flow_id = production['source_flow']
            flow = self.flows.get(flow_id)
            
            if flow:
                # Create decoded flow
                decoded = {
                    'original_flow': flow_id,
                    'intensity': flow['intensity'],
                    'vector': {k: v * random.uniform(0.8, 1.2) 
                             for k, v in flow['vector'].items()},
                    'decoded_characteristics': self._decode_characteristics(flow['characteristics'])
                }
                
                decoded_flows.append(decoded)
        
        return decoded_flows
    
    def _decode_characteristics(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Decode flow characteristics for schizo-process."""
        # Create decoded characteristics
        decoded = {
            'primary': characteristics['primary'],
            'contents': characteristics.get('contents', []),
            'fluidity': characteristics.get('fluidity', 0.5) * random.uniform(1.1, 1.5),
            'deterritorialized': True
        }
        
        return decoded
    
    def _generate_breakthrough_points(self, decoded_flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate breakthrough points from decoded flows."""
        breakthrough_points = []
        
        # Generate from each flow
        for flow in decoded_flows:
            # Only create breakthrough if intensity is high enough
            if flow['intensity'] > 0.6:
                point = {
                    'source_flow': flow['original_flow'],
                    'intensity': flow['intensity'] * random.uniform(0.8, 1.2),
                    'coordinates': self._generate_coordinates(flow['vector']),
                    'breakthrough_type': self._determine_breakthrough_type(flow)
                }
                
                breakthrough_points.append(point)
        
        return breakthrough_points
    
    def _generate_coordinates(self, vector: Dict[str, float]) -> Dict[str, float]:
        """Generate coordinates in conceptual space from vector."""
        # Use vector dimensions as coordinates
        coordinates = {}
        
        for dim, val in vector.items():
            if dim != 'magnitude':
                coordinates[dim] = val
        
        return coordinates
    
    def _determine_breakthrough_type(self, flow: Dict[str, Any]) -> str:
        """Determine the type of breakthrough."""
        # Types based on decoded characteristics
        primary = flow['decoded_characteristics']['primary']
        
        if primary == 'affect':
            return 'affective-breakthrough'
        elif primary == 'concept':
            return 'conceptual-breakthrough'
        elif primary == 'expression':
            return 'expressive-breakthrough'
        elif primary == 'connection':
            return 'connective-breakthrough'
        else:
            return 'intensive-breakthrough'
    
    def _create_schizo_mapping(self, breakthrough_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a schizo-mapping from breakthrough points."""
        # Create mapping
        mapping = {
            'type': 'schizo-mapping',
            'intensity': sum(p['intensity'] for p in breakthrough_points) / len(breakthrough_points)
                        if breakthrough_points else 0.5,
            'dimensions': set(),
            'connections': []
        }
        
        # Collect all dimensions
        for point in breakthrough_points:
            mapping['dimensions'].update(point['coordinates'].keys())
        
        # Create connections between points
        for i, point_a in enumerate(breakthrough_points):
            for point_b in breakthrough_points[i+1:]:
                # Calculate connection intensity (dot product of coordinates)
                intensity = 0
                for dim in set(point_a['coordinates']) & set(point_b['coordinates']):
                    intensity += point_a['coordinates'][dim] * point_b['coordinates'][dim]
                
                # Only create connection if significant
                if intensity > 0.2:
                    connection = {
                        'from': point_a['source_flow'],
                        'to': point_b['source_flow'],
                        'intensity': intensity,
                        'type': 'schizo-connection'
                    }
                    
                    mapping['connections'].append(connection)
        
        return mapping
    
    # Helper methods for machine connections
    def _determine_connection_type(self, machine_a: Dict[str, Any], 
                                 machine_b: Dict[str, Any]) -> str:
        """Determine the type of connection between machines."""
        # Connection types based on machine types
        type_a = machine_a['type']
        type_b = machine_b['type']
        
        if 'rhizomatic' in type_a or 'rhizomatic' in type_b:
            return 'rhizomatic-connection'
        elif 'affective' in type_a and 'conceptual' in type_b:
            return 'affect-concept-connection'
        elif 'conceptual' in type_a and 'affective' in type_b:
            return 'concept-affect-connection'
        elif 'expressive' in type_a or 'expressive' in type_b:
            return 'expressive-connection'
        else:
            return 'productive-connection'
    
    def _calculate_connection_intensity(self, machine_a: Dict[str, Any], 
                                      machine_b: Dict[str, Any]) -> float:
        """Calculate connection intensity between machines."""
        # Base on machine productive capacities
        base_intensity = (machine_a['productive_capacity'] + machine_b['productive_capacity']) / 2
        
        # Adjust based on connection type
        connection_type = self._determine_connection_type(machine_a, machine_b)
        
        if 'rhizomatic' in connection_type:
            base_intensity *= 1.2
        elif 'affect' in connection_type or 'concept' in connection_type:
            base_intensity *= 1.1
            
        # Ensure within range
        return min(1.0, max(0.1, base_intensity))
    
    def _create_inter_machine_flow(self, machine_a: Dict[str, Any], 
                                 machine_b: Dict[str, Any], 
                                 intensity: float) -> Dict[str, Any]:
        """Create flow between machines."""
        # Create flow
        flow_id = f"flow_{machine_a['id']}_{machine_b['id']}_{int(time.time())}"
        
        # Determine flow direction
        if machine_a['productive_capacity'] > machine_b['productive_capacity']:
            direction = {'from': machine_a['id'], 'to': machine_b['id']}
        else:
            direction = {'from': machine_b['id'], 'to': machine_a['id']}
        
        # Create the flow
        flow = {
            'id': flow_id,
            'intensity': intensity,
            'direction': direction,
            'type': self._determine_connection_type(machine_a, machine_b),
            'creation_timestamp': self._get_timestamp()
        }
        
        # Register the flow
        self.flows[flow_id] = flow
        
        return flow
    
    # Utility methods
    def _record_production(self, machine: Dict[str, Any], 
                         flow: Dict[str, Any], 
                         production: Dict[str, Any]) -> None:
        """Record production details for future reference."""
        # Record connection between machine and flow
        self.connections[machine['id']].append({
            'type': 'production',
            'machine': machine['id'],
            'flow': flow['id'],
            'production': production,
            'timestamp': self._get_timestamp()
        })
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return str(int(time.time() * 1000))
    
    # Public interface methods for integration
    def get_active_productions(self) -> List[Dict[str, Any]]:
        """
        Get all active desire productions.
        
        Returns:
            List of active productions
        """
        # Collect from all registers
        productions = []
        for register in self.production_registers.values():
            productions.extend(register[-5:])  # Last 5 from each
        
        # Sort by recency
        if 'creation_timestamp' in productions[0]:
            productions = sorted(productions, 
                               key=lambda p: p['creation_timestamp'], 
                               reverse=True)
        
        return productions
    
    def get_schizo_processes(self) -> List[Dict[str, Any]]:
        """
        Get all schizo-processes.
        
        Returns:
            List of schizo-processes
        """
        return list(self.schizo_process_states.values())
    
    def get_deterritorializations(self) -> List[Dict[str, Any]]:
        """
        Get all deterritorialization vectors.
        
        Returns:
            List of deterritorialization vectors
        """
        return self.deterritorialization_vectors
```
