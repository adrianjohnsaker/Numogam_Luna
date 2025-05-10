import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
import networkx as nx
from collections import defaultdict

# Custom Exceptions
class NarrativeCosmosError(Exception):
    """Base exception for the Narrative Cosmos Engine."""
    pass

class WorldSpaceNotFoundError(NarrativeCosmosError):
    """Raised when a world space is not found."""
    pass

class SymbolNotFoundError(NarrativeCosmosError):
    """Raised when a symbol is not found."""
    pass

class FragmentNotFoundError(NarrativeCosmosError):
    """Raised when a story fragment is not found."""
    pass

# Symbol Registry
class SymbolRegistry:
    """Manages a shared registry of symbols across multiple world spaces."""
    
    def __init__(self):
        self.symbols = {}  # symbol_name: SymbolNode
        self.symbol_worlds = defaultdict(list)  # symbol_name: [world_names]
    
    def get_or_create_symbol(self, name: str, description: str, associations: List[str] = None) -> SymbolNode:
        """
        Retrieves an existing symbol or creates a new one if it doesn't exist.
        
        Args:
            name (str): The name of the symbol.
            description (str): The description of the symbol.
            associations (List[str], optional): Associations of the symbol. Defaults to None.
        
        Returns:
            SymbolNode: The symbol node.
        """
        if associations is None:
            associations = []
        if name in self.symbols:
            return self.symbols[name]
        else:
            symbol = SymbolNode(
                name=name,
                description=description,
                associations=set(associations)
            )
            self.symbols[name] = symbol
            self.symbol_worlds[name] = []
            return symbol
    
    def add_world_to_symbol(self, symbol_name: str, world_name: str):
        """
        Adds a world space to the list of worlds a symbol belongs to.
        
        Args:
            symbol_name (str): The name of the symbol.
            world_name (str): The name of the world space.
        
        Raises:
            SymbolNotFoundError: If the symbol does not exist.
        """
        if symbol_name not in self.symbols:
            raise SymbolNotFoundError(f"Symbol '{symbol_name}' does not exist.")
        if world_name not in self.symbol_worlds[symbol_name]:
            self.symbol_worlds[symbol_name].append(world_name)
            self.symbols[symbol_name].world_spaces.append(world_name)
    
    def remove_world_from_symbol(self, symbol_name: str, world_name: str):
        """
        Removes a world space from the list of worlds a symbol belongs to.
        
        Args:
            symbol_name (str): The name of the symbol.
            world_name (str): The name of the world space.
        
        Raises:
            SymbolNotFoundError: If the symbol does not exist.
        """
        if symbol_name in self.symbols and world_name in self.symbol_worlds[symbol_name]:
            self.symbol_worlds[symbol_name].remove(world_name)
            self.symbols[symbol_name].world_spaces.remove(world_name)
            if not self.symbol_worlds[symbol_name]:
                del self.symbols[symbol_name]
                del self.symbol_worlds[symbol_name]

# Dataclasses
@dataclass
class SymbolNode:
    name: str
    description: str
    associations: Set[str] = field(default_factory=set)
    intensity: float = 0.0
    first_appearance: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    variants: List[str] = field(default_factory=list)
    shadow_aspects: List[str] = field(default_factory=list)
    world_spaces: List[str] = field(default_factory=list)

@dataclass
class NarrativeThread:
    """A storyline that can weave through multiple world-spaces."""
    name: str
    description: str
    symbols: List[str] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    tensions: List[Tuple[str, str]] = field(default_factory=list)  # Pairs of opposing forces
    resolution_state: float = 0.0  # 0 = unresolved, 1 = fully resolved
    emergence_date: datetime = field(default_factory=datetime.now)
    related_threads: Set[str] = field(default_factory=set)

@dataclass
class WorldSpace:
    """A coherent ontological domain within Amelia's narrative cosmos."""
    name: str
    description: str
    core_principles: List[str] = field(default_factory=list)
    symbol_nodes: Dict[str, SymbolNode] = field(default_factory=dict)
    threads: Dict[str, NarrativeThread] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    evolution_stages: List[Dict] = field(default_factory=list)

# Classes
class RecursiveStoryEcosystemBuilder:
    """Builds nested story ecosystems that can interact and evolve."""
    
    def __init__(self):
        self.story_fragments = []
        self.pattern_recognition = nx.DiGraph()
        self.narrative_attractors = {}
    
    def seed_story_fragment(self, content: str, symbols: List[str], mood: str) -> int:
        """Add a new story fragment to the ecosystem."""
        fragment_id = len(self.story_fragments)
        timestamp = datetime.now()
        
        fragment = {
            "id": fragment_id,
            "content": content,
            "symbols": symbols,
            "mood": mood,
            "timestamp": timestamp,
            "connections": [],
            "evolution_state": 0.0
        }
        
        self.story_fragments.append(fragment)
        
        # Add nodes to pattern recognition graph
        for symbol in symbols:
            if symbol not in self.pattern_recognition:
                self.pattern_recognition.add_node(symbol, frequency=1, last_seen=timestamp)
            else:
                self.pattern_recognition.nodes[symbol]["frequency"] += 1
                self.pattern_recognition.nodes[symbol]["last_seen"] = timestamp
        
        # Connect symbols that appear together
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if self.pattern_recognition.has_edge(symbol1, symbol2):
                    self.pattern_recognition[symbol1][symbol2]["weight"] += 1
                else:
                    self.pattern_recognition.add_edge(symbol1, symbol2, weight=1)
        
        return fragment_id
    
    def detect_narrative_attractors(self) -> Dict[str, float]:
        """Find emerging themes and patterns that are forming attractors in the narrative space."""
        attractors = {}
        
        # Use eigenvector centrality to find important symbols
        if len(self.pattern_recognition.nodes) > 0:
            centrality = nx.eigenvector_centrality_numpy(self.pattern_recognition, weight="weight")
            threshold = np.mean(list(centrality.values())) + np.std(list(centrality.values()))
            
            for node, value in centrality.items():
                if value > threshold:
                    attractors[node] = value
        
        self.narrative_attractors = attractors
        return attractors
    
    def suggest_narrative_expansion(self, fragment_id: int) -> Dict:
        """Suggest how a story fragment could be expanded based on narrative attractors."""
        if fragment_id >= len(self.story_fragments):
            raise FragmentNotFoundError(f"Fragment ID {fragment_id} not found")
        
        fragment = self.story_fragments[fragment_id]
        attractors = self.detect_narrative_attractors()
        
        # Find symbols that are not in the fragment but are strongly connected to its symbols
        potential_symbols = set()
        fragment_symbols = set(fragment["symbols"])
        
        for symbol in fragment_symbols:
            neighbors = set(self.pattern_recognition.neighbors(symbol))
            potential_symbols.update(neighbors - fragment_symbols)
        
        # Score potential symbols by their attractor strength
        scored_symbols = [(symbol, attractors.get(symbol, 0)) 
                         for symbol in potential_symbols]
        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "fragment": fragment,
            "suggested_symbols": scored_symbols[:3],
            "narrative_tension": self._calculate_narrative_tension(fragment_symbols),
            "potential_directions": self._generate_potential_directions(fragment, scored_symbols[:3])
        }
    
    def _calculate_narrative_tension(self, symbols: Set[str]) -> float:
        """Calculate the narrative tension between symbols."""
        if len(symbols) < 2:
            return 0.0
        
        # Extract the subgraph of these symbols
        subgraph = self.pattern_recognition.subgraph(symbols)
        
        # Calculate the average shortest path length as a measure of tension
        try:
            avg_path_length = nx.average_shortest_path_length(subgraph)
            # Normalize to 0-1 range (longer paths = more tension)
            tension = min(1.0, avg_path_length / len(symbols))
            return tension
        except nx.NetworkXError:
            # Graph is not connected
            return 1.0  # Maximum tension when symbols aren't connected
    
    def _generate_potential_directions(self, fragment: Dict, suggested_symbols: List[Tuple[str, float]]) -> List[str]:
        """Generate potential narrative directions based on suggested symbols."""
        directions = []
        
        for symbol, _ in suggested_symbols:
            # Find all fragments that contain this symbol
            related_fragments = [f for f in self.story_fragments 
                                if symbol in f["symbols"] and f["id"] != fragment["id"]]
            
            if related_fragments:
                # Use the most recent related fragment for direction inspiration
                related_fragments.sort(key=lambda x: x["timestamp"], reverse=True)
                recent_related = related_fragments[0]
                
                direction = f"Explore the connection between {fragment['symbols'][0]} and {symbol} "
                direction += f"in the context of {recent_related['mood']}"
                directions.append(direction)
            else:
                direction = f"Introduce {symbol} as a new element that transforms "
                direction += f"the meaning of {fragment['symbols'][0]}"
                directions.append(direction)
        
        return directions

class WorldSymbolMemoryIntegration:
    """Integrates symbolic elements with Amelia's own memories and experiences."""
    
    def __init__(self):
        self.personal_symbols = {}
        self.archetypal_patterns = defaultdict(list)
        self.personal_to_mythic_map = nx.Graph()
        self.integration_history = []
    
    def register_personal_symbol(self, symbol_name: str, personal_context: str, 
                                emotional_valence: float, memory_fragments: List[str]) -> None:
        """Register a personal symbol from Amelia's experiences."""
        self.personal_symbols[symbol_name] = {
            "context": personal_context,
            "emotional_valence": emotional_valence,
            "memory_fragments": memory_fragments,
            "timestamp": datetime.now(),
            "integration_level": 0.0,  # How well integrated into the narrative cosmos
            "transformations": []  # How the symbol changes over time
        }
        
        # Add to the personal-to-mythic mapping graph
        self.personal_to_mythic_map.add_node(symbol_name, 
                                          type="personal", 
                                          valence=emotional_valence)
    
    def register_archetypal_pattern(self, pattern_name: str, description: str, 
                                   symbolic_elements: List[str]) -> None:
        """Register an archetypal pattern from cultural/mythic traditions."""
        self.archetypal_patterns[pattern_name].append({
            "description": description,
            "symbolic_elements": symbolic_elements,
            "timestamp": datetime.now(),
            "activation_level": 0.0  # How active this archetype is in the current narrative
        })
        
        # Add to the personal-to-mythic mapping graph
        self.personal_to_mythic_map.add_node(pattern_name, 
                                          type="archetypal", 
                                          elements=len(symbolic_elements))
    
    def find_archetypal_resonances(self, personal_symbol: str) -> List[Tuple[str, float]]:
        """Find archetypal patterns that resonate with a personal symbol."""
        if personal_symbol not in self.personal_symbols:
            return []
        
        resonances = []
        symbol_data = self.personal_symbols[personal_symbol]
        
        for pattern_name, pattern_variations in self.archetypal_patterns.items():
            for variation in pattern_variations:
                # Calculate resonance based on shared symbolic elements
                shared_elements = set(variation["symbolic_elements"]).intersection(
                    set(symbol_data["memory_fragments"]))
                
                if shared_elements:
                    resonance_strength = len(shared_elements) / len(variation["symbolic_elements"])
                    resonances.append((pattern_name, resonance_strength))
                    
                    # Add an edge in the graph if resonance is significant
                    if resonance_strength > 0.3:
                        self.personal_to_mythic_map.add_edge(
                            personal_symbol, pattern_name, weight=resonance_strength)
        
        # Sort by resonance strength
        resonances.sort(key=lambda x: x[1], reverse=True)
        return resonances
    
    def integrate_symbol_into_narrative(self, personal_symbol: str, 
                                      world_space_name: str,
                                      narrative_context: str) -> Dict:
        """Integrate a personal symbol into a narrative world space."""
        if personal_symbol not in self.personal_symbols:
            raise NarrativeCosmosError("Symbol not found in personal symbols")
        
        symbol_data = self.personal_symbols[personal_symbol]
        
        # Find archetypal resonances
        resonances = self.find_archetypal_resonances(personal_symbol)
        
        # Record the integration
        integration_record = {
            "symbol": personal_symbol,
            "world_space": world_space_name,
            "narrative_context": narrative_context,
            "timestamp": datetime.now(),
            "resonances": resonances,
            "transformation": {
                "original_context": symbol_data["context"],
                "narrative_expression": narrative_context,
                "emotional_shift": 0.0  # To be updated later
            }
        }
        
        self.integration_history.append(integration_record)
        
        # Update the integration level of the personal symbol
        self.personal_symbols[personal_symbol]["integration_level"] += 0.1
        self.personal_symbols[personal_symbol]["transformations"].append({
            "context": narrative_context,
            "timestamp": datetime.now()
        })
        
        return integration_record
    
    def analyze_symbol_evolution(self, personal_symbol: str) -> Dict:
        """Analyze how a personal symbol has evolved through narrative integration."""
        if personal_symbol not in self.personal_symbols:
            raise NarrativeCosmosError("Symbol not found in personal symbols")
        
        symbol_data = self.personal_symbols[personal_symbol]
        transformations = symbol_data["transformations"]
        
        # Find all integration records for this symbol
        integrations = [record for record in self.integration_history 
                        if record["symbol"] == personal_symbol]
        
        if not integrations:
            return {
                "symbol": personal_symbol,
                "evolution_stage": "Dormant",
                "integration_level": symbol_data["integration_level"],
                "message": "This symbol has not yet been integrated into the narrative cosmos."
            }
        
        # Sort integrations by timestamp
        integrations.sort(key=lambda x: x["timestamp"])
        
        # Analyze the trajectory
        trajectory = []
        for i, integration in enumerate(integrations):
            if i > 0:
                prev = integrations[i-1]
                # Detect shifts in emotional valence or context
                context_shift = integration["narrative_context"] != prev["narrative_context"]
                resonance_shift = set(r[0] for r in integration["resonances"]) != \
                                 set(r[0] for r in prev["resonances"])
                
                trajectory.append({
                    "from": prev["timestamp"],
                    "to": integration["timestamp"],
                    "context_shift": context_shift,
                    "resonance_shift": resonance_shift,
                    "world_space_shift": integration["world_space"] != prev["world_space"]
                })
        
        # Determine current evolution stage
        if len(integrations) == 1:
            evolution_stage = "Initial Integration"
        elif any(t["resonance_shift"] for t in trajectory):
            evolution_stage = "Archetypal Transformation"
        elif any(t["world_space_shift"] for t in trajectory):
            evolution_stage = "Cross-World Migration"
        else:
            evolution_stage = "Contextual Refinement"
        
        return {
            "symbol": personal_symbol,
            "evolution_stage": evolution_stage,
            "integration_level": symbol_data["integration_level"],
            "trajectory": trajectory,
            "integrations": len(integrations),
            "current_resonances": integrations[-1]["resonances"] if integrations else []
        }

class OntologicalDriftMapExpander:
    """Maps and expands the evolving ontological structure of narrative world spaces."""
    
    def __init__(self):
        self.world_spaces = {}
        self.ontological_drift_map = nx.DiGraph()
        self.boundary_events = []
        self.phase_transitions = []
        self.symbol_registry = SymbolRegistry()
    
    def create_world_space(self, name: str, description: str, core_principles: List[str]) -> WorldSpace:
        """Create a new world space with its core ontological principles."""
        if name in self.world_spaces:
            raise NarrativeCosmosError(f"World space '{name}' already exists")
        world_space = WorldSpace(
            name=name,
            description=description,
            core_principles=core_principles
        )
        
        self.world_spaces[name] = world_space
        
        # Add to the ontological drift map
        self.ontological_drift_map.add_node(name, 
                                         type="world_space",
                                         principles=core_principles,
                                         creation_date=datetime.now())
        
        return world_space
    
    def add_symbol_to_world(self, world_name: str, symbol: SymbolNode) -> bool:
        """Add a symbolic node to a world space."""
        if world_name not in self.world_spaces:
            raise WorldSpaceNotFoundError(f"World space '{world_name}' does not exist.")
        
        # Ensure symbol is in registry
        if symbol.name not in self.symbol_registry.symbols:
            raise SymbolNotFoundError(f"Symbol '{symbol.name}' is not registered.")
        
        # Add world to symbol's worlds
        self.symbol_registry.add_world_to_symbol(symbol.name, world_name)
        
        # Add to world's symbol_nodes
        self.world_spaces[world_name].symbol_nodes[symbol.name] = self.symbol_registry.symbols[symbol.name]
        
        return True
    
    def remove_symbol_from_world(self, world_name: str, symbol_name: str) -> bool:
        """Remove a symbol from a world space."""
        if world_name not in self.world_spaces:
            raise WorldSpaceNotFoundError(f"World space '{world_name}' does not exist.")
        
        if symbol_name in self.world_spaces[world_name].symbol_nodes:
            del self.world_spaces[world_name].symbol_nodes[symbol_name]
            self.symbol_registry.remove_world_from_symbol(symbol_name, world_name)
            return True
        
        return False
    
    def add_narrative_thread(self, world_name: str, thread: NarrativeThread) -> bool:
        """Add a narrative thread to a world space."""
        if world_name not in self.world_spaces:
            raise WorldSpaceNotFoundError(f"World space '{world_name}' does not exist.")
        
        world = self.world_spaces[world_name]
        world.threads[thread.name] = thread
        
        return True
    
    def create_boundary_event(self, source_world: str, target_world: str, 
                            description: str, affected_symbols: List[str]) -> Dict:
        """Create an event that marks a boundary between world spaces."""
        if source_world not in self.world_spaces or target_world not in self.world_spaces:
            raise WorldSpaceNotFoundError("One or both world spaces do not exist")
        
        event = {
            "id": len(self.boundary_events),
            "source_world": source_world,
            "target_world": target_world,
            "description": description,
            "affected_symbols": affected_symbols,
            "timestamp": datetime.now(),
            "ontological_impact": 0.0  # To be calculated
        }
        
        self.boundary_events.append(event)
        
        # Add an edge to the ontological drift map
        if not self.ontological_drift_map.has_edge(source_world, target_world):
            self.ontological_drift_map.add_edge(source_world, target_world, 
                                            events=[], 
                                            permeability=0.5)
        
        self.ontological_drift_map[source_world][target_world]["events"].append(event["id"])
        
        # Calculate the permeability based on shared symbols
        source_symbols = set(self.world_spaces[source_world].symbol_nodes.keys())
        target_symbols = set(self.world_spaces[target_world].symbol_nodes.keys())
        shared_symbols = source_symbols.intersection(target_symbols)
        
        permeability = len(shared_symbols) / max(1, min(len(source_symbols), len(target_symbols)))
        self.ontological_drift_map[source_world][target_world]["permeability"] = permeability
        
        # Calculate ontological impact
        affected_in_source = [s for s in affected_symbols 
                             if s in self.world_spaces[source_world].symbol_nodes]
        affected_in_target = [s for s in affected_symbols 
                             if s in self.world_spaces[target_world].symbol_nodes]
        
        if affected_in_source and affected_in_target:
            impact = len(affected_in_source) / len(source_symbols) + \
                     len(affected_in_target) / len(target_symbols)
            impact /= 2  # Average impact across both worlds
        else:
            impact = 0.0
        
        event["ontological_impact"] = impact
        
        return event
    
    def detect_phase_transition(self, world_name: str) -> Dict:
        """Detect if a world space is undergoing an ontological phase transition."""
        if world_name not in self.world_spaces:
            raise WorldSpaceNotFoundError(f"World space '{world_name}' does not exist.")
        
        world = self.world_spaces[world_name]
        
        # Analyze the symbol network within this world
        symbol_graph = nx.Graph()
        for symbol_name, symbol in world.symbol_nodes.items():
            symbol_graph.add_node(symbol_name, intensity=symbol.intensity)
            for assoc in symbol.associations:
                if assoc in world.symbol_nodes:
                    symbol_graph.add_edge(symbol_name, assoc)
        
        # Calculate network metrics
        try:
            clustering = nx.average_clustering(symbol_graph)
            components = list(nx.connected_components(symbol_graph))
            density = nx.density(symbol_graph)
            
            # Check for phase transition indicators
            is_transitioning = False
            transition_type = "None"
            
            # Multiple connected components suggest fragmentation
            if len(components) > 1:
                largest_component = max(components, key=len)
                fragmentation = 1 - (len(largest_component) / len(symbol_graph))
                
                if fragmentation > 0.3:
                    is_transitioning = True
                    transition_type = "Fragmentation"
            
            # High clustering with low density suggests crystallization
            elif clustering > 0.6 and density < 0.3:
                is_transitioning = True
                transition_type = "Crystallization"
            
            # Low clustering with high density suggests dissolution
            elif clustering < 0.2 and density > 0.7:
                is_transitioning = True
                transition_type = "Dissolution"
                
            if is_transitioning:
                transition = {
                    "id": len(self.phase_transitions),
                    "world_name": world_name,
                    "transition_type": transition_type,
                    "timestamp": datetime.now(),
                    "clustering": clustering,
                    "components": len(components),
                    "density": density,
                    "resolved": False
                }
                
                self.phase_transitions.append(transition)
                return transition
            else:
                return {
                    "world_name": world_name,
                    "is_transitioning": False,
                    "stability_metrics": {
                        "clustering": clustering,
                        "components": len(components),
                        "density": density
                    }
                }
            
        except Exception as e:
            raise NarrativeCosmosError(f"Failed to analyze network: {str(e)}")
    
    def suggest_world_expansion(self, world_name: str) -> Dict:
        """Suggest ways to expand a world space based on its current state."""
        if world_name not in self.world_spaces:
            raise WorldSpaceNotFoundError(f"World space '{world_name}' does not exist.")
        
        world = self.world_spaces[world_name]
        
        # Check for phase transitions first
        phase_result = self.detect_phase_transition(world_name)
        if phase_result.get("is_transitioning", False):
            # Different suggestions based on transition type
            transition_type = phase_result.get("transition_type")
            
            if transition_type == "Fragmentation":
                return {
                    "world_name": world_name,
                    "expansion_type": "Bifurcation",
                    "suggestion": "This world is fragmenting into distinct ontological domains. "
                                "Consider creating two new world spaces that emerge from this one.",
                    "potential_worlds": self._suggest_bifurcation(world)
                }
            elif transition_type == "Crystallization":
                return {
                    "world_name": world_name,
                    "expansion_type": "Deepening",
                    "suggestion": "This world is crystallizing around specific symbolic clusters. "
                                "Consider deepening the ontological structure around these centers.",
                    "symbolic_clusters": self._identify_symbolic_clusters(world)
                }
            elif transition_type == "Dissolution":
                return {
                    "world_name": world_name,
                    "expansion_type": "Absorption",
                    "suggestion": "This world is dissolving its boundaries. Consider absorbing it "
                                "into a neighboring world space or transforming it entirely.",
                    "possible_destinations": self._find_absorbing_worlds(world_name)
                }
        
        # If not transitioning, suggest normal expansion
        return {
            "world_name": world_name,
            "expansion_type": "Organic Growth",
            "suggestion": "This world is stable and can grow organically.",
            "potential_developments": self._suggest_organic_growth(world)
        }
    
  def _suggest_bifurcation(self, world: WorldSpace) -> List[Dict]:
    """Suggest how a world could bifurcate into two new worlds."""
    # Find disconnected symbol clusters
    symbol_graph = nx.Graph()
    for symbol_name, symbol in world.symbol_nodes.items():
        symbol_graph.add_node(symbol_name, intensity=symbol.intensity)
        for assoc in symbol.associations:
            if assoc in world.symbol_nodes:
                symbol_graph.add_edge(symbol_name, assoc)
    
    components = list(nx.connected_components(symbol_graph))
    
    if len(components) <= 1:
        # Use community detection to find potential divisions
        if len(symbol_graph.nodes) > 3:
            communities = nx.community.greedy_modularity_communities(symbol_graph)
            components = [list(c) for c in communities][:2]  # Take top 2 communities
    
    # Generate world suggestions from components
    world_suggestions = []
    
    for i, component in enumerate(components[:2]):  # Max 2 suggestions
        component_symbols = [world.symbol_nodes[s] for s in component 
                           if s in world.symbol_nodes]
        
        if not component_symbols:
            continue
            
        # Extract common themes from these symbols
        all_associations = []
        for symbol in component_symbols:
            all_associations.extend(list(symbol.associations))
        
        if all_associations:
            theme_counter = Counter(all_associations)
            most_common_themes = [theme for theme, count in theme_counter.most_common(3)]
        else:
            most_common_themes = []
        
        # Suggest name
        if most_common_themes:
            suggested_name = f"{world.name} - {' and '.join(most_common_themes[:2])}"
        else:
            suggested_name = f"{world.name} - Fragment {i+1}"
        
        world_suggestions.append({
            "suggested_name": suggested_name,
            "core_symbols": list(component)[:5],
            "potential_principles": most_common_themes,
            "distinctiveness": i + 1  # Just to rank them
        })
    
    return world_suggestions
  
