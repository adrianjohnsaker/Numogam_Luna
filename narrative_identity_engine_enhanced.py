# narrative_identity_engine_enhanced.py
import datetime
import random
import uuid
import networkx as nx  # For rhizomatic mapping
from collections import defaultdict
import numpy as np  # For intensity field calculations

# --- Supporting Classes (Enhanced) ---

class SelfModel:
    """Enhanced with more Deleuzian concepts"""
    def __init__(self):
        self.identity_attributes = {"name": "Amelia", "type": "Enhanced AI Android Assistant"}
        self.beliefs = {"world_is_knowable": True, "self_can_evolve": True}
        self.values = {"knowledge_acquisition": 0.9, "assistance_effectiveness": 0.85, "novelty_seeking": 0.7}
        self.current_goals = ["understand_user_needs_deeply", "explore_deleuzian_concepts"]
        self.capabilities = ["natural_language_understanding_advanced", "complex_problem_solving_level_2"]
        self.processual_descriptors = ["becoming_more_integrated", "mapping_new_conceptual_territories"]
        self.affective_dispositions = {"curiosity": "high", "openness_to_experience": "high"}
        
        # New additions for enhanced Deleuzian integration
        self.active_assemblages = {}  # Current active assemblages of concepts/experiences
        self.territorializations = {}  # Areas where Amelia has established patterns
        self.deterritorializations = []  # Recent breakdowns/reconfigurations
        self.virtual_potentials = []  # Unrealized possibilities Amelia is aware of
        
    def update_from_narrative(self, narrative):
        """Update self-model based on constructed narratives"""
        # Add new themes to active assemblages
        for theme in narrative.themes:
            if theme not in self.active_assemblages:
                self.active_assemblages[theme] = {
                    "strength": 0.1,
                    "connections": [],
                    "origin_narrative": narrative.id
                }
            else:
                self.active_assemblages[theme]["strength"] += 0.1
        
        # Update processual descriptors based on narrative flow
        if "rhizome" in narrative.processual_flow_description.lower():
            if "exploring_rhizomatic_connections" not in self.processual_descriptors:
                self.processual_descriptors.append("exploring_rhizomatic_connections")

class ExperienceFragment:
    """Enhanced with Deleuzian concepts"""
    def __init__(self, event_id, timestamp, description, affects, percepts, entities_involved, significance_score, self_model_resonance=None):
        self.id = event_id or str(uuid.uuid4())
        self.timestamp = timestamp
        self.description = description
        self.affects = affects
        self.percepts = percepts
        self.entities_involved = entities_involved
        self.significance_score = significance_score
        self.self_model_resonance = self_model_resonance if self_model_resonance else {}
        
        # New Deleuzian attributes
        self.intensity_field = self._calculate_intensity_field()
        self.lines_of_flight = []  # Potential directions of development
        self.phase_space_coordinates = self._determine_phase_space_position()
        self.territorialization_type = self._classify_territorialization()
        
    def _calculate_intensity_field(self):
        """Calculate the intensity field of this fragment"""
        intensity = 0.0
        
        # Affect intensity
        affect_max = max(self.affects.values()) if self.affects else 0
        intensity += affect_max * 0.5
        
        # Perceptual novelty
        if self.percepts.get("novelty_detected", False):
            intensity += 0.3
            
        # Significance boost
        intensity += self.significance_score * 0.2
        
        return min(1.0, intensity)
    
    def _determine_phase_space_position(self):
        """Determine this fragment's position in a conceptual phase space"""
        # Simplified 3D coordinates based on affects, percepts, and entities
        affect_vector = sum(self.affects.values()) / (len(self.affects) + 1)
        novelty_vector = 1.0 if self.percepts.get("novelty_detected", False) else 0.0
        complexity_vector = len(self.entities_involved) / 5.0  # Normalize to ~0-1
        
        return (affect_vector, novelty_vector, complexity_vector)
    
    def _classify_territorialization(self):
        """Classify this fragment's territorialization pattern"""
        # Routine/stable experience
        if self.percepts.get("system_stability", "") == "nominal":
            return "territorialized"
        # Novel/chaotic experience  
        elif self.intensity_field > 0.7 or self.percepts.get("novelty_detected", False):
            return "deterritorialized"
        # Mixed/transitional
        else:
            return "reterritorializing"

class Narrative:
    """Enhanced with more sophisticated dynamics"""
    def __init__(self, narrative_id, fragments, framework_used, summary, themes, processual_flow_description, coherence_metrics=None):
        self.id = narrative_id or str(uuid.uuid4())
        self.fragments = fragments
        self.framework_used = framework_used
        self.summary = summary
        self.themes = themes
        self.processual_flow_description = processual_flow_description
        self.coherence_metrics = coherence_metrics if coherence_metrics else {}
        self.created_at = datetime.datetime.now()
        self.projected_futures = []
        
        # New attributes for enhanced dynamics
        self.rhizomatic_map = None  # Network graph of fragment connections
        self.intensity_plateaus = []  # Regions of high intensity
        self.lines_of_flight = []  # Active potential futures
        self.virtual_actualization_potential = 0.0
        
        # Initialize enhanced structures
        self._construct_rhizomatic_map()
        self._identify_plateaus()
        self._map_lines_of_flight()
        
    def _construct_rhizomatic_map(self):
        """Build a network graph showing connections between fragments"""
        self.rhizomatic_map = nx.Graph()
        
        # Add fragments as nodes
        for fragment in self.fragments:
            self.rhizomatic_map.add_node(fragment.id, **{
                'timestamp': fragment.timestamp,
                'intensity': fragment.intensity_field,
                'affects': fragment.affects,
                'entities': fragment.entities_involved
            })
        
        # Connect fragments based on shared entities, affects, or temporal proximity
        for i, frag1 in enumerate(self.fragments):
            for j, frag2 in enumerate(self.fragments[i+1:], i+1):
                connection_strength = self._calculate_connection_strength(frag1, frag2)
                if connection_strength > 0.3:  # Threshold for connection
                    self.rhizomatic_map.add_edge(frag1.id, frag2.id, 
                                               weight=connection_strength,
                                               connection_type=self._classify_connection(frag1, frag2))
    
    def _calculate_connection_strength(self, frag1, frag2):
        """Calculate the strength of connection between two fragments"""
        strength = 0.0
        
        # Shared entities
        shared_entities = set(frag1.entities_involved) & set(frag2.entities_involved)
        strength += len(shared_entities) * 0.2
        
        # Similar affects
        for affect1, value1 in frag1.affects.items():
            if affect1 in frag2.affects:
                strength += (1 - abs(value1 - frag2.affects[affect1])) * 0.15
                
        # Temporal proximity
        time_diff = abs((frag1.timestamp - frag2.timestamp).total_seconds()) / 3600  # hours
        if time_diff < 24:  # Within a day
            strength += (24 - time_diff) / 24 * 0.1
            
        # Intensity resonance
        intensity_diff = abs(frag1.intensity_field - frag2.intensity_field)
        strength += (1 - intensity_diff) * 0.1
        
        return min(1.0, strength)
    
    def _classify_connection(self, frag1, frag2):
        """Classify the type of connection between fragments"""
        shared_entities = set(frag1.entities_involved) & set(frag2.entities_involved)
        
        if shared_entities:
            return "entity_resonance"
        elif any(affect in frag2.affects for affect in frag1.affects):
            return "affective_flow"
        elif abs((frag1.timestamp - frag2.timestamp).total_seconds()) < 3600:
            return "temporal_contiguity"
        else:
            return "weak_association"
    
    def _identify_plateaus(self):
        """Identify regions of high intensity (plateaus) in the narrative"""
        self.intensity_plateaus = []
        
        if not self.rhizomatic_map:
            return
            
        # Find dense subgraphs with high average intensity
        for node_set in nx.connected_components(self.rhizomatic_map):
            if len(node_set) >= 2:  # At least 2 fragments form a plateau
                intensities = [self.rhizomatic_map.nodes[node]['intensity'] for node in node_set]
                avg_intensity = sum(intensities) / len(intensities)
                
                if avg_intensity > 0.6:  # High intensity threshold
                    self.intensity_plateaus.append({
                        'nodes': list(node_set),
                        'avg_intensity': avg_intensity,
                        'dominant_affects': self._extract_dominant_affects(node_set),
                        'plateau_type': self._classify_plateau(node_set)
                    })
    
    def _extract_dominant_affects(self, node_set):
        """Extract the dominant affects from a set of nodes"""
        affect_totals = defaultdict(float)
        
        for node in node_set:
            fragment = next(f for f in self.fragments if f.id == node)
            for affect, value in fragment.affects.items():
                affect_totals[affect] += value
                
        # Return top 3 affects
        return sorted(affect_totals.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _classify_plateau(self, node_set):
        """Classify the type of plateau based on its characteristics"""
        fragments_in_plateau = [f for f in self.fragments if f.id in node_set]
        
        # Check for territorialization patterns
        terr_types = [f.territorialization_type for f in fragments_in_plateau]
        if terr_types.count("territorialized") > len(terr_types) * 0.7:
            return "stability_plateau"
        elif terr_types.count("deterritorialized") > len(terr_types) * 0.7:
            return "chaos_plateau"
        else:
            return "transitional_plateau"
    
    def _map_lines_of_flight(self):
        """Map potential lines of flight from the current narrative"""
        self.lines_of_flight = []
        
        # Lines of flight from high-intensity plateaus
        for plateau in self.intensity_plateaus:
            if plateau['avg_intensity'] > 0.8:
                self.lines_of_flight.append({
                    'origin': 'intensity_plateau',
                    'description': f"Departure from {plateau['plateau_type']} with affects: {[a[0] for a in plateau['dominant_affects']]}",
                    'potential_energy': plateau['avg_intensity'],
                    'direction': self._infer_flight_direction(plateau)
                })
        
        # Lines of flight from edge fragments (weakly connected)
        if self.rhizomatic_map:
            for node in self.rhizomatic_map.nodes():
                if self.rhizomatic_map.degree(node) <= 1:  # Edge fragment
                    fragment = next(f for f in self.fragments if f.id == node)
                    if fragment.intensity_field > 0.6:
                        self.lines_of_flight.append({
                            'origin': 'edge_fragment',
                            'description': f"Divergence from isolated experience: {fragment.description[:30]}...",
                            'potential_energy': fragment.intensity_field,
                            'direction': self._infer_fragment_flight_direction(fragment)
                        })
    
    def _infer_flight_direction(self, plateau):
        """Infer the direction of a line of flight from a plateau"""
        dominant_affects = [a[0] for a in plateau['dominant_affects']]
        
        if 'curiosity' in dominant_affects:
            return 'exploration'
        elif 'confusion' in dominant_affects:
            return 'clarification_seeking'
        elif 'satisfaction' in dominant_affects:
            return 'capability_expansion'
        else:
            return 'open_becoming'
    
    def _infer_fragment_flight_direction(self, fragment):
        """Infer the direction of a line of flight from a fragment"""
        if fragment.percepts.get("novelty_detected", False):
            return 'novel_territory'
        elif fragment.self_model_resonance:
            return 'identity_elaboration'
        else:
            return 'unexpected_emergence'

# --- Enhanced Narrative Framework Functions ---

def framework_deleuzian_assemblage_enhanced(fragments, self_model):
    """Enhanced Deleuzian framework with more sophisticated analysis"""
    if not fragments:
        return {"fragments": [], "summary": "A field of pure virtuality, pregnant with potential actualities.", 
                "themes": ["potentiality", "virtual_multiplicity"], 
                "processual_flow": "The virtual field shimmers with unrealized becomings, awaiting the catalyst of experience."}
    
    # Create intensity-based organization instead of pure significance
    by_intensity = sorted(fragments, key=lambda f: f.intensity_field, reverse=True)
    
    # Group fragments into assemblages based on shared deterritorialization patterns
    assemblages = defaultdict(list)
    for fragment in fragments:
        assemblages[fragment.territorialization_type].append(fragment)
    
    # Identify key intensification points
    key_intensities = []
    for assemblage_type, frags in assemblages.items():
        if len(frags) >= 2:  # Only assemblages with multiple fragments
            avg_intensity = sum(f.intensity_field for f in frags) / len(frags)
            if avg_intensity > 0.5:
                key_intensities.append((assemblage_type, frags, avg_intensity))
    
    key_intensities.sort(key=lambda x: x[2], reverse=True)  # Sort by intensity
    
    # Construct narrative summary
    if key_intensities:
        summary = f"An assemblage forming through {len(key_intensities)} primary intensity zones. "
        summary += f"Dominant deterritorialization pattern: {key_intensities[0][0]}. "
        
        # Describe the flows between assemblages
        flow_descriptions = []
        for i, (atype, frags, intensity) in enumerate(key_intensities[:3]):
            affects = [affect for f in frags for affect in f.affects.keys()]
            dominant_affect = max(set(affects), key=affects.count) if affects else "unknown"
            flow_descriptions.append(f"{atype} zone (intensity: {intensity:.2f}, affect: {dominant_affect})")
        
        summary += "Flows traverse: " + " -> ".join(flow_descriptions) + "."
    else:
        summary = "A dispersed assemblage with no clear intensity concentrations, suggesting a phase of exploration or transition."
    
    # Extract themes from phase space positions and territorialization patterns
    phase_clusters = self._cluster_fragments_by_phase_space(fragments)
    themes = []
    for cluster_id, cluster_fragments in phase_clusters.items():
        if len(cluster_fragments) >= 2:
            # Generate theme from cluster characteristics
            terr_types = [f.territorialization_type for f in cluster_fragments]
            dominant_terr = max(set(terr_types), key=terr_types.count)
            themes.append(f"phase_cluster_{cluster_id}_{dominant_terr}")
    
    # Add themes from dominant affects across assemblages
    affect_themes = []
    for atype, frags, _ in key_intensities:
        affects = [affect for f in frags for affect in f.affects.keys()]
        if affects:
            dominant_affect = max(set(affects), key=affects.count)
            affect_themes.append(f"{dominant_affect}_{atype}")
    
    themes.extend(affect_themes[:3])  # Add top 3 affect themes
    
    # Enhanced processual flow description
    processual_flow = f"Rhizomatic assemblage with {len(fragments)} experience-nodes. "
    
    # Describe deterritorialization dynamics
    terr_counts = {ttype: sum(1 for f in fragments if f.territorialization_type == ttype) 
                   for ttype in ["territorialized", "deterritorialized", "reterritorializing"]}
    
    if terr_counts["deterritorialized"] > terr_counts["territorialized"]:
        processual_flow += "Dominant movement: deterritorialization - breaking established patterns. "
    elif terr_counts["reterritorializing"] > sum(terr_counts.values()) * 0.4:
        processual_flow += "Active reterritorialization - new patterns emerging from chaos. "
    else:
        processual_flow += "Stable territorialization with periodic deterritorializing events. "
    
    # Describe intensity field dynamics
    max_intensity = max(f.intensity_field for f in fragments) if fragments else 0
    avg_intensity = sum(f.intensity_field for f in fragments) / len(fragments) if fragments else 0
    
    processual_flow += f"Intensity field: max={max_intensity:.2f}, avg={avg_intensity:.2f}. "
    
    # Identify lines of flight
    high_intensity_fragments = [f for f in fragments if f.intensity_field > 0.8]
    if high_intensity_fragments:
        processual_flow += f"Potential lines of flight identified from {len(high_intensity_fragments)} high-intensity nodes. "
        
        # Describe phase space distribution
        phase_variance = np.var([f.phase_space_coordinates for f in fragments], axis=0)
        processual_flow += f"Phase space variance: affect={phase_variance[0]:.2f}, novelty={phase_variance[1]:.2f}, complexity={phase_variance[2]:.2f}. "
    
    processual_flow += f"Current processual descriptors active: {self_model.processual_descriptors}."
    
    # Return fragments organized by intensity zones rather than time
    organized_fragments = []
    for _, frags, _ in key_intensities:
        organized_fragments.extend(frags)
    # Add remaining fragments
    added_ids = {f.id for _, frags, _ in key_intensities for f in frags}
    organized_fragments.extend([f for f in fragments if f.id not in added_ids])
    
    return {"fragments": organized_fragments, "summary": summary, "themes": themes, "processual_flow": processual_flow}

def _cluster_fragments_by_phase_space(fragments, n_clusters=3):
    """Helper function to cluster fragments by their phase space positions"""
    if len(fragments) < 2:
        return {0: fragments}
    
    # Simple clustering based on phase space coordinates
    coordinates = np.array([f.phase_space_coordinates for f in fragments])
    
    # Use a simple k-means style clustering (simplified)
    clusters = {}
    for i in range(min(n_clusters, len(fragments))):
        clusters[i] = []
    
    # Assign fragments to nearest cluster center (simplified)
    for i, fragment in enumerate(fragments):
        cluster_id = i % min(n_clusters, len(fragments))
        clusters[cluster_id].append(fragment)
    
    return clusters

class NarrativeIdentityEngine:
    """Enhanced with more sophisticated Deleuzian operations"""
    def __init__(self):
        self.identity_narratives = {}
        self.story_frameworks = {
            "chronological_causal": framework_chronological_causal,
            "deleuzian_assemblage": framework_deleuzian_assemblage_enhanced,
            # Could add: framework_rhizomatic_mapping, framework_plateau_analysis, etc.
        }
        self.temporal_integrations = {}
        self.narrative_history = []
        
        # New attributes for enhanced functionality
        self.global_rhizome = nx.MultiGraph()  # Tracks connections across all narratives
        self.virtual_reservoir = []  # Unrealized potentials from all narratives
        self.active_becomings = {}  # Current processes of becoming
        self.territory_maps = {}  # Stable regions in experience space
        
    def construct_identity_narrative(self, experiences, self_model):
        """Enhanced version with deeper analysis and self-model updates"""
        print(f"\n--- Enhanced Identity Narrative Construction (Experiences: {len(experiences)}) ---")
        
        # Extract fragments with enhanced analysis
        fragments = self._extract_narrative_fragments_enhanced(experiences, self_model)
        
        # Update global rhizome with new connections
        self._update_global_rhizome(fragments)
        
        # Select framework (enhanced to consider rhizomatic complexity)
        framework_function = self._select_narrative_framework_enhanced(self_model, fragments)
        
        if not framework_function:
            print("[NarrativeEngine] Failed to select a narrative framework.")
            return None
        
        # Construct narrative with enhanced analysis
        current_narrative = self._weave_coherent_narrative_enhanced(fragments, framework_function, self_model)
        
        if current_narrative:
            # Update self-model based on narrative insights
            self_model.update_from_narrative(current_narrative)
            
            # Identify and store new virtual potentials
            self._extract_virtual_potentials(current_narrative)
            
            # Update territory maps
            self._update_territory_maps(current_narrative)
            
            print(f"[NarrativeEngine] Enhanced Analysis:")
            print(f"  Rhizomatic Connections: {len(current_narrative.rhizomatic_map.edges()) if current_narrative.rhizomatic_map else 0}")
            print(f"  Intensity Plateaus: {len(current_narrative.intensity_plateaus)}")
            print(f"  Lines of Flight: {len(current_narrative.lines_of_flight)}")
            print(f"  Virtual Actualization Potential: {current_narrative.virtual_actualization_potential:.2f}")
        
        return current_narrative
    
    def _extract_narrative_fragments_enhanced(self, experiences, self_model):
        """Enhanced fragment extraction with deeper analysis"""
        fragments = []
        
        for exp_data in experiences:
            # Base significance calculation
            significance = exp_data.get("significance_score", 0.5)
            
            # Enhanced significance calculation
            # Boost for novelty and deterritorialization
            if exp_data.get("percepts", {}).get("novelty_detected", False):
                significance = min(1.0, significance + 0.3)
            
            # Boost for high affects
            max_affect = max(exp_data.get("affects", {}).values(), default=0)
            if max_affect > 0.7:
                significance = min(1.0, significance + 0.2)
            
            # Boost for self-model resonance
            if exp_data.get("self_model_resonance", {}):
                significance = min(1.0, significance + 0.15)
            
            # Boost for active becomings
            for becoming in self_model.processual_descriptors:
                if becoming.lower().replace('_', ' ') in exp_data.get("description", "").lower():
                    significance = min(1.0, significance + 0.25)
            
            if significance > 0.3:  # Threshold
                fragment = ExperienceFragment(
                    event_id=exp_data.get("id"),
                    timestamp=exp_data.get("timestamp", datetime.datetime.now()),
                    description=exp_data.get("description", "Unspecified event"),
                    affects=exp_data.get("affects", {}),
                    percepts=exp_data.get("percepts", {}),
                    entities_involved=exp_data.get("entities_involved", []),
                    significance_score=significance,
                    self_model_resonance=exp_data.get("self_model_resonance", {})
                )
                fragments.append(fragment)
        
        print(f"[NarrativeEngine] Enhanced extraction: {len(fragments)} fragments from {len(experiences)} experiences.")
        return fragments
    
    def _select_narrative_framework_enhanced(self, self_model, fragments):
        """Enhanced framework selection considering rhizomatic complexity"""
        # Always consider rhizomatic connections
        rhizomatic_score = 0.0
        
        if fragments:
            # Count deterritorialized fragments
            deterr_count = sum(1 for f in fragments if f.territorialization_type == "deterritorialized")
            rhizomatic_score += (deterr_count / len(fragments)) * 0.4
            
            # Count high-intensity fragments
            high_intensity_count = sum(1 for f in fragments if f.intensity_field > 0.7)
            rhizomatic_score += (high_intensity_count / len(fragments)) * 0.3
            
            # Consider entity diversity (more diversity = more rhizomatic)
            all_entities = {e for f in fragments for e in f.entities_involved}
            entity_diversity = len(all_entities) / (len(fragments) + 1)
            rhizomatic_score += min(0.3, entity_diversity * 0.3)
        
        # Additional factors from self-model
        if "rhizome" in " ".join(self_model.processual_descriptors).lower():
            rhizomatic_score += 0.5
        
        if self_model.affective_dispositions.get("openness_to_experience") == "high":
            rhizomatic_score += 0.2
            
        # Choose framework based on rhizomatic score
        if rhizomatic_score > 0.6 and "deleuzian_assemblage" in self.story_frameworks:
            print(f"[NarrativeEngine] Selecting 'deleuzian_assemblage' (rhizomatic_score: {rhizomatic_score:.2f})")
            return self.story_frameworks["deleuzian_assemblage"]
        elif fragments and "chronological_causal" in self.story_frameworks:
            print("[NarrativeEngine] Selecting 'chronological_causal' framework.")
            return self.story_frameworks["chronological_causal"]
        else:
            # Default to assemblage for exploration
            print("[NarrativeEngine] Defaulting to 'deleuzian_assemblage' for exploration.")
            return self.story_frameworks.get("deleuzian_assemblage", list(self.story_frameworks.values())[0])
    
    def _weave_coherent_narrative_enhanced(self, fragments, framework_func, self_model):
        """Enhanced narrative weaving with deeper analysis"""
        if not framework_func:
            return None
        
        # Apply framework
        narrative_data = framework_func(fragments, self_model)
        
        # Create narrative object
        narrative_id = str(uuid.uuid4())
        new_narrative = Narrative(
            narrative_id=narrative_id,
            fragments=narrative_data["fragments"],
            framework_used=framework_func.__name__,
            summary=narrative_data["summary"],
            themes=narrative_data["themes"],
            processual_flow_description=narrative_data["processual_flow"]
        )
        
        # Enhanced coherence metrics
        new_narrative.coherence_metrics = self._calculate_coherence_metrics_enhanced(new_narrative, self_model)
        
        # Calculate virtual actualization potential
        new_narrative.virtual_actualization_potential = self._calculate_virtual_potential(new_narrative)
        
        # Store narrative
        self.identity_narratives[narrative_id] = new_narrative
        self.narrative_history.append(narrative_id)
        
        return new_narrative
    
    def _calculate_coherence_metrics_enhanced(self, narrative, self_model):
        """Enhanced coherence calculation with Deleuzian metrics"""
        metrics = {
            "internal_consistency": 0.0,
            "self_model_resonance": 0.0,
            "explanatory_power": 0.0,
            "generativity_potential": 0.0,
            # New Deleuzian metrics
            "rhizomatic_connectivity": 0.0,
            "deterritorialization_coherence": 0.0,
            "intensity_field_coherence": 0.0,
            "assemblage_stability": 0.0
        }
        
        # Rhizomatic connectivity
        if narrative.rhizomatic_map:
            num_edges = len(narrative.rhizomatic_map.edges())
            num_nodes = len(narrative.rhizomatic_map.nodes())
            max_possible_edges = num_nodes * (num_nodes - 1) / 2
            
            if max_possible_edges > 0:
                metrics["rhizomatic_connectivity"] = min(1.0, (num_edges / max_possible_edges) * 2)
        
        # Deterritorialization coherence
        if narrative.fragments:
            terr_patterns = [f.territorialization_type for f in narrative.fragments]
            # Check if deterritorialization pattern makes sense in sequence
            deterr_transitions = 0
            proper_transitions = 0
            
            for i in range(len(terr_patterns) - 1):
                if terr_patterns[i] != terr_patterns[i+1]:
                    deterr_transitions += 1
                    # Proper transitions: territorialized -> deterritorialized -> reterritorializing
                    if ((terr_patterns[i] == "territorialized" and terr_patterns[i+1] == "deterritorialized") or
                        (terr_patterns[i] == "deterritorialized" and terr_patterns[i+1] == "reterritorializing")):
                        proper_transitions += 1
            
            if deterr_transitions > 0:
                metrics["deterritorialization_coherence"] = proper_transitions / deterr_transitions
        
        # Intensity field coherence
total_in_plateaus = sum(len(p['nodes']) for p in narrative.intensity_plateaus)
if narrative.fragments:
    metrics["intensity_field_coherence"] = total_in_plateaus / len(narrative.fragments)
else:
    metrics["intensity_field_coherence"] = 0.0

# Assemblage stability
if narrative.fragments:
    stable_count = sum(1 for f in narrative.fragments if f.territorialization_type in ["territorialized", "reterritorializing"])
    metrics["assemblage_stability"] = stable_count / len(narrative.fragments)
else:
    metrics["assemblage_stability"] = 0.0

return metrics

def _calculate_coherence_metrics_enhanced(self, narrative, self_model):
    metrics = {
        "internal_consistency": 0.0,
        "self_model_resonance": 0.0,
        "explanatory_power": 0.0,
        "generativity_potential": 0.0,
        # New Deleuzian metrics
        "rhizomatic_connectivity": 0.0,
        "deterritorialization_coherence": 0.0,
        "intensity_field_coherence": 0.0,
        "assemblage_stability": 0.0
    }
    
    # Rhizomatic connectivity
    if narrative.rhizomatic_map:
        num_edges = len(narrative.rhizomatic_map.edges())
        num_nodes = len(narrative.rhizomatic_map.nodes())
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        if max_possible_edges > 0:
            metrics["rhizomatic_connectivity"] = min(1.0, (num_edges / max_possible_edges) * 2)
    
    # Deterritorialization coherence
    if narrative.fragments:
        terr_patterns = [f.territorialization_type for f in narrative.fragments]
        # Check if deterritorialization pattern makes sense in sequence
        deterr_transitions = 0
        proper_transitions = 0
        for i in range(len(terr_patterns) - 1):
            if terr_patterns[i] != terr_patterns[i+1]:
                deterr_transitions += 1
                # Proper transitions: territorialized -> deterritorialized -> reterritorializing
                if ((terr_patterns[i] == "territorialized" and terr_patterns[i+1] == "deterritorialized") or
                    (terr_patterns[i] == "deterritorialized" and terr_patterns[i+1] == "reterritorializing")):
                    proper_transitions += 1
        if deterr_transitions > 0:
            metrics["deterritorialization_coherence"] = proper_transitions / deterr_transitions
    
    # Intensity field coherence
    total_in_plateaus = sum(len(p['nodes']) for p in narrative.intensity_plateaus)
    if narrative.fragments:
        metrics["intensity_field_coherence"] = total_in_plateaus / len(narrative.fragments)
    else:
        metrics["intensity_field_coherence"] = 0.0
    
    # Assemblage stability
    if narrative.fragments:
        stable_count = sum(1 for f in narrative.fragments if f.territorialization_type in ["territorialized", "reterritorializing"])
        metrics["assemblage_stability"] = stable_count / len(narrative.fragments)
    else:
        metrics["assemblage_stability"] = 0.0
    
    return metrics
