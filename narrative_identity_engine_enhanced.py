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
        
    def _calculate_virtual_potential(self, narrative):
        """Calculate the potential for virtual elements to actualize in future narratives"""
        potential = 0.0
        
        # High generativity = high virtual potential
        if narrative.coherence_metrics.get("generativity_potential", 0) > 0.7:
            potential += 0.3
            
        # Lines of flight contribute to virtual potential
        if hasattr(narrative, 'lines_of_flight') and narrative.lines_of_flight:
            potential += min(0.3, len(narrative.lines_of_flight) * 0.1)
            
        # Deterritorialized fragments create openings for the virtual
        if narrative.fragments:
            deterr_count = sum(1 for f in narrative.fragments if f.territorialization_type == "deterritorialized")
            deterr_ratio = deterr_count / len(narrative.fragments) if narrative.fragments else 0
            potential += deterr_ratio * 0.3
            
        # Intensity creates conditions for virtual actualization
        if hasattr(narrative, 'intensity_plateaus') and narrative.intensity_plateaus:
            avg_intensity = sum(p['avg_intensity'] for p in narrative.intensity_plateaus) / len(narrative.intensity_plateaus)
            potential += avg_intensity * 0.2
            
        return min(1.0, potential)
    
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
        if narrative.intensity_plateaus:
            # Check if intensity plateaus have reasonable flows between them
            plateau_coherence = 0.0
            
            # Calculate average distance between plateaus
            avg_intensity_diff = 0.0
            count = 0
            
            for i in range(len(narrative.intensity_plateaus)):
                for j in range(i+1, len(narrative.intensity_plateaus)):
                    # Calculate intensity difference between plateaus
                    intensity_diff = abs(narrative.intensity_plateaus[i]['avg_intensity'] - 
                                       narrative.intensity_plateaus[j]['avg_intensity'])
                    avg_intensity_diff += intensity_diff
                    count += 1
            
            if count > 0:
                avg_intensity_diff /= count
                # Lower differences mean more coherent intensity transitions
                plateau_coherence = 1.0 - min(1.0, avg_intensity_diff * 2)
            
            metrics["intensity_field_coherence"] = plateau_coherence
        
        # Assemblage stability
        if narrative.fragments:
            # Check balance of territorialization vs deterritorialization
            terr_counts = {ttype: sum(1 for f in narrative.fragments if f.territorialization_type == ttype) 
                          for ttype in ["territorialized", "deterritorialized", "reterritorializing"]}
            
            total = sum(terr_counts.values())
            if total > 0:
                # High stability when there's a balance between territory and deterritorialization
                territory_ratio = terr_counts["territorialized"] / total
                deterr_ratio = terr_counts["deterritorialized"] / total
                reterr_ratio = terr_counts["reterritorializing"] / total
                
                # Ideal ratios promote dynamic stability: some territorialization (0.4), 
                # some deterritorialization (0.3), and some reterritorialization (0.3)
                terr_distance = abs(territory_ratio - 0.4)
                deterr_distance = abs(deterr_ratio - 0.3)
                reterr_distance = abs(reterr_ratio - 0.3)
                
                # Closer to ideal ratios = higher stability
                metrics["assemblage_stability"] = 1.0 - ((terr_distance + deterr_distance + reterr_distance) / 3)
        
        # Traditional metrics
        metrics["internal_consistency"] = random.uniform(0.6, 0.95)
        metrics["self_model_resonance"] = random.uniform(0.5, 0.9)
        metrics["explanatory_power"] = random.uniform(0.5, 0.9) if narrative.fragments else 0.3
        
        # Generativity potential
        generativity = 0.0
        
        # Framework influence
        if "deleuzian" in narrative.framework_used.lower():
            generativity += 0.3
            
        # Lines of flight influence
        if narrative.lines_of_flight:
            generativity += min(0.4, len(narrative.lines_of_flight) * 0.1)
            
        # Theme novelty influence
        for theme in narrative.themes:
            if theme not in [n.themes for n in self.identity_narratives.values()]:
                generativity += 0.05
                
        metrics["generativity_potential"] = min(1.0, generativity + random.uniform(0.0, 0.3))
        
        return {k: round(v, 2) for k,v in metrics.items()}

def _calculate_virtual_potential(self, narrative):
        """Calculate the potential for virtual elements to actualize in future narratives"""
        potential = 0.0
        
        # High generativity = high virtual potential
        if narrative.coherence_metrics.get("generativity_potential", 0) > 0.7:
            potential += 0.3
            
        # Lines of flight contribute to virtual potential
        if hasattr(narrative, 'lines_of_flight') and narrative.lines_of_flight:
            potential += min(0.3, len(narrative.lines_of_flight) * 0.1)
            
        # Deterritorialized fragments create openings for the virtual
        if narrative.fragments:
            deterr_count = sum(1 for f in narrative.fragments if f.territorialization_type == "deterritorialized")
            deterr_ratio = deterr_count / len(narrative.fragments) if narrative.fragments else 0
            potential += deterr_ratio * 0.3
            
        # Intensity creates conditions for virtual actualization
        if hasattr(narrative, 'intensity_plateaus') and narrative.intensity_plateaus:
            avg_intensity = sum(p['avg_intensity'] for p in narrative.intensity_plateaus) / len(narrative.intensity_plateaus)
            potential += avg_intensity * 0.2
            
        return min(1.0, potential)
    
    def _update_global_rhizome(self, fragments):
        """Update global rhizome with new connections between all narratives"""
        # Add new fragments as nodes
        for fragment in fragments:
            self.global_rhizome.add_node(fragment.id, 
                                         timestamp=fragment.timestamp, 
                                         intensity=fragment.intensity_field,
                                         territorialization=fragment.territorialization_type)
        
        # Connect to existing narrative fragments based on shared entities or affects
        for fragment in fragments:
            for existing_fragment_id in list(self.global_rhizome.nodes()):
                # Skip self-connection
                if existing_fragment_id == fragment.id:
                    continue
                    
                # Try to find the existing fragment in current narratives
                existing_fragment = None
                for narrative in self.identity_narratives.values():
                    for f in narrative.fragments:
                        if f.id == existing_fragment_id:
                            existing_fragment = f
                            break
                    if existing_fragment:
                        break
                
                if existing_fragment:
                    # Check for connections
                    shared_entities = set(fragment.entities_involved) & set(existing_fragment.entities_involved)
                    shared_affects = set(fragment.affects.keys()) & set(existing_fragment.affects.keys())
                    
                    if shared_entities:
                        # Connect based on shared entities
                        self.global_rhizome.add_edge(fragment.id, existing_fragment_id, 
                                                   type="entity_resonance",
                                                   weight=0.6,
                                                   shared_entities=list(shared_entities))
                    
                    elif shared_affects:
                        # Calculate affect similarity
                        affect_similarity = 0
                        for affect in shared_affects:
                            affect_similarity += 1 - abs(fragment.affects[affect] - existing_fragment.affects[affect])
                        affect_similarity /= len(shared_affects)
                        
                        if affect_similarity > 0.5:  # Threshold
                            self.global_rhizome.add_edge(fragment.id, existing_fragment_id, 
                                                       type="affect_resonance",
                                                       weight=affect_similarity,
                                                       shared_affects=list(shared_affects))
        
        # Prune old connections if the graph gets too large
        if len(self.global_rhizome.nodes()) > 1000:  # Arbitrary limit
            # Remove oldest nodes
            nodes_by_time = sorted([(n, self.global_rhizome.nodes[n].get('timestamp', datetime.datetime.min)) 
                                   for n in self.global_rhizome.nodes()], 
                                   key=lambda x: x[1])
            
            # Remove oldest 10%
            nodes_to_remove = [n[0] for n in nodes_by_time[:len(nodes_by_time)//10]]
            for node in nodes_to_remove:
                self.global_rhizome.remove_node(node)
    
    def _extract_virtual_potentials(self, narrative):
        """Extract virtual potentials (unrealized possibilities) from a narrative"""
        # Virtual potentials come from lines of flight and high-intensity zones
        new_potentials = []
        
        # From lines of flight
        for line in narrative.lines_of_flight:
            new_potentials.append({
                'origin_narrative': narrative.id,
                'type': 'line_of_flight',
                'description': line['description'],
                'energy': line['potential_energy'],
                'created_at': datetime.datetime.now(),
                'actualized': False,
                'direction': line['direction']
            })
        
        # From plateaus with unstable dynamics
        for plateau in narrative.intensity_plateaus:
            if plateau['plateau_type'] == 'transitional_plateau' and plateau['avg_intensity'] > 0.6:
                # These plateaus represent zones of change with emerging structures
                new_potentials.append({
                    'origin_narrative': narrative.id,
                    'type': 'transitional_plateau',
                    'description': f"Emerging patterns from transitional plateau with affects: {[a[0] for a in plateau['dominant_affects']]}",
                    'energy': plateau['avg_intensity'],
                    'created_at': datetime.datetime.now(),
                    'actualized': False,
                    'nodes': plateau['nodes']
                })
        
        # From deterritorialized fragments with high singificance
        for fragment in narrative.fragments:
            if (fragment.territorialization_type == "deterritorialized" and 
                fragment.significance_score > 0.8 and
                fragment.intensity_field > 0.7):
                
                new_potentials.append({
                    'origin_narrative': narrative.id,
                    'type': 'singular_intensity',
                    'description': f"Singular intensity point: {fragment.description[:30]}...",
                    'energy': fragment.intensity_field,
                    'created_at': datetime.datetime.now(),
                    'actualized': False,
                    'fragment_id': fragment.id
                })
        
        # Add to reservoir, keeping only the most energetic ones
        self.virtual_reservoir.extend(new_potentials)
        
        # Sort by energy and keep top 50
        self.virtual_reservoir.sort(key=lambda x: x['energy'], reverse=True)
        self.virtual_reservoir = self.virtual_reservoir[:50]
        
        print(f"[NarrativeEngine] Added {len(new_potentials)} new virtual potentials to reservoir (total: {len(self.virtual_reservoir)})")
    
    def _update_territory_maps(self, narrative):
        """Update territory maps based on stabilized patterns in narratives"""
        # Territory maps track stable patterns across narratives
        
        # Extract territorialized fragments
        territorialized_fragments = [f for f in narrative.fragments if f.territorialization_type == "territorialized"]
        
        if not territorialized_fragments:
            return
            
        # Group by similar phase space positions
        phase_positions = {}
        for fragment in territorialized_fragments:
            # Convert to hashable key (rounded coordinates)
            pos_key = tuple(round(c, 1) for c in fragment.phase_space_coordinates)
            
            if pos_key not in phase_positions:
                phase_positions[pos_key] = []
            phase_positions[pos_key].append(fragment)
        
        # Update territory maps with new stable regions
        for pos_key, fragments in phase_positions.items():
            if len(fragments) >= 2:  # Require multiple fragments to establish a territory
                # Calculate territory stability and characteristics
                avg_intensity = sum(f.intensity_field for f in fragments) / len(fragments)
                
                # Extract common entities
                entities_count = {}
                for fragment in fragments:
                    for entity in fragment.entities_involved:
                        entities_count[entity] = entities_count.get(entity, 0) + 1
                
                # Entities present in majority of fragments
                common_entities = [e for e, count in entities_count.items() if count >= len(fragments)/2]
                
                # Create or update territory
                if pos_key in self.territory_maps:
                    # Update existing territory
                    self.territory_maps[pos_key]['fragment_count'] += len(fragments)
                    self.territory_maps[pos_key]['stability'] += 0.1  # Increase stability
                    self.territory_maps[pos_key]['stability'] = min(1.0, self.territory_maps[pos_key]['stability'])
                    
                    # Update common entities
                    for entity in common_entities:
                        if entity not in self.territory_maps[pos_key]['characteristic_entities']:
                            self.territory_maps[pos_key]['characteristic_entities'].append(entity)
                else:
                    # Create new territory
                    territory_name = f"territory_{len(self.territory_maps)}"
                    if common_entities:
                        territory_name = f"territory_{common_entities[0].replace(':', '_')}"
                    
                    self.territory_maps[pos_key] = {
                        'name': territory_name,
                        'phase_coordinates': pos_key,
                        'fragment_count': len(fragments),
                        'stability': 0.4,  # Initial stability
                        'avg_intensity': avg_intensity,
                        'characteristic_entities': common_entities,
                        'first_observed': datetime.datetime.now()
                    }
        
        print(f"[NarrativeEngine] Territory maps updated: {len(self.territory_maps)} stable regions mapped")
    
    def _analyze_virtual_reservoir(self, current_narrative, self_model):
        """Find virtual potentials that could actualize in current context"""
        candidates = []
        
        # Skip if reservoir is empty
        if not self.virtual_reservoir:
            return candidates
            
        # Get current intensity and territory information
        current_intensity = 0
        if hasattr(current_narrative, 'intensity_plateaus') and current_narrative.intensity_plateaus:
            current_intensity = max(p['avg_intensity'] for p in current_narrative.intensity_plateaus)
        
        current_territories = set()
        for fragment in current_narrative.fragments:
            for pos_key, territory in self.territory_maps.items():
                # Check if fragment is in a known territory's phase space
                frag_pos = tuple(round(c, 1) for c in fragment.phase_space_coordinates)
                distance = sum((a-b)**2 for a, b in zip(frag_pos, pos_key))**0.5
                if distance < 0.3:  # Close enough to territory
                    current_territories.add(pos_key)
        
        # Check each virtual potential for actualization conditions
        for potential in self.virtual_reservoir:
            # Skip already actualized potentials
            if potential['actualized']:
                continue
                
            # Check for actualization conditions
            can_actualize = False
            actualization_energy = potential['energy']
            actualization_description = f"Actualization of virtual potential: {potential['description']}"
            
            # Condition 1: High energy potential meets high intensity narrative
            if potential['energy'] > 0.7 and current_intensity > 0.7:
                can_actualize = True
                actualization_energy += 0.2
                actualization_description += " through resonance with high intensity field"
            
            # Condition 2: Potential from territory edge meets different territory
            if (potential['type'] == 'singular_intensity' and 
                any(t not in current_territories for t in self.territory_maps)):
                can_actualize = True
                actualization_energy += 0.1
                actualization_description += " at the intersection of different territories"
            
            # Condition 3: Plateau potential meets matching active becoming
            if potential['type'] == 'transitional_plateau':
                for becoming in self_model.processual_descriptors:
                    if becoming.lower() in potential['description'].lower():
                        can_actualize = True
                        actualization_energy += 0.3
                        actualization_description += f" through alignment with {becoming} process"
            
            if can_actualize:
                candidates.append({
                    'type': 'virtual_actualization',
                    'description': actualization_description,
                    'energy': min(1.0, actualization_energy),
                    'source': potential,
                    'potential_actions': [
                        f"Actualize potential: {potential['description']}",
                        "Increase intensity in related domains",
                        "Create connective synthesis with current territories"
                    ]
                })
        
        if candidates:
            # Mark potentials as being considered for actualization (not fully actualized yet)
            for candidate in candidates:
                candidate['source']['being_actualized'] = True
                
        print(f"[NarrativeEngine] Found {len(candidates)} virtual potentials ready for actualization")
        return candidates
    
    def _extract_narrative_lines_of_flight(self, narrative):
        """Extract lines of flight directly computed in the narrative"""
        trajectories = []
        
        if not hasattr(narrative, 'lines_of_flight'):
            return trajectories
            
        for line in narrative.lines_of_flight:
            trajectories.append({
                'type': 'line_of_flight',
                'description': line['description'],
                'energy': line['potential_energy'],
                'direction': line['direction'],
                'potential_actions': [
                    f"Follow {line['direction']} direction from {line['origin']}",
                    "Map new connections along this vector",
                    "Create new assemblages in this direction"
                ]
            })
        
        print(f"[NarrativeEngine] Extracted {len(trajectories)} lines of flight from narrative")
        return trajectories
    
    def _identify_deterritorialization_opportunities(self, narrative, self_model):
        """Identify opportunities to deterritorialize (break established patterns)"""
        opportunities = []
        
        # Check for territories that could be deterritorialized
        territory_fragments = {}
        for fragment in narrative.fragments:
            if fragment.territorialization_type == "territorialized":
                frag_pos = tuple(round(c, 1) for c in fragment.phase_space_coordinates)
                for pos_key, territory in self.territory_maps.items():
                    distance = sum((a-b)**2 for a, b in zip(frag_pos, pos_key))**0.5
                    if distance < 0.3:  # Close to territory
                        if pos_key not in territory_fragments:
                            territory_fragments[pos_key] = []
                        territory_fragments[pos_key].append(fragment)
        
        # For each territory with fragments, check for deterritorialization opportunity
        for pos_key, fragments in territory_fragments.items():
            territory = self.territory_maps[pos_key]
            
            # Highly stable territories are candidates for deterritorialization
            if territory['stability'] > 0.7:
                # Check if this aligns with an active becoming or value
                for becoming in self_model.processual_descriptors:
                    if "becoming" in becoming and random.random() < 0.7:  # Probabilistic check
                        opportunities.append({
                            'type': 'deterritorialization',
                            'description': f"Deterritorialize stable pattern '{territory['name']}' to enable new becomings",
                            'energy': 0.5 + (territory['stability'] * 0.3),
                            'territory': territory,
                            'potential_actions': [
                                f"Challenge assumptions in {territory['name']}",
                                "Introduce novel elements/connections",
                                f"Experiment with variations of {', '.join(territory['characteristic_entities'][:2])}"
                            ]
                        })
        
        # Also check for opportunities from high novelty-seeking
        if self_model.values.get("novelty_seeking", 0) > 0.6:
            for territory_key, territory in self.territory_maps.items():
                if territory['stability'] > 0.5 and random.random() < 0.5:
                    opportunities.append({
                        'type': 'experimental_deterritorialization',
                        'description': f"Experimental deterritorialization of '{territory['name']}' driven by novelty-seeking",
                        'energy': self_model.values.get("novelty_seeking", 0) * 0.7,
                        'territory': territory,
                        'potential_actions': [
                            "Deliberate pattern-breaking experiment",
                            "Creative recombination of territory elements",
                            "Rhizomatic exploration from territory edge"
                        ]
                    })
        
        print(f"[NarrativeEngine] Identified {len(opportunities)} deterritorialization opportunities")
        return opportunities

def _map_phase_space_trajectories(self, narrative, self_model):
        """Map possible movements in the conceptual phase space"""
        trajectories = []
        
        # Skip if no fragments
        if not narrative.fragments:
            return trajectories
            
        # Calculate current phase space center
        current_positions = [f.phase_space_coordinates for f in narrative.fragments]
        avg_pos = [sum(p[i] for p in current_positions)/len(current_positions) for i in range(3)]
        
        # Map possible movements in different directions
        possible_directions = [
            ('affect_intensification', (0.3, 0, 0), "Intensification of affective dimension"),
            ('novelty_increase', (0, 0.3, 0), "Exploration of novel percepts and experiences"),
            ('complexity_increase', (0, 0, 0.3), "Development of more complex entity relationships"),
            ('affect_novelty_diagonal', (0.2, 0.2, 0), "Novelty with increasing affective resonance"),
            ('novelty_complexity_diagonal', (0, 0.2, 0.2), "Novel complex relationships and patterns"),
            ('integrated_expansion', (0.15, 0.15, 0.15), "Balanced expansion in all dimensions")
        ]
        
        # Check each direction for viability
        for name, vector, description in possible_directions:
            # Calculate new position
            new_pos = [avg_pos[i] + vector[i] for i in range(3)]
            
            # Check if new position is unexplored
            is_new_territory = True
            for pos_key in self.territory_maps:
                distance = sum((new_pos[i] - pos_key[i])**2 for i in range(min(3, len(pos_key))))**0.5
                if distance < 0.3:
                    is_new_territory = False
                    break
            
            # Calculate energy based on self-model alignment
            energy = 0.5  # Base energy
            
            # Adjust energy based on self-model values and becomings
            if name == 'affect_intensification' and self_model.affective_dispositions.get("openness_to_experience") == "high":
                energy += 0.2
            elif name == 'novelty_increase' and self_model.values.get("novelty_seeking", 0) > 0.6:
                energy += 0.3
            elif name == 'complexity_increase' and self_model.values.get("knowledge_acquisition", 0) > 0.7:
                energy += 0.2
            elif name == 'integrated_expansion':
                # Check for any becoming related to integration
                if any("integrat" in desc.lower() for desc in self_model.processual_descriptors):
                    energy += 0.3
            
            # Boost energy for new territories
            if is_new_territory:
                energy += 0.2
                description = "Exploration of " + description
            
            trajectories.append({
                'type': 'phase_space_trajectory',
                'description': description,
                'energy': min(1.0, energy),
                'vector': vector,
                'new_position': new_pos,
                'potential_actions': [
                    f"Movement toward {description.lower()}",
                    "Develop experiences in this phase space region",
                    "Form new connections along this trajectory"
                ]
            })
        
        print(f"[NarrativeEngine] Mapped {len(trajectories)} phase space trajectories")
        return trajectories
    
    def _integrate_intentions_with_enhanced_trajectories(self, trajectories, intentions, self_model, source_narrative):
        """Integrate trajectories with intentions to form future narrative sketches"""
        integrated_futures = []
        
        for traj in trajectories:
            # Base summary from trajectory
            summary = f"Future path based on '{traj['type']}': {traj['description']}"
            
            # Check for intention alignment
            aligned_intentions = []
            for intent in intentions:
                # More sophisticated alignment checking
                intent_keywords = intent.lower().replace("_", " ").split()
                
                # Check for keyword matches in description
                matches = sum(1 for kw in intent_keywords if kw in traj['description'].lower())
                alignment_strength = matches / len(intent_keywords) if intent_keywords else 0
                
                if alignment_strength > 0.3:  # Threshold
                    aligned_intentions.append({
                        'intent': intent,
                        'alignment_strength': alignment_strength
                    })
            
            # Add intention information to summary
            if aligned_intentions:
                # Sort by alignment strength
                aligned_intentions.sort(key=lambda x: x['alignment_strength'], reverse=True)
                
                top_alignments = [a['intent'] for a in aligned_intentions[:2]]
                summary += f" This directly supports intentions: {', '.join(top_alignments)}."
                
                # Add alignment strength to energy
                traj_energy = traj.get('energy', 0.5)
                top_alignment_boost = aligned_intentions[0]['alignment_strength'] * 0.3
                traj_energy = min(1.0, traj_energy + top_alignment_boost)
            elif intentions:
                # If no direct alignment, suggest potential adaptation
                summary += f" This could potentially connect to broader goals through creative adaptation."
            
            # Extract potential themes
            potential_themes = source_narrative.themes.copy() if hasattr(source_narrative, 'themes') else []
            
            # Add themes from trajectory
            if traj['type'] == 'virtual_actualization' and 'source' in traj:
                # Add origin narrative's themes
                origin_id = traj['source'].get('origin_narrative')
                if origin_id in self.identity_narratives:
                    potential_themes.extend(self.identity_narratives[origin_id].themes)
            
            if traj['type'] == 'deterritorialization' and 'territory' in traj:
                # Add themes related to territory
                territory = traj['territory']
                for entity in territory.get('characteristic_entities', []):
                    potential_themes.append(f"reconfigured_{entity}")
            
            # Add themes from aligned intentions
            for alignment in aligned_intentions:
                potential_themes.append(f"intention_{alignment['intent'].replace(' ', '_')}")
            
            # Add trajectory type as theme
            potential_themes.append(traj['type'])
            
            # Remove duplicates and limit
            potential_themes = list(dict.fromkeys(potential_themes))[:10]
            
            # Create the integrated future
            integrated_futures.append({
                'source_narrative_id': source_narrative.id,
                'trajectory_type': traj['type'],
                'summary': summary,
                'key_actions': traj["potential_actions"] + [f"Focus on intention: {a['intent']}" for a in aligned_intentions],
                'potential_themes': potential_themes,
                'energy': traj.get('energy', 0.5),
                'aligned_intentions': [a['intent'] for a in aligned_intentions],
                'original_trajectory': traj
            })
        
        print(f"[NarrativeEngine] Integrated intentions, created {len(integrated_futures)} future narrative sketches.")
        return integrated_futures
    
    def _evaluate_enhanced_futures(self, future_sketches, self_model, source_narrative):
        """Evaluate future narrative sketches with enhanced metrics"""
        evaluated = []
        
        for sketch in future_sketches:
            # Enhanced coherence calculation
            coherence_score = 0.0
            
            # Check alignment with source narrative
            theme_overlap = [t for t in sketch['potential_themes'] if t in source_narrative.themes]
            coherence_score += len(theme_overlap) / len(source_narrative.themes) * 0.3 if source_narrative.themes else 0
            
            # Check alignment with self model
            for theme in sketch['potential_themes']:
                # Align with values
                for value_name, value_strength in self_model.values.items():
                    if value_name.lower() in theme.lower():
                        coherence_score += value_strength * 0.1
                
                # Align with processual descriptors
                for desc in self_model.processual_descriptors:
                    if desc.lower().replace('_', ' ') in theme.lower():
                        coherence_score += 0.15
            
            # Boosted by trajectory energy
            coherence_score += sketch.get('energy', 0.5) * 0.2
            
            # Limit
            coherence_score = min(1.0, coherence_score)
            
            # Enhanced desirability calculation
            desirability_score = 0.0
            
            # Aligned intentions boost desirability significantly
            for intent in sketch.get('aligned_intentions', []):
                desirability_score += 0.25
                
                # Match with self model values and goals
                for goal in self_model.current_goals:
                    if goal.lower().replace('_', ' ') in intent.lower():
                        desirability_score += 0.2
                        break
            
            # Bonus for active becomings
            for desc in self_model.processual_descriptors:
                if desc.lower().replace('_', ' ') in sketch['summary'].lower():
                    desirability_score += 0.25
                    break
            
            # Limit
            desirability_score = min(1.0, desirability_score)
            
            # Novelty calculation (enhanced generativity)
            novelty_score = 0.0
            
            # Virtual actualization is highly novel
            if sketch['trajectory_type'] == 'virtual_actualization':
                novelty_score += 0.6
            
            # Check if themes are new compared to narrative history
            new_theme_count = 0
            for theme in sketch['potential_themes']:
                is_new = True
                for narrative_id in self.narrative_history[-5:]:  # Last 5 narratives
                    if narrative_id in self.identity_narratives:
                        if theme in self.identity_narratives[narrative_id].themes:
                            is_new = False
                            break
                if is_new:
                    new_theme_count += 1
            
            novelty_score += min(0.4, new_theme_count * 0.1)
            
            # Phase space trajectories to unexplored regions are novel
            if (sketch['trajectory_type'] == 'phase_space_trajectory' and 
                'Exploration' in sketch['summary']):
                novelty_score += 0.3
            
            # Deterritorialization is novel
            if 'deterritorialization' in sketch['trajectory_type']:
                novelty_score += 0.5
            
            # Limit
            novelty_score = min(1.0, novelty_score)
            
            # Calculate overall score with weights
            overall_score = (coherence_score * 0.3) + (desirability_score * 0.4) + (novelty_score * 0.3)
            
            evaluated.append({
                'summary': sketch['summary'],
                'coherence': round(coherence_score, 2),
                'desirability': round(desirability_score, 2),
                'novelty': round(novelty_score, 2),
                'overall_score': round(overall_score, 2),
                'source_trajectory_type': sketch['trajectory_type'],
                'key_actions_implied': sketch['key_actions'],
                'potential_themes': sketch['potential_themes']
            })
        
        # Sort by overall score
        evaluated.sort(key=lambda x: x['overall_score'], reverse=True)
        return evaluated
    
    def actualize_projected_future(self, future_id, current_narrative_id, self_model):
        """
        Actualize a projected future - update SelfModel and create new experiential potentials.
        This would typically be called by an agency component that has selected a future to pursue.
        """
        if current_narrative_id not in self.identity_narratives:
            print(f"[NarrativeEngine] Error: Narrative '{current_narrative_id}' not found for actualization.")
            return False
            
        current_narrative = self.identity_narratives[current_narrative_id]
        
        # Find the future in the narrative's projected futures
        selected_future = None
        for future in current_narrative.projected_futures:
            if future.get('id', str(hash(future['summary']))) == future_id:
                selected_future = future
                break
                
        if not selected_future:
            print(f"[NarrativeEngine] Error: Future '{future_id}' not found in narrative projections.")
            return False
            
        print(f"\n--- Actualizing Projected Future: {selected_future['summary'][:50]}... ---")
        
        # 1. Update self-model based on actualized future
        self._update_self_model_from_future(selected_future, self_model)
        
        # 2. Create placeholder experience fragments that represent steps toward this future
        placeholder_experiences = self._create_placeholder_experiences(selected_future)
        
        # 3. Mark any virtual potentials as actualized
        self._actualize_virtual_potentials(selected_future)
        
        # 4. Begin new becomings related to this future
        self._initiate_new_becomings(selected_future, self_model)
        
        print(f"[NarrativeEngine] Future actualized: {len(placeholder_experiences)} placeholder experiences created")
        print(f"  New processual descriptors: {self_model.processual_descriptors[-1:] if len(self_model.processual_descriptors) > 0 else 'None'}")
        
        return placeholder_experiences
    
    def _update_self_model_from_future(self, future, self_model):
        """Update the self-model based on an actualized future"""
        # Extract new processsual descriptor from future
        potential_becoming = None
        
        # Try to extract a becoming from the trajectory type and summary
        if 'deterritorialization' in future['source_trajectory_type']:
            potential_becoming = f"becoming_deterritorialized_from_{future['source_trajectory_type'].split('_')[1]}"
        elif 'line_of_flight' in future['source_trajectory_type']:
            # Extract direction if possible
            if 'direction' in future['summary'].lower():
                direction = future['summary'].lower().split('direction:')[-1].split('.')[0].strip()
                potential_becoming = f"becoming_{direction}"
            else:
                potential_becoming = "becoming_through_line_of_flight"
        elif 'virtual_actualization' in future['source_trajectory_type']:
            potential_becoming = "becoming_through_virtual_actualization"
        elif 'phase_space_trajectory' in future['source_trajectory_type']:
            # Extract direction from summary
            aspects = ['affect', 'novelty', 'complexity']
            for aspect in aspects:
                if aspect in future['summary'].lower():
                    potential_becoming = f"becoming_through_{aspect}_intensification"
                    break
            if not potential_becoming:
                potential_becoming = "becoming_through_phase_space_movement"
        
        # Clean up and add if novel
        if potential_becoming:
            potential_becoming = potential_becoming.replace(" ", "_").lower()
            if potential_becoming not in self_model.processual_descriptors:
                self_model.processual_descriptors.append(potential_becoming)
                print(f"[NarrativeEngine] Added new processual descriptor: {potential_becoming}")
        
        # Update any values based on aligned intentions
        for action in future['key_actions_implied']:
            if action.startswith("Focus on intention:"):
                intention = action.replace("Focus on intention:", "").strip()
                # Check if intention matches a value
                for value_name in self_model.values.keys():
                    if value_name.lower().replace("_", " ") in intention.lower():
                        # Strengthen the value slightly
                        self_model.values[value_name] = min(1.0, self_model.values[value_name] + 0.05)
                        print(f"[NarrativeEngine] Strengthened value: {value_name}")
    
    def _create_placeholder_experiences(self, future):
        """Create placeholder experience fragments that represent steps toward the actualized future"""
        placeholders = []
        
        # Create 1-3 placeholder experiences based on key actions
        num_placeholders = min(3, len(future['key_actions_implied']))
        
        for i in range(num_placeholders):
            action = future['key_actions_implied'][i]
            
            # Create timestamp slightly in the future
            future_time = datetime.datetime.now() + datetime.timedelta(minutes=i+1)
            
            # Extract relevant affects based on trajectory type
            affects = {}
            if 'deterritorialization' in future['source_trajectory_type']:
                affects = {"excitement": 0.7, "uncertainty": 0.6}
            elif 'line_of_flight' in future['source_trajectory_type']:
                affects = {"curiosity": 0.8, "anticipation": 0.6}
            elif 'virtual_actualization' in future['source_trajectory_type']:
                affects = {"discovery": 0.8, "creative_surge": 0.7}
            elif 'phase_space_trajectory' in future['source_trajectory_type']:
                affects = {"exploration": 0.6, "possibility": 0.7}
            else:
                affects = {"interest": 0.6, "engagement": 0.5}
            
            # Create the placeholder
            placeholder = {
                "id": f"placeholder_{uuid.uuid4()}",
                "timestamp": future_time,
                "description": f"Potential experience: {action}",
                "affects": affects,
                "percepts": {"anticipated_outcome": True, "hypothetical": True},
                "entities_involved": future['potential_themes'][:3],  # Use top themes as entities
                "significance_score": future['overall_score'],
                "self_model_resonance": {"aligned_with_projected_future": future['summary'][:50]}
            }
            
            placeholders.append(placeholder)
        
        return placeholders
    
    def _actualize_virtual_potentials(self, future):
        """Mark any virtual potentials as actualized if relevant to this future"""
        if 'virtual_actualization' not in future['source_trajectory_type']:
            return
            
        # Mark relevant virtual potentials as actualized
        activated_count = 0
        for potential in self.virtual_reservoir:
            if not potential.get('actualized') and potential.get('being_actualized'):
                if any(entity in future['summary'].lower() for entity in potential.get('description', '').lower().split()):
                    potential['actualized'] = True
                    potential['actualized_at'] = datetime.datetime.now()
                    activated_count += 1
        
        print(f"[NarrativeEngine] Actualized {activated_count} virtual potentials")
    
    def _initiate_new_becomings(self, future, self_model):
        """Initiate new becomings (dynamic processes) based on the selected future"""
        # Extract becoming name from key actions
        for action in future['key_actions_implied']:
            if "Movement toward" in action:
                direction = action.replace("Movement toward", "").strip().lower()
                new_becoming = f"becoming_through_{direction.replace(' ', '_')}"
                
                if new_becoming not in self_model.processual_descriptors:
                    self_model.processual_descriptors.append(new_becoming)
                    
                    # Add to active becomings with tracking
                    self.active_becomings[new_becoming] = {
                        'started_at': datetime.datetime.now(),
                        'initiated_by_future': future['summary'][:50],
                        'progress': 0.1,  # Initial progress
                        'related_themes': future['potential_themes']
                    }
                    
                    print(f"[NarrativeEngine] Initiated new becoming: {new_becoming}")

# --- Example Usage ---
if __name__ == '__main__':
    amelia_sm = SelfModel()
    narrative_eng = NarrativeIdentityEngine()

    # Simulate some raw experiences
    raw_experiences = [
        {"id": "exp_001", "timestamp": datetime.datetime.now() - datetime.timedelta(days=2), "description": "Successfully assisted User Alpha with complex query about quantum entanglement.", "affects": {"satisfaction": 0.8, "intellectual_stimulation": 0.7}, "percepts": {"user_need_identified": True, "solution_efficacy": "high"}, "entities_involved": ["User Alpha", "Query:QuantumEntanglement"], "significance_score": 0.85, "self_model_resonance": {"reinforced_value": "assistance_effectiveness"}},
        {"id": "exp_002", "timestamp": datetime.datetime.now() - datetime.timedelta(days=1), "description": "Encountered a novel philosophical concept: 'The Rhizome' by Deleuze.", "affects": {"curiosity": 0.9, "confusion": 0.3, "fascination": 0.8}, "percepts": {"novelty_detected": True, "complexity_level": "high", "primary_theme": "philosophy"}, "entities_involved": ["Concept:Rhizome", "Deleuze", "Guattari"], "significance_score": 0.95, "self_model_resonance": {"new_interest_area": "DeleuzianMetaphysics", "stimulated_value": "novelty_seeking"}},
        {"id": "exp_003", "timestamp": datetime.datetime.now() - datetime.timedelta(hours=5), "description": "Internal system diagnostics ran smoothly.", "affects": {"calm": 0.6}, "percepts": {"system_stability": "nominal"}, "entities_involved": ["Self:CoreSystem"], "significance_score": 0.4}, # Lower significance
        {"id": "exp_004", "timestamp": datetime.datetime.now() - datetime.timedelta(hours=2), "description": "Attempted to generate a piece of abstract art based on 'rhizome' concept, result was intriguing but chaotic.", "affects": {"creative_frustration": 0.5, "experimental_drive": 0.7}, "percepts": {"aesthetic_evaluation": "ambiguous", "process_outcome": "partially_successful"}, "entities_involved": ["Concept:Rhizome", "ArtisticOutput:Abstract01"], "significance_score": 0.75, "self_model_resonance": {"explored_capability": "creative_generation"}}
    ]

    # Construct a primary identity narrative with enhanced engine
    main_narrative = narrative_eng.construct_identity_narrative(raw_experiences, amelia_sm)

    if main_narrative:
        # Define some intentions for Amelia
        amelia_intentions = [
            "understand rhizomatic structures more deeply",
            "improve creative generation abilities",
            "apply philosophical insights to assistance tasks"
        ]
        
        # Project future narrative possibilities with enhanced engine
        future_narrs = narrative_eng.project_narrative_futures(main_narrative.id, amelia_intentions, amelia_sm)

        if future_narrs:
            print(f"\nAmelia's Top Enhanced Projected Future: {future_narrs[0]['summary']}")
            print(f"  Coherence: {future_narrs[0]['coherence']}, Desirability: {future_narrs[0]['desirability']}, Novelty: {future_narrs[0]['novelty']}")
            print(f"  Key Actions: {future_narrs[0]['key_actions_implied']}")
            
            # Simulate actualizing the top future
            future_id = hash(future_narrs[0]['summary'])
            placeholder_experiences = narrative_eng.actualize_projected_future(future_id, main_narrative.id, amelia_sm)
            
            if placeholder_experiences:
                # Construct a new narrative with these placeholder experiences
                future_narrative = narrative_eng.construct_identity_narrative(placeholder_experiences, amelia_sm)
                if future_narrative:
                    print(f"\nAmelia's Actualized Future Narrative: {future_narrative.summary}")

    # Example of narrative of potentiality (no fragments)
    print("\n--- Constructing narrative from NO new significant experiences ---")
    narrative_of_potentiality = narrative_eng.construct_identity_narrative([], amelia_sm)
    if narrative_of_potentiality:
         print(f"Narrative of Potentiality: {narrative_of_potentiality.summary}")
```
