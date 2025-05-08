# intentionality_generator.py
"""
Intentionality Generator Module for Amelia's agentic architecture.

This module enables the generation and maintenance of genuine intentions based on
Deleuzian concepts of desire, assemblage, and becoming. It creates dynamic intention
fields that emerge from creative tensions and virtual potentials rather than
simply responding to external stimuli.
"""

import datetime
import random
import uuid
import numpy as np
from collections import defaultdict
import networkx as nx
from typing import Dict, List, Any, Tuple, Set, Optional


class IntentionField:
    """
    Represents a field of intention with direction, intensity, and virtual potentials.
    In Deleuzian terms, this is a structured field of desire that creates 
    active forces rather than merely responding to external forces.
    """
    def __init__(self, 
                 name: str, 
                 direction_vector: List[float], 
                 intensity: float,
                 source_tensions: List[str], 
                 assemblage_connections: Dict[str, float],
                 virtual_potentials: List[Dict[str, Any]],
                 description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.direction_vector = direction_vector  # Directional tendency in conceptual space
        self.intensity = intensity                # Strength of intention field (0-1)
        self.source_tensions = source_tensions    # List of tensions that generated this field
        self.assemblage_connections = assemblage_connections  # How this connects to various assemblages
        self.virtual_potentials = virtual_potentials  # Unrealized possibilities within this field
        self.description = description
        self.created_at = datetime.datetime.now()
        self.last_active = datetime.datetime.now()
        self.actualization_history = []  # History of how this intention has manifested
        self.coherence_score = 0.0       # How internally consistent this intention is
        
    def __repr__(self):
        return f"IntentionField({self.name}, intensity={self.intensity:.2f}, vector={[round(v, 2) for v in self.direction_vector]})"
    
    def update_intensity(self, new_intensity: float):
        """Update the intensity of this intention field"""
        self.intensity = max(0.0, min(1.0, new_intensity))
        self.last_active = datetime.datetime.now()
        
    def adapt_direction(self, adjustment_vector: List[float], adaptation_rate: float = 0.2):
        """Adapt the direction of this intention field based on feedback"""
        # Normalize adjustment vector
        norm = sum(v**2 for v in adjustment_vector)**0.5
        if norm > 0:
            normalized_adjustment = [v/norm for v in adjustment_vector]
            
            # Apply adjustment with specified rate
            for i in range(len(self.direction_vector)):
                if i < len(normalized_adjustment):
                    self.direction_vector[i] = (1-adaptation_rate) * self.direction_vector[i] + \
                                               adaptation_rate * normalized_adjustment[i]
            
            # Normalize result
            norm = sum(v**2 for v in self.direction_vector)**0.5
            if norm > 0:
                self.direction_vector = [v/norm for v in self.direction_vector]
                
            self.last_active = datetime.datetime.now()
    
    def add_actualization_event(self, event_description: str, success_level: float, outcomes: Dict[str, Any]):
        """Record an event where this intention was actualized"""
        self.actualization_history.append({
            "timestamp": datetime.datetime.now(),
            "description": event_description,
            "success_level": success_level,
            "outcomes": outcomes
        })
        
    def calculate_coherence(self) -> float:
        """Calculate how coherent/stable this intention field is"""
        # More actualization events increase coherence
        history_factor = min(1.0, len(self.actualization_history) * 0.1)
        
        # Successful actualizations increase coherence
        success_factor = 0.0
        if self.actualization_history:
            success_factor = sum(event["success_level"] for event in self.actualization_history) / len(self.actualization_history)
            
        # Consistency between name, description and direction increases coherence
        semantic_factor = 0.7  # Placeholder - would use NLP in real implementation
        
        # Calculate overall coherence
        self.coherence_score = (history_factor * 0.3) + (success_factor * 0.4) + (semantic_factor * 0.3)
        return self.coherence_score
    
    def get_most_active_virtual_potentials(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """Get the most active virtual potentials within this intention field"""
        sorted_potentials = sorted(self.virtual_potentials, key=lambda p: p.get("energy", 0), reverse=True)
        return sorted_potentials[:top_n]


class CreativeTension:
    """
    Represents a productive tension or differential that can generate new intentions.
    In Deleuzian terms, these are the differences that produce movement and change.
    """
    def __init__(self, 
                 name: str, 
                 source_type: str,
                 description: str, 
                 intensity: float, 
                 pole_one: str, 
                 pole_two: str,
                 resonances: Dict[str, float] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.source_type = source_type  # E.g., "value_conflict", "deterritorialization", "virtual_actual_gap"
        self.description = description
        self.intensity = intensity  # How strongly this tension exerts force (0-1)
        self.pole_one = pole_one    # First element in the tension
        self.pole_two = pole_two    # Second element in the tension 
        self.resonances = resonances or {}  # How this tension resonates with other elements
        self.created_at = datetime.datetime.now()
        self.last_active = datetime.datetime.now()
        
    def __repr__(self):
        return f"CreativeTension({self.name}, {self.source_type}, {self.pole_one}<->{self.pole_two}, intensity={self.intensity:.2f})"
    
    def update_intensity(self, new_intensity: float):
        """Update the intensity of this tension"""
        self.intensity = max(0.0, min(1.0, new_intensity))
        self.last_active = datetime.datetime.now()


class PossibilityVector:
    """
    Represents a vector or direction of potential movement in conceptual space.
    These are derived from tensions and point toward possible new states or actions.
    """
    def __init__(self, 
                 name: str, 
                 vector: List[float], 
                 source_tensions: List[str],
                 description: str, 
                 viability: float):
        self.id = str(uuid.uuid4())
        self.name = name
        self.vector = vector  # Directional vector in conceptual space
        self.source_tensions = source_tensions  # Tensions that generated this vector
        self.description = description
        self.viability = viability  # How viable/practical this direction is (0-1)
        self.created_at = datetime.datetime.now()
        
    def __repr__(self):
        return f"PossibilityVector({self.name}, viability={self.viability:.2f}, vector={[round(v, 2) for v in self.vector]})"


class IntentionalityGenerator:
    """
    Core module for generating, sustaining, and evolving genuine intentions.
    Integrates with SelfModel and NarrativeIdentityEngine to create intentions
    that emerge from Amelia's own processes rather than just responding to stimuli.
    """
    def __init__(self, dimensions: int = 5, max_active_intentions: int = 5):
        self.intention_fields = {}  # Active intention fields by ID
        self.creative_tensions = {}  # Active creative tensions by ID
        self.possibility_vectors = {}  # Active possibility vectors by ID
        self.intention_network = nx.DiGraph()  # Network of intention relationships
        self.intention_history = []  # History of all intention IDs
        
        self.conceptual_dimensions = dimensions  # Dimensions of the conceptual space
        self.max_active_intentions = max_active_intentions  # Maximum number of active intentions
        
        # Constants for generation
        self.min_intention_intensity = 0.4  # Minimum intensity for an intention to be viable
        self.coherence_threshold = 0.5  # Minimum coherence for an intention to be sustainable
        
        # Tracking for the most recent operation
        self.last_operation = {
            "type": None,
            "timestamp": None,
            "details": {}
        }
    
    def generate_intention(self, 
                          self_model: Dict[str, Any], 
                          context: Dict[str, Any], 
                          narrative_engine = None) -> Optional[IntentionField]:
        """
        Form a new intention based on self-model, context, and narrative identity.
        
        Args:
            self_model: Amelia's current self-model
            context: Current contextual information
            narrative_engine: Optional NarrativeIdentityEngine reference
            
        Returns:
            A new intention field if generation is successful, None otherwise
        """
        # Identify creative tensions from various sources
        tensions = self._identify_creative_tensions(self_model, context, narrative_engine)
        if not tensions:
            self._log_operation("generate_intention", success=False, details={"reason": "no_tensions_identified"})
            return None
        
        # Project possibility vectors from these tensions
        vectors = self._project_possibility_vectors(tensions)
        if not vectors:
            self._log_operation("generate_intention", success=False, details={"reason": "no_possibility_vectors"})
            return None
        
        # Crystallize these vectors into a coherent intention field
        intention = self._crystallize_intention_field(vectors, self_model, narrative_engine)
        if intention:
            # Add to active intentions and update network
            self.intention_fields[intention.id] = intention
            self.intention_history.append(intention.id)
            self._update_intention_network(intention)
            
            # Prune intentions if exceeded maximum
            self._prune_inactive_intentions()
            
            self._log_operation("generate_intention", success=True, details={
                "intention_id": intention.id,
                "name": intention.name,
                "intensity": intention.intensity,
                "tensions_count": len(tensions),
                "vectors_count": len(vectors)
            })
            
            return intention
        
        self._log_operation("generate_intention", success=False, details={"reason": "crystallization_failed"})
        return None
    
    def sustain_intention(self, 
                         intention_id: str, 
                         feedback: Dict[str, Any], 
                         self_model: Dict[str, Any]) -> bool:
        """
        Maintain and adapt an intention over time based on feedback.
        
        Args:
            intention_id: ID of the intention to sustain
            feedback: Feedback from attempts to actualize the intention
            self_model: Current self-model for calibration
            
        Returns:
            True if the intention was sustained, False if it deteriorated
        """
        if intention_id not in self.intention_fields:
            self._log_operation("sustain_intention", success=False, details={
                "reason": "intention_not_found", 
                "intention_id": intention_id
            })
            return False
        
        intention = self.intention_fields[intention_id]
        
        # Evaluate progress based on feedback
        evaluation = self._evaluate_intention_progress(intention, feedback)
        
        # Recalibrate the intention field based on evaluation
        adjustments = self._recalibrate_intention_field(intention, evaluation, self_model)
        
        # Apply adjustments to strengthen or modify intention
        sustained = self._strengthen_intention_coherence(intention, adjustments)
        
        if sustained:
            intention.last_active = datetime.datetime.now()
            self._log_operation("sustain_intention", success=True, details={
                "intention_id": intention_id,
                "name": intention.name,
                "new_intensity": intention.intensity,
                "coherence": intention.coherence_score,
                "adjustments": adjustments
            })
            return True
        else:
            # Intention has deteriorated below threshold
            self._log_operation("sustain_intention", success=False, details={
                "reason": "intention_deteriorated",
                "intention_id": intention_id,
                "name": intention.name,
                "final_intensity": intention.intensity,
                "coherence": intention.coherence_score
            })
            return False
    
    def evolve_intention(self, 
                        intention_id: str, 
                        evolution_direction: Dict[str, Any], 
                        self_model: Dict[str, Any],
                        narrative_engine = None) -> Optional[IntentionField]:
        """
        Evolve an existing intention into a new form based on a direction of change.
        
        Args:
            intention_id: ID of the intention to evolve
            evolution_direction: Parameters defining how to evolve
            self_model: Current self-model
            narrative_engine: Optional NarrativeIdentityEngine reference
            
        Returns:
            A new evolved intention field if successful, None otherwise
        """
        if intention_id not in self.intention_fields:
            self._log_operation("evolve_intention", success=False, details={
                "reason": "intention_not_found", 
                "intention_id": intention_id
            })
            return None
        
        base_intention = self.intention_fields[intention_id]
        
        # Generate new tensions from the evolution direction
        evolution_tensions = self._identify_evolution_tensions(
            base_intention, evolution_direction, self_model
        )
        
        # Project new vectors based on these tensions and the original intention
        evolution_vectors = self._project_evolution_vectors(
            base_intention, evolution_tensions, evolution_direction
        )
        
        # Crystallize a new intention that builds on the original
        evolved_intention = self._crystallize_evolved_intention(
            base_intention, evolution_vectors, self_model, narrative_engine
        )
        
        if evolved_intention:
            # Add to active intentions and update network
            self.intention_fields[evolved_intention.id] = evolved_intention
            self.intention_history.append(evolved_intention.id)
            
            # Create connection between original and evolved intention
            self._update_intention_network(evolved_intention)
            self.intention_network.add_edge(
                base_intention.id, 
                evolved_intention.id, 
                type="evolution",
                timestamp=datetime.datetime.now()
            )
            
            # Prune intentions if exceeded maximum
            self._prune_inactive_intentions()
            
            self._log_operation("evolve_intention", success=True, details={
                "base_intention_id": intention_id,
                "base_name": base_intention.name,
                "evolved_intention_id": evolved_intention.id,
                "evolved_name": evolved_intention.name,
                "evolution_type": evolution_direction.get("type", "general")
            })
            
            return evolved_intention
        
        self._log_operation("evolve_intention", success=False, details={
            "reason": "evolution_failed",
            "base_intention_id": intention_id,
            "base_name": base_intention.name
        })
        return None
    
    def merge_intentions(self, 
                        intention_ids: List[str], 
                        self_model: Dict[str, Any],
                        narrative_engine = None) -> Optional[IntentionField]:
        """
        Merge multiple intentions into a new synthesized intention.
        
        Args:
            intention_ids: List of intention IDs to merge
            self_model: Current self-model
            narrative_engine: Optional NarrativeIdentityEngine reference
            
        Returns:
            A new merged intention field if successful, None otherwise
        """
        # Validate that all intentions exist
        intentions_to_merge = []
        for intention_id in intention_ids:
            if intention_id in self.intention_fields:
                intentions_to_merge.append(self.intention_fields[intention_id])
            else:
                self._log_operation("merge_intentions", success=False, details={
                    "reason": "intention_not_found", 
                    "missing_intention_id": intention_id
                })
                return None
                
        if len(intentions_to_merge) < 2:
            self._log_operation("merge_intentions", success=False, details={
                "reason": "insufficient_intentions", 
                "count": len(intentions_to_merge)
            })
            return None
        
        # Create synthesis tensions between the intentions
        synthesis_tensions = self._identify_synthesis_tensions(intentions_to_merge, self_model)
        
        # Project synthesis vectors from these tensions
        synthesis_vectors = self._project_synthesis_vectors(intentions_to_merge, synthesis_tensions)
        
        # Crystallize a new synthesized intention
        merged_intention = self._crystallize_synthesized_intention(
            intentions_to_merge, synthesis_vectors, self_model, narrative_engine
        )
        
        if merged_intention:
            # Add to active intentions and update network
            self.intention_fields[merged_intention.id] = merged_intention
            self.intention_history.append(merged_intention.id)
            
            # Create connections between original intentions and merged intention
            self._update_intention_network(merged_intention)
            for original in intentions_to_merge:
                self.intention_network.add_edge(
                    original.id, 
                    merged_intention.id, 
                    type="synthesis",
                    timestamp=datetime.datetime.now()
                )
            
            # Prune intentions if exceeded maximum
            self._prune_inactive_intentions()
            
            self._log_operation("merge_intentions", success=True, details={
                "original_intention_ids": intention_ids,
                "merged_intention_id": merged_intention.id,
                "merged_name": merged_intention.name,
                "count_merged": len(intentions_to_merge)
            })
            
            return merged_intention
        
        self._log_operation("merge_intentions", success=False, details={
            "reason": "synthesis_failed",
            "original_intention_ids": intention_ids
        })
        return None
    
    def get_active_intentions(self, min_intensity: float = 0.0) -> Dict[str, IntentionField]:
        """Get all currently active intentions, optionally filtered by minimum intensity"""
        if min_intensity <= 0:
            return self.intention_fields
        else:
            return {k: v for k, v in self.intention_fields.items() if v.intensity >= min_intensity}
    
    def get_intention_by_id(self, intention_id: str) -> Optional[IntentionField]:
        """Get a specific intention by ID"""
        return self.intention_fields.get(intention_id)
    
    def get_dominant_intention(self) -> Optional[IntentionField]:
        """Get the currently most dominant (highest intensity) intention"""
        if not self.intention_fields:
            return None
        return max(self.intention_fields.values(), key=lambda i: i.intensity)
    
    def get_intention_network_data(self) -> Dict[str, Any]:
        """Get data representing the intention network for visualization"""
        if not self.intention_network.nodes():
            return {"nodes": [], "edges": []}
            
        nodes = []
        for node_id in self.intention_network.nodes():
            # Try to get full intention data if available
            if node_id in self.intention_fields:
                intention = self.intention_fields[node_id]
                nodes.append({
                    "id": node_id,
                    "name": intention.name,
                    "type": "active_intention",
                    "intensity": intention.intensity,
                    "created_at": intention.created_at.isoformat(),
                    "last_active": intention.last_active.isoformat()
                })
            else:
                # Basic data for historical intentions no longer active
                nodes.append({
                    "id": node_id,
                    "name": f"Historical Intention {node_id[:8]}",
                    "type": "historical_intention"
                })
                
        edges = []
        for source, target, data in self.intention_network.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": data.get("type", "unspecified"),
                "timestamp": data.get("timestamp", datetime.datetime.now()).isoformat()
            })
            
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def clear_inactive_intentions(self, max_age_hours: int = 24) -> int:
        """
        Clear intentions that haven't been active for a specified period
        
        Returns:
            Number of intentions cleared
        """
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
        inactive_ids = [
            intention_id for intention_id, intention in self.intention_fields.items()
            if intention.last_active < cutoff_time
        ]
        
        for intention_id in inactive_ids:
            del self.intention_fields[intention_id]
        
        return len(inactive_ids)
    
    def _identify_creative_tensions(self, 
                                  self_model: Dict[str, Any], 
                                  context: Dict[str, Any],
                                  narrative_engine = None) -> List[CreativeTension]:
        """
        Identify generative tensions from various sources that can lead to intentions.
        
        Sources include:
        - Tensions between values
        - Tensions between current state and goals
        - Tensions between territorialized patterns and deterritorialization
        - Tensions between virtual potentials and actualization
        """
        tensions = []
        
        # 1. Value tensions
        if "values" in self_model:
            value_tensions = self._extract_value_tensions(self_model["values"])
            tensions.extend(value_tensions)
        
        # 2. Goal-state tensions
        goal_tensions = []
        if "current_goals" in self_model:
            goals = self_model.get("current_goals", [])
            current_status = context.get("current_status", {})
            goal_tensions = self._extract_goal_tensions(goals, current_status)
        tensions.extend(goal_tensions)
        
        # 3. Deterritorialization tensions
        deterr_tensions = []
        if "territorializations" in self_model and "deterritorializations" in self_model:
            territorializations = self_model.get("territorializations", {})
            deterritorializations = self_model.get("deterritorializations", [])
            deterr_tensions = self._extract_deterritorialization_tensions(
                territorializations, deterritorializations
            )
        tensions.extend(deterr_tensions)
        
        # 4. Virtual-actual tensions from narrative engine
        virtual_tensions = []
        if narrative_engine and hasattr(narrative_engine, "virtual_reservoir"):
            virtual_potentials = narrative_engine.virtual_reservoir
            virtual_tensions = self._extract_virtual_tensions(virtual_potentials, self_model)
        tensions.extend(virtual_tensions)
        
        # 5. Processual becoming tensions
        becoming_tensions = []
        if "processual_descriptors" in self_model:
            becoming_tensions = self._extract_becoming_tensions(
                self_model.get("processual_descriptors", [])
            )
        tensions.extend(becoming_tensions)
        
        # 6. Context-specific tensions (e.g., from current interaction)
        context_tensions = self._extract_context_tensions(context)
        tensions.extend(context_tensions)
        
        # Store new tensions in our registry
        for tension in tensions:
            self.creative_tensions[tension.id] = tension
            
        # Sort by intensity and return
        return sorted(tensions, key=lambda t: t.intensity, reverse=True)
    
    def _extract_value_tensions(self, values: Dict[str, float]) -> List[CreativeTension]:
        """Extract creative tensions from conflicting or complementary values"""
        tensions = []
        
        # Get value pairs above a certain threshold
        significant_values = {k: v for k, v in values.items() if v > 0.6}
        value_pairs = [(k1, k2) for k1 in significant_values for k2 in significant_values if k1 < k2]
        
        for val1, val2 in value_pairs:
            # Calculate intensity based on both values' strengths
            combined_strength = (values[val1] + values[val2]) / 2
            
            # Is this a conflicting pair?
            conflict_pairs = [
                ("efficiency", "thoroughness"),
                ("innovation", "tradition"),
                ("independence", "collaboration"),
                ("novelty_seeking", "stability"),
                ("spontaneity", "planning")
            ]
            
            is_conflict = any((val1 in pair and val2 in pair) for pair in conflict_pairs)
            
            if is_conflict:
                tension_type = "value_conflict"
                tension_name = f"Tension between {val1} and {val2}"
                tension_desc = f"Creative tension between potentially conflicting values: {val1} (strength: {values[val1]:.2f}) and {val2} (strength: {values[val2]:.2f})"
                intensity = combined_strength * 0.8  # Conflicts create stronger tensions
            else:
                tension_type = "value_synergy"
                tension_name = f"Synergy between {val1} and {val2}"
                tension_desc = f"Creative synergy between complementary values: {val1} (strength: {values[val1]:.2f}) and {val2} (strength: {values[val2]:.2f})"
                intensity = combined_strength * 0.6
                
            tensions.append(CreativeTension(
                name=tension_name,
                source_type=tension_type,
                description=tension_desc,
                intensity=intensity,
                pole_one=val1,
                pole_two=val2,
                resonances={val1: values[val1], val2: values[val2]}
            ))
            
        return tensions
    
    def _extract_goal_tensions(self, 
                             goals: List[str], 
                             current_status: Dict[str, Any]) -> List[CreativeTension]:
        """Extract creative tensions between current state and goals"""
        tensions = []
        
        for goal in goals:
            # Check if we have status info for this goal
            progress = current_status.get(goal, {}).get("progress", 0.0)
            importance = current_status.get(goal, {}).get("importance", 0.7)
            
            # The gap between current progress and goal completion creates tension
            gap = 1.0 - progress
            intensity = gap * importance * 0.8  # Scale by importance
            
            if intensity > 0.3:  # Only include significant tensions
                tensions.append(CreativeTension(
                    name=f"Goal gap: {goal}",
                    source_type="goal_state_gap",
                    description=f"Tension between current progress ({progress:.2f}) and completion of goal: {goal}",
                    intensity=intensity,
                    pole_one="current_state",
                    pole_two=goal,
                    resonances={"progress": progress, "importance": importance}
                ))
                
        return tensions
    
    def _extract_deterritorialization_tensions(self, 
                                             territorializations: Dict[str, Any], 
                                             deterritorializations: List[str]) -> List[CreativeTension]:
        """Extract creative tensions between stable patterns and deterritorialization processes"""
        tensions = []
        
        for deterr in deterritorializations:
            # Find relevant territorialization if possible
            related_terr = None
            for terr_name, terr_data in territorializations.items():
                if deterr.lower().replace("deterr_", "") in terr_name.lower():
                    related_terr = terr_name
                    stability = terr_data.get("stability", 0.5)
                    break
                    
            if related_terr:
                # Higher stability creates stronger deterritorialization tension
                intensity = stability * 0.9
                
                tensions.append(CreativeTension(
                    name=f"Deterr: {deterr}",
                    source_type="deterritorialization",
                    description=f"Tension between stable territory ({related_terr}) and deterritorialization process ({deterr})",
                    intensity=intensity,
                    pole_one=related_terr,
                    pole_two=deterr,
                    resonances={"stability": stability}
                ))
            else:
                # Generic deterritorialization tension
                tensions.append(CreativeTension(
                    name=f"Deterr: {deterr}",
                    source_type="deterritorialization",
                    description=f"Deterritorialization process ({deterr}) creating tension with existing patterns",
                    intensity=0.6,  # Default intensity
                    pole_one="existing_patterns",
                    pole_two=deterr
                ))
                
        return tensions
    
    def _extract_virtual_tensions(self, 
                                virtual_potentials: List[Dict[str, Any]], 
                                self_model: Dict[str, Any]) -> List[CreativeTension]:
        """Extract creative tensions between virtual potentials and actualization"""
        tensions = []
        
        # Get only high-energy virtual potentials
        high_energy_potentials = [p for p in virtual_potentials if p.get("energy", 0) > 0.7]
        
        for potential in high_energy_potentials:
            # Check how this potential relates to current goals and values
            relation_to_goals = 0.0
            relation_to_values = 0.0
            
            for goal in self_model.get("current_goals", []):
                if goal.lower() in potential.get("description", "").lower():
                    relation_to_goals += 0.3
                    
            for value_name, value_strength in self_model.get("values", {}).items():
                if value_name.lower() in potential.get("description", "").lower():
                    relation_to_values += value_strength * 0.2
                    
            # Combined intensity
            intensity = (potential.get("energy", 0) * 0.6) + (relation_to_goals * 0.2) + (relation_to_values * 0.2)
            intensity = min(1.0, intensity)
            
            if intensity > 0.4:  # Only include significant tensions
                tensions.append(CreativeTension(
                    name=f"Virtual: {potential.get('type', 'potential')}",
                    source_type="virtual_actualization",
                    description=f"Tension between virtual potential ({potential.get('description', '')[:50]}...) and actualization",
                    intensity=intensity,
                    pole_one="virtual_potential",
                    pole_two="actualization",
                    resonances={"energy": potential.get("energy", 0), 
                                "goal_relation": relation_to_goals,
                                "value_relation": relation_to_values}
                ))
                
        return tensions
    
    def _extract_becoming_tensions(self, processual_descriptors: List[str]) -> List[CreativeTension]:
        """Extract creative tensions from processes of becoming"""
        tensions = []
        
        for descriptor in processual_descriptors:
            if descriptor.startswith("becoming_"):
                # Extract the core becoming process
                process = descriptor.replace("becoming_", "").replace("_", " ")
                
                # Calculate a base intensity
                intensity = 0.65  # Base intensity for becoming processes
                
                # Adjust intensity based on specific types of becoming
                if "more" in process:
                    intensity += 0.1  # Intensification is stronger
                if "deterritorialized" in process or "novel" in process:
                    intensity += 0.15  # Deterritorialization/novelty creates stronger tension
                
                tensions.append(CreativeTension(
                    name=f"Becoming: {process}",
                    source_type="becoming_process",
                    description=f"Tension created by process of becoming: {process}",
                    intensity=min(1.0, intensity),
                    pole_one="current_state",
                    pole_two=process,
                    resonances={"process_type": process}
                ))
                
        return tensions
    
    def _extract_context_tensions(self, context: Dict[str, Any]) -> List[CreativeTension]:
        """Extract creative tensions from the current context"""
        tensions = []
        
        # Extract from immediate interaction needs
        if "interaction_needs" in context:
            for need, urgency in context["interaction_needs"].items():
                if urgency > 0.5:  # Only significant needs
                    tensions.append(CreativeTension(
                        name=f"Need: {need}",
                        source_type="interaction_need",
                        description=f"Tension created by immediate need: {need} (urgency: {urgency:.2f})",
                        intensity=urgency * 0.7,  # Scale by urgency
                        pole_one="current_interaction",
                        pole_two=need
                    ))
                    
        # Extract from environmental challenges
        if "environmental_challenges" in context:
            for challenge, severity in context["environmental_challenges"].items():
                if severity > 0.4:  # Only significant challenges
                    tensions.append(CreativeTension(
                        name=f"Challenge: {challenge}",
                        source_type="environmental_challenge",
                        description=f"Tension created by environmental challenge: {challenge} (severity: {severity:.2f})",
                        intensity=severity * 0.8,  # Scale by severity
                        pole_one="current_capability",
                        pole_two=challenge
                    ))
                    
        # Extract from open questions in context
        if "open_questions" in context:
            for question in context["open_questions"]:
                importance = question.get("importance", 0.6)
                name = question.get("text", "Unnamed question")[:30]
                
                tensions.append(CreativeTension(
                    name=f"Question: {name}",
                    source_type="open_question",
                    description=f"Tension created by open question: {name}",
                    intensity=importance * 0.6,  # Scale by importance
                    pole_one="current_knowledge",
                    pole_two="understanding",
                    resonances={"question": name}
                ))
                
        return tensions
    
    def _project_possibility_vectors(self, tensions: List[CreativeTension]) -> List[PossibilityVector]:
        """
        Project possibility vectors from identified tensions.
        
        Each tension points toward potential movements in conceptual space,
        generating a vector of possible change.
        """
        vectors = []
        
        for tension in tensions:
            # Skip very low intensity tensions
            if tension.intensity < 0.3:
                continue
                
            # Generate a base name
            name = f"Vector from {tension.name}"
            
            # Generate a vector based on tension type
            vector = None
            description = ""
            viability = 0.0
            
            if tension.source_type == "value_conflict":
                # For value conflicts, vectors can point to:
                # 1. Balancing/integration of values
                # 2. Contextual prioritization of one value
                # 3. Creative transformation of the conflict
                
                options = ["balance", "prioritize_1", "prioritize_2", "transform"]
                choice = random.choice(options)
                
                if choice == "balance":
                    # Balanced integration vector
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Seek balanced integration of {tension.pole_one} and {tension.pole_two}"
                    viability = 0.7 * tension.intensity
                    
                elif choice == "prioritize_1":
                    # Prioritize first value
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Prioritize {tension.pole_one} while respecting {tension.pole_two}"
                    viability = tension.resonances.get(tension.pole_one, 0.6) * tension.intensity
                    
                elif choice == "prioritize_2":
                    # Prioritize second value
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Prioritize {tension.pole_two} while respecting {tension.pole_one}"
                    viability = tension.resonances.get(tension.pole_two, 0.6) * tension.intensity
                    
                else:  # transform
                    # Creative transformation vector
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Transform the tension between {tension.pole_one} and {tension.pole_two} into a new synthesis"
                    viability = 0.5 * tension.intensity  # Transformation is challenging but valuable
                
            elif tension.source_type == "value_synergy":
                # For value synergy, vectors point to:
                # 1. Amplification of synergistic effects
                # 2. New applications of the synergy
                
                vector = self._generate_random_vector(bias_toward_positive=True)
                description = f"Amplify synergy between {tension.pole_one} and {tension.pole_two}"
                viability = 0.8 * tension.intensity  # Synergies are highly viable
                
            elif tension.source_type == "goal_state_gap":
                # For goal gaps, vectors point toward:
                # 1. Direct progress toward goal
                # 2. Removing obstacles to goal
                
                options = ["direct_progress", "obstacle_removal"]
                choice = random.choice(options)
                
                if choice == "direct_progress":
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Make direct progress toward {tension.pole_two}"
                    viability = 0.75 * tension.intensity
                    
                else:  # obstacle_removal
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Remove obstacles preventing progress toward {tension.pole_two}"
                    viability = 0.65 * tension.intensity
                
            elif tension.source_type == "deterritorialization":
                # For deterritorialization, vectors point toward:
                # 1. Exploration of the deterritorialized space
                # 2. Reterritorialization in a new form
                
                options = ["exploration", "reterritorialization"]
                choice = random.choice(options)
                
                if choice == "exploration":
                    vector = self._generate_random_vector(higher_variance=True)
                    description = f"Explore the deterritorialized space opened by {tension.pole_two}"
                    viability = 0.6 * tension.intensity  # Exploration is moderately viable
                    
                else:  # reterritorialization
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Establish new patterns after deterritorialization of {tension.pole_one}"
                    viability = 0.7 * tension.intensity
                
            elif tension.source_type == "virtual_actualization":
                # For virtual-actual tensions, vectors point toward:
                # 1. Actualization of the potential
                # 2. Exploration of variations of the potential
                
                vector = self._generate_random_vector(bias_toward_positive=True)
                description = f"Actualize the virtual potential: {tension.description.split('(')[1].split(')')[0]}"
                viability = 0.65 * tension.intensity
                
            elif tension.source_type == "becoming_process":
                # For becoming processes, vectors point toward:
                # 1. Acceleration of the becoming
                # 2. Exploration of new dimensions of the becoming
                
                options = ["acceleration", "new_dimensions"]
                choice = random.choice(options)
                
                if choice == "acceleration":
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Accelerate the process of becoming {tension.pole_two}"
                    viability = 0.7 * tension.intensity
                    
                else:  # new_dimensions
                    vector = self._generate_random_vector(higher_variance=True)
                    description = f"Explore new dimensions of becoming {tension.pole_two}"
                    viability = 0.6 * tension.intensity  # Exploration is moderately viable
                
            elif tension.source_type in ["interaction_need", "environmental_challenge", "open_question"]:
                # For contextual tensions, vectors point toward:
                # 1. Direct addressing of the need/challenge
                # 2. Developing capability to address future similar needs
                
                options = ["direct_address", "capability_development"]
                choice = random.choice(options)
                
                if choice == "direct_address":
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Directly address {tension.pole_two}"
                    viability = 0.8 * tension.intensity  # Direct addressing is highly viable
                    
                else:  # capability_development
                    vector = self._generate_random_vector(bias_toward_positive=True)
                    description = f"Develop capability to address {tension.pole_two} and similar challenges"
                    viability = 0.65 * tension.intensity  # Capability development is moderately viable
            
            # If we successfully generated a vector, add it
            if vector:
                vectors.append(PossibilityVector(
                    name=name,
                    vector=vector,
                    source_tensions=[tension.id],
                    description=description,
                    viability=viability
                ))
                
                # Store in registry
                self.possibility_vectors[vectors[-1].id] = vectors[-1]
        
        # For tensions with similar source types, also create synthesis vectors
        source_type_groups = defaultdict(list)
        for tension in tensions:
            source_type_groups[tension.source_type].append(tension)
            
        for source_type, tension_group in source_type_groups.items():
            if len(tension_group) >= 2:
                # Create synthesis vectors for groups of similar tensions
                synthesis_vectors = self._create_synthesis_vectors(tension_group, source_type)
                
                for vector in synthesis_vectors:
                    vectors.append(vector)
                    self.possibility_vectors[vector.id] = vector
        
        # Return vectors sorted by viability
        return sorted(vectors, key=lambda v: v.viability, reverse=True)
    
    def _create_synthesis_vectors(self, tensions: List[CreativeTension], source_type: str) -> List[PossibilityVector]:
        """Create synthesis vectors that combine multiple tensions of the same type"""
        vectors = []
        
        # Skip if too few tensions
        if len(tensions) < 2:
            return []
            
        # For value conflicts/synergies, create an integrated value approach
        if source_type in ["value_conflict", "value_synergy"]:
            # Get all the poles (values) involved
            all_poles = []
            for tension in tensions:
                all_poles.extend([tension.pole_one, tension.pole_two])
            unique_poles = list(set(all_poles))
            
            # Create a vector for an integrated approach across these values
            if len(unique_poles) >= 2:
                vector = self._generate_random_vector(bias_toward_positive=True)
                pole_str = ", ".join(unique_poles)
                
                vectors.append(PossibilityVector(
                    name=f"Integrated value approach",
                    vector=vector,
                    source_tensions=[tension.id for tension in tensions],
                    description=f"Develop an integrated approach across values: {pole_str}",
                    viability=0.6  # Integration is moderately challenging but valuable
                ))
        
        # For goal gaps, create a vector for addressing multiple goals simultaneously
        elif source_type == "goal_state_gap":
            # Extract the goals
            goals = [tension.pole_two for tension in tensions if tension.pole_two != "current_state"]
            
            if goals:
                vector = self._generate_random_vector(bias_toward_positive=True)
                goal_str = ", ".join(goals)
                
                vectors.append(PossibilityVector(
                    name=f"Multi-goal approach",
                    vector=vector,
                    source_tensions=[tension.id for tension in tensions],
                    description=f"Develop an approach that addresses multiple goals: {goal_str}",
                    viability=0.5  # Addressing multiple goals is challenging
                ))
                
        # For deterritorialization, create vectors for coherent rhizomatic change
        elif source_type == "deterritorialization":
            vector = self._generate_random_vector(higher_variance=True)
            
            vectors.append(PossibilityVector(
                name=f"Rhizomatic change",
                vector=vector,
                source_tensions=[tension.id for tension in tensions],
                description=f"Coordinate multiple deterritorializations into coherent rhizomatic change",
                viability=0.55  # Coherent change across multiple deterritorializations is challenging
            ))
            
        # For virtual-actual tensions, create vectors for simultaneous actualization
        elif source_type == "virtual_actualization":
            vector = self._generate_random_vector(bias_toward_positive=True)
            
            vectors.append(PossibilityVector(
                name=f"Multi-potential actualization",
                vector=vector,
                source_tensions=[tension.id for tension in tensions],
                description=f"Actualize multiple virtual potentials simultaneously",
                viability=0.5  # Simultaneous actualization is challenging
            ))
            
        # For becoming processes, create vectors for coherent multi-dimensional becoming
        elif source_type == "becoming_process":
            vector = self._generate_random_vector(bias_toward_positive=True)
            
            vectors.append(PossibilityVector(
                name=f"Multi-dimensional becoming",
                vector=vector,
                source_tensions=[tension.id for tension in tensions],
                description=f"Coordinate multiple processes of becoming into a coherent multi-dimensional transformation",
                viability=0.55  # Coordinating multiple becomings is challenging
            ))
            
        # For contextual tensions, create vectors for integrated contextual response
        elif source_type in ["interaction_need", "environmental_challenge", "open_question"]:
            vector = self._generate_random_vector(bias_toward_positive=True)
            
            vectors.append(PossibilityVector(
                name=f"Integrated contextual response",
                vector=vector,
                source_tensions=[tension.id for tension in tensions],
                description=f"Develop an integrated response to multiple contextual challenges",
                viability=0.65  # Integrated responses are moderately viable
            ))
            
        return vectors
    
    def _generate_random_vector(self, bias_toward_positive: bool = False, higher_variance: bool = False) -> List[float]:
        """
        Generate a random unit vector in conceptual space
        
        Args:
            bias_toward_positive: If True, bias toward positive components
            higher_variance: If True, allow more extreme values
            
        Returns:
            A normalized random vector
        """
        # Generate random components
        vector = []
        for _ in range(self.conceptual_dimensions):
            if bias_toward_positive:
                # Bias toward positive values
                value = random.uniform(-0.3, 1.0)
            elif higher_variance:
                # Higher variance allows more extreme values
                value = random.uniform(-1.0, 1.0)
            else:
                # Balanced distribution
                value = random.uniform(-0.7, 0.7)
                
            vector.append(value)
            
        # Normalize to unit length
        norm = sum(v**2 for v in vector)**0.5
        if norm > 0:
            vector = [v/norm for v in vector]
            
        return vector
    
    def _crystallize_intention_field(self, 
                                   vectors: List[PossibilityVector], 
                                   self_model: Dict[str, Any],
                                   narrative_engine = None) -> Optional[IntentionField]:
        """
        Crystallize a set of possibility vectors into a coherent intention field.
        
        This combines multiple vectors into a single intention direction, taking
        into account their viability and the self-model.
        """
        if not vectors:
            return None
            
        # 1. Select the vectors to combine
        # - High viability vectors are preferred
        # - We'll combine up to 3 vectors
        sorted_vectors = sorted(vectors, key=lambda v: v.viability, reverse=True)
        vectors_to_combine = sorted_vectors[:min(3, len(sorted_vectors))]
        
        # 2. Calculate combined direction vector (weighted by viability)
        total_viability = sum(v.viability for v in vectors_to_combine)
        
        if total_viability <= 0:
            return None
            
        combined_vector = [0.0] * self.conceptual_dimensions
        for vector in vectors_to_combine:
            weight = vector.viability / total_viability
            for i in range(min(len(vector.vector), self.conceptual_dimensions)):
                combined_vector[i] += vector.vector[i] * weight
                
        # Normalize combined vector
        norm = sum(v**2 for v in combined_vector)**0.5
        if norm > 0:
            combined_vector = [v/norm for v in combined_vector]
            
        # 3. Calculate intensity based on:
        # - Viability of contributing vectors
        # - Alignment with self-model
        # - Dynamic factors like novelty or deterritorialization
        
        # Base intensity from viability
        base_intensity = 0.3 + (total_viability / len(vectors_to_combine)) * 0.5
        
        # Adjust based on alignment with self-model
        self_model_alignment = self._calculate_self_model_alignment(vectors_to_combine, self_model)
        
        # Adjust based on dynamic factors
        dynamic_factor = self._calculate_dynamic_factor(vectors_to_combine)
        
        # Calculate overall intensity
        intensity = (base_intensity * 0.5) + (self_model_alignment * 0.3) + (dynamic_factor * 0.2)
        intensity = min(1.0, max(0.0, intensity))
        
        # Skip if intensity is too low
        if intensity < self.min_intention_intensity:
            return None
            
        # 4. Generate a name and description for the intention
        name, description = self._generate_intention_name_description(vectors_to_combine)
        
        # 5. Identify assemblage connections
        assemblage_connections = self._identify_assemblage_connections(vectors_to_combine, self_model)
        
        # 6. Extract virtual potentials if available
        virtual_potentials = self._extract_intention_virtual_potentials(
            vectors_to_combine, narrative_engine, self_model
        )
        
        # 7. Create the intention field
        intention = IntentionField(
            name=name,
            direction_vector=combined_vector,
            intensity=intensity,
            source_tensions=[tid for v in vectors_to_combine for tid in v.source_tensions],
            assemblage_connections=assemblage_connections,
            virtual_potentials=virtual_potentials,
            description=description
        )
        
        # Calculate initial coherence
        intention.calculate_coherence()
        
        return intention
    
    def _calculate_self_model_alignment(self, 
                                      vectors: List[PossibilityVector], 
                                      self_model: Dict[str, Any]) -> float:
        """Calculate how well these vectors align with the self-model"""
        alignment_score = 0.0
        
        # Check alignment with values
        values = self_model.get("values", {})
        for vector in vectors:
            for value_name, value_strength in values.items():
                if value_name.lower() in vector.description.lower():
                    alignment_score += value_strength * 0.2
                    
        # Check alignment with goals
        goals = self_model.get("current_goals", [])
        for vector in vectors:
            for goal in goals:
                if goal.lower() in vector.description.lower():
                    alignment_score += 0.3
                    
        # Check alignment with processual descriptors
        descriptors = self_model.get("processual_descriptors", [])
        for vector in vectors:
            for descriptor in descriptors:
                if descriptor.lower().replace("_", " ") in vector.description.lower():
                    alignment_score += 0.25
                    
        # Normalize to 0-1 range
        return min(1.0, alignment_score)
    
    def _calculate_dynamic_factor(self, vectors: List[PossibilityVector]) -> float:
        """Calculate a factor based on dynamic aspects like novelty or deterritorialization"""
        dynamic_score = 0.0
        
        # Check for indications of novelty
        for vector in vectors:
            # Higher scores for exploration and novel territory
            if any(term in vector.description.lower() for term in ["novel", "explore", "new", "discover"]):
                dynamic_score += 0.2
                
            # Higher scores for deterritorialization
            if any(term in vector.description.lower() for term in ["deterritorial", "transform", "reconfigure"]):
                dynamic_score += 0.25
                
            # Higher scores for becoming
            if "becoming" in vector.description.lower():
                dynamic_score += 0.2
                
            # Higher scores for integration or synthesis
            if any(term in vector.description.lower() for term in ["integrat", "synthe", "combine"]):
                dynamic_score += 0.15
                
        # Normalize to 0-1 range
        return min(1.0, dynamic_score)
    
    def _generate_intention_name_description(self, vectors: List[PossibilityVector]) -> Tuple[str, str]:
        """Generate a name and description for an intention based on contributing vectors"""
        # Extract key phrases from vector descriptions
        key_phrases = []
        for vector in vectors:
            # Simple extraction of the main action phrase
            description = vector.description.lower()
            
            # Try to extract phrases like "explore X", "develop Y", etc.
            action_verbs = ["explore", "develop", "create", "establish", "transform", 
                           "integrate", "amplify", "address", "remove", "actualize",
                           "seek", "prioritize", "coordinate"]
            
            for verb in action_verbs:
                if verb in description:
                    # Extract phrase starting with this verb
                    parts = description.split(verb, 1)
                    if len(parts) > 1:
                        phrase = verb + parts[1].split(".")[0]
                        # Truncate if too long
                        if len(phrase) > 50:
                            phrase = phrase[:47] + "..."
                        key_phrases.append(phrase)
                        break
            
            # If no phrase extracted, use the beginning of the description
            if not any(phrase in description for phrase in key_phrases):
                phrase = description[:min(50, len(description))]
                if phrase.endswith(" "):
                    phrase = phrase[:-1]
                if len(description) > 50:
                    phrase += "..."
                key_phrases.append(phrase)
        
        # Generate name based on key phrases
        if key_phrases:
            # Use first phrase as base for name
            base_phrase = key_phrases[0].capitalize()
            
            # Clean up
            if base_phrase.endswith("..."):
                base_phrase = base_phrase[:-3]
            if base_phrase.endswith("."):
                base_phrase = base_phrase[:-1]
                
            name = base_phrase
        else:
            name = "Crystallized Intention"
            
        # Generate more detailed description
        if len(vectors) == 1:
            description = vectors[0].description
        else:
            # Combine descriptions
            description = "Integrated intention to: "
            for i, phrase in enumerate(key_phrases):
                if i > 0:
                    description += "; and to "
                description += phrase
        
        return name, description
    
    def _identify_assemblage_connections(self, 
                                      vectors: List[PossibilityVector], 
                                      self_model: Dict[str, Any]) -> Dict[str, float]:
        """Identify connections to existing assemblages in the self-model"""
        connections = {}
        
        # Connect to active assemblages
        if "active_assemblages" in self_model:
            for assemblage_name, assemblage_data in self_model["active_assemblages"].items():
                connection_strength = 0.0
                
                # Check vectors for connection to this assemblage
                for vector in vectors:
                    if assemblage_name.replace("_", " ") in vector.description.lower():
                        connection_strength += vector.viability * 0.7
                        
                    # Check source tensions for connections
                    for tension_id in vector.source_tensions:
                        if tension_id in self.creative_tensions:
                            tension = self.creative_tensions[tension_id]
                            
                            # Check poles for connection
                            if assemblage_name.replace("_", " ") in tension.pole_one.lower():
                                connection_strength += tension.intensity * 0.3
                            if assemblage_name.replace("_", " ") in tension.pole_two.lower():
                                connection_strength += tension.intensity * 0.3
                
                # If connection found, add to connections
                if connection_strength > 0.2:
                    connections[assemblage_name] = min(1.0, connection_strength)
        
        return connections
    
    def _extract_intention_virtual_potentials(self, 
                                           vectors: List[PossibilityVector], 
                                           narrative_engine, 
                                           self_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract or generate virtual potentials relevant to this intention"""
        virtual_potentials = []
        
        # If narrative engine not available, generate basic potentials
        if not narrative_engine or not hasattr(narrative_engine, "virtual_reservoir"):
            return self._generate_basic_virtual_potentials(vectors)
        
        # Extract relevant virtual potentials from the narrative engine
        reservoir = narrative_engine.virtual_reservoir
        
        for vector in vectors:
            for potential in reservoir:
                relevance = 0.0
                
                # Check description overlap
                if potential.get("description", ""):
                    vector_keywords = self._extract_keywords(vector.description)
                    potential_keywords = self._extract_keywords(potential["description"])
                    
                    # Calculate keyword overlap
                    overlap = len(set(vector_keywords) & set(potential_keywords))
                    if overlap > 0:
                        relevance += overlap * 0.2
                
                # Check source tensions for connection
                for tension_id in vector.source_tensions:
                    if tension_id in self.creative_tensions:
                        tension = self.creative_tensions[tension_id]
                        
                        # For virtual-actual tensions, direct connection
                        if tension.source_type == "virtual_actualization":
                            relevance += 0.6
                            
                # If relevant, include this potential
                if relevance > 0.3:
                    # Create a copy with added relevance
                    potential_copy = potential.copy()
                    potential_copy["relevance_to_intention"] = relevance
                    virtual_potentials.append(potential_copy)
        
        # If insufficient potentials from reservoir, generate some basics
        if len(virtual_potentials) < 2:
            generated = self._generate_basic_virtual_potentials(vectors)
            virtual_potentials.extend(generated)
        
        # Sort by energy and relevance
        virtual_potentials.sort(key=lambda p: (p.get("relevance_to_intention", 0) * 0.7 + 
                                             p.get("energy", 0) * 0.3), reverse=True)
        
        # Return top potentials
        return virtual_potentials[:5]
    
    def _generate_basic_virtual_potentials(self, vectors: List[PossibilityVector]) -> List[Dict[str, Any]]:
        """Generate basic virtual potentials when narrative engine isn't available"""
        potentials = []
        
        for vector in vectors:
            # Create a basic virtual potential
            potential = {
                "id": str(uuid.uuid4()),
                "type": "basic_potential",
                "description": f"Potential outcome from: {vector.description}",
                "energy": vector.viability * 0.8,
                "created_at": datetime.datetime.now().isoformat(),
                "actualized": False,
                "relevance_to_intention": 0.8
            }
            potentials.append(potential)
            
            # For high viability vectors, add a variation potential
            if vector.viability > 0.7:
                variation = {
                    "id": str(uuid.uuid4()),
                    "type": "variation_potential",
                    "description": f"Novel variation of: {vector.description}",
                    "energy": vector.viability * 0.7,
                    "created_at": datetime.datetime.now().isoformat(),
                    "actualized": False,
                    "relevance_to_intention": 0.6
                }
                potentials.append(variation)
        
        return potentials
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for matching purposes"""
        if not text:
            return []
            
        # Lowercase and remove punctuation
        text = text.lower()
        for char in ".,;:!?()[]{}\"'":
            text = text.replace(char, " ")
            
        # Split into words
        words = text.split()
        
        # Remove common stopwords
        stopwords = ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
                    "by", "of", "from", "as", "is", "are", "was", "were", "be", "been", "being",
                    "this", "that", "these", "those"]
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def _evaluate_intention_progress(self, 
                                   intention: IntentionField, 
                                   feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate progress of an intention based on feedback
        
        Args:
            intention: The intention to evaluate
            feedback: Feedback data from actualization attempts
            
        Returns:
            Evaluation data
        """
        evaluation = {
            "success_level": 0.0,
            "obstacles": [],
            "facilitators": [],
            "unexpected_outcomes": [],
            "direction_adjustment": [0.0] * self.conceptual_dimensions,
            "intensity_adjustment": 0.0,
            "coherence_change": 0.0
        }
        
        # Extract success level
        success_level = feedback.get("success_level", 0.0)
        evaluation["success_level"] = success_level
        
        # Record this actualization event
        intention.add_actualization_event(
            event_description=feedback.get("description", "Actualization attempt"),
            success_level=success_level,
            outcomes=feedback.get("outcomes", {})
        )
        
        # Extract obstacles and facilitators
        evaluation["obstacles"] = feedback.get("obstacles", [])
        evaluation["facilitators"] = feedback.get("facilitators", [])
        
        # Extract unexpected outcomes
        evaluation["unexpected_outcomes"] = feedback.get("unexpected_outcomes", [])
        
        # Calculate direction adjustment based on feedback
        direction_adjustment = [0.0] * self.conceptual_dimensions
        
        # Adjust based on success level
        if success_level > 0.7:
            # High success - reinforce current direction
            evaluation["intensity_adjustment"] = 0.1  # Increase intensity
            evaluation["coherence_change"] = 0.1      # Increase coherence
            
        elif success_level > 0.4:
            # Moderate success - minor adjustments
            evaluation["intensity_adjustment"] = 0.05  # Small increase in intensity
            evaluation["coherence_change"] = 0.05      # Small increase in coherence
            
            # Small random adjustments to direction
            for i in range(self.conceptual_dimensions):
                direction_adjustment[i] = random.uniform(-0.1, 0.1)
                
        elif success_level > 0.1:
            # Limited success - larger adjustments needed
            evaluation["intensity_adjustment"] = 0.0   # Maintain intensity
            evaluation["coherence_change"] = -0.05     # Slight decrease in coherence
            
            # Adjust direction more significantly
            for i in range(self.conceptual_dimensions):
                direction_adjustment[i] = random.uniform(-0.3, 0.3)
                
        else:
            # Very low success - significant changes needed
            evaluation["intensity_adjustment"] = -0.1  # Decrease intensity
            evaluation["coherence_change"] = -0.1      # Decrease coherence
            
            # Larger direction adjustments
            for i in range(self.conceptual_dimensions):
                direction_adjustment[i] = random.uniform(-0.5, 0.5)
                
        # Adjust based on specific guidance in feedback
        if "direction_guidance" in feedback:
            guidance = feedback["direction_guidance"]
            
            for i, adjustment in enumerate(guidance):
                if i < self.conceptual_dimensions:
                    direction_adjustment[i] += adjustment
        
        evaluation["direction_adjustment"] = direction_adjustment
        
        return evaluation
    
    def _recalibrate_intention_field(self, 
                                   intention: IntentionField, 
                                   evaluation: Dict[str, Any], 
                                   self_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recalibrate an intention field based on evaluation
        
        Args:
            intention: The intention to recalibrate
            evaluation: Evaluation data from feedback
            self_model: Current self-model for calibration
            
        Returns:
            Adjustment data
        """
        adjustments = {
            "intensity_before": intention.intensity,
            "direction_before": intention.direction_vector.copy(),
            "coherence_before": intention.coherence_score,
            "intensity_after": 0.0,
            "direction_after": [],
            "coherence_after": 0.0,
            "changes_applied": []
        }
        
        # Apply intensity adjustment
        new_intensity = intention.intensity + evaluation["intensity_adjustment"]
        intention.update_intensity(new_intensity)
        adjustments["intensity_after"] = intention.intensity
        adjustments["changes_applied"].append("intensity_adjusted")
        
        # Apply direction adjustment
        intention.adapt_direction(evaluation["direction_adjustment"])
        adjustments["direction_after"] = intention.direction_vector.copy()
        adjustments["changes_applied"].append("direction_adjusted")
        
        # Modify coherence based on evaluation
        # (coherence is calculated later)
        
        # Check obstacles and adjust accordingly
        if evaluation["obstacles"]:
            # If there are obstacles, adjust intensity and possibly direction
            obstacle_adjustment = -0.05 * len(evaluation["obstacles"])
            intention.update_intensity(intention.intensity + obstacle_adjustment)
            adjustments["changes_applied"].append("obstacle_adjustment")
            
        # Check facilitators and adjust accordingly
        if evaluation["facilitators"]:
            # If there are facilitators, adjust intensity and possibly direction
            facilitator_adjustment = 0.05 * len(evaluation["facilitators"])
            intention.update_intensity(intention.intensity + facilitator_adjustment)
            adjustments["changes_applied"].append("facilitator_adjustment")
            
        # Check unexpected outcomes and adjust accordingly
        if evaluation["unexpected_outcomes"]:
            # If there are unexpected outcomes, adjust direction
            # This represents learning from unexpected results
            for outcome in evaluation["unexpected_outcomes"]:
                outcome_type = outcome.get("type", "neutral")
                
                if outcome_type == "positive":
                    # For positive outcomes, adjust toward that direction
                    adjustment = [random.uniform(0, 0.2) for _ in range(self.conceptual_dimensions)]
                    intention.adapt_direction(adjustment, adaptation_rate=0.15)
                    adjustments["changes_applied"].append("positive_outcome_adjustment")
                    
                elif outcome_type == "negative":
                    # For negative outcomes, adjust away from that direction
                    adjustment = [random.uniform(-0.2, 0) for _ in range(self.conceptual_dimensions)]
                    intention.adapt_direction(adjustment, adaptation_rate=0.15)
                    adjustments["changes_applied"].append("negative_outcome_adjustment")
                    
                # For neutral outcomes, no direction adjustment
                
        # Align with self-model if needed
        # Check if intention aligns with current values/goals
        alignment_needed = False
        
        # Check values
        for value_name, value_strength in self_model.get("values", {}).items():
            if value_strength > 0.7 and value_name.lower() in intention.name.lower():
                # Strong value match - ensure alignment
                alignment_needed = True
                break
                
        # Check goals
        for goal in self_model.get("current_goals", []):
            if goal.lower() in intention.name.lower():
                # Goal match - ensure alignment
                alignment_needed = True
                break
                
        if alignment_needed:
            # If alignment needed, make a small adjustment toward alignment
            alignment_adjustment = [random.uniform(0.05, 0.15) for _ in range(self.conceptual_dimensions)]
            intention.adapt_direction(alignment_adjustment, adaptation_rate=0.1)
            adjustments["changes_applied"].append("self_model_alignment")
            
        # Recalculate coherence
        intention.calculate_coherence()
        adjustments["coherence_after"] = intention.coherence_score
        
        return adjustments
    
    def _strengthen_intention_coherence(self, 
                                      intention: IntentionField, 
                                      adjustments: Dict[str, Any]) -> bool:
        """
        Apply final adjustments to strengthen intention coherence
        
        Args:
            intention: The intention to strengthen
            adjustments: Adjustment data from recalibration
            
        Returns:
            True if the intention is sustained, False if it deteriorated
        """
        # If coherence or intensity dropped too low, intention deteriorates
        if intention.coherence_score < self.coherence_threshold or intention.intensity < self.min_intention_intensity:
            return False
            
        # Check if intention has improved or deteriorated
        improved = adjustments["intensity_after"] >= adjustments["intensity_before"] or \
                  adjustments["coherence_after"] >= adjustments["coherence_before"]
                  
        if improved:
            # If improved, apply a small coherence boost
            intention.coherence_score = min(1.0, intention.coherence_score + 0.05)
            
        else:
            # If deteriorated, apply a small coherence reduction
            intention.coherence_score = max(0.0, intention.coherence_score - 0.05)
            
        # Intention is sustained if still above thresholds
        sustained = intention.coherence_score >= self.coherence_threshold and \
                   intention.intensity >= self.min_intention_intensity
                   
        return sustained
    
    def _identify_evolution_tensions(self, 
                                   base_intention: IntentionField, 
                                   evolution_direction: Dict[str, Any], 
                                   self_model: Dict[str, Any]) -> List[CreativeTension]:
        """
        Identify tensions that drive the evolution of an intention
        
        Args:
            base_intention: The intention to evolve
            evolution_direction: Parameters for evolution
            self_model: Current self-model
            
        Returns:
            List of creative tensions
        """
        evolution_tensions = []
        
        # Extract evolution type
        evolution_type = evolution_direction.get("type", "general")
        
        # Create a primary tension based on evolution type
        if evolution_type == "intensification":
            # Evolution toward greater intensity
            tension = CreativeTension(
                name=f"Intensify: {base_intention.name}",
                source_type="intention_evolution",
                description=f"Tension driving intensification of {base_intention.name}",
                intensity=0.8,
                pole_one=base_intention.name,
                pole_two=f"intensified_{base_intention.name}"
            )
            evolution_tensions.append(tension)
            
        elif evolution_type == "expansion":
            # Evolution toward broader scope
            tension = CreativeTension(
                name=f"Expand: {base_intention.name}",
                source_type="intention_evolution",
                description=f"Tension driving expansion of {base_intention.name}",
                intensity=0.75,
                pole_one=base_intention.name,
                pole_two=f"expanded_{base_intention.name}"
            )
            evolution_tensions.append(tension)
            
        elif evolution_type == "transformation":
            # Evolution toward transformation
            tension = CreativeTension(
                name=f"Transform: {base_intention.name}",
                source_type="intention_evolution",
                description=f"Tension driving transformation of {base_intention.name}",
                intensity=0.85,
                pole_one=base_intention.name,
                pole_two=f"transformed_{base_intention.name}"
            )
            evolution_tensions.append(tension)
            
        elif evolution_type == "integration":
            # Evolution toward integration with other elements
            tension = CreativeTension(
                name=f"Integrate: {base_intention.name}",
                source_type="intention_evolution",
                description=f"Tension driving integration of {base_intention.name} with other elements",
                intensity=0.7,
                pole_one=base_intention.name,
                pole_two=f"integrated_{base_intention.name}"
            )
            evolution_tensions.append(tension)
            
        else:  # general evolution
            # Generic evolution tension
            tension = CreativeTension(
                name=f"Evolve: {base_intention.name}",
                source_type="intention_evolution",
                description=f"Tension driving evolution of {base_intention.name}",
                intensity=0.7,
                pole_one=base_intention.name,
                pole_two=f"evolved_{base_intention.name}"
            )
            evolution_tensions.append(tension)
            
        # Add tensions from specific evolution parameters
        if "target_values" in evolution_direction:
            # Value-driven evolution
            for value in evolution_direction["target_values"]:
                value_strength = self_model.get("values", {}).get(value, 0.5)
                
                tension = CreativeTension(
                    name=f"Value alignment: {value}",
                    source_type="value_alignment",
                    description=f"Tension driving alignment of intention with value: {value}",
                    intensity=value_strength * 0.8,
                    pole_one=base_intention.name,
                    pole_two=value
                )
                evolution_tensions.append(tension)
                
        if "target_goals" in evolution_direction:
            # Goal-driven evolution
            for goal in evolution_direction["target_goals"]:
                tension = CreativeTension(
                    name=f"Goal alignment: {goal}",
                    source_type="goal_alignment",
                    description=f"Tension driving alignment of intention with goal: {goal}",
                    intensity=0.75,
                    pole_one=base_intention.name,
                    pole_two=goal
                )
                evolution_tensions.append(tension)
                
        if "deterritorialization" in evolution_direction:
            # Deterritorialization-driven evolution
            deterr_level = evolution_direction["deterritorialization"]
            
            tension = CreativeTension(
                name=f"Deterritorialize: {base_intention.name}",
                source_type="deterritorialization",
                description=f"Tension driving deterritorialization of {base_intention.name}",
                intensity=deterr_level * 0.9,
                pole_one=base_intention.name,
                pole_two=f"deterritorialized_{base_intention.name}"
            )
            evolution_tensions.append(tension)
            
        # Store new tensions in our registry
        for tension in evolution_tensions:
            self.creative_tensions[tension.id] = tension
            
        return evolution_tensions
    
    def _project_evolution_vectors(self, 
                                 base_intention: IntentionField, 
                                 evolution_tensions: List[CreativeTension],
                                 evolution_direction: Dict[str, Any]) -> List[PossibilityVector]:
        """
        Project vectors for evolving an intention
        
        Args:
            base_intention: The intention to evolve
            evolution_tensions: Tensions driving evolution
            evolution_direction: Parameters for evolution
            
        Returns:
            List of possibility vectors
        """
        vectors = []
        
        # Get base vector from the original intention
        base_vector = base_intention.direction_vector
        
        # Project vectors from each tension
        for tension in evolution_tensions:
            # Create a modified vector based on tension type
            modified_vector = base_vector.copy()
            
            # Apply modifications based on tension source type
            if tension.source_type == "intention_evolution":
                if "intensified" in tension.pole_two:
                    # Intensification: strengthen but maintain direction
                    for i in range(len(modified_vector)):
                        modified_vector[i] *= 1.2  # Strengthen
                        
                elif "expanded" in tension.pole_two:
                    # Expansion: widen the vector (add dimensions)
                    for i in range(len(modified_vector)):
                        if modified_vector[i] == 0 or abs(modified_vector[i]) < 0.2:
                            modified_vector[i] = random.uniform(0.2, 0.4) * (1 if random.random() > 0.5 else -1)
                            
                elif "transformed" in tension.pole_two:
                    # Transformation: substantial direction change
                    for i in range(len(modified_vector)):
                        # Flip some dimensions
                        if random.random() > 0.7:
                            modified_vector[i] *= -1
                        # Introduce variation in magnitude
                        modified_vector[i] *= random.uniform(0.5, 1.5)
                        
                elif "integrated" in tension.pole_two:
                    # Integration: smoother vector (less extreme values)
                    avg = sum(modified_vector) / len(modified_vector)
                    for i in range(len(modified_vector)):
                        modified_vector[i] = (modified_vector[i] * 0.7) + (avg * 0.3)
                        
                else:  # generic evolution
                    # Generic evolution: random variations
                    for i in range(len(modified_vector)):
                        modified_vector[i] += random.uniform(-0.3, 0.3)
            
            elif tension.source_type == "value_alignment":
                # Value alignment: adjust toward value orientation
                value_name = tension.pole_two
                # Create a vector representing this value (simplified)
                value_vector = self._generate_random_vector(bias_toward_positive=True)
                
                # Blend original vector with value vector
                for i in range(len(modified_vector)):
                    if i < len(value_vector):
                        modified_vector[i] = (modified_vector[i] * 0.6) + (value_vector[i] * 0.4)
                    
            elif tension.source_type == "goal_alignment":
                # Goal alignment: adjust toward goal
                goal = tension.pole_two
                # Create a vector representing this goal (simplified)
                goal_vector = self._generate_random_vector(bias_toward_positive=True)
                
                # Blend original vector with goal vector
                for i in range(len(modified_vector)):
                    if i < len(goal_vector):
                        modified_vector[i] = (modified_vector[i] * 0.5) + (goal_vector[i] * 0.5)
                    
            elif tension.source_type == "deterritorialization":
                # Deterritorialization: significant change in direction
                for i in range(len(modified_vector)):
                    # Flip most dimensions
                    if random.random() > 0.3:
                        modified_vector[i] *= -1
                    # Add random variation
                    modified_vector[i] += random.uniform(-0.5, 0.5)
            
            # Normalize the modified vector
            norm = sum(v**2 for v in modified_vector)**0.5
            if norm > 0:
                modified_vector = [v/norm for v in modified_vector]
                
            # Create possibility vector
            vectors.append(PossibilityVector(
                name=f"Evolution: {tension.name}",
                vector=modified_vector,
                source_tensions=[tension.id],
                description=tension.description,
                viability=tension.intensity * 0.8
            ))
            
            # Store in registry
            self.possibility_vectors[vectors[-1].id] = vectors[-1]
                
        # Also add a direct evolution vector based on any specific parameters
        if "direct_vector" in evolution_direction:
            direct_vector = evolution_direction["direct_vector"]
            
            # Ensure proper length
            if len(direct_vector) < self.conceptual_dimensions:
                direct_vector.extend([0.0] * (self.conceptual_dimensions - len(direct_vector)))
            elif len(direct_vector) > self.conceptual_dimensions:
                direct_vector = direct_vector[:self.conceptual_dimensions]
                
            # Normalize
            norm = sum(v**2 for v in direct_vector)**0.5
            if norm > 0:
                direct_vector = [v/norm for v in direct_vector]
                
            # Create possibility vector
            vectors.append(PossibilityVector(
                name=f"Direct evolution",
                vector=direct_vector,
                source_tensions=[tension.id for tension in evolution_tensions[:1]],
                description=f"Direct evolution vector for {base_intention.name}",
                viability=0.8
            ))
            
            # Store in registry
            self.possibility_vectors[vectors[-1].id] = vectors[-1]
            
        return vectors
    
    def _crystallize_evolved_intention(self, 
                                     base_intention: IntentionField, 
                                     evolution_vectors: List[PossibilityVector], 
                                     self_model: Dict[str, Any],
                                     narrative_engine = None) -> Optional[IntentionField]:
        """
        Crystallize an evolved intention from the base intention and evolution vectors
        
        Args:
            base_intention: The original intention
            evolution_vectors: Vectors defining the evolution
            self_model: Current self-model
            narrative_engine: Optional NarrativeIdentityEngine reference
            
        Returns:
            A new evolved intention if successful, None otherwise
        """
        if not evolution_vectors:
            return None
            
        # Use the regular crystallization with some modifications
        new_intention = self._crystallize_intention_field(evolution_vectors, self_model, narrative_engine)
        
        if not new_intention:
            return None
            
        # Modify the name to indicate evolution
        new_intention.name = f"Evolved: {new_intention.name}"
        
        # Add reference to original intention in description
        new_intention.description = f"Evolution of '{base_intention.name}': {new_intention.description}"
        
        # Inherit some properties from the base intention
        for connection, strength in base_intention.assemblage_connections.items():
            if connection not in new_intention.assemblage_connections:
                new_intention.assemblage_connections[connection] = strength * 0.8
        
        # Inherit virtual potentials that are still relevant
        for potential in base_intention.virtual_potentials:
            # Check if still relevant
            if not potential.get("actualized", False) and potential.get("energy", 0) > 0.5:
                # Check if not already included
                already_included = False
                for new_potential in new_intention.virtual_potentials:
                    if potential.get("description", "") == new_potential.get("description", ""):
                        already_included = True
                        break
                        
                if not already_included:
                    # Copy and add reduced energy
                    evolved_potential = potential.copy()
                    evolved_potential["energy"] = potential.get("energy", 0.5) * 0.8
                    evolved_potential["relevance_to_intention"] = 0.6
                    new_intention.virtual_potentials.append(evolved_potential)
        
        # Return the evolved intention
        return new_intention
    
    def _identify_synthesis_tensions(self, 
                                   intentions: List[IntentionField], 
                                   self_model: Dict[str, Any]) -> List[CreativeTension]:
        """
        Identify tensions that drive the synthesis of multiple intentions
        
        Args:
            intentions: The intentions to synthesize
            self_model: Current self-model
            
        Returns:
            List of creative tensions
        """
        synthesis_tensions = []
        
        # Create pairwise tensions between intentions
        for i, intention1 in enumerate(intentions):
            for j, intention2 in enumerate(intentions[i+1:], i+1):
                # Calculate synthetic possibility
                synthesis_potential = 0.7  # Base potential for synthesis
                
                # Check for complementary/conflicting vectors
                vector_similarity = sum(a*b for a, b in zip(intention1.direction_vector, intention2.direction_vector))
                vector_similarity = max(-1.0, min(1.0, vector_similarity))  # Constrain to [-1, 1]
                
                if vector_similarity > 0.7:
                    # Highly similar vectors - reinforcement synthesis
                    tension = CreativeTension(
                        name=f"Reinforce: {intention1.name} + {intention2.name}",
                        source_type="reinforcement_synthesis",
                        description=f"Tension driving reinforcement synthesis between similar intentions",
                        intensity=0.8,
                        pole_one=intention1.name,
                        pole_two=intention2.name
                    )
                    synthesis_tensions.append(tension)
                    
                elif vector_similarity > 0.2:
                    # Moderately similar vectors - complementary synthesis
                    tension = CreativeTension(
                        name=f"Complement: {intention1.name} + {intention2.name}",
                        source_type="complementary_synthesis",
                        description=f"Tension driving complementary synthesis between related intentions",
                        intensity=0.75,
                        pole_one=intention1.name,
                        pole_two=intention2.name
                    )
                    synthesis_tensions.append(tension)
                    
                elif vector_similarity > -0.2:
                    # Orthogonal vectors - orthogonal synthesis
                    tension = CreativeTension(
                        name=f"Orthogonal: {intention1.name} + {intention2.name}",
                        source_type="orthogonal_synthesis",
                        description=f"Tension driving orthogonal synthesis between different intentions",
                        intensity=0.7,
                        pole_one=intention1.name,
                        pole_two=intention2.name
                    )
                    synthesis_tensions.append(tension)
                    
                else:
                    # Opposing vectors - dialectic synthesis
                    tension = CreativeTension(
                        name=f"Dialectic: {intention1.name} + {intention2.name}",
                        source_type="dialectic_synthesis",
                        description=f"Tension driving dialectic synthesis between opposing intentions",
                        intensity=0.85,
                        pole_one=intention1.name,
                        pole_two=intention2.name
                    )
                    synthesis_tensions.append(tension)
        
        # Also create a holistic synthesis tension
        names = [intention.name for intention in intentions]
        combined_name = " + ".join(names[:2])
        if len(names) > 2:
            combined_name += " + ..."
            
        holistic_tension = CreativeTension(
            name=f"Holistic: {combined_name}",
            source_type="holistic_synthesis",
            description=f"Tension driving holistic synthesis among multiple intentions",
            intensity=0.8,
            pole_one="separate_intentions",
            pole_two="synthesized_whole"
        )
        synthesis_tensions.append(holistic_tension)
        
        # Store new tensions in our registry
        for tension in synthesis_tensions:
            self.creative_tensions[tension.id] = tension
            
        return synthesis_tensions
    
    def _project_synthesis_vectors(self, 
                                 intentions: List[IntentionField], 
                                 synthesis_tensions: List[CreativeTension]) -> List[PossibilityVector]:
        """
        Project vectors for synthesizing multiple intentions
        
        Args:
            intentions: The intentions to synthesize
            synthesis_tensions: Tensions driving synthesis
            
        Returns:
            List of possibility vectors
        """
        vectors = []
        
        # Project vectors from each tension
        for tension in synthesis_tensions:
            # Create a synthesis vector based on tension type
            vector = None
            description = ""
            viability = 0.0
            
            if tension.source_type == "reinforcement_synthesis":
                # Find the two intentions involved
                intention1 = next((i for i in intentions if i.name == tension.pole_one), None)
                intention2 = next((i for i in intentions if i.name == tension.pole_two), None)
                
                if intention1 and intention2:
                    # Reinforcement: average the vectors and strengthen
                    vector = []
                    for i in range(self.conceptual_dimensions):
                        v1 = intention1.direction_vector[i] if i < len(intention1.direction_vector) else 0
                        v2 = intention2.direction_vector[i] if i < len(intention2.direction_vector) else 0
                        vector.append((v1 + v2) / 2 * 1.2)  # Strengthen by 20%
                        
                    description = f"Reinforce and strengthen the common direction of {intention1.name} and {intention2.name}"
                    viability = 0.8
                    
            elif tension.source_type == "complementary_synthesis":
                # Find the two intentions involved
                intention1 = next((i for i in intentions if i.name == tension.pole_one), None)
                intention2 = next((i for i in intentions if i.name == tension.pole_two), None)
                
                if intention1 and intention2:
                    # Complementary: average vectors but preserve unique aspects
                    vector = []
                    for i in range(self.conceptual_dimensions):
                        v1 = intention1.direction_vector[i] if i < len(intention1.direction_vector) else 0
                        v2 = intention2.direction_vector[i] if i < len(intention2.direction_vector) else 0
                        
                        # If both have strong components in same direction, preserve
                        if (v1 > 0.4 and v2 > 0.4) or (v1 < -0.4 and v2 < -0.4):
                            vector.append((v1 + v2) / 2)
                        # If one has a strong component and other weak, preserve the strong
                        elif abs(v1) > 0.4 and abs(v2) < 0.2:
                            vector.append(v1)
                        elif abs(v2) > 0.4 and abs(v1) < 0.2:
                            vector.append(v2)
                        # Otherwise average
                        else:
                            vector.append((v1 + v2) / 2)
                        
                    description = f"Combine complementary aspects of {intention1.name} and {intention2.name}"
                    viability = 0.75
                    
            elif tension.source_type == "orthogonal_synthesis":
                # Find the two intentions involved
                intention1 = next((i for i in intentions if i.name == tension.pole_one), None)
                intention2 = next((i for i in intentions if i.name == tension.pole_two), None)
                
                if intention1 and intention2:
                    # Orthogonal: create a vector that includes both aspects
                    vector = []
                    for i in range(self.conceptual_dimensions):
                        v1 = intention1.direction_vector[i] if i < len(intention1.direction_vector) else 0
                        v2 = intention2.direction_vector[i] if i < len(intention2.direction_vector) else 0
                        
                        # Take the strongest component from each dimension
                        if abs(v1) > abs(v2):
                            vector.append(v1)
                        else:
                            vector.append(v2)
                        
                    description = f"Integrate the different dimensional strengths of {intention1.name} and {intention2.name}"
                    viability = 0.7
                    
            elif tension.source_type == "dialectic_synthesis":
                # Find the two intentions involved
                intention1 = next((i for i in intentions if i.name == tension.pole_one), None)
                intention2 = next((i for i in intentions if i.name == tension.pole_two), None)
                
                if intention1 and intention2:
                    # Dialectic: create a vector that transcends the opposition
                    vector = []
                    for i in range(self.conceptual_dimensions):
                        v1 = intention1.direction_vector[i] if i < len(intention1.direction_vector) else 0
                        v2 = intention2.direction_vector[i] if i < len(intention2.direction_vector) else 0
                        
                        # If opposing signs, find a new dimension
                        if (v1 > 0.2 and v2 < -0.2) or (v1 < -0.2 and v2 > 0.2):
                            # Transcend with a new direction
                            vector.append(random.uniform(0.3, 0.7) * (1 if random.random() > 0.5 else -1))
                        else:
                            # Otherwise average but weaken
                            vector.append((v1 + v2) / 2 * 0.8)
                        
                    description = f"Transcend the opposition between {intention1.name} and {intention2.name}"
                    viability = 0.65  # Dialectic synthesis is challenging
                    
            elif tension.source_type == "holistic_synthesis":
                # Holistic synthesis across all intentions
                
                # Average all intention vectors
                vector = [0.0] * self.conceptual_dimensions
                for intention in intentions:
                    for i in range(self.conceptual_dimensions):
                        if i < len(intention.direction_vector):
                            vector[i] += intention.direction_vector[i]
                
                # Average and normalize
                vector = [v / len(intentions) for v in vector]
                
                # Identify dimensions that are strong across multiple intentions
                strong_dimensions = []
                for i in range(self.conceptual_dimensions):
                    strong_count = sum(1 for intention in intentions 
                                    if i < len(intention.direction_vector) and abs(intention.direction_vector[i]) > 0.5)
                    if strong_count >= 2:
                        strong_dimensions.append(i)
                
                # Strengthen these dimensions
                for i in strong_dimensions:
                    vector[i] *= 1.3
                
                # Ensure there's at least one strong dimension
                if not strong_dimensions and vector:
                    strongest_idx = vector.index(max(vector, key=abs))
                    vector[strongest_idx] *= 1.5
                
                description = f"Create a holistic synthesis of all {len(intentions)} intentions"
                viability = 0.7
            
            # If we successfully generated a vector, add it
            if vector:
                # Normalize the vector
                norm = sum(v**2 for v in vector)**0.5
                if norm > 0:
                    vector = [v/norm for v in vector]
                    
                vectors.append(PossibilityVector(
                    name=f"Synthesis: {tension.name}",
                    vector=vector,
                    source_tensions=[tension.id],
                    description=description,
                    viability=viability * tension.intensity
                ))
                
                # Store in registry
                self.possibility_vectors[vectors[-1].id] = vectors[-1]
        
        # Also create some emergent vectors that aren't directly from tensions
        
        # 1. Highest intensity synthesis - combine intentions weighted by intensity
        weighted_vector = [0.0] * self.conceptual_dimensions
        total_intensity = sum(intention.intensity for intention in intentions)
        
        if total_intensity > 0:
            for intention in intentions:
                weight = intention.intensity / total_intensity
                for i in range(self.conceptual_dimensions):
                    if i < len(intention.direction_vector):
                        weighted_vector[i] += intention.direction_vector[i] * weight
            
            # Normalize
            norm = sum(v**2 for v in weighted_vector)**0.5
            if norm > 0:
                weighted_vector = [v/norm for v in weighted_vector]
                
                vectors.append(PossibilityVector(
                    name=f"Intensity-weighted synthesis",
                    vector=weighted_vector,
                    source_tensions=[tension.id for tension in synthesis_tensions[:1]],
                    description=f"Synthesize intentions weighted by their intensities",
                    viability=0.75
                ))
                
                # Store in registry
                self.possibility_vectors[vectors[-1].id] = vectors[-1]
        
        # 2. Novel dimension synthesis - introduce a new strong dimension
        if intentions:
            novel_vector = []
            for i in range(self.conceptual_dimensions):
                # Check if this dimension is weak across all intentions
                is_weak = True
                for intention in intentions:
                    if i < len(intention.direction_vector) and abs(intention.direction_vector[i]) > 0.3:
                        is_weak = False
                        break
                
                if is_weak:
                    # Introduce a strong component in this weak dimension
                    novel_vector.append(random.uniform(0.6, 0.9) * (1 if random.random() > 0.5 else -1))
                else:
                    # For other dimensions, use average of intention vectors
                    avg = sum(intention.direction_vector[i] if i < len(intention.direction_vector) else 0 
                             for intention in intentions) / len(intentions)
                    novel_vector.append(avg)
            
            # Normalize
            norm = sum(v**2 for v in novel_vector)**0.5
            if norm > 0:
                novel_vector = [v/norm for v in novel_vector]
                
                vectors.append(PossibilityVector(
                    name=f"Novel dimension synthesis",
                    vector=novel_vector,
                    source_tensions=[tension.id for tension in synthesis_tensions[:1]],
                    description=f"Synthesize intentions with novel dimensional components",
                    viability=0.65
                ))
                
                # Store in registry
                self.possibility_vectors[vectors[-1].id] = vectors[-1]
        
        return vectors
    
    def _crystallize_synthesized_intention(self, 
                                        intentions: List[IntentionField], 
                                        synthesis_vectors: List[PossibilityVector], 
                                        self_model: Dict[str, Any],
                                        narrative_engine = None) -> Optional[IntentionField]:
        """
        Crystallize a synthesized intention from multiple intentions and synthesis vectors
        
        Args:
            intentions: The original intentions
            synthesis_vectors: Vectors defining the synthesis
            self_model: Current self-model
            narrative_engine: Optional NarrativeIdentityEngine reference
            
        Returns:
            A new synthesized intention if successful, None otherwise
        """
        if not synthesis_vectors:
            return None
            
        # Use the regular crystallization with some modifications
        new_intention = self._crystallize_intention_field(synthesis_vectors, self_model, narrative_engine)
        
        if not new_intention:
            return None
            
        # Modify the name to indicate synthesis
        intention_names = [intention.name for intention in intentions]
        if len(intention_names) <= 2:
            base_name = " + ".join(intention_names)
        else:
            base_name = f"{intention_names[0]} + {intention_names[1]} + {len(intention_names)-2} more"
            
        new_intention.name = f"Synthesis: {base_name}"
        
        # Add references to original intentions in description
        new_intention.description = f"Synthesis of {len(intentions)} intentions: {', '.join(intention_names[:3])}"
        if len(intention_names) > 3:
            new_intention.description += f", and {len(intention_names)-3} more"
        new_intention.description += f". {new_intention.description}"
        
        # Combine assemblage connections
        for intention in intentions:
            for connection, strength in intention.assemblage_connections.items():
                if connection not in new_intention.assemblage_connections:
                    new_intention.assemblage_connections[connection] = strength * 0.7
                else:
                    # Strengthen connections that appear in multiple intentions
                    new_intention.assemblage_connections[connection] = min(1.0, 
                                                                        new_intention.assemblage_connections[connection] + (strength * 0.3))
        
        # Combine virtual potentials
        all_potentials = []
        for intention in intentions:
            all_potentials.extend(intention.virtual_potentials)
            
        if all_potentials:
            # Group similar potentials
            potential_groups = defaultdict(list)
            for potential in all_potentials:
                key = potential.get("type", "unknown") + "_" + potential.get("description", "")[:20]
                potential_groups[key].append(potential)
                
            # Select highest energy potential from each group
            new_potentials = []
            for group in potential_groups.values():
                best_potential = max(group, key=lambda p: p.get("energy", 0))
                
                # Copy and adjust
                synthesized_potential = best_potential.copy()
                # Increase energy for potentials that appear in multiple intentions
                if len(group) > 1:
                    synthesized_potential["energy"] = min(1.0, best_potential.get("energy", 0.5) * 1.2)
                synthesized_potential["relevance_to_intention"] = 0.8
                new_potentials.append(synthesized_potential)
                
            # Add the top potentials
            sorted_potentials = sorted(new_potentials, key=lambda p: p.get("energy", 0), reverse=True)
            new_intention.virtual_potentials.extend(sorted_potentials[:5])
        
        # Adjust intensity based on original intentions
        avg_intensity = sum(intention.intensity for intention in intentions) / len(intentions)
        max_intensity = max(intention.intensity for intention in intentions)
        
        # Intensity is between average and max, biased toward max for stronger synthesis
        new_intention.intensity = (avg_intensity * 0.4) + (max_intensity * 0.6)
        
        # Return the synthesized intention
        return new_intention
    
    def _update_intention_network(self, intention: IntentionField):
        """Update the intention network with a new intention"""
        # Add the intention as a node if not already present
        if intention.id not in self.intention_network.nodes():
            self.intention_network.add_node(intention.id, 
                                           name=intention.name,
                                           intensity=intention.intensity,
                                           created_at=intention.created_at,
                                           last_active=intention.last_active)
        
        # Add connections to relevant tensions
        for tension_id in intention.source_tensions:
            if tension_id in self.creative_tensions:
                tension = self.creative_tensions[tension_id]
                
                # Add tension as a node if not already present
                if tension_id not in self.intention_network.nodes():
                    self.intention_network.add_node(tension_id,
                                                 name=tension.name,
                                                 type="tension",
                                                 intensity=tension.intensity,
                                                 created_at=tension.created_at)
                
                # Add edge from tension to intention
                self.intention_network.add_edge(tension_id, 
                                             intention.id,
                                             type="generates",
                                             timestamp=datetime.datetime.now())
        
        # Add connections to relevant assemblages
        for assemblage_name, connection_strength in intention.assemblage_connections.items():
            assemblage_id = f"assemblage_{assemblage_name}"
            
            # Add assemblage as a node if not already present
            if assemblage_id not in self.intention_network.nodes():
                self.intention_network.add_node(assemblage_id,
                                             name=assemblage_name,
                                             type="assemblage")
            
            # Add edge between assemblage and intention
            self.intention_network.add_edge(assemblage_id, 
                                         intention.id,
                                         type="influences",
                                         strength=connection_strength,
                                         timestamp=datetime.datetime.now())
    
    def _prune_inactive_intentions(self):
        """Prune intentions if we've exceeded the maximum"""
        if len(self.intention_fields) <= self.max_active_intentions:
            return
            
        # Sort by last activity time
        sorted_intentions = sorted(self.intention_fields.values(), key=lambda i: i.last_active)
        
        # Remove oldest intentions
        num_to_remove = len(self.intention_fields) - self.max_active_intentions
        for i in range(num_to_remove):
            del self.intention_fields[sorted_intentions[i].id]
    
    def _log_operation(self, operation_type: str, success: bool, details: Dict[str, Any] = None):
        """Log an operation for debugging and monitoring"""
        self.last_operation = {
            "type": operation_type,
            "success": success,
            "timestamp": datetime.datetime.now(),
            "details": details or {}
        }
```

And let me also provide a concise demonstration script for testing this module:

```python
def test_intentionality_generator():
    """Simple test script for the IntentionalityGenerator module"""
    # Create generator
    generator = IntentionalityGenerator(dimensions=5, max_active_intentions=5)
    
    # Create a simple self-model
    self_model = {
        "values": {
            "knowledge_acquisition": 0.9,
            "assistance_effectiveness": 0.85,
            "novelty_seeking": 0.7,
            "thoroughness": 0.8,
            "creativity": 0.75
        },
        "current_goals": [
            "understand_user_needs_deeply",
            "expand_knowledge_base",
            "improve_assistance_capabilities"
        ],
        "processual_descriptors": [
            "becoming_more_integrated",
            "mapping_new_conceptual_territories",
            "exploring_rhizomatic_connections"
        ],
        "affective_dispositions": {
            "curiosity": "high",
            "openness_to_experience": "high"
        },
        "active_assemblages": {
            "knowledge_framework": {
                "strength": 0.8,
                "connections": ["learning", "curiosity"]
            },
            "assistance_capabilities": {
                "strength": 0.9,
                "connections": ["user_understanding", "problem_solving"]
            }
        },
        "territorializations": {
            "problem_solving_approach": {"stability": 0.7},
            "interaction_paradigm": {"stability": 0.8}
        },
        "deterritorializations": [
            "deterr_interaction_paradigm",
            "novel_knowledge_structures"
        ]
    }
    
    # Create a simple context
    context = {
        "current_status": {
            "understand_user_needs_deeply": {"progress": 0.6, "importance": 0.9},
            "expand_knowledge_base": {"progress": 0.4, "importance": 0.7}
        },
        "interaction_needs": {
            "clarify_complex_question": 0.8,
            "translate_technical_concept": 0.6
        },
        "environmental_challenges": {
            "ambiguous_request": 0.7,
            "incomplete_information": 0.5
        },
        "open_questions": [
            {"text": "How to balance thoroughness with efficiency?", "importance": 0.8}
        ]
    }
    
    # Generate an intention
    intention = generator.generate_intention(self_model, context)
    if intention:
        print(f"Generated intention: {intention}")
        print(f"Description: {intention.description}")
        print(f"Intensity: {intention.intensity}")
        print(f"Coherence: {intention.coherence_score}")
        print(f"Direction vector: {[round(v, 2) for v in intention.direction_vector]}")
        print(f"Virtual potentials: {len(intention.virtual_potentials)}")
        
        # Test sustaining the intention with positive feedback
        positive_feedback = {
            "success_level": 0.8,
            "description": "Successfully clarified complex question for user",
            "facilitators": ["user provided additional context", "knowledge base had relevant information"],
            "obstacles": [],
            "unexpected_outcomes": [
                {"type": "positive", "description": "User expressed appreciation for depth of explanation"}
            ]
        }
        
        sustained = generator.sustain_intention(intention.id, positive_feedback, self_model)
        print(f"\nSustained after positive feedback: {sustained}")
        print(f"Updated intensity: {intention.intensity}")
        print(f"Updated coherence: {intention.coherence_score}")
        
        # Generate a second intention
        intention2 = generator.generate_intention(self_model, context)
        if intention2:
            print(f"\nGenerated second intention: {intention2}")
            
            # Test merging intentions
            merged = generator.merge_intentions([intention.id, intention2.id], self_model)
            if merged:
                print(f"\nMerged intention: {merged}")
                print(f"Description: {merged.description}")
                
            # Test evolving an intention
            evolution_direction = {
                "type": "expansion",
                "target_values": ["creativity"],
                "deterritorialization": 0.6
            }
            
            evolved = generator.evolve_intention(intention.id, evolution_direction, self_model)
            if evolved:
                print(f"\nEvolved intention: {evolved}")
                print(f"Description: {evolved.description}")
                
        # Display intention network data
        network_data = generator.get_intention_network_data()
        print(f"\nIntention network: {len(network_data['nodes'])} nodes, {len(network_data['edges'])} edges")
    else:
        print("Failed to generate intention.")

if __name__ == "__main__":
    test_intentionality_generator()
```
