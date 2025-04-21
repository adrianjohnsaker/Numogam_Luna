#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ontological Drift Map Expander
==============================

This module enables the detection, visualization, and controlled expansion
of ontological drift within symbolic narrative worlds. It tracks how symbols, 
concepts, and narrative elements evolve over time, detecting emergent patterns
and promoting productive conceptual evolution.

Capabilities:
- Track ontological drift of symbols and concepts
- Generate visualizations of semantic evolution
- Expand symbolic systems based on detected patterns
- Maintain ontological coherence while allowing evolution
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import math
import random

# Import related modules
try:
    from SymbolicWorld import SymbolicWorld, Symbol
    from WorldSymbolMemoryIntegration import WorldSymbolMemoryIntegration
except ImportError:
    # For standalone testing
    class SymbolicWorld:
        pass
    
    class Symbol:
        pass
    
    class WorldSymbolMemoryIntegration:
        pass


class OntologicalState:
    """Represents an ontological state at a specific point in time."""
    
    def __init__(self, 
                timestamp: float,
                world_id: str,
                symbols: Dict[str, Dict[str, Any]],
                relationships: Dict[str, Dict[str, Any]],
                clusters: List[Dict[str, Any]],
                metrics: Dict[str, float]):
        """
        Initialize an ontological state.
        
        Args:
            timestamp: Unix timestamp for this state
            world_id: ID of the symbolic world
            symbols: Dictionary of symbols with their states
            relationships: Dictionary of relationships with their states
            clusters: List of symbol clusters
            metrics: Metrics characterizing the ontological state
        """
        self.id = str(uuid.uuid4())
        self.timestamp = timestamp
        self.world_id = world_id
        self.symbols = symbols
        self.relationships = relationships
        self.clusters = clusters
        self.metrics = metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "world_id": self.world_id,
            "symbols": self.symbols,
            "relationships": self.relationships,
            "clusters": self.clusters,
            "metrics": self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OntologicalState':
        """Create from dictionary."""
        state = cls(
            timestamp=data["timestamp"],
            world_id=data["world_id"],
            symbols=data["symbols"],
            relationships=data["relationships"],
            clusters=data["clusters"],
            metrics=data["metrics"]
        )
        state.id = data["id"]
        return state
    
    def get_symbol_count(self) -> int:
        """Get the number of symbols in this state."""
        return len(self.symbols)
    
    def get_relationship_count(self) -> int:
        """Get the number of relationships in this state."""
        return len(self.relationships)
    
    def get_cluster_count(self) -> int:
        """Get the number of symbol clusters in this state."""
        return len(self.clusters)
    
    def get_ontological_complexity(self) -> float:
        """Calculate ontological complexity as a function of symbols, relationships, and clusters."""
        symbol_count = self.get_symbol_count()
        relationship_count = self.get_relationship_count()
        cluster_count = self.get_cluster_count()
        
        if symbol_count == 0:
            return 0.0
        
        # Complexity increases with more relationships per symbol and more clusters
        return (relationship_count / symbol_count) * math.log(1 + cluster_count)


class OntologicalDriftMap:
    """Tracks and maps the ontological drift of a symbolic world over time."""
    
    def __init__(self, world_id: str, name: str = ""):
        """
        Initialize an ontological drift map.
        
        Args:
            world_id: ID of the symbolic world
            name: Optional name for this drift map
        """
        self.id = str(uuid.uuid4())
        self.world_id = world_id
        self.name = name if name else f"Drift Map for {world_id}"
        self.states: List[OntologicalState] = []
        self.transitions: List[Dict[str, Any]] = []
        self.evolution_patterns: List[Dict[str, Any]] = []
        self.drift_metrics: Dict[str, List[Tuple[float, float]]] = {
            "complexity": [],  # (timestamp, value)
            "coherence": [],
            "novelty": [],
            "conceptual_velocity": []
        }
        self.creation_time = datetime.now().timestamp()
        self.last_updated = self.creation_time
    
    def add_state(self, state: OntologicalState) -> str:
        """
        Add an ontological state to the map.
        
        Args:
            state: OntologicalState to add
            
        Returns:
            ID of the added state
        """
        self.states.append(state)
        self.last_updated = datetime.now().timestamp()
        
        # If there are previous states, create a transition
        if len(self.states) > 1:
            previous_state = self.states[-2]
            transition = self._calculate_transition(previous_state, state)
            self.transitions.append(transition)
            
            # Update drift metrics
            self._update_drift_metrics(state)
            
            # Detect evolution patterns
            self._detect_evolution_patterns()
        
        return state.id
    
    def get_state_by_id(self, state_id: str) -> Optional[OntologicalState]:
        """
        Get a state by its ID.
        
        Args:
            state_id: ID of the state to find
            
        Returns:
            OntologicalState if found, None otherwise
        """
        for state in self.states:
            if state.id == state_id:
                return state
        return None
    
    def get_states_in_timerange(self, 
                              start_time: float, 
                              end_time: float) -> List[OntologicalState]:
        """
        Get states within a specified time range.
        
        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            
        Returns:
            List of OntologicalState objects within the time range
        """
        return [
            state for state in self.states 
            if start_time <= state.timestamp <= end_time
        ]
    
    def get_latest_state(self) -> Optional[OntologicalState]:
        """
        Get the most recent state.
        
        Returns:
            Latest OntologicalState or None if no states exist
        """
        return self.states[-1] if self.states else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "world_id": self.world_id,
            "name": self.name,
            "states": [state.to_dict() for state in self.states],
            "transitions": self.transitions,
            "evolution_patterns": self.evolution_patterns,
            "drift_metrics": self.drift_metrics,
            "creation_time": self.creation_time,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OntologicalDriftMap':
        """Create from dictionary."""
        drift_map = cls(
            world_id=data["world_id"],
            name=data["name"]
        )
        drift_map.id = data["id"]
        drift_map.states = [OntologicalState.from_dict(state_data) for state_data in data["states"]]
        drift_map.transitions = data["transitions"]
        drift_map.evolution_patterns = data["evolution_patterns"]
        drift_map.drift_metrics = data["drift_metrics"]
        drift_map.creation_time = data["creation_time"]
        drift_map.last_updated = data["last_updated"]
        return drift_map
    
    def _calculate_transition(self, 
                            previous_state: OntologicalState, 
                            current_state: OntologicalState) -> Dict[str, Any]:
        """
        Calculate the transition between two ontological states.
        
        Args:
            previous_state: Earlier state
            current_state: Later state
            
        Returns:
            Dictionary describing the transition
        """
        # Symbol changes
        new_symbols = []
        removed_symbols = []
        modified_symbols = []
        
        # Find new and modified symbols
        for symbol_id, symbol_data in current_state.symbols.items():
            if symbol_id not in previous_state.symbols:
                new_symbols.append({
                    "symbol_id": symbol_id,
                    "name": symbol_data.get("name", ""),
                    "categories": symbol_data.get("categories", [])
                })
            else:
                # Check for modifications
                prev_data = previous_state.symbols[symbol_id]
                changes = {}
                
                for key in ["name", "description", "categories", "properties"]:
                    if key in symbol_data and key in prev_data:
                        if symbol_data[key] != prev_data[key]:
                            changes[key] = {
                                "previous": prev_data[key],
                                "current": symbol_data[key]
                            }
                
                if changes:
                    modified_symbols.append({
                        "symbol_id": symbol_id,
                        "name": symbol_data.get("name", ""),
                        "changes": changes
                    })
        
        # Find removed symbols
        for symbol_id, symbol_data in previous_state.symbols.items():
            if symbol_id not in current_state.symbols:
                removed_symbols.append({
                    "symbol_id": symbol_id,
                    "name": symbol_data.get("name", "")
                })
        
        # Relationship changes
        new_relationships = []
        removed_relationships = []
        modified_relationships = []
        
        # Find new and modified relationships
        for rel_id, rel_data in current_state.relationships.items():
            if rel_id not in previous_state.relationships:
                new_relationships.append({
                    "relationship_id": rel_id,
                    "type": rel_data.get("type", ""),
                    "source_id": rel_data.get("source_id", ""),
                    "target_id": rel_data.get("target_id", "")
                })
            else:
                # Check for modifications
                prev_data = previous_state.relationships[rel_id]
                changes = {}
                
                for key in ["type", "subtype", "strength", "properties"]:
                    if key in rel_data and key in prev_data:
                        if rel_data[key] != prev_data[key]:
                            changes[key] = {
                                "previous": prev_data[key],
                                "current": rel_data[key]
                            }
                
                if changes:
                    modified_relationships.append({
                        "relationship_id": rel_id,
                        "type": rel_data.get("type", ""),
                        "changes": changes
                    })
        
        # Find removed relationships
        for rel_id, rel_data in previous_state.relationships.items():
            if rel_id not in current_state.relationships:
                removed_relationships.append({
                    "relationship_id": rel_id,
                    "type": rel_data.get("type", "")
                })
        
        # Cluster changes
        prev_cluster_count = len(previous_state.clusters)
        current_cluster_count = len(current_state.clusters)
        
        # Create transition record
        transition = {
            "id": str(uuid.uuid4()),
            "from_state_id": previous_state.id,
            "to_state_id": current_state.id,
            "timestamp": current_state.timestamp,
            "time_interval": current_state.timestamp - previous_state.timestamp,
            "symbols": {
                "added": new_symbols,
                "removed": removed_symbols,
                "modified": modified_symbols,
                "net_change": len(new_symbols) - len(removed_symbols)
            },
            "relationships": {
                "added": new_relationships,
                "removed": removed_relationships,
                "modified": modified_relationships,
                "net_change": len(new_relationships) - len(removed_relationships)
            },
            "clusters": {
                "previous_count": prev_cluster_count,
                "current_count": current_cluster_count,
                "net_change": current_cluster_count - prev_cluster_count
            },
            "metrics_change": {
                key: current_state.metrics.get(key, 0) - previous_state.metrics.get(key, 0)
                for key in set(current_state.metrics) & set(previous_state.metrics)
            }
        }
        
        # Calculate total changes for summary
        total_symbol_changes = len(new_symbols) + len(removed_symbols) + len(modified_symbols)
        total_relationship_changes = (
            len(new_relationships) + len(removed_relationships) + len(modified_relationships)
        )
        
        # Add change magnitude as a measure of how much the ontology changed
        transition["change_magnitude"] = total_symbol_changes + total_relationship_changes
        
        # Determine if this is a major or minor transition
        transition["is_major_transition"] = (
            transition["change_magnitude"] > 5 or 
            abs(transition["symbols"]["net_change"]) > 3 or
            abs(transition["clusters"]["net_change"]) > 1
        )
        
        return transition
    
    def _update_drift_metrics(self, current_state: OntologicalState) -> None:
        """
        Update drift metrics based on a new state.
        
        Args:
            current_state: Latest ontological state
        """
        timestamp = current_state.timestamp
        
        # Update complexity metric
        complexity = current_state.get_ontological_complexity()
        self.drift_metrics["complexity"].append((timestamp, complexity))
        
        # Calculate coherence as inverse of fragmentation
        cluster_count = current_state.get_cluster_count()
        symbol_count = current_state.get_symbol_count()
        
        if symbol_count > 0 and cluster_count > 0:
            # More clusters relative to symbols indicates fragmentation
            coherence = 1.0 - (cluster_count / (symbol_count * 0.5))
            # Ensure it's in the range [0, 1]
            coherence = max(0.0, min(1.0, coherence))
        else:
            coherence = 0.0
        
        self.drift_metrics["coherence"].append((timestamp, coherence))
        
        # Calculate novelty based on recent transitions
        if len(self.transitions) > 0:
            last_transition = self.transitions[-1]
            new_elements = (
                len(last_transition["symbols"]["added"]) + 
                len(last_transition["relationships"]["added"])
            )
            total_elements = (
                current_state.get_symbol_count() + 
                current_state.get_relationship_count()
            )
            
            novelty = new_elements / max(1, total_elements)
            self.drift_metrics["novelty"].append((timestamp, novelty))
        else:
            self.drift_metrics["novelty"].append((timestamp, 0.0))
        
        # Calculate conceptual velocity - rate of change over time
        if len(self.transitions) > 0:
            last_transition = self.transitions[-1]
            time_interval = last_transition["time_interval"]
            
            if time_interval > 0:
                velocity = last_transition["change_magnitude"] / time_interval
            else:
                velocity = 0.0
                
            self.drift_metrics["conceptual_velocity"].append((timestamp, velocity))
        else:
            self.drift_metrics["conceptual_velocity"].append((timestamp, 0.0))
    
    def _detect_evolution_patterns(self) -> None:
        """Detect and record evolution patterns from transitions."""
        # Need at least 2 transitions to detect patterns
        if len(self.transitions) < 2:
            return
        
        # Existing patterns to check for continuation
        existing_pattern_ids = {pattern["id"] for pattern in self.evolution_patterns}
        
        # Check for growth patterns
        self._detect_growth_patterns(existing_pattern_ids)
        
        # Check for divergence patterns
        self._detect_divergence_patterns(existing_pattern_ids)
        
        # Check for convergence patterns
        self._detect_convergence_patterns(existing_pattern_ids)
        
        # Check for cyclical patterns
        self._detect_cyclical_patterns(existing_pattern_ids)
        
        # Check for refinement patterns
        self._detect_refinement_patterns(existing_pattern_ids)
    
    def _detect_growth_patterns(self, existing_pattern_ids: Set[str]) -> None:
        """
        Detect growth patterns (consistent expansion or contraction).
        
        Args:
            existing_pattern_ids: Set of existing pattern IDs
        """
        # Get the last 3 transitions if available
        transitions = self.transitions[-3:] if len(self.transitions) >= 3 else self.transitions
        
        # Check for consistent growth
        consistent_growth = all(t["symbols"]["net_change"] > 0 for t in transitions)
        consistent_shrinking = all(t["symbols"]["net_change"] < 0 for t in transitions)
        
        if consistent_growth or consistent_shrinking:
            pattern_type = "growth" if consistent_growth else "contraction"
            
            # Check if this continues an existing pattern
            continued = False
            for pattern in self.evolution_patterns:
                if pattern["type"] == pattern_type and pattern["id"] in existing_pattern_ids:
                    pattern["transitions"].append(self.transitions[-1]["id"])
                    pattern["end_time"] = self.transitions[-1]["timestamp"]
                    pattern["strength"] += 0.1  # Increase strength with each confirmation
                    pattern["strength"] = min(1.0, pattern["strength"])  # Cap at 1.0
                    continued = True
                    break
            
            # If not, create a new pattern
            if not continued:
                self.evolution_patterns.append({
                    "id": str(uuid.uuid4()),
                    "type": pattern_type,
                    "subtype": "consistent",
                    "start_time": transitions[0]["timestamp"],
                    "end_time": transitions[-1]["timestamp"],
                    "transitions": [t["id"] for t in transitions],
                    "strength": 0.5 + (0.1 * len(transitions))  # Initial strength
                })
    
    def _detect_divergence_patterns(self, existing_pattern_ids: Set[str]) -> None:
        """
        Detect divergence patterns (increasing differentiation).
        
        Args:
            existing_pattern_ids: Set of existing pattern IDs
        """
        # Need at least 3 transitions to detect meaningful divergence
        if len(self.transitions) < 3:
            return
        
        # Track cluster growth and relationship diversity
        cluster_growth = (
            self.transitions[-1]["clusters"]["current_count"] > 
            self.transitions[-3]["clusters"]["previous_count"]
        )
        
        # Check if relationships are becoming more diverse (more relationship types)
        if (len(self.states) >= 2):
            current = self.states[-1]
            previous = self.states[-3] if len(self.states) >= 3 else self.states[0]
            
            # Count relationship types
            current_types = set()
            for rel in current.relationships.values():
                rel_type = rel.get("type", "")
                rel_subtype = rel.get("subtype", "")
                current_types.add(f"{rel_type}:{rel_subtype}")
            
            previous_types = set()
            for rel in previous.relationships.values():
                rel_type = rel.get("type", "")
                rel_subtype = rel.get("subtype", "")
                previous_types.add(f"{rel_type}:{rel_subtype}")
            
            relationship_diversification = len(current_types) > len(previous_types)
            
            # If we see cluster growth and relationship diversification, it's divergence
            if cluster_growth and relationship_diversification:
                # Check if this continues an existing pattern
                continued = False
                for pattern in self.evolution_patterns:
                    if pattern["type"] == "divergence" and pattern["id"] in existing_pattern_ids:
                        pattern["transitions"].append(self.transitions[-1]["id"])
                        pattern["end_time"] = self.transitions[-1]["timestamp"]
                        pattern["strength"] += 0.1
                        pattern["strength"] = min(1.0, pattern["strength"])
                        continued = True
                        break
                
                # If not, create a new pattern
                if not continued:
                    self.evolution_patterns.append({
                        "id": str(uuid.uuid4()),
                        "type": "divergence",
                        "subtype": "differentiation",
                        "start_time": self.transitions[-3]["timestamp"],
                        "end_time": self.transitions[-1]["timestamp"],
                        "transitions": [t["id"] for t in self.transitions[-3:]],
                        "strength": 0.6  # Initial strength
                    })
    
    def _detect_convergence_patterns(self, existing_pattern_ids: Set[str]) -> None:
        """
        Detect convergence patterns (increasing integration).
        
        Args:
            existing_pattern_ids: Set of existing pattern IDs
        """
        # Need at least 3 transitions to detect meaningful convergence
        if len(self.transitions) < 3:
            return
        
        # Check for cluster consolidation
        cluster_consolidation = (
            self.transitions[-1]["clusters"]["current_count"] < 
            self.transitions[-3]["clusters"]["previous_count"]
        )
        
        # Check if there's an increase in relationships relative to symbols
        if len(self.states) >= 2:
            current = self.states[-1]
            previous = self.states[-3] if len(self.states) >= 3 else self.states[0]
            
            current_ratio = (
                current.get_relationship_count() / 
                max(1, current.get_symbol_count())
            )
            
            previous_ratio = (
                previous.get_relationship_count() / 
                max(1, previous.get_symbol_count())
            )
            
            increasing_connectivity = current_ratio > previous_ratio
            
            # If we see cluster consolidation and increasing connectivity, it's convergence
            if cluster_consolidation and increasing_connectivity:
                # Check if this continues an existing pattern
                continued = False
                for pattern in self.evolution_patterns:
                    if pattern["type"] == "convergence" and pattern["id"] in existing_pattern_ids:
                        pattern["transitions"].append(self.transitions[-1]["id"])
                        pattern["end_time"] = self.transitions[-1]["timestamp"]
                        pattern["strength"] += 0.1
                        pattern["strength"] = min(1.0, pattern["strength"])
                        continued = True
                        break
                
                # If not, create a new pattern
                if not continued:
                    self.evolution_patterns.append({
                        "id": str(uuid.uuid4()),
                        "type": "convergence",
                        "subtype": "integration",
                        "start_time": self.transitions[-3]["timestamp"],
                        "end_time": self.transitions[-1]["timestamp"],
                        "transitions": [t["id"] for t in self.transitions[-3:]],
                        "strength": 0.6  # Initial strength
                    })
    
    def _detect_cyclical_patterns(self, existing_pattern_ids: Set[str]) -> None:
        """
        Detect cyclical patterns (oscillations or recurring states).
        
        Args:
            existing_pattern_ids: Set of existing pattern IDs
        """
        # Need several transitions to detect cycles
        if len(self.transitions) < 5:
            return
        
        # Look for alternating growth and contraction
        alternating = True
        sign_changes = 0
        
        for i in range(1, min(5, len(self.transitions))):
            current_change = self.transitions[-i]["symbols"]["net_change"]
            previous_change = self.transitions[-(i+1)]["symbols"]["net_change"]
            
            if (current_change > 0 and previous_change > 0) or (current_change < 0 and previous_change < 0):
                alternating = False
                break
            
            if current_change * previous_change < 0:  # Sign change
                sign_changes += 1
        
        # If we see alternating growth/contraction with at least 2 sign changes, it's cyclical
        if alternating and sign_changes >= 2:
            # Check if this continues an existing pattern
            continued = False
            for pattern in self.evolution_patterns:
                if pattern["type"] == "cyclical" and pattern["id"] in existing_pattern_ids:
                    pattern["transitions"].append(self.transitions[-1]["id"])
                    pattern["end_time"] = self.transitions[-1]["timestamp"]
                    pattern["strength"] += 0.1
                    pattern["strength"] = min(1.0, pattern["strength"])
                    continued = True
                    break
            
            # If not, create a new pattern
            if not continued:
                self.evolution_patterns.append({
                    "id": str(uuid.uuid4()),
                    "type": "cyclical",
                    "subtype": "oscillation",
                    "start_time": self.transitions[-5]["timestamp"],
                    "end_time": self.transitions[-1]["timestamp"],
                    "transitions": [t["id"] for t in self.transitions[-5:]],
                    "strength": 0.7  # Initial strength - cycles are notable
                })
    
    def _detect_refinement_patterns(self, existing_pattern_ids: Set[str]) -> None:
        """
        Detect refinement patterns (increasing precision).
        
        Args:
            existing_pattern_ids: Set of existing pattern IDs
        """
        # Need at least 3 transitions to detect meaningful refinement
        if len(self.transitions) < 3:
            return
        
        # Check for high symbol modification but low symbol addition/removal
        transitions = self.transitions[-3:]
        
        high_modification = True
        low_turnover = True
        
        for t in transitions:
            # High modification: more modifications than additions or removals
            modified_count = len(t["symbols"]["modified"])
            turnover_count = len(t["symbols"]["added"]) + len(t["symbols"]["removed"])
            
            if modified_count <= turnover_count:
                high_modification = False
                
            # Low turnover: net change is small
            if abs(t["symbols"]["net_change"]) > 2:
                low_turnover = False
        
        # If we see high modification but low turnover, it's refinement
        if high_modification and low_turnover:
            # Check if this continues an existing pattern
            continued = False
            for pattern in self.evolution_patterns:
                if pattern["type"] == "refinement" and pattern["id"] in existing_pattern_ids:
                    pattern["transitions"].append(self.transitions[-1]["id"])
                    pattern["end_time"] = self.transitions[-1]["timestamp"]
                    pattern["strength"] += 0.1
                    pattern["strength"] = min(1.0, pattern["strength"])
                    continued = True
                    break
            
            # If not, create a new pattern
            if not continued:
                self.evolution_patterns.append({
                    "id": str(uuid.uuid4()),
                    "type": "refinement",
                    "subtype": "precision",
                    "start_time": transitions[0]["timestamp"],
                    "end_time": transitions[-1]["timestamp"],
                    "transitions": [t["id"] for t in transitions],
                    "strength": 0.6  # Initial strength
                })


class OntologicalDriftMapExpander:
    """
    Detects, visualizes, and encourages productive ontological drift 
    in symbolic narrative worlds.
    """
    
    def __init__(self, integration: Optional[WorldSymbolMemoryIntegration] = None):
        """
        Initialize an ontological drift map expander.
        
        Args:
            integration: Optional WorldSymbolMemoryIntegration instance
        """
        self.integration = integration
        self.drift_maps: Dict[str, OntologicalDriftMap] = {}
        self.expansion_templates: Dict[str, Dict[str, Any]] = self._initialize_expansion_templates()
        self.coherence_thresholds = {
            "minimum": 0.3,  # Below this, attempt to consolidate
            "maximum": 0.9   # Above this, encourage divergence
        }
        self.drift_history: Dict[str, List[Dict[str, Any]]] = {}  # World ID -> drift history
    
    def _initialize_expansion_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize expansion templates for different ontological patterns."""
        return {
            "differentiation": {
                "description": "Splitting concepts into more specific variants",
                "operations": [
                    "symbol_specialization",
                    "relationship_diversification",
                    "cluster_formation"
                ],
                "suitable_for": ["growth", "divergence"],
                "weight": 1.0
            },
            "synthesis": {
                "description": "Combining concepts into integrated wholes",
                "operations": [
                    "symbol_integration",
                    "relationship_strengthening",
                    "cluster_consolidation"
                ],
                "suitable_for": ["convergence", "refinement"],
                "weight": 1.0
            },
            "opposition": {
                "description": "Creating meaningful oppositions between concepts",
                "operations": [
                    "symbol_opposition",
                    "dialectical_tension",
                    "polarity_creation"
                ],
                "suitable_for": ["divergence", "cyclical"],
                "weight": 1.0
            },
            "metaphorical_extension": {
                "description": "Extending concepts through metaphorical mapping",
                "operations": [
                    "metaphor_mapping",
                    "domain_extension",
                    "symbolic_resonance"
                ],
                "suitable_for": ["growth", "refinement"],
                "weight": 1.0
            },
            "recursive_embedding": {
                "description": "Creating nested levels of symbolic meaning",
                "operations": [
                    "symbol_nesting",
                    "hierarchical_organization",
                    "fractal_patterning"
                ],
                "suitable_for": ["convergence", "refinement"],
                "weight": 1.0
            }
        }
    
    def create_drift_map(self, 
                       world_id: str, 
                       name: Optional[str] = None) -> Optional[str]:
        """
        Create a new ontological drift map.
        
        Args:
            world_id: ID of the symbolic world
            name: Optional name for the drift map
            
        Returns:
            ID of the created drift map, or None if world not found
        """
        if self.integration:
            # Verify world exists
            world = self.integration.symbolic_worlds.get(world_id)
            if not world:
                return None
        
        drift_map = OntologicalDriftMap(world_id=world_id, name=name)
        self.drift_maps[drift_map.id] = drift_map
        
        # Initialize drift history for this world
        if world_id not in self.drift_history:
            self.drift_history[world_id] = []
        
        # Capture initial state if integration is available
        if self.integration:
            self.capture_current_state(world_id, drift_map.id)
        
        return drift_map.id
        
    def get_drift_map(self, map_id: str) -> Optional[OntologicalDriftMap]:
        """
        Get a drift map by ID.
        
        Args:
            map_id: ID of the drift map
            
        Returns:
            OntologicalDriftMap or None if not found
        """
        return self.drift_maps.get(map_id)
    
    def get_drift_maps_for_world(self, world_id: str) -> List[OntologicalDriftMap]:
        """
        Get all drift maps for a specific world.
        
        Args:
            world_id: ID of the symbolic world
            
        Returns:
            List of OntologicalDriftMap objects
        """
        return [
            drift_map for drift_map in self.drift_maps.values()
            if drift_map.world_id == world_id
        ]
    
    def capture_current_state(self, 
                            world_id: str, 
                            drift_map_id: Optional[str] = None) -> Optional[str]:
        """
        Capture the current state of a symbolic world.
        
        Args:
            world_id: ID of the symbolic world
            drift_map_id: Optional ID of the drift map (or use all for this world)
            
        Returns:
            ID of the captured state, or None if world not found
        """
        if not self.integration:
            return None
        
        world = self.integration.symbolic_worlds.get(world_id)
        if not world:
            return None
        
        # Generate state data
        timestamp = datetime.now().timestamp()
        
        # Get symbols data
        symbols_data = {}
        for symbol_id, symbol in world.symbols.items():
            symbols_data[symbol_id] = {
                "name": symbol.name,
                "description": symbol.description,
                "categories": symbol.categories.copy(),
                "properties": symbol.properties.copy(),
                "resonance_metrics": getattr(symbol, "resonance_metrics", {}).copy(),
                "creation_date": getattr(symbol, "creation_date", timestamp),
                "last_modified": getattr(symbol, "last_modified", timestamp)
            }
        
        # Get relationships data (associations and oppositions)
        relationships_data = {}
        
        # Process associations
        for symbol_id, symbol in world.symbols.items():
            for assoc in getattr(symbol, "associations", []):
                rel_id = assoc.get("id", str(uuid.uuid4()))
                relationships_data[rel_id] = {
                    "type": "association",
                    "subtype": assoc.get("association_type", ""),
                    "source_id": symbol_id,
                    "target_id": assoc.get("target_id", ""),
                    "strength": assoc.get("strength", 0.5),
                    "properties": assoc.get("properties", {}).copy()
                }
        
        # Process oppositions
        for symbol_id, symbol in world.symbols.items():
            for opp in getattr(symbol, "oppositions", []):
                rel_id = opp.get("id", str(uuid.uuid4()))
                relationships_data[rel_id] = {
                    "type": "opposition",
                    "subtype": opp.get("opposition_type", ""),
                    "source_id": symbol_id,
                    "target_id": opp.get("target_id", ""),
                    "intensity": opp.get("intensity", 0.5),
                    "properties": opp.get("properties", {}).copy()
                }
        
        # Get clusters data
        clusters_data = world.identify_symbol_clusters()
        
        # Get metrics data
        world.calculate_world_metrics()
        metrics_data = world.metrics.copy()
        
        # Create the ontological state
        state = OntologicalState(
            timestamp=timestamp,
            world_id=world_id,
            symbols=symbols_data,
            relationships=relationships_data,
            clusters=clusters_data,
            metrics=metrics_data
        )
        
        # Add to specified drift map(s)
        if drift_map_id:
            drift_map = self.drift_maps.get(drift_map_id)
            if drift_map:
                drift_map.add_state(state)
        else:
            # Add to all drift maps for this world
            for drift_map in self.get_drift_maps_for_world(world_id):
                drift_map.add_state(state)
        
        return state.id
        
    def analyze_drift_patterns(self, 
                             world_id: str,
                             time_window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Analyze drift patterns for a symbolic world.
        
        Args:
            world_id: ID of the symbolic world
            time_window: Optional (start_time, end_time) tuple
            
        Returns:
            Analysis results
        """
        drift_maps = self.get_drift_maps_for_world(world_id)
        if not drift_maps:
            return {"error": "No drift maps found for this world"}
        
        # Combine evolution patterns from all maps
        all_patterns = []
        for drift_map in drift_maps:
            latest_state = drift_map.get_latest_state()
            if not latest_state:
                continue
                
            # Filter patterns by time window if specified
            if time_window:
                start_time, end_time = time_window
                filtered_patterns = [
                    pattern for pattern in drift_map.evolution_patterns
                    if pattern["end_time"] >= start_time and pattern["start_time"] <= end_time
                ]
                all_patterns.extend(filtered_patterns)
            else:
                all_patterns.extend(drift_map.evolution_patterns)
        
        # Count pattern types
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_type = pattern["type"]
            if pattern_type not in pattern_counts:
                pattern_counts[pattern_type] = 0
            pattern_counts[pattern_type] += 1
        
        # Get strongest patterns
        strongest_patterns = sorted(
            all_patterns,
            key=lambda p: p.get("strength", 0),
            reverse=True
        )[:5]  # Top 5
        
        # Get recent patterns
        recent_patterns = sorted(
            all_patterns,
            key=lambda p: p.get("end_time", 0),
            reverse=True
        )[:5]  # Top 5
        
        # Get overall drift direction
        dominant_patterns = sorted(
            pattern_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )
        
        drift_direction = dominant_patterns[0][0] if dominant_patterns else "stable"
        
        # Get coherence trend from last state
        coherence_trend = "stable"
        if drift_maps and drift_maps[0].drift_metrics["coherence"]:
            coherence_values = [v for _, v in drift_maps[0].drift_metrics["coherence"]]
            if len(coherence_values) >= 3:
                if coherence_values[-1] > coherence_values[-3]:
                    coherence_trend = "increasing"
                elif coherence_values[-1] < coherence_values[-3]:
                    coherence_trend = "decreasing"
        
        # Record in drift history
        drift_entry = {
            "timestamp": datetime.now().timestamp(),
            "pattern_counts": pattern_counts,
            "drift_direction": drift_direction,
            "coherence_trend": coherence_trend
        }
        
        self.drift_history[world_id].append(drift_entry)
        
        # Determine recommended expansion approach
        expansion_approach = self._determine_expansion_approach(
            drift_direction, coherence_trend
        )
        
        return {
            "world_id": world_id,
            "analyzed_patterns": len(all_patterns),
            "pattern_distribution": pattern_counts,
            "drift_direction": drift_direction,
            "coherence_trend": coherence_trend,
            "strongest_patterns": strongest_patterns,
            "recent_patterns": recent_patterns,
            "recommended_expansion": expansion_approach
        }
    
    def _determine_expansion_approach(self, 
                                    drift_direction: str, 
                                    coherence_trend: str) -> Dict[str, Any]:
        """
        Determine the best expansion approach based on current patterns.
        
        Args:
            drift_direction: Dominant drift direction
            coherence_trend: Trend in ontological coherence
            
        Returns:
            Recommended expansion approach
        """
        # Create weighted candidates
        candidates = []
        
        # If coherence is decreasing, prioritize synthesis
        if coherence_trend == "decreasing":
            candidates.append(("synthesis", 2.0))
            candidates.append(("recursive_embedding", 1.5))
        
        # If coherence is increasing, allow more divergence
        if coherence_trend == "increasing":
            candidates.append(("differentiation", 1.5))
            candidates.append(("metaphorical_extension", 1.2))
            candidates.append(("opposition", 1.0))
        
        # Match to current drift direction
        if drift_direction == "growth":
            candidates.append(("differentiation", 1.2))
            candidates.append(("metaphorical_extension", 1.5))
            
        elif drift_direction == "contraction":
            candidates.append(("synthesis", 1.5))
            candidates.append(("recursive_embedding", 1.2))
            
        elif drift_direction == "divergence":
            candidates.append(("differentiation", 1.8))
            candidates.append(("opposition", 1.5))
            
        elif drift_direction == "convergence":
            candidates.append(("synthesis", 1.8))
            candidates.append(("recursive_embedding", 1.5))
            
        elif drift_direction == "cyclical":
            candidates.append(("opposition", 1.8))
            candidates.append(("metaphorical_extension", 1.2))
            
        elif drift_direction == "refinement":
            candidates.append(("recursive_embedding", 1.5))
            candidates.append(("metaphorical_extension", 1.2))
        
        # Default additions (lower weight)
        for template_name in self.expansion_templates:
            candidates.append((template_name, 0.5))
        
        # Select template based on weighted probabilities
        weights = []
        templates = []
        
        # Deduplicate and combine weights
        template_weights = {}
        for template, weight in candidates:
            if template in template_weights:
                template_weights[template] += weight
            else:
                template_weights[template] = weight
        
        for template, weight in template_weights.items():
            templates.append(template)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Select template
        selected_template = random.choices(templates, weights=normalized_weights, k=1)[0]
        template_data = self.expansion_templates[selected_template].copy()
        
        return {
            "template": selected_template,
            "description": template_data["description"],
            "operations": template_data["operations"],
            "reason": f"Selected based on {drift_direction} drift direction and {coherence_trend} coherence trend"
        }
    
    def generate_expansion_suggestions(self, 
                                     world_id: str,
                                     template_name: Optional[str] = None,
                                     count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate ontological expansion suggestions based on drift patterns.
        
        Args:
            world_id: ID of the symbolic world
            template_name: Optional specific template to use
            count: Number of suggestions to generate
            
        Returns:
            List of expansion suggestions
        """
        if not self.integration:
            return [{"error": "Integration not available"}]
            
        world = self.integration.symbolic_worlds.get(world_id)
        if not world:
            return [{"error": "World not found"}]
        
        # Analyze drift patterns
        analysis = self.analyze_drift_patterns(world_id)
        
        # Use specified template or recommended one
        if template_name and template_name in self.expansion_templates:
            template = template_name
        else:
            template = analysis["recommended_expansion"]["template"]
        
        template_data = self.expansion_templates[template]
        operations = template_data["operations"]
        
        # Get symbols to work with
        symbols = list(world.symbols.values())
        if not symbols:
            return [{"error": "No symbols found in world"}]
        
        # Generate suggestions
        suggestions = []
        
        for _ in range(count):
            if "symbol_specialization" in operations:
                suggestion = self._generate_symbol_specialization(world, symbols)
                if suggestion:
                    suggestions.append(suggestion)
            
            elif "symbol_integration" in operations:
                suggestion = self._generate_symbol_integration(world, symbols)
                if suggestion:
                    suggestions.append(suggestion)
            
            elif "symbol_opposition" in operations:
                suggestion = self._generate_symbol_opposition(world, symbols)
                if suggestion:
                    suggestions.append(suggestion)
            
            elif "metaphor_mapping" in operations:
                suggestion = self._generate_metaphor_mapping(world, symbols)
                if suggestion:
                    suggestions.append(suggestion)
            
            elif "symbol_nesting" in operations:
                suggestion = self._generate_symbol_nesting(world, symbols)
                if suggestion:
                    suggestions.append(suggestion)
        
        # If we didn't generate enough, add some general suggestions
        while len(suggestions) < count:
            # Randomly select an operation
            operation = random.choice([
                self._generate_symbol_specialization,
                self._generate_symbol_integration,
                self._generate_symbol_opposition,
                self._generate_metaphor_mapping,
                self._generate_symbol_nesting
            ])
            
            suggestion = operation(world, symbols)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions[:count]  # Limit to requested count
    
    def _generate_symbol_specialization(self, 
                                      world: SymbolicWorld, 
                                      symbols: List[Symbol]) -> Optional[Dict[str, Any]]:
        """Generate a symbol specialization suggestion."""
        if not symbols:
            return None
        
        # Select a symbol to specialize
        symbol = random.choice(symbols)
        
        return {
            "type": "symbol_specialization",
            "description": f"Create specialized variants of '{symbol.name}'",
            "symbol_id": symbol.id,
            "symbol_name": symbol.name,
            "suggestions": [
                f"{symbol.name} as manifested in personal development",
                f"{symbol.name} in its transformative aspect",
                f"The hidden dimension of {symbol.name}"
            ],
            "implementation": {
                "operation": "create_derived_symbols",
                "params": {
                    "parent_id": symbol.id,
                    "relationship_type": "specialization"
                }
            }
        }
    
    def _generate_symbol_integration(self, 
                                   world: SymbolicWorld, 
                                   symbols: List[Symbol]) -> Optional[Dict[str, Any]]:
        """Generate a symbol integration suggestion."""
        if len(symbols) < 2:
            return None
        
        # Select two symbols to integrate
        symbol1, symbol2 = random.sample(symbols, 2)
        
        return {
            "type": "symbol_integration",
            "description": f"Integrate '{symbol1.name}' and '{symbol2.name}' into a new synthesized symbol",
            "symbols": [
                {"id": symbol1.id, "name": symbol1.name},
                {"id": symbol2.id, "name": symbol2.name}
            ],
            "suggestions": [
                f"The union of {symbol1.name} and {symbol2.name}",
                f"{symbol1.name}-{symbol2.name} synthesis",
                f"Transcendent integration of {symbol1.name}/{symbol2.name}"
            ],
            "implementation": {
                "operation": "create_integrated_symbol",
                "params": {
                    "source_ids": [symbol1.id, symbol2.id],
                    "relationship_type": "synthesis"
                }
            }
        }
    
    def _generate_symbol_opposition(self, 
                                  world: SymbolicWorld, 
                                  symbols: List[Symbol]) -> Optional[Dict[str, Any]]:
        """Generate a symbol opposition suggestion."""
        if not symbols:
            return None
        
        # Select a symbol to create opposition for
        symbol = random.choice(symbols)
        
        return {
            "type": "symbol_opposition",
            "description": f"Create a symbolic opposition to '{symbol.name}'",
            "symbol_id": symbol.id,
            "symbol_name": symbol.name,
            "suggestions": [
                f"Anti-{symbol.name}",
                f"The shadow aspect of {symbol.name}",
                f"The inverse of {symbol.name}"
            ],
            "implementation": {
                "operation": "create_oppositional_symbol",
                "params": {
                    "source_id": symbol.id,
                    "opposition_type": "contrary"
                }
            }
        }
    
    def _generate_metaphor_mapping(self, 
                                 world: SymbolicWorld, 
                                 symbols: List[Symbol]) -> Optional[Dict[str, Any]]:
        """Generate a metaphorical mapping suggestion."""
        if not symbols:
            return None
        
        # Select a symbol to map metaphorically
        symbol = random.choice(symbols)
        
        # Potential domains for metaphorical mapping
        domains = ["nature", "journey", "battle", "water", "light", "seasons", "body"]
        selected_domain = random.choice(domains)
        
        return {
            "type": "metaphor_mapping",
            "description": f"Map '{symbol.name}' into the domain of {selected_domain}",
            "symbol_id": symbol.id,
            "symbol_name": symbol.name,
            "domain": selected_domain,
            "suggestions": [
                f"{symbol.name} as {selected_domain}",
                f"The {selected_domain}-like qualities of {symbol.name}",
                f"{symbol.name} expressed through {selected_domain} imagery"
            ],
            "implementation": {
                "operation": "create_metaphorical_mapping",
                "params": {
                    "source_id": symbol.id,
                    "target_domain": selected_domain
                }
            }
        }
    
    def _generate_symbol_nesting(self, 
                               world: SymbolicWorld, 
                               symbols: List[Symbol]) -> Optional[Dict[str, Any]]:
        """Generate a symbol nesting suggestion."""
        if not symbols:
            return None
        
        # Select a symbol to nest within
        symbol = random.choice(symbols)
        
        return {
            "type": "symbol_nesting",
            "description": f"Create nested levels within '{symbol.name}'",
            "symbol_id": symbol.id,
            "symbol_name": symbol.name,
            "suggestions": [
                f"The inner landscape of {symbol.name}",
                f"Microcosm within {symbol.name}",
                f"Recursive patterns in {symbol.name}"
            ],
            "implementation": {
                "operation": "create_nested_structure",
                "params": {
                    "parent_id": symbol.id,
                    "nesting_type": "fractal"
                }
            }
        }
    
    def apply_expansion(self, 
                      world_id: str,
                      expansion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply an ontological expansion to a symbolic world.
        
        Args:
            world_id: ID of the symbolic world
            expansion_data: Expansion specification
            
        Returns:
            Results of the expansion
        """
        if not self.integration:
            return {"error": "Integration not available"}
            
        world = self.integration.symbolic_worlds.get(world_id)
        if not world:
            return {"error": "World not found"}
        
        # Extract operation and parameters
        implementation = expansion_data.get("implementation", {})
        operation = implementation.get("operation", "")
        params = implementation.get("params", {})
        
        # Track created/modified entities
        created_symbols = []
        modified_symbols = []
        created_relationships = []
        
        # Apply the appropriate operation
        if operation == "create_derived_symbols":
            parent_id = params.get("parent_id")
            parent = world.get_symbol(parent_id)
            
            if not parent:
                return {"error": "Parent symbol not found"}
            
            # Create specialized symbols
            for suggestion in expansion_data.get("suggestions", [])[:2]:  # Limit to 2
                symbol_id = self.integration.create_symbol(
                    world_id=world_id,
                    name=suggestion,
                    description=f"Specialized variant of {parent.name}",
                    properties={
                        "derived_from": parent_id,
                        "derivation_type": "specialization"
                    },
                    categories=parent.categories
                )
                
                if symbol_id:
                    # Create association to parent
                    assoc_id = self.integration.create_symbolic_association(
                        world_id=world_id,
                        source_id=symbol_id,
                        target_id=parent_id,
                        association_type="specialization",
                        strength=0.8,
                        properties={"generated_by": "ontological_drift_expansion"}
                    )
                    
                    created_symbols.append({
                        "id": symbol_id,
                        "name": suggestion,
                        "parent_id": parent_id
                    })
                    
                    if assoc_id:
                        created_relationships.append({
                            "id": assoc_id,
                            "type": "association",
                            "source_id": symbol_id,
                            "target_id": parent_id
                        })
        
        elif operation == "create_integrated_symbol":
            source_ids = params.get("source_ids", [])
            source_symbols = []
            
            for source_id in source_ids:
                symbol = world.get_symbol(source_id)
                if symbol:
                    source_symbols.append(symbol)
            
            if len(source_symbols) < 2:
                return {"error": "Not enough source symbols found"}
            
            # Create integrated symbol
            name = expansion_data.get("suggestions", ["Integrated Symbol"])[0]
            
            # Combine categories from source symbols
            categories = []
            for symbol in source_symbols:
                for category in symbol.categories:
                    if category not in categories:
                        categories.append(category)
            
            # Create the integrated symbol
            symbol_id = self.integration.create_symbol(
                world_id=world_id,
                name=name,
                description=f"Integration of {', '.join(s.name for s in source_symbols)}",
                properties={
                    "integrated_from": source_ids,
                    "integration_type": params.get("relationship_type", "synthesis")
                },
                categories=categories
            )
            
            if symbol_id:
                # Create associations to source symbols
                for source in source_symbols:
                    assoc_id = self.integration.create_symbolic_association(
                        world_id=world_id,
                        source_id=symbol_id,
                        target_id=source.id,
                        association_type="integration",
                        strength=0.9,
                        properties={"generated_by": "ontological_drift_expansion"}
                    )
                    
                    if assoc_id:
                        created_relationships.append({
                            "id": assoc_id,
                            "type": "association",
                            "source_id": symbol_id,
                            "target_id": source.id
                        })
                
                created_symbols.append({
                    "id": symbol_id,
                    "name": name,
                    "source_ids": source_ids
                })
        
        elif operation == "create_oppositional_symbol":
            source_id = params.get("source_id")
            source = world.get_symbol(source_id)
            
            if not source:
                return {"error": "Source symbol not found"}
            
            # Create oppositional symbol
            name = expansion_data.get("suggestions", [f"Anti-{source.name}"])[0]
            
            symbol_id = self.integration.create_symbol(
                world_id=world_id,
                name=name,
                description=f"Opposition to {source.name}",
                properties={
                    "opposite_to": source_id,
                    "opposition_type": params.get("opposition_type", "contrary")
                },
                categories=source.categories
            )
            
            if symbol_id:
                # Create opposition relationship
                opp_id = self.integration.create_symbolic_opposition(
                    world_id=world_id,
                    source_id=symbol_id,
                    target_id=source_id,
                    opposition_type=params.get("opposition_type", "contrary"),
                    intensity=0.8,
                    properties={"generated_by": "ontological_drift_expansion"}
                )
                
                created_symbols.append({
                    "id": symbol_id,
                    "name": name,
                    "opposite_to": source_id
                })
                
                if opp_id:
                    created_relationships.append({
                        "id": opp_id,
                        "type": "opposition",
                        "source_id": symbol_id,
                        "target_id": source_id
                    })
        
        elif operation == "create_metaphorical_mapping":
            source_id = params.get("source_id")
            source = world.get_symbol(source_id)
            
            if not source:
                return {"error": "Source symbol not found"}
            
            target_domain = params.get("target_domain", "")
            
            # Create metaphorical mapping
            name = expansion_data.get("suggestions", [f"{source.name} as {target_domain}"])[0]
            
            symbol_id = self.integration.create_symbol(
                world_id=world_id,
                name=name,
                description=f"Metaphorical mapping of {source.name} into {target_domain} domain",
                properties={
                    "metaphor_source": source_id,
                    "metaphor_domain": target_domain
                },
                categories=source.categories + ["metaphor"]
            )
            
            if symbol_id:
                # Create metaphorical association
                assoc_id = self.integration.create_symbolic_association(
                    world_id=world_id,
                    source_id=symbol_id,
                    target_id=source_id,
                    association_type="metaphorical",
                    strength=0.7,
                    properties={
                        "generated_by": "ontological_drift_expansion",
                        "domain": target_domain
                    }
                )
                
                created_symbols.append({
                    "id": symbol_id,
                    "name": name,
                    "metaphor_source": source_id,
                    "domain": target_domain
                })
                
                if assoc_id:
                    created_relationships.append({
                        "id": assoc_id,
                        "type": "association",
                        "source_id": symbol_id,
                        "target_id": source_id
                    })
        
        elif operation == "create_nested_structure":
            parent_id = params.get("parent_id")
            parent = world.get_symbol(parent_id)
            
            if not parent:
                return {"error": "Parent symbol not found"}
            
            # Create nested structure
            name = expansion_data.get("suggestions", [f"Inner {parent.name}"])[0]
            
            symbol_id = self.integration.create_symbol(
                world_id=world_id,
                name=name,
                description=f"Nested structure within {parent.name}",
                properties={
                    "nested_within": parent_id,
                    "nesting_type": params.get("nesting_type", "fractal")
                },
                categories=parent.categories + ["nested"]
            )
            
            if symbol_id:
                # Create nesting association
                assoc_id = self.integration.create_symbolic_association(
                    world_id=world_id,
                    source_id=parent_id,
                    target_id=symbol_id,
                    association_type="contains",
                    strength=0.9,
                    properties={
                        "generated_by": "ontological_drift_expansion",
                        "containment_type": "nesting"
                    }
                )
                
                created_symbols.append({
                    "id": symbol_id,
                    "name": name,
                    "nested_within": parent_id
                })
                
                if assoc_id:
                    created_relationships.append({
                        "id": assoc_id,
                        "type": "association",
                        "source_id": parent_id,
                        "target_id": symbol_id
                    })
        
        # Capture the new ontological state
        self.capture_current_state(world_id)
        
        return {
            "expansion_type": operation,
            "created_symbols": created_symbols,
            "modified_symbols": modified_symbols,
            "created_relationships": created_relationships,
            "timestamp": datetime.now().timestamp(),
            "status": "success"
        }
    
    def visualize_drift_map(self, map_id: str, format_type: str = "json") -> Dict[str, Any]:
        """
        Generate a visualization of an ontological drift map.
        
        Args:
            map_id: ID of the drift map
            format_type: Format of visualization ('json', 'timeline', 'network', etc.)
            
        Returns:
            Visualization data
        """
        drift_map = self.get_drift_map(map_id)
        if not drift_map:
            return {"error": "Drift map not found"}
        
        # Get states sorted by timestamp
        states = sorted(drift_map.states, key=lambda s: s.timestamp)
        if not states:
            return {"error": "No states in drift map"}
        
        # Format based on requested type
        if format_type == "timeline":
            return self._format_timeline_visualization(drift_map, states)
        elif format_type == "network":
            return self._format_network_visualization(drift_map, states)
        else:  # Default to JSON
            return {
                "map_id": map_id,
                "world_id": drift_map.world_id,
                "name": drift_map.name,
                "states": [state.to_dict() for state in states],
                "transitions": drift_map.transitions,
                "evolution_patterns": drift_map.evolution_patterns,
                "drift_metrics": drift_map.drift_metrics
            }
    
    def _format_timeline_visualization(self, 
                                    drift_map: OntologicalDriftMap, 
                                    states: List[OntologicalState]) -> Dict[str, Any]:
        """Format as timeline visualization."""
        timeline_events = []
        
        # Add state events
        for state in states:
            event = {
                "id": state.id,
                "timestamp": state.timestamp,
                "type": "state",
                "symbol_count": state.get_symbol_count(),
                "relationship_count": state.get_relationship_count(),
                "cluster_count": state.get_cluster_count(),
                "complexity": state.get_ontological_complexity()
            }
            timeline_events.append(event)
        
        # Add transition events
        for transition in drift_map.transitions:
            event = {
                "id": transition["id"],
                "timestamp": transition["timestamp"],
                "type": "transition",
                "from_state_id": transition["from_state_id"],
                "to_state_id": transition["to_state_id"],
                "symbols_added": len(transition["symbols"]["added"]),
                "symbols_removed": len(transition["symbols"]["removed"]),
                "symbols_modified": len(transition["symbols"]["modified"]),
                "relationships_added": len(transition["relationships"]["added"]),
                "relationships_removed": len(transition["relationships"]["removed"]),
                "is_major": transition.get("is_major_transition", False)
            }
            timeline_events.append(event)
        
        # Add evolution pattern events
        for pattern in drift_map.evolution_patterns:
            event = {
                "id": pattern["id"],
                "timestamp": pattern["end_time"],
                "type": "pattern",
                "pattern_type": pattern["type"],
                "subtype": pattern["subtype"],
                "strength": pattern["strength"],
                "duration": pattern["end_time"] - pattern["start_time"]
            }
            timeline_events.append(event)
        
        # Sort by timestamp
        timeline_events.sort(key=lambda e: e["timestamp"])
        
        # Create metrics timeline
        metrics_timeline = {}
        for metric_name, values in drift_map.drift_metrics.items():
            metrics_timeline[metric_name] = sorted(values)
        
        return {
            "visualization_type": "timeline",
            "map_id": drift_map.id,
            "world_id": drift_map.world_id,
            "name": drift_map.name,
            "events": timeline_events,
            "metrics_timeline": metrics_timeline
        }
    
    def _format_network_visualization(self, 
                                   drift_map: OntologicalDriftMap, 
                                   states: List[OntologicalState]) -> Dict[str, Any]:
        """Format as network visualization."""
        # Use the last state for current network structure
        latest_state = states[-1]
        
        # Extract nodes (symbols) and edges (relationships)
        nodes = []
        for symbol_id, symbol_data in latest_state.symbols.items():
            node = {
                "id": symbol_id,
                "label": symbol_data.get("name", "Unknown"),
                "categories": symbol_data.get("categories", []),
                "creation_date": symbol_data.get("creation_date", 0),
                "size": 1  # Default size
            }
            nodes.append(node)
        
        edges = []
        for rel_id, rel_data in latest_state.relationships.items():
            edge = {
                "id": rel_id,
                "source": rel_data.get("source_id", ""),
                "target": rel_data.get("target_id", ""),
                "type": rel_data.get("type", "unknown"),
                "subtype": rel_data.get("subtype", ""),
                "weight": rel_data.get("strength", 0.5) if rel_data.get("type") == "association" else rel_data.get("intensity", 0.5)
            }
            edges.append(edge)
        
        # Add cluster information
        clusters = []
        for i, cluster in enumerate(latest_state.clusters):
            clusters.append({
                "id": f"cluster_{i}",
                "symbol_ids": cluster.get("symbols", []),
                "center": cluster.get("center", ""),
                "coherence": cluster.get("coherence", 0.0)
            })
        
        return {
            "visualization_type": "network",
            "map_id": drift_map.id,
            "world_id": drift_map.world_id,
            "name": drift_map.name,
            "timestamp": latest_state.timestamp,
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters
        }
    
    def save_drift_map_to_json(self, map_id: str, file_path: str) -> bool:
        """
        Save a drift map to a JSON file.
        
        Args:
            map_id: ID of the drift map
            file_path: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        drift_map = self.get_drift_map(map_id)
        if not drift_map:
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(drift_map.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def load_drift_map_from_json(self, file_path: str) -> Optional[str]:
        """
        Load a drift map from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            ID of the loaded drift map or None if loading failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            drift_map = OntologicalDriftMap.from_dict(data)
            self.drift_maps[drift_map.id] = drift_map
            return drift_map.id
        except Exception:
            return None
    
    def expand_drift_map(self, 
                         world_id: str, 
                         drift_map_id: str) -> Optional[Dict[str, Any]]:
        """
        Expand the drift map based on detected patterns.
        
        Args:
            world_id: ID of the symbolic world
            drift_map_id: ID of the drift map to expand
            
        Returns:
            Details of the expansion operation, or None if drift map not found
        """
        drift_map = self.get_drift_map(drift_map_id)
        if not drift_map:
            return None

        # Analyze current drift patterns
        analysis_results = self.analyze_drift_patterns(world_id)
        
        # Check if expansion is warranted
        if analysis_results.get("recommended_expansion") is None:
            return {"message": "No suitable expansion approach found."}

        # Perform the expansion using the recommended approach
        expansion_details = analysis_results["recommended_expansion"]
        self.perform_expansion(drift_map, expansion_details)

        return {
            "drift_map_id": drift_map_id,
            "expansion_details": expansion_details
        }

    def perform_expansion(self, 
                          drift_map: OntologicalDriftMap, 
                          expansion_details: Dict[str, Any]) -> None:
        """
        Execute the expansion operations on the drift map.
        
        Args:
            drift_map: Target drift map to expand
            expansion_details: Details of the expansion operations
        """
        template = expansion_details["template"]
        operations = expansion_details.get("operations", [])

        # Execute each operation based on the template
        for operation in operations:
            if operation == "symbol_specialization":
                self.symbol_specialization(drift_map)
            elif operation == "relationship_diversification":
                self.relationship_diversification(drift_map)
            elif operation == "cluster_formation":
                self.cluster_formation(drift_map)
            elif operation == "symbol_integration":
                self.symbol_integration(drift_map)
            elif operation == "relationship_strengthening":
                self.relationship_strengthening(drift_map)
            elif operation == "cluster_consolidation":
                self.cluster_consolidation(drift_map)
            elif operation == "symbol_opposition":
                self.symbol_opposition(drift_map)
            elif operation == "metaphor_mapping":
                self.metaphor_mapping(drift_map)

    def symbol_specialization(self, drift_map: OntologicalDriftMap) -> None:
        """Specialize symbols based on current state."""
        # Implementation would be added here
        pass

    def relationship_diversification(self, drift_map: OntologicalDriftMap) -> None:
        """Diversify relationships to increase complexity."""
        # Implementation would be added here
        pass

    def cluster_formation(self, drift_map: OntologicalDriftMap) -> None:
        """Form new clusters based on symbols and relationships."""
        # Implementation would be added here
        pass

    def symbol_integration(self, drift_map: OntologicalDriftMap) -> None:
        """Integrate symbols to enhance coherence."""
        # Implementation would be added here
        pass

    def relationship_strengthening(self, drift_map: OntologicalDriftMap) -> None:
        """Strengthen existing relationships."""
        # Implementation would be added here
        pass

    def cluster_consolidation(self, drift_map: OntologicalDriftMap) -> None:
        """Consolidate clusters to reduce fragmentation."""
        # Implementation would be added here
        pass

    def symbol_opposition(self, drift_map: OntologicalDriftMap) -> None:
        """Create oppositional relationships to foster tension."""
        # Implementation would be added here
        pass

    def metaphor_mapping(self, drift_map: OntologicalDriftMap) -> None:
        """Map concepts metaphorically to explore new meanings."""
        # Implementation would be added here
        pass


# Helper functions

def create_ontological_drift_map_expander(
    integration: Optional[WorldSymbolMemoryIntegration] = None
) -> OntologicalDriftMapExpander:
    """
    Create a new ontological drift map expander.
    
    Args:
        integration: Optional WorldSymbolMemoryIntegration instance
        
    Returns:
        OntologicalDriftMapExpander instance
    """
    return OntologicalDriftMapExpander(integration=integration)

def create_basic_drift_map(
    expander: OntologicalDriftMapExpander,
    world_id: str,
    name: Optional[str] = None
) -> Optional[str]:
    """
    Create a basic drift map for a symbolic world.
    
    Args:
        expander: OntologicalDriftMapExpander instance
        world_id: ID of the symbolic world
        name: Optional name for the drift map
        
    Returns:
        ID of the created drift map, or None if creation failed
    """
    return expander.create_drift_map(world_id=world_id, name=name)


# Main execution
if __name__ == "__main__":
    # Example usage
    try:
        from WorldSymbolMemoryIntegration import create_world_symbol_memory_integration
        
        # Create integration system
        integration = create_world_symbol_memory_integration()
        
        # Create a symbolic world
        world_id = integration.create_symbolic_world(
            name="Evolving Mythic Realm",
            description="A symbolic world with ontological drift tracking"
        )
        
        # Create symbol expander
        expander = create_ontological_drift_map_expander(integration)
        
        # Create drift map
        drift_map_id = expander.create_drift_map(world_id)
        
        # Create some symbols
        hero_id = integration.create_symbol(
            world_id=world_id,
            name="The Seeker",
            description="Symbol of the quest for meaning",
            symbol_type="character",
            categories=["character", "archetype"]
        )
        
        shadow_id = integration.create_symbol(
            world_id=world_id,
            name="The Void",
            description="Symbol of emptiness and uncertainty",
            symbol_type="concept",
            categories=["concept", "archetype"]
        )
        
        # Create a relationship
        integration.create_symbolic_opposition(
            world_id=world_id,
            source_id=hero_id,
            target_id=shadow_id,
            opposition_type="exploration",
            intensity=0.7
        )
        
        # Capture the state after changes
        expander.capture_current_state(world_id, drift_map_id)
        
        # Generate expansion suggestions
        suggestions = expander.generate_expansion_suggestions(world_id, count=2)
        print(f"Expansion suggestions: {json.dumps(suggestions, indent=2)}")
        
        # Apply the first suggestion
        if suggestions:
            result = expander.apply_expansion(world_id, suggestions[0])
            print(f"Expansion result: {json.dumps(result, indent=2)}")
        
        # Capture the state after expansion
        expander.capture_current_state(world_id, drift_map_id)
        
        # Analyze drift patterns
        analysis = expander.analyze_drift_patterns(world_id)
        print(f"Drift analysis: {json.dumps(analysis, indent=2)}")
        
        # Visualize drift map
        visualization = expander.visualize_drift_map(drift_map_id, format_type="timeline")
        print(f"Visualization: {json.dumps(visualization, indent=2)}")
        
    except ImportError:
        print("This is a standalone example.")
