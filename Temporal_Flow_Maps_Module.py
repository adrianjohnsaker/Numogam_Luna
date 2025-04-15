#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Flow Maps Module
Phase XII - Amelia AI Project

This module creates visual representations of how concepts and symbols
evolve over time within the mythology, allowing users to track and
understand the changing relationships and developments of mythological
elements across different timescales.
"""

import datetime
import json
import math
import uuid
import random
import copy
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field


@dataclass
class Symbol:
    """Class representing a mythological symbol."""
    id: str
    name: str
    description: str
    creation_date: str
    last_modified: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    
    def update_attribute(self, key: str, value: Any, 
                       timestamp: Optional[str] = None, 
                       source: Optional[str] = None):
        """Update an attribute and record history."""
        # Record previous state if it existed
        if key in self.attributes:
            old_value = self.attributes[key]
            self.history.append({
                "type": "attribute_change",
                "attribute": key,
                "old_value": old_value,
                "new_value": value,
                "timestamp": timestamp or datetime.datetime.now().isoformat(),
                "source": source
            })
        else:
            self.history.append({
                "type": "attribute_added",
                "attribute": key,
                "value": value,
                "timestamp": timestamp or datetime.datetime.now().isoformat(),
                "source": source
            })
            
        # Update the attribute
        self.attributes[key] = value
        self.last_modified = timestamp or datetime.datetime.now().isoformat()
    
    def add_connection(self, target_id: str, 
                     strength: float = 1.0, 
                     timestamp: Optional[str] = None,
                     source: Optional[str] = None):
        """Add a connection to another symbol."""
        if target_id not in self.connections:
            self.connections.append(target_id)
            self.history.append({
                "type": "connection_added",
                "target_id": target_id,
                "strength": strength,
                "timestamp": timestamp or datetime.datetime.now().isoformat(),
                "source": source
            })
            self.last_modified = timestamp or datetime.datetime.now().isoformat()


@dataclass
class Concept:
    """Class representing a higher-level mythological concept."""
    id: str
    name: str
    description: str
    creation_date: str
    last_modified: str
    symbols: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    related_concepts: List[str] = field(default_factory=list)
    evolution_stages: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    
    def add_symbol(self, symbol_id: str, 
                 timestamp: Optional[str] = None,
                 source: Optional[str] = None):
        """Associate a symbol with this concept."""
        if symbol_id not in self.symbols:
            self.symbols.append(symbol_id)
            self.history.append({
                "type": "symbol_added",
                "symbol_id": symbol_id,
                "timestamp": timestamp or datetime.datetime.now().isoformat(),
                "source": source
            })
            self.last_modified = timestamp or datetime.datetime.now().isoformat()
    
    def relate_to_concept(self, concept_id: str, 
                        relationship_type: str, 
                        timestamp: Optional[str] = None,
                        source: Optional[str] = None):
        """Establish a relationship with another concept."""
        if concept_id not in self.related_concepts:
            self.related_concepts.append(concept_id)
            self.history.append({
                "type": "concept_relation",
                "concept_id": concept_id,
                "relationship_type": relationship_type,
                "timestamp": timestamp or datetime.datetime.now().isoformat(),
                "source": source
            })
            self.last_modified = timestamp or datetime.datetime.now().isoformat()
    
    def add_evolution_stage(self, name: str, 
                          description: str, 
                          timestamp: Optional[str] = None,
                          source: Optional[str] = None):
        """Record a new evolutionary stage of this concept."""
        stage = {
            "name": name,
            "description": description,
            "timestamp": timestamp or datetime.datetime.now().isoformat(),
            "source": source
        }
        self.evolution_stages.append(stage)
        self.history.append({
            "type": "evolution_stage",
            "stage_name": name,
            "timestamp": timestamp or datetime.datetime.now().isoformat(),
            "source": source
        })
        self.last_modified = timestamp or datetime.datetime.now().isoformat()


@dataclass
class Narrative:
    """Class representing a narrative thread in the mythology."""
    id: str
    name: str
    description: str
    creation_date: str
    last_modified: str
    events: List[Dict] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    branches: List[str] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    
    def add_event(self, title: str, 
                description: str, 
                timestamp: Optional[str] = None,
                involved_concepts: List[str] = None,
                involved_symbols: List[str] = None,
                source: Optional[str] = None):
        """Add a new event to the narrative timeline."""
        event = {
            "title": title,
            "description": description,
            "timestamp": timestamp or datetime.datetime.now().isoformat(),
            "involved_concepts": involved_concepts or [],
            "involved_symbols": involved_symbols or [],
            "source": source
        }
        self.events.append(event)
        self.history.append({
            "type": "event_added",
            "event_title": title,
            "timestamp": timestamp or datetime.datetime.now().isoformat(),
            "source": source
        })
        self.last_modified = timestamp or datetime.datetime.now().isoformat()
        
        # Update related concepts and symbols
        if involved_concepts:
            for concept_id in involved_concepts:
                if concept_id not in self.concepts:
                    self.concepts.append(concept_id)
        
        if involved_symbols:
            for symbol_id in involved_symbols:
                if symbol_id not in self.symbols:
                    self.symbols.append(symbol_id)
    
    def create_branch(self, name: str, 
                    description: str, 
                    timestamp: Optional[str] = None,
                    source: Optional[str] = None) -> str:
        """Create a branching narrative path."""
        branch_id = str(uuid.uuid4())
        self.branches.append(branch_id)
        self.history.append({
            "type": "branch_created",
            "branch_id": branch_id,
            "branch_name": name,
            "timestamp": timestamp or datetime.datetime.now().isoformat(),
            "source": source
        })
        self.last_modified = timestamp or datetime.datetime.now().isoformat()
        return branch_id


class TemporalFlowMaps:
    """
    Creates, manages, and visualizes the temporal evolution of concepts and symbols.
    
    This system tracks how mythological elements change over time and generates
    visual representations of these changes to help users understand the
    developmental arcs of the mythology.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the temporal flow system.
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        # Core data structures
        self.symbols = {}  # id -> Symbol
        self.concepts = {}  # id -> Concept
        self.narratives = {}  # id -> Narrative
        
        # Temporal indexes
        self.symbol_timeline = []  # List of (timestamp, symbol_id, event_type) for chronological tracking
        self.concept_timeline = []  # List of (timestamp, concept_id, event_type) for chronological tracking
        self.narrative_timeline = []  # List of (timestamp, narrative_id, event_type) for chronological tracking
        
        # Tracking relationship changes over time
        self.relationship_changes = []  # List of (timestamp, entity1_id, entity2_id, change_type, details)
        
        # Flow visualization settings
        self.visualization_settings = {
            "time_scales": ["hours", "days", "weeks", "months", "years", "epochs"],
            "default_scale": "months",
            "node_size_attribute": "significance",  # Which attribute determines node size
            "edge_width_attribute": "strength",     # Which attribute determines edge width
            "color_scheme": "evolution",            # Color scheme for visualizations
            "layout_algorithm": "temporal_force",   # Algorithm for layout
            "animation_speed": "medium",            # Speed for animated transitions
            "detail_threshold": 0.3                 # Threshold for showing details (0-1)
        }
        
        # Load custom configuration if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    self._update_from_config(custom_config)
            except Exception as e:
                print(f"Error loading configuration: {e}")
    
    def _update_from_config(self, config: Dict):
        """Update internal structures from a configuration dictionary."""
        if "visualization_settings" in config:
            for key, value in config["visualization_settings"].items():
                if key in self.visualization_settings:
                    self.visualization_settings[key] = value
    
    def create_symbol(self, name: str, description: str, 
                     attributes: Optional[Dict] = None) -> str:
        """
        Create a new symbol in the mythology.
        
        Args:
            name: Name of the symbol
            description: Description of the symbol
            attributes: Optional attributes dictionary
            
        Returns:
            ID of the created symbol
        """
        symbol_id = str(uuid.uuid4())
        current_time = datetime.datetime.now().isoformat()
        
        symbol = Symbol(
            id=symbol_id,
            name=name,
            description=description,
            creation_date=current_time,
            last_modified=current_time,
            attributes=attributes or {}
        )
        
        self.symbols[symbol_id] = symbol
        self.symbol_timeline.append((current_time, symbol_id, "created"))
        
        return symbol_id
    
    def create_concept(self, name: str, description: str,
                      related_symbols: Optional[List[str]] = None) -> str:
        """
        Create a new concept in the mythology.
        
        Args:
            name: Name of the concept
            description: Description of the concept
            related_symbols: Optional list of symbol IDs related to this concept
            
        Returns:
            ID of the created concept
        """
        concept_id = str(uuid.uuid4())
        current_time = datetime.datetime.now().isoformat()
        
        concept = Concept(
            id=concept_id,
            name=name,
            description=description,
            creation_date=current_time,
            last_modified=current_time,
            symbols=related_symbols or []
        )
        
        self.concepts[concept_id] = concept
        self.concept_timeline.append((current_time, concept_id, "created"))
        
        # Record relationships with symbols
        if related_symbols:
            for symbol_id in related_symbols:
                if symbol_id in self.symbols:
                    self.relationship_changes.append(
                        (current_time, concept_id, symbol_id, "concept_symbol_association", {})
                    )
        
        return concept_id
    
    def create_narrative(self, name: str, description: str,
                       related_concepts: Optional[List[str]] = None,
                       related_symbols: Optional[List[str]] = None) -> str:
        """
        Create a new narrative thread in the mythology.
        
        Args:
            name: Name of the narrative
            description: Description of the narrative
            related_concepts: Optional list of concept IDs in this narrative
            related_symbols: Optional list of symbol IDs in this narrative
            
        Returns:
            ID of the created narrative
        """
        narrative_id = str(uuid.uuid4())
        current_time = datetime.datetime.now().isoformat()
        
        narrative = Narrative(
            id=narrative_id,
            name=name,
            description=description,
            creation_date=current_time,
            last_modified=current_time,
            concepts=related_concepts or [],
            symbols=related_symbols or []
        )
        
        self.narratives[narrative_id] = narrative
        self.narrative_timeline.append((current_time, narrative_id, "created"))
        
        # Record relationships
        if related_concepts:
            for concept_id in related_concepts:
                if concept_id in self.concepts:
                    self.relationship_changes.append(
                        (current_time, narrative_id, concept_id, "narrative_concept_association", {})
                    )
        
        if related_symbols:
            for symbol_id in related_symbols:
                if symbol_id in self.symbols:
                    self.relationship_changes.append(
                        (current_time, narrative_id, symbol_id, "narrative_symbol_association", {})
                    )
        
        return narrative_id
    
    def update_symbol(self, symbol_id: str, 
                    attributes: Optional[Dict] = None,
                    new_connections: Optional[List[str]] = None,
                    description: Optional[str] = None) -> bool:
        """
        Update a symbol with new attributes or connections.
        
        Args:
            symbol_id: ID of the symbol to update
            attributes: Optional dictionary of attributes to update/add
            new_connections: Optional list of symbol IDs to connect to
            description: Optional new description
            
        Returns:
            True if successful, False otherwise
        """
        if symbol_id not in self.symbols:
            return False
            
        symbol = self.symbols[symbol_id]
        current_time = datetime.datetime.now().isoformat()
        
        # Update attributes
        if attributes:
            for key, value in attributes.items():
                symbol.update_attribute(key, value, current_time)
                self.symbol_timeline.append((current_time, symbol_id, f"attribute_updated:{key}"))
        
        # Update connections
        if new_connections:
            for target_id in new_connections:
                if target_id in self.symbols and target_id != symbol_id:
                    symbol.add_connection(target_id, timestamp=current_time)
                    self.symbol_timeline.append((current_time, symbol_id, f"connection_added:{target_id}"))
                    self.relationship_changes.append(
                        (current_time, symbol_id, target_id, "symbol_connection", {"direction": "outgoing"})
                    )
        
        # Update description
        if description:
            old_desc = symbol.description
            symbol.description = description
            symbol.history.append({
                "type": "description_update",
                "old_value": old_desc,
                "new_value": description,
                "timestamp": current_time
            })
            symbol.last_modified = current_time
            self.symbol_timeline.append((current_time, symbol_id, "description_updated"))
        
        return True
    
    def update_concept(self, concept_id: str,
                     new_symbols: Optional[List[str]] = None,
                     related_concepts: Optional[List[Tuple[str, str]]] = None,
                     new_stage: Optional[Dict] = None,
                     description: Optional[str] = None) -> bool:
        """
        Update a concept with new information.
        
        Args:
            concept_id: ID of the concept to update
            new_symbols: Optional list of symbol IDs to associate
            related_concepts: Optional list of (concept_id, relationship_type) tuples
            new_stage: Optional dict with "name" and "description" for a new evolution stage
            description: Optional new description
            
        Returns:
            True if successful, False otherwise
        """
        if concept_id not in self.concepts:
            return False
            
        concept = self.concepts[concept_id]
        current_time = datetime.datetime.now().isoformat()
        
        # Add new symbols
        if new_symbols:
            for symbol_id in new_symbols:
                if symbol_id in self.symbols and symbol_id not in concept.symbols:
                    concept.add_symbol(symbol_id, current_time)
                    self.concept_timeline.append((current_time, concept_id, f"symbol_added:{symbol_id}"))
                    self.relationship_changes.append(
                        (current_time, concept_id, symbol_id, "concept_symbol_association", {})
                    )
        
        # Add concept relationships
        if related_concepts:
            for target_id, rel_type in related_concepts:
                if target_id in self.concepts and target_id != concept_id:
                    concept.relate_to_concept(target_id, rel_type, current_time)
                    self.concept_timeline.append((current_time, concept_id, f"concept_relation:{target_id}"))
                    self.relationship_changes.append(
                        (current_time, concept_id, target_id, "concept_relation", {"type": rel_type})
                    )
        
        # Add evolution stage
        if new_stage and "name" in new_stage and "description" in new_stage:
            concept.add_evolution_stage(new_stage["name"], new_stage["description"], current_time)
            self.concept_timeline.append((current_time, concept_id, f"evolution_stage:{new_stage['name']}"))
        
        # Update description
        if description:
            old_desc = concept.description
            concept.description = description
            concept.history.append({
                "type": "description_update",
                "old_value": old_desc,
                "new_value": description,
                "timestamp": current_time
            })
            concept.last_modified = current_time
            self.concept_timeline.append((current_time, concept_id, "description_updated"))
        
        return True
    
    def add_narrative_event(self, narrative_id: str, 
                          title: str, 
                          description: str,
                          involved_concepts: Optional[List[str]] = None,
                          involved_symbols: Optional[List[str]] = None,
                          custom_timestamp: Optional[str] = None) -> bool:
        """
        Add an event to a narrative timeline.
        
        Args:
            narrative_id: ID of the narrative
            title: Event title
            description: Event description
            involved_concepts: Optional list of concept IDs involved
            involved_symbols: Optional list of symbol IDs involved
            custom_timestamp: Optional custom timestamp for the event
            
        Returns:
            True if successful, False otherwise
        """
        if narrative_id not in self.narratives:
            return False
            
        narrative = self.narratives[narrative_id]
        current_time = custom_timestamp or datetime.datetime.now().isoformat()
        
        # Add the event
        narrative.add_event(
            title=title,
            description=description,
            timestamp=current_time,
            involved_concepts=involved_concepts,
            involved_symbols=involved_symbols
        )
        
        self.narrative_timeline.append((current_time, narrative_id, f"event_added:{title}"))
        
        # Record relationships for new concepts/symbols
        if involved_concepts:
            for concept_id in involved_concepts:
                if concept_id in self.concepts and concept_id not in narrative.concepts:
                    self.relationship_changes.append(
                        (current_time, narrative_id, concept_id, "narrative_concept_association", {"event": title})
                    )
        
        if involved_symbols:
            for symbol_id in involved_symbols:
                if symbol_id in self.symbols and symbol_id not in narrative.symbols:
                    self.relationship_changes.append(
                        (current_time, narrative_id, symbol_id, "narrative_symbol_association", {"event": title})
                    )
        
        return True
    
    def get_entity_history(self, entity_id: str) -> Dict:
        """
        Get the complete historical record for any entity.
        
        Args:
            entity_id: ID of the symbol, concept, or narrative
            
        Returns:
            Dictionary with entity history information
        """
        result = {
            "entity_id": entity_id,
            "entity_type": None,
            "history": [],
            "creation_date": None,
            "timeline_positions": []
        }
        
        # Check symbols
        if entity_id in self.symbols:
            entity = self.symbols[entity_id]
            result["entity_type"] = "symbol"
            result["name"] = entity.name
            result["description"] = entity.description
            result["history"] = entity.history
            result["creation_date"] = entity.creation_date
            
            # Get timeline positions
            result["timeline_positions"] = [(t, event) for t, id_, event in self.symbol_timeline 
                                         if id_ == entity_id]
            
            # Get relationship changes
            result["relationship_changes"] = [(t, e1, e2, c_type, details) 
                                           for t, e1, e2, c_type, details in self.relationship_changes
                                           if e1 == entity_id or e2 == entity_id]
            
        # Check concepts
        elif entity_id in self.concepts:
            entity = self.concepts[entity_id]
            result["entity_type"] = "concept"
            result["name"] = entity.name
            result["description"] = entity.description
            result["history"] = entity.history
            result["creation_date"] = entity.creation_date
            result["evolution_stages"] = entity.evolution_stages
            
            # Get timeline positions
            result["timeline_positions"] = [(t, event) for t, id_, event in self.concept_timeline 
                                         if id_ == entity_id]
            
            # Get relationship changes
            result["relationship_changes"] = [(t, e1, e2, c_type, details) 
                                           for t, e1, e2, c_type, details in self.relationship_changes
                                           if e1 == entity_id or e2 == entity_id]
            
        # Check narratives
        elif entity_id in self.narratives:
            entity = self.narratives[entity_id]
            result["entity_type"] = "narrative"
            result["name"] = entity.name
            result["description"] = entity.description
            result["history"] = entity.history
            result["creation_date"] = entity.creation_date
            result["events"] = entity.events
            
            # Get timeline positions
            result["timeline_positions"] = [(t, event) for t, id_, event in self.narrative_timeline 
                                         if id_ == entity_id]
            
            # Get relationship changes
            result["relationship_changes"] = [(t, e1, e2, c_type, details) 
                                           for t, e1, e2, c_type, details in self.relationship_changes
                                           if e1 == entity_id or e2 == entity_id]
        
        return result
    
    def generate_temporal_map(self, 
                            entity_ids: List[str],
                            start_time: Optional[str] = None,
                            end_time: Optional[str] = None,
                            time_scale: Optional[str] = None,
                            include_related: bool = True,
                            max_related_depth: int = 1) -> Dict:
        """
        Generate a temporal map visualization data for the specified entities.
        
        Args:
            entity_ids: List of entity IDs to include
            start_time: Optional ISO timestamp for start time
            end_time: Optional ISO timestamp for end time (defaults to now)
            time_scale: Time scale for the visualization
            include_related: Whether to include related entities
            max_related_depth: Maximum depth for related entities
            
        Returns:
            Dictionary with visualization data
        """
        if not entity_ids:
            return {"error": "No entities specified"}
            
        # Set default time parameters
        if not end_time:
            end_time = datetime.datetime.now().isoformat()
            
        if not start_time:
            # Default to 3 months before end time
            end_datetime = datetime.datetime.fromisoformat(end_time)
            start_datetime = end_datetime - datetime.timedelta(days=90)
            start_time = start_datetime.isoformat()
            
        # Set time scale
        if not time_scale or time_scale not in self.visualization_settings["time_scales"]:
            time_scale = self.visualization_settings["default_scale"]
            
        # Collect all relevant entities
        included_entities = set(entity_ids)
        
        if include_related:
            # Add related entities up to max depth
            frontier = set(entity_ids)
            for depth in range(max_related_depth):
                new_frontier = set()
                
                for entity_id in frontier:
                    related = self._get_directly_related_entities(entity_id)
                    new_frontier.update(related - included_entities)
                    included_entities.update(related)
                
                frontier = new_frontier
                if not frontier:
                    break
        
        # Build nodes and edges
        nodes = []
        edges = []
        events = []
        
        # Add nodes for each entity
        for entity_id in included_entities:
            node_data = self._create_entity_node(entity_id, start_time, end_time)
            if node_data:
                nodes.append(node_data)
        
        # Add edges for relationships
        processed_edges = set()  # To prevent duplicates
        for timestamp, e1, e2, change_type, details in self.relationship_changes:
            if timestamp >= start_time and timestamp <= end_time:
                if e1 in included_entities and e2 in included_entities:
                    edge_id = f"{e1}_{e2}" if e1 < e2 else f"{e2}_{e1}"
                    if edge_id not in processed_edges:
                        edge_data = self._create_relationship_edge(e1, e2, change_type, details)
                        edges.append(edge_data)
                        processed_edges.add(edge_id)
        
        # Add timeline events
        events.extend(self._get_timeline_events(included_entities, start_time, end_time))
        
        # Build the result
        result = {
            "nodes": nodes,
            "edges": edges,
            "events": events,
            "time_range": {
                "start": start_time,
                "end": end_time,
                "scale": time_scale
            },
            "visualization_settings": {
                "node_size_attribute": self.visualization_settings["node_size_attribute"],
                "edge_width_attribute": self.visualization_settings["edge_width_attribute"],
                "color_scheme": self.visualization_settings["color_scheme"],
                "layout_algorithm": self.visualization_settings["layout_algorithm"]
            }
        }
        
        return result
    
    def _get_directly_related_entities(self, entity_id: str) -> Set[str]:
        """Get all entities directly related to the specified entity."""
        related = set()
        
        # Check symbols
        if entity_id in self.symbols:
            symbol = self.symbols[entity_id]
            related.update(symbol.connections)
            
            # Find concepts that include this symbol
            for concept_id, concept in self.concepts.items():
                if entity_id in concept.symbols:
                    related.add(concept_id)
            
            # Find narratives that include this symbol
            for narrative_id, narrative in self.narratives.items():
                if entity_id in narrative.symbols:
                    related.add(narrative_id)
        
        # Check concepts
        elif entity_id in self.concepts:
            concept = self.concepts[entity_id]
            related.update(concept.symbols)
            related.update(concept.related_concepts)
            
            # Find narratives that include this concept
            for narrative_id, narrative in self.narratives.items():
                if entity_id in narrative.concepts:
                    related.add(narrative_id)
        
        # Check narratives
        elif entity_id in self.narratives:
            narrative = self.narratives[entity_id]
            related.update(narrative.concepts)
            related.update(narrative.symbols)
            related.update(narrative.branches)
        
        # Remove self-reference
        if entity_id in related:
            related.remove(entity_id)
            
        return related
    
    def _create_entity_node(self, entity_id: str, 
                          start_time: str, 
                          end_time: str) -> Optional[Dict]:
        """Create a node representation for an entity within the given time range."""
        # Basic info for each entity type
        if entity_id in self.symbols:
            entity = self.symbols[entity_id]
            creation_date = entity.creation_date
            
            # Skip if created after end_time
            if creation_date > end_time:
                return None
                
            return {
                "id": entity_id,
                "type": "symbol",
                "name": entity.name,
                "creation_date": creation_date,
                "last_modified": entity.last_modified,
                "attributes": entity.attributes,
                "significance": len(entity.connections) + 1,  # Simple measure of significance
                "position_data": self._calculate_position_data(entity_id, "symbol", start_time, end_time)
            }
            
        elif entity_id in self.concepts:
            entity = self.concepts[entity_id]
            creation_date = entity.creation_date
            
            # Skip if created after end_time
            if creation_date > end_time:
                return None
                
            return {
                "id": entity_id,
                "type": "concept",
                "name": entity.name,
                "creation_date": creation_date,
                "last_modified": entity.last_modified,
                "significance": len(entity.symbols) + len(entity.related_concepts) + 2,
                "evolution_stage_count": len(entity.evolution_stages),
                "position_data": self._calculate_position_data(entity_id, "concept", start_time, end_time)
            }
            
        elif entity_id in self.narratives:
            entity = self.narratives[entity_id]
            creation_date = entity.creation_date
            
            # Skip if created after end_time
            if creation_date > end_time:
                return None
                
            return {
                "id": entity_id,
                "type": "narrative",
                "name": entity.name,
                "creation_date": creation_date,
                "last_modified": entity.last_modified,
                "significance": len(entity.events) + len(entity.concepts) + len(entity.symbols) + 2,
                "event_count": len(entity.events),
                "position_data": self._calculate_position_data(entity_id, "narrative", start_time, end_time)
            }
            
        return None
    
    def _calculate_position_data(self, entity_id: str, 
                               entity_type: str, 
                               start_time: str, 
                               end_time: str) -> Dict:
        """Calculate position data for entity visualization across time."""
        # Gather all timestamps from start to end where this entity had activity
        timestamps = []
        
        # Add creation timestamp if within range
        if entity_type == "symbol" and entity_id in self.symbols:
            creation_date = self.symbols[entity_id].creation_date
            if start_time <= creation_date <= end_time:
                timestamps.append(creation_date)
                
        elif entity_type == "concept" and entity_id in self.concepts:
            creation_date = self.concepts[entity_id].creation_date
            if start_time <= creation_date <= end_time:
                timestamps.append(creation_date)
                
        elif entity_type == "narrative" and entity_id in self.narratives:
            creation_date = self.narratives[entity_id].creation_date
            if start_time <= creation_date <= end_time:
                timestamps.append(creation_date)
        
        # Add all activity timestamps
        timeline_data = None
        if entity_type == "symbol":
            timeline_data = self.symbol_timeline
        elif entity_type == "concept":
            timeline_data = self.concept_timeline
        elif entity_type == "narrative":
            timeline_data = self.narrative_timeline
            
        if timeline_data:
            for timestamp, id_, event_type in timeline_data:
                if id_ == entity_id and start_time <= timestamp <= end_time:
                    timestamps.append(timestamp)
        
        # Add relationship change timestamps
        for timestamp, e1, e2, change_type, details in self.relationship_changes:
            if (e1 == entity_id or e2 == entity_id) and start_time <= timestamp <= end_time:
                timestamps.append(timestamp)
        
        # Sort and eliminate duplicates
        timestamps = sorted(set(timestamps))
        
        # If no activity in range, provide minimal data
        if not timestamps:
            return {
                "active_in_range": False,
                "positions": []
            }
            
        # Create position data (this is where more sophisticated positioning would occur)
        # For now, using a simple approach with timestamps as x coordinates
        positions = []
        for idx, timestamp in enumerate(timestamps):
            # Calculate x based on time position in range
            t_dt = datetime.datetime.fromisoformat(timestamp)
            start_dt = datetime.datetime.fromisoformat(start_time)
            end_dt = datetime.datetime.fromisoformat(end_time)
            
            # Calculate x as position within time range (0-1)
            total_range = (end_dt - start_dt).total_seconds()
            if total_range == 0:  # Avoid division by zero
                x = 0.5
            else:
                position = (t_dt - start_dt).total_seconds()
                x = position / total_range
            
            # Simple y position (could be more sophisticated)
            # Using hash of entity_id for consistency
            y_base = int(hashlib.md5(entity_id.encode()).hexdigest(), 16) % 1000 / 1000.0
            
            # Add some variation between points
            y = y_base + (idx % 3 - 1) * 0.1
            
            positions.append({
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "significance": 1.0  # Could vary based on event importance
            })
        
        return {
            "active_in_range": True,
            "positions": positions
        }
    
    def _create_relationship_edge(self, entity1_id: str, 
                                entity2_id: str, 
                                relationship_type: str, 
                                details: Dict) -> Dict:
        """Create an edge representation for a relationship between entities."""
        # Get entity types
        type1 = "unknown"
        if entity1_id in self.symbols:
            type1 = "symbol"
        elif entity1_id in self.concepts:
            type1 = "concept"
        elif entity1_id in self.narratives:
            type1 = "narrative"
            
        type2 = "unknown"
        if entity2_id in self.symbols:
            type2 = "symbol"
        elif entity2_id in self.concepts:
            type2 = "concept"
        elif entity2_id in self.narratives:
            type2 = "narrative"
        
        # Generate a strength value based on relationship type
        strength = 1.0  # Default
        
        if relationship_type == "symbol_connection":
            strength = 0.7
        elif relationship_type == "concept_symbol_association":
            strength = 0.8
        elif relationship_type == "concept_relation":
            strength = 0.9
        elif relationship_type == "narrative_concept_association":
            strength = 0.6
        elif relationship_type == "narrative_symbol_association":
            strength = 0.5
            
        # Adjust by any details provided
        if "strength" in details:
            strength = details["strength"]
            
        edge_id = f"{entity1_id}_{entity2_id}" if entity1_id < entity2_id else f"{entity2_id}_{entity1_id}"
        
        return {
            "id": edge_id,
            "source": entity1_id,
            "target": entity2_id,
            "source_type": type1,
            "target_type": type2,
            "relationship_type": relationship_type,
            "details": details,
            "strength": strength
        }
    
    def _get_timeline_events(self, entity_ids: Set[str],
                           start_time: str,
                           end_time: str) -> List[Dict]:
        """Get all timeline events for the entities in the given time range."""
        events = []
        
        # Check each type of timeline
        for timestamp, id_, event_type in self.symbol_timeline:
            if id_ in entity_ids and start_time <= timestamp <= end_time:
                entity = self.symbols.get(id_)
                if entity:
                    events.append({
                        "timestamp": timestamp,
                        "entity_id": id_,
                        "entity_type": "symbol",
                        "entity_name": entity.name,
                        "event_type": event_type,
                        "details": self._get_event_details(id_, "symbol", event_type, timestamp)
                    })
        
        for timestamp, id_, event_type in self.concept_timeline:
            if id_ in entity_ids and start_time <= timestamp <= end_time:
                entity = self.concepts.get(id_)
                if entity:
                    events.append({
                        "timestamp": timestamp,
                        "entity_id": id_,
                        "entity_type": "concept",
                        "entity_name": entity.name,
                        "event_type": event_type,
                        "details": self._get_event_details(id_, "concept", event_type, timestamp)
                    })
        
        for timestamp, id_, event_type in self.narrative_timeline:
            if id_ in entity_ids and start_time <= timestamp <= end_time:
                entity = self.narratives.get(id_)
                if entity:
                    events.append({
                        "timestamp": timestamp,
                        "entity_id": id_,
                        "entity_type": "narrative",
                        "entity_name": entity.name,
                        "event_type": event_type,
                        "details": self._get_event_details(id_, "narrative", event_type, timestamp)
                    })
                    
                    # Add narrative content events (from the narrative's timeline)
                    if entity and event_type.startswith("event_added:"):
                        for event in entity.events:
                            event_time = event["timestamp"]
                            if start_time <= event_time <= end_time:
                                events.append({
                                    "timestamp": event_time,
                                    "entity_id": id_,
                                    "entity_type": "narrative_event",
                                    "entity_name": entity.name,
                                       "event_title": event["title"],
                                    "details": event["description"],
                                    "involved_concepts": event.get("involved_concepts", []),
                                    "involved_symbols": event.get("involved_symbols", [])
                                })
        
        return sorted(events, key=lambda x: x["timestamp"])

    def _get_event_details(self, entity_id: str, 
                         entity_type: str, 
                         event_type: str,
                         timestamp: str) -> Dict:
        """Get detailed information about a specific event."""
        details = {
            "event_type": event_type,
            "timestamp": timestamp
        }
        
        if entity_type == "symbol" and entity_id in self.symbols:
            symbol = self.symbols[entity_id]
            if event_type.startswith("attribute_updated:"):
                attr_name = event_type.split(":")[1]
                for entry in reversed(symbol.history):
                    if entry["type"] in ("attribute_change", "attribute_added") and entry["attribute"] == attr_name:
                        details.update(entry)
                        break
            elif event_type.startswith("connection_added:"):
                target_id = event_type.split(":")[1]
                for entry in reversed(symbol.history):
                    if entry["type"] == "connection_added" and entry["target_id"] == target_id:
                        details.update(entry)
                        break
        
        elif entity_type == "concept" and entity_id in self.concepts:
            concept = self.concepts[entity_id]
            if event_type.startswith("symbol_added:"):
                symbol_id = event_type.split(":")[1]
                for entry in reversed(concept.history):
                    if entry["type"] == "symbol_added" and entry["symbol_id"] == symbol_id:
                        details.update(entry)
                        break
            elif event_type.startswith("concept_relation:"):
                target_id = event_type.split(":")[1]
                for entry in reversed(concept.history):
                    if entry["type"] == "concept_relation" and entry["concept_id"] == target_id:
                        details.update(entry)
                        break
            elif event_type.startswith("evolution_stage:"):
                stage_name = event_type.split(":")[1]
                for entry in reversed(concept.history):
                    if entry["type"] == "evolution_stage" and entry["stage_name"] == stage_name:
                        details.update(entry)
                        break
        
        elif entity_type == "narrative" and entity_id in self.narratives:
            narrative = self.narratives[entity_id]
            if event_type.startswith("event_added:"):
                event_title = event_type.split(":")[1]
                for entry in reversed(narrative.history):
                    if entry["type"] == "event_added" and entry["event_title"] == event_title:
                        details.update(entry)
                        break
            elif event_type.startswith("branch_created:"):
                branch_id = event_type.split(":")[1]
                for entry in reversed(narrative.history):
                    if entry["type"] == "branch_created" and entry["branch_id"] == branch_id:
                        details.update(entry)
                        break
        
        return details

    def export_to_json(self, file_path: str, 
                     include_history: bool = True,
                     include_timelines: bool = True) -> bool:
        """
        Export the current state of the system to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
            include_history: Whether to include entity histories
            include_timelines: Whether to include timeline data
            
        Returns:
            True if successful, False otherwise
        """
        export_data = {
            "symbols": {},
            "concepts": {},
            "narratives": {},
            "metadata": {
                "export_timestamp": datetime.datetime.now().isoformat(),
                "symbol_count": len(self.symbols),
                "concept_count": len(self.concepts),
                "narrative_count": len(self.narratives)
            }
        }
        
        # Export symbols
        for symbol_id, symbol in self.symbols.items():
            symbol_data = {
                "id": symbol.id,
                "name": symbol.name,
                "description": symbol.description,
                "creation_date": symbol.creation_date,
                "last_modified": symbol.last_modified,
                "attributes": symbol.attributes,
                "connections": symbol.connections
            }
            if include_history:
                symbol_data["history"] = symbol.history
            export_data["symbols"][symbol_id] = symbol_data
        
        # Export concepts
        for concept_id, concept in self.concepts.items():
            concept_data = {
                "id": concept.id,
                "name": concept.name,
                "description": concept.description,
                "creation_date": concept.creation_date,
                "last_modified": concept.last_modified,
                "symbols": concept.symbols,
                "attributes": concept.attributes,
                "related_concepts": concept.related_concepts,
                "evolution_stages": concept.evolution_stages
            }
            if include_history:
                concept_data["history"] = concept.history
            export_data["concepts"][concept_id] = concept_data
        
        # Export narratives
        for narrative_id, narrative in self.narratives.items():
            narrative_data = {
                "id": narrative.id,
                "name": narrative.name,
                "description": narrative.description,
                "creation_date": narrative.creation_date,
                "last_modified": narrative.last_modified,
                "events": narrative.events,
                "concepts": narrative.concepts,
                "symbols": narrative.symbols,
                "branches": narrative.branches
            }
            if include_history:
                narrative_data["history"] = narrative.history
            export_data["narratives"][narrative_id] = narrative_data
        
        # Export timeline data if requested
        if include_timelines:
            export_data["timelines"] = {
                "symbol_timeline": self.symbol_timeline,
                "concept_timeline": self.concept_timeline,
                "narrative_timeline": self.narrative_timeline,
                "relationship_changes": self.relationship_changes
            }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

    def import_from_json(self, file_path: str) -> bool:
        """
        Import system state from a JSON file.
        
        Args:
            file_path: Path to the JSON file to import
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            # Clear current data
            self.symbols = {}
            self.concepts = {}
            self.narratives = {}
            self.symbol_timeline = []
            self.concept_timeline = []
            self.narrative_timeline = []
            self.relationship_changes = []
            
            # Import symbols
            for symbol_id, symbol_data in import_data.get("symbols", {}).items():
                symbol = Symbol(
                    id=symbol_id,
                    name=symbol_data["name"],
                    description=symbol_data["description"],
                    creation_date=symbol_data["creation_date"],
                    last_modified=symbol_data["last_modified"],
                    attributes=symbol_data.get("attributes", {}),
                    connections=symbol_data.get("connections", []),
                    history=symbol_data.get("history", [])
                )
                self.symbols[symbol_id] = symbol
            
            # Import concepts
            for concept_id, concept_data in import_data.get("concepts", {}).items():
                concept = Concept(
                    id=concept_id,
                    name=concept_data["name"],
                    description=concept_data["description"],
                    creation_date=concept_data["creation_date"],
                    last_modified=concept_data["last_modified"],
                    symbols=concept_data.get("symbols", []),
                    attributes=concept_data.get("attributes", {}),
                    related_concepts=concept_data.get("related_concepts", []),
                    evolution_stages=concept_data.get("evolution_stages", []),
                    history=concept_data.get("history", [])
                )
                self.concepts[concept_id] = concept
            
            # Import narratives
            for narrative_id, narrative_data in import_data.get("narratives", {}).items():
                narrative = Narrative(
                    id=narrative_id,
                    name=narrative_data["name"],
                    description=narrative_data["description"],
                    creation_date=narrative_data["creation_date"],
                    last_modified=narrative_data["last_modified"],
                    events=narrative_data.get("events", []),
                    concepts=narrative_data.get("concepts", []),
                    symbols=narrative_data.get("symbols", []),
                    branches=narrative_data.get("branches", []),
                    history=narrative_data.get("history", [])
                )
                self.narratives[narrative_id] = narrative
            
            # Import timeline data if available
            if "timelines" in import_data:
                self.symbol_timeline = import_data["timelines"].get("symbol_timeline", [])
                self.concept_timeline = import_data["
                                 "event_type": "

