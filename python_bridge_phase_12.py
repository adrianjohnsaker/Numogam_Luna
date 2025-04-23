#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Bridge
Phase XII - Amelia AI Project

This module serves as the API layer between the Python modules and the Kotlin application.
It exposes the functionality of all three Phase XII modules through a unified interface.
"""

import json
import traceback
import datetime
from typing import Dict, List, Any, Optional, Union, Callable

# Import the Phase XII modules
from circadian_narrative_cycles import CircadianNarrativeCycles
from ritual_interaction_patterns import RitualInteractionPatterns
from temporal_flow_maps import TemporalFlowMaps

class Bridge:
    """
    Bridge class that exposes Python module APIs to the Kotlin application.
    
    This class serves as a unified interface for all three Phase XII modules:
    - Circadian Narrative Cycles
    - Ritual Interaction Patterns
    - Temporal Flow Maps
    
    It handles JSON serialization/deserialization, error handling, and
    provides a clean API for the Kotlin code to interact with.
    """
    
    def __init__(self):
        """Initialize the bridge with empty module instances."""
        self.circadian_instances = {}  # id -> CircadianNarrativeCycles
        self.ritual_instances = {}     # id -> RitualInteractionPatterns
        self.temporal_instances = {}   # id -> TemporalFlowMaps
        
    def _safe_call(self, func: Callable, *args, **kwargs) -> Dict:
        """
        Safely execute a function and catch any exceptions.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Dict with success status and either result or error information
        """
        try:
            result = func(*args, **kwargs)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            error_info = {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            return {
                "success": False,
                "error": error_info
            }
    
    #----------------------------------------
    # Circadian Narrative Cycles Bridge Methods
    #----------------------------------------
    
    def create_circadian_instance(self, config_path: Optional[str] = None) -> Dict:
        """
        Create a new CircadianNarrativeCycles instance.
        
        Args:
            config_path: Optional path to a JSON configuration file
            
        Returns:
            Dict with success status and instance ID if successful
        """
        def _create():
            instance = CircadianNarrativeCycles(config_path)
            instance_id = f"circadian_{len(self.circadian_instances)}"
            self.circadian_instances[instance_id] = instance
            return instance_id
            
        return self._safe_call(_create)
    
    def get_current_narrative_tone(self, instance_id: str, datetime_str: Optional[str] = None) -> Dict:
        """
        Get the narrative tone for the current or specified time.
        
        Args:
            instance_id: The CircadianNarrativeCycles instance ID
            datetime_str: Optional ISO format datetime string
            
        Returns:
            Dict with success status and tone information if successful
        """
        def _get_tone():
            if instance_id not in self.circadian_instances:
                raise ValueError(f"Unknown CircadianNarrativeCycles instance: {instance_id}")
                
            instance = self.circadian_instances[instance_id]
            
            if datetime_str:
                dt = datetime.datetime.fromisoformat(datetime_str)
                return instance.get_current_narrative_tone(dt)
            else:
                return instance.get_current_narrative_tone()
                
        return self._safe_call(_get_tone)
    
    def transform_narrative(self, 
                          instance_id: str, 
                          narrative_text: str,
                          datetime_str: Optional[str] = None,
                          transformation_strength: float = 1.0) -> Dict:
        """
        Transform a narrative text based on the circadian tone.
        
        Args:
            instance_id: The CircadianNarrativeCycles instance ID
            narrative_text: Text to transform
            datetime_str: Optional ISO format datetime string
            transformation_strength: How strongly to apply the transformation (0.0-1.0)
            
        Returns:
            Dict with success status and transformed text if successful
        """
        def _transform():
            if instance_id not in self.circadian_instances:
                raise ValueError(f"Unknown CircadianNarrativeCycles instance: {instance_id}")
                
            instance = self.circadian_instances[instance_id]
            
            if datetime_str:
                dt = datetime.datetime.fromisoformat(datetime_str)
                return instance.transform_narrative(narrative_text, dt, transformation_strength)
            else:
                return instance.transform_narrative(narrative_text, None, transformation_strength)
                
        return self._safe_call(_transform)
    
    def generate_tone_specific_prompt(self,
                                   instance_id: str,
                                   base_prompt: str,
                                   datetime_str: Optional[str] = None) -> Dict:
        """
        Generate a tone-specific version of a base prompt.
        
        Args:
            instance_id: The CircadianNarrativeCycles instance ID
            base_prompt: Original prompt to modify
            datetime_str: Optional ISO format datetime string
            
        Returns:
            Dict with success status and enhanced prompt if successful
        """
        def _generate():
            if instance_id not in self.circadian_instances:
                raise ValueError(f"Unknown CircadianNarrativeCycles instance: {instance_id}")
                
            instance = self.circadian_instances[instance_id]
            
            if datetime_str:
                dt = datetime.datetime.fromisoformat(datetime_str)
                return instance.generate_tone_specific_prompt(base_prompt, dt)
            else:
                return instance.generate_tone_specific_prompt(base_prompt)
                
        return self._safe_call(_generate)
    
    def get_tone_schedule(self, instance_id: str) -> Dict:
        """
        Get the full schedule of tones throughout a 24-hour cycle.
        
        Args:
            instance_id: The CircadianNarrativeCycles instance ID
            
        Returns:
            Dict with success status and tone schedule if successful
        """
        def _get_schedule():
            if instance_id not in self.circadian_instances:
                raise ValueError(f"Unknown CircadianNarrativeCycles instance: {instance_id}")
                
            instance = self.circadian_instances[instance_id]
            return instance.get_tone_schedule()
                
        return self._safe_call(_get_schedule)
    
    #----------------------------------------
    # Ritual Interaction Patterns Bridge Methods
    #----------------------------------------
    
    def create_ritual_instance_manager(self, config_path: Optional[str] = None) -> Dict:
        """
        Create a new RitualInteractionPatterns instance.
        
        Args:
            config_path: Optional path to a JSON configuration file
            
        Returns:
            Dict with success status and instance ID if successful
        """
        def _create():
            instance = RitualInteractionPatterns(config_path)
            instance_id = f"ritual_{len(self.ritual_instances)}"
            self.ritual_instances[instance_id] = instance
            return instance_id
            
        return self._safe_call(_create)
    
    def identify_ritual_opportunity(self,
                                 instance_id: str,
                                 context_json: str,
                                 user_history_json: Optional[str] = None) -> Dict:
        """
        Identify appropriate ritual opportunities based on current context.
        
        Args:
            instance_id: The RitualInteractionPatterns instance ID
            context_json: JSON string with context information
            user_history_json: Optional JSON string with user's ritual history
            
        Returns:
            Dict with success status and ritual suggestion if successful
        """
        def _identify():
            if instance_id not in self.ritual_instances:
                raise ValueError(f"Unknown RitualInteractionPatterns instance: {instance_id}")
                
            instance = self.ritual_instances[instance_id]
            
            # Parse JSON input
            context = json.loads(context_json)
            
            # Convert ISO datetime string to datetime object if present
            if "current_time" in context and isinstance(context["current_time"], str):
                context["current_time"] = datetime.datetime.fromisoformat(context["current_time"])
            
            # Parse user history if provided
            user_history = json.loads(user_history_json) if user_history_json else None
            
            return instance.identify_ritual_opportunity(context, user_history)
                
        return self._safe_call(_identify)
    
    def create_ritual(self,
                   instance_id: str,
                   ritual_type: str,
                   user_id: str,
                   concepts_json: str,
                   context_json: Optional[str] = None) -> Dict:
        """
        Create a new ritual instance for tracking.
        
        Args:
            instance_id: The RitualInteractionPatterns instance ID
            ritual_type: Type of ritual from templates
            user_id: Identifier for the user
            concepts_json: JSON string array of concept IDs
            context_json: Optional JSON string with additional context
            
        Returns:
            Dict with success status and ritual ID if successful
        """
        def _create():
            if instance_id not in self.ritual_instances:
                raise ValueError(f"Unknown RitualInteractionPatterns instance: {instance_id}")
                
            instance = self.ritual_instances[instance_id]
            
            # Parse JSON input
            concepts = json.loads(concepts_json)
            context = json.loads(context_json) if context_json else None
            
            return instance.create_ritual_instance(ritual_type, user_id, concepts, context)
                
        return self._safe_call(_create)
    
    def advance_ritual_stage(self,
                          instance_id: str,
                          ritual_id: str,
                          interaction_data_json: Optional[str] = None) -> Dict:
        """
        Advance a ritual to the next stage based on user interaction.
        
        Args:
            instance_id: The RitualInteractionPatterns instance ID
            ritual_id: The ritual instance ID
            interaction_data_json: Optional JSON string with interaction data
            
        Returns:
            Dict with success status and updated ritual status if successful
        """
        def _advance():
            if instance_id not in self.ritual_instances:
                raise ValueError(f"Unknown RitualInteractionPatterns instance: {instance_id}")
                
            instance = self.ritual_instances[instance_id]
            
            # Parse JSON input if provided
            interaction_data = json.loads(interaction_data_json) if interaction_data_json else None
            
            return instance.advance_ritual_stage(ritual_id, interaction_data)
                
        return self._safe_call(_advance)
    
    def get_ritual_status(self, instance_id: str, ritual_id: str) -> Dict:
        """
        Get the current status of a ritual.
        
        Args:
            instance_id: The RitualInteractionPatterns instance ID
            ritual_id: The ritual instance ID
            
        Returns:
            Dict with success status and ritual status if successful
        """
        def _get_status():
            if instance_id not in self.ritual_instances:
                raise ValueError(f"Unknown RitualInteractionPatterns instance: {instance_id}")
                
            instance = self.ritual_instances[instance_id]
            return instance.get_ritual_status(ritual_id)
                
        return self._safe_call(_get_status)
    
    def get_available_rituals(self, instance_id: str) -> Dict:
        """
        Get information about available ritual templates.
        
        Args:
            instance_id: The RitualInteractionPatterns instance ID
            
        Returns:
            Dict with success status and ritual templates if successful
        """
        def _get_rituals():
            if instance_id not in self.ritual_instances:
                raise ValueError(f"Unknown RitualInteractionPatterns instance: {instance_id}")
                
            instance = self.ritual_instances[instance_id]
            return instance.get_available_rituals()
                
        return self._safe_call(_get_rituals)
    
    def get_user_ritual_history(self, instance_id: str, user_id: str) -> Dict:
        """
        Get a user's ritual history.
        
        Args:
            instance_id: The RitualInteractionPatterns instance ID
            user_id: The user identifier
            
        Returns:
            Dict with success status and ritual history if successful
        """
        def _get_history():
            if instance_id not in self.ritual_instances:
                raise ValueError(f"Unknown RitualInteractionPatterns instance: {instance_id}")
                
            instance = self.ritual_instances[instance_id]
            return instance.get_user_ritual_history(user_id)
                
        return self._safe_call(_get_history)
    
    def generate_ritual_report(self, instance_id: str, ritual_id: str) -> Dict:
        """
        Generate a comprehensive report for a ritual.
        
        Args:
            instance_id: The RitualInteractionPatterns instance ID
            ritual_id: The ritual instance ID
            
        Returns:
            Dict with success status and detailed report if successful
        """
        def _generate_report():
            if instance_id not in self.ritual_instances:
                raise ValueError(f"Unknown RitualInteractionPatterns instance: {instance_id}")
                
            instance = self.ritual_instances[instance_id]
            return instance.generate_ritual_report(ritual_id)
                
        return self._safe_call(_generate_report)
    
    #----------------------------------------
    # Temporal Flow Maps Bridge Methods
    #----------------------------------------
    
    def create_temporal_flow_maps(self, config_path: Optional[str] = None) -> Dict:
        """
        Create a new TemporalFlowMaps instance.
        
        Args:
            config_path: Optional path to a JSON configuration file
            
        Returns:
            Dict with success status and instance ID if successful
        """
        def _create():
            instance = TemporalFlowMaps(config_path)
            instance_id = f"temporal_{len(self.temporal_instances)}"
            self.temporal_instances[instance_id] = instance
            return instance_id
            
        return self._safe_call(_create)
    
    def create_symbol(self,
                    instance_id: str,
                    name: str,
                    description: str,
                    attributes_json: Optional[str] = None) -> Dict:
        """
        Create a new symbol in the mythology.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            name: Name of the symbol
            description: Description of the symbol
            attributes_json: Optional JSON string with attributes dictionary
            
        Returns:
            Dict with success status and symbol ID if successful
        """
        def _create():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            
            # Parse JSON input if provided
            attributes = json.loads(attributes_json) if attributes_json else None
            
            return instance.create_symbol(name, description, attributes)
                
        return self._safe_call(_create)
    
    def create_concept(self,
                     instance_id: str,
                     name: str,
                     description: str,
                     related_symbols_json: Optional[str] = None) -> Dict:
        """
        Create a new concept in the mythology.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            name: Name of the concept
            description: Description of the concept
            related_symbols_json: Optional JSON string array of symbol IDs
            
        Returns:
            Dict with success status and concept ID if successful
        """
        def _create():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            
            # Parse JSON input if provided
            related_symbols = json.loads(related_symbols_json) if related_symbols_json else None
            
            return instance.create_concept(name, description, related_symbols)
                
        return self._safe_call(_create)
    
    def create_narrative(self,
                       instance_id: str,
                       name: str,
                       description: str,
                       related_concepts_json: Optional[str] = None,
                       related_symbols_json: Optional[str] = None) -> Dict:
        """
        Create a new narrative thread in the mythology.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            name: Name of the narrative
            description: Description of the narrative
            related_concepts_json: Optional JSON string array of concept IDs
            related_symbols_json: Optional JSON string array of symbol IDs
            
        Returns:
            Dict with success status and narrative ID if successful
        """
        def _create():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            
            # Parse JSON input if provided
            related_concepts = json.loads(related_concepts_json) if related_concepts_json else None
            related_symbols = json.loads(related_symbols_json) if related_symbols_json else None
            
            return instance.create_narrative(name, description, related_concepts, related_symbols)
                
        return self._safe_call(_create)
    
    def add_narrative_event(self,
                          instance_id: str,
                          narrative_id: str,
                          title: str,
                          description: str,
                          involved_concepts_json: Optional[str] = None,
                          involved_symbols_json: Optional[str] = None,
                          custom_timestamp: Optional[str] = None) -> Dict:
        """
        Add an event to a narrative timeline.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            narrative_id: ID of the narrative
            title: Event title
            description: Event description
            involved_concepts_json: Optional JSON string array of concept IDs
            involved_symbols_json: Optional JSON string array of symbol IDs
            custom_timestamp: Optional ISO format datetime string
            
        Returns:
            Dict with success status and boolean result if successful
        """
        def _add_event():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            
            # Parse JSON input if provided
            involved_concepts = json.loads(involved_concepts_json) if involved_concepts_json else None
            involved_symbols = json.loads(involved_symbols_json) if involved_symbols_json else None
            
            return instance.add_narrative_event(
                narrative_id,
                title,
                description,
                involved_concepts,
                involved_symbols,
                custom_timestamp
            )
                
        return self._safe_call(_add_event)
    
    def generate_temporal_map(self,
                           instance_id: str,
                           entity_ids_json: str,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None,
                           time_scale: Optional[str] = None,
                           include_related: bool = True,
                           max_related_depth: int = 1) -> Dict:
        """
        Generate a temporal map visualization data for the specified entities.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            entity_ids_json: JSON string array of entity IDs to include
            start_time: Optional ISO timestamp for start time
            end_time: Optional ISO timestamp for end time
            time_scale: Time scale for the visualization
            include_related: Whether to include related entities
            max_related_depth: Maximum depth for related entities
            
        Returns:
            Dict with success status and visualization data if successful
        """
        def _generate():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            
            # Parse JSON input
            entity_ids = json.loads(entity_ids_json)
            
            return instance.generate_temporal_map(
                entity_ids,
                start_time,
                end_time,
                time_scale,
                include_related,
                max_related_depth
            )
                
        return self._safe_call(_generate)
    
    def generate_concept_evolution_timeline(self, instance_id: str, concept_id: str) -> Dict:
        """
        Generate a specialized timeline showing the evolution of a concept.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            concept_id: ID of the concept
            
        Returns:
            Dict with success status and timeline data if successful
        """
        def _generate():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            return instance.generate_concept_evolution_timeline(concept_id)
                
        return self._safe_call(_generate)
    
    def generate_symbol_usage_timeline(self, instance_id: str, symbol_id: str) -> Dict:
        """
        Generate a specialized timeline showing how a symbol has been used over time.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            symbol_id: ID of the symbol
            
        Returns:
            Dict with success status and timeline data if successful
        """
        def _generate():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            return instance.generate_symbol_usage_timeline(symbol_id)
                
        return self._safe_call(_generate)
    
    def generate_narrative_flow_map(self, instance_id: str, narrative_id: str) -> Dict:
        """
        Generate a specialized map showing the flow of a narrative over time.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            narrative_id: ID of the narrative
            
        Returns:
            Dict with success status and flow map data if successful
        """
        def _generate():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            return instance.generate_narrative_flow_map(narrative_id)
                
        return self._safe_call(_generate)
    
    def export_temporal_data(self, instance_id: str, format_type: str = "json") -> Dict:
        """
        Export all data for backup or analysis.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            format_type: Export format (currently only 'json' supported)
            
        Returns:
            Dict with success status and export data if successful
        """
        def _export():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            return instance.export_data(format_type)
                
        return self._safe_call(_export)
    
    #----------------------------------------
    # Extended Utility Methods
    #----------------------------------------
    
    def bulk_import_symbols(self, instance_id: str, symbols_json: str) -> Dict:
        """
        Import multiple symbols at once.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            symbols_json: JSON string array of symbol definitions
            
        Returns:
            Dict with success status and array of imported symbol IDs
        """
        def _import():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            symbols = json.loads(symbols_json)
            
            result_ids = []
            for symbol in symbols:
                symbol_id = instance.create_symbol(
                    symbol["name"],
                    symbol["description"],
                    symbol.get("attributes")
                )
                result_ids.append(symbol_id)
                
            return result_ids
                
        return self._safe_call(_import)
    
    def get_entity_details(self, instance_id: str, entity_id: str, entity_type: str = None) -> Dict:
        """
        Get detailed information about an entity (symbol, concept, or narrative).
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            entity_id: ID of the entity
            entity_type: Optional type hint ('symbol', 'concept', or 'narrative')
            
        Returns:
            Dict with success status and entity details if successful
        """
        def _get_details():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            return instance.get_entity_details(entity_id, entity_type)
                
        return self._safe_call(_get_details)
    
    def search_entities(self, 
                      instance_id: str, 
                      query: str, 
                      entity_types: Optional[str] = None,
                      max_results: int = 10) -> Dict:
        """
        Search for entities matching a query string.
        
        Args:
            instance_id: The TemporalFlowMaps instance ID
            query: Search query string
            entity_types: Optional JSON array of entity types to include
            max_results: Maximum number of results to return
            
        Returns:
            Dict with success status and search results if successful
        """
        def _search():
            if instance_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {instance_id}")
                
            instance = self.temporal_instances[instance_id]
            
            # Parse entity types if provided
            types = json.loads(entity_types) if entity_types else None
            
            return instance.search_entities(query, types, max_results)
                
        return self._safe_call(_search)
    
    def merge_circadian_with_ritual(self,
                                  circadian_id: str,
                                  ritual_id: str,
                                  ritual_instance_id: str) -> Dict:
        """
        Integrate circadian narrative tones with ritual interaction patterns.
        
        Args:
            circadian_id: The CircadianNarrativeCycles instance ID
            ritual_id: The RitualInteractionPatterns instance ID
            ritual_instance_id: ID of a specific ritual instance
            
        Returns:
            Dict with success status and enhanced ritual guidance if successful
        """
        def _merge():
            if circadian_id not in self.circadian_instances:
                raise ValueError(f"Unknown CircadianNarrativeCycles instance: {circadian_id}")
                
            if ritual_id not in self.ritual_instances:
                raise ValueError(f"Unknown RitualInteractionPatterns instance: {ritual_id}")
                
            circadian_instance = self.circadian_instances[circadian_id]
            ritual_instance = self.ritual_instances[ritual_id]
            
            # Get current tone
            tone_data = circadian_instance.get_current_narrative_tone()
            
            # Get ritual status
            ritual_status = ritual_instance.get_ritual_status(ritual_instance_id)
            
            # Adapt ritual guidance based on tone
            current_stage = ritual_status["current_stage"]
            stage_guidance = ritual_status["stages"][current_stage]["guidance"]
            
            # Transform the guidance using circadian tone
            enhanced_guidance = circadian_instance.transform_narrative(
                stage_guidance, 
                None,  # Use current time
                0.7    # Moderate transformation strength
            )
            
            return {
                "original_guidance": stage_guidance,
                "enhanced_guidance": enhanced_guidance,
                "current_tone": tone_data["tone"],
                "ritual_stage": current_stage
            }
                
        return self._safe_call(_merge)
    
    def create_temporal_ritual_record(self,
                                   temporal_id: str,
                                   ritual_id: str,
                                   ritual_instance_id: str) -> Dict:
        """
        Record a completed ritual as a narrative event in temporal flow maps.
        
        Args:
            temporal_id: The TemporalFlowMaps instance ID
            ritual_id: The RitualInteractionPatterns instance ID
            ritual_instance_id: ID of a specific ritual instance
            
        Returns:
            Dict with success status and created narrative event ID if successful
        """
        def _create_record():
            if temporal_id not in self.temporal_instances:
                raise ValueError(f"Unknown TemporalFlowMaps instance: {temporal_id}")
                
            if ritual_id not in self.ritual_instances:
                raise ValueError(f"Unknown RitualInteractionPatterns instance: {ritual_id}")
                
            temporal_instance = self.temporal_instances[temporal_id]
            ritual_instance = self.ritual_instances[ritual_id]
            
            # Get ritual details
            ritual_report = ritual_instance.generate_ritual_report(ritual_instance_id)
            ritual_status = ritual_instance.get_ritual_status(ritual_instance_id)
            
            # Create a narrative for this ritual type if it doesn't exist
            ritual_type = ritual_status["ritual_type"]
            ritual_narratives = temporal_instance.search_entities(
                f"ritual:{ritual_type}", 
                ["narrative"],
                1
            )
            
            if not ritual_narratives:
                # Create a new narrative for this ritual type
                narrative_id = temporal_instance.create_narrative(
                    f"Ritual: {ritual_type}",
                    f"Narrative thread for {ritual_type} rituals",
                    [],  # No related concepts initially
                    []   # No related symbols initially
                )
            else:
                narrative_id = ritual_narratives[0]["id"]
            
            # Add the ritual completion as a narrative event
            event_id = temporal_instance.add_narrative_event(
                narrative_id,
                f"{ritual_type} Ritual Completed",
                ritual_report["summary"],
                ritual_status["concepts"],  # Use concepts from ritual
                [],  # No symbols initially
                ritual_status["completion_time"] if "completion_time" in ritual_status else None
            )
            
            return {
                "narrative_id": narrative_id,
                "event_id": event_id,
                "ritual_type": ritual_type
            }
                
        return self._safe_call(_create_record)


# Create a global instance of the bridge for easier access
bridge = Bridge()

# Expose a simple JSON interface for easier integration
def call_method(method_name: str, params_json: str) -> str:
    """
    Call a bridge method using JSON for parameters and return value.
    
    Args:
        method_name: Name of the bridge method to call
        params_json: JSON string of parameters
        
    Returns:
        JSON string with result
    """
    try:
        # Parse parameters
        params = json.loads(params_json)
        
        # Get the method
        if not hasattr(bridge, method_name):
            return json.dumps({
                "success": False,
                "error": {
                    "message": f"Unknown method: {method_name}",
                    "type": "AttributeError"
                }
            })
        
        method = getattr(bridge, method_name)
        
        # Call the method
        result = method(**params)
        
        # Return JSON result
        return json.dumps(result)
    except Exception as e:
        error_info = {
            "message": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        return json.dumps({
            "success": False,
            "error": error_info
        })


# Test if run directly
if __name__ == "__main__":
    # Create instances for testing
    circadian_result = bridge.create_circadian_instance()
    if circadian_result["success"]:
        circadian_id = circadian_result["result"]
        print(f"Created CircadianNarrativeCycles instance: {circadian_id}")
        
        # Test getting current tone
        tone_result = bridge.get_current_narrative_tone(circadian_id)
        if tone_result["success"]:
            print(f"Current tone: {tone_result['result']['tone']}")
        else:
            print(f"Error getting tone: {tone_result['error']['message']}")
    
    ritual_result = bridge.create_ritual_instance_manager()
    if ritual_result["success"]:
        ritual_id = ritual_result["result"]
        print(f"Created RitualInteractionPatterns instance: {ritual_id}")
        
        # Test getting available rituals
        rituals_result = bridge.get_available_rituals(ritual_id)
        if rituals_result["success"]:
            print(f"Available rituals: {list(rituals_result['result'].keys())}")
        else:
            print(f"Error getting rituals: {rituals_result['error']['message']}")
    
    temporal_result = bridge.create_temporal_flow_maps()
    if temporal_result["success"]:
        temporal_id = temporal_result["result"]
        print(f"Created TemporalFlowMaps instance: {temporal_id}")
