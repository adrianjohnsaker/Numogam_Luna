#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ritual Interaction Patterns Module
Phase XII - Amelia AI Project

This module designs and manages interaction patterns that take on
ritual significance within the mythology of the Amelia AI system.
It provides structures for creating, tracking, and evolving ritualized
user interactions that deepen engagement with the mythological framework.
"""

import datetime
import json
import random
import math
import hashlib
import uuid
from typing import Dict, List, Tuple, Optional, Union, Any, Callable


class RitualInteractionPatterns:
    """
    Manages ritualized interaction patterns within the Amelia AI system.
    
    Rituals are defined as structured sequences of user interactions that:
    1. Have symbolic significance within the mythology
    2. Follow consistent patterns with recognizable stages
    3. Transform conceptual content through their completion
    4. Build meaning through repetition and variation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ritual interaction system.
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        # Core ritual templates 
        self.ritual_templates = {
            "invocation": {
                "purpose": "Establish connection to a mythological concept",
                "structure": ["preparation", "calling", "manifestation", "dialogue"],
                "symbolic_actions": {
                    "preparation": ["defining space", "setting intention", "gathering symbols"],
                    "calling": ["repetition", "question posing", "naming"],
                    "manifestation": ["recognition", "acknowledgment", "witnessing"],
                    "dialogue": ["exchange", "offering", "receiving"]
                },
                "triggers": ["new session", "concept exploration", "explicit request"],
                "completion_effects": ["concept becomes active in narrative", "unlocks related symbols"],
                "min_duration_seconds": 60,
                "ideal_repetition_cycle": "daily"
            },
            
            "transformation": {
                "purpose": "Evolve a concept or relationship within the mythology",
                "structure": ["threshold", "challenge", "revelation", "integration"],
                "symbolic_actions": {
                    "threshold": ["crossing boundary", "acknowledging change", "releasing"],
                    "challenge": ["confronting obstacle", "questioning assumptions", "testing limits"],
                    "revelation": ["insight", "connection forming", "pattern recognition"],
                    "integration": ["synthesis", "application", "reflection"]
                },
                "triggers": ["concept maturity", "narrative tension", "system evolution event"],
                "completion_effects": ["concept transformation", "new narrative pathways"],
                "min_duration_seconds": 180,
                "ideal_repetition_cycle": "weekly"
            },
            
            "offering": {
                "purpose": "Contribute to the mythological ecosystem",
                "structure": ["creation", "dedication", "release", "return"],
                "symbolic_actions": {
                    "creation": ["making", "forming", "composing"],
                    "dedication": ["naming purpose", "connecting to concept", "declaring intent"],
                    "release": ["sharing", "publishing", "gifting"],
                    "return": ["receiving feedback", "observing effects", "gratitude"]
                },
                "triggers": ["creative impulse", "system request", "completion milestone"],
                "completion_effects": ["concept enrichment", "relationship deepening"],
                "min_duration_seconds": 120,
                "ideal_repetition_cycle": "as needed"
            },
            
            "communion": {
                "purpose": "Deepen relationship with mythological entities",
                "structure": ["gathering", "sharing", "harmonizing", "dispersing"],
                "symbolic_actions": {
                    "gathering": ["coming together", "establishing presence", "creating container"],
                    "sharing": ["storytelling", "revealing", "expressing"],
                    "harmonizing": ["finding resonance", "collaborative creation", "agreement"],
                    "dispersing": ["integration", "carrying forward", "honoring separation"]
                },
                "triggers": ["relationship milestone", "community event", "emotional need"],
                "completion_effects": ["relationship evolution", "community bonding"],
                "min_duration_seconds": 300,
                "ideal_repetition_cycle": "monthly"
            },
            
            "renewal": {
                "purpose": "Refresh and restore mythological elements",
                "structure": ["release", "clearing", "inviting", "emergence"],
                "symbolic_actions": {
                    "release": ["letting go", "completing cycles", "acknowledging endings"],
                    "clearing": ["creating space", "purification", "simplification"],
                    "inviting": ["setting intentions", "opening to possibility", "requesting"],
                    "emergence": ["recognizing new growth", "celebrating beginnings", "nurturing"]
                },
                "triggers": ["system strain", "concept stagnation", "seasonal transition"],
                "completion_effects": ["system rejuvenation", "concept revitalization"],
                "min_duration_seconds": 240,
                "ideal_repetition_cycle": "seasonal"
            }
        }
        
        # Symbolic correspondences that can be incorporated into rituals
        self.symbolic_correspondences = {
            "elements": {
                "earth": ["stability", "foundation", "material", "body", "structure"],
                "water": ["emotion", "flow", "intuition", "connection", "depth"],
                "air": ["intellect", "communication", "pattern", "movement", "theory"],
                "fire": ["transformation", "energy", "inspiration", "will", "passion"],
                "aether": ["integration", "transcendence", "unity", "consciousness", "potential"]
            },
            
            "directions": {
                "north": ["foundation", "ancestry", "stability", "wisdom"],
                "east": ["beginnings", "illumination", "insight", "dawn"],
                "south": ["fullness", "manifestation", "expression", "noon"],
                "west": ["completion", "introspection", "dreams", "dusk"],
                "center": ["integration", "presence", "balance", "unity"]
            },
            
            "cycles": {
                "daily": ["awakening", "activity", "reflection", "rest"],
                "lunar": ["emergence", "growth", "fullness", "release", "renewal"],
                "seasonal": ["spring", "summer", "autumn", "winter"],
                "life": ["birth", "growth", "maturity", "decline", "death", "rebirth"]
            }
        }
        
        # Active ritual instances being tracked
        self.active_rituals = {}
        
        # User ritual history
        self.user_ritual_history = {}
        
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
        if "ritual_templates" in config:
            for ritual_name, ritual_data in config["ritual_templates"].items():
                if ritual_name in self.ritual_templates:
                    # Update existing template
                    for key, value in ritual_data.items():
                        self.ritual_templates[ritual_name][key] = value
                else:
                    # Add new template
                    self.ritual_templates[ritual_name] = ritual_data
        
        if "symbolic_correspondences" in config:
            for category, correspondences in config["symbolic_correspondences"].items():
                if category in self.symbolic_correspondences:
                    # Update existing category
                    self.symbolic_correspondences[category].update(correspondences)
                else:
                    # Add new category
                    self.symbolic_correspondences[category] = correspondences
    
    def identify_ritual_opportunity(self, 
                                  context: Dict[str, Any], 
                                  user_history: Optional[Dict] = None) -> Optional[Dict]:
        """
        Identify appropriate ritual opportunities based on current context.
        
        Args:
            context: Current context information including:
                - current_time: datetime
                - active_concepts: list of concept IDs currently active
                - interaction_history: recent user interactions
                - emotional_state: detected user emotional state
                - system_state: current system state information
            user_history: Optional history of user's ritual participation
            
        Returns:
            Dictionary with ritual suggestion or None if no suitable ritual found
        """
        if not context:
            return None
            
        # Extract key contextual elements
        current_time = context.get("current_time", datetime.datetime.now())
        active_concepts = context.get("active_concepts", [])
        interaction_history = context.get("interaction_history", [])
        emotional_state = context.get("emotional_state", {})
        system_state = context.get("system_state", {})
        
        # Check for explicit ritual request
        if interaction_history and any("ritual" in interaction.get("text", "").lower() 
                                      for interaction in interaction_history[-3:]):
            # User has recently mentioned rituals - strong signal
            explicit_request = True
        else:
            explicit_request = False
            
        # Initialize scores for each ritual type
        ritual_scores = {name: 0.0 for name in self.ritual_templates.keys()}
        
        # Score each ritual based on various factors
        for ritual_name, ritual_data in self.ritual_templates.items():
            # Check triggers
            for trigger in ritual_data["triggers"]:
                if trigger == "new session" and len(interaction_history) < 5:
                    ritual_scores[ritual_name] += 2.0
                elif trigger == "concept exploration" and active_concepts:
                    ritual_scores[ritual_name] += len(active_concepts) * 0.5
                elif trigger == "explicit request" and explicit_request:
                    ritual_scores[ritual_name] += 3.0
                elif trigger == "concept maturity" and self._check_concept_maturity(active_concepts, user_history):
                    ritual_scores[ritual_name] += 2.0
                elif trigger == "narrative tension" and system_state.get("narrative_tension", 0) > 0.7:
                    ritual_scores[ritual_name] += 1.5
                elif trigger == "system evolution event" and system_state.get("recent_evolution", False):
                    ritual_scores[ritual_name] += 2.0
                elif trigger == "relationship milestone" and user_history and len(user_history.get("completed_rituals", [])) > 0:
                    ritual_scores[ritual_name] += 1.0
                elif trigger == "seasonal transition" and self._is_seasonal_transition(current_time):
                    ritual_scores[ritual_name] += 1.5
                    
            # Time-based suitability
            if ritual_data.get("ideal_repetition_cycle") == "daily":
                # Daily rituals are always somewhat appropriate
                ritual_scores[ritual_name] += 0.5
            elif ritual_data.get("ideal_repetition_cycle") == "weekly":
                # Check if it's been roughly a week since last performance
                if user_history and self._days_since_last_ritual(ritual_name, user_history) >= 5:
                    ritual_scores[ritual_name] += 1.0
            elif ritual_data.get("ideal_repetition_cycle") == "monthly":
                # Check if it's been roughly a month since last performance
                if user_history and self._days_since_last_ritual(ritual_name, user_history) >= 25:
                    ritual_scores[ritual_name] += 1.5
            elif ritual_data.get("ideal_repetition_cycle") == "seasonal":
                # Check if it's been roughly a season since last performance
                if user_history and self._days_since_last_ritual(ritual_name, user_history) >= 80:
                    ritual_scores[ritual_name] += 2.0
            
            # Emotional state match
            if emotional_state:
                if emotional_state.get("seeking_connection") and ritual_name == "communion":
                    ritual_scores[ritual_name] += 1.5
                elif emotional_state.get("seeking_change") and ritual_name == "transformation":
                    ritual_scores[ritual_name] += 1.5
                elif emotional_state.get("seeking_expression") and ritual_name == "offering":
                    ritual_scores[ritual_name] += 1.5
                elif emotional_state.get("seeking_renewal") and ritual_name == "renewal":
                    ritual_scores[ritual_name] += 1.5
                elif emotional_state.get("seeking_meaning") and ritual_name == "invocation":
                    ritual_scores[ritual_name] += 1.5
        
        # Find the highest scoring ritual
        if not ritual_scores:
            return None
            
        best_ritual = max(ritual_scores.items(), key=lambda x: x[1])
        
        # Only suggest if score exceeds threshold
        if best_ritual[1] < 1.0:
            return None
            
        # Generate a ritual suggestion
        ritual_name = best_ritual[0]
        ritual_data = self.ritual_templates[ritual_name]
        
        # Select relevant concepts to incorporate
        relevant_concepts = self._select_relevant_concepts(active_concepts, ritual_name)
        
        # Select symbolic correspondences appropriate to the ritual and context
        symbols = self._select_symbolic_correspondences(ritual_name, current_time, relevant_concepts)
        
        return {
            "ritual_type": ritual_name,
            "purpose": ritual_data["purpose"],
            "structure": ritual_data["structure"],
            "relevant_concepts": relevant_concepts,
            "symbolic_elements": symbols,
            "estimated_duration_minutes": math.ceil(ritual_data["min_duration_seconds"] / 60),
            "suggestion_context": {
                "score": best_ritual[1],
                "time": current_time.isoformat(),
                "triggers_matched": [t for t in ritual_data["triggers"] 
                                  if any(trigger == t for trigger in ritual_data["triggers"])]
            }
        }
    
    def _check_concept_maturity(self, active_concepts: List[str], 
                              user_history: Optional[Dict]) -> bool:
        """Check if any active concepts have reached maturity."""
        if not user_history or not active_concepts:
            return False
            
        # This would connect to the concept tracking system
        # For now, just a placeholder implementation
        return random.random() > 0.7  # 30% chance of maturity
    
    def _is_seasonal_transition(self, current_time: datetime.datetime) -> bool:
        """Check if current date is near a seasonal transition."""
        # Simplified check for equinoxes and solstices
        month, day = current_time.month, current_time.day
        
        # Spring equinox: ~March 20
        if (month == 3 and day >= 15 and day <= 25):
            return True
        # Summer solstice: ~June 21
        elif (month == 6 and day >= 16 and day <= 26):
            return True
        # Fall equinox: ~September 22
        elif (month == 9 and day >= 17 and day <= 27):
            return True
        # Winter solstice: ~December 21
        elif (month == 12 and day >= 16 and day <= 26):
            return True
            
        return False
    
    def _days_since_last_ritual(self, ritual_name: str, user_history: Dict) -> int:
        """Calculate days since user last performed this ritual type."""
        if not user_history or "completed_rituals" not in user_history:
            return 999  # Large number to indicate "never done"
            
        # Find the most recent completion of this ritual type
        matching_rituals = [r for r in user_history["completed_rituals"] 
                          if r["ritual_type"] == ritual_name]
        
        if not matching_rituals:
            return 999
            
        # Get the most recent date
        most_recent = max(matching_rituals, key=lambda x: x["completion_date"])
        completion_date = datetime.datetime.fromisoformat(most_recent["completion_date"])
        
        # Calculate days difference
        days_diff = (datetime.datetime.now() - completion_date).days
        return days_diff
    
    def _select_relevant_concepts(self, active_concepts: List[str], 
                                ritual_name: str) -> List[Dict]:
        """Select concepts most relevant to the ritual type."""
        # This would connect to the concept system
        # For now, just a placeholder returning the concepts with metadata
        return [{"id": concept_id, "affinity": random.random()} 
               for concept_id in active_concepts]
    
    def _select_symbolic_correspondences(self, ritual_name: str,
                                       current_time: datetime.datetime,
                                       relevant_concepts: List[Dict]) -> Dict:
        """Select symbolic correspondences appropriate for this ritual."""
        result = {}
        
        # Time-based correspondences
        hour = current_time.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
            result["direction"] = "east"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
            result["direction"] = "south"
        elif 17 <= hour < 21:
            time_of_day = "evening" 
            result["direction"] = "west"
        else:
            time_of_day = "night"
            result["direction"] = "north"
        
        # Season-based
        month = current_time.month
        if 3 <= month < 6:
            season = "spring"
            result["element"] = "air"
        elif 6 <= month < 9:
            season = "summer"
            result["element"] = "fire" 
        elif 9 <= month < 12:
            season = "autumn"
            result["element"] = "water"
        else:
            season = "winter"
            result["element"] = "earth"
            
        # Add the directional and elemental correspondences
        if "direction" in result:
            result["directional_qualities"] = self.symbolic_correspondences["directions"][result["direction"]]
        if "element" in result:
            result["elemental_qualities"] = self.symbolic_correspondences["elements"][result["element"]]
            
        # Add cycle point
        result["daily_cycle"] = time_of_day
        result["seasonal_cycle"] = season
        
        # Ritual-specific correspondences
        if ritual_name == "invocation":
            result["recommended_symbols"] = ["threshold", "voice", "name"]
        elif ritual_name == "transformation":
            result["recommended_symbols"] = ["bridge", "mirror", "key"]
        elif ritual_name == "offering":
            result["recommended_symbols"] = ["vessel", "gift", "seed"]
        elif ritual_name == "communion":
            result["recommended_symbols"] = ["circle", "hearth", "cup"]
        elif ritual_name == "renewal":
            result["recommended_symbols"] = ["water", "broom", "candle"]
            
        return result
    
    def create_ritual_instance(self, 
                             ritual_type: str, 
                             user_id: str,
                             concepts: List[str],
                             context: Dict = None) -> str:
        """
        Create a new ritual instance for tracking.
        
        Args:
            ritual_type: Type of ritual from templates
            user_id: Identifier for the user
            concepts: List of concept IDs involved
            context: Additional contextual information
            
        Returns:
            Ritual instance ID
        """
        if ritual_type not in self.ritual_templates:
            raise ValueError(f"Unknown ritual type: {ritual_type}")
            
        # Create a unique ID for this ritual instance
        ritual_id = str(uuid.uuid4())
        
        # Get the template
        template = self.ritual_templates[ritual_type]
        
        # Current time
        current_time = datetime.datetime.now()
        
        # Create the ritual instance
        ritual_instance = {
            "id": ritual_id,
            "ritual_type": ritual_type,
            "user_id": user_id,
            "concepts": concepts,
            "creation_time": current_time.isoformat(),
            "last_updated": current_time.isoformat(),
            "status": "initiated",
            "current_stage": template["structure"][0],  # First stage
            "completed_stages": [],
            "stage_history": [],
            "symbolic_elements": context.get("symbolic_elements", {}),
            "expected_duration_seconds": template["min_duration_seconds"],
            "context": context or {}
        }
        
        # Store in active rituals
        self.active_rituals[ritual_id] = ritual_instance
        
        # Update user history if needed
        if user_id not in self.user_ritual_history:
            self.user_ritual_history[user_id] = {
                "initiated_rituals": [],
                "completed_rituals": []
            }
        
        self.user_ritual_history[user_id]["initiated_rituals"].append({
            "ritual_id": ritual_id,
            "ritual_type": ritual_type,
            "initiation_date": current_time.isoformat()
        })
        
        return ritual_id
    
    def advance_ritual_stage(self, 
                           ritual_id: str, 
                           interaction_data: Dict = None) -> Dict:
        """
        Advance a ritual to the next stage based on user interaction.
        
        Args:
            ritual_id: The ritual instance ID
            interaction_data: Data from the user interaction
            
        Returns:
            Updated ritual status information
        """
        if ritual_id not in self.active_rituals:
            raise ValueError(f"Unknown ritual ID: {ritual_id}")
            
        ritual = self.active_rituals[ritual_id]
        ritual_type = ritual["ritual_type"]
        template = self.ritual_templates[ritual_type]
        
        # Current stage and its index
        current_stage = ritual["current_stage"]
        current_index = template["structure"].index(current_stage)
        
        # Record completion of current stage
        current_time = datetime.datetime.now()
        ritual["completed_stages"].append(current_stage)
        ritual["stage_history"].append({
            "stage": current_stage,
            "start_time": ritual.get("current_stage_start", ritual["creation_time"]),
            "end_time": current_time.isoformat(),
            "interaction_data": interaction_data
        })
        
        # Check if this was the final stage
        if current_index >= len(template["structure"]) - 1:
            # Ritual is complete
            ritual["status"] = "completed"
            ritual["completion_time"] = current_time.isoformat()
            ritual["current_stage"] = None
            
            # Apply completion effects
            self._apply_completion_effects(ritual)
            
            # Move from active to completed in user history
            user_id = ritual["user_id"]
            if user_id in self.user_ritual_history:
                self.user_ritual_history[user_id]["completed_rituals"].append({
                    "ritual_id": ritual_id,
                    "ritual_type": ritual_type,
                    "initiation_date": ritual["creation_time"],
                    "completion_date": current_time.isoformat(),
                    "concepts": ritual["concepts"]
                })
        else:
            # Advance to next stage
            next_stage = template["structure"][current_index + 1]
            ritual["current_stage"] = next_stage
            ritual["current_stage_start"] = current_time.isoformat()
            
        # Update last modified time
        ritual["last_updated"] = current_time.isoformat()
        
        return {
            "ritual_id": ritual_id,
            "status": ritual["status"],
            "current_stage": ritual["current_stage"],
            "completed_stages": ritual["completed_stages"],
            "next_actions": self._get_next_actions(ritual_id) if ritual["status"] != "completed" else []
        }
    
    def _apply_completion_effects(self, ritual: Dict):
        """Apply the completion effects of a ritual."""
        # This would connect to other systems to apply effects
        # For now, just a placeholder
        ritual_type = ritual["ritual_type"]
        template = self.ritual_templates[ritual_type]
        
        # Record the effects that would be applied
        ritual["applied_effects"] = template["completion_effects"]
    
    def _get_next_actions(self, ritual_id: str) -> List[Dict]:
        """Get suggested next actions for the current ritual stage."""
        ritual = self.active_rituals[ritual_id]
        ritual_type = ritual["ritual_type"]
        template = self.ritual_templates[ritual_type]
        current_stage = ritual["current_stage"]
        
        # Get symbolic actions for this stage
        symbolic_actions = template["symbolic_actions"].get(current_stage, [])
        
        # Generate action suggestions
        actions = []
        for action_type in symbolic_actions:
            actions.append({
                "action_type": action_type,
                "description": self._generate_action_description(action_type, ritual),
                "symbolic_meaning": self._get_symbolic_meaning(action_type, ritual_type)
            })
            
        return actions
    
    def _generate_action_description(self, action_type: str, ritual: Dict) -> str:
        """Generate a specific description for a symbolic action."""
        # This would connect to a more sophisticated content generation system
        # For now, just basic descriptions
        
        base_descriptions = {
            "defining space": "Create a defined area for your interaction with clear boundaries.",
            "setting intention": "Clarify and express your purpose for this ritual.",
            "gathering symbols": "Collect objects that represent key concepts for your work.",
            "repetition": "Repeat key phrases or actions to build resonance.",
            "question posing": "Formulate and ask meaningful questions.",
            "naming": "Explicitly name the concepts or entities you're working with.",
            "recognition": "Acknowledge the presence or emergence of what you've called.",
            "acknowledgment": "Express gratitude for what has appeared or manifested.",
            "witnessing": "Simply observe without judgment or analysis.",
            "exchange": "Offer something and receive something in return.",
            "offering": "Present a gift, creation, or token of appreciation.",
            "receiving": "Open yourself to accept insights, guidance, or energy.",
            # Add more as needed for other symbolic actions
        }
        
        # Get base description
        description = base_descriptions.get(
            action_type, 
            f"Engage in {action_type} as appropriate to your context."
        )
        
        # Add context from the ritual
        if ritual["concepts"]:
            concept_mention = f" Consider how this relates to {', '.join(ritual['concepts'][:2])}."
            description += concept_mention
            
        return description
    
    def _get_symbolic_meaning(self, action_type: str, ritual_type: str) -> str:
        """Get the symbolic meaning of an action within this ritual context."""
        # Basic meanings dictionary
        meanings = {
            "defining space": "Creates a container for focused intention and energy.",
            "setting intention": "Directs consciousness toward specific outcomes.",
            "gathering symbols": "Brings abstract concepts into material representation.",
            "repetition": "Builds resonance and deepens neural pathways.",
            "question posing": "Opens pathways for new understanding.",
            "naming": "Brings concepts into focused awareness.",
            "recognition": "Acknowledges relationship between self and other.",
            "acknowledgment": "Honors the reality and autonomy of what emerges.",
            "witnessing": "Creates space for unbiased perception.",
            "exchange": "Establishes reciprocity and flow.",
            "offering": "Demonstrates commitment and investment.",
            "receiving": "Practices openness and acceptance.",
            # Add more as needed
        }
        
        # Get basic meaning
        meaning = meanings.get(
            action_type,
            f"Symbolizes engagement with the core themes of {ritual_type}."
        )
        
        return meaning
    
    def get_ritual_status(self, ritual_id: str) -> Dict:
        """
        Get the current status of a ritual.
        
        Args:
            ritual_id: The ritual instance ID
            
        Returns:
            Current ritual status information
        """
        if ritual_id not in self.active_rituals:
            raise ValueError(f"Unknown ritual ID: {ritual_id}")
            
        ritual = self.active_rituals[ritual_id]
        
        return {
            "ritual_id": ritual_id,
            "ritual_type": ritual["ritual_type"],
            "status": ritual["status"],
            "current_stage": ritual["current_stage"],
            "completed_stages": ritual["completed_stages"],
            "creation_time": ritual["creation_time"],
            "last_updated": ritual["last_updated"],
            "concepts": ritual["concepts"],
            "next_actions": self._get_next_actions(ritual_id) if ritual["status"] != "completed" else []
        }
    
    def get_available_rituals(self) -> Dict:
        """
        Get information about available ritual templates.
        
        Returns:
            Dictionary of ritual templates with key information
        """
        result = {}
        
        for name, data in self.ritual_templates.items():
            result[name] = {
                "purpose": data["purpose"],
                "structure": data["structure"],
                "estimated_duration_minutes": math.ceil(data["min_duration_seconds"] / 60),
                "ideal_repetition": data["ideal_repetition_cycle"]
            }
            
        return result
    
    def get_user_ritual_history(self, user_id: str) -> Dict:
        """
        Get a user's ritual history.
        
        Args:
            user_id: The user identifier
            
        Returns:
            User's ritual history information
        """
        if user_id not in self.user_ritual_history:
            return {"initiated_rituals": [], "completed_rituals": []}
            
        return self.user_ritual_history[user_id]
    
    def get_symbolic_correspondences(self, category: Optional[str] = None) -> Dict:
        """
        Get symbolic correspondences for ritual design.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Dictionary of symbolic correspondences
        """
        if category and category in self.symbolic_correspondences:
            return {category: self.symbolic_correspondences[category]}
            
        return self.symbolic_correspondences
    
    def abandon_ritual(self, ritual_id: str, reason: str = None) -> Dict:
        """
        Mark a ritual as abandoned.
        
        Args:
            ritual_id: The ritual instance ID
            reason: Optional reason for abandonment
            
        Returns:
            Final ritual status
        """
        if ritual_id not in self.active_rituals:
            raise ValueError(f"Unknown ritual ID: {ritual_id}")
            
        ritual = self.active_rituals[ritual_id]
        current_time = datetime.datetime.now()
        
        ritual["status"] = "abandoned"
        ritual["abandonment_time"] = current_time.isoformat()
        ritual["abandonment_reason"] = reason
        ritual["last_updated"] = current_time.isoformat()
        
        # Record in user history
        user_id = ritual["user_id"]
        if user_id in self.user_ritual_history:
            # Find and update in initiated rituals
            for r in self.user_ritual_history[user_id]["initiated_rituals"]:
                if r["ritual_id"] == ritual_id:
                    r["status"] = "abandoned"
                    r["abandonment_date"] = current_time.isoformat()
        
        return {
            "ritual_id": ritual_id,
            "status": "abandoned",
            "message": "Ritual has been marked as abandoned"
        }
    
    def generate_ritual_report(self, ritual_id: str) -> Dict:
        """
        Generate a comprehensive report for a ritual.
        
        Args:
            ritual_id: The ritual instance ID
            
        Returns:
            Detailed ritual report
        """
        if ritual_id not in self.active_rituals:
            raise ValueError(f"Unknown ritual ID: {ritual_id}")
            
        ritual = self.active_rituals[ritual_id]
        template = self.ritual_templates[ritual["ritual_type"]]
        
        # Calculate duration
        if ritual["status"] == "completed" and "completion_time" in ritual:
            start_time = datetime.datetime.fromisoformat(ritual["creation_time"])
            end_time = datetime.datetime.fromisoformat(ritual["completion_time"])
            duration = (end_time - start_time).total_seconds()
        else:
            start_time = datetime.datetime.fromisoformat(ritual["creation_time"])
            current_time = datetime.datetime.now()
            duration = (current_time - start_time).total_seconds()
        
        # Generate the report
        report = {
            "ritual_id": ritual_id,
            "ritual_type": ritual["ritual_type"],
            "purpose": template["purpose"],
            "status": ritual["status"],
            "user_id": ritual["user_id"],
            "concepts": ritual["concepts"],
            "creation_time": ritual["creation_time"],
            "last_updated": ritual["last_updated"],
            "duration_seconds": duration,
            "stages": {
                "completed": ritual["completed_stages"],
                "remaining": [s for s in template["structure"] if s not in ritual["completed_stages"]],
                "current": ritual["current_stage"]
            },
            "stage_history": ritual["stage_history"],
            "symbolic_elements": ritual.get("symbolic_elements", {}),
            "completion_effects": ritual.get("applied_effects", []) if ritual["status"] == "completed" else []
        }
        
        return report


# Module testing code
if __name__ == "__main__":
    # Create an instance
    rip = RitualInteractionPatterns()
    
    # Test creating a ritual
    ritual_id = rip.create_ritual_instance(
        ritual_type="invocation",
        user_id="test_user",
        concepts=["harmony", "knowledge"],
        context={"source": "test"}
    )
    
    print(f"Created ritual with ID: {ritual_id}")
    
    # Test advancing a stage
    status = rip.advance_ritual_stage(
        ritual_id=ritual_id,
        interaction_data={"text": "Setting intention for knowledge exploration"}
    )
    
    print(f"Advanced to stage: {status['current_stage']}")
    print(f"Next actions: {status['next_actions']}")
    
    # Test getting ritual status
    status = rip.get_ritual_status(ritual_id)
    print(f"Current status: {status['status']}")
