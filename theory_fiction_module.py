"""
Theory Fiction Module - Enhanced Version

A comprehensive system for generating, exploring, and analyzing theory-fiction narratives 
that blend speculative storytelling with theoretical and philosophical frameworks.
"""

import random
import json
import uuid
import datetime
import math
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import numpy as np
from collections import defaultdict, Counter


class NarrativeMode(Enum):
    """Enumeration of possible narrative modes"""
    SPECULATIVE = auto()
    CRITICAL = auto()
    ANALYTICAL = auto()
    DIALECTICAL = auto()
    EXPLORATORY = auto()
    RECURSIVE = auto()


class TheoreticalDomain(Enum):
    """Major theoretical domains that can be incorporated"""
    POSTHUMANISM = auto()
    ACCELERATION = auto()
    CYBERNETICS = auto()
    PSYCHOANALYSIS = auto()
    PHENOMENOLOGY = auto()
    NEW_MATERIALISM = auto()
    OBJECT_ORIENTED_ONTOLOGY = auto()
    XENOFEMINISM = auto()
    DECOLONIAL_THEORY = auto()
    MEDIA_ARCHAEOLOGY = auto()
    SYSTEMS_THEORY = auto()
    DECONSTRUCTION = auto()


class EventCategory(Enum):
    """Categories of narrative events"""
    TECHNOLOGICAL = auto()
    SOCIAL = auto()
    ECONOMIC = auto()
    ENVIRONMENTAL = auto()
    EPISTEMOLOGICAL = auto()
    ONTOLOGICAL = auto()
    PSYCHOLOGICAL = auto()
    POLITICAL = auto()


@dataclass
class Character:
    """
    Represents a narrative character with detailed attributes and dynamic states.
    """
    name: str
    background: str
    motivations: List[str]
    traits: Dict[str, float] = field(default_factory=dict)
    relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    perspectives: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    arc: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update character's current state"""
        self.state.update(new_state)
        self.arc.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "state": new_state.copy()
        })
    
    def add_relationship(self, target_character_id: str, relation_type: str, 
                         strength: float, attributes: Dict[str, Any] = None) -> None:
        """
        Add or update a relationship to another character
        
        Args:
            target_character_id: ID of the character this relationship points to
            relation_type: Type of relationship (ally, rival, mentor, etc.)
            strength: Strength of relationship (-1.0 to 1.0)
            attributes: Additional attributes describing the relationship
        """
        self.relationships[target_character_id] = {
            "type": relation_type,
            "strength": max(-1.0, min(1.0, strength)),  # Clamp between -1 and 1
            "attributes": attributes or {}
        }
    
    def get_trait(self, trait_name: str, default: float = 0.5) -> float:
        """Get character trait value with default if not set"""
        return self.traits.get(trait_name, default)
    
    def set_perspective(self, concept: str, position: Any) -> None:
        """Set character's perspective on a theoretical concept"""
        self.perspectives[concept] = position
    
    def predict_reaction(self, event: 'NarrativeEvent') -> Dict[str, Any]:
        """
        Predict how the character would react to an event based on traits and motivations
        
        Args:
            event: The narrative event to react to
            
        Returns:
            Dictionary describing the character's likely reaction
        """
        # Calculate emotional response based on traits and event content
        emotional_response = {}
        
        # Basic emotional dimensions
        for emotion in ["fear", "joy", "anger", "surprise", "disgust", "trust"]:
            # Default to mid-range emotional response
            base_intensity = 0.5
            
            # Adjust based on traits that might amplify or diminish the emotion
            trait_modifier = 0.0
            
            # Example trait correlations (simplified)
            if emotion == "fear" and "courage" in self.traits:
                trait_modifier -= self.traits["courage"] * 0.5
            elif emotion == "anger" and "patience" in self.traits:
                trait_modifier -= self.traits["patience"] * 0.5
            elif emotion == "joy" and "optimism" in self.traits:
                trait_modifier += self.traits["optimism"] * 0.3
                
            # Event category can affect emotional responses
            category_modifier = 0.0
            if event.category == EventCategory.TECHNOLOGICAL:
                if "technophobia" in self.traits:
                    if emotion == "fear":
                        category_modifier += self.traits["technophobia"] * 0.4
                    elif emotion == "disgust":
                        category_modifier += self.traits["technophobia"] * 0.3
                elif "technophilia" in self.traits:
                    if emotion == "joy":
                        category_modifier += self.traits["technophilia"] * 0.4
                    elif emotion == "trust":
                        category_modifier += self.traits["technophilia"] * 0.3
            
            # Final calculation with normalization
            intensity = max(0.0, min(1.0, base_intensity + trait_modifier + category_modifier))
            emotional_response[emotion] = intensity
        
        # Determine likely actions based on motivations and event content
        potential_actions = []
        
        # Check if event aligns with or threatens motivations
        for motivation in self.motivations:
            # Simple keyword matching (a more sophisticated system would use semantic analysis)
            if any(kw in event.description.lower() for kw in motivation.lower().split()):
                potential_actions.append({
                    "type": "pursue",
                    "description": f"Pursue opportunities related to motivation: {motivation}",
                    "probability": 0.7
                })
        
        # Return composite reaction
        return {
            "character_id": self.id,
            "character_name": self.name,
            "emotional_response": emotional_response,
            "potential_actions": potential_actions,
            "event_id": event.id
        }


@dataclass
class WorldParameter:
    """
    Represents a configurable parameter of the narrative world.
    """
    name: str
    value: Any
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def update_value(self, new_value: Any) -> None:
        """Update parameter value and record in history"""
        self.history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "previous_value": self.value,
            "new_value": new_value
        })
        self.value = new_value
    
    def get_history_snapshot(self, index: int) -> Any:
        """Get value at a specific point in history"""
        if not self.history:
            return self.value
        if index < 0 or index >= len(self.history):
            raise IndexError(f"History index {index} out of range (0-{len(self.history)-1})")
        return self.history[index]["new_value"]
    
    def within_constraints(self, value: Any = None) -> bool:
        """Check if value is within defined constraints"""
        if value is None:
            value = self.value
            
        if not self.constraints:
            return True
            
        # Numeric constraints
        if "min" in self.constraints and value < self.constraints["min"]:
            return False
        if "max" in self.constraints and value > self.constraints["max"]:
            return False
            
        # Set membership constraints
        if "allowed_values" in self.constraints and value not in self.constraints["allowed_values"]:
            return False
            
        return True


@dataclass
class NarrativeEvent:
    """
    Represents a significant event within the narrative.
    """
    description: str
    category: EventCategory
    probability: float = 1.0
    impact: Dict[str, float] = field(default_factory=dict)
    affected_characters: List[str] = field(default_factory=list)
    affected_parameters: Dict[str, Any] = field(default_factory=dict)
    consequences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: random.random())
    occurred: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def resolve(self, narrative_context: 'NarrativeContext') -> Dict[str, Any]:
        """
        Resolve the event's occurrence and its effects on the narrative
        
        Args:
            narrative_context: The context in which this event occurs
            
        Returns:
            Dictionary containing resolution results
        """
        # Determine if event occurs based on probability
        rolls_occurred = random.random() < self.probability
        self.occurred = rolls_occurred
        
        if not self.occurred:
            return {
                "event_id": self.id,
                "occurred": False,
                "resolution": "Event did not occur based on probability"
            }
        
        # Event has occurred, calculate effects
        results = {
            "event_id": self.id,
            "occurred": True,
            "parameter_changes": {},
            "character_reactions": []
        }
        
        # Apply parameter changes
        for param_id, change in self.affected_parameters.items():
            parameter = narrative_context.get_parameter_by_id(param_id)
            if parameter:
                old_value = parameter.value
                
                # Handle different types of parameter changes
                if isinstance(change, dict) and "operation" in change:
                    if change["operation"] == "add":
                        new_value = old_value + change["value"]
                    elif change["operation"] == "multiply":
                        new_value = old_value * change["value"]
                    elif change["operation"] == "set":
                        new_value = change["value"]
                    else:
                        new_value = old_value
                else:
                    # Direct value assignment
                    new_value = change
                
                # Update parameter if within constraints
                if parameter.within_constraints(new_value):
                    parameter.update_value(new_value)
                    results["parameter_changes"][param_id] = {
                        "parameter_name": parameter.name,
                        "old_value": old_value,
                        "new_value": new_value
                    }
        
        # Calculate character reactions
        for char_id in self.affected_characters:
            character = narrative_context.get_character_by_id(char_id)
            if character:
                reaction = character.predict_reaction(self)
                results["character_reactions"].append(reaction)
        
        return results


@dataclass
class TheoreticalConcept:
    """
    Represents a theoretical concept that can be applied to narrative analysis.
    """
    name: str
    definition: str
    domain: TheoreticalDomain
    implications: Dict[str, str] = field(default_factory=dict)
    related_concepts: Dict[str, float] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def generate_analytical_lens(self) -> Dict[str, Any]:
        """
        Generate an analytical lens based on this concept
        
        Returns:
            Dictionary representing the analytical lens
        """
        return {
            "concept_id": self.id,
            "concept_name": self.name,
            "lens_type": "theoretical",
            "domain": self.domain.name,
            "analytical_questions": [
                f"How does {self.name} manifest in the narrative structure?",
                f"What aspects of the narrative challenge conventional understandings of {self.name}?",
                f"In what ways do characters embody or resist {self.name}?",
                f"How might the narrative be reinterpreted through the lens of {self.name}?"
            ],
            "key_implications": self.implications
        }
    
    def apply_to_event(self, event: NarrativeEvent) -> Dict[str, Any]:
        """
        Apply this theoretical concept to analyze a narrative event
        
        Args:
            event: The narrative event to analyze
            
        Returns:
            Dictionary containing analytical insights
        """
        # Generate interpretations based on theoretical domain
        interpretations = []
        
        # Base interpretation on theoretical domain
        if self.domain == TheoreticalDomain.POSTHUMANISM:
            if event.category == EventCategory.TECHNOLOGICAL:
                interpretations.append(
                    f"This event demonstrates how technology reshapes human experience, "
                    f"challenging traditional humanist boundaries between human and machine."
                )
            elif event.category == EventCategory.ONTOLOGICAL:
                interpretations.append(
                    f"The ontological shift described represents a posthuman understanding "
                    f"of being that transcends traditional humanist categories."
                )
                
        elif self.domain == TheoreticalDomain.ACCELERATION:
            if event.category == EventCategory.TECHNOLOGICAL or event.category == EventCategory.ECONOMIC:
                interpretations.append(
                    f"The {event.category.name.lower()} acceleration depicted in this event "
                    f"illustrates the compounding, non-linear dynamics of change in late capitalism."
                )
                
        # More domain-specific interpretations could be added for other domains
                
        # Generic fallback interpretation if none were generated
        if not interpretations:
            interpretations.append(
                f"From the perspective of {self.name} ({self.domain.name}), "
                f"this {event.category.name.lower()} event illustrates changing paradigms "
                f"that recontextualize our understanding of the subject matter."
            )
        
        return {
            "concept_applied": self.name,
            "event_id": event.id,
            "interpretations": interpretations,
            "theoretical_domain": self.domain.name,
            "implications": list(self.implications.values())
        }


@dataclass
class TheoreticalFramework:
    """
    Represents a coherent theoretical framework composed of multiple concepts.
    """
    name: str
    description: str
    concepts: Dict[str, TheoreticalConcept] = field(default_factory=dict)
    core_principles: List[str] = field(default_factory=list)
    key_thinkers: List[str] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def add_concept(self, concept: TheoreticalConcept) -> None:
        """Add a theoretical concept to the framework"""
        self.concepts[concept.id] = concept
    
    def get_concept(self, concept_id: str) -> Optional[TheoreticalConcept]:
        """Get a concept by ID"""
        return self.concepts.get(concept_id)
    
    def apply_lens(self, narrative_context: 'NarrativeContext') -> Dict[str, Any]:
        """
        Apply this theoretical framework as an analytical lens to a narrative context
        
        Args:
            narrative_context: The narrative context to analyze
            
        Returns:
            Dictionary containing analytical results
        """
        results = {
            "framework_id": self.id,
            "framework_name": self.name,
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "overall_themes": [],
            "character_analyses": [],
            "event_analyses": [],
            "world_parameter_analyses": []
        }
        
        # Set theoretical lens on the narrative context
        narrative_context.theoretical_lens = self.name
        
        # Analyze characters through this framework
        for character in narrative_context.characters.values():
            character_analysis = {
                "character_id": character.id,
                "character_name": character.name,
                "theoretical_insights": []
            }
            
            # Apply each concept to the character
            for concept in self.concepts.values():
                # This is a simplified analysis
                # A more sophisticated version would analyze character traits and motivations
                # in relation to the specific theoretical concepts
                insight = f"Through the lens of {concept.name}, {character.name}'s " \
                          f"motivations around {', '.join(character.motivations[:2])} " \
                          f"can be interpreted as embodying tensions within {self.name}."
                character_analysis["theoretical_insights"].append({
                    "concept": concept.name,
                    "insight": insight
                })
            
            results["character_analyses"].append(character_analysis)
        
        # Analyze events through this framework
        for event in narrative_context.events:
            event_analysis = {
                "event_id": event.id,
                "event_description": event.description,
                "theoretical_interpretations": []
            }
            
            # Apply each concept to the event
            for concept in self.concepts.values():
                analysis = concept.apply_to_event(event)
                event_analysis["theoretical_interpretations"].append({
                    "concept": concept.name,
                    "interpretations": analysis["interpretations"]
                })
            
            results["event_analyses"].append(event_analysis)
        
        # Derive overall themes based on the framework
        # This uses a simplified approach; a more sophisticated version would
        # perform deeper analysis of patterns across characters and events
        theme_candidates = []
        for principle in self.core_principles:
            theme = f"The narrative explores {principle} through its portrayal of " \
                    f"{narrative_context.events[0].category.name.lower() if narrative_context.events else 'events'} " \
                    f"and character dynamics."
            theme_candidates.append(theme)
        
        # Add themes to results
        results["overall_themes"] = theme_candidates[:3]  # Limit to top 3 themes
        
        return results


@dataclass
class NarrativeContext:
    """
    Represents a complex narrative context with multiple dimensions and internal coherence.
    """
    title: str
    core_premise: str
    characters: Dict[str, Character] = field(default_factory=dict)
    world_parameters: Dict[str, WorldParameter] = field(default_factory=dict)
    events: List[NarrativeEvent] = field(default_factory=list)
    theoretical_lens: str = ""
    mode: NarrativeMode = NarrativeMode.SPECULATIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def add_character(self, character: Character) -> str:
        """
        Add a character to the narrative context
        
        Args:
            character: Character to add
            
        Returns:
            ID of the added character
        """
        self.characters[character.id] = character
        return character.id
    
    def get_character_by_id(self, character_id: str) -> Optional[Character]:
        """Get a character by ID"""
        return self.characters.get(character_id)
    
    def get_character_by_name(self, name: str) -> Optional[Character]:
        """Get a character by name (returns first match)"""
        for character in self.characters.values():
            if character.name == name:
                return character
        return None
    
    def add_parameter(self, parameter: WorldParameter) -> str:
        """
        Add a world parameter to the narrative context
        
        Args:
            parameter: Parameter to add
            
        Returns:
            ID of the added parameter
        """
        self.world_parameters[parameter.id] = parameter
        return parameter.id
    
    def get_parameter_by_id(self, parameter_id: str) -> Optional[WorldParameter]:
        """Get a parameter by ID"""
        return self.world_parameters.get(parameter_id)
    
    def get_parameter_by_name(self, name: str) -> Optional[WorldParameter]:
        """Get a parameter by name (returns first match)"""
        for parameter in self.world_parameters.values():
            if parameter.name == name:
                return parameter
        return None
    
    def add_event(self, event: NarrativeEvent) -> str:
        """
        Add an event to the narrative context
        
        Args:
            event: Event to add
            
        Returns:
            ID of the added event
        """
        self.events.append(event)
        return event.id
    
    def get_event_by_id(self, event_id: str) -> Optional[NarrativeEvent]:
        """Get an event by ID"""
        for event in self.events:
            if event.id == event_id:
                return event
        return None
    
    def sort_events_by_timestamp(self) -> None:
        """Sort events by their timestamp"""
        self.events.sort(key=lambda e: e.timestamp)
    
    def resolve_events(self) -> List[Dict[str, Any]]:
        """
        Resolve all events in the narrative
        
        Returns:
            List of event resolution results
        """
        results = []
        self.sort_events_by_timestamp()
        
        for event in self.events:
            resolution = event.resolve(self)
            results.append(resolution)
        
        return results
    
    def generate_synopsis(self) -> str:
        """
        Generate a narrative synopsis based on the current state
        
        Returns:
            Synopsis text
        """
        # Ensure events are in order
        self.sort_events_by_timestamp()
        
        # Collect character names
        character_names = [char.name for char in self.characters.values()]
        character_list = ", ".join(character_names[:-1]) + " and " + character_names[-1] if character_names else "No characters"
        
        # Count occurring events
        occurred_events = [e for e in self.events if e.occurred]
        
        # Create synopsis
        synopsis = f"# {self.title}\n\n"
        synopsis += f"*{self.core_premise}*\n\n"
        
        if self.theoretical_lens:
            synopsis += f"Through the lens of {self.theoretical_lens}, "
        
        synopsis += f"this {self.mode.name.lower()} narrative follows {character_list} "
        synopsis += f"in a world where "
        
        # Add some world parameters
        if self.world_parameters:
            param_descriptions = []
            for param in self.world_parameters.values()[:3]:  # Limit to 3 for readability
                param_descriptions.append(f"{param.name} is {param.value}")
            synopsis += ", ".join(param_descriptions) + ". "
        else:
            synopsis += "various forces shape the characters' experiences. "
        
        # Add events summary
        if occurred_events:
            synopsis += f"\n\nThe narrative explores {len(occurred_events)} key events, "
            synopsis += f"including {occurred_events[0].description.lower()}"
            
            if len(occurred_events) > 1:
                synopsis += f" and culminates with {occurred_events[-1].description.lower()}"
            
            synopsis += "."
        
        return synopsis


class NarrativeGenerator:
    """
    Generates narrative elements using configurable parameters.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the narrative generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed if seed is not None else random.randint(1, 10000)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Templates for narrative generation
        self.character_archetypes = {
            "scholar": {
                "traits": {"curiosity": 0.8, "intellect": 0.7, "caution": 0.6},
                "motivation_templates": [
                    "Uncover the truth about {subject}",
                    "Document the effects of {phenomenon}",
                    "Understand the mechanics of {system}"
                ]
            },
            "pioneer": {
                "traits": {"courage": 0.8, "adaptability": 0.7, "independence": 0.7},
                "motivation_templates": [
                    "Explore the frontiers of {subject}",
                    "Test the limits of {phenomenon}",
                    "Demonstrate the potential of {system}"
                ]
            },
            "critic": {
                "traits": {"skepticism": 0.8, "perception": 0.7, "articulation": 0.7},
                "motivation_templates": [
                    "Challenge assumptions about {subject}",
                    "Expose the dangers of {phenomenon}",
                    "Question the implementation of {system}"
                ]
            },
            "mediator": {
                "traits": {"empathy": 0.8, "diplomacy": 0.7, "patience": 0.7},
                "motivation_templates": [
                    "Bridge divides concerning {subject}",
                    "Mitigate the impact of {phenomenon}",
                    "Find balance within {system}"
                ]
            }
        }
        
        self.event_templates = {
            EventCategory.TECHNOLOGICAL: [
                "A breakthrough in {technology} enables unprecedented {capability}",
                "The failure of {technology} leads to widespread {consequence}",
                "A new form of {technology} emerges, challenging established {convention}"
            ],
            EventCategory.SOCIAL: [
                "A movement advocating for {principle} gains momentum",
                "Tensions between {group_a} and {group_b} reach a breaking point",
                "A new social structure based on {principle} begins to form"
            ],
            EventCategory.ECONOMIC: [
                "The {resource} market collapses, triggering widespread {consequence}",
                "A new economic model based on {principle} disrupts traditional {system}",
                "Resource scarcity in {resource} leads to innovation in {alternative}"
            ],
            EventCategory.ENVIRONMENTAL: [
                "An ecological threshold is crossed, causing {phenomenon}",
                "Adaptation to {phenomenon} creates new forms of {adaptation}",
                "The relationship between {entity_a} and {entity_b} is transformed by environmental change"
            ],
            EventCategory.EPISTEMOLOGICAL: [
                "A paradigm shift in understanding {subject} overturns established {knowledge}",
                "The boundaries between {concept_a} and {concept_b} begin to dissolve",
                "A new framework for understanding {subject} emerges, centered on {principle}"
            ],
            EventCategory.ONTOLOGICAL: [
                "The nature of {entity} fundamentally transforms",
                "A new form of {entity} comes into being, with properties of both {category_a} and {category_b}",
                "The distinction between {category_a} and {category_b} collapses"
            ],
            EventCategory.PSYCHOLOGICAL: [
                "A new form of consciousness emerges, characterized by {quality}",
                "Widespread psychological adaptation to {circumstance} leads to {consequence}",
                "The collective psyche undergoes transformation due to {stimulus}"
            ],
            EventCategory.POLITICAL: [
                "Power structures reorganize around {principle} rather than {old_principle}",
                "Governance systems integrate {technology}, leading to {consequence}",
                "Political agency extends to {entity}, challenging definitions of {concept}"
            ]
        }
        
        # Placeholders for template substitution
        self.subjects = ["digital consciousness", "artificial intelligence", "posthuman embodiment",
                        "biotechnology", "climate engineering", "quantum computing", 
                        "neuro-enhancement", "virtual reality", "genetic modification"]
        
        self.phenomena = ["technological acceleration", "climate change", "automation",
                         "digital convergence", "cognitive augmentation", "social fragmentation",
                         "human-machine integration", "algorithmic governance"]
        
        self.systems = ["capitalism", "democracy", "cognitive architecture", "distributed networks",
                       "planetary computation", "global finance", "digital ecosystems"]
        
        self.technologies = ["neural interfaces", "autonomous systems", "distributed ledgers",
                           "synthetic biology", "immersive realities", "quantum networks"]
        
        self.capabilities = ["thought transmission", "reality manipulation", "identity fluidity",
                           "cognitive enhancement", "memory externalization", "sensory expansion"]
        
        self.consequences = ["social stratification", "cognitive fragmentation", "existential realignment",
                           "cultural transformation", "institutional collapse", "species divergence"]
        
        self.conventions = ["human exceptionalism", "corporeal identity", "cognitive sovereignty",
                          "technological determinism", "social organization", "economic value"]
        
        self.principles = ["distributed agency", "technological embodiment", "cognitive diversity",
                         "algorithmic mediation", "post-scarcity economics", "radical inclusivity"]
        
        self.groups = ["techno-progressives", "bio-conservatives", "digital natives",
                      "human enhancement advocates", "artificial intelligences", "embodied cognitivists",
                      "traditional institutions", "networked collectives"]
        
        self.resources = ["computational power", "attention", "data", "intellectual property",
                        "genetic information", "cognitive labor", "emotional engagement"]
        
        self.entities = ["human consciousness", "artificial intelligence", "distributed cognition",
                       "technological systems", "networked ecosystems", "hybrid intelligences"]
        
        self.categories = ["material", "virtual", "biological", "technological",
                         "individual", "collective", "organic", "synthetic"]
        
        self.qualities = ["distributed awareness", "hyperconnectivity", "temporal flexibility",
                        "recursive self-modification", "empathic expansion", "cognitive fluidity"]
        
        self.stimuli = ["immersive technologies", "global connectivity", "ecological awareness",
                       "technological mediation", "algorithmic personalization"]
    
    def generate_character(self, name: str = None, archetype: str = None) -> Character:
        """
        Generate a character using templates and randomization
        
        Args:
            name: Optional character name (generated if None)
            archetype: Optional character archetype (randomly selected if None)
            
        Returns:
            Generated Character object
        """
        # Generate or use provided name
        if not name:
            # In a real implementation, this would use a more sophisticated name generator
            first_names = ["Aria", "Zhen", "Elian", "Noor", "Soren", "Lyra", "Kai", "Maya", "Jin", "Nova"]
            last_names = ["Wei", "Chaudry", "Okoro", "Chen", "Patel", "Kim", "Nguyen", "Adeyemi", "Singh"]
            name = random.choice(first_names) + " " + random.choice(last_names)
        
        # Select archetype
        if not archetype or archetype not in self.character_archetypes:
            archetype = random.choice(list(self.character_archetypes.keys()))
        
        # Get archetype template
        archetype_template = self.character_archetypes[archetype]
        
        # Generate background
        backgrounds = [
            f"Researcher in {random.choice(self.subjects)}",
            f"Pioneer of {random.choice(self.technologies)}",
            f"Critic of {random.choice(self.conventions)}",
            f"Mediator between {random.choice(self.groups)} and {random.choice(self.groups)}"
        ]
        background = random.choice(backgrounds)
        
        # Generate motivations
        motivations = []
        for template in archetype_template["motivation_templates"][:2]:  # Use first two templates
            # Fill in template with random subjects/phenomena/systems
            filled_template = template.format(
                subject=random.choice(self.subjects),
                phenomenon=random.choice(self.phenomena),
                system=random.choice(self.systems)
            )
            motivations.append(filled_template)
        
        # Create character with generated attributes
        character = Character(
            name=name,
            background=background,
            motivations=motivations,
            traits=archetype_template["traits"].copy()
        )
        
        return character
    
    def generate_world_parameter(self, name: str = None, value: Any = None) -> WorldParameter:
        """
        Generate a world parameter
        
        Args:
            name: Optional parameter name (generated if None)
            value: Optional parameter value (generated if None)
            
        Returns:
            Generated WorldParameter object
        """
        if not name:
            # Generate parameter name
            prefixes = ["technological", "social", "cognitive", "economic", "environmental"]
            suffixes = ["integration", "fragmentation", "acceleration", "transformation", "hybridity"]
            name = f"{random.choice(prefixes)}_{random.choice(suffixes)}"
        
        if value is None:
            # Generate parameter value (default to a float between 0 and 1)
            value = round(random.random(), 2)
        
        # Generate parameter description
        descriptions = [
            f"Measures the degree of {name.replace('_', ' ')} in the narrative world",
            f"Represents the intensity of {name.replace('_', ' ')} experienced by characters",
            f"Quantifies the social impact of {name.replace('_', ' ')} on institutions"
        ]
        description = random.choice(descriptions)
        
        # Generate constraints
        constraints = {}
        if isinstance(value, (int, float)):
            constraints = {"min": 0.0, "max": 1.0}
        
        return WorldParameter(
            name=name,
            value=value,
            description=description,
            constraints=constraints
        )
    
    def generate_narrative_event(self, category: EventCategory = None) -> NarrativeEvent:
        """
        Generate a narrative event
        
        Args:
            category: Optional event category (randomly selected if None)
            
        Returns:
            Generated NarrativeEvent object
        """
        # Select category if not provided
        if category is None:
            category = random.choice(list(EventCategory))
        
        # Select template
        templates = self.event_templates.get(category, ["A significant event occurs"])
        template = random.choice(templates)
        
        # Create placeholder replacements dictionary
        replacements = {
            "technology": random.choice(self.technologies),
            "capability": random.choice(self.capabilities),
            "consequence": random.choice(self.consequences),
            "convention": random.choice(self.conventions),
            "principle": random.choice(self.principles),
            "group_a": random.choice(self.groups),
            "group_b": random.choice(self.groups),
            "resource": random.choice(self.resources),
            "system": random.choice(self.systems),
            "alternative": random.choice(self.technologies),
            "phenomenon": random.choice(self.phenomena),
            "adaptation": random.choice(self.capabilities),
            "entity_a": random.choice(self.entities),
            "entity_b": random.choice(self.entities),
            "subject": random.choice(self.subjects),
            "knowledge": random.choice(self.conventions),
            "concept_a": random.choice(self.subjects),
            "concept_b": random.choice(self.subjects),
            "entity": random.choice(self.entities),
            "category_a": random.choice(self.categories),
            "category_b": random.choice(self.categories),
            "quality": random.choice(self.qualities),
            "circumstance": random.choice(self.phenomena),
            "stimulus": random.choice(self.stimuli),
            "old_principle": random.choice(self.conventions)
        }
        
        # Fill template with replacements
        description = template
        for key, value in replacements.items():
            placeholder = "{" + key + "}"
            if placeholder in description:
                description = description.replace(placeholder, value)
        
        # Generate random probability (weighted toward higher probabilities)
        probability = 0.5 + (random.random() * 0.5)
        
        # Generate random impact
        impact = {
            "social": random.random(),
            "technological": random.random(),
            "economic": random.random(),
            "existential": random.random()
        }
        
        return NarrativeEvent(
            description=description,
            category=category,
            probability=probability,
            impact=impact
        )
    
    def generate_theoretical_concept(self, name: str = None, domain: TheoreticalDomain = None) -> TheoreticalConcept:
        """
        Generate a theoretical concept
        
        Args:
            name: Optional concept name (generated if None)
            domain: Optional theoretical domain (randomly selected if None)
            
        Returns:
            Generated TheoreticalConcept object
        """
        # Select domain if not provided
        if domain is None:
            domain = random.choice(list(TheoreticalDomain))
        
        # Generate name if not provided
        if not name:
            prefixes = {
                TheoreticalDomain.POSTHUMANISM: ["post", "trans", "meta"],
                TheoreticalDomain.ACCELERATION: ["accelerated", "recursive", "exponential"],
                TheoreticalDomain.CYBERNETICS: ["cyber", "feedback", "systems"],
                TheoreticalDomain.PSYCHOANALYSIS: ["psycho", "unconscious", "libidinal"],
                TheoreticalDomain.PHENOMENOLOGY: ["phenomenal", "experiential", "embodied"],
                TheoreticalDomain.NEW_MATERIALISM: ["material", "agential", "vital"],
                TheoreticalDomain.OBJECT_ORIENTED_ONTOLOGY: ["object", "flat", "hyperobject"],
                TheoreticalDomain.XENOFEMINISM: ["xeno", "gender", "techno"],
                TheoreticalDomain.DECOLONIAL_THEORY: ["decolonial", "border", "pluriversal"],
                TheoreticalDomain.MEDIA_ARCHAEOLOGY: ["media", "archival", "technical"],
                TheoreticalDomain.SYSTEMS_THEORY: ["systemic", "emergent", "complex"],
                TheoreticalDomain.DECONSTRUCTION: ["deconstructive", "différance", "trace"]
            }
            
            suffixes = {
                TheoreticalDomain.POSTHUMANISM: ["humanity", "subjectivity", "embodiment"],
                TheoreticalDomain.ACCELERATION: ["acceleration", "growth", "capitalism"],
                TheoreticalDomain.CYBERNETICS: ["system", "control", "information"],
                TheoreticalDomain.PSYCHOANALYSIS: ["desire", "trauma", "fantasy"],
                TheoreticalDomain.PHENOMENOLOGY: ["experience", "perception", "consciousness"],
                TheoreticalDomain.NEW_MATERIALISM: ["materiality", "entanglement", "agency"],
                TheoreticalDomain.OBJECT_ORIENTED_ONTOLOGY: ["ontology", "withdrawal", "existence"],
                TheoreticalDomain.XENOFEMINISM: ["feminism", "technicity", "rationality"],
                TheoreticalDomain.DECOLONIAL_THEORY: ["thinking", "epistemology", "resistance"],
                TheoreticalDomain.MEDIA_ARCHAEOLOGY: ["mediation", "temporality", "archaeology"],
                TheoreticalDomain.SYSTEMS_THEORY: ["emergence", "autopoiesis", "complexity"],
                TheoreticalDomain.DECONSTRUCTION: ["reading", "text", "absence"]
            }
            
            domain_prefixes = prefixes.get(domain, ["theoretical"])
            domain_suffixes = suffixes.get(domain, ["concept"])
            
            name = f"{random.choice(domain_prefixes)}_{random.choice(domain_suffixes)}"
        
        # Generate definition
        definitions = {
            TheoreticalDomain.POSTHUMANISM: f"A framework for understanding {random.choice(self.subjects)} beyond traditional humanist assumptions",
            TheoreticalDomain.ACCELERATION: f"The theory of how {random.choice(self.phenomena)} accelerates under late capitalism",
            TheoreticalDomain.CYBERNETICS: f"The study of {random.choice(self.systems)} as self-regulating systems with feedback mechanisms",
            TheoreticalDomain.PSYCHOANALYSIS: f"Analysis of unconscious processes underlying {random.choice(self.subjects)}",
            TheoreticalDomain.PHENOMENOLOGY: f"The study of subjective experience of {random.choice(self.subjects)} as it appears to consciousness",
            TheoreticalDomain.NEW_MATERIALISM: f"A theoretical approach emphasizing the agency of {random.choice(self.entities)} beyond human intention",
            TheoreticalDomain.OBJECT_ORIENTED_ONTOLOGY: f"A philosophical perspective examining {random.choice(self.subjects)} independent of human access",
            TheoreticalDomain.XENOFEMINISM: f"A technomaterialist, anti-naturalist, gender abolitionist approach to {random.choice(self.subjects)}",
            TheoreticalDomain.DECOLONIAL_THEORY: f"Critical perspective challenging colonial assumptions in {random.choice(self.subjects)}",
            TheoreticalDomain.MEDIA_ARCHAEOLOGY: f"The non-linear investigation of {random.choice(self.technologies)} through technical media",
            TheoreticalDomain.SYSTEMS_THEORY: f"Analysis of {random.choice(self.entities)} as complex, emergent systems with self-organizing properties",
            TheoreticalDomain.DECONSTRUCTION: f"A method of critical analysis exposing assumptions in binary oppositions within {random.choice(self.subjects)}"
        }
        
        definition = definitions.get(domain, f"A theoretical perspective on {random.choice(self.subjects)}")
        
        # Generate implications
        implication_templates = [
            f"Challenges conventional understandings of {random.choice(self.subjects)}",
            f"Reframes the relationship between {random.choice(self.entities)} and {random.choice(self.entities)}",
            f"Offers new perspectives on {random.choice(self.phenomena)}",
            f"Questions established categories of {random.choice(self.categories)} and {random.choice(self.categories)}"
        ]
        
        implications = {}
        for i, template in enumerate(implication_templates[:3]):  # Use first three templates
            key = f"implication_{i+1}"
            implications[key] = template
        
        return TheoreticalConcept(
            name=name,
            definition=definition,
            domain=domain,
            implications=implications
        )
    
    def generate_theoretical_framework(self, name: str = None, num_concepts: int = 3) -> TheoreticalFramework:
        """
        Generate a theoretical framework with multiple concepts
        
        Args:
            name: Optional framework name (generated if None)
            num_concepts: Number of concepts to include in the framework
            
        Returns:
            Generated TheoreticalFramework object
        """
        # Select a primary domain
        primary_domain = random.choice(list(TheoreticalDomain))
        
        # Generate name if not provided
        if not name:
            prefixes = ["Critical", "Speculative", "Theoretical", "Analytical", "Recursive"]
            domain_name = primary_domain.name.replace("_", " ").title()
            name = f"{random.choice(prefixes)} {domain_name}"
        
        # Generate description
        descriptions = [
            f"A framework combining concepts from {primary_domain.name.replace('_', ' ')} to analyze contemporary phenomena",
            f"An analytical approach drawing on {primary_domain.name.replace('_', ' ')} to interpret complex systems",
            f"A theoretical lens applying {primary_domain.name.replace('_', ' ')} concepts to speculative scenarios"
        ]
        description = random.choice(descriptions)
        
        # Generate core principles
        principle_templates = [
            f"The {random.choice(self.qualities)} of {random.choice(self.entities)}",
            f"The relationship between {random.choice(self.categories)} and {random.choice(self.categories)}",
            f"The emergence of {random.choice(self.phenomena)} through {random.choice(self.systems)}",
            f"The transformation of {random.choice(self.conventions)} via {random.choice(self.technologies)}"
        ]
        
        core_principles = random.sample(principle_templates, min(3, len(principle_templates)))
        
        # Generate key thinkers (fictional or real)
        first_names = ["Donna", "Gilles", "Katherine", "Nick", "Rosi", "Bruno", "Karen", "Jacques", "Judith", "Mark"]
        last_names = ["Haraway", "Deleuze", "Hayles", "Land", "Braidotti", "Latour", "Barad", "Derrida", "Butler", "Fisher"]
        
        num_thinkers = random.randint(2, 4)
        key_thinkers = []
        for _ in range(num_thinkers):
            thinker = f"{random.choice(first_names)} {random.choice(last_names)}"
            key_thinkers.append(thinker)
        
        # Create the framework
        framework = TheoreticalFramework(
            name=name,
            description=description,
            core_principles=core_principles,
            key_thinkers=key_thinkers
        )
        
        # Add concepts to the framework
        for _ in range(num_concepts):
            # Generate a concept, mostly from the primary domain
            if random.random() < 0.7:
                domain = primary_domain
            else:
                domain = random.choice(list(TheoreticalDomain))
                
            concept = self.generate_theoretical_concept(domain=domain)
            framework.add_concept(concept)
        
        return framework
    
    def generate_narrative_context(self, title: str = None, core_premise: str = None,
                                 num_characters: int = 3, num_parameters: int = 3,
                                 mode: NarrativeMode = None) -> NarrativeContext:
        """
        Generate a complete narrative context
        
        Args:
            title: Optional narrative title (generated if None)
            core_premise: Optional core premise (generated if None)
            num_characters: Number of characters to generate
            num_parameters: Number of world parameters to generate
            mode: Narrative mode (randomly selected if None)
            
        Returns:
            Generated NarrativeContext object
        """
        # Generate title if not provided
        if not title:
            adjectives = ["Recursive", "Speculative", "Emergent", "Synthetic", "Quantum", "Posthuman"]
            nouns = ["Convergence", "Transformation", "Entanglement", "Acceleration", "Divergence", "Threshold"]
            title = f"{random.choice(adjectives)} {random.choice(nouns)}"
        
        # Generate core premise if not provided
        if not core_premise:
            subjects = random.choice(self.subjects)
            phenomena = random.choice(self.phenomena)
            core_premise = f"Exploring {subjects} in an era of {phenomena}"
        
        # Select mode if not provided
        if mode is None:
            mode = random.choice(list(NarrativeMode))
        
        # Create the narrative context
        context = NarrativeContext(
            title=title,
            core_premise=core_premise,
            mode=mode
        )
        
        # Add characters
        for _ in range(num_characters):
            character = self.generate_character()
            context.add_character(character)
        
        # Add world parameters
        for _ in range(num_parameters):
            parameter = self.generate_world_parameter()
            context.add_parameter(parameter)
        
        return context
    
    def generate_speculative_scenario(self, context: NarrativeContext, complexity: int = 3) -> None:
        """
        Generate a speculative scenario with multiple interconnected events
        
        Args:
            context: Narrative context to add events to
            complexity: Number of interconnected events to generate
        """
        # Generate primary event categories based on the narrative mode
        if context.mode == NarrativeMode.SPECULATIVE:
            primary_categories = [EventCategory.TECHNOLOGICAL, EventCategory.ONTOLOGICAL]
        elif context.mode == NarrativeMode.CRITICAL:
            primary_categories = [EventCategory.SOCIAL, EventCategory.POLITICAL]
        elif context.mode == NarrativeMode.ANALYTICAL:
            primary_categories = [EventCategory.EPISTEMOLOGICAL, EventCategory.PSYCHOLOGICAL]
        elif context.mode == NarrativeMode.DIALECTICAL:
            primary_categories = [EventCategory.SOCIAL, EventCategory.ECONOMIC]
        elif context.mode == NarrativeMode.EXPLORATORY:
            primary_categories = [EventCategory.ENVIRONMENTAL, EventCategory.TECHNOLOGICAL]
        else:  # RECURSIVE mode
            primary_categories = [EventCategory.EPISTEMOLOGICAL, EventCategory.ONTOLOGICAL]
        
        # Generate events with increasing interconnection
        all_character_ids = list(context.characters.keys())
        all_parameter_ids = list(context.world_parameters.keys())
        
        for i in range(complexity):
            # For variety, sometimes use categories outside the primary ones
            if random.random() < 0.7:
                category = random.choice(primary_categories)
            else:
                category = random.choice(list(EventCategory))
            
            # Create the event
            event = self.generate_narrative_event(category=category)
            
            # Set timestamp to create a sequence (with some randomness)
            event.timestamp = (i / complexity) + (random.random() * 0.1)
            
            # Add affected characters with increasing interconnection
            num_affected = min(i + 1, len(all_character_ids))
            event.affected_characters = random.sample(all_character_ids, num_affected)
            
            # Add affected parameters with increasing complexity
            if all_parameter_ids:
                num_affected_params = min(i + 1, len(all_parameter_ids))
                affected_params = random.sample(all_parameter_ids, num_affected_params)
                
                for param_id in affected_params:
                    # Generate a change to the parameter
                    operation = random.choice(["add", "multiply", "set"])
                    
                    if operation == "add":
                        value = random.uniform(-0.2, 0.2)
                    elif operation == "multiply":
                        value = random.uniform(0.8, 1.2)
                    else:  # set
                        value = random.random()
                    
                    event.affected_parameters[param_id] = {
                        "operation": operation,
                        "value": value
                    }
            
            # Add consequences that could lead to future events
            if i < complexity - 1:  # Not the last event
                consequence_templates = [
                    "Leading to increased tensions between {group_a} and {group_b}",
                    "Resulting in accelerated development of {technology}",
                    "Causing widespread reevaluation of {convention}",
                    "Triggering a shift in {system}"
                ]
                
                template = random.choice(consequence_templates)
                replacements = {
                    "group_a": random.choice(self.groups),
                    "group_b": random.choice(self.groups),
                    "technology": random.choice(self.technologies),
                    "convention": random.choice(self.conventions),
                    "system": random.choice(self.systems)
                }
                
                for key, value in replacements.items():
                    placeholder = "{" + key + "}"
                    if placeholder in template:
                        template = template.replace(placeholder, value)
                
                event.consequences.append(template)
            
            # Add the event to the context
            context.add_event(event)


class TheoryFictionSimulation:
    """
    A comprehensive module for generating and exploring theory-fiction narratives.
    Blends speculative storytelling with theoretical and philosophical frameworks.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the Theory-Fiction Simulation module.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        self.seed = seed if seed is not None else random.randint(1, 10000)
        self.generator = NarrativeGenerator(seed=self.seed)
        self.narrative_contexts = {}
        self.theoretical_frameworks = {}
        
        # Analysis tools
        self.narrative_vectors = {}
        self.conceptual_networks = {}
    
    def create_narrative_context(self, title: str, core_premise: str) -> NarrativeContext:
        """
        Create and register a new narrative context.
        
        Args:
            title: Narrative title
            core_premise: Central narrative premise
            
        Returns:
            Created narrative context
        """
        context = self.generator.generate_narrative_context(
            title=title,
            core_premise=core_premise
        )
        self.narrative_contexts[context.id] = context
        return context
    
    def add_theoretical_framework(self, name: str, description: str = None, 
                                concepts: List[Dict[str, Any]] = None) -> TheoreticalFramework:
        """
        Add a new theoretical framework to the simulation.
        
        Args:
            name: Framework name
            description: Optional framework description
            concepts: Optional list of concept dictionaries
            
        Returns:
            Created theoretical framework
        """
        if description is None:
            description = f"A theoretical framework for analyzing {name.lower()} in narrative contexts"
        
        framework = TheoreticalFramework(
            name=name,
            description=description,
            core_principles=[],
            key_thinkers=[]
        )
        
        # Add concepts if provided
        if concepts:
            for concept_data in concepts:
                concept_name = concept_data.get("name", "Unnamed Concept")
                concept_def = concept_data.get("definition", "")
                domain_name = concept_data.get("domain", "POSTHUMANISM")
                
                try:
                    domain = TheoreticalDomain[domain_name]
                except (KeyError, ValueError):
                    domain = TheoreticalDomain.POSTHUMANISM
                
                concept = TheoreticalConcept(
                    name=concept_name,
                    definition=concept_def,
                    domain=domain,
                    implications=concept_data.get("implications", {})
                )
                
                framework.add_concept(concept)
        
        self.theoretical_frameworks[framework.id] = framework
        return framework
    
    def generate_theoretical_framework(self, name: str = None, 
                                     num_concepts: int = 3) -> TheoreticalFramework:
        """
        Generate a theoretical framework with the narrative generator.
        
        Args:
            name: Optional framework name
            num_concepts: Number of concepts to include
            
        Returns:
            Generated framework
        """
        framework = self.generator.generate_theoretical_framework(
            name=name,
            num_concepts=num_concepts
        )
        
        self.theoretical_frameworks[framework.id] = framework
        return framework
    
    def generate_speculative_scenario(self, context_id: str, complexity: int = 3) -> NarrativeContext:
        """
        Generate a speculative scenario within a narrative context.
        
        Args:
            context_id: ID of the narrative context
            complexity: Number of interconnected events to generate
            
        Returns:
            Updated narrative context
        """
        context = self.narrative_contexts.get(context_id)
        if not context:
            raise ValueError(f"No narrative context found with ID {context_id}")
        
        self.generator.generate_speculative_scenario(context, complexity=complexity)
        return context
    
    def apply_theoretical_framework(self, context_id: str, framework_id: str) -> Dict[str, Any]:
        """
        Apply a theoretical framework to a narrative context.
        
        Args:
            context_id: ID of the narrative context
            framework_id: ID of the theoretical framework
            
        Returns:
            Analysis results
        """
        context = self.narrative_contexts.get(context_id)
        if not context:
            raise ValueError(f"No narrative context found with ID {context_id}")
            
        framework = self.theoretical_frameworks.get(framework_id)
        if not framework:
            raise ValueError(f"No theoretical framework found with ID {framework_id}")
        
        return framework.apply_lens(context)
    
    def resolve_narrative(self, context_id: str) -> Dict[str, Any]:
        """
        Resolve all events in a narrative context.
        
        Args:
            context_id: ID of the narrative context
            
        Returns:
            Resolution results
        """
        context = self.narrative_contexts.get(context_id)
        if not context:
            raise ValueError(f"No narrative context found with ID {context_id}")
        
        event_results = context.resolve_events()
        
        # Generate a narrative vector based on resolution results
        vector = self._generate_narrative_vector(context, event_results)
        self.narrative_vectors[context_id] = vector
        
        return {
            "context_id": context_id,
            "title": context.title,
            "events_resolved": len(event_results),
            "events_occurred": sum(1 for e in event_results if e.get("occurred", False)),
            "character_reactions": sum(len(e.get("character_reactions", [])) for e in event_results),
            "parameter_changes": sum(len(e.get("parameter_changes", {})) for e in event_results),
            "narrative_vector": vector
        }
    
    def _generate_narrative_vector(self, context: NarrativeContext, 
                                 event_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Generate a vector representation of the narrative.
        
        Args:
            context: Narrative context
            event_results: Event resolution results
            
        Returns:
            Dictionary mapping dimensions to values
        """
        vector = {
            "complexity": 0.0,
            "coherence": 0.0,
            "novelty": 0.0,
            "emotionality": 0.0,
            "agency": 0.0,
            "determinism": 0.0
        }
        
        # Calculate complexity based on event and character count
        event_count = len([e for e in event_results if e.get("occurred", False)])
        character_count = len(context.characters)
        vector["complexity"] = min(1.0, (event_count * 0.2) + (character_count * 0.1))
        
        # Calculate coherence based on causal connections
        coherence_score = 0.7  # Default medium-high coherence
        vector["coherence"] = coherence_score
        
        # Novelty is partially random but influenced by event categories
        novelty_base = random.random() * 0.5
        novelty_bonus = 0.0
        
        for event in context.events:
            if event.category in [EventCategory.ONTOLOGICAL, EventCategory.EPISTEMOLOGICAL]:
                novelty_bonus += 0.1
        
        vector["novelty"] = min(1.0, novelty_base + novelty_bonus)
        
        # Emotionality based on character reactions
        emotion_intensities = []
        for result in event_results:
            for reaction in result.get("character_reactions", []):
                if "emotional_response" in reaction:
                    for emotion, intensity in reaction["emotional_response"].items():
                        emotion_intensities.append(intensity)
        
        if emotion_intensities:
            vector["emotionality"] = sum(emotion_intensities) / len(emotion_intensities)
        else:
            vector["emotionality"] = 0.5  # Default
        
        # Agency and determinism are complementary
        # Higher complexity generally means higher agency
        vector["agency"] = min(1.0, 0.4 + (vector["complexity"] * 0.5))
        vector["determinism"] = 1.0 - vector["agency"]
        
        return vector
    
    def analyze_narrative_dynamics(self, context_id: str) -> Dict[str, Any]:
        """
        Analyze the dynamics of a narrative.
        
        Args:
            context_id: ID of the narrative context
            
        Returns:
            Analysis results
        """
        context = self.narrative_contexts.get(context_id)
        if not context:
            raise ValueError(f"No narrative context found with ID {context_id}")
            
        vector = self.narrative_vectors.get(context_id)
        if not vector:
            # Generate a vector if none exists
            vector = self._generate_narrative_vector(context, [])
            self.narrative_vectors[context_id] = vector
        
        # Analyze character dynamics
        character_dynamics = []
        for character in context.characters.values():
            dynamics = {
                "character_id": character.id,
                "character_name": character.name,
                "centrality": len(character.relationships) / max(1, len(context.characters) - 1),
                "arc_progression": len(character.arc),
                "motivational_alignment": random.random()  # Placeholder for more sophisticated analysis
            }
            character_dynamics.append(dynamics)
        
        # Analyze event patterns
        event_patterns = {}
        event_categories = [e.category for e in context.events if e.occurred]
        if event_categories:
            category_counts = Counter(event_categories)
            total_events = len(event_categories)
            
            for category, count in category_counts.items():
                event_patterns[category.name] = count / total_events
        
        # Analyze narrative coherence
        coherence_analysis = {
            "overall_coherence": vector["coherence"],
            "causal_density": random.random(),  # Placeholder for more sophisticated analysis
            "thematic_consistency": random.random()  # Placeholder for more sophisticated analysis
        }
        
        return {
            "context_id": context_id,
            "title": context.title,
            "narrative_vector": vector,
            "character_dynamics": character_dynamics,
            "event_patterns": event_patterns,
            "coherence_analysis": coherence_analysis
        }
    
    def generate_conceptual_network(self, framework_id: str) -> Dict[str, Any]:
        """
        Generate a network representation of concepts in a framework.
        
        Args:
            framework_id: ID of the theoretical framework
            
        Returns:
            Network representation
        """
        framework = self.theoretical_frameworks.get(framework_id)
        if not framework:
            raise ValueError(f"No theoretical framework found with ID {framework_id}")
        
        # Generate nodes (concepts)
        nodes = []
        for concept_id, concept in framework.concepts.items():
            nodes.append({
                "id": concept_id,
                "name": concept.name,
                "domain": concept.domain.name,
                "centrality": len(concept.related_concepts) / max(1, len(framework.concepts) - 1)
            })
        
        # Generate edges (relationships between concepts)
        edges = []
        for concept_id, concept in framework.concepts.items():
            for related_id, strength in concept.related_concepts.items():
                if related_id in framework.concepts:
                    edges.append({
                        "source": concept_id,
                        "target": related_id,
                        "strength": strength,
                        "type": "theoretical_relation"
                    })
        
        # Generate network
        network = {
            "framework_id": framework_id,
            "framework_name": framework.name,
            "nodes": nodes,
            "edges": edges,
            "density": len(edges) / max(1, len(nodes) * (len(nodes) - 1))
        }
        
        self.conceptual_networks[framework_id] = network
        return network
    
    def export_narrative(self, context_id: str, format: str = 'json') -> str:
        """
        Export a narrative context to specified format.
        
        Args:
            context_id: ID of narrative context
            format: Export format (json, text, html)
            
        Returns:
            Exported narrative representation
        """
        context = self.narrative_contexts.get(context_id)
        if not context:
            raise ValueError(f"No narrative context found with ID {context_id}")
        
        if format == 'json':
            # Convert dataclasses to dictionaries
            data = {
                "id": context.id,
                "title": context.title,
                "core_premise": context.core_premise,
                "theoretical_lens": context.theoretical_lens,
                "mode": context.mode.name,
                "characters": {cid: asdict(char) for cid, char in context.characters.items()},
                "world_parameters": {pid: asdict(param) for pid, param in context.world_parameters.items()},
                "events": [asdict(event) for event in context.events]
            }
            return json.dumps(data, indent=2)
        
        elif format == 'html':
            # Generate a simple HTML representation
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{context.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2em; }}
        h1, h2, h3 {{ color: #333; }}
        .character {{ margin-bottom: 1em; padding: 0.5em; background: #f8f8f8; }}
        .event {{ margin-bottom: 1em; padding: 0.5em; background: #f0f0f0; }}
        .parameter {{ display: inline-block; margin-right: 1em; padding: 0.3em; background: #e8e8e8; }}
    </style>
</head>
<body>
    <h1>{context.title}</h1>
    <p><em>{context.core_premise}</em></p>
    <p>Narrative Mode: {context.mode.name}</p>
    <p>Theoretical Lens: {context.theoretical_lens or "None"}</p>
    
    <h2>Characters</h2>
    <div class="characters">
"""
            for char in context.characters.values():
                html += f"""
        <div class="character">
            <h3>{char.name}</h3>
            <p><strong>Background:</strong> {char.background}</p>
            <p><strong>Motivations:</strong> {', '.join(char.motivations)}</p>
        </div>
"""
            
            html += """
    </div>
    
    <h2>World Parameters</h2>
    <div class="parameters">
"""
            for param in context.world_parameters.values():
                html += f"""
        <div class="parameter">
            <strong>{param.name}:</strong> {param.value}
        </div>
"""
            
            html += """
    </div>
    
    <h2>Events</h2>
    <div class="events">
"""
            # Sort events by timestamp
            sorted_events = sorted(context.events, key=lambda e: e.timestamp)
            for event in sorted_events:
                occurred = "Occurred" if event.occurred else "Did not occur"
                html += f"""
        <div class="event">
            <p><strong>{event.category.name}:</strong> {event.description}</p>
            <p><strong>Status:</strong> {occurred} (Probability: {event.probability:.2f})</p>
        </div>
"""
            
            html += """
    </div>
</body>
</html>
"""
            return html
        
        # Text representation
        synopsis = context.generate_synopsis()
        
        # Add details about characters
        text = synopsis + "\n\n## Characters\n"
        for char in context.characters.values():
            text += f"\n### {char.name}\n"
            text += f"Background: {char.background}\n"
            text += f"Motivations: {', '.join(char.motivations)}\n"
            if char.traits:
                text += f"Key traits: {', '.join(f'{k}: {v:.2f}' for k, v in char.traits.items())}\n"
        
        # Add details about world parameters
        text += "\n\n## World Parameters\n"
        for param in context.world_parameters.values():
            text += f"\n{param.name}: {param.value}"
            if param.description:
                text += f" - {param.description}"
            text += "\n"
        
        # Add details about events
        text += "\n\n## Events\n"
        context.sort_events_by_timestamp()
        for i, event in enumerate(context.events):
            status = "Occurred" if event.occurred else "Did not occur"
            text += f"\n{i+1}. {event.description}\n"
            text += f"   Category: {event.category.name}\n"
            text += f"   Status: {status} (Probability: {event.probability:.2f})\n"
            
            if event.affected_characters:
                char_names = [context.characters[cid].name for cid in event.affected_characters 
                             if cid in context.characters]
                if char_names:
                    text += f"   Affected characters: {', '.join(char_names)}\n"
        
        return text
    
    def import_narrative(self, data: Dict[str, Any]) -> str:
        """
        Import a narrative from a dictionary.
        
        Args:
            data: Dictionary containing narrative data
            
        Returns:
            ID of the imported narrative context
        """
        title = data.get("title", "Imported Narrative")
        core_premise = data.get("core_premise", "")
        
        # Create the narrative context
        context = NarrativeContext(
            title=title,
            core_premise=core_premise,
            id=data.get("id", str(uuid.uuid4()))
        )
        
        # Set the theoretical lens
        context.theoretical_lens = data.get("theoretical_lens", "")
        
        # Set the mode
        mode_name = data.get("mode", "SPECULATIVE")
        try:
            context.mode = NarrativeMode[mode_name]
        except (KeyError, ValueError):
            context.mode = NarrativeMode.SPECULATIVE
        
        # Import characters
        characters_data = data.get("characters", {})
        for char_id, char_data in characters_data.items():
            character = Character(
                name=char_data.get("name", "Unknown"),
                background=char_data.get("background", ""),
                motivations=char_data.get("motivations", []),
                traits=char_data.get("traits", {}),
                id=char_id
            )
            context.add_character(character)
        
        # Import world parameters
        params_data = data.get("world_parameters", {})
        for param_id, param_data in params_data.items():
            parameter = WorldParameter(
                name=param_data.get("name", "Unknown"),
                value=param_data.get("value", 0.5),
                description=param_data.get("description", ""),
                id=param_id
            )
            context.add_parameter(parameter)
        
        # Import events
        events_data = data.get("events", [])
        for event_data in events_data:
            # Convert category name to enum
            category_name = event_data.get("category", "TECHNOLOGICAL")
            try:
                category = EventCategory[category_name]
            except (KeyError, ValueError):
                category = EventCategory.TECHNOLOGICAL
            
            event = NarrativeEvent(
                description=event_data.get("description", "Unknown event"),
                category=category,
                probability=event_data.get("probability", 0.5),
                timestamp=event_data.get("timestamp", random.random()),
                id=event_data.get("id", str(uuid.uuid4()))
            )
            
            # Import other event data
            event.impact = event_data.get("impact", {})
            event.affected_characters = event_data.get("affected_characters", [])
            event.affected_parameters = event_data.get("affected_parameters", {})
            event.consequences = event_data.get("consequences", [])
            event.occurred = event_data.get("occurred", False)
            
            context.add_event(event)
        
        # Register the context
        self.narrative_contexts[context.id] = context
        
        return context.id


def main():
    """
    Demonstrate the Theory-Fiction Simulation Module's capabilities.
    """
    # Create a simulation instance
    simulation = TheoryFictionSimulation(seed=42)
    
    # Generate a theoretical framework
    framework = simulation.generate_theoretical_framework(
        name="Critical Posthumanism",
        num_concepts=4
    )
    
    # Create a narrative context
    narrative = simulation.create_narrative_context(
        "Transhumanist Convergence", 
        "Exploring human identity in an era of technological integration"
    )
    
    # Generate a speculative scenario
    simulation.generate_speculative_scenario(narrative.id, complexity=5)
    
    # Apply theoretical framework to the narrative
    analysis = simulation.apply_theoretical_framework(narrative.id, framework.id)
    
    # Resolve the narrative
    resolution = simulation.resolve_narrative(narrative.id)
    
    # Export the narrative
    output = simulation.export_narrative(narrative.id, format='text')
    print(output)
    
    # Print some analysis information
    print("\n--- Analysis Results ---")
    print(f"Overall themes: {', '.join(analysis['overall_themes'])}")
    print(f"Events occurred: {resolution['events_occurred']} of {resolution['events_resolved']}")
    print("Narrative vector:", resolution['narrative_vector'])


if __name__ == "__main__":
    main()
