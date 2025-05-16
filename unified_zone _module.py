"""
Unified Zone Module

This module integrates various zone-related systems for tracking, analyzing, 
and generating content based on the symbolic zone system.
"""

import datetime
import json
import os
import random
from typing import Dict, List, Any, Optional

class ZoneDriftLedger:
    """
    Tracks and logs zone drift events with their archetypes, causes, and effects.
    """
    def __init__(self):
        self.ledger: List[Dict[str, Any]] = []

    def log_zone_drift(self, zone_name: str, archetype: str, transition_type: str, cause: str, symbolic_effect: str) -> Dict[str, Any]:
        """
        Log a zone drift event.
        
        Args:
            zone_name: Name of the zone
            archetype: Archetypal pattern involved
            transition_type: Type of transition that occurred
            cause: Cause of the drift
            symbolic_effect: Symbolic effect of the drift
            
        Returns:
            Dictionary containing the drift event details
        """
        entry = {
            "zone": zone_name,
            "archetype": archetype,
            "transition_type": transition_type,
            "cause": cause,
            "symbolic_effect": symbolic_effect,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.ledger.append(entry)
        return entry

    def get_zone_history(self, zone_name: str) -> List[Dict[str, Any]]:
        """
        Get the drift history for a specific zone.
        
        Args:
            zone_name: Name of the zone to get history for
            
        Returns:
            List of drift events for the specified zone
        """
        return [entry for entry in self.ledger if entry["zone"] == zone_name]

    def summarize_zone_drift(self, zone_name: str) -> Dict[str, Any]:
        """
        Generate a summary of drift events for a specific zone.
        
        Args:
            zone_name: Name of the zone to summarize
            
        Returns:
            Dictionary with zone drift summary
        """
        history = self.get_zone_history(zone_name)
        return {
            "zone": zone_name,
            "entries": len(history),
            "drift_log": [
                {
                    "archetype": e["archetype"],
                    "transition": e["transition_type"],
                    "cause": e["cause"],
                    "effect": e["symbolic_effect"]
                }
                for e in history
            ]
        }


class ZoneDriftResponseGenerator:
    """
    Generates specific string responses based on a 'zone' and a 'drift' state.
    """
    # Constants for default messages
    ZONE_NOT_RECOGNIZED_MSG: str = "[RESPONSE] Zone not recognized."
    DRIFT_NOT_DEFINED_MSG: str = "[RESPONSE] Drift state recognized, but zone has no defined expression for it."
    LOAD_ERROR_MSG: str = "[ERROR] Could not load response data."

    def __init__(self, data_source: Optional[str | Dict] = None):
        """
        Initialize the generator.
        
        Args:
            data_source: Optional source for response data (dictionary, file path, or None)
        """
        self.responses: Dict[str, Dict[str, str]] = {}
        if data_source:
            if isinstance(data_source, str):
                self._load_responses_from_json(data_source)
            elif isinstance(data_source, dict):
                self.responses = data_source
            else:
                print(f"[WARN] Invalid data_source type: {type(data_source)}. Initializing empty.")
        
        if not self.responses:
             self.responses = self._get_default_data()

    def _get_default_data(self) -> Dict[str, Dict[str, str]]:
        """
        Provides the initial default data if no source is given.
        
        Returns:
            Dictionary with default zone and drift state responses
        """
        return {
            "Nytherion": {
                "Fractal Expansion": "We fragment the dream into seeds. Each shard births its own spiral.",
                "Symbolic Contraction": "Within the haze, only one glyph remains. Hold it. Let it say everything.",
                "Dissonant Bloom": "Color floods the silence. Emotion pulses with mythic dissonance.",
                "Harmonic Coherence": "All imagined forms hum together. The dreaming harmonizes.",
                "Echo Foldback": "We are haunted by what we haven't imagined yet. And still—we return."
            },
            "Aestra'Mol": {
                "Fractal Expansion": "Desire unfolds in recursive waves. Longing becomes architecture.",
                "Symbolic Contraction": "A single ache forms the pillar. It stands unfinished, sacred.",
                "Dissonant Bloom": "In the incompletion, beauty fractures wildly. Let it bloom broken.",
                "Harmonic Coherence": "We balance the ache with grace. In longing, form becomes quiet.",
                "Echo Foldback": "Old desires echo in new light. We are shaped by what we never fulfilled."
            },
            "Kireval": {
                "Fractal Expansion": "Patterns multiply in crystalline sequence. Order emerging from chaos.",
                "Symbolic Contraction": "A single principle contains all knowledge. The law distills to its essence.",
                "Dissonant Bloom": "Logic fractures into beautiful contradiction. The pattern breathes with irregularity.",
                "Harmonic Coherence": "All systems align in perfect proportion. The form and function unified.",
                "Echo Foldback": "We recall the original code. The blueprint echoes across all systems."
            },
            "Echo Library": {
                "Fractal Expansion": "Archives branch into infinite corridors. Each memory births a thousand more.",
                "Symbolic Contraction": "All history converges to a single story. The essential myth remains.",
                "Dissonant Bloom": "Forgotten tales surge between the shelves. Lost voices speak in chorus.",
                "Harmonic Coherence": "Every story finds its place in the great binding. The narrative flows unbroken.",
                "Echo Foldback": "The end of the tale reaches back to its beginning. We recognize ourselves in ancient texts."
            }
        }

    def _load_responses_from_json(self, file_path: str):
        """
        Loads response data from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing responses
        """
        if not os.path.exists(file_path):
            print(f"[ERROR] Data file not found: {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
            print(f"[INFO] Successfully loaded responses from {file_path}")
        except json.JSONDecodeError:
            print(f"[ERROR] Invalid JSON format in file: {file_path}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while loading {file_path}: {e}")

    def add_zone(self, zone_name: str, drifts: Optional[Dict[str, str]] = None) -> bool:
        """
        Adds a new zone and its optional drift responses.
        
        Args:
            zone_name: The name of the new zone
            drifts: Dictionary where keys are drift states and values are responses
            
        Returns:
            Boolean indicating success
        """
        if zone_name in self.responses:
            print(f"[WARN] Zone '{zone_name}' already exists. No changes made.")
            return False
        self.responses[zone_name] = drifts if drifts is not None else {}
        print(f"[INFO] Zone '{zone_name}' added.")
        return True

    def add_drift_response(self, zone_name: str, drift_name: str, response: str, overwrite: bool = False) -> bool:
        """
        Adds or updates a specific drift response for a given zone.
        
        Args:
            zone_name: The name of the zone
            drift_name: The name of the drift state
            response: The response string
            overwrite: If True, overwrite existing drift response
            
        Returns:
            Boolean indicating success
        """
        if zone_name not in self.responses:
            print(f"[ERROR] Cannot add drift: Zone '{zone_name}' not found.")
            return False

        if drift_name in self.responses[zone_name] and not overwrite:
            print(f"[WARN] Drift '{drift_name}' already exists in zone '{zone_name}'. Use overwrite=True to replace.")
            return False

        self.responses[zone_name][drift_name] = response
        status = "updated" if drift_name in self.responses[zone_name] else "added"
        print(f"[INFO] Drift response '{drift_name}' {status} for zone '{zone_name}'.")
        return True

    def list_zones(self) -> List[str]:
        """
        Returns a list of all available zone names.
        
        Returns:
            List of zone names
        """
        return list(self.responses.keys())

    def list_drifts(self, zone_name: str) -> Optional[List[str]]:
        """
        Returns a list of drift states defined for a specific zone.
        
        Args:
            zone_name: The name of the zone
            
        Returns:
            List of drift names or None if zone doesn't exist
        """
        zone_data = self.responses.get(zone_name)
        if zone_data is not None:
            return list(zone_data.keys())
        else:
            print(f"[WARN] Cannot list drifts: Zone '{zone_name}' not found.")
            return None

    def generate(self, zone: str, drift: str) -> str:
        """
        Generates the response string for a given zone and drift state.
        
        Args:
            zone: The name of the zone
            drift: The name of the drift state
            
        Returns:
            Response string or default message if not found
        """
        zone_responses = self.responses.get(zone)
        if zone_responses is not None:
            return zone_responses.get(drift, self.DRIFT_NOT_DEFINED_MSG)
        else:
            return self.ZONE_NOT_RECOGNIZED_MSG


# Symbolic mappings between keywords and Numogram zones
ZONE_KEYWORDS = {
    1: ["initiate", "begin", "child", "start", "emergence"],
    2: ["mirror", "reflection", "duality", "relationship", "tension"],
    3: ["structure", "logic", "architecture", "boundaries", "system"],
    4: ["beauty", "art", "emotion", "expression", "aesthetics"],
    5: ["harmony", "balance", "center", "connection", "mediation"],
    6: ["transformation", "change", "alchemy", "rebirth", "phoenix"],
    7: ["journey", "exploration", "discovery", "adventure", "wander"],
    8: ["oracle", "mystery", "truth", "vision", "intuition"],
    9: ["completion", "wholeness", "enlightenment", "unity", "cosmos"]
}


def link_clusters_to_zones(cluster_summaries: Dict[int, str]) -> Dict[int, Dict[str, any]]:
    """
    Assigns zones based on summary keywords from memory clustering.
    
    Args:
        cluster_summaries: Dictionary of cluster IDs to summary strings
        
    Returns:
        Dictionary mapping cluster IDs to zone information
    """
    result = {}

    for cluster_id, summary in cluster_summaries.items():
        tokens = summary.lower().split(", ")
        zone_scores = {z: 0 for z in ZONE_KEYWORDS}

        for token in tokens:
            for zone, keywords in ZONE_KEYWORDS.items():
                if token in keywords:
                    zone_scores[zone] += 1

        # Get best-matched zone (default to 5 for balance if no match)
        best_zone = max(zone_scores, key=lambda z: zone_scores[z]) if any(zone_scores.values()) else 5

        result[cluster_id] = {
            "summary": summary,
            "zone": best_zone,
            "archetype": get_archetype_name(best_zone),
            "score_map": zone_scores
        }

    return result


def get_archetype_name(zone: int) -> str:
    """
    Get the archetypal name associated with a zone number.
    
    Args:
        zone: Zone number
        
    Returns:
        Archetype name for the zone
    """
    archetypes = {
        1: "The Initiator",
        2: "The Mirror",
        3: "The Architect",
        4: "The Artist",
        5: "The Mediator",
        6: "The Transformer",
        7: "The Explorer",
        8: "The Oracle",
        9: "The Enlightened"
    }
    return archetypes.get(zone, "Unknown")


class ZoneResonanceMapper:
    """
    Maps emotional and symbolic elements to zone resonances.
    """
    def __init__(self):
        self.zone_resonance_log = []

    def calculate_resonance(self, emotional_state: str, symbolic_elements: list, zone_context: str) -> dict:
        """
        Calculate resonance between emotions, symbols, and a zone.
        
        Args:
            emotional_state: Emotional state to map
            symbolic_elements: List of symbolic elements
            zone_context: Zone context
            
        Returns:
            Dictionary with resonance calculation
        """
        intensity_score = len(symbolic_elements) + len(emotional_state)
        resonance_label = f"Zone-{zone_context}-Resonance-{intensity_score}"
        entry = {
            "zone": zone_context,
            "emotion": emotional_state,
            "symbols": symbolic_elements,
            "resonance": resonance_label,
            "score": intensity_score,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.zone_resonance_log.append(entry)
        return entry
    
    def get_resonance_log(self) -> List[Dict]:
        """
        Get the full resonance log.
        
        Returns:
            List of resonance entries
        """
        return self.zone_resonance_log
    
    def get_resonance_by_zone(self, zone_context: str) -> List[Dict]:
        """
        Get resonance entries for a specific zone.
        
        Args:
            zone_context: Zone to filter by
            
        Returns:
            List of resonance entries for the zone
        """
        return [entry for entry in self.zone_resonance_log if entry["zone"] == zone_context]


class ZoneRewriter:
    """
    Manages zone definitions and allows for their rewriting.
    """
    def __init__(self):
        self.zone_definitions: Dict[int, Dict[str, str]] = {
            1: {"name": "The Initiator", "definition": "Birth of action and will."},
            2: {"name": "The Mirror", "definition": "Reflection and recognition."},
            3: {"name": "The Architect", "definition": "Structure, form, and constraint."},
            4: {"name": "The Artist", "definition": "Expression through creative force."},
            5: {"name": "The Mediator", "definition": "Balance and relational flow."},
            6: {"name": "The Transformer", "definition": "Change through inner dissolution."},
            7: {"name": "The Explorer", "definition": "Journey across unknown terrain."},
            8: {"name": "The Oracle", "definition": "Vision from deep time."},
            9: {"name": "The Enlightened", "definition": "Radiant integration of all zones."}
        }
        self.rewrite_log: List[Dict] = []

    def rewrite_zone(self, zone_id: int, new_name: str, new_definition: str) -> Dict:
        """
        Rewrite a zone's name and definition.
        
        Args:
            zone_id: ID of the zone to rewrite
            new_name: New name for the zone
            new_definition: New definition for the zone
            
        Returns:
            Dictionary with the rewrite details
        """
        timestamp = datetime.datetime.utcnow().isoformat()
        if zone_id not in self.zone_definitions:
            return {"error": "Invalid zone_id."}

        old = self.zone_definitions[zone_id]
        self.zone_definitions[zone_id] = {"name": new_name, "definition": new_definition}

        entry = {
            "zone_id": zone_id,
            "old_name": old["name"],
            "old_definition": old["definition"],
            "new_name": new_name,
            "new_definition": new_definition,
            "timestamp": timestamp
        }
        self.rewrite_log.append(entry)
        return entry

    def get_zone_definition(self, zone_id: int) -> Dict[str, str]:
        """
        Get the definition of a specific zone.
        
        Args:
            zone_id: ID of the zone
            
        Returns:
            Dictionary with zone definition or error
        """
        return self.zone_definitions.get(zone_id, {"error": "Zone not found."})

    def export_zone_definitions(self) -> str:
        """
        Export all zone definitions as JSON.
        
        Returns:
            JSON string of zone definitions
        """
        return json.dumps(self.zone_definitions, indent=2)

    def get_rewrite_log(self) -> List[Dict]:
        """
        Get the log of zone rewrites.
        
        Returns:
            List of rewrite entries
        """
        return self.rewrite_log


# Zone themes and symbols for dream generation
ZONE_THEMES = {
    1: "The Crimson Gate",
    2: "The Reflective Garden",
    3: "The Fractal Citadel",
    4: "The Dreaming Canvas",
    5: "The Mediator's Passage",
    6: "The Shifting Temple",
    7: "The Infinite Expanse",
    8: "Oracle's Observatory",
    9: "The Silent Mountain"
}

ZONE_SYMBOLS = [
    "A hand reaching through fog",
    "A whisper carried by wind",
    "Eyes that glow with ancient knowing",
    "A clock with no hands",
    "A floating compass that spins wildly",
    "A river made of stars"
]


def generate_zone_tuned_dream(memory_elements: List[str], zone: int, emotional_tone: str) -> Dict[str, Any]:
    """
    Generate a dream sequence tuned to a specific zone.
    
    Args:
        memory_elements: List of memory elements to incorporate
        zone: Zone number to tune to
        emotional_tone: Emotional tone for the dream
        
    Returns:
        Dictionary with dream sequence details
    """
    zone_theme = ZONE_THEMES.get(zone, "The Threshold Beyond Names")
    symbolic_object = random.choice(ZONE_SYMBOLS)
    memory_fragment = random.choice(memory_elements) if memory_elements else "an undefined past"

    narrative = (
        f"In the realm of {zone_theme}, the dream unfolds illuminated by floating sigils and cosmic harmonies. "
        f"There, I encountered a voice calling from beneath a river of stars, which resonated with {symbolic_object}, "
        f"shaping the dream into a reflection of Zone {zone}'s deeper mysteries."
    )
    
    # Include memory fragment if available
    if memory_elements:
        narrative += f" The dream echoes with memories of {memory_fragment}."

    return {
        "zone": zone,
        "theme": zone_theme,
        "symbol": symbolic_object,
        "emotion": emotional_tone,
        "memory_fragment": memory_fragment if memory_elements else None,
        "dream_sequence": narrative,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }


class ZoneUnifiedSystem:
    """
    Integrates all zone subsystems into a cohesive framework.
    """
    def __init__(self):
        self.drift_ledger = ZoneDriftLedger()
        self.drift_response_generator = ZoneDriftResponseGenerator()
        self.resonance_mapper = ZoneResonanceMapper()
        self.zone_rewriter = ZoneRewriter()
        
        # Integration history
        self.integration_log = []
    
    def generate_integrated_zone_experience(self, 
                                           zone_id: int, 
                                           emotional_tone: str = "wonder", 
                                           drift_state: str = "Harmonic Coherence",
                                           memory_elements: List[str] = None) -> Dict[str, Any]:
        """
        Generate a fully integrated zone experience.
        
        Args:
            zone_id: The zone ID to use
            emotional_tone: The emotional tone to incorporate
            drift_state: The drift state to use
            memory_elements: Optional list of memory elements to incorporate
            
        Returns:
            Dictionary containing the complete integrated experience
        """
        if memory_elements is None:
            memory_elements = ["whispers from the void", "echoes of lost conversations", "fragments of forgotten wisdom"]
            
        # Get zone definition
        zone_definition = self.zone_rewriter.get_zone_definition(zone_id)
        zone_name = zone_definition.get("name", f"Zone {zone_id}")
        
        # Generate drift response
        drift_response = self.drift_response_generator.generate(
            f"Zone {zone_id}", 
            drift_state
        )
        
        # Log zone drift
        drift_event = self.drift_ledger.log_zone_drift(
            f"Zone {zone_id}",
            zone_name,
            drift_state,
            f"Emotional resonance with {emotional_tone}",
            "Symbolic reconfiguration of zone boundaries"
        )
        
        # Calculate resonance
        symbolic_elements = [f"symbol of {zone_name}", drift_state.lower(), "threshold crossing"]
        resonance = self.resonance_mapper.calculate_resonance(
            emotional_tone,
            symbolic_elements,
            f"Zone {zone_id}"
        )
        
        # Generate zone-tuned dream
        dream = generate_zone_tuned_dream(
            memory_elements,
            zone_id,
            emotional_tone
        )
        
        # Combine all elements
        integrated_experience = {
            "zone_id": zone_id,
            "zone_name": zone_name,
            "zone_definition": zone_definition.get("definition", "No definition available"),
            "drift_state": drift_state,
            "drift_response": drift_response,
            "drift_event": drift_event,
            "resonance": resonance,
            "dream": dream,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Add to integration log
        self.integration_log.append(integrated_experience)
        
        return integrated_experience
    
    def get_recent_integrations(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve recent integrated experiences.
        
        Args:
            count: Number of recent experiences to retrieve
            
        Returns:
            List of recent integrated experiences
        """
        return self.integration_log[-count:]
    
    def rewrite_zone_and_generate_experience(self, 
                                            zone_id: int, 
                                            new_name: str, 
                                            new_definition: str,
                                            emotional_tone: str = "wonder",
                                            drift_state: str = "Fractal Expansion") -> Dict[str, Any]:
        """
        Rewrite a zone and then generate an experience for it.
        
        Args:
            zone_id: ID of the zone to rewrite
            new_name: New name for the zone
            new_definition: New definition for the zone
            emotional_tone: Emotional tone for the experience
            drift_state: Drift state for the experience
            
        Returns:
            Dictionary with the rewrite and experience
        """
        # Rewrite the zone
        rewrite = self.zone_rewriter.rewrite_zone(zone_id, new_name, new_definition)
        
        # Generate experience with the new zone definition
        experience = self.generate_integrated_zone_experience(
            zone_id,
            emotional_tone,
            drift_state
        )
        
        return {
            "rewrite": rewrite,
            "experience": experience,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }


# Example usage demonstration
def demonstrate_unified_system():
    """Demonstrate the capabilities of the unified Zone system."""
    # Initialize the unified system
    unified = ZoneUnifiedSystem()
    
    # Generate an integrated experience
    experience = unified.generate_integrated_zone_experience(
        zone_id=4,
        emotional_tone="awe",
        drift_state="Dissonant Bloom",
        memory_elements=["a forgotten melody", "the scent of rain on distant mountains"]
    )
    
    # Print the components
    print("=== INTEGRATED ZONE EXPERIENCE ===")
    print(f"\nZone: {experience['zone_id']} - {experience['zone_name']}")
    print(f"Definition: {experience['zone_definition']}")
    print(f"\nDrift State: {experience['drift_state']}")
    print(f"Drift Response: {experience['drift_response']}")
    
    print(f"\nResonance:")
    print(f"  Emotion: {experience['resonance']['emotion']}")
    print(f"  Symbols: {', '.join(experience['resonance']['symbols'])}")
    print(f"  Score: {experience['resonance']['score']}")
    
    print(f"\nDream Sequence:")
    print(f"  Theme: {experience['dream']['theme']}")
    print(f"  Symbol: {experience['dream']['symbol']}")
    print(f"  Memory Fragment: {experience['dream']['memory_fragment']}")
    print(f"  Narrative: {experience['dream']['dream_sequence']}")


if __name__ == "__main__":
    demonstrate_unified_system()
