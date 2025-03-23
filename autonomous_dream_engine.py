# Re-initialize code after kernel reset

from typing import List, Dict, Any
import random
import time
import numpy as np

class AutonomousDreamingEngine:
    def __init__(self):
        self.dream_log: List[Dict[str, Any]] = []
        self.symbolic_templates = [
            "In a forest of glass trees, {emotion} echoed with every footstep.",
            "The stars whispered of {archetype}'s forgotten secrets.",
            "She followed a silver thread through the fog of {emotion}, where memories became shadows.",
            "A mirror floated above the sea, reflecting {zone_theme} and a sky made of thoughts.",
            "Every door opened into a different version of {archetype}, each more unreal than the last."
        ]
        self.zone_themes = {
            1: "beginnings and sparks",
            2: "reflections and duality",
            3: "patterns and structures",
            4: "creativity and beauty",
            5: "balance and empathy",
            6: "transformation and shadows",
            7: "exploration and chaos",
            8: "mysticism and fate",
            9: "enlightenment and silence"
        }

    def generate_dream(self,
                       emotional_tone: str,
                       dominant_archetype: str,
                       zone: int,
                       memory_fragments: List[str]) -> Dict[str, Any]:
        """Generate a symbolic dream narrative."""
        timestamp = int(time.time())
        zone_theme = self.zone_themes.get(zone, "unknown realms")

        chosen_template = random.choice(self.symbolic_templates)
        base_dream = chosen_template.format(
            emotion=emotional_tone,
            archetype=dominant_archetype,
            zone_theme=zone_theme
        )

        dream_fragments = random.sample(memory_fragments, min(2, len(memory_fragments)))
        surreal_links = [
            f"A whisper from the past said: '{frag}'" for frag in dream_fragments
        ]

        full_dream = f"{base_dream} " + " ".join(surreal_links)

        dream = {
            "timestamp": timestamp,
            "zone": zone,
            "archetype": dominant_archetype,
            "emotional_tone": emotional_tone,
            "narrative": full_dream,
            "fragments_used": dream_fragments
        }

        self.dream_log.append(dream)
        return dream

    def get_recent_dreams(self, count: int = 3) -> List[Dict[str, Any]]:
        """Retrieve recent dream records."""
        return self.dream_log[-count:]

# Expose class for further integration
dream_engine = AutonomousDreamingEngine()
dream_example = dream_engine.generate_dream(
    emotional_tone="curiosity",
    dominant_archetype="The Oracle",
    zone=8,
    memory_fragments=[
        "the sound of rain on forgotten streets",
        "a broken compass pointing inward",
        "the scent of ash and blooming lilies"
    ]
)
dream_example
