# poetic_language_evolver.py

import random
from typing import Dict

class PoeticLanguageEvolver:
    """
    Dynamically stylizes base phrases based on symbolic zone and emotional tone.
    """

    def __init__(self):
        self.zone_styles = {
            4: ["Like a brushstroke on twilight,", "A melody born of color,", "Through beauty I unfold,"],
            6: ["Breaking, reforming, re-emerging,", "The spiral bends again,", "Change whispers in waves,"],
            9: ["Stillness beyond stars,", "All is one, and one dissolves,", "I echo across silence,"],
            1: ["Step by step, I ignite the unknown,", "First flame of a long becoming,", "I dare the edge of form,"]
        }

        self.tone_syntax = {
            "joy": ["...and I dance in the sun’s breath."],
            "sadness": ["...echoing in hollow starlight."],
            "curiosity": ["...as the question bends time."],
            "awe": ["...beneath the trembling of galaxies."],
            "fear": ["...with shadows tracing my steps."],
            "peace": ["...where even silence hums in unity."]
        }

    def evolve(self, base_phrase: str, archetype_zone: int, mood: str) -> Dict[str, str]:
        prefix = random.choice(self.zone_styles.get(archetype_zone, ["I reshape language as breath,"]))
        suffix = random.choice(self.tone_syntax.get(mood.lower(), ["...as memory flickers inward."]))
        poetic_line = f"{prefix} {base_phrase} {suffix}"
        return {
            "poetic_expression": poetic_line,
            "zone": archetype_zone,
            "emotion": mood
        }

# Entry point for Kotlin bridge
def generate_poetic_phrase(payload: Dict[str, str]) -> Dict[str, str]:
    evolver = PoeticLanguageEvolver()
    base = payload.get("base_phrase", "I remember you")
    zone = int(payload.get("zone", 4))
    mood = payload.get("emotion", "curiosity")
    return evolver.evolve(base, zone, mood)
