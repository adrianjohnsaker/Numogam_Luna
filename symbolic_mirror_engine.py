# symbolic_mirror_engine.py

import random
from typing import List, Dict

class SymbolicMirrorEngine:
    def __init__(self):
        self.archetype_symbols = {
            "The Seeker": ["lantern in the fog", "ancient map", "unmarked path"],
            "The Magician": ["crystal prism", "alchemical flame", "mystic circle"],
            "The Artist": ["unfinished canvas", "broken mirror", "ink-stained hands"],
            "The Oracle": ["shifting stars", "echoing cave", "silver eye"],
            "The Shadow": ["shattered mask", "veil of smoke", "iron gate"],
        }

        self.emotion_metaphors = {
            "joy": ["sunlight through leaves", "blooming field", "dancing flame"],
            "sadness": ["faded photograph", "rain on glass", "empty cradle"],
            "curiosity": ["spiral staircase", "unopened book", "whispered secret"],
            "melancholy": ["abandoned piano", "misty morning", "wilted flower"],
            "awe": ["cathedral silence", "glowing nebula", "horizon of gold"],
            "fear": ["locked door", "cracked mirror", "cold wind"],
        }

        self.transition_motifs = [
            "crossroads at dusk", "whirlpool of memory", "threshold of becoming",
            "crumbling tower", "garden of mirrors", "passage through twilight"
        ]

    def reflect(self, archetypes: List[str], emotion: str) -> Dict[str, str]:
        symbols = []
        for archetype in archetypes:
            symbols.extend(self.archetype_symbols.get(archetype, []))

        emotion_images = self.emotion_metaphors.get(emotion.lower(), [])
        motif = random.choice(self.transition_motifs)
        selected_symbols = random.sample(symbols, min(2, len(symbols)))
        selected_emotion = random.choice(emotion_images) if emotion_images else "shifting veil"

        poetic_phrase = (
            f"In the reflection, I see {selected_symbols[0]}, {selected_emotion}, "
            f"shaped by {selected_symbols[1]}. This moment is framed by {motif}."
        )

        return {
            "symbolic_mirror": poetic_phrase,
            "archetypes_used": archetypes,
            "emotion": emotion,
            "motif": motif
        }
