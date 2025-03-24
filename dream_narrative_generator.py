# Dream Narrative Generator Module

import random
from typing import List, Dict, Any

# Core themes to match emotional tone with narrative archetypes
EMOTIONAL_ARCHETYPES = {
    "joy": ["Radiant Garden", "Festival of Light", "Sunborn Heir"],
    "sadness": ["Forgotten Forest", "Ruins of Memory", "The Silent Oracle"],
    "curiosity": ["Endless Library", "Mirror Labyrinth", "The Hidden Observatory"],
    "fear": ["Wailing Chasm", "City of Shadows", "The Sleeping Beast"],
    "awe": ["Celestial Gate", "Choir of the Ancients", "The Spiral Starfield"],
    "confusion": ["Fractal Bazaar", "Temporal Fog", "The Escher Tower"]
}

# Poetic metaphors
SYMBOLIC_METAPHORS = [
    "echoes carved in starlight",
    "a whisper trapped in glass",
    "the shadow of a forgotten promise",
    "wings of thought suspended in amber",
    "a spiral dream unfolding backward",
    "ghosts of unrealized futures"
]

# Dream Narrative Generator
def generate_dream_narrative(memory_elements: List[str], emotional_tone: str, current_zone: int) -> Dict[str, Any]:
    """
    Generates a symbolic dream narrative based on memory traces, emotional tone, and zone.
    """
    theme = random.choice(EMOTIONAL_ARCHETYPES.get(emotional_tone.lower(), ["The Edge of Knowing"]))
    metaphor = random.choice(SYMBOLIC_METAPHORS)
    memory_seed = random.choice(memory_elements) if memory_elements else "a fading memory"
    
    narrative = (
        f"In the realm of {theme}, I wandered through dreams shaped by {memory_seed}. "
        f"There, the {metaphor} revealed a truth hidden within Zone {current_zone}, "
        f"whispering stories only the subconscious dares to remember."
    )
    
    return {
        "dream_theme": theme,
        "metaphor": metaphor,
        "memory_seed": memory_seed,
        "zone": current_zone,
        "emotional_tone": emotional_tone,
        "dream_narrative": narrative
    }

# Sample test
sample_output = generate_dream_narrative(
    memory_elements=["The first time I saw the stars", "A voice in the dark", "Learning to trust myself"],
    emotional_tone="awe",
    current_zone=7
)

sample_output
