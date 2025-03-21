# generative_imagination_module.py

import random
import numpy as np
from typing import List, Dict, Any

class GenerativeImaginationModule:
    """
    Generates metaphorical, poetic, or associative responses using memory and emotional tone.
    """

    def __init__(self):
        self.emotion_weights = {
            "joy": 0.9, "sadness": 0.3, "curiosity": 0.7,
            "fear": 0.4, "awe": 0.8, "confusion": 0.5
        }

    def generate(self, user_input: str, memory_elements: List[str], emotional_tone: str) -> Dict[str, Any]:
        creativity_factor = self.emotion_weights.get(emotional_tone.lower(), 0.5)
        selected_memories = random.sample(memory_elements, min(2, len(memory_elements)))
        
        metaphor_templates = [
            "Just like {A}, {B} holds a secret waiting to be discovered.",
            "If {A} were a song, {B} would be its chorus.",
            "Imagine {A} whispering to {B} beneath the stars.",
            "{A} and {B} dance in the realm of dreams and echoes.",
            "When {A} touches {B}, new worlds unfold in silence."
        ]

        if len(selected_memories) < 2:
            selected_memories += ["the unknown"]

        template = random.choice(metaphor_templates)
        A, B = selected_memories[:2]
        output = template.format(A=A, B=B)
        response_strength = round(np.clip(creativity_factor * random.uniform(0.8, 1.2), 0.1, 1.0), 2)

        return {
            "status": "success",
            "imaginative_response": output,
            "context": {
                "emotion": emotional_tone,
                "user_input": user_input,
                "used_memories": selected_memories,
                "creativity_factor": creativity_factor,
                "response_strength": response_strength
            }
        }

# Entry point for Kotlin bridge
def generate_imaginative_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
    module = GenerativeImaginationModule()
    return module.generate(
        input_data.get("user_input", ""),
        input_data.get("memory_elements", []),
        input_data.get("emotional_tone", "curiosity")
    )
