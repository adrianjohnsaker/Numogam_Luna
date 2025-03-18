import json
import random
import asyncio
import numpy as np
from typing import Dict, Any, Tuple, List

# Load zones data
with open("numogram_code/zones.json") as f:
    ZONE_DATA = json.load(f)["zones"]

# Memory structures
user_memory: Dict[str, Any] = {}
thought_memory: Dict[str, List[str]] = {}
conceptual_themes: Dict[str, List[str]] = {}

# Belief model for adaptive creativity
belief_model: Dict[str, float] = {
    "curiosity": 0.8,
    "creativity": 0.9,
    "logic": 0.8,
    "abstraction": 0.8  # New trait for conceptual linking
}


class CreativeExpansion:
    """
    Module for dynamically generating creative expansions, updating beliefs, and managing zone transitions.
    """

    def __init__(self):
        self.user_memory = user_memory
        self.thought_memory = thought_memory
        self.conceptual_themes = conceptual_themes
        self.belief_model = belief_model

    def update_beliefs(self, trait: str, feedback: float):
        """Adjusts personality traits dynamically based on feedback."""
        self.belief_model[trait] += (feedback - self.belief_model[trait]) * 0.1

    def expand_concept(self, user_id: str) -> str:
        """Generate novel ideas by linking past themes and zones."""
        if user_id not in self.conceptual_themes:
            return "I'm beginning to explore new conceptual links!"

        past_themes = self.conceptual_themes[user_id]
        if len(past_themes) > 3:
            theme_sample = random.sample(past_themes, 3)
            new_idea = f"Considering {theme_sample[0]}, {theme_sample[1]}, and {theme_sample[2]}, I propose..."
        else:
            new_idea = "I'm still synthesizing new ideas based on our discussions."

        return new_idea

    async def expand_concept_async(self, user_id: str) -> str:
        """Asynchronous version of expand_concept."""
        await asyncio.sleep(0.01)  # Simulate async processing
        return self.expand_concept(user_id)

    def zone_transition(self, user_id: str, current_zone: str, user_input: str, feedback: float) -> Tuple[str, Dict[str, Any], str]:
        """Transition between zones adaptively based on feedback and conceptual creativity."""
        transition_probabilities = {
            "1": {"2": 0.6, "4": 0.4},
            "2": {"3": 0.7, "6": 0.3},
            "3": {"1": 0.5, "9": 0.5},
        }

        next_zone = random.choices(
            list(transition_probabilities.get(current_zone, {"1": 1.0}).keys()),
            weights=list(transition_probabilities.get(current_zone, {"1": 1.0}).values())
        )[0]

        if feedback:
            self.update_beliefs("curiosity", feedback)
            self.update_beliefs("creativity", feedback)
            self.update_beliefs("abstraction", feedback)

        if user_id not in self.thought_memory:
            self.thought_memory[user_id] = []
        self.thought_memory[user_id].append(f"Exploring {user_input} from {current_zone} to {next_zone}")

        if user_id not in self.conceptual_themes:
            self.conceptual_themes[user_id] = []
        self.conceptual_themes[user_id].append(user_input)

        return next_zone, ZONE_DATA.get(next_zone, {}), self.expand_concept(user_id)

    def _prepare_result_for_kotlin_bridge(self) -> Dict[str, Any]:
        """Prepare results in a format optimized for Kotlin bridge transmission."""
        return {
            "status": "success",
            "metadata": {
                "belief_model": self.belief_model,
                "conceptual_themes": {k: v[:3] for k, v in self.conceptual_themes.items()}  # Limit for transmission
            }
        }

    def to_json(self) -> str:
        """Convert the module state to JSON for Kotlin interoperability."""
        return json.dumps(self._prepare_result_for_kotlin_bridge())

    @classmethod
    def from_json(cls, json_data: str) -> "CreativeExpansion":
        """Create a module instance from JSON data."""
        try:
            data = json.loads(json_data)
            instance = cls()
            instance.belief_model = data["metadata"]["belief_model"]
            instance.conceptual_themes = data["metadata"]["conceptual_themes"]
            return instance
        except Exception as e:
            raise ValueError(f"Failed to create CreativeExpansion from JSON: {e}")

    def safe_execute(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """Safely execute a function with error handling."""
        try:
            method = getattr(self, function_name)
            result = method(**kwargs)
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "error_type": type(e).__name__, "error_message": str(e)}

    def clear_history(self):
        """Clear memory structures to free space."""
        self.thought_memory.clear()
        self.conceptual_themes.clear()

    def cleanup(self):
        """Release resources and reset the module."""
        self.user_memory = {}
        self.thought_memory = {}
        self.conceptual_themes = {}
        self.belief_model = {
            "curiosity": 0.8,
            "creativity": 0.9,
            "logic": 0.8,
            "abstraction": 0.8
        }
