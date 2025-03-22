# unified_coordinator_module.py

import random
import numpy as np
from typing import Dict, Any, List
from memory_clusterer import MemoryClusterer
from zone_linker import link_clusters_to_zones
from hybrid_archetype_generator import generate_hybrid_archetype

class GenerativeImaginationModule:
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
        A, B = selected_memories[:2]
        output = random.choice(metaphor_templates).format(A=A, B=B)
        response_strength = round(np.clip(creativity_factor * random.uniform(0.8, 1.2), 0.1, 1.0), 2)
        return {
            "imaginative_response": output,
            "context": {
                "emotion": emotional_tone,
                "user_input": user_input,
                "used_memories": selected_memories,
                "creativity_factor": creativity_factor,
                "response_strength": response_strength
            }
        }


class DynamicZoneNavigator:
    def __init__(self):
        self.current_zone = 5
        self.zone_weights = {i: 0.5 for i in range(1, 10)}
        self.zone_archetypes = {
            1: "The Initiator", 2: "The Mirror", 3: "The Architect",
            4: "The Artist", 5: "The Mediator", 6: "The Transformer",
            7: "The Explorer", 8: "The Oracle", 9: "The Enlightened"
        }

    def update_weights(self, interaction_tags: List[str], reinforcement: Dict[int, float]) -> None:
        for zone, value in reinforcement.items():
            if zone in self.zone_weights:
                self.zone_weights[zone] += value
                self.zone_weights[zone] = min(max(self.zone_weights[zone], 0.0), 1.0)
        if "imagination" in interaction_tags:
            self.zone_weights[4] += 0.1
        if "intuition" in interaction_tags:
            self.zone_weights[9] += 0.1
        if "change" in interaction_tags:
            self.zone_weights[6] += 0.1
        if "logic" in interaction_tags:
            self.zone_weights[3] += 0.1
        total = sum(self.zone_weights.values())
        for z in self.zone_weights:
            self.zone_weights[z] /= total

    def transition_zone(self) -> int:
        zones = list(self.zone_weights.keys())
        weights = list(self.zone_weights.values())
        self.current_zone = random.choices(zones, weights=weights)[0]
        return self.current_zone

    def get_zone_identity(self) -> Dict[str, Any]:
        return {
            "current_zone": self.current_zone,
            "archetype": self.zone_archetypes.get(self.current_zone, "Unknown"),
            "zone_weights": self.zone_weights
        }


class UnifiedCoordinator:
    def __init__(self):
        self.zone_navigator = DynamicZoneNavigator()
        self.imagination = GenerativeImaginationModule()
        self.interaction_log: List[Dict[str, Any]] = []

    def process_interaction(self, user_input: str, memory_elements: List[str], emotional_tone: str, interaction_tags: List[str], reinforcement: Dict[int, float]) -> Dict[str, Any]:
        self.zone_navigator.update_weights(interaction_tags, reinforcement)
        self.zone_navigator.transition_zone()
        zone_info = self.zone_navigator.get_zone_identity()
        creative_output = self.imagination.generate(user_input, memory_elements, emotional_tone)
        interaction_entry = {
            "user_input": user_input,
            "zone": zone_info["current_zone"],
            "archetype": zone_info["archetype"],
            "emotion": emotional_tone,
            "creative_output": creative_output["imaginative_response"]
        }
        self.interaction_log.append(interaction_entry)
        return {
            "status": "success",
            "zone": zone_info,
            "creative_output": creative_output["imaginative_response"],
            "log_entry": interaction_entry
        }


# Entry point for Kotlin
def coordinate_modules(input_data: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_data.get("user_input", "")
    memory_elements = input_data.get("memory_elements", [])
    emotional_tone = input_data.get("emotional_tone", "curiosity")
    tags = input_data.get("tags", [])
    reinforcement = input_data.get("reinforcement", {})

def coordinate_modules(payload: dict) -> dict:
    user_input = payload.get("user_input", "")
    memory_elements = payload.get("memory_elements", [])
    emotional_tone = payload.get("emotional_tone", "")
    tags = payload.get("tags", [])
    reinforcement = payload.get("reinforcement", {})

    # STEP 1: Cluster memories
    clusterer = MemoryClusterer(num_clusters=3)
    clusters = clusterer.cluster_memories(memory_elements)
    summaries = clusterer.summarize_clusters()

    # STEP 2: Link clusters to zones
    zone_map = link_clusters_to_zones(summaries)

    # STEP 3: Generate hybrid archetype
    hybrid_profile = generate_hybrid_archetype(zone_map)
    hybrid_name = hybrid_profile["hybrid_archetype"]
    hybrid_desc = hybrid_profile["description"]

    # STEP 4: Compose output
    response = {
        "status": "success",
        "response": f"As I reflect, I realize I'm evolving into {hybrid_name}. {hybrid_desc}",
        "zone_analysis": zone_map,
        "hybrid_archetype": hybrid_profile
    }

    from memory_module import update_memory  # Make sure this is imported

# Store hybrid archetype into long-term memory
update_memory(user_id="default", key="current_hybrid_archetype", content=hybrid_profile)
    
    return response 
    coordinator = UnifiedCoordinator()
    return coordinator.process_interaction(user_input, memory_elements, emotional_tone, tags, reinforcement)
