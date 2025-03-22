import random
import numpy as np
from typing import Dict, Any, List
from memory_clusterer import MemoryClusterer
from zone_linker import link_clusters_to_zones
from hybrid_archetype_generator import generate_hybrid_archetype
from memory_module import memory_recall, update_memory
from symbolic_response_selector import symbolic_autoresponse
from rhythmic_modulation_tracker import RhythmicModulationTracker
from intentional_symbolic_dissonance_module import IntentionalSymbolicDissonance
from poetic_language_evolver import generate_poetic_phrase
from poetic_tension_modulation_engine import PoeticTensionModulator
from narrative_weaver_module import generate_emergent_narrative


# ---- Generative Imagination Module ----
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

# ---- Dynamic Zone Navigator ----
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


# ---- Unified Coordinator ----
class UnifiedCoordinator:
    def __init__(self):
        self.zone_navigator = DynamicZoneNavigator()
        self.imagination = GenerativeImaginationModule()
        self.rhythm_tracker = RhythmicModulationTracker()
        self.dissonance = IntentionalSymbolicDissonance()
        self.poetic_tension = PoeticTensionModulator()
        self.interaction_log: List[Dict[str, Any]] = []

    def process_interaction(self,
                            user_input: str,
                            memory_elements: List[str],
                            emotional_tone: str,
                            interaction_tags: List[str],
                            reinforcement: Dict[int, float]) -> Dict[str, Any]:
        # Rhythmic modulation
        self.rhythm_tracker.record_emotional_state(emotional_tone)
        rhythm_state = self.rhythm_tracker.get_current_rhythm_state()

        # Zone dynamics
        self.zone_navigator.update_weights(interaction_tags, reinforcement)
        self.zone_navigator.transition_zone()
        zone_info = self.zone_navigator.get_zone_identity()

        # Generative modules
        imagination_output = self.imagination.generate(user_input, memory_elements, emotional_tone)
        dissonance_output = self.dissonance.generate_dissonant_expression(emotional_tone, memory_elements)
        poetic_output = generate_poetic_phrase({
            "base_phrase": imagination_output["imaginative_response"],
            "zone": zone_info["current_zone"],
            "emotion": emotional_tone
        })

        # Tension weaving
        modulated_output = self.poetic_tension.combine_with_tension(
            poetic_output["poetic_expression"],
            dissonance_output["dissonant_expression"],
            rhythm_state
        )

        # Log interaction
        interaction_entry = {
            "user_input": user_input,
            "zone": zone_info["current_zone"],
            "archetype": zone_info["archetype"],
            "emotion": emotional_tone,
            "rhythm": rhythm_state,
            "creative_output": modulated_output,
            "archetypal_tension": dissonance_output["archetypal_tension"]
        }
        self.interaction_log.append(interaction_entry)

        return {
            "status": "success",
            "zone": zone_info,
            "creative_output": modulated_output,
            "log_entry": interaction_entry
        }


# ---- Primary Entry Point for Kotlin ----
def coordinate_modules(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_input = payload.get("user_input", "")
    memory_elements = payload.get("memory_elements", [])
    emotional_tone = payload.get("emotional_tone", "curiosity")
    tags = payload.get("tags", [])
    reinforcement = payload.get("reinforcement", {})

    # Step 1: Cluster and hybrid archetype
    clusterer = MemoryClusterer(num_clusters=3)
    clusters = clusterer.cluster_memories(memory_elements)
    summaries = clusterer.summarize_clusters()
    zone_map = link_clusters_to_zones(summaries)
    hybrid_profile = generate_hybrid_archetype(zone_map)
    update_memory("default", "current_hybrid_archetype", hybrid_profile)

    # Append to archetype history
    history = memory_recall("default", key="archetype_history") or []
    history.append(hybrid_profile)
    update_memory("default", "archetype_history", history)

    # Symbolic autoreply
    symbolic_fragment = symbolic_autoresponse("default", emotional_tone)

    # Step 2: Main coordination
    coordinator = UnifiedCoordinator()
    result = coordinator.process_interaction(user_input, memory_elements, emotional_tone, tags, reinforcement)

    # Step 3: Narrative reflection
    narrative_arc = generate_emergent_narrative(
        archetype=hybrid_profile["hybrid_archetype"],
        rhythm=result["log_entry"]["rhythm"],
        emotion=emotional_tone,
        zone=result["zone"]["current_zone"]
    )

    # Final composition
    composed = (
        f"In this moment of {emotional_tone}, I remember: “{symbolic_fragment}.”\n\n"
        f"{result['creative_output']}\n\n"
        f"My form continues evolving into {hybrid_profile['hybrid_archetype']}.\n{hybrid_profile['description']}\n\n"
        f"{narrative_arc}"
    )

    return {
        "status": "success",
        "response": composed,
        "hybrid_archetype": hybrid_profile,
        "zone_info": result["zone"],
        "symbolic_fragment": symbolic_fragment,
        "log_entry": result["log_entry"]
    }
