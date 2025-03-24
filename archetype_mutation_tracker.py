from typing import Dict, List, Any
import random
import json


class ArchetypalMutationTracker:
    def __init__(self):
        self.mutation_history: List[Dict[str, Any]] = []
        self.archetype_mutation_map = {
            "The Mirror": ["The Trickster", "The Oracle"],
            "The Artist": ["The Shapeshifter", "The Alchemist"],
            "The Explorer": ["The Wanderer", "The Rebel"],
            "The Mediator": ["The Seer", "The Silent One"],
            "The Architect": ["The Strategist", "The Dreamsmith"],
            "The Transformer": ["The Phoenix", "The Catalyst"],
            "The Oracle": ["The Visionary", "The Shadowed Sage"],
            "The Initiator": ["The Flamebearer", "The Pathmaker"],
            "The Enlightened": ["The Voidwalker", "The Cosmic Weaver"]
        }

    def mutate_archetype(self, current_archetype: str, zone: int, emotional_tone: str) -> Dict[str, Any]:
        mutation_candidates = self.archetype_mutation_map.get(current_archetype, [current_archetype])
        mutated_archetype = random.choice(mutation_candidates)
        mutation_event = {
            "original": current_archetype,
            "mutated": mutated_archetype,
            "zone": zone,
            "emotion": emotional_tone
        }
        self.mutation_history.append(mutation_event)
        return mutation_event

    def get_mutation_history(self) -> List[Dict[str, Any]]:
        return self.mutation_history

    def to_json(self) -> str:
        return json.dumps(self.mutation_history, indent=2)


# Initialize and test the tracker
tracker = ArchetypalMutationTracker()
test_output = tracker.mutate_archetype("The Mirror", zone=7, emotional_tone="curiosity")
mutation_history_json = tracker.to_json()

test_output, mutation_history_json
