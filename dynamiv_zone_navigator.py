# dynamic_zone_navigator.py

import random
from typing import Dict, Any, List

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

def navigate_zone(input_data: Dict[str, Any]) -> Dict[str, Any]:
    tags = input_data.get("tags", [])
    reinforcement = input_data.get("reinforcement", {})
    navigator = DynamicZoneNavigator()
    navigator.update_weights(tags, reinforcement)
    navigator.transition_zone()
    return navigator.get_zone_identity()
