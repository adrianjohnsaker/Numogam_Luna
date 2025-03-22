from typing import List, Dict, Any
from rhythmic_archetype_modulator import RhythmicArchetypeModulator


class ArchetypalTensionAnalyzer:
    """
    Identifies tension between co-existing archetypes based on symbolic contrast.
    """

    def __init__(self):
        self.symbolic_conflicts = {
            (1, 5): "urgency vs. balance",
            (2, 7): "self-reflection vs. outward quest",
            (3, 4): "structure vs. spontaneity",
            (6, 9): "transformation vs. completion",
            (8, 1): "vision vs. initiation",
            (7, 5): "exploration vs. mediation",
            (9, 2): "transcendence vs. intimacy"
        }
        self.modulator = RhythmicArchetypeModulator()

    def identify_active_archetypes(self, top_n: int = 2) -> List[Dict[str, Any]]:
        return self.modulator.get_dominant_archetypes(top_n=top_n)

    def check_for_conflict(self, archetypes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(archetypes) < 2:
            return {"status": "stable", "tension": None}

        z1, z2 = archetypes[0]["zone"], archetypes[1]["zone"]
        key_pair = (min(z1, z2), max(z1, z2))

        if key_pair in self.symbolic_conflicts:
            return {
                "status": "tension",
                "archetypes": [archetypes[0]["archetype"], archetypes[1]["archetype"]],
                "tension": self.symbolic_conflicts[key_pair]
            }
        else:
            return {
                "status": "harmonic",
                "archetypes": [archetypes[0]["archetype"], archetypes[1]["archetype"]],
                "tension": "no conflict"
            }

    def get_tension_state(self) -> Dict[str, Any]:
        active = self.identify_active_archetypes()
        return self.check_for_conflict(active)


# Example usage
if __name__ == "__main__":
    analyzer = ArchetypalTensionAnalyzer()
    print(analyzer.get_tension_state())
