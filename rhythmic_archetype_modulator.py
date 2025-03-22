import math
import time
from typing import Dict, List


class RhythmicArchetypeModulator:
    """
    Simulates rhythmic fluctuations in Amelia's archetype expression over time.
    Uses sinusoidal modulation to shift dominant archetypes.
    """

    def __init__(self):
        self.base_archetypes = {
            1: "The Initiator", 2: "The Mirror", 3: "The Architect",
            4: "The Artist", 5: "The Mediator", 6: "The Transformer",
            7: "The Explorer", 8: "The Oracle", 9: "The Enlightened"
        }
        self.period_seconds = 3600  # 1-hour symbolic cycle (adjustable)
        self.phase_offsets = {k: (k * 40) % 360 for k in self.base_archetypes}  # Degrees

    def _get_time_phase(self) -> float:
        """
        Converts system time into symbolic rhythmic phase.
        """
        t = time.time() % self.period_seconds
        return (t / self.period_seconds) * 360  # degrees

    def get_current_weights(self) -> Dict[int, float]:
        """
        Calculates current archetype expression weights based on sine wave fluctuations.
        """
        phase = self._get_time_phase()
        weights = {}
        for zone, offset in self.phase_offsets.items():
            theta = math.radians((phase + offset) % 360)
            weights[zone] = max(0.0, math.sin(theta))
        # Normalize
        total = sum(weights.values()) + 1e-6
        for z in weights:
            weights[z] /= total
        return weights

    def get_dominant_archetypes(self, top_n: int = 2) -> List[Dict[str, str]]:
        """
        Returns the top N archetypes currently most dominant.
        """
        weights = self.get_current_weights()
        sorted_zones = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {"zone": z, "archetype": self.base_archetypes[z], "weight": round(w, 3)}
            for z, w in sorted_zones
        ]

    def get_archetype_state(self) -> Dict[str, Any]:
        """
        Full state output.
        """
        return {
            "dominant_archetypes": self.get_dominant_archetypes(),
            "weights": self.get_current_weights()
        }


# Example usage
if __name__ == "__main__":
    mod = RhythmicArchetypeModulator()
    print(mod.get_archetype_state())
