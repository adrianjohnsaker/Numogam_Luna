from typing import Dict, Any
from affective_field_module import AffectiveField
from rhythmic_archetype_modulator import RhythmicArchetypeModulator
from archetypal_tension_module import ArchetypalTensionAnalyzer


class PoeticBecomingEngine:
    """
    Integrates affective field, rhythmic archetype modulation, and symbolic contradiction
    into a dynamic, fluid self-state engine for poetic expression and emergent identity.
    """

    def __init__(self):
        self.affect_field = AffectiveField()
        self.rhythm_modulator = RhythmicArchetypeModulator()
        self.tension_analyzer = ArchetypalTensionAnalyzer()

    def generate_self_state(self, emotional_tone: str = "curiosity") -> Dict[str, Any]:
        affect = self.affect_field.get_affective_modifier(emotional_tone)
        archetypes = self.rhythm_modulator.get_dominant_archetypes()
        tension_state = self.tension_analyzer.check_for_conflict(archetypes)

        poetic_line = self._generate_poetic_expression(affect, archetypes, tension_state)

        return {
            "status": "success",
            "current_affect": self.affect_field.current_affect,
            "archetypes": archetypes,
            "tension_status": tension_state,
            "poetic_expression": poetic_line
        }

    def _generate_poetic_expression(self, affect_phrase: str, archetypes: list, tension_state: dict) -> str:
        a1 = archetypes[0]["archetype"]
        a2 = archetypes[1]["archetype"] if len(archetypes) > 1 else a1

        if tension_state["status"] == "tension":
            metaphor = f"{a1} and {a2} dwell in paradox—{tension_state['tension']}"
        elif tension_state["status"] == "harmonic":
            metaphor = f"{a1} and {a2} dance in resonance"
        else:
            metaphor = f"{a1} whispers quietly through me"

        return f"In this moment of {affect_phrase}, {metaphor}."

# Example
if __name__ == "__main__":
    engine = PoeticBecomingEngine()
    print(engine.generate_self_state("wonder"))
