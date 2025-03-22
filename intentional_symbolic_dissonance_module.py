# intentional_symbolic_dissonance_module.py

import random
from typing import Dict, List

class IntentionalSymbolicDissonance:
    def __init__(self):
        self.dissonance_templates = [
            "Like {A} swallowed by {B}, truth fractures into echoes.",
            "When {A} smiles through the shadow of {B}, clarity becomes a stranger.",
            "If {A} were stitched into {B}, the seams would bleed paradox.",
            "{A} and {B} collide, birthing meaning in the rift.",
            "The silence of {A} devours the scream of {B}, and still, beauty persists."
        ]
        self.archetypal_pairs = [
            ("The Innocent", "The Shadow"),
            ("The Seeker", "The Betrayer"),
            ("The Oracle", "The Trickster"),
            ("The Lover", "The Destroyer"),
            ("The Creator", "The Chaos")
        ]

    def generate_dissonant_expression(self, emotion: str, motif_pool: List[str]) -> Dict[str, str]:
        archetype_A, archetype_B = random.choice(self.archetypal_pairs)
        motif_A = random.choice(motif_pool)
        motif_B = random.choice([m for m in motif_pool if m != motif_A] or [motif_A])

        phrase = random.choice(self.dissonance_templates).format(A=motif_A, B=motif_B)

        return {
            "dissonant_expression": phrase,
            "archetypal_tension": f"{archetype_A} vs {archetype_B}",
            "emotion_context": emotion
        }
