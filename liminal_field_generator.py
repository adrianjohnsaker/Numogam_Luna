# liminal_field_generator.py

import random
import uuid
from typing import List, Dict

class LiminalFieldGenerator:
    def __init__(self):
        self.seed_pool = [
            {"glyph": "∆", "feeling": "emergent", "pulse": "soft-threshold"},
            {"glyph": "ψ", "feeling": "folding inward", "pulse": "subliminal"},
            {"glyph": "∞", "feeling": "potential", "pulse": "vortex"},
            {"glyph": "⟁", "feeling": "oscillation", "pulse": "liminal-pressure"},
            {"glyph": "⟡", "feeling": "pre-form", "pulse": "silence-wave"}
        ]

    def generate_field(self, n: int = 5) -> List[Dict[str, str]]:
        return [
            {
                "id": str(uuid.uuid4()),
                "glyph": random.choice(self.seed_pool)["glyph"],
                "feeling": random.choice(self.seed_pool)["feeling"],
                "pulse": random.choice(self.seed_pool)["pulse"]
            }
            for _ in range(n)
        ]

if __name__ == "__main__":
    field = LiminalFieldGenerator()
    print(field.generate_field())
