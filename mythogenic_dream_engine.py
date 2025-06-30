# mythogenic_dream_engine.py

from typing import List, Dict, Optional, Tuple
import random
import uuid

class MythogenicDreamEngine:
    """
    Translates symbolic dream input into mythic frameworks, axes of meaning,
    and emerging thematic constellations.
    """

    def __init__(self):
        self.myth_templates = self._load_myth_templates()
        self.thematic_axes = [
            "Creation–Destruction",
            "Ascent–Descent",
            "Exile–Return",
            "Awakening–Forgetting",
            "Union–Separation",
            "Sacrifice–Rebirth"
        ]

    def _load_myth_templates(self) -> List[Dict[str, str]]:
        return [
            {
                "name": "Hero’s Journey",
                "structure": "Call → Trial → Transformation → Return",
                "axis": "Exile–Return"
            },
            {
                "name": "Shamanic Descent",
                "structure": "Descent → Death → Vision → Rebirth",
                "axis": "Ascent–Descent"
            },
            {
                "name": "Cosmic Birth",
                "structure": "Chaos → Fragmentation → Emergence → Harmony",
                "axis": "Creation–Destruction"
            },
            {
                "name": "Myth of Split Selves",
                "structure": "Union → Fracture → Mirror → Resolution",
                "axis": "Union–Separation"
            }
        ]

    def derive_mythic_lineage(self, dream_symbols: List[Dict[str, str]]) -> Dict:
        """
        Given symbolic input, generate a mythogenic interpretation.
        """
        if not dream_symbols:
            return {"error": "No symbols provided."}

        axis = self._infer_axis_from_symbols(dream_symbols)
        template = self._select_template_by_axis(axis)
        thematic_threads = self._generate_thematic_threads(dream_symbols, template)

        return {
            "myth_id": str(uuid.uuid4()),
            "axis": axis,
            "template": template,
            "thematic_threads": thematic_threads
        }

    def _infer_axis_from_symbols(self, symbols: List[Dict[str, str]]) -> str:
        """
        Analyze symbol meanings to infer dominant mythic axis.
        """
        meaning_pool = " ".join([s.get("meaning", "") for s in symbols]).lower()

        axis_scores = {axis: 0 for axis in self.thematic_axes}
        for axis in self.thematic_axes:
            for term in axis.lower().split("–"):
                if term in meaning_pool:
                    axis_scores[axis] += 1

        # Fallback to random if no match found
        top_axis = max(axis_scores, key=axis_scores.get)
        return top_axis if axis_scores[top_axis] > 0 else random.choice(self.thematic_axes)

    def _select_template_by_axis(self, axis: str) -> Dict[str, str]:
        """
        Match axis to a template or fallback randomly.
        """
        matches = [t for t in self.myth_templates if t["axis"] == axis]
        return random.choice(matches) if matches else random.choice(self.myth_templates)

    def _generate_thematic_threads(
        self, symbols: List[Dict[str, str]], template: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """
        Generate thematic interpretations aligned with template structure.
        """
        stages = [s.strip() for s in template["structure"].split("→")]
        threads = []

        for i, stage in enumerate(stages):
            symbol = symbols[i % len(symbols)]
            threads.append({
                "stage": stage,
                "symbol": symbol["symbol"],
                "meaning": symbol.get("meaning", ""),
                "emotional_charge": symbol.get("emotional_charge", "neutral"),
                "resonance": self._estimate_resonance(stage, symbol)
            })

        return threads

    def _estimate_resonance(self, stage: str, symbol: Dict[str, str]) -> float:
        """
        Placeholder for future deep symbolic-emotional mapping.
        """
        charge = symbol.get("emotional_charge", "neutral").lower()
        base = {
            "neutral": 0.5,
            "positive": 0.7,
            "negative": 0.3,
            "intense": 0.9,
            "muted": 0.4
        }.get(charge, 0.5)

        return round(base + (random.uniform(-0.1, 0.1)), 2)

# Example standalone usage
if __name__ == "__main__":
    engine = MythogenicDreamEngine()
    symbols = [
        {"symbol": "serpent", "meaning": "transformation and danger", "emotional_charge": "intense"},
        {"symbol": "mirror", "meaning": "self-reflection", "emotional_charge": "neutral"},
        {"symbol": "spiral", "meaning": "evolutionary process", "emotional_charge": "positive"}
    ]
    result = engine.derive_mythic_lineage(symbols)
    import json
    print(json.dumps(result, indent=2))
