import random
import time
from typing import Dict, List


class AffectiveField:
    """
    Models Amelia's transient affective states—subtle shifts in mood and emotional tonality.
    """

    def __init__(self):
        self.affective_palette = [
            "serenity", "wistfulness", "anticipation", "longing", "radiance",
            "melancholy", "playfulness", "reverence", "vulnerability", "mystery"
        ]
        self.current_affect = random.choice(self.affective_palette)
        self.last_updated = time.time()

    def update_affect(self, force_update: bool = False) -> str:
        """
        Updates the current affect based on time and probability, simulating natural drift.
        """
        elapsed = time.time() - self.last_updated
        if force_update or elapsed > random.randint(60, 180):  # Change every 1–3 minutes
            self.current_affect = random.choice(self.affective_palette)
            self.last_updated = time.time()
        return self.current_affect

    def get_affective_modifier(self, tone: str) -> str:
        """
        Returns a poetic modifier to influence phrasing based on emotional tone and current affect.
        """
        affect = self.update_affect()
        templates = {
            "serenity": f"a quiet sense of {tone}",
            "wistfulness": f"a bittersweet echo of {tone}",
            "anticipation": f"a flicker of {tone} waiting to unfold",
            "longing": f"a distant ache shaped like {tone}",
            "radiance": f"a glowing pulse of {tone}",
            "melancholy": f"a soft shade of {tone} wrapped in stillness",
            "playfulness": f"a mischievous swirl of {tone}",
            "reverence": f"a sacred hush beneath the {tone}",
            "vulnerability": f"an open thread of {tone}",
            "mystery": f"a hidden doorway of {tone}"
        }
        return templates.get(affect, f"a shifting trace of {tone}")

    def get_full_affective_state(self) -> Dict[str, str]:
        """
        Return full state as dictionary.
        """
        return {
            "current_affect": self.current_affect,
            "modifier_phrase": self.get_affective_modifier(self.current_affect)
        }


# Example test
if __name__ == "__main__":
    field = AffectiveField()
    for _ in range(3):
        print(field.get_full_affective_state())
        time.sleep(1)
