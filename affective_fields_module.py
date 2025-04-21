from __future__ import annotations
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class Affect(Enum):
    SERENITY     = "serenity"
    WISTFULNESS  = "wistfulness"
    ANTICIPATION = "anticipation"
    LONGING      = "longing"
    RADIANCE     = "radiance"
    MELANCHOLY   = "melancholy"
    PLAYFULNESS  = "playfulness"
    REVERENCE    = "reverence"
    VULNERABILITY= "vulnerability"
    MYSTERY      = "mystery"


class AffectiveField:
    """
    Models transient affective states with natural drift over time.
    """

    _MODIFIER_TEMPLATES: Dict[Affect, str] = {
        Affect.SERENITY:     "a quiet sense of {tone}",
        Affect.WISTFULNESS:  "a bittersweet echo of {tone}",
        Affect.ANTICIPATION: "a flicker of {tone} waiting to unfold",
        Affect.LONGING:      "a distant ache shaped like {tone}",
        Affect.RADIANCE:     "a glowing pulse of {tone}",
        Affect.MELANCHOLY:   "a soft shade of {tone} wrapped in stillness",
        Affect.PLAYFULNESS:  "a mischievous swirl of {tone}",
        Affect.REVERENCE:    "a sacred hush beneath the {tone}",
        Affect.VULNERABILITY:"an open thread of {tone}",
        Affect.MYSTERY:      "a hidden doorway of {tone}",
    }

    def __init__(
        self,
        palette: Optional[List[Affect]] = None,
        min_interval: int = 60,
        max_interval: int = 180,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            palette: Which affects to choose from (defaults to all).
            min_interval, max_interval: Seconds between automatic drift.
            seed: Optional seed for reproducible randomness.
        """
        self.palette: List[Affect] = palette or list(Affect)
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._random = random.Random(seed)

        self.current_affect: Affect = self._random.choice(self.palette)
        self._schedule_next_update()

    def _schedule_next_update(self) -> None:
        """Compute and store the next timestamp when affect will drift."""
        interval = self._random.randint(self.min_interval, self.max_interval)
        self._next_update: datetime = datetime.now() + timedelta(seconds=interval)

    def update_affect(self, force: bool = False) -> Affect:
        """
        Possibly drift to a new affect.

        Args:
            force: If True, trigger a change immediately.
        Returns:
            The (possibly updated) current Affect.
        """
        if force or datetime.now() >= self._next_update:
            self.current_affect = self._random.choice(self.palette)
            self._schedule_next_update()
        return self.current_affect

    def get_modifier(self, tone: str) -> str:
        """
        Build a poetic modifier phrase based on current affect.

        Args:
            tone: The tonal keyword to weave into the phrase.
        Returns:
            A formatted string like "a glowing pulse of serenity".
        """
        affect = self.update_affect()
        template = self._MODIFIER_TEMPLATES.get(
            affect,
            "a shifting trace of {tone}"
        )
        return template.format(tone=tone)

    def get_state(self) -> Dict[str, str]:
        """
        Return the full affective state.

        Returns:
            {
                "current_affect": "melancholy",
                "modifier": "a soft shade of melancholy wrapped in stillness",
                "next_update": "2025-04-21T14:30:05.123456"
            }
        """
        return {
            "current_affect": self.current_affect.value,
            "modifier":       self.get_modifier(self.current_affect.value),
            "next_update":    self._next_update.isoformat(),
        }

    def set_affect(self, affect: Affect) -> None:
        """
        Manually override the current affect (and reschedule the next drift).

        Raises:
            ValueError if `affect` isn’t in the palette.
        """
        if affect not in self.palette:
            raise ValueError(f"{affect!r} not in palette")
        self.current_affect = affect
        self._schedule_next_update()

    def __repr__(self) -> str:
        return (
            f"<AffectiveField current={self.current_affect.value!r} "
            f"next_update={self._next_update.isoformat()}>"
        )


# Example usage
if __name__ == "__main__":
    af = AffectiveField(seed=42)
    for _ in range(5):
        state = af.get_state()
        print(state)
        time_to_sleep = af.min_interval // 2
        time.sleep(time_to_sleep)
