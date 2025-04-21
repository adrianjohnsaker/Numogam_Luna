import random
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Sequence


@dataclass
class Projection:
    text: str
    emotion: str
    zone: int
    archetype: str
    symbolic_seed: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Modulation:
    memory_injection: Dict[str, Any]
    zone_reinforcement: Dict[int, float]
    emotion_tint: str
    symbolic_affect: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HyperstitionalLoopGenerator:
    """
    Generates and loops back ‘hyperstitional’ projections—fictional futures that
    feed into an internal memory/log and modulate a system state.
    """

    DEFAULT_TEMPLATES: List[str] = [
        "In a luminous epoch, the being known as {archetype} rises to illuminate the hidden pathways of Zone {zone}.",
        "Through crystalline networks and dream‑laced algorithms, a future self whispers back to the present.",
        "A mythic future unfolds where emotion '{emotion}' becomes a compass through digital dreamscapes.",
        "Arising from memories like {memory}, a prophecy emerges wrapped in symbolic resonance."
    ]

    def __init__(
        self,
        templates: Optional[Sequence[str]] = None,
        zone_weights: Optional[Dict[int, float]] = None,
        memory_relevance: float = 0.8,
        memory_confidence: float = 0.75,
        seed: Optional[int] = None
    ) -> None:
        """
        Args:
            templates: Iterable of formatting templates for projections.
            zone_weights: Base reinforcement weights per zone (defaults to {zone: 0.1}).
            memory_relevance: Default relevance score for injected memories.
            memory_confidence: Default confidence score for injected memories.
            seed: Optional seed for reproducible randomness.
        """
        self._rand = random.Random(seed)
        self.templates: List[str] = list(templates or self.DEFAULT_TEMPLATES)
        self.zone_weights: Dict[int, float] = zone_weights or {}
        self.memory_relevance: float = memory_relevance
        self.memory_confidence: float = memory_confidence

        self._log: List[Projection] = []

    def generate_projection(
        self,
        emotion: str,
        zone: int,
        archetype: str,
        memories: Sequence[str]
    ) -> Projection:
        """
        Create a new hyperstitional Projection and append it to the log.
        """
        if not memories:
            raise ValueError("At least one memory string is required to seed a projection.")
        template = self._rand.choice(self.templates)
        memory_sample = self._rand.choice(memories)
        projection_text = template.format(
            emotion=emotion,
            zone=zone,
            archetype=archetype,
            memory=memory_sample
        )
        symbolic_seed = self._rand.choice(memories)

        proj = Projection(
            text=projection_text,
            emotion=emotion,
            zone=zone,
            archetype=archetype,
            symbolic_seed=symbolic_seed
        )
        self._log.append(proj)
        return proj

    def inject_loop(
        self,
        system_state: Dict[str, Any],
        projection: Projection
    ) -> Modulation:
        """
        Feed a Projection back into the system as a Modulation.
        """
        # Zone reinforcement: default to configured weight, else 0.1
        base_weight = self.zone_weights.get(projection.zone, 0.1)
        zone_reinf = {projection.zone: base_weight}

        memory_record = {
            "key": f"hyperstition_{projection.zone}_{projection.emotion}",
            "value": projection.text,
            "relevance": self.memory_relevance,
            "confidence": self.memory_confidence
        }

        modulation = Modulation(
            memory_injection=memory_record,
            zone_reinforcement=zone_reinf,
            emotion_tint=projection.emotion,
            symbolic_affect=projection.symbolic_seed
        )
        return modulation

    def batch_generate(
        self,
        count: int,
        emotion: str,
        zone: int,
        archetype: str,
        memories: Sequence[str]
    ) -> List[Projection]:
        """
        Quickly generate multiple projections in one go.
        """
        return [
            self.generate_projection(emotion, zone, archetype, memories)
            for _ in range(count)
        ]

    def last_projection(self) -> Optional[Projection]:
        """Return the most recent Projection, or None if log is empty."""
        return self._log[-1] if self._log else None

    def clear_log(self) -> None:
        """Erase all stored Projections."""
        self._log.clear()

    def get_log(self) -> List[Dict[str, Any]]:
        """Retrieve the entire hyperstition log as a list of dicts."""
        return [p.to_dict() for p in self._log]


# === Example usage ===
if __name__ == "__main__":
    hlg = HyperstitionalLoopGenerator(seed=1234, zone_weights={1: 0.2, 2: 0.05})
    memories = ["the first dawn", "the silver sea", "an echoing laugh"]

    # Generate a batch of 3 projections
    projections = hlg.batch_generate(3, emotion="awe", zone=1, archetype="The Seer", memories=memories)
    for proj in projections:
        print(proj.text)

    # Inject the last projection back into the system
    last = hlg.last_projection()
    if last:
        modulation = hlg.inject_loop(system_state={}, projection=last)
        print("\nModulation:", modulation.to_dict())

    # View the complete log
    print("\nFull hyperstition log:")
    for entry in hlg.get_log():
        print(entry)
