import json
from typing import Dict, Any, List, Tuple, Optional


class MythicCausalityEngine:
    def __init__(self, cause: str = "", effect: str = ""):
        self.cause: str = cause
        self.effect: str = effect
        self.history: List[Tuple[str, str]] = []

    def link_cause_to_effect(self, cause: str, effect: str) -> Dict[str, Any]:
        """Link a cause to an effect and store it in history."""
        self.cause = cause
        self.effect = effect
        self.history.append((cause, effect))
        return {"linked": True, "cause": cause, "effect": effect}

    def get_last_link(self) -> Dict[str, Optional[str]]:
        """Retrieve the last linked cause and effect."""
        if self.history:
            cause, effect = self.history[-1]
            return {"cause": cause, "effect": effect}
        return {"cause": None, "effect": None}

    def get_history(self) -> List[Dict[str, str]]:
        """Get the history of linked causes and effects."""
        return [{"cause": cause, "effect": effect} for cause, effect in self.history]

    def to_json(self) -> str:
        """Convert the object data to JSON format."""
        result = self._prepare_result_for_kotlin_bridge()
        return json.dumps(result)

    @classmethod
    def from_json(cls, json_data: str) -> 'MythicCausalityEngine':
        """Create an instance from a JSON string."""
        try:
            data = json.loads(json_data)
            instance = cls(
                cause=data.get("cause", ""),
                effect=data.get("effect", "")
            )
            return instance
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to create module from JSON: {e}")

    def _prepare_result_for_kotlin_bridge(self) -> Dict[str, Any]:
        """Prepare the result for Kotlin bridge integration."""
        return {
            "status": "success",
            "data": {
                "cause": self.cause,
                "effect": self.effect
            },
            "metadata": {
                "history_length": len(self.history)
            }
        }

    def clear_history(self) -> None:
        """Clear the history of linked causes and effects."""
        self.history.clear()

    def __str__(self) -> str:
        """Provide a string representation of the object."""
        return f"MythicCausalityEngine(cause='{self.cause}', effect='{self.effect}', history_length={len(self.history)})"
