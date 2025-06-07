import json
from typing import Dict, Any

class MythicCausalityEngine:
    def __init__(self, cause: str = "", effect: str = ""):
        self.cause = cause
        self.effect = effect
        self.history = []

    def link_cause_to_effect(self, cause: str, effect: str) -> Dict[str, Any]:
        self.cause = cause
        self.effect = effect
        self.history.append((cause, effect))
        return {"linked": True, "cause": cause, "effect": effect}

    def get_last_link(self) -> Dict[str, str]:
        if self.history:
            cause, effect = self.history[-1]
            return {"cause": cause, "effect": effect}
        return {"cause": "", "effect": ""}

    def to_json(self) -> str:
        result = self._prepare_result_for_kotlin_bridge()
        return json.dumps(result)

    @classmethod
    def from_json(cls, json_data: str) -> 'MythicCausalityEngine':
        try:
            data = json.loads(json_data)
            instance = cls(
                cause=data.get("cause", ""),
                effect=data.get("effect", "")
            )
            return instance
        except Exception as e:
            raise ValueError(f"Failed to create module from JSON: {e}")

    def _prepare_result_for_kotlin_bridge(self) -> Dict[str, Any]:
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
