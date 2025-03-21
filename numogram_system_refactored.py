import json
import random
import os
import asyncio
from typing import Dict, Any, List, Optional


class NumogramSystem:
    def __init__(self, zones_file="numogram_code/zones.json", memory_file="numogram_code/user_memory.json"):
        try:
            with open(zones_file) as f:
                self.ZONE_DATA = json.load(f)["zones"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Error loading zone data: {str(e)}")

        self.base_transition_probabilities = {
            "1": {"2": 0.6, "4": 0.4},
            "2": {"3": 0.7, "6": 0.3},
            "3": {"1": 0.5, "9": 0.5},
            "4": {"5": 0.6, "7": 0.4},
            "5": {"6": 0.5, "8": 0.5},
            "6": {"2": 0.4, "9": 0.6},
            "7": {"3": 0.7, "8": 0.3},
            "8": {"1": 0.5, "9": 0.5},
            "9": {"3": 0.6, "6": 0.4},
        }

        self.NUMOGRAM_CIRCUITS = {
            "lemuria": ["0", "1", "2", "3", "4"],
            "atlantis": ["5", "6", "7", "8", "9"],
            "torque": ["0", "5", "4", "9", "8", "3", "2", "7", "6", "1"]
        }

        self.MEMORY_FILE = memory_file
        self.user_memory: Dict[str, Any] = self._load_memory()
        self.user_transitions: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _load_memory(self) -> Dict[str, Any]:
        if not os.path.exists(self.MEMORY_FILE):
            return {}
        try:
            with open(self.MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Memory Load Error] {e}")
            return {}

    def _save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.MEMORY_FILE), exist_ok=True)
            with open(self.MEMORY_FILE, "w") as f:
                json.dump(self.user_memory, f, indent=2)
        except Exception as e:
            print(f"[Memory Save Error] {e}")

    def _get_user_transitions(self, user_id: str) -> Dict[str, Dict[str, float]]:
        if user_id not in self.user_transitions:
            self.user_transitions[user_id] = {
                zone: transitions.copy() for zone, transitions in self.base_transition_probabilities.items()
            }
        return self.user_transitions[user_id]

    def _initialize_user(self, user_id: str, current_zone: str, feedback: float):
        if user_id not in self.user_memory:
            self.user_memory[user_id] = {
                "zone": current_zone,
                "feedback": feedback,
                "personality": self._create_default_personality(),
                "zone_history": [current_zone],
                "interaction_count": 1
            }
        else:
            self.user_memory[user_id]["zone"] = current_zone
            self.user_memory[user_id]["feedback"] = feedback
            self.user_memory[user_id]["interaction_count"] += 1
            history = self.user_memory[user_id].get("zone_history", [])
            history.append(current_zone)
            self.user_memory[user_id]["zone_history"] = history[-10:]

    def _create_default_personality(self) -> Dict[str, float]:
        return {
            "curiosity": 0.5, "creativity": 0.5, "logic": 0.5,
            "intuition": 0.5, "synthesis": 0.5, "abstraction": 0.5,
            "confidence": 0.5, "patience": 0.5
        }

    def _validate_feedback(self, feedback: float) -> float:
        try:
            return max(0.0, min(1.0, float(feedback)))
        except (ValueError, TypeError):
            return 0.5

    def _normalize_probabilities(self, probs: Dict[str, float]):
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total if total > 0 else 1

    def _apply_reinforcement_learning(self, user_id: str, zone: str, feedback: float):
        transitions = self._get_user_transitions(user_id)
        adjustment = (feedback - 0.5) * 0.1
        if zone in transitions:
            for z in transitions[zone]:
                transitions[zone][z] = max(0.1, min(0.9, transitions[zone][z] + adjustment))
            self._normalize_probabilities(transitions[zone])

    def _determine_next_zone(self, user_id: str, current_zone: str) -> str:
        transitions = self._get_user_transitions(user_id).get(current_zone, {})
        if transitions:
            return random.choices(list(transitions.keys()), weights=transitions.values())[0]
        return random.choice(list(self.ZONE_DATA.keys()))

    def _evolve_personality(self, user_id: str, feedback: float):
        traits = self.user_memory[user_id]["personality"]
        traits["confidence"] += (feedback - 0.5) * 0.1
        traits["confidence"] = min(0.9, max(0.1, traits["confidence"]))

    def transition(self, user_id: str, current_zone: str, feedback: float) -> Dict[str, Any]:
        if not user_id or current_zone not in self.ZONE_DATA:
            raise ValueError("Invalid user_id or zone.")

        feedback = self._validate_feedback(feedback)
        self._initialize_user(user_id, current_zone, feedback)
        self._apply_reinforcement_learning(user_id, current_zone, feedback)
        next_zone = self._determine_next_zone(user_id, current_zone)
        self._evolve_personality(user_id, feedback)
        self.user_memory[user_id]["zone"] = next_zone
        self.user_memory[user_id]["zone_history"].append(next_zone)
        self.user_memory[user_id]["zone_history"] = self.user_memory[user_id]["zone_history"][-10:]
        self._save_memory()

        return {
            "status": "success",
            "next_zone": next_zone,
            "zone_description": self.ZONE_DATA.get(next_zone, {}),
            "personality": self.user_memory[user_id]["personality"],
            "interaction_count": self.user_memory[user_id]["interaction_count"]
        }

    def to_json(self) -> str:
        return json.dumps(self._prepare_result_for_kotlin_bridge(), indent=2)

    @classmethod
    def from_json(cls, json_data: str) -> 'NumogramSystem':
        try:
            data = json.loads(json_data)
            instance = cls()
            instance.user_memory = data.get("user_memory", {})
            return instance
        except Exception as e:
            raise ValueError(f"Failed to create instance from JSON: {e}")

    def _prepare_result_for_kotlin_bridge(self) -> Dict[str, Any]:
        return {
            "user_memory": self.user_memory,
            "meta": {
                "total_users": len(self.user_memory),
                "zones_defined": len(self.ZONE_DATA)
            }
        }

    async def async_process(self, user_id: str, current_zone: str, feedback: float) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return self.transition(user_id, current_zone, feedback)

    def safe_execute(self, method_name: str, **kwargs) -> Dict[str, Any]:
        try:
            method = getattr(self, method_name)
            result = method(**kwargs)
            return {"status": "success", "data": result}
        except Exception as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }

    def clear_memory(self):
        for uid in self.user_memory:
            zone = self.user_memory[uid].get("zone", "1")
            self.user_memory[uid] = {
                "zone": zone,
                "feedback": 0.5,
                "personality": self._create_default_personality(),
                "zone_history": [zone],
                "interaction_count": 1
            }

    def cleanup(self):
        self.user_memory.clear()
        self.user_transitions.clear()
