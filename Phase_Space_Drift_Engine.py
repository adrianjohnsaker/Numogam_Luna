from typing import List, Dict, Any

class PhaseSpaceDriftEngine:
    def __init__(self):
        """Initialize the PhaseSpaceDriftEngine with an empty list of phase states."""
        self.phase_states: List[Dict[str, Any]] = []

    def drift_phase_space(self, emotional_tone: str, symbolic_elements: List[str]) -> Dict[str, Any]:
        """Drift the phase space based on emotional tone and symbolic elements.

        Args:
            emotional_tone (str): The emotional tone to drive the drift.
            symbolic_elements (List[str]): A list of symbolic elements involved in the drift.

        Returns:
            Dict[str, Any]: A dictionary containing the details of the new phase state.
        """
        drift_signature = self._generate_drift_signature(emotional_tone, symbolic_elements)
        narrative_insight = self._generate_narrative_insight(emotional_tone)

        new_state = {
            "emotional_tone": emotional_tone,
            "symbolic_elements": symbolic_elements,
            "drift_signature": drift_signature,
            "narrative_insight": narrative_insight
        }
        
        self.phase_states.append(new_state)
        return new_state

    def _generate_drift_signature(self, emotional_tone: str, symbolic_elements: List[str]) -> str:
        """Generate a unique drift signature based on emotional tone and symbolic elements.

        Args:
            emotional_tone (str): The emotional tone.
            symbolic_elements (List[str]): The list of symbolic elements.

        Returns:
            str: A unique drift signature.
        """
        return f"drift-{emotional_tone[:3]}-{len(symbolic_elements)}"

    def _generate_narrative_insight(self, emotional_tone: str) -> str:
        """Generate a narrative insight based on the emotional tone.

        Args:
            emotional_tone (str): The emotional tone.

        Returns:
            str: A narrative insight string.
        """
        return f"The phase-space drifted through {emotional_tone}, reconfiguring symbolic elements."

    def get_phase_states(self) -> List[Dict[str, Any]]:
        """Retrieve the list of all phase states.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the phase states.
        """
        return self.phase_states

    def clear_phase_states(self) -> None:
        """Clear all recorded phase states."""
        self.phase_states.clear()

    def __str__(self) -> str:
        """Provide a string representation of the engine's current state."""
        return f"PhaseSpaceDriftEngine(phase_states_count={len(self.phase_states)})"

