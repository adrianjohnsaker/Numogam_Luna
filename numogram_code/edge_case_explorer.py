import json
import random
from typing import List, Dict, Any


class EdgeCaseExplorer:
    """Class to generate and analyze extreme or boundary conditions in scenarios."""

    def __init__(self):
        """
        Initialize the EdgeCaseExplorer with a default significance threshold.
        
        Attributes:
            THRESHOLD: Minimum significance threshold (1-5 scale).
        """
        self.THRESHOLD = 3

    def create_minimum_scenario(self, assumption: Dict, scenario: Dict) -> Dict:
        """Create a scenario where an assumption is minimized."""
        parameter = assumption.get("parameter")
        min_value = assumption.get("min_value", 0)

        # Clone the scenario and modify the relevant parameter
        min_scenario = scenario.copy()
        min_scenario["parameters"] = scenario.get("parameters", {}).copy()
        min_scenario["parameters"][parameter] = min_value
        min_scenario["name"] = f"Minimum {parameter}"
        min_scenario["description"] = f"Scenario where {parameter} is minimized to {min_value}"

        return min_scenario

    def create_maximum_scenario(self, assumption: Dict, scenario: Dict) -> Dict:
        """Create a scenario where an assumption is maximized."""
        parameter = assumption.get("parameter")
        max_value = assumption.get("max_value", 100)

        # Clone the scenario and modify the relevant parameter
        max_scenario = scenario.copy()
        max_scenario["parameters"] = scenario.get("parameters", {}).copy()
        max_scenario["parameters"][parameter] = max_value
        max_scenario["name"] = f"Maximum {parameter}"
        max_scenario["description"] = f"Scenario where {parameter} is maximized to {max_value}"

        return max_scenario

    def create_unexpected_scenario(self, assumption: Dict, scenario: Dict) -> Dict:
        """Create a scenario with an unexpected or unusual condition related to an assumption."""
        parameter = assumption.get("parameter")
        unexpected_conditions = assumption.get("unexpected_conditions", [])

        if not unexpected_conditions:
            # Generate generic unexpected conditions if none provided
            unexpected_conditions = [
                f"{parameter} behaves non-linearly",
                f"{parameter} fluctuates randomly",
                f"{parameter} affects different groups differently",
                f"{parameter} has delayed effects",
                f"{parameter} triggers emergent behavior"
            ]

        # Select a random unexpected condition
        unexpected_condition = random.choice(unexpected_conditions)

        # Clone the scenario and add the unexpected condition
        unexpected_scenario = scenario.copy()
        unexpected_scenario["unexpected_conditions"] = scenario.get("unexpected_conditions", []).copy()
        unexpected_scenario["unexpected_conditions"].append({
            "parameter": parameter,
            "condition": unexpected_condition
        })
        unexpected_scenario["name"] = f"Unexpected {parameter}"
        unexpected_scenario["description"] = f"Scenario where {parameter} exhibits {unexpected_condition}"

        return unexpected_scenario

    def analyze_implications(self, case: Dict) -> List[Dict]:
        """Analyze the implications of an edge case."""
        implications = []

        # Analyze parameter-based implications
        parameters = case.get("parameters", {})
        for param, value in parameters.items():
            is_extreme = False
            for assumption in case.get("baseline_assumptions", []):
                if assumption.get("parameter") == param:
                    min_val = assumption.get("min_value", 0)
                    max_val = assumption.get("max_value", 100)
                    if value <= min_val + (max_val - min_val) * 0.1 or value >= max_val - (max_val - min_val) * 0.9:
                        is_extreme = True
                        break

            if is_extreme:
                implications.append({
                    "type": "parameter_effect",
                    "description": f"Extreme value of {param} ({value}) leads to significant changes",
                    "affected_areas": self._generate_affected_areas(),
                    "severity": random.randint(3, 5)
                })
            else:
                implications.append({
                    "type": "parameter_effect",
                    "description": f"Value of {param} ({value}) has moderate effects",
                    "affected_areas": self._generate_affected_areas(max_areas=2),
                    "severity": random.randint(1, 3)
                })

        # Analyze unexpected condition implications
        for condition in case.get("unexpected_conditions", []):
            implications.append({
                "type": "unexpected_condition",
                "description": f"Unexpected condition '{condition.get('condition')}' for {condition.get('parameter')}",
                "affected_areas": self._generate_affected_areas(),
                "severity": random.randint(3, 5)
            })

        return implications

    def _generate_affected_areas(self, max_areas=3) -> List[str]:
        """Generate a list of affected areas for implications."""
        all_areas = [
            "social_structures", "economic_systems", "technology_development",
            "environmental_conditions", "human_psychology", "political_organization",
            "cultural_norms", "ethical_frameworks", "physical_infrastructure"
        ]
        
        num_areas = random.randint(1, max_areas)
        
