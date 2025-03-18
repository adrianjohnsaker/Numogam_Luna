import json
from typing import List, Dict, Any


class TensionAnalyzer:
    """Class to identify and analyze tensions between stakeholders or principles."""

    def __init__(self):
        """
        Initialize the TensionAnalyzer with common values and their potential conflicts.
        """
        self.common_values = {
            "individual_freedom": {
                "description": "Ability to act according to one's own will",
                "conflicts_with": ["collective_security", "equality", "social_harmony"]
            },
            "equality": {
                "description": "Equal distribution of resources or opportunities",
                "conflicts_with": ["meritocracy", "efficiency", "individual_freedom"]
            },
            "efficiency": {
                "description": "Optimal use of resources",
                "conflicts_with": ["equality", "stability", "deliberation"]
            },
            "sustainability": {
                "description": "Long-term preservation of resources and systems",
                "conflicts_with": ["short_term_growth", "innovation", "consumption"]
            },
            "tradition": {
                "description": "Adherence to established customs and practices",
                "conflicts_with": ["innovation", "adaptation", "individual_expression"]
            },
            "privacy": {
                "description": "Control over personal information and space",
                "conflicts_with": ["transparency", "security", "efficiency"]
            }
        }

    def identify_stakeholders(self, scenario: Dict) -> List[Dict]:
        """
        Identify key stakeholders in a scenario.
        
        Args:
            scenario: The scenario dictionary containing themes and specific stakeholders.
        
        Returns:
            A list of identified stakeholders relevant to the scenario.
        """
        stakeholders = []
        common_stakeholders = [
            {"id": "individuals", "interests": ["individual_freedom", "privacy", "security"]},
            {"id": "communities", "interests": ["social_harmony", "stability", "sustainability"]},
            {"id": "businesses", "interests": ["efficiency", "innovation", "growth"]},
            {"id": "government", "interests": ["order", "equality", "collective_security"]},
            {"id": "future_generations", "interests": ["sustainability", "opportunity", "preservation"]}
        ]

        # Filter stakeholders based on relevance to scenario themes
        for stakeholder in common_stakeholders:
            if any(interest in scenario.get("themes", []) for interest in stakeholder["interests"]):
                stakeholders.append(stakeholder)

        # Add scenario-specific stakeholders
        for specific_stakeholder in scenario.get("specific_stakeholders", []):
            stakeholders.append({
                "id": specific_stakeholder,
                "interests": scenario.get("stakeholder_interests", {}).get(specific_stakeholder, [])
            })

        return stakeholders

    def identify_core_principles(self, scenario: Dict) -> List[Dict]:
        """
        Identify core principles relevant to a scenario.
        
        Args:
            scenario: The scenario dictionary containing themes and specific principles.
        
        Returns:
            A list of identified principles relevant to the scenario.
        """
        principles = []
        
        # Identify common principles based on themes
        for theme in scenario.get("themes", []):
            if theme in self.common_values:
                principles.append({
                    "id": theme,
                    "description": self.common_values[theme]["description"]
                })

        # Add scenario-specific principles
        for specific_principle in scenario.get("specific_principles", []):
            principles.append({
                "id": specific_principle,
                "description": scenario.get("principle_descriptions", {}).get(specific_principle, "")
            })

        return principles

    def identify_conflicts(self, stakeholder1: Dict, stakeholder2: Dict, scenario: Dict) -> List[Dict]:
        """
        Identify conflicts between two stakeholders in the context of the scenario.
        
        Args:
            stakeholder1: The first stakeholder dictionary.
            stakeholder2: The second stakeholder dictionary.
            scenario: The scenario dictionary providing context.
        
        Returns:
            A list of identified conflicts between the two stakeholders.
        """
        conflicts = []

        # Check for conflicting interests
        for interest1 in stakeholder1["interests"]:
            for interest2 in stakeholder2["interests"]:
                if (interest1 in self.common_values and 
                    interest2 in self.common_values.get(interest1, {}).get("conflicts_with", [])):
                    conflicts.append({
                        "type": "interest_conflict",
                        "description": f"Conflict between {stakeholder1['id']}'s interest in {interest1} and "
                                       f"{stakeholder2['id']}'s interest in {interest2}",
                        # Example severity calculation
                        # Adjust severity as per your logic
                        'severity': 3  
                    })
