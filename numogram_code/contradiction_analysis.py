from typing import List, Dict, Any, Tuple


class TensionAnalyzer:
    """Class to identify and analyze tensions between stakeholders or principles."""
    
    def __init__(self):
        # Common values and interests that might come into conflict
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
        """Identify key stakeholders in a scenario."""
        # In a real implementation, this would use NLP or a knowledge graph
        # to extract relevant stakeholders from the scenario description
        
        # Placeholder implementation
        stakeholders = []
        common_stakeholders = [
            {"id": "individuals", "interests": ["individual_freedom", "privacy", "security"]},
            {"id": "communities", "interests": ["social_harmony", "stability", "sustainability"]},
            {"id": "businesses", "interests": ["efficiency", "innovation", "growth"]},
            {"id": "government", "interests": ["order", "equality", "collective_security"]},
            {"id": "future_generations", "interests": ["sustainability", "opportunity", "preservation"]}
        ]
        
        # Filter stakeholders based on relevance to scenario
        # This would be more sophisticated in a real implementation
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
        """Identify core principles relevant to a scenario."""
        # This would use content analysis to extract principles in a real system
        
        # Placeholder implementation
        principles = []
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
        """Identify conflicts between two stakeholders in the context of the scenario."""
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
                        "severity": self._calculate_conflict_severity(interest1, interest2, scenario)
                    })
                    
        # Check for resource competition
        if any(resource in stakeholder1.get("required_resources", []) for resource in stakeholder2.get("required_resources", [])):
            conflicts.append({
                "type": "resource_competition",
                "description": f"Competition for resources between {stakeholder1['id']} and {stakeholder2['id']}",
                "severity": 3  # Default medium severity
            })
            
        return conflicts
    
    def identify_contradictions(self, principle1: Dict, principle2: Dict, scenario: Dict) -> List[Dict]:
        """Identify contradictions between two principles in the context of the scenario."""
        contradictions = []
        
        # Check if principles are known to conflict
        if (principle1["id"] in self.common_values and 
            principle2["id"] in self.common_values.get(principle1["id"], {}).get("conflicts_with", [])):
            contradictions.append({
                "type": "inherent_contradiction",
                "description": f"Inherent tension between {principle1['id']} and {principle2['id']}",
                "severity": self._calculate_contradiction_severity(principle1["id"], principle2["id"], scenario)
            })
            
        # Check for scenario-specific contradictions
        specific_contradictions = scenario.get("specific_contradictions", {})
        if (principle1["id"], principle2["id"]) in specific_contradictions or (principle2["id"], principle1["id"]) in specific_contradictions:
            contradiction_key = (principle1["id"], principle2["id"]) if (principle1["id"], principle2["id"]) in specific_contradictions else (principle2["id"], principle1["id"])
            contradictions.append({
                "type": "scenario_specific",
                "description": specific_contradictions[contradiction_key],
                "severity": 4  # Higher severity for scenario-specific contradictions
            })
            
        return contradictions
    
    def _calculate_conflict_severity(self, interest1: str, interest2: str, scenario: Dict) -> int:
        """Calculate the severity of a conflict between interests."""
        # In a real implementation, this would use more sophisticated reasoning
        # based on the importance of the interests in the scenario context
        
        # Simple implementation based on scenario emphasis
        severity = 3  # Default medium severity
        
        emphasis = scenario.get("emphasis", {})
        if interest1 in emphasis and interest2 in emphasis:
            # If both interests are emphasized, conflict is more severe
            severity += min(emphasis[interest1], emphasis[interest2])
            
        return min(severity, 5)  # Cap at 5
    
    def _calculate_contradiction_severity(self, principle1: str, principle2: str, scenario: Dict) -> int:
        """Calculate the severity of a contradiction between principles."""
        # Similar to conflict severity, but for principles
        severity = 3  # Default medium severity
        
        emphasis = scenario.get("emphasis", {})
        if principle1 in emphasis and principle2 in emphasis:
            # If both principles are emphasized, contradiction is more severe
            severity += min(emphasis[principle1], emphasis[principle2])
            
        return min(severity, 5)  # Cap at 5
    
    def generate_resolution_strategies(self, tensions: List[Dict]) -> List[Dict]:
        """Generate strategies for resolving or managing identified tensions."""
        strategies = []
        
        for tension in tensions:
            if "parties" in tension:  # Stakeholder conflict
                strategies.append({
                    "for_tension": tension["parties"][0]["id"] + " vs " + tension["parties"][1]["id"],
                    "approach": "Stakeholder negotiation",
                    "description": "Facilitated process to find common ground and compromise",
                    "pros": ["Builds relationships", "Creates buy-in", "May find win-win solutions"],
                    "cons": ["Time-consuming", "May result in lowest-common-denominator solutions"]
                })
                
                strategies.append({
                    "for_tension": tension["parties"][0]["id"] + " vs " + tension["parties"][1]["id"],
                    "approach": "Incentive alignment",
                    "description": "Redesign incentives to align interests of conflicting parties",
                    "pros": ["Addresses root causes", "Can be self-sustaining", "Reduces oversight needs"],
                    "cons": ["Complex to design correctly", "May have unintended consequences"]
                })
                
            elif "principles" in tension:  # Principle contradiction
                strategies.append({
                    "for_tension": tension["principles"][0]["id"] + " vs " + tension["principles"][1]["id"],
                    "approach": "Contextual prioritization",
                    "description": "Establish clear hierarchy of principles in specific contexts",
                    "pros": ["Provides clear guidance", "Acknowledges complexity of values"],
                    "cons": ["May still require case-by-case judgment", "Difficult to gain consensus"]
                })
                
                strategies.append({
                    "for_tension": tension["principles"][0]["id"] + " vs " + tension["principles"][1]["id"],
                    "approach": "Synthesis and reframing",
                    "description": "Develop new framework that reconciles seemingly contradictory principles",
                    "pros": ["Can lead to innovative solutions", "Addresses deeper issues"],
                    "cons": ["May be too abstract", "Difficult to implement in practice"]
                })
                
        return strategies
    
    def analyze_competing_interests(self, scenario: Dict) -> Dict:
        """Identify and analyze tensions between different stakeholders or principles."""
        stakeholders = self.identify_stakeholders(scenario)
        principles = self.identify_core_principles(scenario)
        
        # Create tension matrix
        tensions = []
        for i, s1 in enumerate(stakeholders):
            for s2 in stakeholders[i+1:]:
                conflicts = self.identify_conflicts(s1, s2, scenario)
                if conflicts:
                    tensions.append({"parties": [s1, s2], "conflicts": conflicts})
        
        for i, p1 in enumerate(principles):
            for p2 in principles[i+1:]:
                contradictions = self.identify_contradictions(p1, p2, scenario)
                if contradictions:
                    tensions.append({"principles": [p1, p2], "contradictions": contradictions})
        
        resolution_approaches = self.generate_resolution_strategies(tensions)
        
        return {
            "stakeholders": stakeholders,
            "principles": principles,
            "tensions": tensions, 
            "resolution_approaches": resolution_approaches
        }


# Usage example
if __name__ == "__main__":
    analyzer = TensionAnalyzer()
    
    # Example scenario: Photosynthetic humans
    scenario = {
        "themes": ["sustainability", "equality", "efficiency"],
        "specific_stakeholders": ["urban_dwellers", "rural_populations"],
        "stakeholder_interests": {
            "urban_dwellers": ["efficiency", "innovation"],
            "rural_populations": ["sustainability", "tradition"]
        },
        "specific_principles": ["sunlight_access", "energy_independence"],
        "principle_descriptions": {
            "sunlight_access": "Right to adequate sunlight for energy generation",
            "energy_independence": "Freedom from reliance on external energy sources"
        },
        "emphasis": {
            "sustainability": 4,
            "efficiency": 3,
            "sunlight_access": 5
        },
        "specific_contradictions": {
            ("efficiency", "sunlight_access"): "Dense urban living maximizes efficiency but may limit sunlight access"
        }
    }
    
    analysis_result = analyzer.analyze_competing_interests(scenario)
    
    # Print results in a readable format
    import json
    print(json.dumps(analysis_result, indent=2))
