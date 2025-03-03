from typing import List, Dict, Any
import random


class EdgeCaseExplorer:
    """Class to generate and analyze extreme or boundary conditions in scenarios."""
    
    def __init__(self):
        self.THRESHOLD = 3  # Minimum significance threshold (1-5 scale)
    
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
            # Generate generic unexpected condition if none provided
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
        # This would use more sophisticated causal reasoning in a real implementation
        
        implications = []
        
        # Generate implications based on case parameters and conditions
        parameters = case.get("parameters", {})
        unexpected_conditions = case.get("unexpected_conditions", [])
        
        # Analyze parameter-based implications
        for param, value in parameters.items():
            # Check if the parameter is at an extreme value
            is_extreme = False
            for assumption in case.get("baseline_assumptions", []):
                if assumption.get("parameter") == param:
                    min_val = assumption.get("min_value", 0)
                    max_val = assumption.get("max_value", 100)
                    if value <= min_val + (max_val - min_val) * 0.1 or value >= max_val - (max_val - min_val) * 0.1:
                        is_extreme = True
                        break
            
            if is_extreme:
                implications.append({
                    "type": "parameter_effect",
                    "description": f"Extreme value of {param} ({value}) leads to significant changes",
                    "affected_areas": self._generate_affected_areas(),
                    "severity": random.randint(3, 5)  # Higher severity for extreme values
                })
            else:
                implications.append({
                    "type": "parameter_effect",
                    "description": f"Value of {param} ({value}) has moderate effects",
                    "affected_areas": self._generate_affected_areas(max_areas=2),  # Fewer affected areas
                    "severity": random.randint(1, 3)  # Lower severity for non-extreme values
                })
        
        # Analyze unexpected condition implications
        for condition in unexpected_conditions:
            implications.append({
                "type": "unexpected_condition",
                "description": f"Unexpected condition '{condition.get('condition')}' for {condition.get('parameter')}",
                "affected_areas": self._generate_affected_areas(),
                "severity": random.randint(3, 5)  # Higher severity for unexpected conditions
            })
            
        return implications
    
    def _generate_affected_areas(self, max_areas=3):
        """Generate a list of affected areas for implications."""
        all_areas = [
            "social_structures", "economic_systems", "technology_development", 
            "environmental_conditions", "human_psychology", "political_organization",
            "cultural_norms", "ethical_frameworks", "physical_infrastructure"
        ]
        
        # Select a random number of areas up to max_areas
        num_areas = random.randint(1, max_areas)
        return random.sample(all_areas, num_areas)
    
    def estimate_likelihood(self, case: Dict) -> float:
        """Estimate the likelihood of an edge case scenario."""
        # Simple implementation that reduces likelihood for extreme values and unexpected conditions
        
        # Start with moderate likelihood
        likelihood = 0.5
        
        # Adjust based on parameter extremity
        parameters = case.get("parameters", {})
        for param, value in parameters.items():
            for assumption in case.get("baseline_assumptions", []):
                if assumption.get("parameter") == param:
                    min_val = assumption.get("min_value", 0)
                    max_val = assumption.get("max_value", 100)
                    normalized_value = (value - min_val) / (max_val - min_val)
                    extremity = abs(normalized_value - 0.5) * 2  # 0 at middle, 1 at extremes
                    likelihood -= extremity * 0.1  # Reduce likelihood for extreme values
        
        # Adjust based on unexpected conditions
        unexpected_conditions = case.get("unexpected_conditions", [])
        likelihood -= len(unexpected_conditions) * 0.15  # Reduce likelihood for each unexpected condition
        
        # Ensure likelihood is between 0.05 and 0.95
        return max(0.05, min(0.95, likelihood))
    
    def evaluate_significance(self, implications: List[Dict]) -> float:
        """Evaluate the significance of a set of implications."""
        # Simple implementation based on severity and number of affected areas
        
        if not implications:
            return 0
        
        total_severity = sum(implication.get("severity", 0) for implication in implications)
        total_areas = sum(len(implication.get("affected_areas", [])) for implication in implications)
        
        # Calculate significance as a combination of severity and breadth of impact
        significance = (total_severity / len(implications)) * (1 + 0.1 * total_areas)
        
        # Scale to 1-5
        return min(5, max(1, significance))
    
    def explore_edge_cases(self, scenario: Dict, baseline_assumptions: List[Dict]) -> Dict:
        """Generate and analyze extreme or boundary conditions."""
        edge_cases = []
        
        # Add baseline assumptions to scenario for reference
        scenario["baseline_assumptions"] = baseline_assumptions
        
        # Generate edge cases by systematically varying parameters
        for assumption in baseline_assumptions:
            min_case = self.create_minimum_scenario(assumption, scenario)
            max_case = self.create_maximum_scenario(assumption, scenario)
            unusual_case = self.create_unexpected_scenario(assumption, scenario)
            
            edge_cases.extend([min_case, max_case, unusual_case])
        
        # Analyze implications of each edge case
        analyzed_cases = []
        for case in edge_cases:
            implications = self.analyze_implications(case)
            likelihood = self.estimate_likelihood(case)
            significance = self.evaluate_significance(implications)
            
            if significance >= self.THRESHOLD:
                analyzed_cases.append({
                    "case": case,
                    "implications": implications,
                    "likelihood": likelihood,
                    "significance": significance
                })
        
        # Sort cases by significance (descending)
        analyzed_cases.sort(key=lambda x: x["significance"], reverse=True)
        
        return {
            "original_scenario": scenario,
            "edge_cases": analyzed_cases,
            "total_cases_analyzed": len(edge_cases),
            "significant_cases_found": len(analyzed_cases)
        }


# Usage example
if __name__ == "__main__":
    explorer = EdgeCaseExplorer()
    
    # Example scenario: Photosynthetic humans
    scenario = {
        "name": "Photosynthetic Humans",
        "description": "Humans have evolved photosynthetic capabilities similar to plants",
        "parameters": {
            "photosynthetic_efficiency": 50,  # Percentage of energy needs met through photosynthesis
            "sunlight_requirements": 4,  # Hours of direct sunlight needed per day
            "adoption_rate": 70  # Percentage of population with the capability
        }
    }
    
    # Define baseline assumptions
    baseline_assumptions = [
        {
            "parameter": "photosynthetic_efficiency",
            "min_value": 10,
            "max_value": 90,
            "unexpected_conditions": [
                "efficiency varies significantly by individual genetics",
                "efficiency degrades with age",
                "efficiency is enhanced by certain medications"
            ]
        },
        {
            "parameter": "sunlight_requirements",
            "min_value": 1,
            "max_value": 12,
            "unexpected_conditions": [
                "artificial light cannot substitute for sunlight",
                "requirements fluctuate seasonally",
                "requirements increase during periods of stress"
            ]
        },
        {
            "parameter": "adoption_rate",
            "min_value": 5,
            "max_value": 100,
            "unexpected_conditions": [
                "adoption creates new social classes",
                "adoption concentrates in specific geographic regions",
                "adoption leads to religious movements"
            ]
        }
    ]
    
    edge_case_results = explorer.explore_edge_cases(scenario, baseline_assumptions)
    
    # Print results in a readable format
    import json
    # Print just the first two significant cases for brevity
    simplified_results = {
        "original_scenario": edge_case_results["original_scenario"],
        "significant_cases_found": edge_case_results["significant_cases_found"],
        "top_edge_cases": edge_case_results["edge_cases"][:2]
    }
    print(json.dumps(simplified_results, indent=2))
