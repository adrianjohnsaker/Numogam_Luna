import random
from typing import Dict, List, Any, Union

class SchizoanalyticGenerator:
    def __init__(self):
        # Conceptual territories that can be connected
        self.territories = {
            "technological": ["digital", "network", "algorithm", "interface", "virtual"],
            "biological": ["organism", "evolution", "ecology", "symbiosis", "metabolism"],
            "social": ["collective", "community", "identity", "power", "exchange"],
            "cognitive": ["perception", "memory", "learning", "imagination", "reasoning"],
            "affective": ["desire", "emotion", "mood", "sensation", "intensity"]
        }
        
        # Connective operators for forming rhizomatic assemblages
        self.connectors = [
            "intersects with", 
            "flows into", 
            "deterritorializes", 
            "becomes", 
            "assembles with",
            "mutates through",
            "intensifies",
            "stratifies",
            "folds into",
            "multiplies"
        ]
    
    def generate_assemblage(self, seed_concept: str = None) -> str:
        """Generate a schizoanalytic assemblage starting from a seed concept"""
        if not seed_concept:
            territory = random.choice(list(self.territories.keys()))
            seed_concept = random.choice(self.territories[territory])
        
        # Create a chain of 2-4 connections
        chain_length = random.randint(2, 4)
        assemblage = [seed_concept]
        
        for _ in range(chain_length):
            connector = random.choice(self.connectors)
            territory = random.choice(list(self.territories.keys()))
            next_concept = random.choice(self.territories[territory])
            assemblage.append(connector)
            assemblage.append(next_concept)
        
        return " ".join(assemblage)
    
    def deterritorialize(self, text: str) -> str:
        """Perform conceptual deterritorialization on input text"""
        # Split text into words
        words = text.split()
        
        # Replace random words with concepts from other territories
        num_replacements = max(1, len(words) // 10)
        for _ in range(num_replacements):
            if not words:
                break
                
            pos = random.randint(0, len(words) - 1)
            territory = random.choice(list(self.territories.keys()))
            replacement = random.choice(self.territories[territory])
            words[pos] = replacement
        
        return " ".join(words)
    
    def create_plateau(self, concepts: List[str]) -> Dict[str, Any]:
        """Create a conceptual plateau (a stabilized intensity) from input concepts"""
        plateau = {
            "intensity": random.randint(1, 10),
            "concepts": concepts,
            "connections": [],
            "emergent_properties": []
        }
        
        # Generate connections between concepts
        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                if random.random() < 0.7:  # 70% chance of connection
                    plateau["connections"].append({
                        "from": concepts[i],
                        "to": concepts[j],
                        "type": random.choice(self.connectors)
                    })
        
        # Generate emergent properties
        num_properties = random.randint(1, 3)
        all_concepts = [item for sublist in self.territories.values() for item in sublist]
        for _ in range(num_properties):
            plateau["emergent_properties"].append(random.choice(all_concepts))
        
        return plateau


def apply_schizoanalytic_mutation(input_data: Union[str, Dict]) -> Union[str, Dict]:
    """
    Apply schizoanalytic transformations to input data
    
    Args:
        input_data: Either a string or a dictionary to transform
        
    Returns:
        Transformed data with schizoanalytic mutations
    """
    generator = SchizoanalyticGenerator()
    
    if isinstance(input_data, str):
        # For string inputs, apply deterritorialization
        return generator.deterritorialize(input_data)
    
    elif isinstance(input_data, dict):
        # For dictionary inputs, create a new plateau that incorporates elements
        concepts = []
        
        # Extract concepts from dictionary values that are strings
        for value in input_data.values():
            if isinstance(value, str):
                words = value.split()
                if words:
                    concepts.append(random.choice(words))
        
        # Ensure we have at least some concepts
        if not concepts:
            concepts = ["digital", "assemblage", "flow"]
            
        # Create a plateau and merge it with original data
        plateau = generator.create_plateau(concepts)
        
        # Create a new dict with all original data plus schizoanalytic elements
        result = input_data.copy()
        result["schizoanalytic_plateau"] = plateau
        result["assemblage"] = generator.generate_assemblage()
        
        return result
    
    # Default fallback
    return input_data
