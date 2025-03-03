import logging
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SIGNIFICANCE_THRESHOLD = 0.6
SIMILARITY_THRESHOLD = 0.7
FEEDBACK_LOOP_MIN_SIZE = 3
FEEDBACK_LOOP_MAX_SIZE = 7

class Effect:
    """Representation of a consequence or effect."""
    
    def __init__(self, description, domain, magnitude=0.5, likelihood=0.5, timeframe="medium", parent=None):
        """
        Initialize an effect.
        
        Args:
            description (str): Description of the effect
            domain (str): Domain of the effect (social, economic, etc.)
            magnitude (float): Impact magnitude from 0-1
            likelihood (float): Probability of occurrence from 0-1
            timeframe (str): When the effect is expected (short, medium, long)
            parent (Effect, optional): The effect that caused this one
        """
        self.description = description
        self.domain = domain
        self.magnitude = magnitude
        self.likelihood = likelihood
        self.timeframe = timeframe
        self.parent = parent
        self.children = []
        self.id = id(self)  # Unique identifier
        
        # Add this effect as a child to its parent
        if parent:
            parent.children.append(self)
    
    def __repr__(self):
        return f"Effect({self.description[:30]}... [{self.domain}], mag={self.magnitude:.2f}, prob={self.likelihood:.2f})"
    
    @property
    def expected_impact(self):
        """Calculate the expected impact as magnitude * likelihood."""
        return self.magnitude * self.likelihood

class FeedbackLoop:
    """Representation of a feedback loop in consequence chains."""
    
    def __init__(self, effects, is_reinforcing=True):
        """
        Initialize a feedback loop.
        
        Args:
            effects (list): List of Effect objects in the loop
            is_reinforcing (bool): True if reinforcing, False if balancing
        """
        self.effects = effects
        self.is_reinforcing = is_reinforcing
        
    def __repr__(self):
        loop_type = "Reinforcing" if self.is_reinforcing else "Balancing"
        return f"{loop_type}Loop({len(self.effects)} effects, starting with: {self.effects[0].description[:30]}...)"

class EmergentPattern:
    """Representation of an emergent pattern across consequence chains."""
    
    def __init__(self, name, description, related_effects, confidence=0.5):
        """
        Initialize an emergent pattern.
        
        Args:
            name (str): Short name for the pattern
            description (str): Description of the pattern
            related_effects (list): List of Effect objects related to this pattern
            confidence (float): Confidence level in this pattern from 0-1
        """
        self.name = name
        self.description = description
        self.related_effects = related_effects
        self.confidence = confidence
        
    def __repr__(self):
        return f"EmergentPattern({self.name}, {len(self.related_effects)} effects, conf={self.confidence:.2f})"

def identify_social_consequences(effect, num_consequences=2):
    """
    Identify social consequences of an effect.
    
    Args:
        effect (Effect): The cause effect
        num_consequences (int): Number of consequences to generate
        
    Returns:
        list: List of Effect objects
    """
    logger.debug(f"Identifying social consequences for: {effect.description[:50]}...")
    
    # In a real implementation, this would use LLM or knowledge base
    # Here we'll provide a simplified implementation with templates
    
    social_templates = [
        "Changes in community structures as people adapt to {effect}",
        "Shift in social norms regarding {domain} due to {effect}",
        "Creation of new social movements advocating for/against {effect}",
        "Changes in family dynamics resulting from {effect}",
        "Development of new social practices to accommodate {effect}",
        "Emergence of social inequality related to access to {domain}",
        "Changes in educational practices to prepare people for {effect}",
        "Creation of new social identities related to {effect}",
        "Shifts in power dynamics between social groups because of {effect}"
    ]
    
    consequences = []
    selected_templates = np.random.choice(social_templates, size=min(num_consequences, len(social_templates)), replace=False)
    
    for template in selected_templates:
        description = template.format(effect=effect.description.lower(), domain=effect.domain.lower())
        
        # Randomize parameters slightly based on parent
        magnitude = min(1.0, max(0.1, effect.magnitude * np.random.uniform(0.7, 1.3)))
        likelihood = min(1.0, max(0.1, effect.likelihood * np.random.uniform(0.7, 1.3)))
        
        # Timeframe tends to extend
        timeframe_map = {"short": "medium", "medium": "long", "long": "long"}
        timeframe = timeframe_map.get(effect.timeframe, "medium")
        
        consequence = Effect(description, "Social", magnitude, likelihood, timeframe, effect)
        consequences.append(consequence)
    
    return consequences

def identify_economic_consequences(effect, num_consequences=2):
    """
    Identify economic consequences of an effect.
    
    Args:
        effect (Effect): The cause effect
        num_consequences (int): Number of consequences to generate
        
    Returns:
        list: List of Effect objects
    """
    logger.debug(f"Identifying economic consequences for: {effect.description[:50]}...")
    
    economic_templates = [
        "Changes in market dynamics related to {domain} because of {effect}",
        "Creation of new business opportunities in response to {effect}",
        "Shifts in employment patterns as industries adapt to {effect}",
        "Changes in consumer behavior regarding {domain} products",
        "Development of new economic models to account for {effect}",
        "Disruption of supply chains related to {domain}",
        "Changes in investment patterns as markets react to {effect}",
        "Emergence of new economic regulations to address {effect}",
        "Redistribution of wealth due to winners and losers from {effect}"
    ]
    
    consequences = []
    selected_templates = np.random.choice(economic_templates, size=min(num_consequences, len(economic_templates)), replace=False)
    
    for template in selected_templates:
        description = template.format(effect=effect.description.lower(), domain=effect.domain.lower())
        
        # Randomize parameters slightly based on parent
        magnitude = min(1.0, max(0.1, effect.magnitude * np.random.uniform(0.7, 1.3)))
        likelihood = min(1.0, max(0.1, effect.likelihood * np.random.uniform(0.7, 1.3)))
        
        consequence = Effect(description, "Economic", magnitude, likelihood, effect.timeframe, effect)
        consequences.append(consequence)
    
    return consequences

def identify_technological_consequences(effect, num_consequences=2):
    """
    Identify technological consequences of an effect.
    
    Args:
        effect (Effect): The cause effect
        num_consequences (int): Number of consequences to generate
        
    Returns:
        list: List of Effect objects
    """
    logger.debug(f"Identifying technological consequences for: {effect.description[:50]}...")
    
    tech_templates = [
        "Development of new technologies to address challenges from {effect}",
        "Adaptation of existing technologies to accommodate {effect}",
        "Changes in technology adoption patterns due to {effect}",
        "Emergence of technical standards related to {domain} because of {effect}",
        "Integration of {domain} technologies with others to respond to {effect}",
        "Creation of new research directions inspired by {effect}",
        "Technical obsolescence of systems unable to adapt to {effect}",
        "Development of monitoring or measurement systems for {effect}",
        "Creation of new platforms or infrastructure to support changes from {effect}"
    ]
    
    consequences = []
    selected_templates = np.random.choice(tech_templates, size=min(num_consequences, len(tech_templates)), replace=False)
    
    for template in selected_templates:
        description = template.format(effect=effect.description.lower(), domain=effect.domain.lower())
        
        # Technological consequences often have higher variance
        magnitude = min(1.0, max(0.1, effect.magnitude * np.random.uniform(0.5, 1.5)))
        likelihood = min(1.0, max(0.1, effect.likelihood * np.random.uniform(0.5, 1.5)))
        
        consequence = Effect(description, "Technological", magnitude, likelihood, effect.timeframe, effect)
        consequences.append(consequence)
    
    return consequences

def identify_environmental_consequences(effect, num_consequences=2):
    """
    Identify environmental consequences of an effect.
    
    Args:
        effect (Effect): The cause effect
        num_consequences (int): Number of consequences to generate
        
    Returns:
        list: List of Effect objects
    """
    logger.debug(f"Identifying environmental consequences for: {effect.description[:50]}...")
    
    env_templates = [
        "Changes in resource consumption patterns due to {effect}",
        "Impact on biodiversity resulting from {effect}",
        "Changes in land use patterns associated with {effect}",
        "Shifts in pollution patterns related to {domain} activities",
        "Adaptation of ecosystems to accommodate {effect}",
        "Changes in water usage or availability because of {effect}",
        "Impact on climate factors related to {effect}",
        "Changes in energy consumption patterns due to {effect}",
        "Environmental policy responses to address {effect}"
    ]
    
    consequences = []
    selected_templates = np.random.choice(env_templates, size=min(num_consequences, len(env_templates)), replace=False)
    
    for template in selected_templates:
        description = template.format(effect=effect.description.lower(), domain=effect.domain.lower())
        
        # Environmental consequences often have long timeframes
        magnitude = min(1.0, max(0.1, effect.magnitude * np.random.uniform(0.7, 1.3)))
        likelihood = min(1.0, max(0.1, effect.likelihood * np.random.uniform(0.7, 1.3)))
        
        # Environmental effects more likely to be long-term
        timeframe_shift = np.random.choice(["same", "longer", "longer"], p=[0.3, 0.4, 0.3])
        if timeframe_shift == "same":
            timeframe = effect.timeframe
        elif effect.timeframe == "short":
            timeframe = "medium"
        else:
            timeframe = "long"
        
        consequence = Effect(description, "Environmental", magnitude, likelihood, timeframe, effect)
        consequences.append(consequence)
    
    return consequences

def filter_by_significance_and_uniqueness(effects, threshold=SIGNIFICANCE_THRESHOLD):
    """
    Filter effects based on significance and uniqueness.
    
    Args:
        effects (list): List of Effect objects
        threshold (float): Significance threshold
        
    Returns:
        list: Filtered list of Effect objects
    """
    # Filter by significance (expected impact)
    significant = [e for e in effects if e.expected_impact >= threshold]
    
    # Filter for uniqueness
    unique_effects = []
    descriptions = []
    
    for effect in significant:
        is_unique = True
        
        for existing_desc in descriptions:
            # Simple string similarity check
            # In a real implementation, this would use semantic similarity
            if string_similarity(effect.description, existing_desc) > SIMILARITY_THRESHOLD:
                is_unique = False
                break
                
        if is_unique:
            unique_effects.append(effect)
            descriptions.append(effect.description)
