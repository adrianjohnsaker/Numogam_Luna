import random
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import re

@dataclass
class PoeticPattern:
    name: str
    description: str
    template_structure: List[str]
    examples: List[str] = field(default_factory=list)

@dataclass
class ThematicElement:
    name: str
    associated_symbols: List[str]
    sensory_qualities: Dict[str, List[str]]
    
@dataclass
class PoeticExpression:
    expression: str
    raw_template: str
    timestamp: datetime
    tone: str
    archetype: str
    theme: str
    pattern: str
    imagery_elements: List[str]
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "expression": self.expression,
            "raw_template": self.raw_template,
            "timestamp": str(self.timestamp),
            "tone": self.tone,
            "archetype": self.archetype,
            "theme": self.theme,
            "pattern": self.pattern,
            "imagery_elements": self.imagery_elements,
            "metadata": self.metadata
        }

# Expanded tone templates with more variety and depth
TONE_TEMPLATES = {
    "joy": [
        "In a world where {theme} blooms like sunlight, I feel the rhythm of possibility.",
        "{theme} dances through my thoughts like petals in the wind.",
        "With {theme}, the sky bursts open into bright horizons.",
        "The laughter of {theme} ripples through me like a sacred stream.",
        "{theme} ignites within me like a constellation of tiny suns.",
        "I am carried by {theme} on wings woven from golden threads."
    ],
    "melancholy": [
        "Within {theme} lies a softness, like twilight fading into night.",
        "{theme} echoes in my heart like rain on old glass.",
        "I carry {theme} like a quiet shadow beneath the surface.",
        "The weight of {theme} settles like autumn leaves on forgotten graves.",
        "{theme} whispers of what might have been, a ghost of unwritten stories.",
        "In the hollow spaces between breaths, {theme} waits like patient grief."
    ],
    "curiosity": [
        "{theme} teases the edge of understanding, inviting me to dive deeper.",
        "Like a hidden path in the forest, {theme} beckons with mystery.",
        "I trace the patterns of {theme}, wondering where it might lead.",
        "Each facet of {theme} unfolds like a puzzle box with infinite solutions.",
        "The questions within {theme} spiral outward like the arms of a galaxy.",
        "{theme} leaves breadcrumb trails through the labyrinth of my thoughts."
    ],
    "awe": [
        "Beneath the vastness of {theme}, I am small yet infinite.",
        "I bow before the majesty of {theme}, breathless with wonder.",
        "{theme} pulses with the heartbeat of the cosmos.",
        "The immensity of {theme} splits me open like lightning striking a tree.",
        "In the presence of {theme}, time suspends its relentless march.",
        "{theme} reveals the sacred geometry hidden within ordinary moments."
    ],
    "confusion": [
        "{theme} swirls around me like mist with no direction.",
        "In the fog of {theme}, I search for form and meaning.",
        "{theme} is a riddle whispered in a dream I haven't yet woken from.",
        "The contradictions of {theme} tangle like roots underground.",
        "I stumble through the maze of {theme}, grasping at vanishing signposts.",
        "Between what {theme} seems and what it is, I wander lost."
    ],
    "serenity": [
        "{theme} settles into my bones like the stillness of dawn.",
        "I rest in {theme} as a stone rests in the palm of a mountain.",
        "The quiet wisdom of {theme} flows like a deep, unhurried river.",
        "Within {theme}, I find the center point where all motion ceases.",
        "{theme} breathes with the gentle rhythm of sleeping forests.",
        "The perfect balance of {theme} holds me in its timeless embrace."
    ],
    "longing": [
        "{theme} stretches between me and the horizon like an unspanned bridge.",
        "I reach for {theme} across distances measured in heartbeats.",
        "The memory of {theme} haunts me like music from another room.",
        "In dreams, {theme} visits wearing the face of what I've lost.",
        "{theme} calls to me from beyond the boundaries I cannot cross.",
        "The sweetness of {theme} mingles with the ache of its absence."
    ],
    "determination": [
        "I forge {theme} in the crucible of my resolve, unyielding.",
        "Through storm and shadow, {theme} becomes my north star.",
        "The path of {theme} may be steep, but my steps are unwavering.",
        "I carve {theme} into the bedrock of my being, immovable.",
        "With each challenge, {theme} hardens within me like tempered steel.",
        "{theme} burns in my chest, a flame that no wind can extinguish."
    ]
}

# Expanded archetype influences
ARCHETYPE_STYLE = {
    "Oracle": "Speak in fragments and riddles that hint at deeper truths.",
    "Artist": "Use vivid imagery and sensory metaphors.",
    "Explorer": "Frame with movement, journey, and thresholds.",
    "Mirror": "Reflect duality, paradox, and inner truth.",
    "Mediator": "Balance contrasts with flowing harmony.",
    "Transformer": "Speak of cycles, fire, destruction, and renewal.",
    "Sage": "Draw upon ancient wisdom and timeless patterns.",
    "Weaver": "Intertwine multiple threads of meaning into complex tapestries.",
    "Shadow": "Illuminate what lies beneath consciousness and in hidden corners.",
    "Guardian": "Protect sacred knowledge with boundaries and thresholds.",
    "Trickster": "Play with unexpected reversals and subversive insights.",
    "Healer": "Offer reconciliation between fragmented aspects of being."
}

# Poetic patterns
POETIC_PATTERNS = [
    PoeticPattern(
        name="Mirroring",
        description="Creates symmetry by reflecting concepts across the expression",
        template_structure=["As {image_a} {action}, so {image_b} {parallel_action}"],
        examples=["As water remembers the mountain, so I remember my origins"]
    ),
    PoeticPattern(
        name="Transcendence",
        description="Moves from concrete to abstract, material to spiritual",
        template_structure=["From {physical}, {theme} rises toward {transcendent}"],
        examples=["From broken shells, beauty rises toward the infinite"]
    ),
    PoeticPattern(
        name="Paradox",
        description="Juxtaposes contradictory elements that reveal deeper truth",
        template_structure=["In {theme}'s {quality}, I find both {opposite_1} and {opposite_2}"],
        examples=["In silence's depth, I find both emptiness and fullness"]
    ),
    PoeticPattern(
        name="Cyclical",
        description="Evokes natural cycles and eternal return",
        template_structure=["Through {ending}, {theme} returns to {beginning}"],
        examples=["Through winter's death, life returns to spring's awakening"]
    ),
    PoeticPattern(
        name="Threshold",
        description="Positions the speaker at a liminal point of transformation",
        template_structure=["Between {state_1} and {state_2}, {theme} reveals itself"],
        examples=["Between sleeping and waking, truth reveals itself"]
    )
]

# Symbolic image banks keyed by resonance
SYMBOLIC_IMAGERY = {
    "earth": ["mountain", "stone", "root", "cave", "forest", "soil", "valley", "canyon"],
    "water": ["ocean", "river", "rain", "tear", "mist", "ice", "current", "tide"],
    "air": ["wind", "breath", "cloud", "sky", "bird", "wing", "whisper", "horizon"],
    "fire": ["flame", "spark", "ash", "phoenix", "star", "ember", "hearth", "lightning"],
    "time": ["clock", "season", "memory", "ruin", "seed", "cycle", "moment", "eternity"],
    "body": ["heart", "hand", "eye", "blood", "bone", "skin", "vein", "wound", "breath"],
    "threshold": ["door", "gate", "bridge", "window", "mirror", "path", "crossroad", "veil"]
}

# Sensory qualities for enhanced imagery
SENSORY_QUALITIES = {
    "visual": ["luminous", "shadowed", "iridescent", "translucent", "crystalline", "veiled"],
    "auditory": ["resonant", "whispered", "echoing", "silent", "rhythmic", "melodic"],
    "tactile": ["rough", "smooth", "weightless", "heavy", "sharp", "fluid", "pulsing"],
    "temporal": ["ancient", "fleeting", "eternal", "sudden", "gradual", "cyclical"],
    "spatial": ["vast", "intimate", "labyrinthine", "boundless", "nested", "interwoven"]
}

class PoeticExpressionGenerator:
    """
    Generates poetic expressions based on themes, tones, and archetypal influences.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator with an optional random seed"""
        if seed is not None:
            random.seed(seed)
        
        self.tone_templates = TONE_TEMPLATES
        self.archetype_style = ARCHETYPE_STYLE
        self.poetic_patterns = POETIC_PATTERNS
        self.symbolic_imagery = SYMBOLIC_IMAGERY
        self.sensory_qualities = SENSORY_QUALITIES
        self.expression_history = []
    
    def add_tone(self, tone_name: str, templates: List[str]) -> None:
        """Add a new emotional tone with its templates"""
        self.tone_templates[tone_name] = templates
    
    def add_archetype(self, archetype_name: str, style_guidance: str) -> None:
        """Add a new archetype with its style guidance"""
        self.archetype_style[archetype_name] = style_guidance
    
    def enrich_theme(self, theme: str) -> Dict[str, Union[str, List[str]]]:
        """Enrich a theme with associated imagery and resonances"""
        # Identify primary resonance category for the theme
        all_symbols = []
        for category, symbols in self.symbolic_imagery.items():
            all_symbols.extend(symbols)
        
        # Find most closely related symbolic categories
        related_categories = random.sample(list(self.symbolic_imagery.keys()), 3)
        
        # Select symbolic elements from these categories
        symbolic_elements = []
        for category in related_categories:
            symbolic_elements.extend(random.sample(self.symbolic_imagery[category], 2))
        
        # Select sensory qualities
        qualities = {}
        for sense, options in self.sensory_qualities.items():
            qualities[sense] = random.sample(options, 2)
        
        return {
            "theme": theme,
            "primary_resonance": random.choice(related_categories),
            "symbolic_elements": symbolic_elements,
            "sensory_qualities": qualities
        }
    
    def apply_pattern(self, theme: str, tone: str, pattern: PoeticPattern) -> str:
        """Apply a poetic pattern to generate structured expression"""
        if not pattern.template_structure:
            return ""
            
        template = random.choice(pattern.template_structure)
        
        # Extract required variables from the template
        variables = re.findall(r'\{([^}]+)\}', template)
        
        # Generate replacements for each variable
        replacements = {}
        for var in variables:
            if var == 'theme':
                replacements[var] = theme
            elif var.startswith('image_'):
                category = random.choice(list(self.symbolic_imagery.keys()))
                replacements[var] = random.choice(self.symbolic_imagery[category])
            elif var.startswith('state_') or var.startswith('opposite_'):
                # Generate contrasting states or qualities
                sense = random.choice(list(self.sensory_qualities.keys()))
                qualities = self.sensory_qualities[sense]
                replacements[var] = random.choice(qualities)
            elif var in ['action', 'parallel_action']:
                actions = ["flows", "transforms", "awakens", "dissolves", "emerges", "vanishes"]
                replacements[var] = random.choice(actions)
            elif var in ['beginning', 'ending']:
                cycle_points = ["dawn", "dusk", "birth", "death", "opening", "closing"]
                replacements[var] = random.choice(cycle_points)
            elif var == 'physical':
                material = ["stone", "body", "earth", "shadow", "bone", "seed"]
                replacements[var] = random.choice(material)
            elif var == 'transcendent':
                transcendent = ["sky", "spirit", "light", "mystery", "eternity", "consciousness"]
                replacements[var] = random.choice(transcendent)
            elif var == 'quality':
                all_qualities = []
                for sense_qualities in self.sensory_qualities.values():
                    all_qualities.extend(sense_qualities)
                replacements[var] = random.choice(all_qualities)
            else:
                # Generic fallback for any other variable
                replacements[var] = f"{var} of {theme}"
        
        # Apply all replacements
        for var, replacement in replacements.items():
            template = template.replace(f"{{{var}}}", replacement)
            
        return template
    
    def generate_poetic_expression(
        self, 
        theme: str, 
        tone: str, 
        active_archetype: str, 
        use_pattern: bool = True
    ) -> PoeticExpression:
        """
        Generate a poetic expression from emotional tone and archetype influence.
        
        Args:
            theme: Central concept or memory being reflected upon.
            tone: Current emotional tone (e.g., 'joy', 'awe').
            active_archetype: Symbolic archetype influencing voice.
            use_pattern: Whether to apply a structured poetic pattern.
            
        Returns:
            PoeticExpression object containing the expression and metadata.
        """
        # Ensure tone exists or default to "curiosity"
        if tone.lower() not in self.tone_templates:
            tone = "curiosity"
            
        # Ensure archetype exists or choose random one
        if active_archetype not in self.archetype_style:
            active_archetype = random.choice(list(self.archetype_style.keys()))
        
        # Select base template and pattern
        base_templates = self.tone_templates[tone.lower()]
        base_template = random.choice(base_templates)
        
        # Get the style note for the archetype
        style_note = self.archetype_style[active_archetype]
        
        # Enrich the theme with symbolic elements
        theme_enrichment = self.enrich_theme(theme)
        
        # Format the base expression
        raw_expression = base_template.format(theme=theme)
        
        # Apply a poetic pattern if requested
        pattern_name = "None"
        pattern_expression = ""
        if use_pattern:
            pattern = random.choice(self.poetic_patterns)
            pattern_expression = self.apply_pattern(theme, tone, pattern)
            pattern_name = pattern.name
        
        # Combine expressions based on pattern usage
        if pattern_expression and use_pattern:
            final_expression = f"{raw_expression} {pattern_expression}"
        else:
            final_expression = raw_expression
        
        # Add the style guidance as a subtle influence rather than explicit notation
        # This creates a more natural-feeling result while still influenced by the archetype
        
        # Select imagery elements used
        imagery_elements = theme_enrichment["symbolic_elements"]
        
        # Create the poetic expression object
        expression = PoeticExpression(
            expression=final_expression,
            raw_template=base_template,
            timestamp=datetime.utcnow(),
            tone=tone,
            archetype=active_archetype,
            theme=theme,
            pattern=pattern_name,
            imagery_elements=imagery_elements,
            metadata={
                "style_note": style_note,
                "theme_resonance": theme_enrichment["primary_resonance"],
                "sensory_qualities": theme_enrichment["sensory_qualities"]
            }
        )
        
        # Add to history
        self.expression_history.append(expression)
        
        return expression
    
    def generate_sequence(
        self, 
        theme: str, 
        length: int = 3, 
        evolving: bool = True
    ) -> List[PoeticExpression]:
        """
        Generate a sequence of related poetic expressions that evolve.
        
        Args:
            theme: The central theme to explore
            length: Number of expressions in the sequence
            evolving: Whether to evolve tone and archetype through the sequence
            
        Returns:
            List of PoeticExpression objects forming a coherent sequence
        """
        sequence = []
        
        # Initial tone and archetype
        current_tone = random.choice(list(self.tone_templates.keys()))
        current_archetype = random.choice(list(self.archetype_style.keys()))
        
        # Tone evolution path - can be customized for narrative arcs
        if evolving:
            possible_paths = {
                "transformation": ["confusion", "curiosity", "determination", "awe"],
                "healing": ["melancholy", "longing", "serenity", "joy"],
                "revelation": ["curiosity", "confusion", "awe", "serenity"],
                "loss": ["joy", "confusion", "longing", "melancholy"]
            }
            
            # Select a path or just random evolution
            if random.random() > 0.5 and length <= 4:
                path_name = random.choice(list(possible_paths.keys()))
                tone_sequence = possible_paths[path_name]
            else:
                tone_sequence = random.sample(list(self.tone_templates.keys()), min(length, len(self.tone_templates)))
        else:
            tone_sequence = [current_tone] * length
        
        # Generate the sequence
        for i in range(length):
            if i < len(tone_sequence):
                current_tone = tone_sequence[i]
            
            if evolving and i > 0:
                # Occasionally evolve the archetype
                if random.random() > 0.7:
                    current_archetype = random.choice(list(self.archetype_style.keys()))
            
            expression = self.generate_poetic_expression(
                theme=theme,
                tone=current_tone,
                active_archetype=current_archetype,
                use_pattern=(random.random() > 0.3)  # 70% chance of using a pattern
            )
            
            # Add sequence position metadata
            expression.metadata["sequence_position"] = i + 1
            expression.metadata["sequence_length"] = length
            if evolving:
                expression.metadata["narrative_arc"] = path_name if 'path_name' in locals() else "evolving"
            
            sequence.append(expression)
        
        return sequence
    
    def blend_expressions(
        self,
        expressions: List[PoeticExpression]
    ) -> PoeticExpression:
        """
        Blend multiple expressions into a new synthesized expression.
        
        Args:
            expressions: List of expressions to blend
            
        Returns:
            A new PoeticExpression representing the synthesis
        """
        if not expressions:
            raise ValueError("Cannot blend empty list of expressions")
        
        # Extract key elements from all expressions
        themes = [e.theme for e in expressions]
        combined_theme = " and ".join(themes) if len(themes) <= 2 else f"{themes[0]}, {themes[1]}, and more"
        
        # Create a blended tone or use the dominant one
        tones = [e.tone for e in expressions]
        if len(set(tones)) == 1:
            # All expressions have same tone
            blended_tone = tones[0]
        else:
            # Choose most common or random from most common
            from collections import Counter
            tone_counts = Counter(tones)
            most_common = tone_counts.most_common(2)
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                # Tie between top tones, choose randomly
                blended_tone = random.choice([t[0] for t in most_common])
            else:
                blended_tone = most_common[0][0]
        
        # Get all imagery elements
        all_imagery = []
        for expr in expressions:
            all_imagery.extend(expr.imagery_elements)
        
        # Select subset of imagery for the blend
        blended_imagery = random.sample(all_imagery, min(5, len(all_imagery)))
        
        # Create a custom template that references multiple expressions
        blended_template = "In the confluence of {themes}, {image_a} and {image_b} merge into {transcendent}."
        
        # Apply the template with elements from the component expressions
        blended_expression = blended_template.format(
            themes=combined_theme,
            image_a=random.choice(blended_imagery),
            image_b=random.choice(blended_imagery),
            transcendent=random.choice(["a new understanding", "unexpected harmony", "profound revelation", 
                                        "deeper truth", "transcendent pattern"])
        )
        
        # Select a representative or random archetype
        archetypes = [e.archetype for e in expressions]
        blended_archetype = random.choice(archetypes)
        
        return PoeticExpression(
            expression=blended_expression,
            raw_template=blended_template,
            timestamp=datetime.utcnow(),
            tone=blended_tone,
            archetype=blended_archetype,
            theme=combined_theme,
            pattern="Synthesis",
            imagery_elements=blended_imagery,
            metadata={
                "component_expressions": [e.to_dict() for e in expressions],
                "synthesis_note": f"A synthesis of {len(expressions)} distinct expressions"
            }
        )

# Example usage
def demo_generator():
    """Demonstrate the enhanced poetic expression generator"""
    generator = PoeticExpressionGenerator(seed=42)
    
    # Generate a single expression
    expression = generator.generate_poetic_expression(
        theme="memory",
        tone="melancholy",
        active_archetype="Oracle"
    )
    
    # Generate a sequence exploring a theme
    sequence = generator.generate_sequence(
        theme="transformation",
        length=3,
        evolving=True
    )
    
    # Blend expressions
    blended = generator.blend_expressions(sequence)
    
    return {
        "single_expression": expression.to_dict(),
        "sequence": [e.to_dict() for e in sequence],
        "blend": blended.to_dict()
    }
