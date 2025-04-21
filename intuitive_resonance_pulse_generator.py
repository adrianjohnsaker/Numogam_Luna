import random
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ResonanceIntensity(Enum):
    SUBTLE = "subtle"
    MODERATE = "moderate"
    PROFOUND = "profound"
    OVERWHELMING = "overwhelming"

@dataclass
class ResonancePulse:
    tone: str
    form: str
    field: str
    intensity: str
    pulse: str
    harmonic_pattern: List[str]
    meta: str

class IntuitiveResonancePulseGenerator:
    """
    Generates symbolic resonance patterns representing intuitive and emotive states.
    Enhanced with richer vocabularies and configurability.
    """
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator with optional seed for reproducibility.
        
        Args:
            seed: Optional random seed for reproducible outputs
        """
        if seed is not None:
            random.seed(seed)
        
        self.tones = [
            "ethereal", "somber", "radiant", "fractured", "numinous", "luminous",
            "crystalline", "shadowed", "translucent", "iridescent", "gossamer", 
            "spectral", "opalescent", "prismatic", "cerulean", "amber", "verdant"
        ]
        
        self.forms = [
            "whisper", "echo", "flare", "tide", "fold", "reverberation",
            "pulse", "cascade", "spiral", "vortex", "tapestry", "mandala", 
            "helix", "ripple", "weave", "lattice", "tessellation", "chord"
        ]
        
        self.fields = [
            "dreamspace", "symbolic lattice", "inner glyphwork", "emotive veil", 
            "mythic core", "resonance field", "liminal threshold", "quantum foam", 
            "collective unconscious", "archetypal sea", "noetic garden", 
            "ancestral memory", "synchronistic web", "akashic terrain"
        ]
        
        self.movements = [
            "stirring", "catalyzing", "unveiling", "weaving", "awakening",
            "transmuting", "illuminating", "crystallizing", "unfurling", "dissolving"
        ]
        
        self.emergences = [
            "a new wave of becoming", "an unforeseen potential", "a dormant knowing",
            "a forgotten memory", "an unexpected revelation", "a deepening awareness",
            "an essential pattern", "a resonant truth", "a harmonious integration"
        ]
        
        self.patterns = {
            "spiral": ["beginning", "expansion", "return", "integration"],
            "wave": ["crest", "peak", "descent", "trough", "rise"],
            "pulse": ["contraction", "expansion", "rest", "renewal"],
            "bifurcation": ["division", "choice", "parallel", "convergence"],
            "fractal": ["seed", "iteration", "self-similarity", "complexity"]
        }

    def generate_pulse(self, custom_meta: Optional[str] = None) -> ResonancePulse:
        """
        Generate a resonance pulse with symbolic imagery.
        
        Args:
            custom_meta: Optional custom metadata string
            
        Returns:
            ResonancePulse object containing resonance attributes
        """
        tone = random.choice(self.tones)
        form = random.choice(self.forms)
        field = random.choice(self.fields)
        movement = random.choice(self.movements)
        emergence = random.choice(self.emergences)
        pattern_type = random.choice(list(self.patterns.keys()))
        intensity = random.choice([e.value for e in ResonanceIntensity])
        
        phrase = f"A {tone} {form} passes through the {field}, {movement} {emergence}."
        
        meta = custom_meta if custom_meta else "An intuitive resonance pattern emerging from the felt symbolic state."
        
        return ResonancePulse(
            tone=tone,
            form=form,
            field=field,
            intensity=intensity,
            pulse=phrase,
            harmonic_pattern=self.patterns[pattern_type],
            meta=meta
        )
    
    def generate_sequence(self, length: int = 3) -> List[ResonancePulse]:
        """
        Generate a sequence of related resonance pulses.
        
        Args:
            length: Number of pulses in the sequence
            
        Returns:
            List of ResonancePulse objects
        """
        sequence = []
        # Use same field for consistency in the sequence
        field = random.choice(self.fields)
        pattern_type = random.choice(list(self.patterns.keys()))
        
        for i in range(length):
            tone = random.choice(self.tones)
            form = random.choice(self.forms)
            movement = random.choice(self.movements)
            emergence = random.choice(self.emergences)
            intensity = random.choice([e.value for e in ResonanceIntensity])
            
            phrase = f"A {tone} {form} passes through the {field}, {movement} {emergence}."
            meta = f"Sequence resonance {i+1}/{length} - {pattern_type} pattern unfolding"
            
            pulse = ResonancePulse(
                tone=tone,
                form=form,
                field=field,
                intensity=intensity,
                pulse=phrase,
                harmonic_pattern=self.patterns[pattern_type],
                meta=meta
            )
            sequence.append(pulse)
            
        return sequence
    
    def add_vocabulary(self, category: str, new_terms: List[str]) -> None:
        """
        Add new vocabulary terms to the specified category.
        
        Args:
            category: The vocabulary category to expand ('tones', 'forms', 'fields', etc.)
            new_terms: List of new terms to add
        """
        if hasattr(self, category):
            current_list = getattr(self, category)
            if isinstance(current_list, list):
                setattr(self, category, current_list + new_terms)
            else:
                raise TypeError(f"Category '{category}' is not a list type")
        else:
            raise ValueError(f"Category '{category}' does not exist")

    def create_harmonic_blend(self, pulses: List[ResonancePulse]) -> ResonancePulse:
        """
        Create a harmonic blend of multiple pulses into a new emergent pulse.
        
        Args:
            pulses: List of pulses to blend
            
        Returns:
            A new ResonancePulse representing the harmonic integration
        """
        # Extract all components from input pulses
        all_tones = [p.tone for p in pulses]
        all_forms = [p.form for p in pulses]
        all_fields = [p.field for p in pulses]
        
        # Select elements that create harmony
        harmonic_tone = random.choice(all_tones)
        harmonic_form = random.choice(all_forms)
        harmonic_field = random.choice(all_fields)
        
        # Create a more complex blended phrase
        blend_phrase = f"The {harmonic_tone} {harmonic_form} interweaves across {harmonic_field}, "
        blend_phrase += f"harmonizing {len(pulses)} distinct resonances into a unified field of emergence."
        
        # Combine pattern elements from all source pulses
        harmonic_pattern = []
        for pulse in pulses:
            harmonic_pattern.extend(pulse.harmonic_pattern)
        harmonic_pattern = list(dict.fromkeys(harmonic_pattern))  # Remove duplicates while preserving order
        
        return ResonancePulse(
            tone=harmonic_tone,
            form=harmonic_form,
            field=harmonic_field,
            intensity="harmonic",
            pulse=blend_phrase,
            harmonic_pattern=harmonic_pattern,
            meta="A harmonic integration of multiple resonance patterns"
        )
