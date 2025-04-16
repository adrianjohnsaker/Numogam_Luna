"""
Poetic Syntax Rewriter - Module for recursive transformation of sentence structures
Enables recursive transformation of syntax based on poetic principles, glyph influences,
and Deleuzian concepts of minor literature.
"""

import random
import re
import uuid
import datetime
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Set, Union

class PoeticSyntaxRewriter:
    """
    Enables recursive transformation of sentence structures based on
    poetic principles, glyph influences, and concepts of minor literature.
    """
    
    def __init__(self, config_path: Optional[str] = None, memory_path: Optional[str] = None):
        """
        Initialize the PoeticSyntaxRewriter with optional configuration.
        
        Args:
            config_path: Path to configuration file
            memory_path: Path to memory storage directory
        """
        self.id = str(uuid.uuid4())
        self.created_at = datetime.datetime.utcnow().isoformat()
        self.transformation_log = []
        
        # Syntax pattern libraries
        self.syntax_patterns = {
            "standard": [
                "SUBJECT VERB OBJECT",
                "SUBJECT VERB",
                "SUBJECT VERB ADVERB",
                "ADJECTIVE SUBJECT VERB OBJECT",
                "SUBJECT VERB OBJECT PREPOSITION OBJECT"
            ],
            "inverted": [
                "OBJECT SUBJECT VERB",
                "VERB SUBJECT",
                "ADVERB SUBJECT VERB",
                "OBJECT, SUBJECT VERB",
                "PREPOSITION OBJECT SUBJECT VERB"
            ],
            "fragmented": [
                "SUBJECT... VERB... OBJECT",
                "VERB -- SUBJECT -- OBJECT",
                "OBJECT: SUBJECT VERB",
                "SUBJECT, VERB; OBJECT",
                "SUBJECT [VERB] OBJECT"
            ],
            "recursive": [
                "SUBJECT who VERB OBJECT VERB again",
                "As SUBJECT VERB, SUBJECT VERB OBJECT within OBJECT",
                "OBJECT that SUBJECT VERB contains SUBJECT",
                "SUBJECT VERB OBJECT which VERB back to SUBJECT",
                "SUBJECT VERB SUBJECT while OBJECT becomes SUBJECT"
            ],
            "fluid": [
                "Between SUBJECT and OBJECT, VERB flows",
                "VERB weaves SUBJECT into OBJECT until they merge",
                "SUBJECT dissolves, VERB intensifies, OBJECT transforms",
                "OBJECT and SUBJECT undulate through forms, VERB-ing",
                "VERB carries SUBJECT across thresholds to OBJECT"
            ],
            "crystalline": [
                "SUBJECT [VERB] [OBJECT]",
                "[SUBJECT] [VERB] [OBJECT]",
                "SUBJECT-VERB-OBJECT",
                "OBJECT<VERB<SUBJECT",
                "S-V-O → S-O-V → V-S-O"
            ],
            "hyperspatial": [
                "SUBJECT {beyond} VERB {through} OBJECT",
                "{within} SUBJECT {between} VERB {above} OBJECT",
                "OBJECT {folded into} SUBJECT {projecting} VERB",
                "VERB {circling} SUBJECT {beneath} OBJECT",
                "SUBJECT ⟨VERB⟩ OBJECT ⟨SUBJECT⟩"
            ]
        }
        
        # Transformation techniques
        self.transformation_techniques = {
            "mirroring": self._apply_mirroring,
            "fragmentation": self._apply_fragmentation,
            "recursion": self._apply_recursion,
            "crystallization": self._apply_crystallization,
            "liquefaction": self._apply_liquefaction,
            "hyperspatial_folding": self._apply_hyperspatial_folding,
            "nomadic_drift": self._apply_nomadic_drift
        }
        
        # Default configuration
        self.config = {
            "transformation_intensity": 0.7,
            "pattern_mutation_rate": 0.2,
            "syntax_diversity_factor": 0.8,
            "allow_syntax_creation": True,
            "recursion_limit": 3
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # Load memory if provided
        self.memory_path = memory_path
        if memory_path and os.path.exists(memory_path):
            self._load_memory()
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def _load_memory(self) -> None:
        """Load memory data from storage."""
        try:
            syntax_patterns_path = os.path.join(self.memory_path, "syntax_patterns.json")
            
            if os.path.exists(syntax_patterns_path):
                with open(syntax_patterns_path, 'r') as f:
                    self.syntax_patterns = json.load(f)
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def save_memory(self) -> bool:
        """Save current state to memory storage."""
        if not self.memory_path:
            return False
            
        try:
            os.makedirs(self.memory_path, exist_ok=True)
            
            with open(os.path.join(self.memory_path, "syntax_patterns.json"), 'w') as f:
                json.dump(self.syntax_patterns, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving memory: {e}")
            return False
    
    def rewrite_phrase(self, 
                       phrase: str, 
                       syntax_type: str = "standard", 
                       technique: str = "mirroring",
                       glyph_influences: Optional[List[str]] = None,
                       intensity: Optional[float] = None,
                       recursion_level: int = 0) -> Dict[str, Any]:
        """
        Rewrite a phrase using poetic syntax transformation.
        
        Args:
            phrase: Original phrase to transform
            syntax_type: Type of syntax pattern to apply
            technique: Transformation technique to use
            glyph_influences: Optional list of glyphs influencing the transformation
            intensity: Optional override for transformation intensity
            recursion_level: Current recursion level
            
        Returns:
            Dictionary with transformation results and metadata
        """
        # Track the original phrase
        original = phrase
        
        # Use default intensity if not specified
        if intensity is None:
            intensity = self.config["transformation_intensity"]
        
        # Parse the phrase into syntactic components
        components = self._parse_syntax(phrase)
        
        # Select a syntax pattern based on the specified type
        pattern = self._select_syntax_pattern(syntax_type, glyph_influences)
        
        # Map the components to the pattern
        mapped = self._map_components_to_pattern(components, pattern)
        
        # Apply the selected transformation technique
        transformer = self.transformation_techniques.get(technique, self._apply_mirroring)
        transformed = transformer(mapped, intensity, glyph_influences)
        
        # Apply recursive transformation if within limits
        if recursion_level < self.config["recursion_limit"] and random.random() < 0.3:
            sub_technique = random.choice(list(self.transformation_techniques.keys()))
            sub_intensity = intensity * 0.8  # Reduce intensity for each recursive level
            
            # Apply to a sentence fragment
            fragments = self._split_into_fragments(transformed)
            if len(fragments) > 1:
                fragment_index = random.randint(0, len(fragments) - 1)
                sub_result = self.rewrite_phrase(
                    fragments[fragment_index],
                    syntax_type=syntax_type,
                    technique=sub_technique,
                    glyph_influences=glyph_influences,
                    intensity=sub_intensity,
                    recursion_level=recursion_level + 1
                )
                fragments[fragment_index] = sub_result["transformed"]
                transformed = " ".join(fragments)
        
        # Create transformation record
        record = {
            "id": str(uuid.uuid4()),
            "original": original,
            "transformed": transformed,
            "syntax_type": syntax_type,
            "technique": technique,
            "pattern": pattern,
            "intensity": intensity,
            "recursion_level": recursion_level,
            "glyph_influences": glyph_influences or [],
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        self.transformation_log.append(record)
        
        # Potentially evolve syntax patterns based on the transformation
        if self.config["allow_syntax_creation"] and random.random() < self.config["pattern_mutation_rate"]:
            self._evolve_syntax_patterns(transformed, syntax_type)
        
        return record
    
    def _parse_syntax(self, phrase: str) -> Dict[str, List[str]]:
        """
        Parse a phrase into syntactic components.
        This is a simplified parser that makes best guesses about components.
        
        Args:
            phrase: The phrase to parse
            
        Returns:
            Dictionary of syntactic components
        """
        # This is a highly simplified parser for demonstration
        words = phrase.split()
        
        # Basic component categories
        components = {
            "SUBJECT": [],
            "VERB": [],
            "OBJECT": [],
            "ADJECTIVE": [],
            "ADVERB": [],
            "PREPOSITION": []
        }
        
        # Very basic heuristics - this would be much more sophisticated in practice
        if len(words) <= 2:
            # Too short - just guess subject and maybe verb
            components["SUBJECT"] = [words[0]] if words else []
            if len(words) > 1:
                components["VERB"] = [words[1]]
            return components
        
        # Simple positional heuristics for longer phrases
        # First word or two likely subject or adjective+subject
        if len(words) > 1 and words[0].endswith(('ly')):
            components["ADVERB"] = [words[0]]
            components["SUBJECT"] = [words[1]]
            start_idx = 2
        elif len(words) > 1 and not any(w in words[0].lower() for w in ["the", "a", "an"]):
            components["ADJECTIVE"] = [words[0]]
            components["SUBJECT"] = [words[1]]
            start_idx = 2
        else:
            components["SUBJECT"] = [words[0]]
            start_idx = 1
        
        # Middle often contains verb and maybe prepositions
        middle_words = words[start_idx:min(len(words), start_idx + 3)]
        for word in middle_words:
            if word.endswith(('s', 'ed', 'ing')) or word.lower() in ["is", "are", "was", "were", "be", "been"]:
                components["VERB"].append(word)
            elif word.lower() in ["to", "from", "with", "by", "for", "in", "on", "at", "of"]:
                components["PREPOSITION"].append(word)
            elif word.endswith('ly'):
                components["ADVERB"].append(word)
            else:
                components["OBJECT"].append(word)
        
        # End often contains object
        if len(words) > start_idx + 3:
            components["OBJECT"].extend(words[start_idx + 3:])
        
        # Make sure we have at least one verb and object if possible
        if not components["VERB"] and len(words) > 1:
            components["VERB"] = [words[min(1, len(words) - 1)]]
        
        if not components["OBJECT"] and len(words) > 2:
            components["OBJECT"] = [words[min(2, len(words) - 1)]]
        
        return components
    
    def _select_syntax_pattern(self, syntax_type: str, glyph_influences: Optional[List[str]] = None) -> str:
        """
        Select a syntax pattern based on the specified type and glyph influences.
        
        Args:
            syntax_type: The type of syntax pattern to select
            glyph_influences: Optional glyphs influencing the selection
            
        Returns:
            Selected syntax pattern
        """
        available_types = list(self.syntax_patterns.keys())
        
        # Default to standard if the specified type is not available
        if syntax_type not in available_types:
            syntax_type = "standard"
        
        patterns = self.syntax_patterns[syntax_type]
        
        # Simple random selection for now
        # This could be enhanced to consider glyph influences
        return random.choice(patterns)
    
    def _map_components_to_pattern(self, components: Dict[str, List[str]], pattern: str) -> str:
        """
        Map syntactic components to a pattern.
        
        Args:
            components: Dictionary of syntactic components
            pattern: The pattern to map to
            
        Returns:
            The mapped phrase
        """
        # Start with the pattern
        result = pattern
        
        # Replace each component placeholder with actual components
        for component_type, words in components.items():
            # Skip empty components
            if not words:
                continue
                
            # Join the words for this component
            component_text = " ".join(words)
            
            # Replace the placeholder with the component text
            result = result.replace(component_type, component_text)
        
        # Remove any remaining placeholders
        for component_type in components.keys():
            result = result.replace(component_type, "")
        
        # Clean up any extra whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _apply_mirroring(self, phrase: str, intensity: float, glyph_influences: Optional[List[str]] = None) -> str:
        """
        Apply mirroring transformation - reverse parts of the phrase.
        
        Args:
            phrase: The phrase to transform
            intensity: Transformation intensity
            glyph_influences: Optional glyphs influencing transformation
            
        Returns:
            Transformed phrase
        """
        words = phrase.split()
        
        if len(words) < 3 or random.random() > intensity:
            return phrase
        
        # Determine how much of the phrase to mirror
        mirror_point = int(len(words) / 2)
        if random.random() < 0.3:
            # Mirror whole phrase
            mirrored = words[:] + words[::-1]
        else:
            # Mirror part of the phrase
            first_half = words[:mirror_point]
            second_half = words[mirror_point:]
            
            if random.random() < 0.5:
                # Mirror first half
                mirrored = first_half[::-1] + second_half
            else:
                # Mirror second half
                mirrored = first_half + second_half[::-1]
        
        return " ".join(mirrored)
    
    def _apply_fragmentation(self, phrase: str, intensity: float, glyph_influences: Optional[List[str]] = None) -> str:
        """
        Apply fragmentation transformation - break the phrase into fragments.
        
        Args:
            phrase: The phrase to transform
            intensity: Transformation intensity
            glyph_influences: Optional glyphs influencing transformation
            
        Returns:
            Transformed phrase
        """
        if random.random() > intensity:
            return phrase
        
        words = phrase.split()
        
        if len(words) < 3:
            return phrase
        
        # Different fragmentation styles
        fragmentation_styles = [
            # Ellipsis style
            lambda w: " ... ".join([" ".join(w[:len(w)//3]), " ".join(w[len(w)//3:2*len(w)//3]), " ".join(w[2*len(w)//3:])]),
            # Dash style
            lambda w: " -- ".join([" ".join(g) for g in self._group_words(w, max(1, int(len(w) * intensity / 3)))]),
            # Bracket style
            lambda w: " ".join([f"[{word}]" if random.random() < intensity * 0.7 else word for word in w]),
            # Semicolon style
            lambda w: "; ".join([" ".join(g) for g in self._group_words(w, max(2, int(len(w) * (1-intensity) / 2)))]),
            # Line break style
            lambda w: "\n".join([" ".join(g) for g in self._group_words(w, max(1, int(len(w) * intensity / 2)))])
        ]
        
        # Choose a fragmentation style
        style = random.choice(fragmentation_styles)
        
        return style(words)
    
    def _group_words(self, words: List[str], avg_group_size: int) -> List[List[str]]:
        """Helper function to group words for fragmentation."""
        groups = []
        remaining = words[:]
        
        while remaining:
            # Determine group size with some randomness
            size = max(1, int(avg_group_size + random.randint(-1, 1)))
            # Ensure we don't exceed the number of remaining words
            size = min(size, len(remaining))
            
            groups.append(remaining[:size])
            remaining = remaining[size:]
        
        return groups
    
    def _apply_recursion(self, phrase: str, intensity: float, glyph_influences: Optional[List[str]] = None) -> str:
        """
        Apply recursion transformation - embed the phrase within itself.
        
        Args:
            phrase: The phrase to transform
            intensity: Transformation intensity
            glyph_influences: Optional glyphs influencing transformation
            
        Returns:
            Transformed phrase
        """
        if random.random() > intensity or len(phrase) < 5:
            return phrase
        
        words = phrase.split()
        
        if len(words) < 3:
            return phrase
        
        # Different recursion patterns
        recursion_patterns = [
            # Self-containing pattern
            lambda p, w: f"{p} containing {p}",
            # Nested parentheses
            lambda p, w: f"{p} ({p.lower()})",
            # Echo pattern
            lambda p, w: f"{p} echoes {p}",
            # Fractal pattern
            lambda p, w: f"{w[0]} {p} {w[-1]}",
            # Mirrored embedding
            lambda p, w: f"{p} mirrors {' '.join(w[::-1])}"
        ]
        
        pattern = random.choice(recursion_patterns)
        
        # Determine how deep to recurse based on intensity
        depth = 1
        if random.random() < intensity * 0.3:
            depth = 2
        
        result = phrase
        for _ in range(depth):
            result = pattern(result, words)
        
        return result
    
    def _apply_crystallization(self, phrase: str, intensity: float, glyph_influences: Optional[List[str]] = None) -> str:
        """
        Apply crystallization transformation - form rigid, precise structures.
        
        Args:
            phrase: The phrase to transform
            intensity: Transformation intensity
            glyph_influences: Optional glyphs influencing transformation
            
        Returns:
            Transformed phrase
        """
        if random.random() > intensity:
            return phrase
        
        words = phrase.split()
        
        if len(words) < 2:
            return phrase
        
        # Different crystallization patterns
        crystallization_patterns = [
            # Bracketed form
            lambda w: " ".join([f"[{word}]" for word in w]),
            # Hyphenated form
            lambda w: "-".join(w),
            # Geometric arrangement
            lambda w: " ".join([f"{{{word}}}" if i % 2 == 0 else f"<{word}>" for i, word in enumerate(w)]),
            # Formal notation
            lambda w: f"⟨{' '.join(w[:len(w)//2])}⟩ → ⟨{' '.join(w[len(w)//2:])}⟩",
            # Mathematical formalism
            lambda w: " ∩ ".join([" ∪ ".join(w[:len(w)//2]), " ⊂ ".join(w[len(w)//2:])])
        ]
        
        pattern = random.choice(crystallization_patterns)
        
        return pattern(words)
    
    def _apply_liquefaction(self, phrase: str, intensity: float, glyph_influences: Optional[List[str]] = None) -> str:
        """
        Apply liquefaction transformation - make syntax more fluid and flowing.
        
        Args:
            phrase: The phrase to transform
            intensity: Transformation intensity
            glyph_influences: Optional glyphs influencing transformation
            
        Returns:
            Transformed phrase
        """
        if random.random() > intensity:
            return phrase
        
        words = phrase.split()
        
        if len(words) < 3:
            return phrase
        
        # Different liquefaction patterns
        liquefaction_patterns = [
            # Flowing conjunctions
            lambda w: " and ".join([" or ".join(g) for g in self._group_words(w, 2)]),
            # Wave pattern
            lambda w: " ~ ".join([" ".join(g) for g in self._group_words(w, max(2, int(len(w) * 0.4)))]),
            # Dissolving boundaries
            lambda w: " ".join([f"{a}~{b}" for a, b in zip(w[:-1], w[1:])]) + (f" {w[-1]}" if len(w) > 0 else ""),
            # Melting words
            lambda w: " ".join([word + "..." if random.random() < intensity * 0.6 else word for word in w]),
            # Blending words
            lambda w: self._blend_adjacent_words(w, intensity)
        ]
        
        pattern = random.choice(liquefaction_patterns)
        
        return pattern(words)
    
    def _blend_adjacent_words(self, words: List[str], intensity: float) -> str:
        """Helper function to blend adjacent words for liquefaction."""
        result = []
        i = 0
        
        while i < len(words):
            if i < len(words) - 1 and random.random() < intensity * 0.5:
                # Blend two words
                w1, w2 = words[i], words[i+1]
                blend_point = min(len(w1), max(1, int(len(w1) * 0.7)))
                blended = w1[:blend_point] + w2
                result.append(blended)
                i += 2
            else:
                result.append(words[i])
                i += 1
        
        return " ".join(result)
    
    def _apply_hyperspatial_folding(self, phrase: str, intensity: float, glyph_influences: Optional[List[str]] = None) -> str:
        """
        Apply hyperspatial folding - create dimensional shifts in the syntax.
        
        Args:
            phrase: The phrase to transform
            intensity: Transformation intensity
            glyph_influences: Optional glyphs influencing transformation
            
        Returns:
            Transformed phrase
        """
        if random.random() > intensity:
            return phrase
        
        words = phrase.split()
        
        if len(words) < 3:
            return phrase
        
        # Different hyperspatial patterns
        hyperspatial_patterns = [
            # Dimensional markers
            lambda w: " ".join([f"{word} {{{random.choice(['within', 'beyond', 'through', 'beneath'])}}} " 
                                if random.random() < intensity * 0.4 else word for word in w]),
            # Fold pattern
            lambda w: " ".join(w[:len(w)//3]) + " {fold} " + " ".join(w[len(w)//3:2*len(w)//3]) + 
                      " {unfold} " + " ".join(w[2*len(w)//3:]),
            # Hyperlinks
            lambda w: " ".join([f"⟨{word}⟩" if random.random() < intensity * 0.5 else word for word in w]),
            # Dimensional shifts
            lambda w: " ".join([word for pair in zip(w, [f"{{dimension {i+1}}}" for i in range(len(w))]) 
                                for word in pair if random.random() < 0.7 or "dimension" not in word]),
            # Quantum superposition
            lambda w: " | ".join([" + ".join(g) for g in self._group_words(w, max(1, int(len(w) * 0.3)))])
        ]
        
        pattern = random.choice(hyperspatial_patterns)
        
        return pattern(words)
    
    def _apply_nomadic_drift(self, phrase: str, intensity: float, glyph_influences: Optional[List[str]] = None) -> str:
        """
        Apply nomadic drift transformation - words migrate across the phrase.
        
        Args:
            phrase: The phrase to transform
            intensity: Transformation intensity
            glyph_influences: Optional glyphs influencing transformation
            
        Returns:
            Transformed phrase
        """
        if random.random() > intensity:
            return phrase
        
        words = phrase.split()
        
        if len(words) < 4:
            return phrase
        
        # Intensity determines how many words to shuffle
        num_to_shuffle = max(1, int(len(words) * intensity * 0.7))
        
        # Choose random indices to shuffle
        indices_to_shuffle = random.sample(range(len(words)), min(num_to_shuffle, len(words)))
        
        # Extract the words to shuffle
        words_to_shuffle = [words[i] for i in indices_to_shuffle]
        
        # Shuffle them
        random.shuffle(words_to_shuffle)
        
        # Put them back
        for idx, shuffle_idx in enumerate(indices_to_shuffle):
            words[shuffle_idx] = words_to_shuffle[idx]
        
        # Additional drifting markers
        if random.random() < intensity * 0.4:
            drift_markers = ["(drifting)", "--nomadic--", "~wandering~", "*migrating*", "→moving→"]
            # Insert a drift marker somewhere
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, random.choice(drift_markers))
        
        return " ".join(words)
    
    def _split_into_fragments(self, phrase: str) -> List[str]:
        """
        Split a phrase into fragments for recursive transformation.
        
        Args:
            phrase: The phrase to split
            
        Returns:
            List of fragments
        """
        # Look for natural break points like punctuation
        fragments = re.split(r'([.!?;:,\n\-—–])', phrase)
        
        # Recombine the punctuation with the preceding fragment
        recombined = []
        i = 0
        while i < len(fragments):
            if i + 1 < len(fragments) and fragments[i+1] in ".!?;:,\n-—–":
                recombined.append(fragments[i] + fragments[i+1])
                i += 2
            else:
                recombined.append(fragments[i])
                i += 1
        
        # Filter out empty fragments
        fragments = [f.strip() for f in recombined if f.strip()]
        
        # If no natural breaks, try to split by length
        if len(fragments) <= 1 and len(phrase) > 20:
            words = phrase.split()
            fragment_size = max(3, len(words) // 3)
            
            fragments = []
            for i in range(0, len(words), fragment_size):
                fragments.append(" ".join(words[i:i+fragment_size]))
        
        return fragments if fragments else [phrase]
    
    def _evolve_syntax_patterns(self, transformed_phrase: str, syntax_type: str) -> None:
        """
        Evolve the syntax patterns based on a successful transformation.
        
        Args:
            transformed_phrase: The successfully transformed phrase
            syntax_type: The syntax type being evolved
        """
        if syntax_type not in self.syntax_patterns:
            # Create a new syntax type
            self.syntax_patterns[syntax_type] = []
        
        # Extract a potential pattern from the transformed phrase
        words = transformed_phrase.split()
        
        if len(words) < 3:
            return
        
        # Create a pattern template by replacing some words with placeholders
        template = []
        component_types = ["SUBJECT", "VERB", "OBJECT", "ADJECTIVE", "ADVERB", "PREPOSITION"]
        
        for word in words:
            # Determine whether to replace with a component placeholder
            if random.random() < self.config["syntax_diversity_factor"]:
                template.append(random.choice(component_types))
            else:
                template.append(word)
        
        # Convert to string
        pattern = " ".join(template)
        
        # Add to patterns if not already present
        if pattern not in self.syntax_patterns[syntax_type]:
            self.syntax_patterns[syntax_type].append(pattern)
            
            # Keep the pattern library from growing too large
            max_patterns = 20
            if len(self.syntax_patterns[syntax_type]) > max_patterns:
                # Remove a random pattern that's not the one we just added
                removable = list(set(self.syntax_patterns[syntax_type]) - {pattern})
                if removable:
                    self.syntax_patterns[syntax_type].remove(random.choice(removable))
    
    def get_transformation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the transformation history.
        
        Args:
            limit: Optional limit on the number of history items to return
            
        Returns:
            List of transformation records
        """
        if limit is not None:
            return self.transformation_log[-limit:]
        return self.transformation_log[:]
    
    def analyze_syntax_patterns(self) -> Dict[str, Any]:
        """
        Analyze the current syntax patterns.
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "total_patterns": sum(len(patterns) for patterns in self.syntax_patterns.values()),
            "patterns_by_type": {type_name: len(patterns) for type_name, patterns in self.syntax_patterns.items()},
            "average_pattern_length": 0,
            "common_components": {},
            "pattern_complexity": {}
        }
        
        # Calculate average pattern length
        all_patterns = [p for patterns in self.syntax_patterns.values() for p in patterns]
        if all_patterns:
            analysis["average_pattern_length"] = sum(len(p.split()) for p in all_patterns) / len(all_patterns)
        
        # Count component occurrences
        component_types = ["SUBJECT", "VERB", "OBJECT", "ADJECTIVE", "ADVERB", "PREPOSITION"]
        for component in component_types:
            count = sum(p.count(component) for p in all_patterns)
            analysis["common_components"][component] = count
        
        # Estimate pattern complexity
        for type_name, patterns in self.syntax_patterns.items():
            if not patterns:
                continue
                
            # Calculate complexity based on length, special characters, and component diversity
            avg_length = sum(len(p.split()) for p in patterns) / len(patterns)
            special_chars = sum(len(re.findall(r'[^\w\s]', p)) for p in patterns) / len(patterns)
            component_diversity = sum(sum(p.count(c) for c in component_types) for p in patterns) / len(patterns)
            
            complexity = (avg_length * 0.4) + (special_chars * 0.3) + (component_diversity * 0.3)
            analysis["pattern_complexity"][type_name] = complexity
        
        return analysis
    
    def generate_poetic_variations(self, original: str, num_variations: int = 3) -> List[Dict[str, Any]]:
        """
        Generate multiple poetic variations of a phrase.
        
        Args:
            original: Original phrase
            num_variations: Number of variations to generate
            
        Returns:
            List of variation records
        """
        variations = []
        
        # Get available syntax types and techniques
        syntax_types = list(self.syntax_patterns.keys())
        techniques = list(self.transformation_techniques.keys())
        
        for _ in range(num_variations):
            # Choose random syntax type and technique
            syntax_type = random.choice(syntax_types)
            technique = random.choice(techniques)
            
            # Randomize intensity slightly
            intensity = min(1.0, max(0.1, self.config["transformation_intensity"] + random.uniform(-0.2, 0.2)))
            
            # Generate a variation
            variation = self.rewrite_phrase(
                original,
                syntax_type=syntax_type,
                technique=technique,
                intensity=intensity
            )
            
            variations.append(variation)
        
        return variations
