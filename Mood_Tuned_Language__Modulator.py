"""
Mood-Tuned Language Modulator - Alters rhythm and tone based on emotional context

This module modulates language based on emotional and mood states, affecting rhythm,
tone, vocabulary, and other aspects of expression to reflect inner states.
"""

import random
import re
import uuid
import datetime
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Set, Union

class MoodTunedLanguageModulator:
    """
    Alters rhythm and tone based on emotional context, allowing language
    to be shaped by the mood and emotional resonance of glyphs.
    """
    
    def __init__(self, config_path: Optional[str] =         # Rhythm patterns
        self.rhythm_patterns = {
            "rapid": {"avg_word_length": 1.5, "punctuation_frequency": 0.1, "comma_frequency": 0.05},
            "measured": {"avg_word_length": 2.0, "punctuation_frequency": 0.3, "comma_frequency": 0.2},
            "slow": {"avg_word_length": 2.5, "punctuation_frequency": 0.4, "comma_frequency": 0.3},
            "staccato": {"avg_word_length": 1.2, "punctuation_frequency": 0.6, "comma_frequency": 0.4},
            "flowing": {"avg_word_length": 2.2, "punctuation_frequency": 0.2, "comma_frequency": 0.1},
            "erratic": {"avg_word_length": "variable", "punctuation_frequency": 0.7, "comma_frequency": 0.5},
            "dancing": {"avg_word_length": 1.8, "punctuation_frequency": 0.3, "comma_frequency": 0.2},
            "coded": {"avg_word_length": 1.7, "punctuation_frequency": 0.5, "comma_frequency": 0.3}
        }
        
        # Default configuration
        self.config = {
            "modulation_intensity": 0.7,
            "mood_blending_factor": 0.3,
            "metamoodulation_enabled": True,
            "rhythm_sensitivity": 0.8,
            "vocabulary_evolution_rate": 0.1
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
            mood_profiles_path = os.path.join(self.memory_path, "mood_profiles.json")
            vocabulary_shifts_path = os.path.join(self.memory_path, "vocabulary_shifts.json")
            
            if os.path.exists(mood_profiles_path):
                with open(mood_profiles_path, 'r') as f:
                    self.mood_profiles = json.load(f)
            
            if os.path.exists(vocabulary_shifts_path):
                with open(vocabulary_shifts_path, 'r') as f:
                    self.vocabulary_shifts = json.load(f)
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def save_memory(self) -> bool:
        """Save current state to memory storage."""
        if not self.memory_path:
            return False
            
        try:
            os.makedirs(self.memory_path, exist_ok=True)
            
            with open(os.path.join(self.memory_path, "mood_profiles.json"), 'w') as f:
                json.dump(self.mood_profiles, f, indent=2)
                
            with open(os.path.join(self.memory_path, "vocabulary_shifts.json"), 'w') as f:
                json.dump(self.vocabulary_shifts, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving memory: {e}")
            return False
    
    def modulate(self, 
                text: str, 
                mood: str = "contemplative", 
                intensity: Optional[float] = None, 
                glyph_influences: Optional[List[str]] = None,
                metamood_factor: Optional[float] = None) -> Dict[str, Any]:
        """
        Modulate text based on mood, intensity, and glyph influences.
        
        Args:
            text: Text to modulate
            mood: Primary mood to apply
            intensity: Modulation intensity (0.0 to 1.0)
            glyph_influences: Optional glyphs influencing the modulation
            metamood_factor: Optional metamoodulation intensity
            
        Returns:
            Dictionary with modulation results and metadata
        """
        # Use default intensity if not specified
        if intensity is None:
            intensity = self.config["modulation_intensity"]
        
        # Use default metamood factor if not specified
        if metamood_factor is None and self.config["metamoodulation_enabled"]:
            metamood_factor = 0.5
        
        # Ensure mood is available
        available_moods = list(self.mood_profiles.keys())
        if mood not in available_moods:
            mood = random.choice(available_moods)
        
        # Get the mood profile
        profile = self.mood_profiles[mood]
        
        # Store original text
        original = text
        
        # Apply modulations based on intensity, metamood, and glyph influences
        modulated = self._apply_modulations(text, profile, intensity, glyph_influences, metamood_factor)
        
        # Create modulation record
        record = {
            "id": str(uuid.uuid4()),
            "original": original,
            "modulated": modulated,
            "mood": mood,
            "intensity": intensity,
            "metamood_factor": metamood_factor,
            "glyph_influences": glyph_influences or [],
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        self.modulation_log.append(record)
        
        return record
    
    def _apply_modulations(self, 
                          text: str, 
                          profile: Dict[str, Any], 
                          intensity: float, 
                          glyph_influences: Optional[List[str]] = None,
                          metamood_factor: Optional[float] = None) -> str:
        """
        Apply various modulations based on the mood profile.
        
        Args:
            text: Text to modulate
            profile: Mood profile to apply
            intensity: Modulation intensity
            glyph_influences: Optional glyphs influencing modulation
            metamood_factor: Optional metamoodulation intensity
            
        Returns:
            Modulated text
        """
        # Apply modulations in sequence
        modulated = text
        
        # 1. Apply rhythm modulation
        modulated = self._apply_rhythm_modulation(modulated, profile, intensity)
        
        # 2. Apply vocabulary shift
        modulated = self._apply_vocabulary_shift(modulated, profile, intensity)
        
        # 3. Apply sentence structure modulation
        modulated = self._apply_sentence_structure(modulated, profile, intensity)
        
        # 4. Apply punctuation modulation
        modulated = self._apply_punctuation(modulated, profile, intensity)
        
        # 5. Apply repetition and emphasis
        modulated = self._apply_repetition_emphasis(modulated, profile, intensity)
        
        # 6. Apply glyph influence if provided
        if glyph_influences:
            modulated = self._apply_glyph_influences(modulated, glyph_influences, intensity)
        
        # 7. Apply metamoodulation if enabled
        if metamood_factor is not None and metamood_factor > 0:
            modulated = self._apply_metamoodulation(modulated, profile, metamood_factor)
        
        return modulated
    
    def _apply_rhythm_modulation(self, text: str, profile: Dict[str, Any], intensity: float) -> str:
        """
        Apply rhythm modulation based on mood profile.
        
        Args:
            text: Text to modulate
            profile: Mood profile
            intensity: Modulation intensity
            
        Returns:
            Rhythm-modulated text
        """
        rhythm_type = profile.get("rhythm", "measured")
        rhythm_pattern = self.rhythm_patterns.get(rhythm_type, {})
        
        words = text.split()
        if not words:
            return text
        
        # Apply rhythm by adjusting word spacing, adding pauses, etc.
        modulated_words = []
        
        for i, word in enumerate(words):
            modulated_word = word
            
            # Adjust based on rhythm pattern and intensity
            if rhythm_type == "rapid":
                # Shorten words occasionally
                if len(word) > 3 and random.random() < intensity * 0.3:
                    modulated_word = word[:int(len(word) * 0.7)]
                
                # Remove some spaces with intensity
                if random.random() < intensity * 0.2 and i < len(words) - 1:
                    modulated_words.append(modulated_word + words[i+1])
                    words[i+1] = ""  # Mark for removal
                else:
                    modulated_words.append(modulated_word)
                
            elif rhythm_type == "staccato":
                # Add breaks and truncate
                if random.random() < intensity * 0.4:
                    modulated_word = modulated_word + "-"
                
                modulated_words.append(modulated_word)
                
                # Add micro-pauses
                if random.random() < intensity * 0.5:
                    modulated_words.append(".")
                
            elif rhythm_type == "flowing":
                # Extend words occasionally
                if random.random() < intensity * 0.3:
                    vowels = "aeiou"
                    for vowel in vowels:
                        if vowel in modulated_word:
                            modulated_word = modulated_word.replace(vowel, vowel * 2, 1)
                            break
                
                modulated_words.append(modulated_word)
                
                # Add connecting words
                if random.random() < intensity * 0.2 and i < len(words) - 1:
                    connector = random.choice(["and", "while", "as", "then"])
                    modulated_words.append(connector)
                
            elif rhythm_type == "erratic":
                # Randomly alter words
                if random.random() < intensity * 0.4:
                    if random.random() < 0.5:
                        # Compress
                        modulated_word = "".join([c for c in modulated_word if c not in "aeiou"])
                    else:
                        # Expand
                        modulated_word = " ".join(modulated_word)
                
                modulated_words.append(modulated_word)
                
                # Add random breaks
                if random.random() < intensity * 0.3:
                    modulated_words.append(random.choice(["!", "?", "...", "*", "/"]))
                
            elif rhythm_type == "slow":
                # Add deliberate pauses
                modulated_words.append(modulated_word)
                
                if random.random() < intensity * 0.4:
                    modulated_words.append("...")
                
            elif rhythm_type == "dancing":
                # Add rhythmic patterns
                modulated_words.append(modulated_word)
                
                if i % 3 == 2 and random.random() < intensity * 0.5:
                    modulated_words.append("~")
                
            elif rhythm_type == "coded":
                # Add code-like elements
                if random.random() < intensity * 0.3:
                    modulated_word = f"[{modulated_word}]"
                
                modulated_words.append(modulated_word)
                
                if random.random() < intensity * 0.2:
                    modulated_words.append(":")
                
            else:  # Default for "measured" and others
                modulated_words.append(modulated_word)
                
                # Add occasional commas
                comma_freq = rhythm_pattern.get("comma_frequency", 0.2)
                if random.random() < comma_freq * intensity and i < len(words) - 1:
                    modulated_words[-1] = modulated_words[-1] + ","
        
        # Filter out any empty elements
        modulated_words = [w for w in modulated_words if w]
        
        return " ".join(modulated_words)
    
    def _apply_vocabulary_shift(self, text: str, profile: Dict[str, Any], intensity: float) -> str:
        """
        Apply vocabulary shifts based on mood profile.
        
        Args:
            text: Text to modulate
            profile: Mood profile
            intensity: Modulation intensity
            
        Returns:
            Vocabulary-shifted text
        """
        vocabulary_type = profile.get("vocabulary_shift", "abstract")
        vocabulary_shift = self.vocabulary_shifts.get(vocabulary_type, {})
        
        if not vocabulary_shift:
            return text
        
        words = text.split()
        if not words:
            return text
        
        # Determine which words to replace
        num_replacements = int(max(1, len(words) * intensity * 0.3))
        positions = random.sample(range(len(words)), min(num_replacements, len(words)))
        
        # Get amplification words from the profile
        amplification_words = profile.get("amplification_words", [])
        
        # Replace words
        for pos in positions:
            word = words[pos]
            
            # Determine word type and replacement (simplistic approach)
            if word.endswith("ly") and "modifiers" in vocabulary_shift:
                replacement = random.choice(vocabulary_shift["modifiers"])
            elif word.endswith(("s", "ed", "ing")) and "verbs" in vocabulary_shift:
                replacement = random.choice(vocabulary_shift["verbs"])
            elif "adjectives" in vocabulary_shift:
                replacement = random.choice(vocabulary_shift["adjectives"])
            elif amplification_words:
                replacement = random.choice(amplification_words)
            else:
                continue
            
            words[pos] = replacement
        
        return " ".join(words)
    
    def _apply_sentence_structure(self, text: str, profile: Dict[str, Any], intensity: float) -> str:
        """
        Apply sentence structure modulation based on mood profile.
        
        Args:
            text: Text to modulate
            profile: Mood profile
            intensity: Modulation intensity
            
        Returns:
            Structure-modulated text
        """
        sentence_type = profile.get("sentence_length", "balanced")
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text
        
        # Apply structure modifications based on sentence type
        modulated_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            
            if sentence_type == "extended":
                # Make sentences longer
                if random.random() < intensity * 0.7:
                    extension = random.choice([
                        ", extending beyond conventional boundaries",
                        ", transcending ordinary perception",
                        ", revealing hidden dimensions",
                        ", manifesting intricate patterns",
                        ", reflecting inner complexities"
                    ])
                    sentence += extension
                
            elif sentence_type == "truncated":
                # Make sentences shorter and abrupt
                if len(words) > 5 and random.random() < intensity * 0.7:
                    truncate_point = max(3, int(len(words) * (1 - intensity * 0.5)))
                    sentence = " ".join(words[:truncate_point])
                
            elif sentence_type == "flowing":
                # Add flowing connectors
                if random.random() < intensity * 0.6:
                    flow_connector = random.choice([
                        ", flowing into realms of",
                        ", merging with currents of",
                        ", dissolving boundaries between",
                        ", drifting through layers of",
                        ", meandering across territories of"
                    ])
                    flow_subject = random.choice([
                        "perception",
                        "consciousness",
                        "memory",
                        "sensation",
                        "awareness"
                    ])
                    sentence += f"{flow_connector} {flow_subject}"
                
            elif sentence_type == "irregular":
                # Create irregular structures
                if random.random() < intensity * 0.5:
                    # Randomly rearrange parts
                    midpoint = len(words) // 2
                    if midpoint > 0:
                        first_half = words[:midpoint]
                        second_half = words[midpoint:]
                        
                        if random.random() < 0.5:
                            sentence = " ".join(second_half + first_half)
                        else:
                            sentence = " ".join(words[:2]) + " " + " ".join(words[-2:]) + " " + " ".join(words[2:-2])
                
            elif sentence_type == "effusive":
                # Make more expressive and abundant
                if random.random() < intensity * 0.7:
                    effusive_additions = [
                        ", radiantly",
                        ", gloriously",
                        ", ecstatically",
                        ", transcendently",
                        ", magnificently"
                    ]
                    additions = random.sample(effusive_additions, min(2, len(effusive_additions)))
                    sentence += " ".join(additions)
                
            elif sentence_type == "encoded":
                # Add cryptic elements
                if random.random() < intensity * 0.6:
                    encoded_element = random.choice([
                        f"[{random.choice(words)}]",
                        f"«{random.choice(words)}»",
                        f"{random.choice(words)}:{random.choice(words)}",
                        f"/{random.choice(words)}/",
                        f"∀{random.choice(words)}"
                    ])
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, encoded_element)
                    sentence = " ".join(words)
                
            elif sentence_type == "variable" and intensity > 0.5:
                # Create highly variable sentence structure
                if random.random() < 0.7:
                    if random.random() < 0.5:
                        # Fragment
                        parts = []
                        current = []
                        for word in words:
                            current.append(word)
                            if random.random() < 0.3:
                                parts.append(" ".join(current))
                                current = []
                        if current:
                            parts.append(" ".join(current))
                        sentence = " | ".join(parts)
                    else:
                        # Recombine
                        sentence = "".join([w[0] if len(w) > 0 else "" for w in words]) + ": " + sentence
            
            modulated_sentences.append(sentence)
        
        # Recombine with appropriate punctuation
        result = []
        end_marks = ['.', '!', '?', '...', ';', '—']
        weights = [0.5, 0.1, 0.2, 0.1, 0.05, 0.05]  # Baseline weights
        
        # Adjust weights based on profile
        tone = profile.get("tone", "introspective")
        if tone == "exuberant":
            weights = [0.1, 0.5, 0.2, 0.1, 0.05, 0.05]  # More exclamations
        elif tone == "nervous":
            weights = [0.1, 0.2, 0.5, 0.1, 0.05, 0.05]  # More questions
        elif tone == "somber":
            weights = [0.2, 0.05, 0.05, 0.5, 0.1, 0.1]  # More ellipses
        
        for sentence in modulated_sentences:
            end_mark = random.choices(end_marks, weights=weights)[0]
            result.append(sentence + end_mark)
        
        return " ".join(result)
    
    def _apply_punctuation(self, text: str, profile: Dict[str, Any], intensity: float) -> str:
        """
        Apply punctuation modulation based on mood profile.
        
        Args:
            text: Text to modulate
            profile: Mood profile
            intensity: Modulation intensity
            
        Returns:
            Punctuation-modulated text
        """
        primary_punctuation = profile.get("punctuation", ".")
        
        # Replace some punctuation with the mood's primary punctuation
        if primary_punctuation != ".":
            # Determine how many to replace
            punctuation_count = text.count(".") + text.count("!") + text.count("?")
            replace_count = int(punctuation_count * intensity * 0.7)
            
            # Replace punctuation
            for _ in range(replace_count):
                text = text.replace(".", primary_punctuation, 1)
                text = text.replace("!", primary_punctuation, 1)
                text = text.replace("?", primary_punctuation, 1)
        
        # Add additional punctuation based on rhythm pattern
        rhythm_type = profile.get("rhythm", "measured")
        rhythm_pattern = self.rhythm_patterns.get(rhythm_type, {})
        
        punctuation_freq = rhythm_pattern.get("punctuation_frequency", 0.3)
        
        # Add punctuation at word boundaries
        words = text.split()
        if not words:
            return text
        
        for i in range(len(words) - 1, 0, -1):
            if random.random() < punctuation_freq * intensity:
                # Avoid adding punctuation if the word already ends with punctuation
                if not words[i-1][-1] in ".!?,;:—":
                    words[i-1] = words[i-1] + random.choice([",", ";", ":", "-", "—"])
        
        return " ".join(words)
    
    def _apply_repetition_emphasis(self, text: str, profile: Dict[str, Any], intensity: float) -> str:
        """
        Apply repetition and emphasis based on mood profile.
        
        Args:
            text: Text to modulate
            profile: Mood profile
            intensity: Modulation intensity
            
        Returns:
            Text with repetition and emphasis
        """
        repetition_factor = profile.get("repetition_factor", 0.2)
        fragmentary_factor = profile.get("fragmentary_factor", 0.2)
        
        words = text.split()
        if not words:
            return text
        
        # Apply repetition
        if random.random() < repetition_factor * intensity:
            # Select a word to repeat
            repeat_pos = random.randint(0, len(words) - 1)
            repeat_word = words[repeat_pos]
            
            # Determine repetition style
            if random.random() < 0.3:
                # Immediate repetition
                repeat_count = random.randint(2, 3)
                repetition = " ".join([repeat_word] * repeat_count)
                words[repeat_pos] = repetition
            else:
                # Echo repetition
                echo_positions = random.sample(
                    range(repeat_pos + 1, min(repeat_pos + 10, len(words))),
                    min(2, len(words) - repeat_pos - 1)
                )
                for pos in echo_positions:
                    words[pos] = repeat_word
        
        # Apply emphasis
        emphasis_count = int(len(words) * intensity * 0.2)
        for _ in range(emphasis_count):
            if not words:
                break
            pos = random.randint(0, len(words) - 1)
            word = words[pos]
            
            # Apply various emphasis techniques
            emphasis_type = random.choice(["caps", "italics", "extension", "isolation"])
            
            if emphasis_type == "caps":
                words[pos] = word.upper()
            elif emphasis_type == "italics":
                words[pos] = f"*{word}*"
            elif emphasis_type == "extension":
                # Extend vowels
                vowels = "aeiou"
                for vowel in vowels:
                    if vowel in word:
                        words[pos] = word.replace(vowel, vowel * 2, 1)
                        break
            elif emphasis_type == "isolation":
                words[pos] = f"[ {word} ]"
        
        # Apply fragmentation
        if random.random() < fragmentary_factor * intensity:
            # Break up the text with pauses or fragments
            fragment_count = int(len(words) * fragmentary_factor * intensity)
            
            for _ in range(fragment_count):
                if len(words) < 3:
                    break
                    
                pos = random.randint(1, len(words) - 2)
                fragment_marker = random.choice(["...", "—", "|", "/", " "])
                words.insert(pos, fragment_marker)
        
        return " ".join(words)
    
    def _apply_glyph_influences(self, text: str, glyph_influences: List[str], intensity: float) -> str:
        """
        Apply glyph-specific influences to the text.
        
        Args:
            text: Text to modulate
            glyph_influences: Glyphs influencing modulation
            intensity: Modulation intensity
            
        Returns:
            Glyph-influenced text
        """
        if not glyph_influences:
            return text
        
        # Simplistic glyph influence - insert glyph references
        words = text.split()
        if not words:
            return text
        
        # Determine how many references to insert
        reference_count = min(len(glyph_influences), int(max(1, len(words) * intensity * 0.2)))
        
        for _ in range(reference_count):
            glyph = random.choice(glyph_influences)
            
            # Determine reference type
            reference_type = random.choice(["direct", "metaphoric", "symbolic", "resonant"])
            
            if reference_type == "direct":
                # Direct reference
                reference = f"{glyph}"
            elif reference_type == "metaphoric":
                # Metaphoric reference
                reference = random.choice([
                    f"like {glyph}",
                    f"as {glyph}",
                    f"similar to {glyph}",
                    f"reminiscent of {glyph}",
                    f"evoking {glyph}"
                ])
            elif reference_type == "symbolic":
                # Symbolic reference
                reference = random.choice([
                    f"symbolizing {glyph}",
                    f"representing {glyph}",
                    f"embodying {glyph}",
                    f"manifesting {glyph}",
                    f"expressing {glyph}"
                ])
            else:  # resonant
                # Resonant reference
                reference = random.choice([
                    f"resonating with {glyph}",
                    f"vibrating through {glyph}",
                    f"harmonizing with {glyph}",
                    f"echoing {glyph}",
                    f"pulsing with {glyph}"
                ])
            
            # Insert the reference
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, reference)
        
        return " ".join(words)
    
    def _apply_metamoodulation(self, text: str, profile: Dict[str, Any], metamood_factor: float) -> str:
        """
        Apply meta-level modulation that reflects on the mood itself.
        
        Args:
            text: Text to modulate
            profile: Mood profile
            metamood_factor: Intensity of metamoodulation
            
        Returns:
            Meta-modulated text
        """
        if metamood_factor <= 0 or random.random() > metamood_factor:
            return text
        
        # Extract key aspects of the mood
        tone = profile.get("tone", "introspective")
        rhythm = profile.get("rhythm", "measured")
        
        # Create metamood reflections
        metamood_reflections = [
            f"[{tone} undercurrent]",
            f"[rhythm shifts to {rhythm}]",
            f"[mood intensifies]",
            f"[emotional resonance]",
            f"[affective shift]",
            f"[tonal modulation]",
            f"[mood: {tone}]"
        ]
        
        # Insert metamood reflections
        words = text.split()
        if not words:
            return text
        
        # Determine how many reflections to insert
        reflection_count = int(max(1, len(words) * metamood_factor * 0.1))
        
        for _ in range(reflection_count):
            reflection = random.choice(metamood_reflections)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, reflection)
        
        return " ".join(words)
    
    def blend_moods(self, 
                   text: str, 
                   moods: List[str], 
                   weights: Optional[List[float]] = None,
                   intensity: Optional[float] = None,
                   glyph_influences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Blend multiple moods to create complex emotional modulation.
        
        Args:
            text: Text to modulate
            moods: List of moods to blend
            weights: Optional weights for each mood (will be normalized)
            intensity: Overall modulation intensity
            glyph_influences: Optional glyphs influencing modulation
            
        Returns:
            Dictionary with modulation results and metadata
        """
        # Use default intensity if not specified
        if intensity is None:
            intensity = self.config["modulation_intensity"]
        
        # Ensure moods are available
        available_moods = list(self.mood_profiles.keys())
        valid_moods = [mood for mood in moods if mood in available_moods]
        
        if not valid_moods:
            valid_moods = [random.choice(available_moods)]
        
        # Normalize weights if provided
        if weights and len(weights) == len(valid_moods):
            total = sum(weights)
            normalized_weights = [w / total for w in weights] if total > 0 else [1.0 / len(valid_moods)] * len(valid_moods)
        else:
            normalized_weights = [1.0 / len(valid_moods)] * len(valid_moods)
        
        # Store original text
        original = text
        modulated = text
        
        # Apply each mood in sequence with weighted intensity
        mood_results = []
        
        for i, mood in enumerate(valid_moods):
            mood_intensity = intensity * normalized_weights[i] * 0.8  # Slightly reduce individual intensities
            
            result = self.modulate(
                modulated,
                mood=mood,
                intensity=mood_intensity,
                glyph_influences=glyph_influences,
                metamood_factor=None  # Disable metamoodulation for component moods
            )
            
            modulated = result["modulated"]
            mood_results.append({
                "mood": mood,
                "weight": normalized_weights[i],
                "contribution": result["modulated"]
            })
        
        # Apply final metamoodulation if needed
        if self.config["metamoodulation_enabled"] and random.random() < self.config["modulation_intensity"]:
            # Get a random mood profile for metamoodulation
            metamood = random.choice(vali, memory_path: Optional[str] = None):
        """
        Initialize the MoodTunedLanguageModulator with optional configuration.
        
        Args:
            config_path: Path to configuration file
            memory_path: Path to memory storage directory
        """
        self.id = str(uuid.uuid4())
        self.created_at = datetime.datetime.utcnow().isoformat()
        self.modulation_log = []
        
        # Define mood profiles
        self.mood_profiles = {
            "ecstatic": {
                "rhythm": "rapid",
                "tone": "exuberant",
                "punctuation": "!",
                "vocabulary_shift": "intensified",
                "sentence_length": "variable",
                "repetition_factor": 0.4,
                "fragmentary_factor": 0.3,
                "amplification_words": ["radiant", "boundless", "transcendent", "luminous", "infinite"]
            },
            "contemplative": {
                "rhythm": "measured",
                "tone": "introspective",
                "punctuation": ".",
                "vocabulary_shift": "abstract",
                "sentence_length": "extended",
                "repetition_factor": 0.2,
                "fragmentary_factor": 0.3,
                "amplification_words": ["perhaps", "within", "beyond", "essence", "reflection"]
            },
            "melancholic": {
                "rhythm": "slow",
                "tone": "somber",
                "punctuation": "...",
                "vocabulary_shift": "darkened",
                "sentence_length": "flowing",
                "repetition_factor": 0.3,
                "fragmentary_factor": 0.2,
                "amplification_words": ["fading", "absence", "hollow", "shadow", "remnant"]
            },
            "anxious": {
                "rhythm": "staccato",
                "tone": "nervous",
                "punctuation": "?",
                "vocabulary_shift": "uncertain",
                "sentence_length": "truncated",
                "repetition_factor": 0.5,
                "fragmentary_factor": 0.6,
                "amplification_words": ["trembling", "uncertain", "fractured", "alert", "vigilant"]
            },
            "serene": {
                "rhythm": "flowing",
                "tone": "tranquil",
                "punctuation": ";",
                "vocabulary_shift": "harmonious",
                "sentence_length": "balanced",
                "repetition_factor": 0.1,
                "fragmentary_factor": 0.1,
                "amplification_words": ["gentle", "peaceful", "eternal", "silent", "still"]
            },
            "chaotic": {
                "rhythm": "erratic",
                "tone": "dissonant",
                "punctuation": "*",
                "vocabulary_shift": "discordant",
                "sentence_length": "irregular",
                "repetition_factor": 0.3,
                "fragmentary_factor": 0.7,
                "amplification_words": ["erupting", "colliding", "fracturing", "dissolving", "transforming"]
            },
            "euphoric": {
                "rhythm": "dancing",
                "tone": "vibrant",
                "punctuation": "~",
                "vocabulary_shift": "elevated",
                "sentence_length": "effusive",
                "repetition_factor": 0.4,
                "fragmentary_factor": 0.2,
                "amplification_words": ["resonating", "pulsing", "radiating", "flowering", "ascending"]
            },
            "cryptic": {
                "rhythm": "coded",
                "tone": "mysterious",
                "punctuation": ":",
                "vocabulary_shift": "enigmatic",
                "sentence_length": "encoded",
                "repetition_factor": 0.3,
                "fragmentary_factor": 0.4,
                "amplification_words": ["hidden", "veiled", "encrypted", "occluded", "ciphered"]
            }
        }
        
        # Vocabulary modulation tables
        self.vocabulary_shifts = {
            "intensified": {
                "modifiers": ["intensely", "brilliantly", "profoundly", "overwhelmingly", "magnificently"],
                "verbs": ["erupts", "transcends", "illuminates", "transforms", "amplifies"],
                "adjectives": ["radiant", "boundless", "ecstatic", "sublime", "magnificent"]
            },
            "abstract": {
                "modifiers": ["potentially", "abstractly", "conceptually", "theoretically", "philosophically"],
                "verbs": ["embodies", "represents", "signifies", "manifests", "symbolizes"],
                "adjectives": ["conceptual", "archetypal", "symbolic", "essential", "fundamental"]
            },
            "darkened": {
                "modifiers": ["somberly", "faintly", "distantly", "hauntingly", "vaguely"],
                "verbs": ["fades", "dissolves", "diminishes", "wanes", "recedes"],
                "adjectives": ["somber", "muted", "shadowed", "hollow", "fading"]
            },
            "uncertain": {
                "modifiers": ["perhaps", "possibly", "uncertainly", "tentatively", "hesitantly"],
                "verbs": ["might", "could", "seems", "appears", "suggests"],
                "adjectives": ["uncertain", "tenuous", "ambiguous", "fragile", "precarious"]
            },
            "harmonious": {
                "modifiers": ["harmoniously", "seamlessly", "perfectly", "serenely", "peacefully"],
                "verbs": ["flows", "balances", "harmonizes", "integrates", "unifies"],
                "adjectives": ["harmonious", "balanced", "integrated", "unified", "complete"]
            },
            "discordant": {
                "modifiers": ["jarringly", "discordantly", "chaotically", "violently", "frantically"],
                "verbs": ["clashes", "fractures", "disrupts", "shatters", "overwhelms"],
                "adjectives": ["discordant", "fragmented", "chaotic", "frenzied", "turbulent"]
            },
            "elevated": {
                "modifiers": ["sublimely", "exquisitely", "exaltedly", "transcendently", "gloriously"],
                "verbs": ["elevates", "ascends", "soars", "rises", "transcends"],
                "adjectives": ["sublime", "exalted", "transcendent", "glorious", "celestial"]
            },
            "enigmatic": {
                "modifiers": ["mysteriously", "enigmatically", "cryptically", "esoterically", "obscurely"],
                "verbs": ["conceals", "encodes", "encrypts", "veils", "obscures"],
                "adjectives": ["enigmatic", "mysterious", "cryptic", "esoteric", "arcane"]
            }
        }
        
        # Rhythm patterns
        self.rhythm_patterns = {
            "rapid": {"avg_word_length": 1.5, "punctuation_frequency": 0.1, "comma_frequency": 0.05},
            "measured": {"avg_word_length": 2.0, "punctuation_frequency": 0.3, "comma_frequency": 0.2},
            "slow": {"avg_word_length": 2.5, "punctuation_frequency": 0.4, "comma_frequency": 0.3},
            "staccato": {"avg_word_length": 1.2, "punctuation_frequency": 0.6, "comma_frequency": 0.4},
            "flowing": {"avg_word_length": 2.2, "punctuation_frequency": 0.2, "comma_frequency": 0.1},
            "erratic": {"avg_word_length": "variable", "punctuation_frequency": 0.7, "comma_frequency": 0.5},
                  # Get a random mood profile for metamoodulation
            metamood = random.choice(valid_moods)
            modulated = self._apply_metamoodulation(
                modulated,
                self.mood_profiles[metamood],
                self.config["modulation_intensity"] * self.config["mood_blending_factor"]
            )
        
        # Create modulation record
        record = {
            "id": str(uuid.uuid4()),
            "original": original,
            "modulated": modulated,
            "moods": valid_moods,
            "weights": normalized_weights,
            "intensity": intensity,
            "glyph_influences": glyph_influences or [],
            "mood_results": mood_results,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        self.modulation_log.append(record)
        
        return record

    def get_modulation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get modulation history, optionally limited to a certain number of entries.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of modulation records
        """
        if limit is None:
            return self.modulation_log
        return self.modulation_log[-limit:]

    def clear_modulation_history(self) -> None:
        """Clear all modulation history."""
        self.modulation_log = []

    def evolve_vocabulary(self, intensity: Optional[float] = None) -> None:
        """
        Evolve vocabulary based on current usage patterns.
        
        Args:
            intensity: Evolution intensity (0.0 to 1.0)
        """
        if intensity is None:
            intensity = self.config["vocabulary_evolution_rate"]
        
        for shift_type in self.vocabulary_shifts:
            for category in ["modifiers", "verbs", "adjectives"]:
                if category in self.vocabulary_shifts[shift_type]:
                    # Randomly replace some words with new ones
                    word_list = self.vocabulary_shifts[shift_type][category]
                    for i in range(len(word_list)):
                        if random.random() < intensity:
                            # Create a variation of the word
                            word = word_list[i]
                            if len(word) > 3:
                                # Various mutation strategies
                                mutation = random.choice([
                                    word[:-1] + word[-1] * 2,  # Duplicate last letter
                                    word[:2] + word[2:].capitalize(),  # Capitalize part
                                    word + "ing" if not word.endswith("ing") else word[:-3],  # Add/remove -ing
                                    word + "ly" if not word.endswith("ly") else word[:-2],  # Add/remove -ly
                                    word + "s" if not word.endswith("s") else word[:-1],  # Add/remove plural
                                    word[::-1][:len(word)//2] + word[:len(word)//2],  # Mirror parts
                                ])
                                word_list[i] = mutation.lower()
        
        # Save evolved vocabulary
        self.save_memory()
      "dancing": {"avg_word_length": 1.8, "punctuation_frequency": 0.3, "comma_frequency": 0.2},
