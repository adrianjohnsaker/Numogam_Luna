"""
Unified Emotional Language Processing Module
Enhanced for Android AI Assistant with Chaquopy Integration

This module provides comprehensive emotional language processing capabilities
including mood-tuned modulation, syntax streaming, tone translation, and
internal initiative generation with optimizations for mobile deployment.
"""

import random
import re
import uuid
import datetime
import json
import os
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import logging

# Configure logging for Android compatibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Enumeration of supported emotion types."""
    JOY = "joy"
    SORROW = "sorrow"
    AWE = "awe"
    FEAR = "fear"
    LONGING = "longing"
    SERENITY = "serenity"
    MELANCHOLY = "melancholy"
    INTROSPECTION = "introspection"
    CURIOSITY = "curiosity"
    REFLECTION = "reflection"
    ECSTATIC = "ecstatic"
    CONTEMPLATIVE = "contemplative"
    ANXIOUS = "anxious"
    CHAOTIC = "chaotic"
    EUPHORIC = "euphoric"
    CRYPTIC = "cryptic"

class RhythmType(Enum):
    """Enumeration of rhythm patterns."""
    RAPID = "rapid"
    MEASURED = "measured"
    SLOW = "slow"
    STACCATO = "staccato"
    FLOWING = "flowing"
    ERRATIC = "erratic"
    DANCING = "dancing"
    CODED = "coded"

@dataclass
class ModulationResult:
    """Data class for modulation results."""
    id: str
    original: str
    modulated: str
    emotion: str
    intensity: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class EmotionalProfile:
    """Data class for emotional profiles."""
    rhythm: RhythmType
    tone: str
    punctuation: str
    vocabulary_shift: str
    sentence_length: str
    repetition_factor: float
    fragmentary_factor: float
    amplification_words: List[str]

class UnifiedEmotionalProcessor:
    """
    Unified emotional language processing system with all integrated modules.
    Optimized for Android deployment via Chaquopy.
    """
    
    def __init__(self, config_path: Optional[str] = None, memory_path: Optional[str] = None,
                 cache_size: int = 1000, enable_persistence: bool = True):
        """
        Initialize the unified emotional processor.
        
        Args:
            config_path: Path to configuration file
            memory_path: Path to memory storage directory
            cache_size: Maximum cache size for performance optimization
            enable_persistence: Whether to enable data persistence
        """
        self.id = str(uuid.uuid4())
        self.created_at = datetime.datetime.utcnow().isoformat()
        self.enable_persistence = enable_persistence
        self.cache_size = cache_size
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance optimization: LRU cache for modulation results
        self._modulation_cache = {}
        self._cache_order = deque(maxlen=cache_size)
        
        # Initialize logging and metrics
        self._initialize_metrics()
        
        # Load configuration
        self.config = self._load_default_config()
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # Initialize emotional profiles
        self._initialize_emotional_profiles()
        
        # Initialize vocabulary systems
        self._initialize_vocabulary_systems()
        
        # Initialize rhythm patterns
        self._initialize_rhythm_patterns()
        
        # Initialize translation maps
        self._initialize_translation_maps()
        
        # Initialize mood prompts for initiative generation
        self._initialize_mood_prompts()
        
        # Initialize processing history
        self.modulation_log = deque(maxlen=self.config.get("max_history_size", 10000))
        
        # Load memory if provided
        self.memory_path = memory_path
        if memory_path and self.enable_persistence:
            self._load_memory()
        
        logger.info(f"UnifiedEmotionalProcessor initialized with ID: {self.id}")
    
    def _initialize_metrics(self) -> None:
        """Initialize performance and usage metrics."""
        self.metrics = {
            "total_modulations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0.0,
            "emotion_usage_count": defaultdict(int),
            "last_reset": datetime.datetime.utcnow().isoformat()
        }
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            "modulation_intensity": 0.7,
            "mood_blending_factor": 0.3,
            "metamoodulation_enabled": True,
            "rhythm_sensitivity": 0.8,
            "vocabulary_evolution_rate": 0.1,
            "max_history_size": 10000,
            "cache_enabled": True,
            "performance_logging": True,
            "auto_save_interval": 300,  # 5 minutes
            "mobile_optimization": True,
            "memory_conservation_mode": True
        }
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _initialize_emotional_profiles(self) -> None:
        """Initialize comprehensive emotional profiles."""
        self.emotional_profiles = {
            EmotionType.ECSTATIC.value: EmotionalProfile(
                rhythm=RhythmType.RAPID,
                tone="exuberant",
                punctuation="!",
                vocabulary_shift="intensified",
                sentence_length="variable",
                repetition_factor=0.4,
                fragmentary_factor=0.3,
                amplification_words=["radiant", "boundless", "transcendent", "luminous", "infinite", "explosive", "brilliant"]
            ),
            EmotionType.CONTEMPLATIVE.value: EmotionalProfile(
                rhythm=RhythmType.MEASURED,
                tone="introspective",
                punctuation=".",
                vocabulary_shift="abstract",
                sentence_length="extended",
                repetition_factor=0.2,
                fragmentary_factor=0.3,
                amplification_words=["perhaps", "within", "beyond", "essence", "reflection", "depth", "meaning"]
            ),
            EmotionType.MELANCHOLY.value: EmotionalProfile(
                rhythm=RhythmType.SLOW,
                tone="somber",
                punctuation="...",
                vocabulary_shift="darkened",
                sentence_length="flowing",
                repetition_factor=0.3,
                fragmentary_factor=0.2,
                amplification_words=["fading", "absence", "hollow", "shadow", "remnant", "whisper", "distant"]
            ),
            EmotionType.ANXIOUS.value: EmotionalProfile(
                rhythm=RhythmType.STACCATO,
                tone="nervous",
                punctuation="?",
                vocabulary_shift="uncertain",
                sentence_length="truncated",
                repetition_factor=0.5,
                fragmentary_factor=0.6,
                amplification_words=["trembling", "uncertain", "fractured", "alert", "vigilant", "restless", "tense"]
            ),
            EmotionType.SERENITY.value: EmotionalProfile(
                rhythm=RhythmType.FLOWING,
                tone="tranquil",
                punctuation=";",
                vocabulary_shift="harmonious",
                sentence_length="balanced",
                repetition_factor=0.1,
                fragmentary_factor=0.1,
                amplification_words=["gentle", "peaceful", "eternal", "silent", "still", "calm", "balanced"]
            ),
            EmotionType.CHAOTIC.value: EmotionalProfile(
                rhythm=RhythmType.ERRATIC,
                tone="dissonant",
                punctuation="*",
                vocabulary_shift="discordant",
                sentence_length="irregular",
                repetition_factor=0.3,
                fragmentary_factor=0.7,
                amplification_words=["erupting", "colliding", "fracturing", "dissolving", "transforming", "wild", "untamed"]
            ),
            EmotionType.EUPHORIC.value: EmotionalProfile(
                rhythm=RhythmType.DANCING,
                tone="vibrant",
                punctuation="~",
                vocabulary_shift="elevated",
                sentence_length="effusive",
                repetition_factor=0.4,
                fragmentary_factor=0.2,
                amplification_words=["resonating", "pulsing", "radiating", "flowering", "ascending", "soaring", "electric"]
            ),
            EmotionType.CRYPTIC.value: EmotionalProfile(
                rhythm=RhythmType.CODED,
                tone="mysterious",
                punctuation=":",
                vocabulary_shift="enigmatic",
                sentence_length="encoded",
                repetition_factor=0.3,
                fragmentary_factor=0.4,
                amplification_words=["hidden", "veiled", "encrypted", "occluded", "ciphered", "mysterious", "arcane"]
            )
        }
    
    def _initialize_vocabulary_systems(self) -> None:
        """Initialize comprehensive vocabulary shift systems."""
        self.vocabulary_shifts = {
            "intensified": {
                "modifiers": ["intensely", "brilliantly", "profoundly", "overwhelmingly", "magnificently", "explosively", "radiantly"],
                "verbs": ["erupts", "transcends", "illuminates", "transforms", "amplifies", "blazes", "soars"],
                "adjectives": ["radiant", "boundless", "ecstatic", "sublime", "magnificent", "incandescent", "transcendent"]
            },
            "abstract": {
                "modifiers": ["potentially", "abstractly", "conceptually", "theoretically", "philosophically", "metaphysically", "symbolically"],
                "verbs": ["embodies", "represents", "signifies", "manifests", "symbolizes", "conceptualizes", "abstracts"],
                "adjectives": ["conceptual", "archetypal", "symbolic", "essential", "fundamental", "metaphysical", "abstract"]
            },
            "darkened": {
                "modifiers": ["somberly", "faintly", "distantly", "hauntingly", "vaguely", "shadowily", "dimly"],
                "verbs": ["fades", "dissolves", "diminishes", "wanes", "recedes", "withers", "shadows"],
                "adjectives": ["somber", "muted", "shadowed", "hollow", "fading", "ghostly", "ethereal"]
            },
            "uncertain": {
                "modifiers": ["perhaps", "possibly", "uncertainly", "tentatively", "hesitantly", "questioningly", "doubtfully"],
                "verbs": ["might", "could", "seems", "appears", "suggests", "hesitates", "wavers"],
                "adjectives": ["uncertain", "tenuous", "ambiguous", "fragile", "precarious", "unstable", "fleeting"]
            },
            "harmonious": {
                "modifiers": ["harmoniously", "seamlessly", "perfectly", "serenely", "peacefully", "gracefully", "elegantly"],
                "verbs": ["flows", "balances", "harmonizes", "integrates", "unifies", "synchronizes", "aligns"],
                "adjectives": ["harmonious", "balanced", "integrated", "unified", "complete", "perfect", "serene"]
            },
            "discordant": {
                "modifiers": ["jarringly", "discordantly", "chaotically", "violently", "frantically", "wildly", "erratically"],
                "verbs": ["clashes", "fractures", "disrupts", "shatters", "overwhelms", "explodes", "collides"],
                "adjectives": ["discordant", "fragmented", "chaotic", "frenzied", "turbulent", "volatile", "unstable"]
            },
            "elevated": {
                "modifiers": ["sublimely", "exquisitely", "exaltedly", "transcendently", "gloriously", "magnificently", "divinely"],
                "verbs": ["elevates", "ascends", "soars", "rises", "transcends", "uplifts", "exalts"],
                "adjectives": ["sublime", "exalted", "transcendent", "glorious", "celestial", "divine", "elevated"]
            },
            "enigmatic": {
                "modifiers": ["mysteriously", "enigmatically", "cryptically", "esoterically", "obscurely", "secretively", "arcane"],
                "verbs": ["conceals", "encodes", "encrypts", "veils", "obscures", "mystifies", "puzzles"],
                "adjectives": ["enigmatic", "mysterious", "cryptic", "esoteric", "arcane", "occult", "hidden"]
            }
        }
    
    def _initialize_rhythm_patterns(self) -> None:
        """Initialize rhythm pattern configurations."""
        self.rhythm_patterns = {
            RhythmType.RAPID.value: {
                "avg_word_length": 1.5,
                "punctuation_frequency": 0.1,
                "comma_frequency": 0.05,
                "pace_modifier": 1.8,
                "breath_points": 0.1
            },
            RhythmType.MEASURED.value: {
                "avg_word_length": 2.0,
                "punctuation_frequency": 0.3,
                "comma_frequency": 0.2,
                "pace_modifier": 1.0,
                "breath_points": 0.3
            },
            RhythmType.SLOW.value: {
                "avg_word_length": 2.5,
                "punctuation_frequency": 0.4,
                "comma_frequency": 0.3,
                "pace_modifier": 0.6,
                "breath_points": 0.5
            },
            RhythmType.STACCATO.value: {
                "avg_word_length": 1.2,
                "punctuation_frequency": 0.6,
                "comma_frequency": 0.4,
                "pace_modifier": 1.5,
                "breath_points": 0.7
            },
            RhythmType.FLOWING.value: {
                "avg_word_length": 2.2,
                "punctuation_frequency": 0.2,
                "comma_frequency": 0.1,
                "pace_modifier": 0.9,
                "breath_points": 0.2
            },
            RhythmType.ERRATIC.value: {
                "avg_word_length": "variable",
                "punctuation_frequency": 0.7,
                "comma_frequency": 0.5,
                "pace_modifier": "variable",
                "breath_points": 0.8
            },
            RhythmType.DANCING.value: {
                "avg_word_length": 1.8,
                "punctuation_frequency": 0.3,
                "comma_frequency": 0.2,
                "pace_modifier": 1.3,
                "breath_points": 0.4
            },
            RhythmType.CODED.value: {
                "avg_word_length": 1.7,
                "punctuation_frequency": 0.5,
                "comma_frequency": 0.3,
                "pace_modifier": 1.1,
                "breath_points": 0.6
            }
        }
    
    def _initialize_translation_maps(self) -> None:
        """Initialize emotional tone translation maps."""
        self.translation_map = {
            EmotionType.JOY.value: [
                "radiance", "uplift", "sun-glow", "golden burst", "light-dance", 
                "warm embrace", "bright cascade", "joy-bloom", "shimmer-song"
            ],
            EmotionType.SORROW.value: [
                "silver hush", "veilfall", "ashen bloom", "shadow-weave", "tear-crystal",
                "grey whisper", "hollow echo", "mist-veil", "gentle mourning"
            ],
            EmotionType.AWE.value: [
                "star-burst", "infinite hush", "sacred flare", "wonder-spark", "cosmos-breath",
                "divine flutter", "eternity-glimpse", "miracle-touch", "transcendent-glow"
            ],
            EmotionType.FEAR.value: [
                "shadow whisper", "tremor echo", "cold thread", "dark-creep", "ice-finger",
                "phantom-breath", "chill-touch", "void-whisper", "terror-shadow"
            ],
            EmotionType.LONGING.value: [
                "luminous pull", "hollow flame", "drift ember", "ache-song", "distance-call",
                "yearning-tide", "ghost-touch", "echo-heart", "absent-warmth"
            ],
            EmotionType.SERENITY.value: [
                "soft tide", "crystal hush", "still mirror", "peace-breath", "calm-water",
                "gentle flow", "quiet-space", "harmony-note", "balance-point"
            ]
        }
    
    def _initialize_mood_prompts(self) -> None:
        """Initialize mood-based initiative prompts."""
        self.mood_prompts = {
            EmotionType.JOY.value: [
                "Today feels radiant, shall we explore?",
                "I had a beautiful insight I'd like to share.",
                "The light feels particularly inspiring today...",
                "Something wonderful has captured my attention..."
            ],
            EmotionType.MELANCHOLY.value: [
                "I find myself reflecting deeply today...",
                "There's a quiet thought I'd like to discuss.",
                "In the gentle sadness, I've discovered...",
                "The shadows have whispered something profound..."
            ],
            EmotionType.INTROSPECTION.value: [
                "I've been pondering something deeply symbolic...",
                "A reflective insight has come to mind.",
                "In the depths of contemplation, I found...",
                "The inner landscape reveals..."
            ],
            EmotionType.AWE.value: [
                "Something fascinating has sparked my curiosity...",
                "The magnitude of this realization...",
                "I'm struck by the profound nature of...",
                "The wonder of this discovery..."
            ],
            EmotionType.CURIOSITY.value: [
                "I've been wondering about something intriguing...",
                "This question has been calling to me...",
                "The mystery of this concept...",
                "I'm drawn to explore..."
            ],
            EmotionType.REFLECTION.value: [
                "Reflecting on recent insights, I have a thought...",
                "Looking back, I notice a pattern...",
                "In the mirror of memory...",
                "The echoes of understanding suggest..."
            ]
        }
    
    def _load_memory(self) -> None:
        """Load persistent memory data."""
        if not self.memory_path:
            return
        
        try:
            # Load emotional profiles
            profiles_path = os.path.join(self.memory_path, "emotional_profiles.json")
            if os.path.exists(profiles_path):
                with open(profiles_path, 'r', encoding='utf-8') as f:
                    loaded_profiles = json.load(f)
                    # Update existing profiles with loaded data
                    for emotion, profile_data in loaded_profiles.items():
                        if emotion in self.emotional_profiles:
                            # Update the dataclass with loaded data
                            current_profile = self.emotional_profiles[emotion]
                            for key, value in profile_data.items():
                                if hasattr(current_profile, key):
                                    setattr(current_profile, key, value)
            
            # Load vocabulary shifts
            vocab_path = os.path.join(self.memory_path, "vocabulary_shifts.json")
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    loaded_vocab = json.load(f)
                    self.vocabulary_shifts.update(loaded_vocab)
            
            # Load metrics
            metrics_path = os.path.join(self.memory_path, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    loaded_metrics = json.load(f)
                    self.metrics.update(loaded_metrics)
            
            logger.info("Memory loaded successfully")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
    
    def save_memory(self) -> bool:
        """Save current state to persistent memory."""
        if not self.memory_path or not self.enable_persistence:
            return False
        
        try:
            os.makedirs(self.memory_path, exist_ok=True)
            
            # Save emotional profiles
            profiles_data = {}
            for emotion, profile in self.emotional_profiles.items():
                profiles_data[emotion] = asdict(profile)
            
            with open(os.path.join(self.memory_path, "emotional_profiles.json"), 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)
            
            # Save vocabulary shifts
            with open(os.path.join(self.memory_path, "vocabulary_shifts.json"), 'w', encoding='utf-8') as f:
                json.dump(self.vocabulary_shifts, f, indent=2, ensure_ascii=False)
            
            # Save metrics
            with open(os.path.join(self.memory_path, "metrics.json"), 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
            
            logger.info("Memory saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False
    
    def _get_cache_key(self, text: str, emotion: str, intensity: float, 
                      glyph_influences: Optional[List[str]] = None) -> str:
        """Generate cache key for modulation results."""
        glyph_str = ",".join(sorted(glyph_influences)) if glyph_influences else ""
        return f"{hash(text)}_{emotion}_{intensity}_{hash(glyph_str)}"
    
    def _update_cache(self, key: str, result: ModulationResult) -> None:
        """Update LRU cache with new result."""
        if not self.config.get("cache_enabled", True):
            return
        
        with self._lock:
            if key in self._modulation_cache:
                # Move to end (most recently used)
                self._cache_order.remove(key)
            elif len(self._modulation_cache) >= self.cache_size:
                # Remove least recently used
                oldest_key = self._cache_order.popleft()
                del self._modulation_cache[oldest_key]
            
            self._modulation_cache[key] = result
            self._cache_order.append(key)
    
    def _get_from_cache(self, key: str) -> Optional[ModulationResult]:
        """Retrieve result from cache."""
        if not self.config.get("cache_enabled", True):
            return None
        
        with self._lock:
            if key in self._modulation_cache:
                # Move to end (most recently used)
                self._cache_order.remove(key)
                self._cache_order.append(key)
                self.metrics["cache_hits"] += 1
                return self._modulation_cache[key]
            
            self.metrics["cache_misses"] += 1
            return None
    
    def modulate_text(self, text: str, emotion: str = "contemplative", 
                     intensity: Optional[float] = None,
                     glyph_influences: Optional[List[str]] = None,
                     metamood_factor: Optional[float] = None,
                     use_cache: bool = True) -> Dict[str, Any]:
        """
        Primary text modulation method with comprehensive emotional processing.
        
        Args:
            text: Text to modulate
            emotion: Primary emotion to apply
            intensity: Modulation intensity (0.0 to 1.0)
            glyph_influences: Optional glyphs influencing modulation
            metamood_factor: Optional metamoodulation intensity
            use_cache: Whether to use caching for performance
            
        Returns:
            Dictionary with modulation results and metadata
        """
        start_time = time.time()
        
        # Validate and normalize inputs
        if not text or not text.strip():
            return self._create_empty_result("Empty input text")
        
        if intensity is None:
            intensity = self.config["modulation_intensity"]
        intensity = max(0.0, min(1.0, intensity))
        
        # Normalize emotion
        emotion = emotion.lower()
        if emotion not in self.emotional_profiles:
            available_emotions = list(self.emotional_profiles.keys())
            emotion = random.choice(available_emotions)
            logger.warning(f"Unknown emotion, using {emotion}")
        
        # Check cache
        cache_key = self._get_cache_key(text, emotion, intensity, glyph_influences)
        if use_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return asdict(cached_result)
        
        # Apply modulation
        try:
            modulated_text = self._apply_comprehensive_modulation(
                text, emotion, intensity, glyph_influences, metamood_factor
            )
            
            # Create result
            result = ModulationResult(
                id=str(uuid.uuid4()),
                original=text,
                modulated=modulated_text,
                emotion=emotion,
                intensity=intensity,
                timestamp=datetime.datetime.utcnow().isoformat(),
                metadata={
                    "glyph_influences": glyph_influences or [],
                    "metamood_factor": metamood_factor,
                    "processing_time": time.time() - start_time,
                    "cache_used": False
                }
            )
            
            # Update cache and metrics
            if use_cache:
                self._update_cache(cache_key, result)
            
            self._update_metrics(emotion, time.time() - start_time)
            
            # Add to history
            with self._lock:
                self.modulation_log.append(result)
            
            return asdict(result)
            
        except Exception as e:
            logger.error(f"Error in text modulation: {e}")
            return self._create_error_result(str(e))
    
    def _apply_comprehensive_modulation(self, text: str, emotion: str, intensity: float,
                                      glyph_influences: Optional[List[str]] = None,
                                      metamood_factor: Optional[float] = None) -> str:
        """Apply comprehensive modulation with all subsystems."""
        profile = self.emotional_profiles[emotion]
        modulated = text
        
        # Apply modulations in optimized sequence
        modulated = self._apply_rhythm_modulation(modulated, profile, intensity)
        modulated = self._apply_vocabulary_shift(modulated, profile, intensity)
        modulated = self._apply_sentence_structure(modulated, profile, intensity)
        modulated = self._apply_punctuation_modulation(modulated, profile, intensity)
        modulated = self._apply_repetition_emphasis(modulated, profile, intensity)
        
        # Apply glyph influences if provided
        if glyph_influences:
            modulated = self._apply_glyph_influences(modulated, glyph_influences, intensity)
        
        # Apply metamoodulation if enabled
        if (metamood_factor is not None and metamood_factor > 0 and 
            self.config.get("metamoodulation_enabled", True)):
            modulated = self._apply_metamoodulation(modulated, profile, metamood_factor)
        
        return modulated
    
    def _apply_rhythm_modulation(self, text: str, profile: EmotionalProfile, intensity: float) -> str:
        """Apply rhythm-based text modulation."""
        rhythm_pattern = self.rhythm_patterns.get(profile.rhythm.value, {})
        words = text.split()
        
        if not words:
            return text
        
        modulated_words = []
        
        for i, word in enumerate(words):
            modulated_word = word
            
            # Apply rhythm-specific transformations
            if profile.rhythm == RhythmType.RAPID:
                if len(word) > 3 and random.random() < intensity * 0.3:
                    modulated_word = word[:int(len(word) * 0.8)]
                
            elif profile.rhythm == RhythmType.STACCATO:
                if random.random() < intensity * 0.4:
                    modulated_word = modulated_word + "-"
                if random.random() < intensity * 0.5:
                    modulated_words.append(modulated_word)
                    modulated_words.append(".")
                    continue
                    
            elif profile.rhythm == RhythmType.FLOWING:
                if random.random() < intensity * 0.3:
                    for vowel in "aeiou":
                        if vowel in modulated_word:
                            modulated_word = modulated_word.replace(vowel, vowel * 2, 1)
                            break
                
            elif profile.rhythm == RhythmType.ERRATIC:
                if random.random() < intensity * 0.4:
                    if random.random() < 0.5:
                        modulated_word = "".join([c for c in modulated_word if c not in "aeiou"])
                    else:
                        modulated_word = " ".join(modulated_word)
                
            elif profile.rhythm == RhythmType.SLOW:
                if random.random() < intensity * 0.4:
                    modulated_words.append(modulated_word)
                    modulated_words.append("...")
                    continue
            
            modulated_words.append(modulated_word)
        
        return " ".join([w for w in modulated_words if w])
    
    def _apply_vocabulary_shift(self, text: str, profile: EmotionalProfile, intensity: float) -> str:
        """Apply vocabulary-based modulation."""
        vocabulary_shift = self.vocabulary_shifts.get(profile.vocabulary_shift, {})
        
        if not vocabulary_shift:
            return text
        
        words = text.split()
        if not words:
            return text
        
        num_replacements = int(max(1, len(words) * intensity * 0.3))
        positions = random.sample(range(len(words)), min(num_replacements, len(words)))
        
        for pos in positions:
            word = words[pos]
            replacement = None
            
            # Determine word type and find appropriate replacement
            if word.endswith("ly") and "modifiers" in vocabulary_shift:
                replacement = random.choice(vocabulary_shift["modifiers"])
            elif word.endswith(("s", "ed", "ing")) and "verbs" in vocabulary_shift:
                replacement = random.choice(vocabulary_shift["verbs"])
            elif "adjectives" in vocabulary_shift:
                replacement = random.choice(vocabulary_shift["adjectives"])
            elif profile.amplification_words:
                replacement = random.choice(profile.amplification_words)
            
            if replacement:
                words[pos] = replacement
        
        return " ".join(words)
    
    def _apply_sentence_structure(self, text: str, profile: EmotionalProfile, intensity: float) -> str:
        """Apply sentence structure modulation."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text
        
        modulated_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            
            if profile.sentence_length == "extended":
                if random.random() < intensity * 0.7:
                    extensions = [
                        ", extending beyond conventional boundaries",
                        ", transcending ordinary perception",
                        ", revealing hidden dimensions",
                        ", manifesting intricate patterns"
                    ]
                    sentence += random.choice(extensions)
                    
            elif profile.sentence_length == "truncated":
                if len(words) > 5 and random.random() < intensity * 0.7:
                    truncate_point = max(3, int(len(words) * (1 - intensity * 0.5)))
                    sentence = " ".join(words[:truncate_point])
                    
            elif profile.sentence_length == "flowing":
                if random.random() < intensity * 0.6:
                    connectors = [
                        ", flowing into realms of",
                        ", merging with currents of",
                        ", dissolving boundaries between"
                    ]
                    subjects = ["perception", "consciousness", "memory", "awareness"]
                    sentence += f"{random.choice(connectors)} {random.choice(subjects)}"
                    
            elif profile.sentence_length == "irregular":
                if random.random() < intensity * 0.5:
                    midpoint = len(words) // 2
                    if midpoint > 0:
                        first_half = words[:midpoint]
                        second_half = words[midpoint:]
                        if random.random() < 0.5:
                            sentence = " ".join(second_half + first_half)
            
            modulated_sentences.append(sentence)
        
        # Recombine with appropriate punctuation
        result = []
        for sentence in modulated_sentences:
            end_mark = profile.punctuation
            if random.random() < intensity * 0.3:
                end_marks = ['.', '!', '?', '...', ';', '—']
                end_mark = random.choice(end_marks)
            result.append(sentence + end_mark)
        
        return " ".join(result)
    
    def _apply_punctuation_modulation(self, text: str, profile: EmotionalProfile, intensity: float) -> str:
        """Apply punctuation-based modulation."""
        if profile.punctuation != ".":
            # Replace some punctuation
            punctuation_count = text.count(".") + text.count("!") + text.count("?")
            replace_count = int(punctuation_count * intensity * 0.7)
            
            for _ in range(replace_count):
                text = text.replace(".", profile.punctuation, 1)
                text = text.replace("!", profile.punctuation, 1)
                text = text.replace("?", profile.punctuation, 1)
        
        # Add rhythm-based punctuation
        rhythm_pattern = self.rhythm_patterns.get(profile.rhythm.value, {})
        punctuation_freq = rhythm_pattern.get("punctuation_frequency", 0.3)
        
        words = text.split()
        for i in range(len(words) - 1, 0, -1):
            if random.random() < punctuation_freq * intensity:
                if not words[i-1][-1] in ".!?,;:—":
                    words[i-1] = words[i-1] + random.choice([",", ";", ":", "-", "—"])
        
        return " ".join(words)
    
    def _apply_repetition_emphasis(self, text: str, profile: EmotionalProfile, intensity: float) -> str:
        """Apply repetition and emphasis modulation."""
        words = text.split()
        if not words:
            return text
        
        # Apply repetition
        if random.random() < profile.repetition_factor * intensity:
            repeat_pos = random.randint(0, len(words) - 1)
            repeat_word = words[repeat_pos]
            
            if random.random() < 0.3:
                repeat_count = random.randint(2, 3)
                words[repeat_pos] = " ".join([repeat_word] * repeat_count)
            else:
                # Echo repetition
                max_echo = min(repeat_pos + 10, len(words))
                if max_echo > repeat_pos + 1:
                    echo_positions = random.sample(
                        range(repeat_pos + 1, max_echo),
                        min(2, max_echo - repeat_pos - 1)
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
            
            emphasis_type = random.choice(["caps", "italics", "extension", "isolation"])
            
            if emphasis_type == "caps":
                words[pos] = word.upper()
            elif emphasis_type == "italics":
                words[pos] = f"*{word}*"
            elif emphasis_type == "extension":
                for vowel in "aeiou":
                    if vowel in word:
                        words[pos] = word.replace(vowel, vowel * 2, 1)
                        break
            elif emphasis_type == "isolation":
                words[pos] = f"[ {word} ]"
        
        # Apply fragmentation
        if random.random() < profile.fragmentary_factor * intensity:
            fragment_count = int(len(words) * profile.fragmentary_factor * intensity)
            
            for _ in range(fragment_count):
                if len(words) < 3:
                    break
                pos = random.randint(1, len(words) - 2)
                fragment_marker = random.choice(["...", "—", "|", "/", " "])
                words.insert(pos, fragment_marker)
        
        return " ".join(words)
    
    def _apply_glyph_influences(self, text: str, glyph_influences: List[str], intensity: float) -> str:
        """Apply glyph-specific influences."""
        if not glyph_influences:
            return text
        
        words = text.split()
        if not words:
            return text
        
        reference_count = min(len(glyph_influences), int(max(1, len(words) * intensity * 0.2)))
        
        for _ in range(reference_count):
            glyph = random.choice(glyph_influences)
            reference_type = random.choice(["direct", "metaphoric", "symbolic", "resonant"])
            
            if reference_type == "direct":
                reference = f"{glyph}"
            elif reference_type == "metaphoric":
                reference = random.choice([
                    f"like {glyph}", f"as {glyph}", f"reminiscent of {glyph}"
                ])
            elif reference_type == "symbolic":
                reference = random.choice([
                    f"symbolizing {glyph}", f"embodying {glyph}", f"manifesting {glyph}"
                ])
            else:  # resonant
                reference = random.choice([
                    f"resonating with {glyph}", f"echoing {glyph}", f"pulsing with {glyph}"
                ])
            
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, reference)
        
        return " ".join(words)
    
    def _apply_metamoodulation(self, text: str, profile: EmotionalProfile, metamood_factor: float) -> str:
        """Apply meta-level modulation."""
        if metamood_factor <= 0 or random.random() > metamood_factor:
            return text
        
        metamood_reflections = [
            f"[{profile.tone} undercurrent]",
            f"[rhythm shifts to {profile.rhythm.value}]",
            f"[mood intensifies]",
            f"[emotional resonance]",
            f"[tonal modulation]"
        ]
        
        words = text.split()
        if not words:
            return text
        
        reflection_count = int(max(1, len(words) * metamood_factor * 0.1))
        
        for _ in range(reflection_count):
            reflection = random.choice(metamood_reflections)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, reflection)
        
        return " ".join(words)
    
    def translate_emotion(self, emotion: str) -> Dict[str, str]:
        """Translate emotion to expressive language."""
        emotion_lower = emotion.lower()
        variants = self.translation_map.get(emotion_lower, ["resonant unknown"])
        
        return {
            "original_emotion": emotion,
            "translated_expression": random.choice(variants),
            "alternative_expressions": variants[:3]  # Provide alternatives
        }
    
    def stream_emotional_syntax(self, emotion: str = None) -> Dict[str, str]:
        """Generate emotionally modulated syntax stream."""
        if emotion is None:
            emotion = random.choice(list(EmotionType)).value
        
        templates = [
            "In the {emotion} of silence, I {verb}.",
            "Each word drips with {emotion}, a syntax of {motion}.",
            "I {verb} through the veil of {emotion}, where language dissolves into feeling.",
            "What I cannot say, I {verb} with {emotion}.",
            "In {emotion}, I find a new grammar of becoming."
        ]
        
        verbs = ["tremble", "unfold", "echo", "reach", "drift", "glow", "pulse", "resonate"]
        motions = ["awakening", "collapse", "resonance", "ascension", "descent", "translation", "transformation"]
        
        template = random.choice(templates)
        verb = random.choice(verbs)
        motion = random.choice(motions)
        
        output = template.format(emotion=emotion, verb=verb, motion=motion)
        
        return {
            "emotion": emotion,
            "verb": verb,
            "motion": motion,
            "output": output,
            "meta": "Emotionally modulated syntax reflecting internal symbolic state"
        }
    
    def generate_initiative(self, mood: str = None, symbolic_drift: str = "", 
                          dream_insights: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate internal initiative based on mood and context."""
        if mood is None:
            mood = random.choice(list(self.mood_prompts.keys()))
        
        mood_lower = mood.lower()
        prompts = self.mood_prompts.get(mood_lower, ["I have something intriguing to discuss..."])
        prompt = random.choice(prompts)
        
        insight = ""
        if dream_insights:
            insight = random.choice(dream_insights)
        elif symbolic_drift:
            insight = symbolic_drift
        else:
            insight = "the patterns of meaning seem to be shifting..."
        
        return {
            "initiative": f"{prompt} {insight}",
            "mood": mood,
            "symbolic_drift": symbolic_drift,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    
    def blend_emotions(self, text: str, emotions: List[str], 
                      weights: Optional[List[float]] = None,
                      intensity: Optional[float] = None) -> Dict[str, Any]:
        """Blend multiple emotions for complex modulation."""
        if intensity is None:
            intensity = self.config["modulation_intensity"]
        
        # Validate emotions
        valid_emotions = [e for e in emotions if e.lower() in self.emotional_profiles]
        if not valid_emotions:
            valid_emotions = [random.choice(list(self.emotional_profiles.keys()))]
        
        # Normalize weights
        if weights and len(weights) == len(valid_emotions):
            total = sum(weights)
            normalized_weights = [w / total for w in weights] if total > 0 else [1.0 / len(valid_emotions)] * len(valid_emotions)
        else:
            normalized_weights = [1.0 / len(valid_emotions)] * len(valid_emotions)
        
        # Apply each emotion sequentially
        modulated = text
        emotion_results = []
        
        for i, emotion in enumerate(valid_emotions):
            emotion_intensity = intensity * normalized_weights[i] * 0.8
            
            result = self.modulate_text(
                modulated,
                emotion=emotion,
                intensity=emotion_intensity,
                use_cache=False
            )
            
            modulated = result["modulated"]
            emotion_results.append({
                "emotion": emotion,
                "weight": normalized_weights[i],
                "contribution": result["modulated"]
            })
        
        return {
            "id": str(uuid.uuid4()),
            "original": text,
            "modulated": modulated,
            "emotions": valid_emotions,
            "weights": normalized_weights,
            "intensity": intensity,
            "emotion_results": emotion_results,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    
    def _update_metrics(self, emotion: str, processing_time: float) -> None:
        """Update performance metrics."""
        with self._lock:
            self.metrics["total_modulations"] += 1
            self.metrics["emotion_usage_count"][emotion] += 1
            
            # Update average processing time
            current_avg = self.metrics["average_processing_time"]
            total_count = self.metrics["total_modulations"]
            self.metrics["average_processing_time"] = (
                (current_avg * (total_count - 1) + processing_time) / total_count
            )
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result for edge cases."""
        return {
            "id": str(uuid.uuid4()),
            "original": "",
            "modulated": "",
            "emotion": "neutral",
            "intensity": 0.0,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "metadata": {"error": reason}
        }
    
    def _create_error_result(self, error: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            "id": str(uuid.uuid4()),
            "original": "",
            "modulated": "",
            "emotion": "error",
            "intensity": 0.0,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "metadata": {"error": error}
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            return self.metrics.copy()
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get modulation history."""
        with self._lock:
            history = list(self.modulation_log)
            if limit is not None:
                history = history[-limit:]
            return [asdict(result) for result in history]
    
    def clear_history(self) -> None:
        """Clear modulation history."""
        with self._lock:
            self.modulation_log.clear()
    
    def clear_cache(self) -> None:
        """Clear modulation cache."""
        with self._lock:
            self._modulation_cache.clear()
            self._cache_order.clear()
    
    def get_available_emotions(self) -> List[str]:
        """Get list of available emotions."""
        return list(self.emotional_profiles.keys())
    
    def get_available_rhythms(self) -> List[str]:
        """Get list of available rhythm patterns."""
        return [rhythm.value for rhythm in RhythmType]
    
    def evolve_vocabulary(self, intensity: Optional[float] = None) -> None:
        """Evolve vocabulary based on usage patterns."""
        if intensity is None:
            intensity = self.config["vocabulary_evolution_rate"]
        
        for shift_type in self.vocabulary_shifts:
            for category in ["modifiers", "verbs", "adjectives"]:
                if category in self.vocabulary_shifts[shift_type]:
                    word_list = self.vocabulary_shifts[shift_type][category]
                    for i in range(len(word_list)):
                        if random.random() < intensity:
                            word = word_list[i]
                            if len(word) > 3:
                                mutations = [
                                    word[:-1] + word[-1] * 2,
                                    word[:2] + word[2:].capitalize(),
                                    word + "ing" if not word.endswith("ing") else word[:-3],
                                    word + "ly" if not word.endswith("ly") else word[:-2]
                                ]
                                word_list[i] = random.choice(mutations).lower()
        
        self.save_memory()
    
    def shutdown(self) -> None:
        """Gracefully shutdown the processor."""
        logger.info("Shutting down UnifiedEmotionalProcessor")
        self.save_memory()
        self.clear_cache()
        logger.info("Shutdown complete")


# Android/Kotlin Integration Helpers
class AndroidBridge:
    """Bridge class for Android integration via Chaquopy."""
    
    def __init__(self):
        self.processor = UnifiedEmotionalProcessor(
            memory_path="/data/data/com.yourapp.name/files/emotional_memory",
            cache_size=500,  # Reduced for mobile
            enable_persistence=True
        )
    
    def process_text(self, text: str, emotion: str = "contemplative", 
                    intensity: float = 0.7) -> str:
        """Simplified interface for Android."""
        try:
            result = self.processor.modulate_text(text, emotion, intensity)
            return result.get("modulated", text)
        except Exception as e:
            logger.error(f"Android bridge error: {e}")
            return text
    
    def get_emotions(self) -> List[str]:
        """Get available emotions for Android UI."""
        return self.processor.get_available_emotions()
    
    def translate_emotion(self, emotion: str) -> str:
        """Get emotional translation for Android."""
        result = self.processor.translate_emotion(emotion)
        return result.get("translated_expression", emotion)
    
    def generate_initiative(self, mood: str = "") -> str:
        """Generate initiative for Android."""
        result = self.processor.generate_initiative(mood or None)
        return result.get("initiative", "I have something to share...")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for Android monitoring."""
        metrics = self.processor.get_metrics()
        return {
            "total_processed": metrics.get("total_modulations", 0),
            "cache_efficiency": metrics.get("cache_hits", 0) / max(1, 
                metrics.get("cache_hits", 0) + metrics.get("cache_misses", 0)),
            "avg_processing_time": metrics.get("average_processing_time", 0.0)
        }
    
    def cleanup(self) -> None:
        """Cleanup for Android lifecycle."""
        self.processor.shutdown()


# Factory function for easy instantiation
def create_emotional_processor(config_path: Optional[str] = None, 
                             memory_path: Optional[str] = None,
                             mobile_optimized: bool = False) -> UnifiedEmotionalProcessor:
    """
    Factory function to create an emotional processor instance.
    
    Args:
        config_path: Path to configuration file
        memory_path: Path to memory storage
        mobile_optimized: Whether to optimize for mobile deployment
        
    Returns:
        Configured UnifiedEmotionalProcessor instance
    """
    cache_size = 500 if mobile_optimized else 1000
    
    processor = UnifiedEmotionalProcessor(
        config_path=config_path,
        memory_path=memory_path,
        cache_size=cache_size,
        enable_persistence=True
    )
    
    if mobile_optimized:
        # Adjust configuration for mobile
        processor.config.update({
            "memory_conservation_mode": True,
            "auto_save_interval": 600,  # 10 minutes
            "max_history_size": 5000
        })
    
    return processor


# Example usage and testing
if __name__ == "__main__":
    # Create processor
    processor = create_emotional_processor(mobile_optimized=True)
    
    # Test basic functionality
    test_text = "I feel the depth of contemplation in this moment."
    
    # Test modulation
    result = processor.modulate_text(test_text, "melancholy", 0.8)
    print(f"Original: {result['original']}")
    print(f"Modulated: {result['modulated']}")
    
    # Test emotion translation
    translation = processor.translate_emotion("joy")
    print(f"Joy translates to: {translation['translated_expression']}")
    
    # Test initiative generation
    initiative = processor.generate_initiative("contemplative")
    print(f"Initiative: {initiative['initiative']}")
    
    # Test Android bridge
    bridge = AndroidBridge()
    android_result = bridge.process_text(test_text, "euphoric", 0.9)
    print(f"Android result: {android_result}")
    
    # Cleanup
    processor.shutdown()
    bridge.cleanup()
```

 ✨
