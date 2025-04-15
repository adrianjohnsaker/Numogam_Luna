#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circadian Narrative Cycles Module
Phase XIII - Advanced Narrative Intelligence

A sophisticated system for dynamic narrative generation and transformation
based on temporal and contextual modulations.
"""

import datetime
import json
import random
import math
import logging
import threading
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict

# Enhanced Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('circadian_narrative.log', encoding='utf-8')
    ]
)

@dataclass
class NarrativePeriod:
    """Structured representation of a narrative period."""
    start: str
    end: str
    tone: str
    themes: List[str] = field(default_factory=list)
    linguistic_patterns: List[str] = field(default_factory=list)

@dataclass
class NarrativeModifier:
    """Structured representation of narrative modifiers."""
    symbol_emphasis: List[str] = field(default_factory=list)
    color_palette: List[str] = field(default_factory=list)
    tempo: str = ""
    emotional_quality: str = ""

class CircadianNarrativeCyclesError(Exception):
    """Custom exception for Circadian Narrative Cycles."""
    pass

class CircadianNarrativeCycles:
    """
    Advanced narrative generation system that dynamically modulates 
    storytelling based on temporal and contextual factors.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None, 
                 log_level: int = logging.INFO):
        """
        Initialize the Circadian Narrative Cycles system.
        
        Args:
            config_path: Optional path to a JSON configuration file
            log_level: Logging verbosity level
        """
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Default configuration
        self.time_periods = {
            # [Your existing time_periods dictionary]
        }
        
        self.narrative_modifiers = {
            # [Your existing narrative_modifiers dictionary]
        }
        
        # Configuration loading
        if config_path:
            try:
                self.load_configuration(config_path)
            except Exception as e:
                self.logger.error(f"Configuration loading failed: {e}")
                raise CircadianNarrativeCyclesError(f"Configuration error: {e}")
        
        # Initialize caches and transitional states
        self.current_tone_cache = None
        self.cache_timestamp = None
        self.transitions = self._generate_transitional_states()
    def load_configuration(self, config_path: str) -> None:
        """
        Load and validate configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
        
        Raises:
            CircadianNarrativeCyclesError: If configuration is invalid
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate configuration
            if not self._validate_configuration(config):
                raise CircadianNarrativeCyclesError("Invalid configuration structure")
            
            # Update time periods
            if "time_periods" in config:
                for period, settings in config["time_periods"].items():
                    if period in self.time_periods:
                        # Selectively update only valid keys
                        valid_keys = ["start", "end", "tone", "themes", "linguistic_patterns"]
                        self.time_periods[period].update({
                            k: v for k, v in settings.items() if k in valid_keys
                        })
            
            # Update narrative modifiers
            if "narrative_modifiers" in config:
                for tone, modifiers in config["narrative_modifiers"].items():
                    if tone in self.narrative_modifiers:
                        valid_keys = ["symbol_emphasis", "color_palette", "tempo", "emotional_quality"]
                        self.narrative_modifiers[tone].update({
                            k: v for k, v in modifiers.items() if k in valid_keys
                        })
            
            self.logger.info(f"Configuration loaded successfully from {config_path}")
        
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in configuration file: {config_path}")
            raise CircadianNarrativeCyclesError("Configuration file is not valid JSON")
        except IOError:
            self.logger.error(f"Unable to read configuration file: {config_path}")
            raise CircadianNarrativeCyclesError("Cannot access configuration file")
    
    def _validate_configuration(self, config: Dict) -> bool:
        """
        Comprehensive configuration validation.
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Boolean indicating whether configuration is valid
        """
        # Validate time periods
        for period, data in config.get("time_periods", {}).items():
            required_keys = {"start", "end", "tone", "themes", "linguistic_patterns"}
            if not all(key in data for key in required_keys):
                self.logger.warning(f"Invalid configuration for period: {period}")
                return False
        
        # Validate narrative modifiers
        for tone, modifiers in config.get("narrative_modifiers", {}).items():
            required_keys = {"symbol_emphasis", "color_palette", "tempo", "emotional_quality"}
            if not all(key in modifiers for key in required_keys):
                self.logger.warning(f"Invalid configuration for tone: {tone}")
                return False
        
        return True
    
    @lru_cache(maxsize=128)
    def _time_to_minutes(self, time_str: str) -> int:
        """
        Convert a time string to minutes since midnight with caching.
        
        Args:
            time_str: Time in 'HH:MM' format
        
        Returns:
            Minutes since midnight
        """
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except ValueError:
            self.logger.error(f"Invalid time format: {time_str}")
            raise CircadianNarrativeCyclesError(f"Cannot parse time: {time_str}")
    
    def _generate_transitional_states(self) -> Dict:
        """
        Generate sophisticated blended states for period transitions.
        
        Returns:
            Dictionary of nuanced transition states
        """
        transitions = {}
        periods = list(self.time_periods.keys())
        
        for i in range(len(periods)):
            current = periods[i]
            next_idx = (i + 1) % len(periods)
            next_period = periods[next_idx]
            
            transition_key = f"{current}_to_{next_period}"
            transitions[transition_key] = {
                "duration_minutes": 30,
                "tone": f"transitional blend from {self.time_periods[current]['tone']} to {self.time_periods[next_period]['tone']}",
                "themes": list(set(
                    self.time_periods[current]["themes"] + 
                    self.time_periods[next_period]["themes"]
                )),
                "blending_pattern": "graduated_overlay"
            }
        
        return transitions
    
    def _blend_lists(self, 
                     list1: List[str], 
                     list2: List[str], 
                     progress: float, 
                     max_length: int = 5) -> List[str]:
        """
        Intelligently blend two lists based on transition progress.
        
        Args:
            list1: First list of elements
            list2: Second list of elements
            progress: Transition progress (0.0 to 1.0)
            max_length: Maximum number of elements in result
        
        Returns:
            Blended list of unique elements
        """
        if progress < 0.3:
            count1, count2 = math.ceil(len(list1) * 0.7), math.ceil(len(list2) * 0.3)
        elif progress < 0.7:
            count1, count2 = math.ceil(len(list1) * 0.5), math.ceil(len(list2) * 0.5)
        else:
            count1, count2 = math.ceil(len(list1) * 0.3), math.ceil(len(list2) * 0.7)
        
        result = list(set(
            random.sample(list1, min(count1, len(list1))) + 
            random.sample(list2, min(count2, len(list2)))
        ))
        
        return result[:max_length]
    def _current_period(self, current_time: Optional[datetime.datetime] = None) -> Tuple[str, float]:
        """
        Determine the current time period with advanced transition tracking.
        
        Args:
            current_time: Datetime object (defaults to now if None)
            
        Returns:
            Tuple of (period_name, transition_progress)
        """
        if current_time is None:
            current_time = datetime.datetime.now()
            
        current_minutes = current_time.hour * 60 + current_time.minute
        
        for period, data in self.time_periods.items():
            start_minutes = self._time_to_minutes(data["start"])
            end_minutes = self._time_to_minutes(data["end"])
            
            # Handle periods crossing midnight
            if end_minutes < start_minutes:
                end_minutes += 24 * 60
                if current_minutes < start_minutes:
                    current_minutes += 24 * 60
            
            if start_minutes <= current_minutes < end_minutes:
                # Calculate transition zone (30 minutes before end)
                transition_start = end_minutes - 30
                
                if current_minutes >= transition_start:
                    # In transition zone
                    progress = (current_minutes - transition_start) / 30.0
                    return period, progress
                else:
                    # Firmly in this period
                    return period, 0.0
                
        # Fallback
        return list(self.time_periods.keys())[0], 0.0
    
    def get_current_narrative_tone(self, current_time: Optional[datetime.datetime] = None) -> Dict:
        """
        Advanced narrative tone generation with intelligent caching.
        
        Args:
            current_time: Optional datetime object
            
        Returns:
            Comprehensive narrative tone information
        """
        # Thread-safe caching mechanism
        with self._lock:
            # Check cache validity
            if self.current_tone_cache and self.cache_timestamp:
                now = datetime.datetime.now()
                cache_age = (now - self.cache_timestamp).total_seconds() / 60
                if current_time is None and cache_age < 5:
                    return self.current_tone_cache
            
            # Determine current period and transition
            period, transition_progress = self._current_period(current_time)
            
            # Generate tone information
            if transition_progress == 0.0:
                # Stable period state
                result = self._generate_stable_tone(period)
            else:
                # Transitional state
                result = self._generate_transition_tone(period, transition_progress)
            
            # Update cache if using current time
            if current_time is None:
                self.current_tone_cache = result
                self.cache_timestamp = datetime.datetime.now()
            
            return result
    
    def _generate_stable_tone(self, period: str) -> Dict:
        """
        Generate tone information for a stable time period.
        
        Args:
            period: Current time period
        
        Returns:
            Detailed narrative tone dictionary
        """
        return {
            "period": period,
            "tone": self.time_periods[period]["tone"],
            "themes": self.time_periods[period]["themes"],
            "linguistic_patterns": self.time_periods[period]["linguistic_patterns"],
            "modifiers": self.narrative_modifiers[self.time_periods[period]["tone"]],
            "transitioning": False
        }
    
    def _generate_transition_tone(self, period: str, transition_progress: float) -> Dict:
        """
        Generate sophisticated transitional tone information.
        
        Args:
            period: Current period
            transition_progress: Progress through transition (0.0-1.0)
        
        Returns:
            Detailed transitional narrative tone dictionary
        """
        periods = list(self.time_periods.keys())
        current_idx = periods.index(period)
        next_idx = (current_idx + 1) % len(periods)
        next_period = periods[next_idx]
        
        transition_key = f"{period}_to_{next_period}"
        
        return {
            "period": transition_key,
            "tone": self.transitions[transition_key]["tone"],
            "themes": self._blend_lists(
                self.time_periods[period]["themes"],
                self.time_periods[next_period]["themes"],
                transition_progress
            ),
            "linguistic_patterns": self._blend_linguistic_patterns(
                self.time_periods[period]["linguistic_patterns"],
                self.time_periods[next_period]["linguistic_patterns"],
                transition_progress
            ),
            "modifiers": self._blend_modifiers(
                self.narrative_modifiers[self.time_periods[period]["tone"]],
                self.narrative_modifiers[self.time_periods[next_period]["tone"]],
                transition_progress
            ),
            "transitioning": True,
            "transition_progress": transition_progress
        }
    
    def _blend_linguistic_patterns(self, 
                                   patterns1: List[str], 
                                   patterns2: List[str], 
                                   progress: float) -> List[str]:
        """
        Intelligently blend linguistic patterns during transitions.
        
        Args:
            patterns1: First set of linguistic patterns
            patterns2: Second set of linguistic patterns
            progress: Transition progress (0.0-1.0)
        
        Returns:
            Blended linguistic patterns
        """
        if progress < 0.3:
            return patterns1 + [random.choice(patterns2)]
        elif progress < 0.7:
            return random.sample(patterns1, len(patterns1)//2) + random.sample(patterns2, len(patterns2)//2)
        else:
            return [random.choice(patterns1)] + patterns2
    
    def transform_narrative(self, 
                            narrative_text: str, 
                            current_time: Optional[datetime.datetime] = None,
                            transformation_strength: float = 1.0) -> str:
        """
        Advanced narrative transformation based on circadian tone.
        
        Args:
            narrative_text: Original narrative text
            current_time: Optional datetime for tone determination
            transformation_strength: Intensity of transformation (0.0-1.0)
        
        Returns:
            Transformed narrative text
        """
        # Get current tone information
        tone_info = self.get_current_narrative_tone(current_time)
        
        # Transformation prefix based on strength
        prefix_map = {
            (0.0, 0.3): f"[Subtle {tone_info['tone']} tone] ",
            (0.3, 0.7): f"[Moderate {tone_info['tone']} tone] ",
            (0.7, 1.1): f"[Strong {tone_info['tone']} tone] "
        }
        
        # Determine appropriate prefix
        prefix = next(
            prefix for (low, high), prefix in prefix_map.items() 
            if low <= transformation_strength < high
        )
        
        return prefix + narrative_text
    def generate_tone_specific_prompt(self, 
                                      base_prompt: str, 
                                      current_time: Optional[datetime.datetime] = None) -> str:
        """
        Generate a highly contextual, tone-specific prompt enhancement.
        
        Args:
            base_prompt: Original prompt text
            current_time: Optional datetime for tone determination
        
        Returns:
            Enriched, tone-specific prompt
        """
        # Get current narrative tone
        tone_info = self.get_current_narrative_tone(current_time)
        
        # Construct contextual prompt additions
        prompt_enhancements = [
            f"Express this narrative in a {tone_info['tone']} tone, "
            f"emphasizing themes of {', '.join(tone_info['themes'][:2])}.",
            
            f"Utilize {tone_info['modifiers']['tempo']} pacing and "
            f"convey a sense of {tone_info['modifiers']['emotional_quality']}.",
            
            f"Incorporate symbolic elements like {', '.join(tone_info['modifiers'].get('symbol_emphasis', ['unknown'])[:2])}.",
            
            f"Use a color palette reminiscent of {', '.join(tone_info['modifiers'].get('color_palette', ['neutral'])[:2])}."
        ]
        
        # Add transitional context if applicable
        if tone_info.get('transitioning', False):
            transition_note = (
                f"Note: This is a transitional narrative moment "
                f"({tone_info.get('transition_progress', 0):.0%} complete). "
                "Balance emerging and receding narrative qualities."
            )
            prompt_enhancements.append(transition_note)
        
        # Combine base prompt with enhancements
        enhanced_prompt = base_prompt + "\n\n" + "\n".join(prompt_enhancements)
        return enhanced_prompt
    
    def get_tone_schedule(self) -> Dict:
        """
        Generate a comprehensive tone schedule for a 24-hour cycle.
        
        Returns:
            Detailed dictionary of narrative tones and characteristics
        """
        return {
            period: {
                "time": f"{data['start']} - {data['end']}",
                "tone": data["tone"],
                "themes": data["themes"],
                "modifiers": self.narrative_modifiers[data["tone"]]
            }
            for period, data in self.time_periods.items()
        }
    
    def export_configuration(self, output_path: str) -> None:
        """
        Export the current configuration to a JSON file.
        
        Args:
            output_path: Path to save the configuration
        """
        try:
            config = {
                "time_periods": self.time_periods,
                "narrative_modifiers": self.narrative_modifiers
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Configuration exported to {output_path}")
        except IOError as e:
            self.logger.error(f"Failed to export configuration: {e}")
            raise CircadianNarrativeCyclesError(f"Export failed: {e}")

# Comprehensive Module Testing
def test_circadian_narrative_cycles():
    """
    Comprehensive testing function for CircadianNarrativeCycles.
    """
    try:
        # Initialize the system
        cnc = CircadianNarrativeCycles()
        
        # Test time-specific tone generation
        test_times = [
            datetime.datetime(2025, 1, 1, 6, 30),   # Dawn
            datetime.datetime(2025, 1, 1, 10, 15),  # Morning
            datetime.datetime(2025, 1, 1, 14, 45),  # Afternoon
            datetime.datetime(2025, 1, 1, 19, 20),  # Evening
            datetime.datetime(2025, 1, 1, 23, 50)   # Night
        ]
        
        for test_time in test_times:
            tone = cnc.get_current_narrative_tone(test_time)
            print(f"\nTone at {test_time.strftime('%H:%M')}:")
            print(f"Period: {tone['period']}")
            print(f"Tone: {tone['tone']}")
            print(f"Themes: {tone['themes']}")
        
        # Test prompt generation
        base_prompt = "Describe a journey of personal transformation."
        enhanced_prompt = cnc.generate_tone_specific_prompt(base_prompt)
        print("\nEnhanced Prompt:")
        print(enhanced_prompt)
        
        # Test narrative transformation
        original_text = "A traveler stands at the crossroads of destiny."
        transformed_text = cnc.transform_narrative(original_text)
        print("\nTransformed Narrative:")
        print(transformed_text)
        
        # Test tone schedule
        tone_schedule = cnc.get_tone_schedule()
        print("\nTone Schedule:")
        for period, details in tone_schedule.items():
            print(f"{period}: {details}")
        
        print("\n✅ All tests completed successfully!")
    
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        raise

# Module Execution
if __name__ == "__main__":
    test_circadian_narrative_cycles()
