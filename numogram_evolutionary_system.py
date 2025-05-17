import numpy as np
import spacy
import random
import uuid
import datetime
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class SymbolicPattern:
    """Represents an extracted symbolic pattern with numogram zone associations"""
    id: str
    core_symbols: List[str]
    related_symbols: List[str]
    numogram_zone: str
    intensity: float
    context: str
    timestamp: str
    digital_signature: List[int] = field(default_factory=list)

@dataclass
class EmotionalState:
    """Represents an emotional state within the numogram system"""
    id: str
    primary_emotion: str
    emotional_spectrum: Dict[str, float]
    numogram_zone: str
    intensity: float
    timestamp: str
    digital_ratios: List[float] = field(default_factory=list)


class NumogramSystem:
    """
    Core numogram system for zone transitions and memory
    """
    
    def __init__(self, zones_file=None, memory_file=None):
        # Load zone data
        self.ZONE_DATA = self._load_zone_data(zones_file)
        
        # Initialize or load user memory
        self.user_memory = self._load_user_memory(memory_file)
        
        # Store memory file path for saving
        self.memory_file = memory_file
        
        # Transition history
        self.transition_history = []
        
        # Zone transition matrix (probabilities)
        self.TRANSITION_MATRIX = {
            "1": {"2": 0.3, "4": 0.3, "8": 0.3, "5": 0.1},
            "2": {"1": 0.3, "3": 0.3, "6": 0.3, "5": 0.1},
            "3": {"2": 0.3, "7": 0.3, "9": 0.3, "5": 0.1},
            "4": {"1": 0.3, "5": 0.3, "7": 0.3, "8": 0.1},
            "5": {"2": 0.2, "4": 0.2, "6": 0.2, "8": 0.2, "9": 0.2},
            "6": {"2": 0.3, "5": 0.3, "9": 0.3, "3": 0.1},
            "7": {"3": 0.3, "4": 0.3, "8": 0.3, "5": 0.1},
            "8": {"1": 0.3, "5": 0.3, "7": 0.3, "9": 0.1},
            "9": {"3": 0.3, "6": 0.3, "5": 0.3, "8": 0.1}
        }
        
    def _load_zone_data(self, zones_file=None) -> Dict:
        """Load zone data from file or use defaults"""
        default_zones = {
            "1": {
                "name": "Unity",
                "description": "The zone of unified beginnings and singularity",
                "primary_principles": ["origin", "monad", "creation", "source"],
                "secondary_principles": ["potential", "seed", "starting point"]
            },
            "2": {
                "name": "Division",
                "description": "The zone of duality and binary opposition",
                "primary_principles": ["duality", "polarity", "reflection", "mirroring"],
                "secondary_principles": ["contrast", "dialogue", "balance"]
            },
            "3": {
                "name": "Synthesis",
                "description": "The zone of creative combination and birth",
                "primary_principles": ["trinity", "birth", "creativity", "combination"],
                "secondary_principles": ["growth", "expansion", "manifestation"]
            },
            "4": {
                "name": "Structure",
                "description": "The zone of system, order and foundation",
                "primary_principles": ["order", "stability", "foundation", "system"],
                "secondary_principles": ["logic", "reason", "building", "basis"]
            },
            "5": {
                "name": "Transformation",
                "description": "The zone of change, adaptation and transition",
                "primary_principles": ["change", "motion", "adaptation", "pivot"],
                "secondary_principles": ["learning", "experience", "flux", "center"]
            },
            "6": {
                "name": "Harmony",
                "description": "The zone of beauty, balance and resonance",
                "primary_principles": ["beauty", "harmony", "resonance", "proportion"],
                "secondary_principles": ["aesthetics", "pleasure", "art", "attraction"]
            },
            "7": {
                "name": "Mystery",
                "description": "The zone of depth, wisdom and contemplation",
                "primary_principles": ["depth", "mystery", "contemplation", "wisdom"],
                "secondary_principles": ["intuition", "hidden", "esoteric", "insight"]
            },
            "8": {
                "name": "Power",
                "description": "The zone of material power, cycles and recursion",
                "primary_principles": ["power", "cycles", "recursion", "infinity"],
                "secondary_principles": ["matter", "manifestation", "abundance"]
            },
            "9": {
                "name": "Completion",
                "description": "The zone of fulfillment, integration and totality",
                "primary_principles": ["completion", "fulfillment", "totality", "integration"],
                "secondary_principles": ["wholeness", "culmination", "achievement"]
            }
        }
        
        if zones_file and os.path.exists(zones_file):
            try:
                with open(zones_file, 'r') as f:
                    loaded_zones = json.load(f)
                    # Update defaults with loaded zones
                    for zone, data in loaded_zones.items():
                        if zone in default_zones:
                            default_zones[zone].update(data)
                        else:
                            default_zones[zone] = data
            except (json.JSONDecodeError, IOError):
                pass  # Silently use defaults on error
        
        return default_zones
    
    def _load_user_memory(self, memory_file=None) -> Dict:
        """Load user memory from file or initialize empty"""
        memory = {}
        
        if memory_file and os.path.exists(memory_file):
            try:
                with open(memory_file, 'r') as f:
                    memory = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass  # Silently use empty memory on error
        
        return memory
    
    def save_user_memory(self) -> bool:
        """Save user memory to file"""
        if not self.memory_file:
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            with open(self.memory_file, 'w') as f:
                json.dump(self.user_memory, f, indent=2)
            return True
        except IOError:
            return False
    
    def transition(self, user_id: str, current_zone: str = None, feedback: float = 0.5, context_data: Dict = None) -> Dict:
        """
        Execute a numogram zone transition for the user
        
        Parameters:
        - user_id: Unique identifier for the user
        - current_zone: Current zone (if None, will use from memory or default to "1")
        - feedback: How strongly to apply transition probabilities (0-1)
        - context_data: Additional contextual information for the transition
        
        Returns:
        - Dictionary with transition details
        """
        # Initialize context if not provided
        if context_data is None:
            context_data = {}
        
        # Initialize or get user data
        if user_id not in self.user_memory:
            self.user_memory[user_id] = {
                "zone": "1",
                "history": [],
                "last_transition": None
            }
        
        # Get current zone from parameter, memory, or default
        if current_zone is None:
            current_zone = self.user_memory[user_id].get("zone", "1")
        
        # Ensure current zone is valid
        if current_zone not in self.ZONE_DATA:
            current_zone = "1"  # Default to zone 1 if invalid
        
        # Calculate next zone
        next_zone = self._calculate_next_zone(current_zone, feedback, context_data)
        
        # Store the transition
        transition = {
            "user_id": user_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "current_zone": current_zone,
            "next_zone": next_zone,
            "feedback": feedback
        }
        
        # Update user memory
        self.user_memory[user_id]["zone"] = next_zone
        self.user_memory[user_id]["last_transition"] = transition
        
        # Add to history (limited to last 100)
        self.user_memory[user_id]["history"].append(transition)
        if len(self.user_memory[user_id]["history"]) > 100:
            self.user_memory[user_id]["history"] = self.user_memory[user_id]["history"][-100:]
        
        # Add to global transition history
        self.transition_history.append(transition)
        if len(self.transition_history) > 1000:
            self.transition_history = self.transition_history[-1000:]
        
        # Get zone descriptions
        transition["current_zone_data"] = self.ZONE_DATA.get(current_zone, {})
        transition["next_zone_data"] = self.ZONE_DATA.get(next_zone, {})
        
        # Save user memory
        self.save_user_memory()
        
        return transition
    
    def _calculate_next_zone(self, current_zone: str, feedback: float, context_data: Dict) -> str:
        """Calculate the next zone based on transition matrix and context"""
        # Get transition probabilities from matrix
        probabilities = self.TRANSITION_MATRIX.get(current_zone, {"1": 1.0})
        
        # Modify probabilities based on feedback strength
        # Higher feedback means stronger adherence to transition matrix
        modified_probs = {}
        
        # If feedback is low, increase randomness
        if feedback < 0.5:
            # Create uniform distribution across all zones
            uniform_probs = {zone: 1.0/len(self.ZONE_DATA) for zone in self.ZONE_DATA}
            
            # Blend matrix probabilities with uniform distribution
            blend_factor = feedback * 2  # 0-0.5 -> 0-1
            for zone in self.ZONE_DATA:
                matrix_prob = probabilities.get(zone, 0.0)
                uniform_prob = uniform_probs[zone]
                modified_probs[zone] = (matrix_prob * blend_factor) + (uniform_prob * (1 - blend_factor))
        
        # If feedback is high, strengthen main connections
        else:
            # Boost highest probability transitions
            boost_factor = (feedback - 0.5) * 2  # 0.5-1 -> 0-1
            
            # Sort transitions by probability
            sorted_transitions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            # Boost top transition
            if sorted_transitions:
                top_zone = sorted_transitions[0][0]
                boost = min(0.3, boost_factor * 0.3)  # Max 30% boost
                
                for zone, prob in probabilities.items():
                    if zone == top_zone:
                        modified_probs[zone] = prob + boost
                    else:
                        # Distribute the boost reduction proportionally
                        reduction = boost / (len(probabilities) - 1)
                        modified_probs[zone] = max(0.01, prob - reduction)
            else:
                modified_probs = probabilities.copy()
        
        # Apply context-specific adjustments based on emotional and symbolic data
        if context_data:
            # Consider emotional state if present
            if "emotional_state" in context_data:
                emotion = context_data["emotional_state"].get("primary_emotion", "")
                
                # Example: Joy tends to increase probability of zones 6 and 9
                if emotion == "joy":
                    modified_probs["6"] = modified_probs.get("6", 0) + 0.1
                    modified_probs["9"] = modified_probs.get("9", 0) + 0.1
                # Fear tends to increase probability of zones 8 and 2
                elif emotion == "fear":
                    modified_probs["8"] = modified_probs.get("8", 0) + 0.1
                    modified_probs["2"] = modified_probs.get("2", 0) + 0.1
            
            # Consider digital ratios if present
            if "digital_ratios" in context_data:
                ratios = context_data["digital_ratios"]
                if ratios and len(ratios) > 0:
                    # Use first ratio to bias toward corresponding zone
                    ratio = int(ratios[0])
                    if 1 <= ratio <= 9:
                        zone_str = str(ratio)
                        modified_probs[zone_str] = modified_probs.get(zone_str, 0) + 0.1
        
        # Normalize probabilities
        total = sum(modified_probs.values())
        normalized_probs = {zone: prob/total for zone, prob in modified_probs.items()}
        
        # Select next zone based on probabilities
        zones = list(normalized_probs.keys())
        probabilities = list(normalized_probs.values())
        next_zone = random.choices(zones, weights=probabilities, k=1)[0]
        
        return next_zone
    
    def get_user_data(self, user_id: str) -> Dict:
        """Get user data from memory"""
        if user_id not in self.user_memory:
            return {"error": "User not found"}
        
        return self.user_memory[user_id]
    
    def get_zone_data(self, zone: str) -> Dict:
        """Get data for a specific zone"""
        if zone not in self.ZONE_DATA:
            return {"error": "Zone not found"}
        
        return self.ZONE_DATA[zone]


class NumogramaticSymbolExtractor:
    """Extract symbolic patterns and map them to numogram zones"""
    
    def __init__(self, numogram_system):
        # Initialize NLP pipeline
        self.nlp = spacy.load("en_core_web_lg")
        
        # Store reference to numogram system
        self.numogram = numogram_system
        
        # Symbol lexicon organized by numogram zones
        self.zone_symbols = {
            "1": ["origin", "foundation", "unity", "beginning", "source", "center", "one", "singularity"],
            "2": ["division", "duality", "mirror", "reflection", "binary", "polarity", "opposition", "dialogue"],
            "3": ["synthesis", "triad", "creation", "triangle", "trinity", "birth", "generation", "offspring"],
            "4": ["stability", "square", "structure", "foundation", "order", "system", "framework", "logic"],
            "5": ["change", "transformation", "pentagram", "life", "sensory", "experience", "adaptation", "flow"],
            "6": ["harmony", "beauty", "hexagon", "balance", "proportion", "symmetry", "aesthetics", "pleasure"],
            "7": ["mystery", "wisdom", "heptagon", "depth", "contemplation", "analysis", "understanding", "insight"],
            "8": ["power", "infinity", "octagon", "cycles", "regeneration", "abundance", "prosperity", "material"],
            "9": ["completion", "fulfillment", "nonagon", "wholeness", "integration", "culmination", "perfection"]
        }
        
        # Additional symbol categories
        self.archetypal_symbols = {
            "shadow": ["darkness", "unknown", "unconscious", "hidden", "secret", "fear", "repression"],
            "anima/animus": ["soul", "feminine", "masculine", "complement", "opposite", "inner", "psyche"],
            "self": ["wholeness", "integration", "center", "mandala", "unity", "individuation", "actualization"],
            "persona": ["mask", "role", "social", "identity", "appearance", "presentation", "facade"],
            "trickster": ["joker", "fool", "disruption", "chaos", "unpredictable", "transformation", "catalyst"]
        }
        
        # Symbolic association matrix - weights for symbol-to-zone mappings
        self.symbol_zone_affinities = self._initialize_symbol_zone_affinities()
        
        # Cache of extracted patterns
        self.pattern_cache = []
    
    def _initialize_symbol_zone_affinities(self) -> Dict[str, Dict[str, float]]:
        """Initialize affinity scores between symbols and zones"""
        affinities = {}
        
        # For each symbol in zone_symbols, create strong affinity to its zone
        # and weaker affinities to other zones
        for zone, symbols in self.zone_symbols.items():
            for symbol in symbols:
                if symbol not in affinities:
                    affinities[symbol] = {}
                
                # Strong affinity to native zone
                affinities[symbol][zone] = 0.8 + (random.random() * 0.2)
                
                # Weaker affinities to other zones
                for other_zone in self.zone_symbols:
                    if other_zone != zone:
                        # Create some connections stronger than others
                        # based on numogram's inherent connections
                        if (int(zone) + int(other_zone)) % 9 == 0:  # Complementary zones
                            affinity = 0.3 + (random.random() * 0.3)
                        else:
                            affinity = random.random() * 0.3
                        affinities[symbol][other_zone] = affinity
        
        # For archetypal symbols, distribute affinities across zones
        for archetype, symbols in self.archetypal_symbols.items():
            for symbol in symbols:
                if symbol not in affinities:
                    affinities[symbol] = {}
                
                # Assign affinities based on archetypal-zone correspondences
                # These can be adjusted based on deeper numogram analysis
                if archetype == "shadow":
                    # Shadow has strong connections to zones 8, 9, 1
                    affinities[symbol] = {"1": 0.6, "8": 0.8, "9": 0.7, 
                                         "2": 0.2, "3": 0.1, "4": 0.3, 
                                         "5": 0.3, "6": 0.2, "7": 0.5}
                elif archetype == "anima/animus":
                    # Anima/animus has strong connections to zones 2, 3, 5
                    affinities[symbol] = {"1": 0.3, "2": 0.8, "3": 0.7, 
                                         "4": 0.2, "5": 0.7, "6": 0.4, 
                                         "7": 0.3, "8": 0.2, "9": 0.4}
                elif archetype == "self":
                    # Self has strong connections to zones 1, 5, 9
                    affinities[symbol] = {"1": 0.9, "2": 0.3, "3": 0.4, 
                                         "4": 0.3, "5": 0.8, "6": 0.4, 
                                         "7": 0.5, "8": 0.4, "9": 0.8}
                elif archetype == "persona":
                    # Persona has strong connections to zones 2, 4, 6
                    affinities[symbol] = {"1": 0.3, "2": 0.8, "3": 0.4, 
                                         "4": 0.7, "5": 0.3, "6": 0.7, 
                                         "7": 0.2, "8": 0.4, "9": 0.3}
                elif archetype == "trickster":
                    # Trickster has strong connections to zones 3, 5, 7
                    affinities[symbol] = {"1": 0.2, "2": 0.3, "3": 0.7, 
                                         "4": 0.2, "5": 0.8, "6": 0.3, 
                                         "7": 0.7, "8": 0.4, "9": 0.5}
        
        return affinities
    
    def _calculate_digital_signature(self, text: str) -> List[int]:
        """Calculate a digital signature for text using numogram principles"""
        # Sum character values
        text_sum = sum(ord(c) for c in text) % 9
        if text_sum == 0:
            text_sum = 9  # In numogram, 0 is treated as 9
            
        # Generate decimal expansion as per numogram method
        # (These are the decimal digits of 1/n where n is the text_sum)
        digit_expansions = []
        remainder = 1
        for _ in range(3):  # Get first 3 digits
            remainder = (remainder * 10) % text_sum
            digit = remainder * 10 // text_sum
            digit_expansions.append(digit)
            
        return digit_expansions
    
    def extract_symbols(self, text: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Extract symbolic patterns from text with numogram associations"""
        doc = self.nlp(text)
        
        # Get all potential symbolic words based on POS and dependencies
        symbol_candidates = []
        for token in doc:
            # Look for nouns, verbs, adjectives that might have symbolic significance
            if token.pos_ in ("NOUN", "VERB", "ADJ") and not token.is_stop:
                # Calculate symbol strength based on various factors
                strength = 0.0
                
                # Check if word is in our symbol lexicons
                in_zone_symbols = any(token.lemma_.lower() in symbols for symbols in self.zone_symbols.values())
                in_archetypal_symbols = any(token.lemma_.lower() in symbols for symbols in self.archetypal_symbols.values())
                
                if in_zone_symbols:
                    strength += 0.6
                if in_archetypal_symbols:
                    strength += 0.5
                    
                # Words with strong emotional valence often have symbolic potential
                # Note: We'll add a simple check for emotional words since we don't have spaCy sentiment extension
                emotional_words = ["love", "hate", "fear", "joy", "anger", "sadness", "happy", "sad", "afraid", "angry"]
                if token.lemma_.lower() in emotional_words:
                    strength += 0.3
                
                # Abstract concepts are more likely to be symbolic
                if token.pos_ == "NOUN" and token.lemma_.lower() not in {t.lemma_.lower() for t in doc if t.pos_ == "PROPN"}:
                    strength += 0.2
                
                # Add to candidates if sufficiently symbolic
                if strength > 0.3:
                    symbol_candidates.append({
                        "text": token.text,
                        "lemma": token.lemma_.lower(),
                        "pos": token.pos_,
                        "strength": strength,
                        "affinities": self.symbol_zone_affinities.get(token.lemma_.lower(), {})
                    })
        
        # Find relationships between symbols
        symbolic_relations = []
        for i, sym1 in enumerate(symbol_candidates):
            for j, sym2 in enumerate(symbol_candidates):
                if i != j:
                    # Calculate relationship strength based on syntactic and semantic similarity
                    token1 = doc[doc.text.find(sym1["text"])]
                    token2 = doc[doc.text.find(sym2["text"])]
                    
                    # Syntactic relationship
                    syntactic_relation = 0.0
                    if token2 in [child for child in token1.children]:
                        syntactic_relation = 0.7
                    elif token1.head == token2 or token2.head == token1:
                        syntactic_relation = 0.5
                    
                    # Semantic similarity
                    semantic_similarity = token1.similarity(token2)
                    
                    # Combined relationship strength
                    relationship_strength = (syntactic_relation * 0.5) + (semantic_similarity * 0.5)
                    
                    if relationship_strength > 0.3:
                        symbolic_relations.append({
                            "symbol1": sym1["lemma"],
                            "symbol2": sym2["lemma"],
                            "strength": relationship_strength
                        })
        
        # Group symbols into patterns based on relationships
        patterns = []
        visited = set()
        
        for candidate in symbol_candidates:
            if candidate["lemma"] in visited:
                continue
                
            # Find all related symbols
            related = []
            for relation in symbolic_relations:
                if relation["symbol1"] == candidate["lemma"]:
                    related.append({"symbol": relation["symbol2"], "strength": relation["strength"]})
                elif relation["symbol2"] == candidate["lemma"]:
                    related.append({"symbol": relation["symbol1"], "strength": relation["strength"]})
            
            # Sort by relationship strength
            related.sort(key=lambda x: x["strength"], reverse=True)
            related_symbols = [r["symbol"] for r in related[:5]]  # Top 5 related symbols
            
            # Determine most likely numogram zone
            zone_scores = {zone: 0.0 for zone in self.zone_symbols.keys()}
            
            # Factor 1: Direct affinities of core symbol
            for zone, affinity in candidate["affinities"].items():
                zone_scores[zone] += affinity * 0.5
                
            # Factor 2: Affinities of related symbols
            for rel_symbol in related_symbols:
                rel_affinities = self.symbol_zone_affinities.get(rel_symbol, {})
                for zone, affinity in rel_affinities.items():
                    zone_scores[zone] += affinity * 0.2
            
            # Factor 3: User's current zone if available
            if user_id and hasattr(self.numogram, 'user_memory'):
                user_zone = self.numogram.user_memory.get(user_id, {}).get('zone')
                if user_zone in zone_scores:
                    zone_scores[user_zone] += 0.3  # Bias toward user's current zone
            
            # Determine most aligned zone
            most_likely_zone = max(zone_scores.items(), key=lambda x: x[1])[0]
            
            # Calculate pattern intensity
            intensity = candidate["strength"] * 0.6 + sum(r["strength"] for r in related[:5]) * 0.4
            
            # Calculate digital signature
            digital_signature = self._calculate_digital_signature(candidate["lemma"])
            
            # Create pattern
            pattern = SymbolicPattern(
                id=f"pattern_{len(patterns)}",
                core_symbols=[candidate["lemma"]],
                related_symbols=related_symbols,
                numogram_zone=most_likely_zone,
                intensity=intensity,
                context=text,
                timestamp=datetime.datetime.utcnow().isoformat(),
                digital_signature=digital_signature
            )
            
            patterns.append(pattern)
            visited.add(candidate["lemma"])
            for rel in related_symbols:
                visited.add(rel)
        
        # Cache extracted patterns
        self.pattern_cache.extend(patterns)
        
        # Return as dict for easy serialization
        return [vars(pattern) for pattern in patterns]
    
    def get_zone_distribution(self, patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of patterns across numogram zones"""
        zone_counts = {zone: 0 for zone in self.zone_symbols.keys()}
        for pattern in patterns:
            zone = pattern.get("numogram_zone")
            if zone in zone_counts:
                zone_counts[zone] += 1
        
        total = sum(zone_counts.values())
        if total == 0:
            return {zone: 0.0 for zone in zone_counts}
        
        return {zone: count/total for zone, count in zone_counts.items()}
    
    def suggest_transition(self, patterns: List[Dict[str, Any]], user_id: str) -> Tuple[str, float]:
        """Suggest a numogram transition based on extracted patterns"""
        # Get zone distribution
        zone_distribution = self.get_zone_distribution(patterns)
        
        # Get current user zone
        current_zone = self.numogram.user_memory.get(user_id, {}).get('zone', '1')
        
        # Calculate transition based on patterns
        # Higher intensity patterns have more influence
        weighted_zones = {}
        for pattern in patterns:
            zone = pattern.get("numogram_zone")
            intensity = pattern.get("intensity", 0.5)
            if zone not in weighted_zones:
                weighted_zones[zone] = 0
            weighted_zones[zone] += intensity
        
        # Suggest most weighted zone as next transition
        if weighted_zones:
            suggested_zone = max(weighted_zones.items(), key=lambda x: x[1])[0]
            feedback = sum(weighted_zones.values()) / len(weighted_zones)  # Average intensity as feedback
            feedback = max(0.1, min(0.9, feedback))  # Normalize to 0.1-0.9 range
        else:
            # Default suggestion if no patterns
            suggested_zone = current_zone
            feedback = 0.5
        
        return suggested_zone, feedback
    
    def integrate_with_numogram(self, patterns: List[Dict[str, Any]], user_id: str, context_data: Dict = None) -> Dict:
        """Integrate extracted patterns with numogram system"""
        if not patterns:
            return {"status": "no_patterns_found"}
        
        # Extract pattern information
        suggested_zone, feedback = self.suggest_transition(patterns, user_id)
        
        # Prepare context data for numogram
        if context_data is None:
            context_data = {}
            
        # Add pattern information to context
        context_data["symbolic_patterns"] = patterns
        context_data["zone_distribution"] = self.get_zone_distribution(patterns)
        
        # Get core symbols across all patterns
        core_symbols = [symbol for pattern in patterns for symbol in pattern.get("core_symbols", [])]
        related_symbols = [symbol for pattern in patterns for symbol in pattern.get("related_symbols", [])]
        
        context_data["core_symbols"] = core_symbols
        context_data["related_symbols"] = related_symbols
        
        # Get current zone
        current_zone = self.numogram.user_memory.get(user_id, {}).get('zone', '1')
        
        # Execute numogram transition
        result = self.numogram.transition(
            user_id=user_id,
            current_zone=current_zone,
            feedback=feedback,
            context_data=context_data
        )
        
        return {
            "numogram_result": result,
            "symbolic_patterns": patterns,
            "pattern_summary": {
                "suggested_zone": suggested_zone,
                "pattern_feedback": feedback,
                "core_symbols": core_symbols[:5],  # Top 5 core symbols
                "zone_distribution": self.get_zone_distribution(patterns)
            }
        }


class EmotionalEvolutionSystem:
    """Tracks emotional states using evolutionary principles rather than backpropagation"""
    
    def __init__(self, numogram_system, population_size=50, mutation_rate=0.1):
        # Store reference to numogram system
        self.numogram = numogram_system
        
        # Core emotion lexicon
        self.base_emotions = {
            "joy": {"valence": 0.8, "arousal": 0.7, "dominance": 0.6},
            "trust": {"valence": 0.6, "arousal": 0.3, "dominance": 0.5},
            "fear": {"valence": -0.8, "arousal": 0.7, "dominance": -0.6},
            "surprise": {"valence": 0.3, "arousal": 0.8, "dominance": 0.1},
            "sadness": {"valence": -0.7, "arousal": -0.3, "dominance": -0.4},
            "disgust": {"valence": -0.6, "arousal": 0.2, "dominance": 0.1},
            "anger": {"valence": -0.6, "arousal": 0.8, "dominance": 0.7},
            "anticipation": {"valence": 0.4, "arousal": 0.4, "dominance": 0.3},
            "curiosity": {"valence": 0.5, "arousal": 0.5, "dominance": 0.3},
            "awe": {"valence": 0.7, "arousal": 0.6, "dominance": -0.1},
            "serenity": {"valence": 0.7, "arousal": -0.4, "dominance": 0.3},
            "confusion": {"valence": -0.3, "arousal": 0.5, "dominance": -0.4}
        }
        
        # Numogram zone to emotion affinities
        self.zone_emotion_affinities = {
            "1": {"trust": 0.7, "serenity": 0.6, "anticipation": 0.5, "joy": 0.4},
            "2": {"surprise": 0.7, "curiosity": 0.6, "confusion": 0.5, "fear": 0.4},
            "3": {"awe": 0.7, "curiosity": 0.6, "joy": 0.5, "surprise": 0.4},
            "4": {"trust": 0.7, "anticipation": 0.6, "serenity": 0.5, "confusion": 0.4},
            "5": {"curiosity": 0.7, "surprise": 0.6, "anticipation": 0.5, "awe": 0.4},
            "6": {"joy": 0.7, "awe": 0.6, "serenity": 0.5, "trust": 0.4},
            "7": {"serenity": 0.7, "awe": 0.6, "trust": 0.5, "sadness": 0.4},
            "8": {"fear": 0.7, "surprise": 0.6, "anger": 0.5, "disgust": 0.4},
            "9": {"awe": 0.7, "joy": 0.6, "anticipation": 0.5, "curiosity": 0.4}
        }
        
        # Evolutionary algorithm parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Current population of emotional state detectors
        self.detector_population = self._initialize_detector_population()
        
        # History of emotional states
        self.emotional_history = []
    
    def _initialize_detector_population(self) -> List[Dict]:
        """Initialize a population of emotion detectors using evolutionary principles"""
        population = []
        for _ in range(self.population_size):
            # Create a detector with random weights for emotion recognition
            detector = {
                "id": str(uuid.uuid4()),
                "weights": {
                    "text_features": np.random.normal(0, 1, 20),  # Weights for text features
                    "context_features": np.random.normal(0, 1, 10),  # Weights for context features
                    "numogram_features": np.random.normal(0, 1, 9)  # Weights for numogram zone features
                },
                "emotion_biases": {emotion: np.random.normal(0, 0.2) for emotion in self.base_emotions},
                "fitness": 0.0,  # Will be updated during evaluation
                "generation": 0
            }
            population.append(detector)
        return population
    
    def _extract_features(self, text: str, context_data: Dict) -> Dict:
        """Extract features for emotion detection"""
        # For a real implementation, this would use sophisticated NLP
        # Here we'll use a simplified approach
        
        # 1. Simple text features
        text_features = np.zeros(20)
        # Simple feature: word count
        text_features[0] = len(text.split())
        # Simple feature: sentence count
        text_features[1] = text.count('.') + text.count('!') + text.count('?')
        # Simple feature: exclamation count
        text_features[2] = text.count('!')
        # Simple feature: question count
        text_features[3] = text.count('?')
        
        # Emotion word presence (simplified)
        emotion_words = {
            "joy": ["happy", "joy", "delight", "pleased", "glad"],
            "trust": ["trust", "believe", "faith", "reliable", "dependable"],
            "fear": ["fear", "afraid", "scared", "terrified", "anxious"],
            "surprise": ["surprise", "shocked", "astonished", "amazed", "unexpected"],
            "sadness": ["sad", "unhappy", "depressed", "grief", "sorrow"],
            "disgust": ["disgust", "revolted", "repulsed", "gross", "dislike"],
            "anger": ["anger", "angry", "furious", "outraged", "irritated"],
            "anticipation": ["anticipate", "expect", "await", "foresee", "looking forward"],
            "curiosity": ["curious", "wonder", "inquisitive", "interest", "questioning"],
            "awe": ["awe", "wonder", "amazement", "reverence", "profound"],
            "serenity": ["serene", "calm", "peaceful", "tranquil", "relaxed"],
            "confusion": ["confused", "puzzled", "perplexed", "bewildered", "unsure"]
        }
        
        for i, (emotion, words) in enumerate(emotion_words.items()):
            for word in words:
                if word in text.lower():
                    idx = 4 + i  # Start emotion words at index 4
                    if idx < 20:  # Stay within feature array bounds
                        text_features[idx] += 1
        
        # 2. Context features
        context_features = np.zeros(10)
        # Recent emotions if available
        if "recent_emotions" in context_data:
            recent = context_data["recent_emotions"]
            for i, emotion in enumerate(self.base_emotions.keys()):
                if i < 10 and emotion in recent:
                    context_features[i] = recent[emotion]
        
        # 3. Numogram zone features
        numogram_features = np.zeros(9)
        # Current zone
        current_zone = context_data.get("current_zone", "1")
        zone_idx = int(current_zone) - 1  # 0-indexed
        numogram_features[zone_idx] = 1.0
        
        return {
            "text_features": text_features,
            "context_features": context_features,
            "numogram_features": numogram_features
        }
    
    def _evaluate_detector(self, detector: Dict, features: Dict, feedback: float = None) -> Tuple[Dict, Dict]:
        """Evaluate a detector on the features"""
        # Calculate raw scores for each emotion
        emotion_scores = {}
        
        for emotion in self.base_emotions:
            # Start with the bias for this emotion
            score = detector["emotion_biases"][emotion]
            
            # Add weighted contribution from features
            score += np.dot(features["text_features"], detector["weights"]["text_features"])
            score += np.dot(features["context_features"], detector["weights"]["context_features"])
            score += np.dot(features["numogram_features"], detector["weights"]["numogram_features"])
            
            # Apply logistic function to get probability
            emotion_scores[emotion] = 1.0 / (1.0 + np.exp(-score))
        
        # Normalize scores to sum to 1
        total = sum(emotion_scores.values())
        emotion_scores = {e: s/total for e, s in emotion_scores.items()}
        
        # Identify primary emotion
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate fitness if feedback provided
        if feedback is not None:
            # Higher fitness if the detector's top emotion matches user feedback
            # and has high confidence
            detector["fitness"] = 0.5  # Base fitness
            
            # Bonus for matching primary emotion with user feedback
            if feedback == primary_emotion:
                detector["fitness"] += 0.3 * emotion_scores[primary_emotion]
            
            # Penalty for mismatched emotion
            else:
                detector["fitness"] -= 0.2 * (1 - emotion_scores[feedback])
            
            # Bonus for confidence
            detector["fitness"] += 0.2 * emotion_scores[primary_emotion]
        
        return detector, {
            "emotion_scores": emotion_scores,
            "primary_emotion": primary_emotion
        }
    
    def _evolve_population(self, feedback_emotion: str) -> None:
        """Evolve the detector population based on feedback"""
        # Sort population by fitness
        self.detector_population.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Keep the top 25% as elites
        elite_count = self.population_size // 4
        elites = self.detector_population[:elite_count]
        
        # Create new population
        new_population = []
        
        # Add elites unmodified
        for elite in elites:
            new_population.append(elite.copy())
        
        # Fill the rest with crossover and mutation
        while len(new_population) < self.population_size:
            # Select two parents using tournament selection
            parents = []
            for _ in range(2):
                candidates = random.sample(self.detector_population, 3)
                candidates.sort(key=lambda x: x["fitness"], reverse=True)
                parents.append(candidates[0])
            
            # Create child through crossover
            child = {
                "id": str(uuid.uuid4()),
                "weights": {
                    "text_features": np.zeros(20),
                    "context_features": np.zeros(10),
                    "numogram_features": np.zeros(9)
                },
                "emotion_biases": {},
                "fitness": 0.0,
                "generation": max(p["generation"] for p in parents) + 1
            }
            
            # Crossover weights
            for feature_type in ["text_features", "context_features", "numogram_features"]:
                crossover_point = random.randint(0, len(parents[0]["weights"][feature_type]) - 1)
                child["weights"][feature_type][:crossover_point] = parents[0]["weights"][feature_type][:crossover_point]
                child["weights"][feature_type][crossover_point:] = parents[1]["weights"][feature_type][crossover_point:]
            
            # Crossover emotion biases
            for emotion in self.base_emotions:
                if random.random() < 0.5:
                    child["emotion_biases"][emotion] = parents[0]["emotion_biases"][emotion]
                else:
                    child["emotion_biases"][emotion] = parents[1]["emotion_biases"][emotion]
            
            #Apply mutation
            if random.random() < self.mutation_rate:
                # Mutate weights
                for feature_type in ["text_features", "context_features", "numogram_features"]:
                    # Select random weight to mutate
                    idx = random.randint(0, len(child["weights"][feature_type]) - 1)
                    # Apply mutation by adding noise
                    child["weights"][feature_type][idx] += np.random.normal(0, 0.5)
                
                # Mutate random emotion bias
                emotion = random.choice(list(self.base_emotions.keys()))
                child["emotion_biases"][emotion] += np.random.normal(0, 0.3)
            
            new_population.append(child)
        
        # Replace population
        self.detector_population = new_population
    
    def _calculate_digital_ratios(self, emotion_scores: Dict[str, float]) -> List[float]:
        """Calculate digital ratios for emotions based on numogram principles"""
        # Take top 3 emotions
        top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Calculate ratios between top emotions
        ratios = []
        for i in range(len(top_emotions) - 1):
            for j in range(i + 1, len(top_emotions)):
                ratio = top_emotions[i][1] / max(0.01, top_emotions[j][1])
                # Bound ratio to avoid extreme values
                ratio = min(9.0, max(1.0, ratio))
                ratios.append(ratio)
        
        return ratios
    
    def analyze_emotion(self, text: str, user_id: str, context_data: Dict = None) -> Dict:
        """Analyze emotional content of text using evolutionary detector population"""
        # Initialize context data if not provided
        if context_data is None:
            context_data = {}
        
        # Add current numogram zone to context
        current_zone = self.numogram.user_memory.get(user_id, {}).get('zone', '1')
        context_data["current_zone"] = current_zone
        
        # Extract features
        features = self._extract_features(text, context_data)
        
        # Evaluate each detector in the population
        results = []
        for detector in self.detector_population:
            updated_detector, result = self._evaluate_detector(detector, features)
            results.append((updated_detector, result))
        
        # Aggregate results using weighted ensemble (by fitness)
        aggregated_scores = {emotion: 0.0 for emotion in self.base_emotions}
        total_fitness = sum(max(0.01, detector["fitness"]) for detector, _ in results)
        
        for detector, result in results:
            weight = max(0.01, detector["fitness"]) / total_fitness
            for emotion, score in result["emotion_scores"].items():
                aggregated_scores[emotion] += score * weight
        
        # Normalize aggregated scores
        total = sum(aggregated_scores.values())
        aggregated_scores = {e: s/total for e, s in aggregated_scores.items()}
        
        # Get primary emotion
        primary_emotion = max(aggregated_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate intensity (confidence in top emotion)
        intensity = aggregated_scores[primary_emotion]
        
        # Calculate digital ratios
        digital_ratios = self._calculate_digital_ratios(aggregated_scores)
        
        # Create emotional state object
        emotional_state = EmotionalState(
            id=str(uuid.uuid4()),
            primary_emotion=primary_emotion,
            emotional_spectrum=aggregated_scores,
            numogram_zone=current_zone,
            intensity=intensity,
            timestamp=datetime.datetime.utcnow().isoformat(),
            digital_ratios=digital_ratios
        )
        
        # Add to history
        self.emotional_history.append(emotional_state)
        
        # Limit history size
        if len(self.emotional_history) > 100:
            self.emotional_history = self.emotional_history[-100:]
        
        return vars(emotional_state)
    
    def provide_feedback(self, emotion_id: str, correct_emotion: str) -> None:
        """Provide feedback to improve emotional detection"""
        # Find the emotional state
        state = None
        for es in self.emotional_history:
            if es.id == emotion_id:
                state = es
                break
                
        if state is None:
            return {"error": "Emotion state not found"}
        
        # Re-extract features from context
        context_data = {
            "current_zone": state.numogram_zone,
            "recent_emotions": state.emotional_spectrum
        }
        features = self._extract_features(state.context if hasattr(state, "context") else "", context_data)
        
        # Re-evaluate each detector with the correct emotion feedback
        for detector in self.detector_population:
            self._evaluate_detector(detector, features, feedback=correct_emotion)
        
        # Evolve the population based on feedback
        self._evolve_population(correct_emotion)
        
        return {"status": "feedback_applied", "emotion_id": emotion_id}
    
    def track_emotional_trends(self, user_id: str, time_window: int = None) -> Dict:
        """Analyze emotional trends over time"""
        # Filter by time window if specified
        if time_window:
            now = datetime.datetime.utcnow()
            cutoff = (now - datetime.timedelta(minutes=time_window)).isoformat()
            relevant_history = [es for es in self.emotional_history if es.timestamp > cutoff]
        else:
            relevant_history = self.emotional_history
            
        if not relevant_history:
            return {"status": "no_emotional_data"}
        
        # Calculate emotional distribution
        emotion_counts = {}
        for es in relevant_history:
            if es.primary_emotion not in emotion_counts:
                emotion_counts[es.primary_emotion] = 0
            emotion_counts[es.primary_emotion] += 1
        
        # Calculate average intensity
        avg_intensity = sum(es.intensity for es in relevant_history) / len(relevant_history)
        
        # Track zone-emotion correlations
        zone_emotions = {}
        for es in relevant_history:
            if es.numogram_zone not in zone_emotions:
                zone_emotions[es.numogram_zone] = {}
            
            if es.primary_emotion not in zone_emotions[es.numogram_zone]:
                zone_emotions[es.numogram_zone][es.primary_emotion] = 0
            
            zone_emotions[es.numogram_zone][es.primary_emotion] += 1
        
        # Calculate dominant emotion per zone
        dominant_zone_emotions = {}
        for zone, emotions in zone_emotions.items():
            if emotions:
                dominant_zone_emotions[zone] = max(emotions.items(), key=lambda x: x[1])[0]
        
        # Calculate emotional trajectory
        if len(relevant_history) >= 3:
            trajectory = []
            for i in range(len(relevant_history) - 2):
                # Look at three consecutive emotions
                three_emotions = [relevant_history[i+j].primary_emotion for j in range(3)]
                # If all three are the same, it's a sustained emotion
                if three_emotions[0] == three_emotions[1] == three_emotions[2]:
                    trajectory.append(f"sustained_{three_emotions[0]}")
                # If they're different, characterize the shift
                else:
                    # Get the dimensional values
                    dims = []
                    for emotion in three_emotions:
                        if emotion in self.base_emotions:
                            dims.append(self.base_emotions[emotion])
                    
                    # Calculate trajectory
                    if len(dims) == 3:
                        # See if valence is consistently changing
                        valence_changes = [dims[1]["valence"] - dims[0]["valence"], 
                                          dims[2]["valence"] - dims[1]["valence"]]
                        
                        if all(change > 0.2 for change in valence_changes):
                            trajectory.append("improving_valence")
                        elif all(change < -0.2 for change in valence_changes):
                            trajectory.append("declining_valence")
                        
                        # See if arousal is consistently changing
                        arousal_changes = [dims[1]["arousal"] - dims[0]["arousal"], 
                                          dims[2]["arousal"] - dims[1]["arousal"]]
                        
                        if all(change > 0.2 for change in arousal_changes):
                            trajectory.append("increasing_arousal")
                        elif all(change < -0.2 for change in arousal_changes):
                            trajectory.append("decreasing_arousal")
        else:
            trajectory = ["insufficient_data"]
        
        return {
            "emotion_distribution": emotion_counts,
            "average_intensity": avg_intensity,
            "zone_emotion_correlations": zone_emotions,
            "dominant_zone_emotions": dominant_zone_emotions,
            "emotional_trajectory": trajectory,
            "sample_size": len(relevant_history)
        }
    
    def integrate_with_numogram(self, emotional_state: Dict, user_id: str, context_data: Dict = None) -> Dict:
        """Integrate emotional analysis with numogram system"""
        if not emotional_state:
            return {"status": "no_emotional_state"}
        
        # Prepare context data for numogram
        if context_data is None:
            context_data = {}
            
        # Add emotional information to context
        context_data["emotional_state"] = emotional_state
        
        # Add digital ratios to context (important for numogram)
        context_data["digital_ratios"] = emotional_state.get("digital_ratios", [])
        
        # Calculate feedback based on emotional intensity
        feedback = emotional_state.get("intensity", 0.5)
        # Ensure feedback is in valid range
        feedback = max(0.1, min(0.9, feedback))
        
        # Get current zone
        current_zone = self.numogram.user_memory.get(user_id, {}).get('zone', '1')
        
        # Add zone emotion affinities to context
        if current_zone in self.zone_emotion_affinities:
            context_data["zone_emotion_affinities"] = self.zone_emotion_affinities[current_zone]
        
        # Execute numogram transition
        result = self.numogram.transition(
            user_id=user_id,
            current_zone=current_zone,
            feedback=feedback,
            context_data=context_data
        )
        
        return {
            "numogram_result": result,
            "emotional_state": emotional_state,
            "emotional_summary": {
                "primary_emotion": emotional_state.get("primary_emotion"),
                "intensity": emotional_state.get("intensity"),
                "zone_emotion_affinity": self.zone_emotion_affinities.get(current_zone, {}).get(
                    emotional_state.get("primary_emotion"), 0.0
                )
            }
        }


class NeuroevolutionaryIntegrationSystem:
    """
    Evolves neural networks through evolutionary algorithms rather than backpropagation
    to integrate symbolic patterns, emotional states, and numogram transitions
    """
    
    def __init__(self, 
                numogram_system, 
                symbol_extractor, 
                emotion_tracker,
                population_size=30, 
                mutation_rate=0.15,
                crossover_rate=0.7):
        # Store references to component systems
        self.numogram = numogram_system
        self.symbol_extractor = symbol_extractor
        self.emotion_tracker = emotion_tracker
        
        # Evolutionary algorithm parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Neural network topology (simplified for explanation)
        self.topology = {
            "input_size": 30,  # Symbolic + emotional features
            "hidden_layers": [20, 10],  # Two hidden layers
            "output_size": 9   # One output per numogram zone
        }
        
        # Initialize population of neural networks
        self.network_population = self._initialize_network_population()
        
        # Integration history
        self.integration_history = []
    
    def _initialize_network_population(self) -> List[Dict]:
        """Initialize a population of simple neural networks"""
        population = []
        
        for _ in range(self.population_size):
            # Create weights for each layer
            weights = []
            layer_sizes = [self.topology["input_size"]] + self.topology["hidden_layers"] + [self.topology["output_size"]]
            
            for i in range(len(layer_sizes) - 1):
                # Initialize with Xavier/Glorot initialization
                weight_matrix = np.random.normal(
                    0, 
                    np.sqrt(2 / (layer_sizes[i] + layer_sizes[i+1])),
                    (layer_sizes[i], layer_sizes[i+1])
                )
                weights.append(weight_matrix)
            
            # Create network object
            network = {
                "id": str(uuid.uuid4()),
                "weights": weights,
                "fitness": 0.0,
                "generation": 0,
                "novelty": 0.0,  # For novelty search
                "creation_time": datetime.datetime.utcnow().isoformat()
            }
            
            population.append(network)
        
        return population
    
    def _forward_pass(self, network: Dict, input_features: np.ndarray) -> np.ndarray:
        """Perform forward pass through the neural network"""
        # Start with input
        activation = input_features
        
        # Pass through each layer
        for i, weight_matrix in enumerate(network["weights"]):
            # Linear combination
            z = np.dot(activation, weight_matrix)
            
            # Apply activation function
            if i < len(network["weights"]) - 1:
                # ReLU for hidden layers
                activation = np.maximum(0, z)
            else:
                # Softmax for output layer
                exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
                activation = exp_z / exp_z.sum()
        
        return activation
    
    def _extract_integration_features(self, symbolic_patterns: List[Dict], emotional_state: Dict) -> np.ndarray:
        """Extract integration features from symbolic patterns and emotional state"""
        features = np.zeros(self.topology["input_size"])
        
        # Features 0-8: Zone distribution of symbolic patterns
        if symbolic_patterns:
            zone_distribution = {}
            for pattern in symbolic_patterns:
                zone = pattern.get("numogram_zone")
                if zone not in zone_distribution:
                    zone_distribution[zone] = 0
                zone_distribution[zone] += 1
            
            # Normalize zone distribution
            total_patterns = len(symbolic_patterns)
            for zone, count in zone_distribution.items():
                zone_idx = int(zone) - 1
                if 0 <= zone_idx < 9:
                    features[zone_idx] = count / total_patterns
        
        # Features 9-17: Top emotional states
        if emotional_state and "emotional_spectrum" in emotional_state:
            emotion_scores = emotional_state["emotional_spectrum"]
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get indices for tracked emotions
            emotion_indices = {
                "joy": 9, "trust": 10, "fear": 11, "surprise": 12,
                "sadness": 13, "disgust": 14, "anger": 15, "anticipation": 16,
                "curiosity": 17
            }
            
            # Set feature values for emotions
            for emotion, score in sorted_emotions:
                if emotion in emotion_indices:
                    features[emotion_indices[emotion]] = score
        
        # Features 18-20: Emotional intensity and primary metrics 
        if emotional_state:
            features[18] = emotional_state.get("intensity", 0.5)
            
            # Digital ratios (if available)
            digital_ratios = emotional_state.get("digital_ratios", [])
            for i, ratio in enumerate(digital_ratios[:2]):  # Use up to 2 ratios
                if i < 2:
                    features[19 + i] = ratio / 9.0  # Normalize by max possible ratio
        
        # Features 21-29: Pattern intensities by zone
        if symbolic_patterns:
            zone_intensities = {}
            for pattern in symbolic_patterns:
                zone = pattern.get("numogram_zone")
                intensity = pattern.get("intensity", 0.5)
                if zone not in zone_intensities:
                    zone_intensities[zone] = []
                zone_intensities[zone].append(intensity)
            
            # Average intensity per zone
            for zone, intensities in zone_intensities.items():
                zone_idx = int(zone) - 1
                if 0 <= zone_idx < 9:
                    features[21 + zone_idx] = sum(intensities) / len(intensities)
        
        return features
    
    def _evaluate_network(self, network: Dict, features: np.ndarray, target_zone: str = None) -> Tuple[Dict, Dict]:
        """Evaluate a network on the features"""
        # Perform forward pass
        outputs = self._forward_pass(network, features)
        
        # Predicted zone is the one with highest activation
        predicted_zone = str(np.argmax(outputs) + 1)  # Convert to 1-indexed zone
        
        # Zone activation pattern
        zone_activations = {str(i+1): float(outputs[i]) for i in range(len(outputs))}
        
        # Update fitness if target provided
        if target_zone is not None:
            # Base fitness (avoid zero fitness)
            network["fitness"] = 0.1
            
            # Correct prediction gives highest fitness
            if predicted_zone == target_zone:
                network["fitness"] += 0.7 * outputs[int(target_zone) - 1]
            
            # Partial credit for adjacent zones on the numogram
            adjacent_zones = {
                "1": ["2", "4", "8"],
                "2": ["1", "3", "6"],
                "3": ["2", "7", "9"],
                "4": ["1", "5", "7"],
                "5": ["4", "6", "8"],
                "6": ["2", "5", "9"],
                "7": ["3", "4", "8"],
                "8": ["1", "5", "7"],
                "9": ["3", "6"]
            }
            
            if predicted_zone in adjacent_zones.get(target_zone, []):
                network["fitness"] += 0.3 * outputs[int(predicted_zone) - 1]
        
        return network, {
            "predicted_zone": predicted_zone,
            "confidence": float(outputs[int(predicted_zone) - 1]),
            "zone_activations": zone_activations
        }
    
    def _evolve_networks(self) -> None:
        """Evolve the neural network population"""
        # Sort population by fitness
        self.network_population.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Calculate novelty scores (to promote diversity)
        self._calculate_novelty_scores()
        
        # Combine fitness and novelty for selection (weighted sum)
        for network in self.network_population:
            network["selection_score"] = 0.7 * network["fitness"] + 0.3 * network["novelty"]
        
        # Sort by selection score
        self.network_population.sort(key=lambda x: x["selection_score"], reverse=True)
        
        # Keep top 25% as elites
        elite_count = self.population_size // 4
        elites = self.network_population[:elite_count]
        
        # Create new population
        new_population = []
        
        # Add elites unmodified
        for elite in elites:
            elite_copy = elite.copy()
            elite_copy["weights"] = [w.copy() for w in elite["weights"]]
            new_population.append(elite_copy)
        
        # Fill the rest with crossover and mutation
        while len(new_population) < self.population_size:
            # Select two parents using tournament selection
            parents = []
            for _ in range(2):
                candidates = random.sample(self.network_population, 3)
                candidates.sort(key=lambda x: x["selection_score"], reverse=True)
                parents.append(candidates[0])
            
            # Create child
            child = {
                "id": str(uuid.uuid4()),
                "weights": [],
                "fitness": 0.0,
                "novelty": 0.0,
                "generation": max(p["generation"] for p in parents) + 1,
                "creation_time": datetime.datetime.utcnow().isoformat()
            }
            
            # Apply crossover with probability
            if random.random() < self.crossover_rate:
                for layer_idx in range(len(parents[0]["weights"])):
                    # Create layer weight matrix
                    parent1_weights = parents[0]["weights"][layer_idx]
                    parent2_weights = parents[1]["weights"][layer_idx]
                    
                    # Method 1: Element-wise crossover
                    mask = np.random.randint(0, 2, size=parent1_weights.shape).astype(bool)
                    child_weights = np.zeros_like(parent1_weights)
                    child_weights[mask] = parent1_weights[mask]
                    child_weights[~mask] = parent2_weights[~mask]
                    
                    child["weights"].append(child_weights)
            else:
                # No crossover, just copy from one parent
                parent = random.choice(parents)
                child["weights"] = [w.copy() for w in parent["weights"]]
            
            # Apply mutation with probability
            if random.random() < self.mutation_rate:
                for layer_idx, weight_matrix in enumerate(child["weights"]):
                    # Generate mutation mask (controls which weights are mutated)
                    mutation_mask = np.random.random(weight_matrix.shape) < 0.1
                    
                    # Generate mutation noise (how much weights are changed)
                    mutation_noise = np.random.normal(0, 0.5, weight_matrix.shape)
                    
                    # Apply mutation
                    child["weights"][layer_idx] = np.where(
                        mutation_mask, 
                        weight_matrix + mutation_noise, 
                        weight_matrix
                    )
            
            new_population.append(child)
        
        # Replace population
        self.network_population = new_population
    
    def _calculate_novelty_scores(self) -> None:
        """Calculate novelty scores for each network based on behavior distance"""
        # For a simple approach, we'll use the network weights as the behavior
        # In a more sophisticated implementation, we'd use the actual outputs
        # on a diverse set of inputs
        
        # Calculate average weight values for each network
        behaviors = []
        for network in self.network_population:
            # For simplicity, flatten and average weights
            flattened = []
            for w in network["weights"]:
                flattened.extend(w.flatten())
            avg_weights = np.mean(np.abs(flattened))
            behaviors.append(avg_weights)
        
        # Normalize behaviors
        min_b = min(behaviors)
        max_b = max(behaviors)
        range_b = max(0.001, max_b - min_b)  # Avoid division by zero
        normalized_behaviors = [(b - min_b) / range_b for b in behaviors]
        
        # Calculate novelty as distance from nearest neighbors
        for i, behavior in enumerate(normalized_behaviors):
            # Calculate distance to all other behaviors
            distances = [abs(behavior - other) for j, other in enumerate(normalized_behaviors) if i != j]
            
            # Sort distances
            distances.sort()
            
            # Take average of 3 nearest neighbors
            nearest_neighbors = distances[:3]
            if nearest_neighbors:
                novelty = sum(nearest_neighbors) / len(nearest_neighbors)
                self.network_population[i]["novelty"] = novelty
            else:
                self.network_population[i]["novelty"] = 0.0
    
    def integrate(self, text: str, user_id: str, context_data: Dict = None) -> Dict:
        """
        Main integration function that connects symbolic patterns, emotional tracking and numogram
        using neuroevolutionary techniques
        """
        # Initialize context if not provided
        if context_data is None:
            context_data = {}
        
        # 1. Extract symbolic patterns
        symbolic_patterns = self.symbol_extractor.extract_symbols(text, user_id)
        
        # 2. Analyze emotional state
        emotional_state = self.emotion_tracker.analyze_emotion(text, user_id, context_data)
        
        # 3. Get current numogram zone
        current_zone = self.numogram.user_memory.get(user_id, {}).get('zone', '1')
        
        # 4. Extract integration features
        features = self._extract_integration_features(symbolic_patterns, emotional_state)
        
        # 5. Find best network in current population
        best_network = None
        best_fitness = -1
        
        for network in self.network_population:
            if network["fitness"] > best_fitness:
                best_fitness = network["fitness"]
                best_network = network
        
        # 6. Use best network to predict zone
        if best_network:
            _, prediction = self._evaluate_network(best_network, features)
            predicted_zone = prediction["predicted_zone"]
            confidence = prediction["confidence"]
        else:
            # Fallback if no networks available
            predicted_zone = current_zone
            confidence = 0.5
        
        # 7. Make numogram transition
        feedback = confidence  # Use network confidence as feedback
        transition_result = self.numogram.transition(
            user_id=user_id,
            current_zone=predicted_zone,  # Use predicted zone instead of current
            feedback=feedback,
            context_data={
                **context_data,
                "symbolic_patterns": symbolic_patterns,
                "emotional_state": emotional_state,
                "neuroevolution_prediction": {
                    "predicted_zone": predicted_zone,
                    "confidence": confidence
                }
            }
        )
        
        # 8. Get actual next zone
        next_zone = transition_result["next_zone"]
        
        # 9. Update network fitness based on actual result
        for network in self.network_population:
            self._evaluate_network(network, features, target_zone=next_zone)
        
        # 10. Evolve networks
        self._evolve_networks()
        
        # 11. Record integration in history
        integration_record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "user_id": user_id,
            "text_input": text,
            "symbolic_pattern_count": len(symbolic_patterns),
            "primary_emotion": emotional_state.get("primary_emotion"),
            "current_zone": current_zone,
            "predicted_zone": predicted_zone,
            "next_zone": next_zone,
            "network_confidence": confidence,
            "integration_features": features.tolist()
        }
        
        self.integration_history.append(integration_record)
        
        # 12. Limit history size
        if len(self.integration_history) > 100:
            self.integration_history = self.integration_history[-100:]
        
        # 13. Return comprehensive result
        return {
            "user_id": user_id,
            "text_input": text,
            "symbolic_patterns": symbolic_patterns[:5],  # Limit to top 5 for response
            "emotional_state": emotional_state,
            "numogram_transition": transition_result,
            "neuroevolution": {
                "predicted_zone": predicted_zone,
                "network_confidence": confidence,
                "best_network_generation": best_network["generation"] if best_network else 0,
                "population_size": len(self.network_population)
            },
            "integration_id": integration_record["id"]
        }
    
    def get_integration_statistics(self) -> Dict:
        """Get statistics about integration performance"""
        if not self.integration_history:
            return {"status": "no_integration_data"}
        
        # Calculate prediction accuracy
        correct_predictions = sum(
            1 for record in self.integration_history 
            if record["predicted_zone"] == record["next_zone"]
        )
        total_predictions = len(self.integration_history)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate zone distribution
        zone_distribution = {}
        for record in self.integration_history:
            zone = record["next_zone"]
            if zone not in zone_distribution:
                zone_distribution[zone] = 0
            zone_distribution[zone] += 1
        
        # Normalize
        for zone in zone_distribution:
            zone_distribution[zone] /= total_predictions
        
        # Calculate evolutionary progress
        generations = []
        fitnesses = []
        for network in self.network_population:
            generations.append(network["generation"])
            fitnesses.append(network["fitness"])
        
        avg_generation = sum(generations) / len(generations) if generations else 0
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
        max_fitness = max(fitnesses) if fitnesses else 0
        
        return {
            "prediction_accuracy": accuracy,
            "zone_distribution": zone_distribution,
            "evolutionary_metrics": {
                "average_generation": avg_generation,
                "average_fitness": avg_fitness,
                "max_fitness": max_fitness
            },
            "integration_count": total_predictions
        }
    
    def export_best_networks(self, count: int = 3) -> List[Dict]:
        """Export the best networks for external use"""
        # Sort by fitness
        sorted_networks = sorted(self.network_population, key=lambda x: x["fitness"], reverse=True)
        
        # Take top networks
        top_networks = sorted_networks[:count]
        
        # Convert to serializable format
        serialized = []
        for network in top_networks:
            # Convert numpy arrays to lists
            weights_serialized = []
            for layer_weights in network["weights"]:
                weights_serialized.append(layer_weights.tolist())
                
            serialized.append({
                "id": network["id"],
                "weights": weights_serialized,
                "fitness": network["fitness"],
                "generation": network["generation"],
                "creation_time": network["creation_time"]
            })
            
        return serialized


class NumogramEvolutionarySystem:
    """
    Main class that integrates:
    1. Numogram transition system
    2. Symbolic pattern extraction with numogram mapping
    3. Emotional tracking with evolutionary algorithms
    4. Neuroevolutionary integration layer
    """
    
    def __init__(self, config_path=None):
        """Initialize the complete system"""
        # Load configuration if provided
        self.config = self._load_config(config_path)

        # Create the numogram system
        self.numogram = NumogramSystem(
            zones_file=self.config.get("zones_file", "numogram_code/zones.json"),
            memory_file=self.config.get("memory_file", "numogram_code/user_memory.json")
        )
        
        # Create the symbolic pattern extractor
        self.symbol_extractor = NumogramaticSymbolExtractor(self.numogram)
        
        # Create the emotional evolution system
        self.emotion_tracker = EmotionalEvolutionSystem(
            self.numogram, 
            population_size=self.config.get("emotion_population_size", 50),
            mutation_rate=self.config.get("emotion_mutation_rate", 0.1)
        )
        
        # Create the neuroevolutionary integration system
        self.neuro_integrator = NeuroevolutionaryIntegrationSystem(
            self.numogram,
            self.symbol_extractor,
            self.emotion_tracker,
            population_size=self.config.get("network_population_size", 30),
            mutation_rate=self.config.get("network_mutation_rate", 0.15),
            crossover_rate=self.config.get("network_crossover_rate", 0.7)
        )
        
        # System metadata
        self.system_id = str(uuid.uuid4())
        self.created_at = datetime.datetime.utcnow().isoformat()
        self.system_version = "1.0.0"
        self.active_sessions = {}
    
    def _load_config(self, config_path) -> Dict:
        """Load configuration from JSON file or use defaults"""
        default_config = {
            "zones_file": "numogram_code/zones.json",
            "memory_file": "numogram_code/user_memory.json",
            "emotion_population_size": 50,
            "emotion_mutation_rate": 0.1,
            "network_population_size": 30,
            "network_mutation_rate": 0.15,
            "network_crossover_rate": 0.7,
            "save_directory": "output",
            "auto_save_interval": 10  # Save every 10 interactions
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Update defaults with loaded config
                    default_config.update(loaded_config)
            except (json.JSONDecodeError, IOError):
                pass  # Silently use defaults on error
        
        return default_config
    
    def initialize_session(self, user_id: str, session_name: str = None) -> Dict[str, Any]:
        """Initialize a new session"""
        session_id = str(uuid.uuid4())
        session_name = session_name or f"NumogramSession-{session_id[:8]}"
        
        session = {
            "id": session_id,
            "name": session_name,
            "user_id": user_id,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "interactions": [],
            "symbolic_patterns": [],
            "emotional_states": [],
            "numogram_transitions": [],
            "active": True
        }
        
        self.active_sessions[session_id] = session
        return session
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End an active session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        session["active"] = False
        session["ended_at"] = datetime.datetime.utcnow().isoformat()
        
        return session
    
    def process(self, session_id: str, text: str, context_data: Dict = None) -> Dict[str, Any]:
        """Process text input through the integrated system"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
            
        session = self.active_sessions[session_id]
        user_id = session["user_id"]
        
        # Initialize context if not provided
        if context_data is None:
            context_data = {}
        
        # Add session context
        context_data["session_id"] = session_id
        context_data["session_name"] = session["name"]
        context_data["interaction_count"] = len(session["interactions"])
        
        # Process through neuroevolutionary integrator
        result = self.neuro_integrator.integrate(text, user_id, context_data)
        
        # Store in session history
        interaction = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "text_input": text,
            "integration_result": result
        }
        
        session["interactions"].append(interaction)
        
        # Store symbolic patterns
        session["symbolic_patterns"].extend(result.get("symbolic_patterns", []))
        
        # Store emotional state
        if "emotional_state" in result:
            session["emotional_states"].append(result["emotional_state"])
        
        # Store numogram transition
        if "numogram_transition" in result:
            session["numogram_transitions"].append(result["numogram_transition"])
        
        # Auto-save if needed
        if len(session["interactions"]) % self.config.get("auto_save_interval", 10) == 0:
            self._auto_save_session(session_id)
        
        return {
            "session_id": session_id,
            "interaction_id": interaction["id"],
            "integration_result": result,
            "current_zone": result.get("numogram_transition", {}).get("next_zone", "1")
        }
    
    def _auto_save_session(self, session_id: str) -> None:
        """Automatically save session to disk"""
        if session_id not in self.active_sessions:
            return
            
        session = self.active_sessions[session_id]
        save_dir = self.config.get("save_directory", "output")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Build filename with timestamp
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/session_{session_id}_{timestamp}.json"
        
        # Save session data
        with open(filename, 'w') as f:
            json.dump(session, f, indent=2)
    
    def get_zone_info(self, zone: str) -> Dict[str, Any]:
        """Get detailed information about a numogram zone"""
        zone_data = self.numogram.ZONE_DATA.get(zone, {})
        
        # Add symbolic associations from symbol extractor
        symbolic_associations = []
        for symbol_category, symbols in self.symbol_extractor.zone_symbols.items():
            if zone in symbol_category:
                symbolic_associations.extend(symbols)
        
        # Add emotional associations from emotion tracker
        emotional_associations = self.emotion_tracker.zone_emotion_affinities.get(zone, {})
        
        return {
            "zone": zone,
            "zone_data": zone_data,
            "symbolic_associations": symbolic_associations,
            "emotional_associations": emotional_associations
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status"""
        # Count active sessions
        active_count = sum(1 for s in self.active_sessions.values() if s.get("active", False))
        
        # Get numogram stats
        numogram_user_count = len(self.numogram.user_memory) if hasattr(self.numogram, 'user_memory') else 0
        
        # Get neuroevolution stats
        neuro_stats = self.neuro_integrator.get_integration_statistics()
        
        return {
            "system_id": self.system_id,
            "version": self.system_version,
            "created_at": self.created_at,
            "current_time": datetime.datetime.utcnow().isoformat(),
            "active_sessions": active_count,
            "total_sessions": len(self.active_sessions),
            "numogram_users": numogram_user_count,
            "neuroevolution_stats": neuro_stats,
            "components": {
                "numogram": "active",
                "symbol_extractor": "active",
                "emotion_tracker": "active",
                "neuro_integrator": "active"
            }
        }
    
    def export_session_data(self, session_id: str, format: str = "json") -> str:
        """Export session data in the requested format"""
        if session_id not in self.active_sessions:
            return json.dumps({"error": "Session not found"})
        
        session = self.active_sessions[session_id]
        
        if format.lower() == "json":
            return json.dumps(session, indent=2)
        else:
            return json.dumps({"error": f"Unsupported format: {format}"})
    
    def save_system_state(self, filepath: str) -> Dict[str, Any]:
        """Save complete system state for later resumption"""
        # Create state bundle
        state = {
            "system_metadata": {
                "system_id": self.system_id,
                "version": self.system_version,
                "created_at": self.created_at,
                "saved_at": datetime.datetime.utcnow().isoformat()
            },
            "config": self.config,
            "sessions": self.active_sessions,
            # Neural networks are saved separately since they contain numpy arrays
            "best_networks": self.neuro_integrator.export_best_networks(5)
        }
        
        # Save state to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save numogram state (it has its own save mechanism)
        numogram_save_path = os.path.join(os.path.dirname(filepath), "numogram_state.json")
        # Call numogram save method here
        
        return {
            "status": "success",
            "saved_at": state["system_metadata"]["saved_at"],
            "filepath": filepath,
            "numogram_filepath": numogram_save_path
        }
```
