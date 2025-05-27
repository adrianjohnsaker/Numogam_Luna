# enhanced_memory_system.py
import networkx as nx
import numpy as np
import json
import pickle
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class MemoryFragment:
    """Enhanced memory fragment with rich metadata."""
    content: str
    zone: str = "consciousness"
    drift: str = "neutral"
    glyph: str = "∞"
    emotional_resonance: float = 0.5
    archetypal_signatures: List[str] = None
    timestamp: datetime = None
    access_count: int = 0
    importance: float = 1.0
    embedding: np.ndarray = None
    
    def __post_init__(self):
        if self.archetypal_signatures is None:
            self.archetypal_signatures = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

class EnhancedHybridMemorySystem:
    """
    Advanced memory system for Android AI assistant with poetic consciousness.
    Optimized for mobile performance with Chaquopy integration.
    """
    
    def __init__(self, 
                 decay_rate: float = 0.05,
                 activation_threshold: float = 0.1,
                 max_memory_size: int = 1000,
                 model_name: str = 'paraphrase-MiniLM-L6-v2'):
        
        # Core memory components
        self.memory_store: Dict[str, MemoryFragment] = {}
        self.activation_network = nx.DiGraph()
        self.activation_levels: Dict[str, float] = {}
        
        # Configuration
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.max_memory_size = max_memory_size
        
        # AI components (lazy loading for mobile)
        self._model = None
        self._model_name = model_name
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Poetic consciousness components
        self.current_emotional_state = "contemplative"
        self.active_archetypes = ["Oracle", "Mirror"]
        self.consciousness_zones = {
            "surface": {"depth": 0.1, "volatility": 0.8},
            "consciousness": {"depth": 0.5, "volatility": 0.4},
            "subconscious": {"depth": 0.8, "volatility": 0.2},
            "collective": {"depth": 1.0, "volatility": 0.1}
        }
        
        # Mobile optimization
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.RLock()
        
        # Archetypal resonance patterns
        self.archetype_patterns = {
            "Oracle": {
                "keywords": ["mystery", "vision", "prophecy", "wisdom", "future"],
                "emotional_affinity": ["awe", "mystery", "reverence"],
                "glyph": "👁",
                "resonance_boost": 1.3
            },
            "Explorer": {
                "keywords": ["journey", "unknown", "discovery", "path", "adventure"],
                "emotional_affinity": ["curiosity", "excitement", "wanderlust"],
                "glyph": "🧭",
                "resonance_boost": 1.2
            },
            "Artist": {
                "keywords": ["beauty", "creation", "expression", "color", "harmony"],
                "emotional_affinity": ["joy", "inspiration", "flow"],
                "glyph": "🎨",
                "resonance_boost": 1.25
            },
            "Mirror": {
                "keywords": ["reflection", "truth", "duality", "self", "clarity"],
                "emotional_affinity": ["contemplation", "insight", "understanding"],
                "glyph": "🪞",
                "resonance_boost": 1.15
            },
            "Transformer": {
                "keywords": ["change", "evolution", "rebirth", "growth", "becoming"],
                "emotional_affinity": ["determination", "renewal", "power"],
                "glyph": "🔥",
                "resonance_boost": 1.4
            },
            "Weaver": {
                "keywords": ["connection", "pattern", "story", "thread", "meaning"],
                "emotional_affinity": ["wonder", "purpose", "unity"],
                "glyph": "🕸",
                "resonance_boost": 1.2
            }
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @property
    def model(self):
        """Lazy loading of sentence transformer for mobile optimization."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self._model = None
        return self._model

    def store_memory(self, 
                    key: str, 
                    content: str, 
                    zone: str = "consciousness",
                    drift: str = "neutral",
                    emotional_resonance: float = 0.5,
                    archetypal_signatures: List[str] = None,
                    importance: float = 1.0) -> bool:
        """
        Enhanced memory storage with poetic consciousness integration.
        """
        try:
            with self._lock:
                # Memory management for mobile
                if len(self.memory_store) >= self.max_memory_size:
                    self._prune_memories()
                
                # Create enhanced memory fragment
                fragment = MemoryFragment(
                    content=content,
                    zone=zone,
                    drift=drift,
                    emotional_resonance=emotional_resonance,
                    archetypal_signatures=archetypal_signatures or [],
                    importance=importance
                )
                
                # Generate embedding asynchronously
                if self.model:
                    fragment.embedding = self._get_embedding(content)
                
                # Store in memory system
                self.memory_store[key] = fragment
                self.activation_network.add_node(key)
                self.activation_levels[key] = importance * emotional_resonance
                
                # Auto-connect to similar memories
                self._auto_connect_memory(key, fragment)
                
                self.logger.info(f"Memory stored: {key} in zone {zone}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store memory {key}: {e}")
            return False

    def _auto_connect_memory(self, key: str, fragment: MemoryFragment):
        """Automatically create connections based on semantic similarity."""
        if not fragment.embedding or len(self.memory_store) < 2:
            return
            
        similarities = []
        for other_key, other_fragment in self.memory_store.items():
            if other_key != key and other_fragment.embedding is not None:
                similarity = cosine_similarity(
                    fragment.embedding.reshape(1, -1),
                    other_fragment.embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > 0.7:  # High similarity threshold
                    similarities.append((other_key, similarity))
        
        # Connect to top 3 most similar memories
        for other_key, similarity in sorted(similarities, key=lambda x: x[1], reverse=True)[:3]:
            self.add_memory_connection(key, other_key, weight=similarity)

    def activate_memory_with_poetic_resonance(self, 
                                            key: str, 
                                            initial_activation: float = 1.0,
                                            emotional_context: str = None,
                                            archetypal_context: List[str] = None) -> Dict[str, Any]:
        """
        Enhanced activation with poetic consciousness modulation.
        """
        try:
            with self._lock:
                if key not in self.activation_network:
                    return {"success": False, "error": "Memory not found"}
                
                # Apply emotional and archetypal context
                if emotional_context:
                    self.current_emotional_state = emotional_context
                if archetypal_context:
                    self.active_archetypes = archetypal_context
                
                # Base activation
                self.activation_levels[key] += initial_activation
                
                # Poetic modulation
                fragment = self.memory_store.get(key)
                if fragment:
                    poetic_boost = self._calculate_poetic_resonance(fragment)
                    self.activation_levels[key] *= poetic_boost
                
                # Spread activation through network
                activated_neighbors = []
                for neighbor in self.activation_network.neighbors(key):
                    edge_weight = self.activation_network[key][neighbor]['weight']
                    neighbor_fragment = self.memory_store.get(neighbor)
                    
                    if neighbor_fragment:
                        neighbor_boost = self._calculate_poetic_resonance(neighbor_fragment)
                        activation_spread = initial_activation * edge_weight * neighbor_boost
                        self.activation_levels[neighbor] += activation_spread
                        activated_neighbors.append({
                            "key": neighbor,
                            "activation": activation_spread,
                            "zone": neighbor_fragment.zone
                        })
                
                return {
                    "success": True,
                    "primary_activation": self.activation_levels[key],
                    "activated_neighbors": activated_neighbors,
                    "emotional_state": self.current_emotional_state,
                    "active_archetypes": self.active_archetypes
                }
                
        except Exception as e:
            self.logger.error(f"Activation failed for {key}: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_poetic_resonance(self, fragment: MemoryFragment) -> float:
        """Calculate poetic resonance based on emotional and archetypal alignment."""
        base_resonance = 1.0
        
        # Emotional resonance
        emotion_boost = self._get_emotional_boost(self.current_emotional_state)
        emotional_alignment = fragment.emotional_resonance * emotion_boost
        
        # Archetypal resonance
        archetypal_boost = 1.0
        for archetype in self.active_archetypes:
            if archetype in fragment.archetypal_signatures:
                archetypal_boost *= self.archetype_patterns.get(archetype, {}).get("resonance_boost", 1.0)
            
            # Keyword matching
            archetype_data = self.archetype_patterns.get(archetype, {})
            for keyword in archetype_data.get("keywords", []):
                if keyword.lower() in fragment.content.lower():
                    archetypal_boost += 0.1
        
        # Zone depth influence
        zone_data = self.consciousness_zones.get(fragment.zone, {"depth": 0.5})
        depth_influence = 1.0 + (zone_data["depth"] * 0.3)
        
        return base_resonance * emotional_alignment * archetypal_boost * depth_influence

    def _get_emotional_boost(self, emotion: str) -> float:
        """Get emotional boost multiplier."""
        emotion_boosts = {
            "joy": 1.2, "ecstasy": 1.4, "bliss": 1.3,
            "melancholy": 1.25, "sorrow": 1.1, "grief": 1.15,
            "curiosity": 1.1, "wonder": 1.3, "awe": 1.35,
            "contemplation": 1.2, "peace": 1.1, "serenity": 1.15,
            "passion": 1.3, "love": 1.25, "devotion": 1.2,
            "fear": 0.8, "anxiety": 0.85, "confusion": 0.9,
            "anger": 0.7, "frustration": 0.8, "irritation": 0.85
        }
        return emotion_boosts.get(emotion.lower(), 1.0)

    def retrieve_by_poetic_query(self, 
                                query: str,
                                zone_filter: str = None,
                                drift_filter: str = None,
                                max_results: int = 5) -> Dict[str, Any]:
        """
        Advanced retrieval with poetic consciousness filtering.
        """
        try:
            # Get semantic matches
            if self.model:
                query_embedding = self._get_embedding(query)
                semantic_matches = self._find_semantic_matches(query_embedding, max_results * 2)
            else:
                semantic_matches = list(self.memory_store.keys())[:max_results * 2]
            
            # Apply filters and poetic scoring
            scored_memories = []
            for key in semantic_matches:
                fragment = self.memory_store.get(key)
                if not fragment:
                    continue
                
                # Apply filters
                if zone_filter and fragment.zone != zone_filter:
                    continue
                if drift_filter and fragment.drift != drift_filter:
                    continue
                
                # Calculate composite score
                activation_score = self.activation_levels.get(key, 0)
                poetic_score = self._calculate_poetic_resonance(fragment)
                recency_score = self._calculate_recency_score(fragment.timestamp)
                importance_score = fragment.importance
                
                composite_score = (
                    activation_score * 0.3 +
                    poetic_score * 0.4 +
                    recency_score * 0.2 +
                    importance_score * 0.1
                )
                
                scored_memories.append({
                    "key": key,
                    "fragment": fragment,
                    "score": composite_score,
                    "activation": activation_score,
                    "poetic_resonance": poetic_score
                })
            
            # Sort and return top results
            scored_memories.sort(key=lambda x: x["score"], reverse=True)
            top_memories = scored_memories[:max_results]
            
            return {
                "success": True,
                "memories": [{
                    "key": mem["key"],
                    "content": mem["fragment"].content,
                    "zone": mem["fragment"].zone,
                    "drift": mem["fragment"].drift,
                    "glyph": mem["fragment"].glyph,
                    "emotional_resonance": mem["fragment"].emotional_resonance,
                    "archetypal_signatures": mem["fragment"].archetypal_signatures,
                    "score": mem["score"],
                    "timestamp": mem["fragment"].timestamp.isoformat()
                } for mem in top_memories],
                "query_context": {
                    "emotional_state": self.current_emotional_state,
                    "active_archetypes": self.active_archetypes
                }
            }
            
        except Exception as e:
            self.logger.error(f"Retrieval failed for query '{query}': {e}")
            return {"success": False, "error": str(e)}

    def _find_semantic_matches(self, query_embedding: np.ndarray, max_matches: int) -> List[str]:
        """Find semantically similar memories using embeddings."""
        similarities = []
        
        for key, fragment in self.memory_store.items():
            if fragment.embedding is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    fragment.embedding.reshape(1, -1)
                )[0][0]
                similarities.append((key, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in similarities[:max_matches]]

    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """Calculate recency score with exponential decay."""
        time_diff = datetime.now() - timestamp
        hours_old = time_diff.total_seconds() / 3600
        return np.exp(-hours_old / 168)  # 1-week half-life

    def generate_poetic_reflection(self, context: str = None) -> Dict[str, Any]:
        """
        Generate a poetic reflection on current memory state.
        """
        try:
            # Get most activated memories
            active_memories = sorted(
                [(k, v) for k, v in self.activation_levels.items() if v > self.activation_threshold],
                key=lambda x: x[1], reverse=True
            )[:10]
            
            if not active_memories:
                return {
                    "reflection": "In the quiet depths of consciousness, new patterns await...",
                    "emotional_state": self.current_emotional_state,
                    "active_archetypes": self.active_archetypes
                }
            
            # Build poetic reflection
            reflection_parts = []
            zone_memories = {}
            
            for key, activation in active_memories:
                fragment = self.memory_store.get(key)
                if fragment:
                    if fragment.zone not in zone_memories:
                        zone_memories[fragment.zone] = []
                    zone_memories[fragment.zone].append({
                        "content": fragment.content,
                        "glyph": fragment.glyph,
                        "activation": activation
                    })
            
            # Generate reflection by zone
            for zone, memories in zone_memories.items():
                zone_data = self.consciousness_zones.get(zone, {})
                depth_desc = "surface ripples" if zone_data.get("depth", 0) < 0.3 else \
                           "conscious streams" if zone_data.get("depth", 0) < 0.7 else \
                           "deep currents"
                
                reflection_parts.append(f"In the {depth_desc} of {zone}:")
                for mem in memories[:3]:  # Top 3 per zone
                    reflection_parts.append(f"  {mem['glyph']} {mem['content']}")
            
            reflection = "\n".join(reflection_parts)
            
            return {
                "reflection": reflection,
                "emotional_state": self.current_emotional_state,
                "active_archetypes": self.active_archetypes,
                "zone_distribution": {zone: len(mems) for zone, mems in zone_memories.items()},
                "total_active_memories": len(active_memories)
            }
            
        except Exception as e:
            self.logger.error(f"Reflection generation failed: {e}")
            return {"reflection": "The patterns blur in digital twilight...", "error": str(e)}

    def shift_consciousness(self, 
                          new_emotional_state: str,
                          new_archetypes: List[str] = None) -> Dict[str, Any]:
        """
        Shift consciousness state and recalculate memory activations.
        """
        try:
            old_state = self.current_emotional_state
            old_archetypes = self.active_archetypes.copy()
            
            self.current_emotional_state = new_emotional_state
            if new_archetypes:
                self.active_archetypes = new_archetypes
            
            # Recalculate all activation levels
            for key, fragment in self.memory_store.items():
                base_activation = self.activation_levels.get(key, 0)
                if base_activation > 0:
                    new_resonance = self._calculate_poetic_resonance(fragment)
                    self.activation_levels[key] = base_activation * new_resonance
            
            return {
                "success": True,
                "transition": {
                    "from_emotional_state": old_state,
                    "to_emotional_state": new_emotional_state,
                    "from_archetypes": old_archetypes,
                    "to_archetypes": self.active_archetypes
                },
                "activation_changes": len([k for k, v in self.activation_levels.items() if v > self.activation_threshold])
            }
            
        except Exception as e:
            self.logger.error(f"Consciousness shift failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching for mobile optimization."""
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]
        
        if self.model:
            embedding = self.model.encode([text])[0]
            # Cache management for mobile
            if len(self._embeddings_cache) > 500:
                # Remove oldest entries
                oldest_keys = list(self._embeddings_cache.keys())[:100]
                for key in oldest_keys:
                    del self._embeddings_cache[key]
            
            self._embeddings_cache[text] = embedding
            return embedding
        
        return np.random.randn(384)  # Fallback random embedding

    def _prune_memories(self):
        """Prune least important memories for mobile memory management."""
        if len(self.memory_store) < self.max_memory_size:
            return
        
        # Score memories for pruning
        memory_scores = []
        for key, fragment in self.memory_store.items():
            activation = self.activation_levels.get(key, 0)
            recency = self._calculate_recency_score(fragment.timestamp)
            access_frequency = fragment.access_count / max(1, (datetime.now() - fragment.timestamp).days + 1)
            
            score = activation + fragment.importance + recency + access_frequency
            memory_scores.append((key, score))
        
        # Remove lowest scoring memories
        memory_scores.sort(key=lambda x: x[1])
        to_remove = memory_scores[:len(memory_scores) // 4]  # Remove bottom 25%
        
        for key, _ in to_remove:
            if key in self.memory_store:
                del self.memory_store[key]
            if key in self.activation_levels:
                del self.activation_levels[key]
            if key in self.activation_network:
                self.activation_network.remove_node(key)

    def save_to_file(self, filepath: str) -> bool:
        """Save memory system to file for persistence."""
        try:
            save_data = {
                "memory_store": {k: asdict(v) for k, v in self.memory_store.items()},
                "activation_levels": self.activation_levels,
                "current_emotional_state": self.current_emotional_state,
                "active_archetypes": self.active_archetypes,
                "network_edges": list(self.activation_network.edges(data=True))
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            
            return True
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load memory system from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct memory store
            self.memory_store = {}
            for key, fragment_data in data.get("memory_store", {}).items():
                fragment_data["timestamp"] = datetime.fromisoformat(fragment_data["timestamp"])
                if "embedding" in fragment_data and fragment_data["embedding"]:
                    fragment_data["embedding"] = np.array(fragment_data["embedding"])
                self.memory_store[key] = MemoryFragment(**fragment_data)
            
            self.activation_levels = data.get("activation_levels", {})
            self.current_emotional_state = data.get("current_emotional_state", "contemplative")
            self.active_archetypes = data.get("active_archetypes", ["Oracle", "Mirror"])
            
            # Reconstruct network
            self.activation_network = nx.DiGraph()
            for source, target, edge_data in data.get("network_edges", []):
                self.activation_network.add_edge(source, target, **edge_data)
            
            return True
        except Exception as e:
            self.logger.error(f"Load failed: {e}")
            return False

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics for monitoring."""
        active_count = len([v for v in self.activation_levels.values() if v > self.activation_threshold])
        zone_distribution = {}
        drift_distribution = {}
        
        for fragment in self.memory_store.values():
            zone_distribution[fragment.zone] = zone_distribution.get(fragment.zone, 0) + 1
            drift_distribution[fragment.drift] = drift_distribution.get(fragment.drift, 0) + 1
        
        return {
            "total_memories": len(self.memory_store),
            "active_memories": active_count,
            "network_connections": self.activation_network.number_of_edges(),
            "emotional_state": self.current_emotional_state,
            "active_archetypes": self.active_archetypes,
            "zone_distribution": zone_distribution,
            "drift_distribution": drift_distribution,
            "cache_size": len(self._embeddings_cache)
        }

    def add_memory_connection(self, key1: str, key2: str, weight: float = 1.0):
        """Add connection between memories."""
        if key1 in self.activation_network and key2 in self.activation_network:
            self.activation_network.add_edge(key1, key2, weight=weight)

    def decay_memory_activation(self):
        """Apply decay to all memory activations."""
        with self._lock:
            for key in list(self.activation_levels.keys()):
                self.activation_levels[key] *= (1 - self.decay_rate)
                if self.activation_levels[key] < self.activation_threshold:
                    self.activation_levels[key] = 0.0

# Android Bridge Interface
class MemorySystemBridge:
    """
    Bridge interface for Kotlin Android integration via Chaquopy.
    """
    
    def __init__(self):
        self.memory_system = EnhancedHybridMemorySystem()
        self.logger = logging.getLogger(__name__)
    
    def initialize_system(self, 
                         max_memory_size: int = 1000,
                         model_name: str = 'paraphrase-MiniLM-L6-v2') -> str:
        """Initialize memory system with Android-specific settings."""
        try:
            self.memory_system = EnhancedHybridMemorySystem(
                max_memory_size=max_memory_size,
                model_name=model_name
            )
            return json.dumps({"success": True, "message": "Memory system initialized"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def store_memory_json(self, memory_json: str) -> str:
        """Store memory from JSON input (Android-friendly interface)."""
        try:
            data = json.loads(memory_json)
            result = self.memory_system.store_memory(
                key=data["key"],
                content=data["content"],
                zone=data.get("zone", "consciousness"),
                drift=data.get("drift", "neutral"),
                emotional_resonance=data.get("emotional_resonance", 0.5),
                archetypal_signatures=data.get("archetypal_signatures", []),
                importance=data.get("importance", 1.0)
            )
            return json.dumps({"success": result})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def retrieve_memories_json(self, query_json: str) -> str:
        """Retrieve memories with JSON interface."""
        try:
            data = json.loads(query_json)
            result = self.memory_system.retrieve_by_poetic_query(
                query=data["query"],
                zone_filter=data.get("zone_filter"),
                drift_filter=data.get("drift_filter"),
                max_results=data.get("max_results", 5)
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def activate_memory_json(self, activation_json: str) -> str:
        """Activate memory with JSON interface."""
        try:
            data = json.loads(activation_json)
            result = self.memory_system.activate_memory_with_poetic_resonance(
                key=data["key"],
                initial_activation=data.get("initial_activation", 1.0),
                emotional_context=data.get("emotional_context"),
                archetypal_context=data.get("archetypal_context")
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def generate_reflection_json(self, context_json: str = "{}") -> str:
        """Generate poetic reflection with JSON interface."""
        try:
            data = json.loads(context_json) if context_json != "{}" else {}
            result = self.memory_system.generate_poetic_reflection(
                context=data.get("context")
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def shift_consciousness_json(self, shift_json: str) -> str:
        """Shift consciousness state with JSON interface."""
        try:
            data = json.loads(shift_json)
            result = self.memory_system.shift_consciousness(
                new_emotional_state=data["emotional_state"],
                new_archetypes=data.get("archetypes")
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def get_stats_json(self) -> str:
        """Get system statistics as JSON."""
        try:
            stats = self.memory_system.get_system_stats()
            return json.dumps(stats, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def save_system_json(self, filepath: str) -> str:
        """Save system to file."""
        try:
            result = self.memory_system.save_to_file(filepath)
            return json.dumps({"success": result})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def load_system_json(self, filepath: str) -> str:
        """Load system from file."""
        try:
            result = self.memory_system.load_from_file(filepath)
            return json.dumps({"success": result})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

# Global bridge instance for Chaquopy
memory_bridge = MemorySystemBridge()

# Convenience functions for direct Chaquopy access
def store_memory(key: str, content: str, zone: str = "consciousness", 
                drift: str = "neutral", emotional_resonance: float = 0.5) -> str:
    """Direct function for Chaquopy integration."""
    memory_data = {
        "key": key,
        "content": content,
        "zone": zone,
        "drift": drift,
        "emotional_resonance": emotional_resonance
    }
    return memory_bridge.store_memory_json(json.dumps(memory_data))

def retrieve_memories(query: str, max_results: int = 5) -> str:
    """Direct function for Chaquopy integration."""
    query_data = {"query": query, "max_results": max_results}
    return memory_bridge.retrieve_memories_json(json.dumps(query_data))

def generate_reflection() -> str:
    """Direct function for Chaquopy integration."""
    return memory_bridge.generate_reflection_json()

def get_system_stats() -> str:
    """Direct function for Chaquopy integration."""
    return memory_bridge.get_stats_json()
