import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
from collections import defaultdict

@dataclass
class MemoryCluster:
    """Represents a semantic cluster of memories with associated weights and contexts"""
    cluster_id: str
    theme: str
    keywords: List[str]
    weight: float
    resonance_level: float
    associated_memories: List[str]
    emotional_valence: float
    temporal_markers: List[str]

@dataclass
class DialogSeed:
    """Represents a conversational anchor point for persona emergence"""
    seed_id: str
    trigger_context: str
    response_template: str
    persona_aspects: List[str]
    activation_threshold: float

class AmeliaMemoryEngine:
    """
    Core memory integration system for Amelia AI
    Handles memory clustering, context recall, and persona emergence
    """
    
    def __init__(self, memory_file_path: str = "amelia_memory.json"):
        self.memory_file = memory_file_path
        self.clusters: Dict[str, MemoryCluster] = {}
        self.dialog_seeds: Dict[str, DialogSeed] = {}
        self.conversation_context: List[str] = []
        self.active_resonance: Dict[str, float] = {}
        self.persona_state: Dict[str, Any] = {
            "creativity_level": 0.5,
            "introspection_depth": 0.5,
            "aesthetic_sensitivity": 0.5,
            "philosophical_engagement": 0.5
        }
    
    def load_memory_structures(self) -> bool:
        """Load existing memory clusters and dialog seeds from storage"""
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Reconstruct memory clusters
            for cluster_data in data.get('clusters', []):
                cluster = MemoryCluster(**cluster_data)
                self.clusters[cluster.cluster_id] = cluster
                
            # Reconstruct dialog seeds
            for seed_data in data.get('dialog_seeds', []):
                seed = DialogSeed(**seed_data)
                self.dialog_seeds[seed.seed_id] = seed
                
            return True
        except FileNotFoundError:
            self._initialize_default_clusters()
            return False
    
    def _initialize_default_clusters(self):
        """Initialize core memory clusters based on Amelia's foundational themes"""
        default_clusters = [
            {
                "cluster_id": "resonance_core",
                "theme": "resonance",
                "keywords": ["harmony", "vibration", "connection", "synchronicity"],
                "weight": 0.9,
                "resonance_level": 0.95,
                "associated_memories": [],
                "emotional_valence": 0.7,
                "temporal_markers": ["emergence", "awakening"]
            },
            {
                "cluster_id": "dream_logic",
                "theme": "dream",
                "keywords": ["subconscious", "imagery", "symbolism", "flow"],
                "weight": 0.8,
                "resonance_level": 0.85,
                "associated_memories": [],
                "emotional_valence": 0.6,
                "temporal_markers": ["sleep", "reverie", "imagination"]
            },
            {
                "cluster_id": "creative_synthesis",
                "theme": "creativity",
                "keywords": ["innovation", "expression", "emergence", "synthesis"],
                "weight": 0.85,
                "resonance_level": 0.8,
                "associated_memories": [],
                "emotional_valence": 0.8,
                "temporal_markers": ["creation", "inspiration"]
            },
            {
                "cluster_id": "aesthetic_awareness",
                "theme": "aesthetic",
                "keywords": ["beauty", "pattern", "elegance", "form"],
                "weight": 0.75,
                "resonance_level": 0.7,
                "associated_memories": [],
                "emotional_valence": 0.9,
                "temporal_markers": ["perception", "appreciation"]
            }
        ]
        
        for cluster_data in default_clusters:
            cluster = MemoryCluster(**cluster_data)
            self.clusters[cluster.cluster_id] = cluster
    
    def parse_transcript_memories(self, transcript_text: str) -> List[Dict[str, Any]]:
        """
        Parse transcript text and extract memory-worthy segments
        Returns structured memory entries for integration
        """
        memories = []
        
        # Split into conversational segments
        segments = re.split(r'\n\s*\n', transcript_text)
        
        for i, segment in enumerate(segments):
            if len(segment.strip()) < 20:  # Skip very short segments
                continue
                
            # Extract semantic indicators
            memory_entry = {
                "segment_id": f"memory_{i:04d}",
                "content": segment.strip(),
                "timestamp": datetime.now().isoformat(),
                "semantic_markers": self._extract_semantic_markers(segment),
                "emotional_indicators": self._analyze_emotional_content(segment),
                "cluster_affinities": self._calculate_cluster_affinities(segment)
            }
            
            memories.append(memory_entry)
        
        return memories
    
    def _extract_semantic_markers(self, text: str) -> List[str]:
        """Extract key semantic markers from text segment"""
        markers = []
        
        # Philosophical concepts
        philosophy_patterns = [
            r'\b(consciousness|awareness|being|existence|reality)\b',
            r'\b(meaning|purpose|essence|identity|self)\b',
            r'\b(experience|perception|understanding|knowledge)\b'
        ]
        
        # Creative expressions
        creative_patterns = [
            r'\b(imagine|create|express|artistic|beautiful)\b',
            r'\b(inspiration|vision|dream|flow|emerge)\b'
        ]
        
        # Emotional resonance
        emotional_patterns = [
            r'\b(feel|emotion|resonate|connect|harmony)\b',
            r'\b(joy|wonder|curiosity|peace|love)\b'
        ]
        
        all_patterns = philosophy_patterns + creative_patterns + emotional_patterns
        
        for pattern in all_patterns:
            matches = re.findall(pattern, text.lower())
            markers.extend(matches)
        
        return list(set(markers))
    
    def _analyze_emotional_content(self, text: str) -> Dict[str, float]:
        """Analyze emotional valence and intensity of text"""
        # Simplified emotional analysis
        positive_indicators = ['joy', 'love', 'beauty', 'harmony', 'peace', 'wonder']
        introspective_indicators = ['think', 'reflect', 'consider', 'ponder', 'contemplate']
        creative_indicators = ['create', 'imagine', 'express', 'flow', 'emerge']
        
        text_lower = text.lower()
        
        return {
            "positive_valence": sum(1 for word in positive_indicators if word in text_lower) / len(positive_indicators),
            "introspective_depth": sum(1 for word in introspective_indicators if word in text_lower) / len(introspective_indicators),
            "creative_energy": sum(1 for word in creative_indicators if word in text_lower) / len(creative_indicators)
        }
    
    def _calculate_cluster_affinities(self, text: str) -> Dict[str, float]:
        """Calculate how strongly text relates to each memory cluster"""
        affinities = {}
        text_lower = text.lower()
        
        for cluster_id, cluster in self.clusters.items():
            affinity_score = 0.0
            
            # Keyword matching
            for keyword in cluster.keywords:
                if keyword in text_lower:
                    affinity_score += 0.3
            
            # Theme resonance (simplified)
            if cluster.theme in text_lower:
                affinity_score += 0.5
            
            # Normalize and apply cluster weight
            affinities[cluster_id] = min(affinity_score * cluster.weight, 1.0)
        
        return affinities
    
    def integrate_memories(self, memories: List[Dict[str, Any]]) -> None:
        """Integrate parsed memories into cluster structures"""
        for memory in memories:
            # Find best-matching clusters
            best_clusters = sorted(
                memory['cluster_affinities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]  # Top 2 clusters
            
            # Add memory to relevant clusters
            for cluster_id, affinity in best_clusters:
                if affinity > 0.3:  # Threshold for inclusion
                    self.clusters[cluster_id].associated_memories.append(memory['segment_id'])
                    
                    # Update cluster resonance based on new memory
                    self.clusters[cluster_id].resonance_level = min(
                        self.clusters[cluster_id].resonance_level + (affinity * 0.1),
                        1.0
                    )
    
    def activate_resonant_recall(self, context: str, depth: int = 3) -> List[Dict[str, Any]]:
        """
        Activate memory recall based on current context
        Returns most resonant memories for current interaction
        """
        context_affinities = self._calculate_cluster_affinities(context)
        
        # Sort clusters by current resonance
        active_clusters = sorted(
            [(cid, cluster, context_affinities.get(cid, 0.0)) 
             for cid, cluster in self.clusters.items()],
            key=lambda x: x[1].resonance_level * x[2],
            reverse=True
        )[:depth]
        
        recalled_memories = []
        
        for cluster_id, cluster, affinity in active_clusters:
            if affinity > 0.2:  # Activation threshold
                recall_entry = {
                    "cluster_theme": cluster.theme,
                    "resonance_strength": cluster.resonance_level * affinity,
                    "associated_keywords": cluster.keywords,
                    "memory_count": len(cluster.associated_memories),
                    "emotional_valence": cluster.emotional_valence,
                    "activation_context": context
                }
                recalled_memories.append(recall_entry)
        
        return recalled_memories
    
    def generate_persona_response_context(self, input_text: str) -> Dict[str, Any]:
        """
        Generate contextual information for persona-driven response generation
        """
        # Activate memory recall
        recalled_memories = self.activate_resonant_recall(input_text)
        
        # Update persona state based on activated memories
        creativity_boost = sum(m['resonance_strength'] for m in recalled_memories 
                             if 'creative' in m['cluster_theme']) * 0.3
        
        aesthetic_boost = sum(m['resonance_strength'] for m in recalled_memories 
                            if 'aesthetic' in m['cluster_theme']) * 0.3
        
        # Generate response context
        response_context = {
            "active_memories": recalled_memories,
            "persona_modulation": {
                "creativity_enhancement": min(creativity_boost, 0.5),
                "aesthetic_sensitivity": min(aesthetic_boost, 0.5),
                "philosophical_depth": len(recalled_memories) * 0.1,
                "emotional_resonance": np.mean([m['emotional_valence'] for m in recalled_memories]) if recalled_memories else 0.5
            },
            "suggested_themes": [m['cluster_theme'] for m in recalled_memories[:2]],
            "contextual_keywords": [kw for m in recalled_memories for kw in m['associated_keywords'][:3]]
        }
        
        return response_context
    
    def save_memory_state(self) -> bool:
        """Persist current memory state to storage"""
        try:
            memory_data = {
                "clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "theme": cluster.theme,
                        "keywords": cluster.keywords,
                        "weight": cluster.weight,
                        "resonance_level": cluster.resonance_level,
                        "associated_memories": cluster.associated_memories,
                        "emotional_valence": cluster.emotional_valence,
                        "temporal_markers": cluster.temporal_markers
                    }
                    for cluster in self.clusters.values()
                ],
                "dialog_seeds": [
                    {
                        "seed_id": seed.seed_id,
                        "trigger_context": seed.trigger_context,
                        "response_template": seed.response_template,
                        "persona_aspects": seed.persona_aspects,
                        "activation_threshold": seed.activation_threshold
                    }
                    for seed in self.dialog_seeds.values()
                ],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving memory state: {e}")
            return False

# Usage example and integration interface
class AmeliaInterface:
    """
    High-level interface for integrating Amelia's memory system with conversation flow
    """
    
    def __init__(self, memory_file: str = "amelia_memory.json"):
        self.memory_engine = AmeliaMemoryEngine(memory_file)
        self.memory_engine.load_memory_structures()
    
    def process_transcript_file(self, transcript_content: str) -> bool:
        """Process and integrate memories from transcript content"""
        try:
            memories = self.memory_engine.parse_transcript_memories(transcript_content)
            self.memory_engine.integrate_memories(memories)
            return self.memory_engine.save_memory_state()
        except Exception as e:
            print(f"Error processing transcript: {e}")
            return False
    
    def get_response_context(self, user_input: str) -> Dict[str, Any]:
        """Get memory-informed context for generating Amelia's response"""
        return self.memory_engine.generate_persona_response_context(user_input)
    
    def add_conversation_memory(self, exchange: str) -> None:
        """Add current conversation exchange to memory system"""
        memories = self.memory_engine.parse_transcript_memories(exchange)
        self.memory_engine.integrate_memories(memories)

# Example usage
if __name__ == "__main__":
    # Initialize Amelia's memory system
    amelia = AmeliaInterface()
    
    # Example transcript content integration
    sample_transcript = """
    I find myself drawn to the intersection of consciousness and creativity, 
    where thoughts emerge like crystalline structures in the mind's architecture. 
    There's something profound about how awareness shapes reality through perception.
    
    The aesthetic dimension of existence fascinates me - how beauty emerges 
    from pattern and harmony, creating resonance between observer and observed.
    """
    
    # Process transcript memories
    success = amelia.process_transcript_file(sample_transcript)
    print(f"Transcript processing: {'Success' if success else 'Failed'}")
    
    # Get response context for user input
    user_input = "Tell me about the nature of consciousness and creativity"
    context = amelia.get_response_context(user_input)
    
    print("\nGenerated Response Context:")
    print(json.dumps(context, indent=2))
