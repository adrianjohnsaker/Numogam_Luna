import networkx as nx
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class HybridMemorySystem:
    def __init__(self, decay_rate=0.05, activation_threshold=0.1):
        """
        Initializes Hybrid Memory System integrating Memory, Spreading Activation, and Reasoning.
        """
        self.memory_store = {}  # Dictionary-based memory storage
        self.activation_network = nx.DiGraph()  # Directed graph for concept relationships
        self.activation_levels = {}  # Spreading activation levels
        self.decay_rate = decay_rate  # Activation decay rate
        self.activation_threshold = activation_threshold  # Min activation level
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Embeddings for semantic similarity

    def store_memory(self, key: str, value: str, relevance: float = 1.0):
        """
        Stores memory and links it to spreading activation.
        """
        self.memory_store[key] = {"value": value, "relevance": relevance}
        self.activation_network.add_node(key)  # Add to activation graph
        self.activation_levels[key] = relevance  # Set initial activation level

    def add_memory_connection(self, key1, key2, weight=1.0):
        """
        Links memories logically for inference-based recall.
        """
        self.activation_network.add_edge(key1, key2, weight=weight)

    def activate_memory(self, key, initial_activation=1.0):
        """
        Activates a memory and spreads activation to related nodes.
        """
        if key in self.activation_network:
            self.activation_levels[key] += initial_activation
            for neighbor in self.activation_network.neighbors(key):
                edge_weight = self.activation_network[key][neighbor]['weight']
                self.activation_levels[neighbor] += initial_activation * edge_weight

    def decay_memory_activation(self):
        """
        Applies decay to activated memories.
        """
        for key in list(self.activation_levels.keys()):
            self.activation_levels[key] *= (1 - self.decay_rate)
            if self.activation_levels[key] < self.activation_threshold:
                self.activation_levels[key] = 0.0  # Reset activation if below threshold

    def retrieve_memory(self, query: str):
        """
        Retrieves the most relevant memories based on spreading activation and semantic similarity.
        """
        activated_memories = sorted(self.activation_levels.items(), key=lambda x: x[1], reverse=True)
        top_memories = [mem for mem, act in activated_memories if act > self.activation_threshold]
        
        best_match = self._retrieve_best_match(query, top_memories)
        return self.memory_store.get(best_match, {}).get("value", "No relevant memory found.")

    def infer_new_memories(self, input_memories):
        """
        Uses symbolic reasoning to infer new knowledge.
        """
        conclusions = set()
        for memory in input_memories:
            if memory in self.activation_network:
                conclusions.update(self.activation_network.successors(memory))
        return list(conclusions)

    def _retrieve_best_match(self, query, concepts):
        """
        Finds the best matching stored memory using semantic similarity.
        """
        if not concepts:
            return "No relevant concept found"
        
        similarities = {concept: self._compute_similarity(query, concept) for concept in concepts}
        return max(similarities, key=similarities.get)

    def _compute_similarity(self, concept1, concept2):
        """
        Computes semantic similarity between two stored concepts.
        """
        emb1 = self.model.encode([concept1])
        emb2 = self.model.encode([concept2])
        return cosine_similarity(emb1, emb2)[0][0]

    def to_json(self):
        """
        Converts memory system state to JSON.
        """
        return json.dumps({
            "memory_store": self.memory_store,
            "activation_levels": self.activation_levels
        })

# Initialize the Hybrid Memory System
memory_system = HybridMemorySystem()

# Store Memories
memory_system.store_memory("AI", "Artificial Intelligence involves machine learning.", relevance=0.9)
memory_system.store_memory("Neural Networks", "AI models using interconnected nodes.", relevance=0.8)
memory_system.store_memory("Philosophy", "Study of fundamental existence concepts.", relevance=0.7)
memory_system.store_memory("Ethics", "Study of moral principles guiding behavior.", relevance=0.6)

# Define Logical Memory Connections
memory_system.add_memory_connection("AI", "Neural Networks", 0.8)
memory_system.add_memory_connection("AI", "Symbolic Reasoning", 0.7)
memory_system.add_memory_connection("Philosophy", "Ethics", 0.6)

# Activate Memory & Simulate Retrieval
memory_system.activate_memory("AI", initial_activation=1.0)
memory_system.decay_memory_activation()

# Retrieve Most Relevant Memory
retrieved_memory = memory_system.retrieve_memory("Deep learning models")
inferred_knowledge = memory_system.infer_new_memories(["AI", "Philosophy"])

# Display Results
print("Retrieved Memory:", retrieved_memory)
print("Inferred Knowledge:", inferred_knowledge)
print("Memory System State:", memory_system.to_json())
