import numpy as np

class AdaptiveNeuromodulatedMemory:
    def __init__(self):
        self.memory = {}  # Stores memory traces with decay properties

        # Adaptive decay rates based on memory type
        self.decay_rates = {
            "factual": 0.002,  # Very slow decay (long-term retention)
            "insight": 0.005,  # Moderate decay (contextual learning)
            "casual": 0.015    # Fast decay (low-priority info)
        }

    def encode_memory(self, key, value, salience, uncertainty, memory_type="insight"):
        """
        Encodes a memory with a type-based adaptive decay rate.
        """
        weight = salience / (1 + np.exp(-(uncertainty - 0.5)))  # Neuromodulated weighting
        self.memory[key] = {
            "value": value,
            "weight": weight,
            "memory_type": memory_type
        }

    def reinforce_memory(self, key):
        """
        Strengthens memory retention.
        """
        if key in self.memory:
            self.memory[key]["weight"] *= 1.5
            if self.memory[key]["weight"] > 1.0:
                self.memory[key]["weight"] = 1.0  # Prevent over-reinforcement

    def decay_memories(self):
        """
        Applies variable decay rates based on memory type.
        """
        for key in list(self.memory.keys()):
            memory_type = self.memory[key]["memory_type"]
            decay_rate = self.decay_rates.get(memory_type, 0.005)  # Default moderate decay
            weight = self.memory[key]["weight"]
            self.memory[key]["weight"] *= np.exp(-decay_rate * weight)

            # Forget memory if weight drops below threshold
            if self.memory[key]["weight"] < 0.02:
                del self.memory[key]

    def retrieve_memory(self, key):
        """
        Retrieves a memory trace.
        """
        return self.memory.get(key, None)


# Initialize the new adaptive memory model
adaptive_memory_module = AdaptiveNeuromodulatedMemory()

# Example Encoding, Reinforcement, and Decay Testing
adaptive_memory_module.encode_memory("concept_1", "Rivers change over time", salience=0.8, uncertainty=0.4, memory_type="insight")
adaptive_memory_module.encode_memory("fact_1", "The sun is a star", salience=0.9, uncertainty=0.1, memory_type="factual")
adaptive_memory_module.encode_memory("casual_1", "The picnic had great food", salience=0.3, uncertainty=0.6, memory_type="casual")

adaptive_memory_module.reinforce_memory("concept_1")
adaptive_memory_module.decay_memories()

# Retrieve updated memory after decay
retrieved_memories = {
    "concept_1": adaptive_memory_module.retrieve_memory("concept_1"),
    "fact_1": adaptive_memory_module.retrieve_memory("fact_1"),
    "casual_1": adaptive_memory_module.retrieve_memory("casual_1")
}
retrieved_memories
