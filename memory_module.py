import json
import time
from collections import defaultdict

class MemoryModule:
    def __init__(self, memory_file="memory_data.json"):
        self.memory_file = memory_file
        self.load_memory()

    def load_memory(self):
        """Load memory from file or initialize fresh."""
        try:
            with open(self.memory_file, "r") as f:
                self.memory = json.load(f)
        except FileNotFoundError:
            self.memory = defaultdict(lambda: {"interactions": [], "timestamps": []})

    def save_memory(self):
        """Save memory to file for persistence."""
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=4)

    def store_interaction(self, user_id, zone, message):
        """Save user interaction with time-sensitive decay logic."""
        timestamp = time.time()
        if user_id not in self.memory:
            self.memory[user_id] = {"interactions": [], "timestamps": []}
        self.memory[user_id]["interactions"].append({"zone": zone, "message": message, "time": timestamp})
        self.memory[user_id]["timestamps"].append(timestamp)
        self.save_memory()

    def retrieve_context(self, user_id, current_zone, decay_factor=0.85):
        """Retrieve past interactions with priority on recent, relevant ones."""
        if user_id not in self.memory:
            return []

        relevant_interactions = []
        for interaction in self.memory[user_id]["interactions"]:
            if interaction["zone"] == current_zone or current_zone in interaction["zone"]:
                time_passed = time.time() - interaction["time"]
                weight = decay_factor ** (time_passed / 60)
                relevant_interactions.append((interaction["message"], weight))

        # Sort by relevance weighting
        relevant_interactions.sort(key=lambda x: x[1], reverse=True)
        return [msg for msg, _ in relevant_interactions[:5]]  # Limit recall depth

memory_module = MemoryModule()
