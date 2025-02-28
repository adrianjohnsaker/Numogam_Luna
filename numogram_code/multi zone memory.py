import json

class MultiZoneMemory:
    def __init__(self, memory_file="memory.json"):
        self.memory_file = memory_file
        self.memory = self.load_memory()

    def load_memory(self):
        try:
            with open(self.memory_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=4)

    def update_memory(self, user_id, zone, info):
        if user_id not in self.memory:
            self.memory[user_id] = {}
        self.memory[user_id][zone] = info
        self.save_memory()

    def retrieve_memory(self, user_id, zone):
        return self.memory.get(user_id, {}).get(zone, "No relevant memory found.")

# Example usage
memory_manager = MultiZoneMemory()
memory_manager.update_memory("user123", "Zone 4", "Discussed recursive learning techniques.")
print(memory_manager.retrieve_memory("user123", "Zone 4"))
