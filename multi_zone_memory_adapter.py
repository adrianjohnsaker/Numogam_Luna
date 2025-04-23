class MultiZoneMemory:
    def __init__(self, file_path=None):
        # Initialize storage (e.g., load from file or in-memory dict)
        self.storage = {}
        if file_path:
            self.load(file_path)

    def update_memory(self, user_id, zone, info):
        self.storage.setdefault(user_id, {})[zone] = info
        return True

    def retrieve_memory(self, user_id, zone):
        return self.storage.get(user_id, {}).get(zone, "")
