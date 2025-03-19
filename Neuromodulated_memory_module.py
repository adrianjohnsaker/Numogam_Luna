import numpy as np
import json
import time
from scipy.special import expit
from typing import Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Query

class MemoryData:
    def __init__(self, key: str, data: str, salience: float, uncertainty: float):
        self.key = key
        self.data = data
        self.salience = max(0.0, min(salience, 1.0))
        self.uncertainty = max(0.0, min(uncertainty, 1.0))

class NeuromodulatedMemory:
    def __init__(self, decay_rate=0.01, gain_modulation=1.0, retention_threshold=0.05):
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.decay_rate = decay_rate
        self.gain_modulation = gain_modulation
        self.retention_threshold = retention_threshold
        self.last_decay_time = time.time()

    def encode_memory(self, memory: MemoryData) -> str:
        weight = self.compute_memory_weight(memory.salience, memory.uncertainty)
        self.memory_store[memory.key] = {
            "data": memory.data,
            "weight": weight,
            "timestamp": time.time(),
            "access_count": 0,
            "salience": memory.salience,
            "uncertainty": memory.uncertainty
        }
        return f"Memory encoded with weight {weight:.4f}"

    def compute_memory_weight(self, salience: float, uncertainty: float) -> float:
        modulation = 4 * self.gain_modulation * (uncertainty - 0.5)
        weight = salience / (1 + np.exp(-modulation))
        return max(0.01, min(weight, 1.0))

    def decay_memories(self) -> Dict[str, int]:
        current_time = time.time()
        time_elapsed = current_time - self.last_decay_time

        if time_elapsed < 30:
            return {"status": "skipped", "time_since_last": time_elapsed}

        self.last_decay_time = current_time
        removed_count = 0

        for key in list(self.memory_store.keys()):
            memory = self.memory_store[key]
            time_factor = (current_time - memory["timestamp"]) / 3600
            adaptive_decay = self.decay_rate * (1 - 0.5 * memory["salience"])
            memory["weight"] *= np.exp(-adaptive_decay * time_factor)

            if memory["weight"] < self.retention_threshold:
                del self.memory_store[key]
                removed_count += 1

        return {"status": "completed", "memories_removed": removed_count}

    def export_memory(self) -> str:
        return json.dumps(self.memory_store)

# Initialize FastAPI app
app = FastAPI()

memory_module = NeuromodulatedMemory()

@app.post("/store_memory/")
async def store_memory(memory: MemoryData):
    result = memory_module.encode_memory(memory)
    return {"success": True, "message": result}

@app.get("/export_memory/")
async def export_memory():
    exported_data = memory_module.export_memory()
    return {"success": True, "memory_data": exported_data}
