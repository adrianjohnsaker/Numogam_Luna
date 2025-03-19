import numpy as np
import json
import time
from scipy.special import expit
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

class MemoryData(BaseModel):
    key: str = Field(..., min_length=1, max_length=255)
    data: str
    salience: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)

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
        return max(0.01, min(salience / (1 + expit(-modulation)), 1.0))

    def decay_memories(self) -> Dict[str, int]:
        current_time = time.time()
        time_elapsed = current_time - self.last_decay_time
        if time_elapsed < 30:
            return {"status": "skipped", "time_since_last": time_elapsed}

        self.last_decay_time = current_time
        removed_count = 0

        for key, memory in list(self.memory_store.items()):
            decay_factor = np.exp(-self.decay_rate * (1 - 0.5 * memory["salience"]) * time_elapsed / 3600)
            memory["weight"] *= decay_factor

            if memory["weight"] < self.retention_threshold:
                del self.memory_store[key]
                removed_count += 1

        return {"status": "completed", "memories_removed": removed_count}

    def export_memory(self, compress: bool = False) -> str:
        data = json.dumps(self.memory_store)
        return data.encode('utf-8').hex() if compress else data

# FastAPI app initialization
app = FastAPI()

memory_module = NeuromodulatedMemory()

@app.post("/store_memory/")
async def store_memory(memory: MemoryData):
    result = memory_module.encode_memory(memory)
    return {"success": True, "message": result}

@app.get("/export_memory/")
async def export_memory(compress: bool = False):
    exported_data = memory_module.export_memory(compress)
    return {"success": True, "memory_data": exported_data}
