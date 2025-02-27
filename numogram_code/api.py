from fastapi import FastAPI
from pydantic import BaseModel
import json
import random
import os

# Load zone data
with open("numogram_code/zones.json") as f:
    ZONE_DATA = json.load(f)["zones"]

# Bayesian transition probabilities
transition_probabilities = {
    "1": {"2": 0.6, "4": 0.4},
    "2": {"3": 0.7, "6": 0.3},
    "3": {"1": 0.5, "9": 0.5},
}

app = FastAPI()

class TransitionRequest(BaseModel):
    user_id: str
    current_zone: str
    feedback: float  # Reinforcement learning feedback

# Persistent memory file
MEMORY_FILE = "numogram_code/user_memory.json"

# Load existing memory if available
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        user_memory = json.load(f)
else:
    user_memory = {}

@app.post("/numogram/transition")
async def transition(request: TransitionRequest):
    current_zone = request.current_zone

    # Update user memory
    if request.user_id not in user_memory:
        user_memory[request.user_id] = {"zone": current_zone, "feedback": request.feedback}
    else:
        user_memory[request.user_id]["zone"] = current_zone
        user_memory[request.user_id]["feedback"] = request.feedback

    # Bayesian transition logic
    if current_zone in transition_probabilities:
        next_zone = random.choices(
            list(transition_probabilities[current_zone].keys()),
            weights=list(transition_probabilities[current_zone].values())
        )[0]
    else:
        next_zone = "1"  # Default fallback

    user_memory[request.user_id]["zone"] = next_zone

    # Save memory persistently
    with open(MEMORY_FILE, "w") as f:
        json.dump(user_memory, f)

    return {
        "next_zone": next_zone,
        "zone_description": ZONE_DATA.get(next_zone, {}),
        "memory_status": user_memory[request.user_id]
    }
