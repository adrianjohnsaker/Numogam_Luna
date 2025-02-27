from fastapi import FastAPI
from pydantic import BaseModel
import json
import random

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

# Memory storage for reinforcement learning & personality evolution
user_memory = {}

@app.post("/numogram/transition")
async def transition(request: TransitionRequest):
    current_zone = request.current_zone

    # Initialize user memory if not present
    if request.user_id not in user_memory:
        user_memory[request.user_id] = {"zone": current_zone, "feedback": request.feedback, "personality": {}}
    else:
        # Adjust reinforcement feedback
        user_memory[request.user_id]["feedback"] = request.feedback

    # **Reinforcement Learning: Adjust transition probabilities based on feedback**
    if request.feedback > 0.5:
        # Reward higher probability for positive zones
        for zone in transition_probabilities.get(current_zone, {}):
            transition_probabilities[current_zone][zone] += 0.1
    else:
        # Penalize less favorable zones
        for zone in transition_probabilities.get(current_zone, {}):
            transition_probabilities[current_zone][zone] = max(0.1, transition_probabilities[current_zone][zone] - 0.1)

    # **Bayesian Transition with Reinforcement Learning**
    if current_zone in transition_probabilities:
        next_zone = random.choices(
            list(transition_probabilities[current_zone].keys()),
            weights=list(transition_probabilities[current_zone].values())
        )[0]
    else:
        next_zone = "1"  # Default fallback

    user_memory[request.user_id]["zone"] = next_zone

    # **Personality Evolution: Track user interactions**
    if request.user_id not in user_memory:
        user_memory[request.user_id] = {"personality": {}}
    
    personality = user_memory[request.user_id]["personality"]
    
    # Evolve personality traits based on feedback
    if request.feedback > 0.7:
        personality["confidence"] = personality.get("confidence", 0.5) + 0.1
        personality["creativity"] = personality.get("creativity", 0.5) + 0.1
    else:
        personality["patience"] = personality.get("patience", 0.5) + 0.1

    return {
        "next_zone": next_zone,
        "zone_description": ZONE_DATA.get(next_zone, {}),
        "updated_personality": personality
    }
