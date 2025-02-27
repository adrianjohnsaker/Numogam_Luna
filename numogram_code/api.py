from fastapi import FastAPI
from pydantic import BaseModel
from numogram_algorithm import zone_transition, belief_model

app = FastAPI()

class TransitionRequest(BaseModel):
    user_id: str
    current_zone: str
    feedback: float  # User provides feedback (positive/negative)

user_memory = {}

@app.post("/numogram/transition")
async def transition(request: TransitionRequest):
    current_zone = request.current_zone

    if request.user_id not in user_memory:
        user_memory[request.user_id] = {"zone": current_zone, "feedback": request.feedback}
    else:
        user_memory[request.user_id]["feedback"] = request.feedback

    next_zone, description = zone_transition(current_zone, request.feedback)

    user_memory[request.user_id]["zone"] = next_zone

    return {
        "next_zone": next_zone,
        "zone_description": description,
        "updated_beliefs": belief_model
    }
