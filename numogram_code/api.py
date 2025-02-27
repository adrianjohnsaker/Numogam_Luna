from fastapi import FastAPI
from pydantic import BaseModel
from numogram_algorithm import zone_transition

app = FastAPI()

class TransitionRequest(BaseModel):
    current_zone: str
    input: str  # Placeholder for possible future input-based transitions

@app.post("/numogram/transition")
async def transition(request: TransitionRequest):
    result = zone_transition(request.current_zone, request.input)
    return {"transition_result": result}
