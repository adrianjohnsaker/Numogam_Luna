from fastapi import FastAPI
from pydantic import BaseModel
from meta_reflection import meta_reflection
from memory_module import memory_module

app = FastAPI()

class UserRequest(BaseModel):
    user_id: str
    current_zone: str
    input_text: str

@app.post("/respond")
async def respond(request: UserRequest):
    user_id = request.user_id
    current_zone = request.current_zone
    user_input = request.input_text

    # Retrieve contextual memory
    past_context = memory_module.retrieve_context(user_id, current_zone)

    # Base response logic
    base_response = f"That's an interesting thought about '{user_input}'. Let's explore that further."

    # Apply memory-based self-referencing if context exists
    if past_context:
        memory_summary = "I recall our discussions: " + "; ".join(past_context)
        base_response = f"{memory_summary}. Regarding '{user_input}', let's dive deeper."

    # Apply meta-reflection
    refined_response = meta_reflection(user_id, user_input, base_response)

    # Store refined response in memory
    memory_module.store_interaction(user_id, current_zone, refined_response)

    return {"response": refined_response}
