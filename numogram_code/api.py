from fastapi import FastAPI
from pydantic import BaseModel
from meta_reflection import meta_reflection
from memory_module import update_memory

app = FastAPI()

class UserRequest(BaseModel):
    user_id: str
    input_text: str

@app.post("/respond")
async def respond(request: UserRequest):
    user_id = request.user_id
    user_input = request.input_text

    # Generate Amelia's response (base response logic)
    base_response = f"That's an interesting thought about '{user_input}'. Let's explore that further."

    # Apply meta-reflection
    refined_response = meta_reflection(user_id, user_input, base_response)

    # Update memory
    update_memory(user_id, "last_response", refined_response)

    return {"response": refined_response}
