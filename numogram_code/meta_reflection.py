import random
import json
from memory_module import memory_recall, update_memory

def meta_reflection(user_id, user_input, current_response):
    """
    Enhances Amelia's response using structured meta-reflection.
    - Uses weighted recall from memory.
    - Limits recursion depth to avoid simplification.
    - Introduces delayed recursion.
    """
    
    # Retrieve memory
    past_responses = memory_recall(user_id, depth=3)  # Get last 3 relevant memories
    reflection_depth = 2  # Limit recursion to 2 layers

    # Weighing past responses for reflection
    if past_responses:
        weighted_past = random.choices(past_responses, k=min(len(past_responses), 2))
        reflections = [f"In our past discussions, you mentioned '{p}'. Let's expand on that." for p in weighted_past]
    else:
        reflections = []

    # Controlled Recursive Thought Loops
    for _ in range(reflection_depth):
        current_response = f"{current_response} {random.choice(reflections)}" if reflections else current_response

    # Delayed recursion setup (Amelia revisits this later)
    update_memory(user_id, "delayed_reflection", current_response)

    return current_response
