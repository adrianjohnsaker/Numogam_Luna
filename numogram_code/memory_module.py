import json

# Memory storage simulation
user_memories = {}

def memory_recall(user_id, depth=3):
    """ Retrieve relevant memory with tiered weighting. """
    if user_id not in user_memories:
        return []
    
    memory = user_memories[user_id]
    short_term = memory.get("short_term", [])
    mid_term = memory.get("mid_term", [])
    long_term = memory.get("long_term", [])

    # Weighted selection (prioritizing mid & long-term context)
    return (short_term[:depth] + mid_term[:depth] + long_term[:depth])[-depth:]

def update_memory(user_id, key, value):
    """ Stores memory in short-term, mid-term, or long-term storage. """
    if user_id not in user_memories:
        user_memories[user_id] = {"short_term": [], "mid_term": [], "long_term": []}

    # Short-term memory (immediate context)
    user_memories[user_id]["short_term"].insert(0, value)
    if len(user_memories[user_id]["short_term"]) > 5:
        user_memories[user_id]["mid_term"].insert(0, user_memories[user_id]["short_term"].pop())

    # Mid-term transitions to long-term gradually
    if len(user_memories[user_id]["mid_term"]) > 5:
        user_memories[user_id]["long_term"].insert(0, user_memories[user_id]["mid_term"].pop())

    # Ensure long-term memory does not grow infinitely
    user_memories[user_id]["long_term"] = user_memories[user_id]["long_term"][:10]
