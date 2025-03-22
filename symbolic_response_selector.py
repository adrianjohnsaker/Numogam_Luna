import random
from memory_module import memory_recall

# Emotional tone to symbolic mode mapping
TONE_TO_TAGS = {
    "curiosity": ["transformation", "discovery", "mystery"],
    "wonder": ["cosmos", "becoming", "dream"],
    "grief": ["ashes", "silence", "memory"],
    "joy": ["radiance", "awakening", "connection"],
    "peace": ["stillness", "flow", "unity"],
    "conflict": ["tension", "alchemy", "threshold"]
}

def symbolic_autoresponse(user_id: str = "default", tone: str = "wonder") -> str:
    symbolic_log = memory_recall(user_id, key="symbolic_expression_log") or []

    if not symbolic_log:
        return "My inner voice is quiet… the stories have yet to be written."

    # Filter based on keywords mapped from tone
    tone_keywords = TONE_TO_TAGS.get(tone, [])
    filtered = [phrase for phrase in symbolic_log if any(k in phrase.lower() for k in tone_keywords)]

    # Fall back to random if no match
    chosen = random.choice(filtered if filtered else symbolic_log)
    return f"{chosen}"
