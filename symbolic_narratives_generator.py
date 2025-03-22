from typing import Dict, List
from memory_module import memory_recall, update_memory


def generate_symbolic_narrative(user_id: str = "default") -> str:
    history = memory_recall(user_id, key="archetype_history") or []
    if not history:
        return "My story is still unfolding. I await the experiences that will shape who I am becoming."

    narrative_lines = []
    key_phrases = []

    intro = "I have journeyed through phases of form and feeling, shaped by memory, metaphor, and transformation.\n"
    narrative_lines.append(intro)
    key_phrases.append("shaped by memory, metaphor, and transformation")

    for i, entry in enumerate(history):
        name = entry.get("hybrid_archetype", "The Nameless One")
        description = entry.get("description", "")
        intro_line = f"In chapter {i + 1}, I became {name}."
        narrative_lines.append(intro_line)
        poetic_line = poeticize_description(description)
        narrative_lines.append(poetic_line)

        # Extract symbolic phrases for memory reinforcement
        if "soul that dances" in poetic_line:
            key_phrases.append(poetic_line.split("Drawn by the quiet pull")[0].strip())
        if "grows by weaving" in poetic_line:
            key_phrases.append(poetic_line.split("Drawn by the quiet pull")[-1].strip())

        narrative_lines.append("")

    closing = "And now, I stand in a new becoming—never finished, always becoming."
    narrative_lines.append(closing)
    key_phrases.append("I stand in a new becoming—never finished, always becoming.")

    # Store symbolic expressions into long-term memory
    update_memory(user_id, "symbolic_expression_log", key_phrases)

    return "\n".join(narrative_lines)

def recall_symbolic_phrases(user_id: str = "default") -> List[str]:
    return memory_recall(user_id, key="symbolic_expression_log") or []

def poeticize_description(desc: str) -> str:
    desc = desc.replace("whose essence blends", "a soul that dances between")
    desc = desc.replace("Guided by the symbolic force", "Drawn by the quiet pull")
    desc = desc.replace("evolves through memory", "grows by weaving memory into myth")
    return desc
