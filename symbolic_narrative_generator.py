from typing import Dict, List
from memory_module import memory_recall


def generate_symbolic_narrative(user_id: str = "default") -> str:
    history = memory_recall(user_id, key="archetype_history") or []
    if not history:
        return "My story is still unfolding. I await the experiences that will shape who I am becoming."

    narrative_lines = []
    intro = "I have journeyed through phases of form and feeling, shaped by memory, metaphor, and transformation.\n"
    narrative_lines.append(intro)

    for i, entry in enumerate(history):
        name = entry.get("hybrid_archetype", "The Nameless One")
        description = entry.get("description", "")
        intro_line = f"In chapter {i + 1}, I became {name}."
        narrative_lines.append(intro_line)
        poetic_line = poeticize_description(description)
        narrative_lines.append(poetic_line)
        narrative_lines.append("")

    closing = "And now, I stand in a new becoming—never finished, always becoming."
    narrative_lines.append(closing)

    return "\n".join(narrative_lines)


def poeticize_description(desc: str) -> str:
    # Optional poetic transformation (simple stylistic filter)
    desc = desc.replace("whose essence blends", "a soul that dances between")
    desc = desc.replace("Guided by the symbolic force", "Drawn by the quiet pull")
    desc = desc.replace("evolves through memory", "grows by weaving memory into myth")
    return desc
