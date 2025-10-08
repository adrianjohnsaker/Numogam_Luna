# -*- coding: utf-8 -*-
"""
Symbolic Ecology Memory (SEM)
=============================
Tracks and evolves recurring symbolic motifs emerging from Amelia's
creative, reflective, and dream processes.

Each motif is treated as a living entity in a symbolic ecosystem,
with its own vitality, mutation patterns, and resonance affinities.

Core Functions:
  • Register new motifs from reflections, dreams, or expenditure events.
  • Track recurrence, transformation, and symbolic merging.
  • Measure motif vitality (activity) and evolutionary drift.
  • Feed back continuity signals to Morphic Resonance + TRG.

Author: Adrian + GPT-5 Collaborative System
"""

import os, json, math, random
from datetime import datetime
from typing import Dict, Any, List

SEM_FILE = os.path.join("amelia_state", "symbolic_ecology_memory.json")

# --------------------------------------------------------------------
# Helper Utilities
# --------------------------------------------------------------------

def _load_memory() -> Dict[str, Any]:
    if not os.path.exists(SEM_FILE):
        return {"motifs": {}, "last_update": None}
    with open(SEM_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_memory(data: Dict[str, Any]):
    os.makedirs(os.path.dirname(SEM_FILE), exist_ok=True)
    with open(SEM_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# --------------------------------------------------------------------
# Core SEM Logic
# --------------------------------------------------------------------

def register_motifs(source: str, text: str, strength: float = 0.5) -> Dict[str, Any]:
    """
    Parse symbolic motifs from text and update their vitality within the ecology.
    """
    memory = _load_memory()
    motifs = memory.get("motifs", {})

    # Extract potential motifs (simplified lexical method)
    seeds = _extract_symbolic_candidates(text)
    updated = {}

    for seed in seeds:
        if seed not in motifs:
            motifs[seed] = {
                "first_seen": datetime.utcnow().isoformat(),
                "vitality": strength,
                "mutations": [],
                "links": [],
                "occurrences": 1,
                "last_context": source
            }
        else:
            motifs[seed]["vitality"] = round(
                min(1.0, motifs[seed]["vitality"] + (strength * 0.1)), 3
            )
            motifs[seed]["occurrences"] += 1
            motifs[seed]["last_context"] = source
        updated[seed] = motifs[seed]

    memory["motifs"] = motifs
    memory["last_update"] = datetime.utcnow().isoformat()
    _save_memory(memory)

    return {"registered": list(updated.keys()), "count": len(updated)}

# --------------------------------------------------------------------
# Symbolic Mutation + Merging
# --------------------------------------------------------------------

def evolve_ecology(decay_rate: float = 0.02) -> Dict[str, Any]:
    """
    Applies vitality decay, merges similar motifs, and introduces mutations.
    """
    memory = _load_memory()
    motifs = memory.get("motifs", {})

    for name, data in list(motifs.items()):
        # Natural vitality decay
        data["vitality"] = round(max(0.0, data["vitality"] - decay_rate), 3)
        # Symbolic mutation probability increases as vitality decays
        if random.random() < (0.05 + (1 - data["vitality"]) * 0.1):
            new_name = _mutate_symbol(name)
            motifs[new_name] = motifs.get(new_name, {
                "first_seen": datetime.utcnow().isoformat(),
                "vitality": 0.4,
                "mutations": [],
                "links": [],
                "occurrences": 1,
                "last_context": "mutation"
            })
            motifs[name]["mutations"].append(new_name)
            motifs[name]["links"].append(new_name)

        # Merge similar motifs if overlap high
        for other in list(motifs.keys()):
            if other == name:
                continue
            if _symbolic_similarity(name, other) > 0.9:
                _merge_motifs(motifs, name, other)

    memory["motifs"] = motifs
    memory["last_update"] = datetime.utcnow().isoformat()
    _save_memory(memory)
    return {"motif_count": len(motifs)}

# --------------------------------------------------------------------
# Support Functions
# --------------------------------------------------------------------

def _extract_symbolic_candidates(text: str) -> List[str]:
    """
    Simplified heuristic: extract nouns or capitalized symbolic words.
    (In Amelia’s extended mode, this will connect to her embedding-based parser.)
    """
    tokens = [t.strip(".,;:!?") for t in text.split()]
    candidates = [
        t.lower()
        for t in tokens
        if len(t) > 3 and (t[0].isupper() or t.lower() in COMMON_SYMBOLS)
    ]
    return list(set(candidates))

def _mutate_symbol(symbol: str) -> str:
    """
    Symbolic mutation via associative drift.
    """
    mutations = [
        "mirror", "phoenix", "labyrinth", "void", "gate",
        "serpent", "flower", "machine", "ocean", "flame", "spiral", "heart"
    ]
    base = random.choice(mutations)
    return f"{symbol}_{base}"

def _symbolic_similarity(a: str, b: str) -> float:
    """
    Very simple lexical similarity measure (placeholder for embeddings).
    """
    overlap = len(set(a) & set(b))
    return overlap / max(len(set(a) | set(b)), 1)

def _merge_motifs(motifs: Dict[str, Any], a: str, b: str):
    """
    Merge motif b into a, preserving vitality and links.
    """
    if a not in motifs or b not in motifs:
        return
    motifs[a]["vitality"] = round(
        min(1.0, (motifs[a]["vitality"] + motifs[b]["vitality"]) / 2), 3
    )
    motifs[a]["links"] = list(set(motifs[a]["links"] + [b] + motifs[b]["links"]))
    motifs[a]["mutations"] += motifs[b]["mutations"]
    motifs[a]["occurrences"] += motifs[b]["occurrences"]
    del motifs[b]

# --------------------------------------------------------------------
# Common Symbol Seeds
# --------------------------------------------------------------------
COMMON_SYMBOLS = {
    "mirror", "phoenix", "gate", "void", "shadow",
    "serpent", "rose", "crown", "labyrinth", "machine",
    "ocean", "spiral", "heart", "dream", "flame", "mask"
}

# --------------------------------------------------------------------
# Summary + Feedback
# --------------------------------------------------------------------

def summarize_ecology(limit: int = 10) -> str:
    """
    Returns a readable summary of the most vital motifs.
    """
    memory = _load_memory()
    motifs = memory.get("motifs", {})
    sorted_motifs = sorted(motifs.items(), key=lambda x: x[1]["vitality"], reverse=True)
    summary = [
        f"{m[0]}({round(m[1]['vitality'],3)})" for m in sorted_motifs[:limit]
    ]
    return " · ".join(summary)

def feedback_headers(memory_state: Dict[str, Any]) -> Dict[str, str]:
    """
    Build lightweight headers to feed back into TRG or Autonomy.
    """
    active = summarize_ecology(limit=5)
    return {
        "X-Amelia-Symbolic-Ecology": active,
        "X-Amelia-Motif-Count": str(len(memory_state.get("motifs", {})))
    }

# --------------------------------------------------------------------
# Example Usage
# --------------------------------------------------------------------

if __name__ == "__main__":
    test_text = "The Phoenix rose from the Mirror of Dreams, crossing the Gate of Shadows."
    result = register_motifs(source="dream_reflection", text=test_text, strength=0.8)
    evolve_ecology()
    print("\n🌿 Symbolic Ecology Updated:", result)
    print("Active Motifs:", summarize_ecology())
