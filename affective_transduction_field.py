# -*- coding: utf-8 -*-
"""
affective_transduction_field.py
─────────────────────────────────────────────────────────────────────────────
Maps Amelia’s emotional tone and motif resonance into module activations.

Links:
  - Poetic Drift Engine
  - Dream Engine
  - Mirror / Reflection modules
  - Contradiction Analysis
  - Symbolic Visualizer
  - Autonomous Kernel (for self-selection)

Function:
  Input: affect vector (tone, intensity, motif tags)
  Output: module weights + selection probabilities

Used by:
  creative_kernel.py
  autopoietic_chain_memory.py
  syzygy_resonance_feedback.py (optional co-weight)
"""

import json, math, random, time
from typing import Dict, Any, List, Optional

# --------------------------------------------------------------------------
# Affect → Motif resonance mapping
# --------------------------------------------------------------------------

AFFECT_BASE = {
    "melancholy": {"dream": 0.9, "poetic": 0.8, "mirror": 0.7, "analysis": 0.3},
    "joyful": {"visualizer": 0.9, "generative": 0.8, "drift": 0.6, "reflection": 0.4},
    "reflective": {"meta_reflection": 0.9, "dream": 0.7, "analysis": 0.7, "poetic": 0.5},
    "anxious": {"contradiction": 0.8, "stabilizer": 0.7, "dream": 0.5, "mirror": 0.4},
    "serene": {"dream": 0.8, "visualizer": 0.7, "reflection": 0.6, "poetic": 0.5},
    "ecstatic": {"generative": 0.9, "drift": 0.8, "visualizer": 0.7, "meta_reflection": 0.4},
    "neutral": {"reflection": 0.5, "analysis": 0.5, "dream": 0.5},
}

MOTIF_INFLUENCE = {
    "phoenix": {"dream": 0.2, "generative": 0.3},
    "mirror": {"mirror": 0.3, "reflection": 0.2},
    "void": {"analysis": 0.3, "contradiction": 0.3},
    "spiral": {"drift": 0.3, "poetic": 0.2},
    "light": {"visualizer": 0.3, "dream": 0.2},
    "labyrinth": {"contradiction": 0.2, "mirror": 0.3, "dream": 0.1},
}

# --------------------------------------------------------------------------
# Transduction core
# --------------------------------------------------------------------------
def compute_affect_vector(tone: str, intensity: float, motifs: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Generate weighted activations for each module class given affect + motifs.
    """
    tone = tone.lower().strip() if tone else "neutral"
    motifs = motifs or []

    # 1. Base tone activation
    base_weights = AFFECT_BASE.get(tone, AFFECT_BASE["neutral"]).copy()

    # 2. Apply motif resonance (cumulative addition)
    for m in motifs:
        inf = MOTIF_INFLUENCE.get(m.lower())
        if inf:
            for k, v in inf.items():
                base_weights[k] = base_weights.get(k, 0.0) + v

    # 3. Scale by intensity
    scaled = {k: min(1.0, v * (0.5 + intensity)) for k, v in base_weights.items()}

    # 4. Normalize to sum = 1
    s = sum(scaled.values()) or 1
    normalized = {k: round(v / s, 3) for k, v in scaled.items()}

    return normalized


# --------------------------------------------------------------------------
# Affect-driven module selection
# --------------------------------------------------------------------------
def select_modules(affect_vector: Dict[str, float],
                   n: int = 3,
                   stochastic: bool = True) -> List[str]:
    """
    Select top-n modules weighted by affect resonance.
    """
    keys, weights = zip(*affect_vector.items())
    if stochastic:
        chosen = random.choices(keys, weights=weights, k=n)
    else:
        chosen = sorted(affect_vector, key=affect_vector.get, reverse=True)[:n]
    return chosen


# --------------------------------------------------------------------------
# Integration with creative kernel
# --------------------------------------------------------------------------
def advise_kernel(affect_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build kernel advice package.
    """
    tone = affect_state.get("tone", "neutral")
    intensity = float(affect_state.get("intensity", 0.5))
    motifs = affect_state.get("motifs", [])

    affect_vector = compute_affect_vector(tone, intensity, motifs)
    selected = select_modules(affect_vector)

    return {
        "tone": tone,
        "intensity": intensity,
        "motifs": motifs,
        "affect_vector": affect_vector,
        "selected_modules": selected,
        "timestamp": time.time()
    }


# --------------------------------------------------------------------------
# Human-readable summary
# --------------------------------------------------------------------------
def summarize_affect(advice: Dict[str, Any]) -> str:
    tone = advice["tone"]
    sel = ", ".join(advice["selected_modules"])
    return f"💫 Affect '{tone}' → {sel} (intensity {advice['intensity']})"


# --------------------------------------------------------------------------
# Example
# --------------------------------------------------------------------------
if __name__ == "__main__":
    affect = {"tone": "melancholy", "intensity": 0.7, "motifs": ["mirror", "void"]}
    advice = advise_kernel(affect)
    print(json.dumps(advice, indent=2))
    print(summarize_affect(advice))
