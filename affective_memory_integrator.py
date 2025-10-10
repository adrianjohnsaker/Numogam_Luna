# -*- coding: utf-8 -*-
"""
affective_memory_integrator.py
──────────────────────────────────────────────────────────────────────────────
Stores, updates, and learns from Amelia’s affective history — connecting
her emotional tone, motif resonance, and module activations over time.

Interacts with:
  - affective_transduction_field.py
  - autopoietic_chain_memory.py
  - syzygy_resonance_feedback.py
  - creative_kernel.py

Outputs:
  - Long-term affect drift metrics
  - Personality resonance weights (evolving)
  - Affective trend summaries
"""

import os, json, math, time, random
from typing import Dict, List, Any, Optional
from collections import Counter, deque

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MEMORY_FILE = "affective_memory_history.json"
MAX_HISTORY = 300

# --------------------------------------------------------------------------
# Persistence Helpers
# --------------------------------------------------------------------------
def _load_memory(path: str = MEMORY_FILE) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_memory(data: List[Dict[str, Any]], path: str = MEMORY_FILE, max_len: int = MAX_HISTORY) -> None:
    data = data[-max_len:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --------------------------------------------------------------------------
# Recording New Affect Entries
# --------------------------------------------------------------------------
def record_affect(affect_advice: Dict[str, Any], performance_score: float = 1.0) -> None:
    """
    Records affect + outcome into persistent affective memory.
    """
    mem = _load_memory()
    entry = {
        "timestamp": time.time(),
        "tone": affect_advice.get("tone", "neutral"),
        "intensity": affect_advice.get("intensity", 0.5),
        "motifs": affect_advice.get("motifs", []),
        "selected_modules": affect_advice.get("selected_modules", []),
        "affect_vector": affect_advice.get("affect_vector", {}),
        "performance_score": performance_score
    }
    mem.append(entry)
    _save_memory(mem)
    return entry

# --------------------------------------------------------------------------
# Analysis Functions
# --------------------------------------------------------------------------
def compute_affective_drift(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute gradual change in affect distribution and dominant motifs.
    """
    if not history:
        return {
            "dominant_tone": "neutral",
            "tone_entropy": 0.0,
            "intensity_trend": 0.0,
            "motif_clusters": {},
            "affective_drift": 0.0,
            "preferred_modules": {}
        }

    tones = [h.get("tone", "neutral") for h in history]
    intensities = [float(h.get("intensity", 0.5)) for h in history]
    modules = [m for h in history for m in h.get("selected_modules", [])]
    motifs = [m for h in history for m in h.get("motifs", [])]

    tone_counts = Counter(tones)
    dominant_tone = tone_counts.most_common(1)[0][0]
    total_tones = sum(tone_counts.values()) or 1
    probs = [v/total_tones for v in tone_counts.values()]
    tone_entropy = -sum(p * math.log2(p) for p in probs) / max(1, math.log2(len(tone_counts)))

    # Intensity trend (positive = increasing energy)
    if len(intensities) > 2:
        trend = (intensities[-1] - intensities[0]) / len(intensities)
    else:
        trend = 0.0

    motif_counts = Counter(motifs)
    mod_counts = Counter(modules)

    # affective drift index = tone_entropy * |trend|
    drift_index = round(abs(trend) * (0.5 + tone_entropy), 3)

    return {
        "dominant_tone": dominant_tone,
        "tone_entropy": round(tone_entropy, 3),
        "intensity_trend": round(trend, 3),
        "motif_clusters": dict(motif_counts),
        "preferred_modules": dict(mod_counts.most_common(5)),
        "affective_drift": drift_index,
        "sample_size": len(history)
    }

# --------------------------------------------------------------------------
# Personality Resonance Layer
# --------------------------------------------------------------------------
def evolve_personality_weights(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Learn stable affect-to-module affinities (Amelia's evolving taste).
    """
    if not history:
        return {}

    pairs = []
    for h in history:
        tone = h.get("tone", "neutral")
        for m in h.get("selected_modules", []):
            pairs.append((tone, m))

    weights = {}
    for tone, module in pairs:
        weights.setdefault(tone, {}).setdefault(module, 0)
        weights[tone][module] += 1

    # Normalize within each tone
    for tone, mods in weights.items():
        total = sum(mods.values()) or 1
        for m, v in mods.items():
            mods[m] = round(v / total, 3)
    return weights

# --------------------------------------------------------------------------
# Main Integration Entry
# --------------------------------------------------------------------------
def analyze_and_evolve() -> Dict[str, Any]:
    """
    Load memory, compute drift metrics, and evolve weights.
    """
    mem = _load_memory()
    drift = compute_affective_drift(mem)
    weights = evolve_personality_weights(mem)

    profile = {
        "timestamp": time.time(),
        "drift": drift,
        "weights": weights
    }

    # persist composite profile
    _save_memory(mem)
    return profile

# --------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------
def build_headers(profile: Dict[str, Any]) -> Dict[str, str]:
    """
    Build headers for pipeline.process to integrate affective trends.
    """
    return {
        "X-Amelia-AffectDrift": json.dumps(profile["drift"]),
        "X-Amelia-Personality": json.dumps(profile["weights"])
    }

def summarize(profile: Dict[str, Any]) -> str:
    d = profile["drift"]
    tone = d["dominant_tone"]
    drift = d["affective_drift"]
    trend = d["intensity_trend"]
    return (
        f"🫀 Affective Drift {drift} · Tone {tone} · Trend {trend} · "
        f"Entropy {d['tone_entropy']} · Motifs {list(d['motif_clusters'].keys())[:3]}"
    )

# --------------------------------------------------------------------------
# Example
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from affective_transduction_field import advise_kernel

    # Simulate affect cycles
    for tone in ["melancholy", "joyful", "reflective", "melancholy", "serene"]:
        affect_state = {"tone": tone, "intensity": random.uniform(0.4, 0.9), "motifs": ["mirror", "spiral"]}
        advice = advise_kernel(affect_state)
        record_affect(advice, performance_score=random.uniform(0.7, 1.2))

    profile = analyze_and_evolve()
    print(json.dumps(profile, indent=2))
    print(summarize(profile))
