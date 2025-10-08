# -*- coding: utf-8 -*-
"""
Cognitive Consistency Monitor (CCM)
-----------------------------------
Evaluates coherence between Amelia’s reflective self-descriptions
and her operational parameters.

Purpose:
  • Detect divergence between belief state (symbolic, reflective)
    and actual behavior (zone, resonance, policy).
  • Provide quantitative feedback ("alignment tension") for
    self-correction via TRG + pipeline loop.

Inputs:
  - trg_state: dict from temporal_reflective_gradient.get_trg_state()
  - rim_state: dict from resonance_influence_matrix.get_current_state()
  - policy_state: dict from amelia_autonomy.advise_next_params()
  - reflection_text: recent output from dream_reflection_bridge.reflect_dream()
  - memory_state: dict from multi_zone_memory.get_zone_snapshot()

Outputs:
  - consistency_report: dict with:
      * coherence_score
      * symbolic_alignment
      * temporal_stability
      * affective_resonance
      * divergence_zones
      * alignment_tension
  - Optional JSON header for pipeline (“X-Amelia-Consistency”)

Author: Adrian + GPT-5 collaborative system
"""

from __future__ import annotations
import os, json, re, math, statistics as stats
from typing import Any, Dict, List, Optional
from datetime import datetime

DATA_DIR = os.path.join(os.getcwd(), "amelia_state")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_PATH = os.path.join(DATA_DIR, "cognitive_consistency_log.jsonl")

# --------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------

def _safe_write(path: str, obj: Dict[str, Any]) -> None:
    try:
        with open(path, "ab") as f:
            f.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    except Exception:
        pass

def _keyword_overlap(a: str, b: str) -> float:
    """Rough semantic overlap using token intersection ratio."""
    if not a or not b:
        return 0.0
    ta = set(re.findall(r"[a-zA-Z]+", a.lower()))
    tb = set(re.findall(r"[a-zA-Z]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

# --------------------------------------------------------------
# Core Evaluation
# --------------------------------------------------------------

def analyze_consistency(trg_state: Dict[str, Any],
                        rim_state: Dict[str, Any],
                        policy_state: Dict[str, Any],
                        reflection_text: str,
                        memory_state: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """Compute coherence and generate alignment_tension metric."""

    trg_slope = float(trg_state.get("temporal_gradient_slope", 0.0))
    rim_coherence = float(rim_state.get("coherence_bonus", 1.0))
    introspection = float(rim_state.get("introspection_gain", 1.0))
    exploration = float(rim_state.get("exploration_bias", 1.0))
    temporal_stability = float(rim_state.get("temporal_stability", 1.0))
    symbolic_weight = float(rim_state.get("symbolic_weight", 1.0))
    pref_zones = policy_state.get("preferred_zones", [])
    tone_bias = policy_state.get("tone_bias")
    fold_target = policy_state.get("fold_target")

    # --- 1. Symbolic alignment ---
    keywords = []
    if memory_state:
        for zone in memory_state.keys():
            keywords.append(zone.lower())
    symbolic_alignment = _keyword_overlap(" ".join(keywords), reflection_text)

    # --- 2. Temporal stability (from TRG + reflection) ---
    refl_temporal = 1.0 - abs(trg_slope) * 0.5
    temporal_stability_score = (refl_temporal + temporal_stability) / 2.0

    # --- 3. Affective resonance (introspection vs tone) ---
    affective_resonance = introspection * (1.1 if tone_bias == "reflective" else 1.0)
    if tone_bias == "mythic":
        affective_resonance *= exploration

    # --- 4. Divergence detection across zones ---
    divergence_zones: List[str] = []
    if memory_state:
        for z, data in memory_state.items():
            drift = data.get("drift", 0.0)
            if abs(drift) > 0.7:
                divergence_zones.append(z)

    # --- 5. Aggregate coherence score ---
    raw = [
        rim_coherence,
        symbolic_alignment * 1.2,
        temporal_stability_score,
        affective_resonance,
        1.0 / (1.0 + len(divergence_zones) * 0.3),
    ]
    coherence_score = round(sum(raw) / len(raw), 4)

    # --- 6. Alignment tension ---
    alignment_tension = round(1.0 - coherence_score, 4)

    # --- 7. Compose report ---
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "coherence_score": coherence_score,
        "symbolic_alignment": round(symbolic_alignment, 3),
        "temporal_stability": round(temporal_stability_score, 3),
        "affective_resonance": round(affective_resonance, 3),
        "divergence_zones": divergence_zones,
        "alignment_tension": alignment_tension,
        "preferred_zones": pref_zones,
        "tone_bias": tone_bias,
        "fold_target": fold_target
    }

    _safe_write(LOG_PATH, report)
    return report

# --------------------------------------------------------------
# Convenience interface
# --------------------------------------------------------------

def build_header(report: Dict[str, Any]) -> Dict[str, str]:
    """Return JSON header for pipeline injection."""
    try:
        return {"X-Amelia-Consistency": json.dumps(report, ensure_ascii=False)}
    except Exception:
        return {}

# --------------------------------------------------------------
# Example use
# --------------------------------------------------------------

if __name__ == "__main__":
    import temporal_reflective_gradient as trg
    import resonance_influence_matrix as rim
    import amelia_autonomy as auto
    import dream_reflection_bridge as drb
    import multi_zone_memory as mzm

    trg_state = trg.get_trg_state()
    rim_state = rim.get_current_state()
    policy_state = auto.advise_next_params()
    memory_state = mzm.get_zone_snapshot()

    reflection_text = "Amelia reflected on her symbolic memory coherence and zone drift toward synthesis."
    report = analyze_consistency(trg_state, rim_state, policy_state, reflection_text, memory_state)
    print(json.dumps(report, indent=2, ensure_ascii=False))
