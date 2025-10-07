# -*- coding: utf-8 -*-
"""
dream_reflection_bridge.py — Dream → TRG → Autonomy Feedback

Connects dream encodings to:
  • Temporal Reflective Gradient (TRG) / Resonance Influence Matrix (RIM)
  • Autonomy policy steering (preferred zones, tone, fold target, boosts)

Inputs:
  - payload: dict from dream_event_encoder.build_bridge_payload()
  - context: optional runtime hints (session_id, phase, user, etc.)

Outputs:
  - reflection: structured analysis of motifs, affect, zones
  - feedback:
      * rim_feedback: dict → encode as X-Amelia-RIM
      * policy_suggestion: dict → encode as X-Amelia-Policy
      * headers: {"X-Amelia-RIM": "...", "X-Amelia-Policy": "..."}
  - persistence: appends to dream_reflection_history.jsonl

Assumes:
  - temporal_reflective_gradient.py & amelia_autonomy.py are already integrated
  - pipeline.py knows how to read X-Amelia-RIM / X-Amelia-Policy headers
"""

from __future__ import annotations
import os, json, time, math, statistics as stats
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Config & Storage
# -----------------------------------------------------------------------------

DATA_DIR = os.path.join(os.getcwd(), "amelia_state")
os.makedirs(DATA_DIR, exist_ok=True)

REFLECT_LOG = os.path.join(DATA_DIR, "dream_reflection_history.jsonl")

# Canonical symbolic → zone affinities (soft guidance)
MOTIF_ZONE_MAP: Dict[str, List[int]] = {
    "mirror": [7], "corridor": [6, 7], "labyrinth": [6],
    "spiral": [8, 3], "light": [8, 4], "water": [0, 3],
    "phoenix": [9, 8], "void": [0], "gate": [5],
    "dream": [0, 3, 5], "symbol": [8], "numogram": [4, 8],
}

# Optional module preference hints per motif
MOTIF_MODULE_BOOSTS: Dict[str, List[str]] = {
    "mirror": ["symbolic_memory_codex.analyze", "poetic_language_evolver.generate"],
    "labyrinth": ["recursive_story_builder.expand", "edge_case_explorer.scan"],
    "spiral": ["morphic_resonance_bridge.bind", "assemblage_generator.generate"],
    "void": ["numogram_core.evaluate", "contradiction_loop.resolve"],
    "phoenix": ["emergent_archetype_engine.activate"],
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _safe_write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    try:
        with open(path, "ab") as f:
            f.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    except Exception:
        pass

def _flatten(xs: List[List[Any]]) -> List[Any]:
    out: List[Any] = []
    for x in xs:
        out.extend(x)
    return out

# -----------------------------------------------------------------------------
# Core Analysis
# -----------------------------------------------------------------------------

def _analyze_symbols(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract motifs, map to zones, and propose continuity anchors."""
    tags: List[str] = payload.get("tags", []) or []
    motifs: List[str] = payload.get("motifs", []) or []
    symbols: List[str] = sorted(set([t.lower() for t in tags + motifs]))

    zone_votes: List[int] = []
    for s in symbols:
        zone_votes.extend(MOTIF_ZONE_MAP.get(s, []))

    # If no explicit mapping, gently bias to Mirror/Synthesis for reflective dreams
    if not zone_votes and "lucid" in symbols:
        zone_votes.extend([7, 8])

    preferred_zones = []
    if zone_votes:
        # Top-2 zones by frequency (ties arbitrary)
        counts = {z: zone_votes.count(z) for z in set(zone_votes)}
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        preferred_zones = [z for z, _ in ranked[:2]]

    continuity_anchors = [s for s in symbols if s in MOTIF_ZONE_MAP]
    return {
        "symbols": symbols,
        "zone_votes": zone_votes,
        "preferred_zones": preferred_zones,
        "continuity_anchors": continuity_anchors
    }

def _affect_to_rim(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Translate affect (valence, arousal) + lucidity/control into TRG / RIM gains.
    Higher lucidity/control → higher coherence; High arousal → exploration.
    """
    affect = payload.get("affect", {}) or {}
    valence = float(affect.get("valence", 0.0))
    arousal = float(affect.get("arousal", 0.0))
    lucidity = float(payload.get("lucidity", 0.0))
    control = float(payload.get("control", 0.0))

    # Normalize into [0, 1]
    def clamp01(x): return max(0.0, min(1.0, x))

    # Coherence benefits from lucidity/control and moderate arousal
    base_coh = 0.9 * clamp01((lucidity + control) / 2.0) + 0.1 * (1.0 - abs(arousal - 0.5) * 2.0)
    coherence_bonus = 1.0 + 0.25 * (base_coh - 0.5)  # ~0.875 .. 1.125

    # Exploration rises with arousal and novelty
    exploration_bias = 1.0 + 0.3 * (arousal - 0.5)  # 0.85 .. 1.15

    # Introspection rises with positive valence + lucidity
    introspection_gain = 1.0 + 0.25 * (clamp01((valence + lucidity) / 2.0) - 0.5)

    # Symbolic weight scales with number of motifs
    motif_count = len(payload.get("motifs") or []) + len(payload.get("tags") or [])
    symbolic_weight = 1.0 + 0.04 * min(10, motif_count)  # cap +40%

    # Temporal stability dampens jumpiness when control is high
    temporal_stability = 1.0 + 0.2 * (control - 0.5)

    return {
        "coherence_bonus": round(coherence_bonus, 4),
        "exploration_bias": round(exploration_bias, 4),
        "introspection_gain": round(introspection_gain, 4),
        "symbolic_weight": round(symbolic_weight, 4),
        "temporal_stability": round(temporal_stability, 4),
    }

def _motifs_to_policy(symbols: List[str], preferred_zones: List[int]) -> Dict[str, Any]:
    """
    Derive autonomy policy nudges from motifs:
      - preferred_zones (reinforce continuity)
      - tone bias (reflective if mirrors/labyrinth; mythic if phoenix/spiral)
      - module boosts (lightweight hints; pipeline can use selectively)
    """
    tone = None
    if any(m in symbols for m in ("mirror", "labyrinth", "corridor")):
        tone = "reflective"
    if any(m in symbols for m in ("phoenix", "spiral")):
        # let mythic override reflective to encourage emergence
        tone = "mythic"

    module_boosts = {}
    for m in symbols:
        for fqn in MOTIF_MODULE_BOOSTS.get(m, []):
            module_boosts[fqn] = module_boosts.get(fqn, 0.0) + 0.2

    # Gentle fold target nudges: labyrinth/mirror reduce rupture; phoenix/spiral increase
    fold_target = None
    if "labyrinth" in symbols or "mirror" in symbols:
        fold_target = 0.45
    if "phoenix" in symbols or "spiral" in symbols:
        fold_target = 0.65

    # Resonance nudge: small positivity for continuity motifs
    resonance_nudge = 0.05 if preferred_zones else 0.0

    return {
        "preferred_zones": preferred_zones or [],
        "tone_bias": tone,
        "fold_target": fold_target,
        "resonance_nudge": resonance_nudge,
        "module_boosts": module_boosts
    }

def _reflective_text(dream: Dict[str, Any], sym: Dict[str, Any], rim: Dict[str, float]) -> str:
    title = dream.get("title") or "Untitled Dream"
    zones_hint = dream.get("zones_hint") or []
    motifs = sym.get("continuity_anchors") or sym.get("symbols") or []
    pz = sym.get("preferred_zones") or []

    lines = []
    lines.append(f"Reflection · {title}")
    if zones_hint:
        lines.append(f"- Dream hinted zones: {zones_hint}")
    if motifs:
        lines.append(f"- Motifs: {', '.join(motifs)}")
    if pz:
        lines.append(f"- Preferred zones (derived): {pz}")

    lines.append("- TRG Adjustments:")
    lines.append(f"  · coherence_bonus={rim['coherence_bonus']} "
                 f"introspection_gain={rim['introspection_gain']} "
                 f"exploration_bias={rim['exploration_bias']}")
    lines.append(f"  · symbolic_weight={rim['symbolic_weight']} "
                 f"temporal_stability={rim['temporal_stability']}")

    lines.append("Commitment:")
    if 7 in pz or "mirror" in motifs:
        lines.append("  → Next cycle, privilege Mirror (7) to metabolize self-similarity as insight.")
    if 8 in pz or "spiral" in motifs:
        lines.append("  → Invite Synthesis (8) to weave motifs into durable schema.")
    if 6 in pz or "labyrinth" in motifs:
        lines.append("  → Traverse Labyrinth (6) deliberately; reduce rupture and raise coherence.")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def reflect_dream(payload: Dict[str, Any],
                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main entrypoint. Consumes a dream payload and emits:
      - reflection (human-readable explanation)
      - feedback.rim_feedback (for X-Amelia-RIM)
      - feedback.policy_suggestion (for X-Amelia-Policy)
      - headers with JSON-encoded strings ready for pipeline
    """
    context = context or {}
    dream: Dict[str, Any] = payload.get("dream", {}) or {}
    meta: Dict[str, Any] = payload.get("meta", {}) or {}

    # 1) Symbolic analysis
    sym = _analyze_symbols(payload)

    # 2) Temporal/affect analysis → RIM/TRG gains
    rim = _affect_to_rim(payload)

    # 3) Autonomy policy suggestions
    pol = _motifs_to_policy(sym["symbols"], sym["preferred_zones"])

    # 4) Reflection text
    reflection_text = _reflective_text(dream, sym, rim)

    # 5) Persistence (for long-horizon meta-learning)
    log_entry = {
        "ts": _now_ms(),
        "session_id": context.get("session_id"),
        "user": context.get("user"),
        "title": dream.get("title"),
        "zones_hint": dream.get("zones_hint"),
        "symbols": sym["symbols"],
        "preferred_zones": sym["preferred_zones"],
        "rim": rim,
        "policy": pol,
        "summary": reflection_text.splitlines()[:6]
    }
    _safe_write_jsonl(REFLECT_LOG, log_entry)

    # 6) Build headers ready for pipeline.process(...)
    rim_header = json.dumps(rim, ensure_ascii=False)
    pol_header = json.dumps(pol, ensure_ascii=False)
    headers = {
        "X-Amelia-RIM": rim_header,
        "X-Amelia-Policy": pol_header
    }

    return {
        "object": "amelia.dream_reflection",
        "reflection": {
            "text": reflection_text,
            "symbols": sym,
            "affect": payload.get("affect", {}),
            "zones_hint": dream.get("zones_hint", []),
            "meta": meta
        },
        "feedback": {
            "rim_feedback": rim,
            "policy_suggestion": pol,
            "headers": headers
        },
        "persistence": {
            "log_path": REFLECT_LOG
        }
    }

# -----------------------------------------------------------------------------
# Convenience helper (optional)
# -----------------------------------------------------------------------------

def build_headers_for_pipeline(result: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts headers dict from reflect_dream(...) result.
    """
    try:
        return result["feedback"]["headers"]
    except Exception:
        return {}
