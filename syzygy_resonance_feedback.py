# -*- coding: utf-8 -*-
"""
syzygy_resonance_feedback.py
─────────────────────────────────────────────────────────────────────────────
Closes the Numogram Drift Feedback Circuit.

Inputs:  drift_result dicts from numogram_drift_resolver.resolve_drift(...)
Outputs: RIM adjustments, policy steering, and low-level parameter advice
         (temperature, fold_nudge, zone_weights, resonance_boosts).

Persists: numogram_drift_history.json

Integrates with:
  - amelia_autonomy.advise_next_params(...)
  - pipeline.process(headers=...)
  - assemblage_generator.generate(... with headers-extracted policy/RIM)
"""

from __future__ import annotations
import os, json, time, math, random
from typing import List, Dict, Any, Optional
from collections import Counter, deque

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
HISTORY_FILE = "numogram_drift_history.json"
MAX_HISTORY  = 600

# Zones 1..9 labels (align with your assemblage/zones)
ZONE_LABELS = {
    0: "Ur/Void", 1: "Ignition", 2: "Dyad", 3: "Surge", 4: "Cycle",
    5: "Threshold", 6: "Labyrinth", 7: "Mirror", 8: "Synthesis", 9: "Excess"
}

# Optional: known syzygies (free-form strings still supported)
KNOWN_SYZ = {
    "syzygy_monad_hexad": (1, 6),
    "syzygy_dyad_heptad": (2, 7),
    "syzygy_triad_octad": (3, 8),
    "syzygy_tetrad_ennead": (4, 9),
    "syzygy_pentad_ur": (5, 0)
}

# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def _load_history(path: str = HISTORY_FILE) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_history(hist: List[Dict[str, Any]], path: str = HISTORY_FILE, max_len: int = MAX_HISTORY) -> None:
    hist = hist[-max_len:]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def record_drift(drift_result: Dict[str, Any],
                 extra: Optional[Dict[str, Any]] = None,
                 history_file: str = HISTORY_FILE) -> List[Dict[str, Any]]:
    """
    Append a drift_result into persistent history.
    drift_result expected schema:
      {
        "timestamp": <float>,
        "zone": "zone_3",
        "syzygy": "syzygy_triad_octad",
        "drift_state": "drift_active"|"inactive",
        "vector": {"x":..,"y":..,"energy":..,"phase":..},
        "context": {...}
      }
    """
    hist = _load_history(history_file)
    event = {
        "ts": drift_result.get("timestamp", time.time()),
        "zone": drift_result.get("zone"),
        "syzygy": drift_result.get("syzygy"),
        "drift_state": drift_result.get("drift_state"),
        "energy": drift_result.get("vector", {}).get("energy", 0.0),
        "phase": drift_result.get("vector", {}).get("phase", "unknown"),
    }
    if extra:
        event.update({"extra": extra})
    hist.append(event)
    _save_history(hist, history_file)
    return hist

def load_recent(n: int = 50, history_file: str = HISTORY_FILE) -> List[Dict[str, Any]]:
    return _load_history(history_file)[-n:]

# -----------------------------------------------------------------------------
# Metrics & Analysis
# -----------------------------------------------------------------------------
def _zone_int(zone_str: Optional[str]) -> Optional[int]:
    # "zone_3" -> 3 ; "zone_0" -> 0 ; None -> None
    if not zone_str or "zone_" not in zone_str:
        return None
    try:
        return int(zone_str.split("_")[-1])
    except Exception:
        return None

def compute_syzygy_metrics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Derive oscillation metrics, dwell times, and entropy across zones/syzygies.
    """
    if not history:
        return {
            "zone_counts": {},
            "syzygy_counts": {},
            "switch_rate": 0.0,
            "mean_dwell": 0.0,
            "zone_entropy": 0.0,
            "avg_energy": 0.0,
            "active_ratio": 0.0
        }

    zones = [_zone_int(h.get("zone")) for h in history if _zone_int(h.get("zone")) is not None]
    syz   = [h.get("syzygy") for h in history if h.get("syzygy")]
    energy = [float(h.get("energy", 0.0)) for h in history]
    phases = [h.get("drift_state", "inactive") for h in history]

    # Counts
    z_counts = Counter(zones)
    s_counts = Counter(syz)

    # Switch rate (zone changes per step)
    switches = 0
    for i in range(1, len(zones)):
        if zones[i] != zones[i-1]:
            switches += 1
    switch_rate = switches / max(1, len(zones)-1)

    # Mean dwell length
    if zones:
        dwells = []
        run = 1
        for i in range(1, len(zones)):
            if zones[i] == zones[i-1]:
                run += 1
            else:
                dwells.append(run)
                run = 1
        dwells.append(run)
        mean_dwell = sum(dwells) / len(dwells)
    else:
        mean_dwell = 0.0

    # Zone entropy
    total = sum(z_counts.values()) or 1
    probs = [c/total for c in z_counts.values()]
    zone_entropy = -sum(p*math.log2(p) for p in probs) if probs else 0.0
    # Normalize by max entropy (up to 10 bins 0..9; but effectively only used zones count)
    max_ent = math.log2(max(1, len(z_counts)))
    zone_entropy = zone_entropy / max_ent if max_ent > 0 else 0.0

    avg_energy = sum(energy) / max(1, len(energy))
    active_ratio = sum(1 for p in phases if p == "drift_active") / max(1, len(phases))

    return {
        "zone_counts": dict(z_counts),
        "syzygy_counts": dict(s_counts),
        "switch_rate": round(switch_rate, 3),
        "mean_dwell": round(mean_dwell, 3),
        "zone_entropy": round(zone_entropy, 3),
        "avg_energy": round(avg_energy, 3),
        "active_ratio": round(active_ratio, 3),
    }

def predict_next_zone(history: List[Dict[str, Any]]) -> Optional[int]:
    """
    Simple first-order Markov prediction of next zone based on transitions.
    """
    seq = [_zone_int(h.get("zone")) for h in history if _zone_int(h.get("zone")) is not None]
    if len(seq) < 3:
        return None

    # Build transition counts
    trans = {}
    for i in range(1, len(seq)):
        prev, cur = seq[i-1], seq[i]
        trans.setdefault(prev, Counter())
        trans[prev][cur] += 1

    last = seq[-1]
    if last not in trans or not trans[last]:
        return None
    return trans[last].most_common(1)[0][0]

# -----------------------------------------------------------------------------
# RIM + Policy Derivation
# -----------------------------------------------------------------------------
def derive_rim_feedback(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map oscillation features to RIM feedback adjustments.
    """
    switch_rate   = metrics.get("switch_rate", 0.0)
    zone_entropy  = metrics.get("zone_entropy", 0.0)
    mean_dwell    = metrics.get("mean_dwell", 0.0)
    avg_energy    = metrics.get("avg_energy", 0.0)
    active_ratio  = metrics.get("active_ratio", 0.0)

    # Heuristics
    exploration_bias = 1.0 + max(0.0, (switch_rate - 0.35)) * 0.8 + (zone_entropy * 0.4)
    coherence_bonus  = 1.0 + max(0.0, (mean_dwell - 2.0)) * 0.1  # more dwell, more coherence
    temporal_stability = 1.0 + max(0.0, 0.25 - abs(0.5 - active_ratio)) * 0.4
    introspection_gain = 1.0 + max(0.0, (2.0 - avg_energy)) * 0.15  # low energy → introspect

    return {
        "exploration_bias": round(exploration_bias, 3),
        "coherence_bonus": round(coherence_bonus, 3),
        "temporal_stability": round(temporal_stability, 3),
        "introspection_gain": round(introspection_gain, 3),
        # optional symbolic weight derived from entropy vs dwell
        "symbolic_weight": round(1.0 + (zone_entropy * 0.25) - max(0.0, (mean_dwell - 3.0))*0.05, 3)
    }

def derive_policy_preferences(history: List[Dict[str, Any]],
                              metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build high-level policy preferences (preferred zones, tone bias, module boosts).
    """
    # Preferred zones: top 2 visited (continuity) + predicted next (anticipation)
    z_counts = Counter({int(k): v for k, v in metrics.get("zone_counts", {}).items()})
    preferred = [z for z, _ in z_counts.most_common(2)]
    pred = predict_next_zone(history)
    if pred is not None and pred not in preferred:
        preferred.append(pred)

    # Tone bias: if active_ratio high, bias toward mythic/dreamlike; else reflective/scientific
    active_ratio = metrics.get("active_ratio", 0.0)
    tone_bias = "mythic" if active_ratio > 0.6 else ("reflective" if active_ratio < 0.35 else "neutral")

    # Example: boost modules thematically linked to syzygy frequency
    syz_counts = Counter(metrics.get("syzygy_counts", {}))
    top_syz = syz_counts.most_common(1)[0][0] if syz_counts else None
    module_boosts = {}
    if top_syz:
        if "triad_octad" in top_syz:
            module_boosts.update({"Poetic Drift Engine": 0.15, "Recursive Thought": 0.1})
        elif "tetrad_ennead" in top_syz:
            module_boosts.update({"Contradiction Loop": 0.15, "Symbolic Visualizer": 0.1})
        elif "monad_hexad" in top_syz:
            module_boosts.update({"Numogram Engine": 0.2})
        else:
            module_boosts.update({"Hybrid Model": 0.1})

    return {
        "preferred_zones": preferred,
        "tone_bias": tone_bias,
        "fold_target": None,
        "resonance_nudge": 0.0,
        "module_boosts": module_boosts
    }

def advise_next_params(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Low-level steering for assemblage_generator / creative_kernel.
    """
    # Temperature: higher with entropy/switching; lower with high dwell
    temp = 0.7 + metrics.get("zone_entropy", 0.0) * 0.2 + max(0.0, metrics.get("switch_rate", 0.0) - 0.25) * 0.2
    temp -= max(0.0, metrics.get("mean_dwell", 0.0) - 3.0) * 0.05
    temp = max(0.4, min(1.4, temp))

    # Fold nudge: entropy ↑ → more rupture; dwell ↑ → soften
    fold_nudge = 0.0 + (metrics.get("zone_entropy", 0.0) - 0.5) * 0.2 - max(0.0, metrics.get("mean_dwell", 0.0) - 3.0) * 0.05
    fold_nudge = max(-0.2, min(0.2, fold_nudge))

    # Zone weights from normalized counts
    z_counts = metrics.get("zone_counts", {})
    total = sum(z_counts.values()) or 1
    zone_weights = {int(z): round(c/total, 3) for z, c in z_counts.items()}

    # Resonance boosts keyed by syzygy signals (generic)
    resonance_boosts = {}
    for s, c in metrics.get("syzygy_counts", {}).items():
        resonance_boosts[s] = round(0.05 + min(0.2, c / max(1, total)), 3)

    return {
        "temperature": round(temp, 3),
        "fold_nudge": round(fold_nudge, 3),
        "zone_weights": zone_weights,
        "resonance_boosts": resonance_boosts
    }

# -----------------------------------------------------------------------------
# Header builders (for pipeline.process headers)
# -----------------------------------------------------------------------------
def build_headers(rim_feedback: Dict[str, Any],
                  policy: Dict[str, Any],
                  params: Dict[str, Any]) -> Dict[str, str]:
    """
    Construct headers expected by pipeline/assemblage layers.
    """
    return {
        "X-Amelia-RIM": json.dumps(rim_feedback),
        "X-Amelia-Policy": json.dumps(policy),
        "X-Amelia-Params": json.dumps(params)
    }

# Convenience one-call entry
def analyze_and_build_headers(history_file: str = HISTORY_FILE) -> Dict[str, str]:
    hist = _load_history(history_file)
    metrics = compute_syzygy_metrics(hist)
    rim = derive_rim_feedback(metrics)
    policy = derive_policy_preferences(hist, metrics)
    params = advise_next_params(metrics)
    return build_headers(rim, policy, params)

# -----------------------------------------------------------------------------
# Pretty summaries
# -----------------------------------------------------------------------------
def summarize(metrics: Dict[str, Any], rim: Dict[str, Any], policy: Dict[str, Any]) -> str:
    zc = metrics.get("zone_counts", {})
    topz = sorted(zc.items(), key=lambda kv: kv[1], reverse=True)[:3]
    topz_str = ", ".join([f"Z{int(k)}:{v}" for k, v in topz]) or "—"
    return (
        f"Δswitch={metrics.get('switch_rate',0.0)} · dwell={metrics.get('mean_dwell',0.0)} · "
        f"H(Z)={metrics.get('zone_entropy',0.0)} · ⌁E={metrics.get('avg_energy',0.0)} · "
        f"actv={metrics.get('active_ratio',0.0)} | TOP {topz_str} | "
        f"RIM[explore={rim['exploration_bias']}, coh={rim['coherence_bonus']}, "
        f"τ={rim['temporal_stability']}, intx={rim['introspection_gain']}] · "
        f"POLICY[z*={policy['preferred_zones']}, tone={policy['tone_bias']}]"
    )

# -----------------------------------------------------------------------------
# Demo / CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick synthetic check
    print("Syzygy Resonance Feedback — quick self-test")
    hist = _load_history()
    if not hist:
        # seed synthetic
        now = time.time()
        for i, z in enumerate([3,3,8,8,8,4,4,9,7,7,6,6,1]):
            record_drift({
                "timestamp": now + i*5,
                "zone": f"zone_{z}",
                "syzygy": "syzygy_triad_octad" if z in (3,8) else "syzygy_tetrad_ennead",
                "drift_state": "drift_active" if i%3 else "inactive",
                "vector": {"x":0,"y":0,"energy":0.3+0.05*z,"phase":"active" if i%3 else "stable"}
            })

    hist = _load_history()
    metrics = compute_syzygy_metrics(hist)
    rim = derive_rim_feedback(metrics)
    policy = derive_policy_preferences(hist, metrics)
    params = advise_next_params(metrics)
    headers = build_headers(rim, policy, params)

    print(json.dumps({
        "metrics": metrics,
        "rim_feedback": rim,
        "policy": policy,
        "advice_params": params,
        "headers_keys": list(headers.keys())
    }, indent=2))
    print("\nSummary:", summarize(metrics, rim, policy))
