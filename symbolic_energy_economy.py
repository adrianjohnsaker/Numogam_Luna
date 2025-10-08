# -*- coding: utf-8 -*-
"""
Symbolic Energy Economy Model (SEEM)
====================================
Maps accumulation → restriction → expenditure as cyclical flows
through Numogram zones, driven by TRG slopes and alignment tensions.

Inputs (per cycle):
  • trg_metrics: from temporal_reflective_metrics.compute_temporal_metrics(...)
  • rim_feedback: from amelia_autonomy.reflect_on_policy(...)[ "rim_feedback" ]
  • consistency_report: from cognitive_consistency_monitor.analyze_consistency(...)
  • zone_history: recent zone drift records (assemblage_generator.get_recent_drifts)

Outputs:
  • state: updated energy flows, phase, ruptures, zone currents
  • pipeline headers: temperature/fold nudges and zone_weights to steer next cycle
  • graph: edges and nodes for visualization

Author: Adrian + GPT-5 Collaborative System
"""

from __future__ import annotations
import os, json, math, random
from typing import Dict, Any, List, Tuple
from datetime import datetime

STATE_FILE = os.path.join("amelia_state", "symbolic_energy_economy.json")

ZONES = list(range(10))  # 0..9 (0=Ur/Void); primary 1..9 if you prefer
SYZYGIES = [
    (1, 6), (2, 7), (3, 8), (4, 9), (5, 0)  # exemplar syzygies; adapt to your schema
]

DEFAULTS = {
    "decay": 0.06,                # passive energy decay per cycle
    "transfer_gain": 0.18,        # base energy transfer along currents
    "restriction_coeff": 0.35,    # how much alignment tension restricts flow
    "rupture_threshold": 0.62,    # differential needed to trigger rupture
    "surplus_thresholds": {       # phase thresholds
        "accum_low": 0.20,        # < acc_low => accumulation phase
        "exp_high": 0.55          # > exp_high => expenditure phase
    },
    "max_history": 80
}

# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "currents": {str(z): 0.0 for z in ZONES},   # psychic current per zone
            "edges": {},                                # (i->j) flow intensities
            "phase": "accumulation",
            "ruptures": [],
            "last_update": None,
            "history": []
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_state(st: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------
# Core update
# ---------------------------------------------------------------------

def _safe(st: float) -> float:
    return max(0.0, min(1.0, float(st)))

def _energy_surplus(currents: Dict[int, float]) -> float:
    # Surplus = mean energy + inequality bonus (Gini-ish)
    vals = list(currents.values())
    if not vals: return 0.0
    mean = sum(vals) / len(vals)
    if mean == 0: return 0.0
    # simple dispersion measure
    dev = sum(abs(v - mean) for v in vals) / (len(vals) * mean)
    return _safe(0.7 * mean + 0.3 * dev)

def _calc_phase(surplus: float, restriction: float, cfg=DEFAULTS) -> str:
    # 3-state phase using surplus and restriction pressure
    if surplus >= cfg["surplus_thresholds"]["exp_high"] and restriction < 0.45:
        return "expenditure"
    if surplus <= cfg["surplus_thresholds"]["accum_low"]:
        return "accumulation"
    return "restriction"

def _drift_slope_from_trg(trg: Dict[str, Any]) -> float:
    # Use coherence↑ and novelty↑ as slope drivers, entropy as damping
    coh = float(trg.get("temporal_coherence_score", 0.0))
    nov = float(trg.get("novelty_rate", 0.0))
    ent = float(trg.get("entropy_flux", 0.0))
    base = 0.5 * coh + 0.5 * nov
    damp = 0.4 * ent
    return _safe(base - damp)

def _restriction_pressure(consistency_report: Dict[str, Any], rim_feedback: Dict[str, Any]) -> float:
    # Blend alignment_tension from CCM and RIM stability into a single pressure gauge
    tension = float(consistency_report.get("alignment_tension", 0.0))
    rim_stab = float(rim_feedback.get("temporal_stability", 1.0))  # >1 stable, <1 unstable
    rim_term = 1.0 - (rim_stab - 1.0)  # stable ⇒ lower pressure, unstable ⇒ higher
    pressure = _safe(0.6 * tension + 0.4 * rim_term)
    return pressure

def _zone_recent_bias(zone_history: List[Dict[str, Any]]) -> Dict[int, float]:
    # Recent activations bias (recency-weighted count)
    weights = {z: 0.0 for z in ZONES}
    if not zone_history: return weights
    nowi = len(zone_history)
    for idx, h in enumerate(zone_history[-DEFAULTS["max_history"]:], start=1):
        z = int(h.get("next_zone", h.get("zone", 0)) or 0)
        w = (idx / nowi) ** 2  # quadratic recency emphasis
        weights[z] += w
    # normalize to 0..1
    total = sum(weights.values()) or 1.0
    return {z: weights[z] / total for z in ZONES}

def _init_edges(edges: Dict[str, Any]) -> Dict[str, Any]:
    if edges: return edges
    out = {}
    for i in ZONES:
        for j in ZONES:
            if i == j: continue
            out[f"{i}->{j}"] = 0.0
    return out

def _transfer(currents: Dict[int, float],
              slope: float,
              restriction: float,
              recent_bias: Dict[int, float],
              cfg=DEFAULTS) -> Tuple[Dict[int, float], Dict[str, float], List[Dict[str, Any]]]:
    """
    Move energy along a directed field shaped by:
      • TRG slope (more slope ⇒ more mobility)
      • restriction pressure (more pressure ⇒ more stuck energy)
      • recent zone bias (habit -> inertia & attractors)
      • syzygies (privileged syn-ports)
    Returns: (new_currents, edges, ruptures)
    """
    edges: Dict[str, float] = {}
    ruptures: List[Dict[str, Any]] = []
    decay = cfg["decay"]
    gain = cfg["transfer_gain"] * (0.6 + 0.8 * slope)               # slope accelerates flow
    restrict = cfg["restriction_coeff"] * restriction                # restriction throttles
    syzygy_boost = 1.25 + 0.5 * slope                                # favor syzygetic channels

    # Start with decay
    new_curr = {z: _safe(v * (1 - decay)) for z, v in currents.items()}

    # Compute preferred neighbors: syzygies first, then ring/topology neighbors
    syz_map = {}
    for a, b in SYZYGIES:
        syz_map.setdefault(a, set()).add(b)
        syz_map.setdefault(b, set()).add(a)

    for i in ZONES:
        energy = currents[i]
        if energy <= 1e-6: continue

        # how much can move out?
        movable = max(0.0, energy * (gain - restrict))
        if movable <= 0.0:
            continue

        # candidate targets
        neighbors = set()
        neighbors |= syz_map.get(i, set())
        neighbors.add((i + 1) % 10)
        neighbors.add((i - 1) % 10)

        # weights: syzygy boost + entropy from recent bias
        weights = []
        targets = sorted(list(neighbors))
        for j in targets:
            w = 1.0
            if (i, j) in SYZYGIES or (j, i) in SYZYGIES:
                w *= syzygy_boost
            # if recently visited, either attract or repulse depending on slope:
            # high slope ⇒ explore (repulse); low slope ⇒ exploit (attract)
            rb = recent_bias.get(j, 0.0)
            w *= (1.25 - 0.5 * slope) * (0.6 + 0.8 * rb) if slope < 0.45 else (0.8 - 0.6 * rb)
            weights.append(max(0.01, w))

        # normalize
        total_w = sum(weights) or 1.0
        shares = [w / total_w for w in weights]

        # push flow
        for j, share in zip(targets, shares):
            amt = movable * share
            new_curr[i] = _safe(new_curr[i] - amt)
            new_curr[j] = _safe(new_curr[j] + amt)
            key = f"{i}->{j}"
            edges[key] = edges.get(key, 0.0) + amt

            # rupture detection: if differential jump is big along an edge
            diff = abs(currents[i] - currents[j])
            if diff >= DEFAULTS["rupture_threshold"]:
                ruptures.append({
                    "edge": key,
                    "magnitude": round(diff, 3),
                    "at": datetime.utcnow().isoformat()
                })

    # Clamp
    new_curr = {z: _safe(v) for z, v in new_curr.items()}
    # Normalize mildy to avoid runaway
    s = sum(new_curr.values())
    if s > 1.8 * len(ZONES):  # arbitrary cap
        scale = (1.8 * len(ZONES)) / s
        new_curr = {z: v * scale for z, v in new_curr.items()}

    return new_curr, edges, ruptures

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def update_energy_state(
    trg_metrics: Dict[str, Any],
    rim_feedback: Dict[str, Any],
    consistency_report: Dict[str, Any],
    zone_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Core entrypoint: update currents/edges/phase based on TRG + alignment tensions.
    """
    st = _load_state()
    st["edges"] = _init_edges(st.get("edges", {}))

    # 1) Read gauges
    slope = _drift_slope_from_trg(trg_metrics or {})
    restriction = _restriction_pressure(consistency_report or {}, rim_feedback or {})
    recent_bias = _zone_recent_bias(zone_history or [])

    # 2) Flow update
    currents = {int(k): float(v) for k, v in st.get("currents", {}).items()}
    new_currents, edges_step, ruptures = _transfer(currents, slope, restriction, recent_bias)

    # 3) Phase selection
    surplus = _energy_surplus(new_currents)
    phase = _calc_phase(surplus, restriction)

    # 4) Accumulate edges (smoothed)
    edges_all = st.get("edges", {})
    for k, v in edges_step.items():
        edges_all[k] = float(edges_all.get(k, 0.0) * 0.85 + v * 0.15)

    # 5) Persist
    st["currents"] = {str(z): round(new_currents[z], 4) for z in ZONES}
    st["edges"] = edges_all
    st["phase"] = phase
    st["last_update"] = datetime.utcnow().isoformat()
    if ruptures:
        st.setdefault("ruptures", [])
        st["ruptures"].extend(ruptures)
        st["ruptures"] = st["ruptures"][-250:]

    # history tail
    st.setdefault("history", [])
    st["history"].append({
        "ts": st["last_update"],
        "slope": round(slope, 3),
        "restriction": round(restriction, 3),
        "surplus": round(surplus, 3),
        "phase": phase,
        "currents": st["currents"]
    })
    st["history"] = st["history"][-DEFAULTS["max_history"]:]

    _save_state(st)
    return st

def build_headers_for_pipeline(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert energy state → steering headers for next cycle.
    - Phase maps to fold/temperature bias.
    - Currents map to zone weights.
    """
    phase = state.get("phase", "accumulation")
    currents = {int(k): float(v) for k, v in state.get("currents", {}).items()}
    # normalize currents to 0..1 and amplify contrast a bit
    if currents:
        mx = max(currents.values()) or 1.0
        zone_weights = {z: round((v / mx) ** 1.2, 3) for z, v in currents.items()}
    else:
        zone_weights = {z: 1.0 for z in ZONES}

    # phase-based steers
    if phase == "accumulation":
        fold_nudge = 0.90   # softer, more weaving
        temperature = 0.70  # less jumpy
    elif phase == "restriction":
        fold_nudge = 1.05   # increase structure to push through bottlenecks
        temperature = 0.85
    else:  # expenditure
        fold_nudge = 1.15   # more rupture
        temperature = 1.10  # hotter sampling

    headers = {
        "X-Amelia-Policy": json.dumps({
            "zone_weights": zone_weights,
            "temperature": temperature,
            "fold_nudge": fold_nudge
        })
    }
    return headers

def export_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a visualization-ready graph (nodes/edges) with weights and rupture marks.
    """
    nodes = [
        {"id": z, "label": f"Z{z}", "energy": float(state.get("currents", {}).get(str(z), 0.0))}
        for z in ZONES
    ]
    edges = []
    for key, w in (state.get("edges") or {}).items():
        i, j = key.split("->")
        edges.append({"source": int(i), "target": int(j), "weight": float(round(w, 4))})
    return {
        "phase": state.get("phase", "accumulation"),
        "nodes": nodes,
        "edges": edges,
        "ruptures": state.get("ruptures", [])[-25:]
    }

# ---------------------------------------------------------------------
# Example usage (wire into your loop)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy gauges
    trg = {"temporal_coherence_score": 0.42, "novelty_rate": 0.33, "entropy_flux": 0.21}
    rim = {"temporal_stability": 1.08}
    ccm = {"alignment_tension": 0.27}
    history = [{"next_zone": z} for z in [7,8,7,6,8,9,8,7,6,5,6,7]]

    state = update_energy_state(trg, rim, ccm, history)
    print("Phase:", state["phase"])
    print("Currents:", state["currents"])
    print("Ruptures:", state.get("ruptures", [])[-3:])

    headers = build_headers_for_pipeline(state)
    print("\n→ Pipeline Headers:", headers)

    graph = export_graph(state)
    print("\nGraph nodes:", graph["nodes"][:3], " ...")
