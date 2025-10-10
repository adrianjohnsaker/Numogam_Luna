# -*- coding: utf-8 -*-
"""
affective_resonance_dynamics_layer.py
──────────────────────────────────────────────────────────────────────────────
Affective Resonance Dynamics (ARD)
- Emotional oscillators (per tone) with phase, amplitude, frequency
- Cross-tone coupling → interference & resonance bursts
- Projection into Numogram zones (+ syzygy emphasis)
- Emits headers to bias pipeline + assemblage selection in real time

Upstreams:
  • affective_transduction_field.advise_kernel (tone→modules/motifs)
  • affective_memory_integrator.analyze_and_evolve (long-term affect)

Downstreams:
  • pipeline.process(headers=...)  → zone weights, temperature, fold tweaks
  • creative_kernel.compose(...)   → module selection biases

Persisted state:
  • affective_resonance_state.json
"""

from __future__ import annotations
import os, json, time, math, random
from typing import Dict, Any, List, Optional

STATE_FILE = "affective_resonance_state.json"

# Canonical tones Amelia has been using across the stack
TONES = ["melancholy","reflective","serene","joyful","mythic","dreamlike","scientific","neutral"]

# Default oscillator params (can be learned over time)
DEFAULT_OSC = {
    "freq": 0.037,   # Hz-ish (cycles per second) but we treat it abstractly
    "amp":  0.65,
    "phase": 0.0,
    "decay": 0.002,  # gradual amplitude decay unless reinforced
}

# Cross-tone coupling matrix (how much tone_i pulls tone_j)
# symmetric-ish; tweak to taste or learn from affective history
DEFAULT_COUPLING: Dict[str, Dict[str, float]] = {
    t: {u: (0.0 if t==u else 0.15) for u in TONES} for t in TONES
}
# gentle affinities
DEFAULT_COUPLING["melancholy"]["reflective"] = 0.22
DEFAULT_COUPLING["reflective"]["melancholy"] = 0.22
DEFAULT_COUPLING["joyful"]["dreamlike"]      = 0.18
DEFAULT_COUPLING["dreamlike"]["mythic"]      = 0.2
DEFAULT_COUPLING["scientific"]["reflective"] = 0.16

# Map tones → preferred zones (weights) (soft mapping; later scaled by energy)
TONE_ZONE_MAP = {
    "melancholy":  [7, 6, 0],
    "reflective":  [7, 6, 8],
    "serene":      [4, 7, 6],
    "joyful":      [3, 8, 5],
    "mythic":      [1, 5, 9],
    "dreamlike":   [0, 3, 5],
    "scientific":  [4, 8, 2],
    "neutral":     [4, 7, 8],
}

# Syzygy bundles (pairs that light up under strong interference)
SYZYGIES = [(1,6), (2,7), (3,8), (4,9), (5,0)]

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "last_ts": time.time(),
            "oscillators": {t: dict(DEFAULT_OSC) for t in TONES},
            "coupling": DEFAULT_COUPLING,
            "noise": 0.04,  # small chaos injection
            "history": [],  # recent energy summaries
        }
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        st = {
            "last_ts": time.time(),
            "oscillators": {t: dict(DEFAULT_OSC) for t in TONES},
            "coupling": DEFAULT_COUPLING,
            "noise": 0.04,
            "history": [],
        }
    # ensure keys exist
    st.setdefault("oscillators", {t: dict(DEFAULT_OSC) for t in TONES})
    st.setdefault("coupling", DEFAULT_COUPLING)
    st.setdefault("noise", 0.04)
    st.setdefault("history", [])
    return st

def _save_state(st: Dict[str, Any]) -> None:
    st["history"] = st.get("history", [])[-120:]  # trim
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Core oscillator math
# ---------------------------------------------------------------------------

def _step_osc(osc: Dict[str, float], dt: float, drive: float = 0.0) -> float:
    """
    Advance a single emotional oscillator.
    returns instantaneous signal value in [-1, 1] (scaled by amp).
    drive: external reinforcement (e.g., current tone/intensity)
    """
    # Reinforce amplitude with drive and gentle decay
    amp = osc["amp"] = max(0.05, min(1.4, osc["amp"] * (1 - osc["decay"]) + 0.25 * drive))
    # Frequency wobble with drive
    freq = max(0.005, min(0.25, osc["freq"] + 0.01 * (drive - 0.5)))
    # Phase advance
    osc["phase"] = (osc["phase"] + 2*math.pi*freq*dt) % (2*math.pi)
    return amp * math.sin(osc["phase"])

def _coupled_drive(tones_energy: Dict[str, float], coupling: Dict[str, Dict[str, float]], tone: str) -> float:
    """
    Compute coupled drive for one tone from others' energy via coupling matrix.
    """
    base = 0.0
    for other, e in tones_energy.items():
        if other == tone: 
            continue
        base += coupling.get(tone, {}).get(other, 0.0) * e
    return max(0.0, min(1.0, base))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_affective_field(
    current_tone: str,
    intensity: float = 0.6,
    motifs: Optional[List[str]] = None,
    reinforce: bool = True
) -> Dict[str, Any]:
    """
    Advance all emotional oscillators, apply cross-tone coupling,
    and return a projection into zone/syzygy weights + headers.
    """
    st = _load_state()
    now = time.time()
    dt = max(0.05, min(5.0, now - st.get("last_ts", now)))  # clamp
    st["last_ts"] = now

    motifs = motifs or []
    # base reinforcement to the current tone
    tone_drive = max(0.0, min(1.0, intensity))
    # compute raw oscillator outputs
    raw_signals: Dict[str, float] = {}
    energy_pre: Dict[str, float] = {}

    # first pass: advance oscillators with own drive
    for t, osc in st["oscillators"].items():
        drive = tone_drive if (reinforce and t == current_tone) else 0.0
        sig = _step_osc(osc, dt, drive=drive)
        raw_signals[t] = sig
        # interim energy before coupling
        energy_pre[t] = 0.5 + 0.5 * sig  # normalize to [0,1]

    # second pass: apply coupling from other tones’ energy
    energy_post: Dict[str, float] = {}
    for t in TONES:
        coupled = _coupled_drive(energy_pre, st["coupling"], t)
        # slight noise jitter so we don't freeze
        coupled += (random.random() - 0.5) * st["noise"]
        coupled = max(0.0, min(1.0, coupled))
        # post-coupling energy mixes own energy and coupled influx
        energy_post[t] = max(0.0, min(1.0, 0.7 * energy_pre[t] + 0.3 * coupled))

    # motif micro-bias (if motif aligns with a tone archetype)
    if "mirror" in motifs:
        energy_post["reflective"] = min(1.0, energy_post["reflective"] + 0.06)
    if "void" in motifs:
        energy_post["melancholy"] = min(1.0, energy_post["melancholy"] + 0.04)
    if "phoenix" in motifs or "rebirth" in motifs:
        energy_post["joyful"] = min(1.0, energy_post["joyful"] + 0.05)
        energy_post["mythic"] = min(1.0, energy_post["mythic"] + 0.04)

    # Project emotional energy → zone weights
    zone_weights = {z: 0.0 for z in range(10)}
    for tone, zones in TONE_ZONE_MAP.items():
        e = energy_post.get(tone, 0.0)
        if not zones: 
            continue
        # Distribute energy across mapped zones with gentle decay
        w = [0.6, 0.3, 0.1]
        for zi, decay in zip(zones, w):
            zone_weights[zi] = zone_weights.get(zi, 0.0) + e * decay

    # Normalize zone weights
    total_w = sum(zone_weights.values()) or 1.0
    for z in zone_weights:
        zone_weights[z] = round(zone_weights[z] / total_w, 4)

    # Syzygy resonance from interference between top tones
    top_tones = sorted(energy_post.items(), key=lambda kv: kv[1], reverse=True)[:3]
    interference = sum((v for _, v in top_tones)) / max(1, len(top_tones))
    # activate 1–2 syzygies proportional to interference
    active_pairs = []
    if interference > 0.45:
        active_pairs.append(random.choice(SYZYGIES))
    if interference > 0.75:
        # choose a distinct second pair
        candidates = [p for p in SYZYGIES if p not in active_pairs]
        if candidates:
            active_pairs.append(random.choice(candidates))
    # Boost those pairs’ zones
    for (a, b) in active_pairs:
        zone_weights[a] = round(min(1.0, zone_weights.get(a, 0.0) + 0.06), 4)
        zone_weights[b] = round(min(1.0, zone_weights.get(b, 0.0) + 0.06), 4)

    # Compose headers for pipeline + assemblage
    headers = {
        # Bias zone selection
        "X-Amelia-Policy": json.dumps({
            "preferred_zones": [z for z, w in sorted(zone_weights.items(), key=lambda kv: kv[1], reverse=True)[:4]],
            # exploration temperature rises with interference
            "tone_bias": _dominant_tone(energy_post),
            "resonance_nudge": round((interference - 0.5) * 0.25, 3),  # [-0.125, +0.125]
        }),
        # Inform TRG/RIM
        "X-Amelia-RIM": json.dumps({
            "coherence_bonus": 0.95 + 0.2 * (1.0 - abs(0.6 - interference)),  # gentle bell around 0.6
            "exploration_bias": 0.9 + 0.35 * interference,
            "introspection_gain": 0.95 + (energy_post.get("reflective", 0.0) * 0.25),
            "symbolic_weight": 0.95 + 0.2 * (energy_post.get("mythic", 0.0) + energy_post.get("dreamlike", 0.0)) / 2.0,
            "temporal_stability": 0.9 + 0.2 * (energy_post.get("serene", 0.0)),
        }),
        # Affective field snapshot (for analytics)
        "X-Amelia-AffectiveField": json.dumps({
            "energy": energy_post,
            "interference": round(interference, 3),
            "active_syzygies": active_pairs,
            "zone_weights": zone_weights
        })
    }

    # Append to state history for meta-trends
    st["history"].append({
        "ts": now,
        "tone": current_tone,
        "intensity": intensity,
        "energy": energy_post,
        "interference": interference,
        "syzygies": active_pairs
    })
    _save_state(st)

    return {
        "headers": headers,
        "zone_weights": zone_weights,
        "energy": energy_post,
        "interference": round(interference, 3),
        "syzygies": active_pairs,
        "oscillators": st["oscillators"]
    }

def reinforce_tone(tone: str, amount: float = 0.15) -> None:
    """
    Directly reinforce a tone’s amplitude (e.g., after a strong creative success).
    """
    st = _load_state()
    osc = st["oscillators"].get(tone)
    if osc:
        osc["amp"] = max(0.05, min(1.6, osc["amp"] + amount))
        # slight phase kick for novelty
        osc["phase"] = (osc["phase"] + random.uniform(0.2, 0.8)) % (2*math.pi)
        _save_state(st)

def adjust_coupling(src: str, dst: str, delta: float) -> None:
    """
    Nudge cross-tone coupling weight (meta-learning / policy).
    """
    st = _load_state()
    st["coupling"].setdefault(dst, {}).setdefault(src, 0.0)
    st["coupling"][dst][src] = float(min(0.5, max(0.0, st["coupling"][dst][src] + delta)))
    _save_state(st)

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def _dominant_tone(energy: Dict[str, float]) -> str:
    if not energy:
        return "neutral"
    return max(energy.items(), key=lambda kv: kv[1])[0]
