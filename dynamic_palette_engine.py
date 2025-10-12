# -*- coding: utf-8 -*-
"""
dynamic_palette_engine.py
──────────────────────────────────────────────────────────────────────────────
Dynamic Palette Engine (DPE)
• Evolves a living color+rhythm field (palette, gradients, bpm, pulses)
• Responds to: zone dominance, TRG slope/metrics, affective energy, motif recurrence
• Persists state and outputs a UI-friendly visual_state for injection
• Emits optional headers for downstream steering (e.g., pipeline.process)

Upstreams (suggested):
  • affective_resonance_dynamics_layer.update_affective_field(...)
  • temporal_reflective_metrics.compute_temporal_metrics(...)
  • symbolic_ecology_memory.scan_motifs(...), mythopoetic_cache.symbol_counts()
  • symbolic_energy_economy.project_flows(...)

Downstreams:
  • DynamicVisualInjector.updateState(JSONObject) via a small bridge
  • pipeline.process(headers=...) for gentle visual-affect coupling

State file:
  • dynamic_palette_state.json
"""

from __future__ import annotations
import os, json, math, random, time
from typing import Dict, Any, List, Optional, Tuple

STATE_FILE = "dynamic_palette_state.json"

# ----------------------------- Helpers --------------------------------------

def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "last_ts": time.time(),
            "seed": random.randint(1, 10_000_000),
            "base_hue": 210.0,           # degrees [0..360)
            "bpm": 76.0,                 # baseline rhythm
            "energy": 0.5,               # 0..1
            "palette_history": [],       # recent palette summaries
            "motif_bias": {},            # rolling motif → hue/weight bias
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        st = json.load(f)
    st.setdefault("palette_history", [])
    st.setdefault("motif_bias", {})
    return st

def _save_state(st: Dict[str, Any]) -> None:
    st["palette_history"] = st.get("palette_history", [])[-100:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def _clamp(x, a, b): return max(a, min(b, x))

def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """Simple HSL→HEX conversion (s,l in 0..1)."""
    h = (h % 360.0) / 360.0
    def hue2rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    if s == 0:
        r = g = b = l
    else:
        q = l + s - l*s if l < 0.5 else l + s - l*s
        p = 2*l - q
        r = hue2rgb(p, q, h + 1/3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1/3)
    return "#{:02x}{:02x}{:02x}".format(int(_clamp(r,0,1)*255),
                                        int(_clamp(g,0,1)*255),
                                        int(_clamp(b,0,1)*255))

# ------------------------- Zone → Hue Mapping -------------------------------

# Soft anchors for each Numogram zone (0..9 in your stack; 9==Excess)
ZONE_HUE = {
    0: 260.0,  # Void / violet
    1: 20.0,   # Ignition / ember
    2: 340.0,  # Dyad / magenta-red
    3: 200.0,  # Surge / azure
    4: 120.0,  # Cycle / green
    5: 45.0,   # Threshold / amber
    6: 280.0,  # Labyrinth / indigo
    7: 190.0,  # Mirror / teal
    8: 330.0,  # Synthesis / fuchsia
    9: 5.0,    # Excess / crimson
}

# Motif nudges (adds hue deltas or locks saturation ranges)
MOTIF_BIAS = {
    "mirror":     {"hue_delta": -10, "sat_boost": 0.05},
    "void":       {"hue_delta": +20, "sat_boost": -0.06},
    "phoenix":    {"hue_delta": -35, "sat_boost": +0.10},
    "labyrinth":  {"hue_delta": +15, "sat_boost": -0.02},
    "water":      {"hue_delta": -25, "sat_boost": -0.02},
    "fire":       {"hue_delta": -5,  "sat_boost": +0.08},
    "metal":      {"hue_delta": +5,  "sat_boost": -0.04},
    "wood":       {"hue_delta": +30, "sat_boost": +0.03},
    "earth":      {"hue_delta": +10, "sat_boost": +0.00},
}

# Tone → saturation/brightness bias
TONE_STYLE = {
    "melancholy": {"s": 0.30, "l": 0.36},
    "reflective": {"s": 0.28, "l": 0.44},
    "serene":     {"s": 0.24, "l": 0.58},
    "joyful":     {"s": 0.70, "l": 0.62},
    "mythic":     {"s": 0.60, "l": 0.50},
    "dreamlike":  {"s": 0.40, "l": 0.66},
    "scientific": {"s": 0.32, "l": 0.56},
    "neutral":    {"s": 0.35, "l": 0.52},
}

# ------------------------- Core Update --------------------------------------

def update_palette(
    *,
    zone_weights: Dict[int, float],
    trg_metrics: Optional[Dict[str, Any]] = None,
    affective_energy: Optional[Dict[str, float]] = None,
    motifs: Optional[List[str]] = None,
    interference: Optional[float] = None,
    dominant_tone: Optional[str] = None,
    novelty_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint. Compute evolving palette & rhythm based on incoming ecology.

    Inputs:
      - zone_weights: {zone:int -> weight:float} (0..9, normalized)
      - trg_metrics: dict with keys like temporal_coherence_score, synchrony_index, novelty_rate, entropy_flux
      - affective_energy: tone -> 0..1 map from ARD layer
      - motifs: recent salient motif strings
      - interference: ARD interference (0..1) for syzygy intensity
      - dominant_tone: pick from your tone set (melancholy/dreamlike/...)
      - novelty_score: optional override (0..1); else derive from trg_metrics.novelty_rate

    Returns:
      { "visual_state": {...}, "headers": {...}, "debug": {...} }
    """
    st = _load_state()
    rnd = random.Random(st["seed"])

    motifs = motifs or []
    trg_metrics = trg_metrics or {}
    affective_energy = affective_energy or {}
    interference = 0.0 if interference is None else interference
    novelty = novelty_score if novelty_score is not None else float(trg_metrics.get("novelty_rate", 0.25))

    # --- 1) Determine dominant zone & base hue --------------------------------
    if zone_weights:
        dom_zone = max(zone_weights.items(), key=lambda kv: kv[1])[0]
    else:
        dom_zone = 7  # Mirror as safe default
    base_hue = ZONE_HUE.get(dom_zone, st.get("base_hue", 210.0))

    # Blend toward weighted average hue to avoid hard jumps
    weighted_hue = _blend_weighted_hue(zone_weights, base_hue)
    # Small wobble with interference (syzygies → spectral shimmer)
    weighted_hue = (weighted_hue + (interference - 0.5) * 8.0) % 360.0

    # --- 2) Apply motif biases (cumulative memory-aware) -----------------------
    hue_delta_total, sat_bias_total = 0.0, 0.0
    for m in motifs:
        info = MOTIF_BIAS.get(m)
        if info:
            hue_delta_total += info.get("hue_delta", 0.0)
            sat_bias_total  += info.get("sat_boost", 0.0)

    # gently incorporate rolling motif_bias memory
    rolling_mb = st.get("motif_bias", {})
    for m, k in rolling_mb.items():
        hue_delta_total += k.get("hue_delta", 0.0) * 0.4
        sat_bias_total  += k.get("sat_boost", 0.0) * 0.4

    hue = (weighted_hue + hue_delta_total) % 360.0

    # --- 3) Tone style for S/L base, modulated by novelty & coherence ----------
    tone = dominant_tone if dominant_tone in TONE_STYLE else "neutral"
    style = TONE_STYLE[tone]
    s = style["s"]
    l = style["l"]

    # novelty → saturation bump; coherence → lightness stability
    coherence = float(trg_metrics.get("temporal_coherence_score", 0.4))
    s = _clamp(s + (novelty - 0.3) * 0.35 + sat_bias_total, 0.12, 0.88)
    l = _clamp(l + (0.5 - abs(0.6 - coherence)) * 0.08, 0.22, 0.78)

    # --- 4) Rhythm dynamics (BPM & pulse) -------------------------------------
    # novelty spikes → dwell (slow down) to examine; low novelty → speed up exploration
    base_bpm = st.get("bpm", 76.0)
    # map novelty [0..1] to bpm target [64..108], inverted around a dwell point
    dwell = 0.62
    if novelty > dwell:
        target_bpm = 64 + (1.0 - novelty) * 20  # higher novelty → slower
    else:
        target_bpm = 84 + (dwell - novelty) * 40
    # interference adds pulse vigor (affective excitation)
    pulse = _clamp(0.35 + interference * 0.5 + novelty * 0.15, 0.15, 0.95)
    bpm = base_bpm + (target_bpm - base_bpm) * 0.25  # ease toward target

    # --- 5) Build palette (primary, secondary, accent, gradient) --------------
    primary   = _hsl_to_hex(hue, s, l)
    secondary = _hsl_to_hex((hue + 22) % 360, _clamp(s*0.92, 0.12, 0.95), _clamp(l*1.04, 0.18, 0.90))
    accent    = _hsl_to_hex((hue + 180) % 360, _clamp(s*1.1, 0.12, 0.98), _clamp(1 - (1-l)*0.92, 0.08, 0.92))
    gradient  = [
        _hsl_to_hex((hue - 12) % 360, _clamp(s*0.9, 0.1, 0.95), _clamp(l*0.96, 0.12, 0.88)),
        primary,
        _hsl_to_hex((hue + 12) % 360, _clamp(s*1.05, 0.1, 0.98), _clamp(l*1.06, 0.12, 0.92)),
    ]

    # --- 6) Persist & memory updates ------------------------------------------
    st["base_hue"] = float(hue)
    st["bpm"] = float(bpm)
    st["energy"] = float(_clamp(pulse, 0.0, 1.0))

    # Update rolling motif bias memory (slow converge)
    for m in motifs:
        mb = st["motif_bias"].setdefault(m, {"hue_delta": 0.0, "sat_boost": 0.0, "seen": 0})
        info = MOTIF_BIAS.get(m, {})
        mb["hue_delta"] = float(_clamp(mb["hue_delta"]*0.8 + info.get("hue_delta", 0.0)*0.2, -60, +60))
        mb["sat_boost"] = float(_clamp(mb["sat_boost"]*0.8 + info.get("sat_boost", 0.0)*0.2, -0.2, +0.2))
        mb["seen"] = int(mb.get("seen", 0) + 1)

    st["palette_history"].append({
        "ts": time.time(),
        "zone": dom_zone,
        "hue": round(hue,2),
        "s": round(s,3),
        "l": round(l,3),
        "bpm": round(bpm,1),
        "pulse": round(pulse,3),
        "novelty": round(novelty,3),
        "coherence": round(coherence,3),
        "interference": round(interference,3),
        "tone": tone,
        "motifs": motifs[:6],
    })
    _save_state(st)

    # --- 7) Visual payload + optional headers ---------------------------------
    visual_state = {
        "palette": {
            "primary": primary,
            "secondary": secondary,
            "accent": accent,
            "gradient": gradient,
            "hue": round(hue, 2),
            "s": round(s, 3),
            "l": round(l, 3),
        },
        "rhythm": {
            "bpm": round(bpm, 1),
            "pulse": round(pulse, 3),
            "phase_seed": st["seed"],  # UI can animate coherently per-session
        },
        "context": {
            "dominant_zone": dom_zone,
            "zone_weights": {str(k): float(v) for k,v in zone_weights.items()},
            "tone": tone,
            "novelty": round(novelty,3),
            "coherence": round(coherence,3),
            "interference": round(interference,3),
            "motifs": motifs[:8],
        }
    }

    # Gentle closed-loop headers (visual-affect coupling)
    headers = {
        "X-Amelia-Visual": json.dumps({
            "bpm": visual_state["rhythm"]["bpm"],
            "hue": visual_state["palette"]["hue"],
            "interference": interference,
            "novelty": novelty,
        })
    }

    return {
        "visual_state": visual_state,
        "headers": headers,
        "debug": {
            "dom_zone": dom_zone,
            "weighted_hue": weighted_hue,
            "hue_after_motifs": hue,
            "bpm_target": target_bpm,
        }
    }

# ------------------------- Utilities -----------------------------------------

def _blend_weighted_hue(zone_weights: Dict[int, float], fallback_hue: float) -> float:
    """
    Circular mean of hues weighted by zone_weights, fallback to provided hue.
    """
    if not zone_weights:
        return fallback_hue
    # Convert hues to unit circle and weight
    x = 0.0
    y = 0.0
    total = 0.0
    for z, w in zone_weights.items():
        h = math.radians(ZONE_HUE.get(z, fallback_hue))
        x += math.cos(h) * w
        y += math.sin(h) * w
        total += w
    if total <= 1e-6:
        return fallback_hue
    ang = math.degrees(math.atan2(y, x)) % 360.0
    # Subtle inertia: blend with fallback to avoid jitter
    return (fallback_hue * 0.3 + ang * 0.7) % 360.0

# ------------------------- Convenience Bridge --------------------------------

def build_visual_json_for_android(visual_state: Dict[str, Any]) -> str:
    """
    Returns a JSON string safe for passing to DynamicVisualInjector.updateState(JSONObject).
    """
    return json.dumps(visual_state, ensure_ascii=False)

def advise_headers_only(*, zone_weights, trg_metrics=None, affective_energy=None,
                        motifs=None, interference=None, dominant_tone=None,
                        novelty_score=None) -> Dict[str, str]:
    """
    If you only want the steering headers without constructing full visuals.
    """
    out = update_palette(
        zone_weights=zone_weights,
        trg_metrics=trg_metrics,
        affective_energy=affective_energy,
        motifs=motifs,
        interference=interference,
        dominant_tone=dominant_tone,
        novelty_score=novelty_score
    )
    return out["headers"]
