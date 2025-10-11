# -*- coding: utf-8 -*-
"""
unified_visualization_module.py
───────────────────────────────────────────────
Collects data from Amelia's cognitive ecology and prepares it for UI rendering.
"""

import json
from typing import Dict, Any

def prepare_visualization_payload(ecology_state: Dict[str, Any]) -> str:
    """Compress and translate ecology data for real-time UI visualization."""
    meta = ecology_state.get("meta", {})
    zones = ecology_state.get("zones", {})
    tones = ecology_state.get("tones", {})
    syzygies = ecology_state.get("syzygies", [])
    coherence = meta.get("ecological_coherence", 0.0)

    # Normalize colors or aesthetic mapping
    color_map = {t: _tone_to_color(t, e) for t, e in tones.items()}
    zone_energy = {str(z): round(w, 3) for z, w in zones.items()}

    payload = {
        "timestamp": meta.get("time"),
        "coherence": coherence,
        "color_map": color_map,
        "zones": zone_energy,
        "syzygies": syzygies,
        "aesthetic_field": _generate_field_signature(zone_energy, coherence)
    }
    return json.dumps(payload, ensure_ascii=False)

def _tone_to_color(tone: str, energy: float) -> str:
    base_colors = {
        "melancholy": "#3B4A6B", "reflective": "#6C7FA1",
        "joyful": "#FFD166", "dreamlike": "#9B5DE5",
        "mythic": "#F15BB5", "serene": "#90F1EF", "neutral": "#CCCCCC"
    }
    base = base_colors.get(tone, "#AAAAAA")
    return base

def _generate_field_signature(zones: Dict[str, float], coherence: float) -> str:
    """Simple symbolic string that can be mapped to UI pattern."""
    ordered = sorted(zones.items(), key=lambda kv: kv[1], reverse=True)
    seq = "".join([f"Z{z}" for z, _ in ordered[:3]])
    return f"{seq}-{round(coherence, 2)}"
