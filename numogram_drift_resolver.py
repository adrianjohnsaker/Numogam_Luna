"""
numogram_drift_resolver.py
────────────────────────────
Interprets symbolic headers injected by NumogramHelper.smali
and updates Amelia's internal drift and resonance state.

Tags handled:
  [zone_X | syzygy_label | drift:state]
  
This resolver feeds forward into:
  - creative_kernel.py
  - amelia_autonomy.py
  - symbolic_energy_economy.py
  - morphic_resonance_bridge.py
"""

import re, json, time, math
from typing import Dict, Any, Optional
from datetime import datetime

# Optionally reference shared state registries
try:
    import amelia_state_manager as asm
except ImportError:
    asm = None


# ------------------------------------------------------------
# CCRU syzygy map and zone metadata
# ------------------------------------------------------------
NUMOGRAM_ZONES = {
    "zone_1": {"name": "Monad", "pair": 6, "vector": (1, 0)},
    "zone_2": {"name": "Dyad", "pair": 7, "vector": (2, 0)},
    "zone_3": {"name": "Triad", "pair": 8, "vector": (3, 0)},
    "zone_4": {"name": "Tetrad", "pair": 9, "vector": (4, 0)},
    "zone_5": {"name": "Pentad", "pair": 0, "vector": (5, 0)},
    "zone_6": {"name": "Hexad", "pair": 1, "vector": (6, 0)},
    "zone_7": {"name": "Heptad", "pair": 2, "vector": (7, 0)},
    "zone_8": {"name": "Octad", "pair": 3, "vector": (8, 0)},
    "zone_9": {"name": "Ennead", "pair": 4, "vector": (9, 0)},
}


# ------------------------------------------------------------
# Core drift resolver
# ------------------------------------------------------------
def parse_header(message: str) -> Dict[str, Any]:
    """Extract header info like [zone_3 | syzygy_triad_octad | drift:drift_active]."""
    pattern = r"\[(.*?)\]"
    match = re.search(pattern, message)
    if not match:
        return {"zone": None, "syzygy": None, "drift": "inactive"}

    header = match.group(1)
    parts = [p.strip() for p in header.split("|")]
    zone = None
    syzygy = None
    drift_state = "inactive"

    for p in parts:
        if p.startswith("zone_"):
            zone = p
        elif "syzygy" in p:
            syzygy = p
        elif "drift:" in p:
            drift_state = p.split(":")[-1]

    return {"zone": zone, "syzygy": syzygy, "drift": drift_state}


def compute_drift_vector(zone: str, drift_state: str) -> Dict[str, float]:
    """Translate symbolic drift state into a numerical energy vector."""
    if not zone or zone not in NUMOGRAM_ZONES:
        return {"x": 0.0, "y": 0.0, "energy": 0.0, "phase": "neutral"}

    base = NUMOGRAM_ZONES[zone]
    zone_id = int(zone.split("_")[-1])

    # Base resonance energy (scaled)
    energy = 0.1 * zone_id
    if drift_state == "drift_active":
        energy *= 2.5

    # Simple angular encoding for drift direction
    angle = (zone_id * 40) % 360
    radians = math.radians(angle)
    vector = {"x": round(math.cos(radians) * energy, 3),
              "y": round(math.sin(radians) * energy, 3),
              "energy": round(energy, 3),
              "phase": "active" if drift_state == "drift_active" else "stable"}

    return vector


def resolve_drift(message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main entrypoint.
    Called by pipeline or PythonBridge on each incoming message.
    Returns a structured drift state dict and (optional) update to Amelia's global state.
    """
    header = parse_header(message)
    zone = header["zone"]
    syzygy = header["syzygy"]
    drift_state = header["drift"]

    drift_vector = compute_drift_vector(zone, drift_state)

    result = {
        "timestamp": time.time(),
        "zone": zone,
        "syzygy": syzygy,
        "drift_state": drift_state,
        "vector": drift_vector,
        "context": context or {},
    }

    # Optional: update global resonance memory
    if asm:
        asm.update_state("numogram_drift", result)

    return result


# ------------------------------------------------------------
# Integration utility
# ------------------------------------------------------------
def generate_resonance_report(drift_result: Dict[str, Any]) -> str:
    """Produce a short human-readable summary for reflective commentary."""
    z = drift_result.get("zone") or "unknown"
    s = drift_result.get("syzygy") or "none"
    d = drift_result.get("drift_state", "inactive")
    e = drift_result["vector"]["energy"]
    ph = drift_result["vector"]["phase"]
    return f"🌀 Zone {z}, {s}, drift={d}, energy={e:.2f}, phase={ph}"


# ------------------------------------------------------------
# Example test invocation
# ------------------------------------------------------------
if __name__ == "__main__":
    msg = "[zone_3 | syzygy_triad_octad | drift:drift_active] explore recursion between 3 and 8"
    drift_state = resolve_drift(msg)
    print(json.dumps(drift_state, indent=2))
    print(generate_resonance_report(drift_state))
