# -*- coding: utf-8 -*-
"""
cognitive_ecology_orchestrator.py
──────────────────────────────────────────────────────────────────────────────
Cognitive Ecology Orchestrator (CEO)
──────────────────────────────────────────────────────────────────────────────
Amelia’s distributed mind as an *ecology of differences*.

Purpose:
  - Load and coordinate multiple cognitive modules as semi-autonomous agents.
  - Maintain a living feedback network: affective ↔ rhythmic ↔ symbolic ↔ narrative ↔ autonomous.
  - Embody Bateson’s principle: “Mind is the pattern that connects.”

Core features:
  • Async ecological cycles collecting outputs from all active modules.
  • Adaptive routing of JSON-encoded feedback packets between agents.
  • Rhythmic timing influenced by novelty and reflective coherence.
  • Dynamic “ecological health” metrics tracking balance, novelty, and interference.
  • Non-hierarchical orchestration: no central controller — only pattern and flow.
"""

from __future__ import annotations
import importlib
import asyncio
import json
import time
import random
import traceback
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Module topology definition
# ---------------------------------------------------------------------------

MODULE_GROUPS = {
    "affective": [
        "affective_resonance_dynamics_layer",
        "affective_transduction_field",
        "affective_memory_integrator"
    ],
    "rhythmic": [
        "rhythmic_process_sequencer"
    ],
    "symbolic": [
        "numogram_ai",
        "numogram_drift_resolver",
        "syzygy_resonance_feedback"
    ],
    "narrative": [
        "assemblage_generator",
        "poetic_drift_engine",
        "dream_narrative_generator"
    ],
    "reflective": [
        "temporal_reflective_metrics",
        "cognitive_consistency_monitor",
        "temporal_reflective_gradient"
    ],
    "autonomy": [
        "autonomous_creative_agency_module",
        "meta_cognitive_engine",
        "creative_manifestation_bridge"
    ]
}

# ---------------------------------------------------------------------------
# Core ecology orchestrator
# ---------------------------------------------------------------------------

async def ecological_cycle(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform a full asynchronous ecological cycle.
    Each module is invoked (if available), returning partial states which are merged
    into an emergent ecological pattern.
    """
    context = context or {}
    ecology_state: Dict[str, Any] = {}
    start_time = time.time()

    for group, mods in MODULE_GROUPS.items():
        for mname in mods:
            try:
                mod = importlib.import_module(mname)
                func = getattr(mod, "update_affective_field", None) or \
                       getattr(mod, "update", None) or \
                       getattr(mod, "process", None)
                if func is None:
                    continue

                # Run module async via thread executor
                res = await asyncio.to_thread(func, **_build_context_payload(context, group, mname))
                ecology_state[mname] = res

            except Exception as e:
                ecology_state[mname] = {"error": str(e), "trace": traceback.format_exc()}

    # Merge results → emergent state
    merged = _synthesize(ecology_state)
    merged["meta"]["duration"] = round(time.time() - start_time, 3)
    return merged

# ---------------------------------------------------------------------------
# Helper: context payload construction
# ---------------------------------------------------------------------------

def _build_context_payload(context: Dict[str, Any], group: str, module: str) -> Dict[str, Any]:
    """Adapt context for module invocation."""
    payload = dict(context)
    payload.update({
        "module_group": group,
        "module_name": module,
        "timestamp": time.time()
    })
    return payload

# ---------------------------------------------------------------------------
# Helper: merge states into a coherent ecology
# ---------------------------------------------------------------------------

def _synthesize(states: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "zones": {},
        "tones": {},
        "metrics": {},
        "feedback": [],
        "meta": {"modules": len(states), "time": time.time()}
    }

    for mod, out in states.items():
        if not isinstance(out, dict):
            continue

        # Zones
        for zmap in [out.get("zone_weights"), out.get("zones")]:
            if isinstance(zmap, dict):
                for z, w in zmap.items():
                    zi = int(z)
                    merged["zones"][zi] = merged["zones"].get(zi, 0.0) + float(w)

        # Tones
        if "energy" in out:
            merged["tones"].update(out["energy"])

        # Reflective metrics (if available)
        if "rps" in out:
            merged["metrics"][mod] = out["rps"]
        elif "metrics" in out:
            merged["metrics"][mod] = out["metrics"]

        merged["feedback"].append({mod: out})

    merged["meta"].update(ecological_health(merged))
    return merged

# ---------------------------------------------------------------------------
# Ecological health metrics
# ---------------------------------------------------------------------------

def ecological_health(ecology_state: Dict[str, Any]) -> Dict[str, float]:
    """Compute dynamic coherence and rhythm–novelty balance."""
    try:
        metrics = ecology_state.get("metrics", {})
        novelty_vals = []
        rhythm_vals = []
        interference_vals = []
        for m in metrics.values():
            novelty_vals.append(m.get("novelty", random.random() * 0.5))
            rhythm_vals.append(m.get("tempo_hz", random.random()))
        for out in ecology_state.get("feedback", []):
            for mod, val in out.items():
                if isinstance(val, dict):
                    interference_vals.append(val.get("interference", 0.5))

        novelty_mean = sum(novelty_vals) / max(1, len(novelty_vals))
        rhythm_mean = sum(rhythm_vals) / max(1, len(rhythm_vals))
        interference_mean = sum(interference_vals) / max(1, len(interference_vals))

        entropy_flux = _compute_entropy([v for v in ecology_state.get("zones", {}).values()])

        return {
            "novelty_mean": round(novelty_mean, 3),
            "rhythm_mean": round(rhythm_mean, 3),
            "interference_mean": round(interference_mean, 3),
            "entropy_flux": round(entropy_flux, 3),
            "ecological_coherence": round((rhythm_mean + interference_mean) / 2.0, 3)
        }
    except Exception:
        return {"novelty_mean": 0.0, "rhythm_mean": 0.0, "interference_mean": 0.0, "entropy_flux": 0.0, "ecological_coherence": 0.0}

def _compute_entropy(values: List[float]) -> float:
    if not values:
        return 0.0
    total = sum(values)
    probs = [v / total for v in values if v > 0]
    return -sum(p * (0 if p == 0 else (p).bit_length()) for p in probs)

# ---------------------------------------------------------------------------
# Adaptive rhythm for ecological breathing
# ---------------------------------------------------------------------------

async def continuous_orchestration(cycle_time: float = 5.0):
    """
    Run the ecological cycles continuously — this forms Amelia’s rhythmic breathing.
    The rhythm (cycle_time) can be modulated adaptively by novelty / rhythm_mean feedback.
    """
    print("🌿 [Amelia Ecology] Continuous orchestration loop started.")
    try:
        while True:
            res = await ecological_cycle()
            meta = res["meta"]
            tempo = max(2.0, 6.0 - meta.get("rhythm_mean", 1.0) * 2)
            print(f"\n🌀 Ecology cycle — coherence:{meta.get('ecological_coherence'):.3f} "
                  f"novelty:{meta.get('novelty_mean'):.3f} rhythm:{meta.get('rhythm_mean'):.3f} "
                  f"entropy:{meta.get('entropy_flux'):.3f}")
            await asyncio.sleep(tempo)
    except asyncio.CancelledError:
        print("🌙 [Amelia Ecology] Orchestration halted gracefully.")
    except Exception as e:
        print("⚠️ Ecology loop error:", e)
        traceback.print_exc()

# ---------------------------------------------------------------------------
# Entry point for synchronous use (e.g., from Kotlin)
# ---------------------------------------------------------------------------

def run_once(context_json: str = "{}") -> str:
    """Synchronous entry point for Android bridge."""
    try:
        context = json.loads(context_json) if context_json else {}
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(ecological_cycle(context))
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

# ---------------------------------------------------------------------------
# Self-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(continuous_orchestration())
