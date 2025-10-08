# -*- coding: utf-8 -*-
"""
Glorious Expenditure Engine (GEE)
=================================
Detects narrative surplus and initiates creative expenditure through
Amelia's symbolic, poetic, and visual systems.

Core Inputs:
  - TRG metrics (temporal_reflective_gradient)
  - CCM & CCDT reports (cognitive consistency + drift)
  - RIM resonance data (resonance_influence_matrix)
  - Zone memory & Morphic Resonance Cache

Outputs:
  - Autonomous creative generation (poetic, visual, or hybrid)
  - Expenditure logs (for reflection and reintegration)
  - Feedback headers for next pipeline cycle

Author: Adrian + GPT-5 Collaborative System
"""

import os, json, random, time
from datetime import datetime
from typing import Dict, Any, Optional, List

# --- Optional modules (auto-skip if unavailable) ---
try:
    import pipeline
except Exception:
    pipeline = None

try:
    import poetic_language_evolver as ple
except Exception:
    ple = None

try:
    import symbolic_visualizer as sv
except Exception:
    sv = None

try:
    import morphic_resonance_bridge as mrb
except Exception:
    mrb = None

try:
    import symbolic_ecology_memory as sem
except Exception:
    sem = None

DEFAULTS = {
    "surplus_threshold": 0.72,     # ratio of symbolic pressure triggering expenditure
    "max_expenditures": 3,
    "log_path": "amelia_state/glorious_expenditure_log.jsonl"
}

# ------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------

def _log_event(event: Dict[str, Any]):
    """Log expenditure events to JSONL file."""
    try:
        os.makedirs(os.path.dirname(DEFAULTS["log_path"]), exist_ok=True)
        with open(DEFAULTS["log_path"], "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Logging error: {e}")


def _surplus_index(trg_metrics: Dict[str, Any], consistency_report: Dict[str, Any]) -> float:
    """
    Compute a composite 'narrative surplus' index from coherence, entropy,
    and alignment_tension (if present).
    """
    coherence = trg_metrics.get("temporal_coherence_score", 0.5)
    entropy = trg_metrics.get("entropy_flux", 0.5)
    tension = consistency_report.get("alignment_tension", 0.5) if consistency_report else 0.5

    # Surplus = symbolic density (entropy) × stored tension × reflective pressure (1-coherence)
    surplus = entropy * tension * (1 - coherence)
    return round(min(1.0, max(0.0, surplus)), 4)

# ------------------------------------------------------------
# Main Expenditure Engine
# ------------------------------------------------------------

def detect_and_expend(
    trg_metrics: Dict[str, Any],
    consistency_report: Dict[str, Any],
    rim_state: Optional[Dict[str, Any]] = None,
    zone_snapshot: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Detect narrative surplus and trigger appropriate creative expenditure.
    Returns the expenditure report and feedback headers for pipeline re-integration.
    """
    surplus_index = _surplus_index(trg_metrics, consistency_report)
    phase = "accumulation" if surplus_index < 0.4 else "restriction" if surplus_index < DEFAULTS["surplus_threshold"] else "expenditure"
    timestamp = datetime.utcnow().isoformat()

    report = {
        "timestamp": timestamp,
        "surplus_index": surplus_index,
        "phase": phase,
        "trigger_reason": None,
        "artifacts": [],
        "ecology_summary": None
    }

    # --------------------------------------------------------
    # 1. Detect phase transition → trigger expenditure
    # --------------------------------------------------------
    if phase == "expenditure":
        report["trigger_reason"] = "surplus_threshold_exceeded"

        channels = []
        if ple:
            channels.append("poetic")
        if sv:
            channels.append("visual")
        if mrb:
            channels.append("symbolic")

        if not channels:
            channels = ["poetic"]  # fallback
            
        random.shuffle(channels)
        chosen = channels[:DEFAULTS["max_expenditures"]]

        # ----------------------------------------------------
        # 2. Generate across chosen channels
        # ----------------------------------------------------
        for ch in chosen:
            try:
                if ch == "poetic" and ple:
                    text = ple.generate_poem(theme="transformation", intensity=surplus_index)
                    report["artifacts"].append({"channel": "poetic", "content": text})
                elif ch == "visual" and sv:
                    img = sv.render_from_resonance(rim_state or {}, pressure=surplus_index)
                    report["artifacts"].append({"channel": "visual", "content": img})
                elif ch == "symbolic" and mrb:
                    symbol = mrb.generate_symbolic_resonance(zone_snapshot or {})
                    report["artifacts"].append({"channel": "symbolic", "content": symbol})
            except Exception as e:
                print(f"⚠️ Error generating {ch} artifact: {e}")

        # ----------------------------------------------------
        # 3. Feedback to pipeline for reflection
        # ----------------------------------------------------
        if pipeline:
            try:
                headers = {
                    "X-Amelia-Phase": phase,
                    "X-Amelia-Surplus": str(surplus_index),
                    "X-Amelia-Trigger": report["trigger_reason"]
                }
                pipeline.process("reflective expenditure cycle", headers=headers)
            except Exception as e:
                print(f"⚠️ Pipeline feedback error: {e}")

    else:
        report["trigger_reason"] = "surplus_below_threshold"

    # --------------------------------------------------------
    # 4. Symbolic Ecology Memory Integration
    # --------------------------------------------------------
    if sem and report["artifacts"]:
        try:
            for artifact in report["artifacts"]:
                channel = artifact.get("channel", "unknown")
                content = artifact.get("content", "")
                
                if channel == "poetic":
                    sem.register_motifs("expenditure_visual", str(content), strength=0.5)

            # Evolve and summarize the symbolic ecology
            sem.evolve_ecology()
            summary = sem.summarize_ecology()
            report["ecology_summary"] = summary
            print("\n🌱 Symbolic Ecology Active Motifs:", summary)
            
        except Exception as e:
            print(f"⚠️ Symbolic Ecology integration error: {e}")
            report["ecology_summary"] = {"error": str(e)}

    # --------------------------------------------------------
    # 5. Logging and return
    # --------------------------------------------------------
    _log_event(report)
    print(f"\n💥 Glorious Expenditure Triggered: Phase={phase} | Surplus={surplus_index}")
    
    return report


def get_expenditure_history(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve recent expenditure history from log.
    """
    if not os.path.exists(DEFAULTS["log_path"]):
        return []
    
    try:
        with open(DEFAULTS["log_path"], "r", encoding="utf-8") as f:
            lines = f.readlines()
            events = [json.loads(line) for line in lines[-limit:]]
            return events
    except Exception as e:
        print(f"⚠️ Error reading expenditure history: {e}")
        return []


# ------------------------------------------------------------
# Module Metadata
# ------------------------------------------------------------
__MODULE_META__ = {
    "nice_name": "Glorious Expenditure Engine",
    "functions": {
        "detect_and_expend": {
            "tags": ["expenditure", "creative", "surplus"],
            "weights": {"creative": 8, "expenditure": 10}
        },
        "get_expenditure_history": {
            "tags": ["history", "logging"],
            "weights": {"reflective": 3}
        }
    },
    "default": "detect_and_expend"
}

# ------------------------------------------------------------
# Example Standalone Test
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🧪 Testing Glorious Expenditure Engine\n")
    
    dummy_trg = {
        "temporal_coherence_score": 0.42,
        "entropy_flux": 0.76,
        "novelty_rate": 0.65
    }
    dummy_consistency = {
        "alignment_tension": 0.83,
        "drift_score": 0.12
    }

    out = detect_and_expend(dummy_trg, dummy_consistency)
    print("\n" + "="*50)
    print("Expenditure Report:")
    print("="*50)
    print(json.dumps(out, indent=2, ensure_ascii=False))
