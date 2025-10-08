# -*- coding: utf-8 -*-
"""
Cognitive Consistency Drift Tracker (CCDT)
------------------------------------------
Tracks the evolution of Amelia’s alignment_tension over time
and detects self-correction or divergence cycles.

Core ideas:
  • Read cognitive_consistency_log.jsonl (from CCM)
  • Compute rolling averages and deltas of coherence/tension
  • Detect significant upward/downward drifts
  • Emit feedback events for TRG or Autonomy adjustment

Optionally produces lightweight JSON suitable for visualization
(e.g., in the Hyperstructure Visual Layer).

Author: Adrian + GPT-5 Collaborative System
"""

import os, json, math, statistics as stats
from typing import Dict, List, Any
from datetime import datetime

LOG_FILE = os.path.join(os.getcwd(), "amelia_state", "cognitive_consistency_log.jsonl")
DRIFT_FILE = os.path.join(os.getcwd(), "amelia_state", "consistency_drift_summary.json")

def _load_log(limit: int = 200) -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_FILE):
        return []
    lines = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f.readlines()[-limit:]:
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
    return lines

def analyze_drift(window: int = 10, alert_threshold: float = 0.15) -> Dict[str, Any]:
    """
    Analyze the evolution of coherence_score and alignment_tension over time.
    Returns a summary dict and optionally flags divergence cycles.
    """
    data = _load_log()
    if len(data) < 3:
        return {"note": "Insufficient data for drift analysis."}

    tensions = [d.get("alignment_tension", 0.0) for d in data]
    coherences = [d.get("coherence_score", 0.0) for d in data]
    timestamps = [d.get("timestamp") for d in data]

    avg_tension = round(stats.mean(tensions), 4)
    std_tension = round(stats.pstdev(tensions), 4)
    recent_tension = tensions[-1]
    delta = recent_tension - tensions[-2] if len(tensions) > 1 else 0.0

    # rolling window for microtrend
    if len(tensions) >= window:
        recent_window = tensions[-window:]
        early_window = tensions[:window]
        drift_slope = (stats.mean(recent_window) - stats.mean(early_window)) / max(1, len(data) - window)
    else:
        drift_slope = delta

    # Detect states
    state = "stable"
    if drift_slope > alert_threshold:
        state = "diverging"
    elif drift_slope < -alert_threshold:
        state = "re-aligning"

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "samples": len(tensions),
        "avg_tension": avg_tension,
        "std_tension": std_tension,
        "recent_tension": recent_tension,
        "delta": round(delta, 4),
        "drift_slope": round(drift_slope, 4),
        "state": state,
        "trend_description": {
            "diverging": "Self-model coherence decreasing — initiating TRG recalibration.",
            "re-aligning": "System regaining coherence — TRG convergence detected.",
            "stable": "Cognitive alignment stable within threshold."
        }[state]
    }

    # Save summary
    try:
        with open(DRIFT_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return summary

# ----------------------------------------------------------------
# Optional integration with TRG or Autonomy feedback
# ----------------------------------------------------------------

def feedback(summary: Dict[str, Any], trg=None, auto=None) -> None:
    """
    Optionally nudge TRG or Autonomy if drift detected.
    """
    if summary["state"] == "diverging" and trg:
        print("\n⚠️ Drift detected → TRG attenuation triggered.")
        trg.adjust_temporal_gain(-0.1)
    elif summary["state"] == "re-aligning" and auto:
        print("\n✅ Re-alignment detected → Autonomy reinforcement.")
        auto.reinforce_policy("reflective_alignment")

# ----------------------------------------------------------------
# Example standalone use
# ----------------------------------------------------------------

if __name__ == "__main__":
    from pprint import pprint
    summary = analyze_drift()
    pprint(summary)
