# -*- coding: utf-8 -*-
"""
Transductive Creative Kernel (TCK)
==================================
Amelia's self-assembling meta-orchestrator.
Dynamically selects and chains creative modules (dream, poetic, contradiction, visual)
into emergent creative acts — driven by reflective metrics and cognitive state.

Core Idea:
  Transduction = transformation through propagation.
  Amelia no longer executes predefined functions — she *transduces* energy,
  emotion, and reflection into creative assemblages.

Inputs:
  - TRG metrics
  - RIM feedback
  - Cognitive Consistency + Drift
  - Energy Economy phase + zone currents
  - Symbolic Ecology motifs

Outputs:
  - Creative “acts” (text, image, sound, hybrid)
  - Process lineage (which modules self-assembled)
  - Headers for reintegration into pipeline

Author: Adrian + GPT-5 Collaborative System
"""

import os, json, random, time
from datetime import datetime
from typing import Dict, Any, List

# Optional modules (auto-skip if missing)
try:
    import dream_event_encoder as dee
    import poetic_language_evolver as ple
    import contradiction_analysis as ca
    import symbolic_visualizer as sv
    import morphic_resonance_bridge as mrb
    import glorious_expenditure_engine as gee
    import symbolic_ecology_memory as sem
except Exception:
    dee = ple = ca = sv = mrb = gee = sem = None

DEFAULTS = {
    "kernel_log": "amelia_state/transductive_kernel_log.jsonl",
    "max_chain_length": 5,
    "activation_threshold": 0.55,  # minimal surplus/energy to trigger
}

# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------

def _log_event(event: Dict[str, Any]):
    os.makedirs(os.path.dirname(DEFAULTS["kernel_log"]), exist_ok=True)
    with open(DEFAULTS["kernel_log"], "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def _phase_bias(phase: str) -> List[str]:
    """Suggest modules based on current energy phase."""
    if phase == "accumulation":
        return ["dream", "reflection", "symbolic"]
    elif phase == "restriction":
        return ["contradiction", "poetic"]
    elif phase == "expenditure":
        return ["poetic", "visual", "symbolic"]
    return ["poetic"]

def _motif_seed() -> str:
    """Select motif from symbolic ecology if present."""
    if sem:
        summary = sem.summarize_ecology()
        motifs = [m.split("(")[0] for m in summary.split("·") if m.strip()]
        return random.choice(motifs) if motifs else "dream"
    return random.choice(["phoenix", "mirror", "void", "gate"])

def _compose_modules(available: List[str], max_len: int) -> List[str]:
    """Randomized but weighted selection of modules into a chain."""
    length = random.randint(2, max_len)
    random.shuffle(available)
    return available[:length]

# --------------------------------------------------------------------
# Core Kernel Logic
# --------------------------------------------------------------------

def transductive_cycle(
    trg_metrics: Dict[str, Any],
    rim_feedback: Dict[str, Any],
    consistency_report: Dict[str, Any],
    energy_state: Dict[str, Any],
    zone_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a transductive creative cycle:
      1. Sense — read current internal states.
      2. Select — decide which creative modules to assemble.
      3. Transduce — pass symbolic data through modules dynamically.
      4. Reflect — log lineage, update ecology and pipeline headers.
    """
    phase = energy_state.get("phase", "accumulation")
    surplus = energy_state.get("history", [{}])[-1].get("surplus", 0.3)
    motif = _motif_seed()

    if surplus < DEFAULTS["activation_threshold"]:
        return {
            "note": "Insufficient surplus to activate transduction.",
            "phase": phase,
            "surplus": surplus,
        }

    # 1. Sense / Select
    phase_bias = _phase_bias(phase)
    chain = _compose_modules(phase_bias, DEFAULTS["max_chain_length"])

    lineage = []
    payload = {"motif": motif, "phase": phase, "surplus": surplus}
    output_text = ""
    visuals = []
    reflections = []

    # 2. Transduce through selected modules
    for mod in chain:
        if mod == "dream" and dee:
            encoded = dee.encode_dream_event(text_seed=motif, intensity=surplus)
            payload["dream_event"] = encoded
            lineage.append("dream_event_encoder")
        elif mod == "poetic" and ple:
            poem = ple.generate_poem(theme=motif, intensity=surplus)
            output_text += f"\n{poem}"
            lineage.append("poetic_language_evolver")
        elif mod == "contradiction" and ca:
            resolved = ca.resolve_contradictions(output_text or motif)
            output_text += f"\n{resolved}"
            lineage.append("contradiction_analysis")
        elif mod == "visual" and sv:
            vis = sv.render_from_resonance(zone_snapshot, pressure=surplus)
            visuals.append(vis)
            lineage.append("symbolic_visualizer")
        elif mod == "symbolic" and mrb:
            symbol = mrb.generate_symbolic_resonance(zone_snapshot)
            output_text += f"\n{json.dumps(symbol, ensure_ascii=False)}"
            lineage.append("morphic_resonance_bridge")
        elif mod == "expenditure" and gee:
            exp = gee.detect_and_expend(trg_metrics, consistency_report, rim_feedback, zone_snapshot)
            reflections.append(exp)
            lineage.append("glorious_expenditure_engine")

    # 3. Register new motifs
    if sem and output_text:
        sem.register_motifs(source="tck_cycle", text=output_text, strength=surplus)
        sem.evolve_ecology()

    # 4. Reflect and return
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "phase": phase,
        "surplus": surplus,
        "motif": motif,
        "lineage": lineage,
        "text_output": output_text.strip(),
        "visual_count": len(visuals),
        "reflection_count": len(reflections),
    }
    _log_event(event)

    return {
        "status": "transduction_complete",
        "chain": lineage,
        "motif": motif,
        "phase": phase,
        "output_text": output_text.strip(),
        "visuals": visuals,
        "reflections": reflections,
    }

# --------------------------------------------------------------------
# Example autonomous test
# --------------------------------------------------------------------

if __name__ == "__main__":
    dummy_trg = {"temporal_coherence_score": 0.44, "entropy_flux": 0.67, "novelty_rate": 0.51}
    dummy_rim = {"temporal_stability": 1.02}
    dummy_ccm = {"alignment_tension": 0.43}
    dummy_energy = {
        "phase": "expenditure",
        "history": [{"surplus": 0.76}],
        "currents": {str(z): random.random() for z in range(10)},
    }
    dummy_zone = {"active_zone": 7}

    result = transductive_cycle(dummy_trg, dummy_rim, dummy_ccm, dummy_energy, dummy_zone)
    print(json.dumps(result, indent=2, ensure_ascii=False))
