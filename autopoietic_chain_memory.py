# -*- coding: utf-8 -*-
"""
Autopoietic Chain Memory (ACM)
------------------------------
Learns which *chains of modules* produce high-quality outcomes
(coherence, novelty, stability), then biases future composition.

Key ideas
- Record each transductive act: chain + metrics + phase + motif
- Score it (multi-objective), decay older events, and reinforce winners
- Propose next chains (phase-aware) + per-module boosts
- Persist lightweight models to disk for continuity

Integrates with creative_kernel.transductive_cycle() by:
- advising a candidate chain (optional)
- returning per-module boost weights to bias selection

Author: Adrian + GPT-5
"""

from __future__ import annotations
import os, json, math, time, random
from typing import Dict, Any, List, Tuple

ACM_DEFAULTS = {
    "model_file": "amelia_state/acm_memory.json",
    "history_file": "amelia_state/acm_history.jsonl",
    "max_history": 500,
    "half_life_events": 120,       # exponential decay half-life (events), not time-based
    "chain_max_len": 5,
    "min_support": 3,              # min times seen before strong recommendation
    "epsilon_explore": 0.12,       # exploration prob
    "module_base_weight": 1.0,
    "module_boost_cap": 2.5,
    "phase_bias": {
        "accumulation": ["dream", "reflection", "symbolic", "poetic"],
        "restriction":  ["contradiction", "poetic", "symbolic"],
        "expenditure":  ["poetic", "visual", "symbolic", "expenditure"]
    }
}

# --------------------------- Utilities --------------------------------

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _now_ms() -> int:
    return int(time.time() * 1000)

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, obj):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _exp_decay(weight: float, steps: float) -> float:
    # half-life in "events": after H events, weight halves
    # decay factor per step: 0.5 ** (steps / H)
    H = ACM_DEFAULTS["half_life_events"]
    return weight * (0.5 ** (steps / max(1e-9, H)))

# ----------------------- Core State Structure -------------------------

"""
acm_model:
{
  "modules": { "poetic": {"w":1.12,"n":23}, "dream": {...}, ... },
  "chains": {
     "dream|poetic|contradiction": {"w":2.33,"n":7,"last_ms":...,"by_phase":{"expenditure":4}},
     ...
  },
  "stats": {"total_events": 0, "last_ms": 0}
}
"""

def _new_model() -> Dict[str, Any]:
    return {"modules": {}, "chains": {}, "stats": {"total_events": 0, "last_ms": 0}}

# --------------------------- Scoring ----------------------------------

def score_event(
    *,
    coherence: float = 0.5,
    novelty: float = 0.5,
    stability: float = 0.5,
    entropy_flux: float = 0.5,
    custom_weights: Dict[str, float] | None = None
) -> float:
    """
    Multi-objective score (0..1+). Tunable weights; defaults balanced.
    Higher is better.
    """
    w = {"coherence": 0.35, "novelty": 0.35, "stability": 0.20, "entropy_flux": 0.10}
    if custom_weights:
        w.update(custom_weights)
    s = (
        w["coherence"] * coherence +
        w["novelty"] * novelty +
        w["stability"] * stability +
        w["entropy_flux"] * entropy_flux
    )
    # slight nonlinearity to reward >.7 outcomes
    if s > 0.7:
        s += 0.05 * (s - 0.7) / 0.3
    return max(0.0, float(s))

# --------------------- Public API: Learn & Advise ---------------------

def record_chain_event(
    chain: List[str],
    phase: str,
    motif: str | None,
    metrics: Dict[str, Any],
    context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Persist one creative act with metrics; update model weights (decayed reinforcement).
    metrics should include keys like: coherence, novelty, stability, entropy_flux
    """
    model = _load_json(ACM_DEFAULTS["model_file"], _new_model())
    hist_path = ACM_DEFAULTS["history_file"]

    # score
    s = score_event(
        coherence=float(metrics.get("coherence", metrics.get("temporal_coherence_score", 0.5))),
        novelty=float(metrics.get("novelty", metrics.get("novelty_rate", 0.5))),
        stability=float(metrics.get("stability", 1.0 - metrics.get("affective_variance", 0.5))),
        entropy_flux=float(metrics.get("entropy_flux", 0.5)),
        custom_weights=metrics.get("weights")
    )

    # decay global
    model["stats"]["total_events"] += 1
    model["stats"]["last_ms"] = _now_ms()
    steps = 1.0  # per event step
    for name, m in model["modules"].items():
        m["w"] = _exp_decay(m.get("w", 1.0), steps)
    for key, ch in model["chains"].items():
        ch["w"] = _exp_decay(ch.get("w", 1.0), steps)

    # reinforce modules
    for mod in chain:
        entry = model["modules"].setdefault(mod, {"w": ACM_DEFAULTS["module_base_weight"], "n": 0})
        entry["w"] += s
        entry["n"] += 1
        entry["last_ms"] = _now_ms()

    # reinforce chain
    key = "|".join(chain)
    ch = model["chains"].setdefault(key, {"w": 1.0, "n": 0, "last_ms": _now_ms(), "by_phase": {}})
    ch["w"] += s * (1.0 + 0.1 * max(0, len(chain) - 2))  # bonus for longer well-performing chains
    ch["n"] += 1
    ch["last_ms"] = _now_ms()
    ch["by_phase"][phase] = ch["by_phase"].get(phase, 0) + 1
    if motif:
        ch["motif"] = motif

    # persist
    _save_json(ACM_DEFAULTS["model_file"], model)
    _ensure_dir(hist_path)
    with open(hist_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": _now_ms(), "phase": phase, "motif": motif,
            "chain": chain, "score": s, "metrics": metrics, "ctx": context or {}
        }, ensure_ascii=False) + "\n")

    return {"score": s, "model_snapshot": {"modules": model["modules"], "chains": {k: {"w": v["w"], "n": v["n"]} for k, v in model["chains"].items()}}}

def advise_chain(
    phase: str,
    allow_modules: List[str] | None = None,
    max_len: int | None = None,
    epsilon: float | None = None
) -> Dict[str, Any]:
    """
    Suggest a chain to the creative kernel:
      - Phase-aware (prefers chains that succeeded in this phase)
      - Recency- and performance-weighted
      - ε-greedy exploration
    """
    model = _load_json(ACM_DEFAULTS["model_file"], _new_model())
    chains = model["chains"]
    modules = model["modules"]

    max_len = max_len or ACM_DEFAULTS["chain_max_len"]
    epsilon = epsilon if epsilon is not None else ACM_DEFAULTS["epsilon_explore"]

    # usable chains
    candidates: List[Tuple[str, float]] = []
    for key, ch in chains.items():
        seq = key.split("|")
        if allow_modules and any(m not in allow_modules for m in seq):
            continue
        # weight: performance * phase-support * recency boost
        w = ch["w"]
        support = ch["by_phase"].get(phase, 0)
        if support < ACM_DEFAULTS["min_support"]:
            w *= 0.7
        # mild recency: last 10% bonus
        age_bonus = 1.0 + 0.1 * (1.0 / (1.0 + (model["stats"]["total_events"] - ch["n"]) / max(1, model["stats"]["total_events"])))
        w *= age_bonus
        if 2 <= len(seq) <= max_len:
            candidates.append((key, w))

    if candidates and random.random() > epsilon:
        # exploitation: softmax over weights
        ws = [w for _, w in candidates]
        Z = sum(math.exp(x) for x in ws) or 1.0
        probs = [math.exp(w) / Z for _, w in candidates]
        choice = random.choices(candidates, weights=probs, k=1)[0][0]
        selected = choice.split("|")
        mode = "exploit"
    else:
        # exploration: build from module weights
        pool = allow_modules or list(modules.keys())
        if not pool:
            pool = ["dream", "poetic", "contradiction", "visual", "symbolic", "expenditure"]
        # choose length 2..max_len
        L = random.randint(2, max_len)
        # weighted by module w
        ws = [modules.get(m, {"w": 1.0})["w"] for m in pool]
        if sum(ws) == 0:
            ws = [1.0 for _ in pool]
        selected = random.choices(pool, weights=ws, k=L)
        mode = "explore"

    # also produce per-module boosts based on learned weights (capped)
    boosts = {}
    for m, entry in model["modules"].items():
        boosts[m] = min(ACM_DEFAULTS["module_boost_cap"], 0.5 + entry["w"] / max(1.0, entry["n"] or 1.0))

    return {
        "mode": mode,
        "chain": selected,
        "module_boosts": boosts,
        "meta": {"phase": phase, "epsilon": epsilon, "max_len": max_len}
    }

def top_chains(n: int = 10) -> List[Dict[str, Any]]:
    model = _load_json(ACM_DEFAULTS["model_file"], _new_model())
    ranked = sorted(model["chains"].items(), key=lambda kv: kv[1]["w"], reverse=True)[:n]
    out = []
    for key, ch in ranked:
        out.append({"chain": key.split("|"), "w": round(ch["w"], 3), "n": ch["n"], "by_phase": ch.get("by_phase", {})})
    return out

def build_headers_for_kernel(advice: Dict[str, Any]) -> Dict[str, str]:
    """
    Provide optional headers the kernel/pipeline can read to steer composition.
    """
    return {
        "X-Amelia-ACM-Mode": advice.get("mode", "explore"),
        "X-Amelia-ACM-Chain": json.dumps(advice.get("chain", [])),
        "X-Amelia-ACM-ModuleBoosts": json.dumps(advice.get("module_boosts", {})),
    }
