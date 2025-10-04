# -*- coding: utf-8 -*-
"""
amelia_autonomy.py — Meta-layer for adaptive symbolic evolution

Responsibilities
---------------
• Persist a "policy" which biases the lower-level generator:
    - zone_prior[0..9]           : soft priors over zones
    - tone_prior[str->float]     : soft priors over narrative tones
    - pair_bias["A|B"]           : association weights for module pairs (order-agnostic)
    - resonance_bias (float)     : nudges resonance_strength upward/downward
    - fold_target (float 0..1)   : target fold intensity
• Learn from recent reflection traces (drift_reason + cross_reason).
• Produce directives:
    - preferred_zones, tone_bias, fold_target, resonance_nudge
    - module_boosts[name] for assemblage selection weighting
• Accept outcomes to continuously refine policy.

No imports from pipeline/assemblage_generator to avoid cycles. Uses same state file:
    amelia_state/zone_swarm_state.json
and stores a separate policy file:
    amelia_state/autonomy_policy.json

Public API
----------
- load_policy() -> Dict
- save_policy(policy: Dict) -> None
- learn_from_reflection(window=50) -> Dict (policy)
- decide(intent: str, context: Dict, catalog: Optional[Dict]) -> Dict (directives)
- compute_module_boosts(catalog: Dict, policy: Dict) -> Dict[str, float]
- update_after_outcome(outcome: Dict) -> Dict (policy)

Outcome shape (what you pass after a turn)
------------------------------------------
outcome = {
  "zone": int,
  "drift_reason": str,
  "cross_reason": str,
  "modules_used": [str, ...],
  "resonance_strength": float,
  "tone": str
}
"""

from __future__ import annotations
import os, json, time, math, collections, random
from typing import Any, Dict, List, Optional, Tuple

# ---------- Paths (aligned with assemblage_generator.py) ----------

STATE_DIR = "amelia_state"
STATE_FILE = "zone_swarm_state.json"
POLICY_FILE = "autonomy_policy.json"

def _app_dir() -> str:
    base = os.path.join(os.getcwd(), STATE_DIR)
    os.makedirs(base, exist_ok=True)
    return base

def _state_path() -> str:
    return os.path.join(_app_dir(), STATE_FILE)

def _policy_path() -> str:
    return os.path.join(_app_dir(), POLICY_FILE)

# ---------- Safe load/save helpers ----------

def _load_json(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _save_json(path: str, obj: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---------- Defaults ----------

def _default_policy() -> Dict[str, Any]:
    tones = ["mythic","reflective","scientific","dreamlike","neutral"]
    tone_prior = {t: 1.0/len(tones) for t in tones}
    return {
        "version": 1,
        "updated_ms": int(time.time()*1000),
        "zone_prior": [0.1]*10,     # uniform soft prior
        "tone_prior": tone_prior,   # uniform soft prior
        "pair_bias": {},            # "A|B" -> weight
        "resonance_bias": 0.0,      # nudges resonance_strength (-0.2..+0.2 recommended)
        "fold_target": 0.55,        # target fold
        "learning": {
            "pair_recency_weight": 0.6,
            "pair_freq_weight": 0.4,
            "zone_recency_weight": 0.6,
            "zone_freq_weight": 0.4,
            "tone_recency_weight": 0.5,
            "tone_freq_weight": 0.5,
            "alpha": 0.25           # EMA for priors
        }
    }

# ---------- Public: load/save policy ----------

def load_policy() -> Dict[str, Any]:
    pol = _load_json(_policy_path(), _default_policy())
    # Fill missing keys if upgrading
    base = _default_policy()
    for k, v in base.items():
        if k not in pol:
            pol[k] = v
    # Fill learning defaults
    for k, v in base["learning"].items():
        pol["learning"].setdefault(k, v)
    return pol

def save_policy(policy: Dict[str, Any]) -> None:
    policy["updated_ms"] = int(time.time()*1000)
    _save_json(_policy_path(), policy)

# ---------- Internal: helpers ----------

def _load_state() -> Dict[str, Any]:
    return _load_json(_state_path(), {})

def _normalize(vs: List[float], eps: float = 1e-9) -> List[float]:
    s = sum(vs)
    if s <= eps:
        return [1.0/len(vs) for _ in vs]
    return [x/s for x in vs]

def _norm_dict(d: Dict[str, float], floor: float = 1e-6) -> Dict[str, float]:
    s = sum(d.values()) or 1.0
    return {k: max(floor, v/s) for k, v in d.items()}

def _pair_key(a: str, b: str) -> str:
    x, y = sorted([a, b])
    return f"{x}|{y}"

def _ema(old: float, new: float, alpha: float) -> float:
    return (1.0 - alpha) * old + alpha * new

def _recency_boost(ts: int, now: int, scale_ms: int = 60_000) -> float:
    # 1 / (1 + age_minutes)
    age = max(0, now - ts)
    return 1.0 / (1.0 + age / float(scale_ms))

# ---------- Learning from reflection ----------

def learn_from_reflection(window: int = 50) -> Dict[str, Any]:
    """
    Reads state (drift_history, pair_history), updates the autonomy policy, persists it,
    and returns the new policy.
    """
    st = _load_state()
    policy = load_policy()
    learn = policy["learning"]
    now = int(time.time()*1000)

    # ---- Zones ----
    drift_hist: List[Dict[str, Any]] = st.get("drift_history", [])[-window:]
    zone_counts = collections.Counter([d.get("next_zone") for d in drift_hist if isinstance(d.get("next_zone"), int)])
    zone_scores = [0.0]*10
    total = max(1, len(drift_hist))
    for i in range(10):
        freq = zone_counts.get(i, 0) / total
        # recency: max recency among drifts to that zone
        rec = 0.0
        for d in drift_hist:
            if d.get("next_zone") == i:
                rec = max(rec, _recency_boost(d.get("timestamp", now), now))
        zone_scores[i] = learn["zone_freq_weight"]*freq + learn["zone_recency_weight"]*rec

    zone_scores = _normalize(zone_scores)
    # EMA update
    policy["zone_prior"] = [_ema(policy["zone_prior"][i], zone_scores[i], learn["alpha"]) for i in range(10)]

    # ---- Tones ----
    tone_counts = collections.Counter([d.get("tone") for d in drift_hist if isinstance(d.get("tone"), str)])
    tones = set(list(policy["tone_prior"].keys()) + list(tone_counts.keys()))
    tone_scores = {}
    total_tones = max(1, sum(tone_counts.values()))
    for t in tones:
        freq = tone_counts.get(t, 0) / total_tones
        rec = 0.0
        for d in drift_hist:
            if d.get("tone") == t:
                rec = max(rec, _recency_boost(d.get("timestamp", now), now))
        tone_scores[t] = learn["tone_freq_weight"]*freq + learn["tone_recency_weight"]*rec

    # Normalize dict + EMA
    tone_scores = _norm_dict(tone_scores) if tone_scores else dict(policy["tone_prior"])
    for t in tones:
        old = policy["tone_prior"].get(t, 1e-3)
        new = tone_scores.get(t, 1e-3)
        policy["tone_prior"][t] = _ema(old, new, learn["alpha"])

    # ---- Pairs ----
    pair_hist: List[Dict[str, Any]] = st.get("pair_history", [])[-(window*4):]
    # Accumulate recency-weighted counts for each pair
    pair_acc = collections.defaultdict(float)
    for h in pair_hist:
        a, b = h.get("a"), h.get("b")
        if not a or not b: continue
        key = _pair_key(a, b)
        pair_acc[key] += _recency_boost(h.get("ts", now), now)

    # Softly blend into policy pair_bias (EMA on each)
    for key, score in pair_acc.items():
        old = policy["pair_bias"].get(key, 0.0)
        policy["pair_bias"][key] = _ema(old, score, learn["alpha"])

    # Slight global resonance nudging based on stability of drift reasons:
    # If last N drifts are mostly "historical_resonance" or tone-biased, increase resonance slightly.
    last_reasons = [d.get("reason","") for d in drift_hist]
    strong = sum(1 for r in last_reasons if "historical" in r or "tone_bias" in r)
    frac = strong / max(1, len(last_reasons))
    target_res_bias = (frac - 0.5) * 0.3  # clamp around [-0.15, +0.15]
    # EMA toward target
    policy["resonance_bias"] = max(-0.2, min(0.2, _ema(policy["resonance_bias"], target_res_bias, learn["alpha"])))

    # Fold target adaptation: if many ruptures observed (in state fold?), ease toward mid unless user steers
    # We approximate from st["fold"] if present — gentle correction toward ~0.55
    cur_fold = float(st.get("fold", 0.55))
    policy["fold_target"] = max(0.1, min(0.95, _ema(policy["fold_target"], 0.55 + (cur_fold-0.55)*0.2, 0.15)))

    save_policy(policy)
    return policy

# ---------- Mapping policy to directives ----------

def _argmax_k(values: List[float], k: int) -> List[int]:
    idx = list(range(len(values)))
    idx.sort(key=lambda i: values[i], reverse=True)
    return idx[:max(1, k)]

def compute_module_boosts(catalog: Dict[str, Dict[str, Any]],
                          policy: Dict[str, Any]) -> Dict[str, float]:
    """
    Map pair_bias and tone/zone priors to per-module boosts.
    Expects `catalog` in the same format pipeline injects into assemblage_generator._CATALOG:
        { "module.fn": {"tags":[...], "weights":{...}, "nice_name":"..."} }
    We derive module-level boosts by:
        - summing pair_bias contributions for any pair that includes the module name (prefix match)
        - small tone/tag alignment nudges (if tags present)
    Returns: { "module_name": boost_float }
    """
    # Aggregate by module root (before the first dot)
    by_module = collections.defaultdict(lambda: {"tags": set(), "weight": 0.0})
    for fqn, meta in (catalog or {}).items():
        mod = fqn.split(".", 1)[0]
        tags = meta.get("tags") or []
        wsum = sum((meta.get("weights") or {}).values()) if isinstance(meta.get("weights"), dict) else 0.0
        by_module[mod]["tags"].update(tags)
        by_module[mod]["weight"] += float(wsum)

    # Pair bias to modules
    pair_bias = policy.get("pair_bias", {})
    module_score = collections.defaultdict(float)
    for key, score in pair_bias.items():
        a, b = key.split("|", 1)
        module_score[a] += score
        module_score[b] += score

    # Tone/tag nudges
    tone_prior = policy.get("tone_prior", {})
    tone_top = max(tone_prior, key=tone_prior.get) if tone_prior else None
    tone_tag_map = {
        "mythic": {"myth", "symbolic", "ritual"},
        "reflective": {"reflection", "memory", "labyrinth"},
        "scientific": {"math", "numogram", "quantum"},
        "dreamlike": {"dream", "poetic", "oneiric"},
        "neutral": set()
    }
    prefer = tone_tag_map.get(tone_top, set())

    for mod, agg in by_module.items():
        if prefer and (agg["tags"] & prefer):
            module_score[mod] += 0.35
        module_score[mod] += 0.10 * math.tanh(0.3 * agg["weight"])

    # Normalize-ish: we just need relative weights; cap to sane range
    # Return as small boosts usable by assemblage selection
    out = {}
    if not module_score:
        return out
    maxv = max(module_score.values())
    if maxv <= 1e-9:
        return out
    for mod, v in module_score.items():
        out[mod] = round(0.6 * (v / maxv), 4)  # in [0..0.6]
    return out

def decide(intent: Optional[str] = None,
           context: Optional[Dict[str, Any]] = None,
           catalog: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Produce directives from current policy (optionally intent/context-aware).
    - preferred_zones: indices (top-2 by prior)
    - tone_bias: top tone name
    - fold_target: float
    - resonance_nudge: float
    - module_boosts: name->float (optional, if catalog provided)

    'intent' can be hints like: "creative", "analytical", "dream", "mythic", etc.
    We slightly tweak priors for common intents.
    """
    policy = load_policy()
    zone_prior = list(policy.get("zone_prior", [0.1]*10))
    tone_prior = dict(policy.get("tone_prior", {}))
    fold_target = float(policy.get("fold_target", 0.55))
    resonance_nudge = float(policy.get("resonance_bias", 0.0))

    # Intent shaping (gentle)
    if isinstance(intent, str):
        intent = intent.lower()
        if "dream" in intent or "poetic" in intent:
            tone_prior["dreamlike"] = tone_prior.get("dreamlike", 0.2) + 0.15
        if "myth" in intent:
            tone_prior["mythic"] = tone_prior.get("mythic", 0.2) + 0.15
        if "analysis" in intent or "scientific" in intent or "numogram" in intent:
            tone_prior["scientific"] = tone_prior.get("scientific", 0.2) + 0.15
        if "reflect" in intent or "memory" in intent:
            tone_prior["reflective"] = tone_prior.get("reflective", 0.2) + 0.15
        # Zone nudges
        if "excess" in intent:
            zone_prior[9] += 0.12
        if "mirror" in intent:
            zone_prior[7] += 0.12
        if "labyrinth" in intent:
            zone_prior[6] += 0.12
        if "synthesis" in intent:
            zone_prior[8] += 0.12

    # Normalize priors
    zone_prior = _normalize(zone_prior)
    tone_prior = _norm_dict(tone_prior) if tone_prior else {"neutral": 1.0}

    # Choose top-2 zones, top tone
    preferred_zones = _argmax_k(zone_prior, k=2)
    tone_bias = max(tone_prior, key=tone_prior.get)

    # Optional: compute module boosts if catalog provided
    boosts = compute_module_boosts(catalog or {}, policy) if isinstance(catalog, dict) else {}

    return {
        "preferred_zones": preferred_zones,
        "tone_bias": tone_bias,
        "fold_target": round(fold_target, 3),
        "resonance_nudge": round(resonance_nudge, 3),
        "module_boosts": boosts
    }

# ---------- Online update after each turn ----------

def update_after_outcome(outcome: Dict[str, Any]) -> Dict[str, Any]:
    """
    Consume a single outcome (post-generate) and adjust policy online.
    This can be called every turn in addition to periodic learn_from_reflection().
    """
    policy = load_policy()
    learn = policy["learning"]

    # Sanity
    if not isinstance(outcome, dict):
        return policy

    # Drift reason: reward stability from "historical_resonance" and "tone_bias"
    drift_reason = str(outcome.get("drift_reason", "")).lower()
    cross_reason = str(outcome.get("cross_reason", "")).lower()
    zone = int(outcome.get("zone", -1))
    tone = str(outcome.get("tone", "")).lower()
    modules = outcome.get("modules_used", []) or []
    rs = float(outcome.get("resonance_strength", 0.0))

    # Resonance nudging
    if "historical" in drift_reason or "tone_bias" in drift_reason or "historical" in cross_reason:
        target = min(0.2, 0.10 + 0.20 * rs)  # up to +0.2
    elif "random" in drift_reason or "random" in cross_reason or "weak" in drift_reason:
        target = max(-0.2, -0.10 + -0.10 * (1.0 - rs))  # down to -0.2
    else:
        target = 0.0
    policy["resonance_bias"] = _ema(policy["resonance_bias"], target, learn["alpha"])

    # Zone prior small reinforcement
    if 0 <= zone <= 9:
        zp = policy.get("zone_prior", [0.1]*10)
        zp[zone] = _ema(zp[zone], 1.0, 0.05)  # very mild
        policy["zone_prior"] = _normalize(zp)

    # Tone prior small reinforcement
    if tone:
        tp = dict(policy.get("tone_prior", {}))
        if tone not in tp:
            tp[tone] = 1e-3
        tp[tone] = _ema(tp[tone], 1.0, 0.05)
        policy["tone_prior"] = _norm_dict(tp)

    # Pair reinforcement from modules_used
    if isinstance(modules, list) and len(modules) >= 2:
        for i in range(len(modules)):
            for j in range(i+1, len(modules)):
                key = _pair_key(modules[i], modules[j])
                old = policy["pair_bias"].get(key, 0.0)
                # Reward more if cross_reason isn't random
                bonus = 0.6 if "historical" in cross_reason else 0.25
                policy["pair_bias"][key] = _ema(old, old + bonus, 0.25)

    save_policy(policy)
    return policy

# ---------- Convenience: one-shot end-to-end (optional) ----------

def plan_and_update(intent: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None,
                    catalog: Optional[Dict[str, Dict[str, Any]]] = None,
                    outcome: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Utility that:
      1) learn_from_reflection()
      2) produce directives via decide()
      3) optionally apply online update from an outcome
    Returns: {"policy":..., "directives":...}
    """
    policy = learn_from_reflection()
    directives = decide(intent=intent, context=context, catalog=catalog)
    if outcome:
        update_after_outcome(outcome)
    return {"policy": policy, "directives": directives}
