# morphic_resonance_bridge.py
# -*- coding: utf-8 -*-
"""
Morphic Resonance Bridge

Responsibilities:
  - Mythopoetic Encoding Cache (lightweight archetype store)
  - Resonance Influence Matrix (RIM): co-occurrence and weighting of
    (drift_reason, cross_reason) across zones + temporal smoothing
  - Produce advise_next_params() (temperature, fold_nudge, zone_weights, resonance_boosts)
    which other modules (pipeline/assemblage_generator/amelia_autonomy) can consume.
  - Simple motif detection + archetype merging / mutation to bias continuity.
  - Persist/Load state to the app directory.

Design notes:
  - Stateless functions return pure data; the module persists an internal state file.
  - All outputs are JSON-serializable primitives/structures.
"""

from __future__ import annotations
import os, json, time, math, random, hashlib, re
from typing import Any, Dict, List, Optional, Tuple

_STATE_DIR = "amelia_state"
_STATE_FILE = "morphic_resonance_state.json"
_ARCHIVE_FILE = "mythopoetic_cache.jsonl"

# Tunables
MAX_PAIR_HISTORY = 1000
RIM_DECAY_HALF_LIFE_S = 60.0 * 60.0 * 24.0  # 1 day half-life by default
MUTATION_RATE = 0.04  # small chance archetype mutates on update
MERGE_SIM_THRESH = 0.7  # similarity threshold to merge motifs

# --- Utility I/O -----------------------------------------------------------

def _app_dir() -> str:
    base = os.path.join(os.getcwd(), _STATE_DIR)
    os.makedirs(base, exist_ok=True)
    return base

def _state_path() -> str:
    return os.path.join(_app_dir(), _STATE_FILE)

def _archive_path() -> str:
    return os.path.join(_app_dir(), _ARCHIVE_FILE)

def _now_ms() -> int:
    return int(time.time() * 1000)

# --- Simple text similarity (lightweight) ---------------------------------

def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", s)]

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))

def _semantic_sim(a: str, b: str) -> float:
    # lightweight heuristic: jaccard + shared prefix boost
    ja = _jaccard(_tokenize(a), _tokenize(b))
    prefix = 1.0 if a.strip().lower().startswith(b.strip().lower()[:3]) or b.strip().lower().startswith(a.strip().lower()[:3]) else 0.0
    return min(1.0, 0.7 * ja + 0.3 * prefix)

# --- State management ------------------------------------------------------

def _load_state() -> Dict[str, Any]:
    path = _state_path()
    if not os.path.exists(path):
        return {
            "pair_history": [],            # [{"a":str,"b":str,"zone":int,"ts":ms}, ...]
            "rim": {},                     # { "drift|cross": { "zone": { "partner": weight } } }
            "archetypes": [],              # [ {"id":str,"label":str,"snippets":[..],"score":float,"ts":ms}, ... ]
            "last_updated": _now_ms()
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # fallback default
        return {
            "pair_history": [],
            "rim": {},
            "archetypes": [],
            "last_updated": _now_ms()
        }

def _save_state(st: Dict[str, Any]) -> None:
    try:
        with open(_state_path(), "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- Mythopoetic Cache helpers ---------------------------------------------

def _persist_archetype_arc(label: str, snippet: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    try:
        rec = {"ts": _now_ms(), "label": label, "snippet": snippet, "meta": metadata or {}}
        with open(_archive_path(), "ab") as f:
            f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    except Exception:
        pass

def _new_archetype(label: str, snippet: str) -> Dict[str, Any]:
    return {"id": hashlib.sha1((label + snippet + str(time.time())).encode()).hexdigest()[:12],
            "label": label,
            "snippets": [snippet],
            "score": 1.0,
            "ts": _now_ms()}

# --- RIM (Resonance Influence Matrix) functions ---------------------------

def _decay_weight(old_weight: float, age_s: float, half_life_s: float = RIM_DECAY_HALF_LIFE_S) -> float:
    if old_weight <= 0:
        return 0.0
    # exponential decay toward zero
    lam = math.log(2) / max(1e-6, half_life_s)
    return old_weight * math.exp(-lam * age_s)

def _rim_key(drift_reason: str, cross_reason: str) -> str:
    # canonical composite key used in RIM
    return f"{drift_reason}||{cross_reason}"

# --- Core analysis / update routines --------------------------------------

def record_pairing(drift_reason: str, cross_reason: str, zone: int, st: Dict[str, Any]) -> None:
    """
    Record a module pair / reason co-occurrence for later analysis.
    """
    rec = {"a": drift_reason, "b": cross_reason, "zone": int(zone), "ts": _now_ms()}
    st.setdefault("pair_history", []).append(rec)
    # cap history
    if len(st["pair_history"]) > MAX_PAIR_HISTORY:
        st["pair_history"] = st["pair_history"][-MAX_PAIR_HISTORY:]

def update_rim(drift_reason: str, cross_reason: str, zone: int, st: Dict[str, Any]) -> None:
    """
    Update RIM cell weights for this (drift_reason, cross_reason, zone).
    Uses temporal smoothing and decay.
    """
    rim = st.setdefault("rim", {})
    key = _rim_key(drift_reason, cross_reason)
    zone_map = rim.setdefault(key, {})
    now = _now_ms()
    existing = zone_map.get(str(zone), {"w": 0.0, "ts": now})
    age_s = max(0.0, (now - existing.get("ts", now)) / 1000.0)
    # apply decay to existing weight then add a small reinforcement
    decayed = _decay_weight(existing.get("w", 0.0), age_s)
    increment = 1.0  # base reinforcement
    new_w = decayed + increment
    zone_map[str(zone)] = {"w": new_w, "ts": now}
    st["rim"][key] = zone_map

def query_rim_influence(drift_reason: str, cross_reason: str, st: Dict[str, Any]) -> Dict[int, float]:
    """
    Return normalized weights per zone for given reason pair.
    """
    key = _rim_key(drift_reason, cross_reason)
    rim = st.get("rim", {})
    zone_map = rim.get(key, {})
    out = {}
    total = 0.0
    now = _now_ms()
    for z_str, info in zone_map.items():
        w = info.get("w", 0.0)
        age_s = max(0.0, (now - info.get("ts", now)) / 1000.0)
        w = _decay_weight(w, age_s)
        if w > 0:
            out[int(z_str)] = w
            total += w
    # normalize
    if total > 0:
        for z in list(out.keys()):
            out[z] = out[z] / total
    return out

# --- Archetype detection / mutation ---------------------------------------

def detect_archetype(snippet: str, st: Dict[str, Any]) -> Tuple[str, float]:
    """
    Detect the closest archetype in the cache. Returns (label, similarity).
    If no match above threshold, returns (new_label_guess, 0.0).
    """
    archetypes = st.get("archetypes", [])
    if not archetypes:
        guess = "motif"
        return guess, 0.0

    best_score = 0.0
    best_label = archetypes[0]["label"] if archetypes else "motif"
    for arc in archetypes:
        # compare snippet to each stored snippet (take best)
        for s in arc.get("snippets", [])[-3:]:
            sim = _semantic_sim(snippet, s)
            if sim > best_score:
                best_score = sim
                best_label = arc["label"]
    return best_label, best_score

def merge_or_create_archetype(snippet: str, st: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge snippet into an existing archetype if similar, otherwise create new one.
    Returns the archetype dict (new or updated).
    """
    label_guess = re.sub(r"\W+", "_", snippet.strip().split()[:3][0] if snippet else "motif")[:20]
    archetypes = st.setdefault("archetypes", [])
    best_idx = -1
    best_sim = 0.0
    for idx, arc in enumerate(archetypes):
        # compare to arc label and few snippets
        sim_label = _semantic_sim(label_guess, arc.get("label", ""))
        sim_snip = max((_semantic_sim(snippet, s) for s in arc.get("snippets", [])[-4:]), default=0.0)
        sim = max(sim_label, sim_snip)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    if best_sim >= MERGE_SIM_THRESH and best_idx >= 0:
        # merge
        arc = archetypes[best_idx]
        arc["snippets"].append(snippet)
        arc["score"] = min(10.0, arc.get("score", 1.0) + 0.25 + best_sim)
        arc["ts"] = _now_ms()
        # small mutation chance
        if random.random() < MUTATION_RATE:
            arc["label"] = arc["label"] + "-" + hashlib.sha1(snippet.encode()).hexdigest()[:4]
        _persist_archetype_arc(arc["label"], snippet, {"merged": True, "sim": best_sim})
        return arc
    else:
        # create new archetype
        arc = _new_archetype(label_guess, snippet)
        archetypes.append(arc)
        _persist_archetype_arc(arc["label"], snippet, {"merged": False})
        return arc

# --- Scoring / advice generation ------------------------------------------

def _score_zone_bias_from_rim(drift_reason: str, cross_reason: str, st: Dict[str, Any], zone_count: int = 10) -> Dict[int, float]:
    """
    Convert RIM query into zone_weights for advise_next_params.
    Combines direct RIM influence with simple smoothing across neighboring zones.
    """
    rim_map = query_rim_influence(drift_reason, cross_reason, st)
    if not rim_map:
        # default flat distribution
        return {z: 1.0 / max(1, zone_count) for z in range(zone_count)}

    # base weights from rim_map (sparse)
    weights = {z: rim_map.get(z, 0.0) for z in range(zone_count)}
    # simple neighbor smoothing (circular)
    smoothed = {}
    for z in range(zone_count):
        left = weights.get((z - 1) % zone_count, 0.0)
        right = weights.get((z + 1) % zone_count, 0.0)
        smoothed[z] = 0.5 * weights.get(z, 0.0) + 0.25 * left + 0.25 * right
    # normalize
    total = sum(smoothed.values()) or 1.0
    for z in smoothed:
        smoothed[z] = smoothed[z] / total
    return smoothed

def _suggest_resonance_boosts_from_pairs(chosen_modules: List[Dict[str, Any]], st: Dict[str, Any]) -> Dict[str, float]:
    """
    Small heuristic: increase resonance_boost for modules that participate in frequently occurring pairings.
    Returns mapping {module_name: boost}
    """
    boosts = {}
    history = st.get("pair_history", [])[-200:]
    name_set = set(m.get("name") for m in chosen_modules)
    now = _now_ms()
    for m in chosen_modules:
        name = m.get("name")
        times = [h for h in history if name in (h.get("a"), h.get("b"))]
        if not times:
            boosts[name] = 0.0
            continue
        # recency-weighted frequency
        score = 0.0
        for h in times:
            age_s = (now - h.get("ts", now)) / 1000.0
            score += 1.0 / (1.0 + age_s / 60.0)  # favors recent events
        # normalize into a small boost range
        boosts[name] = min(0.5, 0.05 + 0.001 * score)
    return boosts

def advise_next_params(drift_reason: str,
                       cross_reason: str,
                       zone: int,
                       chosen_modules: List[Dict[str, Any]],
                       snippets: List[str],
                       st_override: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    """
    Primary function to call each cycle.
    Inputs:
      - drift_reason, cross_reason: strings from last cycle
      - zone: current zone (int)
      - chosen_modules: list of module dicts used in cycle
      - snippets: textual fragments produced in cycle
      - st_override: optional pre-loaded state (for testing)
    Returns a dict with:
      - temperature: suggested temperature (float)
      - fold_nudge: additive adjustment to fold in [-0.2,0.2]
      - zone_weights: dict zone->weight (normalized)
      - resonance_boosts: dict module_name->float (additive)
      - cross_feedback: short reason string for introspection
    """
    st = st_override if isinstance(st_override, dict) else _load_state()

    # record pairing
    record_pairing(drift_reason, cross_reason, zone, st)
    update_rim(drift_reason, cross_reason, zone, st)

    # Merge snippets into archetype cache
    archetype_pairs = []
    for s in snippets[-4:]:
        label, sim = detect_archetype(s, st)
        arc = merge_or_create_archetype(s, st)
        archetype_pairs.append((arc["label"], sim))

    # compute base zone bias from RIM
    zone_weights = _score_zone_bias_from_rim(drift_reason, cross_reason, st, zone_count=10)

    # compute resonance boosts for chosen modules
    resonance_boosts = _suggest_resonance_boosts_from_pairs(chosen_modules, st)

    # determine temperature: if RIM shows concentrated distribution -> lower temp (exploit)
    concentration = max(zone_weights.values()) if zone_weights else 0.0
    temperature = 0.9 - 0.5 * concentration  # lower if concentrated
    temperature = max(0.2, min(1.5, temperature))

    # fold_nudge: if archetype similarity high, nudge toward stronger fold (continuity)
    avg_sim = sum(sim for _, sim in archetype_pairs) / max(1, len(archetype_pairs))
    fold_nudge = (avg_sim - 0.35) * 0.25  # map sim -> small adjustment
    fold_nudge = max(-0.2, min(0.2, fold_nudge))

    # cross_feedback reason summarizer
    if avg_sim > 0.8:
        cross_feedback = "historical_resonance"
    elif avg_sim > 0.45:
        cross_feedback = "partial_resonance"
    else:
        cross_feedback = "random_link"

    # small meta learning: if many recent pairings contain same pair, slightly boost exploitation
    recent_pairs = [(h["a"], h["b"]) for h in st.get("pair_history", [])[-100:]]
    freq_map = {}
    for a, b in recent_pairs:
        freq_map[(a, b)] = freq_map.get((a, b), 0) + 1
    most_common_count = max(freq_map.values()) if freq_map else 0
    # if there is a dominant repeated pair, bias temperature down a touch
    if most_common_count > 4:
        temperature *= 0.9
        temperature = max(0.15, temperature)

    # update timestamps and persist
    st["last_updated"] = _now_ms()
    _save_state(st)

    return {
        "temperature": round(float(temperature), 3),
        "fold_nudge": round(float(fold_nudge), 3),
        "zone_weights": {int(k): float(v) for k, v in zone_weights.items()},
        "resonance_boosts": resonance_boosts,
        "cross_feedback": cross_feedback,
        "meta": {
            "avg_archetype_sim": round(float(avg_sim), 3),
            "most_common_pair_count": int(most_common_count),
            "arch_count": len(st.get("archetypes", []))
        }
    }

# --- Convenience wrappers & CLI test harness -------------------------------

def analyze_and_feedback(cycle_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper for pipeline:
      cycle_info expected keys:
        - drift_reason (str)
        - cross_reason (str)
        - zone (int)
        - chosen_modules (list of module dicts)
        - snippets (list of strings)
    Returns the advise_next_params structure.
    """
    drift = cycle_info.get("drift_reason", "phase_engine")
    cross = cycle_info.get("cross_reason", "none")
    zone = int(cycle_info.get("zone", 0))
    chosen = cycle_info.get("chosen_modules", []) or []
    snippets = cycle_info.get("snippets", []) or []
    return advise_next_params(drift, cross, zone, chosen, snippets)

if __name__ == "__main__":
    # quick smoke test
    st = _load_state()
    sample = {
        "drift_reason": "phase_engine",
        "cross_reason": "random_link",
        "zone": 7,
        "chosen_modules": [{"name": "numogram_core"}, {"name": "poetic_drift"}, {"name": "morphic_resonator"}],
        "snippets": ["mirror of the night", "phoenix descent", "labyrinth of memory"]
    }
    out = analyze_and_feedback(sample)
    print(json.dumps(out, indent=2, ensure_ascii=False))
