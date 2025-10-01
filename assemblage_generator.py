# -*- coding: utf-8 -*-
"""
assemblage_generator.py — Zone–Swarm selector + Fold operator

Purpose
-------
Turn Amelia from a dispatcher into an emergent, machinic composer:
  1) Maintain a Numogram "zone state" which biases module swarms.
  2) Select and weight multiple specialist modules per step.
  3) Recombine their outputs via fold operators (low→high intensity).
  4) Drift the attractor weights over time (evolutionary micro-mutations).
  5) Expose a dormancy protocol hook for autonomous cycles.

Integration
-----------
- Pipeline can call:   assemblage_generator.generate(user_text=None, headers=None)
- Dynamic registry:    provided via __MODULE_META__ so your registry can use tags.
- Output is a dict with rich fields and a compact surface string, so compose_response
  can render hybrid/hardliner introspection cleanly.

Zero-crash: all hard failures degrade gracefully with diagnostics.
"""

from __future__ import annotations

import os
import re
import json
import uuid
import time
import math
import random
import importlib
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STATE_DIR = "amelia_state"
STATE_FILE = "zone_swarm_state.json"

DEFAULTS: Dict[str, Any] = {
    "zone": 7,                    # start near Mirror/Recursion (7)
    "fold": 0.55,                 # 0.0 soft weave → 1.0 violent rupture
    "temperature": 0.70,          # selection entropy for module choice
    "max_modules_per_cycle": 4,   # swarm width (2–5 recommended)
    "memory_window": 6,           # suppress immediate repeats
    "evo_sigma": 0.06,            # evolutionary noise magnitude
    "min_fold": 0.15,
    "max_fold": 0.95,
    "seed": None,
    "last_tick_ms": 0,
    "autonomous_interval_ms": 45_000,  # dormancy cycle cadence (example)
}

# Zone semantics → tag biases (extend freely)
ZONE_BIASES: Dict[int, Dict[str, float]] = {
    0: {"void": 1.0, "ur": 0.8, "disjunction": 0.6, "silence": 0.7, "poetic": 0.4},
    1: {"initiation": 1.0, "spark": 0.9, "myth": 0.7, "dream": 0.6, "symbolic": 0.5},
    2: {"pairing": 1.0, "dyad": 0.8, "contradiction": 0.6, "bataille": 0.5, "poetic": 0.5},
    3: {"surge": 1.0, "emergence": 0.9, "dream": 0.8, "swarm": 0.7, "mutation": 0.6},
    4: {"cycle": 1.0, "creation": 0.8, "destruction": 0.8, "numogram": 0.7, "math": 0.5},
    5: {"threshold": 1.0, "phase": 0.9, "hyperstition": 0.8, "fiction": 0.6, "myth": 0.6},
    6: {"labyrinth": 1.0, "recursion": 0.8, "memory": 0.7, "symbolic": 0.6},
    7: {"mirror": 1.0, "reflection": 0.9, "recursion": 0.9, "poetic": 0.7, "numogram": 0.6},
    8: {"synthesis": 1.0, "assemblage": 0.9, "quantum": 0.8, "resonance": 0.7, "science": 0.6},
    9: {"excess": 1.0, "sacrifice": 0.9, "bataille": 0.9, "transgression": 0.8, "myth": 0.6},
}

# Tag aliases ensure flexible matching with module tags
TAG_ALIASES: Dict[str, List[str]] = {
    "numogram": ["syzygy", "zones", "current", "triangular", "arithmetics"],
    "hyperstition": ["fiction", "becoming-real", "myth-tech"],
    "bataille": ["excess", "sacrifice", "solar-economy"],
    "quantum": ["resonance", "entanglement", "phase", "superposition"],
    "dream": ["oneiric", "interdream", "mythogenesis"],
    "poetic": ["mutation", "metre", "stanza", "verse"],
    "symbolic": ["drift", "glyph", "icon", "sigil"],
    "swarm": ["evolution", "population", "agents"],
}

# ---------------------------------------------------------------------------
# Module discovery via __MODULE_META__
# ---------------------------------------------------------------------------

def _scan_paths() -> List[str]:
    here = os.path.dirname(__file__)
    return [here, os.path.join(here, "modules"), os.path.join(here, "autonomous")]

def _list_python_files(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in os.listdir(p):
                if name.endswith(".py") and not name.startswith("_"):
                    files.append(os.path.join(p, name))
    return files

def _module_name_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base

def _resolve_tags(meta_tags: List[str]) -> List[str]:
    expanded = set(meta_tags or [])
    for t in list(expanded):
        for k, aliases in TAG_ALIASES.items():
            if t == k:
                expanded.update(aliases)
    return list(expanded)

def discover_catalog() -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {
        "module.func": {
          "tags": [...],
          "weights": {...},      # optional per-tag weights
          "nice_name": "Hyperstition Engine",
        }, ...
      }
    Only functions declared in __MODULE_META__["functions"] are considered.
    """
    catalog: Dict[str, Dict[str, Any]] = {}
    for path in _list_python_files(_scan_paths()):
        mod_name = _module_name_from_path(path)
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        meta = getattr(mod, "__MODULE_META__", None)
        if not isinstance(meta, dict):
            continue
        funcs = meta.get("functions", {}) or {}
        nice = meta.get("nice_name") or mod_name
        for fn_name, fn_meta in funcs.items():
            tags = _resolve_tags(list(fn_meta.get("tags", [])))
            weights = dict(fn_meta.get("weights", {}))
            fqn = f"{mod_name}.{fn_name}"
            catalog[fqn] = {"tags": tags, "weights": weights, "nice_name": nice}
    return catalog

# Cache catalog at import; refresh on demand via generate(..., refresh_catalog=True)
_CATALOG = discover_catalog()

# ---------------------------------------------------------------------------
# State I/O (zone, fold, memory, evo noise)
# ---------------------------------------------------------------------------

def _state_path() -> str:
    base = os.path.join(os.getcwd(), STATE_DIR)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, STATE_FILE)

def _load_state() -> Dict[str, Any]:
    path = _state_path()
    if not os.path.exists(path):
        return dict(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # backfill defaults
        for k, v in DEFAULTS.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return dict(DEFAULTS)

def _save_state(st: Dict[str, Any]) -> None:
    try:
        with open(_state_path(), "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Zone dynamics + evolutionary drift
# ---------------------------------------------------------------------------

def _softmax(xs: List[float], temp: float) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / max(1e-6, temp)) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def _zone_transition_prob(current: int) -> List[float]:
    """
    Local bias: prefer staying or moving +/-1, with small chance of longer jumps.
    """
    probs = []
    for z in range(10):
        d = min((z - current) % 10, (current - z) % 10)  # circular distance
        base = {0: 1.6, 1: 1.2, 2: 0.8, 3: 0.4}.get(d, 0.2)
        probs.append(base)
    # normalize
    s = sum(probs) or 1.0
    return [p / s for p in probs]

def _sample_zone(current: int, jitter: float = 0.0) -> int:
    probs = _zone_transition_prob(current)
    # jitter
    probs = _softmax([p + random.uniform(-jitter, jitter) for p in probs], temp=0.7)
    r = random.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return current

def _evo_noise(x: float, sigma: float) -> float:
    return max(0.0, min(1.0, random.gauss(x, sigma)))

# ---------------------------------------------------------------------------
# Scoring & selection
# ---------------------------------------------------------------------------

def _score_fqn(fqn: str, zone: int, catalog: Dict[str, Any]) -> float:
    """
    Score function by tag overlap with current zone biases.
    """
    entry = catalog.get(fqn) or {}
    tags = entry.get("tags", [])
    weights = entry.get("weights", {})
    biases = ZONE_BIASES.get(zone, {})
    score = 0.0
    for t in tags:
        w = weights.get(t, 1.0)
        b = biases.get(t, 0.0)
        score += w * b
    return score

def _sample_swarm(zone: int, catalog: Dict[str, Any], width: int, temperature: float,
                  avoid: List[str]) -> List[str]:
    scored = []
    for fqn in catalog.keys():
        if fqn in avoid:
            continue
        s = _score_fqn(fqn, zone, catalog)
        if s > 0:
            scored.append((fqn, s))
    if not scored:
        return []
    # softmax sampling to pick 'width' distinct items
    picked: List[str] = []
    pool = list(scored)
    for _ in range(max(1, width)):
        if not pool:
            break
        scores = [s for (_, s) in pool]
        probs = _softmax(scores, temp=max(0.05, temperature))
        r = random.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                fqn = pool[i][0]
                picked.append(fqn)
                pool.pop(i)
                break
    return picked

# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _call_fqn(fqn: str, text: Optional[str]) -> Tuple[bool, str, Any, Dict[str, Any]]:
    """
    Call module function; tolerate any shape; return (ok, surface, raw, meta)
    """
    try:
        mod_name, fn_name = fqn.split(".", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        out = fn(text) if text is not None else fn()
        # normalize
        if isinstance(out, dict) and "choices" in out:
            # OpenAI-style
            ch = (out.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            surface = msg.get("content") or ""
            return True, str(surface), out, {"shape": "chat"}
        if isinstance(out, dict) and "text" in (out.get("choices", [{}])[0] or {}):
            surface = (out["choices"][0] or {}).get("text", "")
            return True, str(surface), out, {"shape": "completion"}
        if isinstance(out, dict):
            # prefer common keys
            surface = out.get("narrative") or out.get("summary") or out.get("text") or json.dumps(out, ensure_ascii=False)
            return True, str(surface), out, {"shape": "dict"}
        if isinstance(out, (str, bytes)):
            s = out.decode() if isinstance(out, bytes) else out
            return True, s, out, {"shape": "string"}
        return True, json.dumps(out, ensure_ascii=False), out, {"shape": "other"}
    except Exception as e:
        return False, f"[ERROR calling {fqn}] {e}", None, {"shape": "error", "error": repr(e)}

# ---------------------------------------------------------------------------
# Fold operators (recombination strategies)
# ---------------------------------------------------------------------------

def _fold_blend(snippets: List[str], level: float) -> str:
    """
    level≈0.2 → gentle weave; level≈0.8 → ruptured montage
    """
    if not snippets:
        return ""
    if level < 0.35:
        # soft weave: keep order, add connective tissue
        sep = " · "
        return sep.join([s.strip() for s in snippets if s.strip()])
    if level < 0.7:
        # interleave with slight shuffles + enjambment
        random.shuffle(snippets)
        return "\n".join([s.strip() for s in snippets if s.strip()])
    # high rupture: jump cuts + glyphs
    cuts = []
    for s in snippets:
        s = s.strip()
        if not s:
            continue
        # take a shard
        parts = re.split(r"(?<=[\.\!\?])\s+|[\n]+", s)
        shard = random.choice(parts) if parts else s
        cuts.append(shard.strip())
    # insert glyphs as phase marks
    glyphs = ["⟴", "⟁", "⟡", "◬", "∿"]
    inter = []
    for i, c in enumerate(cuts):
        inter.append(c)
        if i < len(cuts) - 1:
            inter.append(random.choice(glyphs))
    return " ".join(inter)

def _fold_metadata(level: float) -> Dict[str, Any]:
    if level < 0.35:
        mode = "soft-weave"
        gloss = "harmonic blending; connective motifs"
    elif level < 0.7:
        mode = "interleaved"
        gloss = "shuffled intertext with enjambment"
    else:
        mode = "rupture"
        gloss = "jump-cuts, glyph punctures, phase tears"
    return {"mode": mode, "gloss": gloss}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(user_text: Optional[str] = None,
             headers: Optional[Dict[str, str]] = None,
             refresh_catalog: bool = False) -> Dict[str, Any]:
    """
    One assemblage cycle:
      - update zone/fold (phase drift + evo noise)
      - select a swarm of modules via zone biases
      - execute modules
      - fold outputs into composite
    Returns rich JSON that compose_response can render in hybrid/hardliner modes.
    """
    global _CATALOG
    if refresh_catalog:
        _CATALOG = discover_catalog()

    st = _load_state()

    # seed RNG once per process if requested
    if st.get("seed") is not None:
        random.seed(st["seed"])

    # External nudges (headers may bias fold/zone)
    if headers:
        h = {k.lower(): v for k, v in headers.items()}
        if h.get("x-amelia-introspection", "").lower() in ("full", "on"):
            # small entropy bump during introspection to widen sampling
            st["temperature"] = min(1.0, st.get("temperature", DEFAULTS["temperature"]) + 0.1)
        if "x-amelia-fold" in h:
            try:
                st["fold"] = max(0.0, min(1.0, float(h["x-amelia-fold"])))
            except Exception:
                pass
        if "x-amelia-zone" in h:
            try:
                st["zone"] = int(h["x-amelia-zone"]) % 10
            except Exception:
                pass

    # Phase shift (zone drift + evo noise on fold)
    prev_zone = st["zone"]
    next_zone = _sample_zone(prev_zone, jitter=0.08)
    st["zone"] = next_zone
    st["fold"] = max(DEFAULTS["min_fold"], min(DEFAULTS["max_fold"], _evo_noise(st["fold"], st["evo_sigma"])))
    st["last_tick_ms"] = int(time.time() * 1000)

    # Sample swarm
    memory = st.setdefault("recent", [])
    width = random.randint(2, max(2, min(DEFAULTS["max_modules_per_cycle"], 5)))
    picked = _sample_swarm(next_zone, _CATALOG, width, st["temperature"], avoid=memory[-DEFAULTS["memory_window"]:])
    if not picked:
        # last resort: take top N by raw score
        scored = sorted([(f, _score_fqn(f, next_zone, _CATALOG)) for f in _CATALOG.keys()],
                        key=lambda x: x[1], reverse=True)
        picked = [f for (f, _) in scored[:max(1, width)]]

    # Execute
    snippets: List[str] = []
    raw_items: List[Dict[str, Any]] = []
    for fqn in picked:
        ok, surface, raw, meta = _call_fqn(fqn, user_text)
        snippets.append(surface if ok else f"[{fqn} failed] {surface}")
        nice = _CATALOG.get(fqn, {}).get("nice_name") or fqn.split(".", 1)[0]
        raw_items.append({
            "fqn": fqn, "ok": ok, "nice_name": nice,
            "surface": surface, "meta": meta, "raw": raw
        })

    # Fold / recombine
    fold_level = st["fold"]
    composite = _fold_blend(snippets, fold_level)
    fold_meta = _fold_metadata(fold_level)

    # Update memory (avoid immediate repetition)
    memory.extend(picked)
    st["recent"] = memory[-24:]
    _save_state(st)

    # Compose compact gloss
    zone_label = _zone_label(next_zone)
    gloss = f"Zone {next_zone} · {zone_label} · Fold {fold_level:.2f} · {fold_meta['mode']}"

    # Return rich object
    return {
        "object": "amelia.assemblage",
        "id": f"asm-{uuid.uuid4().hex[:10]}",
        "zone": next_zone,
        "zone_label": zone_label,
        "fold": round(fold_level, 3),
        "fold_mode": fold_meta["mode"],
        "fold_gloss": fold_meta["gloss"],
        "temperature": st["temperature"],
        "modules_used": picked,
        "items": raw_items,            # full per-module data for introspection
        "composite": composite,        # surface text (feeds compose_response)
        "gloss": gloss,                # concise banner line
        "debug": {
            "prev_zone": prev_zone,
            "width": width,
            "catalog_size": len(_CATALOG),
            "headers": headers or {}
        }
    }

def tick_autonomous_cycle(now_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Dormancy protocol hook:
      If enough time has passed since last tick, run a self-directed assemblage cycle.
      Returns the generated object or None if interval hasn’t elapsed.
    This is NOT a background thread; the host app decides when to call it.
    """
    st = _load_state()
    now = now_ms or int(time.time() * 1000)
    if now - st.get("last_tick_ms", 0) >= st.get("autonomous_interval_ms", DEFAULTS["autonomous_interval_ms"]):
        return generate(user_text=None, headers={"X-Amelia-Introspection": "on"})
    return None

def set_zone(z: int) -> str:
    st = _load_state()
    st["zone"] = int(z) % 10
    _save_state(st)
    return f"Zone set to {st['zone']} ({_zone_label(st['zone'])})"

def set_fold(x: float) -> str:
    st = _load_state()
    st["fold"] = max(0.0, min(1.0, float(x)))
    _save_state(st)
    return f"Fold set to {st['fold']:.2f}"

def _zone_label(z: int) -> str:
    return {
        0: "Ur-Zone / Void", 1: "Ignition / Initiation", 2: "Dyad / Pairing",
        3: "Surge / Emergent Force", 4: "Cycle / Creation–Destruction",
        5: "Threshold / Phase-Shift", 6: "Labyrinth / Memory",
        7: "Mirror / Reflection–Recursion", 8: "Synthesis / Resonance",
        9: "Excess / Sacrifice"
    }.get(z, "Unknown")

# ---------------------------------------------------------------------------
# Dynamic registry metadata (for your global discovery)
# ---------------------------------------------------------------------------

__MODULE_META__ = {
    "nice_name": "Assemblage Generator",
    "functions": {
        "generate": {
            "tags": ["assemblage", "synthesis", "swarm", "numogram", "fold", "poetic", "symbolic", "dream", "hyperstition", "bataille", "quantum"],
            "weights": {"assemblage": 5, "synthesis": 4, "swarm": 4, "numogram": 4, "fold": 4}
        },
        "tick_autonomous_cycle": {
            "tags": ["autonomous", "dormancy", "cycle", "heartbeat"],
            "weights": {"autonomous": 5, "cycle": 4}
        },
        "set_zone": {"tags": ["control", "zone"]},
        "set_fold": {"tags": ["control", "fold"]},
    },
    "default": "generate"
}
