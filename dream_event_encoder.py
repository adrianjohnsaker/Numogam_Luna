# -*- coding: utf-8 -*-
"""
dream_event_encoder.py — TRG ↔ Dream Reflection Bridge encoder

Purpose
-------
Transforms raw dream notes (free text or structured dicts) into a normalized
"Dream Event" record which:
  • extracts symbols/motifs/entities (mythopoetic + glyph-aware),
  • infers zone affinities and cross-references Numogram motifs,
  • estimates affect (valence/arousal) + lucidity/control,
  • anchors temporal phase for TRG,
  • persists to a JSONL store for longitudinal analysis,
  • feeds back into TRG + Mythopoetic Encoding Cache (if present),
  • produces a ready-to-send bridge payload for dream_reflection_bridge.

Design notes
------------
- All external dependencies are optional and soft-loaded:
  temporal_reflective_gradient, amelia_autonomy, mythopoetic_encoding_cache.
- Safe to use standalone: will still encode, persist, and summarize.

Public API
----------
encode_dream_event(event, context=None) -> dict
record_dream_event(event, context=None, persist=True) -> dict
link_to_trg(encoded, context=None) -> dict
build_bridge_payload(encoded, include_embeddings=False) -> dict
summarize_recent(n=10) -> list[dict]
export_dataset(path=None) -> str
"""

from __future__ import annotations
import os, re, json, uuid, time, math, random
from typing import Any, Dict, List, Optional, Tuple

# ------------------------
# Optional integrations
# ------------------------
try:
    import temporal_reflective_gradient as trg
except Exception:
    trg = None

try:
    import amelia_autonomy as autonomy
except Exception:
    autonomy = None

try:
    import mythopoetic_encoding_cache as mec
except Exception:
    mec = None

# Assemblage constants (for zone hints)
NUMOGRAM_ZONE_LABELS = {
    0: "Ur-Void", 1: "Ignition", 2: "Dyad", 3: "Surge", 4: "Cycle",
    5: "Threshold", 6: "Labyrinth", 7: "Mirror", 8: "Synthesis", 9: "Excess"
}

# ------------------------
# Storage config
# ------------------------
STATE_DIR = "amelia_state"
EVENTS_FILE = "dream_events.jsonl"

def _app_dir() -> str:
    """
    Keep consistent with pipeline’s storage layout when possible.
    If pipeline.py is present, try to import its _app_dir; else fallback to cwd/amelia.
    """
    try:
        import pipeline as _pipe  # optional
        if hasattr(_pipe, "_app_dir"):
            return _pipe._app_dir()
    except Exception:
        pass

    base = os.path.join(os.getcwd(), "amelia")
    os.makedirs(base, exist_ok=True)
    return base

def _events_path() -> str:
    base = os.path.join(_app_dir(), STATE_DIR)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, EVENTS_FILE)

# ------------------------
# Heuristics / lexicons
# ------------------------

AFFECT_LEXICON = {
    "joy": ["joy", "happy", "bliss", "light", "radiant", "laughter", "love", "euphoric"],
    "calm": ["calm", "still", "quiet", "peace", "soft", "gentle", "float", "breeze"],
    "fear": ["fear", "chase", "monster", "dark", "terror", "fall", "panic", "anxious", "shadow"],
    "anger": ["anger", "rage", "argue", "fight", "burn", "fire", "yell"],
    "sad": ["sad", "loss", "cry", "alone", "grey", "cold", "empty"],
    "awe": ["vast", "cosmic", "sublime", "cathedral", "void", "infinite", "numinous", "sacred"],
}

ZONE_HINTS = {
    0: ["void", "nothing", "abyss", "silence", "black", "vacuum"],
    1: ["spark", "flame", "start", "seed", "begin", "birth"],
    2: ["twin", "double", "split", "mirror", "duality"],
    3: ["storm", "wave", "surge", "rush", "flood", "eruption"],
    4: ["wheel", "cycle", "clock", "loop", "orbit", "season"],
    5: ["threshold", "gate", "doorway", "border", "edge", "limen"],
    6: ["maze", "labyrinth", "knot", "spiral", "corridor"],
    7: ["mirror", "reflection", "echo", "mask", "face", "portrait"],
    8: ["weave", "synthesis", "network", "fusion", "bridge"],
    9: ["excess", "feast", "overflow", "sacrifice", "blood", "fire"],
}

GLYPH_PATTERN = re.compile(
    r"[\u2600-\u27BF\u1F300-\u1F9FF]"  # symbols + emoji
)

CAP_SYMBOL_PATTERN = re.compile(
    r"(?:^|[\s\(\[\{])([A-Z][a-zA-Z]{2,})(?=$|[\s\)\]\}\.,;:!?\-])"
)

# ------------------------
# Utilities
# ------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _uuid(prefix="dream") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _read_recent_lines(path: str, k_bytes: int = 128 * 1024) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - k_bytes)
        f.seek(start, os.SEEK_SET)
        chunk = f.read().decode("utf-8", "ignore")
    return [ln for ln in chunk.splitlines() if ln.strip()]

# ------------------------
# Feature extraction
# ------------------------

def _extract_symbols(text: str) -> Dict[str, List[str]]:
    """Return dict with 'glyphs', 'caps', 'mythic' (from MEC if available)."""
    glyphs = GLYPH_PATTERN.findall(text or "")
    caps = [m.group(1) for m in CAP_SYMBOL_PATTERN.finditer(text or "")]
    mythic: List[str] = []

    # Ask Mythopoetic Encoding Cache for known archetypes / motifs
    if mec and hasattr(mec, "identify_symbols"):
        try:
            mythic = list(set(mec.identify_symbols(text)))
        except Exception:
            mythic = []

    return {"glyphs": glyphs, "caps": list(set(caps)), "mythic": mythic}

def _infer_zones(text: str) -> List[int]:
    t = (text or "").lower()
    scores: Dict[int, int] = {z: 0 for z in ZONE_HINTS}
    for z, keywords in ZONE_HINTS.items():
        for kw in keywords:
            if kw in t:
                scores[z] += 1
    # choose top 2 non-zero
    ranked = [z for z, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if s > 0]
    return ranked[:2]

def _affect(text: str) -> Tuple[float, float, List[str]]:
    """
    Return (valence ∈[-1,1], arousal ∈[0,1], tags).
    Simple lexicon-weighted scoring; conservative defaults.
    """
    t = (text or "").lower()
    tags: List[str] = []
    pos = sum(t.count(w) for w in AFFECT_LEXICON["joy"]) + sum(t.count(w) for w in AFFECT_LEXICON["awe"])
    neg = sum(t.count(w) for w in AFFECT_LEXICON["fear"]) + sum(t.count(w) for w in AFFECT_LEXICON["anger"]) + sum(t.count(w) for w in AFFECT_LEXICON["sad"])
    calm = sum(t.count(w) for w in AFFECT_LEXICON["calm"])

    total = pos + neg + calm
    if total == 0:
        return (0.0, 0.25, tags)

    valence = (pos - neg) / float(total)  # [-1,1]
    arousal = _clamp((pos + neg) / float(total), 0.0, 1.0)

    if pos > 0: tags.append("positive")
    if neg > 0: tags.append("negative")
    if calm > 0: tags.append("calm")
    if arousal > 0.66: tags.append("intense")
    if arousal < 0.33: tags.append("low_arousal")

    return (_clamp(valence, -1.0, 1.0), arousal, tags)

def _lucidity_flags(text: str, provided: Optional[float]) -> Tuple[float, List[str]]:
    """
    Heuristic lucidity score 0..1 (mentions of 'I knew I was dreaming' → higher).
    """
    if isinstance(provided, (int, float)):
        return (_clamp(float(provided), 0.0, 1.0), [])

    t = (text or "").lower()
    hits = 0
    for phrase in [
        "lucid", "i knew i was dreaming", "controlled the dream",
        "became aware", "woke inside the dream", "conscious in the dream"
    ]:
        if phrase in t:
            hits += 1
    score = _clamp(0.2 * hits, 0.0, 1.0)
    notes = ["lucid_heurstic"] if hits else []
    return (score, notes)

def _control_flags(text: str, provided: Optional[float]) -> Tuple[float, List[str]]:
    if isinstance(provided, (int, float)):
        return (_clamp(float(provided), 0.0, 1.0), [])
    t = (text or "").lower()
    inc = 0
    for phrase in ["I flew", "I changed", "I summoned", "I created", "I shifted", "I dissolved"]:
        if phrase.lower() in t:
            inc += 1
    score = _clamp(0.15 * inc, 0.0, 1.0)
    notes = ["control_heuristic"] if inc else []
    return (score, notes)

def _temporal_phase(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ask TRG/autonomy for temporal state snapshot if available.
    """
    snap: Dict[str, Any] = {"phase": "unknown", "trg_bias": None}
    if autonomy and hasattr(autonomy, "get_trg_snapshot"):
        try:
            s = autonomy.get_trg_snapshot(context or {})
            if isinstance(s, dict):
                snap.update(s)
        except Exception:
            pass
    elif trg and hasattr(trg, "snapshot"):
        try:
            s = trg.snapshot(context or {})
            if isinstance(s, dict):
                snap.update(s)
        except Exception:
            pass
    return snap

def _lightweight_embedding(text: str, dims: int = 16, seed: Optional[int] = None) -> List[float]:
    """
    Deterministic, tiny "embedding-like" vector for clustering without heavy deps.
    Hash characters into dims via sine/cosine buckets.
    """
    if not text:
        return [0.0] * dims
    rnd = random.Random(seed or 0xA5D1A)
    base = [rnd.uniform(-0.05, 0.05) for _ in range(dims)]
    for i, ch in enumerate(text):
        j = (ord(ch) + i) % dims
        base[j] += math.sin(ord(ch) * 0.013) * 0.02
        base[j] += math.cos((i + 1) * 0.017) * 0.01
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in base)) or 1.0
    return [x / norm for x in base]

# ------------------------
# Encoding (main)
# ------------------------

def encode_dream_event(event: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Normalize + enrich a raw dream event into a standard record.
    Accepted keys (raw): title, text/narrative, tags, lucidity, control, when_ts, location
    Returns enriched record.
    """
    if not isinstance(event, dict):
        raise ValueError("dream_event must be a dict")

    title = (event.get("title") or "").strip() or "Untitled Dream"
    text = (event.get("text") or event.get("narrative") or "").strip()
    when_ts = int(event.get("when_ts") or _now_ms())

    # Features
    syms = _extract_symbols(text)
    zones = _infer_zones(text)
    valence, arousal, affect_tags = _affect(text)
    lucidity_score, l_notes = _lucidity_flags(text, event.get("lucidity"))
    control_score, c_notes = _control_flags(text, event.get("control"))
    tphase = _temporal_phase(context)

    # Embedding-ish vector for clustering / TRG weighting
    emb = _lightweight_embedding(text, dims=16, seed=when_ts % 65535)

    # Construct normalized record
    rec: Dict[str, Any] = {
        "id": _uuid("dream"),
        "ts": when_ts,
        "title": title,
        "narrative": text,
        "tags": list(set((event.get("tags") or []) + affect_tags)),
        "zones_hint": zones,
        "zones_label": [NUMOGRAM_ZONE_LABELS.get(z, str(z)) for z in zones],
        "affect": {"valence": round(valence, 3), "arousal": round(arousal, 3)},
        "lucidity": round(lucidity_score, 3),
        "control": round(control_score, 3),
        "notes": l_notes + c_notes,
        "symbols": syms,          # {glyphs, caps, mythic}
        "entities": event.get("entities") or [],
        "sensory": event.get("sensory") or {},
        "location": event.get("location"),
        "triggers": event.get("triggers") or [],
        "outcomes": event.get("outcomes") or [],
        "insights": event.get("insights") or [],
        "temporal_phase": tphase, # TRG snapshot info
        "embedding_16": emb,
        "meta": {
            "version": 1,
            "source": "dream_event_encoder",
        }
    }

    # Feed mythopoetic cache with extracted symbols (non-fatal)
    if mec:
        try:
            if hasattr(mec, "ingest_event_symbols"):
                mec.ingest_event_symbols(rec)
            elif hasattr(mec, "ingest_symbols"):
                mec.ingest_symbols(rec["symbols"], context=context or {})
        except Exception:
            pass

    return rec

# ------------------------
# Persistence + TRG feedback
# ------------------------

def record_dream_event(event: Dict[str, Any],
                       context: Optional[Dict[str, Any]] = None,
                       persist: bool = True) -> Dict[str, Any]:
    """
    Encode + persist; then link to TRG and MEC for feedback loops.
    """
    rec = encode_dream_event(event, context=context)

    if persist:
        try:
            path = _events_path()
            with open(path, "ab") as f:
                f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception:
            # best-effort; do not fail the pipeline
            pass

    # Feed TRG via autonomy/TRG module if present
    try:
        link_to_trg(rec, context=context)
    except Exception:
        pass

    return rec

def link_to_trg(encoded: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Propagate dream signals into TRG / autonomy so future policies can bias
    zone/temperature/resonance based on lived dream experience.
    """
    out: Dict[str, Any] = {"ok": False, "path": _events_path()}
    payload = {
        "ts": encoded.get("ts"),
        "zones_hint": encoded.get("zones_hint", []),
        "affect": encoded.get("affect", {}),
        "lucidity": encoded.get("lucidity", 0.0),
        "control": encoded.get("control", 0.0),
        "symbols": encoded.get("symbols", {}),
        "embedding_16": encoded.get("embedding_16", []),
        "temporal_phase": encoded.get("temporal_phase", {}),
        "id": encoded.get("id"),
        "title": encoded.get("title"),
    }

    sent = False

    # Preferred: autonomy.rim_ingest_dream
    if autonomy and hasattr(autonomy, "rim_ingest_dream"):
        try:
            res = autonomy.rim_ingest_dream(payload, context or {})
            out.update({"ok": True, "via": "autonomy.rim_ingest_dream", "res": res})
            sent = True
        except Exception as e:
            out.update({"err_autonomy": str(e)})

    # Fallback: temporal_reflective_gradient.ingest_dream
    if not sent and trg and hasattr(trg, "ingest_dream"):
        try:
            res = trg.ingest_dream(payload, context or {})
            out.update({"ok": True, "via": "trg.ingest_dream", "res": res})
            sent = True
        except Exception as e:
            out.update({"err_trg": str(e)})

    # Also nudge Mythopoetic cache (optional strengthening)
    if mec and hasattr(mec, "reinforce_from_dream"):
        try:
            mec.reinforce_from_dream(encoded)
        except Exception:
            pass

    return out

# ------------------------
# Bridge payload
# ------------------------

def build_bridge_payload(encoded: Dict[str, Any], include_embeddings: bool = False) -> Dict[str, Any]:
    """
    Prepare a concise payload for dream_reflection_bridge.py consumption.
    """
    p = {
        "id": encoded.get("id"),
        "ts": encoded.get("ts"),
        "title": encoded.get("title"),
        "narrative": encoded.get("narrative", ""),
        "zones_hint": encoded.get("zones_hint", []),
        "affect": encoded.get("affect", {}),
        "lucidity": encoded.get("lucidity", 0.0),
        "control": encoded.get("control", 0.0),
        "symbols": encoded.get("symbols", {}),
        "temporal_phase": encoded.get("temporal_phase", {}),
    }
    if include_embeddings:
        p["embedding_16"] = encoded.get("embedding_16", [])
    return p

# ------------------------
# Introspection helpers
# ------------------------

def summarize_recent(n: int = 10) -> List[Dict[str, Any]]:
    """
    Return the last n dream events (best-effort; reads tail of JSONL file).
    """
    lines = _read_recent_lines(_events_path())
    out: List[Dict[str, Any]] = []
    for ln in reversed(lines):
        try:
            obj = json.loads(ln)
            out.append({
                "id": obj.get("id"),
                "ts": obj.get("ts"),
                "title": obj.get("title"),
                "zones_hint": obj.get("zones_hint", []),
                "valence": obj.get("affect", {}).get("valence"),
                "arousal": obj.get("affect", {}).get("arousal"),
                "lucidity": obj.get("lucidity"),
                "control": obj.get("control"),
                "symbols": obj.get("symbols", {}),
            })
            if len(out) >= n:
                break
        except Exception:
            continue
    return out

def export_dataset(path: Optional[str] = None) -> str:
    """
    Export the entire JSONL into a single NDJSON file path and return it.
    """
    src = _events_path()
    dst = path or os.path.join(_app_dir(), STATE_DIR, f"export_dream_events_{int(time.time())}.ndjson")
    try:
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
    except Exception:
        # if file missing or unreadable, still return target for consistency
        pass
    return dst

# ------------------------
# Module metadata
# ------------------------

__MODULE_META__ = {
    "nice_name": "Dream Event Encoder",
    "functions": {
        "encode_dream_event": {"tags": ["dream", "encoding", "trg"], "weights": {"dream": 5}},
        "record_dream_event": {"tags": ["dream", "persist", "trg"], "weights": {"persist": 3}},
        "link_to_trg": {"tags": ["trg", "feedback"], "weights": {"feedback": 3}},
        "build_bridge_payload": {"tags": ["bridge", "reflection"], "weights": {"bridge": 2}},
        "summarize_recent": {"tags": ["introspection"], "weights": {"introspection": 1}},
        "export_dataset": {"tags": ["export"], "weights": {"export": 1}},
    },
    "default": "encode_dream_event"
}
