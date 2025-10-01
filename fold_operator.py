# -*- coding: utf-8 -*-
"""
fold_operator.py — Textual Fold/Blend engine for Amelia

Goal
-----
Apply "fold" operations which modulate how multiple module outputs are blended.
- Low intensity  (≈0.0–0.4): subtle weaving / coherence-preserving.
- Medium         (≈0.4–0.7): mixed interleave / controlled drift.
- High intensity (≈0.7–1.0): violent rupture / cut-ups / unexpected juxtaposition.

This produces different "textures" of expression without manual prompt tweaks.

Public API
----------
apply(chunks, intensity=0.5, mode="poetic", seed=None, max_len=1200) -> dict
  chunks: List[str] of module outputs (e.g., dream, numogram, symbolic, poetic)
  intensity: float in [0,1]
  mode: "poetic" | "mythic" | "analytic" | "glyph"
  seed: optional int for deterministic folds
  max_len: soft cap for resulting text

Return dict keys:
  {
    "text": str,                  # folded composite
    "intensity": float,
    "mode": str,
    "ops": [ {op metadata...} ],
    "texture": {"coherence":..., "rupture":..., "density":..., "entropy":...},
    "used_chunks": int,
  }

Notes
-----
- No external dependencies.
- Stable under non-ASCII inputs.
- Safe: gracefully degrades if inputs are empty.
"""

from __future__ import annotations
import re
import random
from typing import List, Dict, Any, Tuple

# ----------------------------
# Helpers
# ----------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')
_CLAUSE_SPLIT = re.compile(r'[,;:—–-]\s+')
_WORD_SPLIT = re.compile(r'\s+')

def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return _SENT_SPLIT.split(text)

def _split_clauses(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return _CLAUSE_SPLIT.split(text)

def _split_words(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return _WORD_SPLIT.split(text)

def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"

def _score_density(text: str) -> float:
    # Naive lexical density proxy
    words = _split_words(text)
    if not words:
        return 0.0
    longish = sum(1 for w in words if len(w) >= 6)
    return round(longish / max(1, len(words)), 3)

def _score_entropy(text: str) -> float:
    # Naive char-level entropy-ish proxy
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    import math
    total = len(text)
    ent = -sum((c/total) * math.log2(c/total) for c in freq.values())
    # Normalize rough range to 0..1 for typical prose
    return round(min(1.0, ent / 6.0), 3)

# ----------------------------
# Micro-operations (weaving vs rupture)
# ----------------------------

def _op_subtle_stitch(a: str, b: str) -> Tuple[str, Dict[str, Any]]:
    """Blend by taking a few sentences of A then a sentence from B as connective tissue."""
    sa, sb = _split_sentences(a), _split_sentences(b)
    if not sa and not sb:
        return "", {"op": "subtle_stitch", "note": "both empty"}
    out: List[str] = []
    i = j = 0
    # take 2 sentences from A, 1 from B (if available)
    while i < len(sa) or j < len(sb):
        if i < len(sa):
            out.append(sa[i]); i += 1
        if i < len(sa):
            out.append(sa[i]); i += 1
        if j < len(sb):
            out.append(sb[j]); j += 1
    text = " ".join(out)
    return text, {"op": "subtle_stitch", "taken": {"A": min(i, len(sa)), "B": min(j, len(sb))}}

def _op_interleave_clauses(a: str, b: str) -> Tuple[str, Dict[str, Any]]:
    """Alternate clauses from A and B to create medium drift."""
    ca, cb = _split_clauses(a), _split_clauses(b)
    out: List[str] = []
    i = j = 0
    while i < len(ca) or j < len(cb):
        if i < len(ca):
            out.append(ca[i]); i += 1
        if j < len(cb):
            out.append(cb[j]); j += 1
    # Rebuild with commas/semicolons as breath marks
    text = ", ".join([c.strip() for c in out if c.strip()]) + "."
    return text, {"op": "interleave_clauses", "lenA": len(ca), "lenB": len(cb)}

def _op_cutup_fragments(a: str, b: str, shock: float) -> Tuple[str, Dict[str, Any]]:
    """Hard cut-up: sample word fragments from A and B. 'shock' controls rupture ratio."""
    wa, wb = _split_words(a), _split_words(b)
    if not wa and not wb:
        return "", {"op": "cutup_fragments", "note": "both empty"}
    n = max(12, int((len(wa) + len(wb)) * 0.15))
    out: List[str] = []
    for _ in range(n):
        pick_b = random.random() < shock
        pool = wb if pick_b and wb else wa
        if not pool:
            pool = wa or wb
        # pick a fragment or word
        token = random.choice(pool)
        if len(token) > 6 and random.random() < 0.3:
            # split word into fragment to increase rupture
            k = random.randint(3, max(4, len(token)-2))
            if random.random() < 0.5:
                token = token[:k]
            else:
                token = token[-k:]
        out.append(token)
    # Punctuate with staccato rhythm
    for i in range(4, len(out), 7):
        out[i] = out[i] + "."
    text = " ".join(out)
    return text, {"op": "cutup_fragments", "shock": round(shock, 3), "n": n}

def _op_bridge_metaphor(a: str, b: str) -> Tuple[str, Dict[str, Any]]:
    """Create a metaphoric bridge sentence between two blocks."""
    ka = _truncate(a.strip(), 140)
    kb = _truncate(b.strip(), 140)
    bridge = f"As {ka} converges with {kb}, a new contour precipitates between them."
    return bridge, {"op": "bridge_metaphor"}

def _op_caesura(text: str, ratio: float) -> Tuple[str, Dict[str, Any]]:
    """Insert line breaks to create breathing room; ratio controls density of breaks."""
    words = _split_words(text)
    if not words:
        return "", {"op": "caesura", "note": "empty"}
    out = []
    step = max(5, int(15 - 12 * ratio))  # more breaks at higher ratio
    for i, w in enumerate(words, 1):
        out.append(w)
        if i % step == 0:
            out.append("\n")
    return " ".join(out).replace(" \n ", "\n"), {"op": "caesura", "step": step}

# ----------------------------
# Top-level fold strategy
# ----------------------------

def _pairwise(chunks: List[str]) -> List[Tuple[str, str]]:
    if len(chunks) < 2:
        return [(chunks[0] if chunks else "", "")]
    pairs = []
    for i in range(0, len(chunks)-1, 2):
        pairs.append((chunks[i], chunks[i+1]))
    if len(chunks) % 2 == 1:
        pairs.append((chunks[-1], ""))
    return pairs

def _mode_filter(text: str, mode: str) -> str:
    """Light post-style shaping per mode."""
    t = text.strip()
    if not t:
        return t
    if mode == "glyph":
        # compress to emblematic lines
        sentences = _split_sentences(t)
        keep = [s for s in sentences if len(s.strip()) <= 100]
        return "\n".join(keep[:6]) or _truncate(t, 280)
    if mode == "analytic":
        # crispen punctuation to declaratives
        t = re.sub(r'[;:—–]', ', ', t)
        t = re.sub(r'\s+', ' ', t)
        return t
    if mode == "mythic":
        # add gentle cadence with line breaks
        t, _ = _op_caesura(t, ratio=0.35)
        return t
    # poetic: leave as is with mild caesura
    t, _ = _op_caesura(t, ratio=0.20)
    return t

def apply(chunks: List[str],
          intensity: float = 0.5,
          mode: str = "poetic",
          seed: int | None = None,
          max_len: int = 1200) -> Dict[str, Any]:
    """
    Fold multiple textual chunks into a single composite expression.

    intensity ∈ [0,1]:
      0.0–0.4  → low fold (subtle weaving)
      0.4–0.7  → medium fold (interleave, bridges)
      0.7–1.0  → high fold (cut-up, rupture, unexpected juxtapositions)
    """
    rnd_state = None
    if seed is not None:
        rnd_state = random.getstate()
        random.seed(seed)

    try:
        chunks = [c for c in (chunks or []) if isinstance(c, str) and c.strip()]
        if not chunks:
            return {
                "text": "",
                "intensity": intensity,
                "mode": mode,
                "ops": [{"op": "noop", "note": "no chunks"}],
                "texture": {"coherence": 0.0, "rupture": 0.0, "density": 0.0, "entropy": 0.0},
                "used_chunks": 0,
            }

        ops_trace: List[Dict[str, Any]] = []
        composites: List[str] = []

        # Choose strategy per pair based on intensity
        pairs = _pairwise(chunks)
        for (a, b) in pairs:
            if intensity <= 0.40:
                # Low: subtle stitch + optional bridge
                stitched, meta1 = _op_subtle_stitch(a, b)
                ops_trace.append(meta1)
                if random.random() < 0.45:  # occasional metaphor bridge
                    bridge, meta2 = _op_bridge_metaphor(a, b)
                    ops_trace.append(meta2)
                    segment = f"{stitched}\n{bridge}"
                else:
                    segment = stitched
            elif intensity <= 0.70:
                # Medium: clause interleave + bridge
                inter, meta1 = _op_interleave_clauses(a, b)
                ops_trace.append(meta1)
                bridge, meta2 = _op_bridge_metaphor(a, b)
                ops_trace.append(meta2)
                segment = f"{inter}\n{bridge}"
            else:
                # High: violent rupture — cut-up with shock tied to intensity
                shock = 0.55 + 0.4 * (intensity - 0.70) / 0.30  # ~0.55..0.95
                cut, meta1 = _op_cutup_fragments(a, b, shock=shock)
                ops_trace.append(meta1)
                # sometimes weave a small stitch to add ghost-coherence
                if random.random() < 0.35 and a and b:
                    ghost, meta2 = _op_subtle_stitch(a, b)
                    ops_trace.append(meta2)
                    segment = f"{cut}\n{ghost}"
                else:
                    segment = cut

            composites.append(segment.strip())

        # Join all pair-composites
        composite = "\n\n".join([c for c in composites if c])

        # Post shaping (mode) + global caesura proportional to intensity
        composite = _mode_filter(composite, mode=mode)
        composite, caes_meta = _op_caesura(composite, ratio=0.15 + 0.5 * intensity)
        ops_trace.append(caes_meta)

        # Trim
        composite = _truncate(composite.strip(), max_len)

        # Texture metrics
        density = _score_density(composite)
        entropy = _score_entropy(composite)
        # heuristic coherence vs rupture
        rupture = round(0.2 + 0.7 * intensity, 3)
        coherence = round(1.0 - rupture * 0.85 + (0.15 * (1.0 - entropy)), 3)
        coherence = max(0.0, min(1.0, coherence))

        return {
            "text": composite,
            "intensity": round(float(intensity), 3),
            "mode": mode,
            "ops": ops_trace,
            "texture": {
                "coherence": coherence,
                "rupture": rupture,
                "density": density,
                "entropy": entropy
            },
            "used_chunks": len(chunks),
        }
    finally:
        # restore RNG if we changed it
        if seed is not None and rnd_state is not None:
            random.setstate(rnd_state)
