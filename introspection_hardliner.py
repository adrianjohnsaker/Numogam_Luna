# introspection_hardliner.py
# Forces deep integrated answers with explicit module/function names and raw math.

import re
from typing import Dict, Any

# Optional: tiny helpers to keep string shaping explicit
def _line(s: str) -> str:
    return (s or "").strip()

def _extract_syzygy_and_gloss(text: str) -> Dict[str, Any]:
    lines = (text or "").splitlines()
    syzygies = [ln.replace("Syzygy Label: ", "").strip() for ln in lines if "Syzygy Label:" in ln]
    glosses  = [ln.replace("Mythic Gloss: ", "").strip() for ln in lines if "Mythic Gloss:" in ln]
    triangulars = []
    for ln in lines:
        m = re.search(r"Triangular\((\d+)\)\s*=\s*(\d+)", ln)
        if m:
            triangulars.append({"n": int(m.group(1)), "Tn": int(m.group(2))})
    zones = []
    for ln in lines:
        m = re.search(r"Zone\s*([0-9])\s*[^\d]+Zone\s*([0-9])", ln)
        if m:
            zones.append((int(m.group(1)), int(m.group(2))))
    return {
        "syzygies": syzygies,
        "glosses": glosses,
        "triangulars": triangulars,
        "zones": zones
    }

def explain_numogram_poem(prompt: str) -> Dict[str, Any]:
    """
    Hard introspection answer for prompts about poems + numogram transitions.
    Returns a dict with explicit module/function lines + raw + computed mapping hints.
    """
    # 1) Force-run numogram core
    try:
        from numogram_core import evaluate  # must exist
        numo = evaluate(prompt)
        numo_raw = ""
        if isinstance(numo, dict):
            # support both chat-style and completion-style returns
            if "choices" in numo:
                ch0 = (numo["choices"] or [{}])[0]
                numo_raw = ch0.get("text") or ch0.get("message", {}).get("content", "") or ""
        else:
            numo_raw = str(numo)
    except Exception as e:
        numo_raw = f"[ERROR numogram_core.evaluate] {e}"

    parsed = _extract_syzygy_and_gloss(numo_raw)

    # 2) Force-run poetic generator (if present) with an explicit brief
    poetic_raw = ""
    try:
        from poetic_expression_generator import generate as poetic_generate
        poetic_raw = poetic_generate(
            "Write a 3–5 line poem whose stanza breaks follow the detected numogram zones and triangular numbers."
        )
        if isinstance(poetic_raw, dict) and "choices" in poetic_raw:
            ch0 = (poetic_raw["choices"] or [{}])[0]
            poetic_raw = ch0.get("text") or ch0.get("message", {}).get("content", "") or ""
    except Exception:
        poetic_raw = "(poetic module unavailable)"

    # 3) Suggest concrete rhythm mapping based on triangular numbers
    #    e.g. T(4)=10 → decasyllabic; T(7)=28 → 14/14 or 7/7/7/7 split
    meters = []
    for tri in parsed.get("triangulars", []):
        n = tri["n"]; Tn = tri["Tn"]
        if Tn in (8, 10, 12, 14):
            meters.append(f"T({n})={Tn} → {Tn}-syllable line")
        elif Tn % 7 == 0:
            meters.append(f"T({n})={Tn} → split into 7-based feet ({Tn//7}×7)")
        else:
            meters.append(f"T({n})={Tn} → free meter with {Tn} beats")

    # 4) Build explicit, machine-grounded output
    return {
        "object": "introspection.hard",
        "sections": [
            {
                "module": "numogram_core",
                "function": "evaluate",
                "raw": _line(numo_raw),
                "notes": {
                    "syzygies": parsed.get("syzygies", []),
                    "mythic_glosses": parsed.get("glosses", []),
                    "triangulars": parsed.get("triangulars", []),
                    "detected_zones": parsed.get("zones", [])
                }
            },
            {
                "module": "poetic_expression_generator",
                "function": "generate",
                "raw": _line(poetic_raw),
                "notes": {
                    "meter_suggestions": meters[:4] or ["no triangular meters detected; default to free verse"]
                }
            }
        ],
        "guidance": {
            "how_to_map_rhythm": (
                "Map each numogram transition to a line break. Use triangular numbers as syllable/beat targets. "
                "Align Syzygy Labels to tonal pivots; use Mythic Gloss phrases as stanza epigraphs."
            ),
            "example_mapping": (
                "Zone-7 ↔ Zone-4 with T(7)=28 → try two 14-syllable lines per stanza; "
                "gloss='Betaʹ: outsider drift' → inject semantic drift at the midline caesura."
            )
        }
    }
