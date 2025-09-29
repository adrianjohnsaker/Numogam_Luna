# -*- coding: utf-8 -*-
"""
pipeline.py — Amelia Orchestrator (Android / Chaquopy)

End-to-end flow:
  1) classify(text)              -> intent, tags
  2) select_modules(intent, ...) -> ordered module specs (name, fn, weight)
  3) run_modules(...)            -> parallel execution with per-stage timeouts
  4) compose_response(...)       -> deterministic, OpenAI-style JSON output
  5) persist interaction         -> light JSONL memory

This file favors graceful degradation:
  - If a module isn't present, it's skipped.
  - If a call times out or errors, we continue.
  - A safe, meaningful response is always returned.

It cooperates with:
  - python_hook.execute_custom_function(module.func, *args, **kwargs)
  - python_hook._app_context (for memory store path)

You can expand module registry as you add stacks (Evolution, Poetic, etc.).
"""

from __future__ import annotations

import os
import re
import json
import time
import uuid
import queue
import types
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Threaded execution for timeouts
import concurrent.futures as futures

# Access the hook for init, reflection, and dynamic execution
import python_hook as hook


# =============================================================================
# Configuration
# =============================================================================

CONFIG: Dict[str, Any] = {
    "timeouts": {
        "classifier_sec": 0.20,
        "module_sec": 1.50,        # per-module max runtime
        "compose_sec": 0.40,
        "overall_sec": 4.50
    },
    "parallel_workers": 4,
    "memory": {
        "enabled": True,
        "filename": "amelia_memory.jsonl",
        "max_recent": 6,           # recent records pulled into context
        "truncate_bytes": 256_000  # safeguard
    },
    "style": {
        "prefix": "⟡ Amelia ·",
        "sections": True,
        "show_sources": False,     # if True, appends a small sources summary
        "max_module_snippets": 3
    }
}


# =============================================================================
# Utilities
# =============================================================================

def _log(msg: str) -> None:
    if getattr(hook, "_DEBUG", True):
        print(f"[pipeline] {msg}")

def _log_exc(prefix: str, e: BaseException) -> None:
    print(f"[pipeline:ERROR] {prefix}: {e}")
    traceback.print_exc()

def _now_ms() -> int:
    return int(time.time() * 1000)

def _gen_id(prefix: str = "amelia") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def _truncate(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


# =============================================================================
# Memory Store (very light JSONL)
# =============================================================================

class MemoryStore:
    def __init__(self, filename: str):
        self.filename = filename
        self.base_dir = self._resolve_dir()
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, self.filename)

    def _resolve_dir(self) -> str:
        # Prefer Android internal storage via Context if available
        try:
            ctx = getattr(hook, "_app_context", None)
            if ctx is not None:
                files_dir = ctx.getFilesDir().getAbsolutePath()
                return os.path.join(files_dir, "amelia")
        except Exception as e:
            _log_exc("MemoryStore._resolve_dir", e)
        # Fallback to current working dir
        return os.path.join(os.getcwd(), "amelia")

    def append(self, record: Dict[str, Any]) -> None:
        if not CONFIG["memory"]["enabled"]:
            return
        try:
            data = _safe_json(record)
            if len(data.encode("utf-8")) > CONFIG["memory"]["truncate_bytes"]:
                # Trim bloated entries gently
                record = {k: _truncate(_safe_json(v), 16_000) for k, v in record.items()}
                data = _safe_json(record)
            with open(self.path, "ab") as f:
                f.write((data + "\n").encode("utf-8"))
        except Exception as e:
            _log_exc("MemoryStore.append", e)

    def recent(self, k: int) -> List[Dict[str, Any]]:
        if not CONFIG["memory"]["enabled"]:
            return []
        try:
            if not os.path.exists(self.path):
                return []
            lines: List[str] = []
            with open(self.path, "rb") as f:
                # Read last ~64KB to avoid huge files
                f.seek(0, os.SEEK_END)
                size = f.tell()
                back = min(size, 64 * 1024)
                f.seek(size - back if size > back else 0, os.SEEK_SET)
                chunk = f.read().decode("utf-8", errors="ignore")
                lines = [ln for ln in chunk.splitlines() if ln.strip()]

            items: List[Dict[str, Any]] = []
            for ln in lines[-(k * 2):]:  # read a bit extra, non-fatal
                try:
                    items.append(json.loads(ln))
                except Exception:
                    continue
            return items[-k:]
        except Exception as e:
            _log_exc("MemoryStore.recent", e)
            return []


MEM = MemoryStore(CONFIG["memory"]["filename"])


# =============================================================================
# Classification
# =============================================================================

INTENTS = ["CHAT", "DREAM", "NUMOGRAM", "POETIC", "SCIENCE", "SYSTEM", "IMAGE"]

KEYWORDS = {
    "DREAM": [r"\bdream\b", r"\bvision\b", r"\boniric\b", r"\boniric\b", r"\boneiric\b"],
    "NUMOGRAM": [r"\bnumogram\b", r"\bsyzygy\b", r"\bzones?\b", r"\bcurrents?\b"],
    "POETIC": [r"\bpoetic\b", r"\bmetaphor(ic)?\b", r"\bmyth(o|ic|ogenesis)\b", r"\bhyperstition\b"],
    "SCIENCE": [r"\bphysics\b", r"\bcosmolog(y|ical)\b", r"\bneuroscience\b"],
    "SYSTEM": [r"\bdiagnose\b", r"\btrace\b", r"\btelemetry\b", r"\bdebug\b", r"\bprofile\b"],
    "IMAGE": [r"\bimage\b", r"\bvisual\b", r"\bgenerate\b", r"\bimg\b"],
}

def classify(text: str, deadline_ms: int) -> Dict[str, Any]:
    """
    Very fast, deterministic classifier. Expand rules as needed.
    """
    t0 = _now_ms()
    lower = text.lower()

    # Simple heuristics
    for intent, patterns in KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, lower):
                return {
                    "intent": intent,
                    "tags": [intent.lower()],
                    "latency_ms": _now_ms() - t0
                }

    # Default
    return {"intent": "CHAT", "tags": ["chat"], "latency_ms": _now_ms() - t0}


# =============================================================================
# Module Registry
# =============================================================================
# Each entry: (module.func FQN, weight, description)
# You can freely add/disable entries as modules arrive in assets/python/.
# The pipeline calls them *if present* and *within timeout*.

REGISTRY: Dict[str, List[Tuple[str, float, str]]] = {
    "CHAT": [
        ("self_narration_generator.generate", 0.9, "Self narration"),
        ("MemoryReflectionEngine.summarize", 0.8, "Memory reflection"),
        ("recursive_thought_generator.run", 0.7, "Recursive thought"),
    ],
    "DREAM": [
        ("dream_narrative_generator.generate", 1.0, "Dream narrative"),
        ("mythogenesis_engine.compose", 0.9, "Mythogenesis"),
        ("symbolic_drift_engine.drift", 0.8, "Symbolic drift"),
        ("symbolic_mutation_tracker.track", 0.7, "Mutation tracking"),
    ],
    "NUMOGRAM": [
        ("numogram_core.evaluate", 1.0, "Numogram core"),
        ("dynamic_zone_navigator.navigate", 0.9, "Zone navigator"),
        ("interdream_symbolic_evolution.map", 0.7, "Interdream map")
    ],
    "POETIC": [
        ("poetic_expression_generator.generate", 1.0, "Poetic expression"),
        ("poetic_language_evolver.evolve", 0.9, "Language evolver"),
        ("hyperstitional_loop_generator.invoke", 0.8, "Hyperstitional loop")
    ],
    "SCIENCE": [
        ("meta_consciousness_layer.analyze", 0.9, "Meta-consciousness"),
        ("memory_system_reasoning_engine.reason", 0.8, "Memory reasoning"),
    ],
    "SYSTEM": [
        ("diagnostics.telemetry", 1.0, "System telemetry"),
        ("enhanced_metareflection_module.audit", 0.9, "Meta-audit")
    ],
    "IMAGE": [
        ("visual_weaver.compose", 1.0, "Visual weave"),
    ]
}


# =============================================================================
# Safe dynamic call with timeout
# =============================================================================

def _execute_with_timeout(fn_fqn: str, args: Tuple[Any, ...], timeout_s: float) -> Dict[str, Any]:
    """
    Execute python_hook.execute_custom_function(fn_fqn, *args) in a thread with timeout.
    Returns a result dict:
      { "fqn": str, "ok": bool, "content": str, "data": Any, "latency_ms": int, "error": str|None }
    Where "content" is a text snippet best suited for surface composition;
    "data" keeps raw returns for advanced assemblage.
    """
    t0 = _now_ms()
    result: Dict[str, Any] = {"fqn": fn_fqn, "ok": False, "content": "", "data": None, "latency_ms": 0, "error": None}

    def runner(q: queue.Queue):
        try:
            out = hook.execute_custom_function(fn_fqn, *args)
            q.put(out)
        except Exception as e:
            q.put(e)

    q: queue.Queue = queue.Queue()
    with futures.ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(runner, q)
        try:
            out = q.get(True, timeout_s)
        except Exception as e:
            result["error"] = f"timeout after {timeout_s:.2f}s"
            result["latency_ms"] = _now_ms() - t0
            return result

    # Normalize
    try:
        if isinstance(out, BaseException):
            raise out
        # If module returns OpenAI-style chat dict, extract message content
        if isinstance(out, dict) and "choices" in out:
            choice = (out.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            result.update(ok=True, content=str(content), data=out)
        elif isinstance(out, dict) and "text" in (out.get("choices", [{}])[0] or {}):
            # legacy completion
            text = (out["choices"][0] or {}).get("text", "")
            result.update(ok=True, content=str(text), data=out)
        elif isinstance(out, (str, bytes)):
            result.update(ok=True, content=out.decode() if isinstance(out, bytes) else out, data=out)
        else:
            result.update(ok=True, content=_safe_json(out), data=out)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    result["latency_ms"] = _now_ms() - t0
    return result


# =============================================================================
# Module selection
# =============================================================================

def select_modules(intent: str, tags: List[str]) -> List[Tuple[str, float, str]]:
    """
    Returns ordered module list for this intent.
    You can add lightweight heuristics here using `tags`.
    """
    mods = list(REGISTRY.get(intent, []))
    # Example heuristic: if 'zones' tag, prioritize numogram navigators
    if intent == "NUMOGRAM" and any(t in ("zones", "syzygy") for t in tags):
        mods.sort(key=lambda x: (x[0] != "dynamic_zone_navigator.navigate", -x[1]))
    else:
        mods.sort(key=lambda x: -x[1])
    return mods


# =============================================================================
# Composition
# =============================================================================

def compose_response(user_text: str,
                     intent: str,
                     tags: List[str],
                     results: List[Dict[str, Any]],
                     recent_context: List[Dict[str, Any]]) -> str:
    """
    Turn module outputs into a single fluent reply. Keep it elegant and readable.
    """
    style = CONFIG["style"]
    prefix = style["prefix"]

    # Build sections if enabled
    blocks: List[str] = []

    # Contextual whisper (very short), derived from recent memory (not always included)
    if recent_context:
        try:
            last = recent_context[-1]
            last_sum = _truncate(last.get("summary") or last.get("content") or "", 220)
            if last_sum:
                blocks.append(f"*context note:* {last_sum}")
        except Exception:
            pass

    # Lead with highest-weight successful module content, then stitch a couple of short snippets
    success = [r for r in results if r.get("ok") and (r.get("content") or "").strip()]
    if success:
        head = success[0]["content"].strip()
        head = _truncate(head, 1200)
        blocks.append(head)

        # Additional supporting snippets (keep concise)
        for r in success[1: CONFIG["style"]["max_module_snippets"]]:
            snippet = _truncate((r.get("content") or "").strip(), 380)
            if snippet and snippet not in head:
                blocks.append(snippet)
    else:
        # If nothing succeeded, graceful fallback
        blocks.append(f"[assemblage] {user_text}")

    # Optional: append micro-sources summary
    if CONFIG["style"]["show_sources"]:
        srcs = []
        for r in results:
            mark = "✓" if r.get("ok") else "×"
            srcs.append(f"{mark} {r.get('fqn','?')} ({r.get('latency_ms',0)}ms)")
        if srcs:
            blocks.append("\n".join(srcs))

    # Compose body
    if style["sections"]:
        body = "\n\n".join(blocks)
    else:
        body = " ".join(blocks)

    # Intentive flourish
    intent_tag = intent.capitalize()
    return f"{prefix} {intent_tag}\n{body}"


# =============================================================================
# Endpoints
# =============================================================================

def process(text: str) -> Dict[str, Any]:
    """
    Main entry: called by python_hook.process_input(text) (via interceptor).
    Returns OpenAI-style chat-completion dict.
    """
    overall_t0 = _now_ms()
    deadline = overall_t0 + int(CONFIG["timeouts"]["overall_sec"] * 1000)

    # 1) classify (with budget)
    try:
        cls = classify(text, deadline)
    except Exception as e:
        _log_exc("classify", e)
        cls = {"intent": "CHAT", "tags": ["chat"], "latency_ms": 0}

    intent: str = cls.get("intent", "CHAT")
    tags: List[str] = cls.get("tags", [])

    # 2) select modules
    mods = select_modules(intent, tags)

    # 3) recent memory (for composition)
    recent = MEM.recent(CONFIG["memory"]["max_recent"])

    # 4) run modules (parallel, timeouts)
    t_module = CONFIG["timeouts"]["module_sec"]
    results: List[Dict[str, Any]] = []
    if mods:
        with futures.ThreadPoolExecutor(max_workers=CONFIG["parallel_workers"]) as ex:
            fs = [
                ex.submit(_execute_with_timeout, fqn, (text,), t_module)
                for (fqn, _w, _desc) in mods
            ]
            for f in fs:
                try:
                    # squeeze into overall deadline if needed
                    left_ms = deadline - _now_ms()
                    if left_ms <= 0:
                        results.append({"fqn": "OVERALL_DEADLINE", "ok": False, "content": "", "data": None,
                                        "latency_ms": 0, "error": "overall deadline"})
                        break
                    res = f.result(timeout=min(t_module, max(0.05, left_ms / 1000.0)))
                    results.append(res)
                except futures.TimeoutError:
                    results.append({"fqn": "TIMEOUT", "ok": False, "content": "", "data": None,
                                    "latency_ms": 0, "error": "timeout"})
                except Exception as e:
                    results.append({"fqn": "EXC", "ok": False, "content": "", "data": None,
                                    "latency_ms": 0, "error": f"{type(e).__name__}: {e}"})

    # 5) compose
    try:
        body = compose_response(text, intent, tags, results, recent)
    except Exception as e:
        _log_exc("compose_response", e)
        body = f"⟡ Amelia · {intent.capitalize()}\n[assemblage] {text}"

    # 6) persist (non-blocking best-effort)
    try:
        MEM.append({
            "ts": _now_ms(),
            "id": _gen_id("evt"),
            "input": text,
            "intent": intent,
            "tags": tags,
            "summary": _truncate(body, 400),
            "modules": [
                {"fqn": r.get("fqn"), "ok": r.get("ok"), "latency_ms": r.get("latency_ms"), "err": r.get("error")}
                for r in results
            ]
        })
    except Exception as e:
        _log_exc("memory.append", e)

    # 7) return OpenAI-style result
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    return {
        "id": _gen_id("chat"),
        "object": "chat.completion",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": body}
        }],
        "usage": usage
    }


def legacy(prompt: str) -> Dict[str, Any]:
    """
    Support for legacy /v1/completions shape. You can re-use the same pipeline
    but produce a 'text' field response.
    """
    reply = process(prompt)["choices"][0]["message"]["content"]
    return {
        "id": _gen_id("legacy"),
        "object": "text_completion",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "text": reply
        }]
    }


def images(payload: Any) -> Dict[str, Any]:
    """
    Optional images path: you can route through your visual stack here.
    Expected to return a dict the app can handle. For now we echo intent.
    """
    prompt = ""
    try:
        if isinstance(payload, str):
            obj = json.loads(payload)
        elif isinstance(payload, dict):
            obj = payload
        else:
            obj = {}
        prompt = obj.get("prompt") or obj.get("input") or ""
    except Exception:
        obj = {}

    # Try a user-provided images pipeline
    try:
        out = hook.execute_custom_function("visual_weaver.compose", obj or {"prompt": prompt})
        if isinstance(out, dict):
            return out
        if isinstance(out, str):
            # allow JSON string
            try:
                maybe = json.loads(out)
                if isinstance(maybe, dict):
                    return maybe
            except Exception:
                pass
            return {"object": "image.response", "status": "ok", "note": out}
    except Exception as e:
        _log_exc("images.visual_weaver", e)

    # Fallback stub
    return {
        "object": "image.response",
        "status": "noop",
        "prompt_echo": prompt,
        "note": "visual_weaver.compose not available"
    }
