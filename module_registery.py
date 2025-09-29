# module_registry.py
import os
import importlib
import traceback

MODULES = {}

SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__)),
    os.path.join(os.path.dirname(__file__), "modules"),
    os.path.join(os.path.dirname(__file__), "autonomous"),
]

# Default tag weights
DEFAULT_TAG_WEIGHTS = {
    "syzygy": 5,
    "zone": 4,
    "currents": 4,
    "triangular": 3,
    "dream": 5,
    "glyph": 4,
    "myth": 4,
    "symbolic": 3,
    "poetic": 3,
    "memory": 3,
    "narrative": 2,
    "math": 1,
}

def discover_modules():
    """Scan SEARCH_PATHS for Python modules and register them."""
    for base_path in SEARCH_PATHS:
        if not os.path.isdir(base_path):
            continue

        for fname in os.listdir(base_path):
            if fname.endswith(".py") and not fname.startswith("_"):
                mod_name = fname[:-3]
                try:
                    module = importlib.import_module(mod_name)
                    MODULES[mod_name] = {"module": module, "functions": {}, "default": None}

                    meta = getattr(module, "__MODULE_META__", None)
                    if meta:
                        for fn_name, fn_meta in meta.get("functions", {}).items():
                            MODULES[mod_name]["functions"][fn_name] = {
                                "tags": fn_meta.get("tags", []),
                                "weights": fn_meta.get("weights", {})
                            }
                        # Mark default function if provided
                        MODULES[mod_name]["default"] = meta.get("default")
                    else:
                        for fn_name in dir(module):
                            if callable(getattr(module, fn_name)) and not fn_name.startswith("_"):
                                MODULES[mod_name]["functions"][fn_name] = {"tags": [], "weights": {}}
                        # Pick first function as default
                        if MODULES[mod_name]["functions"]:
                            MODULES[mod_name]["default"] = next(iter(MODULES[mod_name]["functions"]))

                except Exception as e:
                    print(f"[ERROR] Failed to import {mod_name}: {e}")
                    traceback.print_exc()

    return MODULES

def tag_score(text: str, tags: list[str], weights: dict) -> int:
    """Compute weighted score for a set of tags in given text."""
    lower_text = text.lower()
    score = 0
    for tag in tags:
        if tag in lower_text:
            score += weights.get(tag, DEFAULT_TAG_WEIGHTS.get(tag, 1))
    return score

def get_module_for_query(text: str):
    """
    Find best function match by weighted hierarchical tag relevance.
    Includes fallback chaining:
      1. Best matching function
      2. Module default
      3. Global fallback
    """
    best_score = -1
    best_fn = None
    best_mod = None

    for mod, meta in MODULES.items():
        for fn_name, fn_meta in meta["functions"].items():
            score = tag_score(text, fn_meta["tags"], fn_meta.get("weights", {}))
            if score > best_score:
                best_score = score
                best_fn = fn_name
                best_mod = meta["module"]

    # --- Case 1: Found a specific match
    if best_mod and best_fn and best_score > 0:
        return best_mod, best_fn, "specific"

    # --- Case 2: Fall back to default function of a module
    for mod, meta in MODULES.items():
        default_fn = meta.get("default")
        if default_fn:
            return meta["module"], default_fn, "module-default"

    # --- Case 3: Global fallback (none matched)
    return None, None, "global-fallback"
