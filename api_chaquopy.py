# api.py (for Chaquopy)
import os
import sys
import importlib
import logging

# Ensure assets/python is on sys.path
ASSET_DIR = os.path.dirname(__file__)
if ASSET_DIR not in sys.path:
    sys.path.insert(0, ASSET_DIR)

# Dynamically import every .py except this one
_MODULES = {}
for fname in os.listdir(ASSET_DIR):
    if fname.endswith(".py") and fname != "api.py":
        name = fname[:-3]
        try:
            mod = importlib.import_module(name)
            _MODULES[name] = mod
        except Exception as e:
            logging.error(f"Import failed for {name}: {e}")

def list_modules() -> list:
    return list(_MODULES.keys())

def list_functions(module: str) -> list:
    mod = _MODULES.get(module)
    if not mod: return []
    return [f for f in dir(mod)
            if callable(getattr(mod, f)) and not f.startswith("_")]

def call(module: str, func: str, *args):
    mod = _MODULES.get(module)
    if not mod:
        raise ImportError(f"No module named {module}")
    fn = getattr(mod, func, None)
    if not callable(fn):
        raise AttributeError(f"{func} not in {module}")
    return fn(*args)
