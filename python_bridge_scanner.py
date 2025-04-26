# assets/pybridge/scanner.py
import importlib
import pkgutil
from pathlib import Path

def scan_all_modules():
    """Auto-discovers all Python modules in assets"""
    modules = []
    base_path = str(Path(__file__).parent.parent)
    
    for finder, name, _ in pkgutil.iter_modules([base_path]):
        try:
            if not name.startswith('_'):
                spec = finder.find_spec(name)
                if spec and spec.origin:
                    modules.append(f"python.modules.{name}")
        except:
            continue
            
    return modules

def get_callables(module_name):
    """Lists all callable functions in a module"""
    module = importlib.import_module(module_name)
    return [name for name in dir(module) 
            if callable(getattr(module, name)) 
            and not name.startswith('_')]
