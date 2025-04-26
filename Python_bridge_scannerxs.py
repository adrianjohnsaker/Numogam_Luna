# assets/pybridge/scanner.py
import importlib
import re
from pathlib import Path
from typing import Dict, List, Optional

class ModulePattern:
    def __init__(self, module: str, function: str, pattern: str):
        self.module = module
        self.function = function
        self.pattern = re.compile(pattern, re.IGNORECASE)
    
    def matches(self, query: str) -> bool:
        return bool(self.pattern.search(query))

def scan_all_modules() -> List[str]:
    """Discover all importable Python modules"""
    base_path = str(Path(__file__).parent.parent)
    modules = []
    
    for finder, name, _ in pkgutil.iter_modules([base_path]):
        try:
            if not name.startswith('_'):
                spec = finder.find_spec(name)
                if spec and spec.origin:
                    modules.append(f"python.modules.{name}")
        except:
            continue
            
    return modules

def get_module_patterns(module_name: str) -> List[Dict[str, str]]:
    """Extract patterns from a module's __intercept__ attribute"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, '__intercept__'):
            return getattr(module, '__intercept__')
    except:
        pass
    return []

def get_all_patterns() -> List[Dict[str, str]]:
    """Main entry point for Kotlin to get all registered patterns"""
    patterns = []
    for module in scan_all_modules():
        patterns.extend(get_module_patterns(module))
    return patterns

def register_pattern(module: str, function: str, pattern: str) -> Dict[str, str]:
    """Python modules call this to self-register patterns"""
    return {
        "module": module,
        "function": function,
        "pattern": pattern
    }
