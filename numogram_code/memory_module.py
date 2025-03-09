#!/usr/bin/env python3
"""
Enhanced Module Loader for Android Applications
Efficiently loads Python modules from specified asset directories.
"""

import os
import sys
import importlib
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ModuleLoader")

class ModuleLoader:
    """Module loader that handles dynamic loading from asset directories"""

    def __init__(self, base_path: str):
        """
        Initialize the module loader
        
        Args:
            base_path: Base directory path for modules
        """
        self.base_path = base_path
        self.loaded_modules: Dict[str, object] = {}
        
        # Validate path exists
        if not os.path.exists(self.base_path):
            logger.error(f"Error: Module path {self.base_path} does not exist.")
            return
            
        # Add to Python path if not already there
        if self.base_path not in sys.path:
            sys.path.append(self.base_path)
            logger.debug(f"Added {self.base_path} to system path")

    def load_modules(self, exclude_files: List[str] = None) -> Dict[str, object]:
        """
        Recursively load Python modules from the assets folder
        
        Args:
            exclude_files: List of filenames to exclude from loading
            
        Returns:
            Dictionary of loaded module objects by module name
        """
        if exclude_files is None:
            exclude_files = ["module_loader.py", "__init__.py", "__pycache__"]
        
        loaded_count = 0
        failed_count = 0
        
        for root, _, files in os.walk(self.base_path):
            for filename in files:
                if not filename.endswith(".py") or filename in exclude_files:
                    continue
                    
                # Convert file path to module path
                module_path = os.path.relpath(os.path.join(root, filename), self.base_path)
                module_name = module_path.replace(os.sep, ".")[:-3]  # Remove .py extension
                
                # Skip if already loaded
                if module_name in self.loaded_modules:
                    continue
                    
                try:
                    module = importlib.import_module(module_name)
                    self.loaded_modules[module_name] = module
                    loaded_count += 1
                    logger.info(f"✅ Loaded: {module_name}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ Failed: {module_name} - {str(e)}")
        
        logger.info(f"🔹 Module Loading Complete: {loaded_count} loaded, {failed_count} failed")
        return self.loaded_modules
    
    def get_module(self, module_name: str) -> object:
        """
        Get a specific loaded module by name
        
        Args:
            module_name: Name of the module to retrieve
            
        Returns:
            Module object if found, None otherwise
        """
        return self.loaded_modules.get(module_name)
    
    def reload_module(self, module_name: str) -> bool:
        """
        Reload a specific module
        
        Args:
            module_name: Name of the module to reload
            
        Returns:
            True if successful, False otherwise
        """
        if module_name not in self.loaded_modules:
            logger.warning(f"Cannot reload {module_name}: Not previously loaded")
            return False
            
        try:
            self.loaded_modules[module_name] = importlib.reload(self.loaded_modules[module_name])
            logger.info(f"♻️ Reloaded: {module_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reload: {module_name} - {str(e)}")
            return False

    def reload_all(self) -> int:
        """
        Reload all previously loaded modules
        
        Returns:
            Count of successfully reloaded modules
        """
        success_count = 0
        for module_name in list(self.loaded_modules.keys()):
            if self.reload_module(module_name):
                success_count += 1
        return success_count


# Implementation example
if __name__ == "__main__":
    ASSETS_PATH = "/storage/emulated/0/Android/data/com.yourapp.luna/assets/"
    loader = ModuleLoader(ASSETS_PATH)
    modules = loader.load_modules()
    
    # Print loaded module info
    print(f"\nSuccessfully loaded {len(modules)} modules:")
    for name in sorted(modules.keys()):
        print(f"  - {name}")
