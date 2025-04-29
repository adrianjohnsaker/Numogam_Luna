
```python
# test_module.py
# Basic test module to verify Python integration is working

import os
import sys
import json
import time
import traceback

class TestModule:
    def __init__(self):
        self.name = "TestModule"
        self.version = "1.0.0"
        self.initialized_at = time.time()
    
    def get_info(self):
        """Return basic module information"""
        return {
            "name": self.name,
            "version": self.version,
            "python_version": sys.version,
            "platform": sys.platform,
            "initialized_at": self.initialized_at,
            "uptime": time.time() - self.initialized_at
        }
    
    def list_available_modules(self):
        """List Python modules that are available"""
        return sorted(sys.modules.keys())
    
    def get_environment(self):
        """Return environment information"""
        return {k: v for k, v in os.environ.items()}
    
    def test_json(self, data):
        """Test JSON serialization/deserialization"""
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
                return {"input_type": "string", "parsed": parsed, "success": True}
            else:
                serialized = json.dumps(data)
                return {"input_type": str(type(data)), "serialized": serialized, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def add_numbers(self, a, b):
        """Simple test function to add two numbers"""
        return a + b
    
    def get_file_list(self, directory=None):
        """List files in a directory (or current directory if None)"""
        try:
            if directory is None:
                directory = os.getcwd()
            files = os.listdir(directory)
            return {"directory": directory, "files": files, "count": len(files)}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

# Module-level functions
def test_function(message):
    """Simple test function at module level"""
    return f"Python received: {message} (Python {sys.version})"

def get_system_info():
    """Return basic system information"""
    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "path": sys.path,
        "executable": sys.executable,
        "modules": len(sys.modules)
    }

# Initialize module when imported
module = TestModule()
print(f"Test module initialized: {module.get_info
