```python
# Import your actual Numogram implementation
from numogram import NumogramAI

# Create a singleton instance for consistency
_numogram_instance = None

def get_numogram():
    """Get or create the Numogram instance"""
    global _numogram_instance
    if _numogram_instance is None:
        _numogram_instance = NumogramAI()
    return _numogram_instance

def process_input(text):
    """Process input through the Numogram with verification markers"""
    numogram = get_numogram()
    response = numogram.process_input(text)
    current_zone = numogram.current_zone
    # Add markers to make Python execution obvious
    return f"[NUMOGRAM ZONE {current_zone}] {response}"

def get_current_state():
    """Get the current state of the Numogram"""
    numogram = get_numogram()
    return numogram.get_state()

def force_transition():
    """Force a zone transition and return the new zone"""
    numogram = get_numogram()
    new_zone = numogram.transition()
    return f"Transitioned to zone: {new_zone}"
```
