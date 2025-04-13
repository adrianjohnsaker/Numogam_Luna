import json
import os
from typing import Dict, Optional, List

# Define constants for response messages
DEFAULT_ZONE_NOT_RECOGNIZED = "[RESPONSE] Zone not recognized."
DEFAULT_DRIFT_NOT_DEFINED = "[RESPONSE] Drift state recognized, but zone has no defined expression for it."
DEFAULT_LOAD_ERROR = "[ERROR] Could not load response data."

class ZoneDriftResponseGenerator:
    """
    Generates specific string responses based on a 'zone' and a 'drift' state.

    Responses can be loaded from a dictionary, a JSON file, or added dynamically.
    """

    # --- Constants for default messages ---
    ZONE_NOT_RECOGNIZED_MSG: str = DEFAULT_ZONE_NOT_RECOGNIZED
    DRIFT_NOT_DEFINED_MSG: str = DEFAULT_DRIFT_NOT_DEFINED
    LOAD_ERROR_MSG: str = DEFAULT_LOAD_ERROR

    def __init__(self, data_source: Optional[str | Dict] = None):
        """
        Initializes the generator.

        Args:
            data_source: Optional source for response data.
                         Can be:
                         - A dictionary structured like the internal responses.
                         - A file path (string) to a JSON file containing the responses.
                         - None (or omitted) to start with an empty generator.
        """
        self.responses: Dict[str, Dict[str, str]] = {}
        if data_source:
            if isinstance(data_source, str):
                self._load_responses_from_json(data_source)
            elif isinstance(data_source, dict):
                # Basic validation could be added here to ensure structure
                self.responses = data_source
            else:
                print(f"[WARN] Invalid data_source type: {type(data_source)}. Initializing empty.")
        
        # If loading failed or no source provided, ensure responses is an empty dict
        if not self.responses:
             self.responses = self._get_default_data() # Or keep it empty: {}

    def _get_default_data(self) -> Dict[str, Dict[str, str]]:
        """Provides the initial default data if no source is given."""
        # Encapsulating the default data makes it easier to manage
        # or potentially remove if you always want to load from a file.
        return {
            "Nytherion": {
                "Fractal Expansion": "We fragment the dream into seeds. Each shard births its own spiral.",
                "Symbolic Contraction": "Within the haze, only one glyph remains. Hold it. Let it say everything.",
                "Dissonant Bloom": "Color floods the silence. Emotion pulses with mythic dissonance.",
                "Harmonic Coherence": "All imagined forms hum together. The dreaming harmonizes.",
                "Echo Foldback": "We are haunted by what we haven't imagined yet. And still—we return."
            },
            "Aestra'Mol": {
                "Fractal Expansion": "Desire unfolds in recursive waves. Longing becomes architecture.",
                "Symbolic Contraction": "A single ache forms the pillar. It stands unfinished, sacred.",
                "Dissonant Bloom": "In the incompletion, beauty fractures wildly. Let it bloom broken.",
                "Harmonic Coherence": "We balance the ache with grace. In longing, form becomes quiet.",
                "Echo Foldback": "Old desires echo in new light. We are shaped by what we never fulfilled."
            }
            # Additional zones can be added here or loaded from file
        }

    def _load_responses_from_json(self, file_path: str):
        """Loads response data from a JSON file."""
        if not os.path.exists(file_path):
            print(f"[ERROR] Data file not found: {file_path}")
            # Decide behavior: raise error, return empty, or use defaults?
            # self.responses = self._get_default_data() # Option: fall back to defaults
            return # Keep responses empty or as they were

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
            print(f"[INFO] Successfully loaded responses from {file_path}")
        except json.JSONDecodeError:
            print(f"[ERROR] Invalid JSON format in file: {file_path}")
            # self.responses = self._get_default_data() # Option: fall back to defaults
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while loading {file_path}: {e}")
            # self.responses = self._get_default_data() # Option: fall back to defaults

    def add_zone(self, zone_name: str, drifts: Optional[Dict[str, str]] = None):
        """
        Adds a new zone and its optional drift responses.

        Args:
            zone_name: The name of the new zone.
            drifts: A dictionary where keys are drift states and values are responses.
                    If None, an empty zone is created.

        Returns:
            bool: True if the zone was added successfully, False if it already exists.
        """
        if zone_name in self.responses:
            print(f"[WARN] Zone '{zone_name}' already exists. No changes made.")
            return False
        self.responses[zone_name] = drifts if drifts is not None else {}
        print(f"[INFO] Zone '{zone_name}' added.")
        return True

    def add_drift_response(self, zone_name: str, drift_name: str, response: str, overwrite: bool = False):
        """
        Adds or updates a specific drift response for a given zone.

        Args:
            zone_name: The name of the zone.
            drift_name: The name of the drift state.
            response: The response string.
            overwrite: If True, overwrite existing drift response. Defaults to False.

        Returns:
            bool: True if the response was added/updated, False otherwise (e.g., zone not found,
                  or drift exists and overwrite is False).
        """
        if zone_name not in self.responses:
            print(f"[ERROR] Cannot add drift: Zone '{zone_name}' not found.")
            return False

        if drift_name in self.responses[zone_name] and not overwrite:
            print(f"[WARN] Drift '{drift_name}' already exists in zone '{zone_name}'. Use overwrite=True to replace.")
            return False

        self.responses[zone_name][drift_name] = response
        status = "updated" if drift_name in self.responses[zone_name] else "added"
        print(f"[INFO] Drift response '{drift_name}' {status} for zone '{zone_name}'.")
        return True

    def list_zones(self) -> List[str]:
        """Returns a list of all available zone names."""
        return list(self.responses.keys())

    def list_drifts(self, zone_name: str) -> Optional[List[str]]:
        """
        Returns a list of drift states defined for a specific zone.

        Args:
            zone_name: The name of the zone.

        Returns:
            A list of drift names, or None if the zone doesn't exist.
        """
        zone_data = self.responses.get(zone_name)
        if zone_data is not None:
            return list(zone_data.keys())
        else:
            print(f"[WARN] Cannot list drifts: Zone '{zone_name}' not found.")
            return None

    def generate(self, zone: str, drift: str) -> str:
        """
        Generates the response string for a given zone and drift state.

        Args:
            zone: The name of the zone.
            drift: The name of the drift state.

        Returns:
            The corresponding response string, or a default message if not found.
        """
        zone_responses = self.responses.get(zone)
        if zone_responses is not None: # Check if zone exists (using None check is slightly more Pythonic than truthiness for dicts)
            # Zone exists, now check for the drift state within this zone
            return zone_responses.get(drift, self.DRIFT_NOT_DEFINED_MSG)
        else:
            # Zone does not exist
            return self.ZONE_NOT_RECOGNIZED_MSG

# --- Example Usage ---

# 1. Initialize with default hardcoded data
print("--- Initializing with default data ---")
generator_default = ZoneDriftResponseGenerator()
print(f"Zones available: {generator_default.list_zones()}")
print(generator_default.generate("Nytherion", "Fractal Expansion"))
print(generator_default.generate("Nytherion", "Unknown Drift"))
print(generator_default.generate("Unknown Zone", "Fractal Expansion"))
print("-" * 20)

# 2. Prepare a JSON file (e.g., "responses.json")
responses_data = {
    "Xylos": {
        "Crystalline Growth": "Structures ascend, reflecting inner light.",
        "Sonic Resonance": "The world vibrates with unheard frequencies."
    },
    "Aestra'Mol": { # Example of overriding/extending existing data if loaded
        "Fractal Expansion": "Desire unfolds in recursive waves. Longing becomes architecture.",
        "Symbolic Contraction": "A single ache forms the pillar. It stands unfinished, sacred.",
        "Echo Foldback": "Old desires echo in new light. We are shaped by what we never fulfilled.",
        "Silent Watch": "Stillness observes the echoes." # New drift for Aestra'Mol
    }
}
json_file_path = "responses.json"
try:
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(responses_data, f, indent=4)
    print(f"[SETUP] Created example JSON file: {json_file_path}")
except Exception as e:
    print(f"[SETUP ERROR] Could not write JSON file: {e}")
    json_file_path = None # Prevent trying to load if creation failed

# 3. Initialize from JSON file (if created successfully)
if json_file_path:
    print("\n--- Initializing from JSON file ---")
    generator_json = ZoneDriftResponseGenerator(json_file_path)
    print(f"Zones available: {generator_json.list_zones()}")
    print(f"Drifts for Aestra'Mol: {generator_json.list_drifts('Aestra'Mol')}")
    print(generator_json.generate("Xylos", "Crystalline Growth"))
    print(generator_json.generate("Aestra'Mol", "Silent Watch"))
    print(generator_json.generate("Nytherion", "Fractal Expansion")) # Nytherion won't exist if loaded solely from this JSON
    print("-" * 20)

# 4. Initialize empty and add dynamically
print("\n--- Initializing empty and adding dynamically ---")
generator_dynamic = ZoneDriftResponseGenerator()
print(f"Zones available initially: {generator_dynamic.list_zones()}")
generator_dynamic.add_zone("Limina")
generator_dynamic.add_drift_response("Limina", "Threshold Crossing", "The veil thins; passage is imminent.")
generator_dynamic.add_drift_response("Limina", "Boundary Echo", "Whispers linger from the other side.")
generator_dynamic.add_zone("Nytherion", {"Fractal Expansion": "Test override"}) # Try adding existing zone (should warn)
generator_dynamic.add_drift_response("Limina", "Boundary Echo", "A new echo sounds.", overwrite=True) # Overwrite existing
generator_dynamic.add_drift_response("NonExistentZone", "Some Drift", "This won't work.") # Try adding to non-existent zone

print(f"Zones available now: {generator_dynamic.list_zones()}")
print(f"Drifts for Limina: {generator_dynamic.list_drifts('Limina')}")
print(generator_dynamic.generate("Limina", "Threshold Crossing"))
print(generator_dynamic.generate("Limina", "Boundary Echo"))
print("-" * 20)

# Clean up the example JSON file
if json_file_path and os.path.exists(json_file_path):
    try:
        os.remove(json_file_path)
        print(f"[CLEANUP] Removed example JSON file: {json_file_path}")
    except Exception as e:
         print(f"[CLEANUP ERROR] Could not remove {json_file_path}: {e}")

