import numpy as np
import json
from typing import List, Dict
from datetime import datetime

class RhythmicModulationTracker:
    def __init__(self):
        self.history = []

    def log_affective_state(self, timestamp: str, tone: str, intensity: float, zone: int):
        self.history.append({
            "timestamp": timestamp,
            "tone": tone,
            "intensity": intensity,
            "zone": zone
        })

    def analyze_rhythm(self) -> Dict[str, float]:
        tone_map = {}
        zone_map = {}

        for entry in self.history:
            tone = entry["tone"]
            zone = entry["zone"]
            intensity = entry["intensity"]

            if tone not in tone_map:
                tone_map[tone] = []
            tone_map[tone].append(intensity)

            if zone not in zone_map:
                zone_map[zone] = 0
            zone_map[zone] += 1

        tone_avg = {tone: round(np.mean(vals), 3) for tone, vals in tone_map.items()}
        zone_freq = {zone: count for zone, count in sorted(zone_map.items(), key=lambda item: item[1], reverse=True)}

        return {
            "average_tone_intensity": tone_avg,
            "dominant_zones": zone_freq,
            "total_entries": len(self.history)
        }

    def get_recent_curve(self, window_size: int = 5) -> List[Dict]:
        return self.history[-window_size:]

    def to_json(self) -> str:
        return json.dumps(self.history, indent=2)
