from typing import List, Dict, Any
from collections import Counter
from memory_module import memory_recall

def track_archetype_evolution(user_id: str = "default") -> Dict[str, Any]:
    history = memory_recall(user_id, key="archetype_history") or []
    if not history:
        return {"status": "empty", "message": "No archetype history found."}

    archetype_names = [entry["hybrid_archetype"] for entry in history]
    symbolic_titles = [name.split(" ")[1] for name in archetype_names if len(name.split(" ")) > 1]
    primary_zones = extract_zones(history)

    # Frequency analysis
    title_freq = Counter(symbolic_titles)
    zone_freq = Counter(primary_zones)

    # Determine current phase
    current_archetype = history[-1]

    return {
        "status": "success",
        "total_evolutions": len(history),
        "most_common_title": title_freq.most_common(1)[0][0],
        "most_common_zone": zone_freq.most_common(1)[0][0],
        "zone_distribution": dict(zone_freq),
        "archetype_names": archetype_names,
        "current_archetype": current_archetype
    }

def extract_zones(history: List[Dict]) -> List[int]:
    zones = []
    for entry in history:
        zones.extend(entry.get("description", "").split("Zone "))
    return [int(z[0]) for z in zones if z and z[0].isdigit()]
