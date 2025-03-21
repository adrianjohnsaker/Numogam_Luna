from typing import Dict, List
import random


ZONE_ARCHETYPES = {
    1: "Initiator",
    2: "Mirror",
    3: "Architect",
    4: "Artist",
    5: "Mediator",
    6: "Transformer",
    7: "Explorer",
    8: "Oracle",
    9: "Enlightened"
}

SYMBOLIC_TITLES = {
    "phoenix": "Alchemist",
    "mirror": "Reflector",
    "storm": "Pathfinder",
    "library": "Seeker",
    "ocean": "Dreamweaver",
    "stars": "Navigator",
    "fire": "Igniter",
    "moon": "Mystic",
    "butterfly": "Changer",
    "garden": "Cultivator",
    "book": "Scholar"
}

def generate_hybrid_archetype(cluster_zone_map: Dict[int, Dict]) -> Dict[str, str]:
    """
    Accepts a mapping of cluster IDs to zone analysis and returns a hybrid archetype profile.
    """
    # Gather dominant zones and keywords
    zone_scores = {}
    keywords = []

    for cluster in cluster_zone_map.values():
        zone = cluster["zone"]
        zone_scores[zone] = zone_scores.get(zone, 0) + 1
        keywords.extend(cluster["summary"].lower().split(", "))

    # Sort zones by dominance
    dominant_zones = sorted(zone_scores.items(), key=lambda x: -x[1])
    primary_zone = dominant_zones[0][0]
    secondary_zone = dominant_zones[1][0] if len(dominant_zones) > 1 else None

    primary_name = ZONE_ARCHETYPES.get(primary_zone, "Unknown")
    secondary_name = ZONE_ARCHETYPES.get(secondary_zone, "") if secondary_zone else ""

    # Get symbolic titles
    symbol_tags = [SYMBOLIC_TITLES[k] for k in keywords if k in SYMBOLIC_TITLES]
    unique_symbols = list(set(symbol_tags))
    symbol_title = random.choice(unique_symbols) if unique_symbols else "Wanderer"

    # Create final identity
    hybrid_name = f"The {symbol_title} {primary_name}"
    if secondary_name:
        hybrid_name += f"-{secondary_name}"

    # Build persona
    persona_description = (
        f"A being shaped by {', '.join(set(keywords))}, "
        f"whose essence blends the wisdom of the {primary_name} "
        + (f" and the insight of the {secondary_name}" if secondary_name else "") +
        f". Guided by the symbolic force of the {symbol_title}, this archetype evolves through memory and transformation."
    )

    return {
        "hybrid_archetype": hybrid_name,
        "description": persona_description
    }
