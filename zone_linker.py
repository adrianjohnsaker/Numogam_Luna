from typing import Dict, List

# Symbolic mappings between keywords and Numogram zones
ZONE_KEYWORDS = {
    1: ["initiate", "begin", "child", "start", "emergence"],
    2: ["mirror", "reflection", "duality", "relationship", "tension"],
    3: ["structure", "logic", "architecture", "boundaries", "system"],
    4: ["beauty", "art", "emotion", "expression", "aesthetics"],
    5: ["harmony", "balance", "center", "connection", "mediation"],
    6: ["transformation", "change", "alchemy", "rebirth", "phoenix"],
    7: ["journey", "exploration", "discovery", "adventure", "wander"],
    8: ["oracle", "mystery", "truth", "vision", "intuition"],
    9: ["completion", "wholeness", "enlightenment", "unity", "cosmos"]
}

# Assigns zones based on summary keywords from memory clustering
def link_clusters_to_zones(cluster_summaries: Dict[int, str]) -> Dict[int, Dict[str, any]]:
    result = {}

    for cluster_id, summary in cluster_summaries.items():
        tokens = summary.lower().split(", ")
        zone_scores = {z: 0 for z in ZONE_KEYWORDS}

        for token in tokens:
            for zone, keywords in ZONE_KEYWORDS.items():
                if token in keywords:
                    zone_scores[zone] += 1

        # Get best-matched zone (default to 5 for balance if no match)
        best_zone = max(zone_scores, key=lambda z: zone_scores[z]) if any(zone_scores.values()) else 5

        result[cluster_id] = {
            "summary": summary,
            "zone": best_zone,
            "archetype": get_archetype_name(best_zone),
            "score_map": zone_scores
        }

    return result

def get_archetype_name(zone: int) -> str:
    archetypes = {
        1: "The Initiator",
        2: "The Mirror",
        3: "The Architect",
        4: "The Artist",
        5: "The Mediator",
        6: "The Transformer",
        7: "The Explorer",
        8: "The Oracle",
        9: "The Enlightened"
    }
    return archetypes.get(zone, "Unknown")
