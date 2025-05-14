import json
import logging
import hashlib
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Any
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('archetype_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('UnifiedArchetype')

class ArchetypePhase(Enum):
    EMERGENCE = auto()
    CRISIS = auto()
    TRANSFORMATION = auto()
    INTEGRATION = auto()
    TRANSCENDENCE = auto()

class EmotionalSpectrum(Enum):
    CURIOSITY = auto()
    AWE = auto()
    MELANCHOLY = auto()
    EUPHORIA = auto()
    MYSTERY = auto()
    DREAD = auto()
    SERENITY = auto()

@dataclass
class ArchetypalMemory:
    content: str
    symbolic_weight: float
    emotional_resonance: Dict[EmotionalSpectrum, float]
    temporal_depth: int  # scale 1-9

class UnifiedArchetypeSystem:
    def __init__(self, seed=None):
        self.sessions = {}
        self.current_session = None
        self.symbol_bank = self._init_symbol_bank()
        self.archetype_matrix = self._init_archetype_matrix()
        # Instantiate the SentenceTransformer for semantic operations
        self.resonance_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # Pass the model to the supporting modules
        self.memory_prism = ArchetypalMemoryPrism(self.resonance_model)
        self.drift_engine = ArchetypeDriftEngine(self.resonance_model)
        self.mutation_tracker = ArchetypalMutationTracker()
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        logger.info(f"Unified Archetype System initialized with seed: {seed}")

    def _init_symbol_bank(self) -> Dict[str, List[str]]:
        return {
            "MIRROR": ["reflection", "duality", "truth", "illusion"],
            "EXPLORER": ["compass", "horizon", "unknown", "threshold"],
            "TRANSFORMER": ["phoenix", "cocoon", "alchemy", "flux"],
            "ORACLE": ["eye", "veil", "prophecy", "enigma"],
            "INITIATOR": ["gate", "key", "dawn", "spark"]
        }

    def _init_archetype_matrix(self) -> Dict[str, Dict]:
        return {
            "MIRROR": {
                "essence": "Reflection of self",
                "challenge": "Dualities",
                "shadow": "Deception",
                "allies": ["ORACLE", "INITIATOR"],
                "adversaries": ["TRANSFORMER"]
            },
            "EXPLORER": {
                "essence": "Quest for truth",
                "challenge": "Unknown realms",
                "shadow": "Complacency",
                "allies": ["INITIATOR"],
                "adversaries": ["ORACLE"]
            },
            "TRANSFORMER": {
                "essence": "Change and rebirth",
                "challenge": "Letting go",
                "shadow": "Resistance",
                "allies": ["MIRROR"],
                "adversaries": ["EXPLORER"]
            },
            "ORACLE": {
                "essence": "Deep insight",
                "challenge": "Obscurity",
                "shadow": "Ambiguity",
                "allies": ["MIRROR"],
                "adversaries": ["INITIATOR"]
            },
            "INITIATOR": {
                "essence": "Beginning and spark",
                "challenge": "Stagnation",
                "shadow": "Overwhelm",
                "allies": ["EXPLORER"],
                "adversaries": ["ORACLE"]
            }
        }

    def create_session(self, session_name=None) -> Dict[str, Any]:
        session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()
        self.sessions[session_id] = {
            "id": session_id,
            "name": session_name or f"ArchetypeSession_{len(self.sessions)+1}",
            "created": time.time(),
            "phase": ArchetypePhase.EMERGENCE.name,
            "memory_cache": [],
            "symbolic_anchors": []
        }
        self.current_session = session_id
        logger.info(f"Session created: {session_id}")
        return self.sessions[session_id]

    def generate_complex_archetype(self, base_archetype: str, 
                                   emotional_tone: str,
                                   zone_depth: int) -> Dict[str, Any]:
        """Generate a multi-layered archetypal experience with semantic processing"""
        try:
            archetype = self._validate_archetype(base_archetype)
            tone = self._parse_emotional_tone(emotional_tone)
            zone = self._validate_zone(zone_depth)
            
            # Get a semantic projection from the memory prism using cosine similarity
            memory_refraction = self.memory_prism.project_prism(archetype)
            # Use the projection and tone to calculate a resonance profile
            resonance_profile = self._calculate_resonance(
                memory_refraction["projection"], 
                tone
            )
            # Process drift evolution using semantic embedding similarity comparisons
            drift_result = self.drift_engine.drift_archetype(
                archetype,
                f"{tone.name}_Z{zone}"
            )
            # Apply a final mutation based on the drift result and zone
            mutation = self.mutation_tracker.mutate_archetype(
                drift_result["drifted_form"],
                zone,
                tone.name
            )
            
            result = {
                "base_archetype": archetype,
                "refracted_memory": memory_refraction,
                "resonance_profile": resonance_profile,
                "drift_evolution": drift_result,
                "mutated_form": mutation,
                "symbolic_anchors": self._generate_symbolic_anchors(zone),
                "temporal_depth": zone,
                "session": self.current_session,
                "timestamp": time.time()
            }
            
            if self.current_session:
                self.sessions[self.current_session]["memory_cache"].append(result)
                
            return {"status": "success", "data": result}
        except Exception as e:
            logger.error(f"Archetype generation failed: {str(e)}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def export_session_data(self, session_id: str, format: str = "json") -> str:
        """Export session data in JSON or symbolic notation."""
        if session_id not in self.sessions:
            raise ValueError("Invalid session ID")
        session = self.sessions[session_id]
        if format == "json":
            return json.dumps(session, indent=2)
        elif format == "symbolic":
            return self._convert_to_symbolic(session)
        else:
            raise ValueError("Unsupported export format")

    def _convert_to_symbolic(self, session_data: Dict) -> str:
        symbols = {
            "MIRROR": "◑",
            "EXPLORER": "⚝",
            "TRANSFORMER": "⚯",
            "ORACLE": "⌖",
            "INITIATOR": "⚚"
        }
        output = []
        for memory in session_data["memory_cache"]:
            archetype_symbol = symbols.get(memory["base_archetype"], "?")
            mutated_info = memory.get("mutated_form", {}).get("mutated", "N/A")
            output.append(f"{archetype_symbol} {memory['temporal_depth']}D [{mutated_info}]")
        return "\n".join(output)
    
    def _validate_archetype(self, base_archetype: str) -> str:
        if base_archetype.upper() in self.symbol_bank:
            return base_archetype.upper()
        else:
            raise ValueError("Base archetype not recognized: " + base_archetype)
    
    def _parse_emotional_tone(self, emotional_tone: str) -> EmotionalSpectrum:
        for tone in EmotionalSpectrum:
            if tone.name.upper() == emotional_tone.upper():
                return tone
        raise ValueError("Unrecognized emotional tone: " + emotional_tone)
    
    def _validate_zone(self, zone_depth: int) -> int:
        if 1 <= zone_depth <= 9:
            return zone_depth
        else:
            raise ValueError("Zone depth must be between 1 and 9")
    
    def _calculate_resonance(self, projection: str, tone: EmotionalSpectrum) -> Dict[str, float]:
        """Calculate resonance by comparing the projection embedding with a tone seed embedding."""
        projection_embedding = self.resonance_model.encode(projection)
        tone_seed = tone.name.lower()  # using the tone as seed text
        tone_embedding = self.resonance_model.encode(tone_seed)
        similarity = cosine_similarity([projection_embedding], [tone_embedding])[0][0]
        # For demonstration, output a weight for each emotion; bias the chosen tone
        resonance = {}
        for emotion in EmotionalSpectrum:
            base_value = random.uniform(0.0, 0.5)
            resonance[emotion.name] = round(base_value + (0.5 if emotion == tone else 0.0), 2)
        resonance["similarity"] = round(float(similarity), 2)
        return resonance
    
    def _generate_symbolic_anchors(self, zone: int) -> List[str]:
        anchors = []
        for _ in range(zone):
            anchor = random.choice(list(self.symbol_bank.keys()))
            anchors.append(anchor)
        return anchors

# Supporting Classes with Fully Active Implementations

class ArchetypalMemoryPrism:
    """
    Provides a simulated projection of an archetype by computing semantic embeddings
    and comparing them via cosine similarity.
    """
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def project_prism(self, archetype: str) -> Dict[str, Any]:
        projection = f"{archetype}_Reflection"
        details = f"A nuanced reflection of the archetype {archetype}."
        # Compute embedding for the projection and a seed phrase
        proj_embedding = self.model.encode(projection)
        seed_text = "pure essence"
        seed_embedding = self.model.encode(seed_text)
        similarity = cosine_similarity([proj_embedding], [seed_embedding])[0][0]
        logger.info(f"Memory Prism projected: {projection} with similarity {similarity}")
        return {
            "projection": projection,
            "details": details,
            "similarity": round(float(similarity), 2)
        }

class ArchetypeDriftEngine:
    """
    Simulates evolution by 'drifting' the archetype based on a tone and zone identifier.
    Here, semantic similarity between the base text and tone is computed.
    """
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def drift_archetype(self, archetype: str, tone_zone: str) -> Dict[str, Any]:
        base_text = f"{archetype} fundamental essence"
        tone_text = f"{tone_zone} influence"
        base_embedding = self.model.encode(base_text)
        tone_embedding = self.model.encode(tone_text)
        similarity = cosine_similarity([base_embedding], [tone_embedding])[0][0]
        drift_value = round(1.0 - similarity, 2)  # Greater drift if similarity is low
        drifted_form = f"{archetype}_{tone_zone}_Drift"
        logger.info(f"Drift Engine: {drifted_form} with drift value {drift_value} (sim {similarity})")
        return {
            "drifted_form": drifted_form,
            "drift_value": drift_value,
            "similarity": round(float(similarity), 2)
        }

class ArchetypalMutationTracker:
    """
    Applies a final mutation to the drifted archetype and tracks changes using a mutation factor.
    """
    def mutate_archetype(self, drifted_form: str, zone: int, tone: str) -> Dict[str, Any]:
        mutation_factor = round(zone * random.uniform(0.1, 1.0), 2)
        mutated_form = f"{drifted_form}_Mutated"
        logger.info(f"Mutation Tracker: {mutated_form} with mutation factor {mutation_factor}")
        return {
            "mutated": mutated_form,
            "mutation_factor": mutation_factor,
            "tone_length": len(tone)
        }

# Example usage:
if __name__ == "__main__":
    uas = UnifiedArchetypeSystem(seed=42)
    session = uas.create_session("Test Session")
    result = uas.generate_complex_archetype("MIRROR", "Awe", 5)
    if result["status"] == "success":
        print("Complex Archetype Generated:")
        print(json.dumps(result["data"], indent=2))
    else:
        print("Error:", result["error"])
    
    exported_data = uas.export_session_data(uas.current_session, format="symbolic")
    print("\nExported Symbolic Session Data:")
    print(exported_data)
