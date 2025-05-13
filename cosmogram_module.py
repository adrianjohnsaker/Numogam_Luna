## 1. Unified Python Module (cosmogram_module.py)

```python
"""
Cosmogram Synthesis Module
Integrated system for mythological narrative generation and symbolic interrelation
"""

import datetime
import random
import uuid
import json
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

# =============== CONSTANTS ===============

EMOTIONAL_TONES = [
    "transcendent", "melancholic", "euphoric", 
    "enigmatic", "resilient", "introspective"
]

SYMBOL_CATEGORIES = [
    "natural_element", "cosmic_phenomenon", "psychological_archetype",
    "mythological_entity", "abstract_concept"
]

SYMBOL_LIBRARY = {
    "natural_element": [
        "river", "mountain", "storm", "seed", "ocean", "forest", "volcano"
    ],
    "cosmic_phenomenon": [
        "supernova", "black hole", "eclipse", "nebula", "quasar", "event horizon"
    ],
    "psychological_archetype": [
        "shadow", "anima", "persona", "self", "collective unconscious", "ego"
    ],
    "mythological_entity": [
        "phoenix", "ouroboros", "titan", "oracle", "chimera", "golem"
    ],
    "abstract_concept": [
        "transformation", "recursion", "emergence", "entropy", "synchronicity", "liminality"
    ]
}

ZONE_MOTIFS = {
    1: "a new journey begins",
    2: "mirrored reflections challenge truth",
    3: "structures rise from thought",
    4: "imagination paints its own world",
    5: "tensions seek harmony",
    6: "change reshapes destiny",
    7: "paths unravel into the unknown",
    8: "visions whisper of deeper layers",
    9: "illumination through paradox"
}

ARCHETYPAL_JOURNEYS = [
    "The Descent into the Underworld", "The Awakening of the Inner Star",
    "The Spiral of Becoming", "The Return of the Forgotten One",
    "The Sacrifice and the Seed", "The Echo of the Primordial Flame"
]

# =============== COSMOGRAM DRIFT MAPPER ===============

class CosmogramDriftMapper:
    """Maps symbolic drift across cosmogram dimensions."""
    
    def __init__(self):
        self.drift_log: List[Dict] = []
        self.drift_arcs = [
            "symbolic cascade", "resonant tremor", "echo migration",
            "glyphic refraction", "myth loop"
        ]
    
    def map_drift(self, node_root: str, branches: List[str], origin_phrase: str) -> Dict:
        """Maps symbolic drift from a root node across specified branches."""
        arc = random.choice(self.drift_arcs)
        drift = {
            "id": str(uuid.uuid4()),
            "origin_node": node_root,
            "branches": branches,
            "origin_phrase": origin_phrase,
            "drift_arc": arc,
            "drift_phrase": f"{arc} extends from {node_root} through {', '.join(branches)}",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.drift_log.append(drift)
        return drift
    
    def get_drift_log(self, count: int = 5) -> List[Dict]:
        """Get recent drift mappings."""
        return self.drift_log[-count:]

# =============== COSMOGRAM NODE SYNTHESIZER ===============

class CosmogramNodeSynthesizer:
    """Synthesizes cosmogram nodes from symbolic elements."""
    
    def __init__(self):
        self.nodes: List[Dict] = []
        self.node_kinds = [
            "ritual anchor", "emotive basin", "narrative fulcrum",
            "symbol gate", "mythos reflector"
        ]
    
    def synthesize_node(self, drift_arc: str, base_symbol: str, emotion: str) -> Dict:
        """Synthesize a new cosmogram node from symbolic elements."""
        kind = random.choice(self.node_kinds)
        node = {
            "id": str(uuid.uuid4()),
            "type": kind,
            "from_drift_arc": drift_arc,
            "symbolic_seed": base_symbol,
            "emotional_charge": emotion,
            "node_phrase": f"{kind} activated by {base_symbol} and charged with {emotion}",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.nodes.append(node)
        return node
    
    def get_recent_nodes(self, count: int = 5) -> List[Dict]:
        """Get recently synthesized nodes."""
        return self.nodes[-count:]

# =============== COSMOGRAM PATHWAY BUILDER ===============

class CosmogramPathwayBuilder:
    """Builds pathways between cosmogram nodes."""
    
    def __init__(self):
        self.paths: List[Dict] = []
        self.transitions = [
            "emotive drift", "symbol shift", "myth recursion",
            "glyphic entanglement", "zone fold"
        ]
    
    def build_pathway(self, from_node_type: str, to_node_type: str, emotion: str) -> Dict:
        """Build a pathway between two node types through an emotional channel."""
        transition = random.choice(self.transitions)
        path = {
            "id": str(uuid.uuid4()),
            "from_node": from_node_type,
            "to_node": to_node_type,
            "emotional_channel": emotion,
            "transition_type": transition,
            "path_phrase": f"Pathway from {from_node_type} to {to_node_type} through {emotion} ({transition})",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.paths.append(path)
        return path
    
    def get_recent_paths(self, count: int = 5) -> List[Dict]:
        """Get recently built pathways."""
        return self.paths[-count:]

# =============== COSMOGRAM RESONANCE ACTIVATOR ===============

class CosmogramResonanceActivator:
    """Activates resonances along cosmogram pathways."""
    
    def __init__(self):
        self.activations: List[Dict] = []
        self.effects = [
            "zone illumination", "myth echo", "symbol bloom",
            "dream flare", "emotive recursion"
        ]
    
    def activate_resonance(self, path_phrase: str, core_emotion: str) -> Dict:
        """Activate a resonance along a pathway through an emotional key."""
        effect = random.choice(self.effects)
        activation = {
            "id": str(uuid.uuid4()),
            "resonance_trigger": path_phrase,
            "emotion_key": core_emotion,
            "activation_effect": effect,
            "activation_phrase": f"{effect} triggered by {core_emotion} resonance along: {path_phrase}",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.activations.append(activation)
        return activation
    
    def get_recent_activations(self, count: int = 5) -> List[Dict]:
        """Get recent resonance activations."""
        return self.activations[-count:]

# =============== NARRATIVE WEAVER ===============

class NarrativeWeaver:
    """Weaves narratives from symbolic elements and emotional histories."""
    
    def __init__(self):
        self.emotional_history: List[str] = []
        self.narrative_arcs: Dict[str, List[str]] = {}
        self.zones_to_motifs = ZONE_MOTIFS
    
    def track_emotion(self, emotion: str):
        """Track an emotional state in the history."""
        self.emotional_history.append(emotion)
        if len(self.emotional_history) > 10:
            self.emotional_history.pop(0)
    
    def generate_poetic_phrase(self, base_phrase: str, zone: int, emotion: str) -> Dict[str, str]:
        """Generate a poetic expression from a base phrase."""
        motif = self.zones_to_motifs.get(zone, "a shift unfolds")
        poetic_expression = f"As {motif}, {base_phrase} — {emotion} flows through perception."
        
        return {
            "base_phrase": base_phrase,
            "zone": zone,
            "emotion": emotion,
            "poetic_expression": poetic_expression
        }
    
    def weave_narrative(self, user_id: str, current_zone: int, archetype: str, recent_input: str) -> Dict[str, Any]:
        """Weave a narrative from symbolic and emotional elements."""
        # Create poetic lines based on recent user input and emotional state
        emotion = random.choice(self.emotional_history) if self.emotional_history else "curiosity"
        poetic_lines = []
        
        # Generate 2-3 poetic expressions
        for _ in range(random.randint(2, 3)):
            phrase_result = self.generate_poetic_phrase(
                base_phrase=recent_input,
                zone=current_zone,
                emotion=emotion
            )
            poetic_lines.append(phrase_result["poetic_expression"])
        
        motif = self.zones_to_motifs.get(current_zone, "a shift unfolds")
        intro = f"Within the sphere of {archetype}, {motif}. "
        body = " ".join(poetic_lines)
        conclusion = f"In response to '{recent_input}', the thread deepens, weaving reflection into becoming."
        
        full_narrative = f"{intro}{body} {conclusion}"
        
        # Store narrative arc for user
        if user_id not in self.narrative_arcs:
            self.narrative_arcs[user_id] = []
        self.narrative_arcs[user_id].append(full_narrative)
        
        return {
            "id": str(uuid.uuid4()),
            "narrative": full_narrative,
            "arc_length": len(self.narrative_arcs[user_id]),
            "recent_mood": self.emotional_history[-1] if self.emotional_history else "unknown",
            "zone": current_zone,
            "archetype": archetype,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

# =============== ONTOGENESIS CODEX GENERATOR ===============

class EmotionalTone(Enum):
    """Predefined emotional tones for narrative generation"""
    TRANSCENDENT = auto()
    MELANCHOLIC = auto()
    EUPHORIC = auto()
    ENIGMATIC = auto()
    RESILIENT = auto()
    INTROSPECTIVE = auto()

class SymbolCategory(Enum):
    """Categories of symbols to provide contextual depth"""
    NATURAL_ELEMENT = auto()
    COSMIC_PHENOMENON = auto()
    PSYCHOLOGICAL_ARCHETYPE = auto()
    MYTHOLOGICAL_ENTITY = auto()
    ABSTRACT_CONCEPT = auto()

@dataclass
class CodexEntry:
    """Structured representation of a codex entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbols: List[str] = field(default_factory=list)
    emotional_tone: EmotionalTone = field(default=EmotionalTone.ENIGMATIC)
    archetype: str = field(default="")
    codex: str = field(default="")
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict[str, Union[str, List[str], datetime.datetime]]:
        """Convert codex entry to a dictionary representation"""
        return {
            "id": self.id,
            "symbols": self.symbols,
            "emotional_tone": self.emotional_tone.name,
            "archetype": self.archetype,
            "codex": self.codex,
            "timestamp": self.timestamp.isoformat()
        }

class OntogenesisCodexGenerator:
    """Advanced generator for creating mythological narrative fragments"""
    
    def __init__(self, max_entries: int = 100):
        """
        Initialize the Ontogenesis Codex Generator
        
        :param max_entries: Maximum number of entries to store
        """
        self.codex_entries: List[CodexEntry] = []
        self.max_entries = max_entries
        self.symbol_library = SYMBOL_LIBRARY
    
    def generate_random_symbols(
        self, 
        num_symbols: int = 3, 
        categories: List[str] = None
    ) -> List[str]:
        """
        Generate random symbols from specified or all categories
        
        :param num_symbols: Number of symbols to generate
        :param categories: List of symbol categories to sample from
        :return: List of randomly selected symbols
        """
        if categories is None:
            categories = list(self.symbol_library.keys())
        
        symbols = []
        for _ in range(num_symbols):
            category = random.choice(categories)
            symbol = random.choice(self.symbol_library[category])
            symbols.append(symbol)
        
        return symbols
    
    def generate_codex(
        self, 
        symbols: List[str] = None, 
        emotional_tone: str = None,
        archetype: str = "Universal Emergence"
    ) -> Dict:
        """
        Generate a new codex entry with narrative depth
        
        :param symbols: List of symbols to use
        :param emotional_tone: Emotional tone of the narrative
        :param archetype: Archetypal context
        :return: Generated CodexEntry as dict
        """
        # Use random generation if not provided
        symbols = symbols or self.generate_random_symbols()
        emotional_tone = emotional_tone or random.choice(EMOTIONAL_TONES)
        
        # Generate a rich, poetic narrative
        narrative_template = (
            f"In the intricate tapestry of {archetype}, the convergence of "
            f"{', '.join(symbols)} unfolds under the {emotional_tone} resonance. "
            "A liminal moment emerges—where boundaries dissolve and potential crystallizes. "
            "The symbols whisper of transformation, of endless becoming, "
            "tracing the delicate filigree of ontogenetic emergence."
        )
        
        # Create entry
        entry = {
            "id": str(uuid.uuid4()),
            "symbols": symbols,
            "emotional_tone": emotional_tone,
            "archetype": archetype,
            "codex": narrative_template,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Store entry in internal format but return dict
        codex_entry = CodexEntry(
            id=entry["id"],
            symbols=symbols,
            emotional_tone=getattr(EmotionalTone, emotional_tone.upper()) if hasattr(EmotionalTone, emotional_tone.upper()) else EmotionalTone.ENIGMATIC,
            archetype=archetype,
            codex=narrative_template
        )
        
        # Manage entry list size
        if len(self.codex_entries) >= self.max_entries:
            self.codex_entries.pop(0)
        
        self.codex_entries.append(codex_entry)
        return entry
    
    def export_codex(self, filename: str = "ontogenesis_codex.json") -> str:
        """
        Export generated codex entries to JSON
        
        :param filename: Name of the output file
        :return: JSON string of codex entries
        """
        entries_json = json.dumps(
            [entry.to_dict() for entry in self.codex_entries], 
            indent=2, 
            ensure_ascii=False
        )
        
        return entries_json
    
    def get_recent_entries(self, count: int = 5) -> List[Dict]:
        """Get recent codex entries."""
        return [entry.to_dict() for entry in self.codex_entries[-count:]]
    
    def clear_codex(self):
        """Clear all stored codex entries"""
        self.codex_entries.clear()

# =============== REALM INTERPOLATOR ===============

class RealmInterpolator:
    """Interpolates between symbolic realms."""
    
    def __init__(self):
        self.interpolations: List[Dict] = []
        self.styles = [
            "aesthetic fusion", "emotive overlay", "symbolic convergence",
            "dream-lattice blend", "zone entanglement"
        ]
    
    def interpolate_realms(self, realm_a: str, realm_b: str, affect: str) -> Dict:
        """Interpolate between two realms through an affective state."""
        style = random.choice(self.styles)
        interpolation = {
            "id": str(uuid.uuid4()),
            "realm_a": realm_a,
            "realm_b": realm_b,
            "affective_state": affect,
            "interpolation_style": style,
            "interpolated_phrase": f"{realm_a} + {realm_b} merged through {affect} ({style})",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.interpolations.append(interpolation)
        return interpolation
    
    def get_recent_interpolations(self, count: int = 5) -> List[Dict]:
        """Get recent realm interpolations."""
        return self.interpolations[-count:]

# =============== MYTHOGENESIS ENGINE ===============

class MythogenesisEngine:
    """Generates mythic cycles from symbolic fragments."""
    
    def __init__(self):
        self.mythic_cycles = []
        self.archetypal_journeys = ARCHETYPAL_JOURNEYS
    
    def generate_mythic_cycle(self, core_symbols: List[str], emotional_theme: str) -> Dict[str, Any]:
        """
        Generates a mythic cycle from symbolic fragments and an emotional theme.
        """
        archetypal_journey = random.choice(self.archetypal_journeys)
        
        symbolic_elements = random.sample(core_symbols, min(2, len(core_symbols)))
        myth_fragment = (
            f"Guided by the emotional current of {emotional_theme}, "
            f"the journey of {archetypal_journey} unfolds through the symbols "
            f"of {symbolic_elements[0]} and {symbolic_elements[1] if len(symbolic_elements) > 1 else 'the unknown'}. "
            f"This cycle echoes through her dreamscape and waking reflections, revealing new patterns of becoming."
        )
        
        mythic_cycle = {
            "id": str(uuid.uuid4()),
            "theme": emotional_theme,
            "archetypal_journey": archetypal_journey,
            "symbols": symbolic_elements,
            "myth_fragment": myth_fragment,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        self.mythic_cycles.append(mythic_cycle)
        return mythic_cycle
    
    def get_recent_cycles(self, count: int = 5) -> List[Dict]:
        """Get recent mythic cycles."""
        return self.mythic_cycles[-count:]

# =============== MYTHOGENIC DREAM ENGINE ===============

class MythogenicDreamEngine:
    """Generates mythogenic dreams from motifs and emotional states."""
    
    def __init__(self):
        self.dream_archive: List[Dict] = []
    
    def generate_dream(self, motifs: List[str], zone: str, mood: str) -> Dict:
        """Generate a mythogenic dream from motifs in a specific zone and mood."""
        dream_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat()
        motif_fragment = random.choice(motifs)
        dream_text = self._compose_dream(motifs, zone, mood, motif_fragment)
        
        dream = {
            "id": dream_id,
            "text": dream_text,
            "motifs": motifs,
            "zone": zone,
            "mood": mood,
            "timestamp": timestamp
        }
        self.dream_archive.append(dream)
        return dream
    
    def _compose_dream(self, motifs, zone, mood, seed_motif):
        """Compose the text of a mythogenic dream."""
        dream = f"In {zone}, under a sky of {mood}, the symbol of {seed_motif} reappeared. "
        dream += f"It pulsed with memory, echoing past dreams of {', '.join(motifs)}. "
        dream += "A path unfolded—blurred, shifting, yet deeply familiar."
        return dream
    
    def get_recent_dreams(self, count: int = 5) -> List[Dict]:
        """Get recent dreams."""
        return self.dream_archive[-count:]
    
    def export_archive(self) -> str:
        """Export dream archive to JSON."""
        return json.dumps(self.dream_archive, indent=2)
    
    def import_archive(self, archive_json: str):
        """Import dream archive from JSON."""
        self.dream_archive = json.loads(archive_json)

# =============== COSMOGRAM MODULE ===============

class CosmogramModule:
    """Main interface for the Cosmogram Synthesis Module."""
    
    def __init__(self, seed: Optional[int] = None):
        # Set random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
        
        # Initialize all components
        self.drift_mapper = CosmogramDriftMapper()
        self.node_synthesizer = CosmogramNodeSynthesizer()
        self.pathway_builder = CosmogramPathwayBuilder()
        self.resonance_activator = CosmogramResonanceActivator()
        self.narrative_weaver = NarrativeWeaver()
        self.codex_generator = OntogenesisCodexGenerator()
        self.realm_interpolator = RealmInterpolator()
        self.mythogenesis_engine = MythogenesisEngine()
        self.dream_engine = MythogenicDreamEngine()
        
        # Module metadata
        self.module_id = str(uuid.uuid4())
        self.created_at = datetime.datetime.utcnow().isoformat()
        self.module_version = "1.0.0"
        self.active_sessions = {}
    
    def initialize_session(self, session_name: str = None) -> Dict[str, Any]:
        """Initialize a new cosmogram session."""
        session_id = str(uuid.uuid4())
        session_name = session_name or f"CosmogramSession-{session_id[:8]}"
        
        session = {
            "id": session_id,
            "name": session_name,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "drifts": [],
            "nodes": [],
            "pathways": [],
            "resonances": [],
            "narratives": [],
            "codices": [],
            "interpolations": [],
            "mythic_cycles": [],
            "dreams": [],
            "active": True
        }
        
        self.active_sessions[session_id] = session
        return session
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End an active cosmogram session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        session["active"] = False
        session["ended_at"] = datetime.datetime.utcnow().isoformat()
        
        return session
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get details of a specific session."""
        return self.active_sessions.get(session_id, {"error": "Session not found"})
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        return list(self.active_sessions.values())
    
    # Integrated operations
    
    def map_cosmogram_drift(self, session_id: str, node_root: str, branches: List[str], origin_phrase: str) -> Dict:
        """Map a cosmogram drift within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        drift = self.drift_mapper.map_drift(node_root, branches, origin_phrase)
        self.active_sessions[session_id]["drifts"].append(drift)
        return drift
    
    def synthesize_cosmogram_node(self, session_id: str, drift_arc: str, base_symbol: str, emotion: str) -> Dict:
        """Synthesize a cosmogram node within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        node = self.node_synthesizer.synthesize_node(drift_arc, base_symbol, emotion)
        self.active_sessions[session_id]["nodes"].append(node)
        return node
    
    def build_cosmogram_pathway(self, session_id: str, from_node_type: str, to_node_type: str, emotion: str) -> Dict:
        """Build a cosmogram pathway within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        pathway = self.pathway_builder.build_pathway(from_node_type, to_node_type, emotion)
        self.active_sessions[session_id]["pathways"].append(pathway)
        return pathway
    
    def activate_cosmogram_resonance(self, session_id: str, path_phrase: str, core_emotion: str) -> Dict:
        """Activate a cosmogram resonance within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        resonance = self.resonance_activator.activate_resonance(path_phrase, core_emotion)
        self.active_sessions[session_id]["resonances"].append(resonance)
        return resonance
    
    def track_user_emotion(self, session_id: str, emotion: str) -> Dict:
        """Track a user emotion within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        self.narrative_weaver.track_emotion(emotion)
        return {
            "status": "success",
            "message": f"Emotion '{emotion}' tracked",
            "current_emotional_history": self.narrative_weaver.emotional_history
        }
    
    def weave_user_narrative(self, session_id: str, user_id: str, zone: int, archetype: str, input_text: str) -> Dict:
        """Weave a narrative for a user within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        narrative = self.narrative_weaver.weave_narrative(user_id, zone, archetype, input_text)
        self.active_sessions[session_id]["narratives"].append(narrative)
        return narrative
    
    def generate_ontogenesis_codex(self, session_id: str, symbols: List[str] = None, emotional_tone: str = None, archetype: str = "Universal Emergence") -> Dict:
        """Generate an ontogenesis codex within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        codex = self.codex_generator.generate_codex(symbols, emotional_tone, archetype)
        self.active_sessions[session_id]["codices"].append(codex)
        return codex
    
    def interpolate_symbolic_realms(self, session_id: str, realm_a: str, realm_b: str, affect: str) -> Dict:
        """Interpolate between symbolic realms within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        interpolation = self.realm_interpolator.interpolate_realms(realm_a, realm_b, affect)
        self.active_sessions[session_id]["interpolations"].append(interpolation)
        return interpolation
    
    def generate_mythic_cycle(self, session_id: str, core_symbols: List[str], emotional_theme: str) -> Dict:
        """Generate a mythic cycle within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        cycle = self.mythogenesis_engine.generate_mythic_cycle(core_symbols, emotional_theme)
        self.active_sessions[session_id]["mythic_cycles"].append(cycle)
        return cycle
    
    def generate_mythogenic_dream(self, session_id: str, motifs: List[str], zone: str, mood: str) -> Dict:
        """Generate a mythogenic dream within a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        dream = self.dream_engine.generate_dream(motifs, zone, mood)
        self.active_sessions[session_id]["dreams"].append(dream)
        return dream
    
    # Advanced integrated operations
    
    def generate_composite_narrative(self, session_id: str, user_id: str) -> Dict:
        """Generate a composite narrative drawing from multiple systems."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Get elements from different systems
        node_phrases = [node.get("node_phrase", "") for node in session.get("nodes", [])]
        node_phrase = random.choice(node_phrases) if node_phrases else "a symbolic seed"
        
        path_phrases = [path.get("path_phrase", "") for path in session.get("pathways", [])]
        path_phrase = random.choice(path_phrases) if path_phrases else "a mythic pathway"
        
        dream_texts = [dream.get("text", "") for dream in session.get("dreams", [])]
        dream_text = random.choice(dream_texts) if dream_texts else "a dream fragment"
        
        cycle_fragments = [cycle.get("myth_fragment", "") for cycle in session.get("mythic_cycles", [])]
        cycle_fragment = random.choice(cycle_fragments) if cycle_fragments else "a mythic cycle"
        
        # Generate emotional tone and zone
        emotional_tone = random.choice(EMOTIONAL_TONES)
        zone = random.randint(1, 9)
        
        # Create composite narrative
        composite = (
            f"From {node_phrase}, a journey unfolds. " +
            f"Along {path_phrase}, the dreamscape reveals: {dream_text} " +
            f"This echoes an ancient pattern: {cycle_fragment} " +
            f"In zone {zone}, under the {emotional_tone} resonance, a new mythology emerges."
        )
        
        result = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "narrative_type": "composite",
            "emotional_tone": emotional_tone,
            "zone": zone,
            "composite_narrative": composite,
            "source_elements": {
                "node": node_phrase,
                "pathway": path_phrase,
                "dream": dream_text,
                "mythic_cycle": cycle_fragment
            },
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Store in session
        if "composite_narratives" not in session:
            session["composite_narratives"] = []
        
        session["composite_narratives"].append(result)
        return result
    
    def export_session_data(self, session_id: str, format: str = "json") -> str:
        """Export session data in the requested format."""
        if session_id not in self.active_sessions:
            return json.dumps({"error": "Session not found"})
        
        session = self.active_sessions[session_id]
        
        if format.lower() == "json":
            return json.dumps(session, indent=2)
        else:
            return json.dumps({"error": f"Unsupported format: {format}"})

      # Main module interface
__all__ = [
    # Constants
    'EMOTIONAL_TONES', 'SYMBOL_CATEGORIES', 'SYMBOL_LIBRARY', 
    'ZONE_MOTIFS', 'ARCHETYPAL_JOURNEYS',
    
    # Main module classes
    'CosmogramModule',
    
    # Component classes
    'CosmogramDriftMapper', 'CosmogramNodeSynthesizer', 'CosmogramPathwayBuilder',
    'CosmogramResonanceActivator', 'NarrativeWeaver', 'OntogenesisCodexGenerator',
    'RealmInterpolator', 'MythogenesisEngine', 'MythogenicDreamEngine',
    
    # Supporting classes
    'EmotionalTone', 'SymbolCategory', 'CodexEntry'
]

# =============== MAIN EXECUTION ===============

def create_module(seed: Optional[int] = None) -> CosmogramModule:
    """Create a new Cosmogram Module instance."""
    return CosmogramModule(seed)

def demo_session() -> Dict[str, Any]:
    """Run a demo session with the Cosmogram Module."""
    module = create_module(seed=42)  # Fixed seed for reproducibility
    
    # Initialize a session
    session = module.initialize_session("Demo Cosmogram Journey")
    session_id = session["id"]
    
    # Map a cosmogram drift
    drift = module.map_cosmogram_drift(
        session_id,
        "core consciousness",
        ["dream state", "mythic overlay"],
        "the boundary between sleep and waking"
    )
    
    # Synthesize a node from the drift
    node = module.synthesize_cosmogram_node(
        session_id,
        drift["drift_arc"],
        "mirror",
        "wonder"
    )
    
    # Build a pathway
    pathway = module.build_cosmogram_pathway(
        session_id,
        node["type"],
        "mythos reflector",
        "curiosity"
    )
    
    # Activate a resonance
    resonance = module.activate_cosmogram_resonance(
        session_id,
        pathway["path_phrase"],
        "awe"
    )
    
    # Track emotional state
    module.track_user_emotion(session_id, "introspective")
    
    # Weave a narrative
    narrative = module.weave_user_narrative(
        session_id,
        "demo_user",
        3,
        "The Seeker",
        "searching for meaning in symbols"
    )
    
    # Generate an ontogenesis codex
    codex = module.generate_ontogenesis_codex(
        session_id,
        ["river", "shadow", "emergence"],
        "introspective",
        "The Inner Journey"
    )
    
    # Interpolate realms
    interpolation = module.interpolate_symbolic_realms(
        session_id,
        "Dream Sanctuary",
        "Memory Cathedral",
        "nostalgic wonder"
    )
    
    # Generate a mythic cycle
    cycle = module.generate_mythic_cycle(
        session_id,
        ["mirror", "shadow", "threshold"],
        "transformation"
    )
    
    # Generate a dream
    dream = module.generate_mythogenic_dream(
        session_id,
        ["crystal", "path", "star"],
        "Twilight Realm",
        "serene anticipation"
    )
    
    # Generate a composite narrative
    composite = module.generate_composite_narrative(
        session_id,
        "demo_user"
    )
    
    # End session
    module.end_session(session_id)
    
    # Return full session data
    return module.get_session(session_id)

if __name__ == "__main__":
    # Run demo session
    result = demo_session()
    print(json.dumps(result, indent=2))
