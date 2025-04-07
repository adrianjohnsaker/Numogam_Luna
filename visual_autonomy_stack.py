import json
from typing import Dict, Any

from intent_generator import IntentGenerator
from symbolic_attractor_engine import SymbolicAttractorEngine
from drift_priority_assessor import DriftPriorityAssessor
from mythic_identity_manager import MythicIdentityManager
from reflective_state_tracker import ReflectiveStateTracker
from symbolic_conflict_resolver import SymbolicConflictResolver
from internal_guidance_oracle import InternalGuidanceOracle
from zone_preference_weaver import ZonePreferenceWeaver
from glyph_visual_generator import GlyphVisualGenerator
from zone_emotion_mapper import ZoneEmotionMapper
from visual_hyperstructure_synthesizer import VisualHyperstructureSynthesizer
from render_json_out import RenderJsonOut

class VisualAutonomyStack:
    def __init__(self):
        self.intent_gen = IntentGenerator()
        self.attractor = SymbolicAttractorEngine()
        self.drift_assessor = DriftPriorityAssessor()
        self.mythic_id = MythicIdentityManager()
        self.state_tracker = ReflectiveStateTracker()
        self.conflict_resolver = SymbolicConflictResolver()
        self.guidance_oracle = InternalGuidanceOracle()
        self.zone_weaver = ZonePreferenceWeaver()
        self.glyph_generator = GlyphVisualGenerator()
        self.zone_emotion = ZoneEmotionMapper()
        self.visual_synth = VisualHyperstructureSynthesizer()
        self.renderer = RenderJsonOut()

    def process_stack(self, user_id: str, zone: int, emotion: str, recent_input: str) -> Dict[str, Any]:
        try:
            intent = self.intent_gen.generate_intent(recent_input)
            attractor = self.attractor.evaluate_attraction(intent)
            drift_priority = self.drift_assessor.assess_priority(intent)
            mythic_self = self.mythic_id.update_identity(user_id, intent)
            state_reflection = self.state_tracker.track_state(user_id, intent)
            resolution = self.conflict_resolver.resolve_conflict(user_id, intent)
            guidance = self.guidance_oracle.provide_guidance(intent)
            zone_pref = self.zone_weaver.weave_preference(user_id, zone, emotion)
            glyph = self.glyph_generator.generate_glyph(zone, emotion)
            zone_emotion_data = self.zone_emotion.map_zone_emotion(zone, emotion)
            structure = self.visual_synth.synthesize_structure(
                zone, emotion, state_reflection["self_state"], resolution, glyph["symbol"]
            )
            rendered = self.renderer.render_json(structure)

            return {
                "status": "success",
                "intent": intent,
                "attractor": attractor,
                "drift_priority": drift_priority,
                "mythic_self": mythic_self,
                "reflected_state": state_reflection,
                "conflict_resolution": resolution,
                "guidance": guidance,
                "zone_preference": zone_pref,
                "glyph_data": glyph,
                "zone_emotion": zone_emotion_data,
                "visual_structure": structure,
                "rendered_output": rendered
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "type": type(e).__name__
            }
