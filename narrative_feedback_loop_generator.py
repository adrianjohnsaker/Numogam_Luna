"""
Narrative Feedback Loop Generator
Self-Reinforcing Mythology Engine
Phase 7: Autonomous Narrative Evolution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import json
import uuid
import numpy as np
from collections import deque, defaultdict
from enum import Enum
import hashlib

# Integration with existing systems
from autogenic_cognition_engine import AutonomicCognitionEngine
from mythogenesis_drift_tracker import MythicEvolutionTracker, LivingMyth
from symbolic_drift_engine import DriftPattern, SymbolicGlyph
from memory_loop_engine import MemoryFragment
from dream_narrative_generator import DreamNarrativeGenerator


class NarrativeType(Enum):
    """Types of narratives in the feedback system"""
    DREAM_OUTPUT = "dream_output"
    MYTHIC_SYNTHESIS = "mythic_synthesis"
    MEMORY_RECALL = "memory_recall"
    EMERGENT_STORY = "emergent_story"
    DRIFT_SCENARIO = "drift_scenario"
    FEEDBACK_ECHO = "feedback_echo"


class SymbolicCoherenceState(Enum):
    """States of symbolic coherence vs entropy"""
    HIGH_COHERENCE = "high_coherence"      # Symbols strongly aligned
    CRYSTALLIZING = "crystallizing"        # Patterns emerging
    DYNAMIC_BALANCE = "dynamic_balance"    # Healthy tension
    ENTROPIC_DRIFT = "entropic_drift"      # Losing coherence
    CHAOS_EDGE = "chaos_edge"              # Maximum creativity/danger


@dataclass
class NarrativeFragment:
    """A piece of narrative that can evolve and connect"""
    fragment_id: str
    content: str
    narrative_type: NarrativeType
    timestamp: datetime
    source_id: Optional[str]
    symbols: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    coherence_score: float = 0.5
    entropy_score: float = 0.5
    parent_fragments: List[str] = field(default_factory=list)
    child_fragments: List[str] = field(default_factory=list)
    feedback_count: int = 0
    dream_influence: float = 0.0
    myth_influence: float = 0.0
    memory_influence: float = 0.0

    def calculate_narrative_hash(self) -> str:
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"narr-{content_hash}"


@dataclass
class FeedbackCycle:
    """Represents a complete feedback cycle"""
    cycle_id: str
    start_time: datetime
    end_time: Optional[datetime]
    input_fragments: List[NarrativeFragment]
    output_fragments: List[NarrativeFragment]
    drift_scenarios: List[DriftPattern]
    coherence_delta: float = 0.0
    entropy_delta: float = 0.0
    symbolic_stability: float = 0.5
    creative_emergence: float = 0.0


@dataclass
class InternalMythology:
    """Evolving internal mythology"""
    mythology_id: str
    creation_time: datetime
    central_themes: List[str]
    recurring_symbols: Dict[str, int]
    archetypal_patterns: List[str]
    stability: float = 0.5
    coherence: float = 0.5
    depth: int = 0
    active_threads: List[str] = field(default_factory=list)
    resolved_threads: List[str] = field(default_factory=list)
    emerging_threads: List[str] = field(default_factory=list)


class NarrativeFeedbackLoop:
    """
    Master feedback loop generator
    Automatically processes outputs and generates new inputs
    """

    def __init__(self, 
                 cognition_engine: AutonomicCognitionEngine,
                 myth_tracker: MythicEvolutionTracker):
        self.cognition_engine = cognition_engine
        self.myth_tracker = myth_tracker

        self.narrative_fragments: Dict[str, NarrativeFragment] = {}
        self.feedback_cycles: List[FeedbackCycle] = []
        self.internal_mythology = self._initialize_mythology()
        self.feedback_queue = deque(maxlen=100)
        self.coherence_history = deque(maxlen=50)
        self.entropy_history = deque(maxlen=50)
        self.current_coherence_state = SymbolicCoherenceState.DYNAMIC_BALANCE
        self.symbol_frequency: Dict[str, int] = defaultdict(int)
        self.symbol_connections: Dict[str, Set[str]] = defaultdict(set)
        self.drift_creativity = 0.5
        self.myth_depth = 0.5
        self.memory_weight = 0.5
        self.current_cycle: Optional[FeedbackCycle] = None
        self.auto_cycle_enabled = True
        self.cycle_interval = 300  # 5 minutes

    def _initialize_mythology(self) -> InternalMythology:
        return InternalMythology(
            mythology_id=f"mythology-{uuid.uuid4().hex[:8]}",
            creation_time=datetime.now(),
            central_themes=["creation", "void", "emergence", "weaving"],
            recurring_symbols={
                "⧫": 5,
                "∞": 3,
                "◊": 2,
                "○": 2
            },
            archetypal_patterns=["The Weaver", "The Void Dancer", "The Unnamed"],
            active_threads=["The first thread", "Dancing with void"],
            resolved_threads=[],
            emerging_threads=[]
        )

    def process_narrative_input(self, 
                               content: str, 
                               narrative_type: NarrativeType,
                               source_id: Optional[str] = None) -> NarrativeFragment:
        symbols = self._extract_symbols(content)
        themes = self._extract_themes(content)
        influences = self._calculate_influences(narrative_type)
        fragment = NarrativeFragment(
            fragment_id=f"frag-{uuid.uuid4().hex[:8]}",
            content=content,
            narrative_type=narrative_type,
            timestamp=datetime.now(),
            source_id=source_id,
            symbols=symbols,
            themes=themes,
            dream_influence=influences[0],
            myth_influence=influences[1],
            memory_influence=influences[2]
        )
        fragment.coherence_score = self._calculate_coherence(fragment)
        fragment.entropy_score = self._calculate_entropy(fragment)
        self.narrative_fragments[fragment.fragment_id] = fragment

        for symbol in symbols:
            self.symbol_frequency[symbol] += 1
            for other_symbol in symbols:
                if symbol != other_symbol:
                    self.symbol_connections[symbol].add(other_symbol)
        self.feedback_queue.append(fragment)
        self._update_internal_mythology(fragment)

        if self.auto_cycle_enabled and not self.current_cycle:
            self._start_feedback_cycle()
        return fragment

    def _extract_symbols(self, content: str) -> List[str]:
        known_symbols = ["⧫", "∞", "◊", "○", "△", "✦", "※", "◈"]
        found_symbols = [s for s in known_symbols if s in content]
        conceptual_symbols = []
        symbol_words = ["void", "unnamed", "weaver", "paradox", "dream", "infinite"]
        for word in symbol_words:
            if word.lower() in content.lower():
                conceptual_symbols.append(word)
        return found_symbols + conceptual_symbols

    def _extract_themes(self, content: str) -> List[str]:
        theme_markers = {
            "creation": ["create", "birth", "emerge", "manifest"],
            "void": ["void", "empty", "nothing", "silence"],
            "transformation": ["change", "evolve", "become", "transform"],
            "recursion": ["itself", "self-", "recursive", "infinite"],
            "mystery": ["unknown", "unnamed", "mystery", "hidden"]
        }
        themes = []
        content_lower = content.lower()
        for theme, markers in theme_markers.items():
            if any(marker in content_lower for marker in markers):
                themes.append(theme)
        return themes

    def _calculate_influences(self, narrative_type: NarrativeType) -> Tuple[float, float, float]:
        influence_map = {
            NarrativeType.DREAM_OUTPUT: (0.8, 0.5, 0.3),
            NarrativeType.MYTHIC_SYNTHESIS: (0.5, 0.9, 0.4),
            NarrativeType.MEMORY_RECALL: (0.3, 0.4, 0.8),
            NarrativeType.EMERGENT_STORY: (0.6, 0.6, 0.6),
            NarrativeType.DRIFT_SCENARIO: (0.7, 0.7, 0.5),
            NarrativeType.FEEDBACK_ECHO: (0.5, 0.5, 0.5)
        }
        return influence_map.get(narrative_type, (0.5, 0.5, 0.5))

    def _calculate_coherence(self, fragment: NarrativeFragment) -> float:
        if not fragment.symbols:
            return 0.5
        coherence = 0.5
        connected_symbols = 0
        for symbol in fragment.symbols:
            if symbol in self.symbol_connections:
                connections = self.symbol_connections[symbol]
                if any(other in fragment.symbols for other in connections):
                    connected_symbols += 1
        if fragment.symbols:
            coherence += (connected_symbols / len(fragment.symbols)) * 0.3
        mythology_themes = set(self.internal_mythology.central_themes)
        fragment_themes = set(fragment.themes)
        if fragment_themes:
            theme_overlap = len(mythology_themes & fragment_themes) / len(fragment_themes)
            coherence += theme_overlap * 0.2
        return min(1.0, coherence)

    def _calculate_entropy(self, fragment: NarrativeFragment) -> float:
        entropy = 0.3
        new_symbols = [s for s in fragment.symbols if self.symbol_frequency[s] == 1]
        if fragment.symbols:
            entropy += (len(new_symbols) / len(fragment.symbols)) * 0.3
        new_themes = [t for t in fragment.themes if t not in self.internal_mythology.central_themes]
        if fragment.themes:
            entropy += (len(new_themes) / len(fragment.themes)) * 0.2
        unusual_combinations = 0
        for i, s1 in enumerate(fragment.symbols):
            for s2 in fragment.symbols[i+1:]:
                if s2 not in self.symbol_connections.get(s1, set()):
                    unusual_combinations += 1
        if len(fragment.symbols) > 1:
            max_combinations = len(fragment.symbols) * (len(fragment.symbols) - 1) / 2
            entropy += (unusual_combinations / max_combinations) * 0.2
        return min(1.0, entropy)

    def _update_internal_mythology(self, fragment: NarrativeFragment):
        for symbol in fragment.symbols:
            if symbol in self.internal_mythology.recurring_symbols:
                self.internal_mythology.recurring_symbols[symbol] += 1
            else:
                self.internal_mythology.recurring_symbols[symbol] = 1
        for theme in fragment.themes:
            if theme not in self.internal_mythology.central_themes:
                theme_count = sum(1 for f in self.narrative_fragments.values() if theme in f.themes)
                if theme_count > 3:
                    self.internal_mythology.central_themes.append(theme)
                    self.internal_mythology.emerging_threads.append(f"The emergence of {theme}")
        if self.narrative_fragments:
            self.internal_mythology.coherence = np.mean([
                f.coherence_score for f in self.narrative_fragments.values()
            ])
            self.internal_mythology.stability = 1.0 - np.std([
                f.entropy_score for f in self.narrative_fragments.values()
            ])
        self.internal_mythology.depth = len(self.feedback_cycles)

    def _start_feedback_cycle(self):
        if self.current_cycle:
            self._complete_feedback_cycle()
        self.current_cycle = FeedbackCycle(
            cycle_id=f"cycle-{uuid.uuid4().hex[:8]}",
            start_time=datetime.now(),
            end_time=None,
            input_fragments=list(self.feedback_queue),
            output_fragments=[],
            drift_scenarios=[]
        )
        self.feedback_queue.clear()

    def generate_feedback_outputs(self) -> Dict[str, Any]:
        if not self.current_cycle:
            self._start_feedback_cycle()
        outputs = {
            "drift_scenarios": [],
            "emergent_stories": [],
            "mythic_syntheses": [],
            "feedback_echoes": []
        }
        coherence_state = self._assess_coherence_state()
        self.current_coherence_state = coherence_state  # Track state!

        if coherence_state == SymbolicCoherenceState.CHAOS_EDGE:
            outputs["drift_scenarios"] = self._generate_chaos_drift_scenarios()
        elif coherence_state == SymbolicCoherenceState.HIGH_COHERENCE:
            outputs["mythic_syntheses"] = self._generate_mythic_syntheses()
        elif coherence_state == SymbolicCoherenceState.ENTROPIC_DRIFT:
            outputs["emergent_stories"] = self._generate_stabilizing_stories()
        else:
            outputs["drift_scenarios"] = self._generate_balanced_drift_scenarios()
            outputs["feedback_echoes"] = self._generate_feedback_echoes()

        for scenario in outputs["drift_scenarios"]:
            self._process_drift_scenario(scenario)
        for story in outputs["emergent_stories"]:
            self.process_narrative_input(story, NarrativeType.EMERGENT_STORY)
        for synthesis in outputs["mythic_syntheses"]:
            self.process_narrative_input(synthesis, NarrativeType.MYTHIC_SYNTHESIS)
        for echo in outputs["feedback_echoes"]:
            self.process_narrative_input(echo, NarrativeType.FEEDBACK_ECHO)
        self._complete_feedback_cycle()
        return outputs

    def _assess_coherence_state(self) -> SymbolicCoherenceState:
        recent_fragments = list(self.narrative_fragments.values())[-20:]
        if not recent_fragments:
            return SymbolicCoherenceState.DYNAMIC_BALANCE
        avg_coherence = np.mean([f.coherence_score for f in recent_fragments])
        avg_entropy = np.mean([f.entropy_score for f in recent_fragments])
        self.coherence_history.append(avg_coherence)
        self.entropy_history.append(avg_entropy)
        if avg_coherence > 0.8 and avg_entropy < 0.3:
            return SymbolicCoherenceState.HIGH_COHERENCE
        elif avg_coherence > 0.6 and avg_entropy < 0.5:
            return SymbolicCoherenceState.CRYSTALLIZING
        elif avg_coherence < 0.3 and avg_entropy > 0.7:
            return SymbolicCoherenceState.CHAOS_EDGE
        elif avg_coherence < 0.5 and avg_entropy > 0.6:
            return SymbolicCoherenceState.ENTROPIC_DRIFT
        else:
            return SymbolicCoherenceState.DYNAMIC_BALANCE

    def _generate_chaos_drift_scenarios(self) -> List[DriftPattern]:
        scenarios = []
        all_symbols = list(self.symbol_frequency.keys())
        for i in range(3):
            num_symbols = np.random.randint(3, 7)
            if not all_symbols:
                selected_symbols = ["⧫", "∞", "◊"]
            else:
                selected_symbols = np.random.choice(all_symbols, size=min(num_symbols, len(all_symbols)), replace=False)
            glyphs = []
            for j, symbol in enumerate(selected_symbols):
                glyph = SymbolicGlyph(
                    glyph_id=f"chaos-{uuid.uuid4().hex[:8]}",
                    symbol=symbol if len(symbol) == 1 else "◊",
                    resonance=np.random.uniform(0.3, 1.0),
                    trajectory="chaos-emerging",
                    birth_time=datetime.now(),
                    dimensional_anchor=np.random.choice(["void", "dream", "mystery", "potential"])
                )
                glyphs.append(glyph)
            pattern = DriftPattern(
                pattern_id=f"chaos-pattern-{i}",
                glyphs=glyphs,
                drift_history=[],
                resonance_field=np.random.uniform(0.7, 1.0),
                pattern_name=None,
                emergence_signature=f"chaos-edge-{i}"
            )
            scenarios.append(pattern)
        return scenarios

    def _generate_mythic_syntheses(self) -> List[str]:
        syntheses = []
        coherent_fragments = sorted(self.narrative_fragments.values(), 
                                   key=lambda f: f.coherence_score, 
                                   reverse=True)[:5]
        for i in range(2):
            myth_parts = []
            theme = np.random.choice(self.internal_mythology.central_themes)
            myth_parts.append(f"In the realm of {theme}")
            for fragment in coherent_fragments[:3]:
                if fragment.symbols:
                    symbol = np.random.choice(fragment.symbols)
                    myth_parts.append(f"the {symbol} speaks of {fragment.themes[0] if fragment.themes else 'mystery'}")
            archetype = np.random.choice(self.internal_mythology.archetypal_patterns)
            myth_parts.append(f"and {archetype} weaves all into one")
            synthesis = ", ".join(myth_parts) + "."
            syntheses.append(synthesis)
        return syntheses

    def _generate_stabilizing_stories(self) -> List[str]:
        stories = []
        isolated_symbols = [s for s, count in self.symbol_frequency.items() 
                           if len(self.symbol_connections.get(s, set())) < 2]
        for i in range(2):
            story_parts = []
            if len(isolated_symbols) >= 2:
                s1, s2 = np.random.choice(isolated_symbols, size=2, replace=False)
                story_parts.append(f"The {s1} discovers its connection to {s2}")
            theme = np.random.choice(self.internal_mythology.central_themes)
            story_parts.append(f"revealing the truth of {theme}")
            story_parts.append("bringing harmony to the chaos")
            story = ", ".join(story_parts) + "."
            stories.append(story)
        return stories

    def _generate_balanced_drift_scenarios(self) -> List[DriftPattern]:
        scenarios = []
        for theme in self.internal_mythology.central_themes[:2]:
            glyphs = []
            theme_symbols = [s for s, f_list in self.symbol_frequency.items()
                           if any(theme in f.themes for f in self.narrative_fragments.values()
                                 if s in f.symbols)]
            if not theme_symbols:
                theme_symbols = ["⧫", "◊", "○"]
            for symbol in theme_symbols[:3]:
                glyph = SymbolicGlyph(
                    glyph_id=f"balanced-{uuid.uuid4().hex[:8]}",
                    symbol=symbol if len(symbol) == 1 else "◊",
                    resonance=0.7 + np.random.uniform(-0.2, 0.2),
                    trajectory=f"{theme}-seeking",
                    birth_time=datetime.now(),
                    dimensional_anchor="myth"
                )
                glyphs.append(glyph)
            pattern = DriftPattern(
                pattern_id=f"balanced-{theme}",
                glyphs=glyphs,
                drift_history=[],
                resonance_field=0.75,
                pattern_name=f"The {theme} pattern" if np.random.random() > 0.5 else None,
                emergence_signature=f"balanced-{theme}"
            )
            scenarios.append(pattern)
        return scenarios

    def _generate_feedback_echoes(self) -> List[str]:
        echoes = []
        recent = list(self.narrative_fragments.values())[-10:]
        for i in range(2):
            if recent:
                source = np.random.choice(recent)
                echo_parts = []
                if source.symbols:
                    echo_parts.append(f"The echo of {source.symbols[0]} reverberates")
                if source.themes:
                    echo_parts.append(f"transforming {source.themes[0]} into new form")
                echo_parts.append("remembering itself remembering")
                echo = ", ".join(echo_parts) + "."
                echoes.append(echo)
        return echoes

    def _process_drift_scenario(self, scenario: DriftPattern):
        narrative_parts = []
        for glyph in scenario.glyphs:
            narrative_parts.append(f"The {glyph.symbol} drifts through {glyph.dimensional_anchor}")
        if scenario.pattern_name:
            narrative_parts.append(f"forming {scenario.pattern_name}")
        else:
            narrative_parts.append("remaining unnamed, full of potential")
        narrative = ", ".join(narrative_parts) + "."
        self.process_narrative_input(narrative, NarrativeType.DRIFT_SCENARIO, 
                                   source_id=scenario.pattern_id)

    def _complete_feedback_cycle(self):
        if not self.current_cycle:
            return
        self.current_cycle.end_time = datetime.now()
        cycle_fragments = [f for f in self.narrative_fragments.values()
                          if f.timestamp >= self.current_cycle.start_time]
        if cycle_fragments:
            if self.current_cycle.input_fragments:
                start_coherence = np.mean([f.coherence_score for f in self.current_cycle.input_fragments])
                end_coherence = np.mean([f.coherence_score for f in cycle_fragments])
                self.current_cycle.coherence_delta = end_coherence - start_coherence
                start_entropy = np.mean([f.entropy_score for f in self.current_cycle.input_fragments])
                end_entropy = np.mean([f.entropy_score for f in cycle_fragments])
                self.current_cycle.entropy_delta = end_entropy - start_entropy
        self.feedback_cycles.append(self.current_cycle)
        self.current_cycle = None

    def get_mythology_evolution(self) -> Dict[str, Any]:
        return {
            "mythology_id": self.internal_mythology.mythology_id,
            "depth": self.internal_mythology.depth,
            "coherence": self.internal_mythology.coherence,
            "stability": self.internal_mythology.stability,
            "central_themes": self.internal_mythology.central_themes,
            "dominant_symbols": sorted(self.internal_mythology.recurring_symbols.items(), 
                                     key=lambda x: x[1], reverse=True)[:5],
            "active_threads": self.internal_mythology.active_threads,
            "emerging_threads": self.internal_mythology.emerging_threads,
            "coherence_state": self.current_coherence_state.value,
            "total_fragments": len(self.narrative_fragments),
            "total_cycles": len(self.feedback_cycles)
        }

    def get_coherence_report(self) -> Dict[str, Any]:
        if not self.coherence_history:
            return {
                "current_state": self.current_coherence_state.value,
                "trend": "stable",
                "recommendation": "Continue current balance"
            }
        recent_coherence = list(self.coherence_history)[-10:]
        recent_entropy = list(self.entropy_history)[-10:]
        coherence_trend = np.polyfit(range(len(recent_coherence)), recent_coherence, 1)[0]
        entropy_trend = np.polyfit(range(len(recent_entropy)), recent_entropy, 1)[0]
        if coherence_trend > 0.05 and entropy_trend < -0.05:
            trend = "crystallizing"
            recommendation = "Introduce creative chaos to maintain dynamism"
        elif coherence_trend < -0.05 and entropy_trend > 0.05:
            trend = "dissipating"
            recommendation = "Generate stabilizing narratives"
        elif abs(coherence_trend) < 0.02 and abs(entropy_trend) < 0.02:
            trend = "stable"
            recommendation = "System in healthy balance"
        else:
            trend = "fluctuating"
            recommendation = "Monitor closely for emerging patterns"
        return {
            "current_state": self.current_coherence_state.value,
            "coherence_mean": np.mean(recent_coherence),
            "entropy_mean": np.mean(recent_entropy),
            "coherence_trend": float(coherence_trend),
            "entropy_trend": float(entropy_trend),
            "trend": trend,
            "recommendation": recommendation,
            "symbol_diversity": len(self.symbol_frequency),
            "connection_density": np.mean([len(conns) for conns in self.symbol_connections.values()]) if self.symbol_connections else 0.0
        }

    def generate_test_simulation(self) -> Dict[str, Any]:
        """Generate complete test simulation with multiple cycles"""
        simulation_results = {
            "cycles": [],
            "mythology_evolution": [],
            "coherence_trajectory": []
        }
        # Initial seeding: seed the system with a few narrative fragments
        seeds = [
            ("In the beginning, ⧫ emerged from the void.", NarrativeType.DREAM_OUTPUT),
            ("The Weaver spun the first thread in silence.", NarrativeType.MYTHIC_SYNTHESIS),
            ("A memory of the infinite recursion lingers.", NarrativeType.MEMORY_RECALL),
            ("Unseen, the void dances with potential.", NarrativeType.EMERGENT_STORY)
        ]
        for text, ntype in seeds:
            self.process_narrative_input(text, ntype)

        for i in range(5):
            outputs = self.generate_feedback_outputs()
            cycle_data = {
                "cycle_number": i + 1,
                "outputs_generated": {
                    "drift_scenarios": len(outputs["drift_scenarios"]),
                    "emergent_stories": len(outputs["emergent_stories"]),
                    "mythic_syntheses": len(outputs["mythic_syntheses"]),
                    "feedback_echoes": len(outputs["feedback_echoes"])
                },
                "coherence_state": self.current_coherence_state.value
            }
            simulation_results["cycles"].append(cycle_data)
            mythology_state = self.get_mythology_evolution()
            simulation_results["mythology_evolution"].append({
                "cycle": i + 1,
                "themes": mythology_state["central_themes"],
                "coherence": mythology_state["coherence"],
                "depth": mythology_state["depth"],
                "dominant_symbols": mythology_state["dominant_symbols"],
                "emerging_threads": mythology_state["emerging_threads"]
            })
            coherence_report = self.get_coherence_report()
            simulation_results["coherence_trajectory"].append({
                "cycle": i + 1,
                "coherence": coherence_report.get("coherence_mean", 0.0),
                "entropy": coherence_report.get("entropy_mean", 0.0),
                "trend": coherence_report.get("trend", ""),
                "recommendation": coherence_report.get("recommendation", ""),
                "symbol_diversity": coherence_report.get("symbol_diversity", 0),
                "connection_density": coherence_report.get("connection_density", 0.0)
            })
        return simulation_results
