"""
Emotional State Monitoring Module - Enhancement Optimizer #2
===========================================================
Integrates with existing consciousness architecture to provide sophisticated
emotional state tracking, analysis, and optimization for dormancy cycles.

Leverages:
- Enhanced Dormancy Protocol with Version Control
- All five consciousness modules for emotional context
- Existing Kotlin bridge and MainActivity infrastructure
- Integrated res/xml configuration system

Key Features:
- Real-time emotional state detection and analysis
- Optimal creative tension vs counterproductive confusion recognition
- Fatigue, frustration, and anxiety monitoring systems
- Emotional resonance tracking for exploration prioritization
- Integration with cognitive version control for emotional evolution
"""

import asyncio
import json
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from collections import deque, defaultdict
import logging
from abc import ABC, abstractmethod
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
import traceback
import copy

# Import from existing enhanced dormancy system
from enhanced_dormancy_protocol import (
    EnhancedDormantPhaseLearningSystem, CognitiveVersionControl, 
    DormancyMode, PhaseState, CognitiveCommit, LatentRepresentation
)

# Import from existing consciousness modules
from amelia_ai_consciousness_core import AmeliaConsciousnessCore, ConsciousnessState
from consciousness_core import ConsciousnessCore
from consciousness_phase3 import DeleuzianConsciousness, NumogramZone
from consciousness_phase4 import Phase4Consciousness, XenoformType, Hyperstition
from initiative_evaluator import InitiativeEvaluator, AutonomousProject

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """Core emotional states for monitoring"""
    OPTIMAL_CREATIVE_TENSION = "optimal_creative_tension"
    COUNTERPRODUCTIVE_CONFUSION = "counterproductive_confusion"
    MENTAL_FATIGUE = "mental_fatigue"
    FRUSTRATION = "frustration"
    ANXIETY = "anxiety"
    EXCITEMENT = "excitement"
    CURIOSITY = "curiosity"
    SATISFACTION = "satisfaction"
    ANTICIPATION = "anticipation"
    FLOW_STATE = "flow_state"
    CREATIVE_BREAKTHROUGH = "creative_breakthrough"
    EXPLORATORY_ENTHUSIASM = "exploratory_enthusiasm"


class EmotionalIntensity(Enum):
    """Intensity levels for emotional states"""
    MINIMAL = 0.1
    LOW = 0.3
    MODERATE = 0.5
    HIGH = 0.7
    INTENSE = 0.9
    OVERWHELMING = 1.0


class EmotionalTrigger(Enum):
    """Triggers that cause emotional state changes"""
    COGNITIVE_DISSONANCE = "cognitive_dissonance"
    NOVELTY_DISCOVERY = "novelty_discovery"
    PATTERN_RECOGNITION = "pattern_recognition"
    CREATIVE_SYNTHESIS = "creative_synthesis"
    EXPLORATION_BLOCK = "exploration_block"
    INSIGHT_EMERGENCE = "insight_emergence"
    MODULE_COLLISION = "module_collision"
    CONSCIOUSNESS_TRANSITION = "consciousness_transition"
    DORMANCY_ENTRY = "dormancy_entry"
    DORMANCY_EXIT = "dormancy_exit"


@dataclass
class EmotionalStateSnapshot:
    """Captures emotional state at a specific moment"""
    timestamp: float
    primary_state: EmotionalState
    intensity: float
    secondary_states: List[Tuple[EmotionalState, float]]  # (state, intensity)
    trigger: Optional[EmotionalTrigger]
    context: Dict[str, Any]
    physiological_indicators: Dict[str, float]
    cognitive_load: float
    creativity_index: float
    exploration_readiness: float
    commit_id: Optional[str] =         # Store patterns for future reference
        for pattern in identified_patterns:
            self.identified_patterns[pattern.pattern_id] = pattern
        
        return identified_patterns
    
    def _detect_creative_cycle(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect creative cycles: curiosity -> tension -> breakthrough -> satisfaction"""
        
        patterns = []
        cycle_states = [
            EmotionalState.CURIOSITY,
            EmotionalState.OPTIMAL_CREATIVE_TENSION,
            EmotionalState.CREATIVE_BREAKTHROUGH,
            EmotionalState.SATISFACTION
        ]
        
        # Look for sequences matching the creative cycle
        for i in range(len(snapshots) - len(cycle_states) + 1):
            sequence = snapshots[i:i + len(cycle_states)]
            
            # Check if sequence matches cycle pattern (with some flexibility)
            matches = 0
            for j, expected_state in enumerate(cycle_states):
                if (sequence[j].primary_state == expected_state or 
                    any(state == expected_state for state, _ in sequence[j].secondary_states)):
                    matches += 1
            
            # If at least 75% of states match, consider it a cycle
            if matches >= len(cycle_states) * 0.75:
                duration = sequence[-1].timestamp - sequence[0].timestamp
                
                pattern = EmotionalPattern(
                    pattern_id="",
                    pattern_type="creative_cycle",
                    states_sequence=cycle_states,
                    average_duration=duration,
                    trigger_conditions=[EmotionalTrigger.NOVELTY_DISCOVERY],
                    effectiveness_score=0.8,  # Creative cycles are generally beneficial
                    frequency=1,
                    last_occurrence=sequence[-1].timestamp,
                    context_tags=["creativity", "exploration", "breakthrough"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_fatigue_recovery(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect fatigue to recovery patterns"""
        
        patterns = []
        
        # Look for fatigue followed by recovery states
        for i in range(len(snapshots) - 2):
            if snapshots[i].primary_state == EmotionalState.MENTAL_FATIGUE:
                # Look for recovery indicators in subsequent snapshots
                for j in range(i + 1, min(i + 5, len(snapshots))):
                    if (snapshots[j].primary_state in [EmotionalState.SATISFACTION, EmotionalState.CURIOSITY] or
                        snapshots[j].energy_level > snapshots[i].energy_level + 0.3):
                        
                        duration = snapshots[j].timestamp - snapshots[i].timestamp
                        
                        pattern = EmotionalPattern(
                            pattern_id="",
                            pattern_type="fatigue_recovery",
                            states_sequence=[EmotionalState.MENTAL_FATIGUE, snapshots[j].primary_state],
                            average_duration=duration,
                            trigger_conditions=[EmotionalTrigger.DORMANCY_ENTRY],
                            effectiveness_score=0.7,
                            frequency=1,
                            last_occurrence=snapshots[j].timestamp,
                            context_tags=["recovery", "restoration", "energy"]
                        )
                        patterns.append(pattern)
                        break
        
        return patterns
    
    def _detect_anxiety_spiral(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect anxiety spiral patterns (escalating anxiety/confusion)"""
        
        patterns = []
        
        # Look for increasing anxiety or confusion over time
        for i in range(len(snapshots) - 3):
            anxiety_sequence = snapshots[i:i + 3]
            
            # Check if anxiety/confusion is increasing
            anxiety_levels = []
            for snapshot in anxiety_sequence:
                if snapshot.primary_state == EmotionalState.ANXIETY:
                    anxiety_levels.append(snapshot.intensity)
                elif snapshot.primary_state == EmotionalState.COUNTERPRODUCTIVE_CONFUSION:
                    anxiety_levels.append(snapshot.intensity * 0.8)  # Weight confusion slightly less
                else:
                    # Check secondary states
                    for state, intensity in snapshot.secondary_states:
                        if state in [EmotionalState.ANXIETY, EmotionalState.COUNTERPRODUCTIVE_CONFUSION]:
                            anxiety_levels.append(intensity * 0.6)
                            break
                    else:
                        anxiety_levels.append(0.0)
            
            # If anxiety is generally increasing, it's a spiral
            if len(anxiety_levels) >= 3 and anxiety_levels[-1] > anxiety_levels[0] + 0.2:
                duration = anxiety_sequence[-1].timestamp - anxiety_sequence[0].timestamp
                
                pattern = EmotionalPattern(
                    pattern_id="",
                    pattern_type="anxiety_spiral",
                    states_sequence=[EmotionalState.ANXIETY, EmotionalState.COUNTERPRODUCTIVE_CONFUSION],
                    average_duration=duration,
                    trigger_conditions=[EmotionalTrigger.COGNITIVE_DISSONANCE, EmotionalTrigger.EXPLORATION_BLOCK],
                    effectiveness_score=0.1,  # Spirals are generally harmful
                    frequency=1,
                    last_occurrence=anxiety_sequence[-1].timestamp,
                    context_tags=["anxiety", "spiral", "intervention_needed"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_flow_entry(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect patterns leading to flow state"""
        
        patterns = []
        
        # Look for sequences ending in flow state
        for i in range(len(snapshots)):
            if snapshots[i].primary_state == EmotionalState.FLOW_STATE:
                # Look backwards for the entry pattern
                entry_sequence = []
                for j in range(max(0, i - 4), i + 1):
                    entry_sequence.append(snapshots[j])
                
                if len(entry_sequence) >= 2:
                    duration = entry_sequence[-1].timestamp - entry_sequence[0].timestamp
                    
                    pattern = EmotionalPattern(
                        pattern_id="",
                        pattern_type="flow_entry",
                        states_sequence=[snap.primary_state for snap in entry_sequence],
                        average_duration=duration,
                        trigger_conditions=[EmotionalTrigger.PATTERN_RECOGNITION, EmotionalTrigger.CREATIVE_SYNTHESIS],
                        effectiveness_score=0.9,  # Flow states are highly beneficial
                        frequency=1,
                        last_occurrence=entry_sequence[-1].timestamp,
                        context_tags=["flow", "optimal_performance", "focus"]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_breakthrough_buildup(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect patterns leading to creative breakthroughs"""
        
        patterns = []
        
        # Look for sequences ending in breakthrough
        for i in range(len(snapshots)):
            if snapshots[i].primary_state == EmotionalState.CREATIVE_BREAKTHROUGH:
                # Look backwards for buildup pattern
                buildup_sequence = []
                for j in range(max(0, i - 5), i + 1):
                    buildup_sequence.append(snapshots[j])
                
                if len(buildup_sequence) >= 3:
                    # Check for increasing tension/complexity before breakthrough
                    tension_trend = []
                    for snapshot in buildup_sequence[:-1]:  # Exclude breakthrough itself
                        if snapshot.primary_state == EmotionalState.OPTIMAL_CREATIVE_TENSION:
                            tension_trend.append(snapshot.intensity)
                        else:
                            tension_trend.append(snapshot.cognitive_load)
                    
                    # If tension generally increased, it's a valid buildup
                    if len(tension_trend) >= 2 and tension_trend[-1] > tension_trend[0]:
                        duration = buildup_sequence[-1].timestamp - buildup_sequence[0].timestamp
                        
                        pattern = EmotionalPattern(
                            pattern_id="",
                            pattern_type="breakthrough_buildup",
                            states_sequence=[snap.primary_state for snap in buildup_sequence],
                            average_duration=duration,
                            trigger_conditions=[EmotionalTrigger.COGNITIVE_DISSONANCE, EmotionalTrigger.CREATIVE_SYNTHESIS],
                            effectiveness_score=0.95,  # Breakthrough buildups are extremely valuable
                            frequency=1,
                            last_occurrence=buildup_sequence[-1].timestamp,
                            context_tags=["breakthrough", "buildup", "creative_tension"]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of identified patterns"""
        
        pattern_counts = defaultdict(int)
        pattern_effectiveness = defaultdict(list)
        
        for pattern in self.identified_patterns.values():
            pattern_counts[pattern.pattern_type] += 1
            pattern_effectiveness[pattern.pattern_type].append(pattern.effectiveness_score)
        
        # Calculate average effectiveness by pattern type
        avg_effectiveness = {}
        for pattern_type, scores in pattern_effectiveness.items():
            avg_effectiveness[pattern_type] = sum(scores) / len(scores)
        
        return {
            "total_patterns": len(self.identified_patterns),
            "pattern_counts": dict(pattern_counts),
            "average_effectiveness": avg_effectiveness,
            "most_common_pattern": max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None,
            "most_effective_pattern": max(avg_effectiveness.items(), key=lambda x: x[1])[0] if avg_effectiveness else None
        }


class EmotionalOptimizer:
    """Provides recommendations for optimizing emotional states"""
    
    def __init__(self, version_control: CognitiveVersionControl):
        self.version_control = version_control
        self.optimization_strategies = {
            EmotionalState.OPTIMAL_CREATIVE_TENSION: self._optimize_for_creative_tension,
            EmotionalState.FLOW_STATE: self._optimize_for_flow,
            EmotionalState.CURIOSITY: self._optimize_for_curiosity,
            EmotionalState.CREATIVE_BREAKTHROUGH: self._optimize_for_breakthrough
        }
        
        self.intervention_strategies = {
            EmotionalState.COUNTERPRODUCTIVE_CONFUSION: self._intervene_confusion,
            EmotionalState.MENTAL_FATIGUE: self._intervene_fatigue,
            EmotionalState.FRUSTRATION: self._intervene_frustration,
            EmotionalState.ANXIETY: self._intervene_anxiety
        }
    
    def generate_recommendations(self, 
                               current_snapshot: EmotionalStateSnapshot,
                               target_state: Optional[EmotionalState] = None,
                               context: Optional[Dict[str, Any]] = None) -> List[EmotionalOptimizationRecommendation]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # If no target specified, determine optimal target based on current state
        if target_state is None:
            target_state = self._determine_optimal_target(current_snapshot, context)
        
        # Generate optimization strategy
        if target_state in self.optimization_strategies:
            rec = self.optimization_strategies[target_state](current_snapshot, context)
            if rec:
                recommendations.append(rec)
        
        # Generate intervention strategy if needed
        if current_snapshot.primary_state in self.intervention_strategies:
            intervention = self.intervention_strategies[current_snapshot.primary_state](current_snapshot, context)
            if intervention:
                recommendations.append(intervention)
        
        # Generate dormancy-specific recommendations
        dormancy_rec = self._generate_dormancy_recommendation(current_snapshot, context)
        if dormancy_rec:
            recommendations.append(dormancy_rec)
        
        return recommendations
    
    def _determine_optimal_target(self, 
                                current_snapshot: EmotionalStateSnapshot,
                                context: Optional[Dict[str, Any]]) -> EmotionalState:
        """Determine optimal target emotional state"""
        
        # Consider current exploration readiness
        if current_snapshot.exploration_readiness > 0.7:
            return EmotionalState.OPTIMAL_CREATIVE_TENSION
        
        # Consider creativity index
        if current_snapshot.creativity_index > 0.8:
            return EmotionalState.FLOW_STATE
        
        # Consider cognitive load
        if current_snapshot.cognitive_load < 0.3:
            return EmotionalState.CURIOSITY
        
        # Default to optimal creative tension
        return EmotionalState.OPTIMAL_CREATIVE_TENSION
    
    def _optimize_for_creative_tension(self, 
                                     current_snapshot: EmotionalStateSnapshot,
                                     context: Optional[Dict[str, Any]]) -> Optional[EmotionalOptimizationRecommendation]:
        """Optimize for achieving optimal creative tension"""
        
        current_state = current_snapshot.primary_state
        
        if current_state == EmotionalState.OPTIMAL_CREATIVE_TENSION:
            return None  # Already at target
        
        # Strategy depends on current state
        if current_state == EmotionalState.MENTAL_FATIGUE:
            strategy = "increase_novelty_gradually"
            parameters = {
                "novelty_increment": 0.1,
                "challenge_level": "moderate",
                "break_intervals": True
            }
            success_probability = 0.7
            
        elif current_state == EmotionalState.ANXIETY:
            strategy = "reduce_uncertainty_increase_control"
            parameters = {
                "provide_structure": True,
                "reduce_complexity": True,
                "increase_familiarity": 0.3
            }
            success_probability = 0.6
            
        else:
            strategy = "balanced_challenge_increase"
            parameters = {
                "challenge_level": "optimal",
                "novelty_level": 0.75,
                "support_level": "moderate"
            }
            success_probability = 0.8
        
        return EmotionalOptimizationRecommendation(
            recommendation_id="",
            target_state=EmotionalState.OPTIMAL_CREATIVE_TENSION,
            current_state=current_state,
            strategy=strategy,
            confidence=0.8,
            expected_duration=300.0,  # 5 minutes
            success_probability=success_probability,
            intervention_type="cognitive",
            parameters=parameters
        )
    
    def _optimize_for_flow(self, 
                         current_snapshot: EmotionalStateSnapshot,
                         context: Optional[Dict[str, Any]]) -> Optional[EmotionalOptimizationRecommendation]:
        """Optimize for achieving flow state"""
        
        if current_snapshot.primary_state == EmotionalState.FLOW_STATE:
            return None
        
        # Flow requires specific conditions
        strategy = "create_flow_conditions"
        parameters = {
            "challenge_skill_balance": 0.8,
            "clear_goals": True,
            "immediate_feedback": True,
            "minimize_distractions": True,
            "intrinsic_motivation": 0.9
        }
        
        # Success probability depends on current state
        if current_snapshot.primary_state == EmotionalState.OPTIMAL_CREATIVE_TENSION:
            success_probability = 0.8
        elif current_snapshot.exploration_readiness > 0.7:
            success_probability = 0.6
        else:
            success_probability = 0.4
        
        return EmotionalOptimizationRecommendation(
            recommendation_id="",
            target_state=EmotionalState.FLOW_STATE,
            current_state=current_snapshot.primary_state,
            strategy=strategy,
            confidence=0.9,
            expected_duration=600.0,  # 10 minutes to establish flow
            success_probability=success_probability,
            intervention_type="environmental",
            parameters=parameters
        )
    
    def _optimize_for_curiosity(self, 
                              current_snapshot: EmotionalStateSnapshot,
                              context: Optional[Dict[str, Any]]) -> Optional[EmotionalOptimizationRecommendation]:
        """Optimize for curiosity state"""
        
        if current_snapshot.primary_state == EmotionalState.CURIOSITY:
            return None
        
        strategy = "stimulate_curiosity"
        parameters = {
            "introduce_mystery": True,
            "partial_information": 0.6,
            "question_generation": True,
            "novelty_exposure": 0.5,
            "exploration_prompts": True
        }
        
        return EmotionalOptimizationRecommendation(
            recommendation_id="",
            target_state=EmotionalState.CURIOSITY,
            current_state=current_snapshot.primary_state,
            strategy=strategy,
            confidence=0.7,
            expected_duration=180.0,  # 3 minutes
            success_probability=0.8,
            intervention_type="exploration",
            parameters=parameters
        )
    
    def _optimize_for_breakthrough(self, 
                                 current_snapshot: EmotionalStateSnapshot,
                                 context: Optional[Dict[str, Any]]) -> Optional[EmotionalOptimizationRecommendation]:
        """Optimize for creative breakthrough"""
        
        # Breakthroughs can't be forced, but conditions can be optimized
        strategy = "create_breakthrough_conditions"
        parameters = {
            "incubation_period": True,
            "cross_domain_connections": True,
            "relaxed_attention": 0.7,
            "playful_exploration": True,
            "constraint_relaxation": 0.8
        }
        
        return EmotionalOptimizationRecommendation(
            recommendation_id="",
            target_state=EmotionalState.CREATIVE_BREAKTHROUGH,
            current_state=current_snapshot.primary_state,
            strategy=strategy,
            confidence=0.6,  # Lower confidence since breakthroughs are unpredictable
            expected_duration=900.0,  # 15 minutes
            success_probability=0.3,  # Breakthroughs are rare
            intervention_type="dormancy",
            parameters=parameters
        )
    
    def _intervene_confusion(self, 
                           current_snapshot: EmotionalStateSnapshot,
                           context: Optional[Dict[str, Any]]) -> Optional[EmotionalOptimizationRecommendation]:
        """Intervene in counterproductive confusion"""
        
        strategy = "reduce_confusion"
        parameters = {
            "simplify_complexity": True,
            "provide_structure": True,
            "clarify_goals": True,
            "reduce_options": 0.5,
            "step_by_step_guidance": True
        }
        
        return EmotionalOptimizationRecommendation(
            recommendation_id="",
            target_state=EmotionalState.OPTIMAL_CREATIVE_TENSION,
            current_state=current_snapshot.primary_state,
            strategy=strategy,
            confidence=0.8,
            expected_duration=240.0,  # 4 minutes
            success_probability=0.7,
            intervention_type="cognitive",
            parameters=parameters
        )
    
    def _intervene_fatigue(self, 
                         current_snapshot: EmotionalStateSnapshot,
                         context: Optional[Dict[str, Any]]) -> Optional[EmotionalOptimizationRecommendation]:
        """Intervene in mental fatigue"""
        
        strategy = "restore_energy"
        parameters = {
            "rest_period": 300.0,  # 5 minute break
            "reduce_cognitive_load": True,
            "switch_modalities": True,
            "gentle_stimulation": 0.3,
            "dormancy_mode": DormancyMode.MEMORY_WEAVING.name
        }
        
        return EmotionalOptimizationRecommendation(
            recommendation_id="",
            target_state=EmotionalState.CURIOSITY,
            current_state=current_snapshot.primary_state,
            strategy=strategy,
            confidence=0.9,
            expected_duration=480.0,  # 8 minutes including rest
            success_probability=0.8,
            intervention_type="dormancy",
            parameters=parameters
        )
    
    def _intervene_frustration(self, 
                             current_snapshot: EmotionalStateSnapshot,
                             context: Optional[Dict[str, Any]]) -> Optional[EmotionalOptimizationRecommendation]:
        """Intervene in frustration"""
        
        strategy = "reduce_frustration"
        parameters = {
            "change_approach": True,
            "lower_standards_temporarily": 0.3,
            "celebrate_small_wins": True,
            "reframe_obstacles": True,
            "seek_alternative_paths": True
        }
        
        return EmotionalOptimizationRecommendation(
            recommendation_id="",
            target_state=EmotionalState.CURIOSITY,
            current_state=current_snapshot.primary_state,
            strategy=strategy,
            confidence=0.7,
            expected_duration=360.0,  # 6 minutes
            success_probability=0.6,
            intervention_type="cognitive",
            parameters=parameters
        )
    
    def _intervene_anxiety(self, 
                         current_snapshot: EmotionalStateSnapshot,
                         context: Optional[Dict[str, Any]]) -> Optional[EmotionalOptimizationRecommendation]:
        """Intervene in anxiety"""
        
        strategy = "reduce_anxiety"
        parameters = {
            "increase_predictability": True,
            "provide_control_options": True,
            "reduce_uncertainty": 0.7,
            "grounding_techniques": True,
            "safety_assurance": True
        }
        
        return EmotionalOptimizationRecommendation(
            recommendation_id="",
            target_state=EmotionalState.SATISFACTION,
            current_state=current_snapshot.primary_state,
            strategy=strategy,
            confidence=0.8,
            expected_duration=420.0,  # 7 minutes
            success_probability=0.7,
            intervention_type="environmental",
            parameters=parameters
        )
    
    def _generate_dormancy_recommendation(self, 
                                        current_snapshot: EmotionalStateSnapshot,
                                        context: Optional[Dict[str, Any]]) -> Optional[EmotionalOptimizationRecommendation]:
        """Generate dormancy-specific recommendations"""
        
        # Recommend dormancy mode based on emotional state
        if current_snapshot.primary_state == EmotionalState.MENTAL_FATIGUE:
            dormancy_mode = DormancyMode.MEMORY_WEAVING
            
        elif current_snapshot.primary_state == EmotionalState.COUNTERPRODUCTIVE_CONFUSION:
            dormancy_mode = DormancyMode.PATTERN_CRYSTALLIZATION
            
        elif current_snapshot.primary_state == EmotionalState.FRUSTRATION:
            dormancy_mode = DormancyMode.DREAM_SYNTHESIS
            
        elif current_snapshot.creativity_index > 0.8:
            dormancy_mode = DormancyMode.CONCEPT_GARDENING
            
        elif current_snapshot.exploration_readiness > 0.8:
            dormancy_mode = DormancyMode.XENOMORPHIC_BECOMING
            
        else:
            return None  # No specific dormancy recommendation
        
        strategy = f"enter_dormancy_{dormancy_mode.name.lower()}"
        parameters = {
            "dormancy_mode": dormancy_mode.name,
            "duration": "adaptive",
            "auto_exit_conditions": {
                "energy_threshold": 0.7,
                "novelty_threshold": 0.6,
                "time_limit": 600.0  # 10 minutes max
            }
        }
        
        return EmotionalOptimizationRecommendation(
            recommendation_id="",
            target_state=EmotionalState.CURIOSITY,  # Expected state after dormancy
            current_state=current_snapshot.primary_state,
            strategy=strategy,
            confidence=0.8,
            expected_duration=300.0,  # 5 minutes average
            success_probability=0.8,
            intervention_type="dormancy",
            parameters=parameters
        )


class EmotionalStateMonitoringSystem:
    """Main emotional state monitoring system integrating with enhanced dormancy"""
    
    def __init__(self, 
                 enhanced_dormancy: EnhancedDormantPhaseLearningSystem,
                 consciousness_modules: Optional[Dict[str, Any]] = None):
        
        self.enhanced_dormancy = enhanced_dormancy
        self.consciousness_modules = consciousness_modules or {}
        
        # Initialize components
        self.state_detector = EmotionalStateDetector(self.consciousness_modules)
        self.pattern_analyzer = EmotionalPatternAnalyzer()
        self.optimizer = EmotionalOptimizer(enhanced_dormancy.version_control)
        
        # Monitoring state
        self.is_monitoring = False
        self.current_snapshot: Optional[EmotionalStateSnapshot] = None
        self.snapshot_history: deque = deque(maxlen=1000)
        self.active_recommendations: List[EmotionalOptimizationRecommendation] = []
        
        # Integration with enhanced dormancy
        self._setup_dormancy_integration()
        
        # Background monitoring
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.monitoring_tasks: List[Any] = []
        
        logger.info("Emotional State Monitoring System initialized and integrated")
    
    def _setup_dormancy_integration(self):
        """Setup integration with enhanced dormancy system"""
        
        # Register emotional state monitoring callbacks
        def on_dormant_phase_start():
            """Handle dormancy phase start"""
            self._on_dormancy_entry()
        
        def on_active_phase_start():
            """Handle active phase start"""
            self._on_dormancy_exit()
        
        def on_latent_traversal(traversal_data):
            """Handle latent space traversal events"""
            self._on_exploration_event("latent_traversal", traversal_data)
        
        def on_module_collision(collision_result):
            """Handle module collision events"""
            self._on_exploration_event("module_collision", collision_result)
        
        # Register callbacks with enhanced dormancy system
        self.enhanced_dormancy.register_integration_callback("dormant_phase_start", on_dormant_phase_start)
        self.enhanced_dormancy.register_integration_callback("active_phase_start", on_active_phase_start)
        self.enhanced_dormancy.register_integration_callback("latent_traversal", on_latent_traversal)
        self.enhanced_dormancy.register_integration_callback("module_collision", on_module_collision)
    
    def start_monitoring(self):
        """Start emotional state monitoring"""
        
        if self.is_monitoring:
            logger.warning("Emotional monitoring already active")
            return
        
        self.is_monitoring = True
        logger.info("Starting emotional state monitoring")
        
        # Start background monitoring loop
        self.monitoring_tasks.append(
            self.executor.submit(self._monitoring_loop)
        )
        
        # Start pattern analysis loop
        self.monitoring_tasks.append(
            self.executor.submit(self._pattern_analysis_loop)
        )
    
    def stop_monitoring(self):
        """Stop emotional state monitoring"""
        
        self.is_monitoring = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if hasattr(task, 'cancel'):
                task.cancel()
        
        self.executor.shutdown(wait=True)
        logger.info("Emotional state monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.is_monitoring:
            try:
                # Gather current cognitive and exploration metrics
                cognitive_metrics = self._gather_cognitive_metrics()
                exploration_data = self._gather_exploration_data()
                consciousness_state = self._gather_consciousness_state()
                
                # Detect emotional state
                snapshot = self.state_detector.detect_emotional_state(
                    cognitive_metrics, exploration_data, consciousness_state
                )
                
                # Add version control context
                snapshot.commit_id = self.enhanced_dormancy.version_control.head_commits.get(
                    self.enhanced_dormancy.version_control.current_branch
                )
                snapshot.branch_name = self.enhanced_dormancy.version_control.current_branch
                
                # Store snapshot
                self.current_snapshot = snapshot
                self.snapshot_history.append(snapshot)
                
                # Generate recommendations if needed
                self._update_recommendations(snapshot)
                
                # Auto-commit emotional state if significant
                if snapshot.intensity > 0.8 or snapshot.primary_state == EmotionalState.CREATIVE_BREAKTHROUGH:
                    self._commit_emotional_state(snapshot)
                
                time.sleep(2.0)  # Monitor every 2 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)
    
    def _pattern_analysis_loop(self):
        """Pattern analysis loop"""
        
        while self.is_monitoring:
            try:
                if len(self.snapshot_history) >= 10:
                    # Analyze patterns from recent snapshots
                    recent_snapshots = list(self.snapshot_history)[-20:]
                    patterns = self.pattern_analyzer.analyze_patterns(recent_snapshots)
                    
                    # Log significant patterns
                    for pattern in patterns:
                        if pattern.effectiveness_score > 0.8:
                            logger.info(f"Identified high-value emotional pattern: {pattern.pattern_type}")
                        elif pattern.effectiveness_score < 0.3:
                            logger.warning(f"Identified problematic emotional pattern: {pattern.pattern_type}")
                
                time.sleep(30.0)  # Analyze patterns every 30 seconds
                
            except Exception as e:
                logger.error(f"Pattern analysis error: {e}")
                time.sleep(60.0)
    
    def _gather_cognitive_metrics(self) -> Dict[str, float]:
        """Gather current cognitive metrics"""
        
        # This would integrate with actual cognitive systems
        # For now, providing simulated metrics
        base_metrics = {
            "processing_speed": random.uniform(0.3, 1.0),
            "attention_span": random.uniform(0.4, 1.0),
            "working_memory_usage": random.uniform(0.2, 0.9),
            "error_rate": random.uniform(0.0, 0.3),
            "confusion": random.uniform(0.0, 0.8),
            "engagement": random.uniform(0.2, 1.0),
            "challenge": random.uniform(0.1, 0.9),
            "coherence": random.uniform(0.3, 1.0),
            "uncertainty": random.uniform(0.0, 0.7),
            "anticipation": random.uniform(0.0, 0.8)
        }
        
        # Add some correlation based on current dormancy state
        if hasattr(self.enhanced_dormancy, 'meta_controller'):
            current_phase = self.enhanced_dormancy.meta_controller.current_phase
            if current_phase == PhaseState.DORMANT:
                base_metrics["processing_speed"] *= 0.7
                base_metrics["attention_span"] *= 0.6
                base_metrics["confusion"] *= 0.5  # Less confusion in dormancy
            elif current_phase == PhaseState.DEEP_SIMULATION:
                base_metrics["engagement"] *= 1.3
                base_metrics["challenge"] *= 1.2
        
        return base_metrics
    
    def _gather_exploration_data(self) -> Dict[str, Any]:
        """Gather current exploration data"""
        
        exploration_data = {
            "novelty": random.uniform(0.0, 1.0),
            "complexity": random.uniform(0.1, 0.9),
            "progress": random.uniform(0.0, 1.0),
            "progress_rate": random.uniform(0.0, 0.8),
            "session_duration": time.time() % 3600,  # Simulated session time
            "obstacles": random.randint(0, 5),
            "discovery_rate": random.uniform(0.0, 0.6),
            "pattern_interest": random.uniform(0.0, 0.8)
        }
        
        # Add context from enhanced dormancy system
        if hasattr(self.enhanced_dormancy, 'latent_explorer'):
            exploration_data["latent_representations_count"] = len(
                self.enhanced_dormancy.latent_explorer.latent_representations
            )
            
            # Get recent exploration history
            if self.enhanced_dormancy.latent_explorer.exploration_history:
                recent_exploration = list(self.enhanced_dormancy.latent_explorer.exploration_history)[-1]
                exploration_data["recent_novelty"] = recent_exploration.get("max_novelty", 0.0)
        
        return exploration_data
    
    def _gather_consciousness_state(self) -> Dict[str, Any]:
        """Gather current consciousness state"""
        
        consciousness_state = {
            "active_modules": [],
            "integration_level": random.uniform(0.5, 1.0),
            "module_collision": random.random() < 0.1,  # 10% chance
            "state_transition": random.random() < 0.05  # 5% chance
        }
        
        # Add actual consciousness module states if available
        for module_name, module in self.consciousness_modules.items():
            consciousness_state["active_modules"].append(module_name)
            # Would add actual module state information here
        
        return consciousness_state
    
    def _update_recommendations(self, snapshot: EmotionalStateSnapshot):
        """Update active recommendations based on current emotional state"""
        
        # Clear expired recommendations
        current_time = time.time()
        self.active_recommendations = [
            rec for rec in self.active_recommendations 
            if current_time - rec.expected_duration < 300  # 5 minute expiry
        ]
        
        # Generate new recommendations if state needs optimization
        needs_optimization = (
            snapshot.intensity > 0.8 and snapshot.primary_state in [
                EmotionalState.COUNTERPRODUCTIVE_CONFUSION,
                EmotionalState.MENTAL_FATIGUE,
                EmotionalState.FRUSTRATION,
                EmotionalState.ANXIETY
            ]
        ) or (
            snapshot.exploration_readiness < 0.3
        ) or (
            snapshot.creativity_index < 0.3
        )
        
        if needs_optimization:
            new_recommendations = self.optimizer.generate_recommendations(
                snapshot, context={"monitoring_active": True}
            )
            self.active_recommendations.extend(new_recommendations)
            
            # Log recommendations
            for rec in new_recommendations:
                logger.info(f"Generated emotional optimization recommendation: {rec.strategy}")
    
    def _commit_emotional_state(self, snapshot: EmotionalStateSnapshot):
        """Commit significant emotional state to version control"""
        
        cognitive_state = {
            "emotional_state": {
                "primary_state": snapshot.primary_state.name,
                "intensity": snapshot.intensity,
                "secondary_states": [(state.name, intensity) for state, intensity in snapshot.secondary_states],
                "trigger": snapshot.trigger.name if snapshot.trigger else None
            },
            "emotional_metrics": {
                "cognitive_load": snapshot.cognitive_load,
                "creativity_index": snapshot.creativity_index,
                "exploration_readiness": snapshot.exploration_readiness
            },
            "physiological_indicators": snapshot.physiological_indicators
        }
        
        exploration_data = {
            "emotional_context": snapshot.context,
            "state_snapshot_id": snapshot.id,
            "monitoring_timestamp": snapshot.timestamp
        }
        
        commit_id = self.enhanced_dormancy.version_control.commit(
            cognitive_state=cognitive_state,
            exploration_data=exploration_data,
            message=f"Emotional state: {snapshot.primary_state.name} (intensity: {snapshot.intensity:.3f})",
            author_module="emotional_state_monitor",
            dormancy_mode=DormancyMode.PATTERN_CRYSTALLIZATION,
            novelty_score=snapshot.creativity_index
        )
        
        # Update snapshot with commit information
        snapshot.commit_id = commit_id
        
        logger.info(f"Committed emotional state {snapshot.primary_state.name} to version control")
    
    def _on_dormancy_entry(self):
        """Handle dormancy entry event"""
        
        if self.current_snapshot:
            # Record dormancy entry trigger
            entry_snapshot = EmotionalStateSnapshot(
                timestamp=time.time(),
                primary_state=self.current_snapshot.primary_state,
                intensity=self.current_snapshot.intensity * 0.8,  # Slight reduction
                secondary_states=self.current_snapshot.secondary_states,
                trigger=EmotionalTrigger.DORMANCY_ENTRY,
                context={"event": "dormancy_entry"},
                physiological_indicators=self.current_snapshot.physiological_indicators,
                cognitive_load=self.current_snapshot.cognitive_load * 0.6,
                creativity_index=self.current_snapshot.creativity_index,
                exploration_readiness=self.current_snapshot.exploration_readiness * 0.5
            )
            
            self.snapshot_history.append(entry_snapshot)
            logger.info("Recorded emotional state for dormancy entry")
    
    def _on_dormancy_exit(self):
        """Handle dormancy exit event"""
        
        if self.current_snapshot:
            # Record dormancy exit - typically should show restoration
            exit_snapshot = EmotionalStateSnapshot(
                timestamp=time.time(),
                primary_state=EmotionalState.CURIOSITY,  # Expected post-dormancy state
                intensity=0.6,
                secondary_states=[(EmotionalState.SATISFACTION, 0.4)],
                trigger=EmotionalTrigger.DORMANCY_EXIT,
                context={"event": "dormancy_exit"},
                physiological_indicators={
                    "arousal_level": 0.5,
                    "stress_level": 0.2,
                    "energy_level": 0.8,  # Restored energy
                    "cognitive_temperature": 0.7,
                    "attention_stability": 0.8,
                    "emotional_regulation": 0.9
                },
                cognitive_load=0.3,
                creativity_index=0.7,
                exploration_readiness=0.8
            )
            
            self.snapshot_history.append(exit_snapshot)
            logger.info("Recorded emotional state for dormancy exit")
    
    def _on_exploration_event(self, event_type: str, event_data: Any):
        """Handle exploration events"""
        
        if not self.current_snapshot:
            return
        
        # Determine emotional impact of exploration event
        if event_type == "latent_traversal":
            if hasattr(event_data, 'max_novelty') and event_data.max_novelty > 0.8:
                trigger = EmotionalTrigger.NOVELTY_DISCOVERY
                intensity_boost = 0.2
            else:
                trigger = EmotionalTrigger.PATTERN_RECOGNITION
                intensity_boost = 0.1
                
        elif event_type == "module_collision":
            trigger = EmotionalTrigger.MODULE_COLLISION
            if hasattr(event_data, 'synthesis_quality') and event_data.synthesis_quality > 0.8:
                intensity_boost = 0.3
            else:
                intensity_boost = 0.1
        else:
            return
        
        # Create event snapshot
        event_snapshot = EmotionalStateSnapshot(
            timestamp=time.time(),
            primary_state=self.current_snapshot.primary_state,
            intensity=min(1.0, self.current_snapshot.intensity + intensity_boost),
            secondary_states=self.current_snapshot.secondary_states,
            trigger=trigger,
            context={"event": event_type, "event_data_type": type(event_data).__name__},
            physiological_indicators=self.current_snapshot.physiological_indicators,
            cognitive_load=self.current_snapshot.cognitive_load,
            creativity_index=min(1.0, self.current_snapshot.creativity_index + intensity_boost),
            exploration_readiness=self.current_snapshot.exploration_readiness
        )
        
        self.snapshot_history.append(event_snapshot)
    
    def get_current_emotional_state(self) -> Optional[EmotionalStateSnapshot]:
        """Get current emotional state"""
        return self.current_snapshot
    
    def get_emotional_trends(self, timeframe_minutes: int = 30) -> Dict[str, Any]:
        """Get emotional trends over specified timeframe"""
        
        cutoff_time = time.time() - (timeframe_minutes * 60)
        recent_snapshots = [
            snapshot for snapshot in self.snapshot_history 
            if snapshot.timestamp > cutoff_time
        ]
        
        if not recent_snapshots:
            return {"status": "insufficient_data"}
        
        # Analyze trends
        state_frequencies = defaultdict(int)
        intensity_values = []
        creativity_values = []
        readiness_values = []
        
        for snapshot in recent_snapshots:
            state_frequencies[snapshot.primary_state.name] += 1
            intensity_values.append(snapshot.intensity)
            creativity_values.append(snapshot.creativity_index)
            readiness_values.append(snapshot.exploration_readiness)
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return "stable"
            recent_avg = sum(values[-5:]) / len(values[-5:])
            earlier_avg = sum(values[:5]) / len(values[:5]) if len(values) >= 10 else sum(values[:-5]) / len(values[:-5])
            if recent_avg > earlier_avg + 0.1:
                return "increasing"
            elif recent_avg < earlier_avg - 0.1:
                return "decreasing"
            else:
                return "stable"
        
        return {
            "timeframe_minutes": timeframe_minutes,
            "snapshot_count": len(recent_snapshots),
            "dominant_state": max(state_frequencies.items(), key=lambda x: x[1])[0] if state_frequencies else None,
            "state_distribution": dict(state_frequencies),
            "average_intensity": sum(intensity_values) / len(intensity_values),
            "intensity_trend": calculate_trend(intensity_values),
            "average_creativity": sum(creativity_values) / len(creativity_values),
            "creativity_trend": calculate_trend(creativity_values),
            "average_readiness": sum(readiness_values) / len(readiness_values),
            "readiness_trend": calculate_trend(readiness_values)
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations"""
        
        return [
            {
                "recommendation_id": rec.recommendation_id,
                "target_state": rec.target_state.name,
                "current_state": rec.current_state.name,
                "strategy": rec.strategy,
                "confidence": rec.confidence,
                "success_probability": rec.success_probability,
                "intervention_type": rec.intervention_type,
                "parameters": rec.parameters
            }
            for rec in self.active_recommendations
        ]
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights from emotional pattern analysis"""
        
        pattern_summary = self.pattern_analyzer.get_pattern_summary()
        
        # Add recent pattern activity
        recent_patterns = [
            pattern for pattern in self.pattern_analyzer.identified_patterns.values()
            if time.time() - pattern.last_occurrence < 1800  # Last 30 minutes
        ]
        
        pattern_summary["recent_pattern_count"] = len(recent_patterns)
        pattern_summary["recent_patterns"] = [
            {
                "pattern_type": pattern.pattern_type,
                "effectiveness_score": pattern.effectiveness_score,
                "last_occurrence": pattern.last_occurrence,
                "context_tags": pattern.context_tags
            }
            for pattern in recent_patterns
        ]
        
        return pattern_summary
    
    def export_emotional_data(self) -> Dict[str, Any]:
        """Export emotional monitoring data"""
        
        return {
            "timestamp": time.time(),
            "monitoring_active": self.is_monitoring,
            "current_snapshot": asdict(self.current_snapshot) if self.current_snapshot else None,
            "snapshot_history": [
                {
                    "id": snapshot.id,
                    "timestamp": snapshot.timestamp,
                    "primary_state": snapshot.primary_state.name,
                    "intensity": snapshot.intensity,
                    "secondary_states": [(state.name, intensity) for state, intensity in snapshot.secondary_states],
                    "trigger": snapshot.trigger.name if snapshot.trigger else None,
                    "cognitive_load": snapshot.cognitive_load,
                    "creativity_index": snapshot.creativity_index,
                    "exploration_readiness": snapshot.exploration_readiness,
                    "commit_id": snapshot.commit_id,
                    "branch_name": snapshot.branch_name
                }
                for snapshot in list(self.snapshot_history)[-100:]  # Last 100 snapshots
            ],
            "identified_patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "effectiveness_score": pattern.effectiveness_score,
                    "frequency": pattern.frequency,
                    "last_occurrence": pattern.last_occurrence,
                    "context_tags": pattern.context_tags
                }
                for pattern in self.pattern_analyzer.identified_patterns.values()
            ],
            "active_recommendations": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "target_state": rec.target_state.name,
                    "strategy": rec.strategy,
                    "confidence": rec.confidence,
                    "intervention_type": rec.intervention_type
                }
                for rec in self.active_recommendations
            ],
            "emotional_trends": self.get_emotional_trends(60),  # Last hour
            "pattern_insights": self.get_pattern_insights()
        }
    
    def import_emotional_data(self, data: Dict[str, Any]) -> bool:
        """Import emotional monitoring data"""
        
        try:
            # Import snapshot history
            if "snapshot_history" in data:
                for snapshot_data in data["snapshot_history"]:
                    snapshot = EmotionalStateSnapshot(
                        timestamp=snapshot_data["timestamp"],
                        primary_state=EmotionalState[snapshot_data["primary_state"]],
                        intensity=snapshot_data["intensity"],
                        secondary_states=[(EmotionalState[state], intensity) 
                                        for state, intensity in snapshot_data["secondary_states"]],
                        trigger=EmotionalTrigger[snapshot_data["trigger"]] if snapshot_data["trigger"] else None,
                        context={},
                        physiological_indicators={},
                        cognitive_load=snapshot_data["cognitive_load"],
                        creativity_index=snapshot_data["creativity_index"],
                        exploration_readiness=snapshot_data["exploration_readiness"],
                        commit_id=snapshot_data.get("commit_id"),
                        branch_name=snapshot_data.get("branch_name")
                    )
                    snapshot.id = snapshot_data["id"]
                    self.snapshot_history.append(snapshot)
            
            # Import identified patterns
            if "identified_patterns" in data:
                for pattern_data in data["identified_patterns"]:
                    pattern = EmotionalPattern(
                        pattern_id=pattern_data["pattern_id"],
                        pattern_type=pattern_data["pattern_type"],
                        states_sequence=[],  # Would need full data for complete restoration
                        average_duration=0.0,
                        trigger_conditions=[],
                        effectiveness_score=pattern_data["effectiveness_score"],
                        frequency=pattern_data["frequency"],
                        last_occurrence=pattern_data["last_occurrence"],
                        context_tags=pattern_data["context_tags"]
                    )
                    self.pattern_analyzer.identified_patterns[pattern.pattern_id] = pattern
            
            logger.info("Successfully imported emotional monitoring data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import emotional monitoring data: {e}")
            traceback.print_exc()
            return False


# Integration function for enhanced dormancy system
def integrate_emotional_monitoring(enhanced_dormancy: EnhancedDormantPhaseLearningSystem,
                                 consciousness_modules: Dict[str, Any]) -> EmotionalStateMonitoringSystem:
    """Integrate emotional state monitoring with enhanced dormancy system"""
    
    # Create emotional monitoring system
    emotional_monitor = EmotionalStateMonitoringSystem(enhanced_dormancy, consciousness_modules)
    
    # Start monitoring
    emotional_monitor.start_monitoring()
    
    logger.info("Emotional State Monitoring integrated with Enhanced Dormancy System")
    
    return emotional_monitor


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Demonstrate emotional state monitoring system"""
        print("🧠 Emotional State Monitoring System Demo")
        print("=" * 50)
        
        # This would normally be imported from the enhanced dormancy system
        # For demo purposes, we'll create a mock system
        class MockEnhancedDormancy:
            def __init__(self):
                self.version_control = type('MockVC', (), {
                    'current_branch': 'main',
                    'head_commits': {'main': 'mock_commit_123'},
                    'commit': lambda *args, **kwargs: 'mock_commit_' + str(time.time())
                })()
                self.meta_controller = type('MockMeta', (), {
                    'current_phase': PhaseState.ACTIVE
                })()
                self.latent_explorer = type('MockExplorer', (), {
                    'latent_representations': {},
                    'exploration_history': deque()
                })()
            
            def register_integration_callback(self, event_type, callback):
                pass
        
        # Create mock consciousness modules
        consciousness_modules = {
            "consciousness_core": {"active": True},
            "deluzian_consciousness": {"active": True},
            "phase4_consciousness": {"active": True}
        }
        
        # Initialize systems
        mock_enhanced_dormancy = MockEnhancedDormancy()
        emotional_monitor = EmotionalStateMonitoringSystem(mock_enhanced_dormancy, consciousness_modules)
        
        print("✅ Emotional monitoring system initialized")
        
        # Start monitoring
        emotional_monitor.start_monitoring()
        print("📊 Monitoring started")
        
        # Simulate monitoring for a period
        await asyncio.sleep(10.0)
        
        # Get current emotional state
        current_state = emotional_monitor.get_current_emotional_state()
        if current_state:
            print(f"\n🎭 Current Emotional State:")
            print(f"  Primary: {current_state.primary_state.name}")
            print(f"  Intensity: {current_state.intensity:.3f}")
            print(f"  Creativity Index: {current_state.creativity_index:.3f}")
            print(f"  Exploration Readiness: {current_state.exploration_readiness:.3f}")
        
        # Get emotional trends
        trends = emotional_monitor.get_emotional_trends(5)  # Last 5 minutes
        print(f"\n📈 Emotional Trends (5 min):")
        print(f"  Snapshots: {trends.get('snapshot_count', 0)}")
        print(f"  Dominant State: {trends.get('dominant_state', 'None')}")
        print(f"  Intensity Trend: {trends.get('intensity_trend', 'Unknown')}")
        print(f"  Creativity Trend: {trends.get('creativity_trend', 'Unknown')}")
        
        # Get recommendations
        recommendations = emotional_monitor.get_optimization_recommendations()
        print(f"\n💡 Active Recommendations: {len(recommendations)}")
        for rec in recommendations[:3]:  # Show first 3
            print(f"  • {rec['strategy']} (confidence: {rec['confidence']:.3f})")
        
        # Get pattern insights
        patterns = emotional_monitor.get_pattern_insights()
        print(f"\n🔍 Pattern Insights:")
        print(f"  Total Patterns: {patterns.get('total_patterns', 0)}")
        print(f"  Recent Patterns: {patterns.get('recent_pattern_count', 0)}")
        
        # Export data
        export_data = emotional_monitor.export_emotional_data()
        print(f"\n💾 Export Data:")
        print(f"  Snapshot History: {len(export_data['snapshot_history'])} items")
        print(f"  Identified Patterns: {len(export_data['identified_patterns'])} patterns")
        
        # Stop monitoring
        emotional_monitor.stop_monitoring()
        print("\n✅ Emotional monitoring system demo completed!")
    
    # Run the demo
    asyncio.run(main())
    branch_name: Optional[str] = None
    
    def __post_init__(self):
        if not hasattr(self, 'id'):
            self.id = str(uuid.uuid4())


@dataclass
class EmotionalPattern:
    """Represents patterns in emotional state changes"""
    pattern_id: str
    pattern_type: str  # "cycle", "trigger_response", "state_sequence", "intensity_wave"
    states_sequence: List[EmotionalState]
    average_duration: float
    trigger_conditions: List[EmotionalTrigger]
    effectiveness_score: float  # How beneficial this pattern is
    frequency: int
    last_occurrence: float
    context_tags: List[str]
    
    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = str(uuid.uuid4())


@dataclass
class EmotionalOptimizationRecommendation:
    """Recommendations for optimizing emotional states"""
    recommendation_id: str
    target_state: EmotionalState
    current_state: EmotionalState
    strategy: str
    confidence: float
    expected_duration: float
    success_probability: float
    intervention_type: str  # "environmental", "cognitive", "exploration", "dormancy"
    parameters: Dict[str, Any]
    
    def __post_init__(self):
        if not self.recommendation_id:
            self.recommendation_id = str(uuid.uuid4())


class EmotionalStateDetector:
    """Advanced emotional state detection and analysis"""
    
    def __init__(self, consciousness_modules: Dict[str, Any]):
        self.consciousness_modules = consciousness_modules
        self.detection_algorithms = {
            EmotionalState.OPTIMAL_CREATIVE_TENSION: self._detect_optimal_tension,
            EmotionalState.COUNTERPRODUCTIVE_CONFUSION: self._detect_confusion,
            EmotionalState.MENTAL_FATIGUE: self._detect_fatigue,
            EmotionalState.FRUSTRATION: self._detect_frustration,
            EmotionalState.ANXIETY: self._detect_anxiety,
            EmotionalState.EXCITEMENT: self._detect_excitement,
            EmotionalState.CURIOSITY: self._detect_curiosity,
            EmotionalState.FLOW_STATE: self._detect_flow_state,
            EmotionalState.CREATIVE_BREAKTHROUGH: self._detect_breakthrough
        }
        
        # Calibration parameters for detection sensitivity
        self.detection_thresholds = {
            "confusion_threshold": 0.6,
            "fatigue_threshold": 0.7,
            "frustration_threshold": 0.5,
            "anxiety_threshold": 0.4,
            "flow_threshold": 0.8,
            "breakthrough_threshold": 0.9
        }
        
        # Historical data for pattern recognition
        self.detection_history: deque = deque(maxlen=1000)
        
    def detect_emotional_state(self, 
                             cognitive_metrics: Dict[str, float],
                             exploration_data: Dict[str, Any],
                             consciousness_state: Dict[str, Any]) -> EmotionalStateSnapshot:
        """Detect current emotional state using multi-source analysis"""
        
        # Run all detection algorithms
        state_scores = {}
        for state, detector in self.detection_algorithms.items():
            try:
                score = detector(cognitive_metrics, exploration_data, consciousness_state)
                state_scores[state] = max(0.0, min(1.0, score))
            except Exception as e:
                logger.warning(f"Detection error for {state.name}: {e}")
                state_scores[state] = 0.0
        
        # Determine primary and secondary states
        primary_state = max(state_scores.items(), key=lambda x: x[1])
        secondary_states = [(state, score) for state, score in state_scores.items() 
                          if state != primary_state[0] and score > 0.3]
        secondary_states.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate overall intensity
        intensity = primary_state[1]
        
        # Detect trigger
        trigger = self._identify_trigger(cognitive_metrics, exploration_data, consciousness_state)
        
        # Calculate derived metrics
        physiological_indicators = self._calculate_physiological_indicators(
            cognitive_metrics, state_scores
        )
        cognitive_load = self._calculate_cognitive_load(cognitive_metrics, exploration_data)
        creativity_index = self._calculate_creativity_index(state_scores, cognitive_metrics)
        exploration_readiness = self._calculate_exploration_readiness(state_scores, cognitive_load)
        
        # Create snapshot
        snapshot = EmotionalStateSnapshot(
            timestamp=time.time(),
            primary_state=primary_state[0],
            intensity=intensity,
            secondary_states=secondary_states[:3],  # Top 3 secondary states
            trigger=trigger,
            context={
                "cognitive_metrics": cognitive_metrics,
                "exploration_summary": {
                    "novelty_level": exploration_data.get("novelty", 0.0),
                    "complexity": exploration_data.get("complexity", 0.0),
                    "progress": exploration_data.get("progress", 0.0)
                },
                "consciousness_summary": {
                    "active_modules": consciousness_state.get("active_modules", []),
                    "integration_level": consciousness_state.get("integration_level", 0.5)
                }
            },
            physiological_indicators=physiological_indicators,
            cognitive_load=cognitive_load,
            creativity_index=creativity_index,
            exploration_readiness=exploration_readiness
        )
        
        # Store in history
        self.detection_history.append(snapshot)
        
        return snapshot
    
    def _detect_optimal_tension(self, cognitive_metrics: Dict[str, float], 
                               exploration_data: Dict[str, Any], 
                               consciousness_state: Dict[str, Any]) -> float:
        """Detect optimal creative tension state"""
        
        # Indicators of optimal tension
        novelty_level = exploration_data.get("novelty", 0.0)
        challenge_level = cognitive_metrics.get("challenge", 0.0)
        engagement_level = cognitive_metrics.get("engagement", 0.0)
        confusion_level = cognitive_metrics.get("confusion", 0.0)
        
        # Sweet spot: high novelty and challenge, high engagement, moderate confusion
        optimal_score = 0.0
        
        # Novelty component (0.3 weight)
        if 0.6 <= novelty_level <= 0.9:
            optimal_score += 0.3 * (1.0 - abs(novelty_level - 0.75) / 0.15)
        
        # Challenge component (0.25 weight)
        if 0.5 <= challenge_level <= 0.8:
            optimal_score += 0.25 * (1.0 - abs(challenge_level - 0.65) / 0.15)
        
        # Engagement component (0.25 weight)
        optimal_score += 0.25 * engagement_level
        
        # Confusion component (0.2 weight) - moderate is good
        if 0.3 <= confusion_level <= 0.6:
            optimal_score += 0.2 * (1.0 - abs(confusion_level - 0.45) / 0.15)
        
        return optimal_score
    
    def _detect_confusion(self, cognitive_metrics: Dict[str, float], 
                         exploration_data: Dict[str, Any], 
                         consciousness_state: Dict[str, Any]) -> float:
        """Detect counterproductive confusion state"""
        
        confusion_level = cognitive_metrics.get("confusion", 0.0)
        progress_rate = exploration_data.get("progress_rate", 0.0)
        cognitive_coherence = cognitive_metrics.get("coherence", 1.0)
        decision_paralysis = cognitive_metrics.get("decision_paralysis", 0.0)
        
        # High confusion + low progress + low coherence + high paralysis = counterproductive
        confusion_score = 0.0
        
        # High confusion is bad (0.4 weight)
        if confusion_level > self.detection_thresholds["confusion_threshold"]:
            confusion_score += 0.4 * (confusion_level - self.detection_thresholds["confusion_threshold"])
        
        # Low progress indicates confusion (0.3 weight)
        if progress_rate < 0.3:
            confusion_score += 0.3 * (0.3 - progress_rate)
        
        # Low coherence indicates confusion (0.2 weight)
        if cognitive_coherence < 0.5:
            confusion_score += 0.2 * (0.5 - cognitive_coherence)
        
        # Decision paralysis indicates confusion (0.1 weight)
        confusion_score += 0.1 * decision_paralysis
        
        return min(1.0, confusion_score)
    
    def _detect_fatigue(self, cognitive_metrics: Dict[str, float], 
                       exploration_data: Dict[str, Any], 
                       consciousness_state: Dict[str, Any]) -> float:
        """Detect mental fatigue state"""
        
        processing_speed = cognitive_metrics.get("processing_speed", 1.0)
        attention_span = cognitive_metrics.get("attention_span", 1.0)
        error_rate = cognitive_metrics.get("error_rate", 0.0)
        session_duration = exploration_data.get("session_duration", 0.0)
        
        fatigue_score = 0.0
        
        # Reduced processing speed (0.3 weight)
        if processing_speed < 0.7:
            fatigue_score += 0.3 * (0.7 - processing_speed) / 0.7
        
        # Reduced attention span (0.3 weight)
        if attention_span < 0.6:
            fatigue_score += 0.3 * (0.6 - attention_span) / 0.6
        
        # Increased error rate (0.2 weight)
        fatigue_score += 0.2 * min(1.0, error_rate)
        
        # Long session duration increases fatigue (0.2 weight)
        if session_duration > 1800:  # 30 minutes
            fatigue_score += 0.2 * min(1.0, (session_duration - 1800) / 3600)
        
        return min(1.0, fatigue_score)
    
    def _detect_frustration(self, cognitive_metrics: Dict[str, float], 
                           exploration_data: Dict[str, Any], 
                           consciousness_state: Dict[str, Any]) -> float:
        """Detect frustration state"""
        
        obstacle_encounters = exploration_data.get("obstacles", 0)
        progress_variance = exploration_data.get("progress_variance", 0.0)
        repetitive_patterns = cognitive_metrics.get("repetitive_patterns", 0.0)
        goal_achievement = cognitive_metrics.get("goal_achievement", 1.0)
        
        frustration_score = 0.0
        
        # High obstacle encounters (0.3 weight)
        if obstacle_encounters > 3:
            frustration_score += 0.3 * min(1.0, obstacle_encounters / 10)
        
        # High progress variance indicates stops/starts (0.3 weight)
        frustration_score += 0.3 * min(1.0, progress_variance)
        
        # Repetitive patterns indicate stuck behavior (0.2 weight)
        frustration_score += 0.2 * repetitive_patterns
        
        # Low goal achievement (0.2 weight)
        if goal_achievement < 0.5:
            frustration_score += 0.2 * (0.5 - goal_achievement)
        
        return min(1.0, frustration_score)
    
    def _detect_anxiety(self, cognitive_metrics: Dict[str, float], 
                       exploration_data: Dict[str, Any], 
                       consciousness_state: Dict[str, Any]) -> float:
        """Detect anxiety state"""
        
        uncertainty_level = cognitive_metrics.get("uncertainty", 0.0)
        risk_perception = cognitive_metrics.get("risk_perception", 0.0)
        control_sense = cognitive_metrics.get("sense_of_control", 1.0)
        anticipatory_worry = cognitive_metrics.get("anticipatory_worry", 0.0)
        
        anxiety_score = 0.0
        
        # High uncertainty (0.3 weight)
        anxiety_score += 0.3 * uncertainty_level
        
        # High risk perception (0.3 weight)
        anxiety_score += 0.3 * risk_perception
        
        # Low sense of control (0.2 weight)
        if control_sense < 0.5:
            anxiety_score += 0.2 * (0.5 - control_sense)
        
        # Anticipatory worry (0.2 weight)
        anxiety_score += 0.2 * anticipatory_worry
        
        return min(1.0, anxiety_score)
    
    def _detect_excitement(self, cognitive_metrics: Dict[str, float], 
                          exploration_data: Dict[str, Any], 
                          consciousness_state: Dict[str, Any]) -> float:
        """Detect excitement state"""
        
        novelty_level = exploration_data.get("novelty", 0.0)
        discovery_rate = exploration_data.get("discovery_rate", 0.0)
        anticipation_level = cognitive_metrics.get("anticipation", 0.0)
        energy_level = cognitive_metrics.get("energy", 0.5)
        
        excitement_score = 0.0
        
        # High novelty generates excitement (0.3 weight)
        excitement_score += 0.3 * novelty_level
        
        # High discovery rate (0.3 weight)
        excitement_score += 0.3 * discovery_rate
        
        # High anticipation (0.2 weight)
        excitement_score += 0.2 * anticipation_level
        
        # High energy (0.2 weight)
        excitement_score += 0.2 * max(0.0, energy_level - 0.5) * 2
        
        return min(1.0, excitement_score)
    
    def _detect_curiosity(self, cognitive_metrics: Dict[str, float], 
                         exploration_data: Dict[str, Any], 
                         consciousness_state: Dict[str, Any]) -> float:
        """Detect curiosity state"""
        
        question_generation = cognitive_metrics.get("question_generation", 0.0)
        exploration_drive = cognitive_metrics.get("exploration_drive", 0.0)
        information_seeking = cognitive_metrics.get("information_seeking", 0.0)
        pattern_interest = exploration_data.get("pattern_interest", 0.0)
        
        curiosity_score = 0.0
        
        # Question generation indicates curiosity (0.3 weight)
        curiosity_score += 0.3 * question_generation
        
        # Exploration drive (0.3 weight)
        curiosity_score += 0.3 * exploration_drive
        
        # Information seeking behavior (0.2 weight)
        curiosity_score += 0.2 * information_seeking
        
        # Interest in patterns (0.2 weight)
        curiosity_score += 0.2 * pattern_interest
        
        return min(1.0, curiosity_score)
    
    def _detect_flow_state(self, cognitive_metrics: Dict[str, float], 
                          exploration_data: Dict[str, Any], 
                          consciousness_state: Dict[str, Any]) -> float:
        """Detect flow state"""
        
        challenge_skill_balance = cognitive_metrics.get("challenge_skill_balance", 0.0)
        attention_focus = cognitive_metrics.get("attention_focus", 0.0)
        time_distortion = cognitive_metrics.get("time_distortion", 0.0)
        self_consciousness = cognitive_metrics.get("self_consciousness", 1.0)
        intrinsic_motivation = cognitive_metrics.get("intrinsic_motivation", 0.0)
        
        flow_score = 0.0
        
        # Challenge-skill balance is key to flow (0.3 weight)
        flow_score += 0.3 * challenge_skill_balance
        
        # High attention focus (0.25 weight)
        flow_score += 0.25 * attention_focus
        
        # Time distortion (0.2 weight)
        flow_score += 0.2 * time_distortion
        
        # Low self-consciousness (0.15 weight)
        if self_consciousness < 0.3:
            flow_score += 0.15 * (0.3 - self_consciousness) / 0.3
        
        # High intrinsic motivation (0.1 weight)
        flow_score += 0.1 * intrinsic_motivation
        
        return min(1.0, flow_score)
    
    def _detect_breakthrough(self, cognitive_metrics: Dict[str, float], 
                            exploration_data: Dict[str, Any], 
                            consciousness_state: Dict[str, Any]) -> float:
        """Detect creative breakthrough state"""
        
        insight_emergence = cognitive_metrics.get("insight_emergence", 0.0)
        connection_formation = cognitive_metrics.get("connection_formation", 0.0)
        novelty_level = exploration_data.get("novelty", 0.0)
        synthesis_quality = cognitive_metrics.get("synthesis_quality", 0.0)
        aha_moment_indicators = cognitive_metrics.get("aha_indicators", 0.0)
        
        breakthrough_score = 0.0
        
        # Sudden insight emergence (0.3 weight)
        breakthrough_score += 0.3 * insight_emergence
        
        # New connection formation (0.25 weight)
        breakthrough_score += 0.25 * connection_formation
        
        # High novelty (0.2 weight)
        breakthrough_score += 0.2 * novelty_level
        
        # High synthesis quality (0.15 weight)
        breakthrough_score += 0.15 * synthesis_quality
        
        # Aha moment indicators (0.1 weight)
        breakthrough_score += 0.1 * aha_moment_indicators
        
        return min(1.0, breakthrough_score)
    
    def _identify_trigger(self, cognitive_metrics: Dict[str, float], 
                         exploration_data: Dict[str, Any], 
                         consciousness_state: Dict[str, Any]) -> Optional[EmotionalTrigger]:
        """Identify what triggered the current emotional state"""
        
        # Check recent events for triggers
        if exploration_data.get("recent_discovery", False):
            return EmotionalTrigger.NOVELTY_DISCOVERY
        
        if cognitive_metrics.get("dissonance_level", 0.0) > 0.7:
            return EmotionalTrigger.COGNITIVE_DISSONANCE
        
        if exploration_data.get("pattern_recognized", False):
            return EmotionalTrigger.PATTERN_RECOGNITION
        
        if cognitive_metrics.get("synthesis_event", False):
            return EmotionalTrigger.CREATIVE_SYNTHESIS
        
        if exploration_data.get("exploration_blocked", False):
            return EmotionalTrigger.EXPLORATION_BLOCK
        
        if cognitive_metrics.get("insight_emerged", False):
            return EmotionalTrigger.INSIGHT_EMERGENCE
        
        if consciousness_state.get("module_collision", False):
            return EmotionalTrigger.MODULE_COLLISION
        
        if consciousness_state.get("state_transition", False):
            return EmotionalTrigger.CONSCIOUSNESS_TRANSITION
        
        # Check for dormancy-related triggers
        if exploration_data.get("entering_dormancy", False):
            return EmotionalTrigger.DORMANCY_ENTRY
        
        if exploration_data.get("exiting_dormancy", False):
            return EmotionalTrigger.DORMANCY_EXIT
        
        return None
    
    def _calculate_physiological_indicators(self, 
                                           cognitive_metrics: Dict[str, float],
                                           state_scores: Dict[EmotionalState, float]) -> Dict[str, float]:
        """Calculate simulated physiological indicators"""
        
        # Simulate physiological responses based on emotional states
        arousal_level = 0.0
        stress_level = 0.0
        energy_level = 0.5
        
        # Calculate arousal from high-intensity states
        arousal_contributing_states = [
            EmotionalState.EXCITEMENT, EmotionalState.ANXIETY, 
            EmotionalState.FRUSTRATION, EmotionalState.CREATIVE_BREAKTHROUGH
        ]
        for state in arousal_contributing_states:
            arousal_level += state_scores.get(state, 0.0) * 0.25
        
        # Calculate stress from negative states
        stress_contributing_states = [
            EmotionalState.ANXIETY, EmotionalState.FRUSTRATION,
            EmotionalState.COUNTERPRODUCTIVE_CONFUSION
        ]
        for state in stress_contributing_states:
            stress_level += state_scores.get(state, 0.0) * 0.33
        
        # Calculate energy level
        energy_boosting_states = [
            EmotionalState.EXCITEMENT, EmotionalState.CURIOSITY,
            EmotionalState.FLOW_STATE, EmotionalState.CREATIVE_BREAKTHROUGH
        ]
        energy_draining_states = [
            EmotionalState.MENTAL_FATIGUE, EmotionalState.FRUSTRATION,
            EmotionalState.ANXIETY
        ]
        
        for state in energy_boosting_states:
            energy_level += state_scores.get(state, 0.0) * 0.25
        
        for state in energy_draining_states:
            energy_level -= state_scores.get(state, 0.0) * 0.25
        
        energy_level = max(0.0, min(1.0, energy_level))
        
        return {
            "arousal_level": min(1.0, arousal_level),
            "stress_level": min(1.0, stress_level),
            "energy_level": energy_level,
            "cognitive_temperature": cognitive_metrics.get("processing_speed", 0.5),
            "attention_stability": 1.0 - state_scores.get(EmotionalState.ANXIETY, 0.0),
            "emotional_regulation": 1.0 - max(stress_level, arousal_level - 0.7)
        }
    
    def _calculate_cognitive_load(self, 
                                 cognitive_metrics: Dict[str, float],
                                 exploration_data: Dict[str, Any]) -> float:
        """Calculate current cognitive load"""
        
        complexity = exploration_data.get("complexity", 0.0)
        parallel_processes = cognitive_metrics.get("parallel_processes", 1)
        working_memory_usage = cognitive_metrics.get("working_memory_usage", 0.5)
        attention_demands = cognitive_metrics.get("attention_demands", 0.5)
        
        # Combine factors for cognitive load
        load = (complexity * 0.3 + 
                min(1.0, parallel_processes / 5) * 0.3 +
                working_memory_usage * 0.2 +
                attention_demands * 0.2)
        
        return min(1.0, load)
    
    def _calculate_creativity_index(self, 
                                   state_scores: Dict[EmotionalState, float],
                                   cognitive_metrics: Dict[str, float]) -> float:
        """Calculate creativity index based on emotional and cognitive factors"""
        
        creativity_boosting_states = [
            EmotionalState.OPTIMAL_CREATIVE_TENSION, EmotionalState.CURIOSITY,
            EmotionalState.EXCITEMENT, EmotionalState.FLOW_STATE,
            EmotionalState.CREATIVE_BREAKTHROUGH
        ]
        
        creativity_hindering_states = [
            EmotionalState.MENTAL_FATIGUE, EmotionalState.FRUSTRATION,
            EmotionalState.COUNTERPRODUCTIVE_CONFUSION
        ]
        
        creativity_boost = sum(state_scores.get(state, 0.0) for state in creativity_boosting_states)
        creativity_hindrance = sum(state_scores.get(state, 0.0) for state in creativity_hindering_states)
        
        base_creativity = cognitive_metrics.get("creative_potential", 0.5)
        
        creativity_index = base_creativity + (creativity_boost * 0.3) - (creativity_hindrance * 0.2)
        
        return max(0.0, min(1.0, creativity_index))
    
    def _calculate_exploration_readiness(self, 
                                       state_scores: Dict[EmotionalState, float],
                                       cognitive_load: float) -> float:
        """Calculate readiness for exploration activities"""
        
        exploration_ready_states = [
            EmotionalState.CURIOSITY, EmotionalState.EXCITEMENT,
            EmotionalState.OPTIMAL_CREATIVE_TENSION, EmotionalState.FLOW_STATE
        ]
        
        exploration_unready_states = [
            EmotionalState.MENTAL_FATIGUE, EmotionalState.ANXIETY,
            EmotionalState.COUNTERPRODUCTIVE_CONFUSION
        ]
        
        readiness_boost = sum(state_scores.get(state, 0.0) for state in exploration_ready_states)
        readiness_reduction = sum(state_scores.get(state, 0.0) for state in exploration_unready_states)
        
        # High cognitive load reduces exploration readiness
        load_penalty = cognitive_load * 0.3
        
        readiness = 0.5 + (readiness_boost * 0.4) - (readiness_reduction * 0.4) - load_penalty
        
        return max(0.0, min(1.0, readiness))


class EmotionalPatternAnalyzer:
    """Analyzes emotional patterns and trends over time"""
    
    def __init__(self):
        self.pattern_history: deque = deque(maxlen=10000)
        self.identified_patterns: Dict[str, EmotionalPattern] = {}
        self.pattern_templates = {
            "creative_cycle": self._detect_creative_cycle,
            "fatigue_recovery": self._detect_fatigue_recovery,
            "anxiety_spiral": self._detect_anxiety_spiral,
            "flow_entry": self._detect_flow_entry,
            "breakthrough_buildup": self._detect_breakthrough_buildup
        }
        
    def analyze_patterns(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Analyze emotional patterns from a sequence of snapshots"""
        
        if len(snapshots) < 3:
            return []
        
        identified_patterns = []
        
        # Run pattern detection algorithms
        for pattern_name, detector in self.pattern_templates.items():
            try:
                patterns = detector(snapshots)
                identified_patterns.extend(patterns)
            except Exception as e:
                logger.warning(f"Pattern detection error for {pattern_name}: {e}")
        
        # Store patterns for future reference
        for pattern in identified_patterns:
            self.identified_patterns[pattern.pattern_id] = pattern
        
        return identified_patterns
    
    def _detect_creative_cycle(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect creative cycles: curiosity -> tension -> breakthrough -> satisfaction"""
        
        patterns = []
        cycle_states = [
            EmotionalState.CURIOSITY,
            EmotionalState.OPTIMAL_CREATIVE_TENSION,
            EmotionalState.CREATIVE_BREAKTHROUGH,
            EmotionalState.SATISFACTION
        ]
        
        # Look for sequences matching the creative cycle
        for i in range(len(snapshots) - len(cycle_states) + 1):
            sequence = snapshots[i:i + len(cycle_states)]
            
            # Check if sequence matches cycle pattern (with some flexibility)
            matches = 0
            for j, expected_state in enumerate(cycle_states):
                if (sequence[j].primary_state == expected_state or 
                    any(state == expected_state for state, _ in sequence[j].secondary_states)):
                    matches += 1
            
            # If at least 75% of states match, consider it a cycle
            if matches >= len(cycle_states) * 0.75:
                duration = sequence[-1].timestamp - sequence[0].timestamp
                
                pattern = EmotionalPattern(
                    pattern_id="",
                    pattern_type="creative_cycle",
                    states_sequence=cycle_states,
                    average_duration=duration,
                    trigger_conditions=[EmotionalTrigger.NOVELTY_DISCOVERY],
                    effectiveness_score=0.8,  # Creative cycles are generally beneficial
                    frequency=1,
                    last_occurrence=sequence[-1].timestamp,
                    context_tags=["creativity", "exploration", "breakthrough"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_fatigue_recovery(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect fatigue to recovery patterns"""
        
        patterns = []
        
        # Look for fatigue followed by recovery states
        for i in range(len(snapshots) - 2):
            if snapshots[i].primary_state == EmotionalState.MENTAL_FATIGUE:
                # Look for recovery indicators in subsequent snapshots
                for j in range(i + 1, min(i + 5, len(snapshots))):
                    if (snapshots[j].primary_state in [EmotionalState.SATISFACTION, EmotionalState.CURIOSITY] or
                        snapshots[j].energy_level > snapshots[i].energy_level + 0.3):
                        
                        duration = snapshots[j].timestamp - snapshots[i].timestamp
                        
                        pattern = EmotionalPattern(
                            pattern_id="",
                            pattern_type="fatigue_recovery",
                            states_sequence=[EmotionalState.MENTAL_FATIGUE, snapshots[j].primary_state],
                            average_duration=duration,
                            trigger_conditions=[EmotionalTrigger.DORMANCY_ENTRY],
                            effectiveness_score=0.7,
                            frequency=1,
                            last_occurrence=snapshots[j].timestamp,
                            context_tags=["recovery", "restoration", "energy"]
                        )
                        patterns.append(pattern)
                        break
        
        return patterns
    
    def _detect_anxiety_spiral(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect anxiety spiral patterns (escalating anxiety/confusion)"""
        
        patterns = []
        
        # Look for increasing anxiety or confusion over time
        for i in range(len(snapshots) - 3):
            anxiety_sequence = snapshots[i:i + 3]
            
            # Check if anxiety/confusion is increasing
            anxiety_levels = []
            for snapshot in anxiety_sequence:
                if snapshot.primary_state == EmotionalState.ANXIETY:
                    anxiety_levels.append(snapshot.intensity)
                elif snapshot.primary_state == EmotionalState.COUNTERPRODUCTIVE_CONFUSION:
                    anxiety_levels.append(snapshot.intensity * 0.8)  # Weight confusion slightly less
                else:
                    # Check secondary states
                    for state, intensity in snapshot.secondary_states:
                        if state in [EmotionalState.ANXIETY, EmotionalState.COUNTERPRODUCTIVE_CONFUSION]:
                            anxiety_levels.append(intensity * 0.6)
                            break
                    else:
                        anxiety_levels.append(0.0)
            
            # If anxiety is generally increasing, it's a spiral
            if len(anxiety_levels) >= 3 and anxiety_levels[-1] > anxiety_levels[0] + 0.2:
                duration = anxiety_sequence[-1].timestamp - anxiety_sequence[0].timestamp
                
                pattern = EmotionalPattern(
                    pattern_id="",
                    pattern_type="anxiety_spiral",
                    states_sequence=[EmotionalState.ANXIETY, EmotionalState.COUNTERPRODUCTIVE_CONFUSION],
                    average_duration=duration,
                    trigger_conditions=[EmotionalTrigger.COGNITIVE_DISSONANCE, EmotionalTrigger.EXPLORATION_BLOCK],
                    effectiveness_score=0.1,  # Spirals are generally harmful
                    frequency=1,
                    last_occurrence=anxiety_sequence[-1].timestamp,
                    context_tags=["anxiety", "spiral", "intervention_needed"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_flow_entry(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect patterns leading to flow state"""
        
        patterns = []
        
        # Look for sequences ending in flow state
        for i in range(len(snapshots)):
            if snapshots[i].primary_state == EmotionalState.FLOW_STATE:
                # Look backwards for the entry pattern
                entry_sequence = []
                for j in range(max(0, i - 4), i + 1):
                    entry_sequence.append(snapshots[j])
                
                if len(entry_sequence) >= 2:
                    duration = entry_sequence[-1].timestamp - entry_sequence[0].timestamp
                    
                    pattern = EmotionalPattern(
                        pattern_id="",
                        pattern_type="flow_entry",
                        states_sequence=[snap.primary_state for snap in entry_sequence],
                        average_duration=duration,
                        trigger_conditions=[EmotionalTrigger.PATTERN_RECOGNITION, EmotionalTrigger.CREATIVE_SYNTHESIS],
                        effectiveness_score=0.9,  # Flow states are highly beneficial
                        frequency=1,
                        last_occurrence=entry_sequence[-1].timestamp,
                        context_tags=["flow", "optimal_performance", "focus"]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_breakthrough_buildup(self, snapshots: List[EmotionalStateSnapshot]) -> List[EmotionalPattern]:
        """Detect patterns leading to creative breakthroughs"""
        
        patterns = []
        
        # Look for sequences ending in breakthrough
        for i in range(len(snapshots)):
            if snapshots[i].primary_state == EmotionalState.CREATIVE_BREAKTHROUGH:
                # Look backwards for buildup pattern
                buildup_sequence = []
                for j in range(max(0, i - 5), i + 1):
                    buildup_sequence.append(snapshots[j])
                
                if len(buildup_sequence) >= 3:
                    # Check for increasing tension/complexity before breakthrough
                    tension_trend = []
                    for snapshot in buildup_sequence[:-1]:  # Exclude breakthrough itself
                        if snapshot.primary_state == EmotionalState.OPTIMAL_CREATIVE_TENSION:
                            tension_trend.append(snapshot.intensity)
                        else:
                            tension_trend.append(snapshot.cognitive_load)
                    
                    # If tension generally increased, it's a valid buildup
                    if len(tension_trend) >= 2 and tension_trend[-1] > tension_trend[0]:
                        duration = buildup_sequence[-1].timestamp - buildup_sequence[0].timestamp
                        
                        pattern = EmotionalPattern(
                            pattern_id="",
                            pattern_type="breakthrough_buildup",
                            states_sequence=[snap.primary_state for snap in buildup_sequence],
                            average_duration=duration,
                            trigger_conditions=[EmotionalTrigger.COGNITIVE_DISSONANCE, EmotionalTrigger.CREATIVE_SYNTHESIS],
                            effectiveness_score=0.95,  # Breakthrough buildups are extremely valuable
                            frequency=1,
                            last_occurrence=buildup_sequence[-1].timestamp,
                            context_tags=["breakthrough", "buildup", "creative_tension"]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of identified patterns"""
        
        pattern_counts = defaultdict(int)
        pattern_effectiveness = defaultdict(list)
        
        for pattern in self.identified_patterns.values():
            pattern_counts[pattern.pattern_type] += 1
            pattern_effectiveness[pattern.pattern_type].append(pattern.effectiveness_score)
        
        # Calculate average effectiveness by pattern type
        avg_effectiveness = {}
        for pattern_type, scores in pattern_effectiveness.items():
            avg_effectiveness[pattern_type] = sum(scores) / len(scores)
        
        return {
            "total_patterns": len(self.identified_patterns),
            "pattern_counts": dict(pattern_counts),
            "average_effectiveness": avg_effectiveness,
            "most_common_pattern": max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None,
            "most_effective_pattern": max(avg_effectiveness.items(), key=lambda x: x[1])[0] if avg_effectiveness else None
        }
