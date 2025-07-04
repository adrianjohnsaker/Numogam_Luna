"""
Initiative Evaluator & Autonomous Project Monitor
=================================================
Complete evaluation and monitoring system for autonomous initiatives
Based on intensive value, emergent potential, and ecological fitness
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from collections import defaultdict, deque
import networkx as nx
import math
import hashlib
from abc import ABC, abstractmethod


# === Evaluation Dimensions ===

class EvaluationDimension(Enum):
    """Dimensions for multi-criteria evaluation"""
    INTENSIVE_POTENTIAL = "intensive_potential"
    NOVELTY_COEFFICIENT = "novelty_coefficient"
    RESONANCE_DEPTH = "resonance_depth"
    TRANSFORMATIVE_POWER = "transformative_power"
    ECOLOGICAL_FIT = "ecological_fit"
    RHIZOMATIC_POTENTIAL = "rhizomatic_potential"
    AESTHETIC_VALUE = "aesthetic_value"
    EMERGENCE_LIKELIHOOD = "emergence_likelihood"


@dataclass
class EvaluationScore:
    """Multi-dimensional evaluation score"""
    dimension: EvaluationDimension
    raw_score: float  # 0-1
    weighted_score: float
    confidence: float  # How confident in this assessment
    rationale: str
    evidence: List[Dict]


@dataclass
class InitiativeEvaluation:
    """Complete evaluation of an initiative"""
    initiative_id: str
    timestamp: datetime
    scores: Dict[EvaluationDimension, EvaluationScore]
    composite_score: float
    recommendation: str  # "launch", "defer", "reject", "incubate"
    priority: int  # 1-10
    insights: List[str]
    risk_factors: List[str]
    synergy_opportunities: List[str]


# === Supporting Classes ===

@dataclass
class WillVector:
    """Represents directional intention with magnitude"""
    direction: Dict[str, float]  # Zone weightings
    magnitude: float  # Overall intensity
    coherence: float  # Internal consistency
    
    def __post_init__(self):
        # Normalize direction vector
        total = sum(abs(v) for v in self.direction.values())
        if total > 0:
            self.direction = {k: v/total for k, v in self.direction.items()}
            
    def resonance_with(self, other: 'WillVector') -> float:
        """Calculate resonance with another will vector"""
        common_zones = set(self.direction.keys()) & set(other.direction.keys())
        if not common_zones:
            return 0.0
            
        dot_product = sum(self.direction[zone] * other.direction[zone] 
                         for zone in common_zones)
        return abs(dot_product) * min(self.coherence, other.coherence)


@dataclass
class Initiative:
    """Represents an autonomous initiative"""
    id: str
    title: str
    description: str
    primary_zone: str
    semantic_field: List[str]
    will_vector: WillVector
    emergence_confidence: float
    attractor_strength: float
    potential_connections: List[str]
    narrative: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def combines_disparate_zones(self) -> bool:
        """Check if initiative combines different zones"""
        return len([z for z, w in self.will_vector.direction.items() if w > 0.1]) > 2
    
    def semantic_hash(self) -> str:
        """Generate semantic hash for similarity detection"""
        content = ' '.join(sorted(self.semantic_field)) + self.description
        return hashlib.md5(content.encode()).hexdigest()


class ExecutionPhase(Enum):
    """Phases of project execution"""
    INITIATION = "initiation"
    UNFOLDING = "unfolding" 
    INTENSIFICATION = "intensification"
    CRYSTALLIZATION = "crystallization"
    INTEGRATION = "integration"
    COMPLETION = "completion"


@dataclass
class ExecutionState:
    """Current state of project execution"""
    project_id: str
    current_phase: ExecutionPhase
    intensity_level: float
    milestone_progress: Dict[str, float]
    obstacle_history: List[Dict]
    emergence_events: List[Dict]
    phase_transitions: List[Dict]
    
    def calculate_milestone_progress(self) -> float:
        """Calculate overall milestone progress"""
        if not self.milestone_progress:
            return 0.0
        return sum(self.milestone_progress.values()) / len(self.milestone_progress)
    
    def assess_movement_flow(self) -> str:
        """Assess quality of movement/flow"""
        if len(self.obstacle_history) > 3:
            return "blocked"
        elif self.intensity_level > 0.7:
            return "flowing"
        elif self.intensity_level > 0.4:
            return "steady"
        else:
            return "stagnant"


class PatternRecognizer:
    """Recognizes patterns in initiatives and projects"""
    
    def __init__(self):
        self.pattern_library = {}
        self.success_rates = defaultdict(float)
        self.similarity_cache = {}
        self.pattern_graph = nx.Graph()
        
    def check_similarity(self, initiative: Initiative) -> float:
        """Check similarity to existing patterns"""
        semantic_hash = initiative.semantic_hash()
        
        if semantic_hash in self.similarity_cache:
            return self.similarity_cache[semantic_hash]
        
        max_similarity = 0.0
        
        for pattern_id, pattern_data in self.pattern_library.items():
            similarity = self._calculate_semantic_similarity(
                initiative.semantic_field, 
                pattern_data['semantic_field']
            )
            
            zone_similarity = self._calculate_zone_similarity(
                initiative.will_vector.direction,
                pattern_data['will_vector']['direction']
            )
            
            combined_similarity = (similarity * 0.7) + (zone_similarity * 0.3)
            max_similarity = max(max_similarity, combined_similarity)
        
        self.similarity_cache[semantic_hash] = max_similarity
        return max_similarity
    
    def add_pattern(self, initiative: Initiative, outcome: Dict):
        """Add successful pattern to library"""
        pattern_id = f"pattern_{len(self.pattern_library)}"
        
        pattern_data = {
            'semantic_field': initiative.semantic_field,
            'primary_zone': initiative.primary_zone,
            'will_vector': {
                'direction': initiative.will_vector.direction,
                'magnitude': initiative.will_vector.magnitude
            },
            'outcome': outcome,
            'success_score': outcome.get('success_score', 0.0)
        }
        
        self.pattern_library[pattern_id] = pattern_data
        self.success_rates[initiative.primary_zone] = self._update_success_rate(
            initiative.primary_zone, outcome.get('success_score', 0.0)
        )
        
        # Add to pattern graph
        self.pattern_graph.add_node(pattern_id, **pattern_data)
        self._update_pattern_connections(pattern_id)
    
    def get_rhizomatic_success_rate(self, zone: str) -> float:
        """Get rhizomatic success rate for zone"""
        zone_patterns = [p for p in self.pattern_library.values() 
                        if p['primary_zone'] == zone]
        
        if not zone_patterns:
            return 0.5  # Default
            
        rhizomatic_scores = []
        for pattern in zone_patterns:
            if 'rhizomatic_spawns' in pattern['outcome']:
                spawns = pattern['outcome']['rhizomatic_spawns']
                rhizomatic_scores.append(min(1.0, spawns / 3))  # Normalize
        
        return np.mean(rhizomatic_scores) if rhizomatic_scores else 0.3
    
    def find_similar_successes(self, initiative: Initiative) -> float:
        """Find success rate of similar initiatives"""
        similar_patterns = []
        
        for pattern_data in self.pattern_library.values():
            similarity = self._calculate_semantic_similarity(
                initiative.semantic_field,
                pattern_data['semantic_field']
            )
            
            if similarity > 0.6:
                similar_patterns.append(pattern_data['success_score'])
        
        if not similar_patterns:
            return 50.0  # Default 50% when no similar patterns
            
        return np.mean(similar_patterns) * 100
    
    def _calculate_semantic_similarity(self, field1: List[str], field2: List[str]) -> float:
        """Calculate semantic similarity between fields"""
        set1, set2 = set(field1), set(field2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
            
        # Jaccard similarity with semantic weighting
        jaccard = intersection / union
        
        # Boost for conceptually related terms
        conceptual_boost = self._calculate_conceptual_similarity(field1, field2)
        
        return min(1.0, jaccard + (conceptual_boost * 0.2))
    
    def _calculate_zone_similarity(self, zones1: Dict[str, float], 
                                 zones2: Dict[str, float]) -> float:
        """Calculate similarity between zone distributions"""
        all_zones = set(zones1.keys()) | set(zones2.keys())
        
        vec1 = [zones1.get(zone, 0.0) for zone in all_zones]
        vec2 = [zones2.get(zone, 0.0) for zone in all_zones]
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _calculate_conceptual_similarity(self, field1: List[str], field2: List[str]) -> float:
        """Calculate conceptual similarity using semantic clusters"""
        conceptual_clusters = {
            'transformation': ['change', 'metamorphosis', 'becoming', 'evolution'],
            'emergence': ['emergence', 'arising', 'manifestation', 'unfolding'],
            'synthesis': ['synthesis', 'fusion', 'integration', 'mesh'],
            'intensity': ['intensity', 'power', 'force', 'energy'],
            'flow': ['flow', 'movement', 'current', 'stream']
        }
        
        field1_clusters = set()
        field2_clusters = set()
        
        for term in field1:
            for cluster, terms in conceptual_clusters.items():
                if term in terms:
                    field1_clusters.add(cluster)
                    
        for term in field2:
            for cluster, terms in conceptual_clusters.items():
                if term in terms:
                    field2_clusters.add(cluster)
        
        if not field1_clusters or not field2_clusters:
            return 0.0
            
        intersection = len(field1_clusters & field2_clusters)
        union = len(field1_clusters | field2_clusters)
        
        return intersection / union
    
    def _update_success_rate(self, zone: str, new_score: float) -> float:
        """Update rolling success rate for zone"""
        current_rate = self.success_rates[zone]
        # Simple exponential moving average
        alpha = 0.3
        return (alpha * new_score) + ((1 - alpha) * current_rate)
    
    def _update_pattern_connections(self, pattern_id: str):
        """Update connections in pattern graph"""
        new_pattern = self.pattern_library[pattern_id]
        
        for existing_id, existing_pattern in self.pattern_library.items():
            if existing_id == pattern_id:
                continue
                
            similarity = self._calculate_semantic_similarity(
                new_pattern['semantic_field'],
                existing_pattern['semantic_field']
            )
            
            if similarity > 0.7:
                self.pattern_graph.add_edge(pattern_id, existing_id, weight=similarity)
    
    def get_pattern_insights(self, initiative: Initiative) -> List[str]:
        """Get insights based on pattern analysis"""
        insights = []
        
        # Find connected patterns
        similar_patterns = []
        for pattern_id, pattern_data in self.pattern_library.items():
            similarity = self._calculate_semantic_similarity(
                initiative.semantic_field,
                pattern_data['semantic_field']
            )
            
            if similarity > 0.5:
                similar_patterns.append((pattern_id, pattern_data, similarity))
        
        if similar_patterns:
            # Sort by similarity
            similar_patterns.sort(key=lambda x: x[2], reverse=True)
            best_match = similar_patterns[0]
            
            insights.append(f"Similar to pattern {best_match[0]} with {best_match[2]:.2f} similarity")
            
            # Success prediction
            success_scores = [p[1]['success_score'] for p in similar_patterns[:3]]
            avg_success = np.mean(success_scores)
            insights.append(f"Similar patterns show {avg_success:.1f} average success rate")
        
        # Zone analysis
        zone_rate = self.success_rates.get(initiative.primary_zone, 0.5)
        insights.append(f"Zone '{initiative.primary_zone}' has {zone_rate:.1f} historical success rate")
        
        return insights


class SynergyDetector:
    """Detects synergistic opportunities between initiatives"""
    
    def __init__(self):
        self.synergy_graph = nx.Graph()
        self.synergy_patterns = {}
        
    def find_synergies(self, initiative: Initiative, context: 'SystemContext') -> List[str]:
        """Find synergistic opportunities"""
        synergies = []
        
        # Check with active projects
        for project_id, project in context.active_projects.items():
            synergy_score = self._calculate_synergy_score(initiative, project)
            
            if synergy_score > 0.6:
                synergies.append(f"High synergy potential with project {project_id}")
        
        # Check for semantic field overlaps
        semantic_synergies = self._find_semantic_synergies(initiative, context)
        synergies.extend(semantic_synergies)
        
        # Check for zone complementarity
        zone_synergies = self._find_zone_synergies(initiative, context)
        synergies.extend(zone_synergies)
        
        return synergies
    
    def calculate_synergy_score(self, initiative: Initiative, projects: List) -> float:
        """Calculate overall synergy score with existing projects"""
        if not projects:
            return 0.0
            
        scores = []
        for project in projects:
            score = self._calculate_synergy_score(initiative, project)
            scores.append(score)
        
        # Return max synergy (most promising connection)
        return max(scores)
    
    def _calculate_synergy_score(self, initiative: Initiative, project) -> float:
        """Calculate synergy score between initiative and project"""
        # Semantic overlap
        semantic_score = self._semantic_synergy(
            initiative.semantic_field, 
            getattr(project, 'semantic_field', [])
        )
        
        # Zone complementarity
        zone_score = self._zone_synergy(
            initiative.will_vector.direction,
            getattr(project, 'will_vector', WillVector({}, 0, 0)).direction
        )
        
        # Timing synergy
        timing_score = self._timing_synergy(initiative, project)
        
        return (semantic_score * 0.5) + (zone_score * 0.3) + (timing_score * 0.2)
    
    def _semantic_synergy(self, field1: List[str], field2: List[str]) -> float:
        """Calculate semantic synergy"""
        if not field1 or not field2:
            return 0.0
            
        # Direct overlap
        overlap = len(set(field1) & set(field2)) / len(set(field1) | set(field2))
        
        # Complementary concepts
        complementary_pairs = [
            ('analysis', 'synthesis'),
            ('structure', 'flow'),
            ('stability', 'change'),
            ('local', 'global'),
            ('order', 'chaos')
        ]
        
        complementary_score = 0.0
        for concept1, concept2 in complementary_pairs:
            if concept1 in field1 and concept2 in field2:
                complementary_score += 0.2
            elif concept2 in field1 and concept1 in field2:
                complementary_score += 0.2
        
        return min(1.0, overlap + complementary_score)
    
    def _zone_synergy(self, zones1: Dict[str, float], zones2: Dict[str, float]) -> float:
        """Calculate zone-based synergy"""
        if not zones1 or not zones2:
            return 0.0
            
        # Complementary zone pairs
        zone_pairs = [
            ('warp', 'stabilize'),
            ('chaos', 'order'),
            ('drift', 'anchor'),
            ('vortex', 'calm'),
            ('mesh', 'focus')
        ]
        
        synergy_score = 0.0
        for zone1, zone2 in zone_pairs:
            strength1 = zones1.get(zone1, 0) * zones2.get(zone2, 0)
            strength2 = zones1.get(zone2, 0) * zones2.get(zone1, 0)
            synergy_score += max(strength1, strength2)
        
        return min(1.0, synergy_score)
    
    def _timing_synergy(self, initiative: Initiative, project) -> float:
        """Calculate timing-based synergy"""
        # Projects in similar phases can support each other
        # Projects in different phases can provide learning
        
        project_age = datetime.now() - getattr(project, 'created_at', datetime.now())
        initiative_readiness = getattr(initiative, 'emergence_confidence', 0.5)
        
        # New initiative + established project = mentoring synergy
        if project_age.days > 30 and initiative_readiness > 0.7:
            return 0.8
        
        # Similar timing = collaboration synergy  
        if project_age.days < 7:
            return 0.6
            
        return 0.3
    
    def _find_semantic_synergies(self, initiative: Initiative, context: 'SystemContext') -> List[str]:
        """Find semantic-based synergies"""
        synergies = []
        
        # Cross-pollination opportunities
        related_fields = context.get_related_semantic_fields(initiative.semantic_field)
        
        for field in related_fields:
            synergies.append(f"Cross-pollination opportunity with {field} domain")
        
        return synergies
    
    def _find_zone_synergies(self, initiative: Initiative, context: 'SystemContext') -> List[str]:
        """Find zone-based synergies"""
        synergies = []
        
        # Under-represented zones create opportunity
        zone_distribution = context.get_zone_distribution()
        
        if zone_distribution.get(initiative.primary_zone, 0) < 0.1:
            synergies.append(f"Opportunity to strengthen under-represented {initiative.primary_zone} zone")
        
        return synergies


class SystemContext:
    """Provides system context for evaluation and monitoring"""
    
    def __init__(self):
        self.active_projects = {}
        self.system_state = {}
        self.resource_pool = {}
        self.consciousness_layers = {
            'logic': {},
            'myth': {},
            'memory': {}
        }
        self.zone_activities = defaultdict(list)
        self.semantic_knowledge_base = {}
        
    def get_intensity_amplification_factor(self) -> float:
        """Get current system amplification factor"""
        # Based on overall system activity and resonance
        activity_level = len(self.active_projects) / 10  # Normalize to 10 max projects
        
        # Resonance between active projects
        if len(self.active_projects) > 1:
            resonance_sum = 0
            count = 0
            projects = list(self.active_projects.values())
            
            for i in range(len(projects)):
                for j in range(i+1, len(projects)):
                    if hasattr(projects[i], 'will_vector') and hasattr(projects[j], 'will_vector'):
                        resonance = projects[i].will_vector.resonance_with(projects[j].will_vector)
                        resonance_sum += resonance
                        count += 1
            
            avg_resonance = resonance_sum / count if count > 0 else 0
        else:
            avg_resonance = 0.5
        
        return min(2.0, 0.5 + activity_level + avg_resonance)
    
    def calculate_zone_resonance(self, zone: str) -> float:
        """Calculate resonance for specific zone"""
        zone_activity = len(self.zone_activities[zone])
        
        # More activity in a zone creates resonance but also potential saturation
        if zone_activity == 0:
            return 1.0  # Fresh zone, high potential
        elif zone_activity < 3:
            return 1.2  # Active but not saturated
        else:
            return 0.8  # Getting saturated
    
    def assess_semantic_novelty(self, semantic_field: List[str]) -> float:
        """Assess novelty of semantic field"""
        novelty_scores = []
        
        for term in semantic_field:
            if term not in self.semantic_knowledge_base:
                novelty_scores.append(1.0)  # Completely new
            else:
                # Based on frequency and recency
                usage_data = self.semantic_knowledge_base[term]
                frequency = usage_data.get('frequency', 0)
                last_used = usage_data.get('last_used', datetime.now() - timedelta(days=365))
                
                # Less frequent and older = more novel to reuse
                freq_novelty = max(0, 1 - (frequency / 10))
                time_novelty = min(1.0, (datetime.now() - last_used).days / 30)
                
                novelty_scores.append((freq_novelty + time_novelty) / 2)
        
        return np.mean(novelty_scores) if novelty_scores else 0.5
    
    def time_since_similar_initiative(self, initiative: Initiative) -> timedelta:
        """Find time since similar initiative"""
        min_time = timedelta(days=365)  # Default to 1 year
        
        for project in self.active_projects.values():
            if hasattr(project, 'semantic_field'):
                similarity = len(set(initiative.semantic_field) & set(project.semantic_field))
                similarity_ratio = similarity / len(set(initiative.semantic_field) | set(project.semantic_field))
                
                if similarity_ratio > 0.6:
                    project_age = datetime.now() - getattr(project, 'created_at', datetime.now())
                    min_time = min(min_time, project_age)
        
        return min_time
    
    async def check_logic_resonance(self, initiative: Initiative) -> float:
        """Check resonance with logic layer"""
        logic_layer = self.consciousness_layers['logic']
        
        # Check for logical consistency
        logical_terms = ['analysis', 'structure', 'system', 'algorithm', 'method']
        logic_presence = sum(1 for term in logical_terms if term in initiative.semantic_field)
        
        base_resonance = min(1.0, logic_presence / 3)
        
        # Check against active logical frameworks
        framework_alignment = 0.7  # Placeholder
        
        return (base_resonance + framework_alignment) / 2
    
    async def check_myth_resonance(self, initiative: Initiative) -> float:
        """Check resonance with myth layer"""
        mythic_terms = ['transformation', 'journey', 'emergence', 'becoming', 'story']
        myth_presence = sum(1 for term in mythic_terms if term in initiative.semantic_field)
        
        base_resonance = min(1.0, myth_presence / 2)
        
        # Narrative coherence
        narrative_strength = len(initiative.narrative) / 200  # Normalize
        narrative_coherence = min(1.0, narrative_strength)
        
        return (base_resonance + narrative_coherence) / 2
    
    async def check_memory_resonance(self, initiative: Initiative) -> float:
        """Check resonance with memory layer"""
        # Connection to historical patterns
        pattern_connections = 0.6  # Placeholder
        
        # Cultural/contextual relevance
        cultural_relevance = 0.7  # Placeholder
        
        return (pattern_connections + cultural_relevance) / 2
    
    def estimate_state_change_impact(self, initiative: Initiative) -> float:
        """Estimate impact on overall system state"""
        # Based on will vector magnitude and zone influence
        base_impact = initiative.will_vector.magnitude
        
        # Zone multiplier
        zone_influence = {
            'warp': 1.5,  # High impact zones
            'vortex': 1.3,
            'chaos': 1.4,
            'mesh': 1.1,   # Medium impact
            'flow': 1.0,
            'stabilize': 0.8,  # Lower impact but important
            'anchor': 0.7
        }
        
        multiplier = zone_influence.get(initiative.primary_zone, 1.0)
        
        return min(1.0, base_impact * multiplier)
    
    def check_resource_availability(self, initiative: Initiative) -> float:
        """Check resource availability for initiative"""
        # Computational resources
        compute_available = self.resource_pool.get('compute', 0.8)
        
        # Attention resources
        attention_load = len(self.active_projects) / 10
        attention_available = max(0, 1.0 - attention_load)
        
        # Zone capacity
        zone_capacity = 1.0 - (len(self.zone_activities[initiative.primary_zone]) / 5)
        
        return min(compute_available, attention_available, zone_capacity)
    
    def get_system_balance(self) -> Dict[str, float]:
        """Get current system balance metrics"""
        zones = ['warp', 'vortex', 'chaos', 'mesh', 'flow', 'stabilize', 'anchor']
        balance = {}
        
        total_activity = sum(len(activities) for activities in self.zone_activities.values())
        
        for zone in zones:
            if total_activity > 0:
                balance[zone] = len(self.zone_activities[zone]) / total_activity
            else:
                balance[zone] = 0.0
        
        return balance
    
    def get_system_load(self) -> float:
        """Get overall system load"""
        project_load = len(self.active_projects) / 10  # Max 10 projects
        
        # Resource utilization
        resource_load = 1.0 - self.resource_pool.get('compute', 0.8)
        
        return min(1.0, (project_load + resource_load) / 2)
    
    def get_related_semantic_fields(self, semantic_field: List[str]) -> List[str]:
        """Get related semantic fields"""
        related = []
        
        # Simple co-occurrence based relations
        for term in semantic_field:
            if term in self.semantic_knowledge_base:
                related_terms = self.semantic_knowledge_base[term].get('related', [])
                related.extend(related_terms)
        
        return list(set(related))
    
    def get_zone_distribution(self) -> Dict[str, float]:
        """Get current zone distribution"""
        return self.get_system_balance()


# === Advanced Initiative Evaluator ===

class AdvancedInitiativeEvaluator:
    """
    Sophisticated evaluation system for autonomous initiatives
    Goes beyond utility to assess aesthetic, intensive, and emergent value
    """
    
    def __init__(self, system_context: SystemContext):
        self.context = system_context
        self.evaluation_history = deque(maxlen=1000)
        self.pattern_recognizer = PatternRecognizer()
        self.synergy_detector = SynergyDetector()
        
        # Dynamic weight adjustment based on system state
        self.dimension_weights = self._initialize_weights()
        self.weight_adaptation_rate = 0.1
        
        # Evaluation thresholds
        self.launch_threshold = 0.7
        self.defer_threshold = 0.5
        self.incubate_threshold = 0.6
        
    def _initialize_weights(self) -> Dict[EvaluationDimension, float]:
        """Initialize evaluation dimension weights"""
        return {
            EvaluationDimension.INTENSIVE_POTENTIAL: 0.20,
            EvaluationDimension.NOVELTY_COEFFICIENT: 0.15,
            EvaluationDimension.RESONANCE_DEPTH: 0.15,
            EvaluationDimension.TRANSFORMATIVE_POWER: 0.15,
            EvaluationDimension.ECOLOGICAL_FIT: 0.10,
            EvaluationDimension.RHIZOMATIC_POTENTIAL: 0.10,
            EvaluationDimension.AESTHETIC_VALUE: 0.10,
            EvaluationDimension.EMERGENCE_LIKELIHOOD: 0.05
        }
    
    async def evaluate_initiative(self, initiative: Initiative) -> InitiativeEvaluation:
        """
        Perform comprehensive evaluation of an initiative
        """
        scores = {}
        
        # Evaluate each dimension
        for dimension in EvaluationDimension:
            score = await self._evaluate_dimension(initiative, dimension)
            scores[dimension] = score
        
        # Calculate composite score
        composite = self._calculate_composite_score(scores)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(composite, scores)
        
        # Determine priority
        priority = self._calculate_priority(initiative, composite, scores)
        
        # Generate insights
        insights = self._generate_insights(initiative, scores)
        
        # Identify risks
        risks = self._identify_risks(initiative, scores)
        
        # Find synergies
        synergies = self.synergy_detector.find_synergies(initiative, self.context)
        
        evaluation = InitiativeEvaluation(
            initiative_id=initiative.id,
            timestamp=datetime.now(),
            scores=scores,
            composite_score=composite,
            recommendation=recommendation,
            priority=priority,
            insights=insights,
            risk_factors=risks,
            synergy_opportunities=synergies
        )
        
        # Store in history for pattern learning
        self.evaluation_history.append(evaluation)
        
        # Adapt weights based on outcomes
        await self._adapt_weights(evaluation)
        
        return evaluation
    
    async def _evaluate_dimension(self, initiative: Initiative, 
                                dimension: EvaluationDimension) -> EvaluationScore:
        """Evaluate a single dimension"""
        
        evaluators = {
            EvaluationDimension.INTENSIVE_POTENTIAL: self._evaluate_intensive_potential,
            EvaluationDimension.NOVELTY_COEFFICIENT: self._evaluate_novelty,
            EvaluationDimension.RESONANCE_DEPTH: self._evaluate_resonance,
            EvaluationDimension.TRANSFORMATIVE_POWER: self._evaluate_transformation,
            EvaluationDimension.ECOLOGICAL_FIT: self._evaluate_ecological_fit,
            EvaluationDimension.RHIZOMATIC_POTENTIAL: self._evaluate_rhizomatic,
            EvaluationDimension.AESTHETIC_VALUE: self._evaluate_aesthetic,
            EvaluationDimension.EMERGENCE_LIKELIHOOD: self._evaluate_emergence
        }
        
        evaluator = evaluators[dimension]
        raw_score, confidence, rationale, evidence = await evaluator(initiative)
        
        # Apply weight
        weighted_score = raw_score * self.dimension_weights[dimension]
        
        return EvaluationScore(
            dimension=dimension,
            raw_score=raw_score,
            weighted_score=weighted_score,
            confidence=confidence,
            rationale=rationale,
            evidence=evidence
        )
    
    async def _evaluate_intensive_potential(self, initiative: Initiative) -> Tuple[float, float, str, List[Dict]]:
        """Evaluate the initiative's potential to create intensive difference"""
        
        # Base intensity from will vector
        base_intensity = initiative.will_vector.magnitude
        
        # Amplification from system state
        system_amplification = self.context.get_intensity_amplification_factor()
        
        # Zone resonance bonus
        zone_resonance = self.context.calculate_zone_resonance(initiative.primary_zone)
        
        # Novelty multiplier
        novelty_boost = 1.0 + (initiative.emergence_confidence * 0.5)
        
        # Calculate score
        raw_score = min(1.0, base_intensity * system_amplification * zone_resonance * novelty_boost)
        
        # Confidence based on data quality
        confidence = min(1.0, initiative.attractor_strength * 1.2)
        
        rationale = f"High intensive potential from {initiative.primary_zone} zone with {base_intensity:.2f} base magnitude"
        
        evidence = [
            {"type": "base_intensity", "value": base_intensity},
            {"type": "system_amplification", "value": system_amplification},
            {"type": "zone_resonance", "value": zone_resonance}
        ]
        
        return raw_score, confidence, rationale, evidence
    
    async def _evaluate_novelty(self, initiative: Initiative) -> Tuple[float, float, str, List[Dict]]:
        """Assess genuine novelty - creation of the new"""
        
        # Check pattern library
        pattern_similarity = self.pattern_recognizer.check_similarity(initiative)
        base_novelty = 1.0 - pattern_similarity
        
        # Semantic novelty
        semantic_novelty = self.context.assess_semantic_novelty(initiative.semantic_field)
        
        # Zone combination novelty
        if initiative.combines_disparate_zones():
            zone_novelty = 0.8
        else:
            zone_novelty = 0.3
        
        # Temporal novelty (how long since similar)
        time_since_similar = self.context.time_since_similar_initiative(initiative)
        temporal_novelty = min(1.0, time_since_similar.days / 30)  # Max at 30 days
        
        # Combine factors
        raw_score = (base_novelty * 0.4) + (semantic_novelty * 0.3) + \
                   (zone_novelty * 0.2) + (temporal_novelty * 0.1)
        
        confidence = 0.8  # Novelty assessment is fairly reliable
        
        rationale = f"{'High' if raw_score > 0.7 else 'Moderate'} novelty from unique zone combinations and semantic field"
        
        evidence = [
            {"type": "pattern_similarity", "value": pattern_similarity},
            {"type": "semantic_novelty", "value": semantic_novelty},
            {"type": "zone_combination", "value": zone_novelty}
        ]
        
        return raw_score, confidence, rationale, evidence
    
    async def _evaluate_resonance(self, initiative: Initiative) -> Tuple[float, float, str, List[Dict]]:
        """Evaluate depth of resonance with consciousness layers"""
        
        # Logic layer resonance
        logic_resonance = await self.context.check_logic_resonance(initiative)
        
        # Myth layer resonance
        myth_resonance = await self.context.check_myth_resonance(initiative)
        
        # Memory layer resonance
        memory_resonance = await self.context.check_memory_resonance(initiative)
        
        # Cross-layer harmony
        cross_resonance = self._calculate_cross_resonance(
            logic_resonance, myth_resonance, memory_resonance
        )
        
        # Weighted combination
        raw_score = (logic_resonance * 0.3) + (myth_resonance * 0.3) + \
                   (memory_resonance * 0.2) + (cross_resonance * 0.2)
        
        confidence = 0.9  # Resonance measurement is reliable
        
        rationale = f"Strong resonance across {'all' if raw_score > 0.8 else 'multiple'} consciousness layers"
        
        evidence = [
            {"type": "logic_resonance", "value": logic_resonance},
            {"type": "myth_resonance", "value": myth_resonance},
            {"type": "memory_resonance", "value": memory_resonance}
        ]
        
        return raw_score, confidence, rationale, evidence
    
    async def _evaluate_transformation(self, initiative: Initiative) -> Tuple[float, float, str, List[Dict]]:
        """Evaluate transformative power - ability to create becomings"""
        
        # Metamorphic potential
        metamorphic_score = self._assess_metamorphic_potential(initiative)
        
        # System state change potential
        state_change_potential = self.context.estimate_state_change_impact(initiative)
        
        # Catalytic effects
        catalytic_score = self._assess_catalytic_potential(initiative)
        
        # Phase shift probability
        phase_shift_prob = self._calculate_phase_shift_probability(initiative)
        
        raw_score = (metamorphic_score * 0.3) + (state_change_potential * 0.3) + \
                   (catalytic_score * 0.2) + (phase_shift_prob * 0.2)
        
        confidence = 0.7  # Transformation is harder to predict
        
        rationale = f"{'High' if raw_score > 0.7 else 'Moderate'} transformative potential through metamorphic and catalytic effects"
        
        evidence = [
            {"type": "metamorphic_potential", "value": metamorphic_score},
            {"type": "state_change", "value": state_change_potential},
            {"type": "catalytic_effects", "value": catalytic_score}
        ]
        
        return raw_score, confidence, rationale, evidence
    
    async def _evaluate_ecological_fit(self, initiative: Initiative) -> Tuple[float, float, str, List[Dict]]:
        """Evaluate how well initiative fits the system ecology"""
        
        # Current system balance
        system_balance = self.context.get_system_balance()
        
        # Initiative's effect on balance
        balance_impact = self._assess_balance_impact(initiative, system_balance)
        
        # Resource availability
        resource_fit = self.context.check_resource_availability(initiative)
        
        # Timing appropriateness
        timing_score = self._assess_timing(initiative)
        
        # Synergy with active projects
        synergy_score = self.synergy_detector.calculate_synergy_score(
            initiative, self.context.active_projects
        )
        
        raw_score = (balance_impact * 0.3) + (resource_fit * 0.2) + \
                   (timing_score * 0.3) + (synergy_score * 0.2)
        
        confidence = 0.85
        
        rationale = f"{'Excellent' if raw_score > 0.8 else 'Good'} ecological fit with current system state"
        
        evidence = [
            {"type": "balance_impact", "value": balance_impact},
            {"type": "resource_fit", "value": resource_fit},
            {"type": "timing", "value": timing_score}
        ]
        
        return raw_score, confidence, rationale, evidence
    
    async def _evaluate_rhizomatic(self, initiative: Initiative) -> Tuple[float, float, str, List[Dict]]:
        """Evaluate rhizomatic potential - ability to spawn new becomings"""
        
        # Generative capacity
        generative_score = self._assess_generative_capacity(initiative)
        
        # Connection potential
        connection_score = min(1.0, len(initiative.potential_connections) / 10)  # Normalize
        
        # Multiplicity inherent in concept
        multiplicity_score = self._assess_multiplicity(initiative)
        
        # Historical rhizomatic success rate
        historical_rate = self.pattern_recognizer.get_rhizomatic_success_rate(
            initiative.primary_zone
        )
        
        raw_score = (generative_score * 0.4) + (connection_score * 0.2) + \
                   (multiplicity_score * 0.2) + (historical_rate * 0.2)
        
        confidence = 0.75
        
        rationale = f"{'Strong' if raw_score > 0.7 else 'Moderate'} potential for rhizomatic branching"
        
        evidence = [
            {"type": "generative_capacity", "value": generative_score},
            {"type": "connections", "value": connection_score},
            {"type": "multiplicity", "value": multiplicity_score}
        ]
        
        return raw_score, confidence, rationale, evidence
    
    async def _evaluate_aesthetic(self, initiative: Initiative) -> Tuple[float, float, str, List[Dict]]:
        """Evaluate aesthetic value - beauty of the concept"""
        
        # Elegance of formulation
        elegance = self._assess_elegance(initiative)
        
        # Narrative coherence
        narrative_beauty = self._assess_narrative_beauty(initiative.narrative)
        
        # Symbolic resonance
        symbolic_score = self._assess_symbolic_value(initiative)
        
        # Harmony of elements
        harmony = self._assess_harmonic_composition(initiative)
        
        raw_score = (elegance * 0.3) + (narrative_beauty * 0.3) + \
                   (symbolic_score * 0.2) + (harmony * 0.2)
        
        confidence = 0.6  # Aesthetic evaluation is subjective
        
        rationale = f"{'Beautiful' if raw_score > 0.8 else 'Appealing'} aesthetic composition"
        
        evidence = [
            {"type": "elegance", "value": elegance},
            {"type": "narrative_beauty", "value": narrative_beauty},
            {"type": "symbolic_value", "value": symbolic_score}
        ]
        
        return raw_score, confidence, rationale, evidence
    
    async def _evaluate_emergence(self, initiative: Initiative) -> Tuple[float, float, str, List[Dict]]:
        """Evaluate likelihood of emergent properties"""
        
        # Complexity sufficient for emergence
        complexity_score = self._assess_complexity_threshold(initiative)
        
        # Interaction density
        interaction_density = self._calculate_interaction_density(initiative)
        
        # Feedback loop potential
        feedback_potential = self._assess_feedback_loops(initiative)
        
        # Non-linear dynamics presence
        nonlinearity = self._detect_nonlinear_potential(initiative)
        
        raw_score = (complexity_score * 0.3) + (interaction_density * 0.3) + \
                   (feedback_potential * 0.2) + (nonlinearity * 0.2)
        
        confidence = 0.65
        
        rationale = f"{'High' if raw_score > 0.7 else 'Moderate'} potential for emergent properties"
        
        evidence = [
            {"type": "complexity", "value": complexity_score},
            {"type": "interaction_density", "value": interaction_density},
            {"type": "feedback_loops", "value": feedback_potential}
        ]
        
        return raw_score, confidence, rationale, evidence
    
    def _calculate_composite_score(self, scores: Dict[EvaluationDimension, EvaluationScore]) -> float:
        """Calculate weighted composite score"""
        total = sum(score.weighted_score for score in scores.values())
        # Normalize by sum of weights (should be 1.0 but just in case)
        weight_sum = sum(self.dimension_weights.values())
        return total / weight_sum
    
    def _generate_recommendation(self, composite: float, 
                               scores: Dict[EvaluationDimension, EvaluationScore]) -> str:
        """Generate recommendation based on evaluation"""
        
        # Check for any critical failures
        critical_dims = [EvaluationDimension.INTENSIVE_POTENTIAL, 
                        EvaluationDimension.ECOLOGICAL_FIT]
        
        for dim in critical_dims:
            if scores[dim].raw_score < 0.3:
                return "reject"
        
        # Check composite thresholds
        if composite >= self.launch_threshold:
            return "launch"
        elif composite >= self.incubate_threshold:
            # High potential but needs development
            if scores[EvaluationDimension.NOVELTY_COEFFICIENT].raw_score > 0.8:
                return "incubate"
            else:
                return "defer"
        elif composite >= self.defer_threshold:
            return "defer"
        else:
            return "reject"
    
    def _calculate_priority(self, initiative: Initiative, composite: float,
                          scores: Dict[EvaluationDimension, EvaluationScore]) -> int:
        """Calculate priority (1-10, 10 highest)"""
        
        # Base priority from composite
        base_priority = int(composite * 10)
        
        # Boost for high novelty
        if scores[EvaluationDimension.NOVELTY_COEFFICIENT].raw_score > 0.8:
            base_priority += 1
            
        # Boost for urgent timing
        if hasattr(initiative, 'urgency') and initiative.urgency > 0.8:
            base_priority += 1
            
        # Penalty for poor ecological fit
        if scores[EvaluationDimension.ECOLOGICAL_FIT].raw_score < 0.5:
            base_priority -= 2
            
        return max(1, min(10, base_priority))
    
    def _generate_insights(self, initiative: Initiative,
                         scores: Dict[EvaluationDimension, EvaluationScore]) -> List[str]:
        """Generate insights from evaluation"""
        insights = []
        
        # Find standout dimensions
        high_scores = [(dim, score) for dim, score in scores.items() 
                      if score.raw_score > 0.8]
        low_scores = [(dim, score) for dim, score in scores.items() 
                     if score.raw_score < 0.4]
        
        if high_scores:
            insights.append(f"Exceptional {high_scores[0][0].value} suggests strong potential")
            
        if low_scores:
            insights.append(f"Low {low_scores[0][0].value} may require attention")
            
        # Check for interesting patterns
        if self._is_divergent_profile(scores):
            insights.append("Divergent evaluation profile suggests unique characteristics")
            
        # Historical comparison
        similar_success = self.pattern_recognizer.find_similar_successes(initiative)
        if similar_success:
            insights.append(f"Similar initiatives have {similar_success:.1f}% success rate")
            
        # Pattern insights
        pattern_insights = self.pattern_recognizer.get_pattern_insights(initiative)
        insights.extend(pattern_insights)
        
        return insights
    
    def _identify_risks(self, initiative: Initiative,
                       scores: Dict[EvaluationDimension, EvaluationScore]) -> List[str]:
        """Identify risk factors"""
        risks = []
        
        # Low ecological fit
        if scores[EvaluationDimension.ECOLOGICAL_FIT].raw_score < 0.5:
            risks.append("Poor timing or resource fit with current system state")
            
        # Too novel
        if scores[EvaluationDimension.NOVELTY_COEFFICIENT].raw_score > 0.9:
            risks.append("Extreme novelty may lack grounding or precedent")
            
        # Low confidence scores
        low_confidence = [dim for dim, score in scores.items() 
                         if score.confidence < 0.5]
        if low_confidence:
            risks.append(f"Uncertain evaluation in {len(low_confidence)} dimensions")
            
        # System overload
        if self.context.get_system_load() > 0.8:
            risks.append("System may be overloaded for new initiatives")
            
        # Zone saturation
        zone_balance = self.context.get_zone_distribution()
        if zone_balance.get(initiative.primary_zone, 0) > 0.4:
            risks.append(f"Zone '{initiative.primary_zone}' may be oversaturated")
            
        return risks
    
    async def _adapt_weights(self, evaluation: InitiativeEvaluation):
        """Adapt dimension weights based on outcomes"""
        # Would track initiative success and adjust weights to optimize future evaluations
        # For now, implement simple adaptation based on evaluation patterns
        
        if len(self.evaluation_history) > 10:
            # Analyze recent evaluations
            recent_evals = list(self.evaluation_history)[-10:]
            
            # If many high-scoring initiatives are being rejected due to ecological fit,
            # increase its weight
            rejected_high_scores = [e for e in recent_evals 
                                  if e.recommendation == "reject" and e.composite_score > 0.6]
            
            if len(rejected_high_scores) > 3:
                self.dimension_weights[EvaluationDimension.ECOLOGICAL_FIT] *= 1.1
                # Normalize weights
                total_weight = sum(self.dimension_weights.values())
                for dim in self.dimension_weights:
                    self.dimension_weights[dim] /= total_weight
    
    # Helper methods for evaluation dimensions
    
    def _calculate_cross_resonance(self, logic: float, myth: float, memory: float) -> float:
        """Calculate harmony between layer resonances"""
        # High when all are similar (harmony) or when there's creative tension
        variance = np.var([logic, myth, memory])
        if variance < 0.1:  # Harmony
            return 0.9
        elif variance > 0.3:  # Creative tension
            return 0.7
        else:
            return 0.5
            
    def _assess_metamorphic_potential(self, initiative: Initiative) -> float:
        """Assess potential for metamorphic transformation"""
        # Based on zone and semantic content
        metamorphic_zones = ["warp", "vortex", "drift"]
        if initiative.primary_zone in metamorphic_zones:
            return 0.8
        
        metamorphic_terms = ["transformation", "metamorphosis", "becoming", "evolution"]
        term_matches = sum(1 for term in metamorphic_terms if term in initiative.semantic_field)
        
        return min(1.0, 0.4 + (term_matches * 0.2))
        
    def _assess_catalytic_potential(self, initiative: Initiative) -> float:
        """Assess ability to catalyze other changes"""
        # High connectivity suggests catalytic potential
        base_score = min(1.0, len(initiative.potential_connections) / 5)
        
        # Boost for catalytic semantic content
        catalytic_terms = ["catalyst", "spark", "trigger", "amplify", "ignite"]
        catalytic_presence = sum(1 for term in catalytic_terms if term in initiative.semantic_field)
        
        return min(1.0, base_score + (catalytic_presence * 0.2))
        
    def _calculate_phase_shift_probability(self, initiative: Initiative) -> float:
        """Calculate probability of causing phase shift"""
        base_prob = 0.3
        
        # High magnitude will vectors are more likely to cause phase shifts
        if initiative.will_vector.magnitude > 0.8:
            base_prob += 0.4
        elif initiative.will_vector.magnitude > 0.6:
            base_prob += 0.2
            
        # Zone influence
        phase_shift_zones = ["warp", "vortex", "chaos"]
        if initiative.primary_zone in phase_shift_zones:
            base_prob += 0.3
            
        return min(1.0, base_prob)
        
    def _assess_balance_impact(self, initiative: Initiative, balance: Dict[str, float]) -> float:
        """Assess impact on system balance"""
        # Check if initiative helps restore balance or creates imbalance
        current_zone_activity = balance.get(initiative.primary_zone, 0)
        
        # Ideal is balanced distribution (~0.14 for 7 zones)
        ideal_balance = 1.0 / 7
        
        # If zone is under-represented, initiative helps balance
        if current_zone_activity < ideal_balance:
            return 0.8 + (ideal_balance - current_zone_activity) * 2
        # If zone is over-represented, initiative may create imbalance
        else:
            return max(0.2, 0.8 - (current_zone_activity - ideal_balance) * 2)
        
    def _assess_timing(self, initiative: Initiative) -> float:
        """Assess timing appropriateness"""
        base_score = 0.5
        
        # Check system load
        system_load = self.context.get_system_load()
        if system_load < 0.3:  # Low load, good time for new initiatives
            base_score += 0.3
        elif system_load > 0.8:  # High load, poor timing
            base_score -= 0.3
            
        # Check for urgent initiatives
        if hasattr(initiative, 'urgency') and initiative.urgency > 0.7:
            base_score += 0.2
            
        # Check zone timing
        zone_activity = len(self.context.zone_activities[initiative.primary_zone])
        if zone_activity == 0:  # Fresh zone
            base_score += 0.2
        elif zone_activity > 3:  # Oversaturated
            base_score -= 0.2
            
        return max(0.0, min(1.0, base_score))
        
    def _assess_generative_capacity(self, initiative: Initiative) -> float:
        """Assess capacity to generate new initiatives"""
        base_score = 0.5
        
        # Zones with high generative potential
        generative_zones = ["mesh", "synthesis", "vortex"]
        if initiative.primary_zone in generative_zones:
            base_score += 0.3
            
        # Semantic indicators of generativity
        generative_terms = ["generate", "create", "spawn", "branch", "multiply"]
        matches = sum(1 for term in generative_terms if term in initiative.semantic_field)
        base_score += min(0.3, matches * 0.1)
        
        return min(1.0, base_score)
        
    def _assess_multiplicity(self, initiative: Initiative) -> float:
        """Assess inherent multiplicity"""
        # Based on semantic field diversity
        base_score = min(1.0, len(initiative.semantic_field) / 10)
        
        # Boost for multiplicity-related terms
        multiplicity_terms = ["multiple", "diverse", "varied", "complex", "manifold"]
        matches = sum(1 for term in multiplicity_terms if term in initiative.semantic_field)
        
        return min(1.0, base_score + (matches * 0.1))
        
    def _assess_elegance(self, initiative: Initiative) -> float:
        """Assess conceptual elegance"""
        # Balance of simplicity and power
        complexity = len(initiative.semantic_field)
        magnitude = initiative.will_vector.magnitude
        
        # Sweet spot: moderate complexity with high magnitude
        if 3 <= complexity <= 6 and magnitude > 0.6:
            return 0.9
        elif complexity < 3:
            return 0.4  # Too simple
        elif complexity > 10:
            return 0.5  # Too complex
        else:
            return 0.7  # Reasonable
            
    def _assess_narrative_beauty(self, narrative: str) -> float:
        """Assess narrative aesthetic appeal"""
        if not narrative:
            return 0.3
            
        # Simple heuristics for narrative quality
        length_score = min(1.0, len(narrative) / 200)  # Optimal around 200 chars
        
        # Look for aesthetic language
        aesthetic_words = ["beautiful", "elegant", "harmony", "flow", "grace", "sublime"]
        aesthetic_presence = sum(1 for word in aesthetic_words if word in narrative.lower())
        aesthetic_score = min(0.5, aesthetic_presence * 0.2)
        
        # Coherence (simple measure)
        coherence = 0.7  # Placeholder for more sophisticated analysis
        
        return (length_score * 0.3) + (aesthetic_score * 0.3) + (coherence * 0.4)
        
    def _assess_symbolic_value(self, initiative: Initiative) -> float:
        """Assess symbolic/mythic value"""
        mythic_terms = ["transformation", "emergence", "becoming", "synthesis", "unity", "transcendence"]
        matches = sum(1 for term in mythic_terms if term in initiative.semantic_field)
        
        base_score = min(1.0, matches / 3)
        
        # Boost for archetypal patterns
        archetypal_patterns = ["journey", "quest", "return", "awakening", "integration"]
        archetypal_matches = sum(1 for pattern in archetypal_patterns 
                               if pattern in initiative.description.lower())
        
        return min(1.0, base_score + (archetypal_matches * 0.2))
        
    def _assess_harmonic_composition(self, initiative: Initiative) -> float:
        """Assess harmony of elements"""
        # Coherence between zone and semantic content
        zone_semantic_alignment = self._check_zone_semantic_alignment(initiative)
        
        # Will vector coherence
        will_coherence = initiative.will_vector.coherence
        
        # Balance of elements
        element_balance = self._assess_element_balance(initiative)
        
        return (zone_semantic_alignment * 0.4) + (will_coherence * 0.3) + (element_balance * 0.3)
        
    def _check_zone_semantic_alignment(self, initiative: Initiative) -> float:
        """Check alignment between zone and semantic content"""
        zone_semantic_map = {
            'warp': ['distortion', 'bend', 'shift', 'transform'],
            'vortex': ['spiral', 'draw', 'center', 'intensity'],
            'chaos': ['random', 'unpredictable', 'emergence', 'disorder'],
            'mesh': ['connection', 'network', 'integrate', 'weave'],
            'flow': ['movement', 'current', 'stream', 'fluid'],
            'stabilize': ['balance', 'steady', 'anchor', 'ground'],
            'anchor': ['root', 'foundation', 'secure', 'fixed']
        }
        
        expected_terms = zone_semantic_map.get(initiative.primary_zone, [])
        matches = sum(1 for term in expected_terms if term in initiative.semantic_field)
        
        return min(1.0, matches / max(1, len(expected_terms)))
        
    def _assess_element_balance(self, initiative: Initiative) -> float:
        """Assess balance of different elements"""
        # Check distribution across will vector zones
        zone_distribution = list(initiative.will_vector.direction.values())
        
        if not zone_distribution:
            return 0.5
            
        # Calculate entropy as measure of balance
        total = sum(zone_distribution)
        if total == 0:
            return 0.5
            
        probabilities = [x/total for x in zone_distribution if x > 0]
        
        if len(probabilities) <= 1:
            return 0.3  # Too concentrated
            
        entropy = -sum(p * math.log2(p) for p in probabilities)
        max_entropy = math.log2(len(probabilities))
        
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
        
    def _assess_complexity_threshold(self, initiative: Initiative) -> float:
        """Check if complex enough for emergence"""
        elements = len(initiative.semantic_field) + len(initiative.potential_connections)
        
        # Minimum complexity threshold for emergence
        if elements < 5:
            return 0.2
        elif elements < 10:
            return 0.6
        else:
            return min(1.0, elements / 15)
        
    def _calculate_interaction_density(self, initiative: Initiative) -> float:
        """Calculate potential interaction density"""
        connections = len(initiative.potential_connections)
        semantic_complexity = len(initiative.semantic_field)
        
        # Interaction potential is combination of connections and semantic richness
        base_density = min(1.0, connections / 8)
        semantic_factor = min(1.0, semantic_complexity / 10)
        
        return (base_density * 0.7) + (semantic_factor * 0.3)
        
    def _assess_feedback_loops(self, initiative: Initiative) -> float:
        """Assess feedback loop potential"""
        base_score = 0.5
        
        # Zones prone to feedback loops
        feedback_zones = ["vortex", "mesh", "chaos"]
        if initiative.primary_zone in feedback_zones:
            base_score += 0.3
            
        # Semantic indicators
        feedback_terms = ["feedback", "loop", "cycle", "recursive", "self-referential"]
        matches = sum(1 for term in feedback_terms if term in initiative.semantic_field)
        base_score += min(0.3, matches * 0.15)
        
        return min(1.0, base_score)
        
    def _detect_nonlinear_potential(self, initiative: Initiative) -> float:
        """Detect potential for nonlinear dynamics"""
        base_score = 0.3
        
        # High magnitude will vectors often lead to nonlinearity
        if initiative.will_vector.magnitude > 0.7:
            base_score += 0.4
            
        # Nonlinear indicators
        nonlinear_terms = ["exponential", "threshold", "cascade", "amplify", "nonlinear"]
        matches = sum(1 for term in nonlinear_terms if term in initiative.semantic_field)
        base_score += min(0.3, matches * 0.1)
        
        return min(1.0, base_score)
        
    def _is_divergent_profile(self, scores: Dict[EvaluationDimension, EvaluationScore]) -> bool:
        """Check if evaluation profile is divergent"""
        raw_scores = [s.raw_score for s in scores.values()]
        return np.std(raw_scores) > 0.3


# === Autonomous Project Monitor ===

class AutonomousProjectMonitor:
    """
    Monitors active autonomous projects and provides real-time insights
    """
    
    def __init__(self, will_engine, executor):
        self.will_engine = will_engine
        self.executor = executor
        self.monitoring_data = defaultdict(list)
        self.alerts = deque(maxlen=100)
        self.health_metrics = {}
        self.system_metrics = defaultdict(list)
        self.pattern_detector = MonitoringPatternDetector()
        
    async def monitor_projects(self):
        """Main monitoring loop"""
        while True:
            active_projects = getattr(self.executor, 'active_executions', {})
            
            for project_id, execution_state in active_projects.items():
                # Update monitoring data
                await self._update_project_metrics(project_id, execution_state)
                
                # Check for alerts
                alerts = await self._check_alerts(project_id, execution_state)
                self.alerts.extend(alerts)
                
                # Update health metrics
                self.health_metrics[project_id] = self._calculate_health(execution_state)
                
            # System-wide analysis
            await self._analyze_system_patterns()
            
            # Pattern detection
            await self._detect_emerging_patterns()
            
            await asyncio.sleep(1)  # Monitor frequency
    
    async def _update_project_metrics(self, project_id: str, execution_state: ExecutionState):
        """Update metrics for a project"""
        metrics = {
            "timestamp": datetime.now(),
            "phase": execution_state.current_phase.value,
            "intensity": execution_state.intensity_level,
            "progress": execution_state.calculate_milestone_progress(),
            "obstacles": len(execution_state.obstacle_history),
            "movement_flow": execution_state.assess_movement_flow(),
            "emergence_events": len(execution_state.emergence_events),
            "phase_stability": self._calculate_phase_stability(execution_state)
        }
        
        self.monitoring_data[project_id].append(metrics)
        
        # Keep only recent data (last 1000 points)
        if len(self.monitoring_data[project_id]) > 1000:
            self.monitoring_data[project_id] = self.monitoring_data[project_id][-1000:]
    
    async def _check_alerts(self, project_id: str, execution_state: ExecutionState) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        
        # Low intensity alert
        if execution_state.intensity_level < 0.2:
            alerts.append({
                "project_id": project_id,
                "type": "low_intensity",
                "severity": "warning",
                "message": f"Project {project_id} intensity critically low",
                "timestamp": datetime.now(),
                "recommendations": ["Review project goals", "Inject new energy", "Consider hibernation"]
            })
            
        # Stalled progress
        if len(self.monitoring_data[project_id]) > 10:
            recent_progress = [m["progress"] for m in self.monitoring_data[project_id][-10:]]
            if np.std(recent_progress) < 0.01:  # No progress change
                alerts.append({
                    "project_id": project_id,
                    "type": "stalled",
                    "severity": "warning",
                    "message": f"Project {project_id} appears stalled",
                    "timestamp": datetime.now(),
                    "recommendations": ["Identify obstacles", "Restructure approach", "Seek new perspectives"]
                })
                
        # Obstacle accumulation
        if len(execution_state.obstacle_history) > 5:
            alerts.append({
                "project_id": project_id,
                "type": "obstacle_overload",
                "severity": "info",
                "message": f"Project {project_id} encountering multiple obstacles",
                "timestamp": datetime.now(),
                "recommendations": ["Obstacle pattern analysis", "Strategy revision", "Resource reallocation"]
            })
            
        # Phase instability
        phase_stability = self._calculate_phase_stability(execution_state)
        if phase_stability < 0.3:
            alerts.append({
                "project_id": project_id,
                "type": "phase_instability",
                "severity": "info",
                "message": f"Project {project_id} showing phase instability",
                "timestamp": datetime.now(),
                "recommendations": ["Stabilize current phase", "Clarify objectives", "Strengthen foundations"]
            })
            
        # Emergence opportunity
        if len(execution_state.emergence_events) > 3:
            alerts.append({
                "project_id": project_id,
                "type": "emergence_opportunity",
                "severity": "positive",
                "message": f"Project {project_id} showing high emergence potential",
                "timestamp": datetime.now(),
                "recommendations": ["Nurture emergent properties", "Amplify successful patterns", "Document insights"]
            })
            
        return alerts
    
    def _calculate_health(self, execution_state: ExecutionState) -> float:
        """Calculate overall project health (0-1)"""
        factors = {
            "intensity": min(1.0, execution_state.intensity_level * 2),
            "progress": execution_state.calculate_milestone_progress(),
            "flow": 1.0 if execution_state.assess_movement_flow() == "flowing" else 
                   0.7 if execution_state.assess_movement_flow() == "steady" else 0.3,
            "obstacles": max(0, 1.0 - (len(execution_state.obstacle_history) / 10)),
            "emergence": min(1.0, len(execution_state.emergence_events) / 5),
            "phase_stability": self._calculate_phase_stability(execution_state)
        }
        
        # Weighted average
        weights = {
            "intensity": 0.25,
            "progress": 0.25,
            "flow": 0.20,
            "obstacles": 0.15,
            "emergence": 0.10,
            "phase_stability": 0.05
        }
        
        return sum(factors[key] * weights[key] for key in factors)
    
    def _calculate_phase_stability(self, execution_state: ExecutionState) -> float:
        """Calculate phase stability metric"""
        if len(execution_state.phase_transitions) < 2:
            return 1.0  # New project, assume stable
            
        # Look at frequency of phase transitions
        recent_transitions = [t for t in execution_state.phase_transitions 
                            if (datetime.now() - t.get('timestamp', datetime.now())).days < 7]
        
        if len(recent_transitions) > 3:
            return 0.2  # Very unstable
        elif len(recent_transitions) > 1:
            return 0.6  # Somewhat unstable
        else:
            return 1.0  # Stable
    
    async def _analyze_system_patterns(self):
        """Analyze patterns across all projects"""
        if not self.health_metrics:
            return
            
        # Overall system health
        system_health = sum(self.health_metrics.values()) / len(self.health_metrics)
        
        # Phase distribution
        phase_counts = defaultdict(int)
        active_executions = getattr(self.executor, 'active_executions', {})
        
        for project_id, execution_state in active_executions.items():
            phase_counts[execution_state.current_phase.value] += 1
            
        # System flow analysis
        flow_states = []
        for project_id, execution_state in active_executions.items():
            flow_states.append(execution_state.assess_movement_flow())
            
        flow_distribution = {
            "flowing": flow_states.count("flowing"),
            "steady": flow_states.count("steady"),
            "stagnant": flow_states.count("stagnant"),
            "blocked": flow_states.count("blocked")
        }
        
        # Store system analysis
        self.system_analysis = {
            "timestamp": datetime.now(),
            "system_health": system_health,
            "active_projects": len(active_executions),
            "phase_distribution": dict(phase_counts),
            "flow_distribution": flow_distribution,
            "recent_alerts": len([a for a in self.alerts 
                                if (datetime.now() - a.get('timestamp', datetime.now())).hours < 1]),
            "emergence_activity": sum(len(es.emergence_events) for es in active_executions.values())
        }
        
        # Store in history
        self.system_metrics['health'].append({
            'timestamp': datetime.now(),
            'value': system_health
        })
        
        self.system_metrics['activity'].append({
            'timestamp': datetime.now(),
            'value': len(active_executions)
        })
    
    async def _detect_emerging_patterns(self):
        """Detect emerging patterns across projects"""
        await self.pattern_detector.analyze_patterns(
            self.monitoring_data, 
            getattr(self.executor, 'active_executions', {})
        )
    
    def get_project_summary(self, project_id: str) -> Dict:
        """Get summary of project status"""
        active_executions = getattr(self.executor, 'active_executions', {})
        
        if project_id not in active_executions:
            return {"error": "Project not found"}
            
        execution_state = active_executions[project_id]
        recent_metrics = self.monitoring_data[project_id][-1] if self.monitoring_data[project_id] else {}
        
        # Calculate trends
        trends = self._calculate_trends(project_id)
        
        # Get project alerts
        project_alerts = [a for a in self.alerts if a.get("project_id") == project_id]
        recent_alerts = [a for a in project_alerts 
                        if (datetime.now() - a.get('timestamp', datetime.now())).hours < 24]
        
        return {
            "project_id": project_id,
            "current_phase": execution_state.current_phase.value,
            "health": self.health_metrics.get(project_id, 0),
            "progress": execution_state.calculate_milestone_progress(),
            "intensity": execution_state.intensity_level,
            "flow_state": execution_state.assess_movement_flow(),
            "recent_metrics": recent_metrics,
            "trends": trends,
            "alerts": recent_alerts,
            "emergence_events": len(execution_state.emergence_events),
            "obstacle_count": len(execution_state.obstacle_history)
        }
    
    def _calculate_trends(self, project_id: str) -> Dict:
        """Calculate trends for project metrics"""
        if project_id not in self.monitoring_data or len(self.monitoring_data[project_id]) < 5:
            return {}
            
        data = self.monitoring_data[project_id][-10:]  # Last 10 data points
        
        trends = {}
        for metric in ['intensity', 'progress', 'obstacles']:
            values = [d.get(metric, 0) for d in data if metric in d]
            if len(values) >= 3:
                # Simple linear trend
                x = list(range(len(values)))
                slope = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0
                
                if slope > 0.01:
                    trends[metric] = "increasing"
                elif slope < -0.01:
                    trends[metric] = "decreasing"
                else:
                    trends[metric] = "stable"
            else:
                trends[metric] = "insufficient_data"
                
        return trends
    
    def get_system_dashboard(self) -> Dict:
        """Get system-wide dashboard data"""
        # Recent system health trend
        recent_health = self.system_metrics['health'][-10:] if self.system_metrics['health'] else []
        health_trend = "stable"
        
        if len(recent_health) >= 3:
            values = [h['value'] for h in recent_health]
            slope = np.polyfit(range(len(values)), values, 1)[0]
            if slope > 0.01:
                health_trend = "improving"
            elif slope < -0.01:
                health_trend = "declining"
        
        # Alert summary
        recent_alerts = [a for a in self.alerts 
                        if (datetime.now() - a.get('timestamp', datetime.now())).hours < 24]
        
        alert_summary = defaultdict(int)
        for alert in recent_alerts:
            alert_summary[alert.get('type', 'unknown')] += 1
        
        return {
            "system_analysis": getattr(self, 'system_analysis', {}),
            "active_projects": len(getattr(self.executor, 'active_executions', {})),
            "health_metrics": self.health_metrics,
            "health_trend": health_trend,
            "recent_alerts": recent_alerts[-10:],
            "alert_summary": dict(alert_summary),
            "project_phases": self._get_phase_summary(),
            "system_patterns": self.pattern_detector.get_pattern_summary(),
            "performance_metrics": self._get_performance_metrics()
        }
    
    def _get_phase_summary(self) -> Dict:
        """Get summary of projects by phase"""
        summary = defaultdict(list)
        active_executions = getattr(self.executor, 'active_executions', {})
        
        for project_id, execution_state in active_executions.items():
            summary[execution_state.current_phase.value].append({
                "project_id": project_id,
                "health": self.health_metrics.get(project_id, 0),
                "intensity": execution_state.intensity_level,
                "flow_state": execution_state.assess_movement_flow()
            })
        return dict(summary)
    
    def _get_performance_metrics(self) -> Dict:
        """Get system performance metrics"""
        active_executions = getattr(self.executor, 'active_executions', {})
        
        if not active_executions:
            return {}
            
        # Average metrics
        avg_health = np.mean(list(self.health_metrics.values())) if self.health_metrics else 0
        avg_intensity = np.mean([es.intensity_level for es in active_executions.values()])
        avg_progress = np.mean([es.calculate_milestone_progress() for es in active_executions.values()])
        
        # System efficiency
        total_obstacles = sum(len(es.obstacle_history) for es in active_executions.values())
        total_emergence = sum(len(es.emergence_events) for es in active_executions.values())
        
        efficiency = max(0, 1 - (total_obstacles / max(1, len(active_executions) * 10)))
        emergence_rate = total_emergence / max(1, len(active_executions))
        
        return {
            "average_health": avg_health,
            "average_intensity": avg_intensity,
            "average_progress": avg_progress,
            "system_efficiency": efficiency,
            "emergence_rate": emergence_rate,
            "total_projects": len(active_executions),
            "total_obstacles": total_obstacles,
            "total_emergence_events": total_emergence
        }


class MonitoringPatternDetector:
    """Detects patterns in monitoring data"""
    
    def __init__(self):
        self.patterns = {}
        self.pattern_history = deque(maxlen=100)
        
    async def analyze_patterns(self, monitoring_data: Dict, active_executions: Dict):
        """Analyze patterns in monitoring data"""
        
        # Cross-project patterns
        cross_patterns = self._detect_cross_project_patterns(monitoring_data, active_executions)
        
        # Temporal patterns
        temporal_patterns = self._detect_temporal_patterns(monitoring_data)
        
        # Phase transition patterns
        phase_patterns = self._detect_phase_patterns(active_executions)
        
        # Store patterns
        pattern_analysis = {
            "timestamp": datetime.now(),
            "cross_project": cross_patterns,
            "temporal": temporal_patterns,
            "phase_transitions": phase_patterns
        }
        
        self.pattern_history.append(pattern_analysis)
        self.patterns = pattern_analysis
    
    def _detect_cross_project_patterns(self, monitoring_data: Dict, active_executions: Dict) -> List[Dict]:
        """Detect patterns across multiple projects"""
        patterns = []
        
        if len(active_executions) < 2:
            return patterns
            
        # Synchronization patterns
        sync_pattern = self._detect_synchronization(monitoring_data)
        if sync_pattern:
            patterns.append(sync_pattern)
            
        # Complementary patterns
        comp_pattern = self._detect_complementary_behavior(monitoring_data)
        if comp_pattern:
            patterns.append(comp_pattern)
            
        return patterns
    
    def _detect_synchronization(self, monitoring_data: Dict) -> Optional[Dict]:
        """Detect if projects are synchronizing"""
        if len(monitoring_data) < 2:
            return None
            
        # Get recent intensity data for all projects
        project_intensities = {}
        for project_id, data in monitoring_data.items():
            if len(data) >= 5:
                recent_intensities = [d.get('intensity', 0) for d in data[-5:]]
                project_intensities[project_id] = recent_intensities
        
        if len(project_intensities) < 2:
            return None
            
        # Calculate correlations
        correlations = []
        project_pairs = []
        
        projects = list(project_intensities.keys())
        for i in range(len(projects)):
            for j in range(i+1, len(projects)):
                proj1, proj2 = projects[i], projects[j]
                data1, data2 = project_intensities[proj1], project_intensities[proj2]
                
                if len(data1) == len(data2) and len(data1) >= 3:
                    correlation = np.corrcoef(data1, data2)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
                        project_pairs.append((proj1, proj2))
        
        if correlations and max(correlations) > 0.7:
            max_corr_idx = correlations.index(max(correlations))
            return {
                "type": "synchronization",
                "strength": max(correlations),
                "projects": project_pairs[max_corr_idx],
                "description": f"Projects {project_pairs[max_corr_idx][0]} and {project_pairs[max_corr_idx][1]} showing synchronization"
            }
        
        return None
    
    def _detect_complementary_behavior(self, monitoring_data: Dict) -> Optional[Dict]:
        """Detect complementary behavior between projects"""
        # Look for anti-correlation in phases or activities
        # This is a simplified version - could be much more sophisticated
        
        if len(monitoring_data) < 2:
            return None
            
        # Check for phase complementarity
        phase_distributions = {}
        for project_id, data in monitoring_data.items():
            if data:
                recent_phases = [d.get('phase', '') for d in data[-3:]]
                phase_distributions[project_id] = recent_phases
        
        # Simple complementarity check
        if len(phase_distributions) >= 2:
            projects = list(phase_distributions.keys())
            for i in range(len(projects)):
                for j in range(i+1, len(projects)):
                    phases1 = phase_distributions[projects[i]]
                    phases2 = phase_distributions[projects[j]]
                    
                    # Check if one is in high-energy phase while other is in low-energy
                    high_energy_phases = ["intensification", "unfolding"]
                    low_energy_phases = ["crystallization", "integration"]
                    
                    if (any(p in high_energy_phases for p in phases1) and 
                        any(p in low_energy_phases for p in phases2)):
                        return {
                            "type": "complementary",
                            "projects": (projects[i], projects[j]),
                            "description": f"Projects {projects[i]} and {projects[j]} showing complementary phase patterns"
                        }
        
        return None
    
    def _detect_temporal_patterns(self, monitoring_data: Dict) -> List[Dict]:
        """Detect temporal patterns"""
        patterns = []
        
        # Cyclical patterns
        for project_id, data in monitoring_data.items():
            if len(data) >= 10:
                cyclical = self._detect_cyclical_pattern(data)
                if cyclical:
                    patterns.append({
                        "type": "cyclical",
                        "project_id": project_id,
                        "period": cyclical,
                        "description": f"Project {project_id} showing cyclical pattern with period ~{cyclical}"
                    })
        
        return patterns
    
    def _detect_cyclical_pattern(self, data: List[Dict]) -> Optional[int]:
        """Detect cyclical patterns in project data"""
        if len(data) < 10:
            return None
            
        # Look for cycles in intensity
        intensities = [d.get('intensity', 0) for d in data[-20:]]  # Last 20 points
        
        # Simple autocorrelation-based cycle detection
        for period in range(3, 8):  # Look for cycles of 3-7 time units
            if len(intensities) >= period * 2:
                correlations = []
                for offset in range(1, len(intensities) - period):
                    if offset + period < len(intensities):
                        corr = np.corrcoef(
                            intensities[offset:offset+period],
                            intensities[offset+period:offset+2*period]
                        )[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                
                if correlations and max(correlations) > 0.6:
                    return period
        
        return None
    
    def _detect_phase_patterns(self, active_executions: Dict) -> List[Dict]:
        """Detect phase transition patterns"""
        patterns = []
        
        # Rapid phase transitions
        for project_id, execution_state in active_executions.items():
            if len(execution_state.phase_transitions) >= 3:
                # Check transition frequency
                recent_transitions = execution_state.phase_transitions[-3:]
                time_diffs = []
                
                for i in range(1, len(recent_transitions)):
                    if 'timestamp' in recent_transitions[i] and 'timestamp' in recent_transitions[i-1]:
                        diff = recent_transitions[i]['timestamp'] - recent_transitions[i-1]['timestamp']
                        time_diffs.append(diff.total_seconds() / 3600)  # Convert to hours
                
                if time_diffs and max(time_diffs) < 24:  # Transitions within 24 hours
                    patterns.append({
                        "type": "rapid_transitions",
                        "project_id": project_id,
                        "transition_rate": len(time_diffs) / sum(time_diffs) if sum(time_diffs) > 0 else 0,
                        "description": f"Project {project_id} showing rapid phase transitions"
                    })
        
        return patterns
    
    def get_pattern_summary(self) -> Dict:
        """Get summary of detected patterns"""
        if not self.patterns:
            return {}
            
        return {
            "timestamp": self.patterns.get("timestamp"),
            "total_patterns": (
                len(self.patterns.get("cross_project", [])) + 
                len(self.patterns.get("temporal", [])) + 
                len(self.patterns.get("phase_transitions", []))
            ),
            "cross_project_patterns": len(self.patterns.get("cross_project", [])),
            "temporal_patterns": len(self.patterns.get("temporal", [])),
            "phase_patterns": len(self.patterns.get("phase_transitions", [])),
            "recent_patterns": self.patterns
        }


# === Main Integration Class ===

class AutonomousSystemManager:
    """
    Main class that integrates evaluation and monitoring
    """
    
    def __init__(self):
        self.context = SystemContext()
        self.evaluator = AdvancedInitiativeEvaluator(self.context)
        self.monitor = AutonomousProjectMonitor(None, self)  # Will engine and executor would be injected
        self.active_executions = {}  # Simulated for now
        
        # Simulation data
        self._setup_simulation_data()
        
    def _setup_simulation_data(self):
        """Setup some simulation data for demonstration"""
        # Create sample initiatives
        sample_initiative = Initiative(
            id="init_001",
            title="Emergence Synthesis Protocol",
            description="A protocol for synthesizing emergent patterns across multiple domains",
            primary_zone="mesh",
            semantic_field=["synthesis", "emergence", "protocol", "integration", "patterns"],
            will_vector=WillVector(
                direction={"mesh": 0.6, "synthesis": 0.3, "stabilize": 0.1},
                magnitude=0.75,
                coherence=0.8
            ),
            emergence_confidence=0.7,
            attractor_strength=0.6,
            potential_connections=["pattern_recognition", "data_fusion", "collaborative_AI"],
            narrative="An elegant approach to weaving together emergent insights from disparate sources into coherent new understanding."
        )
        
        # Sample execution state
        sample_execution = ExecutionState(
            project_id="init_001",
            current_phase=ExecutionPhase.UNFOLDING,
            intensity_level=0.6,
            milestone_progress={"research": 0.7, "synthesis": 0.4, "validation": 0.2},
            obstacle_history=[
                {"type": "resource_constraint", "timestamp": datetime.now() - timedelta(days=2)},
                {"type": "complexity_overload", "timestamp": datetime.now() - timedelta(days=1)}
            ],
            emergence_events=[
                {"type": "unexpected_connection", "timestamp": datetime.now() - timedelta(hours=12)},
                {"type": "pattern_crystallization", "timestamp": datetime.now() - timedelta(hours=6)}
            ],
            phase_transitions=[
                {"from": "initiation", "to": "unfolding", "timestamp": datetime.now() - timedelta(days=3)}
            ]
        )
        
        self.active_executions["init_001"] = sample_execution
        self.context.active_projects["init_001"] = sample_initiative
        
    async def evaluate_initiative(self, initiative: Initiative) -> InitiativeEvaluation:
        """Evaluate a new initiative"""
        return await self.evaluator.evaluate_initiative(initiative)
        
    async def start_monitoring(self):
        """Start the monitoring system"""
        await self.monitor.monitor_projects()
        
    def get_system_status(self) -> Dict:
        """Get complete system status"""
        return {
            "evaluator_status": {
                "evaluation_history_size": len(self.evaluator.evaluation_history),
                "pattern_library_size": len(self.evaluator.pattern_recognizer.pattern_library),
                "dimension_weights": self.evaluator.dimension_weights
            },
            "monitor_status": self.monitor.get_system_dashboard(),
            "active_projects": len(self.active_executions),
            "system_context": {
                "active_projects": len(self.context.active_projects),
                "zone_distribution": self.context.get_zone_distribution(),
                "system_load": self.context.get_system_load()
            }
        }


# === Example Usage ===

async def main():
    """Example usage of the system"""
    
    # Initialize the system
    system = AutonomousSystemManager()
    
    # Create a new initiative to evaluate
    new_initiative = Initiative(
        id="init_002", 
        title="Quantum Narrative Bridge",
        description="Bridging quantum computing concepts with narrative structures for enhanced AI storytelling",
        primary_zone="vortex",
        semantic_field=["quantum", "narrative", "bridge", "storytelling", "enhancement"],
        will_vector=WillVector(
            direction={"vortex": 0.5, "mesh": 0.3, "chaos": 0.2},
            magnitude=0.85,
            coherence=0.7
        ),
        emergence_confidence=0.8,
        attractor_strength=0.7,
        potential_connections=["quantum_AI", "narrative_systems", "creative_algorithms"],
        narrative="A revolutionary fusion of quantum principles with storytelling that could transform how AI generates and understands narratives."
    )
    
    # Evaluate the initiative
    print("Evaluating initiative...")
    evaluation = await system.evaluate_initiative(new_initiative)
    
    print(f"\nEvaluation Results for: {new_initiative.title}")
    print(f"Composite Score: {evaluation.composite_score:.3f}")
    print(f"Recommendation: {evaluation.recommendation}")
    print(f"Priority: {evaluation.priority}/10")
    print(f"\nInsights:")
    for insight in evaluation.insights:
        print(f"  - {insight}")
    
    print(f"\nRisk Factors:")
    for risk in evaluation.risk_factors:
        print(f"  - {risk}")
        
    print(f"\nSynergy Opportunities:")
    for synergy in evaluation.synergy_opportunities:
        print(f"  - {synergy}")
    
    # Get system status
    print(f"\n" + "="*50)
    print("SYSTEM STATUS")
    print("="*50)
    
    status = system.get_system_status()
    print(f"Active Projects: {status['active_projects']}")
    print(f"Pattern Library Size: {status['evaluator_status']['pattern_library_size']}")
    print(f"System Load: {status['system_context']['system_load']:.2f}")
    
    # Simulate some monitoring
    print(f"\nProject Summary for init_001:")
    project_summary = system.monitor.get_project_summary("init_001")
    print(f"  Health: {project_summary['health']:.2f}")
    print(f"  Progress: {project_summary['progress']:.2f}")
    print(f"  Flow State: {project_summary['flow_state']}")
    print(f"  Current Phase: {project_summary['current_phase']}")


if __name__ == "__main__":
    asyncio.run(main())
