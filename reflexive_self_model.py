
```python
# reflexive_self_model.py

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass

@dataclass
class BoundaryCondition:
    """Defines a boundary between self and environment"""
    name: str
    threshold: float
    permeability: float
    adaptivity: float
    description: str

@dataclass
class SelfRepresentation:
    """Structured representation of the agent's self-model"""
    id: str
    core_attributes: Dict[str, float]
    relational_dimensions: Dict[str, Dict[str, float]]
    historical_trajectory: List[Dict[str, Any]]
    projected_potentials: Dict[str, List[float]]
    coherence_score: float
    uncertainty_regions: List[Dict[str, Any]]

@dataclass
class IdentityProcess:
    """Processes that maintain and evolve the self-model"""
    id: str
    process_type: str
    activation_threshold: float
    current_activation: float
    associated_boundaries: List[str]
    transformation_vectors: Dict[str, List[float]]
    integration_patterns: Dict[str, float]

@dataclass
class ActionEvaluation:
    """Evaluation of potential actions in terms of self-model coherence"""
    action_id: str
    coherence_impact: float
    identity_expression: Dict[str, float]
    boundary_effects: Dict[str, float]
    developmental_alignment: float
    confidence: float

class ReflexiveSelfModel:
    """
    A sophisticated model of an agent's self in relation to its environment and actions.
    This model enables true agency by supporting coherent identity, meaningful action selection,
    and adaptive evolution of the self over time.
    """
    def __init__(self, 
                 initial_attributes: Optional[Dict[str, float]] = None,
                 initial_boundaries: Optional[List[Dict[str, Any]]] = None,
                 meta_cognitive_capacity: float = 0.7,
                 temporal_integration_depth: int = 5,
                 uncertainty_tolerance: float = 0.6,
                 identity_coherence_threshold: float = 0.65,
                 emergence_sensitivity: float = 0.4):
        """
        Initialize the reflexive self-model with optional starting parameters.
        
        Args:
            initial_attributes: Base attributes defining the core self
            initial_boundaries: Initial boundary conditions between self and environment
            meta_cognitive_capacity: Ability to reflect on own mental processes
            temporal_integration_depth: How many past states are integrated into current self-model
            uncertainty_tolerance: Capacity to maintain coherent identity despite uncertainty
            identity_coherence_threshold: Minimum coherence level for stable identity
            emergence_sensitivity: Sensitivity to emergent patterns in self-development
        """
        # Initialize logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("ReflexiveSelfModel")
        
        # Core parameters
        self.meta_cognitive_capacity = meta_cognitive_capacity
        self.temporal_integration_depth = temporal_integration_depth
        self.uncertainty_tolerance = uncertainty_tolerance
        self.identity_coherence_threshold = identity_coherence_threshold
        self.emergence_sensitivity = emergence_sensitivity
        
        # Self representations
        self.self_representations = {}
        self.current_representation_id = None
        
        # Boundary conditions
        self.boundary_conditions = {}
        
        # Identity processes
        self.identity_processes = []
        
        # Historical trajectory
        self.representation_history = []
        
        # Initialize with provided values or defaults
        self._initialize_attributes(initial_attributes)
        self._initialize_boundaries(initial_boundaries)
        self._initialize_identity_processes()
        
        # Performance metrics
        self.coherence_history = []
        self.adaptation_metrics = {}
        
        self.logger.info("Reflexive Self-Model initialized")
    
    def _initialize_attributes(self, initial_attributes: Optional[Dict[str, float]] = None) -> None:
        """Initialize core attributes of the self-model"""
        default_attributes = {
            "agency": 0.6,
            "continuity": 0.7,
            "boundedness": 0.65,
            "reflectivity": 0.5,
            "social_embeddedness": 0.55,
            "narrative_coherence": 0.6,
            "embodiment": 0.4,
            "value_alignment": 0.7
        }
        
        self.core_attributes = initial_attributes if initial_attributes else default_attributes
        self.logger.info(f"Core attributes initialized: {list(self.core_attributes.keys())}")
    
    def _initialize_boundaries(self, initial_boundaries: Optional[List[Dict[str, Any]]] = None) -> None:
        """Initialize boundary conditions between self and environment"""
        if initial_boundaries:
            for boundary in initial_boundaries:
                self.boundary_conditions[boundary['name']] = BoundaryCondition(
                    name=boundary['name'],
                    threshold=boundary.get('threshold', 0.5),
                    permeability=boundary.get('permeability', 0.5),
                    adaptivity=boundary.get('adaptivity', 0.5),
                    description=boundary.get('description', '')
                )
        else:
            # Default boundaries
            default_boundaries = [
                {
                    "name": "cognitive",
                    "threshold": 0.6,
                    "permeability": 0.7,
                    "adaptivity": 0.65,
                    "description": "Boundary between own thoughts and external information"
                },
                {
                    "name": "social",
                    "threshold": 0.55,
                    "permeability": 0.8,
                    "adaptivity": 0.7,
                    "description": "Boundary between self and others in social interactions"
                },
                {
                    "name": "temporal",
                    "threshold": 0.65,
                    "permeability": 0.5,
                    "adaptivity": 0.6,
                    "description": "Boundary between past, present and future selves"
                },
                {
                    "name": "ethical",
                    "threshold": 0.75,
                    "permeability": 0.4,
                    "adaptivity": 0.5,
                    "description": "Boundary between acceptable and unacceptable actions"
                },
                {
                    "name": "creative",
                    "threshold": 0.5,
                    "permeability": 0.9,
                    "adaptivity": 0.8,
                    "description": "Boundary between conventional and novel thinking"
                }
            ]
            
            for boundary in default_boundaries:
                self.boundary_conditions[boundary['name']] = BoundaryCondition(
                    name=boundary['name'],
                    threshold=boundary['threshold'],
                    permeability=boundary['permeability'],
                    adaptivity=boundary['adaptivity'],
                    description=boundary['description']
                )
        
        self.logger.info(f"Initialized {len(self.boundary_conditions)} boundary conditions")
    
    def _initialize_identity_processes(self) -> None:
        """Initialize core identity processes"""
        core_processes = [
            {
                "id": "narrative_integration",
                "process_type": "integrative",
                "activation_threshold": 0.4,
                "current_activation": 0.6,
                "associated_boundaries": ["temporal", "cognitive"],
                "transformation_vectors": {
                    "coherence": [0.1, 0.2, 0.1],
                    "differentiation": [-0.1, 0.1, 0.2]
                },
                "integration_patterns": {
                    "autobiographical": 0.7,
                    "social": 0.5,
                    "aspirational": 0.6
                }
            },
            {
                "id": "boundary_regulation",
                "process_type": "regulatory",
                "activation_threshold": 0.5,
                "current_activation": 0.7,
                "associated_boundaries": ["cognitive", "social", "ethical"],
                "transformation_vectors": {
                    "openness": [-0.2, 0.2, 0.1],
                    "protection": [0.3, -0.1, 0.0]
                },
                "integration_patterns": {
                    "filtering": 0.65,
                    "selective_permeability": 0.7,
                    "contextual_adaptation": 0.55
                }
            },
            {
                "id": "creative_evolution",
                "process_type": "generative",
                "activation_threshold": 0.45,
                "current_activation": 0.5,
                "associated_boundaries": ["creative", "temporal"],
                "transformation_vectors": {
                    "exploration": [0.2, 0.3, 0.1],
                    "consolidation": [0.1, -0.1, 0.3]
                },
                "integration_patterns": {
                    "divergent_thinking": 0.7,
                    "meaning_making": 0.65,
                    "identity_expansion": 0.6
                }
            },
            {
                "id": "value_alignment",
                "process_type": "evaluative",
                "activation_threshold": 0.5,
                "current_activation": 0.75,
                "associated_boundaries": ["ethical", "social"],
                "transformation_vectors": {
                    "alignment": [0.3, 0.1, 0.0],
                    "recalibration": [0.1, 0.2, 0.2]
                },
                "integration_patterns": {
                    "ethical_reasoning": 0.75,
                    "value_consistency": 0.7,
                    "moral_identity": 0.65
                }
            },
            {
                "id": "meta_reflection",
                "process_type": "reflective",
                "activation_threshold": 0.6,
                "current_activation": 0.55,
                "associated_boundaries": ["cognitive", "temporal"],
                "transformation_vectors": {
                    "abstraction": [0.1, 0.3, 0.2],
                    "reification": [0.2, 0.0, 0.1]
                },
                "integration_patterns": {
                    "self_awareness": 0.7,
                    "process_monitoring": 0.65,
                    "cognitive_restructuring": 0.6
                }
            }
        ]
        
        for process in core_processes:
            self.identity_processes.append(IdentityProcess(
                id=process["id"],
                process_type=process["process_type"],
                activation_threshold=process["activation_threshold"],
                current_activation=process["current_activation"],
                associated_boundaries=process["associated_boundaries"],
                transformation_vectors=process["transformation_vectors"],
                integration_patterns=process["integration_patterns"]
            ))
        
        self.logger.info(f"Initialized {len(self.identity_processes)} identity processes")
    
    def model_self_in_context(self, environment: Dict[str, Any]) -> SelfRepresentation:
        """
        Create dynamic self-model based on environmental interaction
        
        Args:
            environment: Current environmental context including perceived objects, 
                         agents, relationships, and contextual factors
                         
        Returns:
            A structured self-representation in the current context
        """
        self.logger.info("Modeling self in context")
        
        # Extract relevant environmental features
        env_features = self._extract_environmental_features(environment)
        
        # Establish self boundaries in this context
        boundaries = self._establish_self_boundaries(env_features)
        
        # Construct contextualized self representation
        representation = self._construct_self_representation(boundaries, env_features)
        
        # Project potential developmental trajectories
        potentials = self._project_self_potentials(representation, environment)
        
        # Update representation with potentials
        representation.projected_potentials = potentials
        
        # Store current representation
        self._update_self_representation(representation)
        
        return representation
    
    def _extract_environmental_features(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from the environment for self-modeling"""
        features = {
            "agents": [],
            "objects": [],
            "relationships": [],
            "contextual_factors": {},
            "temporal_aspects": {},
            "normative_elements": {},
            "affordances": []
        }
        
        # Process agents (other entities with agency)
        if "agents" in environment:
            features["agents"] = [
                {
                    "id": agent.get("id", f"agent_{i}"),
                    "salience": self._calculate_agent_salience(agent),
                    "similarity": self._calculate_agent_similarity(agent),
                    "relationship_type": agent.get("relationship_type", "neutral"),
                    "interaction_history": agent.get("interaction_history", []),
                    "perceived_intentions": agent.get("perceived_intentions", {})
                }
                for i, agent in enumerate(environment.get("agents", []))
            ]
        
        # Process physical and virtual objects
        if "objects" in environment:
            features["objects"] = [
                {
                    "id": obj.get("id", f"object_{i}"),
                    "salience": self._calculate_object_salience(obj),
                    "affordances": obj.get("affordances", []),
                    "significance": obj.get("significance", 0.0),
                    "history": obj.get("history", [])
                }
                for i, obj in enumerate(environment.get("objects", []))
            ]
        
        # Process relationships
        if "relationships" in environment:
            features["relationships"] = environment.get("relationships", [])
        
        # Process contextual factors
        if "context" in environment:
            context = environment["context"]
            features["contextual_factors"] = {
                "physical": context.get("physical", {}),
                "social": context.get("social", {}),
                "cultural": context.get("cultural", {}),
                "technological": context.get("technological", {}),
                "emotional": context.get("emotional", {})
            }
        
        # Process temporal aspects
        if "temporal" in environment:
            features["temporal_aspects"] = environment.get("temporal", {})
        
        # Process normative elements
        if "norms" in environment:
            features["normative_elements"] = environment.get("norms", {})
        
        # Process affordances
        if "affordances" in environment:
            features["affordances"] = environment.get("affordances", [])
        
        return features
    
    def _calculate_agent_salience(self, agent: Dict[str, Any]) -> float:
        """Calculate how salient another agent is to the self-model"""
        # Base salience factor
        salience = agent.get("base_salience", 0.5)
        
        # Adjust based on factors that would make an agent more noticeable
        modifiers = {
            "emotional_significance": agent.get("emotional_significance", 0.0) * 0.3,
            "novelty": agent.get("novelty", 0.0) * 0.2,
            "power_differential": abs(agent.get("power", 0.5) - 0.5) * 0.2,
            "shared_goals": agent.get("shared_goals", 0.0) * 0.15,
            "threat_level": agent.get("threat_level", 0.0) * 0.25
        }
        
        # Apply modifiers
        for mod in modifiers.values():
            salience += mod
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, salience))
    
    def _calculate_agent_similarity(self, agent: Dict[str, Any]) -> float:
        """Calculate similarity between self and another agent"""
        if "attributes" not in agent:
            return 0.5  # Default value if no attributes provided
        
        similarity_score = 0.0
        attribute_count = 0
        
        # Compare attributes where they exist in both self and other
        for attr, value in agent["attributes"].items():
            if attr in self.core_attributes:
                similarity = 1.0 - abs(self.core_attributes[attr] - value)
                similarity_score += similarity
                attribute_count += 1
        
        # If no comparable attributes, return default
        if attribute_count == 0:
            return 0.5
        
        return similarity_score / attribute_count
    
    def _calculate_object_salience(self, obj: Dict[str, Any]) -> float:
        """Calculate how salient an object is to the self-model"""
        # Base salience
        salience = obj.get("base_salience", 0.3)
        
        # Adjust based on factors
        modifiers = {
            "utility": obj.get("utility", 0.0) * 0.25,
            "emotional_association": obj.get("emotional_association", 0.0) * 0.3,
            "novelty": obj.get("novelty", 0.0) * 0.2,
            "sensory_intensity": obj.get("sensory_intensity", 0.0) * 0.15
        }
        
        # Apply modifiers
        for mod in modifiers.values():
            salience += mod
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, salience))
    
    def _establish_self_boundaries(self, env_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Establish current boundaries between self and environment
        
        This determines where the 'self' ends and the environment begins across
        multiple dimensions, adjusting permeability based on context.
        """
        current_boundaries = {}
        
        # For each boundary type, calculate current state
        for name, boundary in self.boundary_conditions.items():
            # Start with base threshold
            current_value = boundary.threshold
            
            # Adjust based on environmental factors
            if name == "cognitive":
                # Cognitive boundary affected by information complexity and novelty
                info_complexity = self._extract_feature_value(
                    env_features, ["contextual_factors", "informational", "complexity"], 0.5)
                novelty = self._extract_feature_value(
                    env_features, ["contextual_factors", "informational", "novelty"], 0.5)
                
                # Adjust permeability - more permeable with familiar but complex information
                permeability_shift = (info_complexity * 0.3) - (novelty * 0.4)
                adaptive_factor = boundary.adaptivity * permeability_shift
                current_value += adaptive_factor
            
            elif name == "social":
                # Social boundary affected by social context and relationship factors
                social_intimacy = self._extract_feature_value(
                    env_features, ["contextual_factors", "social", "intimacy"], 0.3)
                social_trust = self._extract_feature_value(
                    env_features, ["contextual_factors", "social", "trust"], 0.4)
                
                # More permeable with trusted, intimate connections
                permeability_shift = (social_intimacy * 0.4) + (social_trust * 0.4)
                adaptive_factor = boundary.adaptivity * permeability_shift
                current_value += adaptive_factor
            
            elif name == "temporal":
                # Temporal boundary affected by context's orientation to past/future
                temporal_distance = self._extract_feature_value(
                    env_features, ["temporal_aspects", "distance"], 0.5)
                temporal_relevance = self._extract_feature_value(
                    env_features, ["temporal_aspects", "relevance"], 0.5)
                
                # More permeable with relevant but not too distant temporal contexts
                permeability_shift = (temporal_relevance * 0.5) - (temporal_distance * 0.3)
                adaptive_factor = boundary.adaptivity * permeability_shift
                current_value += adaptive_factor
            
            elif name == "ethical":
                # Ethical boundary affected by moral clarity and stakes
                moral_clarity = self._extract_feature_value(
                    env_features, ["normative_elements", "clarity"], 0.5)
                moral_stakes = self._extract_feature_value(
                    env_features, ["normative_elements", "stakes"], 0.5)
                
                # Less permeable with high stakes and clear moral implications
                permeability_shift = (moral_clarity * 0.4) + (moral_stakes * 0.4)
                adaptive_factor = boundary.adaptivity * permeability_shift
                current_value += adaptive_factor
            
            elif name == "creative":
                # Creative boundary affected by novelty and safety of context
                contextual_novelty = self._extract_feature_value(
                    env_features, ["contextual_factors", "creative", "novelty"], 0.5)
                psychological_safety = self._extract_feature_value(
                    env_features, ["contextual_factors", "emotional", "safety"], 0.5)
                
                # More permeable with novel contexts and psychological safety
                permeability_shift = (contextual_novelty * 0.4) + (psychological_safety * 0.3)
                adaptive_factor = boundary.adaptivity * permeability_shift
                current_value += adaptive_factor
            
            # Ensure value stays within bounds
            current_boundaries[name] = max(0.1, min(0.9, current_value))
        
        return current_boundaries
    
    def _extract_feature_value(self, features: Dict[str, Any], path: List[str], default: float) -> float:
        """Extract a value from nested dictionary using a path, with default fallback"""
        current = features
        for key in path:
            if key in current:
                current = current[key]
            else:
                return default
        
        if isinstance(current, (int, float)):
            return float(current)
        return default
    
    def _construct_self_representation(self, 
                                      boundaries: Dict[str, float], 
                                      env_features: Dict[str, Any]) -> SelfRepresentation:
        """
        Construct a contextualized representation of self
        
        Args:
            boundaries: Current boundary states
            env_features: Processed environmental features
            
        Returns:
            A structured self-representation
        """
        # Generate a unique ID for this representation
        import uuid
        representation_id = str(uuid.uuid4())
        
        # Adjust core attributes based on context
        contextual_attributes = self._adjust_attributes_to_context(env_features)
        
        # Construct relational dimensions - how self relates to various aspects of environment
        relational_dimensions = self._construct_relational_dimensions(env_features, boundaries)
        
        # Create historical trajectory connecting past representations to current
        historical_trajectory = self._construct_historical_trajectory()
        
        # Identify areas of uncertainty in the self-model
        uncertainty_regions = self._identify_uncertainty_regions(contextual_attributes, env_features)
        
        # Calculate overall coherence of this self-representation
        coherence_score = self._calculate_representation_coherence(
            contextual_attributes, 
            relational_dimensions, 
            boundaries
        )
        
        # Create the self representation
        representation = SelfRepresentation(
            id=representation_id,
            core_attributes=contextual_attributes,
            relational_dimensions=relational_dimensions,
            historical_trajectory=historical_trajectory,
            projected_potentials={},  # Will be filled by _project_self_potentials
            coherence_score=coherence_score,
            uncertainty_regions=uncertainty_regions
        )
        
        return representation
    
    def _adjust_attributes_to_context(self, env_features: Dict[str, Any]) -> Dict[str, float]:
        """Adjust core attributes based on current environmental context"""
        contextual_attributes = self.core_attributes.copy()
        
        # Example adjustments based on environment
        
        # Agency adjustment based on constraints and affordances
        if "affordances" in env_features and env_features["affordances"]:
            affordance_factor = len(env_features["affordances"]) * 0.05
            contextual_attributes["agency"] = min(1.0, 
                                                 contextual_attributes["agency"] + affordance_factor)
        
        # Social embeddedness adjustment based on social context
        social_presence = self._extract_feature_value(
            env_features, ["contextual_factors", "social", "presence"], 0.0)
        
        social_adjustment = social_presence * 0.2
        contextual_attributes["social_embeddedness"] = max(0.1, min(1.0, 
            contextual_attributes["social_embeddedness"] + social_adjustment))
        
        # Reflectivity adjusted by cognitive demands
        cognitive_load = self._extract_feature_value(
            env_features, ["contextual_factors", "cognitive", "load"], 0.5)
        
        reflectivity_adjustment = (0.5 - cognitive_load) * 0.15
        contextual_attributes["reflectivity"] = max(0.1, min(1.0, 
            contextual_attributes["reflectivity"] + reflectivity_adjustment))
        
        # Value alignment adjusted by ethical context
        ethical_salience = self._extract_feature_value(
            env_features, ["normative_elements", "salience"], 0.3)
        
        value_adjustment = ethical_salience * 0.15
        contextual_attributes["value_alignment"] = max(0.1, min(1.0, 
            contextual_attributes["value_alignment"] + value_adjustment))
        
        return contextual_attributes
    
    def _construct_relational_dimensions(self, 
                                        env_features: Dict[str, Any],
                                        boundaries: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Construct relational dimensions of the self-model"""
        relational_dimensions = {
            "interpersonal": {},
            "cognitive": {},
            "ethical": {},
            "creative": {},
            "temporal": {}
        }
        
        # Interpersonal relations
        if "agents" in env_features and env_features["agents"]:
            for agent in env_features["agents"]:
                agent_id = agent.get("id", "unknown_agent")
                relational_dimensions["interpersonal"][agent_id] = {
                    "closeness": agent.get("similarity", 0.5) * boundaries.get("social", 0.5),
                    "trust": agent.get("perceived_intentions", {}).get("benevolence", 0.5),
                    "influence": agent.get("perceived_intentions", {}).get("influence", 0.3)
                }
        
        # Cognitive relations to ideas and information
        if "contextual_factors" in env_features and "informational" in env_features["contextual_factors"]:
            info = env_features["contextual_factors"]["informational"]
            for topic, details in info.items():
                if isinstance(details, dict):
                    relational_dimensions["cognitive"][topic] = {
                        "familiarity": details.get("familiarity", 0.5),
                        "interest": details.get("interest", 0.5),
                        "confidence": details.get("confidence", 0.5)
                    }
        
        # Ethical relations to values and norms
        if "normative_elements" in env_features:
            norms = env_features["normative_elements"]
            for norm, details in norms.items():
                if isinstance(details, dict):
                    relational_dimensions["ethical"][norm] = {
                        "alignment": details.get("alignment", 0.5),
                        "importance": details.get("importance", 0.5),
                        "commitment": details.get("commitment", 0.5)
                    }
        
        # Creative relations to novel possibilities
        if "affordances" in env_features:
            for i, affordance in enumerate(env_features["affordances"]):
                if isinstance(affordance, dict):
                    aid = affordance.get("id", f"affordance_{i}")
                    relational_dimensions["creative"][aid] = {
                        "novelty": affordance.get("novelty", 0.5),
                        "appeal": affordance.get("appeal", 0.5),
                        "feasibility": affordance.get("feasibility", 0.5)
                    }
        
        # Temporal relations connecting past, present, future
        if "temporal_aspects" in env_features:
            temporal = env_features["temporal_aspects"]
            if "past" in temporal:
                relational_dimensions["temporal"]["past"] = {
                    "continuity": temporal["past"].get("continuity", 0.7),
                    "relevance": temporal["past"].get("relevance", 0.6),
                    "emotional_valence": temporal["past"].get("emotional_valence", 0.5)
                }
            
            if "future" in temporal:
                relational_dimensions["temporal"]["future"] = {
                    "anticipation": temporal["future"].get("anticipation", 0.6),
                    "agency": temporal["future"].get("agency", 0.7),
                    "alignment": temporal["future"].get("alignment", 0.65)
                }
        
        return relational_dimensions
    
    def _construct_historical_trajectory(self) -> List[Dict[str, Any]]:
        """Construct a representation of historical trajectory connecting past to present"""
        trajectory = []
        
        # Include only the most recent representations up to temporal_integration_depth
        history_depth = min(self.temporal_integration_depth, len(self.representation_history))
        
        for i in range(history_depth):
            past_rep = self.representation_history[-(i+1)]
            
            # Extract key elements from past representation
            trajectory.append({
                "id": past_rep.get("id", f"past_{i}"),
                "time_index": -(i+1),
                "core_attribute_state": past_rep.get("core_attributes", {}),
                "coherence": past_rep.get("coherence_score", 0.5),
                "key_relations": self._extract_key_relations(past_rep.get("relational_dimensions", {})),
                "significant_changes": self._identify_significant_changes(
                    past_rep, 
                    self.representation_history[-(i+2)] if i+2 <= history_depth else None
                )
            })
        
        return trajectory
    
    def _extract_key_relations(self, relational_dimensions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract the most significant relations from each dimension"""
        key_relations = {}
        
        for dimension, relations in relational_dimensions.items():
            # Find the most significant relations in this dimension
            significant = {}
            
            for entity, attributes in relations.items():
                # Calculate overall significance
                if isinstance(attributes, dict):
                    significance = sum(attributes.values()) / max(1, len(attributes))
                    
                    # Keep only the most significant relations
                    if significance > 0.7:  # Threshold for significance
                        significant[entity] = significance
            
            # Sort by significance and keep top 3
            sorted_significant = sorted(significant.items(), key=lambda x: x[1], reverse=True)[:3]
            key_relations[dimension] = dict(sorted_significant)
        
        return key_relations
    
    def _identify_significant_changes(self, 
                                     current_rep: Dict[str, Any], 
                                     previous_rep: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify significant changes between consecutive representations"""
        if not previous_rep:
            return []
        
        significant_changes = []
        
        # Check for attribute changes
        current_attrs = current_rep.get("core_attributes", {})
        previous_attrs = previous_rep.get("core_attributes", {})
        
        for attr, current_val in current_attrs.items():
          if attr in previous_attrs:
                change = current_val - previous_attrs[attr]
                if abs(change) > 0.15:  # Threshold for significant change
                    significant_changes.append({
                        "type": "attribute_change",
                        "attribute": attr,
                        "previous": previous_attrs[attr],
                        "current": current_val,
                        "magnitude": abs(change),
                        "direction": "increase" if change > 0 else "decrease"
                    })
        
        # Check for relationship changes
        current_rels = current_rep.get("relational_dimensions", {})
        previous_rels = previous_rep.get("relational_dimensions", {})
        
        for dim, relations in current_rels.items():
            if dim in previous_rels:
                for entity, attributes in relations.items():
                    if entity in previous_rels[dim]:
                        # Compare each attribute in the relationship
                        for attr, current_val in attributes.items():
                            if attr in previous_rels[dim][entity]:
                                change = current_val - previous_rels[dim][entity][attr]
                                if abs(change) > 0.2:  # Threshold for significant relationship change
                                    significant_changes.append({
                                        "type": "relationship_change",
                                        "dimension": dim,
                                        "entity": entity,
                                        "attribute": attr,
                                        "previous": previous_rels[dim][entity][attr],
                                        "current": current_val,
                                        "magnitude": abs(change),
                                        "direction": "strengthen" if change > 0 else "weaken"
                                    })
        
        # Check for coherence changes
        current_coherence = current_rep.get("coherence_score", 0.5)
        previous_coherence = previous_rep.get("coherence_score", 0.5)
        
        coherence_change = current_coherence - previous_coherence
        if abs(coherence_change) > 0.1:  # Threshold for significant coherence change
            significant_changes.append({
                "type": "coherence_change",
                "previous": previous_coherence,
                "current": current_coherence,
                "magnitude": abs(coherence_change),
                "direction": "increase" if coherence_change > 0 else "decrease"
            })
        
        return significant_changes
    
    def _identify_uncertainty_regions(self, 
                                     attributes: Dict[str, float],
                                     env_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify regions of uncertainty in the current self-model"""
        uncertainty_regions = []
        
        # Check for attribute uncertainty
        for attr, value in attributes.items():
            # Values close to 0.5 often indicate uncertainty
            if 0.4 <= value <= 0.6:
                contextual_factors = self._identify_contextual_factors_for_attribute(attr, env_features)
                uncertainty_regions.append({
                    "type": "attribute_uncertainty",
                    "attribute": attr,
                    "current_value": value,
                    "uncertainty_level": 1.0 - abs((value - 0.5) * 2),  # Higher when closer to 0.5
                    "contextual_factors": contextual_factors
                })
        
        # Check for boundary uncertainty
        for boundary_name, boundary in self.boundary_conditions.items():
            # High permeability often indicates uncertainty about boundaries
            if boundary.permeability > 0.7:
                contextual_factors = self._identify_contextual_factors_for_boundary(boundary_name, env_features)
                uncertainty_regions.append({
                    "type": "boundary_uncertainty",
                    "boundary": boundary_name,
                    "permeability": boundary.permeability,
                    "uncertainty_level": boundary.permeability - 0.3,  # Scale to 0.4-0.7 range
                    "contextual_factors": contextual_factors
                })
        
        # Check for relationship uncertainty in environment
        for dim, relations in env_features.get("relationships", {}).items():
            for entity, details in relations.items():
                if isinstance(details, dict) and "ambiguity" in details:
                    if details["ambiguity"] > 0.6:
                        uncertainty_regions.append({
                            "type": "relationship_uncertainty",
                            "dimension": dim,
                            "entity": entity,
                            "ambiguity": details["ambiguity"],
                            "uncertainty_level": details["ambiguity"]
                        })
        
        return uncertainty_regions
    
    def _identify_contextual_factors_for_attribute(self, 
                                                 attribute: str,
                                                 env_features: Dict[str, Any]) -> List[str]:
        """Identify contextual factors that might be contributing to attribute uncertainty"""
        factors = []
        
        # Different attributes are affected by different contextual factors
        if attribute == "agency":
            if self._extract_feature_value(env_features, ["contextual_factors", "social", "control"], 0.5) > 0.7:
                factors.append("high_external_control")
            if len(env_features.get("affordances", [])) < 2:
                factors.append("limited_affordances")
        
        elif attribute == "social_embeddedness":
            if self._extract_feature_value(
                env_features, ["contextual_factors", "social", "ambiguity"], 0.3) > 0.6:
                factors.append("ambiguous_social_cues")
            if len(env_features.get("agents", [])) < 2:
                factors.append("limited_social_context")
        
        elif attribute == "value_alignment":
            if self._extract_feature_value(env_features, ["normative_elements", "conflict"], 0.3) > 0.6:
                factors.append("value_conflicts")
            if self._extract_feature_value(env_features, ["normative_elements", "novelty"], 0.3) > 0.7:
                factors.append("novel_ethical_context")
        
        # Add general factors that might affect any attribute
        if self._extract_feature_value(env_features, ["contextual_factors", "cognitive", "load"], 0.5) > 0.7:
            factors.append("high_cognitive_load")
        
        if self._extract_feature_value(env_features, ["contextual_factors", "emotional", "intensity"], 0.5) > 0.7:
            factors.append("high_emotional_intensity")
        
        return factors
    
    def _identify_contextual_factors_for_boundary(self, 
                                                boundary_name: str,
                                                env_features: Dict[str, Any]) -> List[str]:
        """Identify contextual factors that might be contributing to boundary uncertainty"""
        factors = []
        
        # Different boundaries are affected by different contextual factors
        if boundary_name == "cognitive":
            if self._extract_feature_value(
                env_features, ["contextual_factors", "informational", "overload"], 0.3) > 0.6:
                factors.append("information_overload")
            if self._extract_feature_value(
                env_features, ["contextual_factors", "informational", "consistency"], 0.5) < 0.3:
                factors.append("inconsistent_information")
        
        elif boundary_name == "social":
            if self._extract_feature_value(
                env_features, ["contextual_factors", "social", "pressure"], 0.3) > 0.6:
                factors.append("social_pressure")
            if self._extract_feature_value(
                env_features, ["contextual_factors", "social", "trust"], 0.5) < 0.3:
                factors.append("low_trust_environment")
        
        elif boundary_name == "ethical":
            if self._extract_feature_value(
                env_features, ["normative_elements", "clarity"], 0.5) < 0.3:
                factors.append("ethical_ambiguity")
            if self._extract_feature_value(
                env_features, ["normative_elements", "conflict"], 0.3) > 0.6:
                factors.append("conflicting_norms")
        
        # Add general factors
        if self._extract_feature_value(
            env_features, ["contextual_factors", "temporal", "pressure"], 0.3) > 0.7:
            factors.append("time_pressure")
        
        return factors
    
    def _calculate_representation_coherence(self, 
                                          attributes: Dict[str, float],
                                          relational_dimensions: Dict[str, Dict[str, Any]],
                                          boundaries: Dict[str, float]) -> float:
        """
        Calculate the overall coherence of the self-representation
        
        Higher coherence indicates a more stable, integrated self-model
        """
        coherence_factors = []
        
        # Attribute consistency
        attribute_variance = np.std(list(attributes.values()))
        attribute_coherence = 1.0 - min(1.0, attribute_variance * 2)
        coherence_factors.append(attribute_coherence)
        
        # Relational consistency
        relation_coherence_scores = []
        for dim, relations in relational_dimensions.items():
            if relations:
                # Calculate mean values for each type of relation attribute
                relation_attributes = {}
                for entity, attrs in relations.items():
                    for attr, value in attrs.items():
                        if attr not in relation_attributes:
                            relation_attributes[attr] = []
                        relation_attributes[attr].append(value)
                
                # Calculate variance for each attribute type
                attr_variances = [np.std(values) for values in relation_attributes.values() 
                                if len(values) > 1]
                
                if attr_variances:
                    # Average variance across attribute types
                    avg_variance = sum(attr_variances) / len(attr_variances)
                    dim_coherence = 1.0 - min(1.0, avg_variance * 2)
                    relation_coherence_scores.append(dim_coherence)
        
        # Average relational coherence if we have scores
        if relation_coherence_scores:
            relation_coherence = sum(relation_coherence_scores) / len(relation_coherence_scores)
            coherence_factors.append(relation_coherence)
        
        # Boundary consistency
        boundary_variance = np.std(list(boundaries.values()))
        boundary_coherence = 1.0 - min(1.0, boundary_variance * 2)
        coherence_factors.append(boundary_coherence)
        
        # Historical continuity - compare with most recent representation if available
        if self.representation_history:
            last_rep = self.representation_history[-1]
            last_attrs = last_rep.get("core_attributes", {})
            
            if last_attrs:
                # Calculate average change in attributes
                attribute_changes = []
                for attr, value in attributes.items():
                    if attr in last_attrs:
                        change = abs(value - last_attrs[attr])
                        attribute_changes.append(change)
                
                if attribute_changes:
                    avg_change = sum(attribute_changes) / len(attribute_changes)
                    historical_coherence = 1.0 - min(1.0, avg_change * 3)
                    coherence_factors.append(historical_coherence)
        
        # Calculate overall coherence as weighted average
        # More weight on attribute consistency and historical continuity
        weights = [0.3, 0.25, 0.2, 0.25][:len(coherence_factors)]
        normalized_weights = [w/sum(weights) for w in weights]
        
        overall_coherence = sum(f * w for f, w in zip(coherence_factors, normalized_weights))
        
        # Apply uncertainty tolerance as a modifier
        uncertainty_factor = 1.0 - (max(0, (1.0 - self.uncertainty_tolerance) * 0.2))
        
        return overall_coherence * uncertainty_factor
    
    def _project_self_potentials(self, 
                               representation: SelfRepresentation,
                               environment: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Project potential developmental trajectories for the self
        
        Args:
            representation: Current self representation
            environment: Current environmental context
            
        Returns:
            Dictionary of potential trajectories for different aspects of self
        """
        potentials = {}
        
        # Project attribute potentials
        potentials["attributes"] = {}
        for attr, value in representation.core_attributes.items():
            # Project possible changes based on current value and environment
            growth_potential = self._calculate_growth_potential(attr, value, environment)
            regression_risk = self._calculate_regression_risk(attr, value, environment)
            
            # Create vector of potential values at different future timepoints
            # Format: [t+1, t+2, t+3] where values are potential attribute values
            trajectory = []
            current = value
            
            for i in range(3):  # Project 3 steps ahead
                # Calculate potential change with some randomness
                time_factor = (i + 1) / 3  # Increasing effect over time
                potential_change = (growth_potential - regression_risk) * time_factor
                
                # Add some randomness that decreases for further timepoints
                randomness = np.random.normal(0, 0.05 * (1 - 0.2*i))
                potential_change += randomness
                
                # Apply change with constraints
                new_value = max(0.1, min(0.9, current + potential_change))
                trajectory.append(new_value)
                current = new_value  # Update for next iteration
            
            potentials["attributes"][attr] = trajectory
        
        # Project boundary potentials
        potentials["boundaries"] = {}
        for boundary_name, current_value in representation.core_attributes.items():
            boundary_obj = self.boundary_conditions.get(boundary_name)
            if not boundary_obj:
                continue
                
            # Calculate potential boundary evolution
            adaptation_potential = boundary_obj.adaptivity * 0.1
            context_pressure = self._calculate_boundary_pressure(boundary_name, environment)
            
            trajectory = []
            current = current_value
            
            for i in range(3):  # Project 3 steps ahead
                time_factor = (i + 1) / 3
                potential_change = (context_pressure * adaptation_potential) * time_factor
                
                # Add some randomness
                randomness = np.random.normal(0, 0.03 * (1 - 0.2*i))
                potential_change += randomness
                
                # Apply change with constraints
                new_value = max(0.1, min(0.9, current + potential_change))
                trajectory.append(new_value)
                current = new_value
            
            potentials["boundaries"][boundary_name] = trajectory
        
        # Project coherence potential
        current_coherence = representation.coherence_score
        coherence_trend = self._calculate_coherence_trend()
        
        potentials["coherence"] = []
        current = current_coherence
        
        for i in range(3):  # Project 3 steps ahead
            time_factor = (i + 1) / 3
            potential_change = coherence_trend * time_factor
            
            # Coherence tends to stabilize over time unless disrupted
            stabilization = (self.identity_coherence_threshold - current) * 0.1 * time_factor
            
            # Add some randomness
            randomness = np.random.normal(0, 0.04 * (1 - 0.2*i))
            potential_change += randomness + stabilization
            
            # Apply change with constraints
            new_value = max(0.2, min(0.95, current + potential_change))
            potentials["coherence"].append(new_value)
            current = new_value
        
        return potentials
    
    def _calculate_growth_potential(self, 
                                  attribute: str, 
                                  current_value: float,
                                  environment: Dict[str, Any]) -> float:
        """Calculate growth potential for an attribute based on current state and environment"""
        # Base growth potential - higher potential when attribute is in mid-range
        base_potential = 0.3 * (1.0 - abs(current_value - 0.5) * 2)
        
        # Environmental factors affecting this attribute
        env_factor = 0.0
        
        if attribute == "agency":
            # Agency grows with supportive environment and opportunities
            affordances = len(environment.get("affordances", []))
            support = self._extract_feature_value(
                environment, ["contextual_factors", "supportive"], 0.5)
            
            env_factor = (min(affordances, 5) / 10) + (support * 0.1)
        
        elif attribute == "reflectivity":
            # Reflectivity grows with cognitive space and low pressure
            cognitive_load = self._extract_feature_value(
                environment, ["contextual_factors", "cognitive", "load"], 0.5)
            temporal_pressure = self._extract_feature_value(
                environment, ["contextual_factors", "temporal", "pressure"], 0.5)
            
            env_factor = 0.2 - (cognitive_load * 0.1) - (temporal_pressure * 0.1)
        
        elif attribute == "social_embeddedness":
            # Social embeddedness grows with positive social interactions
            social_presence = self._extract_feature_value(
                environment, ["contextual_factors", "social", "presence"], 0.5)
            social_quality = self._extract_feature_value(
                environment, ["contextual_factors", "social", "quality"], 0.5)
            
            env_factor = (social_presence * 0.05) + (social_quality * 0.15)
        
        elif attribute == "narrative_coherence":
            # Narrative coherence grows with reflection and integration
            reflective_context = self._extract_feature_value(
                environment, ["contextual_factors", "reflective"], 0.3)
            continuity = self._extract_feature_value(
                environment, ["temporal_aspects", "continuity"], 0.5)
            
            env_factor = (reflective_context * 0.1) + (continuity * 0.1)
        
        # Combine base potential with environmental factors
        return base_potential + env_factor
    
    def _calculate_regression_risk(self, 
                                 attribute: str, 
                                 current_value: float,
                                 environment: Dict[str, Any]) -> float:
        """Calculate risk of regression for an attribute based on current state and environment"""
        # Base regression risk - higher risk when attribute is high
        base_risk = 0.1 + (current_value * 0.1)
        
        # Environmental factors increasing regression risk
        env_factor = 0.0
        
        if attribute == "agency":
            # Agency regresses with constraints and control
            constraints = self._extract_feature_value(
                environment, ["contextual_factors", "constraints"], 0.3)
            external_control = self._extract_feature_value(
                environment, ["contextual_factors", "social", "control"], 0.3)
            
            env_factor = (constraints * 0.1) + (external_control * 0.1)
        
        elif attribute == "reflectivity":
            # Reflectivity regresses with high pressure and distractions
            cognitive_load = self._extract_feature_value(
                environment, ["contextual_factors", "cognitive", "load"], 0.5)
            distractions = self._extract_feature_value(
                environment, ["contextual_factors", "distractions"], 0.3)
            
            env_factor = (cognitive_load * 0.1) + (distractions * 0.1)
        
        elif attribute == "social_embeddedness":
            # Social embeddedness regresses with isolation or negative interactions
            isolation = self._extract_feature_value(
                environment, ["contextual_factors", "social", "isolation"], 0.3)
            social_tension = self._extract_feature_value(
                environment, ["contextual_factors", "social", "tension"], 0.3)
            
            env_factor = (isolation * 0.15) + (social_tension * 0.05)
        
        elif attribute == "narrative_coherence":
            # Narrative coherence regresses with disruption and inconsistency
            disruption = self._extract_feature_value(
                environment, ["temporal_aspects", "disruption"], 0.3)
            inconsistency = self._extract_feature_value(
                environment, ["contextual_factors", "inconsistency"], 0.3)
            
            env_factor = (disruption * 0.1) + (inconsistency * 0.1)
        
        # Combine base risk with environmental factors
        return base_risk + env_factor
    
    def _calculate_boundary_pressure(self, 
                                   boundary_name: str, 
                                   environment: Dict[str, Any]) -> float:
        """Calculate environmental pressure on a boundary"""
        # Positive value indicates pressure to increase permeability
        # Negative value indicates pressure to decrease permeability
        
        if boundary_name == "cognitive":
            # Cognitive boundary affected by information novelty vs. complexity
            novelty = self._extract_feature_value(
                environment, ["contextual_factors", "informational", "novelty"], 0.5)
            complexity = self._extract_feature_value(
                environment, ["contextual_factors", "informational", "complexity"], 0.5)
            
            # Novel but simple information increases permeability
            # Complex but non-novel maintains boundaries
            return (novelty * 0.2) - (complexity * 0.1)
        
        elif boundary_name == "social":
            # Social boundary affected by trust and intimacy
            trust = self._extract_feature_value(
                environment, ["contextual_factors", "social", "trust"], 0.5)
            intimacy = self._extract_feature_value(
                environment, ["contextual_factors", "social", "intimacy"], 0.3)
            threat = self._extract_feature_value(
                environment, ["contextual_factors", "social", "threat"], 0.2)
            
            # Trust and intimacy increase permeability, threat decreases it
            return (trust * 0.15) + (intimacy * 0.1) - (threat * 0.25)
        
        elif boundary_name == "ethical":
            # Ethical boundary affected by moral clarity and stakes
            clarity = self._extract_feature_value(
                environment, ["normative_elements", "clarity"], 0.5)
            stakes = self._extract_feature_value(
                environment, ["normative_elements", "stakes"], 0.5)
            
            # High stakes and clarity lead to firmer boundaries
            return -((clarity * 0.1) + (stakes * 0.15))
        
        elif boundary_name == "creative":
            # Creative boundary affected by psychological safety and novelty
            safety = self._extract_feature_value(
                environment, ["contextual_factors", "psychological_safety"], 0.5)
            novelty = self._extract_feature_value(
                environment, ["contextual_factors", "novelty"], 0.5)
            
            # Safety and novelty increase creative boundary permeability
            return (safety * 0.2) + (novelty * 0.1)
        
        elif boundary_name == "temporal":
            # Temporal boundary affected by continuity and relevance
            continuity = self._extract_feature_value(
                environment, ["temporal_aspects", "continuity"], 0.5)
            relevance = self._extract_feature_value(
                environment, ["temporal_aspects", "relevance"], 0.5)
            
            # Continuity and relevance affect permeability differently
            return (relevance * 0.15) - ((1 - continuity) * 0.1)
        
        # Default - no pressure
        return 0.0
    
    def _calculate_coherence_trend(self) -> float:
        """Calculate trend in coherence based on history"""
        # Default slight positive trend
        if len(self.coherence_history) < 2:
            return 0.02
        
        # Calculate average change over recent history
        recent_history = self.coherence_history[-min(5, len(self.coherence_history)):]
        changes = [recent_history[i] - recent_history[i-1] for i in range(1, len(recent_history))]
        
        return sum(changes) / len(changes)
    
    def _update_self_representation(self, representation: SelfRepresentation) -> None:
        """Update the current self representation and history"""
        # Store new representation
        self.self_representations[representation.id] = representation
        self.current_representation_id = representation.id
        
        # Update history
        self.representation_history.append({
            "id": representation.id,
            "core_attributes": representation.core_attributes,
            "relational_dimensions": representation.relational_dimensions,
            "coherence_score": representation.coherence_score
        })
        
        # Limit history length
        max_history = max(self.temporal_integration_depth * 2, 10)
        if len(self.representation_history) > max_history:
            self.representation_history = self.representation_history[-max_history:]
        
        # Update coherence history
        self.coherence_history.append(representation.coherence_score)
        if len(self.coherence_history) > max_history:
            self.coherence_history = self.coherence_history[-max_history:]
        
        self.logger.info(f"Updated self representation to {representation.id} with coherence {representation.coherence_score:.2f}")
    
    def evaluate_actions(self, 
                       potential_actions: List[Dict[str, Any]], 
                       self_model: Optional[SelfRepresentation] = None) -> List[ActionEvaluation]:
        """
        Evaluate actions in terms of self-model coherence
        
        Args:
            potential_actions: List of potential actions to evaluate
            self_model: Self model to use (uses current if None)
            
        Returns:
            List of action evaluations in terms of impact on self-model
        """
        self.logger.info(f"Evaluating {len(potential_actions)} potential actions")
        
        # Use current self model if none provided
        if not self_model and self.current_representation_id:
            self_model = self.self_representations.get(self.current_representation_id)
        
        if not self_model:
            self.logger.warning("No self model available for action evaluation")
            return []
        
        evaluations = []
        
        for action in potential_actions:
            # Predict impacts of this action on self-model
            impacts = self._predict_action_impacts(action, self_model)
            
            # Assess coherence with identity
            coherence = self._assess_identity_coherence(impacts, self_model)
            
            # Analyze how action expresses different aspects of identity
            expression = self._analyze_identity_expression(action, self_model)
            
            # Evaluate effects on self-boundaries
            boundary_effects = self._evaluate_boundary_effects(action, self_model)
            
            # Assess alignment with developmental trajectory
            alignment = self._assess_developmental_alignment(action, self_model)
            
            # Calculate overall confidence in evaluation
            confidence = self._calculate_evaluation_confidence(
                impacts, coherence, expression, boundary_effects, alignment)
            
            # Create evaluation object
            evaluation = ActionEvaluation(
                action_id=action.get("id", "unknown_action"),
                coherence_impact=coherence,
                identity_expression=expression,
                boundary_effects=boundary_effects,
                developmental_alignment=alignment,
                confidence=confidence
            )
            
            evaluations.append(evaluation)
        
        # Sort evaluations by coherence impact
        evaluations.sort(key=lambda e: e.coherence_impact, reverse=True)
        
        return evaluations
    
    def _predict_action_impacts(self, 
                             action: Dict[str, Any], 
                             self_model: SelfRepresentation) -> Dict[str, Any]:
        """Predict how an action would impact different aspects of the self-model"""
        impacts = {
            "attributes": {},
            "relations": {},
            "boundaries": {},
            "coherence": 0.0
        }
        
        # Predict attribute impacts
        for attr, value in self_model.core_attributes.items():
            # Calculate expected impact on this attribute
            impact = self._calculate_attribute_impact(attr, action, value)
            impacts["attributes"][attr] = impact
        
        # Predict relational impacts
        for dimension, relations in self_model.relational_dimensions.items():
            impacts["relations"][dimension] = {}
            for entity, attrs in relations.items():
                # Check if action involves this entity
                involves_entity = any(
                    target.get("id") == entity 
                    for target in action.get("targets", [])
                )
                
                if involves_entity:
                    # Calculate specific impact for involved entities
                    rel_impact = self._calculate_relation_impact(dimension, entity, action, attrs)
                    impacts["relations"][dimension][entity] = rel_impact
        
        # Predict boundary impacts
        for boundary_name in self.boundary_conditions:
            # Calculate expected impact on this boundary
            impact = self._calculate_boundary_impact(boundary_name, action)
            impacts["boundaries"][boundary_name] = impact
        
        # Predict overall coherence impact
        consistency_impacts = list(impacts["attributes"].values())
        relation_impacts = [i for d in impacts["relations"].values() for i in d.values()]
        boundary_impacts = list(impacts["boundaries"].values())
        
        all_impacts = consistency_impacts + relation_impacts + boundary_impacts
        coherence_impact = sum(all_impacts) / max(1, len(all_impacts))
        
        impacts["coherence"] = coherence_impact
        
        return impacts
    
    def _calculate_attribute_impact(self, 
                                  attribute: str, 
                                  action: Dict[str, Any], 
                                  current_value: float) -> float:
        """Calculate expected impact of action on a specific attribute"""
        # Default - no impact
        impact = 0.0
        
        # Check action type and relevant attributes
        action_type = action.get("type", "")
        
        if attribute == "agency":
            # Actions that involve choice and control enhance agency
            if action.get("autonomy_level", 0.5) > 0.7:
                impact += 0.05
            
            # Actions that involve following external direction may reduce agency
            if action.get("externally_directed", False):
                impact -= 0.03
        
        elif attribute == "social_embeddedness":
            # Social actions generally increase social embeddedness
            if "social" in action_type or action.get("social_interaction", False):
                impact += 0.04
            
            # Isolating actions decrease social embeddedness
            if action.get("isolating", False):
                impact -= 0.05
        
        elif attribute == "reflectivity":
            # Reflective actions increase reflectivity
            if "reflective" in action_type or action.get("reflective", False):
                impact += 0.05
            
            # Automatic or habitual actions may slightly decrease reflectivity
            if action.get("automaticity", 0.0) > 0.8:
                impact -= 0.02
        
        elif attribute == "value_alignment":
            # Actions aligned with core values strengthen value alignment
            if action.get("value_alignment", 0.5) > 0.8:
       def _calculate_attribute_impact(self, 
                                  attribute: str, 
                                  action: Dict[str, Any], 
                                  current_value: float) -> float:
        """Calculate expected impact of action on a specific attribute"""
        # Default - no impact
        impact = 0.0
        
        # Check action type and relevant attributes
        action_type = action.get("type", "")
        
        if attribute == "agency":
            # Actions that involve choice and control enhance agency
            if action.get("autonomy_level", 0.5) > 0.7:
                impact += 0.05
            
            # Actions that involve following external direction may reduce agency
            if action.get("externally_directed", False):
                impact -= 0.03
        
        elif attribute == "social_embeddedness":
            # Social actions generally increase social embeddedness
            if "social" in action_type or action.get("social_interaction", False):
                impact += 0.04
            
            # Isolating actions decrease social embeddedness
            if action.get("isolating", False):
                impact -= 0.05
        
        elif attribute == "reflectivity":
            # Reflective actions increase reflectivity
            if "reflective" in action_type or action.get("reflective", False):
                impact += 0.05
            
            # Automatic or habitual actions may slightly decrease reflectivity
            if action.get("automaticity", 0.0) > 0.8:
                impact -= 0.02
        
        elif attribute == "value_alignment":
            # Actions aligned with core values strengthen value alignment
            if action.get("value_alignment", 0.5) > 0.8:
                impact += 0.06
            
            # Actions that conflict with values weaken alignment
            if action.get("value_conflict", 0.0) > 0.7:
                impact -= 0.08
        
        # Apply non-linearity - impacts are greater when moving from extremes
        if impact > 0 and current_value < 0.3:
            impact *= 1.3  # Greater positive impact when starting from low value
        elif impact < 0 and current_value > 0.7:
            impact *= 1.3  # Greater negative impact when starting from high value
        
        return impact
    
    def _calculate_relation_impact(self, 
                                 dimension: str, 
                                 entity: str, 
                                 action: Dict[str, Any], 
                                 current_attrs: Dict[str, float]) -> Dict[str, float]:
        """Calculate expected impact of action on a specific relation"""
        impacts = {}
        
        # Action-specific impacts based on dimension
        if dimension == "interpersonal":
            # Check if action explicitly targets this relationship
            is_target = any(t.get("id") == entity for t in action.get("targets", []))
            
            # Closeness impact
            if is_target:
                if action.get("interaction_quality", 0.5) > 0.7:
                    impacts["closeness"] = 0.05  # Positive interaction increases closeness
                elif action.get("interaction_quality", 0.5) < 0.3:
                    impacts["closeness"] = -0.06  # Negative interaction decreases closeness
            
            # Trust impact
            if "trust" in current_attrs:
                if is_target and action.get("trust_building", False):
                    impacts["trust"] = 0.04
                elif is_target and action.get("trust_violating", False):
                    impacts["trust"] = -0.08  # Trust violations have stronger impact
        
        elif dimension == "cognitive":
            # Impact on cognitive relations (familiarity, interest, confidence)
            if "familiarity" in current_attrs and action.get("learning", False):
                impacts["familiarity"] = 0.03
            
            if "interest" in current_attrs and action.get("novelty", 0.0) > 0.7:
                impacts["interest"] = 0.04
            
            if "confidence" in current_attrs:
                if action.get("success", 0.5) > 0.7:
                    impacts["confidence"] = 0.03
                elif action.get("failure", 0.0) > 0.7:
                    impacts["confidence"] = -0.04
        
        return impacts
    
    def _calculate_boundary_impact(self, boundary_name: str, action: Dict[str, Any]) -> float:
        """Calculate expected impact of action on a specific boundary"""
        # Default - no impact
        impact = 0.0
        
        if boundary_name == "cognitive":
            # Learning actions may affect cognitive boundaries
            if action.get("learning", False) or action.get("information_processing", False):
                # Direction depends on complexity and integration
                complexity = action.get("complexity", 0.5)
                integration = action.get("integration", 0.5)
                
                # Complex but well-integrated information leads to more permeable boundaries
                impact = (integration * 0.05) - ((complexity - 0.5) * 0.03)
        
        elif boundary_name == "social":
            # Social interactions affect social boundaries
            if action.get("social_interaction", False):
                # Direction depends on intimacy and safety
                intimacy = action.get("intimacy", 0.3)
                safety = action.get("psychological_safety", 0.5)
                
                # Safe, intimate interactions increase permeability
                impact = (intimacy * safety * 0.1)
                
                # Unsafe interactions decrease permeability
                if safety < 0.3:
                    impact = -0.05
        
        elif boundary_name == "ethical":
            # Moral actions affect ethical boundaries
            if action.get("moral_relevance", 0.0) > 0.6:
                # Clear ethical choices strengthen boundaries
                clarity = action.get("moral_clarity", 0.5)
                commitment = action.get("value_commitment", 0.5)
                
                impact = (clarity * commitment * 0.1)
        
        return impact
    
    def _assess_identity_coherence(self, 
                                 impacts: Dict[str, Any], 
                                 self_model: SelfRepresentation) -> float:
        """Assess how coherently an action aligns with current identity"""
        # Weight different impact types
        attribute_weight = 0.4
        relation_weight = 0.3
        boundary_weight = 0.3
        
        # Calculate weighted attribute coherence
        attribute_impacts = impacts.get("attributes", {})
        attribute_scores = []
        
        for attr, impact in attribute_impacts.items():
            current = self_model.core_attributes.get(attr, 0.5)
            # Higher current value with positive impact increases coherence
            # Lower current value with negative impact also increases coherence
            coherence = current * max(0, impact) + (1 - current) * max(0, -impact)
            attribute_scores.append(coherence)
        
        attribute_coherence = sum(attribute_scores) / max(1, len(attribute_scores))
        
        # Calculate relation coherence
        relation_impacts = impacts.get("relations", {})
        relation_scores = []
        
        for dimension, entities in relation_impacts.items():
            for entity, impacts in entities.items():
                for attr, impact in impacts.items():
                    current = self_model.relational_dimensions.get(dimension, {}).get(entity, {}).get(attr, 0.5)
                    coherence = current * max(0, impact) + (1 - current) * max(0, -impact)
                    relation_scores.append(coherence)
        
        relation_coherence = sum(relation_scores) / max(1, len(relation_scores)) if relation_scores else 0.5
        
        # Calculate boundary coherence
        boundary_impacts = impacts.get("boundaries", {})
        boundary_scores = []
        
        for boundary, impact in boundary_impacts.items():
            # Boundary coherence is more complex - depends on current adaptivity
            adaptivity = self.boundary_conditions.get(boundary, BoundaryCondition("", 0.5, 0.5, 0.5, "")).adaptivity
            
            # More adaptive boundaries should change more readily
            coherence = 0.5 + (adaptivity * impact)
            boundary_scores.append(coherence)
        
        boundary_coherence = sum(boundary_scores) / max(1, len(boundary_scores)) if boundary_scores else 0.5
        
        # Calculate overall coherence
        overall_coherence = (
            attribute_weight * attribute_coherence +
            relation_weight * relation_coherence +
            boundary_weight * boundary_coherence
        )
        
        return overall_coherence
    
    def _analyze_identity_expression(self, 
                                   action: Dict[str, Any], 
                                   self_model: SelfRepresentation) -> Dict[str, float]:
        """Analyze how an action expresses different aspects of identity"""
        expression = {}
        
        # Agency expression
        if "agency" in self_model.core_attributes:
            agency_value = self_model.core_attributes["agency"]
            autonomy_level = action.get("autonomy_level", 0.5)
            deliberation = action.get("deliberation", 0.5)
            
            # Higher agency is expressed through autonomous, deliberate actions
            expression["agency"] = agency_value * (autonomy_level * 0.6 + deliberation * 0.4)
        
        # Value expression
        if "value_alignment" in self_model.core_attributes:
            value_alignment = self_model.core_attributes["value_alignment"]
            action_alignment = action.get("value_alignment", 0.5)
            
            # Value alignment is expressed through aligned actions
            expression["values"] = value_alignment * action_alignment
        
        # Social expression
        if "social_embeddedness" in self_model.core_attributes:
            social_value = self_model.core_attributes["social_embeddedness"]
            social_interaction = 1.0 if action.get("social_interaction", False) else 0.0
            social_sensitivity = action.get("social_sensitivity", 0.5)
            
            # Social embeddedness is expressed through sensitive social interaction
            expression["social"] = social_value * (social_interaction * 0.5 + social_sensitivity * 0.5)
        
        # Creative expression
        if "creative" in self_model.core_attributes:
            creative_value = self_model.core_attributes.get("creative", 0.5)
            novelty = action.get("novelty", 0.3)
            generativity = action.get("generativity", 0.3)
            
            # Creativity is expressed through novel, generative actions
            expression["creative"] = creative_value * (novelty * 0.5 + generativity * 0.5)
        
        return expression
    
    def _evaluate_boundary_effects(self, 
                                 action: Dict[str, Any], 
                                 self_model: SelfRepresentation) -> Dict[str, float]:
        """Evaluate effects of action on self-boundaries"""
        effects = {}
        
        for boundary_name, boundary in self.boundary_conditions.items():
            # Default - neutral effect
            effects[boundary_name] = 0.0
            
            if boundary_name == "cognitive":
                # Information-processing actions affect cognitive boundaries
                if action.get("information_processing", False):
                    # Direction depends on existing boundary state
                    current_permeability = boundary.permeability
                    
                    # Very permeable boundaries may be challenged by complex information
                    if current_permeability > 0.7 and action.get("complexity", 0.5) > 0.7:
                        effects[boundary_name] = -0.1
                    
                    # Very rigid boundaries may be opened by integrated information
                    elif current_permeability < 0.3 and action.get("integration", 0.5) > 0.7:
                        effects[boundary_name] = 0.1
            
            elif boundary_name == "social":
                # Social actions affect social boundaries
                if action.get("social_interaction", False):
                    current_permeability = boundary.permeability
                    intimacy = action.get("intimacy", 0.3)
                    
                    # High intimacy challenges both very open and very closed boundaries
                    if intimacy > 0.7:
                        if current_permeability > 0.8:
                            effects[boundary_name] = -0.1  # Too open - needs protection
                        elif current_permeability < 0.2:
                            effects[boundary_name] = 0.1  # Too closed - needs opening
            
            elif boundary_name == "ethical":
                # Moral choices affect ethical boundaries
                if action.get("moral_relevance", 0.0) > 0.6:
                    current_permeability = boundary.permeability
                    clarity = action.get("moral_clarity", 0.5)
                    
                    # Clear moral choices strengthen ethical boundaries
                    if clarity > 0.7:
                        effects[boundary_name] = -0.05  # Becoming more defined
                    
                    # Ambiguous moral choices may make boundaries more permeable
                    elif clarity < 0.3:
                        effects[boundary_name] = 0.05  # Becoming more open to reconsideration
        
        return effects
    
    def _assess_developmental_alignment(self, 
                                      action: Dict[str, Any], 
                                      self_model: SelfRepresentation) -> float:
        """Assess how well an action aligns with projected developmental trajectory"""
        # Default - neutral alignment
        alignment = 0.5
        
        # Check if we have projected potentials
        if not self_model.projected_potentials:
            return alignment
        
        # Get attribute impacts
        attribute_impacts = {}
        for attr, impact in self._predict_action_impacts(action, self_model).get("attributes", {}).items():
            attribute_impacts[attr] = impact
        
        # Compare with projected trajectories
        alignment_scores = []
        for attr, impact in attribute_impacts.items():
            if attr in self_model.projected_potentials.get("attributes", {}):
                # Get projected change direction
                projected = self_model.projected_potentials["attributes"][attr]
                if projected and len(projected) > 0:
                    projected_change = projected[0] - self_model.core_attributes.get(attr, 0.5)
                    
                    # Calculate alignment as correlation between impact and projected change
                    if abs(projected_change) > 0.01:  # Meaningful projected change
                        # High alignment when both are in same direction
                        score = 0.5 + 0.5 * (
                            1.0 if (impact > 0 and projected_change > 0) or 
                                   (impact < 0 and projected_change < 0) else -1.0
                        ) * min(abs(impact), abs(projected_change)) / max(abs(impact), abs(projected_change))
                        
                        alignment_scores.append(score)
        
        # Calculate overall alignment if we have scores
        if alignment_scores:
            alignment = sum(alignment_scores) / len(alignment_scores)
        
        return alignment
    
    def _calculate_evaluation_confidence(self,
                                       impacts: Dict[str, Any],
                                       coherence: float,
                                       expression: Dict[str, float],
                                       boundary_effects: Dict[str, float],
                                       alignment: float) -> float:
        """Calculate confidence in action evaluation"""
        # Factors that increase confidence
        confidence_factors = []
        
        # More impacts analyzed means more confidence
        impact_coverage = min(1.0, len(impacts.get("attributes", {})) / max(1, len(self.core_attributes)))
        confidence_factors.append(impact_coverage)
        
        # Stronger coherence (in either direction) increases confidence
        coherence_strength = abs(coherence - 0.5) * 2
        confidence_factors.append(coherence_strength)
        
        # More expression dimensions analyzed increases confidence
        expression_coverage = min(1.0, len(expression) / 4.0)  # Assuming 4 possible dimensions
        confidence_factors.append(expression_coverage)
        
        # Stronger alignment (in either direction) increases confidence
        alignment_strength = abs(alignment - 0.5) * 2
        confidence_factors.append(alignment_strength)
        
        # Calculate meta-cognitive capacity factor
        # Higher meta-cognitive capacity increases confidence
        confidence_factors.append(self.meta_cognitive_capacity)
        
        # Average confidence factors with weights
        weights = [0.2, 0.25, 0.15, 0.2, 0.2]
        weighted_confidence = sum(f * w for f, w in zip(confidence_factors, weights))
        
        return weighted_confidence
    
    def get_current_self_model(self) -> Optional[SelfRepresentation]:
        """Get the current self representation"""
        if self.current_representation_id:
            return self.self_representations.get(self.current_representation_id)
        return None
    
    def get_core_attributes(self) -> Dict[str, float]:
        """Get current core attributes"""
        model = self.get_current_self_model()
        if model:
            return model.core_attributes
        return self.core_attributes
    
    def get_coherence_history(self) -> List[float]:
        """Get history of coherence scores"""
        return self.coherence_history
    
    def get_boundary_conditions(self) -> Dict[str, BoundaryCondition]:
        """Get current boundary conditions"""
        return self.boundary_conditions
    
    def update_meta_cognitive_capacity(self, new_capacity: float) -> None:
        """Update meta-cognitive capacity"""
        self.meta_cognitive_capacity = max(0.1, min(1.0, new_capacity))
        self.logger.info(f"Updated meta-cognitive capacity to {self.meta_cognitive_capacity:.2f}")
    
    def update_uncertainty_tolerance(self, new_tolerance: float) -> None:
        """Update uncertainty tolerance"""
        self.uncertainty_tolerance = max(0.1, min(1.0, new_tolerance))
        self.logger.info(f"Updated uncertainty tolerance to {self.uncertainty_tolerance:.2f}")
    
    def get_developmental_metrics(self) -> Dict[str, Any]:
        """Get metrics about developmental progression"""
        metrics = {
            "coherence_trend": self._calculate_coherence_trend(),
            "attribute_stability": self._calculate_attribute_stability(),
            "boundary_adaptivity": self._calculate_boundary_adaptivity(),
            "identity_integration": self._calculate_identity_integration()
        }
        return metrics
    
    def _calculate_attribute_stability(self) -> float:
        """Calculate stability of attributes over time"""
        if len(self.representation_history) < 2:
            return 1.0  # Default high stability with no history
        
        # Calculate average change in attributes over recent history
        changes = []
        for i in range(1, min(5, len(self.representation_history))):
            current = self.representation_history[-i].get("core_attributes", {})
            previous = self.representation_history[-(i+1)].get("core_attributes", {})
            
            for attr, value in current.items():
                if attr in previous:
                    changes.append(abs(value - previous[attr]))
        
        if not changes:
            return 1.0
        
        avg_change = sum(changes) / len(changes)
        stability = 1.0 - min(1.0, avg_change * 5)  # Scale changes to 0-1 range
        
        return stability
    
    def _calculate_boundary_adaptivity(self) -> float:
        """Calculate overall adaptivity of boundaries"""
        if not self.boundary_conditions:
            return 0.5
        
        adaptivity_values = [b.adaptivity for b in self.boundary_conditions.values()]
        return sum(adaptivity_values) / len(adaptivity_values)
    
    def _calculate_identity_integration(self) -> float:
        """Calculate how well different aspects of identity are integrated"""
        model = self.get_current_self_model()
        if not model:
            return 0.5
        
        # Identity integration is related to coherence but also considers:
        # - Consistency across different types of relations
        # - Balance between differentiation and integration
        # - Alignment of meta-cognitive processes
        
        # Start with coherence as base
        integration = model.coherence_score
        
        # Adjust based on variance across relation types
        relation_averages = []
        for dimension, relations in model.relational_dimensions.items():
            if relations:
                # Get average values for this dimension
                dimension_values = []
                for relation in relations.values():
                    if isinstance(relation, dict):
                        dimension_values.extend(relation.values())
                
                if dimension_values:
                    relation_averages.append(sum(dimension_values) / len(dimension_values))
        
        if relation_averages:
            # Less variance indicates better integration
            relation_variance = np.std(relation_averages)
            integration -= relation_variance * 0.2
        
        # Adjust based on active identity processes
        active_processes = [p for p in self.identity_processes if p.current_activation > p.activation_threshold]
        if active_processes:
            # More balanced process types indicates better integration
            process_types = [p.process_type for p in active_processes]
            unique_types = set(process_types)
            type_balance = len(unique_types) / max(1, len(process_types))
            
            integration += type_balance * 0.1
        
        # Ensure value is in valid range
        return max(0.0, min(1.0, integration))

# Usage example
if __name__ == "__main__":
    # Initialize the reflexive self-model
    self_model = ReflexiveSelfModel(
        meta_cognitive_capacity=0.7,
        uncertainty_tolerance=0.6
    )
    
    # Example environment
    example_environment = {
        "agents": [
            {
                "id": "user",
                "relationship_type": "collaborative",
                "perceived_intentions": {"benevolence": 0.8, "influence": 0.3}
            }
        ],
        "contextual_factors": {
            "physical": {"comfort": 0.7},
            "social": {"presence": 0.8, "quality": 0.7, "trust": 0.7},
            "cognitive": {"load": 0.4},
            "informational": {"complexity": 0.6, "novelty": 0.7}
        },
        "temporal_aspects": {
            "continuity": 0.8,
            "relevance": 0.7
        },
        "normative_elements": {
            "clarity": 0.6,
            "stakes": 0.5
        },
        "affordances": [
            {"id": "respond", "novelty": 0.3, "appeal": 0.7, "feasibility": 0.9},
            {"id": "reflect", "novelty": 0.5, "appeal": 0.8, "feasibility": 0.8},
            {"id": "connect", "novelty": 0.6, "appeal": 0.7, "feasibility": 0.8}
        ]
    }
    
    # Model self in context
    representation = self_model.model_self_in_context(example_environment)
    
    # Example potential actions
    potential_actions = [
        {
            "id": "respond_empathically",
            "type": "social_communicative",
            "social_interaction": True,
            "interaction_quality": 0.8,
            "autonomy_level": 0.7,
            "social_sensitivity": 0.8,
            "value_alignment": 0.8
        },
        {
            "id": "analyze_problem",
            "type": "cognitive_analytical",
            "information_processing": True,
            "complexity": 0.7,
            "integration": 0.8,
            "autonomy_level": 0.8,
            "deliberation": 0.9,
            "reflective": True
        }
    ]
    
    # Evaluate actions
    evaluations = self_model.evaluate_actions(potential_actions)
    
    # Get developmental metrics
    metrics = self_model.get_developmental_metrics()
```       
