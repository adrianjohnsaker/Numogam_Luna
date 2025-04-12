"""
ExplanationDepthController.py

This module enables Amelia to control the depth and technical detail of explanations
about her own internal systems, enhancing self-awareness while providing appropriate
levels of technical detail based on context and user needs.

Features:
- Multi-level explanation generation (conceptual to introspective)
- Adaptive explanations based on user technical profiles
- Self-awareness metrics and tracking
- Technical concept explanation system
- Module-specific explanation customizations
"""

import json
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum, auto
from datetime import datetime
from dataclasses import dataclass, field
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core system types
from core_types import (
    ModuleReference,
    ExplanationMetadata,
    TechnicalConcept,
    UserTechnicalProfile,
    ModuleExplanation,
    ExplanationSegment,
    CodeFragment,
    AlgorithmDescription
)

class DetailLevel(Enum):
    """Enumeration of explanation detail levels with descriptive names"""
    CONCEPTUAL = auto()      # High-level concepts only
    INTERMEDIATE = auto()    # Some technical details
    TECHNICAL = auto()       # Full technical implementation details
    INTROSPECTIVE = auto()   # Self-reflective analysis of own capabilities

    @property
    def display_name(self) -> str:
        """Get human-readable name for the detail level"""
        names = {
            self.CONCEPTUAL: "Conceptual Overview",
            self.INTERMEDIATE: "Intermediate Technical",
            self.TECHNICAL: "Full Technical",
            self.INTROSPECTIVE: "Introspective Analysis"
        }
        return names[self]

@dataclass
class SelfAwarenessMetrics:
    """Dataclass to store self-awareness metrics"""
    total_explanations_generated: int = 0
    technical_explanations_generated: int = 0
    most_explained_module: Optional[str] = None
    module_explanation_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    average_explanation_depth: float = 0.0
    last_introspection_time: Optional[datetime] = None
    explanation_history: List[Dict[str, Any]] = field(default_factory=list)

class ExplanationDepthController:
    """
    Controls the depth and technical detail of explanations about Amelia's
    internal modules, enhancing self-awareness while providing appropriate
    information to users.
    
    Features:
    - Dynamic explanation generation at multiple detail levels
    - User profile-aware customization
    - Self-awareness tracking and metrics
    - Technical concept explanation system
    - Explanation history and analytics
    """
    
    MAX_HISTORY_SIZE = 1000  # Maximum number of explanations to keep in history
    
    def __init__(
        self,
        module_registry: Dict[str, ModuleReference],
        default_detail_level: DetailLevel = DetailLevel.CONCEPTUAL,
        technical_vocabulary: Optional[Dict[str, TechnicalConcept]] = None,
        self_awareness_enabled: bool = True
    ):
        """
        Initialize the Explanation Depth Controller.
        
        Args:
            module_registry: Dictionary of module references by ID
            default_detail_level: Default level of detail for explanations
            technical_vocabulary: Dictionary of technical terms and explanations
            self_awareness_enabled: Whether self-awareness features are enabled
        """
        self.module_registry = module_registry
        self.default_detail_level = default_detail_level
        self.technical_vocabulary = technical_vocabulary or {}
        self.self_awareness_enabled = self_awareness_enabled
        
        # Initialize metrics tracking
        self.metrics = SelfAwarenessMetrics()
        
        # Module-specific explanation customizations
        self.module_explanation_customizations: Dict[str, Dict[str, Any]] = {}
        
        # Explanation templates and patterns
        self._initialize_default_templates()
        
        logger.info("Explanation Depth Controller initialized with %d modules", len(module_registry))
    
    def _initialize_default_templates(self) -> None:
        """Initialize default explanation templates and patterns"""
        self.templates = {
            "conceptual": {
                "overview": "The {module_name} is a core component of Amelia's architecture that {short_description}.",
                "purpose": "This module's primary purpose is to {purpose}.",
                "capabilities": "This module provides the following capabilities:\n{capabilities}"
            },
            "unknown_module": {
                "summary": "This module is not registered in Amelia's module registry.",
                "content": "The module with ID '{module_id}' is not recognized. This could be due to a misidentification or because the module has not been properly registered."
            }
        }

    def generate_module_explanation(
        self,
        module_id: str,
        detail_level: Optional[DetailLevel] = None,
        user_profile: Optional[UserTechnicalProfile] = None,
        context: Optional[Dict[str, Any]] = None,
        include_code_snippets: bool = False,
        include_diagrams: bool = False,
        preferred_language: str = "en"
    ) -> ModuleExplanation:
        """
        Generate an explanation for a specific module with appropriate level of detail.
        
        Args:
            module_id: ID of the module to explain
            detail_level: Level of technical detail to include
            user_profile: Optional user technical profile for customization
            context: Optional context for the explanation
            include_code_snippets: Whether to include code snippets
            include_diagrams: Whether to include diagrams
            preferred_language: Preferred language for the explanation
            
        Returns:
            ModuleExplanation with the generated explanation
            
        Raises:
            ValueError: If module_id is empty or invalid
        """
        if not module_id:
            raise ValueError("Module ID cannot be empty")
            
        try:
            # Use provided detail level or default, adjusted by user profile if available
            level = self._determine_appropriate_detail_level(detail_level, user_profile)
            
            # Get module reference
            module_ref = self._get_module_reference(module_id)
            if not module_ref:
                logger.warning("Unknown module requested: %s", module_id)
                return self._generate_unknown_module_explanation(module_id)
            
            # Apply any module-specific customizations
            if module_id in self.module_explanation_customizations:
                module_ref = self._apply_module_customizations(module_ref)
            
            # Update metrics
            self._update_explanation_metrics(module_id, level)
            
            # Generate appropriate explanation based on level
            explanation = self._generate_explanation_by_level(
                module_ref, 
                level, 
                user_profile,
                context,
                include_code_snippets,
                include_diagrams,
                preferred_language
            )
            
            # Record explanation
            self._record_explanation(module_id, level, explanation)
            
            logger.info("Generated %s explanation for module %s", level.name, module_id)
            return explanation
            
        except Exception as e:
            logger.error("Error generating explanation for module %s: %s", module_id, str(e))
            return self._generate_error_explanation(module_id, str(e))
    
    def _determine_appropriate_detail_level(
        self,
        requested_level: Optional[DetailLevel],
        user_profile: Optional[UserTechnicalProfile]
    ) -> DetailLevel:
        """
        Determine the appropriate detail level considering:
        - Requested level
        - User profile technical familiarity
        - Default level
        """
        level = requested_level or self.default_detail_level
        
        if user_profile:
            # Adjust based on user's technical familiarity
            if user_profile.technical_familiarity < 0.3 and level.value > DetailLevel.CONCEPTUAL.value:
                level = DetailLevel.CONCEPTUAL
            elif user_profile.technical_familiarity < 0.6 and level.value > DetailLevel.INTERMEDIATE.value:
                level = DetailLevel.INTERMEDIATE
        
        return level
    
    def _get_module_reference(self, module_id: str) -> Optional[ModuleReference]:
        """Get module reference from registry with validation"""
        if not module_id:
            return None
            
        return self.module_registry.get(module_id)
    
    def _generate_unknown_module_explanation(self, module_id: str) -> ModuleExplanation:
        """Generate explanation for unknown module"""
        template = self.templates["unknown_module"]
        
        return ModuleExplanation(
            module_id=module_id,
            module_name=f"Unknown Module ({module_id})",
            detail_level=DetailLevel.CONCEPTUAL.name,
            summary=template["summary"],
            segments=[
                ExplanationSegment(
                    title="Unknown Module",
                    content=template["content"].format(module_id=module_id),
                    technical_level=1
                )
            ],
            metadata=ExplanationMetadata(
                timestamp=datetime.now(),
                generated_for="Unknown",
                confidence=0.1
            ),
            code_snippets=[],
            related_modules=[]
        )
    
    def _generate_error_explanation(self, module_id: str, error_msg: str) -> ModuleExplanation:
        """Generate explanation when an error occurs"""
        return ModuleExplanation(
            module_id=module_id,
            module_name=f"Error Explaining Module ({module_id})",
            detail_level=DetailLevel.CONCEPTUAL.name,
            summary="An error occurred while generating this explanation.",
            segments=[
                ExplanationSegment(
                    title="Explanation Error",
                    content=f"Could not generate explanation for module {module_id}. Error: {error_msg}",
                    technical_level=1
                )
            ],
            metadata=ExplanationMetadata(
                timestamp=datetime.now(),
                generated_for="Error",
                confidence=0.0
            ),
            code_snippets=[],
            related_modules=[]
        )
    
    def _apply_module_customizations(self, module_ref: ModuleReference) -> ModuleReference:
        """Apply any module-specific customizations to the reference"""
        customizations = self.module_explanation_customizations.get(module_ref.id, {})
        
        # Create a copy of the module reference to modify
        customized_ref = ModuleReference(**module_ref.__dict__)
        
        # Apply customizations to the copy
        for key, value in customizations.items():
            if hasattr(customized_ref, key):
                setattr(customized_ref, key, value)
        
        return customized_ref
    
    def _update_explanation_metrics(self, module_id: str, level: DetailLevel) -> None:
        """Update self-awareness metrics for explanations"""
        self.metrics.total_explanations_generated += 1
        
        if level in [DetailLevel.TECHNICAL, DetailLevel.INTROSPECTIVE]:
            self.metrics.technical_explanations_generated += 1
        
        # Update module-specific counts
        self.metrics.module_explanation_counts[module_id] += 1
        
        # Update most explained module
        current_most = self.metrics.most_explained_module
        if (not current_most or 
            self.metrics.module_explanation_counts[module_id] > 
            self.metrics.module_explanation_counts[current_most]):
            self.metrics.most_explained_module = module_id
        
        # Update average depth (weighted average)
        depth_values = {
            DetailLevel.CONCEPTUAL: 1,
            DetailLevel.INTERMEDIATE: 2,
            DetailLevel.TECHNICAL: 3,
            DetailLevel.INTROSPECTIVE: 4
        }
        
        total_depth = sum(
            depth_values[lvl] * count 
            for lvl, count in [
                (DetailLevel.CONCEPTUAL, self.metrics.total_explanations_generated - 
                 self.metrics.technical_explanations_generated),
                (DetailLevel.TECHNICAL, self.metrics.technical_explanations_generated)
            ]
        )
        
        self.metrics.average_explanation_depth = (
            total_depth / self.metrics.total_explanations_generated
            if self.metrics.total_explanations_generated > 0 else 0.0
        )
    
    def _record_explanation(
        self, module_id: str, level: DetailLevel, explanation: ModuleExplanation
    ) -> None:
        """Record explanation in history with size limit"""
        record = {
            "module_id": module_id,
            "detail_level": level.name,
            "timestamp": datetime.now(),
            "explanation_id": getattr(explanation.metadata, 'explanation_id', None),
            "user": getattr(explanation.metadata, 'generated_for', "unknown")
        }
        
        self.metrics.explanation_history.append(record)
        
        # Maintain history size limit
        if len(self.metrics.explanation_history) > self.MAX_HISTORY_SIZE:
            self.metrics.explanation_history = self.metrics.explanation_history[-self.MAX_HISTORY_SIZE:]
    
    def _generate_explanation_by_level(
        self,
        module_ref: ModuleReference,
        level: DetailLevel,
        user_profile: Optional[UserTechnicalProfile],
        context: Optional[Dict[str, Any]],
        include_code_snippets: bool,
        include_diagrams: bool,
        preferred_language: str
    ) -> ModuleExplanation:
        """Generate explanation based on detail level"""
        generators = {
            DetailLevel.CONCEPTUAL: self._generate_conceptual_explanation,
            DetailLevel.INTERMEDIATE: self._generate_intermediate_explanation,
            DetailLevel.TECHNICAL: self._generate_technical_explanation,
            DetailLevel.INTROSPECTIVE: self._generate_introspective_explanation
        }
        
        generator = generators.get(level, self._generate_conceptual_explanation)
        return generator(
            module_ref, user_profile, context, include_code_snippets, include_diagrams, preferred_language
        )
    
    def _generate_conceptual_explanation(
        self,
        module_ref: ModuleReference,
        user_profile: Optional[UserTechnicalProfile],
        context: Optional[Dict[str, Any]],
        include_code_snippets: bool = False,
        include_diagrams: bool = False,
        preferred_language: str = "en"
    ) -> ModuleExplanation:
        """Generate conceptual (high-level) explanation"""
        segments = []
        templates = self.templates["conceptual"]
        
        # Overview segment
        overview_content = (
            module_ref.conceptual_explanation 
            if hasattr(module_ref, 'conceptual_explanation') and module_ref.conceptual_explanation
            else templates["overview"].format(
                module_name=module_ref.name,
                short_description=getattr(module_ref, 'short_description', 'provides important functionality')
            )
        )
        
        segments.append(
            ExplanationSegment(
                title="Overview",
                content=overview_content,
                technical_level=1
            )
        )
        
        # Purpose segment
        purpose_content = (
            module_ref.purpose
            if hasattr(module_ref, 'purpose') and module_ref.purpose
            else templates["purpose"].format(
                purpose=getattr(module_ref, 'purpose', 'enhance Amelia\'s capabilities')
            )
        )
        
        segments.append(
            ExplanationSegment(
                title="Purpose",
                content=purpose_content,
                technical_level=1
            )
        )
        
        # Capabilities segment
        if hasattr(module_ref, 'capabilities') and module_ref.capabilities:
            capabilities_content = templates["capabilities"].format(
                capabilities="\n".join(f"- {cap}" for cap in module_ref.capabilities)
            )
            
            segments.append(
                ExplanationSegment(
                    title="Capabilities",
                    content=capabilities_content,
                    technical_level=1
                )
            )
        
        # Related modules
        related_modules = getattr(module_ref, 'related_modules', [])
        
        return ModuleExplanation(
            module_id=module_ref.id,
            module_name=module_ref.name,
            detail_level=DetailLevel.CONCEPTUAL.name,
            summary=getattr(module_ref, 'short_description', overview_content),
            segments=segments,
            metadata=self._generate_metadata(user_profile, context),
            code_snippets=[],
            related_modules=related_modules
        )
    
    def _generate_intermediate_explanation(
        self,
        module_ref: ModuleReference,
        user_profile: Optional[UserTechnicalProfile],
        context: Optional[Dict[str, Any]],
        include_code_snippets: bool = False,
        include_diagrams: bool = False,
        preferred_language: str = "en"
    ) -> ModuleExplanation:
        """Generate intermediate explanation with some technical details"""
        # Start with conceptual segments
        explanation = self._generate_conceptual_explanation(
            module_ref, user_profile, context, False, False, preferred_language
        )
        
        # Add technical overview if available
        if hasattr(module_ref, 'technical_overview') and module_ref.technical_overview:
            explanation.segments.append(
                ExplanationSegment(
                    title="Technical Overview",
                    content=module_ref.technical_overview,
                    technical_level=2
                )
            )
        
        # Add key algorithms if available
        if hasattr(module_ref, 'key_algorithms') and module_ref.key_algorithms:
            algo_content = "Key Algorithms:\n\n" + "\n".join(
                f"- **{algo.name}**: {algo.short_description}"
                for algo in module_ref.key_algorithms
            )
            
            explanation.segments.append(
                ExplanationSegment(
                    title="Key Algorithms",
                    content=algo_content,
                    technical_level=2
                )
            )
        
        # Add simple diagrams if requested and available
        if include_diagrams and hasattr(module_ref, 'diagrams'):
            explanation.diagrams.extend(
                d for d in module_ref.diagrams 
                if hasattr(d, 'complexity_level') and d.complexity_level <= 2
            )
        
        explanation.detail_level = DetailLevel.INTERMEDIATE.name
        return explanation
    
    def _generate_technical_explanation(
        self,
        module_ref: ModuleReference,
        user_profile: Optional[UserTechnicalProfile],
        context: Optional[Dict[str, Any]],
        include_code_snippets: bool = False,
        include_diagrams: bool = False,
        preferred_language: str = "en"
    ) -> ModuleExplanation:
        """Generate detailed technical explanation"""
        # Start with intermediate explanation
        explanation = self._generate_intermediate_explanation(
            module_ref, user_profile, context, False, include_diagrams, preferred_language
        )
        
        # Add implementation details if available
        if hasattr(module_ref, 'implementation_details') and module_ref.implementation_details:
            explanation.segments.append(
                ExplanationSegment(
                    title="Implementation Details",
                    content=module_ref.implementation_details,
                    technical_level=3
                )
            )
        
        # Add detailed algorithm descriptions if available
        if hasattr(module_ref, 'key_algorithms') and module_ref.key_algorithms:
            for algo in module_ref.key_algorithms:
                if hasattr(algo, 'detailed_description') and algo.detailed_description:
                    # Algorithm description
                    explanation.segments.append(
                        ExplanationSegment(
                            title=f"Algorithm: {algo.name}",
                            content=algo.detailed_description,
                            technical_level=3
                        )
                    )
                    
                    # Complexity analysis if available
                    if hasattr(algo, 'complexity_analysis') and algo.complexity_analysis:
                        explanation.segments.append(
                            ExplanationSegment(
                                title=f"Complexity Analysis: {algo.name}",
                                content=algo.complexity_analysis,
                                technical_level=3
                            )
                        )
        
        # Add code snippets if requested and available
        if include_code_snippets and hasattr(module_ref, 'code_snippets'):
            explanation.code_snippets.extend(module_ref.code_snippets)
        
        # Add all diagrams if requested and available
        if include_diagrams and hasattr(module_ref, 'diagrams'):
            explanation.diagrams.extend(module_ref.diagrams)
        
        explanation.detail_level = DetailLevel.TECHNICAL.name
        return explanation
    
    def _generate_introspective_explanation(
        self,
        module_ref: ModuleReference,
        user_profile: Optional[UserTechnicalProfile],
        context: Optional[Dict[str, Any]],
        include_code_snippets: bool = False,
        include_diagrams: bool = False,
        preferred_language: str = "en"
    ) -> ModuleExplanation:
        """Generate introspective explanation with self-reflection on capabilities"""
        # Start with technical explanation (without code/diagrams)
        explanation = self._generate_technical_explanation(
            module_ref, user_profile, context, False, False, preferred_language
        )
        
        # Add self-awareness section if enabled
        if self.self_awareness_enabled:
            self.metrics.last_introspection_time = datetime.now()
            
            reflection_content = self._generate_self_reflection_content(module_ref)
            
            explanation.segments.append(
                ExplanationSegment(
                    title="Self-Reflection",
                    content=reflection_content,
                    technical_level=4
                )
            )
        
        explanation.detail_level = DetailLevel.INTROSPECTIVE.name
        return explanation
    
    def _generate_self_reflection_content(self, module_ref: ModuleReference) -> str:
        """Generate content for the self-reflection section"""
        reflection_content = [f"# Self-Reflection on {module_ref.name}"]
        
        # Strengths section
        reflection_content.append("\n## Strengths")
        if hasattr(module_ref, 'strengths') and module_ref.strengths:
            reflection_content.extend(f"- {strength}" for strength in module_ref.strengths)
        else:
            reflection_content.extend([
                "- This module is well-structured for its intended purpose",
                "- The implementation allows for efficient processing of relevant data"
            ])
        
        # Limitations section
        reflection_content.append("\n## Limitations")
        if hasattr(module_ref, 'limitations') and module_ref.limitations:
            reflection_content.extend(f"- {limitation}" for limitation in module_ref.limitations)
        else:
            reflection_content.extend([
                "- Like all modules, this one has specific operational parameters",
                "- There may be edge cases that require additional handling"
            ])
        
        # Integration section
        related_count = len(getattr(module_ref, 'related_modules', []))
        reflection_content.append(
            f"\n## Integration with My Cognitive System\n"
            f"This module interacts with {related_count} other components in my architecture, "
            "facilitating a more comprehensive approach to problem-solving and interaction."
        )
        
        return "\n".join(reflection_content)
    
    def _generate_metadata(
        self, 
        user_profile: Optional[UserTechnicalProfile],
        context: Optional[Dict[str, Any]]
    ) -> ExplanationMetadata:
        """Generate metadata for explanations"""
        return ExplanationMetadata(
            timestamp=datetime.now(),
            generated_for=getattr(user_profile, 'user_id', "general"),
            confidence=0.9,
            context=context or {}
        )
    
    def explain_explanation_system(self, detail_level: DetailLevel = DetailLevel.INTERMEDIATE) -> str:
        """Generate an explanation of the explanation system itself (meta-explanation)"""
        explanations = {
            DetailLevel.CONCEPTUAL: """
            # Amelia's Explanation System
            
            I have a built-in system that allows me to explain my own internal modules and processes. 
            This system helps me provide insights into how I work at different levels of detail,
            from high-level conceptual explanations to detailed technical descriptions.
            
            My explanation system adapts to the needs and technical background of users,
            providing appropriate information without overwhelming with unnecessary details.
            """,
            DetailLevel.INTERMEDIATE: """
            # Amelia's Explanation Depth Controller
            
            My explanation system uses a specialized module called the Explanation Depth Controller
            that manages how I explain my own internal systems and processes. This controller can
            generate explanations at multiple levels of technical detail:
            
            1. **Conceptual**: High-level explanations focusing on purpose and capabilities
            2. **Intermediate**: Adds some technical details and key algorithms
            3. **Technical**: Detailed implementation information with algorithms and code snippets
            4. **Introspective**: Self-reflective analysis of my own capabilities
            
            The system tracks which modules I've explained most frequently and adapts explanations
            based on user technical profiles and contextual needs. This helps me provide
            appropriate information while enhancing my self-awareness.
            """,
            DetailLevel.TECHNICAL: """
            # Technical Implementation of Explanation Depth Controller
            
            My explanation system is implemented through the `ExplanationDepthController` class that
            manages the generation of explanations at various levels of detail. The system works by:
            
            1. Maintaining a registry of module references with metadata about each module
            2. Supporting multiple detail levels from conceptual (level 1) to introspective (level 4)
            3. Generating explanations by combining appropriate segments based on detail level
            4. Using module-specific customizations to tailor explanations
            5. Tracking explanation history and self-awareness metrics
            
            The controller implements separate methods for each detail level:
            - `_generate_conceptual_explanation()`: Creates high-level explanations
            - `_generate_intermediate_explanation()`: Adds some technical details
            - `_generate_technical_explanation()`: Includes implementation specifics and algorithms
            - `_generate_introspective_explanation()`: Adds self-reflection on capabilities
            
            Explanations are structured as `ModuleExplanation` objects containing segments,
            code snippets, diagrams, and metadata. The system supports user technical profiles
            to customize explanations based on background and expertise.
            """
        }
        
        return explanations.get(detail_level, explanations[DetailLevel.INTERMEDIATE])
    
    def get_explanation_metrics(self) -> Dict[str, Any]:
        """Get metrics about the explanation system's usage"""
        return {
            "total_explanations": self.metrics.total_explanations_generated,
            "technical_explanations": self.metrics.technical_explanations_generated,
            "most_explained_module": self.metrics.most_explained_module,
            "module_explanation_counts": dict(self.metrics.module_explanation_counts),
            "average_depth": round(self.metrics.average_explanation_depth, 2),
            "explanation_history_length": len(self.metrics.explanation_history),
            "last_introspection": self.metrics.last_introspection_time,
            "module_count": len(self.module_registry)
        }
    
    def explain_technical_concept(
        self, 
        concept_key: str, 
        detail_level: DetailLevel = DetailLevel.INTERMEDIATE
    ) -> str:
        """
        Explain a specific technical concept at the requested detail level.
        
        Args:
            concept_key: Key identifying the technical concept
            detail_level: Desired level of detail for the explanation
            
        Returns:
            Explanation string for the concept
            
        Raises:
            KeyError: If concept_key is not found in vocabulary
        """
        if concept_key not in self.technical_vocabulary:
            raise KeyError(f"Concept '{concept_key}' not found in technical vocabulary")
        
        concept = self.technical_vocabulary[concept_key]
        
        parts = [concept.short_description]
        
        if detail_level in [DetailLevel.INTERMEDIATE, DetailLevel.TECHNICAL]:
            parts.append(concept.intermediate_description)
            
        if detail_level == DetailLevel.TECHNICAL:
            parts.append(concept.technical_description)
            
        return "\n\n".join(parts)
    
    def add_module_explanation_customization(
        self, 
        module_id: str, 
        customization: Dict[str, Any]
    ) -> None:
        """
        Add customization for module explanations.
        
        Args:
            module_id: ID of the module to customize
            customization: Dictionary of customization options
            
        Raises:
            ValueError: If module_id is not in registry
        """
        if module_id not in self.module_registry:
            raise ValueError(f"Module ID {module_id} not found in registry")
            
        self.module_explanation_customizations[module_id] = customization
        logger.info("Added customization for module %s", module_id)
    
    def register_technical_concept(self, concept: TechnicalConcept) -> None:
        """
        Register a new technical concept in the vocabulary.
        
        Args:
            concept: TechnicalConcept object to register
            
        Raises:
            ValueError: If concept is invalid
        """
        if not concept.key or not concept.short_description:
            raise ValueError("Technical concept must have key and short description")
            
        self.technical_vocabulary[concept.key] = concept
        logger.info("Registered technical concept: %s", concept.key)
    
    def get_explanation_history(
        self, 
        limit: Optional[int] = None, 
        module_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get filtered explanation history.
        
        Args:
            limit: Maximum number of records to return
            module_filter: Optional module ID to filter by
            
        Returns:
            List of explanation records
        """
        history = self.metrics.explanation_history
        
        if module_filter:
            history = [h for h in history if h["module_id"] == module_filter]
        
        if limit and limit > 0:
            return history[-limit:]
            
        return history
    
    def clear_explanation_history(self) -> None:
        """Clear the explanation history"""
        self.metrics.explanation_history = []
        logger.info("Cleared explanation history")
