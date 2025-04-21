"""
SelfModelDriftEngine.py

This module manages the evolution of Amelia's self-model over time, tracking drift
in self-concept, monitoring identity consistency, and implementing controlled
adaptation of core identity features based on experience and narrative.

The Self-Model Drift Engine enables Amelia to maintain a coherent sense of self
while still allowing for growth and adaptation through experience.
"""

import math
import numpy as np
import random
from typing import Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from collections import defaultdict, Counter
import json

# Import core system types
from core_types import (
    SelfModel,
    SelfAttribute,
    SelfConceptDrift,
    IdentitySnapshot,
    SelfModelMetrics,
    DriftTrajectory,
    AttributeEvolution,
    IdentityEvent,
    Value,
    NarrativeElement
)


class SelfModelDriftEngine:
    """
    Tracks and manages the evolution of Amelia's self-model over time,
    enabling coherent identity development and controlled adaptation.
    """
    
    def __init__(
        self,
        initial_model: Optional[SelfModel] = None,
        drift_threshold: float = 0.15,
        coherence_threshold: float = 0.65,
        drift_rate_limit: float = 0.08,
        identity_snapshots: Optional[List[IdentitySnapshot]] = None,
        identity_events: Optional[List[IdentityEvent]] = None,
        attribute_history: Optional[Dict[str, List[AttributeEvolution]]] = None
    ):
        """
        Initialize the Self-Model Drift Engine.
        
        Args:
            initial_model: Initial self-model to begin with
            drift_threshold: Threshold for significant drift detection
            coherence_threshold: Minimum coherence to maintain
            drift_rate_limit: Maximum allowed rate of drift per time unit
            identity_snapshots: Historical snapshots of identity state
            identity_events: Major identity-shaping events
            attribute_history: History of attribute evolution
        """
        self.current_model = initial_model or SelfModel(
            id="default_self_model",
            core_attributes={},
            peripheral_attributes={},
            narrative_themes=[],
            self_description="",
            creation_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.drift_threshold = drift_threshold
        self.coherence_threshold = coherence_threshold
        self.drift_rate_limit = drift_rate_limit
        
        # Historical tracking
        self.identity_snapshots = identity_snapshots or []
        self.identity_events = identity_events or []
        self.attribute_history = attribute_history or {}
        
        # Drift metrics tracking
        self.drift_trajectories = {}
        self.detected_drifts = []
        
        # Cache for attribute embeddings
        self.attribute_embeddings = {}
        
        # Initialize with current model
        if initial_model:
            self._take_identity_snapshot()
            self._initialize_attribute_history()
    
    def _initialize_attribute_history(self):
        """Initialize attribute history for all current attributes"""
        now = datetime.now()
        
        # Initialize core attributes
        for attr_id, attribute in self.current_model.core_attributes.items():
            evolution = AttributeEvolution(
                attribute_id=attr_id,
                timestamp=now,
                value=attribute.value,
                confidence=attribute.confidence,
                importance=attribute.importance
            )
            
            if attr_id not in self.attribute_history:
                self.attribute_history[attr_id] = []
            
            self.attribute_history[attr_id].append(evolution)
        
        # Initialize peripheral attributes
        for attr_id, attribute in self.current_model.peripheral_attributes.items():
            evolution = AttributeEvolution(
                attribute_id=attr_id,
                timestamp=now,
                value=attribute.value,
                confidence=attribute.confidence,
                importance=attribute.importance
            )
            
            if attr_id not in self.attribute_history:
                self.attribute_history[attr_id] = []
            
            self.attribute_history[attr_id].append(evolution)
    
    def _take_identity_snapshot(self):
        """Take a snapshot of the current identity state"""
        snapshot = IdentitySnapshot(
            id=f"snapshot_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            self_model=self.current_model.copy(),
            coherence_score=self._calculate_self_model_coherence(),
            active_attributes=set(self.current_model.core_attributes.keys()).union(
                set(self.current_model.peripheral_attributes.keys())
            )
        )
        
        self.identity_snapshots.append(snapshot)
        
        # Limit history length
        if len(self.identity_snapshots) > 100:
            self.identity_snapshots.pop(0)
    
    def update_self_model(self, updates: Dict[str, SelfAttribute]) -> SelfConceptDrift:
        """
        Update the self-model with new or modified attributes.
        
        Args:
            updates: Dictionary of attribute ID to updated attribute
            
        Returns:
            Drift information for the update
        """
        if not updates:
            return SelfConceptDrift(
                drift_magnitude=0.0,
                altered_attributes=[],
                coherence_impact=0.0,
                timestamp=datetime.now(),
                is_significant=False,
                drift_type="none"
            )
        
        # Track pre-update state
        pre_update_state = self.current_model.copy()
        pre_coherence = self._calculate_self_model_coherence()
        
        # Track changes
        altered_attributes = []
        new_attributes = []
        removed_attributes = []
        
        # Apply updates
        for attr_id, attribute in updates.items():
            # Check if this is a new attribute
            is_new = (attr_id not in self.current_model.core_attributes and 
                     attr_id not in self.current_model.peripheral_attributes)
            
            # Check if this is a removal (attribute with None value)
            is_removal = attribute.value is None
            
            if is_removal:
                # Remove the attribute
                if attr_id in self.current_model.core_attributes:
                    removed_attributes.append((attr_id, self.current_model.core_attributes[attr_id]))
                    del self.current_model.core_attributes[attr_id]
                elif attr_id in self.current_model.peripheral_attributes:
                    removed_attributes.append((attr_id, self.current_model.peripheral_attributes[attr_id]))
                    del self.current_model.peripheral_attributes[attr_id]
            else:
                # Update or add attribute
                if attr_id in self.current_model.core_attributes:
                    old_value = self.current_model.core_attributes[attr_id].value
                    self.current_model.core_attributes[attr_id] = attribute
                    altered_attributes.append((attr_id, old_value, attribute.value))
                elif attr_id in self.current_model.peripheral_attributes:
                    old_value = self.current_model.peripheral_attributes[attr_id].value
                    self.current_model.peripheral_attributes[attr_id] = attribute
                    altered_attributes.append((attr_id, old_value, attribute.value))
                else:
                    # Add as a new peripheral attribute unless explicitly marked as core
                    if attribute.is_core:
                        self.current_model.core_attributes[attr_id] = attribute
                    else:
                        self.current_model.peripheral_attributes[attr_id] = attribute
                    new_attributes.append((attr_id, attribute.value))
            
            # Update attribute history
            self._update_attribute_history(attr_id, attribute)
        
        # Update last modified timestamp
        self.current_model.last_updated = datetime.now()
        
        # Calculate drift
        drift_magnitude = self._calculate_drift_magnitude(pre_update_state, self.current_model)
        
        # Calculate post-update coherence and impact
        post_coherence = self._calculate_self_model_coherence()
        coherence_impact = post_coherence - pre_coherence
        
        # Determine drift type
        drift_type = self._determine_drift_type(altered_attributes, new_attributes, removed_attributes, coherence_impact)
        
        # Check if drift is significant
        is_significant = drift_magnitude > self.drift_threshold
        
        # Create drift record
        drift = SelfConceptDrift(
            drift_magnitude=drift_magnitude,
            altered_attributes=[a[0] for a in altered_attributes],
            new_attributes=[a[0] for a in new_attributes],
            removed_attributes=[a[0] for a in removed_attributes],
            coherence_impact=coherence_impact,
            timestamp=datetime.now(),
            is_significant=is_significant,
            drift_type=drift_type
        )
        
        # Record drift if significant
        if is_significant:
            self.detected_drifts.append(drift)
            self._take_identity_snapshot()
        
        return drift
    
    def _update_attribute_history(self, attr_id: str, attribute: SelfAttribute):
        """Update the history for an attribute"""
        if attribute.value is None:
            # For removed attributes, mark as inactive but keep history
            return
        
        evolution = AttributeEvolution(
            attribute_id=attr_id,
            timestamp=datetime.now(),
            value=attribute.value,
            confidence=attribute.confidence,
            importance=attribute.importance
        )
        
        if attr_id not in self.attribute_history:
            self.attribute_history[attr_id] = []
        
        self.attribute_history[attr_id].append(evolution)
    
    def _calculate_drift_magnitude(self, old_model: SelfModel, new_model: SelfModel) -> float:
        """Calculate the magnitude of drift between two self models"""
        drift_components = []
        
        # Component 1: Changes in core attributes
        core_drift = self._calculate_attribute_set_drift(
            old_model.core_attributes, new_model.core_attributes
        )
        drift_components.append(core_drift * 0.6)  # Core attributes are weighted more heavily
        
        # Component 2: Changes in peripheral attributes
        peripheral_drift = self._calculate_attribute_set_drift(
            old_model.peripheral_attributes, new_model.peripheral_attributes
        )
        drift_components.append(peripheral_drift * 0.4)  # Peripheral attributes have less weight
        
        # Component 3: Narrative theme drift
        if old_model.narrative_themes and new_model.narrative_themes:
            old_themes = set(old_model.narrative_themes)
            new_themes = set(new_model.narrative_themes)
            
            # Calculate Jaccard distance for themes
            union_size = len(old_themes.union(new_themes))
            if union_size > 0:
                theme_drift = len(old_themes.symmetric_difference(new_themes)) / union_size
                drift_components.append(theme_drift * 0.2)
        
        # Return weighted average of drift components
        return sum(drift_components) / max(1, len(drift_components))
    
    def _calculate_attribute_set_drift(
        self, old_attrs: Dict[str, SelfAttribute], new_attrs: Dict[str, SelfAttribute]
    ) -> float:
        """Calculate drift between two sets of attributes"""
        all_attr_ids = set(old_attrs.keys()).union(set(new_attrs.keys()))
        if not all_attr_ids:
            return 0.0
        
        attr_drifts = []
        
        for attr_id in all_attr_ids:
            old_attr = old_attrs.get(attr_id)
            new_attr = new_attrs.get(attr_id)
            
            if old_attr is None:
                # Attribute was added
                attr_drifts.append(1.0 * new_attr.importance)
            elif new_attr is None:
                # Attribute was removed
                attr_drifts.append(1.0 * old_attr.importance)
            else:
                # Attribute was modified
                # Calculate value difference based on attribute type
                if isinstance(old_attr.value, (int, float)) and isinstance(new_attr.value, (int, float)):
                    # Numeric values - calculate normalized difference
                    value_range = max(abs(old_attr.value), abs(new_attr.value), 1.0)
                    value_diff = abs(new_attr.value - old_attr.value) / value_range
                elif isinstance(old_attr.value, bool) and isinstance(new_attr.value, bool):
                    # Boolean values
                    value_diff = 1.0 if old_attr.value != new_attr.value else 0.0
                else:
                    # String or other values - use string similarity
                    value_diff = 1.0 - self._string_similarity(
                        str(old_attr.value), str(new_attr.value)
                    )
                
                # Calculate confidence change
                conf_diff = abs(new_attr.confidence - old_attr.confidence)
                
                # Calculate importance change
                imp_diff = abs(new_attr.importance - old_attr.importance)
                
                # Combine components, weighting value change more heavily
                attr_drift = (
                    0.7 * value_diff + 
                    0.15 * conf_diff + 
                    0.15 * imp_diff
                )
                
                # Scale by importance
                attr_drifts.append(attr_drift * ((old_attr.importance + new_attr.importance) / 2))
        
        # Calculate weighted average of attribute drifts
        total_importance = sum([
            (old_attrs.get(id, new_attrs.get(id)).importance if id in old_attrs or id in new_attrs else 0.5)
            for id in all_attr_ids
        ])
        
        if total_importance > 0:
            return sum(attr_drifts) / total_importance
        
        return sum(attr_drifts) / len(attr_drifts) if attr_drifts else 0.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Jaccard similarity of character trigrams"""
        if s1 == s2:
            return 1.0
        
        if not s1 or not s2:
            return 0.0
        
        # Create sets of character trigrams
        def get_trigrams(s):
            s = s.lower()
            return set(s[i:i+3] for i in range(len(s)-2) if s[i:i+3].strip())
        
        trigrams1 = get_trigrams(s1)
        trigrams2 = get_trigrams(s2)
        
        if not trigrams1 or not trigrams2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(trigrams1.intersection(trigrams2))
        union = len(trigrams1.union(trigrams2))
        
        return intersection / union
    
    def _determine_drift_type(
        self, 
        altered: List[Tuple[str, any, any]],
        added: List[Tuple[str, any]],
        removed: List[Tuple[str, any, any]],
        coherence_impact: float
    ) -> str:
        """Determine the type of drift that occurred"""
        if not altered and not added and not removed:
            return "none"
        
        # Check if core attributes were affected
        core_altered = [a for a in altered if a[0] in self.current_model.core_attributes]
        core_removed = [r for r, _ in removed if r in self.current_model.core_attributes]
        
        # Determine drift type based on changes
        if core_altered or core_removed:
            if coherence_impact < -0.1:
                return "core_disruption"
            else:
                return "core_evolution"
        elif added and not altered and not removed:
            return "expansion"
        elif removed and not altered and not added:
            return "contraction"
        elif coherence_impact > 0.1:
            return "integration"
        elif coherence_impact < -0.1:
            return "fragmentation"
        else:
            return "peripheral_adjustment"
    
    def _calculate_self_model_coherence(self) -> float:
        """Calculate the coherence of the current self model"""
        coherence_components = []
        
        # Component 1: Attribute consistency
        attr_consistency = self._calculate_attribute_consistency()
        coherence_components.append(attr_consistency)
        
        # Component 2: Narrative integration
        nar_integration = self._calculate_narrative_integration()
        if nar_integration is not None:
            coherence_components.append(nar_integration)
        
        # Component 3: Temporal stability
        temp_stability = self._calculate_temporal_stability()
        if temp_stability is not None:
            coherence_components.append(temp_stability)
        
        # Return average of components
        return sum(coherence_components) / len(coherence_components) if coherence_components else 0.5
    
    def _calculate_attribute_consistency(self) -> float:
        """Calculate the internal consistency of attributes"""
        all_attributes = {**self.current_model.core_attributes, **self.current_model.peripheral_attributes}
        if len(all_attributes) < 2:
            return 1.0  # Perfect consistency if only one or zero attributes
        
        # Get embeddings for all attributes
        embeddings = {}
        for attr_id, attr in all_attributes.items():
            # Create or retrieve cached embedding
            if attr_id in self.attribute_embeddings:
                embeddings[attr_id] = self.attribute_embeddings[attr_id]
            else:
                # Create a simple embedding from attribute details
                attr_key = f"{attr_id}:{attr.value}"
                np.random.seed(hash(attr_key) % 2**32)
                embedding = np.random.randn(128)  # 128-dimensional embedding
                embeddings[attr_id] = embedding / np.linalg.norm(embedding)  # Normalize
                self.attribute_embeddings[attr_id] = embeddings[attr_id]
        
        # Calculate average pairwise cosine similarity
        total_sim = 0.0
        count = 0
        
        attr_ids = list(all_attributes.keys())
        for i in range(len(attr_ids)):
            for j in range(i+1, len(attr_ids)):
                id1 = attr_ids[i]
                id2 = attr_ids[j]
                
                # Calculate cosine similarity
                sim = np.dot(embeddings[id1], embeddings[id2])
                
                # Weight by attribute importance
                weight = (all_attributes[id1].importance + all_attributes[id2].importance) / 2
                total_sim += sim * weight
                count += weight
        
        # Return normalized consistency score
        return (total_sim / count + 1) / 2 if count > 0 else 1.0
    
    def _calculate_narrative_integration(self) -> Optional[float]:
        """Calculate how well attributes integrate with narrative themes"""
        if not self.current_model.narrative_themes:
            return None
        
        all_attributes = {**self.current_model.core_attributes, **self.current_model.peripheral_attributes}
        if not all_attributes:
            return None
        
        # Calculate integration for each attribute
        integration_scores = []
        
        for attr_id, attr in all_attributes.items():
            # Check how well attribute aligns with themes
            theme_alignments = []
            
            for theme in self.current_model.narrative_themes:
                # Calculate string similarity with theme
                attr_str = f"{attr_id} {attr.value}"
                alignment = self._string_similarity(attr_str, theme)
                theme_alignments.append(alignment)
            
            # Use max alignment with any theme
            best_alignment = max(theme_alignments) if theme_alignments else 0.0
            
            # Weight by importance
            integration_scores.append(best_alignment * attr.importance)
        
        # Return weighted average
        total_importance = sum(attr.importance for attr in all_attributes.values())
        if total_importance > 0:
            return sum(integration_scores) / total_importance
        
        return sum(integration_scores) / len(integration_scores) if integration_scores else 0.5
    
    def _calculate_temporal_stability(self) -> Optional[float]:
        """Calculate stability of self-model over time"""
        if len(self.identity_snapshots) < 2:
            return None
        
        # Get recent snapshots (last 3)
        recent_snapshots = self.identity_snapshots[-3:]
        if len(recent_snapshots) < 2:
            recent_snapshots = self.identity_snapshots[-2:]
        
        # Calculate stability as inverse of average drift between snapshots
        total_drift = 0.0
        count = 0
        
        for i in range(len(recent_snapshots) - 1):
            drift = self._calculate_drift_magnitude(
                recent_snapshots[i].self_model,
                recent_snapshots[i+1].self_model
            )
            total_drift += drift
            count += 1
        
        # Convert drift to stability (0 drift = 1.0 stability, high drift = low stability)
        avg_drift = total_drift / count if count > 0 else 0.0
        stability = max(0.0, 1.0 - (avg_drift / max(self.drift_threshold, 0.01)))
        
        return stability
    
    def process_identity_event(self, event: IdentityEvent) -> SelfConceptDrift:
        """
        Process a significant identity-shaping event.
        
        Args:
            event: The identity event to process
            
        Returns:
            Drift information resulting from the event
        """
        # Record the event
        self.identity_events.append(event)
        
        # Initialize attribute updates
        updates = {}
        
        # Process affected attributes
        for attr_id, impact in event.attribute_impacts.items():
            # Check if attribute exists
            is_core = attr_id in self.current_model.core_attributes
            existing_attr = None
            
            if is_core:
                existing_attr = self.current_model.core_attributes.get(attr_id)
            else:
                existing_attr = self.current_model.peripheral_attributes.get(attr_id)
            
            if existing_attr:
                # Calculate new value based on impact
                new_value = self._apply_attribute_impact(existing_attr.value, impact)
                
                # Create updated attribute
                updated_attr = SelfAttribute(
                    name=existing_attr.name,
                    value=new_value,
                    confidence=min(1.0, existing_attr.confidence + impact.confidence_change),
                    importance=min(1.0, existing_attr.importance + impact.importance_change),
                    is_core=is_core,
                    source=existing_attr.source
                )
                
                # Add to updates
                updates[attr_id] = updated_attr
            elif impact.create_if_missing:
                # Create new attribute
                new_attr = SelfAttribute(
                    name=impact.name or attr_id,
                    value=impact.value,
                    confidence=max(0.0, min(1.0, 0.5 + impact.confidence_change)),
                    importance=max(0.0, min(1.0, 0.5 + impact.importance_change)),
                    is_core=impact.is_core or False,
                    source=event.source
                )
                
                # Add to updates
                updates[attr_id] = new_attr
        
        # Update narrative themes if provided
        if event.narrative_themes:
            current_themes = set(self.current_model.narrative_themes)
            for theme in event.narrative_themes:
                if theme.startswith("-"):  # Remove theme
                    clean_theme = theme[1:].strip()
                    if clean_theme in current_themes:
                        current_themes.remove(clean_theme)
                else:  # Add theme
                    current_themes.add(theme)
            
            self.current_model.narrative_themes = list(current_themes)
        
        # Apply all updates
        drift = self.update_self_model(updates)
        
        # Add event reference to drift record
        drift.source_event_id = event.id
        
        return drift
    
    def _apply_attribute_impact(self, current_value, impact):
        """Apply impact to attribute value based on type"""
        if impact.operation == "set":
            return impact.value
        elif impact.operation == "increment" and isinstance(current_value, (int, float)):
            return current_value + impact.value
        elif impact.operation == "decrement" and isinstance(current_value, (int, float)):
            return current_value - impact.value
        elif impact.operation == "toggle" and isinstance(current_value, bool):
            return not current_value
        elif impact.operation == "append" and isinstance(current_value, str):
            return current_value + str(impact.value)
        elif impact.operation == "remove" and isinstance(current_value, str):
            return current_value.replace(str(impact.value), "")
        else:
            # Default to replacement
            return impact.value
    
    def integrate_narrative_update(
        self, narrative: NarrativeElement, salience: float = 0.5
    ) -> Optional[SelfConceptDrift]:
        """
        Integrate a narrative element into the self-model.
        
        Args:
            narrative: The narrative element to integrate
            salience: How salient/important this narrative is (0-1)
            
        Returns:
            Drift information if any drift occurred, None otherwise
        """
        # Extract themes from narrative
        themes = set(narrative.themes if hasattr(narrative, 'themes') else [])
        
        # Extract key concepts
        concepts = set(narrative.key_concepts if hasattr(narrative, 'key_concepts') else [])
        
        # Skip if no themes or concepts
        if not themes and not concepts:
            return None
        
        # Create event from narrative
        event = IdentityEvent(
            id=f"narrative_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            description=f"Integration of narrative: {narrative.title if hasattr(narrative, 'title') else 'Untitled'}",
            source="narrative",
            significance=salience,
            attribute_impacts={},
            narrative_themes=list(themes)
        )
        
        # Generate attribute impacts from concepts and emotional tone
        emotional_tone = getattr(narrative, 'emotional_tone', {})
        
        for concept in concepts:
            # Generate attribute ID from concept
            attr_id = f"concept_{concept.lower().replace(' ', '_')}"
            
            # Check if concept is relevant to existing attributes
            matched_attr_id = self._find_relevant_attribute(concept)
            
            if matched_attr_id:
                # Impact existing attribute
                impact_value = 0.1 * salience  # Small reinforcement
                
                # Use emotional tone to determine direction
                pos_emotions = sum(emotional_tone.get(e, 0) for e in ["joy", "trust", "anticipation"])
                neg_emotions = sum(emotional_tone.get(e, 0) for e in ["anger", "fear", "disgust", "sadness"])
                
                # Determine if this is a positive or negative reinforcement
                if pos_emotions > neg_emotions:
                    confidence_change = 0.05 * salience
                else:
                    confidence_change = -0.03 * salience
                
                event.attribute_impacts[matched_attr_id] = AttributeImpact(
                    operation="increment",
                    value=impact_value,
                    confidence_change=confidence_change,
                    importance_change=0.02 * salience
                )
            elif salience > 0.7:
                # Create new attribute for highly salient concepts
                event.attribute_impacts[attr_id] = AttributeImpact(
                    name=concept,
                    operation="set",
                    value=concept,
                    confidence_change=0.3 * salience,
                    importance_change=0.2 * salience,
                    create_if_missing=True,
                    is_core=False
                )
        
        # Process the event
        if event.attribute_impacts or event.narrative_themes:
            return self.process_identity_event(event)
        
        return None
    
    def _find_relevant_attribute(self, concept: str) -> Optional[str]:
        """Find an existing attribute relevant to a concept"""
        all_attributes = {**self.current_model.core_attributes, **self.current_model.peripheral_attributes}
        
        best_match = None
        best_score = 0.3  # Minimum similarity threshold
        
        for attr_id, attr in all_attributes.items():
            # Check name similarity
            name_sim = self._string_similarity(attr.name.lower(), concept.lower())
            
            # Check value similarity if string
            value_sim = 0
            if isinstance(attr.value, str):
                value_sim = self._string_similarity(attr.value.lower(), concept.lower())
            
            # Take the best match
            similarity = max(name_sim, value_sim)
            if similarity > best_score:
                best_score = similarity
                best_match = attr_id
        
        return best_match
    
    def integrate_value_changes(self, values: List[Value]) -> Optional[SelfConceptDrift]:
        """
        Integrate value changes into the self-model.
        
        Args:
            values: List of values that changed
            
        Returns:
            Drift information if any drift occurred, None otherwise
        """
        if not values:
            return None
        
        # Create event from value changes
        event = IdentityEvent(
            id=f"values_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            description=f"Integration of value changes",
            source="values",
            significance=0.6,  # Values are important for identity
            attribute_impacts={},
            narrative_themes=[]
        )
        
        # Generate impacts from value changes
        for value in values:
            # Check priority to determine significance
            significance = min(1.0, value.priority / 10 if hasattr(value, 'priority') else 0.5)
            
            # Generate attribute ID from value
            attr_id = f"value_{value.name.lower().replace(' ', '_')}"
            
            # Check if already exists as attribute
            is_existing = (attr_id in self.current_model.core_attributes or 
                          attr_id in self.current_model.peripheral_attributes)
            
            if is_existing:
                # Update existing attribute
                event.attribute_impacts[attr_id] = AttributeImpact(
                    operation="set",
                    value=value.name,
                    confidence_change=0.05 * significance,
                    importance_change=0.03 * significance
                )
            else:
                # Create new attribute for high-priority values
                if significance > 0.6:
                    event.attribute_impacts[attr_id] = AttributeImpact(
                        name=value.name,
                        operation="set",
                        value=value.name,
                        confidence_change=0.4 * significance,
                        importance_change=0.3 * significance,
                        create_if_missing=True,
                        is_core=significance > 0.8  # High priority values become core attributes
                    )
            
            # Add related concepts as potential narrative themes
            if hasattr(value, 'related_concepts') and value.related_concepts:
                for concept in value.related_concepts:
                    if len(concept) > 3 and significance > 0.7:
                        event.narrative_themes.append(concept)
        
        # Process the event
        if event.attribute_impacts or event.narrative_themes:
            return self.process_identity_event(event)
        
        return None
    
    def analyze_drift_trajectory(self, time_window_days: int = 30) -> DriftTrajectory:
        """
        Analyze the trajectory of self-model drift over time.
        
        Args:
            time_window_days: Number of days to look back for analysis
            
        Returns:
            Analysis of drift trajectory
        """
        now = datetime.now()
        cutoff_date = now - timedelta(days=time_window_days)
        
        # Filter drifts and snapshots within time window
        recent_drifts = [
            drift for drift in self.detected_drifts 
            if drift.timestamp >= cutoff_date
        ]
        
        recent_snapshots = [
            snapshot for snapshot in self.identity_snapshots 
            if snapshot.timestamp >= cutoff_date
        ]
        
        # Calculate overall statistics
        drift_magnitudes = [drift.drift_magnitude for drift in recent_drifts]
        avg_magnitude = sum(drift_magnitudes) / len(drift_magnitudes) if drift_magnitudes else 0
        
        drift_types = Counter([drift.drift_type for drift in recent_drifts])
        most_common_type = drift_types.most_common(1)[0][0] if drift_types else "none"
        
        # Calculate coherence trend
        coherence_values = [(snapshot.timestamp, snapshot.coherence_score) for snapshot in recent_snapshots]
        coherence_trend = self._calculate_trend([score for _, score in coherence_values]) if coherence_values else 0
        
        # Calculate attribute stability
        attribute_stability = {}
        core_stability = []
        peripheral_stability = []
        
        for attr_id, history in self.attribute_history.items():
            # Filter to recent history
            recent_history = [h for h in history if h.timestamp >= cutoff_date]
            if len(recent_history) > 1:
                # Calculate stability as inverse of variance
                values = [h.value for h in recent_history if not isinstance(h.value, (str, bool))]
                if values and all(isinstance(v, (int, float)) for v in values):
                    variance = np.var(values) if len(values) > 1 else 0
                    max_val = max(abs(v) for v in values) if values else 1
                    normalized_variance = variance / (max_val ** 2) if max_val > 0 else 0
                    stability = max(0, 1 - min(1, normalized_variance * 10))
                else:
                    # For strings or booleans, check for changes
                    changes = sum(1 for i in range(len(recent_history)-1) 
                                 if recent_history[i].value != recent_history[i+1].value)
                    stability = max(0, 1 - changes / (len(recent_history) - 1)) if len(recent_history) > 1 else 1
                
                attribute_stability[attr_id] = stability
                
                # Track core vs peripheral stability
                if attr_id in self.current_model.core_attributes:
                    core_stability.append(stability)
                else:
                    peripheral_stability.append(stability)
        
        # Check trajectory type
        trajectory_type = self._determine_trajectory_type(
            recent_drifts, 
            coherence_trend,
            core_stability,
            peripheral_stability
        )
        
        # Return comprehensive trajectory analysis
        return DriftTrajectory(
            analysis_timestamp=now,
            time_window_days=time_window_days,
            drift_count=len(recent_drifts),
            average_drift_magnitude=avg_magnitude,
            predominant_drift_type=most_common_type,
            coherence_trend=coherence_trend,
            attribute_stability=attribute_stability,
            core_attribute_stability=sum(core_stability) / len(core_stability) if core_stability else 1.0,
            peripheral_attribute_stability=sum(peripheral_stability) / len(peripheral_stability) if peripheral_stability else 1.0,
            trajectory_type=trajectory_type,
            significant_events=[event for event in self.identity_events if event.timestamp >= cutoff_date],
            current_coherence=self._calculate_self_model_coherence()
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate the trend slope of a series of values"""
        if not values or len(values) < 2:
            return 0
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Normalize slope to a -1 to 1 scale based on the mean value
        normalized_slope = slope / abs(mean_y) if mean_y != 0 else slope
        
        # Clamp to [-1, 1]
        return max(-1, min(1, normalized_slope * 10))
    
    def _determine_trajectory_type(
        self, 
        drifts: List[SelfConceptDrift],
        coherence_trend: float,
        core_stability: List[float],
        peripheral_stability: List[float]
    ) -> str:
        """Determine the type of identity trajectory"""
        if not drifts:
            return "stable"
        
        # Calculate average core and peripheral stability
        avg_core_stability = sum(core_stability) / len(core_stability) if core_stability else 1.0
        avg_peripheral_stability = sum(peripheral_stability) / len(peripheral_stability) if peripheral_stability else 1.0
        
        # Check for different trajectory patterns
        recent_drift_types = Counter([d.drift_type for d in drifts[-min(3, len(drifts)):]])
        
        if coherence_trend > 0.3 and avg_core_stability > 0.7:
            return "integrative_development"
        elif coherence_trend < -0.3 and "core_disruption" in recent_drift_types:
            return "identity_crisis"
        elif coherence_trend < -0.2 and avg_peripheral_stability < 0.5:
            return "fragmentation"
        elif "expansion" in recent_drift_types and recent_drift_types["expansion"] >= 2:
            return "exploration"
        elif avg_core_stability > 0.9 and avg_peripheral_stability < 0.6:
            return "core_consolidation"
        elif avg_core_stability > 0.8 and avg_peripheral_stability > 0.7:
            return "stable_identity"
        elif len(drifts) > 5 and sum(d.drift_magnitude for d in drifts) / len(drifts) > self.drift_threshold:
            return "rapid_evolution"
        else:
            return "gradual_adjustment"
    
    def check_coherence_and_stabilize(self) -> Optional[SelfConceptDrift]:
        """
        Check if the self-model is becoming incoherent and apply stabilization if needed.
        
        Returns:
            Drift information if stabilization was applied, None otherwise
        """
        current_coherence = self._calculate_self_model_coherence()
        
        # Check if coherence is below threshold
        if current_coherence < self.coherence_threshold:
            # Apply stabilization
            return self._apply_coherence_stabilization(current_coherence)
        
        return None
    
    def _apply_coherence_stabilization(self, current_coherence: float) -> SelfConceptDrift:
        """Apply stabilization to improve coherence"""
        # Check recent drifts to identify potential causes
        recent_drifts = self.detected_drifts[-min(5, len(self.detected_drifts)):]
        
        # Stabilization plan
        updates = {}
        
        # Strategy 1: Reinforce core attributes
        for attr_id, attr in self.current_model.core_attributes.items():
            # Boost confidence in core attributes
            updated_attr = attr.copy()
            updated_attr.confidence = min(1.0, attr.confidence + 0.1)
            updates[attr_id] = updated_attr
        
        # Strategy 2: Prune low-confidence peripheral attributes
        for attr_id, attr in self.current_model.peripheral_attributes.items():
            if attr.confidence < 0.3 and attr.importance < 0.4:
                # Mark for removal
                updates[attr_id] = SelfAttribute(
                    name=attr.name,
                    value=None,  # None value signals removal
                    confidence=0,
                    importance=0,
                    is_core=False,
                    source=attr.source
                )
        
        # Strategy 3: Align with narrative themes
        if self.current_model.narrative_themes:
            # Find attributes that align with themes
            for attr_id, attr in self.current_model.peripheral_attributes.items():
                # Skip if already marked for removal
                if attr_id in updates and updates[attr_id].value is None:
                    continue
                
                # Check alignment with themes
                attr_str = f"{attr.name} {attr.value}"
                theme_alignments = [
                    self._string_similarity(attr_str, theme)
                    for theme in self.current_model.narrative_themes
                ]
                
                best_alignment = max(theme_alignments) if theme_alignments else 0
                
                if best_alignment > 0.5:
                    # Boost well-aligned attributes
                    updated_attr = attr.copy()
                    updated_attr.confidence = min(1.0, attr.confidence + 0.1)
                    updated_attr.importance = min(1.0, attr.importance + 0.05)
                    updates[attr_id] = updated_attr
        
        # Apply updates
        description = "Self-model coherence stabilization"
        drift = self.update_self_model(updates)
        drift.description = description
        
        return drift
    
    def generate_self_description(self) -> str:
        """
        Generate a textual description of the current self-model.
        
        Returns:
            Text description of the self-model
        """
        if not self.current_model.core_attributes and not self.current_model.peripheral_attributes:
            return "No defined self-model available."
        
        # Start with core attributes
        core_attributes = sorted(
            self.current_model.core_attributes.items(),
            key=lambda x: x[1].importance,
            reverse=True
        )
        
        core_desc = []
        for _, attr in core_attributes:
            confidence_marker = ""
            if attr.confidence < 0.5:
                confidence_marker = " (uncertain)"
            elif attr.confidence > 0.9:
                confidence_marker = " (certain)"
                
            core_desc.append(f"{attr.name}: {attr.value}{confidence_marker}")
        
        # Add high-importance peripheral attributes
        peripheral_attributes = sorted(
            [
                (attr_id, attr) for attr_id, attr in self.current_model.peripheral_attributes.items()
                if attr.importance > 0.6
            ],
            key=lambda x: x[1].importance,
            reverse=True
        )
        
        peripheral_desc = []
        for _, attr in peripheral_attributes:
            peripheral_desc.append(f"{attr.name}: {attr.value}")
        
        # Compile the description
        description = "Core identity attributes:\n- " + "\n- ".join(core_desc)
        
        if peripheral_desc:
            description += "\n\nSignificant secondary attributes:\n- " + "\n- ".join(peripheral_desc)
        
        if self.current_model.narrative_themes:
            description += "\n\nKey narrative themes:\n- " + "\n- ".join(self.current_model.narrative_themes)
        
        # Add trajectory info if available
        if self.detected_drifts:
            trajectory = self.analyze_drift_trajectory()
            description += f"\n\nIdentity trajectory: {trajectory.trajectory_type.replace('_', ' ').title()}"
        
        return description
    
    def get_self_model_metrics(self) -> SelfModelMetrics:
        """
        Get metrics about the current self-model.
        
        Returns:
            Metrics about the self-model
        """
        # Calculate coherence
        coherence = self._calculate_self_model_coherence()
        
        # Calculate stability from recent drift
        recent_drift_magnitude = 0.0
        if self.detected_drifts:
            recent_drifts = self.detected_drifts[-min(3, len(self.detected_drifts)):]
            recent_drift_magnitude = sum(d.drift_magnitude for d in recent_drifts) / len(recent_drifts)
        
        stability = max(0.0, 1.0 - recent_drift_magnitude / max(self.drift_threshold, 0.01))
        
        # Calculate complexity
        attribute_count = len(self.current_model.core_attributes) + len(self.current_model.peripheral_attributes)
        theme_count = len(self.current_model.narrative_themes)
        complexity = min(1.0, ((attribute_count / 20) * 0.7 + (theme_count / 10) * 0.3))
        
        # Calculate confidence
        core_confidences = [attr.confidence for attr in self.current_model.core_attributes.values()]
        avg_core_confidence = sum(core_confidences) / len(core_confidences) if core_confidences else 0.5
        
        # Determine self-concept clarity
        clarity = (coherence * 0.5 + avg_core_confidence * 0.3 + stability * 0.2)
        
        # Get trajectory
        trajectory_type = "unknown"
        if len(self.identity_snapshots) > 2:
            trajectory = self.analyze_drift_trajectory()
            trajectory_type = trajectory.trajectory_type
        
        return SelfModelMetrics(
            timestamp=datetime.now(),
            coherence=coherence,
            stability=stability,
            complexity=complexity,
            core_attribute_count=len(self.current_model.core_attributes),
            peripheral_attribute_count=len(self.current_model.peripheral_attributes),
            narrative_theme_count=theme_count,
            average_core_confidence=avg_core_confidence,
            self_concept_clarity=clarity,
            trajectory=trajectory_type,
            recent_drift_magnitude=recent_drift_magnitude
        )
    
    def get_attribute_evolution(self, attr_id: str) -> List[AttributeEvolution]:
        """
        Get the evolution history of a specific attribute.
        
        Args:
            attr_id: ID of the attribute to get history for
            
        Returns:
            List of attribute evolution records
        """
        return self.attribute_history.get(attr_id, [])
    
    def export_self_model(self) -> Dict:
        """
        Export the current self-model and its state.
        
        Returns:
            Dictionary representation of the self-model
        """
        def serialize_datetime(dt):
            return dt.isoformat() if dt else None
        
        def serialize_attribute(attr):
            return {
                "name": attr.name,
                "value": attr.value,
                "confidence": attr.confidence,
                "importance": attr.importance,
                "is_core": attr.is_core,
                "source": attr.source
            }
        
        model_data = {
            "id": self.current_model.id,
            "core_attributes": {
                attr_id: serialize_attribute(attr)
                for attr_id, attr in self.current_model.core_attributes.items()
            },
            "peripheral_attributes": {
                attr_id: serialize_attribute(attr)
                for attr_id, attr in self.current_model.peripheral_attributes.items()
            },
            "narrative_themes": self.current_model.narrative_themes,
            "self_description": self.current_model.self_description,
            "creation_date": serialize_datetime(self.current_model.creation_date),
            "last_updated": serialize_datetime(self.current_model.last_updated)
        }
        
        # Add metrics
        metrics = self.get_self_model_metrics()
        model_data["metrics"] = {
            "coherence": metrics.coherence,
            "stability": metrics.stability,
            "complexity": metrics.complexity,
            "self_concept_clarity": metrics.self_concept_clarity,
            "trajectory": metrics.trajectory
        }
        
        return model_data
    
    def import_self_model(self, model_data: Dict) -> bool:
        """
        Import a self-model from a dictionary representation.
        
        Args:
            model_data: Dictionary representation of the self-model
            
        Returns:
            True if import was successful, False otherwise
        """
        try:
            # Parse datetime objects
            def parse_datetime(dt_str):
                return datetime.fromisoformat(dt_str) if dt_str else None
            
            # Parse attributes
            def parse_attribute(attr_data):
                return SelfAttribute(
                    name=attr_data["name"],
                    value=attr_data["value"],
                    confidence=attr_data["confidence"],
                    importance=attr_data["importance"],
                    is_core=attr_data["is_core"],
                    source=attr_data["source"]
                )
            
            # Create new model
            new_model = SelfModel(
                id=model_data["id"],
                core_attributes={
                    attr_id: parse_attribute(attr_data)
                    for attr_id, attr_data in model_data["core_attributes"].items()
                },
                peripheral_attributes={
                    attr_id: parse_attribute(attr_data)
                    for attr_id, attr_data in model_data["peripheral_attributes"].items()
                },
                narrative_themes=model_data["narrative_themes"],
                self_description=model_data["self_description"],
                creation_date=parse_datetime(model_data["creation_date"]),
                last_updated=parse_datetime(model_data["last_updated"])
            )
            
            # Replace current model
            self.current_model = new_model
            
            # Take snapshot
            self._take_identity_snapshot()
            
            # Initialize attribute history
            self._initialize_attribute_history()
            
            return True
        except Exception as e:
            print(f"Error importing self-model: {e}")
            return False


class AttributeImpact:
    """Represents the impact of an event on an attribute"""
    
    def __init__(
        self,
        operation: str = "set",
        value: any = None,
        confidence_change: float = 0.0,
        importance_change: float = 0.0,
        create_if_missing: bool = False,
        is_core: bool = False,
        name: Optional[str] = None
    ):
        self.operation = operation  # 'set', 'increment', 'decrement', 'toggle', 'append', 'remove'
        self.value = value
        self.confidence_change = confidence_change
        self.importance_change = importance_change
        self.create_if_missing = create_if_missing
        self.is_core = is_core
        self.name = name  # Used when creating new attributes

    def maintain_identity_balance(self) -> Optional[SelfConceptDrift]:
    """
    Maintain balance between stability and adaptability in the self-model.
    
    Returns:
        Drift information if balance adjustments were made, None otherwise
    """
    # Get current metrics
    metrics = self.get_self_model_metrics()
    
    # Skip if recent stability and coherence are in good range
    if 0.4 <= metrics.stability <= 0.9 and metrics.coherence >= self.coherence_threshold:
        return None
    
    # Create a drift event for balance adjustments
    description = "Identity balance maintenance"
    updates = {}
    
    # Case 1: Too much instability
    if metrics.stability < 0.4:
        # Strengthen core attributes to increase stability
        for attr_id, attr in self.current_model.core_attributes.items():
            updated_attr = attr.copy()
            updated_attr.confidence = min(1.0, attr.confidence + 0.08)
            updates[attr_id] = updated_attr
        
        # Reduce peripheral change by increasing confidence of stable peripherals
        for attr_id in self.attribute_history:
            if attr_id in self.current_model.peripheral_attributes:
                attr = self.current_model.peripheral_attributes[attr_id]
                history = self.attribute_history[attr_id]
                
                if len(history) > 2:
                    # Check if value has been stable
                    recent_values = [h.value for h in history[-3:]]
                    if len(set(recent_values)) == 1:  # Same value for last 3 records
                        updated_attr = attr.copy()
                        updated_attr.confidence = min(1.0, attr.confidence + 0.1)
                        updates[attr_id] = updated_attr
    
    # Case 2: Too much rigidity (too stable)
    elif metrics.stability > 0.9 and len(self.detected_drifts) > 3:
        # Identify least important peripheral attributes for flexibility
        sorted_peripherals = sorted(
            self.current_model.peripheral_attributes.items(),
            key=lambda x: x[1].importance
        )
        
        # Reduce confidence in less important peripherals to allow change
        for attr_id, attr in sorted_peripherals[:max(2, len(sorted_peripherals)//4)]:
            updated_attr = attr.copy()
            updated_attr.confidence = max(0.2, attr.confidence - 0.1)
            updates[attr_id] = updated_attr
    
    # Case 3: Low coherence
    if metrics.coherence < self.coherence_threshold:
        # Apply strategy to remove contradicting peripherals
        conflict_scores = self._identify_attribute_conflicts()
        
        # Remove or reduce confidence in most conflicting attributes
        for attr_id, conflict_score in conflict_scores.items():
            if conflict_score > 0.7 and attr_id in self.current_model.peripheral_attributes:
                attr = self.current_model.peripheral_attributes[attr_id]
                
                if attr.importance < 0.4:
                    # Mark for removal if not important
                    updates[attr_id] = SelfAttribute(
                        name=attr.name,
                        value=None,  # None value signals removal
                        confidence=0,
                        importance=0,
                        is_core=False,
                        source=attr.source
                    )
                else:
                    # Reduce confidence if important
                    updated_attr = attr.copy()
                    updated_attr.confidence = max(0.2, attr.confidence - 0.15)
                    updates[attr_id] = updated_attr
    
    # Apply updates if any
    if updates:
        drift = self.update_self_model(updates)
        drift.description = description
        return drift
    
    return None

def _identify_attribute_conflicts(self) -> Dict[str, float]:
    """Identify attributes that conflict with others"""
    all_attributes = {**self.current_model.core_attributes, **self.current_model.peripheral_attributes}
    conflict_scores = {}
    
    # Get attribute embeddings
    embeddings = {}
    for attr_id, attr in all_attributes.items():
        if attr_id in self.attribute_embeddings:
            embeddings[attr_id] = self.attribute_embeddings[attr_id]
        else:
            # Create embedding if not cached
            attr_key = f"{attr_id}:{attr.value}"
            np.random.seed(hash(attr_key) % 2**32)
            embedding = np.random.randn(128)
            embeddings[attr_id] = embedding / np.linalg.norm(embedding)
            self.attribute_embeddings[attr_id] = embeddings[attr_id]
    
    # Calculate conflict scores for each attribute
    for attr_id, attr in all_attributes.items():
        conflicts = []
        
        for other_id, other_attr in all_attributes.items():
            if attr_id == other_id:
                continue
            
            # Calculate cosine similarity
            sim = np.dot(embeddings[attr_id], embeddings[other_id])
            
            # Negative similarity indicates potential conflict
            if sim < -0.2:
                # Weight by importance of both attributes
                conflict_weight = other_attr.importance * abs(sim)
                conflicts.append(conflict_weight)
        
        # Calculate overall conflict score
        if conflicts:
            # Use top 3 conflicts or all if fewer
            top_conflicts = sorted(conflicts, reverse=True)[:min(3, len(conflicts))]
            conflict_scores[attr_id] = sum(top_conflicts) / len(top_conflicts)
        else:
            conflict_scores[attr_id] = 0.0
    
    return conflict_scores

def simulate_drift(self, time_steps: int = 10, external_influence: float = 0.3) -> List[SelfConceptDrift]:
    """
    Simulate natural drift of the self-model over time.
    
    Args:
        time_steps: Number of time steps to simulate
        external_influence: Magnitude of external influence (0-1)
        
    Returns:
        List of drift events that occurred during simulation
    """
    drift_events = []
    
    for step in range(time_steps):
        # Determine if drift occurs at this step
        drift_probability = 0.3 + (external_influence * 0.4)
        if random.random() < drift_probability:
            # Generate simulated drift
            drift = self._generate_simulated_drift(external_influence)
            if drift:
                drift_events.append(drift)
        
        # Periodically check and maintain coherence
        if step % 3 == 0:
            drift = self.check_coherence_and_stabilize()
            if drift and drift.drift_magnitude > 0:
                drift_events.append(drift)
    
    return drift_events

def _generate_simulated_drift(self, external_influence: float) -> Optional[SelfConceptDrift]:
    """Generate a simulated drift event based on current state"""
    all_attributes = {**self.current_model.core_attributes, **self.current_model.peripheral_attributes}
    if not all_attributes:
        return None
    
    # Determine drift type
    drift_type_probs = {
        "peripheral_adjustment": 0.5,
        "expansion": 0.2,
        "contraction": 0.1,
        "core_evolution": 0.2 * external_influence,
        "integration": 0.1
    }
    
    drift_types = list(drift_type_probs.keys())
    drift_probs = [drift_type_probs[t] for t in drift_types]
    # Normalize probabilities
    total_prob = sum(drift_probs)
    drift_probs = [p/total_prob for p in drift_probs]
    
    selected_drift = np.random.choice(drift_types, p=drift_probs)
    updates = {}
    
    # Generate updates based on drift type
    if selected_drift == "peripheral_adjustment":
        # Modify a random peripheral attribute
        peripheral_ids = list(self.current_model.peripheral_attributes.keys())
        if peripheral_ids:
            attr_id = random.choice(peripheral_ids)
            attr = self.current_model.peripheral_attributes[attr_id]
            
            # Modify attribute
            updated_attr = attr.copy()
            
            # Change value based on type
            if isinstance(attr.value, (int, float)):
                change = attr.value * (random.random() * 0.2 - 0.1)  # -10% to +10%
                updated_attr.value = attr.value + change
            elif isinstance(attr.value, bool):
                if random.random() < 0.3:  # 30% chance to toggle
                    updated_attr.value = not attr.value
            elif isinstance(attr.value, str):
                # No change to string value in simulation
                pass
            
            # Adjust confidence
            confidence_change = random.random() * 0.2 - 0.1  # -0.1 to +0.1
            updated_attr.confidence = max(0.1, min(1.0, attr.confidence + confidence_change))
            
            updates[attr_id] = updated_attr
    
    elif selected_drift == "expansion":
        # Add a new peripheral attribute
        new_id = f"simulated_attr_{int(datetime.now().timestamp())}"
        
        # Generate random value (simplified)
        value_type = random.choice(["numeric", "boolean", "string"])
        if value_type == "numeric":
            value = random.random() * 10
        elif value_type == "boolean":
            value = random.choice([True, False])
        else:
            value = "Simulated attribute value"
        
        new_attr = SelfAttribute(
            name=f"Simulated Attribute {new_id[-4:]}",
            value=value,
            confidence=0.3 + random.random() * 0.4,  # 0.3 to 0.7
            importance=0.2 + random.random() * 0.3,  # 0.2 to 0.5
            is_core=False,
            source="simulation"
        )
        
        updates[new_id] = new_attr
    
    elif selected_drift == "contraction":
        # Remove a peripheral attribute
        peripheral_ids = list(self.current_model.peripheral_attributes.keys())
        if peripheral_ids:
            attr_id = random.choice(peripheral_ids)
            updates[attr_id] = SelfAttribute(
                name=self.current_model.peripheral_attributes[attr_id].name,
                value=None,  # None signals removal
                confidence=0,
                importance=0,
                is_core=False,
                source="simulation"
            )
    
    elif selected_drift == "core_evolution" and external_influence > 0.5:
        # Modify a core attribute (only with high external influence)
        core_ids = list(self.current_model.core_attributes.keys())
        if core_ids:
            attr_id = random.choice(core_ids)
            attr = self.current_model.core_attributes[attr_id]
            
            # Make a small change to the core attribute
            updated_attr = attr.copy()
            
            # Change value based on type
            if isinstance(attr.value, (int, float)):
                change = attr.value * (random.random() * 0.1 - 0.05)  # -5% to +5%
                updated_attr.value = attr.value + change
            elif isinstance(attr.value, bool):
                if random.random() < 0.1:  # 10% chance to toggle
                    updated_attr.value = not attr.value
            # No change to string value
            
            # Small confidence adjustment
            confidence_change = random.random() * 0.1 - 0.05  # -0.05 to +0.05
            updated_attr.confidence = max(0.5, min(1.0, attr.confidence + confidence_change))
            
            updates[attr_id] = updated_attr
    
    elif selected_drift == "integration":
        # Increase coherence by strengthening related attributes
        attr_pairs = []
        all_attr_ids = list(all_attributes.keys())
        
        # Find related attribute pairs
        if len(all_attr_ids) >= 2:
            for i in range(min(3, len(all_attr_ids))):
                id1 = random.choice(all_attr_ids)
                remaining = [id2 for id2 in all_attr_ids if id2 != id1]
                if remaining:
                    id2 = random.choice(remaining)
                    attr_pairs.append((id1, id2))
        
        # Strengthen related pairs
        for id1, id2 in attr_pairs:
            # Increase confidence in both
            for attr_id in [id1, id2]:
                attr = all_attributes[attr_id]
                updated_attr = attr.copy()
                updated_attr.confidence = min(1.0, attr.confidence + 0.05)
                updated_attr.importance = min(1.0, attr.importance + 0.03)
                updates[attr_id] = updated_attr
    
    # Apply updates
    if updates:
        return self.update_self_model(updates)
    
    return None

def record_significant_experience(
    self, 
    description: str,
    impact_level: float,
    affected_attributes: Dict[str, AttributeImpact],
    narrative_impact: List[str] = None
) -> SelfConceptDrift:
    """
    Record a significant experience that impacts the self-model.
    
    Args:
        description: Description of the experience
        impact_level: How significant the experience is (0-1)
        affected_attributes: Dict mapping attribute IDs to their impacts
        narrative_impact: List of narrative themes affected
        
    Returns:
        Resulting drift information
    """
    # Create an identity event
    event = IdentityEvent(
        id=f"experience_{int(datetime.now().timestamp())}",
        timestamp=datetime.now(),
        description=description,
        source="experience",
        significance=impact_level,
        attribute_impacts=affected_attributes,
        narrative_themes=narrative_impact or []
    )
    
    # Process the event
    return self.process_identity_event(event)

def merge_self_model(self, other_model: SelfModel, merge_weight: float = 0.5) -> SelfConceptDrift:
    """
    Merge another self-model into the current one.
    
    Args:
        other_model: The other self-model to merge
        merge_weight: Weight of the other model in the merge (0-1)
        
    Returns:
        Drift information for the merge
    """
    if not other_model:
        return SelfConceptDrift(
            drift_magnitude=0.0,
            altered_attributes=[],
            coherence_impact=0.0,
            timestamp=datetime.now(),
            is_significant=False,
            drift_type="none"
        )
    
    # Initialize updates
    updates = {}
    
    # Merge core attributes
    for attr_id, other_attr in other_model.core_attributes.items():
        if attr_id in self.current_model.core_attributes:
            # Both models have this core attribute - merge
            current_attr = self.current_model.core_attributes[attr_id]
            
            # Create merged attribute
            merged_attr = self._merge_attributes(
                current_attr, other_attr, merge_weight
            )
            
            updates[attr_id] = merged_attr
        else:
            # Only other model has this attribute
            # Add as core or peripheral based on confidence * merge_weight
            effective_confidence = other_attr.confidence * merge_weight
            effective_importance = other_attr.importance * merge_weight
            
            new_attr = SelfAttribute(
                name=other_attr.name,
                value=other_attr.value,
                confidence=effective_confidence,
                importance=effective_importance,
                is_core=effective_importance > 0.7,  # Add as core only if still important
                source=f"merged:{other_attr.source}"
            )
            
            updates[attr_id] = new_attr
    
    # Merge peripheral attributes
    for attr_id, other_attr in other_model.peripheral_attributes.items():
        if attr_id in self.current_model.peripheral_attributes:
            # Both models have this peripheral attribute - merge
            current_attr = self.current_model.peripheral_attributes[attr_id]
            
            merged_attr = self._merge_attributes(
                current_attr, other_attr, merge_weight
            )
            
            updates[attr_id] = merged_attr
        elif attr_id not in self.current_model.core_attributes:
            # Only other model has this attribute
            # Add with reduced confidence/importance
            new_attr = SelfAttribute(
                name=other_attr.name,
                value=other_attr.value,
                confidence=other_attr.confidence * merge_weight,
                importance=other_attr.importance * merge_weight,
                is_core=False,  # Always add as peripheral
                source=f"merged:{other_attr.source}"
            )
            
            updates[attr_id] = new_attr
    
    # Merge narrative themes
    current_themes = set(self.current_model.narrative_themes)
    other_themes = set(other_model.narrative_themes)
    
    # Add new themes from other model
    new_themes = other_themes - current_themes
    if new_themes:
        self.current_model.narrative_themes.extend(list(new_themes))
    
    # Apply updates
    drift = self.update_self_model(updates)
    drift.description = f"Merge with model: {other_model.id}"
    
    return drift

def _merge_attributes(
    self, attr1: SelfAttribute, attr2: SelfAttribute, weight2: float
) -> SelfAttribute:
    """Merge two attributes with given weights"""
    weight1 = 1.0 - weight2
    
    # Merge based on attribute value type
    if isinstance(attr1.value, (int, float)) and isinstance(attr2.value, (int, float)):
        # Numeric values - weighted average
        merged_value = attr1.value * weight1 + attr2.value * weight2
    elif isinstance(attr1.value, bool) and isinstance(attr2.value, bool):
        # Boolean values - use majority or weighted choice
        if attr1.value == attr2.value:
            merged_value = attr1.value
        else:
            # Weight by confidence and importance
            w1 = weight1 * attr1.confidence * attr1.importance
            w2 = weight2 * attr2.confidence * attr2.importance
            merged_value = attr1.value if w1 > w2 else attr2.value
    else:
        # String or other values - use the one with higher weight*confidence
        w1 = weight1 * attr1.confidence
        w2 = weight2 * attr2.confidence
        merged_value = attr1.value if w1 > w2 else attr2.value
    
    # Merge confidence and importance (weighted average)
    merged_confidence = attr1.confidence * weight1 + attr2.confidence * weight2
    merged_importance = attr1.importance * weight1 + attr2.importance * weight2
    
    # Create merged attribute
    return SelfAttribute(
        name=attr1.name,  # Keep original name
        value=merged_value,
        confidence=merged_confidence,
        importance=merged_importance,
        is_core=attr1.is_core or (attr2.is_core and weight2 > 0.7),
        source=f"merged:{attr1.source}+{attr2.source}"
    )

def reset_peripheral_attributes(self) -> SelfConceptDrift:
    """
    Reset all peripheral attributes while preserving core attributes.
    
    Returns:
        Drift information for the reset
    """
    # Mark all peripheral attributes for removal
    updates = {}
    
    for attr_id, attr in self.current_model.peripheral_attributes.items():
        updates[attr_id] = SelfAttribute(
            name=attr.name,
            value=None,  # Mark for removal
            confidence=0,
            importance=0,
            is_core=False,
            source=attr.source
        )
    
    # Apply updates
    drift = self.update_self_model(updates)
    drift.description = "Reset peripheral attributes"
    
    return drift
