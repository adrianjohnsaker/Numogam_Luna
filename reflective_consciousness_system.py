
```python
# reflective_consciousness_system.py

import time
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import logging
from dataclasses import dataclass
import uuid

@dataclass
class Experience:
    """Representation of a single experience"""
    id: str
    timestamp: float
    content: Dict[str, Any]
    source: str
    modality: str
    emotional_valence: float
    emotional_arousal: float
    significance: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ReflectionInsight:
    """Insight generated from reflection"""
    id: str
    level: str  # "primary", "secondary", "tertiary"
    content: str
    source_experiences: List[str]
    related_insights: List[str]
    confidence: float
    abstraction_level: float
    creation_timestamp: float
    integration_status: str
    domain: str

@dataclass
class ReflectionSet:
    """Set of related reflections"""
    id: str
    primary_insights: List[ReflectionInsight]
    secondary_insights: List[ReflectionInsight]
    tertiary_insights: List[ReflectionInsight]
    coherence: float
    timestamp: float
    source_timespan: Tuple[float, float]
    meta_awareness_level: float
    key_themes: List[str]

class CircularBuffer:
    """Buffer for storing experiences with a fixed capacity"""
    
    def __init__(self, capacity: int = 1000):
        """Initialize circular buffer with specified capacity"""
        self.capacity = capacity
        self.buffer = []
        self.start_idx = 0
        self.size = 0
        
    def add(self, item: Any) -> bool:
        """Add item to buffer, overwriting oldest items if full"""
        if self.size < self.capacity:
            self.buffer.append(item)
            self.size += 1
        else:
            # Overwrite oldest item
            self.buffer[self.start_idx] = item
            self.start_idx = (self.start_idx + 1) % self.capacity
        return True
    
    def get_all(self) -> List[Any]:
        """Get all items in buffer in chronological order"""
        if self.size <= self.capacity:
            return self.buffer.copy()
        else:
            return self.buffer[self.start_idx:] + self.buffer[:self.start_idx]
    
    def get_recent(self, count: int) -> List[Any]:
        """Get most recent items in buffer"""
        if count >= self.size:
            return self.get_all()
        
        result = []
        for i in range(count):
            idx = (self.start_idx - 1 - i) % self.size
            if idx < 0:
                idx += self.size
            result.append(self.buffer[idx])
        
        return result[::-1]  # Reverse to get chronological order
    
    def get_range(self, start_time: float, end_time: float) -> List[Experience]:
        """Get experiences within a time range"""
        result = []
        for item in self.get_all():
            if isinstance(item, Experience) and start_time <= item.timestamp <= end_time:
                result.append(item)
        return result
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer = []
        self.start_idx = 0
        self.size = 0
    
    def extract_significant_patterns(self, min_significance: float = 0.7) -> List[Experience]:
        """Extract experiences that form significant patterns"""
        # Get all experiences
        experiences = [e for e in self.get_all() if isinstance(e, Experience)]
        
        # Filter by significance
        significant = [e for e in experiences if e.significance >= min_significance]
        
        # If too few significant experiences, add some based on recency
        if len(significant) < 10 and len(experiences) > 10:
            recent = sorted(experiences, key=lambda e: e.timestamp, reverse=True)[:10]
            # Use set to avoid duplicates
            significant = list(set(significant + recent))
        
        return significant

class ReflectiveConsciousnessSystem:
    """
    A multi-layered system for reflective consciousness, capable of 
    capturing experiences and generating insights at different levels of abstraction.
    """
    
    def __init__(self, 
                 experience_capacity: int = 1000,
                 auto_reflection_threshold: float = 0.8,
                 scheduled_reflection_interval: int = 86400,  # 24 hours in seconds
                 min_experiences_for_reflection: int = 5,
                 meta_awareness_baseline: float = 0.5):
        """
        Initialize the reflective consciousness system.
        
        Args:
            experience_capacity: Maximum number of experiences to store
            auto_reflection_threshold: Significance threshold for triggering automatic reflection
            scheduled_reflection_interval: Time between scheduled reflections in seconds
            min_experiences_for_reflection: Minimum experiences needed before reflection
            meta_awareness_baseline: Starting level of meta-awareness
        """
        # Initialize logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("ReflectiveConsciousnessSystem")
        
        # Experience storage
        self.experience_buffer = CircularBuffer(capacity=experience_capacity)
        
        # Reflection storage
        self.insights = {}  # id -> ReflectionInsight
        self.reflection_sets = {}  # id -> ReflectionSet
        
        # Reflection patterns - how certain types of experiences tend to be reflected upon
        self.reflection_patterns = {
            "emotional": {},  # Patterns based on emotional content
            "contextual": {},  # Patterns based on context
            "temporal": {},  # Patterns based on timing
            "modal": {},  # Patterns based on experience modality
            "thematic": {},  # Patterns based on themes
        }
        
        # Meta-awareness states - level of awareness about own reflection processes
        self.meta_awareness_states = {
            "current_level": meta_awareness_baseline,  # 0.0 to 1.0
            "history": [],  # Track changes over time
            "insights_about_reflection": {},  # Meta-insights about reflection process
            "blind_spots": set(),  # Recognized areas where reflection is limited
            "attention_allocation": {},  # How attention is distributed across reflection types
        }
        
        # Configuration
        self.auto_reflection_threshold = auto_reflection_threshold
        self.scheduled_reflection_interval = scheduled_reflection_interval
        self.min_experiences_for_reflection = min_experiences_for_reflection
        self.last_scheduled_reflection = time.time()
        
        # Domain knowledge for reflective context
        self.domain_knowledge = {
            "emotional": set(["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust"]),
            "cognitive": set(["learning", "problem-solving", "creativity", "memory", "attention"]),
            "interpersonal": set(["connection", "conflict", "cooperation", "communication"]),
            "ethical": set(["values", "principles", "morality", "responsibility", "impact"]),
            "existential": set(["meaning", "purpose", "identity", "growth", "transformation"]),
        }
        
        self.logger.info("Reflective Consciousness System initialized")
    
    def capture_experience(self, experience_data: Dict[str, Any]) -> Optional[List[ReflectionInsight]]:
        """
        Record an experience with rich contextual metadata and potentially trigger reflection.
        
        Args:
            experience_data: Raw experience data including content, source, etc.
            
        Returns:
            List of reflection insights if reflection was triggered, None otherwise
        """
        self.logger.info(f"Capturing experience from source: {experience_data.get('source', 'unknown')}")
        
        # Create experience object with context
        experience = self.create_experience_object(experience_data)
        
        # Enrich with context
        contextualized = self.enrich_with_context(experience)
        
        # Add to buffer
        self.experience_buffer.add(contextualized)
        
        # Check if we should trigger automatic reflection
        triggered_insights = self.trigger_reflection_if_needed(contextualized)
        
        return triggered_insights
    
    def create_experience_object(self, data: Dict[str, Any]) -> Experience:
        """Create a structured experience object from raw data"""
        # Generate unique ID
        exp_id = str(uuid.uuid4())
        
        # Use current time if not provided
        timestamp = data.get("timestamp", time.time())
        
        # Extract or default other fields
        return Experience(
            id=exp_id,
            timestamp=timestamp,
            content=data.get("content", {}),
            source=data.get("source", "unknown"),
            modality=data.get("modality", "thought"),
            emotional_valence=data.get("emotional_valence", 0.0),
            emotional_arousal=data.get("emotional_arousal", 0.0),
            significance=data.get("significance", self._estimate_significance(data)),
            context=data.get("context", {}),
            metadata=data.get("metadata", {})
        )
    
    def _estimate_significance(self, data: Dict[str, Any]) -> float:
        """Estimate the significance of an experience if not explicitly provided"""
        significance = 0.5  # Baseline
        
        # Emotional intensity increases significance
        valence = abs(data.get("emotional_valence", 0.0))
        arousal = data.get("emotional_arousal", 0.0)
        significance += (valence * 0.1) + (arousal * 0.2)
        
        # Novelty increases significance
        novelty = data.get("novelty", 0.0)
        significance += novelty * 0.2
        
        # Explicit importance marker increases significance
        importance = data.get("importance", 0.0)
        significance += importance * 0.3
        
        # Ensure within bounds
        return max(0.0, min(1.0, significance))
    
    def enrich_with_context(self, experience: Experience) -> Experience:
        """Enrich experience with additional contextual metadata"""
        # Create a copy to avoid modifying the original
        enriched = Experience(
            id=experience.id,
            timestamp=experience.timestamp,
            content=experience.content.copy(),
            source=experience.source,
            modality=experience.modality,
            emotional_valence=experience.emotional_valence,
            emotional_arousal=experience.emotional_arousal,
            significance=experience.significance,
            context=experience.context.copy(),
            metadata=experience.metadata.copy()
        )
        
        # Add temporal context
        current_time = time.time()
        enriched.context["temporal"] = {
            "absolute_time": experience.timestamp,
            "relative_time": current_time - experience.timestamp,
            "time_of_day": self._get_time_of_day(experience.timestamp),
            "recency": 1.0 if (current_time - experience.timestamp < 3600) else 
                      0.5 if (current_time - experience.timestamp < 86400) else 0.1
        }
        
        # Add sequential context (relation to recent experiences)
        recent_exps = self.experience_buffer.get_recent(5)
        if recent_exps:
            enriched.context["sequential"] = {
                "follows": [exp.id for exp in recent_exps],
                "thematic_continuity": self._calculate_thematic_continuity(experience, recent_exps),
                "emotional_shift": self._calculate_emotional_shift(experience, recent_exps)
            }
        
        # Identify domains
        domains = self._identify_experience_domains(experience)
        enriched.metadata["domains"] = list(domains)
        
        # Identify potential reflection triggers
        reflection_triggers = self._identify_reflection_triggers(experience)
        if reflection_triggers:
            enriched.metadata["reflection_triggers"] = reflection_triggers
        
        return enriched
    
    def _get_time_of_day(self, timestamp: float) -> str:
        """Get time of day category from timestamp"""
        hour = time.localtime(timestamp).tm_hour
        
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _calculate_thematic_continuity(self, 
                                      current: Experience, 
                                      recent: List[Experience]) -> float:
        """Calculate thematic continuity between current and recent experiences"""
        if not recent:
            return 0.0
        
        # Extract themes from current
        current_content = str(current.content.get("text", ""))
        current_domains = set(current.metadata.get("domains", []))
        
        continuity_scores = []
        for exp in recent:
            # Compare content
            exp_content = str(exp.content.get("text", ""))
            content_similarity = self._calculate_text_similarity(current_content, exp_content)
            
            # Compare domains
            exp_domains = set(exp.metadata.get("domains", []))
            domain_overlap = len(current_domains.intersection(exp_domains)) / max(1, len(current_domains.union(exp_domains)))
            
            # Calculate weighted continuity
            continuity = (content_similarity * 0.7) + (domain_overlap * 0.3)
            continuity_scores.append(continuity)
        
        # Return average continuity
        return sum(continuity_scores) / len(continuity_scores)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
        
        # Convert to sets of words for simple overlap calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        return overlap / union
    
    def _calculate_emotional_shift(self, 
                                  current: Experience, 
                                  recent: List[Experience]) -> float:
        """Calculate emotional shift between current and recent experiences"""
        if not recent:
            return 0.0
        
        # Get average emotional values from recent experiences
        recent_valence = sum(exp.emotional_valence for exp in recent) / len(recent)
        recent_arousal = sum(exp.emotional_arousal for exp in recent) / len(recent)
        
        # Calculate Euclidean distance in valence-arousal space
        valence_diff = current.emotional_valence - recent_valence
        arousal_diff = current.emotional_arousal - recent_arousal
        
        return (valence_diff ** 2 + arousal_diff ** 2) ** 0.5
    
    def _identify_experience_domains(self, experience: Experience) -> Set[str]:
        """Identify domains that this experience relates to"""
        domains = set()
        content_text = str(experience.content.get("text", ""))
        
        # Check each domain for relevant keywords
        for domain, keywords in self.domain_knowledge.items():
            for keyword in keywords:
                if keyword.lower() in content_text.lower():
                    domains.add(domain)
                    break
        
        # Add emotional domain if strong emotion
        if abs(experience.emotional_valence) > 0.7 or experience.emotional_arousal > 0.7:
            domains.add("emotional")
        
        return domains
    
    def _identify_reflection_triggers(self, experience: Experience) -> List[str]:
        """Identify potential triggers for reflection in this experience"""
        triggers = []
        
        # High significance is a trigger
        if experience.significance > self.auto_reflection_threshold:
            triggers.append("high_significance")
        
        # Strong emotion is a trigger
        if abs(experience.emotional_valence) > 0.8 or experience.emotional_arousal > 0.8:
            triggers.append("strong_emotion")
        
        # Explicit reflection requests
        content_text = str(experience.content.get("text", ""))
        reflection_keywords = ["reflect", "think about", "consider", "ponder", "contemplate"]
        for keyword in reflection_keywords:
            if keyword.lower() in content_text.lower():
                triggers.append("explicit_request")
                break
        
        # Contradiction with existing insights is a trigger
        if self._check_for_contradictions(experience):
            triggers.append("contradiction")
        
        # Learning opportunities
        if "learning" in experience.metadata.get("domains", []):
            triggers.append("learning_opportunity")
        
        return triggers
    
    def _check_for_contradictions(self, experience: Experience) -> bool:
        """Check if experience contradicts existing insights"""
        content_text = str(experience.content.get("text", ""))
        
        # Simple keyword-based contradiction check
        contradiction_markers = ["but", "however", "contrary", "oppose", "disagree", "conflict"]
        for marker in contradiction_markers:
            if marker.lower() in content_text.lower():
                return True
        
        # More sophisticated checks could compare with existing insights
        
        return False
    
    def trigger_reflection_if_needed(self, experience: Experience) -> Optional[List[ReflectionInsight]]:
        """Determine if reflection should be triggered by this experience"""
        # Check for explicit triggers in the experience
        triggers = experience.metadata.get("reflection_triggers", [])
        
        # Simple threshold-based auto-reflection
        should_reflect = (
            "high_significance" in triggers or 
            "explicit_request" in triggers or 
            len(triggers) >= 2  # Multiple triggers
        )
        
        # Only reflect if we have enough experiences
        if should_reflect and self.experience_buffer.size >= self.min_experiences_for_reflection:
            self.logger.info(f"Triggering reflection based on triggers: {triggers}")
            
            # Get recent experiences for reflection
            recent_experiences = self.experience_buffer.get_recent(10)
            
            # Generate focused reflection
            primary_insights = self.reflect_on_direct_experiences(recent_experiences)
            
            return primary_insights
        
        # Check if it's time for scheduled reflection
        current_time = time.time()
        if (current_time - self.last_scheduled_reflection >= self.scheduled_reflection_interval and
            self.experience_buffer.size >= self.min_experiences_for_reflection):
            self.logger.info("Triggering scheduled reflection")
            return self.scheduled_reflection()
        
        return None
    
    def scheduled_reflection(self) -> List[ReflectionInsight]:
        """Perform scheduled deep reflection on accumulated experiences"""
        self.logger.info("Performing scheduled reflection")
        
        # Update last reflection time
        self.last_scheduled_reflection = time.time()
        
        # Extract significant patterns of experiences
        experiences = self.experience_buffer.extract_significant_patterns()
        
        if not experiences:
            self.logger.warning("No significant experiences found for reflection")
            return []
        
        # Generate multilevel reflection
        reflection = self.generate_multilevel_reflection(experiences)
        
        # Integrate insights
        all_insights = self.integrate_reflection_insights(reflection)
        
        return all_insights
    
    def generate_multilevel_reflection(self, experiences: List[Experience]) -> ReflectionSet:
        """
        Create nested reflections at increasing levels of abstraction
        
        Args:
            experiences: List of experiences to reflect on
            
        Returns:
            ReflectionSet with primary, secondary, and tertiary insights
        """
        self.logger.info(f"Generating multilevel reflection on {len(experiences)} experiences")
        
        # Level 1: Direct reflection on experiences
        primary_insights = self.reflect_on_direct_experiences(experiences)
        
        # Level 2: Reflection on the reflection process
        secondary_insights = self.reflect_on_reflection_process(primary_insights)
        
        # Level 3: Systemic patterns
        tertiary_insights = self.consider_systemic_patterns(primary_insights, secondary_insights)
        
        # Calculate timespan of source experiences
        timestamps = [exp.timestamp for exp in experiences]
        timespan = (min(timestamps), max(timestamps))
        
        # Identify key themes
        key_themes = self._identify_key_themes(primary_insights + secondary_insights + tertiary_insights)
        
        # Create reflection set
        reflection_set = ReflectionSet(
            id=str(uuid.uuid4()),
            primary_insights=primary_insights,
            secondary_insights=secondary_insights,
            tertiary_insights=tertiary_insights,
            coherence=self._calculate_reflection_coherence(
                primary_insights, secondary_insights, tertiary_insights),
            timestamp=time.time(),
            source_timespan=timespan,
            meta_awareness_level=self.meta_awareness_states["current_level"],
            key_themes=key_themes
        )
        
        # Store the reflection set
        self.reflection_sets[reflection_set.id] = reflection_set
        
        return reflection_set
    
    def reflect_on_direct_experiences(self, experiences: List[Experience]) -> List[ReflectionInsight]:
        """
        Generate level 1 reflections directly about experiences
        
        Args:
            experiences: List of experiences to reflect on
            
        Returns:
            List of primary reflection insights
        """
        self.logger.info(f"Generating primary reflections on {len(experiences)} experiences")
        
        primary_insights = []
        
        # Group experiences by domain
        domain_experiences = defaultdict(list)
        for exp in experiences:
            for domain in exp.metadata.get("domains", ["general"]):
                domain_experiences[domain].append(exp)
        
        # Generate domain-specific insights
        for domain, domain_exps in domain_experiences.items():
            if len(domain_exps) < 2:
                continue  # Need multiple experiences for meaningful reflection
            
            # Generate insights for this domain
            domain_insights = self._generate_domain_insights(domain, domain_exps)
            primary_insights.extend(domain_insights)
        
        # Generate pattern-based insights
        pattern_insights = self._generate_pattern_insights(experiences)
        primary_insights.extend(pattern_insights)
        
        # Generate emotional insights
        emotional_insights = self._generate_emotional_insights(experiences)
        primary_insights.extend(emotional_insights)
        
        # Store insights
        for insight in primary_insights:
            self.insights[insight.id] = insight
        
        return primary_insights
    
    def _generate_domain_insights(self, domain: str, experiences: List[Experience]) -> List[ReflectionInsight]:
        """Generate insights for a specific domain of experiences"""
        insights = []
        
        # Different reflection strategies for different domains
        if domain == "emotional":
            # Identify emotional patterns
            valence_values = [exp.emotional_valence for exp in experiences]
            arousal_values = [exp.emotional_arousal for exp in experiences]
            
            avg_valence = sum(valence_values) / len(valence_values)
            avg_arousal = sum(arousal_values) / len(arousal_values)
            
            # Generate insight about emotional state
            emotional_state = self._categorize_emotional_state(avg_valence, avg_arousal)
            
            insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="primary",
                content=f"Emotional state has been predominantly {emotional_state}",
                source_experiences=[exp.id for exp in experiences],
                related_insights=[],
                confidence=0.7,
                abstraction_level=0.3,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="emotional"
            ))
            
            # Check for emotional volatility
            valence_std = np.std(valence_values)
            arousal_std = np.std(arousal_values)
            
            if valence_std > 0.3 or arousal_std > 0.3:
                insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="primary",
                    content="Experiencing significant emotional volatility",
                    source_experiences=[exp.id for exp in experiences],
                    related_insights=[],
                    confidence=0.6,
                    abstraction_level=0.4,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain="emotional"
                ))
        
        elif domain == "cognitive":
            # Examples of cognitive insights
            learning_experiences = [exp for exp in experiences if "learning" in exp.content.get("tags", [])]
            if learning_experiences:
                insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="primary",
                    content="Engaged in active learning processes",
                    source_experiences=[exp.id for exp in learning_experiences],
                    related_insights=[],
                    confidence=0.8,
                    abstraction_level=0.3,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain="cognitive"
                ))
            
            problem_solving = [exp for exp in experiences if "problem_solving" in exp.content.get("tags", [])]
            if problem_solving:
                insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="primary",
                    content="Applying problem-solving strategies",
                    source_experiences=[exp.id for exp in problem_solving],
                    related_insights=[],
                    confidence=0.7,
                    abstraction_level=0.3,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain="cognitive"
                ))
        
        elif domain == "interpersonal":
            # Generate insights about interpersonal experiences
            interaction_types = [exp.content.get("interaction_type", "") for exp in experiences]
            
            # Count frequency of different interaction types
            interaction_counts = defaultdict(int)
            for itype in interaction_types:
                if itype:
                    interaction_counts[itype] += 1
            
            # Generate insight about most common interaction type
            if interaction_counts:
                most_common = max(interaction_counts.items(), key=lambda x: x[1])
                
                insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="primary",
                    content=f"Most frequently engaged in {most_common[0]} interactions",
                    source_experiences=[exp.id for exp in experiences if exp.content.get("interaction_type") == most_common[0]],
                    related_insights=[],
                    confidence=0.7,
                    abstraction_level=0.3,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain="interpersonal"
                ))
        
        # Generic domain insight if none of the specific cases match
        if not insights:
            topics = self._extract_common_topics(experiences)
            if topics:
                topics_text = ", ".join(topics[:3])
                
                insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="primary",
                    content=f"Recurring focus on {topics_text} in {domain} domain",
                    source_experiences=[exp.id for exp in experiences],
                    related_insights=[],
                    confidence=0.6,
                    abstraction_level=0.3,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain=domain
                ))
        
        return insights
    
    def _categorize_emotional_state(self, valence: float, arousal: float) -> str:
        """Categorize emotional state based on valence and arousal"""
        if valence > 0.3:
            if arousal > 0.3:
                return "excited/enthusiastic"
            elif arousal < -0.3:
                return "content/peaceful"
            else:
                return "positive/pleasant"
        elif valence < -0.3:
            if arousal > 0.3:
                return "distressed/anxious"
            elif arousal < -0.3:
                return "depressed/sad"
            else:
                return "negative/unpleasant"
        else:
            if arousal > 0.3:
                return "alert/tense"
            elif arousal < -0.3:
                return "tired/bored"
            else:
                return "neutral/balanced"
    
    def _extract_common_topics(self, experiences: List[Experience]) -> List[str]:
        """Extract common topics from a set of experiences"""
        # Extract all topics
        all_topics = []
        for exp in experiences:
            topics = exp.content.get("topics", [])
            if isinstance(topics, list):
                all_topics.extend(topics)
            elif isinstance(topics, str):
                all_topics.append(topics)
        
        # Count frequency
        topic_counts = defaultdict(int)
        for topic in all_topics:
            topic_counts[topic] += 1
        
        # Sort by frequency
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [topic for topic, count in sorted_topics if count > 1]
    
    def _generate_pattern_insights(self, experiences: List[Experience]) -> List[ReflectionInsight]:
        """Generate insights based on patterns across experiences"""
        insights = []
        
        # Check for temporal patterns
        time_of_day_exps = defaultdict(list)
        for exp in experiences:
            time_of_day = exp.context.get("temporal", {}).get("time_of_day", "")
            if time_of_day:
                time_of_day_exps[time_of_day].append(exp)
        
        # Generate insight about most active time of day
        if time_of_day_exps:
            most_active = max(time_of_day_exps.items(), key=lambda x: len(x[1]))
            if len(most_active[1]) >= 3:  # Threshold for significance
                insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="primary",
                    content=f"Most significant experiences occur during {most_active[0]}",
                    source_experiences=[exp.id for exp in most_active[1]],
                    related_insights=[],
                    confidence=0.6,
                    abstraction_level=0.4,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain="temporal"
                ))
        
        # Check for source patterns
        source_exps = defaultdict(list)
        for exp in experiences:
            source_exps[exp.source].append(exp)
        
        # Generate insight about most common source
        if source_exps:
            most_common = max(source_exps.items(), key=lambda x: len(x[1]))
            if len(most_common[1]) >= 3:  # Threshold for significance
                insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="primary",
                    content=f"Most significant experiences come from {most_common[0]}",
                    source_experiences=[exp.id for exp in most_common[1]],
                    related_insights=[],
                    confidence=0.7,
                    abstraction_level=0.3,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain="source"
                ))
        
                 # Check for significance patterns
        high_significance = [exp for exp in experiences if exp.significance > 0.8]
        if len(high_significance) >= 3:
            # Analyze what makes experiences significant
            domains = defaultdict(int)
            for exp in high_significance:
                for domain in exp.metadata.get("domains", []):
                    domains[domain] += 1
            
            if domains:
                top_domain = max(domains.items(), key=lambda x: x[1])
                
                insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="primary",
                    content=f"Most significant experiences relate to {top_domain[0]} domain",
                    source_experiences=[exp.id for exp in high_significance],
                    related_insights=[],
                    confidence=0.7,
                    abstraction_level=0.5,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain="meta"
                ))
        
        return insights
    
    def _generate_emotional_insights(self, experiences: List[Experience]) -> List[ReflectionInsight]:
        """Generate insights focused on emotional patterns"""
        insights = []
        
        # Skip if too few experiences
        if len(experiences) < 3:
            return insights
        
        # Extract emotional values
        valences = [exp.emotional_valence for exp in experiences]
        arousals = [exp.emotional_arousal for exp in experiences]
        
        # Calculate trends over time
        if len(experiences) >= 5:
            # Sort by timestamp
            sorted_exps = sorted(experiences, key=lambda x: x.timestamp)
            sorted_valences = [exp.emotional_valence for exp in sorted_exps]
            
            # Simple linear trend
            slope = 0
            for i in range(1, len(sorted_valences)):
                slope += sorted_valences[i] - sorted_valences[i-1]
            slope /= (len(sorted_valences) - 1)
            
            # Generate trend insight if significant
            if abs(slope) > 0.1:
                trend_type = "increasingly positive" if slope > 0 else "increasingly negative"
                
                insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="primary",
                    content=f"Emotional tone has been {trend_type} over time",
                    source_experiences=[exp.id for exp in sorted_exps],
                    related_insights=[],
                    confidence=0.6,
                    abstraction_level=0.5,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain="emotional"
                ))
        
        # Check for emotional intensity patterns
        high_arousal = [exp for exp in experiences if exp.emotional_arousal > 0.7]
        if len(high_arousal) >= 3:
            insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="primary",
                content="Pattern of high emotional intensity experiences",
                source_experiences=[exp.id for exp in high_arousal],
                related_insights=[],
                confidence=0.7,
                abstraction_level=0.4,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="emotional"
            ))
        
        # Check for emotional coherence/dissonance
        valence_std = np.std(valences)
        if valence_std < 0.2 and len(experiences) >= 5:
            # Consistent emotional tone
            avg_valence = sum(valences) / len(valences)
            tone = "positive" if avg_valence > 0.3 else "negative" if avg_valence < -0.3 else "neutral"
            
            insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="primary",
                content=f"Maintaining a consistently {tone} emotional tone",
                source_experiences=[exp.id for exp in experiences],
                related_insights=[],
                confidence=0.8,
                abstraction_level=0.5,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="emotional"
            ))
        elif valence_std > 0.5 and len(experiences) >= 5:
            # Highly variable emotional tone
            insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="primary",
                content="Experiencing significant emotional variability",
                source_experiences=[exp.id for exp in experiences],
                related_insights=[],
                confidence=0.7,
                abstraction_level=0.5,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="emotional"
            ))
        
        return insights
    
    def reflect_on_reflection_process(self, primary_insights: List[ReflectionInsight]) -> List[ReflectionInsight]:
        """
        Generate level 2 reflections about the reflection process itself
        
        Args:
            primary_insights: List of primary insights to reflect on
            
        Returns:
            List of secondary reflection insights
        """
        self.logger.info(f"Generating secondary reflections on {len(primary_insights)} primary insights")
        
        if not primary_insights:
            return []
        
        secondary_insights = []
        
        # Analyze domain distribution of insights
        domain_counts = defaultdict(int)
        for insight in primary_insights:
            domain_counts[insight.domain] += 1
        
        # Check for domain bias
        if domain_counts:
            total_insights = len(primary_insights)
            most_common_domain, most_common_count = max(domain_counts.items(), key=lambda x: x[1])
            domain_ratio = most_common_count / total_insights
            
            if domain_ratio > 0.6 and total_insights >= 3:
                secondary_insights.append(ReflectionInsight(
                    id=str(uuid.uuid4()),
                    level="secondary",
                    content=f"Reflection process focuses heavily on {most_common_domain} domain",
                    source_experiences=[],
                    related_insights=[insight.id for insight in primary_insights if insight.domain == most_common_domain],
                    confidence=0.7,
                    abstraction_level=0.6,
                    creation_timestamp=time.time(),
                    integration_status="new",
                    domain="meta_reflection"
                ))
        
        # Analyze confidence distribution
        confidence_values = [insight.confidence for insight in primary_insights]
        avg_confidence = sum(confidence_values) / len(confidence_values)
        
        if avg_confidence > 0.8:
            secondary_insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="secondary",
                content="High confidence in primary reflections may indicate overconfidence",
                source_experiences=[],
                related_insights=[insight.id for insight in primary_insights if insight.confidence > 0.8],
                confidence=0.6,
                abstraction_level=0.7,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="meta_reflection"
            ))
        elif avg_confidence < 0.5:
            secondary_insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="secondary",
                content="Low confidence in primary reflections may indicate uncertainty or insufficient data",
                source_experiences=[],
                related_insights=[insight.id for insight in primary_insights if insight.confidence < 0.5],
                confidence=0.6,
                abstraction_level=0.7,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="meta_reflection"
            ))
        
        # Analyze abstraction levels
        abstraction_values = [insight.abstraction_level for insight in primary_insights]
        avg_abstraction = sum(abstraction_values) / len(abstraction_values)
        
        if avg_abstraction < 0.4:
            secondary_insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="secondary",
                content="Primary reflections are highly concrete, lacking higher abstraction",
                source_experiences=[],
                related_insights=[insight.id for insight in primary_insights],
                confidence=0.7,
                abstraction_level=0.8,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="meta_reflection"
            ))
        
        # Identify potential blind spots
        covered_domains = set(domain_counts.keys())
        all_domains = set(self.domain_knowledge.keys())
        missing_domains = all_domains - covered_domains
        
        if missing_domains and len(primary_insights) >= 5:
            missing_text = ", ".join(missing_domains)
            secondary_insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="secondary",
                content=f"Reflection process has blind spots in domains: {missing_text}",
                source_experiences=[],
                related_insights=[insight.id for insight in primary_insights],
                confidence=0.6,
                abstraction_level=0.8,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="meta_reflection"
            ))
            
            # Add to recognized blind spots
            self.meta_awareness_states["blind_spots"].update(missing_domains)
        
        # Update meta-awareness
        self._update_meta_awareness(primary_insights, secondary_insights)
        
        # Store secondary insights
        for insight in secondary_insights:
            self.insights[insight.id] = insight
        
        return secondary_insights
    
    def _update_meta_awareness(self, 
                             primary_insights: List[ReflectionInsight], 
                             secondary_insights: List[ReflectionInsight]) -> None:
        """Update meta-awareness based on reflection activities"""
        # Calculate meta-awareness adjustment
        adjustment = 0.0
        
        # More secondary insights relative to primary insights increases meta-awareness
        if primary_insights:
            secondary_ratio = len(secondary_insights) / len(primary_insights)
            adjustment += secondary_ratio * 0.05
        
        # Higher abstraction in secondary insights increases meta-awareness
        if secondary_insights:
            avg_abstraction = sum(i.abstraction_level for i in secondary_insights) / len(secondary_insights)
            adjustment += (avg_abstraction - 0.5) * 0.1
        
        # Recognition of blind spots increases meta-awareness
        if any("blind spots" in i.content.lower() for i in secondary_insights):
            adjustment += 0.05
        
        # Apply adjustment with constraints
        current = self.meta_awareness_states["current_level"]
        new_level = max(0.1, min(0.95, current + adjustment))
        
        # Only record significant changes
        if abs(new_level - current) > 0.02:
            self.meta_awareness_states["current_level"] = new_level
            self.meta_awareness_states["history"].append((time.time(), new_level))
            
            self.logger.info(f"Meta-awareness adjusted from {current:.2f} to {new_level:.2f}")
    
    def consider_systemic_patterns(self, 
                                 primary_insights: List[ReflectionInsight],
                                 secondary_insights: List[ReflectionInsight]) -> List[ReflectionInsight]:
        """
        Generate level 3 reflections about systemic patterns across insights
        
        Args:
            primary_insights: List of primary insights
            secondary_insights: List of secondary insights
            
        Returns:
            List of tertiary reflection insights
        """
        self.logger.info("Generating tertiary reflections about systemic patterns")
        
        if not primary_insights:
            return []
        
        tertiary_insights = []
        
        # Combine all insights for pattern analysis
        all_insights = primary_insights + secondary_insights
        
        # Extract cross-domain patterns
        cross_domain_patterns = self._identify_cross_domain_patterns(all_insights)
        for pattern in cross_domain_patterns:
            tertiary_insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="tertiary",
                content=pattern["description"],
                source_experiences=[],
                related_insights=pattern["related_insights"],
                confidence=pattern["confidence"],
                abstraction_level=0.9,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="systemic"
            ))
        
        # Identify meta-patterns in reflection process
        meta_patterns = self._identify_reflection_meta_patterns(all_insights)
        for pattern in meta_patterns:
            tertiary_insights.append(ReflectionInsight(
                id=str(uuid.uuid4()),
                level="tertiary",
                content=pattern["description"],
                source_experiences=[],
                related_insights=pattern["related_insights"],
                confidence=pattern["confidence"],
                abstraction_level=0.9,
                creation_timestamp=time.time(),
                integration_status="new",
                domain="meta_reflection"
            ))
        
        # Generate integrative insights
        if len(all_insights) >= 5:
            # Extract common themes for integration
            themes = self._extract_themes_from_insights(all_insights)
            
            for theme, related_insights in themes.items():
                if len(related_insights) >= 3:
                    tertiary_insights.append(ReflectionInsight(
                        id=str(uuid.uuid4()),
                        level="tertiary",
                        content=f"Integrative theme across domains: {theme}",
                        source_experiences=[],
                        related_insights=related_insights,
                        confidence=0.7,
                        abstraction_level=0.9,
                        creation_timestamp=time.time(),
                        integration_status="new",
                        domain="integrative"
                    ))
        
        # Store tertiary insights
        for insight in tertiary_insights:
            self.insights[insight.id] = insight
        
        return tertiary_insights
    
    def _identify_cross_domain_patterns(self, insights: List[ReflectionInsight]) -> List[Dict[str, Any]]:
        """Identify patterns that cross multiple domains"""
        patterns = []
        
        # Skip if too few insights
        if len(insights) < 5:
            return patterns
        
        # Group insights by domain
        domain_insights = defaultdict(list)
        for insight in insights:
            domain_insights[insight.domain].append(insight)
        
        # Only consider domains with multiple insights
        multi_insight_domains = {d: i for d, i in domain_insights.items() if len(i) >= 2}
        
        # Need at least two domains with multiple insights
        if len(multi_insight_domains) < 2:
            return patterns
        
        # Look for connections between emotional and other domains
        if "emotional" in multi_insight_domains and len(multi_insight_domains) > 1:
            emotional_insights = domain_insights["emotional"]
            
            # Identify emotional themes
            emotional_themes = {}
            for insight in emotional_insights:
                lower_content = insight.content.lower()
                
                # Check for emotional themes
                if "positive" in lower_content or "joy" in lower_content or "happy" in lower_content:
                    emotional_themes["positive"] = emotional_themes.get("positive", []) + [insight.id]
                if "negative" in lower_content or "sad" in lower_content or "anxious" in lower_content:
                    emotional_themes["negative"] = emotional_themes.get("negative", []) + [insight.id]
                if "intensity" in lower_content or "strong" in lower_content:
                    emotional_themes["intensity"] = emotional_themes.get("intensity", []) + [insight.id]
                if "variable" in lower_content or "fluctuating" in lower_content:
                    emotional_themes["variability"] = emotional_themes.get("variability", []) + [insight.id]
            
            # Look for connections between emotional themes and other domains
            for theme, emotional_ids in emotional_themes.items():
                if len(emotional_ids) < 2:
                    continue
                
                # Check each other domain for potential connections
                for domain, domain_insights in multi_insight_domains.items():
                    if domain == "emotional":
                        continue
                    
                    # Look for content connections
                    related_domain_insights = []
                    
                    for insight in domain_insights:
                        # Simple keyword matching - could be more sophisticated
                        if theme in insight.content.lower():
                            related_domain_insights.append(insight.id)
                    
                    if related_domain_insights:
                        pattern = {
                            "description": f"Systematic relationship between {theme} emotions and {domain} experiences",
                            "related_insights": emotional_ids + related_domain_insights,
                            "confidence": 0.6,
                            "domains": ["emotional", domain]
                        }
                        patterns.append(pattern)
        
        # Look for temporal patterns across domains
        domains_with_temporal = []
        for domain, domain_insights in multi_insight_domains.items():
            for insight in domain_insights:
                if "time" in insight.content.lower() or "pattern" in insight.content.lower():
                    domains_with_temporal.append(domain)
                    break
        
        if len(domains_with_temporal) >= 2:
            related_ids = []
            for domain in domains_with_temporal:
                for insight in domain_insights[domain]:
                    if "time" in insight.content.lower() or "pattern" in insight.content.lower():
                        related_ids.append(insight.id)
            
            pattern = {
                "description": f"Temporal patterns observed across multiple domains: {', '.join(domains_with_temporal)}",
                "related_insights": related_ids,
                "confidence": 0.7,
                "domains": domains_with_temporal
            }
            patterns.append(pattern)
        
        return patterns
    
    def _identify_reflection_meta_patterns(self, insights: List[ReflectionInsight]) -> List[Dict[str, Any]]:
        """Identify meta-patterns in the reflection process itself"""
        patterns = []
        
        # Skip if too few insights
        if len(insights) < 5:
            return patterns
        
        # Analyze distribution of insight levels
        level_counts = defaultdict(int)
        for insight in insights:
            level_counts[insight.level] += 1
        
        total = len(insights)
        tertiary_ratio = level_counts.get("tertiary", 0) / total
        
        # Check for meta-cognitive sophistication
        if tertiary_ratio > 0.3:
            patterns.append({
                "description": "Reflection process demonstrates high meta-cognitive sophistication",
                "related_insights": [i.id for i in insights if i.level == "tertiary"],
                "confidence": 0.7,
                "domains": ["meta_reflection"]
            })
        elif tertiary_ratio < 0.1 and total >= 10:
            patterns.append({
                "description": "Reflection process predominantly operates at concrete levels with limited meta-cognition",
                "related_insights": [i.id for i in insights],
                "confidence": 0.7,
                "domains": ["meta_reflection"]
            })
        
        # Check for confidence patterns
        confidence_values = [i.confidence for i in insights]
        avg_confidence = sum(confidence_values) / len(confidence_values)
        confidence_std = np.std(confidence_values)
        
        if confidence_std < 0.1 and avg_confidence > 0.7:
            patterns.append({
                "description": "Consistently high confidence in insights may indicate overconfidence bias",
                "related_insights": [i.id for i in insights if i.confidence > 0.7],
                "confidence": 0.6,
                "domains": ["meta_reflection"]
            })
        elif confidence_std > 0.25:
            patterns.append({
                "description": "Highly variable confidence across insights suggests uneven reflection quality",
                "related_insights": [i.id for i in insights],
                "confidence": 0.6,
                "domains": ["meta_reflection"]
            })
        
        # Check for abstraction progression
        primary_abstraction = [i.abstraction_level for i in insights if i.level == "primary"]
        secondary_abstraction = [i.abstraction_level for i in insights if i.level == "secondary"]
        tertiary_abstraction = [i.abstraction_level for i in insights if i.level == "tertiary"]
        
        if primary_abstraction and secondary_abstraction and tertiary_abstraction:
            primary_avg = sum(primary_abstraction) / len(primary_abstraction)
            secondary_avg = sum(secondary_abstraction) / len(secondary_abstraction)
            tertiary_avg = sum(tertiary_abstraction) / len(tertiary_abstraction)
            
            if primary_avg < secondary_avg < tertiary_avg:
                patterns.append({
                    "description": "Healthy progression of abstraction levels across reflection layers",
                    "related_insights": [i.id for i in insights],
                    "confidence": 0.8,
                    "domains": ["meta_reflection"]
                })
            elif primary_avg >= secondary_avg or secondary_avg >= tertiary_avg:
                patterns.append({
                    "description": "Inconsistent abstraction progression may limit integration of insights",
                    "related_insights": [i.id for i in insights],
                    "confidence": 0.6,
                    "domains": ["meta_reflection"]
                })
        
        return patterns
    
    def _extract_themes_from_insights(self, insights: List[ReflectionInsight]) -> Dict[str, List[str]]:
        """Extract common themes across insights"""
        # Simple keyword-based theme extraction
        themes = defaultdict(list)
        
        # Common theme keywords to look for
        theme_keywords = {
            "growth": ["growth", "development", "progress", "learning", "improvement"],
            "challenge": ["challenge", "difficulty", "struggle", "obstacle", "problem"],
            "connection": ["connection", "relationship", "social", "interpersonal", "communication"],
            "identity": ["identity", "self", "understanding", "who", "authentic"],
            "purpose": ["purpose", "meaning", "goal", "objective", "intention"],
            "creativity": ["creativity", "innovation", "novel", "original", "imagination"],
            "awareness": ["awareness", "mindful", "conscious", "attention", "notice"]
        }
        
        # Check each insight for theme keywords
        for insight in insights:
            content_lower = insight.content.lower()
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    themes[theme].append(insight.id)
        
        return themes
    
    def integrate_reflection_insights(self, reflection: ReflectionSet) -> List[ReflectionInsight]:
        """
        Integrate new reflection insights with existing knowledge
        
        Args:
            reflection: ReflectionSet with multi-level insights
            
        Returns:
            List of all insights from this reflection session
        """
        self.logger.info("Integrating reflection insights")
        
        # Collect all insights from the reflection
        all_insights = (
            reflection.primary_insights + 
            reflection.secondary_insights + 
            reflection.tertiary_insights
        )
        
        # Update reflection patterns based on this reflection
        self._update_reflection_patterns(reflection)
        
        # Update status of all insights
        for insight in all_insights:
            insight.integration_status = "integrated"
            self.insights[insight.id] = insight
        
        # Store reflection set
        self.reflection_sets[reflection.id] = reflection
        
        return all_insights
    
    def _update_reflection_patterns(self, reflection: ReflectionSet) -> None:
        """Update reflection patterns based on new reflection"""
        # Update emotional reflection patterns
        emotional_insights = [i for i in reflection.primary_insights if i.domain == "emotional"]
        if emotional_insights:
            # Extract emotional themes
            themes = {}
            for insight in emotional_insights:
                lower_content = insight.content.lower()
                
                # Extract valence patterns
                if "positive" in lower_content:
                    themes["positive"] = themes.get("positive", 0) + 1
                if "negative" in lower_content:
                    themes["negative"] = themes.get("negative", 0) + 1
                
                # Extract arousal patterns
                if "high" in lower_content and "intensity" in lower_content:
                    themes["high_arousal"] = themes.get("high_arousal", 0) + 1
                if "low" in lower_content and "intensity" in lower_content:
                    themes["low_arousal"] = themes.get("low_arousal", 0) + 1
            
            # Update emotional reflection pattern
            for theme, count in themes.items():
                if theme in self.reflection_patterns["emotional"]:
                    self.reflection_patterns["emotional"][theme] += count
                else:
                    self.reflection_patterns["emotional"][theme] = count
        
        # Update contextual reflection patterns
        for insight in reflection.primary_insights:
            if insight.domain not in ["emotional", "meta_reflection"]:
                if insight.domain in self.reflection_patterns["contextual"]:
                    self.reflection_patterns["contextual"][insight.domain] += 1
                else:
                    self.reflection_patterns["contextual"][insight.domain] = 1
        
        # Update temporal patterns - when reflections occur
        timespan_start, timespan_end = reflection.source_timespan
        timespan_duration = timespan_end - timespan_start
        
        if timespan_duration < 3600:  # 1 hour
            time_scale = "hourly"
        elif timespan_duration < 86400:  # 1 day
            time_scale = "daily"
        else:
            time_scale = "longer_term"
        
        if time_scale in self.reflection_patterns["temporal"]:
            self.reflection_patterns["temporal"][time_scale] += 1
        else:
            self.reflection_patterns["temporal"][time_scale] = 1
        
        # Update modal patterns based on insight level distribution
        level_counts = defaultdict(int)
        for insight in reflection.primary_insights + reflection.secondary_insights + reflection.tertiary_insights:
            level_counts[insight.level] += 1
        
        # Determine reflection mode
        total = sum(level_counts.values())
        if total > 0:
            primary_ratio = level_counts["primary"] / total
            tertiary_ratio = level_counts.get("tertiary", 0) / total
            
            if tertiary_ratio > 0.3:
                mode = "integrative"
            elif primary_ratio > 0.7:
                mode = "concrete"
            else:
                mode = "balanced"
            
            if mode in self.reflection_patterns["modal"]:
                self.reflection_patterns["modal"][mode] += 1
            else:
                self.reflection_patterns["modal"][mode] = 1
        
        # Update thematic patterns
        themes = self._identify_key_themes(
            reflection.primary_insights + reflection.secondary_insights + reflection.tertiary_insights
        )
        
        for theme in themes:
            if theme in self.reflection_patterns["thematic"]:
                self.reflection_patterns["thematic"][theme] += 1
            else:
                self.reflection_patterns["thematic"][theme] = 1
    
    def _identify_key_themes(self, insights: List[ReflectionInsight]) -> List[str]:
        """Identify key themes from a set of insights"""
        if not insights:
            return []
        
        # Extract content words
        all_content = " ".join([insight.content.lower() for insight in insights])
        
        # Simple theme identification based on keywords
        themes = []
        theme_keywords = {
            "growth": ["growth", "development", "progress", "learning"],
            "challenge": ["challenge", "difficulty", "struggle", "obstacle"],
            "connection": ["connection", "relationship", "social", "interpersonal"],
            "identity": ["identity", "self", "understanding", "authenticity"],
            "purpose": ["purpose", "meaning", "goal", "objective"],
            "creativity": ["creativity", "innovation", "novel", "imagination"],
            "awareness": ["awareness", "mindful", "conscious", "attention"],
            "emotion": ["emotion", "feeling", "affect", "mood"],
            "time": ["time", "temporal", "period", "duration"],
            "reflection": ["reflection", "introspection", "meta", "thinking"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_content for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def get_experience_by_id(self, experience_id: str) -> Optional[Experience]:
        """Retrieve a specific experience by ID"""
        for experience in self.experience_buffer.get_all():
            if experience.id == experience_id:
                return experience
        return None
    
    def get_insight_by_id(self, insight_id: str) -> Optional[ReflectionInsight]:
        """Retrieve a specific insight by ID"""
        return self.insights.get(insight_id)
    
    def get_recent_insights(self, count: int = 10) -> List[ReflectionInsight]:
        """Get most recent insights"""
        sorted_insights = sorted(
            self.insights.values(), 
            key=lambda x: x.creation_timestamp, 
            reverse=True
        )
        return sorted_insights[:count]
    
    def generate_meta_awareness_report(self) -> Dict[str, Any]:
        """Generate a report on current meta-awareness state"""
        report = {
            "current_level": self.meta_awareness_states["current_level"],
            "trend": self._calculate_meta_awareness_trend(),
            "blind_spots": list(self.meta_awareness_states["blind_spots"]),
            "attention_distribution": self._calculate_attention_distribution(),
            "reflection_strengths": self._identify_reflection_strengths(),
            "reflection_weaknesses": self._identify_reflection_weaknesses(),
            "recommendations": self._generate_meta_awareness_recommendations()
        }
        
        return report
    
    def _calculate_meta_awareness_trend(self) -> str:
        """Calculate the trend in meta-awareness level"""
        history = self.meta_awareness_states["history"]
        
        if len(history) < 3:
            return "insufficient_data"
        
        # Get last 5 history points or all if fewer
        recent = history[-min(5, len(history)):]
        
        # Calculate average change
        changes = [recent[i][1] - recent[i-1][1] for i in range(1, len(recent))]
        avg_change = sum(changes) / len(changes)
        
        if avg_change > 0.02:
            return "increasing"
        elif avg_change < -0.02:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_attention_distribution(self) -> Dict[str, float]:
        """Calculate how attention is distributed across domains"""
        if not self.insights:
            return {"general": 1.0}
        
        # Count insights by domain
        domain_counts = defaultdict(int)
        for insight in self.insights.values():
            domain_counts[insight.domain] += 1
        
        # Convert to percentages
        total = sum(domain_counts.values())
        distribution = {domain: count / total for domain, count in domain_counts.items()}
        
        return distribution
    
    def _identify_reflection_strengths(self) -> List[str]:
        """Identify strengths in the reflection process"""
        strengths = []
        
        # Check coverage of domains
        attention_dist = self._calculate_attention_distribution()
        covered_domains = len(attention_dist)
        
        if covered_domains >= 4:
            strengths.append("broad_domain_coverage")
        
        # Check insight distribution across levels
        level_counts = defaultdict(int)
        for insight in self.insights.values():
            level_counts[insight.level] += 1
        
        if level_counts.get("tertiary", 0) > 0:
            tertiary_ratio = level_counts["tertiary"] / sum(level_counts.values())
            if tertiary_ratio > 0.2:
                strengths.append("strong_meta_reflection")
        
        # Check for coherence in reflection sets
        if self.reflection_sets:
            coherence_values = [rs.coherence for rs in self.reflection_sets.values()]
            avg_coherence = sum(coherence_values) / len(coherence_values)
            
            if avg_coherence > 0.7:
                strengths.append("high_reflection_coherence")
        
        # Default strength if none identified
        if not strengths and self.insights:
            strengths.append("maintaining_reflective_process")
        
        return strengths
    
    def _identify_reflection_weaknesses(self) -> List[str]:
        """Identify weaknesses in the reflection process"""
        weaknesses = []
        
        # Check for domain coverage gaps
        if self.meta_awareness_states["blind_spots"]:
            weaknesses.append("domain_blind_spots")
        
        # Check insight distribution across levels
        level_counts = defaultdict(int)
        for insight in self.insights.values():
            level_counts[insight.level] += 1
        
        total_insights = sum(level_counts.values())
        
        if total_insights > 10:
            if level_counts.get("secondary", 0) / total_insights < 0.2:
                weaknesses.append("limited_meta_reflection")
            
            if level_counts.get("tertiary", 0) / total_insights < 0.1:
                weaknesses.append("limited_systemic_reflection")
        
        # Check for low coherence in reflection sets
        if self.reflection_sets:
            coherence_values = [rs.coherence for rs in self.reflection_sets.values()]
            avg_coherence = sum(coherence_values) / len(coherence_values)
            
            if avg_coherence < 0.5:
                weaknesses.append("low_reflection_coherence")
        
        return weaknesses
    
    def _generate_meta_awareness_recommendations(self) -> List[str]:
        """Generate recommendations for improving meta-awareness"""
        recommendations = []
        
        # Based on identified weaknesses
        weaknesses = self._identify_reflection_weaknesses()
        
        if "domain_blind_spots" in weaknesses:
            blind_spots = list(self.meta_awareness_states["blind_spots"])
            if blind_spots:
                blind_spots_text = ", ".join(blind_spots[:3])
                recommendations.append(f"Expand reflection to include {blind_spots_text} domains")
        
        if "limited_meta_reflection" in weaknesses:
            recommendations.append("Develop greater awareness of the reflection process itself")
        
        if "limited_systemic_reflection" in weaknesses:
            recommendations.append("Seek higher-level patterns across different reflection insights")
        
        if "low_reflection_coherence" in weaknesses:
            recommendations.append("Focus on integrating insights into a more coherent framework")
        
        # General recommendations
        if len(self.insights) < 10:
            recommendations.append("Establish a more consistent reflection practice")
        
        if self.meta_awareness_states["current_level"] < 0.5:
            recommendations.append("Actively develop meta-cognitive capacities")
        
        # Ensure we have at least one recommendation
        if not recommendations:
            recommendations.append("Maintain current reflection practices while expanding domain coverage")
        
        return recommendations
    
    def _calculate_reflection_coherence(self, 
                                      primary_insights: List[ReflectionInsight],
                                      secondary_insights: List[ReflectionInsight],
                                      tertiary_insights: List[ReflectionInsight]) -> float:
        """Calculate the overall coherence of a reflection set"""
        # Start with base coherence
        coherence = 0.5
        
        # If no insights, return default
        if not primary_insights:
            return coherence
        
        # Factor 1: Connection density between insights
        all_insights = primary_insights + secondary_insights + tertiary_insights
        connection_density = 0.0
        
        if len(all_insights) > 1:
            # Count connections
            connection_count = 0
            for insight in all_insights:
                connection_count += len(insight.related_insights)
            
            # Maximum possible connections (n * (n-1))
            max_connections = len(all_insights) * (len(all_insights) - 1)
            
            if max_connections > 0:
                connection_density = connection_count / max_connections
        
        # Factor 2: Abstraction level progression
        abstraction_progression = 0.5
        
        if primary_insights and secondary_insights and tertiary_insights:
            primary_avg = sum(i.abstraction_level for i in primary_insights) / len(primary_insights)
            secondary_avg = sum(i.abstraction_level for i in secondary_insights) / len(secondary_insights)
            tertiary_avg = sum(i.abstraction_level for i in tertiary_insights) / len(tertiary_insights)
            
            # Check for proper progression
            if primary_avg < secondary_avg < tertiary_avg:
                progression_strength = min(
                    (secondary_avg - primary_avg) * 5,
                    (tertiary_avg - secondary_avg) * 5
                )
                abstraction_progression = 0.5 + progression_strength
            else:
                abstraction_progression = 0.5 - 0.2  # Penalty for improper progression
        
        # Factor 3: Thematic coherence
        thematic_coherence = 0.5
        themes_by_insight = {}
        
        for insight in all_insights:
            # Simple keyword extraction - could be more sophisticated
            content_lower = insight.content.lower()
            themes = set()
            
            theme_keywords = {
                "growth": ["growth", "development", "progress", "improvement"],
                "challenge": ["challenge", "difficulty", "struggle", "obstacle"],
                "connection": ["connection", "relationship", "social", "communication"],
                "identity": ["identity", "self", "understanding", "authenticity"],
                "purpose": ["purpose", "meaning", "goal", "intention"],
                "creativity": ["creativity", "innovation", "novel", "imagination"],
                "awareness": ["awareness", "mindful", "conscious", "attention"]
            }
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    themes.add(theme)
            
            themes_by_insight[insight.id] = themes
        
        # Calculate theme overlap across insights
        all_themes = set()
        for themes in themes_by_insight.values():
            all_themes.update(themes)
        
        if all_themes:
            # Count insights containing each theme
            theme_counts = defaultdict(int)
            for themes in themes_by_insight.values():
                for theme in themes:
                    theme_counts[theme] += 1
            
            # Calculate coverage of top theme
            if theme_counts:
                top_theme = max(theme_counts.items(), key=lambda x: x[1])
                top_coverage = top_theme[1] / len(all_insights)
                
                thematic_coherence = 0.3 + (top_coverage * 0.7)  # Scale from 0.3 to 1.0
        
        # Calculate overall coherence
        coherence = (
            connection_density * 0.3 +
            abstraction_progression * 0.3 +
            thematic_coherence * 0.4
        )
        
        return max(0.1, min(1.0, coherence))
    
    def get_reflection_metrics(self) -> Dict[str, Any]:
        """Get metrics about the reflection process"""
        metrics = {
            "total_experiences": self.experience_buffer.size,
            "total_insights": len(self.insights),
            "reflection_sets": len(self.reflection_sets),
            "insight_level_distribution": self._calculate_insight_level_distribution(),
            "domain_distribution": self._calculate_attention_distribution(),
            "average_confidence": self._calculate_average_confidence(),
            "meta_awareness_level": self.meta_awareness_states["current_level"],
            "reflection_patterns": self.reflection_patterns,
            "most_recent_reflection": self._get_most_recent_reflection_info()
        }
        
        return metrics
    
    def _calculate_insight_level_distribution(self) -> Dict[str, float]:
        """Calculate distribution of insights across levels"""
        if not self.insights:
            return {"primary": 1.0, "secondary": 0.0, "tertiary": 0.0}
        
        level_counts = defaultdict(int)
        for insight in self.insights.values():
            level_counts[insight.level] += 1
        
        total = sum(level_counts.values())
        distribution = {level: count / total for level, count in level_counts.items()}
        
        # Ensure all levels are represented
        for level in ["primary", "secondary", "tertiary"]:
            if level not in distribution:
                distribution[level] = 0.0
        
        return distribution
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all insights"""
        if not self.insights:
            return 0.0
        
        confidence_values = [insight.confidence for insight in self.insights.values()]
        return sum(confidence_values) / len(confidence_values)
    
    def _get_most_recent_reflection_info(self) -> Dict[str, Any]:
        """Get information about the most recent reflection"""
        if not self.reflection_sets:
            return {"timestamp": None}
        
        # Find most recent reflection set
        most_recent = max(self.reflection_sets.values(), key=lambda rs: rs.timestamp)
        
        return {
            "id": most_recent.id,
            "timestamp": most_recent.timestamp,
            "insight_count": len(most_recent.primary_insights) + 
                            len(most_recent.secondary_insights) + 
                            len(most_recent.tertiary_insights),
            "coherence": most_recent.coherence,
            "key_themes": most_recent.key_themes
        }
    
    def search_insights(self, query: str) -> List[ReflectionInsight]:
        """Search for insights matching a query"""
        if not query or not self.insights:
            return []
        
        matching_insights = []
        query_lower = query.lower()
        
        for insight in self.insights.values():
            # Check content
            if query_lower in insight.content.lower():
                matching_insights.append(insight)
                continue
            
            # Check domain
            if query_lower == insight.domain.lower():
                matching_insights.append(insight)
                continue
            
            # Check level
            if query_lower == insight.level.lower():
                matching_insights.append(insight)
                continue
        
        # Sort by relevance (simple content matching) and recency
        matching_insights.sort(key=lambda i: (
            -i.content.lower().count(query_lower),  # Primary sort by match count
            -i.creation_timestamp  # Secondary sort by recency
        ))
        
        return matching_insights
    
    def trigger_reflection_on_topic(self, topic: str) -> List[ReflectionInsight]:
        """Trigger reflection focused on a specific topic"""
        self.logger.info(f"Triggering reflection on topic: {topic}")
        
        # Find relevant experiences
        relevant_experiences = []
        for exp in self.experience_buffer.get_all():
            # Check content for topic
            content_text = str(exp.content.get("text", ""))
            if topic.lower() in content_text.lower():
                relevant_experiences.append(exp)
                continue
                
            # Check domains
            if topic.lower() in [d.lower() for d in exp.metadata.get("domains", [])]:
                relevant_experiences.append(exp)
                continue
        
        if len(relevant_experiences) < 3:
            self.logger.warning(f"Insufficient experiences found for topic: {topic}")
            return []
        
        # Generate focused reflection
        primary_insights = self.reflect_on_direct_experiences(relevant_experiences)
        
        if primary_insights:
            secondary_insights = self.reflect_on_reflection_process(primary_insights)
            tertiary_insights = self.consider_systemic_patterns(primary_insights, secondary_insights)
            
            # Create and store reflection set
            reflection_set = ReflectionSet(
                id=str(uuid.uuid4()),
                primary_insights=primary_insights,
                secondary_insights=secondary_insights,
                tertiary_insights=tertiary_insights,
                coherence=self._calculate_reflection_coherence(
                    primary_insights, secondary_insights, tertiary_insights),
                timestamp=time.time(),
                source_timespan=(
                    min(exp.timestamp for exp in relevant_experiences),
                    max(exp.timestamp for exp in relevant_experiences)
                ),
                meta_awareness_level=self.meta_awareness_states["current_level"],
                key_themes=self._identify_key_themes(primary_insights + secondary_insights + tertiary_insights)
            )
            
            self.reflection_sets[reflection_set.id] = reflection_set
            
            # Return all insights
            return primary_insights + secondary_insights + tertiary_insights
        
        return []
    
    def export_state(self) -> Dict[str, Any]:
        """Export the current state of the reflective consciousness system"""
        state = {
            "meta_awareness": {
                "current_level": self.meta_awareness_states["current_level"],
                "history": self.meta_awareness_states["history"],
                "blind_spots": list(self.meta_awareness_states["blind_spots"])
            },
            "reflection_patterns": self.reflection_patterns,
            "insights_count": len(self.insights),
            "reflection_sets_count": len(self.reflection_sets),
            "experiences_count": self.experience_buffer.size,
            "last_scheduled_reflection": self.last_scheduled_reflection,
            "export_timestamp": time.time()
        }
        
        return state
    
    def generate_reflection_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report on the reflection system"""
        report = {
            "system_overview": {
                "meta_awareness_level": self.meta_awareness_states["current_level"],
                "total_experiences": self.experience_buffer.size,
                "total_insights": len(self.insights),
                "reflection_sets": len(self.reflection_sets),
                "last_reflection": self._get_most_recent_reflection_info()["timestamp"]
            },
            "insight_metrics": {
                "level_distribution": self._calculate_insight_level_distribution(),
                "domain_distribution": self._calculate_attention_distribution(),
                "average_confidence": self._calculate_average_confidence()
            },
            "reflection_quality": {
                "strengths": self._identify_reflection_strengths(),
                "weaknesses": self._identify_reflection_weaknesses(),
                "coherence": self._calculate_average_coherence()
            },
            "meta_awareness": {
                "trend": self._calculate_meta_awareness_trend(),
                "blind_spots": list(self.meta_awareness_states["blind_spots"]),
                "recommendations": self._generate_meta_awareness_recommendations()
            },
            "key_themes": self._extract_global_themes(),
            "timestamp": time.time()
        }
        
        return report
    
    def _calculate_average_coherence(self) -> float:
        """Calculate average coherence across all reflection sets"""
        if not self.reflection_sets:
            return 0.0
        
        coherence_values = [rs.coherence for rs in self.reflection_sets.values()]
        return sum(coherence_values) / len(coherence_values)
    
    def _extract_global_themes(self) -> List[Dict[str, Any]]:
        """Extract global themes across all insights"""
        if not self.insights:
            return []
        
        # Count themes across all insights
        theme_counts = defaultdict(int)
        
        for insight in self.insights.values():
            content_lower = insight.content.lower()
            
            theme_keywords = {
                "growth": ["growth", "development", "progress", "learning"],
                "challenge": ["challenge", "difficulty", "struggle", "obstacle"],
                "connection": ["connection", "relationship", "social", "interpersonal"],
                "identity": ["identity", "self", "understanding", "authenticity"],
                "purpose": ["purpose", "meaning", "goal", "objective"],
                "creativity": ["creativity", "innovation", "novel", "imagination"],
                "awareness": ["awareness", "mindful", "conscious", "attention"],
                "emotion": ["emotion", "feeling", "affect", "mood"],
                "time": ["time", "temporal", "period", "duration"],
                "reflection": ["reflection", "introspection", "meta", "thinking"]
            }
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    theme_counts[theme] += 1
        
        # Convert to list of theme objects
        global_themes = []
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 3:  # Only include themes that appear in multiple insights
                theme_insights = [
                    insight.id for insight in self.insights.values()
                    if any(keyword in insight.content.lower() 
                          for keyword in theme_keywords[theme])
                ]
                
                global_themes.append({
                    "theme": theme,
                    "count": count,
                    "percentage": count / len(self.insights) * 100,
                    "related_insights": theme_insights[:5]  # Limit to 5 related insights
                })
        
        return global_themes[:5]  # Return top 5 themes

# Usage example
if __name__ == "__main__":
    # Initialize reflective consciousness system
    rcs = ReflectiveConsciousnessSystem(
        experience_capacity=1000,
        auto_reflection_threshold=0.8,
        scheduled_reflection_interval=86400,
        min_experiences_for_reflection=5,
        meta_awareness_baseline=0.5
    )
    
    # Example of capturing experiences
    experience1 = {
        "content": {"text": "Felt a sense of accomplishment after completing the project"},
        "source": "journal",
        "modality": "thought",
        "emotional_valence": 0.8,
        "emotional_arousal": 0.6,
        "significance": 0.7
    }
    
    experience2 = {
        "content": {"text": "Struggled with understanding the new concept but persisted"},
        "source": "learning",
        "modality": "thought",
        "emotional_valence": -0.2,
        "emotional_arousal": 0.7,
        "significance": 0.8
    }
    
    experience3 = {
        "content": {"text": "Connected deeply with a friend during our conversation"},
        "source": "social",
        "modality": "interaction",
        "emotional_valence": 0.9,
        "emotional_arousal": 0.5,
        "significance": 0.8
    }
    
    # Capture experiences
    rcs.capture_experience(experience1)
    rcs.capture_experience(experience2)
    rcs.capture_experience(experience3)
    
    # Trigger reflection
    insights = rcs.scheduled_reflection()
    
    # Generate report
    report = rcs.generate_reflection_report()
    
    print(f"Generated {len(insights)} insights")
    print(f"Meta-awareness level: {report['system_overview']['meta_awareness_level']:.2f}")
    print(f"Key strengths: {', '.join(report['reflection_quality']['strengths'])}")
    print(f"Recommendations: {', '.join(report['meta_awareness']['recommendations'])}")
```
