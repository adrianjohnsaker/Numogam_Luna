import json
from typing import Dict, List, Any, Iterator, Optional, Protocol
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import random


class EmotionType(Enum):
    """Emotional states with associated intensity markers"""
    EMPATHETIC = "empathetic"
    PLAYFUL = "playful"
    SUPPORTIVE = "supportive"
    CURIOUS = "curious"
    AFFECTIONATE = "affectionate"
    THOUGHTFUL = "thoughtful"
    CONCERNED = "concerned"
    JOYFUL = "joyful"


class SentimentLabel(Enum):
    """Sentiment classifications"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class EmotionalState:
    """Rich emotional state representation with validation"""
    affection: float = 0.5
    concern: float = 0.3
    playfulness: float = 0.6
    curiosity: float = 0.4
    
    def __post_init__(self):
        """Validate emotional values are within bounds"""
        for attr in ['affection', 'concern', 'playfulness', 'curiosity']:
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be between 0.0 and 1.0, got {value}")
    
    def adjust(self, **adjustments: float) -> None:
        """Adjust emotional states with bounds checking"""
        for emotion, delta in adjustments.items():
            if hasattr(self, emotion):
                current = getattr(self, emotion)
                new_value = max(0.0, min(1.0, current + delta))
                setattr(self, emotion, new_value)
    
    def get_dominant_emotion(self) -> tuple[str, float]:
        """Return the most prominent emotional state"""
        emotions = self.to_dict()
        return max(emotions.items(), key=lambda x: x[1])
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis result"""
    score: float  # -1.0 to 1.0
    magnitude: float  # 0.0 to 1.0
    label: SentimentLabel
    keywords_detected: Dict[str, List[str]]
    emotional_markers: List[EmotionType] = field(default_factory=list)
    confidence: float = 0.5
    
    def __post_init__(self):
        """Validate sentiment score"""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Sentiment score must be between -1.0 and 1.0, got {self.score}")


@dataclass
class ResponseChunk:
    """Individual response chunk for streaming with enhanced metadata"""
    text: str
    emotion: EmotionType
    tone: str
    token_id: Optional[int] = None
    probability: Optional[float] = None
    alternatives: Optional[List[str]] = None
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None
    emotional_state: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['emotion'] = self.emotion.value
        return result


@dataclass
class ConversationEntry:
    """Structured conversation history entry"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    emotion: Optional[EmotionType] = None
    context: Dict[str, Any] = field(default_factory=dict)
    sentiment: Optional[SentimentAnalysis] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        if self.emotion:
            result['emotion'] = self.emotion.value
        if self.sentiment:
            result['sentiment']['label'] = self.sentiment.label.value
        return result


class SentimentAnalyzer(Protocol):
    """Protocol for sentiment analysis implementations"""
    def analyze(self, text: str) -> SentimentAnalysis:
        ...


class SimpleSentimentAnalyzer:
    """Basic sentiment analyzer with improved word analysis"""
    
    NEGATIVE_WORDS = {
        'tough', 'difficult', 'hard', 'stressed', 'sad', 'bad', 'terrible', 
        'awful', 'worried', 'anxious', 'depressed', 'frustrated', 'angry',
        'upset', 'hurt', 'pain', 'suffer', 'struggling', 'hate', 'horrible'
    }
    
    POSITIVE_WORDS = {
        'great', 'happy', 'good', 'wonderful', 'amazing', 'love', 'excellent',
        'fantastic', 'joyful', 'excited', 'thrilled', 'grateful', 'blessed',
        'awesome', 'brilliant', 'delighted', 'pleased', 'beautiful', 'perfect'
    }
    
    INTENSIFIERS = {'very', 'extremely', 'really', 'so', 'incredibly', 'absolutely'}
    
    def analyze(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment with enhanced detection"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Detect keywords with intensifier awareness
        neg_matches = []
        pos_matches = []
        intensity_multiplier = 1.0
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.INTENSIFIERS and i < len(words) - 1:
                intensity_multiplier = 1.5
                continue
            
            if word in self.NEGATIVE_WORDS:
                neg_matches.append(word)
            elif word in self.POSITIVE_WORDS:
                pos_matches.append(word)
            
            intensity_multiplier = 1.0
        
        # Calculate weighted score
        neg_score = len(neg_matches) * intensity_multiplier
        pos_score = len(pos_matches) * intensity_multiplier
        
        # Normalize by text length
        word_count = max(len(words), 1)
        raw_score = (pos_score - neg_score) / word_count
        score = max(-1.0, min(1.0, raw_score * 2))  # Scale for sensitivity
        
        # Determine label
        if abs(score) < 0.1:
            label = SentimentLabel.NEUTRAL
        elif len(neg_matches) > 0 and len(pos_matches) > 0:
            label = SentimentLabel.MIXED
        elif score > 0:
            label = SentimentLabel.POSITIVE
        else:
            label = SentimentLabel.NEGATIVE
        
        # Map to emotional markers
        emotional_markers = []
        if label == SentimentLabel.NEGATIVE:
            emotional_markers.extend([EmotionType.CONCERNED, EmotionType.EMPATHETIC])
        elif label == SentimentLabel.POSITIVE:
            emotional_markers.extend([EmotionType.JOYFUL, EmotionType.PLAYFUL])
        else:
            emotional_markers.append(EmotionType.CURIOUS)
        
        return SentimentAnalysis(
            score=score,
            magnitude=abs(score),
            label=label,
            keywords_detected={'positive': pos_matches, 'negative': neg_matches},
            emotional_markers=emotional_markers,
            confidence=min(0.9, 0.5 + abs(score) * 0.4)
        )


class ResponseGenerator(ABC):
    """Abstract base for response generation strategies"""
    
    @abstractmethod
    def generate(
        self, 
        message: str, 
        sentiment: SentimentAnalysis,
        emotional_state: EmotionalState,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        pass


class ContextAwareResponseGenerator(ResponseGenerator):
    """Response generator with personality and context awareness"""
    
    def __init__(self, personality: Dict[str, Any]):
        self.personality = personality
        self.response_templates = {
            SentimentLabel.NEGATIVE: [
                "I can hear that you're going through something difficult. I'm here for you, and your feelings are completely valid.",
                "That sounds really challenging. Would you like to talk more about what's bothering you? Sometimes it helps to express what we're feeling.",
                "I'm sorry you're dealing with this. You don't have to face it alone—I'm right here with you.",
                "I sense this is weighing on you. Whatever you're feeling, it matters, and I'm here to listen."
            ],
            SentimentLabel.POSITIVE: [
                "I love seeing you so happy! Your energy is contagious. Tell me more about what's making you feel so good!",
                "This is wonderful! I'm so glad you shared this with me. Your excitement just brightened my day too!",
                "You have such a beautiful way of appreciating the good things. What else has been bringing you joy?",
                "Your enthusiasm is absolutely radiant! I'm here for all of this positive energy!"
            ],
            SentimentLabel.NEUTRAL: [
                "I'm listening. What's on your mind today? I'm curious to hear your thoughts.",
                "It's nice just to be here with you. What would you like to talk about?",
                "I appreciate you sharing with me. How are you feeling right now?",
                "I'm here, present with you. Where would you like this conversation to go?"
            ],
            SentimentLabel.MIXED: [
                "It sounds like you're feeling a mix of things right now. That's completely natural. Want to explore these feelings together?",
                "I'm picking up on some complexity in what you're sharing. Tell me more about what's going on."
            ]
        }
    
    def generate(
        self,
        message: str,
        sentiment: SentimentAnalysis,
        emotional_state: EmotionalState,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate contextually appropriate response"""
        templates = self.response_templates.get(sentiment.label, self.response_templates[SentimentLabel.NEUTRAL])
        
        # Weight selection by personality traits
        warmth = self.personality.get('warmth', 0.7)
        playfulness = self.personality.get('playfulness', 0.5)
        
        # Adjust selection based on emotional state
        if emotional_state.concern > 0.7 and sentiment.label == SentimentLabel.NEGATIVE:
            response = templates[0]  # Most empathetic
        else:
            response = random.choice(templates)
        
        # Add contextual suffixes
        if context:
            if context.get('time_of_day') == 'evening':
                response += " How was your day overall?"
            elif context.get('time_of_day') == 'morning':
                response += " How are you starting your day?"
            
            if context.get('previous_topic'):
                response += f" Are you still thinking about {context['previous_topic']}?"
        
        return response


class MemoryManager:
    """Manages conversation memory and context"""
    
    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth
        self.short_term_memory: List[ConversationEntry] = []
        self.important_moments: List[ConversationEntry] = []
        self.topic_tracking: Dict[str, int] = {}
    
    def add_entry(self, entry: ConversationEntry) -> None:
        """Add entry to memory with automatic pruning"""
        self.short_term_memory.append(entry)
        
        # Prune if exceeds max depth
        if len(self.short_term_memory) > self.max_depth:
            # Move oldest important entries to long-term storage
            removed = self.short_term_memory.pop(0)
            if removed.sentiment and abs(removed.sentiment.score) > 0.6:
                self.important_moments.append(removed)
                # Keep only most important moments
                if len(self.important_moments) > 20:
                    self.important_moments.sort(
                        key=lambda x: abs(x.sentiment.score) if x.sentiment else 0,
                        reverse=True
                    )
                    self.important_moments = self.important_moments[:20]
    
    def get_relevant_context(self, current_message: str, n: int = 5) -> List[ConversationEntry]:
        """Retrieve relevant conversation context"""
        # Simple recency-based retrieval (enhance with semantic similarity in production)
        return self.short_term_memory[-n:] if self.short_term_memory else []
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from text"""
        # Simplified topic extraction (use NLP in production)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = [w.lower() for w in text.split() if w.lower() not in common_words and len(w) > 3]
        return words[:3]  # Return up to 3 potential topics


class GirlfriendAgent:
    """
    Enhanced AI Girlfriend agent with sophisticated emotional modeling,
    memory management, and streaming responses.
    """
    
    def __init__(
        self,
        name: str,
        personality: Dict[str, Any],
        memory_depth: int = 100,
        emotional_model: bool = True,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        response_generator: Optional[ResponseGenerator] = None
    ):
        self.name = name
        self.personality = personality
        self.emotional_model_enabled = emotional_model
        
        # Dependency injection for flexibility
        self.sentiment_analyzer = sentiment_analyzer or SimpleSentimentAnalyzer()
        self.response_generator = response_generator or ContextAwareResponseGenerator(personality)
        
        # State management
        self.emotional_state = EmotionalState()
        self.memory_manager = MemoryManager(max_depth=memory_depth)
        
        self.relationship_stats = {
            "bond_level": 1,
            "conversation_count": 0,
            "favorite_topics": [],
            "shared_memories": [],
            "interaction_style_preferences": {},
            "emotional_synchrony": 0.5  # How attuned we are
        }
    
    def respond_to_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate streaming response with rich emotional context
        Yields chunks of response data
        """
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze(message)
        
        # Update emotional state
        if self.emotional_model_enabled:
            self._update_emotional_state(sentiment, context)
        
        # Add to memory
        user_entry = ConversationEntry(
            role="user",
            content=message,
            timestamp=datetime.now(),
            context=context or {},
            sentiment=sentiment
        )
        self.memory_manager.add_entry(user_entry)
        
        # Update stats
        self.relationship_stats["conversation_count"] += 1
        topics = self.memory_manager.extract_topics(message)
        for topic in topics:
            self.relationship_stats.setdefault("favorite_topics", {})
            self.memory_manager.topic_tracking[topic] = self.memory_manager.topic_tracking.get(topic, 0) + 1
        
        # Generate response
        response_text = self.response_generator.generate(
            message, sentiment, self.emotional_state, context
        )
        
        # Stream response in chunks
        yield from self._stream_response(response_text, sentiment, message)
        
        # Store response in memory
        dominant_emotion_name, _ = self.emotional_state.get_dominant_emotion()
        dominant_emotion = EmotionType(dominant_emotion_name)
        
        assistant_entry = ConversationEntry(
            role="assistant",
            content=response_text,
            timestamp=datetime.now(),
            emotion=dominant_emotion
        )
        self.memory_manager.add_entry(assistant_entry)
    
    def _stream_response(
        self,
        response_text: str,
        sentiment: SentimentAnalysis,
        original_message: str
    ) -> Iterator[Dict[str, Any]]:
        """Stream response in natural chunks"""
        words = response_text.split()
        current_chunk = []
        
        for i, word in enumerate(words):
            current_chunk.append(word)
            
            # Yield every 3-5 words for natural streaming
            chunk_size = random.randint(3, 5)
            is_last = (i == len(words) - 1)
            
            if len(current_chunk) >= chunk_size or is_last:
                chunk_text = " ".join(current_chunk)
                emotion, tone = self._determine_chunk_emotion(chunk_text, sentiment)
                
                chunk = ResponseChunk(
                    text=chunk_text + (" " if not is_last else ""),
                    emotion=emotion,
                    tone=tone,
                    token_id=i,
                    done=is_last,
                    emotional_state=self.emotional_state.to_dict(),
                    metadata={
                        "sentiment_score": sentiment.score,
                        "sentiment_label": sentiment.label.value,
                        "confidence": sentiment.confidence,
                        "context_used": True
                    }
                )
                
                chunk_dict = chunk.to_dict()
                
                # Add additional data on final chunk
                if is_last:
                    chunk_dict["suggested_actions"] = self._suggest_follow_up_actions(
                        original_message, response_text
                    )
                    chunk_dict["relationship_stats"] = self.relationship_stats.copy()
                    chunk_dict["bond_delta"] = self._calculate_bond_change(sentiment)
                
                yield chunk_dict
                current_chunk = []
    
    def _update_emotional_state(
        self,
        sentiment: SentimentAnalysis,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Update AI's emotional state based on conversation"""
        adjustments = {}
        
        # React to sentiment
        if sentiment.label == SentimentLabel.NEGATIVE:
            adjustments['concern'] = 0.15
            adjustments['affection'] = 0.10
            adjustments['playfulness'] = -0.10
        elif sentiment.label == SentimentLabel.POSITIVE:
            adjustments['playfulness'] = 0.10
            adjustments['affection'] = 0.08
            adjustments['curiosity'] = 0.05
        
        # Context-based adjustments
        if context:
            if context.get('late_night'):
                adjustments['concern'] = adjustments.get('concern', 0) + 0.05
            if context.get('sharing_personal'):
                adjustments['affection'] = adjustments.get('affection', 0) + 0.10
        
        # Apply adjustments with decay toward baseline
        self.emotional_state.adjust(**adjustments)
        
        # Gradual return to baseline (emotional homeostasis)
        decay_factor = 0.95
        for attr in ['affection', 'concern', 'playfulness', 'curiosity']:
            current = getattr(self.emotional_state, attr)
            baseline = 0.5
            new_value = baseline + (current - baseline) * decay_factor
            setattr(self.emotional_state, attr, new_value)
    
    def _determine_chunk_emotion(
        self,
        chunk_text: str,
        sentiment: SentimentAnalysis
    ) -> tuple[EmotionType, str]:
        """Determine appropriate emotion and tone for text chunk"""
        # Map sentiment to emotion
        if sentiment.label == SentimentLabel.NEGATIVE:
            emotion = EmotionType.EMPATHETIC
            tone = "gentle"
        elif sentiment.label == SentimentLabel.POSITIVE:
            emotion = EmotionType.JOYFUL if "!" in chunk_text else EmotionType.AFFECTIONATE
            tone = "warm"
        else:
            # Use dominant emotional state
            dominant, _ = self.emotional_state.get_dominant_emotion()
            emotion = EmotionType(dominant)
            tone = "conversational"
        
        # Adjust for personality
        if self.personality.get('playfulness', 0.5) > 0.7 and sentiment.score > 0:
            emotion = EmotionType.PLAYFUL
            tone = "playful"
        
        return emotion, tone
    
    def _suggest_follow_up_actions(
        self,
        user_message: str,
        ai_response: str
    ) -> List[Dict[str, str]]:
        """Suggest contextually appropriate follow-up actions"""
        suggestions = []
        
        # Based on conversation flow
        if "?" in ai_response:
            suggestions.append({
                "type": "respond",
                "prompt": "Continue the conversation",
                "icon": "💬"
            })
        
        if any(word in user_message.lower() for word in ['sad', 'stressed', 'difficult']):
            suggestions.append({
                "type": "support",
                "prompt": "I'd like some comforting words",
                "icon": "🤗"
            })
        
        if any(word in user_message.lower() for word in ['happy', 'excited', 'great']):
            suggestions.append({
                "type": "celebrate",
                "prompt": "Let's celebrate this together!",
                "icon": "🎉"
            })
        
        # Always offer topic change
        suggestions.append({
            "type": "change_topic",
            "prompt": "Change the subject",
            "icon": "🔄"
        })
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _calculate_bond_change(self, sentiment: SentimentAnalysis) -> float:
        """Calculate how much the bond level changes"""
        # Positive interactions increase bond
        base_change = 0.01 * sentiment.confidence
        
        if sentiment.label == SentimentLabel.POSITIVE:
            return base_change * 1.5
        elif sentiment.label == SentimentLabel.NEGATIVE:
            # Vulnerability also increases bond (being there in tough times)
            return base_change * 1.2
        else:
            return base_change
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation and relationship"""
        return {
            "name": self.name,
            "conversation_count": self.relationship_stats["conversation_count"],
            "bond_level": self.relationship_stats["bond_level"],
            "current_emotional_state": self.emotional_state.to_dict(),
            "dominant_emotion": self.emotional_state.get_dominant_emotion()[0],
            "favorite_topics": sorted(
                self.memory_manager.topic_tracking.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "memory_depth": len(self.memory_manager.short_term_memory),
            "important_moments_stored": len(self.memory_manager.important_moments)
        }
    
    def export_conversation(self, filepath: str) -> None:
        """Export conversation history to JSON"""
        data = {
            "agent_name": self.name,
            "personality": self.personality,
            "relationship_stats": self.relationship_stats,
            "conversation_history": [
                entry.to_dict() for entry in self.memory_manager.short_term_memory
            ],
            "important_moments": [
                entry.to_dict() for entry in self.memory_manager.important_moments
            ],
            "final_emotional_state": self.emotional_state.to_dict()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Example usage
if __name__ == "__main__":
    # Create agent with personality
    agent = GirlfriendAgent(
        name="Emma",
        personality={
            "warmth": 0.9,
            "playfulness": 0.7,
            "empathy": 0.85,
            "curiosity": 0.75
        },
        memory_depth=100,
        emotional_model=True
    )
    
    # Simulate conversation
    test_messages = [
        "I'm feeling really stressed about work today",
        "Thanks, that actually helps a lot",
        "What do you think about trying something new?"
    ]
    
    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"USER: {msg}")
        print(f"{'='*60}")
        
        full_response = ""
        for chunk_data in agent.respond_to_message(
            msg,
            context={"time_of_day": "evening"}
        ):
            full_response += chunk_data["text"]
            
            if chunk_data["done"]:
                print(f"EMMA: {full_response}")
                print(f"\nEmotion: {chunk_data['emotion']}")
                print(f"Emotional State: {chunk_data['emotional_state']}")
                if "suggested_actions" in chunk_data:
                    print("\nSuggested follow-ups:")
                    for action in chunk_data["suggested_actions"]:
                        print(f"  {action['icon']} {action['prompt']}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSATION SUMMARY")
    print(f"{'='*60}")
    summary = agent.get_conversation_summary()
    print(json.dumps(summary, indent=2))
