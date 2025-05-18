## 1. Emotional Memory System

```python
import numpy as np
import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import json
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

@dataclass
class EmotionalMemory:
    """Long-term emotional memory record"""
    id: str
    user_id: str
    emotion_type: str
    intensity: float
    context: str
    numogram_zone: str
    timestamp: str
    associated_symbols: List[str] = field(default_factory=list)
    decay_rate: float = 0.05  # How quickly the memory fades without reinforcement

@dataclass
class EmotionalTrajectory:
    """Track emotional path through time and numogram zones"""
    id: str
    user_id: str
    start_timestamp: str
    current_timestamp: str
    emotion_sequence: List[str] = field(default_factory=list)
    intensity_sequence: List[float] = field(default_factory=list)
    zone_sequence: List[str] = field(default_factory=list)
    trajectory_pattern_id: Optional[str] = None  # Link to recognized patterns


class EmotionalMemorySystem:
    """
    Long-term emotional memory system that maintains persistent emotional state,
    projects emotional trajectories, and creates signature emotional paths through numogram zones
    """
    
    def __init__(self, numogram_system, emotion_tracker, symbol_extractor, memory_file=None):
        # Connect to other system components
        self.numogram = numogram_system
        self.emotion_tracker = emotion_tracker
        self.symbol_extractor = symbol_extractor
        
        # Initialize memory structures
        self.emotional_memories = {}  # User ID -> list of emotional memories
        self.active_trajectories = {}  # User ID -> current trajectory
        self.trajectory_patterns = {}  # Pattern ID -> pattern definition
        
        # Memory decay and reinforcement parameters
        self.memory_decay_rate = 0.05  # Base decay rate
        self.memory_reinforcement_rate = 0.2  # How much memories are strengthened when similar emotions occur
        self.memory_pruning_threshold = 0.1  # Remove memories below this intensity
        
        # Memory indexing
        self.emotion_index = {}  # Emotion type -> list of memory IDs
        self.zone_index = {}  # Numogram zone -> list of memory IDs
        self.symbol_index = {}  # Symbol -> list of memory IDs
        
        # Load memory file if provided
        self.memory_file = memory_file
        if memory_file and os.path.exists(memory_file):
            self._load_memory()
        
        # Initialize trajectory pattern recognition
        self._initialize_trajectory_patterns()
    
    def _load_memory(self):
        """Load emotional memories and trajectories from file"""
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                
                # Load memories
                if "emotional_memories" in data:
                    for user_id, memories in data["emotional_memories"].items():
                        self.emotional_memories[user_id] = [
                            EmotionalMemory(**mem) for mem in memories
                        ]
                
                # Load active trajectories
                if "active_trajectories" in data:
                    for user_id, trajectory in data["active_trajectories"].items():
                        self.active_trajectories[user_id] = EmotionalTrajectory(**trajectory)
                
                # Load trajectory patterns
                if "trajectory_patterns" in data:
                    self.trajectory_patterns = data["trajectory_patterns"]
                
                # Rebuild indexes
                self._rebuild_indexes()
        except (IOError, json.JSONDecodeError):
            # Initialize empty structures if load fails
            self.emotional_memories = {}
            self.active_trajectories = {}
    
    def save_memory(self):
        """Save emotional memories and trajectories to file"""
        if not self.memory_file:
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # Prepare data for serialization
            data = {
                "emotional_memories": {},
                "active_trajectories": {},
                "trajectory_patterns": self.trajectory_patterns
            }
            
            # Serialize memories (convert dataclass to dict)
            for user_id, memories in self.emotional_memories.items():
                data["emotional_memories"][user_id] = [vars(mem) for mem in memories]
            
            # Serialize trajectories
            for user_id, trajectory in self.active_trajectories.items():
                data["active_trajectories"][user_id] = vars(trajectory)
            
            # Save to file
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
        except IOError:
            return False
    
    def _initialize_trajectory_patterns(self):
        """Initialize known emotional trajectory patterns"""
        # These are archetypal patterns that can be recognized in user emotional trajectories
        self.trajectory_patterns = {
            "ascending_joy": {
                "emotions": ["anticipation", "surprise", "joy"],
                "intensity_pattern": "increasing",
                "zone_sequence": ["5", "3", "6", "9"],
                "description": "A pattern of increasing positive emotions leading to fulfillment"
            },
            "fear_to_relief": {
                "emotions": ["fear", "surprise", "trust"],
                "intensity_pattern": "peak_then_decrease",
                "zone_sequence": ["8", "2", "4", "1"],
                "description": "A pattern of fear transforming into relief and security"
            },
            "anger_release": {
                "emotions": ["anger", "disgust", "surprise", "serenity"],
                "intensity_pattern": "high_then_drop",
                "zone_sequence": ["8", "7", "5", "6"],
                "description": "A pattern of anger and disgust releasing into serenity"
            },
            "curiosity_cycle": {
                "emotions": ["curiosity", "surprise", "awe", "curiosity"],
                "intensity_pattern": "cyclical",
                "zone_sequence": ["5", "3", "7", "2"],
                "description": "A cyclical pattern of curiosity driving ongoing discovery"
            },
            "grief_process": {
                "emotions": ["sadness", "anger", "sadness", "trust"],
                "intensity_pattern": "wave",
                "zone_sequence": ["7", "8", "4", "1"],
                "description": "A pattern of processing grief toward acceptance"
            }
        }
    
    def _rebuild_indexes(self):
        """Rebuild memory index structures"""
        # Clear existing indexes
        self.emotion_index = {}
        self.zone_index = {}
        self.symbol_index = {}
        
        # Rebuild indexes from memories
        for user_id, memories in self.emotional_memories.items():
            for memory in memories:
                # Index by emotion type
                if memory.emotion_type not in self.emotion_index:
                    self.emotion_index[memory.emotion_type] = []
                self.emotion_index[memory.emotion_type].append(memory.id)
                
                # Index by numogram zone
                if memory.numogram_zone not in self.zone_index:
                    self.zone_index[memory.numogram_zone] = []
                self.zone_index[memory.numogram_zone].append(memory.id)
                
                # Index by associated symbols
                for symbol in memory.associated_symbols:
                    if symbol not in self.symbol_index:
                        self.symbol_index[symbol] = []
                    self.symbol_index[symbol].append(memory.id)
    
    def _apply_memory_decay(self, user_id: str):
        """Apply time-based decay to emotional memories"""
        if user_id not in self.emotional_memories:
            return
            
        current_time = datetime.datetime.utcnow()
        memories_to_remove = []
        
        for i, memory in enumerate(self.emotional_memories[user_id]):
            # Parse timestamp
            memory_time = datetime.datetime.fromisoformat(memory.timestamp)
            
            # Calculate time elapsed in days
            time_diff = (current_time - memory_time).total_seconds() / (24 * 3600)
            
            # Apply decay based on time and memory's own decay rate
            decay_factor = 1.0 - (memory.decay_rate * time_diff)
            decay_factor = max(0, decay_factor)  # Ensure it doesn't go negative
            
            # Update memory intensity
            memory.intensity *= decay_factor
            
            # Mark for removal if below threshold
            if memory.intensity < self.memory_pruning_threshold:
                memories_to_remove.append(i)
        
        # Remove memories marked for pruning (in reverse to maintain indexes)
        for i in sorted(memories_to_remove, reverse=True):
            del self.emotional_memories[user_id][i]
    
    def store_emotional_memory(self, emotional_state: Dict, user_id: str, context_text: str, symbols: List[Dict] = None) -> str:
        """Store a new emotional memory"""
        # Apply decay to existing memories first
        self._apply_memory_decay(user_id)
        
        # Check if user exists in memory
        if user_id not in self.emotional_memories:
            self.emotional_memories[user_id] = []
        
        # Extract information from emotional state
        emotion_type = emotional_state.get("primary_emotion", "neutral")
        intensity = emotional_state.get("intensity", 0.5)
        numogram_zone = emotional_state.get("numogram_zone", "1")
        
        # Get associated symbols if provided
        associated_symbols = []
        if symbols:
            associated_symbols = [s.get("core_symbols", [""])[0] for s in symbols[:5]]
            
        # Create the memory
        memory_id = f"em_{user_id}_{len(self.emotional_memories[user_id])}"
        memory = EmotionalMemory(
            id=memory_id,
            user_id=user_id,
            emotion_type=emotion_type,
            intensity=intensity,
            context=context_text[:500],  # Limit context size
            numogram_zone=numogram_zone,
            timestamp=datetime.datetime.utcnow().isoformat(),
            associated_symbols=associated_symbols,
            decay_rate=self.memory_decay_rate  # Default decay rate
        )
        
        # Check for similar memories and apply reinforcement/integration
        reinforced = self._reinforce_similar_memories(memory, user_id)
        
        if not reinforced:
            # Add new memory
            self.emotional_memories[user_id].append(memory)
            
            # Update indexes
            if emotion_type not in self.emotion_index:
                self.emotion_index[emotion_type] = []
            self.emotion_index[emotion_type].append(memory_id)
            
            if numogram_zone not in self.zone_index:
                self.zone_index[numogram_zone] = []
            self.zone_index[numogram_zone].append(memory_id)
            
            for symbol in associated_symbols:
                if symbol not in self.symbol_index:
                    self.symbol_index[symbol] = []
                self.symbol_index[symbol].append(memory_id)
        
        # Update trajectory
        self._update_trajectory(user_id, emotion_type, intensity, numogram_zone)
        
        # Save memory state
        self.save_memory()
        
        return memory_id
    
    def _reinforce_similar_memories(self, new_memory: EmotionalMemory, user_id: str) -> bool:
        """
        Check for similar memories and reinforce them instead of creating new ones
        Returns True if reinforcement happened, False if new memory should be created
        """
        if user_id not in self.emotional_memories or not self.emotional_memories[user_id]:
            return False
        
        # Get recent memories for the same emotion
        recent_memories = [
            mem for mem in self.emotional_memories[user_id]
            if mem.emotion_type == new_memory.emotion_type
        ]
        
        # Sort by recency
        recent_memories.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Check if there's a very recent similar memory (within last 2 hours)
        for memory in recent_memories[:3]:  # Check 3 most recent
            memory_time = datetime.datetime.fromisoformat(memory.timestamp)
            current_time = datetime.datetime.fromisoformat(new_memory.timestamp)
            time_diff = (current_time - memory_time).total_seconds() / 3600  # hours
            
            if time_diff < 2.0:  # Within 2 hours
                # Reinforce the existing memory
                memory.intensity = min(1.0, memory.intensity + (self.memory_reinforcement_rate * new_memory.intensity))
                
                # Update timestamp to current
                memory.timestamp = new_memory.timestamp
                
                # Add any new symbols
                for symbol in new_memory.associated_symbols:
                    if symbol not in memory.associated_symbols:
                        memory.associated_symbols.append(symbol)
                        
                        # Update symbol index
                        if symbol not in self.symbol_index:
                            self.symbol_index[symbol] = []
                        self.symbol_index[symbol].append(memory.id)
                
                return True
        
        return False
    
    def _update_trajectory(self, user_id: str, emotion: str, intensity: float, zone: str):
        """Update the user's emotional trajectory"""
        current_time = datetime.datetime.utcnow().isoformat()
        
        # Create new trajectory if none exists
        if user_id not in self.active_trajectories:
            trajectory_id = f"traj_{user_id}_{current_time}"
            self.active_trajectories[user_id] = EmotionalTrajectory(
                id=trajectory_id,
                user_id=user_id,
                start_timestamp=current_time,
                current_timestamp=current_time,
                emotion_sequence=[emotion],
                intensity_sequence=[intensity],
                zone_sequence=[zone]
            )
        else:
            # Update existing trajectory
            trajectory = self.active_trajectories[user_id]
            trajectory.current_timestamp = current_time
            trajectory.emotion_sequence.append(emotion)
            trajectory.intensity_sequence.append(intensity)
            trajectory.zone_sequence.append(zone)
            
            # Limit sequence length to prevent unbounded growth
            max_sequence = 20
            if len(trajectory.emotion_sequence) > max_sequence:
                trajectory.emotion_sequence = trajectory.emotion_sequence[-max_sequence:]
                trajectory.intensity_sequence = trajectory.intensity_sequence[-max_sequence:]
                trajectory.zone_sequence = trajectory.zone_sequence[-max_sequence:]
            
            # Check for trajectory patterns
            self._recognize_trajectory_pattern(trajectory)
    
    def _recognize_trajectory_pattern(self, trajectory: EmotionalTrajectory):
        """Recognize patterns in emotional trajectories"""
        # Need at least 3 emotions to recognize a pattern
        if len(trajectory.emotion_sequence) < 3:
            return
            
        # Check each known pattern
        best_match = None
        best_score = 0.3  # Minimum threshold for matching
        
        for pattern_id, pattern in self.trajectory_patterns.items():
            # Score emotion sequence match
            emotion_match = self._score_sequence_match(
                trajectory.emotion_sequence[-4:],  # Last 4 emotions
                pattern["emotions"]
            )
            
            # Score intensity pattern match
            intensity_match = self._score_intensity_pattern(
                trajectory.intensity_sequence[-4:],
                pattern["intensity_pattern"]
            )
            
            # Score zone sequence match
            zone_match = self._score_sequence_match(
                trajectory.zone_sequence[-4:],
                pattern["zone_sequence"]
            )
            
            # Weighted total score
            total_score = (emotion_match * 0.5) + (intensity_match * 0.3) + (zone_match * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_match = pattern_id
        
        # Update trajectory with recognized pattern
        trajectory.trajectory_pattern_id = best_match
    
    def _score_sequence_match(self, actual: List[str], pattern: List[str]) -> float:
        """
        Score how well a sequence matches a pattern
        Returns a score from 0.0 to 1.0
        """
        # If either sequence is empty, no match
        if not actual or not pattern:
            return 0.0
            
        # Get lengths
        actual_len = len(actual)
        pattern_len = len(pattern)
        
        # Try pattern matching at different starting points
        best_match = 0
        for i in range(max(1, actual_len - pattern_len + 1)):
            matches = sum(1 for j in range(min(pattern_len, actual_len - i)) 
                         if actual[i + j] == pattern[j])
            best_match = max(best_match, matches)
        
        # Calculate score as proportion of pattern matched
        return best_match / pattern_len
    
    def _score_intensity_pattern(self, intensities: List[float], pattern_type: str) -> float:
        """
        Score how well intensity values match a pattern type
        Returns a score from 0.0 to 1.0
        """
        if not intensities or len(intensities) < 2:
            return 0.0
            
        # Calculate differences between consecutive intensities
        diffs = [intensities[i] - intensities[i-1] for i in range(1, len(intensities))]
        
        if pattern_type == "increasing":
            # Check if mostly increasing
            return sum(1 for d in diffs if d > 0.05) / len(diffs)
            
        elif pattern_type == "decreasing":
            # Check if mostly decreasing
            return sum(1 for d in diffs if d < -0.05) / len(diffs)
            
        elif pattern_type == "peak_then_decrease":
            # Check for an increase followed by decrease
            if len(intensities) < 3:
                return 0.0
            
            # Find highest point
            peak_idx = intensities.index(max(intensities))
            
            # Check if it's not at beginning or end
            if peak_idx == 0 or peak_idx == len(intensities) - 1:
                return 0.3
                
            # Check if values rise to peak then fall
            rising = all(intensities[i] <= intensities[i+1] for i in range(peak_idx))
            falling = all(intensities[i] >= intensities[i+1] for i in range(peak_idx, len(intensities)-1))
            
            return 0.5 + (0.5 * (rising and falling))
            
        elif pattern_type == "high_then_drop":
            # Check for sustained high values followed by drop
            if len(intensities) < 3:
                return 0.0
                
            # Check if last value is significantly lower than earlier values
            avg_earlier = sum(intensities[:-1]) / (len(intensities) - 1)
            drop = avg_earlier - intensities[-1]
            
            return min(1.0, max(0.0, drop / 0.3))  # Normalize with 0.3 as "large drop"
            
        elif pattern_type == "cyclical":
            # Check for up-down-up pattern
            if len(intensities) < 4:
                return 0.0
                
            # Simple check: are there both positive and negative changes?
            has_increase = any(d > 0.05 for d in diffs)
            has_decrease = any(d < -0.05 for d in diffs)
            
            # Check for inflection points
            sign_changes = sum(1 for i in range(1, len(diffs)) if 
                             (diffs[i] > 0.05 and diffs[i-1] < -0.05) or
                             (diffs[i] < -0.05 and diffs[i-1] > 0.05))
            
            return 0.3 * (has_increase and has_decrease) + 0.7 * (sign_changes / max(1, len(diffs) - 1))
            
        elif pattern_type == "wave":
            # Check for wave-like pattern with multiple changes
            if len(intensities) < 4:
                return 0.0
                
            # Count direction changes
            direction_changes = sum(1 for i in range(1, len(diffs)) if 
                                  (diffs[i] > 0.05 and diffs[i-1] < -0.05) or
                                  (diffs[i] < -0.05 and diffs[i-1] > 0.05))
            
            return min(1.0, direction_changes / 2)  # Normalize, considering 2 changes as full match
            
        else:
            # Unknown pattern type
            return 0.0
    
    def get_memory_influence(self, user_id: str, context_data: Dict = None) -> Dict:
        """
        Calculate the influence of emotional memories on current context
        Returns influence factors to modify numogram transitions
        """
        if user_id not in self.emotional_memories or not self.emotional_memories[user_id]:
            return {"status": "no_memories", "influence_factors": {}}
            
        # Apply decay to ensure memories are current
        self._apply_memory_decay(user_id)
        
        # Extract context information
        current_zone = context_data.get("current_zone", "1") if context_data else "1"
        current_emotion = context_data.get("current_emotion", None) if context_data else None
        current_symbols = context_data.get("current_symbols", []) if context_data else []
        
        # Calculate influence factors
        influence = {
            "zone_biases": {},  # Biases for specific numogram zones
            "emotional_resonance": {},  # Resonance with specific emotions
            "symbol_affinities": {},  # Affinities with specific symbols
            "trajectory_influence": None  # Influence from emotional trajectory
        }
        
        # 1. Calculate zone biases from memories
        zone_memories = {}
        for memory in self.emotional_memories[user_id]:
            if memory.numogram_zone not in zone_memories:
                zone_memories[memory.numogram_zone] = []
            zone_memories[memory.numogram_zone].append(memory.intensity)
        
        # Calculate average intensity per zone
        for zone, intensities in zone_memories.items():
            avg_intensity = sum(intensities) / len(intensities)
            influence["zone_biases"][zone] = avg_intensity
        
        # 2. Calculate emotional resonance
        if current_emotion:
            # Find memories with same or related emotions
            related_emotions = self._get_related_emotions(current_emotion)
            
            for emotion in related_emotions:
                if emotion in self.emotion_index:
                    memory_ids = self.emotion_index[emotion]
                    memories = [m for m in self.emotional_memories[user_id] 
                               if m.id in memory_ids]
                    
                    if memories:
                        # Calculate weighted average based on recency and intensity
                        current_time = datetime.datetime.utcnow()
                        
                        weighted_sum = 0
                        weight_sum = 0
                        
                        for memory in memories:
                            memory_time = datetime.datetime.fromisoformat(memory.timestamp)
                            recency = 1.0 / max(1, (current_time - memory_time).total_seconds() / 86400)  # Days
                            
                            weight = recency * memory.intensity
                            weighted_sum += weight
                            weight_sum += weight
                        
                        if weight_sum > 0:
                            resonance = weighted_sum / weight_sum
                            influence["emotional_resonance"][emotion] = resonance
        
        # 3. Calculate symbol affinities
        for symbol in current_symbols:
            if symbol in self.symbol_index:
                memory_ids = self.symbol_index[symbol]
                memories = [m for m in self.emotional_memories[user_id] 
                           if m.id in memory_ids]
                
                if memories:
                    # Calculate average intensity
                    avg_intensity = sum(m.intensity for m in memories) / len(memories)
                    influence["symbol_affinities"][symbol] = avg_intensity
        
        # 4. Get trajectory influence
        if user_id in self.active_trajectories:
            trajectory = self.active_trajectories[user_id]
            
            if trajectory.trajectory_pattern_id:
                pattern = self.trajectory_patterns.get(trajectory.trajectory_pattern_id)
                
                if pattern:
                    # Get next predicted zone in sequence
                    current_pattern_idx = None
                    
                    # Try to find where in pattern we currently are
                    for i, zone in enumerate(pattern["zone_sequence"]):
                        if zone == current_zone and i < len(pattern["zone_sequence"]) - 1:
                            current_pattern_idx = i
                            break
                    
                    if current_pattern_idx is not None:
                        next_zone = pattern["zone_sequence"][current_pattern_idx + 1]
                        influence["trajectory_influence"] = {
                            "pattern_id": trajectory.trajectory_pattern_id,
                            "pattern_name": trajectory.trajectory_pattern_id.replace("_", " ").title(),
                            "next_zone": next_zone,
                            "confidence": 0.7  # Fixed confidence for now
                        }
        
        return {
            "status": "success",
            "influence_factors": influence,
            "memory_count": len(self.emotional_memories[user_id])
        }
    
    def _get_related_emotions(self, emotion: str) -> List[str]:
        """Get emotions related to the given emotion"""
        # Define emotion relationships
        emotion_relationships = {
            "joy": ["anticipation", "serenity", "awe"],
            "trust": ["serenity", "anticipation", "joy"],
            "fear": ["surprise", "anger", "sadness"],
            "surprise": ["curiosity", "joy", "fear"],
            "sadness": ["fear", "disgust", "trust"],
            "disgust": ["anger", "sadness", "fear"],
            "anger": ["disgust", "fear", "anticipation"],
            "anticipation": ["joy", "trust", "curiosity"],
            "curiosity": ["surprise", "anticipation", "awe"],
            "awe": ["joy", "surprise", "curiosity"],
            "serenity": ["trust", "joy", "anticipation"],
            "confusion": ["surprise", "curiosity", "fear"]
        }
        
        # Return related emotions plus the original
        related = emotion_relationships.get(emotion, [])
        return [emotion] + related
    
    def predict_emotional_trajectory(self, user_id: str, steps: int = 3) -> Dict:
        """Predict future emotional trajectory based on pattern recognition"""
        if user_id not in self.active_trajectories:
            return {"status": "no_trajectory"}
            
        trajectory = self.active_trajectories[user_id]
        
        # Need minimum sequence length for prediction
        if len(trajectory.emotion_sequence) < 3:
            return {"status": "insufficient_data"}
            
        # If we have a recognized pattern, use it for prediction
        if trajectory.trajectory_pattern_id and trajectory.trajectory_pattern_id in self.trajectory_patterns:
            pattern = self.trajectory_patterns[trajectory.trajectory_pattern_id]
            
            # Find where in pattern we currently are
            pattern_len = len(pattern["emotions"])
            traj_len = len(trajectory.emotion_sequence)
            
            best_match_pos = 0
            best_match_score = 0
            
            for i in range(min(pattern_len, traj_len)):
                # Check how well the last i elements of trajectory match pattern
                match_count = sum(1 for j in range(min(i, pattern_len))
                                if trajectory.emotion_sequence[traj_len - i + j] == pattern["emotions"][j])
                
                score = match_count / min(i, pattern_len) if min(i, pattern_len) > 0 else 0
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_pos = i
            
            # Generate predictions based on pattern
            predictions = []
            current_pos = best_match_pos % pattern_len
            
            for i in range(steps):
                next_pos = (current_pos + i + 1) % pattern_len
                predictions.append({
                    "emotion": pattern["emotions"][next_pos],
                    "confidence": best_match_score * (0.9 ** (i+1))  # Decreasing confidence
                })
            
            return {
                "status": "pattern_based_prediction",
                "pattern_id": trajectory.trajectory_pattern_id,
                "pattern_name": trajectory.trajectory_pattern_id.replace("_", " ").title(),
                "match_confidence": best_match_score,
                "predictions": predictions
            }
            
        else:
            # Use sequence prediction based on recent history
            # Simple Markov chain-like prediction
            emotion_transitions = {}
            
            # Build transition probabilities
            for i in range(len(trajectory.emotion_sequence) - 1):
                curr = trajectory.emotion_sequence[i]
                next_e = trajectory.emotion_sequence[i+1]
                
                if curr not in emotion_transitions:
                    emotion_transitions[curr] = {}
                    
                if next_e not in emotion_transitions[curr]:
                    emotion_transitions[curr][next_e] = 0
                    
                emotion_transitions[curr][next_e] += 1
            
            # Normalize probabilities
            for curr, transitions in emotion_transitions.items():
                total = sum(transitions.values())
                for next_e in transitions:
                    transitions[next_e] /= total
            
            # Generate predictions
            predictions = []
            current = trajectory.emotion_sequence[-1]
            
            for i in range(steps):
                if current not in emotion_transitions:
                    # No data for current state, use global distribution
                    emotion_counts = {}
                    for e in trajectory.emotion_sequence:
                        if e not in emotion_counts:
                            emotion_counts[e] = 0
                        emotion_counts[e] += 1
                    
                    total = sum(emotion_counts.values())
                    most_common = max(emotion_counts.items(), key=lambda x: x[1])[0]
                    confidence = emotion_counts[most_common] / total
                    
                    predictions.append({
                        "emotion": most_common,
                        "confidence": confidence * (0.7 ** (i+1))  # Decreasing confidence
                    })
                    
                    current = most_common
                    
                else:
                    # Use transition probabilities
                    transitions = emotion_transitions[current]
                    most_likely = max(transitions.items(), key=lambda x: x[1])[0]
                    confidence = transitions[most_likely]
                    
                    predictions.append({
                        "emotion": most_likely,
                        "confidence": confidence * (0.8 ** (i+1))  # Decreasing confidence
                    })
                    
                    current = most_likely
            
            return {
                "status": "statistical_prediction",
                "predictions": predictions
            }
    
    def visualize_emotional_landscape(self, user_id: str) -> Dict:
        """Generate visualization data for emotional landscape"""
        if user_id not in self.emotional_memories or not self.emotional_memories[user_id]:
            return {"status": "no_memories"}
            
        # Apply decay to ensure memories are current
        self._apply_memory_decay(user_id)
        
        # Collect emotional memories
        memories = self.emotional_memories[user_id]
        
        # Prepare visualization data
        visualization_data = {
            "emotion_distribution": {},    # Distribution of emotions
            "zone_heatmap": {},            # Heatmap of numogram zones
            "temporal_evolution": [],      # Emotion evolution over time
            "dominant_symbols": {},        # Most common symbols per emotion
            "emotional_network": {         # Network of related emotions
                "nodes": [],
                "links": []
            }
        }
        
        # Calculate emotion distribution
        for memory in memories:
            if memory.emotion_type not in visualization_data["emotion_distribution"]:
                visualization_data["emotion_distribution"][memory.emotion_type] = 0
            visualization_data["emotion_distribution"][memory.emotion_type] += memory.intensity
        
        # Normalize distribution
        total_intensity = sum(visualization_data["emotion_distribution"].values())
        if total_intensity > 0:
            for emotion in visualization_data["emotion_distribution"]:
                visualization_data["emotion_distribution"][emotion] /= total_intensity
        
        # Calculate zone heatmap
        for memory in memories:
            if memory.numogram_zone not in visualization_data["zone_heatmap"]:
                visualization_data["zone_heatmap"][memory.numogram_zone] = 0
            visualization_data["zone_heatmap"][memory.numogram_zone] += memory.intensity
        
        # Build temporal evolution
        time_ordered_memories = sorted(memories, key=lambda x: x.timestamp)
        
        # Group by day for temporal view
        day_emotions = {}
        for memory in time_ordered_memories:
            date = memory.timestamp.split('T')[0]  # Extract date part
            
            if date not in day_emotions:
                day_emotions[date] = {}
                
            if memory.emotion_type not in day_emotions[date]:
                day_emotions[date][memory.emotion_type] = 0
                
            day_emotions[date][memory.emotion_type] += memory.intensity
        
        # Convert to array format for visualization
        for date, emotions in day_emotions.items():
            entry = {"date": date}
            entry.update(emotions)
            visualization_data["temporal_evolution"].append(entry)
        
        # Find dominant symbols per emotion
        emotion_symbols = {}
        for memory in memories:
            if memory.emotion_type not in emotion_symbols:
                emotion_symbols[memory.emotion_type] = {}
                
            for symbol in memory.associated_symbols:
                if symbol not in emotion_symbols[memory.emotion_type]:
                    emotion_symbols[memory.emotion_type][symbol] = 0
                emotion_symbols[memory.emotion_type][symbol] += memory.intensity
        
        # Get top symbols per emotion
        for emotion, symbols in emotion_symbols.items():
            sorted_symbols = sorted(symbols.items(), key=lambda x: x[1], reverse=True)
            visualization_data["dominant_symbols"][emotion] = [s[0] for s in sorted_symbols[:5]]
        
        # Build emotional network
        # Nodes
        processed_emotions = set()
        for memory in memories:
            if memory.emotion_type not in processed_emotions:
                visualization_data["emotional_network"]["nodes"].append({
                    "id": memory.emotion_type,
                    "intensity": visualization_data["emotion_distribution"].get(memory.emotion_type, 0)
                })
                processed_emotions.add(memory.emotion_type)
        
        # Links (based on co-occurrence in trajectory)
        if user_id in self.active_trajectories:
            trajectory = self.active_trajectories[user_id]
            
            # Count emotion pairs
            emotion_pairs = {}
            for i in range(len(trajectory.emotion_sequence) - 1):
                curr = trajectory.emotion_sequence[i]
                next_e = trajectory.emotion_sequence[i+1]
                
                pair_key = f"{curr}-{next_e}"
                if pair_key not in emotion_pairs:
                    emotion_pairs[pair_key] = 0
                emotion_pairs[pair_key] += 1
            
            # Convert to links
            for pair, count in emotion_pairs.items():
                source, target = pair.split('-')
                visualization_data["emotional_network"]["links"].append({
                    "source": source,
                    "target": target,
                    "value": count
                })
        
        return {
            "status": "success",
            "visualization_data": visualization_data,
            "user_id": user_id,
            "memory_count": len(memories)
        }
    
    def create_signature_path(self, user_id: str) -> Dict:
        """
        Create a signature emotional path through numogram zones
        based on user's emotional history
        """
        if user_id not in self.emotional_memories or not self.emotional_memories[user_id]:
            return {"status": "no_memories"}
            
        # Get transitions from trajectory if available
        if user_id in self.active_trajectories:
            trajectory = self.active_trajectories[user_id]
            
            if len(trajectory.zone_sequence) >= 3:
                # Extract most common zone transitions
                transitions = {}
                
                for i in range(len(trajectory.zone_sequence) - 1):
                    curr = trajectory.zone_sequence[i]
                    next_z = trajectory.zone_sequence[i+1]
                    
                    key = f"{curr}-{next_z}"
                    if key not in transitions:
                        transitions[key] = 0
                    transitions[key] += 1
                
                # Sort by frequency
                sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
                
                # Get top transitions
                signature_transitions = [t[0] for t in sorted_transitions[:5]]
                
                # Build signature path
                path_zones = []
                path_emotions = []
                
                # Start with most frequent zone
                zone_counts = {}
                for z in trajectory.zone_sequence:
                    if z not in zone_counts:
                        zone_counts[z] = 0
                    zone_counts[z] += 1
                
                start_zone = max(zone_counts.items(), key=lambda x: x[1])[0]
                path_zones.append(start_zone)
                
                # Find emotion for zone
                zone_emotions = {}
                for i, zone in enumerate(trajectory.zone_sequence):
                    if zone not in zone_emotions:
                        zone_emotions[zone] = {}
                    
                    emotion = trajectory.emotion_sequence[i]
                    if emotion not in zone_emotions[zone]:
                        zone_emotions[zone][emotion] = 0
                    zone_emotions[zone][emotion] += 1
                
                # Add start emotion
                if start_zone in zone_emotions:
                    start_emotion = max(zone_emotions[start_zone].items(), key=lambda x: x[1])[0]
                    path_emotions.append(start_emotion)
                else:
                    path_emotions.append("neutral")
                
                # Build path using transitions
                for _ in range(min(4, len(signature_transitions))):  # Limit to 5 zones total
                    curr_zone = path_zones[-1]
                    
                    # Find next zone from transitions
                    next_zone = None
                    for trans in signature_transitions:
                        parts = trans.split('-')
                        if parts[0] == curr_zone:
                            next_zone = parts[1]
                            break
                    
                    if next_zone and next_zone not in path_zones:  # Avoid repeats
                        path_zones.append(next_zone)
                        
                        # Add emotion for this zone
                        if next_zone in zone_emotions:
                            emotion = max(zone_emotions[next_zone].items(), key=lambda x: x[1])[0]
                            path_emotions.append(emotion)
                        else:
                            path_emotions.append("neutral")
                
                # Create signature pathway
                signature_path = {
                    "name": f"User {user_id[:5]} Signature Path",
                    "zone_sequence": path_zones,
                    "emotion_sequence": path_emotions,
                    "description": "Emotional signature path based on user's emotional history",
                    "confidence": 0.7
                }
                
                # Add zone names
                zone_names = {
                    "1": "Unity", "2": "Division", "3": "Synthesis",
                    "4": "Structure", "5": "Transformation", "6": "Harmony",
                    "7": "Mystery", "8": "Power", "9": "Completion"
                }
                
                signature_path["zone_names"] = [zone_names.get(z, f"Zone {z}") for z in path_zones]
                
                return {
                    "status": "success",
                    "signature_path": signature_path
                }
        
        # Fallback if no trajectory or insufficient transitions
        # Use most intense emotions in each zone
        zone_emotions = {}
        for memory in self.emotional_memories[user_id]:
            if memory.numogram_zone not in zone_emotions:
                zone_emotions[memory.numogram_zone] = {}
                
            if memory.emotion_type not in zone_emotions[memory.numogram_zone]:
                zone_emotions[memory.numogram_zone][memory.emotion_type] = 0
                
            zone_emotions[memory.numogram_zone][memory.emotion_type] += memory.intensity
        
        # Sort zones by total emotional intensity
        zone_intensities = {}
        for zone, emotions in zone_emotions.items():
            zone_intensities[zone] = sum(emotions.values())
        
        # Get top 5 most emotionally intense zones
        top_zones = sorted(zone_intensities.items(), key=lambda x: x[1], reverse=True)[:5]
        path_zones = [z[0] for z in top_zones]
        
        # Get dominant emotion for each zone
        path_emotions = []
        for zone in path_zones:
            emotions = zone_emotions[zone]
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            path_emotions.append(dominant_emotion)
        
        # Create signature pathway
        zone_names = {
            "1": "Unity", "2": "Division", "3": "Synthesis",
            "4": "Structure", "5": "Transformation", "6": "Harmony",
            "7": "Mystery", "8": "Power", "9": "Completion"
        }
        
        signature_path = {
            "name": f"User {user_id[:5]} Emotional Resonance Path",
            "zone_sequence": path_zones,
            "emotion_sequence": path_emotions,
            "zone_names": [zone_names.get(z, f"Zone {z}") for z in path_zones],
            "description": "Emotional resonance path based on user's most intense emotional zones",
            "confidence": 0.5
        }
        
        return {
            "status": "success",
            "signature_path": signature_path
        }
    
    def integrate_with_numogram(self, emotional_state: Dict, user_id: str, context_data: Dict = None) -> Dict:
        """
        Integrate emotional memory system with numogram system for transitions
        that are informed by emotional history
        """
        # Store emotional memory
        context_text = context_data.get("text_input", "") if context_data else ""
        symbols = context_data.get("symbolic_patterns", []) if context_data else None
        memory_id = self.store_emotional_memory(emotional_state, user_id, context_text, symbols)
        
        # Get memory influence factors
        influence = self.get_memory_influence(user_id, {
            "current_zone": emotional_state.get("numogram_zone", "1"),
            "current_emotion": emotional_state.get("primary_emotion"),
            "current_symbols": [s.get("core_symbols", [""])[0] for s in symbols[:5]] if symbols else []
        })
        
        # Add to context data
        if context_data is None:
            context_data = {}
            
        context_data["emotional_memory"] = {
            "memory_id": memory_id,
            "influence_factors": influence.get("influence_factors", {})
        }
        
        # Add emotional trajectory prediction
        prediction = self.predict_emotional_trajectory(user_id, steps=2)
        if prediction.get("status") != "no_trajectory":
            context_data["emotional_trajectory"] = prediction
        
        # Get signature path if it exists
        signature_path = self.create_signature_path(user_id)
        if signature_path.get("status") == "success":
            context_data["signature_path"] = signature_path["signature_path"]
        
        # Calculate feedback based on emotional intensity and memory resonance
        feedback = emotional_state.get("intensity", 0.5)
        
        # Modify feedback based on memory influence
        influence_factors = influence.get("influence_factors", {})
        
        if "emotional_resonance" in influence_factors:
            resonance = influence_factors["emotional_resonance"].get(
                emotional_state.get("primary_emotion", "neutral"), 0
            )
            # Blend with resonance (strong resonance increases feedback)
            feedback = (feedback * 0.7) + (resonance * 0.3)
        
        # Ensure feedback is in valid range
        feedback = max(0.1, min(0.9, feedback))
        
        # Get current zone
        current_zone = emotional_state.get("numogram_zone", "1")
        
        # Check if trajectory suggests a specific next zone
        next_zone = None
        if "trajectory_influence" in influence_factors and influence_factors["trajectory_influence"]:
            trajectory_influence = influence_factors["trajectory_influence"]
            next_zone = trajectory_influence.get("next_zone")
            # Only use if confidence is sufficient
            if trajectory_influence.get("confidence", 0) < 0.5:
                next_zone = None
        
        # Execute numogram transition
        result = self.numogram.transition(
            user_id=user_id,
            current_zone=current_zone,
            feedback=feedback,
            context_data=context_data
        )
        
        # Check if next_zone from trajectory was followed
        followed_trajectory = False
        if next_zone and result["next_zone"] == next_zone:
            followed_trajectory = True
        
        return {
            "numogram_result": result,
            "emotional_memory": {
                "memory_id": memory_id,
                "influence_applied": True,
                "trajectory_followed": followed_trajectory
            }
        }
```
