"""
Python Bridge Module
Serves as the central coordinator for all Python modules in the application.
"""

import time
import json
from MetacognitiveArchitecture import MetacognitiveArchitecture
from BoundaryAwareness import BoundaryAwareness

class PythonBridge:
    def __init__(self):
        self.metacognitive_arch = MetacognitiveArchitecture()
        self.boundary_awareness = BoundaryAwareness()
        self.module_registry = {
            "metacognitive": self.metacognitive_arch,
            "boundary": self.boundary_awareness
        }
        self.interaction_history = []
        self.init_timestamp = time.time()
    
    def process_message(self, user_input, context=None):
        """
        Process user input through all modules and return consolidated results
        
        Args:
            user_input (str): The text input from the user
            context (dict): Additional context information
            
        Returns:
            dict: Consolidated processing results
        """
        timestamp = time.time()
        context = context or {}
        
        # Update interaction timestamps
        self.boundary_awareness.update_interaction_timestamp(timestamp)
        
        # Process through metacognitive architecture
        meta_results = self.metacognitive_arch.process_input(user_input)
        
        # Analyze boundary cues
        boundary_cues = self.boundary_awareness.interpret_user_boundary_cues(user_input)
        
        # Get appropriate response style
        response_style = self.boundary_awareness.get_appropriate_response_style()
        
        # Retrieve relevant memories
        relevant_memories = self.metacognitive_arch.retrieve_relevant_memories(user_input)
        
        # Adapt to relationship stage
        relationship_adaptation = self.boundary_awareness.adapt_to_relationship_stage()
        
        # Consolidate results
        result = {
            "timestamp": timestamp,
            "user_input": user_input,
            "processed_input": meta_results,
            "boundary_cues": boundary_cues,
            "response_style": response_style,
            "relationship_stage": {
                "duration_days": self.boundary_awareness.boundaries["temporal"]["relationship_duration"],
                "adapted_style": relationship_adaptation
            },
            "relevant_memories": relevant_memories,
            "context": context
        }
        
        # Store in interaction history
        self.interaction_history.append({
            "timestamp": timestamp,
            "user_input": user_input,
            "processing_result": result
        })
        
        return result
    
    def record_response(self, ai_response, user_feedback=None):
        """
        Record AI response and optional user feedback
        
        Args:
            ai_response (str): The response given to the user
            user_feedback (float, optional): User feedback score (-1.0 to 1.0)
            
        Returns:
            bool: Success status
        """
        if not self.interaction_history:
            return False
            
        # Get the last interaction
        last_interaction = self.interaction_history[-1]
        last_interaction["ai_response"] = ai_response
        
        # Record in metacognitive system for learning
        user_input = last_interaction["user_input"]
        self.metacognitive_arch.reflect_on_interaction(user_input, ai_response, user_feedback)
        
        return True
    
    def get_module(self, module_name):
        """
        Get reference to a specific module
        
        Args:
            module_name (str): Name of the module
            
        Returns:
            object: Module instance or None if not found
        """
        return self.module_registry.get(module_name)
    
    def register_module(self, name, module):
        """
        Register a new module
        
        Args:
            name (str): Module name
            module (object): Module instance
            
        Returns:
            bool: Success status
        """
        if name not in self.module_registry:
            self.module_registry[name] = module
            return True
        return False
    
    def export_state(self):
        """
        Export the current state of all modules
        
        Returns:
            dict: State data for all modules
        """
        return {
            "metacognitive": {
                "personality_traits": self.metacognitive_arch.personality_traits,
                "metacognitive_state": self.metacognitive_arch.metacognitive_state,
                "short_term_memory_size": len(self.metacognitive_arch.short_term_memory),
                "long_term_memory_size": len(self.metacognitive_arch.long_term_memory)
            },
            "boundary_awareness": {
                "boundaries": self.boundary_awareness.boundaries,
                "boundary_shifts": len(self.boundary_awareness.boundary_shifts),
                "relationship_duration": self.boundary_awareness.boundaries["temporal"]["relationship_duration"]
            },
            "interaction_count": len(self.interaction_history),
            "uptime": time.time() - self.init_timestamp
        }
    
    def import_state(self, state_data):
        """
        Import a previously exported state
        
        Args:
            state_data (dict): State data to import
            
        Returns:
            bool: Success status
        """
        try:
            # Import metacognitive state
            if "metacognitive" in state_data:
                meta_data = state_data["metacognitive"]
                if "personality_traits" in meta_data:
                    self.metacognitive_arch.personality_traits = meta_data["personality_traits"]
                if "metacognitive_state" in meta_data:
                    self.metacognitive_arch.metacognitive_state = meta_data["metacognitive_state"]
            
            # Import boundary awareness state
            if "boundary_awareness" in state_data and "boundaries" in state_data["boundary_awareness"]:
                self.boundary_awareness.boundaries = state_data["boundary_awareness"]["boundaries"]
            
            return True
        except Exception as e:
            print(f"Error importing state: {str(e)}")
            return False
            
    def get_recent_interactions(self, count=5):
        """
        Get the most recent interactions from history
        
        Args:
            count (int): Number of recent interactions to return
            
        Returns:
            list: Recent interactions
        """
        return self.interaction_history[-count:] if len(self.interaction_history) >= count else self.interaction_history[:]
    
    def clear_history(self, older_than_days=None):
        """
        Clear interaction history, optionally keeping recent entries
        
        Args:
            older_than_days (float, optional): If provided, only clear entries older than this many days
            
        Returns:
            int: Number of entries cleared
        """
        if older_than_days is None:
            count = len(self.interaction_history)
            self.interaction_history = []
            return count
        
        current_time = time.time()
        cutoff_time = current_time - (older_than_days * 86400)  # 86400 seconds per day
        
        original_count = len(self.interaction_history)
        self.interaction_history = [
            interaction for interaction in self.interaction_history 
            if interaction["timestamp"] >= cutoff_time
        ]
        
        return original_count - len(self.interaction_history)
    
    def analyze_interaction_patterns(self):
        """
        Analyze patterns in user interactions
        
        Returns:
            dict: Analysis results
        """
        if not self.interaction_history:
            return {"error": "No interaction history to analyze"}
            
        # Calculate average time between messages
        timestamps = [item["timestamp"] for item in self.interaction_history]
        time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        avg_time_between = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Count interactions by time of day
        hour_distribution = {}
        for item in self.interaction_history:
            hour = time.localtime(item["timestamp"]).tm_hour
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
            
        # Analyze common topics using metacognitive architecture
        topics = self.metacognitive_arch.extract_common_topics(
            [item["user_input"] for item in self.interaction_history]
        )
        
        return {
            "total_interactions": len(self.interaction_history),
            "avg_time_between_messages_seconds": avg_time_between,
            "hour_distribution": hour_distribution,
            "common_topics": topics,
            "relationship_duration_days": self.boundary_awareness.boundaries["temporal"]["relationship_duration"]
        }
    
    def emergency_reset(self):
        """
        Emergency reset of the system state
        
        Returns:
            bool: Success status
        """
        try:
            # Reset metacognitive architecture
            self.metacognitive_arch = MetacognitiveArchitecture()
            
            # Reset boundary awareness
            self.boundary_awareness = BoundaryAwareness()
            
            # Reset module registry
            self.module_registry = {
                "metacognitive": self.metacognitive_arch,
                "boundary": self.boundary_awareness
            }
            
            # Clear interaction history
            self.interaction_history = []
            
            # Reset initialization timestamp
            self.init_timestamp = time.time()
            
            return True
        except Exception as e:
            print(f"Error during emergency reset: {str(e)}")
            return False
    
    def save_state_to_file(self, filepath):
        """
        Save current state to a JSON file
        
        Args:
            filepath (str): Path to save file
            
        Returns:
            bool: Success status
        """
        try:
            state = self.export_state()
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving state to file: {str(e)}")
            return False
    
    def load_state_from_file(self, filepath):
        """
        Load state from a JSON file
        
        Args:
            filepath (str): Path to load file
            
        Returns:
            bool: Success status
        """
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            return self.import_state(state_data)
        except Exception as e:
            print(f"Error loading state from file: {str(e)}")
            return False
