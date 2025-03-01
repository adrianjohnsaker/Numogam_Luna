import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


class MemoryManager:
    def __init__(self, memory_path: str = "./memory"):
        """
        Initialize the MemoryManager.
        
        Args:
            memory_path: Path to the directory where memory files are stored
        """
        self.memory_path = memory_path
        os.makedirs(memory_path, exist_ok=True)
    
    def _load_memory(self, user_id: str) -> Dict[str, Any]:
        """
        Load user memory from file.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing user memory data
        """
        memory_file = os.path.join(self.memory_path, f"{user_id}.json")
        
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._initialize_memory()
        else:
            return self._initialize_memory()
    
    def _save_memory(self, user_id: str, memory_data: Dict[str, Any]) -> None:
        """
        Save user memory to file.
        
        Args:
            user_id: Unique identifier for the user
            memory_data: Dictionary containing user memory data
        """
        memory_file = os.path.join(self.memory_path, f"{user_id}.json")
        
        with open(memory_file, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def _initialize_memory(self) -> Dict[str, Any]:
        """
        Initialize a new memory structure.
        
        Returns:
            Dictionary with initialized memory structure
        """
        return {
            "conversation_history": [],
            "preferences": {},
            "last_interaction": None,
            "session_count": 0
        }
    
    def update_memory(self, user_id: str, key: str, value: Any) -> None:
        """
        Update a specific memory field for a user.
        
        Args:
            user_id: Unique identifier for the user
            key: Memory field to update
            value: New value to store
        """
        memory_data = self._load_memory(user_id)
        
        # Update the specific field
        memory_data[key] = value
        
        # Also update the last interaction timestamp
        memory_data["last_interaction"] = datetime.now().isoformat()
        
        # Save the updated memory
        self._save_memory(user_id, memory_data)
    
    def add_to_conversation_history(self, user_id: str, user_input: str, response: str) -> None:
        """
        Add a conversation exchange to the user's history.
        
        Args:
            user_id: Unique identifier for the user
            user_input: The user's input text
            response: The AI's response
        """
        memory_data = self._load_memory(user_id)
        
        # Add the new exchange
        memory_data["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response
        })
        
        # Limit history size to avoid excessive storage
        if len(memory_data["conversation_history"]) > 50:
            memory_data["conversation_history"] = memory_data["conversation_history"][-50:]
        
        # Update session count and last interaction
        memory_data["session_count"] += 1
        memory_data["last_interaction"] = datetime.now().isoformat()
        
        # Save the updated memory
        self._save_memory(user_id, memory_data)
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history for a user.
        
        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of conversation exchanges to return
            
        Returns:
            List of conversation exchanges
        """
        memory_data = self._load_memory(user_id)
        return memory_data["conversation_history"][-limit:]
    
    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get stored preferences for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary of user preferences
        """
        memory_data = self._load_memory(user_id)
        return memory_data.get("preferences", {})
    
    def update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """
        Update preferences for a user.
        
        Args:
            user_id: Unique identifier for the user
            preferences: Dictionary of preferences to update
        """
        memory_data = self._load_memory(user_id)
        
        # Merge new preferences with existing ones
        current_prefs = memory_data.get("preferences", {})
        current_prefs.update(preferences)
        memory_data["preferences"] = current_prefs
        
        # Save the updated memory
        self._save_memory(user_id, memory_data)


# Create a singleton instance for easy import
memory_manager = MemoryManager()

def update_memory(user_id: str, key: str, value: Any) -> None:
    """
    Convenience function to update a memory field.
    
    Args:
        user_id: Unique identifier for the user
        key: Memory field to update
        value: New value to store
    """
    memory_manager.update_memory(user_id, key, value)

def add_conversation(user_id: str, user_input: str, response: str) -> None:
    """
    Convenience function to add a conversation exchange.
    
    Args:
        user_id: Unique identifier for the user
        user_input: The user's input text
        response: The AI's response
    """
    memory_manager.add_to_conversation_history(user_id, user_input, response)

def get_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience function to get conversation history.
    
    Args:
        user_id: Unique identifier for the user
        limit: Maximum number of conversation exchanges to return
        
    Returns:
        List of conversation exchanges
    """
    return memory_manager.get_conversation_history(user_id, limit)
