#!/usr/bin/env python3
"""
Enhanced Memory Management System
Provides tiered memory storage with efficient persistence capabilities.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Union
import logging
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MemoryManager")

class MemoryManager:
    """
    Memory management system with tiered storage (short-term, mid-term, long-term)
    Thread-safe implementation with optional persistence
    """

    def __init__(self, 
                 persist_path: Optional[str] = None,
                 short_term_limit: int = 5,
                 mid_term_limit: int = 10,
                 long_term_limit: int = 20):
        """
        Initialize the memory manager
        
        Args:
            persist_path: Directory path for persistent storage (None for in-memory only)
            short_term_limit: Maximum number of items in short-term memory
            mid_term_limit: Maximum number of items in mid-term memory
            long_term_limit: Maximum number of items in long-term memory
        """
        self.persist_path = persist_path
        self.short_term_limit = short_term_limit
        self.mid_term_limit = mid_term_limit
        self.long_term_limit = long_term_limit
        
        # Initialize memory structure
        self.memories: Dict[str, Dict[str, List[Any]]] = {}
        
        # Thread safety
        self.lock = Lock()
        
        # Load persisted memories if available
        if self.persist_path and os.path.exists(self.persist_path):
            self._load_persistent_memories()

    def _get_user_memory(self, user_id: str) -> Dict[str, List[Any]]:
        """
        Get or initialize user memory structure
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            User memory dictionary with initialized tiers
        """
        if user_id not in self.memories:
            self.memories[user_id] = {
                "short_term": [],
                "mid_term": [],
                "long_term": [],
                "last_updated": time.time()
            }
        return self.memories[user_id]

    def recall(self, 
               user_id: str, 
               depth: int = 3, 
               memory_type: Optional[str] = None) -> List[Any]:
        """
        Retrieve memory entries for a user with tiered priority
        
        Args:
            user_id: Unique identifier for the user
            depth: Number of memories to recall per tier
            memory_type: Specific memory tier to recall from (all tiers if None)
            
        Returns:
            List of memory entries
        """
        with self.lock:
            if user_id not in self.memories:
                return []
            
            user_memory = self.memories[user_id]
            
            # If specific memory type requested
            if memory_type and memory_type in user_memory:
                return user_memory[memory_type][:depth]
            
            # Combine memories from different tiers with weighting
            short_term = user_memory.get("short_term", [])[:depth]
            mid_term = user_memory.get("mid_term", [])[:depth]
            long_term = user_memory.get("long_term", [])[:depth]
            
            # Prioritize recent short-term memories but include context from other tiers
            combined = short_term + mid_term + long_term
            return combined[-depth:]

    def store(self, 
              user_id: str, 
              value: Any, 
              memory_type: str = "short_term") -> None:
        """
        Store memory directly in a specific tier
        
        Args:
            user_id: Unique identifier for the user
            value: Value to store in memory
            memory_type: Memory tier to store in
        """
        with self.lock:
            user_memory = self._get_user_memory(user_id)
            
            if memory_type not in user_memory:
                logger.warning(f"Invalid memory type: {memory_type}, defaulting to short_term")
                memory_type = "short_term"
                
            # Add timestamp if not present
            if isinstance(value, dict) and "timestamp" not in value:
                value["timestamp"] = time.time()
                
            # Add to specified memory tier
            user_memory[memory_type].insert(0, value)
            
            # Enforce tier limits
            if memory_type == "short_term" and len(user_memory["short_term"]) > self.short_term_limit:
                user_memory["short_term"] = user_memory["short_term"][:self.short_term_limit]
            elif memory_type == "mid_term" and len(user_memory["mid_term"]) > self.mid_term_limit:
                user_memory["mid_term"] = user_memory["mid_term"][:self.mid_term_limit]
            elif memory_type == "long_term" and len(user_memory["long_term"]) > self.long_term_limit:
                user_memory["long_term"] = user_memory["long_term"][:self.long_term_limit]
                
            # Update timestamp
            user_memory["last_updated"] = time.time()
            
            # Auto-persist if path is configured
            if self.persist_path:
                self._persist_memories()

    def update_memory(self, 
                      user_id: str, 
                      key: str, 
                      value: Any) -> None:
        """
        Update memory with automatic tier progression
        
        Args:
            user_id: Unique identifier for the user
            key: Memory key (not used in current implementation but kept for API compatibility)
            value: Value to store in memory
        """
        with self.lock:
            user_memory = self._get_user_memory(user_id)
            
            # Format memory item if needed
            memory_item = value
            if not isinstance(value, dict):
                memory_item = {"content": value, "timestamp": time.time()}
            elif "timestamp" not in memory_item:
                memory_item["timestamp"] = time.time()
            
            # Store in short-term memory
            user_memory["short_term"].insert(0, memory_item)
            
            # Move items between tiers when limits are reached
            if len(user_memory["short_term"]) > self.short_term_limit:
                overflow = user_memory["short_term"].pop()
                user_memory["mid_term"].insert(0, overflow)
                
            if len(user_memory["mid_term"]) > self.mid_term_limit:
                overflow = user_memory["mid_term"].pop()
                user_memory["long_term"].insert(0, overflow)
                
            # Limit long-term memory size
            if len(user_memory["long_term"]) > self.long_term_limit:
                user_memory["long_term"] = user_memory["long_term"][:self.long_term_limit]
            
            # Update timestamp
            user_memory["last_updated"] = time.time()
            
            # Auto-persist if path is configured
            if self.persist_path:
                self._persist_memories()

    def clear_memory(self, 
                     user_id: str, 
                     memory_type: Optional[str] = None) -> bool:
        """
        Clear user memory
        
        Args:
            user_id: Unique identifier for the user
            memory_type: Specific memory tier to clear (all if None)
            
        Returns:
            True if operation was successful
        """
        with self.lock:
            if user_id not in self.memories:
                return False
                
            if memory_type:
                if memory_type in self.memories[user_id]:
                    self.memories[user_id][memory_type] = []
                    
                    # Update timestamp
                    self.memories[user_id]["last_updated"] = time.time()
                    
                    # Auto-persist if path is configured
                    if self.persist_path:
                        self._persist_memories()
                    return True
                return False
            
            # Clear all memories
            self.memories[user_id] = {
                "short_term": [],
                "mid_term": [],
                "long_term": [],
                "last_updated": time.time()
            }
            
            # Auto-persist if path is configured
            if self.persist_path:
                self._persist_memories()
            return True

    def _persist_memories(self) -> bool:
        """
        Save memories to persistent storage
        
        Returns:
            True if operation was successful
        """
        if not self.persist_path:
            return False
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            
            # Write to temporary file first to prevent corruption
            temp_path = f"{self.persist_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(self.memories, f)
                
            # Rename to target file (atomic operation)
            os.replace(temp_path, self.persist_path)
            return True
        except Exception as e:
            logger.error(f"Failed to persist memories: {str(e)}")
            return False

    def _load_persistent_memories(self) -> bool:
        """
        Load memories from persistent storage
        
        Returns:
            True if operation was successful
        """
        if not self.persist_path or not os.path.exists(self.persist_path):
            return False
            
        try:
            with open(self.persist_path, 'r') as f:
                loaded_memories = json.load(f)
                
                # Validate structure
                if not isinstance(loaded_memories, dict):
                    logger.error("Invalid memory file format")
                    return False
                    
                self.memories = loaded_memories
                return True
        except Exception as e:
            logger.error(f"Failed to load memories: {str(e)}")
            return False


# Compatibility functions (for backward compatibility)
_memory_manager = MemoryManager()

def memory_recall(user_id, depth=3):
    """Legacy compatibility function for memory recall"""
    return _memory_manager.recall(user_id, depth)

def update_memory(user_id, key, value):
    """Legacy compatibility function for memory update"""
    return _memory_manager.update_memory(user_id, key, value)


# Example usage
if __name__ == "__main__":
    # Example with persistence
    memory = MemoryManager(persist_path="/storage/emulated/0/Android/data/com.yourapp.luna/memory/user_memory.json")
    
    # Store some memories
    memory.update_memory("user123", "preference", {"theme": "dark"})
    memory.update_memory("user123", "conversation", {"topic": "weather", "sentiment": "positive"})
    
    # Recall memories
    recalled = memory.recall("user123")
    print("Recalled memories:", recalled)
