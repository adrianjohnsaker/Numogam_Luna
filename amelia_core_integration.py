# amelia_core_integration.py
"""
Integration layer that connects Amelia's core response system 
with the Numogrammatic Memory Module
"""

import json
from typing import Dict, Any, Optional
from numogram_memory_module import NumogrammaticMemory
from amelia_numogram_bridge import AmeliaNumogramBridge

class AmeliaCoreIntegration:
    """
    Main integration point for Amelia's response pipeline
    """
    
    def __init__(self):
        self.memory = None
        self.bridge = None
        self.initialized = False
        self.current_session = None
        
    def initialize(self, storage_path: str = "numogram_memory") -> bool:
        """Initialize the complete memory system"""
        try:
            # Create memory module
            self.memory = NumogrammaticMemory(storage_path)
            
            # Create bridge
            self.bridge = AmeliaNumogramBridge(self.memory)
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize: {e}")
            return False
    
    def process_amelia_exchange(self, 
                               user_input: str,
                               amelia_base_response: str,
                               session_id: str) -> str:
        """
        Main entry point for processing Amelia's responses
        This should be called from wherever Amelia generates responses
        """
        if not self.initialized:
            return amelia_base_response
        
        try:
            # Process through numogrammatic system
            retrieval_data = self.bridge.process_user_input(user_input, session_id)
            
            # Generate enhanced response
            enhanced_response = self.bridge.generate_enhanced_response(
                amelia_base_response, 
                retrieval_data
            )
            
            # Store the enhanced response
            self.bridge.store_amelia_response(enhanced_response, session_id)
            
            return enhanced_response
            
        except Exception as e:
            print(f"Error in processing: {e}")
            # Fallback to base response
            return amelia_base_response
    
    def get_response_with_memory(self,
                                user_input: str,
                                session_id: str,
                                response_generator_func) -> str:
        """
        Wrapper for any response generation function
        response_generator_func should be the function that generates Amelia's base response
        """
        # Generate base response using provided function
        base_response = response_generator_func(user_input)
        
        # Enhance with memory
        return self.process_amelia_exchange(user_input, base_response, session_id)


# Global instance for easy access
amelia_core = AmeliaCoreIntegration()

# Integration functions for different Amelia implementations

def integrate_with_pytorch_model(model, tokenizer, device="cpu"):
    """
    Integration for PyTorch-based Amelia
    """
    def generate_response_with_memory(user_input: str, session_id: str) -> str:
        # Original PyTorch response generation
        def pytorch_generator(text):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Wrap with memory integration
        return amelia_core.get_response_with_memory(
            user_input, 
            session_id,
            pytorch_generator
        )
    
    return generate_response_with_memory

def integrate_with_api_model(api_func):
    """
    Integration for API-based Amelia (e.g., calling an external AI service)
    """
    def generate_response_with_memory(user_input: str, session_id: str) -> str:
        # Wrap API call with memory
        return amelia_core.get_response_with_memory(
            user_input,
            session_id,
            lambda text: api_func(text)
        )
    
    return generate_response_with_memory

def integrate_with_custom_model(response_function):
    """
    Generic integration for any custom response generation function
    """
    def enhanced_response_function(user_input: str, session_id: str) -> str:
        return amelia_core.get_response_with_memory(
            user_input,
            session_id,
            lambda text: response_function(text)
        )
    
    return enhanced_response_function

# Direct integration point for Kotlin/Android
def process_amelia_message(user_input: str, 
                          base_response: str,
                          session_id: str) -> Dict[str, Any]:
    """
    Direct integration for Kotlin bridge
    Returns structured data including the enhanced response
    """
    if not amelia_core.initialized:
        amelia_core.initialize()
    
    enhanced_response = amelia_core.process_amelia_exchange(
        user_input,
        base_response,
        session_id
    )
    
    # Get current status
    status = {}
    if amelia_core.bridge:
        status = {
            "current_zone": amelia_core.bridge.response_context["zone_position"],
            "temporal_phase": amelia_core.bridge.response_context["temporal_phase"],
            "active_contagions": len(amelia_core.bridge.response_context["active_contagions"])
        }
    
    return {
        "response": enhanced_response,
        "status": status,
        "enhanced": enhanced_response != base_response
    }
