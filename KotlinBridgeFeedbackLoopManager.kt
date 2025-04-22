# Kotlin bridge for the Feedback Loop Manager
class FeedbackLoopBridge:
    """
    Bridge to expose the Feedback Loop Manager to Kotlin code.
    """
    
    def __init__(self, feedback_manager: FeedbackLoopManager = None):
        """
        Initialize the Kotlin bridge.
        
        Args:
            feedback_manager: FeedbackLoopManager instance
        """
        try:
            self.feedback_manager = feedback_manager or setup_feedback_loop()
            logger.info("Feedback loop bridge initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing feedback loop bridge: {e}")
            self.feedback_manager = FeedbackLoopManager()
    
    def process_technical_response_json(self, concept: str, statement: str, detail_type: str = None) -> str:
        """
        Process a technical response and return as JSON.
        
        Args:
            concept: The concept being discussed
            statement: The technical statement to process
            detail_type: Optional detail type to focus on
            
        Returns:
            json_result: JSON string with processed response information
        """
        try:
            processed = self.feedback_manager.process_technical_response(concept, statement, detail_type)
            return json.dumps(processed)
        except Exception as e:
            logger.error(f"Error in process_technical_response_json: {e}")
            return json.dumps({"error": str(e)})
    
    def format_response_with_citations_json(self, response_elements_json: str) -> str:
        """
        Format a complete technical response with multiple citations from JSON.
        
        Args:
            response_elements_json: JSON string with response elements
            
        Returns:
            json_result: JSON string with formatted response
        """
        try:
            response_elements = json.loads(response_elements_json)
            formatted = self.feedback_manager.format_technical_response_with_citations(response_elements)
            return json.dumps({"formatted_response": formatted})
        except Exception as e:
            logger.error(f"Error in format_response_with_citations_json: {e}")
            return json.dumps({"error": str(e)})
    
    def get_session_metrics_json(self) -> str:
        """
        Get metrics for the current feedback loop session as JSON.
        
        Returns:
            json_result: JSON string with session metrics
        """
        try:
            metrics = self.feedback_manager.get_session_metrics()
            return json.dumps(metrics)
        except Exception as e:
            logger.error(f"Error in get_session_metrics_json: {e}")
            return json.dumps({"error": str(e)})
    
    def reset_session_json(self) -> str:
        """
        Reset the current feedback loop session and return status as JSON.
        
        Returns:
            json_result: JSON string with reset status
        """
        try:
            self.feedback_manager.reset_session()
            return json.dumps({"status": "success", "message": "Session reset successfully"})
        except Exception as e:
            logger.error(f"Error in reset_session_json: {e}")
            return json.dumps({"error": str(e)})


# Initialize the bridge for external access
feedback_loop_bridge = None
try:
    # Try to import and use the introspection module
    try:
        from system_introspection import SystemIntrospection
        introspection = SystemIntrospection()
        feedback_manager = setup_feedback_loop(introspection)
    except ImportError:
        feedback_manager = setup_feedback_loop()
    
    feedback_loop_bridge = FeedbackLoopBridge(feedback_manager)
    logger.info("Feedback loop bridge initialized and ready for use")
except Exception as e:
    logger.error(f"Failed to initialize feedback loop bridge: {e}")


# Example usage
def test_feedback_loop():
    """Test the feedback loop functionality."""
    print("Testing Feedback Loop with Source Citation...")
    
    # Initialize
    try:
        from system_introspection import SystemIntrospection
        introspection = SystemIntrospection()
        feedback_manager = setup_feedback_loop(introspection)
    except ImportError:
        print("Warning: SystemIntrospection module not available")
        feedback_manager = setup_feedback_loop()
    
    # Test processing a technical response
    concepts = ["AgentDecisionMaking", "EnergyManagement", "AgentCommunication"]
    statements = [
        "The decide_action method calculates utility scores for different actions and selects the action with the highest utility.",
        "The energy cost for the 'avoid' action is 1.5 times the base cost, making it more expensive than regular movement.",
        "Agents prioritize messages based on their importance and process up to 5 messages per step."
    ]
    
    response_elements = []
    
    for concept, statement in zip(concepts, statements):
        print(f"\nProcessing statement about {concept}...")
        processed = feedback_manager.process_technical_response(concept, statement)
        
        print(f"Processed text: {processed['processed_text']}")
        print(f"Verification: {processed['verification']['message']}")
        print(f"Accuracy: {processed['accuracy']['message']}")
        
        if processed.get("needs_correction", False):
            print(f"Needs correction: {processed['correction'].get('corrections', [])}")
        
        response_elements.append(processed)
    
    # Format complete response
    print("\nFormatted complete response:")
    formatted = feedback_manager.format_technical_response_with_citations(response_elements)
    print(formatted)
    
    # Get session metrics
    print("\nSession metrics:")
    metrics = feedback_manager.get_session_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test the bridge
    bridge = FeedbackLoopBridge(feedback_manager)
    print("\nTesting Kotlin bridge...")
    json_result = bridge.process_technical_response_json(
        "TaskAllocation", 
        "Tasks track progress using a value from 0.0 to 1.0 and are marked complete when this value reaches 1.0."
    )
    print(f"Bridge result: {json_result[:100]}...")
    
    print("\nTesting complete!")


if __name__ == "__main__":
    test_feedback_loop()
```
