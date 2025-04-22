class FeedbackLoopManager:
    """
    Manages the feedback loop for source citation and verification.
    """
    
    def __init__(self, introspection_module=None):
        """
        Initialize the feedback loop manager.
        
        Args:
            introspection_module: Optional reference to the SystemIntrospection module
        """
        self.citation_protocol = CitationProtocol(introspection_module)
        self.response_templates = self._load_response_templates()
        self.current_session = {
            "citations": [],
            "verifications": [],
            "accuracy_checks": [],
            "corrections": []
        }
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates for generating formatted responses."""
        # These would typically be loaded from a file, but are included here for simplicity
        return {
            "citation": "According to the implementation in {file_name}, specifically in the {class_name}.{method_name} method, {statement} {citation}",
            "verification_success": "I've verified my understanding by analyzing the implementation details. {message}",
            "verification_failure": "I notice my explanation may not fully align with the implementation. Let me revise based on the actual code.",
            "accuracy_correction": "I need to correct my previous statement. {corrections} Let me provide the accurate information based on the code implementation.",
            "confidence_high": "I'm highly confident about this implementation detail as I'm directly referencing the code.",
            "confidence_medium": "I have reasonable confidence in this implementation detail based on the code, though some aspects are inferred.",
            "confidence_low": "I'm describing this at a conceptual level, as the specific implementation details aren't fully accessible to me.",
        }
    
    def process_technical_response(self, concept: str, statement: str, 
                                  detail_type: str = None) -> Dict[str, Any]:
        """
        Process a technical response by adding citations, verifying understanding,
        and checking accuracy.
        
        Args:
            concept: The concept being discussed
            statement: The technical statement to process
            detail_type: Optional detail type to focus on
            
        Returns:
            processed_response: Dictionary with processed response information
        """
        # Generate citation
        citation_result = self.citation_protocol.generate_citation(concept, detail_type)
        citation = citation_result.get("citation", "")
        
        # Verify understanding
        verification = self.citation_protocol.verify_understanding(statement, concept)
        
        # Check accuracy
        accuracy = self.citation_protocol.check_statement_accuracy(statement, concept)
        
        # Determine if correction is needed
        needs_correction = not accuracy.get("accurate", True)
        correction = None
        
        if needs_correction:
            correction = self.citation_protocol.get_correction_suggestion(concept)
        
        # Format response with citation
        formatted_statement = statement
        
        # Add confidence indicator
        confidence = citation_result.get("info", {}).get("confidence", ConfidenceLevel.MEDIUM)
        confidence_statement = self.response_templates.get(f"confidence_{confidence}", "")
        
        # Add citation
        cited_statement = self.response_templates["citation"].format(
            file_name=citation_result.get("info", {}).get("file_name", "the implementation"),
            class_name=citation_result.get("info", {}).get("class_name", "the class"),
            method_name=citation_result.get("info", {}).get("method_name", "method"),
            statement=formatted_statement,
            citation=citation
        )
        
        # Add verification feedback if needed
        if not verification.get("verified", True):
            verification_feedback = self.response_templates["verification_failure"]
        else:
            verification_feedback = self.response_templates["verification_success"].format(
                message=verification.get("message", "")
            )
        
        # Add correction if needed
        correction_text = ""
        if needs_correction and correction and correction.get("has_correction", False):
            corrections = ", ".join(correction.get("corrections", []))
            correction_text = self.response_templates["accuracy_correction"].format(
                corrections=corrections
            )
            suggested_statement = correction.get("suggested_statement", "")
        
        # Combine all parts
        processed_text = cited_statement
        
        if confidence_statement:
            processed_text += f" {confidence_statement}"
            
        if verification_feedback:
            processed_text += f" {verification_feedback}"
            
        if correction_text:
            processed_text += f" {correction_text}"
            
        # Record in current session
        self.current_session["citations"].append(citation_result)
        self.current_session["verifications"].append(verification)
        self.current_session["accuracy_checks"].append(accuracy)
        
        if correction:
            self.current_session["corrections"].append(correction)
        
        # Return processed response
        return {
            "original_statement": statement,
            "processed_text": processed_text,
            "citation": citation_result,
            "verification": verification,
            "accuracy": accuracy,
            "correction": correction,
            "needs_correction": needs_correction
        }
    
    def format_technical_response_with_citations(self, response_elements: List[Dict[str, Any]]) -> str:
        """
        Format a complete technical response with multiple citations.
        
        Args:
            response_elements: List of processed response elements
            
        Returns:
            formatted_response: Formatted response text with citations
        """
        formatted_response = ""
        
        for i, element in enumerate(response_elements):
            formatted_response += element.get("processed_text", "")
            
            if i < len(response_elements) - 1:
                formatted_response += "\n\n"
        
        return formatted_response
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the current feedback loop session.
        
        Returns:
            metrics: Dictionary with session metrics
        """
        citations = len(self.current_session["citations"])
        verifications = sum(1 for v in self.current_session["verifications"] if v.get("verified", False))
        accurate_statements = sum(1 for a in self.current_session["accuracy_checks"] if a.get("accurate", False))
        corrections = len(self.current_session["corrections"])
        
        return {
            "total_citations": citations,
            "successful_verifications": verifications,
            "accurate_statements": accurate_statements,
            "corrections_made": corrections,
            "verification_rate": verifications / max(1, citations),
            "accuracy_rate": accurate_statements / max(1, citations)
        }
    
    def reset_session(self) -> None:
        """Reset the current feedback loop session."""
        self.current_session = {
            "citations": [],
            "verifications": [],
            "accuracy_checks": [],
            "corrections": []
        }


# Example implementation with integration to the introspection module
def setup_feedback_loop(introspection_module=None):
    """
    Set up the feedback loop manager with introspection module.
    
    Args:
        introspection_module: Optional SystemIntrospection instance
        
    Returns:
        feedback_manager: Configured FeedbackLoopManager
    """
    try:
        if introspection_module is None:
            # Try to import SystemIntrospection
            try:
                from system_introspection import SystemIntrospection
                introspection_module = SystemIntrospection()
            except ImportError:
                logger.warning("Could not import SystemIntrospection module")
        
        feedback_manager = FeedbackLoopManager(introspection_module)
        logger.info("Feedback loop manager initialized successfully")
        return feedback_manager
    except Exception as e:
        logger.error(f"Error setting up feedback loop: {e}")
        return FeedbackLoopManager()  # Fallback with no introspection

