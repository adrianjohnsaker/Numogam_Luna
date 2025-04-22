```python
"""
Feedback Loop with Source Citation

This module implements a citation protocol for Amelia to accurately reference
code implementations, provide confidence indicators, verify understanding,
and enable error correction when discussing technical details.
"""

import json
import re
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/citation_feedback.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('citation_feedback')


class ConfidenceLevel:
    """Enumeration of confidence levels for technical statements."""
    HIGH = "high"      # Direct code observation with high certainty
    MEDIUM = "medium"  # Inferred from code with reasonable certainty
    LOW = "low"        # Conceptual understanding with limited code evidence
    UNKNOWN = "unknown" # No specific code evidence


class CitationProtocol:
    """
    Implements a protocol for citing code sources in technical explanations.
    """
    
    def __init__(self, introspection_module=None):
        """
        Initialize the citation protocol.
        
        Args:
            introspection_module: Optional reference to the SystemIntrospection module
        """
        self.introspection = introspection_module
        self.citation_history = []
        self.verification_status = {}
        self.error_corrections = {}
    
    def format_citation(self, file_name: str, class_name: str = None, 
                       method_name: str = None, line_numbers: List[int] = None, 
                       confidence: str = ConfidenceLevel.MEDIUM) -> str:
        """
        Format a code citation in a standardized way.
        
        Args:
            file_name: Name of the source file
            class_name: Optional name of the class
            method_name: Optional name of the method
            line_numbers: Optional list of line numbers
            confidence: Confidence level in the citation
            
        Returns:
            formatted_citation: Formatted citation string
        """
        citation = f"[Source: {file_name}"
        
        if class_name:
            citation += f", class: {class_name}"
            
        if method_name:
            citation += f", method: {method_name}"
            
        if line_numbers:
            line_str = ", ".join(str(line) for line in line_numbers)
            citation += f", lines: {line_str}"
            
        citation += f", confidence: {confidence}]"
        
        return citation
    
    def generate_citation(self, concept: str, detail_type: str = None) -> Dict[str, Any]:
        """
        Generate a citation for a specific concept and detail type.
        
        Args:
            concept: The concept to cite (e.g., "AgentDecisionMaking")
            detail_type: Optional detail type to focus on (e.g., "methods", "parameters")
            
        Returns:
            citation_info: Dictionary with citation information
        """
        if not self.introspection:
            return {
                "error": "Introspection module not available",
                "citation": self.format_citation("unknown", confidence=ConfidenceLevel.LOW)
            }
        
        try:
            # Get implementation details
            details = self.introspection.get_implementation_details(concept, detail_type)
            
            if "error" in details:
                return {
                    "error": details["error"],
                    "citation": self.format_citation("unknown", confidence=ConfidenceLevel.LOW)
                }
            
            # Extract implementation information
            implementation = details.get("implementation", {})
            
            # Find file name
            file_name = None
            if "classes" in implementation:
                class_name = implementation["classes"][0] if implementation["classes"] else None
                if class_name:
                    class_info = self.introspection.code_parser.find_class_definition(class_name)
                    if class_info:
                        module_name = class_info.get("module")
                        if module_name in self.introspection.code_parser.module_paths:
                            file_name = os.path.basename(self.introspection.code_parser.module_paths[module_name])
            
            if not file_name:
                file_name = "adaptive_swarm_intelligence_algorithm_with_dynamic_communication.py"
            
            # Find class and method names
            class_name = None
            method_name = None
            line_numbers = None
            confidence = ConfidenceLevel.MEDIUM
            
            if "classes" in implementation and implementation["classes"]:
                class_name = implementation["classes"][0]
                confidence = ConfidenceLevel.HIGH
            
            if "methods" in implementation and implementation["methods"]:
                method_info = implementation["methods"][0]
                if "." in method_info:
                    class_name, method_name = method_info.split(".")
                    confidence = ConfidenceLevel.HIGH
            
            # Record the citation
            citation_info = {
                "concept": concept,
                "detail_type": detail_type,
                "file_name": file_name,
                "class_name": class_name,
                "method_name": method_name,
                "line_numbers": line_numbers,
                "confidence": confidence,
                "details": details
            }
            
            self.citation_history.append(citation_info)
            
            return {
                "citation": self.format_citation(
                    file_name, class_name, method_name, line_numbers, confidence
                ),
                "info": citation_info
            }
            
        except Exception as e:
            logger.error(f"Error generating citation for {concept}: {e}")
            return {
                "error": str(e),
                "citation": self.format_citation("unknown", confidence=ConfidenceLevel.LOW)
            }
    
    def verify_understanding(self, statement: str, concept: str) -> Dict[str, Any]:
        """
        Verify understanding of a code concept based on a textual explanation.
        
        Args:
            statement: The explanation to verify
            concept: The concept being explained
            
        Returns:
            verification: Dictionary with verification results
        """
        if not self.introspection:
            return {
                "verified": False,
                "confidence": ConfidenceLevel.LOW,
                "message": "Cannot verify without introspection module"
            }
        
        try:
            # Get implementation details
            details = self.introspection.get_implementation_details(concept)
            
            if "error" in details:
                return {
                    "verified": False,
                    "confidence": ConfidenceLevel.LOW,
                    "message": f"Cannot verify concept '{concept}': {details['error']}"
                }
            
            # Extract key terms from implementation
            key_terms = set()
            
            # Add class names
            if "classes" in details.get("implementation", {}):
                key_terms.update(details["implementation"]["classes"])
            
            # Add method names
            if "methods" in details.get("implementation", {}):
                for method in details["implementation"]["methods"]:
                    if "." in method:
                        _, method_name = method.split(".")
                        key_terms.add(method_name)
            
            # Add parameter names if available
            if "parameters" in details.get("implementation", {}):
                key_terms.update(details["implementation"]["parameters"].keys())
            
            # Check if key terms appear in the statement
            term_count = 0
            found_terms = []
            
            for term in key_terms:
                if term.lower() in statement.lower():
                    term_count += 1
                    found_terms.append(term)
            
            # Calculate verification score
            if not key_terms:
                score = 0
            else:
                score = term_count / len(key_terms)
            
            # Determine verification status and confidence
            if score >= 0.7:
                verified = True
                confidence = ConfidenceLevel.HIGH
                message = "Explanation includes most key implementation terms"
            elif score >= 0.4:
                verified = True
                confidence = ConfidenceLevel.MEDIUM
                message = "Explanation includes some key implementation terms"
            else:
                verified = False
                confidence = ConfidenceLevel.LOW
                message = "Explanation may be conceptual rather than implementation-specific"
            
            verification = {
                "verified": verified,
                "confidence": confidence,
                "score": score,
                "found_terms": found_terms,
                "total_terms": len(key_terms),
                "message": message
            }
            
            # Store verification status
            self.verification_status[concept] = verification
            
            return verification
            
        except Exception as e:
            logger.error(f"Error verifying understanding for {concept}: {e}")
            return {
                "verified": False,
                "confidence": ConfidenceLevel.LOW,
                "message": f"Error during verification: {str(e)}"
            }
    
    def check_statement_accuracy(self, statement: str, concept: str) -> Dict[str, Any]:
        """
        Check the accuracy of a technical statement against code implementation.
        
        Args:
            statement: The technical statement to check
            concept: The concept being discussed
            
        Returns:
            accuracy_check: Dictionary with accuracy check results
        """
        if not self.introspection:
            return {
                "accurate": False,
                "confidence": ConfidenceLevel.LOW,
                "message": "Cannot check accuracy without introspection module"
            }
        
        try:
            # Get implementation details
            details = self.introspection.get_implementation_details(concept)
            
            if "error" in details:
                return {
                    "accurate": False,
                    "confidence": ConfidenceLevel.LOW,
                    "message": f"Cannot check concept '{concept}': {details['error']}"
                }
            
            # Extract specific values to check for
            implementation = details.get("implementation", {})
            parameters = implementation.get("parameters", {})
            
            # Check for numeric value claims in the statement
            # This is a simple regex-based approach; more sophisticated NLP could be used
            numeric_claims = re.findall(r'(\w+)\s+(?:is|equals|has a value of|costs?|value)\s+(\d+\.?\d*)', statement)
            
            errors = []
            correct_values = []
            
            for term, claimed_value in numeric_claims:
                # Normalize term to match parameter keys
                normalized_term = term.lower()
                
                # Try to convert claimed value to float for comparison
                try:
                    claimed_value = float(claimed_value)
                except ValueError:
                    continue
                
                # Check if term exists in parameters
                found = False
                for param_key, param_value in parameters.items():
                    if normalized_term in param_key.lower():
                        found = True
                        # Compare values (with some tolerance for float comparison)
                        actual_value = float(param_value)
                        if abs(claimed_value - actual_value) < 0.001:
                            correct_values.append((term, claimed_value, param_key))
                        else:
                            errors.append((term, claimed_value, param_key, actual_value))
                
                if not found and claimed_value is not None:
                    # Term not found in parameters, might be an incorrect claim
                    errors.append((term, claimed_value, None, None))
            
            # Determine accuracy status
            if errors:
                accurate = False
                confidence = ConfidenceLevel.HIGH
                message = "Statement contains numeric errors"
                corrections = [
                    f"'{error[0]}' is claimed to be {error[1]}, but actual value is {error[3]}"
                    for error in errors if error[2] is not None
                ]
                
                # Record error for correction
                self.error_corrections[concept] = {
                    "statement": statement,
                    "errors": errors,
                    "corrections": corrections
                }
            else:
                accurate = True
                confidence = ConfidenceLevel.MEDIUM if correct_values else ConfidenceLevel.LOW
                message = "No numeric errors detected"
                corrections = []
            
            accuracy_check = {
                "accurate": accurate,
                "confidence": confidence,
                "message": message,
                "errors": errors,
                "correct_values": correct_values,
                "corrections": corrections
            }
            
            return accuracy_check
            
        except Exception as e:
            logger.error(f"Error checking statement accuracy for {concept}: {e}")
            return {
                "accurate": False,
                "confidence": ConfidenceLevel.LOW,
                "message": f"Error during accuracy check: {str(e)}"
            }
    
    def get_correction_suggestion(self, concept: str) -> Dict[str, Any]:
        """
        Get a suggestion for correcting an erroneous statement.
        
        Args:
            concept: The concept with an error to correct
            
        Returns:
            suggestion: Dictionary with correction suggestion
        """
        if concept not in self.error_corrections:
            return {
                "has_correction": False,
                "message": f"No recorded errors for concept '{concept}'"
            }
        
        error_info = self.error_corrections[concept]
        
        if not error_info.get("errors"):
            return {
                "has_correction": False,
                "message": "No specific errors to correct"
            }
        
        # Generate correction suggestion
        suggestion = {
            "has_correction": True,
            "original_statement": error_info["statement"],
            "errors": error_info["errors"],
            "corrections": error_info["corrections"],
            "suggested_statement": error_info["statement"]
        }
        
        # Try to create a corrected version of the statement
        corrected = error_info["statement"]
        for term, claimed, param_key, actual in error_info["errors"]:
            if actual is not None:
                # Replace claimed value with actual value
                pattern = r'(\b' + re.escape(term) + r'\s+(?:is|equals|has a value of|costs?|value)\s+)' + str(claimed)
                replacement = r'\1' + str(actual)
                corrected = re.sub(pattern, replacement, corrected)
        
        suggestion["suggested_statement"] = corrected
        
        return suggestion
