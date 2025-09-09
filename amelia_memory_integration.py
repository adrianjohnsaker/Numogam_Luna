# amelia_memory_integration.py
"""
Integration layer between Amelia AI and the Memory Module
This module intercepts Amelia's responses and enhances them with memory context
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import json

class AmeliaMemoryIntegration:
    """
    Integrates memory retrieval into Amelia's response generation
    """
    
    def __init__(self, memory_module):
        self.memory = memory_module
        self.memory_triggers = [
            r'\b(remember|recall|previous|earlier|last time|we discussed|we talked)\b',
            r'\b(our conversation|our discussion|we explored|we examined)\b',
            r'\b(you mentioned|I mentioned|you said|I asked)\b',
            r'\b(continuing|continue our|back to|return to)\b',
            r'\b(history|past|before|previously)\b'
        ]
        self.context_window = 10  # Number of recent messages to consider
        
    def should_search_memory(self, user_input: str) -> bool:
        """
        Determine if the user's input requires memory search
        """
        input_lower = user_input.lower()
        
        # Check for explicit memory triggers
        for pattern in self.memory_triggers:
            if re.search(pattern, input_lower, re.IGNORECASE):
                return True
                
        # Check for topic continuity phrases
        continuity_phrases = ['as we', 'like we', 'that we', 'when we', 'how we']
        if any(phrase in input_lower for phrase in continuity_phrases):
            return True
            
        return False
    
    def extract_search_terms(self, user_input: str) -> List[str]:
        """
        Extract relevant search terms from user input
        """
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 
                     'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should',
                     'may', 'might', 'can', 'remember', 'recall', 'our', 'we'}
        
        # Extract meaningful words
        words = re.findall(r'\b\w+\b', user_input.lower())
        search_terms = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Look for quoted phrases
        quoted = re.findall(r'"([^"]+)"', user_input) + re.findall(r"'([^']+)'", user_input)
        search_terms.extend(quoted)
        
        # Extract capitalized terms (likely important concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_input)
        search_terms.extend([term.lower() for term in capitalized])
        
        return list(set(search_terms))[:5]  # Limit to 5 terms
    
    def search_relevant_memories(self, user_input: str) -> Dict[str, Any]:
        """
        Search for relevant past conversations
        """
        search_terms = self.extract_search_terms(user_input)
        
        results = {
            'found_memories': False,
            'conversations': [],
            'relevant_quotes': [],
            'topics': [],
            'context_summary': None
        }
        
        if not search_terms:
            return results
        
        # Search for each term
        all_results = []
        for term in search_terms:
            try:
                term_results = self.memory.search_by_keyword(term)
                all_results.extend(term_results)
            except:
                continue
        
        # Deduplicate and sort by relevance
        seen_sessions = set()
        unique_results = []
        for result in all_results:
            session_id = result.get('session_id')
            if session_id and session_id not in seen_sessions:
                seen_sessions.add(session_id)
                unique_results.append(result)
        
        if unique_results:
            results['found_memories'] = True
            results['conversations'] = unique_results[:3]  # Top 3 conversations
            
            # Extract relevant quotes
            for conv in unique_results[:2]:
                for msg in conv.get('matching_messages', [])[:2]:
                    results['relevant_quotes'].append({
                        'role': msg['role'],
                        'content': msg['content'][:200],  # Truncate long messages
                        'session_id': conv['session_id']
                    })
            
            # Extract topics from found conversations
            topics_set = set()
            for conv in unique_results:
                if 'summary' in conv and conv['summary']:
                    # Extract topics from summary
                    summary_words = re.findall(r'\b\w+\b', conv['summary'].lower())
                    meaningful = [w for w in summary_words if len(w) > 4][:3]
                    topics_set.update(meaningful)
            
            results['topics'] = list(topics_set)[:5]
        
        return results
    
    def get_current_context(self) -> List[Dict[str, str]]:
        """
        Get recent conversation context
        """
        try:
            return self.memory.get_conversation_context(message_limit=self.context_window)
        except:
            return []
    
    def enhance_response_with_memory(self, user_input: str, 
                                   amelia_base_response: str) -> str:
        """
        Enhance Amelia's response with memory context
        """
        # Check if memory search is needed
        if not self.should_search_memory(user_input):
            return amelia_base_response
        
        # Search memories
        memory_results = self.search_relevant_memories(user_input)
        
        if not memory_results['found_memories']:
            # If asking about memories but none found
            if re.search(r'\b(remember|recall|previous)\b', user_input.lower()):
                return (f"I don't have any recorded memories of discussing that specific topic. "
                       f"{amelia_base_response}")
            return amelia_base_response
        
        # Build enhanced response
        enhanced_response = []
        
        # Add memory context introduction
        conv_count = len(memory_results['conversations'])
        if conv_count == 1:
            enhanced_response.append("I found a relevant conversation in my memory.")
        else:
            enhanced_response.append(f"I found {conv_count} relevant conversations in my memory.")
        
        # Add specific quotes if highly relevant
        relevant_quotes = memory_results['relevant_quotes']
        if relevant_quotes:
            enhanced_response.append("\nFrom our previous discussions:")
            for quote in relevant_quotes[:2]:
                if quote['role'] == 'user':
                    enhanced_response.append(f"• You mentioned: \"{quote['content']}...\"")
                else:
                    enhanced_response.append(f"• I responded: \"{quote['content']}...\"")
        
        # Add the base response
        enhanced_response.append(f"\n{amelia_base_response}")
        
        # Add topics for context
        if memory_results['topics']:
            topics_str = ", ".join(memory_results['topics'][:3])
            enhanced_response.append(f"\n(Related topics from our history: {topics_str})")
        
        return "\n".join(enhanced_response)
    
    def process_amelia_response(self, user_input: str, 
                              amelia_response: str,
                              save_to_memory: bool = True) -> str:
        """
        Main integration point - process Amelia's response with memory enhancement
        """
        # Get current context for continuity
        current_context = self.get_current_context()
        
        # Enhance response with memory if needed
        enhanced_response = self.enhance_response_with_memory(user_input, amelia_response)
        
        # Save the exchange to memory if requested
        if save_to_memory and self.memory.current_session_id:
            self.memory.add_message("user", user_input)
            self.memory.add_message("amelia", enhanced_response)
        
        return enhanced_response


# Bridge functions for Chaquopy
def create_integration(memory_module) -> AmeliaMemoryIntegration:
    """Create memory integration instance"""
    return AmeliaMemoryIntegration(memory_module)

def process_with_memory(integration: AmeliaMemoryIntegration,
                       user_input: str,
                       amelia_response: str,
                       save_to_memory: bool = True) -> str:
    """Process Amelia's response with memory enhancement"""
    return integration.process_amelia_response(user_input, amelia_response, save_to_memory)

def check_memory_needed(integration: AmeliaMemoryIntegration, user_input: str) -> bool:
    """Check if memory search is needed for this input"""
    return integration.should_search_memory(user_input)

def search_memories_for_input(integration: AmeliaMemoryIntegration, 
                            user_input: str) -> Dict[str, Any]:
    """Search memories relevant to user input"""
    return integration.search_relevant_memories(user_input)

def get_conversation_context(integration: AmeliaMemoryIntegration) -> List[Dict[str, str]]:
    """Get current conversation context"""
    return integration.get_current_context()
