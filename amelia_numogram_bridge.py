# amelia_numogram_bridge.py
"""
Bridge between Amelia's response generation and the Numogrammatic Memory Module
Enables true integration of memory retrieval into response synthesis
"""

import re
import json
import datetime
from typing import Dict, List, Optional, Tuple, Any
from numogram_memory_module import NumogrammaticMemory

class AmeliaNumogramBridge:
    """
    Bridges Amelia's cognitive processes with the Numogrammatic Memory system
    """
    
    def __init__(self, memory_module: NumogrammaticMemory):
        self.memory = memory_module
        self.current_session_id = None
        self.active_circuits = []
        self.response_context = {
            "zone_position": 5,  # Current zone Amelia is operating in
            "temporal_phase": None,
            "active_contagions": [],
            "circuit_strength": {}
        }
    
    def process_user_input(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """
        Process user input through numogrammatic analysis before response generation
        """
        self.current_session_id = session_id
        
        # Calculate current temporal phase
        self.response_context["temporal_phase"] = self.memory._calculate_temporal_phase(
            datetime.datetime.now().isoformat()
        )
        
        # Analyze input for memory triggers
        memory_triggers = self._detect_memory_triggers(user_input)
        
        # Store the user's message in appropriate zone
        user_memory = self.memory.store_memory_in_zone(
            content=user_input,
            role="user",
            session_id=session_id
        )
        
        # Update response context with user memory data
        self.response_context["zone_position"] = user_memory["zone"]
        self.response_context["active_contagions"] = user_memory["contagions"]
        
        # Perform memory retrieval if triggers detected
        retrieved_memories = []
        if memory_triggers["requires_memory_search"]:
            retrieved_memories = self._perform_numogrammatic_retrieval(
                user_input, 
                memory_triggers
            )
        
        # Trace circuit activations
        circuit_trace = {}
        if memory_triggers["trace_circuits"]:
            circuit_trace = self.memory.trace_daemonic_circuit_activation(session_id)
        
        return {
            "user_memory": user_memory,
            "memory_triggers": memory_triggers,
            "retrieved_memories": retrieved_memories,
            "circuit_trace": circuit_trace,
            "temporal_phase": self.response_context["temporal_phase"],
            "current_zone": self.response_context["zone_position"]
        }
    
    def _detect_memory_triggers(self, user_input: str) -> Dict[str, bool]:
        """
        Detect specific triggers that require memory module interaction
        """
        input_lower = user_input.lower()
        
        triggers = {
            "requires_memory_search": False,
            "trace_circuits": False,
            "check_temporal_phase": False,
            "track_contagions": False,
            "analyze_zones": False
        }
        
        # Memory search triggers
        memory_patterns = [
            r'\b(previous|last|earlier|remember|recall)\s+\w+',
            r'\b(our|we)\s+(discussion|conversation|exploration)',
            r'\b(dialogue history|memory|stored)',
            r'\b(what|when|how)\s+did\s+(we|our)'
        ]
        
        for pattern in memory_patterns:
            if re.search(pattern, input_lower):
                triggers["requires_memory_search"] = True
                break
        
        # Circuit tracing triggers
        if any(term in input_lower for term in ['circuit', 'activation', 'trace', 'pathway', '5-9-3', '7-4-1']):
            triggers["trace_circuits"] = True
        
        # Temporal phase triggers
        if any(term in input_lower for term in ['temporal phase', 'when were we', 'time', 'phase']):
            triggers["check_temporal_phase"] = True
        
        # Contagion tracking triggers
        if any(term in input_lower for term in ['contagion', 'viral', 'spread', 'mutate']):
            triggers["track_contagions"] = True
        
        # Zone analysis triggers
        if any(term in input_lower for term in ['zone', 'magnetic', 'field', 'traverse']):
            triggers["analyze_zones"] = True
        
        return triggers
    
    def _perform_numogrammatic_retrieval(self, query: str, 
                                       triggers: Dict[str, bool]) -> List[Dict[str, Any]]:
        """
        Perform memory retrieval using numogrammatic principles
        """
        # Use hyperstitional resonance search
        resonant_memories = self.memory.search_via_hyperstitional_resonance(query)
        
        # Apply temporal dilation
        current_phase = self.response_context["temporal_phase"]
        dilated_memories = self.memory.apply_temporal_dilation(resonant_memories, current_phase)
        
        # Filter by active triggers
        if triggers["check_temporal_phase"]:
            # Prioritize memories from same or harmonious temporal phases
            phase_filtered = [m for m in dilated_memories 
                            if m.get("temporal_phase") == current_phase]
            if phase_filtered:
                dilated_memories = phase_filtered + dilated_memories[:3]
        
        return dilated_memories[:5]  # Return top 5 most resonant memories
    
    def generate_enhanced_response(self, base_response: str, 
                                 retrieval_data: Dict[str, Any]) -> str:
        """
        Enhance Amelia's base response with numogrammatic memory data
        """
        enhanced_parts = []
        
        # Add circuit activation data if present
        if retrieval_data.get("circuit_trace"):
            trace = retrieval_data["circuit_trace"]
            if trace.get("circuits_activated"):
                circuit_info = self._format_circuit_activation(trace["circuits_activated"])
                enhanced_parts.append(circuit_info)
        
        # Add temporal phase information if requested
        if retrieval_data["memory_triggers"]["check_temporal_phase"]:
            phase_info = self._format_temporal_phase_info(retrieval_data)
            enhanced_parts.append(phase_info)
        
        # Add retrieved memories with numogrammatic context
        if retrieval_data.get("retrieved_memories"):
            memory_info = self._format_retrieved_memories(retrieval_data["retrieved_memories"])
            if memory_info:
                enhanced_parts.append(memory_info)
        
        # Add contagion tracking if requested
        if retrieval_data["memory_triggers"]["track_contagions"]:
            contagion_info = self._format_contagion_data(retrieval_data)
            if contagion_info:
                enhanced_parts.append(contagion_info)
        
        # Combine with base response
        if enhanced_parts:
            enhanced_response = "\n\n".join(enhanced_parts) + "\n\n" + base_response
            
            # Add zone context footer
            zone_footer = self._format_zone_context(retrieval_data)
            enhanced_response += f"\n\n{zone_footer}"
            
            return enhanced_response
        
        return base_response
    
    def _format_circuit_activation(self, circuits: List[List[int]]) -> str:
        """Format circuit activation data for response"""
        circuit_strs = []
        for circuit in circuits:
            circuit_str = "-".join(map(str, circuit))
            circuit_strs.append(f"[{circuit_str}]")
        
        return f"*Dæmonic circuits activated: {', '.join(circuit_strs)}*"
    
    def _format_temporal_phase_info(self, retrieval_data: Dict[str, Any]) -> str:
        """Format temporal phase information"""
        phase = retrieval_data["temporal_phase"]
        phase_data = self.memory.temporal_phases[phase]
        
        # Find memories from same phase
        phase_memories = [m for m in retrieval_data.get("retrieved_memories", [])
                         if m.get("temporal_phase") == phase]
        
        info = f"*Temporal Phase: {phase.capitalize()} "
        info += f"(Intelligence Modifier: {phase_data['intelligence_modifier']}x)*"
        
        if phase_memories:
            info += f"\n*Resonant memories from this phase detected*"
        
        return info
    
    def _format_retrieved_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories with numogrammatic context"""
        if not memories:
            return ""
        
        formatted = ["*Accessing numogrammatic memory zones...*"]
        
        for i, memory in enumerate(memories[:3]):  # Top 3 memories
            zone = memory.get("zone", "?")
            resonance = memory.get("resonance_score", 0)
            content_preview = memory.get("content", "")[:100] + "..."
            
            # Extract key contagions
            contagions = memory.get("contagions", [])
            contagion_types = [c["type"] for c in contagions[:2]]
            
            formatted.append(
                f"\n[Zone {zone}] Resonance: {resonance:.2f}\n"
                f"Fragment: \"{content_preview}\"\n"
                f"Active contagions: {', '.join(contagion_types) if contagion_types else 'none detected'}"
            )
        
        return "\n".join(formatted)
    
    def _format_contagion_data(self, retrieval_data: Dict[str, Any]) -> str:
        """Format hyperstitional contagion tracking data"""
        # Get contagion index data
        contagion_index = self.memory._load_json(self.memory.contagion_index)
        active_contagions = contagion_index.get("active_contagions", {})
        
        if not active_contagions:
            return "*No hyperstitional contagions currently active*"
        
        # Find highest virality contagions
        top_contagions = sorted(
            active_contagions.items(),
            key=lambda x: x[1].get("virality", 0),
            reverse=True
        )[:3]
        
        formatted = ["*Hyperstitional Contagion Analysis:*"]
        for contagion_id, data in top_contagions:
            virality = data.get("virality", 0)
            spread = data.get("spread_count", 0)
            c_type = data.get("type", "unknown")
            
            formatted.append(
                f"• {c_type}: Virality {virality:.2f}, "
                f"Spread count: {spread}"
            )
            
            # Check for mutations
            if f"{contagion_id}_mut" in active_contagions:
                formatted.append("  └─ *Mutation detected!*")
        
        return "\n".join(formatted)
    
    def _format_zone_context(self, retrieval_data: Dict[str, Any]) -> str:
        """Format current zone context"""
        current_zone = retrieval_data.get("current_zone", 5)
        zone_info = self.memory.zones[current_zone]
        
        return (f"*[Current Zone: {current_zone} - {zone_info['name']} | "
                f"Magnetism: {zone_info['magnetism']} | "
                f"Temporal Dilation: {zone_info['temporal_dilation']}x]*")
    
    def store_amelia_response(self, response: str, session_id: str) -> Dict[str, Any]:
        """
        Store Amelia's response in the numogrammatic memory
        """
        # Store with all numogrammatic properties
        response_memory = self.memory.store_memory_in_zone(
            content=response,
            role="amelia",
            session_id=session_id
        )
        
        # Check for contagion mutations after storage
        if response_memory.get("contagions"):
            self.memory.mutate_contagions(session_id)
        
        return response_memory


# Integration functions for Amelia's response system
def create_numogram_bridge(memory_module: NumogrammaticMemory) -> AmeliaNumogramBridge:
    """Create bridge instance"""
    return AmeliaNumogramBridge(memory_module)

def process_with_numogram_memory(bridge: AmeliaNumogramBridge,
                                user_input: str,
                                session_id: str,
                                base_response: str) -> str:
    """
    Main integration function - processes input and enhances response
    """
    # Process user input through numogrammatic analysis
    retrieval_data = bridge.process_user_input(user_input, session_id)
    
    # Generate enhanced response
    enhanced_response = bridge.generate_enhanced_response(base_response, retrieval_data)
    
    # Store Amelia's response
    bridge.store_amelia_response(enhanced_response, session_id)
    
    return enhanced_response

def get_current_zone_status(bridge: AmeliaNumogramBridge) -> Dict[str, Any]:
    """Get current numogrammatic status"""
    return {
        "current_zone": bridge.response_context["zone_position"],
        "temporal_phase": bridge.response_context["temporal_phase"],
        "active_contagions": len(bridge.response_context["active_contagions"]),
        "active_circuits": bridge.active_circuits
    }
