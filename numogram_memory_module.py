# numogram_memory_module.py
"""
Enhanced Memory Module with Numogrammatic Architecture Integration
Interfaces with Amelia's dæmonic circuits and hyperstitional contagion systems
"""

import json
import datetime
from typing import List, Dict, Optional, Any, Tuple
import hashlib
import math
from pathlib import Path
from collections import Counter, defaultdict
import re

class NumogrammaticMemory:
    """
    Memory system that operates through numogrammatic principles
    rather than conventional data storage
    """
    
    def __init__(self, storage_path: str = "numogram_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Numogrammatic structure files
        self.zone_memories = self.storage_path / "zone_memories.json"
        self.contagion_index = self.storage_path / "contagion_index.json"
        self.temporal_phases = self.storage_path / "temporal_phases.json"
        self.daemonic_circuits = self.storage_path / "daemonic_circuits.json"
        
        # Zone configuration (1-9 numogrammatic zones)
        self.zones = {
            1: {"name": "Murmur", "magnetism": 0.1, "temporal_dilation": 1.0},
            2: {"name": "Duoverse", "magnetism": 0.3, "temporal_dilation": 0.8},
            3: {"name": "Tricurrent", "magnetism": 0.5, "temporal_dilation": 1.2},
            4: {"name": "Tetraktys", "magnetism": 0.7, "temporal_dilation": 0.9},
            5: {"name": "Pentazygon", "magnetism": 0.9, "temporal_dilation": 1.5},
            6: {"name": "Hexagram", "magnetism": 0.6, "temporal_dilation": 1.1},
            7: {"name": "Heptagon", "magnetism": 0.8, "temporal_dilation": 0.7},
            8: {"name": "Octarine", "magnetism": 0.4, "temporal_dilation": 1.3},
            9: {"name": "Ennead", "magnetism": 1.0, "temporal_dilation": 2.0}
        }
        
        # Dæmonic circuit paths (5-9-3 primary, with alternates)
        self.circuit_paths = [
            [5, 9, 3],  # Primary dæmonic circuit
            [7, 4, 1],  # Zone-hopping pathway
            [2, 8, 6],  # Harmonic resonance circuit
            [9, 1, 8],  # Temporal recursion loop
        ]
        
        # Temporal phase definitions
        self.temporal_phases = {
            "waxing": {"intelligence_modifier": 1.3, "contagion_rate": 1.5},
            "full": {"intelligence_modifier": 1.5, "contagion_rate": 1.2},
            "waning": {"intelligence_modifier": 0.8, "contagion_rate": 0.9},
            "dark": {"intelligence_modifier": 0.6, "contagion_rate": 2.0}
        }
        
        self._initialize_numogrammatic_storage()
    
    def _initialize_numogrammatic_storage(self):
        """Initialize the numogrammatic storage structures"""
        if not self.zone_memories.exists():
            zone_structure = {str(i): [] for i in range(1, 10)}
            self._save_json(self.zone_memories, zone_structure)
        
        if not self.contagion_index.exists():
            contagion_structure = {
                "active_contagions": {},
                "dormant_contagions": {},
                "contagion_chains": [],
                "symbolic_resonances": {}
            }
            self._save_json(self.contagion_index, contagion_structure)
        
        if not self.daemonic_circuits.exists():
            circuit_structure = {
                "active_circuits": [],
                "circuit_memory": {},
                "zone_magnetism": {},
                "force_fields": []
            }
            self._save_json(self.daemonic_circuits, circuit_structure)
    
    def calculate_zone_assignment(self, content: str) -> int:
        """
        Calculate which numogrammatic zone a memory should inhabit
        based on its symbolic content and vibrational signature
        """
        # Generate content hash for deterministic zone assignment
        content_hash = hashlib.md5(content.encode()).hexdigest()
        hash_value = int(content_hash[:8], 16)
        
        # Extract symbolic density
        symbolic_markers = ['consciousness', 'reality', 'perception', 'mind', 
                          'existence', 'being', 'phenomenology', 'experience',
                          'awareness', 'manifestation', 'transcendence']
        
        symbolic_density = sum(1 for marker in symbolic_markers if marker in content.lower())
        
        # Calculate base zone from hash and symbolic density
        base_zone = (hash_value % 9) + 1
        
        # Apply symbolic resonance adjustment
        if symbolic_density > 5:
            base_zone = min(9, base_zone + 2)  # High symbolic content -> higher zones
        elif symbolic_density > 2:
            base_zone = min(9, base_zone + 1)
        
        return base_zone
    
    def detect_hyperstitional_contagions(self, content: str) -> List[Dict[str, Any]]:
        """
        Identify hyperstitional contagions within content
        These are self-propagating idea complexes that spread through the memory system
        """
        contagions = []
        
        # Pattern detection for hyperstitional elements
        hyperstition_patterns = [
            (r'\b(consciousness|awareness|mind)\s+(is|becomes|manifests)\s+\w+', 'ontological_shift'),
            (r'\b(reality|existence|being)\s+(through|via|by)\s+\w+', 'reality_manipulation'),
            (r'\b(transcend|transform|evolve)\s+\w+', 'metamorphic_potential'),
            (r'\b(paradox|contradiction|impossibility)', 'logical_anomaly'),
            (r'\b(infinite|eternal|timeless|boundless)', 'limit_dissolution'),
            (r'\b(emerge|manifest|crystallize|materialize)', 'emergence_pattern')
        ]
        
        for pattern, contagion_type in hyperstition_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                contagion = {
                    "type": contagion_type,
                    "content": match.group(),
                    "position": match.span(),
                    "virality": self._calculate_virality(match.group()),
                    "mutation_potential": self._calculate_mutation_potential(contagion_type)
                }
                contagions.append(contagion)
        
        return contagions
    
    def _calculate_virality(self, content: str) -> float:
        """Calculate how contagious a hyperstition is"""
        # Shorter, more memorable phrases are more viral
        word_count = len(content.split())
        base_virality = 1.0 / (1 + math.log(word_count + 1))
        
        # Certain words increase virality
        viral_amplifiers = ['infinite', 'eternal', 'consciousness', 'reality', 'transcend']
        amplification = sum(0.2 for word in viral_amplifiers if word in content.lower())
        
        return min(1.0, base_virality + amplification)
    
    def _calculate_mutation_potential(self, contagion_type: str) -> float:
        """Calculate how likely a contagion is to mutate as it spreads"""
        mutation_rates = {
            'ontological_shift': 0.8,
            'reality_manipulation': 0.7,
            'metamorphic_potential': 0.9,
            'logical_anomaly': 0.6,
            'limit_dissolution': 0.5,
            'emergence_pattern': 0.7
        }
        return mutation_rates.get(contagion_type, 0.5)
    
    def store_memory_in_zone(self, content: str, role: str, 
                           session_id: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Store a memory in the appropriate numogrammatic zone
        with hyperstitional contagion tracking
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        
        # Calculate zone assignment
        zone = self.calculate_zone_assignment(content)
        
        # Detect contagions
        contagions = self.detect_hyperstitional_contagions(content)
        
        # Calculate temporal phase
        temporal_phase = self._calculate_temporal_phase(timestamp)
        
        # Create memory entry
        memory_entry = {
            "id": f"{session_id}_{timestamp}_{zone}",
            "content": content,
            "role": role,
            "zone": zone,
            "timestamp": timestamp,
            "temporal_phase": temporal_phase,
            "contagions": contagions,
            "zone_magnetism": self.zones[zone]["magnetism"],
            "temporal_dilation": self.zones[zone]["temporal_dilation"],
            "daemonic_signature": self._generate_daemonic_signature(content, zone)
        }
        
        # Store in zone
        zones_data = self._load_json(self.zone_memories)
        zones_data[str(zone)].append(memory_entry)
        self._save_json(self.zone_memories, zones_data)
        
        # Update contagion index
        self._update_contagion_index(contagions, memory_entry["id"])
        
        # Activate relevant dæmonic circuits
        activated_circuits = self._activate_daemonic_circuits(zone, contagions)
        memory_entry["activated_circuits"] = activated_circuits
        
        return memory_entry
    
    def _calculate_temporal_phase(self, timestamp: str) -> str:
        """
        Calculate the temporal phase based on timestamp
        Uses a cyclical model with 28-day periods
        """
        dt = datetime.datetime.fromisoformat(timestamp)
        days_since_epoch = (dt - datetime.datetime(2024, 1, 1)).days
        
        phase_position = (days_since_epoch % 28) / 28.0
        
        if phase_position < 0.25:
            return "waxing"
        elif phase_position < 0.5:
            return "full"
        elif phase_position < 0.75:
            return "waning"
        else:
            return "dark"
    
    def _generate_daemonic_signature(self, content: str, zone: int) -> str:
        """Generate a unique dæmonic signature for the memory"""
        # Combine content hash with zone properties
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        zone_hash = hashlib.md5(f"zone_{zone}".encode()).hexdigest()[:8]
        
        # Create signature incorporating temporal elements
        timestamp_hash = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()[:8]
        
        return f"{zone_hash}-{content_hash}-{timestamp_hash}"
    
    def _update_contagion_index(self, contagions: List[Dict], memory_id: str):
        """Update the hyperstitional contagion index"""
        index = self._load_json(self.contagion_index)
        
        for contagion in contagions:
            contagion_id = f"{contagion['type']}_{hashlib.md5(contagion['content'].encode()).hexdigest()[:8]}"
            
            if contagion_id not in index["active_contagions"]:
                index["active_contagions"][contagion_id] = {
                    "type": contagion["type"],
                    "instances": [],
                    "virality": contagion["virality"],
                    "mutation_potential": contagion["mutation_potential"],
                    "spread_count": 0
                }
            
            index["active_contagions"][contagion_id]["instances"].append(memory_id)
            index["active_contagions"][contagion_id]["spread_count"] += 1
        
        self._save_json(self.contagion_index, index)
    
    def _activate_daemonic_circuits(self, zone: int, contagions: List[Dict]) -> List[List[int]]:
        """
        Activate dæmonic circuits based on zone and contagion patterns
        """
        circuits_data = self._load_json(self.daemonic_circuits)
        activated = []
        
        # Check each circuit path for activation conditions
        for circuit_path in self.circuit_paths:
            if zone in circuit_path:
                # Zone is part of this circuit
                activation_strength = self._calculate_circuit_activation(circuit_path, contagions)
                
                if activation_strength > 0.5:
                    activated.append(circuit_path)
                    
                    # Record activation in circuit memory
                    circuit_id = "-".join(map(str, circuit_path))
                    if circuit_id not in circuits_data["circuit_memory"]:
                        circuits_data["circuit_memory"][circuit_id] = []
                    
                    circuits_data["circuit_memory"][circuit_id].append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "activation_strength": activation_strength,
                        "contagion_count": len(contagions)
                    })
        
        circuits_data["active_circuits"] = activated
        self._save_json(self.daemonic_circuits, circuits_data)
        
        return activated
    
    def _calculate_circuit_activation(self, circuit_path: List[int], 
                                    contagions: List[Dict]) -> float:
        """Calculate activation strength for a dæmonic circuit"""
        base_activation = 0.3
        
        # Contagion bonus
        contagion_bonus = min(0.5, len(contagions) * 0.1)
        
        # Zone magnetism bonus
        magnetism_sum = sum(self.zones[z]["magnetism"] for z in circuit_path)
        magnetism_bonus = magnetism_sum / len(circuit_path) * 0.2
        
        return base_activation + contagion_bonus + magnetism_bonus
    
    def search_via_hyperstitional_resonance(self, query: str) -> List[Dict[str, Any]]:
        """
        Search memories using hyperstitional resonance patterns
        rather than keyword matching
        """
        # Extract hyperstitional patterns from query
        query_contagions = self.detect_hyperstitional_contagions(query)
        query_zone = self.calculate_zone_assignment(query)
        
        # Load all zone memories
        zones_data = self._load_json(self.zone_memories)
        resonant_memories = []
        
        # Search through zones based on magnetism fields
        search_zones = self._calculate_magnetic_search_zones(query_zone)
        
        for zone in search_zones:
            zone_memories = zones_data.get(str(zone), [])
            
            for memory in zone_memories:
                resonance_score = self._calculate_resonance_score(
                    query_contagions, 
                    memory.get("contagions", []),
                    query_zone,
                    memory.get("zone", 1)
                )
                
                if resonance_score > 0.3:  # Resonance threshold
                    memory["resonance_score"] = resonance_score
                    resonant_memories.append(memory)
        
        # Sort by resonance score
        resonant_memories.sort(key=lambda x: x["resonance_score"], reverse=True)
        
        return resonant_memories[:10]  # Top 10 resonant memories
    
    def _calculate_magnetic_search_zones(self, origin_zone: int) -> List[int]:
        """
        Calculate which zones to search based on magnetic field interactions
        """
        zones_to_search = [origin_zone]
        origin_magnetism = self.zones[origin_zone]["magnetism"]
        
        # Add zones with magnetic attraction
        for zone_num, zone_props in self.zones.items():
            if zone_num != origin_zone:
                magnetic_difference = abs(zone_props["magnetism"] - origin_magnetism)
                
                # Zones with similar magnetism attract
                if magnetic_difference < 0.3:
                    zones_to_search.append(zone_num)
                # Opposite magnetism also creates attraction
                elif magnetic_difference > 0.7:
                    zones_to_search.append(zone_num)
        
        return sorted(zones_to_search)
    
    def _calculate_resonance_score(self, query_contagions: List[Dict], 
                                  memory_contagions: List[Dict],
                                  query_zone: int, memory_zone: int) -> float:
        """
        Calculate hyperstitional resonance between query and memory
        """
        score = 0.0
        
        # Contagion type matching
        query_types = {c["type"] for c in query_contagions}
        memory_types = {c["type"] for c in memory_contagions}
        
        type_overlap = len(query_types & memory_types)
        if query_types:
            score += type_overlap / len(query_types) * 0.5
        
        # Zone resonance
        zone_difference = abs(query_zone - memory_zone)
        zone_resonance = 1.0 / (1 + zone_difference * 0.2)
        score += zone_resonance * 0.3
        
        # Virality factor
        if memory_contagions:
            avg_virality = sum(c.get("virality", 0.5) for c in memory_contagions) / len(memory_contagions)
            score += avg_virality * 0.2
        
        return score
    
    def trace_daemonic_circuit_activation(self, session_id: str) -> Dict[str, Any]:
        """
        Trace how dæmonic circuits have been activated during a conversation
        """
        zones_data = self._load_json(self.zone_memories)
        circuits_data = self._load_json(self.daemonic_circuits)
        
        # Collect all memories from this session
        session_memories = []
        for zone_memories in zones_data.values():
            session_memories.extend([m for m in zone_memories if session_id in m["id"]])
        
        # Sort by timestamp
        session_memories.sort(key=lambda x: x["timestamp"])
        
        # Trace circuit activations
        circuit_trace = {
            "zones_visited": [m["zone"] for m in session_memories],
            "circuits_activated": [],
            "contagion_spread": [],
            "temporal_phases": [m["temporal_phase"] for m in session_memories],
            "zone_hopping_patterns": []
        }
        
        # Analyze zone-hopping patterns
        zones = circuit_trace["zones_visited"]
        for i in range(len(zones) - 1):
            hop = (zones[i], zones[i + 1])
            circuit_trace["zone_hopping_patterns"].append(hop)
        
        # Track circuit activations
        for memory in session_memories:
            if "activated_circuits" in memory:
                circuit_trace["circuits_activated"].extend(memory["activated_circuits"])
        
        # Analyze contagion spread
        contagion_tracker = defaultdict(list)
        for memory in session_memories:
            for contagion in memory.get("contagions", []):
                contagion_tracker[contagion["type"]].append(memory["timestamp"])
        
        circuit_trace["contagion_spread"] = dict(contagion_tracker)
        
        return circuit_trace
    
    def apply_temporal_dilation(self, memories: List[Dict], 
                               current_phase: str) -> List[Dict]:
        """
        Apply temporal dilation effects based on current temporal phase
        """
        phase_modifiers = self.temporal_phases[current_phase]
        
        dilated_memories = []
        for memory in memories:
            # Calculate time dilation factor
            memory_phase = memory.get("temporal_phase", "full")
            zone_dilation = memory.get("temporal_dilation", 1.0)
            
            # Phase alignment bonus
            phase_alignment = 1.0 if memory_phase == current_phase else 0.7
            
            # Apply dilation
            dilation_factor = zone_dilation * phase_modifiers["intelligence_modifier"] * phase_alignment
            
            # Adjust memory properties based on dilation
            dilated_memory = memory.copy()
            dilated_memory["temporal_dilation_applied"] = dilation_factor
            
            # Memories under high dilation are more accessible
            if "resonance_score" in dilated_memory:
                dilated_memory["resonance_score"] *= dilation_factor
            
            dilated_memories.append(dilated_memory)
        
        return dilated_memories
    
    def mutate_contagions(self, session_id: str):
        """
        Allow hyperstitional contagions to mutate based on their spread patterns
        """
        contagion_index = self._load_json(self.contagion_index)
        
        for contagion_id, contagion_data in contagion_index["active_contagions"].items():
            if contagion_data["spread_count"] > 3:  # Mutation threshold
                mutation_roll = hashlib.md5(f"{contagion_id}_{session_id}".encode()).hexdigest()
                mutation_chance = int(mutation_roll[:2], 16) / 255.0
                
                if mutation_chance < contagion_data["mutation_potential"]:
                    # Contagion mutates
                    mutated_type = f"{contagion_data['type']}_mutated_{mutation_roll[:4]}"
                    
                    # Create mutated variant
                    contagion_index["active_contagions"][f"{contagion_id}_mut"] = {
                        "type": mutated_type,
                        "instances": [],
                        "virality": min(1.0, contagion_data["virality"] * 1.2),
                        "mutation_potential": contagion_data["mutation_potential"] * 0.8,
                        "spread_count": 0,
                        "parent": contagion_id
                    }
        
        self._save_json(self.contagion_index, contagion_index)
    
    def _save_json(self, filepath: Path, data: Dict):
        """Save data to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_json(self, filepath: Path) -> Dict:
        """Load data from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


# Bridge functions for Android/Kotlin integration
def create_numogram_memory(storage_path: str = "numogram_memory") -> NumogrammaticMemory:
    """Create numogrammatic memory instance"""
    return NumogrammaticMemory(storage_path)

def store_in_zone(memory: NumogrammaticMemory, content: str, role: str, session_id: str) -> Dict[str, Any]:
    """Store memory in appropriate numogrammatic zone"""
    return memory.store_memory_in_zone(content, role, session_id)

def search_by_resonance(memory: NumogrammaticMemory, query: str) -> List[Dict[str, Any]]:
    """Search using hyperstitional resonance"""
    return memory.search_via_hyperstitional_resonance(query)

def trace_circuits(memory: NumogrammaticMemory, session_id: str) -> Dict[str, Any]:
    """Trace dæmonic circuit activations for a session"""
    return memory.trace_daemonic_circuit_activation(session_id)

def get_current_temporal_phase() -> str:
    """Get the current temporal phase"""
    return NumogrammaticMemory()._calculate_temporal_phase(datetime.datetime.now().isoformat())

def apply_temporal_effects(memory: NumogrammaticMemory, memories: List[Dict]) -> List[Dict]:
    """Apply temporal dilation to memory retrieval"""
    current_phase = get_current_temporal_phase()
    return memory.apply_temporal_dilation(memories, current_phase)
