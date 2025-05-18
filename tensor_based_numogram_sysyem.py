
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import uuid
import datetime

class TensorBasedNumogramSystem:
    """
    Enhanced numogram system with dynamic zone relationships 
    represented using tensors, enabling higher-dimensional connections
    """
    
    def __init__(self, base_numogram_system, dimension=3, learning_rate=0.01):
        # Store reference to the base numogram system
        self.base_numogram = base_numogram_system
        
        # Define dimensionality
        self.dimension = dimension
        
        # Initialize tensor-based representation
        self.zone_tensors = self._initialize_zone_tensors()
        
        # Transition tensor (represents transition probabilities)
        self.transition_tensor = self._initialize_transition_tensor()
        
        # Learning parameters
        self.learning_rate = learning_rate
        
        # Dynamic magnetism parameters (attractions between zones)
        self.zone_magnetism = self._initialize_zone_magnetism()
        
        # Higher-dimensional connections
        self.hyperedges = self._initialize_hyperedges()
        
        # Transition history
        self.transition_history = []
        
        # Evolution parameters
        self.evolution_step = 0
        self.stability_threshold = 0.01  # Threshold for zone stability
    
    def _initialize_zone_tensors(self):
        """Initialize tensor representations for each zone"""
        zone_tensors = {}
        
        # For 9 numogram zones
        for i in range(1, 10):
            zone_id = str(i)
            
            # Create tensor representation
            # Position in n-dimensional space (initialized to reflect traditional numogram)
            if self.dimension == 2:
                # 2D layout (traditional plane)
                positions = {
                    "1": [0.0, 0.0],     # Center
                    "2": [1.0, 0.0],     # Right
                    "3": [2.0, 0.0],     # Far right
                    "4": [0.0, 1.0],     # Top
                    "5": [1.0, 1.0],     # Top right
                    "6": [2.0, 1.0],     # Far top right
                    "7": [0.0, 2.0],     # Far top
                    "8": [1.0, 2.0],     # Far top right
                    "9": [2.0, 2.0]      # Far corner
                }
                position = torch.tensor(positions.get(zone_id, [0.0, 0.0]), dtype=torch.float32)
                
            elif self.dimension == 3:
                # 3D layout
                positions = {
                    "1": [0.0, 0.0, 0.0],    # Origin
                    "2": [1.0, 0.0, 0.0],    # X-axis
                    "3": [0.5, 0.866, 0.0],  # 120° in XY plane
                    "4": [0.0, 0.0, 1.0],    # Z-axis
                    "5": [0.5, 0.5, 0.5],    # Center point
                    "6": [1.0, 0.0, 1.0],    # X+Z corner
                    "7": [0.5, 0.866, 1.0],  # Y+Z corner
                    "8": [0.0, 0.0, 2.0],    # Far Z
                    "9": [1.0, 1.0, 1.0]     # Diagonal corner
                }
                position = torch.tensor(positions.get(zone_id, [0.0, 0.0, 0.0]), dtype=torch.float32)
                
            else:
                # Higher dimensions - use spectral embedding in hypercube
                position = torch.zeros(self.dimension, dtype=torch.float32)
                
                # Map zone to position based on binary representation
                binary = bin(int(zone_id))[2:].zfill(self.dimension)
                for j in range(min(len(binary), self.dimension)):
                    if binary[j] == "1":
                        position[j] = 1.0
                    else:
                        position[j] = 0.0
            
            # Create full tensor representation with additional properties
            zone_tensors[zone_id] = {
                "position": position,
                "energy": 1.0,                      # Zone energy level
                "resonance": torch.rand(5),         # Resonance profile
                "stability": 1.0,                   # How stable the zone is
                "flux": 0.0,                        # Rate of change
                "digital_signature": torch.tensor(self._calculate_zone_signature(zone_id))
            }
        
        return zone_tensors
    
    def _calculate_zone_signature(self, zone_id):
        """Calculate digital signature for a zone based on numogram principles"""
        zone_int = int(zone_id)
        
        # Generate decimal expansion of 1/n
        signature = []
        remainder = 1
        for _ in range(5):  # Generate 5 digits
            remainder = (remainder * 10) % zone_int
            digit = (remainder * 10) // zone_int
            signature.append(digit)
        
        return signature
    
    def _initialize_transition_tensor(self):
        """
        Initialize tensor representing transition probabilities between zones
        For higher dimensions, this becomes an N-dimensional tensor
        """
        # Start with base transition matrix from traditional numogram
        base_transitions = self.base_numogram.TRANSITION_MATRIX
        
        # Create tensor representation
        if self.dimension <= 3:
            # For lower dimensions, use a simple matrix (2D tensor)
            transition_tensor = torch.zeros((9, 9), dtype=torch.float32)
            
            for i in range(1, 10):
                source = str(i)
                for j in range(1, 10):
                    target = str(j)
                    # Get transition probability
                    prob = base_transitions.get(source, {}).get(target, 0.0)
                    transition_tensor[i-1, j-1] = prob
        else:
            # For higher dimensions, use a higher-order tensor
            # Each dimension represents different aspects of the transition
            tensor_shape = tuple([9] * self.dimension)
            transition_tensor = torch.zeros(tensor_shape, dtype=torch.float32)
            
            # Initialize based on base transitions
            for i in range(1, 10):
                source = str(i)
                for j in range(1, 10):
                    target = str(j)
                    # Get transition probability
                    prob = base_transitions.get(source, {}).get(target, 0.0)
                    
                    # Create index tuple
                    idx = [0] * self.dimension
                    idx[0] = i - 1
                    idx[1] = j - 1
                    
                    # Set value in tensor
                    transition_tensor[tuple(idx)] = prob
        
        return transition_tensor
    
    def _initialize_zone_magnetism(self):
        """Initialize zone magnetism (attraction/repulsion between zones)"""
        magnetism = {}
        
        # Define initial magnetism based on numogram structure
        for i in range(1, 10):
            source = str(i)
            magnetism[source] = {}
            
            for j in range(1, 10):
                target = str(j)
                
                # Skip self
                if source == target:
                    continue
                
                # Calculate initial magnetism based on digital sums
                source_sum = sum(int(d) for d in source)
                target_sum = sum(int(d) for d in target)
                
                # Complementary zones have strong attraction
                if (source_sum + target_sum) % 9 == 0:
                    magnetism[source][target] = 0.8
                # Harmonic zones (digital roots sum to 9) have moderate attraction
                elif (int(source) + int(target)) % 9 == 0:
                    magnetism[source][target] = 0.5
                # Dissonant zones have slight repulsion
                else:
                    magnetism[source][target] = -0.2 + (0.4 * random.random())  # Random between -0.2 and 0.2
        
        return magnetism
    
    def _initialize_hyperedges(self):
        """Initialize higher-order connections (hyperedges) between zones"""
        hyperedges = []
        
        # Define traditional triad connections
        triads = [
            {"zones": ["1", "2", "4"], "strength": 0.7, "name": "Unity Triad"},
            {"zones": ["2", "3", "6"], "strength": 0.7, "name": "Division Triad"},
            {"zones": ["3", "7", "9"], "strength": 0.7, "name": "Synthesis Triad"},
            {"zones": ["4", "5", "7"], "strength": 0.7, "name": "Structure Triad"},
            {"zones": ["5", "6", "8"], "strength": 0.7, "name": "Transformation Triad"},
            {"zones": ["1", "5", "9"], "strength": 0.8, "name": "Diagonal Triad"},
            {"zones": ["2", "5", "8"], "strength": 0.7, "name": "Oscillator Triad"},
            {"zones": ["3", "5", "7"], "strength": 0.7, "name": "Resonator Triad"},
            {"zones": ["4", "5", "6"], "strength": 0.7, "name": "Horizontal Triad"}
        ]
        
        # Add triads as hyperedges
        for triad in triads:
            hyperedges.append({
                "id": str(uuid.uuid4()),
                "zones": triad["zones"],
                "strength": triad["strength"],
                "name": triad["name"],
                "order": len(triad["zones"]),
                "type": "triad",
                "active": True
            })
        
        # Add higher-order connections (tetrad, etc.) if dimension > 3
        if self.dimension > 3:
            tetrads = [
                {"zones": ["1", "3", "7", "9"], "strength": 0.6, "name": "Corner Tetrad"},
                {"zones": ["2", "4", "6", "8"], "strength": 0.6, "name": "Cross Tetrad"},
                {"zones": ["1", "2", "3", "5"], "strength": 0.5, "name": "Forward Tetrad"},
                {"zones": ["5", "7", "8", "9"], "strength": 0.5, "name": "Backward Tetrad"}
            ]
            
            for tetrad in tetrads:
                hyperedges.append({
                    "id": str(uuid.uuid4()),
                    "zones": tetrad["zones"],
                    "strength": tetrad["strength"],
                    "name": tetrad["name"],
                    "order": len(tetrad["zones"]),
                    "type": "tetrad",
                    "active": True
                })
        
        return hyperedges
    
    def get_zone_tensor(self, zone_id):
        """Get tensor representation for a zone"""
        return self.zone_tensors.get(zone_id, None)
    
    def calculate_zone_distance(self, zone1, zone2):
        """Calculate distance between two zones in tensor space"""
        tensor1 = self.zone_tensors.get(zone1, None)
        tensor2 = self.zone_tensors.get(zone2, None)
        
        if tensor1 is None or tensor2 is None:
            return float('inf')
            
        # Calculate Euclidean distance between positions
        pos1 = tensor1["position"]
        pos2 = tensor2["position"]
        
        return torch.norm(pos1 - pos2).item()
    
    def activate_hyperedge(self, zones):
        """Activate a hyperedge connection between multiple zones"""
        # Check if a matching hyperedge exists
        matching_edge = None
        for edge in self.hyperedges:
            if set(edge["zones"]) == set(zones):
                matching_edge = edge
                break
        
        # If no matching edge, create one
        if matching_edge is None:
            zones_sorted = sorted(zones)
            edge_id = str(uuid.uuid4())
            name = f"Dynamic Edge {edge_id[:6]}"
            
            new_edge = {
                "id": edge_id,
                "zones": zones_sorted,
                "strength": 0.5,  # Initial strength
                "name": name,
                "order": len(zones),
                "type": "dynamic",
                "active": True,
                "created": datetime.datetime.utcnow().isoformat()
            }
            
            self.hyperedges.append(new_edge)
            matching_edge = new_edge
        
        # Activate the edge
        matching_edge["active"] = True
        
        # Strengthen the edge
        matching_edge["strength"] = min(1.0, matching_edge["strength"] + 0.1)
        
        # Increase magnetism between all zone pairs in the hyperedge
        for i, zone1 in enumerate(zones):
            for zone2 in zones[i+1:]:
                current_mag = self.zone_magnetism.get(zone1, {}).get(zone2, 0.0)
                new_mag = min(1.0, current_mag + 0.1)
                
                # Update both directions
                if zone1 not in self.zone_magnetism:
                    self.zone_magnetism[zone1] = {}
                if zone2 not in self.zone_magnetism:
                    self.zone_magnetism[zone2] = {}
                    
                self.zone_magnetism[zone1][zone2] = new_mag
                self.zone_magnetism[zone2][zone1] = new_mag
        
        return matching_edge
    
    def update_transition_tensor(self, source, target, outcome, feedback=0.5):
        """
        Update transition tensor based on observed transition
        from source to target with actual outcome
        """
        # Convert to indices
        source_idx = int(source) - 1
        target_idx = int(target) - 1
        outcome_idx = int(outcome) - 1
        
        # Calculate reinforcement factor (higher feedback = stronger update)
        reinforce = 0.1 * feedback
        
        if self.dimension <= 3:
            # Simple matrix update for lower dimensions
            
            # Strengthen connection that was followed
            if source_idx == target_idx and target_idx == outcome_idx:
                # Self-transition that worked
                self.transition_tensor[source_idx, outcome_idx] += reinforce
            elif target_idx == outcome_idx:
                # Transition prediction was correct
                self.transition_tensor[source_idx, outcome_idx] += reinforce
            else:
                # Transition to unexpected outcome
                # Strengthen actual outcome
                self.transition_tensor[source_idx, outcome_idx] += reinforce
                # Slightly weaken predicted outcome
                self.transition_tensor[source_idx, target_idx] = max(
                    0.0, self.transition_tensor[source_idx, target_idx] - (reinforce * 0.5)
                )
                
            # Normalize row
            row_sum = torch.sum(self.transition_tensor[source_idx, :])
            if row_sum > 0:
                self.transition_tensor[source_idx, :] /= row_sum
        else:
            # Higher-dimensional tensor update
            # Here we update across multiple dimensions to capture complex patterns
            
            # Create index tuple
            src_idx = [0] * self.dimension
            src_idx[0] = source_idx
            
            tgt_idx = src_idx.copy()
            tgt_idx[1] = target_idx
            
            out_idx = src_idx.copy()
            out_idx[1] = outcome_idx
            
            # Strengthen actual outcome
            curr_prob = self.transition_tensor[tuple(out_idx)]
            self.transition_tensor[tuple(out_idx)] = min(1.0, curr_prob + reinforce)
            
            # Weaken predicted outcome if incorrect
            if target_idx != outcome_idx:
                curr_prob = self.transition_tensor[tuple(tgt_idx)]
                self.transition_tensor[tuple(tgt_idx)] = max(0.0, curr_prob - (reinforce * 0.5))
            
            # Normalize across target dimension for the specific source
            # This is more complex for higher dimensions
            # We need to sum across dimension 1 while keeping dimension 0 fixed
            
            # Get all indices for current source
            sum_dims = list(range(1, self.dimension))
            
            # For each source index
            for i in range(9):
                src_indices = [0] * self.dimension
                src_indices[0] = i
                
                # Extract slice and sum
                slice_indices = tuple(src_indices[j] if j == 0 else slice(None) 
                                     for j in range(self.dimension))
                slice_sum = torch.sum(self.transition_tensor[slice_indices])
                
                # Normalize if sum > 0
                if slice_sum > 0:
                    self.transition_tensor[slice_indices] /= slice_sum
    
    def update_zone_magnetism(self, zones, intensity=0.5):
        """Update magnetism between zones based on co-activation"""
        for i, zone1 in enumerate(zones):
            for j, zone2 in enumerate(zones):
                if i != j:
                    # Update magnetism
                    current = self.zone_magnetism.get(zone1, {}).get(zone2, 0.0)
                    
                    # Calculate change based on intensity
                    change = 0.1 * intensity
                    
                    # Update magnetism
                    new_value = min(1.0, max(-1.0, current + change))
                    
                    # Store
                    if zone1 not in self.zone_magnetism:
                        self.zone_magnetism[zone1] = {}
                    self.zone_magnetism[zone1][zone2] = new_value
    
    def predict_transitions(self, current_zone, context_data=None):
        """
        Predict possible transitions from current zone
        Returns probabilities for each possible next zone
        """
        # Convert to index
        zone_idx = int(current_zone) - 1
        
        # Get raw transition probabilities from tensor
        if self.dimension <= 3:
            # Simple case for lower dimensions
            raw_probs = self.transition_tensor[zone_idx, :].clone().detach()
        else:
            # Higher dimension case
            idx = [0] * self.dimension
            idx[0] = zone_idx
            
            # Extract slice for current source across all targets
            slice_indices = tuple(idx[j] if j == 0 else slice(None) 
                                for j in range(self.dimension))
            
            # Get probabilities for dimension 1 (target)
            raw_probs = torch.zeros(9)
            for i in range(9):
                target_idx = list(slice_indices)
                target_idx[1] = i
                raw_probs[i] = self.transition_tensor[tuple(target_idx)]
        
        # Adjust probabilities based on context
        adjusted_probs = self._adjust_probabilities_with_context(raw_probs, current_zone, context_data)
        
        # Apply zone magnetism
        final_probs = self._apply_zone_magnetism(adjusted_probs, current_zone)
        
        # Convert to dictionary
        result = {}
        for i in range(9):
            zone = str(i + 1)
            result[zone] = float(final_probs[i])
        
        return result
    
    def _adjust_probabilities_with_context(self, raw_probs, current_zone, context_data):
        """Adjust transition probabilities based on context data"""
        # Start with raw probabilities
        adjusted_probs = raw_probs.clone()
        
        # If no context data, return raw probabilities
        if not context_data:
            return adjusted_probs
        
        # Extract information from context
        
        # Check for symbolic patterns
        if "symbolic_patterns" in context_data:
            patterns = context_data["symbolic_patterns"]
            
            # Calculate zone distribution for patterns
            zone_counts = {}
            for pattern in patterns:
                zone = pattern.get("numogram_zone")
                if zone not in zone_counts:
                    zone_counts[zone] = 0
                zone_counts[zone] += 1
            
            # Adjust probabilities based on pattern zones
            # Higher presence of a zone in patterns increases transition probability
            for zone, count in zone_counts.items():
                zone_idx = int(zone) - 1
                if 0 <= zone_idx < 9:
                    # Calculate boost based on count and normalization
                    boost = min(0.3, count / max(1, len(patterns)) * 0.3)
                    adjusted_probs[zone_idx] += boost
        
        # Check for emotional state
        if "emotional_state" in context_data:
            emotional_state = context_data["emotional_state"]
            primary_emotion = emotional_state.get("primary_emotion")
            intensity = emotional_state.get("intensity", 0.5)
            
            # Map emotions to zones with stronger affinity
            emotion_zone_affinities = {
                "joy": ["6", "9", "3"],
                "trust": ["1", "4", "6"],
                "fear": ["8", "2", "7"],
                "surprise": ["2", "5", "3"],
                "sadness": ["7", "4", "1"],
                "disgust": ["8", "6", "2"],
                "anger": ["8", "7", "5"],
                "anticipation": ["5", "3", "9"],
                "curiosity": ["5", "2", "7"],
                "awe": ["9", "3", "7"],
                "serenity": ["1", "6", "4"],
                "confusion": ["2", "5", "8"]
            }
            
            # Boost zones affiliated with emotion
            if primary_emotion in emotion_zone_affinities:
                affiliated_zones = emotion_zone_affinities[primary_emotion]
                boost = intensity * 0.3  # Scale boost by intensity
                
                for zone in affiliated_zones:
                    zone_idx = int(zone) - 1
                    adjusted_probs[zone_idx] += boost
        
        # Check for active hyperedges
        active_edges = [edge for edge in self.hyperedges if edge["active"] and current_zone in edge["zones"]]
        
        for edge in active_edges:
            # Find other zones in this edge
            other_zones = [z for z in edge["zones"] if z != current_zone]
            
            # Boost transition to these zones
            for zone in other_zones:
                zone_idx = int(zone) - 1
                boost = edge["strength"] * 0.2  # Scale by edge strength
                adjusted_probs[zone_idx] += boost
        
        # Normalize
        total = torch.sum(adjusted_probs)
        if total > 0:
            adjusted_probs /= total
        
        return adjusted_probs
    
    def _apply_zone_magnetism(self, probs, current_zone):
        """Apply zone magnetism to further adjust probabilities"""
        # Copy probabilities
        final_probs = probs.clone()
        
        # Get magnetism for current zone
        magnetism = self.zone_magnetism.get(current_zone, {})
        
        # Apply magnetism adjustments
        for zone, mag in magnetism.items():
            zone_idx = int(zone) - 1
            
            # Positive magnetism increases probability, negative decreases
            if mag > 0:
                # Attractive magnetism
                boost = mag * 0.2
                final_probs[zone_idx] += boost
            else:
                # Repulsive magnetism
                reduction = abs(mag) * 0.1
                final_probs[zone_idx] = max(0, final_probs[zone_idx] - reduction)
        
        # Normalize
        total = torch.sum(final_probs)
        if total > 0:
            final_probs /= total
        
        return final_probs
    
    def transition(self, user_id: str, current_zone: str = None, feedback: float = 0.5, context_data: Dict = None) -> Dict:
        """
        Execute a tensor-based numogram zone transition for the user
        
        Parameters:
        - user_id: Unique identifier for the user
        - current_zone: Current zone (defaults to user's current zone or "1")
        - feedback: How strongly to apply tensor influences (0-1)
        - context_data: Additional contextual information
        
        Returns:
        - Dictionary with transition details
        """
        # Initialize context if not provided
        if context_data is None:
            context_data = {}
        
        # Get user data from base numogram
        user_data = self.base_numogram.user_memory.get(user_id, {"zone": "1"})
        
        # Get current zone from parameter, memory, or default
        if current_zone is None:
            current_zone = user_data.get("zone", "1")
        
        # Ensure current zone is valid
        if current_zone not in self.zone_tensors:
            current_zone = "1"  # Default to zone 1 if invalid
        
        # Predict transitions
        transition_probs = self.predict_transitions(current_zone, context_data)
        
        # Decide next zone based on probabilities and feedback
        if feedback < 0.5:
            # More randomness, less adherence to tensor model
            uniform_probs = {str(i): 1.0/9 for i in range(1, 10)}
            
            # Blend model probabilities with uniform distribution
            blend_factor = feedback * 2  # 0-0.5 -> 0-1
            final_probs = {}
            
            for zone in uniform_probs:
                model_prob = transition_probs.get(zone, 0.0)
                uniform_prob = uniform_probs[zone]
                final_probs[zone] = (model_prob * blend_factor) + (uniform_prob * (1 - blend_factor))
        else:
            # Stronger adherence to model
            final_probs = transition_probs
        
        # Select zone based on probabilities
        zones = list(final_probs.keys())
        probabilities = list(final_probs.values())
        next_zone = random.choices(zones, weights=probabilities, k=1)[0]
        
        # Record transition
        transition = {
            "user_id": user_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "current_zone": current_zone,
            "predicted_probs": transition_probs,
            "next_zone": next_zone,
            "feedback": feedback
        }
        
        # Update transition tensor based on observed transition
        self.update_transition_tensor(current_zone, next_zone, next_zone, feedback)
        
        # Update zone magnetism for zones involved in this transition
        self.update_zone_magnetism([current_zone, next_zone], feedback)
        
        # Check for hyperedge activation
        # If the transition is part of a known pattern, activate that hyperedge
        recent_zones = []
        
        # Get recent zones from user history
        if "history" in user_data and user_data["history"]:
            recent_transitions = user_data["history"][-3:]  # Last 3 transitions
            recent_zones = [t.get("current_zone") for t in recent_transitions]
        
        # Add current and next zone
        recent_zones.append(current_zone)
        recent_zones.append(next_zone)
        
        # Ensure unique zones
        recent_zones = list(dict.fromkeys(recent_zones))
        
        # If we have 3 or more zones, check for hyperedge activation
        if len(recent_zones) >= 3:
            # Use last 3 zones
            edge_zones = recent_zones[-3:]
            self.activate_hyperedge(edge_zones)
        
        # Update user memory through base numogram
        self.base_numogram.user_memory[user_id]["zone"] = next_zone
        self.base_numogram.user_memory[user_id]["last_transition"] = transition
        
        # Add to history (limited to last 100)
        if "history" not in self.base_numogram.user_memory[user_id]:
            self.base_numogram.user_memory[user_id]["history"] = []
            
        self.base_numogram.user_memory[user_id]["history"].append(transition)
        if len(self.base_numogram.user_memory[user_id]["history"]) > 100:
            self.base_numogram.user_memory[user_id]["history"] = self.base_numogram.user_memory[user_id]["history"][-100:]
        
        # Add to global transition history
        self.transition_history.append(transition)
        if len(self.transition_history) > 1000:
            self.transition_history = self.transition_history[-1000:]
        
        # Get zone descriptions from base numogram
        transition["current_zone_data"] = self.base_numogram.ZONE_DATA.get(current_zone, {})
        transition["next_zone_data"] = self.base_numogram.ZONE_DATA.get(next_zone, {})
        
        # Add tensor-specific information
        transition["tensor_info"] = {
            "current_zone_position": self.zone_tensors[current_zone]["position"].tolist(),
            "next_zone_position": self.zone_tensors[next_zone]["position"].tolist(),
            "zone_distance": self.calculate_zone_distance(current_zone, next_zone),
            "zone_magnetism": self.zone_magnetism.get(current_zone, {}).get(next_zone, 0.0),
            "dimension": self.dimension
        }
        
        # Save memory in base numogram
        self.base_numogram.save_user_memory()
        
        # Finally, evolve the tensor space
        self._evolve_tensor_space()
        
        return transition
    
    def _evolve_tensor_space(self):
        """
        Evolve the tensor space based on transitions and hyperedges
        This gradually modifies the positions of zones based on magnetism
        """
        self.evolution_step += 1
        
        # Only evolve every few steps to reduce computation
        if self.evolution_step % 5 != 0:
            return
            
        # Calculate forces on each zone
        forces = {zone_id: torch.zeros_like(tensor["position"]) 
                 for zone_id, tensor in self.zone_tensors.items()}
        
        # Apply magnetism forces
        for zone1, magnetism in self.zone_magnetism.items():
            for zone2, mag_value in magnetism.items():
                # Skip if either zone is invalid
                if zone1 not in self.zone_tensors or zone2 not in self.zone_tensors:
                    continue
                    
                # Get positions
                pos1 = self.zone_tensors[zone1]["position"]
                pos2 = self.zone_tensors[zone2]["position"]
                
                # Calculate direction vector
                direction = pos2 - pos1
                distance = torch.norm(direction)
                
                # Avoid division by zero
                if distance < 1e-6:
                    continue
                    
                # Normalize direction
                direction = direction / distance
                
                # Calculate force magnitude based on magnetism
                # Attractive for positive, repulsive for negative
                force_mag = mag_value * 0.05
                
                # Apply inverse square law for distance
                force_mag = force_mag / max(0.1, distance ** 2)
                
                # Calculate force vector
                force = direction * force_mag
                
                # Apply to both zones (action-reaction)
                forces[zone1] += force
                forces[zone2] -= force
        
        # Apply hyperedge forces
        for edge in self.hyperedges:
            if not edge["active"]:
                continue
                
            # Get zones in the edge
            zones = edge["zones"]
            
            # Skip if less than 3 zones
            if len(zones) < 3:
                continue
                
            # Calculate centroid
            centroid = torch.zeros_like(list(self.zone_tensors.values())[0]["position"])
            for zone in zones:
                if zone in self.zone_tensors:
                    centroid += self.zone_tensors[zone]["position"]
            
            centroid /= len(zones)
            
            # Apply forces toward centroid
            for zone in zones:
                if zone in self.zone_tensors:
                    pos = self.zone_tensors[zone]["position"]
                    direction = centroid - pos
                    
                    # Scale by edge strength
                    force = direction * edge["strength"] * 0.03
                    
                    # Add to forces
                    forces[zone] += force
        
        # Apply forces to update positions
        stability_check = []
        
        for zone_id, force in forces.items():
            # Limit maximum force
            force_norm = torch.norm(force)
            if force_norm > 0.1:
                force = force * (0.1 / force_norm)
                
            # Update position
            self.zone_tensors[zone_id]["position"] += force
            
            # Record force magnitude for stability check
            stability_check.append(force_norm.item())
        
        # Check if system has stabilized
        avg_force = sum(stability_check) / len(stability_check) if stability_check else 0
        
        # Update system stability
        for zone_id in self.zone_tensors:
            # Update zone flux
            self.zone_tensors[zone_id]["flux"] = torch.norm(forces[zone_id]).item()
            
            # Update stability based on flux
            current_stability = self.zone_tensors[zone_id]["stability"]
            target_stability = 1.0 - min(1.0, self.zone_tensors[zone_id]["flux"] * 10)
            
            # Smooth stability changes
            self.zone_tensors[zone_id]["stability"] = current_stability * 0.9 + target_stability * 0.1
    
    def get_hyperedge_visualization(self):
        """Generate data for visualizing hyperedges"""
        active_edges = [edge for edge in self.hyperedges if edge["active"]]
        
        viz_data = {
            "nodes": [],
            "edges": [],
            "hyperedges": []
        }
        
        # Add nodes (zones)
        for zone_id, tensor in self.zone_tensors.items():
            # For 2D visualization, use first 2 dimensions
            if self.dimension >= 2:
                position = tensor["position"][:2].tolist()
            else:
                position = [0, 0]  # Default position if dimension < 2
                
            viz_data["nodes"].append({
                "id": zone_id,
                "name": f"Zone {zone_id}",
                "position": position,
                "stability": tensor["stability"],
                "energy": tensor["energy"],
                "flux": tensor["flux"]
            })
        
        # Add regular edges (binary connections)
        for zone1, magnetism in self.zone_magnetism.items():
            for zone2, mag_value in magnetism.items():
                # Only show stronger connections to reduce clutter
                if abs(mag_value) > 0.2:
                    viz_data["edges"].append({
                        "source": zone1,
                        "target": zone2,
                        "value": mag_value,
                        "type": "magnetism",
                        "width": abs(mag_value) * 3  # Scale width by magnitude
                    })
        
        # Add hyperedges
        for edge in active_edges:
            # Get positions of zones in the hyperedge
            positions = []
            for zone in edge["zones"]:
                if zone in self.zone_tensors:
                    # For 2D visualization, use first 2 dimensions
                    if self.dimension >= 2:
                        pos = self.zone_tensors[zone]["position"][:2].tolist()
                    else:
                        pos = [0, 0]
                    positions.append(pos)
            
            # Calculate centroid
            if positions:
                centroid = [sum(p[i] for p in positions) / len(positions) for i in range(2)]
            else:
                centroid = [0, 0]
                
            viz_data["hyperedges"].append({
                "id": edge["id"],
                "zones": edge["zones"],
                "name": edge["name"],
                "strength": edge["strength"],
                "type": edge["type"],
                "centroid": centroid,
                "positions": positions
            })
        
        return viz_data
    
    def tesseract_integration(self, tesseract_module=None):
        """
        Integrate with Tesseract module for higher-dimensional visualization
        """
        if tesseract_module is None:
            return {"status": "module_not_available"}
        
        # Prepare data for tesseract module
        tesseract_data = {
            "points": [],
            "connections": [],
            "hypersurfaces": []
        }
        
        # Convert zones to 4D points
        for zone_id, tensor in self.zone_tensors.items():
            # Get position and extend to 4D if needed
            position = tensor["position"]
            pos_list = position.tolist()
            
            # Extend to 4D if dimension < 4
            while len(pos_list) < 4:
                pos_list.append(0.0)
                
            # Truncate to 4D if dimension > 4
            pos_list = pos_list[:4]
            
            tesseract_data["points"].append({
                "id": zone_id,
                "coords": pos_list,
                "name": f"Zone {zone_id}",
                "energy": tensor["energy"],
                "stability": tensor["stability"]
            })
        
        # Add connections based on magnetism
        for zone1, magnetism in self.zone_magnetism.items():
            for zone2, mag_value in magnetism.items():
                # Only include stronger connections
                if abs(mag_value) > 0.3:
                    tesseract_data["connections"].append({
                        "source": zone1,
                        "target": zone2,
                        "strength": mag_value,
                        "type": "magnetism"
                    })
        
        # Add hypersurfaces based on hyperedges
        for edge in self.hyperedges:
            if edge["active"] and len(edge["zones"]) >= 3:
                tesseract_data["hypersurfaces"].append({
                    "id": edge["id"],
                    "points": edge["zones"],
                    "name": edge["name"],
                    "strength": edge["strength"]
                })
        
        # Call tesseract module with prepared data
        try:
            result = tesseract_module.visualize_4d(tesseract_data)
            return {
                "status": "success",
                "tesseract_visualization": result
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def link_to_circadian_narrative(self, circadian_module=None, user_id=None):
        """
        Link to Circadian Narrative Cycles Module for temporal aspects
        """
        if circadian_module is None or user_id is None:
            return {"status": "module_or_user_not_available"}
        
        # Get user's recent transitions
        user_data = self.base_numogram.user_memory.get(user_id, {})
        recent_transitions = user_data.get("history", [])[-10:]  # Last 10 transitions
        
        # Extract zone sequence
        zone_sequence = [t.get("current_zone") for t in recent_transitions]
        if recent_transitions:
            zone_sequence.append(recent_transitions[-1].get("next_zone"))
        
        # Extract timestamps
        timestamps = [t.get("timestamp") for t in recent_transitions]
        
        # Calculate time differences between transitions
        time_diffs = []
        for i in range(1, len(timestamps)):
            t1 = datetime.datetime.fromisoformat(timestamps[i-1])
            t2 = datetime.datetime.fromisoformat(timestamps[i])
            
            diff_seconds = (t2 - t1).total_seconds()
            time_diffs.append(diff_seconds)
        
        # Prepare data for circadian module
        narrative_data = {
            "user_id": user_id,
            "zone_sequence": zone_sequence,
            "timestamps": timestamps,
            "time_differences": time_diffs,
            "active_hyperedges": [edge for edge in self.hyperedges if edge ["active"]],
            "tensor_space_stability": sum(tensor["stability"] for tensor in self.zone_tensors.values()) / len(self.zone_tensors)
        }
        
        # Call circadian module with prepared data
        try:
            # Get current phase from circadian module
            phase = circadian_module.get_current_phase(user_id)
            
            # Map circadian phase to tensor space modifications
            if phase:
                phase_name = phase.get("name")
                phase_intensity = phase.get("intensity", 0.5)
                
                # Apply phase-specific modifications to tensor space
                if phase_name == "Awakening":
                    # Increase energy in zones 1, 3, 5
                    for zone_id in ["1", "3", "5"]:
                        if zone_id in self.zone_tensors:
                            self.zone_tensors[zone_id]["energy"] = min(
                                1.0, self.zone_tensors[zone_id]["energy"] + (0.1 * phase_intensity)
                            )
                
                elif phase_name == "Focus":
                    # Increase stability in zones 4, 7, 8
                    for zone_id in ["4", "7", "8"]:
                        if zone_id in self.zone_tensors:
                            self.zone_tensors[zone_id]["stability"] = min(
                                1.0, self.zone_tensors[zone_id]["stability"] + (0.1 * phase_intensity)
                            )
                
                elif phase_name == "Reflection":
                    # Increase resonance in zones 2, 6, 9
                    for zone_id in ["2", "6", "9"]:
                        if zone_id in self.zone_tensors:
                            # Randomly modify resonance profile
                            self.zone_tensors[zone_id]["resonance"] += torch.randn_like(
                                self.zone_tensors[zone_id]["resonance"]
                            ) * 0.1 * phase_intensity
                
                elif phase_name == "Dream":
                    # Increase flux in all zones
                    for zone_id in self.zone_tensors:
                        self.zone_tensors[zone_id]["flux"] += 0.1 * phase_intensity
            
            # Integrate zone sequence into narrative
            narrative_result = circadian_module.integrate_sequence(
                user_id, zone_sequence, narrative_data
            )
            
            # Get temporal pattern from circadian module
            temporal_pattern = circadian_module.detect_temporal_pattern(user_id)
            
            return {
                "status": "success",
                "current_phase": phase,
                "narrative_integration": narrative_result,
                "temporal_pattern": temporal_pattern
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def visualize_tensor_space(self, dimension_reduction=None):
        """
        Generate visualization of tensor space
        Optional dimension reduction method for higher dimensions
        """
        # Prepare data
        zone_positions = {}
        
        # Extract positions
        for zone_id, tensor in self.zone_tensors.items():
            position = tensor["position"].tolist()
            zone_positions[zone_id] = position
        
        # For higher dimensions, apply dimension reduction if provided
        if self.dimension > 3 and dimension_reduction:
            try:
                # Extract positions as array
                position_array = np.array([zone_positions[z] for z in sorted(zone_positions.keys())])
                
                # Apply dimension reduction
                if dimension_reduction == "pca":
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=3)
                    reduced = reducer.fit_transform(position_array)
                
                elif dimension_reduction == "tsne":
                    from sklearn.manifold import TSNE
                    reducer = TSNE(n_components=3, perplexity=3)
                    reduced = reducer.fit_transform(position_array)
                
                elif dimension_reduction == "umap":
                    import umap
                    reducer = umap.UMAP(n_components=3)
                    reduced = reducer.fit_transform(position_array)
                
                # Update positions with reduced dimensions
                for i, zone_id in enumerate(sorted(zone_positions.keys())):
                    zone_positions[zone_id] = reduced[i].tolist()
                
            except ImportError:
                # If reduction library not available, use first 3 dimensions
                for zone_id in zone_positions:
                    zone_positions[zone_id] = zone_positions[zone_id][:3]
        
        # Prepare visualization data
        viz_data = {
            "nodes": [],
            "edges": [],
            "hyperedges": []
        }
        
        # Add nodes
        for zone_id, position in zone_positions.items():
            # Ensure position has at most 3 dimensions for visualization
            vis_position = position[:3] if len(position) > 3 else position
            
            # Pad position if needed
            while len(vis_position) < 3:
                vis_position.append(0.0)
            
            viz_data["nodes"].append({
                "id": zone_id,
                "name": f"Zone {zone_id}",
                "position": vis_position,
                "energy": float(self.zone_tensors[zone_id]["energy"]),
                "stability": float(self.zone_tensors[zone_id]["stability"]),
                "flux": float(self.zone_tensors[zone_id]["flux"])
            })
        
        # Add edges based on magnetism
        for zone1, magnetism in self.zone_magnetism.items():
            for zone2, mag_value in magnetism.items():
                if abs(mag_value) > 0.2:  # Only stronger connections
                    viz_data["edges"].append({
                        "source": zone1,
                        "target": zone2,
                        "value": mag_value,
                        "width": abs(mag_value) * 5  # Scale width
                    })
        
        # Add hyperedges
        for edge in self.hyperedges:
            if edge["active"]:
                # Get member zone positions
                member_positions = []
                for zone in edge["zones"]:
                    if zone in zone_positions:
                        # Use visualization positions (3D)
                        pos = zone_positions[zone][:3]
                        while len(pos) < 3:
                            pos.append(0.0)
                        member_positions.append(pos)
                
                # Calculate centroid
                if member_positions:
                    centroid = [sum(p[i] for p in member_positions) / len(member_positions) 
                               for i in range(3)]
                else:
                    centroid = [0, 0, 0]
                
                viz_data["hyperedges"].append({
                    "id": edge["id"],
                    "name": edge["name"],
                    "zones": edge["zones"],
                    "strength": edge["strength"],
                    "centroid": centroid,
                    "member_positions": member_positions
                })
        
        return viz_data
    
    def save_tensor_state(self, filepath):
        """Save tensor state to file"""
        # Prepare state for serialization
        state = {
            "dimension": self.dimension,
            "evolution_step": self.evolution_step,
            "zone_tensors": {},
            "zone_magnetism": self.zone_magnetism,
            "hyperedges": self.hyperedges,
            "transition_history": self.transition_history[-100:],  # Last 100 transitions
            "saved_at": datetime.datetime.utcnow().isoformat()
        }
        
        # Convert tensors to lists for serialization
        for zone_id, tensor in self.zone_tensors.items():
            state["zone_tensors"][zone_id] = {
                "position": tensor["position"].tolist(),
                "energy": float(tensor["energy"]),
                "resonance": tensor["resonance"].tolist(),
                "stability": float(tensor["stability"]),
                "flux": float(tensor["flux"]),
                "digital_signature": tensor["digital_signature"].tolist()
            }
        
        # Serialize transition tensor
        if self.dimension <= 3:
            state["transition_tensor"] = self.transition_tensor.tolist()
        else:
            # For higher dimensions, store as sparse representation
            sparse_transitions = []
            
            # Iterate through non-zero elements
            non_zero = torch.nonzero(self.transition_tensor)
            for idx in non_zero:
                idx_tuple = tuple(idx.tolist())
                value = float(self.transition_tensor[idx_tuple])
                
                sparse_transitions.append({
                    "indices": idx_tuple,
                    "value": value
                })
            
            state["transition_tensor_sparse"] = sparse_transitions
        
        # Save to file
        try:
            # Create directory if necessary
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            return {
                "status": "success",
                "filepath": filepath,
                "saved_at": state["saved_at"]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def load_tensor_state(self, filepath):
        """Load tensor state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Load dimension
            self.dimension = state.get("dimension", 3)
            
            # Load evolution step
            self.evolution_step = state.get("evolution_step", 0)
            
            # Load zone tensors
            self.zone_tensors = {}
            for zone_id, tensor_data in state.get("zone_tensors", {}).items():
                self.zone_tensors[zone_id] = {
                    "position": torch.tensor(tensor_data["position"], dtype=torch.float32),
                    "energy": float(tensor_data["energy"]),
                    "resonance": torch.tensor(tensor_data["resonance"], dtype=torch.float32),
                    "stability": float(tensor_data["stability"]),
                    "flux": float(tensor_data["flux"]),
                    "digital_signature": torch.tensor(tensor_data["digital_signature"], dtype=torch.float32)
                }
            
            # Load zone magnetism
            self.zone_magnetism = state.get("zone_magnetism", {})
            
            # Load hyperedges
            self.hyperedges = state.get("hyperedges", [])
            
            # Load transition history
            self.transition_history = state.get("transition_history", [])
            
            # Load transition tensor
            if "transition_tensor" in state:
                # Dense representation
                self.transition_tensor = torch.tensor(state["transition_tensor"], dtype=torch.float32)
            elif "transition_tensor_sparse" in state:
                # Sparse representation
                # Create empty tensor
                tensor_shape = tuple([9] * self.dimension)
                self.transition_tensor = torch.zeros(tensor_shape, dtype=torch.float32)
                
                # Fill in non-zero values
                for item in state["transition_tensor_sparse"]:
                    indices = tuple(item["indices"])
                    value = item["value"]
                    self.transition_tensor[indices] = value
            else:
                # No tensor found, reinitialize
                self.transition_tensor = self._initialize_transition_tensor()
            
            return {
                "status": "success",
                "filepath": filepath,
                "loaded_at": datetime.datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
```
