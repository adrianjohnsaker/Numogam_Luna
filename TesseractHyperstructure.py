from typing import Dict, List, Any, Optional, Union, Set, Tuple
import hashlib
import json
from collections import defaultdict
import math


class TesseractHyperstructure:
    """
    TesseractHyperstructure: A multi-dimensional data processing system that manages
    abstract symbolic relationships and temporal-spatial transformations.
    
    This structure operates through interconnected layers that process symbolic,
    numeric, and temporal data to generate emergent patterns and transformations.
    """
    
    # Constants for configuration
    MAX_HISTORY_SIZE = 100
    DIMENSIONAL_AXES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    PHASE_CYCLES = 9
    
    def __init__(self, dimension_count: int = 4, initialization_seed: Optional[str] = None):
        """
        Initialize the TesseractHyperstructure with configurable dimensions.
        
        Args:
            dimension_count: Number of dimensions to model (default: 4)
            initialization_seed: Optional seed to initialize the structure state
        """
        if dimension_count < 2:
            raise ValueError("Dimension count must be at least 2")
            
        self.dimension_count = dimension_count
        self.layers = {
            "numogram_core": {},
            "bwo_processor": {},
            "emergent_flows": [],
            "symbolic_resonance": [],
            "temporal_states": [],
            "integration_matrix": defaultdict(dict),
            "dimensional_vectors": {axis: [] for axis in self.DIMENSIONAL_AXES[:dimension_count]}
        }
        
        self.state_hash = None
        self.cycle_count = 0
        self.stability_coefficient = 1.0
        
        # Initialize with seed if provided
        if initialization_seed:
            self._initialize_from_seed(initialization_seed)

    def _initialize_from_seed(self, seed: str) -> None:
        """Initialize the structure state from a seed string."""
        # Generate deterministic initial state from seed
        seed_hash = hashlib.sha256(seed.encode()).hexdigest()
        
        # Use the hash to initialize various components
        for i, char in enumerate(seed_hash[:16]):
            value = int(char, 16)
            axis = i % len(self.DIMENSIONAL_AXES[:self.dimension_count])
            axis_name = self.DIMENSIONAL_AXES[axis]
            self.layers["dimensional_vectors"][axis_name].append(value)
            
        # Initialize numogram_core with seed-derived values
        self.layers["numogram_core"] = {
            "seed_origin": seed,
            "prime_attractor": int(seed_hash[:8], 16) % 9,
            "dimensional_bias": [int(seed_hash[i:i+2], 16) / 255 for i in range(0, 16, 2)]
        }

    def receive_input(self, 
                      zone_state: Dict[str, Any], 
                      symbolic_map: Dict[str, Any], 
                      drift_pattern: Dict[str, Any], 
                      affect_signal: Union[float, Dict[str, float]]) -> None:
        """
        Process and integrate new input data into the hyperstructure.
        
        Args:
            zone_state: Current state of the zone being processed
            symbolic_map: Mapping of symbols and their relationships
            drift_pattern: Temporal drift patterns
            affect_signal: Signal strength or emotional/intensity values
        """
        # Validate inputs
        if not isinstance(zone_state, dict):
            raise TypeError("zone_state must be a dictionary")
            
        if not isinstance(symbolic_map, dict):
            raise TypeError("symbolic_map must be a dictionary")
        
        # Deep copy and update numogram_core with new zone_state
        self.layers["numogram_core"].update(zone_state)
        
        # Manage history lengths
        if len(self.layers["symbolic_resonance"]) >= self.MAX_HISTORY_SIZE:
            self.layers["symbolic_resonance"].pop(0)
        self.layers["symbolic_resonance"].append(symbolic_map)
        
        if len(self.layers["temporal_states"]) >= self.MAX_HISTORY_SIZE:
            self.layers["temporal_states"].pop(0)
        self.layers["temporal_states"].append(drift_pattern)
        
        # Process and store the emergent flow data
        flow_data = {
            "timestamp": self.cycle_count,
            "symbolic_map": symbolic_map,
            "drift_pattern": drift_pattern,
            "affect_signal": affect_signal,
            "dimensional_projection": self._calculate_dimensional_projection(symbolic_map)
        }
        
        if len(self.layers["emergent_flows"]) >= self.MAX_HISTORY_SIZE:
            self.layers["emergent_flows"].pop(0)
        self.layers["emergent_flows"].append(flow_data)
        
        # Update integration matrix
        self._update_integration_matrix(symbolic_map, drift_pattern, affect_signal)
        
        # Update state hash
        self._update_state_hash()
        
        self.cycle_count += 1

    def _update_integration_matrix(self, 
                                  symbolic_map: Dict[str, Any],
                                  drift_pattern: Dict[str, Any],
                                  affect_signal: Union[float, Dict[str, float]]) -> None:
        """
        Update the integration matrix with new relationships between inputs.
        
        This creates connections between symbols, patterns and affective states.
        """
        # Extract keys for correlation
        symbolic_keys = set(symbolic_map.keys())
        drift_keys = set(drift_pattern.keys())
        
        # Create relationships between elements
        for s_key in symbolic_keys:
            for d_key in drift_keys:
                # Create a correlation strength based on values
                if isinstance(symbolic_map[s_key], (int, float)) and isinstance(drift_pattern[d_key], (int, float)):
                    correlation = abs(float(symbolic_map[s_key]) - float(drift_pattern[d_key])) / 10.0
                    correlation = min(1.0, max(0.0, correlation))
                    
                    # Add to integration matrix
                    key_pair = f"{s_key}:{d_key}"
                    
                    # Update with exponential moving average
                    if key_pair in self.layers["integration_matrix"]:
                        old_value = self.layers["integration_matrix"][key_pair].get("correlation", 0)
                        self.layers["integration_matrix"][key_pair]["correlation"] = old_value * 0.7 + correlation * 0.3
                    else:
                        self.layers["integration_matrix"][key_pair] = {"correlation": correlation}
                    
                    # Add affect association
                    if isinstance(affect_signal, (int, float)):
                        self.layers["integration_matrix"][key_pair]["affect"] = affect_signal
                    elif isinstance(affect_signal, dict):
                        self.layers["integration_matrix"][key_pair]["affect_vector"] = affect_signal

    def _calculate_dimensional_projection(self, symbolic_map: Dict[str, Any]) -> Dict[str, float]:
        """Calculate projections of symbolic data onto dimensional axes."""
        result = {}
        
        # Simple hash-based projection
        for axis in self.DIMENSIONAL_AXES[:self.dimension_count]:
            # Create a projection based on symbol hashing
            axis_value = 0
            for key, value in symbolic_map.items():
                hash_val = int(hashlib.md5(f"{key}:{value}:{axis}".encode()).hexdigest(), 16)
                axis_value += hash_val % 1000 / 1000  # Normalized to 0-1
            
            result[axis] = axis_value / max(1, len(symbolic_map))
            
            # Store in dimensional vectors
            if len(self.layers["dimensional_vectors"][axis]) >= self.MAX_HISTORY_SIZE:
                self.layers["dimensional_vectors"][axis].pop(0)
            self.layers["dimensional_vectors"][axis].append(result[axis])
            
        return result

    def _update_state_hash(self) -> None:
        """Update the hash representation of the current state."""
        # Create a hashable representation of the current state
        state_repr = json.dumps({
            "numogram": self.layers["numogram_core"],
            "last_flow": self.layers["emergent_flows"][-1] if self.layers["emergent_flows"] else None,
            "cycle": self.cycle_count
        }, sort_keys=True)
        
        self.state_hash = hashlib.sha256(state_repr.encode()).hexdigest()

    def process(self) -> Dict[str, Any]:
        """
        Process the accumulated data in all layers and generate transformative outputs.
        
        Returns:
            A dictionary containing processed results and emergent properties.
        """
        # Calculate stability based on recent changes
        self.stability_coefficient = self._calculate_stability()
        
        # Perform abstract recombination and multi-dimensional evaluation
        fused_output = {
            "restructured_zone": self._fold_zone(),
            "emergent_symbol": self._generate_symbolic_merge(),
            "temporal_shift": self._evaluate_temporal_phase(),
            "dimensional_vectors": self._extract_dimensional_vectors(),
            "stability_coefficient": self.stability_coefficient,
            "integration_density": self._calculate_integration_density(),
            "cycle_state": self.cycle_count % self.PHASE_CYCLES,
            "event_horizon": self._calculate_event_horizon()
        }
        
        # Store the processed output
        self.layers["bwo_processor"] = fused_output
        
        return fused_output

    def _fold_zone(self) -> Dict[str, Any]:
        """
        Implement tesseract-like folding logic to restructure the zone.
        
        This creates higher-dimensional folding of the input space.
        """
        # Enhanced folding logic using numogram_core data
        try:
            # Create a hashable representation of numogram_core
            items = sorted(self.layers["numogram_core"].items())
            zone_hash = hash(frozenset(items))
            
            # Create a more complex folding pattern
            primary_fold = zone_hash % self.PHASE_CYCLES + 1
            
            # Calculate secondary dimensions
            secondary_dims = {}
            for i, axis in enumerate(self.DIMENSIONAL_AXES[:self.dimension_count]):
                # Create variations based on position in the cycle
                axis_value = (zone_hash >> (i * 4)) % 16
                axis_phase = (self.cycle_count + i) % self.PHASE_CYCLES
                secondary_dims[axis] = {
                    "value": axis_value,
                    "phase": axis_phase,
                    "resonance": axis_value / 16 * math.sin(math.pi * axis_phase / self.PHASE_CYCLES)
                }
            
            # Integration with recent emergent flows
            recent_flows = self.layers["emergent_flows"][-3:] if len(self.layers["emergent_flows"]) >= 3 else []
            flow_integration = {}
            
            if recent_flows:
                # Extract patterns from recent flows
                flow_keys = set()
                for flow in recent_flows:
                    if isinstance(flow["symbolic_map"], dict):
                        flow_keys.update(flow["symbolic_map"].keys())
                
                # Create integration values for recurring keys
                for key in flow_keys:
                    occurrences = sum(1 for flow in recent_flows if key in flow.get("symbolic_map", {}))
                    if occurrences > 1:  # Only include recurring patterns
                        flow_integration[key] = occurrences / len(recent_flows)
            
            # Return the folded zone structure
            return {
                "folded_zone": primary_fold,
                "dimensional_folds": secondary_dims,
                "flow_integration": flow_integration,
                "fold_complexity": len(self.layers["numogram_core"]) * primary_fold / self.PHASE_CYCLES
            }
        except Exception as e:
            # Fallback to simpler folding in case of errors
            return {"folded_zone": hash(str(self.layers["numogram_core"])) % self.PHASE_CYCLES + 1, 
                    "error": str(e)}

    def _generate_symbolic_merge(self) -> str:
        """
        Generate emergent symbols by merging symbolic resonances.
        
        Creates new symbolic representations from the combination of existing symbols.
        """
        if not self.layers["symbolic_resonance"]:
            return "unknown"
            
        try:
            # Collect all values from symbolic resonance dictionaries
            all_symbols: Set[str] = set()
            
            for resonance in self.layers["symbolic_resonance"]:
                if not isinstance(resonance, dict):
                    continue
                    
                # Extract symbols, converting non-string values to strings
                for key, value in resonance.items():
                    if isinstance(value, str):
                        all_symbols.add(value)
                    elif isinstance(value, (int, float)):
                        # Convert numbers to symbolic representations
                        symbol = f"{key}:{value:.2f}" if isinstance(value, float) else f"{key}:{value}"
                        all_symbols.add(symbol)
                    elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, str)) for x in value):
                        # Handle simple collections of basic types
                        symbol = f"{key}:[{','.join(str(x) for x in value[:3])}{'...' if len(value) > 3 else ''}]"
                        all_symbols.add(symbol)
            
            # Calculate frequencies
            symbol_freq = {}
            for resonance in self.layers["symbolic_resonance"]:
                if not isinstance(resonance, dict):
                    continue
                
                for value in resonance.values():
                    if isinstance(value, str) and value in all_symbols:
                        symbol_freq[value] = symbol_freq.get(value, 0) + 1
            
            # Select most frequent symbols (up to 5)
            top_symbols = sorted(symbol_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
            if not top_symbols:
                # Fallback if no frequency data
                return "-".join(sorted(list(all_symbols)[:5]))
                
            # Create merged symbol using most frequent symbols
            merged = "-".join(symbol for symbol, _ in top_symbols)
            
            # Add cycle marker
            cycle_marker = chr(9311 + (self.cycle_count % 10 + 1))  # Unicode circled numbers
            
            return f"{merged}{cycle_marker}"
            
        except Exception as e:
            # Fallback in case of error
            return f"symbol-error-{self.cycle_count % self.PHASE_CYCLES}"

    def _evaluate_temporal_phase(self) -> Dict[str, Any]:
        """
        Evaluate temporal phase relationships and cycles.
        
        Analyzes temporal patterns to identify cycles, progressions, and phase shifts.
        """
        # Count phases
        phase_count = len(self.layers["temporal_states"])
        
        if phase_count == 0:
            return {"phase": 0, "confidence": 0}
            
        try:
            # Basic cyclic phase calculation
            current_phase = phase_count % self.PHASE_CYCLES
            
            # Advanced temporal analysis
            temporal_density = min(1.0, phase_count / self.MAX_HISTORY_SIZE)
            
            # Calculate phase velocity by comparing recent states
            phase_velocity = 0
            if phase_count >= 2:
                recent_states = self.layers["temporal_states"][-2:]
                
                # Compare state sizes as a simple metric
                if all(isinstance(state, dict) for state in recent_states):
                    size_diff = len(recent_states[1]) - len(recent_states[0])
                    phase_velocity = size_diff / max(1, len(recent_states[0]))
            
            # Phase prediction based on velocity
            predicted_next_phase = (current_phase + math.ceil(phase_velocity * self.PHASE_CYCLES)) % self.PHASE_CYCLES
            
            # Calculate confidence based on stability
            confidence = self.stability_coefficient * temporal_density
            
            return {
                "phase": current_phase,
                "velocity": phase_velocity,
                "predicted_next": predicted_next_phase,
                "cycle_position": phase_count % self.PHASE_CYCLES,
                "confidence": confidence
            }
        except Exception as e:
            # Fallback to simple phase calculation
            return {"phase": phase_count % 3, "error": str(e)}

    def _extract_dimensional_vectors(self) -> Dict[str, List[float]]:
        """Extract the current dimensional vectors for analysis."""
        result = {}
        
        # Calculate averaged vectors for each dimension
        for axis, values in self.layers["dimensional_vectors"].items():
            if not values:
                result[axis] = 0.0
                continue
                
            # Get recent values (last 5 or less)
            recent = values[-5:] if len(values) >= 5 else values
            
            # Calculate average and trend
            avg = sum(recent) / len(recent)
            
            # Calculate trend (slope) if enough points
            trend = 0
            if len(recent) >= 3:
                # Simple linear regression slope approximation
                n = len(recent)
                x_mean = (n - 1) / 2  # x values are just indices 0, 1, 2...
                y_mean = sum(recent) / n
                
                numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                
                trend = numerator / denominator if denominator != 0 else 0
            
            result[axis] = {
                "value": avg,
                "trend": trend,
                "recent": recent[-3:] if len(recent) >= 3 else recent
            }
            
        return result

    def _calculate_stability(self) -> float:
        """Calculate the stability coefficient based on recent state changes."""
        if len(self.layers["emergent_flows"]) < 2:
            return 1.0
            
        # Get the two most recent flows
        recent_flows = self.layers["emergent_flows"][-2:]
        
        try:
            # Calculate similarity between recent symbolic maps
            symbolic_similarity = 0.0
            if all("symbolic_map" in flow and isinstance(flow["symbolic_map"], dict) for flow in recent_flows):
                map1 = recent_flows[0]["symbolic_map"]
                map2 = recent_flows[1]["symbolic_map"]
                
                # Keys in both maps
                common_keys = set(map1.keys()) & set(map2.keys())
                all_keys = set(map1.keys()) | set(map2.keys())
                
                # Basic similarity: proportion of common keys
                key_similarity = len(common_keys) / max(1, len(all_keys))
                
                # Value similarity for common keys
                value_similarity = 0.0
                for key in common_keys:
                    if map1[key] == map2[key]:
                        value_similarity += 1.0
                
                value_similarity = value_similarity / max(1, len(common_keys))
                
                symbolic_similarity = (key_similarity + value_similarity) / 2
            
            # Factor in cycle position
            cycle_factor = 1.0 - (0.1 * (self.cycle_count % self.PHASE_CYCLES) / self.PHASE_CYCLES)
            
            # Combine factors with weights
            stability = (symbolic_similarity * 0.7) + (cycle_factor * 0.3)
            
            # Ensure value is between 0 and 1
            return max(0.1, min(1.0, stability))
        except Exception:
            # Default stability if calculation fails
            return 0.5

    def _calculate_integration_density(self) -> float:
        """Calculate the density of integrations between different data sources."""
        if not self.layers["integration_matrix"]:
            return 0.0
            
        # Count active connections
        active_connections = sum(1 for v in self.layers["integration_matrix"].values() 
                                if v.get("correlation", 0) > 0.3)
                                
        # Calculate potential connection count based on recent inputs
        potential_connections = 0
        if self.layers["symbolic_resonance"] and self.layers["temporal_states"]:
            latest_symbolic = self.layers["symbolic_resonance"][-1]
            latest_temporal = self.layers["temporal_states"][-1]
            
            if isinstance(latest_symbolic, dict) and isinstance(latest_temporal, dict):
                potential_connections = len(latest_symbolic) * len(latest_temporal)
        
        # Calculate density ratio
        if potential_connections > 0:
            return active_connections / potential_connections
        else:
            return 0.0

    def _calculate_event_horizon(self) -> Dict[str, Any]:
        """Calculate the event horizon - the boundary of predictable outcomes."""
        # Based on stability and cycle position
        base_horizon = self.stability_coefficient * 10  # Max 10 cycles ahead
        
        # Adjust based on integration density
        integration_density = self._calculate_integration_density()
        adjusted_horizon = base_horizon * (0.5 + integration_density)
        
        # Current position in the cycle affects predictability
        cycle_position = self.cycle_count % self.PHASE_CYCLES
        cycle_factor = 1.0 - (cycle_position / self.PHASE_CYCLES)
        
        # Final horizon calculation
        final_horizon = math.ceil(adjusted_horizon * cycle_factor)
        
        return {
            "cycles_ahead": final_horizon,
            "confidence": self.stability_coefficient * integration_density,
            "max_visibility": self.PHASE_CYCLES if final_horizon >= self.PHASE_CYCLES else final_horizon
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the hyperstructure."""
        return {
            "cycle": self.cycle_count,
            "phase": self.cycle_count % self.PHASE_CYCLES,
            "stability": self.stability_coefficient,
            "state_hash": self.state_hash,
            "active_dimensions": list(self.layers["dimensional_vectors"].keys()),
            "last_output": self.layers.get("bwo_processor", {})
        }
    
    def reset(self, preserve_history: bool = False) -> None:
        """
        Reset the hyperstructure to initial state.
        
        Args:
            preserve_history: If True, maintains historical data but resets processing state
        """
        if not preserve_history:
            self.__init__(dimension_count=self.dimension_count)
            return
        
        # Preserve history but reset processing state
        self.layers["numogram_core"] = {}
        self.layers["bwo_processor"] = {}
        self.state_hash = None
        self.stability_coefficient = 1.0
