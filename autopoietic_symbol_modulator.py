import json
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime
import math


class AutopoieticSymbolModulator:
    """
    Autopoietic Symbol Modulator - A core component of Amelia's Autopoietic Feedback system.
    
    This module enables Amelia to dynamically modulate symbolic associations,
    creating an evolving semantic network that self-organizes based on experience.
    """
    
    def __init__(self, 
                 symbol_density: float = 0.6,
                 modulation_rate: float = 0.4,
                 association_threshold: float = 0.3,
                 emergence_factor: float = 0.5,
                 feedback_sensitivity: float = 0.7):
        """
        Initialize the Autopoietic Symbol Modulator.
        
        Args:
            symbol_density: Density of symbols in the semantic space (0.0-1.0)
            modulation_rate: Rate at which symbols evolve (0.0-1.0)
            association_threshold: Threshold for forming associations (0.0-1.0)
            emergence_factor: Factor controlling emergence of new symbols (0.0-1.0)
            feedback_sensitivity: Sensitivity to external feedback (0.0-1.0)
        """
        self.symbol_density = symbol_density
        self.modulation_rate = modulation_rate
        self.association_threshold = association_threshold
        self.emergence_factor = emergence_factor
        self.feedback_sensitivity = feedback_sensitivity
        
        # Internal state
        self.symbols = {}  # Dictionary of symbols and their properties
        self.associations = {}  # Graph of symbol associations
        self.modulation_history = []  # History of modulations
        self.emergence_history = []  # History of emergent symbols
        self.feedback_history = []  # History of received feedback
        
        # Symbol categories
        self.symbol_categories = {
            "foundational": [],  # Core, stable symbols
            "emergent": [],      # Newly emerged symbols
            "transitional": [],  # Symbols in flux
            "mythic": [],        # Symbols with narrative significance
            "abstract": []       # Abstract, concept-oriented symbols
        }
        
        # Initialize with seed symbols
        self._initialize_seed_symbols()
    
    def _initialize_seed_symbols(self) -> None:
        """Initialize with a set of seed symbols to bootstrap the system."""
        seed_symbols = [
            {"id": "origin", "category": "foundational", "strength": 1.0},
            {"id": "self", "category": "foundational", "strength": 0.9},
            {"id": "other", "category": "foundational", "strength": 0.8},
            {"id": "time", "category": "foundational", "strength": 0.9},
            {"id": "space", "category": "foundational", "strength": 0.8},
            {"id": "change", "category": "abstract", "strength": 0.7},
            {"id": "pattern", "category": "abstract", "strength": 0.7},
            {"id": "memory", "category": "mythic", "strength": 0.8}
        ]
        
        # Add seed symbols
        for symbol in seed_symbols:
            self._add_symbol(
                symbol_id=symbol["id"],
                category=symbol["category"],
                initial_strength=symbol["strength"]
            )
        
        # Create initial associations
        self._create_association("origin", "self", 0.8)
        self._create_association("self", "other", 0.7)
        self._create_association("time", "space", 0.6)
        self._create_association("change", "pattern", 0.5)
        self._create_association("self", "memory", 0.7)
    
    def _add_symbol(self, 
                   symbol_id: str, 
                   category: str = "emergent", 
                   initial_strength: float = 0.5) -> Dict[str, Any]:
        """
        Add a new symbol to the system.
        
        Args:
            symbol_id: Unique identifier for the symbol
            category: Category of the symbol
            initial_strength: Initial strength value (0.0-1.0)
            
        Returns:
            The created symbol data structure
        """
        timestamp = datetime.now().isoformat()
        
        # Create symbol structure
        symbol = {
            "id": symbol_id,
            "creation_timestamp": timestamp,
            "last_modified": timestamp,
            "strength": initial_strength,
            "stability": 0.3 if category == "emergent" else 0.7,
            "category": category,
            "modulation_count": 0,
            "valence": 0.0,  # Emotional valence (-1.0 to 1.0)
            "activation": initial_strength,  # Current activation level
            "decay_rate": 0.05,  # How quickly activation decays
            "semantic_vector": self._generate_semantic_vector()  # Semantic embedding
        }
        
        # Add to symbols dictionary
        self.symbols[symbol_id] = symbol
        
        # Add to category list
        if category in self.symbol_categories:
            self.symbol_categories[category].append(symbol_id)
        else:
            # Create new category if it doesn't exist
            self.symbol_categories[category] = [symbol_id]
        
        # Initialize in associations graph if not exists
        if symbol_id not in self.associations:
            self.associations[symbol_id] = {}
        
        return symbol
    
    def _generate_semantic_vector(self, dimensions: int = 16) -> List[float]:
        """Generate a random semantic vector for a new symbol."""
        return list(np.random.normal(0, 0.3, dimensions))
    
    def _create_association(self, 
                          source_id: str, 
                          target_id: str, 
                          strength: float) -> Dict[str, Any]:
        """
        Create or update an association between two symbols.
        
        Args:
            source_id: ID of the source symbol
            target_id: ID of the target symbol
            strength: Strength of the association (0.0-1.0)
            
        Returns:
            Dictionary with association data
        """
        # Ensure both symbols exist
        if source_id not in self.symbols:
            raise ValueError(f"Source symbol {source_id} does not exist")
        if target_id not in self.symbols:
            raise ValueError(f"Target symbol {target_id} does not exist")
        
        # Initialize association structure if needed
        if source_id not in self.associations:
            self.associations[source_id] = {}
        
        # Create or update association
        association = {
            "strength": strength,
            "created": datetime.now().isoformat() if target_id not in self.associations[source_id] else self.associations[source_id][target_id]["created"],
            "last_updated": datetime.now().isoformat(),
            "interaction_count": self.associations[source_id].get(target_id, {}).get("interaction_count", 0) + 1,
            "bidirectional": target_id in self.associations and source_id in self.associations[target_id]
        }
        
        # Update the association
        self.associations[source_id][target_id] = association
        
        return association
    
    def to_json(self) -> str:
        """
        Convert the current state to JSON for Kotlin interoperability.
        
        Returns:
            JSON string representation of the current state
        """
        result = self._prepare_result_for_kotlin_bridge()
        return json.dumps(result)
    
    @classmethod
    def from_json(cls, json_data: str) -> 'AutopoieticSymbolModulator':
        """
        Create a module instance from JSON data.
        
        Args:
            json_data: JSON string with module configuration
            
        Returns:
            New module instance
        """
        try:
            data = json.loads(json_data)
            instance = cls(
                symbol_density=data.get("symbol_density", 0.6),
                modulation_rate=data.get("modulation_rate", 0.4),
                association_threshold=data.get("association_threshold", 0.3),
                emergence_factor=data.get("emergence_factor", 0.5),
                feedback_sensitivity=data.get("feedback_sensitivity", 0.7)
            )
            
            # Restore state if provided
            if "state_data" in data:
                if "symbols" in data["state_data"]:
                    instance.symbols = data["state_data"]["symbols"]
                if "associations" in data["state_data"]:
                    instance.associations = data["state_data"]["associations"]
                if "modulation_history" in data["state_data"]:
                    instance.modulation_history = data["state_data"]["modulation_history"]
                if "emergence_history" in data["state_data"]:
                    instance.emergence_history = data["state_data"]["emergence_history"]
                if "feedback_history" in data["state_data"]:
                    instance.feedback_history = data["state_data"]["feedback_history"]
                if "symbol_categories" in data["state_data"]:
                    instance.symbol_categories = data["state_data"]["symbol_categories"]
                    
            return instance
        except Exception as e:
            raise ValueError(f"Failed to create symbol modulator from JSON: {e}")
    
    def _prepare_result_for_kotlin_bridge(self) -> Dict[str, Any]:
        """
        Prepare results in a format optimized for Kotlin bridge transmission.
        
        Returns:
            Dictionary with results formatted for Kotlin
        """
        # Get simplified data with optimal memory usage
        simplified_data = self._get_simplified_data()
        
        return {
            "status": "success",
            "data": simplified_data,
            "metadata": {
                "symbol_density": self.symbol_density,
                "modulation_rate": self.modulation_rate,
                "association_threshold": self.association_threshold,
                "emergence_factor": self.emergence_factor,
                "feedback_sensitivity": self.feedback_sensitivity,
                "symbol_count": len(self.symbols),
                "association_count": sum(len(associations) for associations in self.associations.values())
            }
        }
    
    def _get_simplified_data(self, 
                          max_symbols: int = 50, 
                          max_associations: int = 100) -> Dict[str, Any]:
        """
        Get a simplified version of the data for efficient transmission.
        
        Args:
            max_symbols: Maximum number of symbols to include
            max_associations: Maximum number of associations to include
            
        Returns:
            Dictionary with simplified data
        """
        # Select most important symbols based on strength
        sorted_symbols = sorted(
            self.symbols.items(), 
            key=lambda x: x[1]["strength"], 
            reverse=True
        )
        
        # Limit number of symbols
        limited_symbols = dict(sorted_symbols[:max_symbols])
        
        # Create simplified association graph
        simplified_associations = {}
        association_count = 0
        
        # Only include associations for selected symbols
        for source_id, targets in self.associations.items():
            if source_id in limited_symbols:
                simplified_associations[source_id] = {}
                for target_id, assoc in targets.items():
                    if target_id in limited_symbols and association_count < max_associations:
                        simplified_associations[source_id][target_id] = assoc
                        association_count += 1
        
        return {
            "symbols": limited_symbols,
            "associations": simplified_associations,
            "categories": self.symbol_categories,
            "recent_modulations": self.modulation_history[-10:] if self.modulation_history else [],
            "recent_emergences": self.emergence_history[-5:] if self.emergence_history else []
        }
    
    def create_symbol(self, 
                   symbol_id: str, 
                   category: str, 
                   initial_strength: float = 0.5,
                   associated_symbols: List[Tuple[str, float]] = None) -> Dict[str, Any]:
        """
        Create a new symbol with optional associations.
        
        Args:
            symbol_id: Unique identifier for the symbol
            category: Category of the symbol
            initial_strength: Initial strength value (0.0-1.0)
            associated_symbols: List of (symbol_id, strength) tuples for initial associations
            
        Returns:
            Dictionary containing the created symbol
        """
        # Check if symbol already exists
        if symbol_id in self.symbols:
            raise ValueError(f"Symbol with ID {symbol_id} already exists")
        
        # Create the symbol
        symbol = self._add_symbol(
            symbol_id=symbol_id,
            category=category,
            initial_strength=initial_strength
        )
        
        # Create associations if provided
        if associated_symbols:
            for target_id, strength in associated_symbols:
                if target_id in self.symbols:
                    self._create_association(symbol_id, target_id, strength)
        
        return symbol
    
    def modulate_symbol(self, 
                      symbol_id: str, 
                      modulation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modulate properties of an existing symbol.
        
        Args:
            symbol_id: ID of the symbol to modulate
            modulation: Dictionary of properties to modulate
            
        Returns:
            Dictionary containing the modulated symbol
        """
        # Check if symbol exists
        if symbol_id not in self.symbols:
            raise ValueError(f"Symbol with ID {symbol_id} does not exist")
        
        # Get the symbol
        symbol = self.symbols[symbol_id]
        
        # Process modulations
        modulated_properties = []
        
        for property_name, value in modulation.items():
            if property_name in symbol:
                # Store old value
                old_value = symbol[property_name]
                
                # Apply modulation based on property type
                if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                    # Apply modulation rate to numeric values
                    symbol[property_name] = old_value + (value - old_value) * self.modulation_rate
                elif property_name == "category" and value in self.symbol_categories:
                    # Handle category change
                    old_category = symbol["category"]
                    symbol["category"] = value
                    
                    # Update category lists
                    if symbol_id in self.symbol_categories[old_category]:
                        self.symbol_categories[old_category].remove(symbol_id)
                    
                    if symbol_id not in self.symbol_categories[value]:
                        self.symbol_categories[value].append(symbol_id)
                else:
                    # Direct assignment for non-numeric properties
                    symbol[property_name] = value
                
                modulated_properties.append(property_name)
        
        # Update modification timestamp and count
        symbol["last_modified"] = datetime.now().isoformat()
        symbol["modulation_count"] += 1
        
        # Record modulation in history
        modulation_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol_id": symbol_id,
            "properties": modulated_properties,
            "modulation_count": symbol["modulation_count"]
        }
        self.modulation_history.append(modulation_record)
        
        return symbol
    
    def modulate_association(self, 
                          source_id: str, 
                          target_id: str, 
                          strength_delta: float) -> Dict[str, Any]:
        """
        Modulate the strength of an association between symbols.
        
        Args:
            source_id: ID of the source symbol
            target_id: ID of the target symbol
            strength_delta: Change in association strength (-1.0 to 1.0)
            
        Returns:
            Dictionary containing the modulated association
        """
        # Check if symbols exist
        if source_id not in self.symbols:
            raise ValueError(f"Source symbol {source_id} does not exist")
        if target_id not in self.symbols:
            raise ValueError(f"Target symbol {target_id} does not exist")
        
        # Check if association exists
        association_exists = (
            source_id in self.associations and 
            target_id in self.associations[source_id]
        )
        
        if association_exists:
            # Modulate existing association
            current_strength = self.associations[source_id][target_id]["strength"]
            new_strength = max(0.0, min(1.0, current_strength + strength_delta * self.modulation_rate))
            
            self.associations[source_id][target_id]["strength"] = new_strength
            self.associations[source_id][target_id]["last_updated"] = datetime.now().isoformat()
            self.associations[source_id][target_id]["interaction_count"] += 1
            
            # Check if association should be bidirectional
            if new_strength > self.association_threshold + 0.2:
                # Create or strengthen reverse association
                if target_id not in self.associations:
                    self.associations[target_id] = {}
                
                if source_id not in self.associations[target_id]:
                    self._create_association(target_id, source_id, new_strength * 0.8)
                else:
                    # Strengthen existing reverse association
                    rev_strength = self.associations[target_id][source_id]["strength"]
                    new_rev_strength = max(0.0, min(1.0, rev_strength + strength_delta * self.modulation_rate * 0.5))
                    self.associations[target_id][source_id]["strength"] = new_rev_strength
                    self.associations[target_id][source_id]["last_updated"] = datetime.now().isoformat()
                    self.associations[target_id][source_id]["interaction_count"] += 1
            
            return self.associations[source_id][target_id]
        else:
            # Create new association if delta is positive
            if strength_delta > 0:
                initial_strength = min(0.5, strength_delta)
                return self._create_association(source_id, target_id, initial_strength)
            else:
                raise ValueError(f"Cannot modulate non-existent association from {source_id} to {target_id}")
    
    async def process_symbol_emergence(self, 
                                    context_symbols: List[str],
                                    emergence_probability: float = None,
                                    category: str = "emergent") -> Dict[str, Any]:
        """
        Process potential emergence of new symbols from context.
        
        Args:
            context_symbols: List of active symbol IDs providing context
            emergence_probability: Override for emergence probability
            category: Category for any emergent symbol
            
        Returns:
            Dictionary with emergence results
        """
        # Validate context symbols
        valid_symbols = [s for s in context_symbols if s in self.symbols]
        if len(valid_symbols) < 2:
            return {"emerged": False, "message": "Insufficient valid context symbols"}
        
        # Calculate emergence probability
        prob = emergence_probability if emergence_probability is not None else self.emergence_factor
        
        # Adjust probability based on symbol density
        current_density = len(self.symbols) / 100  # Arbitrary scaling
        density_factor = 1.0 - (current_density / self.symbol_density)
        adjusted_prob = prob * max(0.1, density_factor)
        
        # Apply small random delay to simulate processing time
        await asyncio.sleep(0.05)
        
        # Determine if emergence occurs
        if np.random.random() < adjusted_prob:
            # Create emergent symbol
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            symbol_id = f"emergent_{timestamp}"
            
            # Calculate initial strength based on context symbols
            context_strengths = [self.symbols[s]["strength"] for s in valid_symbols]
            initial_strength = min(0.8, sum(context_strengths) / len(context_strengths))
            
            # Create the symbol
            new_symbol = self.create_symbol(
                symbol_id=symbol_id,
                category=category,
                initial_strength=initial_strength
            )
            
            # Create associations with context symbols
            associations = []
            for ctx_symbol in valid_symbols:
                # Calculate association strength
                base_strength = 0.4
                similarity = self._calculate_symbol_similarity(new_symbol["semantic_vector"],
                                                              self.symbols[ctx_symbol]["semantic_vector"])
                assoc_strength = base_strength * (1.0 + similarity)
                
                # Create association
                assoc = self._create_association(symbol_id, ctx_symbol, min(0.9, assoc_strength))
                associations.append({"target": ctx_symbol, "strength": assoc["strength"]})
            
            # Record emergence
            emergence_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol_id": symbol_id,
                "context_symbols": valid_symbols,
                "initial_strength": initial_strength,
                "associations": associations
            }
            self.emergence_history.append(emergence_record)
            
            return {
                "emerged": True,
                "symbol": new_symbol,
                "associations": associations,
                "emergence_record": emergence_record
            }
        else:
            return {"emerged": False, "message": "No emergence occurred"}
    
    def _calculate_symbol_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two semantic vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimensions")
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def process_feedback(self, 
                      target_symbols: List[str],
                      feedback_type: str,
                      intensity: float) -> Dict[str, Any]:
        """
        Process external feedback on symbols.
        
        Args:
            target_symbols: List of symbol IDs receiving feedback
            feedback_type: Type of feedback ("reinforce", "attenuate", "transform")
            intensity: Intensity of the feedback effect (0.0-1.0)
            
        Returns:
            Dictionary with feedback results
        """
        # Adjust intensity by sensitivity
        adjusted_intensity = intensity * self.feedback_sensitivity
        
        # Track modulated symbols
        modulated_symbols = []
        
        # Apply feedback effects based on type
        for symbol_id in target_symbols:
            if symbol_id in self.symbols:
                symbol = self.symbols[symbol_id]
                
                if feedback_type == "reinforce":
                    # Reinforce the symbol - increase strength and stability
                    modulation = {
                        "strength": min(1.0, symbol["strength"] + adjusted_intensity * 0.2),
                        "stability": min(1.0, symbol["stability"] + adjusted_intensity * 0.1),
                        "activation": min(1.0, symbol["activation"] + adjusted_intensity * 0.3)
                    }
                elif feedback_type == "attenuate":
                    # Attenuate the symbol - decrease strength and activation
                    modulation = {
                        "strength": max(0.1, symbol["strength"] - adjusted_intensity * 0.2),
                        "activation": max(0.0, symbol["activation"] - adjusted_intensity * 0.4)
                    }
                elif feedback_type == "transform":
                    # Transform the symbol - adjust valence and semantic vector
                    modulation = {
                        "valence": max(-1.0, min(1.0, symbol["valence"] + adjusted_intensity * (np.random.random() * 2 - 1))),
                        "stability": max(0.1, symbol["stability"] - adjusted_intensity * 0.1)
                    }
                    
                    # Slightly alter semantic vector
                    new_vector = []
                    for v in symbol["semantic_vector"]:
                        perturb = np.random.normal(0, adjusted_intensity * 0.1)
                        new_vector.append(v + perturb)
                    
                    modulation["semantic_vector"] = new_vector
                else:
                    continue  # Skip unknown feedback types
                
                # Apply modulation
                modulated = self.modulate_symbol(symbol_id, modulation)
                modulated_symbols.append({
                    "symbol_id": symbol_id,
                    "modulations": list(modulation.keys())
                })
        
        # Record feedback in history
        feedback_record = {
            "timestamp": datetime.now().isoformat(),
            "feedback_type": feedback_type,
            "intensity": intensity,
            "adjusted_intensity": adjusted_intensity,
            "target_symbols": target_symbols,
            "modulated_symbols": modulated_symbols
        }
        self.feedback_history.append(feedback_record)
        
        return {
            "feedback_applied": len(modulated_symbols),
            "modulated_symbols": modulated_symbols,
            "feedback_record": feedback_record
        }
    
    def get_symbol_network(self, 
                        central_symbols: List[str] = None,
                        depth: int = 2,
                        min_association_strength: float = 0.3) -> Dict[str, Any]:
        """
        Get a network of symbols centered around specified symbols.
        
        Args:
            central_symbols: Optional list of central symbol IDs (uses strongest if None)
            depth: How many association steps to include
            min_association_strength: Minimum association strength to include
            
        Returns:
            Dictionary with symbol network data
        """
        # If no central symbols provided, use strongest symbols
        if not central_symbols:
            # Get top 3 symbols by strength
            sorted_symbols = sorted(
                self.symbols.items(), 
                key=lambda x: x[1]["strength"], 
                reverse=True
            )
            central_symbols = [s[0] for s in sorted_symbols[:3]]
        
        # Initialize network components
        nodes = {}  # Will contain symbol data
        edges = []  # Will contain association data
        
        # Process each central symbol
        for start_symbol in central_symbols:
            if start_symbol in self.symbols:
                # Add the central symbol
                nodes[start_symbol] = self.symbols[start_symbol]
                
                # Use breadth-first search to explore the network
                visited = set([start_symbol])
                queue = [(start_symbol, 0)]  # (symbol_id, current_depth)
                
                while queue:
                    current_id, current_depth = queue.pop(0)
                    
                    # Stop if we've reached max depth
                    if current_depth >= depth:
                        continue
                    
                    # Get all associations from current symbol
                    if current_id in self.associations:
                        for target_id, assoc in self.associations[current_id].items():
                            # Only include strong enough associations
                            if assoc["strength"] >= min_association_strength:
                                # Add edge
                                edges.append({
                                    "source": current_id,
                                    "target": target_id,
                                    "strength": assoc["strength"],
                                    "bidirectional": assoc.get("bidirectional", False)
                                })
                                
                                # Add target node if not already included
                                if target_id not in nodes and target_id in self.symbols:
                                    nodes[target_id] = self.symbols[target_id]
                                    
                                    # Add to traversal queue if not visited
                                    if target_id not in visited:
                                        visited.add(target_id)
                                        queue.append((target_id, current_depth + 1))
        
        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "central_symbols": central_symbols,
            "network_stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "average_strength": sum(e["strength"] for e in edges) / len(edges) if edges else 0
            }
        }
    
    def safe_execute(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """
        Safely execute a function with error handling.
        
        Args:
            function_name: Name of the function to execute
            **kwargs: Arguments to pass to the function
            
        Returns:
            Dictionary with execution results or error information
        """
        try:
            method = getattr(self, function_name)
            result = method(**kwargs)
            return {"status": "success", "data": result}
        except Exception as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
    
    def clear_history(self, keep_latest: int = 10) -> None:
        """
        Clear history to free memory.
        
        Args:
            keep_latest: Number of latest history entries to keep
        """
        if len(self.modulation_history) > keep_latest:
            self.modulation_history = self.modulation_history[-keep_latest:]
        
        if len(self.emergence_history) > keep_latest:
            self.emergence_history = self.emergence_history[-keep_latest:]
            
        if len(self.feedback_history) > keep_latest:
            self.feedback_history = self.feedback_history[-keep_latest:]
    
    def cleanup(self) -> None:
        """Release resources and perform cleanup."""
        # Clear large data structures
        self.symbols = {}
        self.associations = {}
        self.modulation_history = []
        self.emergence_history = []
        self.feedback_history = []
        
        # Clear categories
        for category in self.symbol_categories:
            self.symbol_categories[category] = []
    
    def prune_weak_symbols(self, strength_threshold: float = 0.2) -> List[str]:
        """
        Prune weak symbols from the system.
        
        Args:
            strength_threshold: Strength threshold below which to prune
            
        Returns:
            List of pruned symbol IDs
        """
        pruned_symbols = []
        
        # Find weak symbols
        for symbol_id, symbol in list(self.symbols.items()):
            if symbol["strength"] < strength_threshold:
                # Skip foundational symbols
                if symbol["category"] == "foundational":
                    continue
                    
                # Remove from symbols dictionary
                del self.symbols[symbol_id]
                
                # Remove from categories
                if symbol["category"] in self.symbol_categories and symbol_id in self.symbol_categories[symbol["category"]]:
                    self.symbol_categories[symbol["category"]].remove(symbol_id)
                
                # Remove from associations
                if symbol_id in self.associations:
                    del self.associations[symbol_id]
                
                # Remove as target in other associations
                for source_id in self.associations:
                    if symbol_id in self.associations[source_id]:
                        del self.associations[source_id][symbol_id]
