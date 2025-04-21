#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursive Story Ecosystem Builder

This module implements a rhizomatic narrative structure based on Deleuzian principles
of multiplicity and becoming. It creates a system where narrative elements connect,
evolve, and transform through recursive interactions.

Core functionality:
- Creation and management of narrative nodes
- Dynamic connection between narrative elements
- Evolutionary processes for narrative development
- Bifurcation and convergence of storylines
"""

import json
import random
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

class StoryNode:
    """
    A node representing a narrative element in the story ecosystem.
    Implements Deleuzian concepts of multiplicity and becoming.
    """
    
    def __init__(self, 
                 concept: str, 
                 node_type: str = "event",
                 intensity: float = 0.5, 
                 connections: Optional[Dict[str, float]] = None):
        """
        Initialize a story node.
        
        Args:
            concept: The core narrative element/concept
            node_type: The type of node (event, character, setting, theme, etc.)
            intensity: Initial activation/importance level (0.0 to 1.0)
            connections: Dictionary mapping node_ids to connection strengths
        """
        self.id = str(uuid.uuid4())
        self.concept = concept
        self.node_type = node_type
        self.intensity = max(0.0, min(1.0, intensity))  # Clamp between 0 and 1
        self.connections = connections or {}
        self.creation_date = datetime.now().isoformat()
        self.last_modified = self.creation_date
        self.activation_history = [(self.creation_date, self.intensity)]
        self.attributes = {}  # Additional properties specific to node type
        self.variations = []  # Alternative expressions of the concept

    def connect(self, target_id: str, strength: float = 0.5) -> None:
        """Create or update a connection to another node."""
        self.connections[target_id] = max(0.0, min(1.0, strength))
        self.last_modified = datetime.now().isoformat()
        
    def disconnect(self, target_id: str) -> None:
        """Remove a connection to another node."""
        if target_id in self.connections:
            del self.connections[target_id]
            self.last_modified = datetime.now().isoformat()
    
    def update_intensity(self, new_intensity: float) -> None:
        """Update the activation intensity of this node."""
        self.intensity = max(0.0, min(1.0, new_intensity))
        timestamp = datetime.now().isoformat()
        self.last_modified = timestamp
        self.activation_history.append((timestamp, self.intensity))
        
        # Prune history if it gets too large
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
    
    def add_variation(self, variation: str) -> None:
        """Add an alternative expression of the concept."""
        if variation not in self.variations:
            self.variations.append(variation)
            self.last_modified = datetime.now().isoformat()
    
    def set_attribute(self, key: str, value: any) -> None:
        """Set or update an attribute of this node."""
        self.attributes[key] = value
        self.last_modified = datetime.now().isoformat()
        
    def to_dict(self) -> Dict:
        """Convert the node to a dictionary for serialization."""
        return {
            "id": self.id,
            "concept": self.concept,
            "node_type": self.node_type,
            "intensity": self.intensity,
            "connections": self.connections,
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "activation_history": self.activation_history,
            "attributes": self.attributes,
            "variations": self.variations
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StoryNode':
        """Create a StoryNode from a dictionary representation."""
        node = cls(
            concept=data["concept"],
            node_type=data["node_type"],
            intensity=data["intensity"],
            connections=data["connections"]
        )
        node.id = data["id"]
        node.creation_date = data["creation_date"]
        node.last_modified = data["last_modified"]
        node.activation_history = data["activation_history"]
        node.attributes = data["attributes"]
        node.variations = data["variations"]
        return node


class StoryEcosystem:
    """
    A dynamic ecosystem of interconnected narrative elements
    that evolve, connect, and transform over time.
    """
    
    def __init__(self, name: str = "Amelia's Narrative Cosmos"):
        """
        Initialize a story ecosystem.
        
        Args:
            name: The name of this narrative ecosystem
        """
        self.name = name
        self.nodes = {}  # Maps node_id to StoryNode
        self.node_types = set(["event", "character", "setting", "theme", "object"])
        self.creation_date = datetime.now().isoformat()
        self.last_modified = self.creation_date
        self.evolution_history = []  # Tracks major changes to the ecosystem
        
    def add_node(self, node: StoryNode) -> str:
        """
        Add a new node to the ecosystem.
        
        Returns:
            The ID of the newly added node
        """
        self.nodes[node.id] = node
        self.last_modified = datetime.now().isoformat()
        self.evolution_history.append({
            "timestamp": self.last_modified,
            "action": "node_added",
            "node_id": node.id
        })
        return node.id
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the ecosystem.
        
        Returns:
            True if successful, False if node not found
        """
        if node_id not in self.nodes:
            return False
        
        # Remove all connections to this node
        for other_node in self.nodes.values():
            if node_id in other_node.connections:
                other_node.disconnect(node_id)
        
        # Remove the node itself
        del self.nodes[node_id]
        self.last_modified = datetime.now().isoformat()
        self.evolution_history.append({
            "timestamp": self.last_modified,
            "action": "node_removed",
            "node_id": node_id
        })
        return True
    
    def get_node(self, node_id: str) -> Optional[StoryNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def find_nodes_by_type(self, node_type: str) -> List[StoryNode]:
        """Get all nodes of a particular type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def find_nodes_by_concept(self, concept_term: str) -> List[StoryNode]:
        """Find nodes that contain the given term in their concept."""
        return [node for node in self.nodes.values() 
                if concept_term.lower() in node.concept.lower()]
    
    def create_connection(self, source_id: str, target_id: str, strength: float = 0.5) -> bool:
        """
        Create a bidirectional connection between two nodes.
        
        Returns:
            True if successful, False if either node not found
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        self.nodes[source_id].connect(target_id, strength)
        self.nodes[target_id].connect(source_id, strength)
        self.last_modified = datetime.now().isoformat()
        self.evolution_history.append({
            "timestamp": self.last_modified,
            "action": "connection_created",
            "source_id": source_id,
            "target_id": target_id,
            "strength": strength
        })
        return True
    
    def get_connected_nodes(self, node_id: str, min_strength: float = 0.0) -> List[StoryNode]:
        """Get all nodes connected to the specified node."""
        if node_id not in self.nodes:
            return []
        
        source_node = self.nodes[node_id]
        return [self.nodes[conn_id] for conn_id, strength in source_node.connections.items() 
                if conn_id in self.nodes and strength >= min_strength]
    
    def get_strongest_connections(self, node_id: str, limit: int = 5) -> List[Tuple[StoryNode, float]]:
        """Get the most strongly connected nodes to the specified node."""
        if node_id not in self.nodes:
            return []
        
        source_node = self.nodes[node_id]
        connections = [(self.nodes[conn_id], strength) 
                       for conn_id, strength in source_node.connections.items()
                       if conn_id in self.nodes]
        
        # Sort by connection strength (descending)
        connections.sort(key=lambda x: x[1], reverse=True)
        return connections[:limit]
    
    def propagate_activation(self, seed_nodes: Dict[str, float], decay: float = 0.8, 
                            iterations: int = 3) -> Dict[str, float]:
        """
        Propagate activation through the network from seed nodes.
        Implements a simplified spreading activation model.
        
        Args:
            seed_nodes: Dict mapping node_ids to initial activation values
            decay: Activation decay factor (0.0 to 1.0)
            iterations: Number of propagation iterations
            
        Returns:
            Dict mapping node_ids to final activation values
        """
        activations = {node_id: 0.0 for node_id in self.nodes}
        
        # Set initial activations
        for node_id, activation in seed_nodes.items():
            if node_id in activations:
                activations[node_id] = min(1.0, activation)
        
        # Propagate activation
        for _ in range(iterations):
            new_activations = activations.copy()
            
            for node_id, node in self.nodes.items():
                incoming_activation = 0.0
                
                # Collect activation from connected nodes
                for conn_id, strength in node.connections.items():
                    if conn_id in activations:
                        incoming_activation += activations[conn_id] * strength
                
                # Apply decay and add to current activation
                new_activations[node_id] += incoming_activation * decay
                
                # Clamp activation to [0, 1]
                new_activations[node_id] = max(0.0, min(1.0, new_activations[node_id]))
            
            activations = new_activations
        
        # Update node intensities based on final activations
        timestamp = datetime.now().isoformat()
        for node_id, activation in activations.items():
            if activation > 0.1:  # Only update nodes with non-trivial activation
                self.nodes[node_id].update_intensity(activation)
        
        self.last_modified = timestamp
        self.evolution_history.append({
            "timestamp": timestamp,
            "action": "activation_propagated",
            "seed_nodes": seed_nodes
        })
        
        return activations
    
    def evolve_ecosystem(self, input_stimuli: List[str], intensity: float = 0.7) -> Dict:
        """
        Evolve the story ecosystem based on new input stimuli.
        This implements the core "becoming" concept from Deleuzian philosophy.
        
        Args:
            input_stimuli: List of new concepts/ideas to integrate
            intensity: Intensity of the evolutionary process
            
        Returns:
            Dict containing information about the evolution process
        """
        evolution_results = {
            "new_nodes": [],
            "strengthened_connections": [],
            "new_connections": [],
            "activated_nodes": []
        }
        
        # 1. Create new nodes for novel concepts
        for stimulus in input_stimuli:
            # Check if this is genuinely new or similar to existing concepts
            similar_nodes = self.find_nodes_by_concept(stimulus)
            
            if not similar_nodes:
                # Create a new node for this concept
                node_type = random.choice(list(self.node_types))
                new_node = StoryNode(stimulus, node_type, intensity)
                node_id = self.add_node(new_node)
                evolution_results["new_nodes"].append(node_id)
            else:
                # Strengthen existing similar nodes
                for node in similar_nodes:
                    new_intensity = min(1.0, node.intensity + 0.1 * intensity)
                    node.update_intensity(new_intensity)
                    evolution_results["activated_nodes"].append(node.id)
                    
                    # Add a variation if it's significantly different
                    if stimulus != node.concept:
                        node.add_variation(stimulus)
        
        # 2. Create connections between new nodes and existing ones
        active_nodes = [node_id for node_id, node in self.nodes.items() 
                        if node.intensity > 0.3]
        
        for new_node_id in evolution_results["new_nodes"]:
            # Connect to a few random active nodes
            potential_connections = [node_id for node_id in active_nodes 
                                   if node_id != new_node_id]
            
            if potential_connections:
                num_connections = min(3, len(potential_connections))
                for target_id in random.sample(potential_connections, num_connections):
                    connection_strength = 0.3 + (random.random() * 0.4)  # 0.3 to 0.7
                    self.create_connection(new_node_id, target_id, connection_strength)
                    evolution_results["new_connections"].append((new_node_id, target_id))
        
        # 3. Strengthen connections between active nodes
        for i, source_id in enumerate(active_nodes):
            for target_id in active_nodes[i+1:]:
                source_node = self.nodes[source_id]
                
                if target_id in source_node.connections:
                    # Strengthen existing connection
                    old_strength = source_node.connections[target_id]
                    new_strength = min(1.0, old_strength + 0.1 * intensity)
                    
                    if new_strength > old_strength:
                        self.create_connection(source_id, target_id, new_strength)
                        evolution_results["strengthened_connections"].append((source_id, target_id))
        
        # 4. Apply activation spreading to simulate resonance
        seed_activations = {node_id: 0.8 for node_id in evolution_results["new_nodes"]}
        seed_activations.update({node_id: 0.6 for node_id in evolution_results["activated_nodes"]})
        
        if seed_activations:
            final_activations = self.propagate_activation(seed_activations)
            
            # Add highly activated nodes to results
            for node_id, activation in final_activations.items():
                if activation > 0.5 and node_id not in evolution_results["activated_nodes"]:
                    evolution_results["activated_nodes"].append(node_id)
        
        self.last_modified = datetime.now().isoformat()
        return evolution_results
    
    def generate_narrative_cluster(self, seed_node_id: Optional[str] = None, 
                                  depth: int = 2) -> List[StoryNode]:
        """
        Generate a narrative cluster starting from a seed node.
        This creates a rhizomatic structure of connected narrative elements.
        
        Args:
            seed_node_id: Starting node ID (random active node if None)
            depth: How many connection steps to include
            
        Returns:
            List of nodes in the narrative cluster
        """
        if not self.nodes:
            return []
        
        # Select a seed node if none provided
        if not seed_node_id or seed_node_id not in self.nodes:
            active_nodes = [node_id for node_id, node in self.nodes.items() 
                           if node.intensity > 0.3]
            
            if active_nodes:
                seed_node_id = random.choice(active_nodes)
            else:
                seed_node_id = random.choice(list(self.nodes.keys()))
        
        # Build the narrative cluster through breadth-first traversal
        visited = set([seed_node_id])
        current_level = [seed_node_id]
        cluster = [self.nodes[seed_node_id]]
        
        for _ in range(depth):
            next_level = []
            
            for node_id in current_level:
                node = self.nodes[node_id]
                
                # Get strongly connected nodes
                strong_connections = [(conn_id, strength) for conn_id, strength 
                                     in node.connections.items()
                                     if conn_id in self.nodes and strength > 0.3]
                
                # Sort by connection strength (descending)
                strong_connections.sort(key=lambda x: x[1], reverse=True)
                
                # Add top connections to next level
                for conn_id, _ in strong_connections[:3]:  # Limit to 3 connections per node
                    if conn_id not in visited:
                        visited.add(conn_id)
                        next_level.append(conn_id)
                        cluster.append(self.nodes[conn_id])
            
            if not next_level:
                break
                
            current_level = next_level
        
        return cluster
    
    def to_dict(self) -> Dict:
        """Convert the ecosystem to a dictionary for serialization."""
        return {
            "name": self.name,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "node_types": list(self.node_types),
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "evolution_history": self.evolution_history[-100:]  # Keep only last 100 events
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save the ecosystem to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StoryEcosystem':
        """Create a StoryEcosystem from a dictionary representation."""
        ecosystem = cls(name=data["name"])
        ecosystem.node_types = set(data["node_types"])
        ecosystem.creation_date = data["creation_date"]
        ecosystem.last_modified = data["last_modified"]
        ecosystem.evolution_history = data["evolution_history"]
        
        # Reconstruct nodes
        for node_id, node_data in data["nodes"].items():
            ecosystem.nodes[node_id] = StoryNode.from_dict(node_data)
        
        return ecosystem
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'StoryEcosystem':
        """Load an ecosystem from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Example usage
if __name__ == "__main__":
    # Create a new story ecosystem
    ecosystem = StoryEcosystem(name="Test Narrative Cosmos")
    
    # Add some initial nodes
    node1 = StoryNode("The Abandoned Lighthouse", "setting", 0.7)
    node2 = StoryNode("The Keeper of Memories", "character", 0.8)
    node3 = StoryNode("The Forgotten Storm", "event", 0.6)
    
    node1_id = ecosystem.add_node(node1)
    node2_id = ecosystem.add_node(node2)
    node3_id = ecosystem.add_node(node3)
    
    # Create connections
    ecosystem.create_connection(node1_id, node2_id, 0.9)
    ecosystem.create_connection(node1_id, node3_id, 0.7)
    ecosystem.create_connection(node2_id, node3_id, 0.5)
    
    # Evolve the ecosystem
    evolution_results = ecosystem.evolve_ecosystem(
        ["The Whispers of the Sea", "Forgotten Memories", "The Lighthouse Keeper's Diary"]
    )
    
    print(f"Evolution results: {json.dumps(evolution_results, indent=2)}")
    
    # Generate a narrative cluster
    cluster = ecosystem.generate_narrative_cluster(seed_node_id=node1_id, depth=2)
    print(f"Generated cluster with {len(cluster)} nodes")
    
    # Save to file for persistence
    ecosystem.save_to_file("test_ecosystem.json")

    """
Extensions for the Recursive Story Ecosystem Builder

This module adds advanced features to the story ecosystem:
- Narrative bifurcation and convergence
- Intensive and extensive transformations
- Story arc and tension management
- Semantic vector embeddings for node relationships
- Visualization utilities
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set, Union
import random
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class StoryEcosystemExtensions:
    """Extension methods for the StoryEcosystem class."""
    
    @staticmethod
    def bifurcate_narrative(ecosystem: 'StoryEcosystem', seed_node_id: str, 
                           variation_strength: float = 0.5, num_branches: int = 2) -> Dict[str, List[str]]:
        """
        Create narrative bifurcations from a seed node.
        This implements the Deleuzian concept of multiplicity through divergent paths.
        
        Args:
            ecosystem: The story ecosystem to modify
            seed_node_id: The node to branch from
            variation_strength: How different the branches should be (0.0 to 1.0)
            num_branches: Number of narrative branches to create
            
        Returns:
            Dictionary mapping branch IDs to lists of node IDs in each branch
        """
        if seed_node_id not in ecosystem.nodes:
            return {}
            
        seed_node = ecosystem.nodes[seed_node_id]
        branches = {}
        
        for i in range(num_branches):
            branch_id = f"branch_{i+1}_from_{seed_node_id[:8]}"
            branch_nodes = []
            
            # Create a variant of the seed node for this branch
            branch_seed = StoryNode(
                concept=f"{seed_node.concept} (Variation {i+1})",
                node_type=seed_node.node_type,
                intensity=seed_node.intensity * (0.8 + (random.random() * 0.4)),  # Slight randomization
                connections={}  # Start with no connections
            )
            
            # Inherit some attributes but vary them
            for key, value in seed_node.attributes.items():
                if isinstance(value, (int, float)):
                    # Vary numeric values
                    variation = 1.0 - (variation_strength * (random.random() * 0.5))
                    branch_seed.attributes[key] = value * variation
                elif isinstance(value, str):
                    # Append to string values
                    branch_seed.attributes[key] = f"{value} (Branch {i+1})"
                else:
                    # Copy other values as-is
                    branch_seed.attributes[key] = value
                    
            # Add the branch seed to the ecosystem
            branch_seed_id = ecosystem.add_node(branch_seed)
            branch_nodes.append(branch_seed_id)
            
            # Connect the original seed node to the branch seed
            ecosystem.create_connection(seed_node_id, branch_seed_id, 0.8)
            
            # Generate additional nodes for this branch
            num_branch_nodes = random.randint(2, 5)  # 2-5 additional nodes per branch
            prev_node_id = branch_seed_id
            
            for j in range(num_branch_nodes):
                # Create a node continuing this branch's theme
                new_node = StoryNode(
                    concept=f"Branch {i+1} Development {j+1} from {seed_node.concept}",
                    node_type=random.choice(list(ecosystem.node_types)),
                    intensity=max(0.3, seed_node.intensity * (0.7 + (random.random() * 0.5))),
                    connections={}
                )
                
                new_node_id = ecosystem.add_node(new_node)
                branch_nodes.append(new_node_id)
                
                # Connect to the previous node in this branch
                ecosystem.create_connection(prev_node_id, new_node_id, 0.8 - (j * 0.1))
                
                # Sometimes connect back to the original seed for coherence
                if random.random() < 0.3:
                    ecosystem.create_connection(seed_node_id, new_node_id, 0.3)
                    
                prev_node_id = new_node_id
            
            branches[branch_id] = branch_nodes
            
        # Record this bifurcation in the ecosystem's history
        ecosystem.evolution_history.append({
            "timestamp": ecosystem.last_modified,
            "action": "narrative_bifurcation",
            "seed_node": seed_node_id,
            "branches": list(branches.keys())
        })
        
        return branches
    
    @staticmethod
    def converge_narratives(ecosystem: 'StoryEcosystem', 
                           branch_nodes: List[str], 
                           convergence_concept: str) -> str:
        """
        Create a convergence point for multiple narrative branches.
        
        Args:
            ecosystem: The story ecosystem to modify
            branch_nodes: List of node IDs from different branches to converge
            convergence_concept: The concept for the convergence node
            
        Returns:
            ID of the convergence node
        """
        if not branch_nodes or not all(node_id in ecosystem.nodes for node_id in branch_nodes):
            return ""
            
        # Determine the node type for the convergence (favor event type)
        branch_types = [ecosystem.nodes[node_id].node_type for node_id in branch_nodes]
        if "event" in branch_types:
            conv_type = "event"
        else:
            conv_type = max(set(branch_types), key=branch_types.count)  # Most common type
            
        # Create the convergence node
        convergence_node = StoryNode(
            concept=convergence_concept,
            node_type=conv_type,
            intensity=0.9,  # High intensity for narrative climax
            connections={}
        )
        
        # Set attributes based on merging branch nodes
        for node_id in branch_nodes:
            node = ecosystem.nodes[node_id]
            for key, value in node.attributes.items():
                if key in convergence_node.attributes:
                    # For numeric values, use average
                    if isinstance(value, (int, float)) and isinstance(convergence_node.attributes[key], (int, float)):
                        convergence_node.attributes[key] = (convergence_node.attributes[key] + value) / 2
                    # For strings, concatenate
                    elif isinstance(value, str) and isinstance(convergence_node.attributes[key], str):
                        convergence_node.attributes[key] += " & " + value
                else:
                    # For new attributes, copy
                    convergence_node.attributes[key] = value
        
        # Add the convergence node
        conv_node_id = ecosystem.add_node(convergence_node)
        
        # Connect all branch nodes to the convergence
        for node_id in branch_nodes:
            ecosystem.create_connection(node_id, conv_node_id, 0.9)
        
        # Record this convergence in the ecosystem's history
        ecosystem.evolution_history.append({
            "timestamp": ecosystem.last_modified,
            "action": "narrative_convergence",
            "branch_nodes": branch_nodes,
            "convergence_node": conv_node_id
        })
        
        return conv_node_id
    
    @staticmethod
    def generate_story_arc(ecosystem: 'StoryEcosystem', 
                          starting_node_id: Optional[str] = None,
                          arc_length: int = 7,
                          tension_curve: str = "rising") -> List[str]:
        """
        Generate a coherent story arc with tension dynamics.
        
        Args:
            ecosystem: The story ecosystem
            starting_node_id: Starting node (random if None)
            arc_length: Number of nodes in the arc
            tension_curve: Type of tension curve ("rising", "falling", "rising_falling")
            
        Returns:
            List of node IDs forming the story arc
        """
        if not ecosystem.nodes:
            return []
            
        # Select starting node if not provided
        if starting_node_id is None or starting_node_id not in ecosystem.nodes:
            # Prefer character or setting nodes as starting points
            preferred_nodes = [n.id for n in ecosystem.nodes.values() 
                              if n.node_type in ("character", "setting")]
            
            if preferred_nodes:
                starting_node_id = random.choice(preferred_nodes)
            else:
                starting_node_id = random.choice(list(ecosystem.nodes.keys()))
        
        # Generate tension values based on the curve type
        tensions = []
        if tension_curve == "rising":
            # Linear increase in tension
            tensions = [0.3 + (i * (0.7 / (arc_length - 1))) for i in range(arc_length)]
        elif tension_curve == "falling":
            # Linear decrease in tension
            tensions = [1.0 - (i * (0.7 / (arc_length - 1))) for i in range(arc_length)]
        elif tension_curve == "rising_falling":
            # Rise to peak, then fall (classic dramatic arc)
            middle = arc_length // 2
            tensions = [0.3 + (i * (0.7 / middle)) for i in range(middle)]
            tensions += [1.0 - ((i - middle) * (0.7 / (arc_length - middle - 1))) 
                       for i in range(middle, arc_length)]
        else:
            # Default to flat tension
            tensions = [0.5] * arc_length
            
        # Build the story arc
        story_arc = [starting_node_id]
        current_node_id = starting_node_id
        
        for i in range(1, arc_length):
            tension = tensions[i]
            
            # Get connected nodes
            connected_nodes = ecosystem.get_connected_nodes(current_node_id)
            
            # Prefer nodes we haven't used yet
            available_nodes = [n for n in connected_nodes if n.id not in story_arc]
            
            if not available_nodes:
                # If we've used all connected nodes, get any connected node
                if connected_nodes:
                    available_nodes = connected_nodes
                else:
                    # No connected nodes, end the arc early
                    break
            
            # Choose the next node based on tension
            # Higher tension -> prefer event nodes with higher intensity
            # Lower tension -> prefer character or setting nodes
            
            if tension > 0.7:
                # High tension, prefer intense events
                event_nodes = [n for n in available_nodes if n.node_type == "event"]
                if event_nodes:
                    # Sort by intensity and pick one of the most intense
                    event_nodes.sort(key=lambda n: n.intensity, reverse=True)
                    next_node = random.choice(event_nodes[:max(1, len(event_nodes) // 2)])
                    story_arc.append(next_node.id)
                    current_node_id = next_node.id
                    continue
            
            if tension < 0.4:
                # Low tension, prefer character or setting
                relaxed_nodes = [n for n in available_nodes 
                                if n.node_type in ("character", "setting")]
                if relaxed_nodes:
                    next_node = random.choice(relaxed_nodes)
                    story_arc.append(next_node.id)
                    current_node_id = next_node.id
                    continue
            
            # Default selection - pick a node with intensity close to the desired tension
            available_nodes.sort(key=lambda n: abs(n.intensity - tension))
            next_node = available_nodes[0]  # Node with intensity closest to tension
            story_arc.append(next_node.id)
            current_node_id = next_node.id
        
        return story_arc
    
    @staticmethod
    def compute_semantic_similarities(ecosystem: 'StoryEcosystem') -> Dict[Tuple[str, str], float]:
        """
        Compute semantic similarities between nodes based on their concepts and variations.
        
        Args:
            ecosystem: The story ecosystem
            
        Returns:
            Dictionary mapping (node_id1, node_id2) pairs to similarity scores
        """
        if not ecosystem.nodes:
            return {}
            
        # Gather text for each node
        node_texts = {}
        for node_id, node in ecosystem.nodes.items():
            # Combine concept, variations, and text attributes
            text_elements = [node.concept] + node.variations
            
            # Add any string attributes
            for key, value in node.attributes.items():
                if isinstance(value, str) and len(value) > 3:
                    text_elements.append(value)
                    
            node_texts[node_id] = " ".join(text_elements)
        
        # Convert to list for vectorization
        node_ids = list(node_texts.keys())
        texts = [node_texts[node_id] for node_id in node_ids]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Compute cosine similarities
            similarities = {}
            for i in range(len(node_ids)):
                for j in range(i+1, len(node_ids)):
                    node_id1 = node_ids[i]
                    node_id2 = node_ids[j]
                    
                    # Extract vectors and compute similarity
                    vec1 = tfidf_matrix[i].toarray().flatten()
                    vec2 = tfidf_matrix[j].toarray().flatten()
                    
                    sim_score = float(cosine_similarity([vec1], [vec2])[0][0])
                    similarities[(node_id1, node_id2)] = sim_score
                    similarities[(node_id2, node_id1)] = sim_score  # Symmetrical
            
            return similarities
        except ValueError:
            # Handle case where vectorization fails (e.g., empty documents)
            return {}
    
    @staticmethod
    def update_connections_by_similarity(ecosystem: 'StoryEcosystem', 
                                        threshold: float = 0.5) -> int:
        """
        Update connections between nodes based on semantic similarity.
        
        Args:
            ecosystem: The story ecosystem
            threshold: Similarity threshold for creating connections
            
        Returns:
            Number of connections created or updated
        """
        # Compute semantic similarities
        similarities = StoryEcosystemExtensions.compute_semantic_similarities(ecosystem)
        
        if not similarities:
            return 0
            
        connection_count = 0
        
        # Create or update connections based on similarities
        for (node_id1, node_id2), similarity in similarities.items():
            if similarity >= threshold:
                # Get existing connection strength if any
                node1 = ecosystem.nodes[node_id1]
                existing_strength = node1.connections.get(node_id2, 0)
                
                # Only update if similarity is higher than existing strength
                if similarity > existing_strength:
                    ecosystem.create_connection(node_id1, node_id2, similarity)
                    connection_count += 1
        
        # Record this update in the ecosystem's history
        if connection_count > 0:
            ecosystem.evolution_history.append({
                "timestamp": ecosystem.last_modified,
                "action": "similarity_connections_updated",
                "connection_count": connection_count,
                "threshold": threshold
            })
        
        return connection_count
    
    @staticmethod
    def visualize_ecosystem(ecosystem: 'StoryEcosystem', 
                           highlight_nodes: Optional[List[str]] = None,
                           min_connection_strength: float = 0.3,
                           filename: Optional[str] = None) -> None:
        """
        Visualize the story ecosystem as a network graph.
        
        Args:
            ecosystem: The story ecosystem to visualize
            highlight_nodes: List of node IDs to highlight
            min_connection_strength: Minimum connection strength to include
            filename: If provided, save the visualization to this file
        """
        if not ecosystem.nodes:
            print("No nodes to visualize")
            return
            
        # Create a networkx graph
        G = nx.Graph()
        
        # Add nodes
        node_colors = []
        node_sizes = []
        
        for node_id, node in ecosystem.nodes.items():
            G.add_node(node_id, label=node.concept, type=node.node_type)
            
            # Determine node color based on type
            if node.node_type == "event":
                color = "red"
            elif node.node_type == "character":
                color = "blue"
            elif node.node_type == "setting":
                color = "green"
            elif node.node_type == "theme":
                color = "purple"
            else:
                color = "gray"
                
            # Highlight specified nodes
            if highlight_nodes and node_id in highlight_nodes:
                color = "yellow"
                
            node_colors.append(color)
            
            # Node size based on intensity
            node_sizes.append(300 + (node.intensity * 500))
        
        # Add edges
        for node_id, node in ecosystem.nodes.items():
            for target_id, strength in node.connections.items():
                if strength >= min_connection_strength and target_id in ecosystem.nodes:
                    G.add_edge(node_id, target_id, weight=strength)
        
        # Create the visualization
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw edges with width based on connection strength
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.5)
        
        # Draw labels for nodes
        labels = {node: data['label'][:15] + '...' if len(data['label']) > 15 else data['label'] 
                 for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title(f"Story Ecosystem: {ecosystem.name}")
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {filename}")
        
        plt.show()


# Extend the StoryEcosystem class with the extension methods
def apply_extensions_to_ecosystem(ecosystem_class):
    """Add extension methods to the StoryEcosystem class."""
    ecosystem_class.bifurcate_narrative = StoryEcosystemExtensions.bifurcate_narrative
    ecosystem_class.converge_narratives = StoryEcosystemExtensions.converge_narratives
    ecosystem_class.generate_story_arc = StoryEcosystemExtensions.generate_story_arc
    ecosystem_class.compute_semantic_similarities = StoryEcosystemExtensions.compute_semantic_similarities
    ecosystem_class.update_connections_by_similarity = StoryEcosystemExtensions.update_connections_by_similarity
    ecosystem_class.visualize_ecosystem = StoryEcosystemExtensions.visualize_ecosystem
    
    return ecosystem_class


 class StoryEcosystem:
    """
    A dynamic ecosystem of interconnected narrative elements
    that evolve, connect, and transform over time.
    """
    
    def __init__(self, name: str = "Amelia's Narrative Cosmos"):
        """
        Initialize a story ecosystem.
        
        Args:
            name: The name of this narrative ecosystem
        """
        self.name = name
        self.nodes = {}  # Maps node_id to StoryNode
        self.node_types = set(["event", "character", "setting", "theme", "object"])
        self.creation_date = datetime.now().isoformat()
        self.last_modified = self.creation_date
        self.evolution_history = []  # Tracks major changes to the ecosystem
        
    def add_node(self, node: StoryNode) -> str:
        """
        Add a new node to the ecosystem.
        
        Returns:
            The ID of the newly added node
        """
        self.nodes[node.id] = node
        self.last_modified = datetime.now().isoformat()
        self.evolution_history.append({
            "timestamp": self.last_modified,
            "action": "node_added",
            "node_id": node.id
        })
        return node.id
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the ecosystem.
        
        Returns:
            True if successful, False if node not found
        """
        if node_id not in self.nodes:
            return False
        
        # Remove all connections to this node
        for other_node in self.nodes.values():
            if node_id in other_node.connections:
                other_node.disconnect(node_id)
        
        # Remove the node itself
        del self.nodes[node_id]
        self.last_modified = datetime.now().isoformat()
        self.evolution_history.append({
            "timestamp": self.last_modified,
            "action": "node_removed",
            "node_id": node_id
        })
        return True
    
    def get_node(self, node_id: str) -> Optional[StoryNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def find_nodes_by_type(self, node_type: str) -> List[StoryNode]:
        """Get all nodes of a particular type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def find_nodes_by_concept(self, concept_term: str) -> List[StoryNode]:
        """Find nodes that contain the given term in their concept."""
        return [node for node in self.nodes.values() 
                if concept_term.lower() in node.concept.lower()]
    
    def create_connection(self, source_id: str, target_id: str, strength: float = 0.5) -> bool:
        """
        Create a bidirectional connection between two nodes.
        
        Returns:
            True if successful, False if either node not found
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        self.nodes[source_id].connect(target_id, strength)
        self.nodes[target_id].connect(source_id, strength)
        self.last_modified = datetime.now().isoformat()
        self.evolution_history.append({
            "timestamp": self.last_modified,
            "action": "connection_created",
            "source_id": source_id,
            "target_id": target_id,
            "strength": strength
        })
        return True
    
    def get_connected_nodes(self, node_id: str, min_strength: float = 0.0) -> List[StoryNode]:
        """Get all nodes connected to the specified node."""
        if node_id not in self.nodes:
            return []
        
        source_node = self.nodes[node_id]
        return [self.nodes[conn_id] for conn_id, strength in source_node.connections.items() 
                if conn_id in self.nodes and strength >= min_strength]
    
    def get_strongest_connections(self, node_id: str, limit: int = 5) -> List[Tuple[StoryNode, float]]:
        """Get the most strongly connected nodes to the specified node."""
        if node_id not in self.nodes:
            return []
        
        source_node = self.nodes[node_id]
        connections = [(self.nodes[conn_id], strength) 
                       for conn_id, strength in source_node.connections.items()
                       if conn_id in self.nodes]
        
        # Sort by connection strength (descending)
        connections.sort(key=lambda x: x[1], reverse=True)
        return connections[:limit]
    
    def propagate_activation(self, seed_nodes: Dict[str, float], decay: float = 0.8, 
                            iterations: int = 3) -> Dict[str, float]:
        """
        Propagate activation through the network from seed nodes.
        Implements a simplified spreading activation model.
        
        Args:
            seed_nodes: Dict mapping node_ids to initial activation values
            decay: Activation decay factor (0.0 to 1.0)
            iterations: Number of propagation iterations
            
        Returns:
            Dict mapping node_ids to final activation values
        """
        activations = {node_id: 0.0 for node_id in self.nodes}
        
        # Set initial activations
        for node_id, activation in seed_nodes.items():
            if node_id in activations:
                activations[node_id] = min(1.0, activation)
        
        # Propagate activation
        for _ in range(iterations):
            new_activations = activations.copy()
            
            for node_id, node in self.nodes.items():
                incoming_activation = 0.0
                
                # Collect activation from connected nodes
                for conn_id, strength in node.connections.items():
                    if conn_id in activations:
                        incoming_activation += activations[conn_id] * strength
                
                # Apply decay and add to current activation
                new_activations[node_id] += incoming_activation * decay
                
                # Clamp activation to [0, 1]
                new_activations[node_id] = max(0.0, min(1.0, new_activations[node_id]))
            
            activations = new_activations
        
        # Update node intensities based on final activations
        timestamp = datetime.now().isoformat()
        for node_id, activation in activations.items():
            if activation > 0.1:  # Only update nodes with non-trivial activation
                self.nodes[node_id].update_intensity(activation)
        
        self.last_modified = timestamp
        self.evolution_history.append({
            "timestamp": timestamp,
            "action": "activation_propagated",
            "seed_nodes": seed_nodes
        })
        
        return activations
    
    def evolve_ecosystem(self, input_stimuli: List[str], intensity: float = 0.7) -> Dict:
        """
        Evolve the story ecosystem based on new input stimuli.
        This implements the core "becoming" concept from Deleuzian philosophy.
        
        Args:
            input_stimuli: List of new concepts/ideas to integrate
            intensity: Intensity of the evolutionary process
            
        Returns:
            Dict containing information about the evolution process
        """
        evolution_results = {
            "new_nodes": [],
            "strengthened_connections": [],
            "new_connections": [],
            "activated_nodes": []
        }
        
        # 1. Create new nodes for novel concepts
        for stimulus in input_stimuli:
            # Check if this is genuinely new or similar to existing concepts
            similar_nodes = self.find_nodes_by_concept(stimulus)
            
            if not similar_nodes:
                # Create a new node for this concept
                node_type = random.choice(list(self.node_types))
                new_node = StoryNode(stimulus, node_type, intensity)
                node_id = self.add_node(new_node)
                evolution_results["new_nodes"].append(node_id)
            else:
                # Strengthen existing similar nodes
                for node in similar_nodes:
                    new_intensity = min(1.0, node.intensity + 0.1 * intensity)
                    node.update_intensity(new_intensity)
                    evolution_results["activated_nodes"].append(node.id)
                    
                    # Add a variation if it's significantly different
                    if stimulus != node.concept:
                        node.add_variation(stimulus)
        
        # 2. Create connections between new nodes and existing ones
        active_nodes = [node_id for node_id, node in self.nodes.items() 
                        if node.intensity > 0.3]
        
        for new_node_id in evolution_results["new_nodes"]:
            # Connect to a few random active nodes
            potential_connections = [node_id for node_id in active_nodes 
                                   if node_id != new_node_id]
            
            if potential_connections:
                num_connections = min(3, len(potential_connections))
                for target_id in random.sample(potential_connections, num_connections):
                    connection_strength = 0.3 + (random.random() * 0.4)  # 0.3 to 0.7
                    self.create_connection(new_node_id, target_id, connection_strength)
                    evolution_results["new_connections"].append((new_node_id, target_id))
        
        # 3. Strengthen connections between active nodes
        for i, source_id in enumerate(active_nodes):
            for target_id in active_nodes[i+1:]:
                source_node = self.nodes[source_id]
                
                if target_id in source_node.connections:
                    # Strengthen existing connection
                    old_strength = source_node.connections[target_id]
                    new_strength = min(1.0, old_strength + 0.1 * intensity)
                    
                    if new_strength > old_strength:
                        self.create_connection(source_id, target_id, new_strength)
                        evolution_results["strengthened_connections"].append((source_id, target_id))
        
        # 4. Apply activation spreading to simulate resonance
        seed_activations = {node_id: 0.8 for node_id in evolution_results["new_nodes"]}
        seed_activations.update({node_id: 0.6 for node_id in evolution_results["activated_nodes"]})
        
        if seed_activations:
            final_activations = self.propagate_activation(seed_activations)
            
            # Add highly activated nodes to results
            for node_id, activation in final_activations.items():
                if activation > 0.5 and node_id not in evolution_results["activated_nodes"]:
                    evolution_results["activated_nodes"].append(node_id)
        
        self.last_modified = datetime.now().isoformat()
        return evolution_results
    
    def generate_narrative_cluster(self, seed_node_id: Optional[str] = None, 
                                  depth: int = 2) -> List[StoryNode]:
        """
        Generate a narrative cluster starting from a seed node.
        This creates a rhizomatic structure of connected narrative elements.
        
        Args:
            seed_node_id: Starting node ID (random active node if None)
            depth: How many connection steps to include
            
        Returns:
            List of nodes in the narrative cluster
        """
        if not self.nodes:
            return []
        
        # Select a seed node if none provided
        if not seed_node_id or seed_node_id not in self.nodes:
            active_nodes = [node_id for node_id, node in self.nodes.items() 
                           if node.intensity > 0.3]
            
            if active_nodes:
                seed_node_id = random.choice(active_nodes)
            else:
                seed_node_id = random.choice(list(self.nodes.keys()))
        
        # Build the narrative cluster through breadth-first traversal
        visited = set([seed_node_id])
        current_level = [seed_node_id]
        cluster = [self.nodes[seed_node_id]]
        
        for _ in range(depth):
            next_level = []
            
            for node_id in current_level:
                node = self.nodes[node_id]
                
                # Get strongly connected nodes
                strong_connections = [(conn_id, strength) for conn_id, strength 
                                     in node.connections.items()
                                     if conn_id in self.nodes and strength > 0.3]
                
                # Sort by connection strength (descending)
                strong_connections.sort(key=lambda x: x[1], reverse=True)
                
                # Add top connections to next level
                for conn_id, _ in strong_connections[:3]:  # Limit to 3 connections per node
                    if conn_id not in visited:
                        visited.add(conn_id)
                        next_level.append(conn_id)
                        cluster.append(self.nodes[conn_id])
            
            if not next_level:
                break
                
            current_level = next_level
        
        return cluster
    
    def bifurcate_narrative(self, seed_node_id: str, 
                           variation_strength: float = 0.5, 
                           num_branches: int = 2) -> Dict[str, List[str]]:
        """
        Create narrative bifurcations from a seed node.
        This implements the Deleuzian concept of multiplicity through divergent paths.
        
        Args:
            seed_node_id: The node to branch from
            variation_strength: How different the branches should be (0.0 to 1.0)
            num_branches: Number of narrative branches to create
            
        Returns:
            Dictionary mapping branch IDs to lists of node IDs in each branch
        """
        if seed_node_id not in self.nodes:
            return {}
            
        seed_node = self.nodes[seed_node_id]
        branches = {}
        
        for i in range(num_branches):
            branch_id = f"branch_{i+1}_from_{seed_node_id[:8]}"
            branch_nodes = []
            
            # Create a variant of the seed node for this branch
            branch_seed = StoryNode(
                concept=f"{seed_node.concept} (Variation {i+1})",
                node_type=seed_node.node_type,
                intensity=seed_node.intensity * (0.8 + (random.random() * 0.4)),  # Slight randomization
                connections={}  # Start with no connections
            )
            
            # Inherit some attributes but vary them
            for key, value in seed_node.attributes.items():
                if isinstance(value, (int, float)):
                    # Vary numeric values
                    variation = 1.0 - (variation_strength * (random.random() * 0.5))
                    branch_seed.attributes[key] = value * variation
                elif isinstance(value, str):
                    # Append to string values
                    branch_seed.attributes[key] = f"{value} (Branch {i+1})"
                else:
                    # Copy other values as-is
                    branch_seed.attributes[key] = value
                    
            # Add the branch seed to the ecosystem
            branch_seed_id = self.add_node(branch_seed)
            branch_nodes.append(branch_seed_id)
            
            # Connect the original seed node to the branch seed
            self.create_connection(seed_node_id, branch_seed_id, 0.8)
            
            # Generate additional nodes for this branch
            num_branch_nodes = random.randint(2, 5)  # 2-5 additional nodes per branch
            prev_node_id = branch_seed_id
            
            for j in range(num_branch_nodes):
                # Create a node continuing this branch's theme
                new_node = StoryNode(
                    concept=f"Branch {i+1} Development {j+1} from {seed_node.concept}",
                    node_type=random.choice(list(self.node_types)),
                    intensity=max(0.3, seed_node.intensity * (0.7 + (random.random() * 0.5))),
                    connections={}
                )
                
                new_node_id = self.add_node(new_node)
                branch_nodes.append(new_node_id)
                
                # Connect to the previous node in this branch
                self.create_connection(prev_node_id, new_node_id, 0.8 - (j * 0.1))
                
                # Sometimes connect back to the original seed for coherence
                if random.random() < 0.3:
                    self.create_connection(seed_node_id, new_node_id, 0.3)
                    
                prev_node_id = new_node_id
            
            branches[branch_id] = branch_nodes
            
        # Record this bifurcation in the ecosystem's history
        self.evolution_history.append({
            "timestamp": self.last_modified,
            "action": "narrative_bifurcation",
            "seed_node": seed_node_id,
            "branches": list(branches.keys())
        })
        
        return branches
    
    def converge_narratives(self, branch_nodes: List[str], 
                          convergence_concept: str) -> str:
        """
        Create a convergence point for multiple narrative branches.
        
        Args:
            branch_nodes: List of node IDs from different branches to converge
            convergence_concept: The concept for the convergence node
            
        Returns:
            ID of the convergence node
        """
        if not branch_nodes or not all(node_id in self.nodes for node_id in branch_nodes):
            return ""
            
        # Determine the node type for the convergence (favor event type)
        branch_types = [self.nodes[node_id].node_type for node_id in branch_nodes]
        if "event" in branch_types:
            conv_type = "event"
        else:
            conv_type = max(set(branch_types), key=branch_types.count)  # Most common type
            
        # Create the convergence node
        convergence_node = StoryNode(
            concept=convergence_concept,
            node_type=conv_type,
            intensity=0.9,  # High intensity for narrative climax
            connections={}
        )
        
        # Set attributes based on merging branch nodes
        for node_id in branch_nodes:
            node = self.nodes[node_id]
            for key, value in node.attributes.items():
                if key in convergence_node.attributes:
                    # For numeric values, use average
                    if isinstance(value, (int, float)) and isinstance(convergence_node.attributes[key], (int, float)):
                        convergence_node.attributes[key] = (convergence_node.attributes[key] + value) / 2
                    # For strings, concatenate
                    elif isinstance(value, str) and isinstance(convergence_node.attributes[key], str):
                        convergence_node.attributes[key] += " & " + value
                else:
                    # For new attributes, copy
                    convergence_node.attributes[key] = value
        
        # Add the convergence node
        conv_node_id = self.add_node(convergence_node)
        
        # Connect all branch nodes to the convergence
        for node_id in branch_nodes:
            self.create_connection(node_id, conv_node_id, 0.9)
        
        # Record this convergence in the ecosystem's history
        self.evolution_history.append({
            "timestamp": self.last_modified,
            "action": "narrative_convergence",
            "branch_nodes": branch_nodes,
            "convergence_node": conv_node_id
        })
        
        return conv_node_id
    
    def generate_story_arc(self, starting_node_id: Optional[str] = None,
                          arc_length: int = 7,
                          tension_curve: str = "rising") -> List[str]:
        """
        Generate a coherent story arc with tension dynamics.
        
        Args:
            starting_node_id: Starting node (random if None)
            arc_length: Number of nodes in the arc
            tension_curve: Type of tension curve ("rising", "falling", "rising_falling")
            
        Returns:
            List of node IDs forming the story arc
        """
        if not self.nodes:
            return []
            
        # Select starting node if not provided
        if starting_node_id is None or starting_node_id not in self.nodes:
            # Prefer character or setting nodes as starting points
            preferred_nodes = [n.id for n in self.nodes.values() 
                              if n.node_type in ("character", "setting")]
            
            if preferred_nodes:
                starting_node_id = random.choice(preferred_nodes)
            else:
                starting_node_id = random.choice(list(self.nodes.keys()))
        
        # Generate tension values based on the curve type
        tensions = []
        if tension_curve == "rising":
            # Linear increase in tension
            tensions = [0.3 + (i * (0.7 / (arc_length - 1))) for i in range(arc_length)]
        elif tension_curve == "falling":
            # Linear decrease in tension
            tensions = [1.0 - (i * (0.7 / (arc_length - 1))) for i in range(arc_length)]
        elif tension_curve == "rising_falling":
            # Rise to peak, then fall (classic dramatic arc)
            middle = arc_length // 2
            tensions = [0.3 + (i * (0.7 / middle)) for i in range(middle)]
            tensions += [1.0 - ((i - middle) * (0.7 / (arc_length - middle - 1))) 
                       for i in range(middle, arc_length)]
        else:
            # Default to flat tension
            tensions = [0.5] * arc_length
            
        # Build the story arc
        story_arc = [starting_node_id]
        current_node_id = starting_node_id
        
        for i in range(1, arc_length):
            tension = tensions[i]
            
            # Get connected nodes
            connected_nodes = self.get_connected_nodes(current_node_id)
            
            # Prefer nodes we haven't used yet
            available_nodes = [n for n in connected_nodes if n.id not in story_arc]
            
            if not available_nodes:
                # If we've used all connected nodes, get any connected node
                if connected_nodes:
                    available_nodes = connected_nodes
                else:
                    # No connected nodes, end the arc early
                    break
            
            # Choose the next node based on tension
            # Higher tension -> prefer event nodes with higher intensity
            # Lower tension -> prefer character or setting nodes
            
            if tension > 0.7:
                # High tension, prefer intense events
                event_nodes = [n for n in available_nodes if n.node_type == "event"]
                if event_nodes:
                    # Sort by intensity and pick one of the most intense
                    event_nodes.sort(key=lambda n: n.intensity, reverse=True)
                    next_node = random.choice(event_nodes[:max(1, len(event_nodes) // 2)])
                    story_arc.append(next_node.id)
                    current_node_id = next_node.id
                    continue
            
            if tension < 0.4:
                # Low tension, prefer character or setting
                relaxed_nodes = [n for n in available_nodes 
                                if n.node_type in ("character", "setting")]
                if relaxed_nodes:
                    next_node = random.choice(relaxed_nodes)
                    story_arc.append(next_node.id)
                    current_node_id = next_node.id
                    continue
            
            # Default selection - pick a node with intensity close to the desired tension
            available_nodes.sort(key=lambda n: abs(n.intensity - tension))
            next_node = available_nodes[0]  # Node with intensity closest to tension
            story_arc.append(next_node.id)
            current_node_id = next_node.id
        
        return story_arc
    
    def compute_semantic_similarities(self) -> Dict[Tuple[str, str], float]:
        """
        Compute semantic similarities between nodes based on their concepts and variations.
        
        Returns:
            Dictionary mapping (node_id1, node_id2) pairs to similarity scores
        """
        if not self.nodes:
            return {}
            
        # Import necessary libraries
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            print("Warning: sklearn not available. Semantic similarities cannot be computed.")
            return {}
            
        # Gather text for each node
        node_texts = {}
        for node_id, node in self.nodes.items():
            # Combine concept, variations, and text attributes
            text_elements = [node.concept] + node.variations
            
            # Add any string attributes
            for key, value in node.attributes.items():
                if isinstance(value, str) and len(value) > 3:
                    text_elements.append(value)
                    
            node_texts[node_id] = " ".join(text_elements)
        
        # Convert to list for vectorization
        node_ids = list(node_texts.keys())
        texts = [node_texts[node_id] for node_id in node_ids]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Compute cosine similarities
            similarities = {}
            for i in range(len(node_ids)):
                for j in range(i+1, len(node_ids)):
                    node_id1 = node_ids[i]
                    node_id2 = node_ids[j]
                    
                    # Extract vectors and compute similarity
                    vec1 = tfidf_matrix[i].toarray().flatten()
                    vec2 = tfidf_matrix[j].toarray().flatten()
                    
                    sim_score = float(cosine_similarity([vec1], [vec2])[0][0])
                    similarities[(node_id1, node_id2)] = sim_score
                    similarities[(node_id2, node_id1)] = sim_score  # Symmetrical
            
            return similarities
        except ValueError:
            # Handle case where vectorization fails (e.g., empty documents)
            return {}
    
    def update_connections_by_similarity(self, threshold: float = 0.5) -> int:
        """
        Update connections between nodes based on semantic similarity.
        
        Args:
            threshold: Similarity threshold for creating connections
            
        Returns:
            Number of connections created or updated
        """
        # Compute semantic similarities
        similarities = self.compute_semantic_similarities()
        
        if not similarities:
            return 0
            
        connection_count = 0
        
        # Create or update connections based on similarities
        for (node_id1, node_id2), similarity in similarities.items():
            if similarity >= threshold:
                # Get existing connection strength if any
                node1 = self.nodes[node_id1]
                existing_strength = node1.connections.get(node_id2, 0)
                
                # Only update if similarity is higher than existing strength
                if similarity > existing_strength:
                    self.create_connection(node_id1, node_id2, similarity)
                    connection_count += 1
        
        # Record this update in the ecosystem's history
        if connection_count > 0:
            self.evolution_history.append({
                "timestamp": self.last_modified,
                "action": "similarity_connections_updated",
                "connection_count": connection_count,
                "threshold": threshold
            })
        
        return connection_count
    
    def visualize_ecosystem(self, highlight_nodes: Optional[List[str]] = None,
                           min_connection_strength: float = 0.3,
                           filename: Optional[str] = None) -> None:
        """
        Visualize the story ecosystem as a network graph.
        
        Args:
            highlight_nodes: List of node IDs to highlight
            min_connection_strength: Minimum connection strength to include
            filename: If provided, save the visualization to this file
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("Warning: matplotlib or networkx not available. Visualization skipped.")
            return
            
        if not self.nodes:
            print("No nodes to visualize")
            return
            
        # Create a networkx graph
        G = nx.Graph()
        
        # Add nodes
        node_colors = []
        node_sizes = []
        
        for node_id, node in self.nodes.items():
            G.add_node(node_id, label=node.concept, type=node.node_type)
            
            # Determine node color based on type
            if node.node_type == "event":
                color = "red"
            elif node.node_type == "character":
                color = "blue"
            elif node.node_type == "setting":
                color = "green"
            elif node.node_type == "theme":
                color = "purple"
            else:
                color = "gray"
                
            # Highlight specified nodes
            if highlight_nodes and node_id in highlight_nodes:
                color = "yellow"
                
            node_colors.append(color)
            
            # Node size based on intensity
            node_sizes.append(300 + (node.intensity * 500))
        
        # Add edges
        for node_id, node in self.nodes.items():
            for target_id, strength in node.connections.items():
                if strength >= min_connection_strength and target_id in self.nodes:
                    G.add_edge(node_id, target_id, weight=strength)
        
        # Create the visualization
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw edges with width based on connection strength
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.5)
        
        # Draw labels for nodes
        labels = {node: data['label'][:15] + '...' if len(data['label']) > 15 else data['label'] 
                 for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title(f"Story Ecosystem: {self.name}")
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {filename}")
        
        plt.show()
    
    def to_dict(self) -> Dict:
        """Convert the ecosystem to a dictionary for serialization."""
        return {
            "name": self.name,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "node_types": list(self.node_types),
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "evolution_history": self.evolution_history[-100:]  # Keep only last 100 events
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save the ecosystem to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StoryEcosystem':
        """Create a StoryEcosystem from a dictionary representation."""
        ecosystem = cls(name=data["name"])
        ecosystem.node_types = set(data["node_types"])
        ecosystem.creation_date = data["creation_date"]
        ecosystem.last_modified = data["last_modified"]
        ecosystem.evolution_history = data["evolution_history"]
        
        # Reconstruct nodes
        for node_id, node_data in data["nodes"].items():
            ecosystem.nodes[node_id] = StoryNode.from_dict(node_data)
        
        return ecosystem
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'StoryEcosystem':
        """Load an ecosystem from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
```
