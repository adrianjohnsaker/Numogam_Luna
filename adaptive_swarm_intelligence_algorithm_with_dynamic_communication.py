"""
Adaptive Swarm Intelligence Algorithm with Dynamic Communication

This module implements a sophisticated swarm intelligence system based on natural systems
like ant colony optimization, bees algorithms, and flocking behaviors. It enables emergent
behavior through dynamic communication, adaptive decision-making, and reinforcement learning.

Key features:
1. Dynamic Communication Protocol - Agents exchange information based on proximity and relevance
2. Adaptive Decision Making - Agents adjust behavior based on environmental feedback
3. Reinforcement Learning - Agents evolve strategies over time
4. Pheromone-like Signal System - For indirect communication and path optimization
5. Self-organizing Task Allocation - Emergent division of labor
"""

import numpy as np
import random
from enum import Enum
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
import math

class MessagePriority(Enum):
    """Defines priority levels for agent communication."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class MessageType(Enum):
    """Defines types of messages agents can exchange."""
    RESOURCE_LOCATION = 1
    THREAT_ALERT = 2
    PATH_OPTIMIZATION = 3
    TASK_REQUEST = 4
    TASK_COMPLETION = 5
    BOUNDARY_INFO = 6
    COORDINATION_SIGNAL = 7

class EnvironmentState(Enum):
    """Possible states of environment cells."""
    EMPTY = 0
    OBSTACLE = 1
    RESOURCE = 2
    THREAT = 3
    GOAL = 4

class TaskState(Enum):
    """Possible states for tasks."""
    UNASSIGNED = 0
    ASSIGNED = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4

class Agent:
    """
    Individual agent in the swarm with sensing, decision-making, 
    communication and learning capabilities.
    """
    
    def __init__(self, agent_id: int, position: Tuple[float, float], 
                 specialization: float = 0.5, learning_rate: float = 0.01):
        """
        Initialize an agent with identity and starting position.
        
        Args:
            agent_id: Unique identifier for the agent
            position: (x, y) coordinates in the environment
            specialization: Value between 0-1 representing specialization vs. generalization
            learning_rate: Rate at which agent learns from experiences
        """
        # Basic properties
        self.id = agent_id
        self.position = position
        self.velocity = (0.0, 0.0)
        self.energy = 100.0
        self.specialization = specialization
        self.learning_rate = learning_rate
        
        # Sensing and memory
        self.perception_radius = 10.0
        self.memory = defaultdict(lambda: {"value": 0.0, "timestamp": 0})
        self.knowledge_base = {}
        
        # Communication
        self.message_queue = []
        self.connections = set()  # Other agents this agent can communicate with
        self.trust_map = defaultdict(lambda: 0.5)  # Trust levels for other agents
        
        # Decision making
        self.current_task = None
        self.task_progress = 0.0
        self.strategy = {
            "exploration_rate": 0.3,
            "risk_tolerance": 0.5,
            "cooperation_bias": 0.7
        }
        
        # Learning components
        self.experience = 0
        self.skill_levels = {
            "resource_gathering": 0.5,
            "threat_avoidance": 0.5,
            "path_finding": 0.5,
            "cooperation": 0.5
        }
        
        # Action history for reinforcement learning
        self.action_history = []
        self.reward_history = []
        
        # Pheromone-like signals
        self.signal_strength = 0.0
        self.signal_type = None
        
    def sense_environment(self, environment: 'Environment', other_agents: List['Agent']) -> Dict:
        """
        Perceive the local environment within perception radius.
        
        Args:
            environment: The environment object containing state information
            other_agents: List of all other agents in the swarm
            
        Returns:
            Dictionary containing sensed information
        """
        sensed_data = {
            "resources": [],
            "obstacles": [],
            "threats": [],
            "nearby_agents": [],
            "signals": [],
            "goals": []
        }
        
        # Sense environment cells
        for x in range(int(self.position[0] - self.perception_radius),
                       int(self.position[0] + self.perception_radius + 1)):
            for y in range(int(self.position[1] - self.perception_radius),
                          int(self.position[1] + self.perception_radius + 1)):
                
                # Check if coordinates are within environment bounds
                if environment.is_valid_position((x, y)):
                    distance = math.sqrt((x - self.position[0])**2 + (y - self.position[1])**2)
                    
                    # Only sense if within perception radius
                    if distance <= self.perception_radius:
                        cell_state = environment.get_cell_state((x, y))
                        signal_strength = environment.get_signal_strength((x, y))
                        signal_type = environment.get_signal_type((x, y))
                        
                        if cell_state == EnvironmentState.RESOURCE:
                            sensed_data["resources"].append(((x, y), distance))
                        elif cell_state == EnvironmentState.OBSTACLE:
                            sensed_data["obstacles"].append(((x, y), distance))
                        elif cell_state == EnvironmentState.THREAT:
                            sensed_data["threats"].append(((x, y), distance))
                        elif cell_state == EnvironmentState.GOAL:
                            sensed_data["goals"].append(((x, y), distance))
                        
                        if signal_strength > 0:
                            sensed_data["signals"].append(((x, y), signal_type, signal_strength))
        
        # Sense other agents
        for agent in other_agents:
            if agent.id != self.id:
                distance = math.sqrt((agent.position[0] - self.position[0])**2 + 
                                    (agent.position[1] - self.position[1])**2)
                if distance <= self.perception_radius:
                    sensed_data["nearby_agents"].append((agent, distance))
                    # Add to connections if within communication range
                    if distance <= self.perception_radius * 0.8:  # Communication range slightly shorter
                        self.connections.add(agent.id)
                    else:
                        if agent.id in self.connections:
                            self.connections.remove(agent.id)
        
        return sensed_data
    
    def decide_action(self, sensed_data: Dict, swarm: 'Swarm', 
                      time_step: int) -> Tuple[str, Dict]:
        """
        Determine the next action based on sensed data and current state.
        Uses a combination of rule-based and utility-based decision making.
        
        Args:
            sensed_data: Dictionary of sensory information
            swarm: Reference to the overall swarm
            time_step: Current time step in the simulation
            
        Returns:
            Tuple of (action_name, action_parameters)
        """
        # Calculate various motivations/utilities
        utility_scores = {
            "explore": self._calc_exploration_utility(sensed_data, time_step),
            "gather": self._calc_gathering_utility(sensed_data),
            "avoid": self._calc_avoidance_utility(sensed_data),
            "communicate": self._calc_communication_utility(sensed_data, time_step),
            "rest": self._calc_rest_utility(),
            "cooperate": self._calc_cooperation_utility(sensed_data, swarm)
        }
        
        # Apply agent's strategy preferences
        utility_scores["explore"] *= (0.5 + self.strategy["exploration_rate"])
        utility_scores["avoid"] *= (1.0 - self.strategy["risk_tolerance"])
        utility_scores["cooperate"] *= self.strategy["cooperation_bias"]
        
        # Apply specialization effect
        for skill, level in self.skill_levels.items():
            if skill == "resource_gathering":
                utility_scores["gather"] *= (0.5 + level)
            elif skill == "threat_avoidance":
                utility_scores["avoid"] *= (0.5 + level)
            elif skill == "path_finding":
                utility_scores["explore"] *= (0.5 + level)
            elif skill == "cooperation":
                utility_scores["cooperate"] *= (0.5 + level)
                utility_scores["communicate"] *= (0.5 + level)
        
        # Select action with highest utility
        action_name = max(utility_scores, key=utility_scores.get)
        
        # Apply some exploration for learning
        if random.random() < max(0.05, self.strategy["exploration_rate"] - self.experience/1000):
            action_name = random.choice(list(utility_scores.keys()))
        
        # Determine parameters for the selected action
        action_params = self._get_action_parameters(action_name, sensed_data, swarm)
        
        # Record action for learning
        self.action_history.append((action_name, action_params, utility_scores))
        
        return action_name, action_params
    
    def _calc_exploration_utility(self, sensed_data: Dict, time_step: int) -> float:
        """Calculate utility of exploration based on memory and current data."""
        # Higher utility if we haven't explored much recently
        exploration_need = 0.2 + 0.8 * math.exp(-len(self.memory) / 50)
        
        # Higher utility if there are no known resources or goals
        if not sensed_data["resources"] and not sensed_data["goals"]:
            exploration_need += 0.3
            
        # Higher utility if we see signals suggesting exploration
        for _, signal_type, strength in sensed_data["signals"]:
            if signal_type == "exploration":
                exploration_need += 0.2 * strength
                
        # Periodic exploration impulse
        if time_step % 50 == 0:
            exploration_need += 0.2
            
        return min(1.0, exploration_need)
    
    def _calc_gathering_utility(self, sensed_data: Dict) -> float:
        """Calculate utility of gathering resources."""
        if not sensed_data["resources"]:
            return 0.1  # Low baseline if no resources
        
        # Resources detected - utility based on distance and energy
        closest_resource = min(sensed_data["resources"], key=lambda x: x[1])
        distance_factor = 1.0 - (closest_resource[1] / (self.perception_radius * 1.5))
        energy_need = 1.0 - (self.energy / 100.0)
        
        return 0.3 + 0.7 * distance_factor * (0.3 + 0.7 * energy_need)
    
    def _calc_avoidance_utility(self, sensed_data: Dict) -> float:
        """Calculate utility of avoiding threats."""
        if not sensed_data["threats"]:
            return 0.1  # Low baseline if no threats
        
        # Threats detected - high utility based on proximity
        closest_threat = min(sensed_data["threats"], key=lambda x: x[1])
        threat_proximity = 1.0 - (closest_threat[1] / self.perception_radius)
        
        # Almost maximum priority if threat is very close
        if closest_threat[1] < self.perception_radius * 0.3:
            return 0.9
        
        return 0.2 + 0.7 * threat_proximity
    
    def _calc_communication_utility(self, sensed_data: Dict, time_step: int) -> float:
        """Calculate utility of communicating with other agents."""
        # Higher utility with more nearby agents
        nearby_agent_count = len(sensed_data["nearby_agents"])
        
        # Higher utility if we have important info to share
        important_info = (len(self.memory) > 5 and 
                         time_step - max([info["timestamp"] for info in self.memory.values()]) < 10)
        
        # Higher utility if message queue has high priority messages
        has_high_priority = any(msg["priority"] in [MessagePriority.HIGH, MessagePriority.URGENT] 
                              for msg in self.message_queue)
        
        base_utility = 0.1
        if nearby_agent_count > 0:
            base_utility += 0.2 * min(1.0, nearby_agent_count / 5)
        if important_info:
            base_utility += 0.3
        if has_high_priority:
            base_utility += 0.3
            
        return min(0.9, base_utility)
    
    def _calc_rest_utility(self) -> float:
        """Calculate utility of resting to conserve/regain energy."""
        # Higher utility when energy is low
        energy_factor = 1.0 - (self.energy / 100.0)
        return 0.1 + 0.8 * max(0, energy_factor - 0.3)  # Only significant when energy < 70%
    
    def _calc_cooperation_utility(self, sensed_data: Dict, swarm: 'Swarm') -> float:
        """Calculate utility of cooperating with other agents on tasks."""
        # Check if there are tasks that need cooperation
        cooperative_tasks = swarm.get_cooperative_tasks()
        if not cooperative_tasks:
            return 0.1
            
        # Higher utility if there are nearby agents to cooperate with
        nearby_agents = len(sensed_data["nearby_agents"])
        
        # Find closest cooperative task
        task_distances = []
        for task in cooperative_tasks:
            task_pos = task.get("position", (0, 0))
            distance = math.sqrt((task_pos[0] - self.position[0])**2 + 
                                (task_pos[1] - self.position[1])**2)
            task_distances.append((task, distance))
            
        if not task_distances:
            return 0.1
            
        closest_task, distance = min(task_distances, key=lambda x: x[1])
        distance_factor = max(0, 1.0 - (distance / (self.perception_radius * 3)))
        
        return 0.2 + 0.6 * distance_factor * min(1.0, nearby_agents / 3)
    
    def _get_action_parameters(self, action_name: str, sensed_data: Dict, 
                              swarm: 'Swarm') -> Dict:
        """Generate parameters for the selected action."""
        params = {}
        
        if action_name == "explore":
            # Choose direction with fewer visited cells in memory
            visited_positions = set(self.memory.keys())
            candidate_directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1), 
                (0.7, 0.7), (0.7, -0.7), (-0.7, 0.7), (-0.7, -0.7)
            ]
            
            # Score each direction by how many new cells it might reveal
            direction_scores = []
            for direction in candidate_directions:
                test_pos = (self.position[0] + direction[0] * 5, 
                           self.position[1] + direction[1] * 5)
                nearby_visited = sum(1 for pos in visited_positions 
                                   if math.sqrt((pos[0] - test_pos[0])**2 + 
                                              (pos[1] - test_pos[1])**2) < 5)
                # Avoid obstacles
                obstacle_penalty = 0
                for obs_pos, _ in sensed_data["obstacles"]:
                    obs_direction = (obs_pos[0] - self.position[0], obs_pos[1] - self.position[1])
                    similarity = (obs_direction[0] * direction[0] + obs_direction[1] * direction[1]) / \
                                (math.sqrt(obs_direction[0]**2 + obs_direction[1]**2) * 
                                 math.sqrt(direction[0]**2 + direction[1]**2))
                    if similarity > 0.7:  # Direction points toward obstacle
                        obstacle_penalty += 10 * similarity
                
                direction_scores.append((direction, -nearby_visited - obstacle_penalty))
            
            best_direction = max(direction_scores, key=lambda x: x[1])[0]
            params["direction"] = best_direction
            params["speed"] = 1.0
            
        elif action_name == "gather":
            # Move toward closest resource
            closest_resource = min(sensed_data["resources"], key=lambda x: x[1])
            resource_pos = closest_resource[0]
            direction = (resource_pos[0] - self.position[0], resource_pos[1] - self.position[1])
            
            # Normalize direction vector
            magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
            if magnitude > 0:
                direction = (direction[0] / magnitude, direction[1] / magnitude)
            
            params["direction"] = direction
            params["speed"] = 1.0
            params["target_resource"] = resource_pos
            
        elif action_name == "avoid":
            # Move away from closest threat
            closest_threat = min(sensed_data["threats"], key=lambda x: x[1])
            threat_pos = closest_threat[0]
            
            # Direction vector away from threat
            direction = (self.position[0] - threat_pos[0], self.position[1] - threat_pos[1])
            
            # Normalize direction vector
            magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
            if magnitude > 0:
                direction = (direction[0] / magnitude, direction[1] / magnitude)
            
            params["direction"] = direction
            params["speed"] = 1.5  # Move faster when avoiding threats
            params["threat"] = threat_pos
            
        elif action_name == "communicate":
            # Choose agents to communicate with
            nearby_agents = sensed_data["nearby_agents"]
            if nearby_agents:
                # Prioritize agents with highest trust
                trusted_agents = sorted(nearby_agents, 
                                      key=lambda a: self.trust_map[a[0].id], 
                                      reverse=True)
                target_agents = [agent[0].id for agent in trusted_agents[:3]]
                
                # Choose message content based on knowledge
                message_content = {}
                if sensed_data["resources"]:
                    message_content["resources"] = [pos for pos, _ in sensed_data["resources"]]
                if sensed_data["threats"]:
                    message_content["threats"] = [pos for pos, _ in sensed_data["threats"]]
                
                # Information about memory is also valuable
                recent_memories = {k: v for k, v in self.memory.items() 
                                 if v["timestamp"] > swarm.time_step - 50}
                if recent_memories:
                    message_content["memory"] = dict(recent_memories)
                
                params["target_agents"] = target_agents
                params["content"] = message_content
                params["priority"] = MessagePriority.MEDIUM
                
                # Urgent if threats are present
                if sensed_data["threats"]:
                    params["priority"] = MessagePriority.HIGH
            else:
                # No nearby agents to communicate with
                params["leave_signal"] = True
            
        elif action_name == "rest":
            # Find safest nearby position to rest
            safe_positions = []
            for x in range(int(self.position[0] - 2), int(self.position[0] + 3)):
                for y in range(int(self.position[1] - 2), int(self.position[1] + 3)):
                    # Check if this position is far from threats
                    is_safe = True
                    for threat_pos, _ in sensed_data["threats"]:
                        if math.sqrt((x - threat_pos[0])**2 + (y - threat_pos[1])**2) < 5:
                            is_safe = False
                            break
                    
                    if is_safe:
                        safe_positions.append((x, y))
            
            if safe_positions:
                rest_pos = random.choice(safe_positions)
            else:
                rest_pos = self.position
                
            params["rest_position"] = rest_pos
            
        elif action_name == "cooperate":
            # Find nearest cooperative task
            cooperative_tasks = swarm.get_cooperative_tasks()
            task_distances = []
            for task in cooperative_tasks:
                task_pos = task.get("position", (0, 0))
                distance = math.sqrt((task_pos[0] - self.position[0])**2 + 
                                    (task_pos[1] - self.position[1])**2)
                task_distances.append((task, distance))
                
            if task_distances:
                closest_task, _ = min(task_distances, key=lambda x: x[1])
                task_pos = closest_task.get("position", (0, 0))
                
                # Move toward task
                direction = (task_pos[0] - self.position[0], task_pos[1] - self.position[1])
                magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
                if magnitude > 0:
                    direction = (direction[0] / magnitude, direction[1] / magnitude)
                
                params["direction"] = direction
                params["speed"] = 1.0
                params["task_id"] = closest_task.get("id")
                
                # Signal to other agents
                params["signal"] = {
                    "type": "cooperation",
                    "strength": 0.8,
                    "task_id": closest_task.get("id")
                }
            
        return params
    
    def execute_action(self, action_name: str, action_params: Dict, 
                       environment: 'Environment', swarm: 'Swarm') -> Dict:
        """
        Execute the decided action and return results.
        
        Args:
            action_name: Name of action to execute
            action_params: Parameters for the action
            environment: Environment object for interaction
            swarm: Swarm object for coordination
            
        Returns:
            Dictionary of results from the action
        """
        results = {"success": False, "reward": 0, "energy_change": 0}
        
        # Default energy cost for any action
        energy_cost = 1.0
        
        if action_name == "explore":
            # Move in the specified direction
            direction = action_params.get("direction", (0, 0))
            speed = action_params.get("speed", 1.0)
            
            new_position = (
                self.position[0] + direction[0] * speed,
                self.position[1] + direction[1] * speed
            )
            
            # Check if new position is valid
            if environment.is_valid_position(new_position):
                self.position = new_position
                self.velocity = (direction[0] * speed, direction[1] * speed)
                
                # Discover what's at the new position
                cell_state = environment.get_cell_state(new_position)
                self.memory[new_position] = {
                    "value": cell_state.value,
                    "timestamp": swarm.time_step
                }
                
                # Leave exploration signal
                environment.add_signal(self.position, "exploration", 0.5, 30)
                
                results["success"] = True
                results["new_position"] = new_position
                results["discovered"] = cell_state.name
                
                # Small reward for exploration
                results["reward"] = 0.1
                
                # Energy cost proportional to speed
                energy_cost = 1.0 * speed
            
        elif action_name == "gather":
            # Move toward resource
            direction = action_params.get("direction", (0, 0))
            speed = action_params.get("speed", 1.0)
            target_resource = action_params.get("target_resource")
            
            new_position = (
                self.position[0] + direction[0] * speed,
                self.position[1] + direction[1] * speed
            )
            
            # Check if new position is valid
            if environment.is_valid_position(new_position):
                self.position = new_position
                self.velocity = (direction[0] * speed, direction[1] * speed)
                
                # Check if reached resource
                distance_to_resource = math.sqrt(
                    (self.position[0] - target_resource[0])**2 +
                    (self.position[1] - target_resource[1])**2
                )
                
                if distance_to_resource < 1.0:
                    # Collect the resource
                    resource_value = environment.collect_resource(target_resource)
                    if resource_value > 0:
                        self.energy += resource_value
                        
                        # Leave resource signal for others
                        environment.add_signal(self.position, "resource", 0.7, 50)
                        
                        results["success"] = True
                        results["collected"] = resource_value
                        
                        # High reward for gathering
                        results["reward"] = 0.7 * resource_value
                        
                        # Energy gained
                        results["energy_change"] = resource_value
                        energy_cost = 0  # Offset by resource gain
                else:
                    results["success"] = True
                    results["new_position"] = new_position
                    results["reward"] = 0.1
                    
                    # Energy cost proportional to speed
                    energy_cost = 1.0 * speed
            
        elif action_name == "avoid":
            # Move away from threat
            direction = action_params.get("direction", (0, 0))
            speed = action_params.get("speed", 1.5)
            
            new_position = (
                self.position[0] + direction[0] * speed,
                self.position[1] + direction[1] * speed
            )
            
            # Check if new position is valid
            if environment.is_valid_position(new_position):
                self.position = new_position
                self.velocity = (direction[0] * speed, direction[1] * speed)
                
                # Leave threat signal for others
                environment.add_signal(action_params.get("threat"), "threat", 0.9, 70)
                
                results["success"] = True
                results["new_position"] = new_position
                
                # Reward for successful avoidance
                results["reward"] = 0.3
                
                # Higher energy cost due to faster movement
                energy_cost = 1.5 * speed
            
        elif action_name == "communicate":
            # Send messages to target agents
            target_agents = action_params.get("target_agents", [])
            content = action_params.get("content", {})
            priority = action_params.get("priority", MessagePriority.MEDIUM)
            
            if target_agents:
                for agent_id in target_agents:
                    message = {
                        "sender_id": self.id,
                        "receiver_id": agent_id,
                        "content": content,
                        "priority": priority,
                        "timestamp": swarm.time_step
                    }
                    swarm.send_message(message)
                
                results["success"] = True
                results["messages_sent"] = len(target_agents)
                
                # Small reward for communication
                results["reward"] = 0.2
                
                # Lower energy cost
                energy_cost = 0.5
            
            # Leave communication signal if no agents
            if action_params.get("leave_signal", False):
                environment.add_signal(self.position, "communication", 0.6, 40)
            
        elif action_name == "rest":
            # Move to rest position if needed
            rest_position = action_params.get("rest_position", self.position)
            
            if rest_position != self.position:
                direction = (
                    rest_position[0] - self.position[0],
                    rest_position[1] - self.position[1]
                )
                magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
                if magnitude > 0:
                    direction = (direction[0] / magnitude, direction[1] / magnitude)
                    
                new_position = (
                    self.position[0] + direction[0] * 0.5,  # Slow movement
                    self.position[1] + direction[1] * 0.5
                )
                
                if environment.is_valid_position(new_position):
                    self.position = new_position
                    self.velocity = (direction[0] * 0.5, direction[1] * 0.5)
            
            # Regain energy
            energy_gain = 5.0
            self.energy = min(100.0, self.energy + energy_gain)
            
            results["success"] = True
            results["energy_gained"] = energy_gain
            
            # Small reward for resting when needed
            results["reward"] = 0.1
            
            # Energy gain instead of cost
            results["energy_change"] = energy_gain
            energy_cost = -energy_gain  # Negative cost = gain
            
        elif action_name == "cooperate":
            # Move toward task
            direction = action_params.get("direction", (0, 0))
            speed = action_params.get("speed", 1.0)
            task_id = action_params.get("task_id")
            
            new_position = (
                self.position[0] + direction[0] * speed,
                self.position[1] + direction[1] * speed
            )
            
            # Check if new position is valid
            if environment.is_valid_position(new_position):
                self.position = new_position
                self.velocity = (direction[0] * speed, direction[1] * speed)
                
                # Contribute to task
                if task_id is not None:
                    contribution = swarm.contribute_to_task(task_id, self.id, 0.1)
                    
                    # Leave cooperation signal
                    if action_params.get("signal"):
                        signal_info = action_params.get("signal")
                        environment.add_signal(
                            self.position, 
                            signal_info["type"], 
                            signal_info["strength"],
                            60
                        )
                    
                    if contribution > 0:
                        results["success"] = True
                        results["task_contribution"] = contribution
                        
                        # Reward based on contribution
                        results["reward"] = 0.5 * contribution
                    else:
                        results["success"] = True
                        results["new_position"] = new_position
                        results["reward"] = 0.1
                else:
                    results["success"] = True
                    results["new_position"] = new_position
                    results["reward"] = 0.1
                    
                # Energy cost proportional to speed


        # Apply energy cost
        self.energy = max(0.0, self.energy - energy_cost)
        results["energy_change"] -= energy_cost
        
        # Record reward for learning
        self.reward_history.append(results["reward"])
        
        return results
    
    def process_messages(self, swarm_time: int) -> None:
        """
        Process incoming messages from the message queue.
        
        Args:
            swarm_time: Current time step in the simulation
        """
        # Sort by priority
        self.message_queue.sort(key=lambda x: x["priority"].value, reverse=True)
        
        # Process important messages first (up to a limit)
        messages_processed = 0
        max_messages_per_step = 5
        
        while self.message_queue and messages_processed < max_messages_per_step:
            message = self.message_queue.pop(0)
            
            # Skip if too old (except for urgent messages)
            if (message["priority"] != MessagePriority.URGENT and 
                swarm_time - message["timestamp"] > 20):
                continue
                
            # Process message based on content
            self._process_message_content(message)
            
            # Update trust in sender based on message usefulness
            self._update_trust(message)
            
            messages_processed += 1
    
    def _process_message_content(self, message: Dict) -> None:
        """Process the content of a message and update knowledge."""
        content = message.get("content", {})
        
        # Process resource information
        if "resources" in content:
            for resource_pos in content["resources"]:
                # Add to memory if not already known
                if resource_pos not in self.memory:
                    self.memory[resource_pos] = {
                        "value": EnvironmentState.RESOURCE.value,
                        "timestamp": message["timestamp"],
                        "source": message["sender_id"]
                    }
        
        # Process threat information
        if "threats" in content:
            for threat_pos in content["threats"]:
                # Always update threat information as it's critical
                self.memory[threat_pos] = {
                    "value": EnvironmentState.THREAT.value,
                    "timestamp": message["timestamp"],
                    "source": message["sender_id"]
                }
                
        # Process memory sharing
        if "memory" in content:
            for pos, mem_data in content["memory"].items():
                # Only update if it's newer information than what we have
                existing = self.memory.get(pos)
                if not existing or mem_data["timestamp"] > existing["timestamp"]:
                    self.memory[pos] = mem_data.copy()
                    self.memory[pos]["source"] = message["sender_id"]
    
    def _update_trust(self, message: Dict) -> None:
        """Update trust level for the message sender based on usefulness."""
        sender_id = message["sender_id"]
        content = message.get("content", {})
        
        # Calculate message value based on information freshness and relevance
        message_value = 0.0
        
        # Resource information is valuable
        if "resources" in content:
            message_value += 0.05 * len(content["resources"])
            
        # Threat information is highly valuable
        if "threats" in content:
            message_value += 0.1 * len(content["threats"])
            
        # Memory sharing value depends on how much new info
        if "memory" in content:
            new_info_count = sum(1 for pos, data in content["memory"].items() 
                                if pos not in self.memory)
            message_value += 0.02 * new_info_count
            
        # Adjust trust based on message value
        current_trust = self.trust_map[sender_id]
        if message_value > 0:
            # Increase trust (with diminishing returns at high trust)
            trust_increase = message_value * (1.0 - current_trust/2)
            self.trust_map[sender_id] = min(1.0, current_trust + trust_increase)
        else:
            # Small trust decrease for useless messages
            self.trust_map[sender_id] = max(0.0, current_trust - 0.02)
    
    def receive_message(self, message: Dict) -> None:
        """Add an incoming message to the queue."""
        self.message_queue.append(message)
    
    def update_learning(self) -> None:
        """
        Update agent's learning parameters based on recent experiences.
        Uses a form of reinforcement learning to adapt behavior.
        """
        if not self.action_history or not self.reward_history:
            return
            
        # Calculate recent average reward
        recent_rewards = self.reward_history[-min(50, len(self.reward_history)):]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        # Count action frequencies
        action_counts = {}
        for action, _, _ in self.action_history[-min(50, len(self.action_history)):]:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate average rewards per action
        action_rewards = {}
        for i, (action, _, _) in enumerate(self.action_history[-min(50, len(self.action_history)):]):
            if i < len(self.reward_history):
                reward = self.reward_history[i]
                action_rewards[action] = action_rewards.get(action, []) + [reward]
        
        action_avg_rewards = {
            action: sum(rewards) / len(rewards) 
            for action, rewards in action_rewards.items()
        }
        
        # Update strategy based on rewards
        for action, avg_action_reward in action_avg_rewards.items():
            if avg_action_reward > avg_reward:
                # This action performs better than average
                if action == "explore":
                    self.strategy["exploration_rate"] = min(
                        0.9, self.strategy["exploration_rate"] + self.learning_rate
                    )
                elif action == "avoid":
                    self.strategy["risk_tolerance"] = max(
                        0.1, self.strategy["risk_tolerance"] - self.learning_rate
                    )
                elif action == "cooperate":
                    self.strategy["cooperation_bias"] = min(
                        0.9, self.strategy["cooperation_bias"] + self.learning_rate
                    )
            else:
                # This action performs worse than average
                if action == "explore":
                    self.strategy["exploration_rate"] = max(
                        0.1, self.strategy["exploration_rate"] - self.learning_rate
                    )
                elif action == "avoid":
                    self.strategy["risk_tolerance"] = min(
                        0.9, self.strategy["risk_tolerance"] + self.learning_rate
                    )
                elif action == "cooperate":
                    self.strategy["cooperation_bias"] = max(
                        0.1, self.strategy["cooperation_bias"] - self.learning_rate
                    )
        
        # Update skill levels based on action performance
        for action, avg_action_reward in action_avg_rewards.items():
            if action == "gather" and avg_action_reward > 0:
                self.skill_levels["resource_gathering"] = min(
                    1.0, self.skill_levels["resource_gathering"] + self.learning_rate * avg_action_reward
                )
            elif action == "avoid" and avg_action_reward > 0:
                self.skill_levels["threat_avoidance"] = min(
                    1.0, self.skill_levels["threat_avoidance"] + self.learning_rate * avg_action_reward
                )
            elif action == "explore" and avg_action_reward > 0:
                self.skill_levels["path_finding"] = min(
                    1.0, self.skill_levels["path_finding"] + self.learning_rate * avg_action_reward
                )
            elif action in ["cooperate", "communicate"] and avg_action_reward > 0:
                self.skill_levels["cooperation"] = min(
                    1.0, self.skill_levels["cooperation"] + self.learning_rate * avg_action_reward
                )
        
        # Increment experience
        self.experience += 1


class Environment:
    """
    Represents the environment where agents operate.
    Contains resources, obstacles, signals, and threats.
    """
    
    def __init__(self, width: int, height: int, resource_density: float = 0.05,
                obstacle_density: float = 0.1, threat_density: float = 0.03):
        """
        Initialize the environment with given dimensions and features.
        
        Args:
            width: Width of the environment grid
            height: Height of the environment grid
            resource_density: Density of resources (0-1)
            obstacle_density: Density of obstacles (0-1)
            threat_density: Density of threats (0-1)
        """
        self.width = width
        self.height = height
        self.grid = {}  # Sparse grid representation: (x,y) -> EnvironmentState
        self.signals = {}  # (x,y) -> {"type": str, "strength": float, "decay_time": int}
        self.resources = {}  # (x,y) -> resource_value
        
        # Initialize with random features
        self._initialize_environment(resource_density, obstacle_density, threat_density)
        
    def _initialize_environment(self, resource_density: float, obstacle_density: float,
                               threat_density: float) -> None:
        """Populate the environment with initial features."""
        # Create clusters of obstacles
        num_obstacle_clusters = int(self.width * self.height * obstacle_density / 10)
        for _ in range(num_obstacle_clusters):
            cluster_x = random.randint(0, self.width - 1)
            cluster_y = random.randint(0, self.height - 1)
            cluster_size = random.randint(3, 8)
            
            for dx in range(-cluster_size, cluster_size + 1):
                for dy in range(-cluster_size, cluster_size + 1):
                    x, y = cluster_x + dx, cluster_y + dy
                    if (0 <= x < self.width and 0 <= y < self.height and
                        random.random() < 0.3):  # Sparse clusters
                        self.grid[(x, y)] = EnvironmentState.OBSTACLE
        
        # Create resource patches
        num_resource_patches = int(self.width * self.height * resource_density / 5)
        for _ in range(num_resource_patches):
            patch_x = random.randint(0, self.width - 1)
            patch_y = random.randint(0, self.height - 1)
            
            # Don't place on obstacles
            if (patch_x, patch_y) in self.grid:
                continue
                
            patch_size = random.randint(2, 5)
            
            for dx in range(-patch_size, patch_size + 1):
                for dy in range(-patch_size, patch_size + 1):
                    x, y = patch_x + dx, patch_y + dy
                    if (0 <= x < self.width and 0 <= y < self.height and
                        (x, y) not in self.grid and random.random() < 0.4):
                        self.grid[(x, y)] = EnvironmentState.RESOURCE
                        self.resources[(x, y)] = random.uniform(5.0, 15.0)
        
        # Add threats
        num_threats = int(self.width * self.height * threat_density)
        threat_positions = set()
        
        for _ in range(num_threats):
            # Try to place threats away from each other
            for attempt in range(10):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                
                # Don't place on obstacles or resources
                if (x, y) in self.grid:
                    continue
                    
                # Check if far enough from other threats
                too_close = False
                for tx, ty in threat_positions:
                    if math.sqrt((x - tx)**2 + (y - ty)**2) < 15:
                        too_close = True
                        break
                
                if not too_close:
                    self.grid[(x, y)] = EnvironmentState.THREAT
                    threat_positions.add((x, y))
                    break
        
        # Add a few goals
        num_goals = max(1, int(self.width * self.height * 0.001))
        for _ in range(num_goals):
            for attempt in range(20):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                
                # Don't place on other features
                if (x, y) in self.grid:
                    continue
                    
                self.grid[(x, y)] = EnvironmentState.GOAL
                break
    
    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Check if a position is within environment bounds."""
        x, y = position
        
        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
            
        # Check if not obstacle
        grid_x, grid_y = int(x), int(y)
        return self.get_cell_state((grid_x, grid_y)) != EnvironmentState.OBSTACLE
    
    def get_cell_state(self, position: Tuple[int, int]) -> EnvironmentState:
        """Get the state of a cell at given position."""
        return self.grid.get(position, EnvironmentState.EMPTY)
    
    def collect_resource(self, position: Tuple[int, int]) -> float:
        """
        Collect resource at given position.
        
        Args:
            position: (x, y) coordinates to collect from
            
        Returns:
            Value of collected resource, or 0 if none
        """
        # Round to nearest grid cell
        grid_pos = (int(position[0]), int(position[1]))
        
        if grid_pos in self.resources and self.grid.get(grid_pos) == EnvironmentState.RESOURCE:
            resource_value = self.resources.pop(grid_pos)
            self.grid.pop(grid_pos)
            return resource_value
        return 0
    
    def add_signal(self, position: Tuple[float, float], signal_type: str, 
                  strength: float, decay_time: int) -> None:
        """
        Add a pheromone-like signal to the environment.
        
        Args:
            position: (x, y) coordinates for the signal
            signal_type: Type of signal (e.g., "resource", "threat", "exploration")
            strength: Initial strength of the signal (0-1)
            decay_time: Time steps until signal disappears
        """
        # Round to nearest grid cell
        grid_pos = (int(position[0]), int(position[1]))
        
        # Update existing signal or add new one
        if grid_pos in self.signals:
            # Take the stronger signal
            if strength > self.signals[grid_pos]["strength"]:
                self.signals[grid_pos] = {
                    "type": signal_type,
                    "strength": strength,
                    "decay_time": decay_time
                }
        else:
            self.signals[grid_pos] = {
                "type": signal_type,
                "strength": strength,
                "decay_time": decay_time
            }
    
    def get_signal_strength(self, position: Tuple[int, int]) -> float:
        """Get signal strength at given position."""
        return self.signals.get(position, {}).get("strength", 0.0)
    
    def get_signal_type(self, position: Tuple[int, int]) -> Optional[str]:
        """Get signal type at given position."""
        return self.signals.get(position, {}).get("type", None)
    
    def update_signals(self) -> None:
        """Update all signals, reducing strength and removing expired ones."""
        expired_signals = []
        
        for pos, signal in self.signals.items():
            # Reduce decay time
            signal["decay_time"] -= 1
            
            # Reduce strength
            signal["strength"] *= 0.95
            
            # Mark for removal if expired or too weak
            if signal["decay_time"] <= 0 or signal["strength"] < 0.1:
                expired_signals.append(pos)
                
        # Remove expired signals
        for pos in expired_signals:
            self.signals.pop(pos)
    
    def diffuse_signals(self) -> None:
        """Diffuse signals to neighboring cells, simulating pheromone spreading."""
        # Create a copy of current signals to avoid modifying while iterating
        current_signals = dict(self.signals)
        
        for pos, signal in current_signals.items():
            # Skip weak signals
            if signal["strength"] < 0.2:
                continue
                
            # Diffuse to neighbors
            diffusion_strength = signal["strength"] * 0.2
            neighbors = self._get_neighbors(pos)
            
            for neighbor in neighbors:
                # Don't diffuse through obstacles
                if self.get_cell_state(neighbor) == EnvironmentState.OBSTACLE:
                    continue
                    
                # Add diffused signal
                self.add_signal(
                    neighbor,
                    signal["type"],
                    diffusion_strength,
                    signal["decay_time"]
                )
    
    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        x, y = position
        neighbors = []
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
                
        return neighbors
    
    def add_resource(self, position: Tuple[int, int], value: float) -> bool:
        """
        Add a new resource to the environment.
        
        Args:
            position: (x, y) coordinates for the resource
            value: Value of the resource
            
        Returns:
            True if successfully added, False otherwise
        """
        # Check if position is valid
        if not (0 <= position[0] < self.width and 0 <= position[1] < self.height):
            return False
            
        # Check if position is empty
        if position in self.grid:
            return False
            
        # Add resource
        self.grid[position] = EnvironmentState.RESOURCE
        self.resources[position] = value
        return True
    
    def regenerate_resources(self, probability: float = 0.01, 
                            min_val: float = 3.0, max_val: float = 10.0) -> None:
        """
        Randomly regenerate some resources.
        
        Args:
            probability: Chance of generating a new resource per cell
            min_val: Minimum resource value
            max_val: Maximum resource value
        """
        # Count current resources
        current_resource_count = sum(1 for state in self.grid.values() 
                                   if state == EnvironmentState.RESOURCE)
        
        # Don't regenerate if already above threshold
        max_resources = int(self.width * self.height * 0.1)  # 10% maximum density
        if current_resource_count >= max_resources:
            return
            
        # Try to add a few new resources
        attempts = min(20, max_resources - current_resource_count)
        
        for _ in range(attempts):
            if random.random() < probability:
                # Find valid position
                for attempt in range(10):
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)
                    
                    if (x, y) not in self.grid:
                        value = random.uniform(min_val, max_val)
                        self.add_resource((x, y), value)
                        break


class Task:
    """
    Represents a task that can be performed by one or more agents.
    Tasks may require multiple agents to complete.
    """
    
    def __init__(self, task_id: int, position: Tuple[float, float], 
                difficulty: float, required_agents: int = 1):
        """
        Initialize a task.
        
        Args:
            task_id: Unique identifier for the task
            position: (x, y) coordinates in the environment
            difficulty: How difficult the task is (affects completion time/effort)
            required_agents: Minimum number of agents needed
        """
        self.id = task_id
        self.position = position
        self.difficulty = difficulty
        self.required_agents = required_agents
        self.progress = 0.0  # 0 to 1.0 for completion progress
        self.state = TaskState.UNASSIGNED
        self.assigned_agents = set()
        self.contributors = {}  # agent_id -> contribution_amount
        self.creation_time = 0
        self.completion_time = None
        self.failure_time = None
        
    def assign_agent(self, agent_id: int) -> bool:
        """
        Assign an agent to this task.
        
        Args:
            agent_id: ID of the agent to assign
            
        Returns:
            True if assignment was successful
        """
        if len(self.assigned_agents) < self.required_agents:
            self.assigned_agents.add(agent_id)
            
            # Update task state
            if len(self.assigned_agents) >= self.required_agents:
                self.state = TaskState.IN_PROGRESS
            else:
                self.state = TaskState.ASSIGNED
                
            return True
        return False
    
    def unassign_agent(self, agent_id: int) -> None:
        """Remove an agent from this task."""
        if agent_id in self.assigned_agents:
            self.assigned_agents.remove(agent_id)
            
            # Update task state if no longer enough agents
            if len(self.assigned_agents) < self.required_agents:
                self.state = TaskState.ASSIGNED if self.assigned_agents else TaskState.UNASSIGNED
    
    def add_contribution(self, agent_id: int, amount: float) -> float:
        """
        Add agent's contribution toward task completion.
        
        Args:
            agent_id: ID of contributing agent
            amount: Amount of contribution (typically based on action effectiveness)
            
        Returns:
            Actual contribution amount (may be scaled)
        """
        # Only count contributions from assigned agents
        if agent_id not in self.assigned_agents and self.required_agents > 1:
            return 0.0
            
        # Scale contribution based on difficulty
        actual_contribution = amount / self.difficulty
        
        # Store agent's contribution
        self.contributors[agent_id] = self.contributors.get(agent_id, 0.0) + actual_contribution
        
        # Update progress
        old_progress = self.progress
        self.progress = min(1.0, self.progress + actual_contribution)
        
        # Check if completed
        if self.progress >= 1.0 and self.state != TaskState.COMPLETED:
            self.state = TaskState.COMPLETED
            
        return self.progress - old_progress
    
    def get_completion_percentage(self) -> float:
        """Get the task completion percentage."""
        return self.progress * 100.0
    
    def is_cooperative(self) -> bool:
        """Determine if this is a cooperative task requiring multiple agents."""
        return self.required_agents > 1
    
    def mark_failed(self, time: int) -> None:
        """Mark task as failed at given time."""
        self.state = TaskState.FAILED
        self.failure_time = time
    
    def mark_completed(self, time: int) -> None:
        """Mark task as completed at given time."""
        self.state = TaskState.COMPLETED
        self.completion_time = time
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "position": self.position,
            "difficulty": self.difficulty,
            "required_agents": self.required_agents,
            "progress": self.progress,
            "state": self.state.name,
            "assigned_agents": list(self.assigned_agents),
            "creation_time": self.creation_time,
            "completion_time": self.completion_time,
            "failure_time": self.failure_time
        }


class Swarm:
    """
    Represents a swarm of agents working together.
    Coordinates communication and collective behavior.
    """
    
    def __init__(self, environment: Environment, num_agents: int = 10):
        """
        Initialize a swarm with given number of agents.
        
        Args:
            environment: Environment object where the swarm operates
            num_agents: Number of agents in the swarm
        """
        self.environment = environment
        self.agents = []
        self.time_step = 0
        self.message_buffer = []
        self.tasks = {}  # task_id -> Task
        self.task_counter = 0
        self.completed_tasks = []
        self.failed_tasks = []
        self.metrics = {
            "resources_collected": 0,
            "communication_events": 0,
            "agent_interactions": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "exploration_coverage": 0.0
        }
        
        # Create agents with different specializations
        self._initialize_agents(num_agents)
    
    def _initialize_agents(self, num_agents: int) -> None:
        """Create initial set of agents with diverse properties."""
        for i in range(num_agents):
            # Random starting position
            while True:
                x = random.uniform(0, self.environment.width)
                y = random.uniform(0, self.environment.height)
                
                # Ensure valid starting position
                if self.environment.is_valid_position((x, y)):
                    break
            
            # Assign different specializations
            specialization = random.uniform(0.3, 0.8)
            learning_rate = random.uniform(0.01, 0.05)
            
            agent = Agent(i, (x, y), specialization, learning_rate)
            
            # Randomize initial strategy slightly
            agent.strategy["exploration_rate"] = max(0.1, min(0.9, 
                                                           agent.strategy["exploration_rate"] + 
                                                           random.uniform(-0.2, 0.2)))
            agent.strategy["risk_tolerance"] = max(0.1, min(0.9, 
                                                          agent.strategy["risk_tolerance"] + 
                                                          random.uniform(-0.2, 0.2)))
            agent.strategy["cooperation_bias"] = max(0.1, min(0.9, 
                                                            agent.strategy["cooperation_bias"] + 
                                                            random.uniform(-0.2, 0.2)))
            
            self.agents.append(agent)
    
    def step(self) -> Dict:
        """
        Advance the simulation by one time step.
        
        Returns:
            Dictionary of metrics for this step
        """
        self.time_step += 1
        step_metrics = {
            "resources_collected": 0,
            "messages_sent": 0,
            "tasks_assigned": 0,
            "tasks_completed": 0,
            "tasks_created": 0
        }
        
        # Environment updates
        self.environment.update_signals()
        if self.time_step % 5 == 0:  # Diffuse less frequently for efficiency
            self.environment.diffuse_signals()
        
        if self.time_step % 50 == 0:  # Regenerate resources periodically
            self.environment.regenerate_resources()
        
        # Process message queue
        self._deliver_messages()
        
        # Process tasks
        task_updates = self._update_tasks()
        step_metrics.update(task_updates)
        
        # Create new tasks periodically
        if self.time_step % 20 == 0 or random.random() < 0.05:
            num_new_tasks = random.randint(1, 3)
            for _ in range(num_new_tasks):
                self._create_random_task()
                step_metrics["tasks_created"] += 1
        
        # Agent decision and action phase
        for agent in self.agents:
            # Process messages
            agent.process_messages(self.time_step)
            
            # Sense environment
            other_agents = [a for a in self.agents if a.id != agent.id]
            sensed_data = agent.sense_environment(self.environment, other_agents)
            
            # Decide action
            action_name, action_params = agent.decide_action(sensed_data, self, self.time_step)
            
            # Execute action
            results = agent.execute_action(action_name, action_params, self.environment, self)
            
            # Update metrics
            if "collected" in results:
                step_metrics["resources_collected"] += results["collected"]
                self.metrics["resources_collected"] += results["collected"]
                
            # Occasional learning update
            if self.time_step % 10 == 0:
                agent.update_learning()
                
        # Update coverage metrics
        self._update_metrics()
        
        return step_metrics
    
    def send_message(self, message: Dict) -> None:
        """
        Queue a message for delivery.
        
        Args:
            message: Message dictionary with sender, receiver, content, etc.
        """
        self.message_buffer.append(message)
        self.metrics["communication_events"] += 1
    
    def _deliver_messages(self) -> None:
        """Process message buffer and deliver to recipients."""
        # Group messages by recipient
        recipient_messages = defaultdict(list)
        
        for message in self.message_buffer:
            recipient_id = message["receiver_id"]
            recipient_messages[recipient_id].append(message)
        
        # Deliver to each agent
        for agent_id, messages in recipient_messages.items():
            # Find the agent
            recipient = next((a for a in self.agents if a.id == agent_id), None)
            if recipient:
                for message in messages:
                    recipient.receive_message(message)
        
        # Clear the buffer
        self.message_buffer = []
    
    def _create_random_task(self) -> Task:
        """Create a new random task in the environment."""
        # Generate random position
        while True:
            x = random.uniform(0, self.environment.width)
            y = random.uniform(0, self.environment.height)
            
            # Ensure valid position
            if self.environment.is_valid_position((x, y)):
                break
        
        # Determine task properties
        difficulty = random.uniform(1.0, 5.0)
        
        # Higher chance of cooperative tasks as time progresses
        coop_probability = min(0.5, 0.1 + (self.time_step / 1000))
        required_agents = 1
        if random.random() < coop_probability:
            required_agents = random.randint(2, min(4, len(self.agents) // 2))
        
        # Create the task
        task_id = self.task_counter
        self.task_counter += 1
        
        task = Task(task_id, (x, y), difficulty, required_agents)
        task.creation_time = self.time_step
        
        self.tasks[task_id] = task
        return task
    
    def _update_tasks(self) -> Dict:
        """
        Update all tasks, checking for completion or timeout.
        
        Returns:
            Dictionary with task metrics
        """
        metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_assigned": 0
        }
        
        # Check task states
        completed_ids = []
        failed_ids = []
        
        for task_id, task in self.tasks.items():
            # Check for completion
            if task.progress >= 1.0 and task.state != TaskState.COMPLETED:
                task.mark_completed(self.time_step)
                completed_ids.append(task_id)
                metrics["tasks_completed"] += 1
                self.metrics["tasks_completed"] += 1
                
            # Check for timeout or abandonment
            elif (self.time_step - task.creation_time > 200 or  # Timeout
                  (task.is_cooperative() and len(task.assigned_agents) == 0 and 
                   self.time_step - task.creation_time > 50)):  # Abandoned
                
                task.mark_failed(self.time_step)
                failed_ids.append(task_id)
                metrics["tasks_failed"] += 1
                self.metrics["tasks_failed"] += 1
                
            # Auto-assign nearby unassigned agents to tasks
            if task.state in [TaskState.UNASSIGNED, TaskState.ASSIGNED]:
                # Only assign if more agents needed
                if len(task.assigned_agents) < task.required_agents:
                    # Find nearby unassigned agents
                    for agent in self.agents:
                        if agent.id not in task.assigned_agents and agent.current_task is None:
                            # Check if agent is close to task
                            distance = math.sqrt(
                                (agent.position[0] - task.position[0])**2 +
                                (agent.position[1] - task.position[1])**2
                            )
                            
                            # Only assign if reasonably close
                            if distance < 15:
                                if task.assign_agent(agent.id):
                                    agent.current_task = task_id
                                    metrics["tasks_assigned"] += 1
                                    
                                    # Only assign up to required number
                                    if len(task.assigned_agents) >= task.required_agents:
                                        break
        
        # Move completed tasks to archive
        for task_id in completed_ids:
            self.completed_tasks.append(self.tasks[task_id])
            del self.tasks[task_id]
            
            # Update agents assigned to this task
            for agent in self.agents:
                if agent.current_task == task_id:
                    agent.current_task = None
        
        # Move failed tasks to archive
        for task_id in failed_ids:
            self.failed_tasks.append(self.tasks[task_id])
            del self.tasks[task_id]
            
            # Update agents assigned to this task
            for agent in self.agents:
                if agent.current_task == task_id:
                    agent.current_task = None
        
        return metrics
    
    def get_cooperative_tasks(self) -> List[Dict]:
        """Get all current cooperative tasks that need multiple agents."""
        cooperative_tasks = []
        
        for task_id, task in self.tasks.items():
            if task.is_cooperative() and task.state != TaskState.COMPLETED:
                # Include position and ID for agent decision making
                cooperative_tasks.append({
                    "id": task_id,
                    "position": task.position,
                    "required_agents": task.required_agents,
                    "current_agents": len(task.assigned_agents),
                    "progress": task.progress
                })
                
        return cooperative_tasks
    
    def contribute_to_task(self, task_id: int, agent_id: int, contribution_amount: float) -> float:
        """
        Add agent contribution to a task.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the contributing agent
            contribution_amount: Amount of contribution
            
        Returns:
            Actual contribution applied
        """
        # Check if task exists
        if task_id not in self.tasks:
            return 0.0
            
        task = self.tasks[task_id]
        
        # Auto-assign agent if not yet assigned
        if agent_id not in task.assigned_agents:
            if len(task.assigned_agents) < task.required_agents:
                task.assign_agent(agent_id)
                
                # Update agent's current task
                for agent in self.agents:
                    if agent.id == agent_id:
                        agent.current_task = task_id
                        break
        
        # Add contribution
        actual_contribution = task.add_contribution(agent_id, contribution_amount)
        return actual_contribution
    
    def _update_metrics(self) -> None:
        """Update various swarm metrics."""
        # Calculate exploration coverage
        total_cells = self.environment.width * self.environment.height
        explored_cells = set()
        
        for agent in self.agents:
            explored_cells.update(agent.memory.keys())
            
        coverage = len(explored_cells) / total_cells
        self.metrics["exploration_coverage"] = coverage
        
        # Count agent interactions
        interaction_count = 0
        for agent in self.agents:
            interaction_count += len(agent.connections)
            
        self.metrics["agent_interactions"] = interaction_count
    
    def get_metrics(self) -> Dict:
        """Get current metrics dictionary."""
        return dict(self.metrics)
    
    def get_task_summary(self) -> Dict:
        """Get summary of current task states."""
        summary = {
            "active_tasks": len(self.tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "cooperative_tasks": sum(1 for t in self.tasks.values() if t.is_cooperative()),
            "unassigned_tasks": sum(1 for t in self.tasks.values() 
                                  if t.state == TaskState.UNASSIGNED),
            "in_progress_tasks": sum(1 for t in self.tasks.values() 
                                   if t.state == TaskState.IN_PROGRESS)
        }
        return summary
    
    def get_agent_status(self) -> List[Dict]:
        """Get status summary for all agents."""
        agent_status = []
        
        for agent in self.agents:
            status = {
                "id": agent.id,
                "position": agent.position,
                "energy": agent.energy,
                "task": agent.current_task,
                "memory_size": len(agent.memory),
                "connections": len(agent.connections),
                "specialization": agent.specialization,
                "strategy": dict(agent.strategy),
                "skill_levels": dict(agent.skill_levels)
            }
            agent_status.append(status)
            
        return agent_status


class AdaptiveSwarmSimulation:
    """
    Main simulation class that runs the Adaptive Swarm Intelligence System.
    """
    
    def __init__(self, width: int = 100, height: int = 100, num_agents: int = 20):
        """
        Initialize the simulation.
        
        Args:
            width: Width of the environment
            height: Height of the environment
            num_agents: Number of agents in the swarm
        """
        self.environment = Environment(width, height)
        self.swarm = Swarm(self.environment, num_agents)
        self.current_step = 0
        self.history = {
            "metrics": [],
            "task_summaries": [],
            "significant_events": []
        }
    
    def run(self, num_steps: int) -> Dict:
        """
        Run the simulation for a specified number of steps.
        
        Args:
            num_steps: Number of steps to run
            
        Returns:
            Dictionary of results
        """
        results = []
        
        for _ in range(num_steps):
            step_result = self.step()
            results.append(step_result)
            
        return {
            "num_steps": num_steps,
            "final_metrics": self.swarm.get_metrics(),
            "final_task_summary": self.swarm.get_task_summary(),
            "step_results": results
        }
    
    def step(self) -> Dict:
        """
        Execute a single simulation step.
        
        Returns:
            Dictionary of step results
        """
        self.current_step += 1
        
        # Execute swarm step
        step_metrics = self.swarm.step()
        
        # Record history
        self.history["metrics"].append(self.swarm.get_metrics())
        self.history["task_summaries"].append(self.swarm.get_task_summary())
        
        # Check for significant events
        self._check_for_events(step_metrics)
        
        return {
            "step": self.current_step,
            "metrics": step_metrics,
            "swarm_metrics": self.swarm.get_metrics(),
            "task_summary": self.swarm.get_task_summary()
        }
    
    def _check_for_events(self, step_metrics: Dict) -> None:
        """Check for and record significant events."""
        events = []
        
        # Resource collection milestone
        if step_metrics.get("resources_collected", 0) > 10:
            events.append({
                "type": "resource_milestone",
                "step": self.current_step,
                "details": f"Collected {step_metrics['resources_collected']} resources in one step"
            })
            
        # Task completion
        if step_metrics.get("tasks_completed", 0) > 0:
            events.append({
                "type": "task_completion",
                "step": self.current_step,
                "details": f"Completed {step_metrics['tasks_completed']} tasks"
            })
            
        # Record events
        for event in events:
            self.history["significant_events"].append(event)
    
    def get_stats(self) -> Dict:
        """Get overall statistics from the simulation."""
        # Calculate various statistics from history
        metrics_history = self.history["metrics"]
        task_history = self.history["task_summaries"]
        
        if not metrics_history:
            return {"error": "No simulation data available"}
            
        stats = {
            "total_steps": self.current_step,
            "resources_collected": metrics_history[-1]["resources_collected"],
            "tasks_completed": metrics_history[-1]["tasks_completed"],
            "tasks_failed": metrics_history[-1]["tasks_failed"],
            "exploration_coverage": metrics_history[-1]["exploration_coverage"],
            "communication_events": metrics_history[-1]["communication_events"],
            "agent_interactions": metrics_history[-1]["agent_interactions"],
            "efficiency": self._calculate_efficiency()
        }
        
        return stats
    
    def _calculate_efficiency(self) -> float:
        """Calculate overall swarm efficiency metric."""
        if not self.history["metrics"]:
            return 0.0
            
        last_metrics = self.history["metrics"][-1]
        
        # Efficiency based on resources gathered per step and task completion rate
        resources_per_step = last_metrics["resources_collected"] / max(1, self.current_step)
        
        task_completion_rate = 0.0
        total_tasks = last_metrics["tasks_completed"] + last_metrics["tasks_failed"]
        if total_tasks > 0:
            task_completion_rate = last_metrics["tasks_completed"] / total_tasks
            
        # Weighted combination
        efficiency = (0.6 * resources_per_step * 0.1) + (0.4 * task_completion_rate)
        
        return efficiency
    
    def get_agent_positions(self) -> List[Tuple[float, float]]:
        """Get current positions of all agents."""
        return [agent.position for agent in self.swarm.agents]
    
    def get_environment_state(self) -> Dict:
        """Get a representation of the current environment state."""
        env = self.environment
        
        state = {
            "dimensions": (env.width, env.height),
            "obstacles": [pos for pos, state in env.grid.items() 
                         if state == EnvironmentState.OBSTACLE],
            "resources": [pos for pos, state in env.grid.items() 
                         if state == EnvironmentState.RESOURCE],
            "threats": [pos for pos, state in env.grid.items() 
                       if state == EnvironmentState.THREAT],
            "goals": [pos for pos, state in env.grid.items() 
                     if state == EnvironmentState.GOAL],
            "signals": [
                {"position": pos, "type": data["type"], "strength": data["strength"]}
                for pos, data in env.signals.items()
            ],
            "agent_positions": self.get_agent_positions()
        }
        
        return state


# Example usage
if __name__ == "__main__":
    # Create and run a simulation
    sim = AdaptiveSwarmSimulation(width=100, height=100, num_agents=30)
    results = sim.run(num_steps=500)
    
    # Print final statistics
    print("\nSimulation Complete!")
    print("-------------------")
    print(f"Steps: {results['num_steps']}")
    print(f"Resources Collected: {results['final_metrics']['resources_collected']}")
    print(f"Tasks Completed: {results['final_metrics']['tasks_completed']}")
    print(f"Tasks Failed: {results['final_metrics']['tasks_failed']}")
    print(f"Exploration Coverage: {results['final_metrics']['exploration_coverage']*100:.2f}%")
    print(f"Communication Events: {results['final_metrics']['communication_events']}")
    
    # Print task summary
    print("\nTask Summary:")
    task_summary = results['final_task_summary']
    for key, value in task_summary.items():
        print(f"  {key}: {value}")


