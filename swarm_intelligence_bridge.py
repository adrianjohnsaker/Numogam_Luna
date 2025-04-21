import json
import numpy as np
import random
import math
from enum import Enum
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Import core swarm intelligence classes
from swarm_intelligence import (
    Agent, Environment, Task, Swarm, TaskState, 
    MessagePriority, MessageType, EnvironmentState,
    AdaptiveSwarmSimulation
)

class SwarmIntelligenceBridge:
    """
    Main bridge class for communication between Kotlin and Python
    components of the Swarm Intelligence system.
    """
    
    def __init__(self, width=100, height=100, num_agents=20):
        """Initialize the swarm intelligence system and bridge"""
        # Initialize the main simulation
        self.simulation = AdaptiveSwarmSimulation(
            width=width, 
            height=height,
            num_agents=num_agents
        )
        
        # Configuration settings
        self.config = {
            "step_delay": 100,  # Milliseconds between steps
            "auto_run": False,
            "visualization_enabled": True,
            "log_level": "INFO",
            "export_metrics": True
        }
        
        # Cached data for better performance
        self.cached_state = None
        self.cache_time = 0
        self.cache_duration = 500  # Milliseconds
        
        # Metadata for tracking
        self.last_command = None
        self.error_log = []
    
    def initialize_simulation(self, width=None, height=None, num_agents=None, seed=None):
        """
        Initialize or reinitialize the simulation with given parameters
        
        Args:
            width: Width of environment grid
            height: Height of environment grid
            num_agents: Number of agents in swarm
            seed: Random seed for reproducibility
            
        Returns:
            JSON string with initialization status
        """
        try:
            # Set random seed if provided
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            
            # Use provided parameters or defaults
            width = width or self.simulation.environment.width
            height = height or self.simulation.environment.height
            num_agents = num_agents or len(self.simulation.swarm.agents)
            
            # Create new simulation
            self.simulation = AdaptiveSwarmSimulation(
                width=width,
                height=height,
                num_agents=num_agents
            )
            
            # Clear cache
            self.cached_state = None
            
            return json.dumps({
                "status": "success",
                "environment": {
                    "width": width,
                    "height": height
                },
                "agents": num_agents,
                "message": "Simulation initialized successfully"
            })
            
        except Exception as e:
            self.error_log.append(str(e))
            return json.dumps({
                "status": "error",
                "message": f"Failed to initialize simulation: {str(e)}"
            })
    
    def step(self, num_steps=1):
        """
        Execute simulation step(s)
        
        Args:
            num_steps: Number of steps to execute
            
        Returns:
            JSON string with step results
        """
        try:
            results = []
            
            for _ in range(num_steps):
                step_result = self.simulation.step()
                results.append(step_result)
            
            # Clear cache since state changed
            self.cached_state = None
            
            # For performance, only return minimal data when multiple steps
            if num_steps > 1:
                return json.dumps({
                    "status": "success",
                    "steps_executed": num_steps,
                    "final_metrics": self.simulation.swarm.get_metrics(),
                    "message": f"Executed {num_steps} simulation steps"
                })
            else:
                # For single step, return full data
                return json.dumps({
                    "status": "success",
                    "steps_executed": 1,
                    "step_data": results[0],
                    "metrics": self.simulation.swarm.get_metrics(),
                    "message": "Executed 1 simulation step"
                })
                
        except Exception as e:
            self.error_log.append(str(e))
            return json.dumps({
                "status": "error",
                "message": f"Error during simulation step: {str(e)}"
            })
    
    def get_environment_state(self, include_agents=True, include_signals=True):
        """
        Get current environment state
        
        Args:
            include_agents: Whether to include agent positions
            include_signals: Whether to include pheromone signals
            
        Returns:
            JSON string with environment state
        """
        # Check if we can use cached state
        current_time = self._get_current_time()
        if (self.cached_state is not None and 
            current_time - self.cache_time < self.cache_duration):
            return self.cached_state
        
        try:
            # Get raw state data
            state = self.simulation.get_environment_state()
            
            # Create response based on requested components
            response = {
                "status": "success",
                "environment": {
                    "dimensions": state["dimensions"],
                    "obstacles": self._format_positions(state["obstacles"]),
                    "resources": self._format_positions(state["resources"]),
                    "threats": self._format_positions(state["threats"]),
                    "goals": self._format_positions(state["goals"]),
                }
            }
            
            if include_agents:
                response["agents"] = self._format_positions(state["agent_positions"])
            
            if include_signals:
                response["signals"] = [
                    {
                        "position": [s["position"][0], s["position"][1]],
                        "type": s["type"],
                        "strength": s["strength"]
                    } for s in state["signals"]
                ]
            
            # Cache result
            json_response = json.dumps(response)
            self.cached_state = json_response
            self.cache_time = current_time
            
            return json_response
            
        except Exception as e:
            self.error_log.append(str(e))
            return json.dumps({
                "status": "error",
                "message": f"Failed to get environment state: {str(e)}"
            })
    
    def get_agent_details(self, agent_id=None):
        """
        Get detailed information about agents
        
        Args:
            agent_id: Specific agent ID or None for all agents
            
        Returns:
            JSON string with agent details
        """
        try:
            agent_status = self.simulation.swarm.get_agent_status()
            
            if agent_id is not None:
                # Return details for specific agent
                for agent in agent_status:
                    if agent["id"] == agent_id:
                        return json.dumps({
                            "status": "success",
                            "agent": agent
                        })
                
                return json.dumps({
                    "status": "error",
                    "message": f"Agent with ID {agent_id} not found"
                })
            else:
                # Return summary for all agents
                return json.dumps({
                    "status": "success",
                    "agents": agent_status
                })
                
        except Exception as e:
            self.error_log.append(str(e))
            return json.dumps({
                "status": "error",
                "message": f"Failed to get agent details: {str(e)}"
            })
    
    def get_tasks(self, include_completed=False, include_failed=False):
        """
        Get information about tasks
        
        Args:
            include_completed: Whether to include completed tasks
            include_failed: Whether to include failed tasks
            
        Returns:
            JSON string with task details
        """
        try:
            # Get active tasks
            active_tasks = {}
            for task_id, task in self.simulation.swarm.tasks.items():
                active_tasks[task_id] = task.to_dict()
            
            response = {
                "status": "success",
                "active_tasks": active_tasks,
                "task_summary": self.simulation.swarm.get_task_summary()
            }
            
            # Add completed tasks if requested
            if include_completed:
                completed_tasks = {}
                for task in self.simulation.swarm.completed_tasks:
                    completed_tasks[task.id] = task.to_dict()
                response["completed_tasks"] = completed_tasks
            
            # Add failed tasks if requested
            if include_failed:
                failed_tasks = {}
                for task in self.simulation.swarm.failed_tasks:
                    failed_tasks[task.id] = task.to_dict()
                response["failed_tasks"] = failed_tasks
            
            return json.dumps(response)
            
        except Exception as e:
            self.error_log.append(str(e))
            return json.dumps({
                "status": "error",
                "message": f"Failed to get tasks: {str(e)}"
            })
    
    def add_resource(self, x, y, value=10.0):
        """
        Add a new resource to the environment
        
        Args:
            x: X coordinate
            y: Y coordinate
            value: Resource value
            
        Returns:
            JSON string with operation result
        """
        try:
            result = self.simulation.environment.add_resource((int(x), int(y)), float(value))
            
            # Clear cache since environment changed
            self.cached_state = None
            
            if result:
                return json.dumps({
                    "status": "success",
                    "message": f"Resource added at position ({x}, {y})"
                })
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to add resource at position ({x}, {y})"
                })
                
        except Exception as e:
            self.error_log.append(str(e))
            return json.dumps({
                "status": "error",
                "message": f"Error adding resource: {str(e)}"
            })
    
    def add_task(self, x, y, difficulty=2.0, required_agents=1):
        """
        Add a new task to the environment
        
        Args:
            x: X coordinate
            y: Y coordinate
            difficulty: Task difficulty (1.0 to 5.0)
            required_agents: Number of agents required
            
        Returns:
            JSON string with operation result
        """
        try:
            task_id = self.simulation.swarm.task_counter
            self.simulation.swarm.task_counter += 1
            
            task = Task(
                task_id=task_id,
                position=(float(x), float(y)),
                difficulty=float(difficulty),
                required_agents=int(required_agents)
            )
            task.creation_time = self.simulation.swarm.time_step
            
            self.simulation.swarm.tasks[task_id] = task
            
            return json.dumps({
                "status": "success",
                "task_id": task_id,
                "message": f"Task created at position ({x}, {y})"
            })
                
        except Exception as e:
            self.error_log.append(str(e))
            return json.dumps({
                "status": "error",
                "message": f"Error adding task: {str(e)}"
            })
    
    def get_metrics(self):
        """
        Get current simulation metrics
        
        Returns:
            JSON string with metrics
        """
        try:
            metrics = self.simulation.swarm.get_metrics()
            stats = self.simulation.get_stats()
            
            return json.dumps({
                "status": "success",
                "metrics": metrics,
                "statistics": stats,
                "current_step": self.simulation.current_step
            })
                
        except Exception as e:
            self.error_log.append(str(e))
            return json.dumps({
                "status": "error",
                "message": f"Error retrieving metrics: {str(e)}"
            })
    
    def export_state(self):
        """
        Export full simulation state for saving
        
        Returns:
            Dictionary with serializable state
        """
        # Create a serializable representation of simulation state
        state = {
            "metadata": {
                "version": "1.0",
                "timestamp": self._get_current_time(),
                "steps": self.simulation.current_step
            },
            "environment": {
                "width": self.simulation.environment.width,
                "height": self.simulation.environment.height,
                "resource_positions": [(pos[0], pos[1], self.simulation.environment.resources[pos]) 
                                      for pos in self.simulation.environment.resources],
                "obstacle_positions": [(pos[0], pos[1]) for pos, state in 
                                      self.simulation.environment.grid.items() 
                                      if state == EnvironmentState.OBSTACLE],
                "threat_positions": [(pos[0], pos[1]) for pos, state in 
                                    self.simulation.environment.grid.items() 
                                    if state == EnvironmentState.THREAT]
            },
            "agents": [],
            "tasks": [],
            "config": self.config
        }
        
        # Add agents
        for agent in self.simulation.swarm.agents:
            state["agents"].append({
                "id": agent.id,
                "position": agent.position,
                "energy": agent.energy,
                "strategy": agent.strategy,
                "skill_levels": agent.skill_levels
            })
        
        # Add active tasks
        for task_id, task in self.simulation.swarm.tasks.items():
            state["tasks"].append(task.to_dict())
        
        return state
    
    def import_state(self, state_dict):
        """
        Import previously exported state
        
        Args:
            state_dict: Dictionary with state data
            
        Returns:
            Boolean success status
        """
        try:
            # Validate minimal required data
            if not isinstance(state_dict, dict):
                return False
                
            if "metadata" not in state_dict or "environment" not in state_dict:
                return False
            
            # Create new simulation with saved dimensions
            env_data = state_dict["environment"]
            width = env_data.get("width", 100)
            height = env_data.get("height", 100)
            
            # Initialize with empty agent list (we'll add them from saved state)
            self.simulation = AdaptiveSwarmSimulation(
                width=width,
                height=height,
                num_agents=0
            )
            
            # Add resources
            for res_data in env_data.get("resource_positions", []):
                self.simulation.environment.add_resource(
                    (int(res_data[0]), int(res_data[1])),
                    float(res_data[2])
                )
            
            # Add obstacles and threats
            for obs_pos in env_data.get("obstacle_positions", []):
                self.simulation.environment.grid[obs_pos] = EnvironmentState.OBSTACLE
                
            for threat_pos in env_data.get("threat_positions", []):
                self.simulation.environment.grid[threat_pos] = EnvironmentState.THREAT
            
            # Restore config
            if "config" in state_dict:
                self.config = state_dict["config"]
            
            # Clear cache
            self.cached_state = None
                
            return True
            
        except Exception as e:
            self.error_log.append(str(e))
            return False
    
    def update_config(self, config_updates):
        """
        Update configuration settings
        
        Args:
            config_updates: Dictionary with settings to update
            
        Returns:
            Boolean success status
        """
        try:
            for key, value in config_updates.items():
                if key in self.config:
                    self.config[key] = value
            return True
        except Exception:
            return False
    
    def _format_positions(self, positions):
        """Convert position tuples to lists for JSON serialization"""
        return [[pos[0], pos[1]] for pos in positions]
    
    def _get_current_time(self):
        """Get current time in milliseconds - platform independent"""
        try:
            # Try to use time module
            import time
            return int(time.time() * 1000)
        except:
            # Fallback to simulation step as proxy for time
            return self.simulation.current_step * 100
