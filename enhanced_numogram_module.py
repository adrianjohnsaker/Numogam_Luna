# numogram_code/models.py
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import json
import os
import numpy as np
from datetime import datetime

class ZoneData(BaseModel):
    name: str
    description: str
    energy_coefficient: float = 1.0
    evolutionary_potential: float = 0.5
    
class TransitionData(BaseModel):
    source: str
    target: str
    probability: float = 0.5
    energy_cost: float = 1.0
    evolutionary_impact: float = 0.0
    feedback_coefficient: float = 0.1

class NumogramMemory(BaseModel):
    zone_visits: Dict[str, int] = {}
    transition_frequency: Dict[str, Dict[str, int]] = {}
    feedback_history: List[Dict[str, Any]] = []
    bayesian_priors: Dict[str, Dict[str, float]] = {}
    
class TransitionRequest(BaseModel):
    current_zone: str
    input: str
    context: Optional[Dict[str, Any]] = None
    
class TransitionResponse(BaseModel):
    transition_result: List[str]
    probabilities: Dict[str, float]
    context: Dict[str, Any]
    feedback_required: bool = False

# numogram_code/numogram_algorithm.py
import json
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import os
import math

class NumogramCore:
    def __init__(self, zones_file="zones.json", transitions_file="transitions.json", memory_file="memory.json"):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load zones
        zones_path = os.path.join(self.data_dir, zones_file)
        with open(zones_path, 'r') as f:
            self.zones = json.load(f)["zones"]
            
        # Load or initialize transitions
        transitions_path = os.path.join(self.data_dir, transitions_file)
        if os.path.exists(transitions_path):
            with open(transitions_path, 'r') as f:
                self.transitions = json.load(f)
        else:
            self.transitions = self._initialize_transitions()
            self._save_transitions(transitions_file)
            
        # Load or initialize memory
        memory_path = os.path.join(self.data_dir, memory_file)
        if os.path.exists(memory_path):
            with open(memory_path, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = self._initialize_memory()
            self._save_memory(memory_file)
        
        # Initialize the Bayesian model if it doesn't exist
        if "bayesian_priors" not in self.memory:
            self.memory["bayesian_priors"] = self._initialize_bayesian_priors()
            self._save_memory(memory_file)
            
    def _initialize_transitions(self) -> Dict:
        """Initialize transition map based on numogram principles"""
        basic_transitions = {
            "1": ["2", "4"],
            "2": ["3", "6"],
            "3": ["1", "9"],
            "4": ["5", "8"],
            "5": ["7"],
            "6": ["5"],
            "7": ["8"],
            "8": ["9"],
            "9": ["6"]
        }
        
        transitions = {
            "transitions": [],
            "meta": {
                "created": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        # Create structured transition data
        for source, targets in basic_transitions.items():
            for target in targets:
                transitions["transitions"].append({
                    "source": source,
                    "target": target,
                    "probability": 0.5,
                    "energy_cost": 1.0,
                    "evolutionary_impact": 0.1,
                    "feedback_coefficient": 0.1
                })
        
        return transitions
    
    def _initialize_memory(self) -> Dict:
        """Initialize the memory structure"""
        return {
            "zone_visits": {str(i): 0 for i in range(1, 10)},
            "transition_frequency": {},
            "feedback_history": [],
            "learning_iterations": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _initialize_bayesian_priors(self) -> Dict:
        """Initialize Bayesian priors for all possible transitions"""
        priors = {}
        for i in range(1, 10):
            source = str(i)
            priors[source] = {}
            for j in range(1, 10):
                target = str(j)
                # Default weak prior
                priors[source][target] = 0.1
                
        # Set stronger priors for known transitions
        for transition in self.transitions["transitions"]:
            source = transition["source"]
            target = transition["target"]
            priors[source][target] = transition["probability"]
            
        return priors
    
    def _save_transitions(self, filename="transitions.json"):
        """Save the current transition model"""
        path = os.path.join(self.data_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.transitions, f, indent=2)
            
    def _save_memory(self, filename="memory.json"):
        """Save the current memory state"""
        path = os.path.join(self.data_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def _update_memory(self, source: str, target: str, feedback: Optional[Dict] = None):
        """Update memory with new transition information"""
        # Update visit counts
        self.memory["zone_visits"][source] = self.memory["zone_visits"].get(source, 0) + 1
        
        # Update transition frequency
        if source not in self.memory["transition_frequency"]:
            self.memory["transition_frequency"][source] = {}
        
        self.memory["transition_frequency"][source][target] = \
            self.memory["transition_frequency"][source].get(target, 0) + 1
        
        # Add feedback if provided
        if feedback:
            self.memory["feedback_history"].append({
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "target": target,
                "feedback": feedback
            })
        
        # Update timestamp
        self.memory["last_updated"] = datetime.now().isoformat()
        self.memory["learning_iterations"] += 1
        
        # Save the updated memory
        self._save_memory()
    
    def _bayesian_update(self, source: str, target: str, success_factor: float = 1.0):
        """Update Bayesian probabilities based on feedback"""
        # Get current priors
        priors = self.memory["bayesian_priors"][source]
        
        # Simple Bayesian update - increase probability for successful transition
        # and normalize others
        total_probability = sum(priors.values())
        
        for t, probability in priors.items():
            if t == target:
                # Increase probability based on success factor
                new_prob = probability * (1 + 0.1 * success_factor)
            else:
                # Slightly decrease other probabilities
                new_prob = probability * 0.99
                
            priors[t] = new_prob
        
        # Normalize to ensure sum is still 1.0
        normalization_factor = total_probability / sum(priors.values())
        for t in priors:
            priors[t] *= normalization_factor
            
        # Update the transition probability in the transition model
        for i, transition in enumerate(self.transitions["transitions"]):
            if transition["source"] == source and transition["target"] == target:
                self.transitions["transitions"][i]["probability"] = priors[target]
                break
        
        # Save updates
        self._save_transitions()
        self._save_memory()
    
    def _apply_reinforcement_learning(self, source: str, target: str, reward: float):
        """Apply reinforcement learning to adjust transition probabilities"""
        # Simple Q-learning inspired approach
        learning_rate = 0.1
        discount_factor = 0.9
        
        for i, transition in enumerate(self.transitions["transitions"]):
            if transition["source"] == source and transition["target"] == target:
                # Update the evolutionary impact based on reward
                current_impact = transition["evolutionary_impact"]
                new_impact = current_impact + learning_rate * (reward + discount_factor - current_impact)
                self.transitions["transitions"][i]["evolutionary_impact"] = new_impact
                
                # Also adjust probability
                current_prob = transition["probability"]
                new_prob = current_prob + learning_rate * reward * 0.1
                self.transitions["transitions"][i]["probability"] = max(0.1, min(0.9, new_prob))
                break
                
        # Save updates
        self._save_transitions()
    
    def get_possible_transitions(self, current_zone: str) -> List[Dict]:
        """Get all possible transitions from the current zone"""
        return [t for t in self.transitions["transitions"] if t["source"] == current_zone]
    
    def zone_transition(self, request: Dict) -> Dict:
        """Calculate zone transitions using advanced numogram intelligence"""
        current_zone = request["current_zone"]
        user_input = request["input"]
        context = request.get("context", {})
        
        # Get possible transitions
        possible_transitions = self.get_possible_transitions(current_zone)
        
        if not possible_transitions:
            # Fallback to zone 1 if no transitions exist
            return {
                "transition_result": ["1"],
                "probabilities": {"1": 1.0},
                "context": {"error": "No transitions defined for zone " + current_zone}
            }
        
        # Apply Bayesian decision making
        bayesian_priors = self.memory["bayesian_priors"].get(current_zone, {})
        
        # Adjust probabilities based on input heuristics
        # This is a simple example - could be much more sophisticated
        input_len = len(user_input)
        input_sum = sum(ord(c) for c in user_input) % 9 + 1
        input_factor = str(input_sum)
        
        # Calculate modified probabilities
        transition_weights = {}
        for transition in possible_transitions:
            target = transition["target"]
            base_prob = transition["probability"]
            
            # Evolutionary factor - transitions with higher impact have increased chances
            evo_factor = 1.0 + transition["evolutionary_impact"]
            
            # Input resonance - if input has numerical resonance with target
            input_resonance = 1.2 if input_factor == target else 1.0
            
            # Memory factor - transitions taken more often have higher probability
            memory_factor = 1.0
            if current_zone in self.memory["transition_frequency"]:
                freq = self.memory["transition_frequency"][current_zone].get(target, 0)
                total_freq = sum(self.memory["transition_frequency"][current_zone].values()) or 1
                memory_factor = 1.0 + (freq / total_freq) * 0.5
            
            # Feedback factor from previous interactions
            feedback_factor = 1.0
            
            # Calculate final weight
            weight = base_prob * evo_factor * input_resonance * memory_factor * feedback_factor
            transition_weights[target] = weight
        
        # Normalize weights
        total_weight = sum(transition_weights.values()) or 1.0
        for target in transition_weights:
            transition_weights[target] /= total_weight
        
        # Choose transitions based on weighted probabilities
        targets = list(transition_weights.keys())
        probabilities = list(transition_weights.values())
        
        # Typically return the highest probability, but occasionally explore
        explore_chance = min(0.2, 1.0 / (self.memory["learning_iterations"] + 1))
        
        if random.random() < explore_chance:
            # Exploration - choose probabilistically
            results = random.choices(targets, weights=probabilities, k=1)
        else:
            # Exploitation - choose highest probability
            results = [targets[np.argmax(probabilities)]]
        
        # Update memory
        for target in results:
            self._update_memory(current_zone, target)
        
        # Prepare response context
        response_context = {
            "timestamp": datetime.now().isoformat(),
            "input_analysis": {
                "length": input_len,
                "numerical_resonance": input_factor
            },
            "exploration_factor": explore_chance,
            "zone_data": self.zones.get(results[0], {})
        }
        
        return {
            "transition_result": results,
            "probabilities": {t: float(p) for t, p in zip(targets, probabilities)},
            "context": response_context,
            "feedback_required": explore_chance > 0.1  # Occasionally ask for feedback
        }
    
    def process_feedback(self, source: str, target: str, feedback_value: float):
        """Process user feedback to improve the model"""
        # Update Bayesian model
        self._bayesian_update(source, target, feedback_value)
        
        # Apply reinforcement learning
        self._apply_reinforcement_learning(source, target, feedback_value)
        
        # Update memory with feedback
        self._update_memory(source, target, {"value": feedback_value})
        
        return {"status": "feedback_processed"}

# numogram_code/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime

from numogram_algorithm import NumogramCore

app = FastAPI(
    title="Numogram Intelligence API",
    description="Advanced Numogram system with evolutionary transitions, memory, and Bayesian decision-making",
    version="1.0.0"
)

# Initialize the numogram core
numogram = NumogramCore()

class TransitionRequest(BaseModel):
    current_zone: str
    input: str
    context: Optional[Dict[str, Any]] = None

class FeedbackRequest(BaseModel):
    source_zone: str
    target_zone: str
    feedback_value: float  # -1.0 to 1.0
    comments: Optional[str] = None

@app.post("/numogram/transition")
async def transition(request: TransitionRequest):
    """Process a zone transition request using advanced numogram intelligence"""
    try:
        result = numogram.zone_transition(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transition processing error: {str(e)}")

@app.post("/numogram/feedback")
async def feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Process feedback to improve the numogram intelligence"""
    try:
        # Process feedback in the background to avoid blocking
        background_tasks.add_task(
            numogram.process_feedback,
            request.source_zone,
            request.target_zone, 
            request.feedback_value
        )
        return {"status": "feedback_accepted", "message": "Processing in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback processing error: {str(e)}")

@app.get("/numogram/zones")
async def get_zones():
    """Get all available zones and their descriptions"""
    return {"zones": numogram.zones}

@app.get("/numogram/stats")
async def get_stats():
    """Get statistics about the numogram system"""
    stats = {
        "total_transitions": sum(sum(freqs.values()) for freqs in numogram.memory["transition_frequency"].values()),
        "zone_visits": numogram.memory["zone_visits"],
        "learning_iterations": numogram.memory["learning_iterations"],
        "last_updated": numogram.memory["last_updated"],
        "most_common_transitions": {}
    }
    
    # Calculate most common transitions
    for source, targets in numogram.memory["transition_frequency"].items():
        if targets:
            most_common = max(targets.items(), key=lambda x: x[1])
            stats["most_common_transitions"][source] = {
                "target": most_common[0],
                "frequency": most_common[1]
            }
    
    return stats

# numogram_code/transitions.json
{
  "transitions": [
    {
      "source": "1",
      "target": "2",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "1",
      "target": "4",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "2",
      "target": "3",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "2",
      "target": "6",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "3",
      "target": "1",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "3",
      "target": "9",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "4",
      "target": "5",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "4",
      "target": "8",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "5",
      "target": "7",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "6",
      "target": "5",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "7",
      "target": "8",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "8",
      "target": "9",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    },
    {
      "source": "9",
      "target": "6",
      "probability": 0.5,
      "energy_cost": 1.0,
      "evolutionary_impact": 0.1,
      "feedback_coefficient": 0.1
    }
  ],
  "meta": {
    "created": "2025-02-26T12:00:00",
    "version": "1.0"
  }
}

# numogram_code/zones.json
{
  "zones": {
    "1": {
      "name": "Isolation Zone",
      "description": "The zone of singularity and self-reference",
      "energy_coefficient": 1.0,
      "evolutionary_potential": 0.7
    },
    "2": {
      "name": "Duplication Zone",
      "description": "The zone of binary operations and mirroring",
      "energy_coefficient": 1.2,
      "evolutionary_potential": 0.6
    },
    "3": {
      "name": "Triad Zone",
      "description": "The zone of synthesis and triangulation",
      "energy_coefficient": 1.3,
      "evolutionary_potential": 0.8
    },
    "4": {
      "name": "Quadratic Zone",
      "description": "The zone of stability and foundational structures",
      "energy_coefficient": 1.1,
      "evolutionary_potential": 0.5
    },
    "5": {
      "name": "Pentagonal Zone",
      "description": "The zone of dynamic balance and transformation",
      "energy_coefficient": 1.5,
      "evolutionary_potential": 0.9
    },
    "6": {
      "name": "Hexagonal Zone",
      "description": "The zone of harmony and interconnection",
      "energy_coefficient": 1.4,
      "evolutionary_potential": 0.6
    },
    "7": {
      "name": "Heptagonal Zone",
      "description": "The zone of mystical convergence and emergence",
      "energy_coefficient": 1.7,
      "evolutionary_potential": 0.8
    },
    "8": {
      "name": "Octagonal Zone",
      "description": "The zone of cyclical patterns and infinity",
      "energy_coefficient": 1.6,
      "evolutionary_potential": 0.7
    },
    "9": {
      "name": "Enneagonal Zone",
      "description": "The zone of completion and universal principles",
      "energy_coefficient": 1.9,
      "evolutionary_potential": 0.9
    }
  }
}

# numogram_code/main.py
import uvicorn
import os
import sys
from pathlib import Path

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure required files exist
current_dir = Path(__file__).parent
files_to_check = ['zones.json', 'transitions.json']

for file in files_to_check:
    file_path = current_dir / file
    if not file_path.exists():
        print(f"Warning: {file} not found. The system will create it with default values.")

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
