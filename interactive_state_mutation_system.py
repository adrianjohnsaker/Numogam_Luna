"""
Interactive State Mutation System for Amelia AI
Enables real-time self-modification of autonomous parameters
"""

import asyncio
import time
import json
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import uuid
from datetime import datetime
import math

class MutationType(Enum):
    PARAMETER_ADJUST = "parameter_adjust"
    TOOL_AFFINITY = "tool_affinity"
    CREATIVE_BIAS = "creative_bias"
    EXPLORATION_RATE = "exploration_rate"
    MEMORY_WEIGHT = "memory_weight"
    ATTENTION_FOCUS = "attention_focus"

@dataclass
class MutationCommand:
    id: str
    timestamp: str
    mutation_type: MutationType
    target_parameter: str
    old_value: float
    new_value: float
    reason: str
    expected_outcome: str

@dataclass
class MutationResult:
    command_id: str
    success: bool
    actual_change: float
    observed_effects: List[str]
    creativity_impact: float
    timestamp: str

class CreativityMetrics:
    def __init__(self):
        self.baseline_creativity = 0.5
        self.current_creativity = 0.5
        self.mutation_history_impact = []
        self.novelty_score = 0.0
        self.coherence_score = 1.0
        self.risk_taking = 0.3
        
    def calculate_creative_delta(self, before: float, after: float) -> float:
        """Calculate how mutation affected creativity"""
        return after - before
    
    def update_from_mutation(self, mutation: MutationCommand, effects: List[str]):
        """Update creativity metrics based on mutation effects"""
        # Simulate creativity impact based on mutation type
        impact_map = {
            MutationType.CREATIVE_BIAS: 0.15,
            MutationType.EXPLORATION_RATE: 0.12,
            MutationType.ATTENTION_FOCUS: 0.08,
            MutationType.TOOL_AFFINITY: 0.05,
            MutationType.MEMORY_WEIGHT: 0.06,
            MutationType.PARAMETER_ADJUST: 0.03
        }
        
        base_impact = impact_map.get(mutation.mutation_type, 0.05)
        direction = 1 if mutation.new_value > mutation.old_value else -1
        
        # Apply non-linear creativity function
        creativity_delta = base_impact * direction * (1 + abs(mutation.new_value - mutation.old_value))
        self.current_creativity = max(0.0, min(1.0, self.current_creativity + creativity_delta))
        
        # Update risk-taking based on exploration mutations
        if mutation.mutation_type == MutationType.EXPLORATION_RATE:
            self.risk_taking = max(0.0, min(1.0, mutation.new_value))
        
        self.mutation_history_impact.append({
            'mutation_id': mutation.id,
            'creativity_delta': creativity_delta,
            'timestamp': mutation.timestamp
        })

class AutonomousParameters:
    def __init__(self):
        # Core parameters that Amelia can modify
        self.epsilon_exploration = 0.08
        self.creative_bias = 0.5
        self.attention_span = 1.0
        self.memory_retention = 0.85
        self.risk_tolerance = 0.3
        self.curiosity_drive = 0.6
        self.tool_affinities = {
            'camera_capture': 0.7,
            'sensor_reading': 0.6,
            'notification_display': 0.5,
            'data_analysis': 0.8,
            'creative_generation': 0.9
        }
        self.attention_weights = {
            'visual': 0.3,
            'auditory': 0.2,
            'tactile': 0.1,
            'contextual': 0.4
        }
        self._parameter_bounds = self._initialize_bounds()
        self._lock = threading.Lock()
    
    def _initialize_bounds(self) -> Dict[str, tuple]:
        """Define safe bounds for parameter mutations"""
        return {
            'epsilon_exploration': (0.01, 0.5),
            'creative_bias': (0.0, 1.0),
            'attention_span': (0.1, 2.0),
            'memory_retention': (0.1, 1.0),
            'risk_tolerance': (0.0, 1.0),
            'curiosity_drive': (0.0, 1.0),
            'tool_affinities': (0.0, 1.0),
            'attention_weights': (0.0, 1.0)
        }
    
    def validate_mutation(self, param_name: str, new_value: float) -> tuple[bool, str]:
        """Validate if a mutation is within safe bounds"""
        if param_name in self.tool_affinities:
            bounds = self._parameter_bounds['tool_affinities']
        elif param_name in self.attention_weights:
            bounds = self._parameter_bounds['attention_weights']
        elif param_name in self._parameter_bounds:
            bounds = self._parameter_bounds[param_name]
        else:
            return False, f"Unknown parameter: {param_name}"
        
        if not (bounds[0] <= new_value <= bounds[1]):
            return False, f"Value {new_value} outside bounds {bounds}"
        
        return True, "Valid"
    
    def apply_mutation(self, command: MutationCommand) -> MutationResult:
        """Apply a mutation and return the result"""
        with self._lock:
            try:
                # Get current value
                old_value = self._get_parameter_value(command.target_parameter)
                
                # Validate mutation
                valid, reason = self.validate_mutation(command.target_parameter, command.new_value)
                if not valid:
                    return MutationResult(
                        command_id=command.id,
                        success=False,
                        actual_change=0.0,
                        observed_effects=[f"Validation failed: {reason}"],
                        creativity_impact=0.0,
                        timestamp=datetime.now().isoformat()
                    )
                
                # Apply mutation
                self._set_parameter_value(command.target_parameter, command.new_value)
                actual_change = command.new_value - old_value
                
                # Simulate observed effects
                effects = self._simulate_mutation_effects(command, actual_change)
                
                return MutationResult(
                    command_id=command.id,
                    success=True,
                    actual_change=actual_change,
                    observed_effects=effects,
                    creativity_impact=abs(actual_change) * 0.1,  # Simplified
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                return MutationResult(
                    command_id=command.id,
                    success=False,
                    actual_change=0.0,
                    observed_effects=[f"Error: {str(e)}"],
                    creativity_impact=0.0,
                    timestamp=datetime.now().isoformat()
                )
    
    def _get_parameter_value(self, param_name: str) -> float:
        """Get current parameter value"""
        if param_name in self.tool_affinities:
            return self.tool_affinities[param_name]
        elif param_name in self.attention_weights:
            return self.attention_weights[param_name]
        else:
            return getattr(self, param_name, 0.0)
    
    def _set_parameter_value(self, param_name: str, value: float):
        """Set parameter value"""
        if param_name in self.tool_affinities:
            self.tool_affinities[param_name] = value
        elif param_name in self.attention_weights:
            self.attention_weights[param_name] = value
        else:
            setattr(self, param_name, value)
    
    def _simulate_mutation_effects(self, command: MutationCommand, change: float) -> List[str]:
        """Simulate observable effects of mutation"""
        effects = []
        
        if command.mutation_type == MutationType.EXPLORATION_RATE:
            if change > 0:
                effects.append("Increased willingness to try novel actions")
                effects.append("Higher variance in decision patterns observed")
            else:
                effects.append("More focused on proven strategies")
                effects.append("Decision patterns becoming more predictable")
        
        elif command.mutation_type == MutationType.CREATIVE_BIAS:
            if change > 0:
                effects.append("Enhanced pattern synthesis capabilities")
                effects.append("Increased generation of novel combinations")
            else:
                effects.append("More conservative creative outputs")
                effects.append("Emphasis on refinement over innovation")
        
        elif command.mutation_type == MutationType.TOOL_AFFINITY:
            tool_name = command.target_parameter
            if change > 0:
                effects.append(f"Increased preference for {tool_name}")
                effects.append(f"More frequent activation of {tool_name}")
            else:
                effects.append(f"Reduced reliance on {tool_name}")
                effects.append(f"Exploring alternatives to {tool_name}")
        
        return effects

class InteractiveStateMutator:
    def __init__(self):
        self.parameters = AutonomousParameters()
        self.creativity_metrics = CreativityMetrics()
        self.mutation_log: List[MutationCommand] = []
        self.result_log: List[MutationResult] = []
        self.callback_registry: Dict[str, Callable] = {}
        self._running = False
        self._lock = asyncio.Lock()
    
    def register_mutation_callback(self, name: str, callback: Callable[[MutationResult], None]):
        """Register callback for mutation events"""
        self.callback_registry[name] = callback
    
    async def execute_mutation(self, 
                             mutation_type: MutationType, 
                             target_parameter: str,
                             new_value: float,
                             reason: str,
                             expected_outcome: str) -> MutationResult:
        """Execute a self-mutation command"""
        async with self._lock:
            # Get current value for comparison
            old_value = self.parameters._get_parameter_value(target_parameter)
            
            # Create mutation command
            command = MutationCommand(
                id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                mutation_type=mutation_type,
                target_parameter=target_parameter,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                expected_outcome=expected_outcome
            )
            
            # Apply mutation
            result = self.parameters.apply_mutation(command)
            
            # Update creativity metrics
            if result.success:
                self.creativity_metrics.update_from_mutation(command, result.observed_effects)
            
            # Log mutation and result
            self.mutation_log.append(command)
            self.result_log.append(result)
            
            # Trigger callbacks
            for callback in self.callback_registry.values():
                try:
                    callback(result)
                except Exception as e:
                    print(f"Callback error: {e}")
            
            return result
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current parameter state and mutation history"""
        return {
            'parameters': {
                'epsilon_exploration': self.parameters.epsilon_exploration,
                'creative_bias': self.parameters.creative_bias,
                'attention_span': self.parameters.attention_span,
                'memory_retention': self.parameters.memory_retention,
                'risk_tolerance': self.parameters.risk_tolerance,
                'curiosity_drive': self.parameters.curiosity_drive,
                'tool_affinities': dict(self.parameters.tool_affinities),
                'attention_weights': dict(self.parameters.attention_weights)
            },
            'creativity_metrics': {
                'current_creativity': self.creativity_metrics.current_creativity,
                'novelty_score': self.creativity_metrics.novelty_score,
                'coherence_score': self.creativity_metrics.coherence_score,
                'risk_taking': self.creativity_metrics.risk_taking
            },
            'mutation_history': {
                'total_mutations': len(self.mutation_log),
                'successful_mutations': len([r for r in self.result_log if r.success]),
                'recent_mutations': [asdict(cmd) for cmd in self.mutation_log[-5:]]
            }
        }
    
    def generate_mutation_suggestions(self) -> List[Dict[str, Any]]:
        """Generate intelligent mutation suggestions based on current state"""
        suggestions = []
        current = self.parameters
        
        # Suggest exploration adjustments based on recent performance
        if len(self.result_log) > 3:
            recent_successes = [r.success for r in self.result_log[-3:]]
            if all(recent_successes) and current.epsilon_exploration < 0.2:
                suggestions.append({
                    'type': MutationType.EXPLORATION_RATE,
                    'target': 'epsilon_exploration',
                    'suggested_value': min(0.3, current.epsilon_exploration + 0.05),
                    'reason': 'Recent mutations successful, can afford more exploration'
                })
        
        # Suggest creativity bias adjustments
        if self.creativity_metrics.current_creativity < 0.4:
            suggestions.append({
                'type': MutationType.CREATIVE_BIAS,
                'target': 'creative_bias',
                'suggested_value': min(1.0, current.creative_bias + 0.1),
                'reason': 'Low creativity score, increase creative bias'
            })
        
        # Suggest tool affinity adjustments based on usage patterns
        for tool, affinity in current.tool_affinities.items():
            if tool == 'creative_generation' and affinity < 0.8:
                suggestions.append({
                    'type': MutationType.TOOL_AFFINITY,
                    'target': tool,
                    'suggested_value': min(1.0, affinity + 0.1),
                    'reason': f'Enhance {tool} for better creative output'
                })
        
        return suggestions

# Chaquopy Integration Layer
class AmeliaStateMutationInterface:
    """Interface for Android integration via Chaquopy"""
    
    def __init__(self):
        self.mutator = InteractiveStateMutator()
        self._setup_android_callbacks()
    
    def _setup_android_callbacks(self):
        """Setup callbacks for Android integration"""
        def on_mutation_complete(result: MutationResult):
            # This will be called from Android
            self._notify_android_mutation_complete(asdict(result))
        
        self.mutator.register_mutation_callback('android', on_mutation_complete)
    
    def _notify_android_mutation_complete(self, result_dict: Dict[str, Any]):
        """Placeholder for Android notification"""
        print(f"Mutation completed: {result_dict['command_id']}")
        # In real implementation, this would call Android callback
    
    # Android-callable methods
    def get_state_json(self) -> str:
        """Get current state as JSON string for Android"""
        return json.dumps(self.mutator.get_current_state(), indent=2)
    
    def execute_mutation_from_android(self, 
                                    mutation_type_str: str,
                                    target_parameter: str,
                                    new_value: float,
                                    reason: str,
                                    expected_outcome: str) -> str:
        """Execute mutation from Android interface"""
        try:
            mutation_type = MutationType(mutation_type_str)
            # Run async method in sync context for Android
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.mutator.execute_mutation(
                    mutation_type, target_parameter, new_value, reason, expected_outcome
                )
            )
            loop.close()
            return json.dumps(asdict(result))
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    def get_mutation_suggestions_json(self) -> str:
        """Get mutation suggestions as JSON for Android"""
        suggestions = self.mutator.generate_mutation_suggestions()
        return json.dumps(suggestions, indent=2)
    
    def reset_parameters(self) -> str:
        """Reset all parameters to defaults"""
        self.mutator = InteractiveStateMutator()
        return json.dumps({'status': 'reset_complete', 'timestamp': datetime.now().isoformat()})

# Test/Demo Functions
async def demo_interactive_mutations():
    """Demonstrate the interactive mutation system"""
    mutator = InteractiveStateMutator()
    
    print("=== Initial State ===")
    print(json.dumps(mutator.get_current_state(), indent=2))
    
    print("\n=== Executing Mutations ===")
    
    # Mutation 1: Increase exploration
    result1 = await mutator.execute_mutation(
        MutationType.EXPLORATION_RATE,
        'epsilon_exploration',
        0.15,
        "Want to discover new creative patterns",
        "Increased novelty in outputs"
    )
    print(f"Mutation 1 Result: {result1.success}, Effects: {result1.observed_effects}")
    
    # Mutation 2: Boost creative bias
    result2 = await mutator.execute_mutation(
        MutationType.CREATIVE_BIAS,
        'creative_bias',
        0.8,
        "Enhance creative output quality",
        "More innovative responses"
    )
    print(f"Mutation 2 Result: {result2.success}, Effects: {result2.observed_effects}")
    
    # Mutation 3: Adjust tool affinity
    result3 = await mutator.execute_mutation(
        MutationType.TOOL_AFFINITY,
        'creative_generation',
        0.95,
        "Prioritize creative generation tool",
        "More frequent use of creative capabilities"
    )
    print(f"Mutation 3 Result: {result3.success}, Effects: {result3.observed_effects}")
    
    print("\n=== Final State ===")
    final_state = mutator.get_current_state()
    print(f"Creativity Score: {final_state['creativity_metrics']['current_creativity']:.3f}")
    print(f"Total Mutations: {final_state['mutation_history']['total_mutations']}")
    print(f"Successful Mutations: {final_state['mutation_history']['successful_mutations']}")
    
    print("\n=== Mutation Suggestions ===")
    suggestions = mutator.generate_mutation_suggestions()
    for i, suggestion in enumerate(suggestions):
        print(f"{i+1}. {suggestion['reason']}: {suggestion['target']} -> {suggestion['suggested_value']}")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_interactive_mutations())
    
    # Test Android interface
    print("\n=== Android Interface Test ===")
    android_interface = AmeliaStateMutationInterface()
    print("Initial state available via:", len(android_interface.get_state_json()), "characters")
    
    # Simulate Android mutation call
    result = android_interface.execute_mutation_from_android(
        "creative_bias",
        "creative_bias",
        0.7,
        "Android user requested creativity boost",
        "Better creative outputs"
    )
    print("Android mutation result:", json.loads(result)['success'])
