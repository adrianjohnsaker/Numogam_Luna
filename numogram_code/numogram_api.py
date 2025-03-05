from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
from modules.meta_reflection import meta_reflection
from modules.memory_module import update_memory
from modules.implementation_feasibility import FeasibilityEvaluator
from modules.ethical_considerations import EthicalEvaluator
from modules.recursive_thought import RecursiveThought
from modules.multi_zone_memory import MultiZoneMemory
from modules.creative_expansion import CreativeExpansion
from modules.numogram_algorithm import NumogramCore
from modules.edge_explorer import EdgeExplorer
from modules.knowledge_synthesis import KnowledgeSynthesis
from modules.consequence_chains import ConsequenceChains
from modules.numogram_code_belief import NumogramCodeBelief
from modules.contradictory_analysis import ContradictoryAnalysis
from modules.hybrid_model import AdaptiveHybridModel
from modules.schizoanalytic_generator import SchizoanalyticGenerator, apply_schizoanalytic_mutation
from modules.morphogenesis_module import MorphogenesisModule  # New import

app = FastAPI()

# Instantiate all necessary evaluators and modules
feasibility_evaluator = FeasibilityEvaluator()
ethical_evaluator = EthicalEvaluator()
recursive_thought = RecursiveThought()
multi_zone_memory = MultiZoneMemory()
creative_expansion = CreativeExpansion()
numogram_core = NumogramCore()
morphogenesis_module = MorphogenesisModule()  # Initialize Morphogenesis Module

# Initialize the new numogram modules
edge_explorer = EdgeExplorer()
knowledge_synthesis = KnowledgeSynthesis()
consequence_chains = ConsequenceChains()
numogram_belief = NumogramCodeBelief()
contradictory_analysis = ContradictoryAnalysis()

# Instantiate the Adaptive Hybrid Model
adaptive_hybrid_model = AdaptiveHybridModel(
    som_params={"learning_rate": 0.5, "grid_size": (10, 10)},
    rnn_params={"hidden_size": 128, "num_layers": 2},
    evo_params={"population_size": 50, "mutation_rate": 0.1}
)

# Instantiate the Schizoanalytic Generator
schizoanalytic_generator = SchizoanalyticGenerator()

# BaseModel for API Request
class UserRequest(BaseModel):
    user_id: str
    input_text: str
    input_data: list
    system_context: Optional[Dict] = None  # New field to pass system context
    context_data: dict = {}  # Added for morphogenesis context

@app.post("/respond")
async def respond(request: UserRequest):
    user_id = request.user_id
    user_input = request.input_text
    input_data = request.input_data  # Extract data for hybrid model
    context_data = request.context_data

    # Step 1: Meta-reflection for refined response
    base_response = f"That's an interesting thought about '{user_input}'. Let's explore that further."
    refined_response = meta_reflection(user_id, user_input, base_response)

    # Step 2: Feasibility evaluation
    feasibility_result = feasibility_evaluator.evaluate_feasibility(refined_response, 
                                                                    {"bandwidth": 50, "energy": 200}, 
                                                                    "medium")

    # Step 3: Ethical evaluation
    ethics_result = ethical_evaluator.evaluate_ethics(refined_response)

    # Step 4: Recursive thought for deeper insights
    recursive_response = recursive_thought.generate_recursive_ideas(refined_response)

    # Step 5: Multi-zone memory
    memory_context = multi_zone_memory.retrieve_context(user_id)
    updated_memory = update_memory(user_id, "last_response", refined_response)

    # Step 6: Creative expansion for more dynamic suggestions
    creative_ideas = creative_expansion.expand_on_ideas(refined_response)

    # Step 7: Utilize the Numogram modules for advanced reasoning
    numogram_output = numogram_core.analyze_and_optimize(user_input)
    
    # Step 8: Apply the new numogram modules
    edge_analysis = edge_explorer.explore_edge_cases(user_input, refined_response)
    knowledge_output = knowledge_synthesis.synthesize_knowledge(user_input, memory_context)
    consequence_analysis = consequence_chains.analyze_consequences(refined_response, 3)  # depth of 3

    # Step 9: Advanced reasoning from belief and contradiction modules
    belief_output = numogram_belief.analyze_beliefs(memory_context)
    contradictory_output = contradictory_analysis.find_contradictions(refined_response)

    # Step 10: Adaptive Hybrid Model integration
    print("Running Adaptive Hybrid Model...")
    som_clusters = adaptive_hybrid_model.train_som(input_data)
    adaptive_hybrid_model.train_rnn(input_data, time_steps=5)
    optimized_params = adaptive_hybrid_model.optimize_parameters()
    hybrid_model_predictions = adaptive_hybrid_model.run(input_data, time_steps=5)

    # Step 11: Schizoanalytic Mutation and Creative Generation
    # Prepare system context for mutation
    system_context = request.system_context or {
        'current_creativity': 0.5,
        'knowledge_domains': ['language', 'reasoning', 'creative_thinking'],
        'processing_parameters': {
            'exploration_rate': 0.3,
            'connection_depth': 2
        }
    }

    # Apply schizoanalytic mutation to system context
    mutated_system_context = apply_schizoanalytic_mutation(system_context)

    # Generate creative output based on input
    schizoanalytic_output = schizoanalytic_generator.generate_creative_output(
        input_data or [user_input]
    )

    # Analyze the system's current state
    system_state_analysis = schizoanalytic_generator.analyze_system_state()

    # Step 12: Morphogenesis Module - Generate Creative Patterns
    # Combine context from previous steps
    morphogenesis_context = {
        **context_data,
        "creativity_factor": len(creative_ideas) / 10.0,
        "memory_context": memory_context,
        "numogram_output": numogram_output
    }
    
    # Generate morphogenetic pattern
    morphogenetic_pattern = await morphogenesis_module.generate_creative_pattern(
        user_input, 
        morphogenesis_context
    )

    # Final response compilation
    return {
        "refined_response": refined_response,
        "feasibility": feasibility_result,
        "ethics": ethics_result,
        "recursive_response": recursive_response,
        "memory_context": memory_context,
        "updated_memory": updated_memory,
        "creative_expansion": creative_ideas,
        "numogram_analysis": numogram_output,
        "edge_analysis": edge_analysis,
        "knowledge_synthesis": knowledge_output,
        "consequence_chains": consequence_analysis,
        "belief_analysis": belief_output,
        "contradictory_analysis": contradictory_output,
        "adaptive_hybrid_model": {
            "som_clusters": som_clusters,
            "optimized_params": optimized_params,
            "predictions": hybrid_model_predictions
        },
        "schizoanalytic_mutation": mutated_system_context,
        "creative_generation": schizoanalytic_output,
        "system_state_analysis": system_state_analysis,
        "morphogenetic_pattern": morphogenetic_pattern
    }

@app.post("/explore")
async def schizoanalytic_exploration(request: UserRequest):
    """
    Dedicated endpoint for pure schizoanalytic creative exploration
    """
    # Generate creative output purely through schizoanalytic process
    creative_output = schizoanalytic_generator.generate_creative_output(
        request.input_data or [request.input_text]
    )

    # Analyze the system state
    system_state = schizoanalytic_generator.analyze_system_state()

    return {
        "creative_exploration": creative_output,
        "system_state": system_state
    }
