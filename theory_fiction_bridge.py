# theory_fiction_bridge.py
import json
import traceback
import random
from typing import Dict, List, Any, Optional

# Import the TheoryFictionSimulation module
try:
    from theory_fiction_module import TheoryFictionSimulation, NarrativeMode
except ImportError:
    # This will be handled by importing the complete implementation
    pass

# Create a singleton instance
theory_fiction_simulation = None

def init_simulation(seed=None):
    """Initialize the theory fiction simulation with optional seed"""
    global theory_fiction_simulation
    if theory_fiction_simulation is None:
        try:
            from theory_fiction_module import TheoryFictionSimulation
            theory_fiction_simulation = TheoryFictionSimulation(seed=seed)
        except ImportError as e:
            return {"error": f"Failed to import TheoryFictionSimulation: {str(e)}"}
    return {"status": "initialized"}

def test_connection():
    """Simple function to test if Python bridge is working"""
    return json.dumps({"status": "connected"})

def execute_theory_fiction(command, params_json):
    """
    Execute a command in the TheoryFictionSimulation module
    
    Args:
        command: String command to execute
        params_json: JSON string of parameters
    
    Returns:
        JSON string with the results
    """
    try:
        # Initialize if not already done
        if theory_fiction_simulation is None:
            init_result = init_simulation()
            if "error" in init_result:
                return json.dumps(init_result)
        
        params = json.loads(params_json)
        result = {}
        
        if command == "create_narrative_context":
            title = params.get("title", "Untitled Narrative")
            core_premise = params.get("core_premise", "")
            
            context = theory_fiction_simulation.create_narrative_context(
                title=title,
                core_premise=core_premise
            )
            
            result = {
                "context_id": context.id,
                "title": context.title,
                "core_premise": context.core_premise
            }
            
        elif command == "add_character":
            context_id = params.get("context_id")
            context = theory_fiction_simulation.narrative_contexts.get(context_id)
            
            if not context:
                return json.dumps({"error": f"No narrative context found with ID {context_id}"})
            
            # Create character from parameters
            from theory_fiction_module import Character
            character = Character(
                name=params.get("name", "Unnamed Character"),
                background=params.get("background", ""),
                motivations=params.get("motivations", []),
                traits=params.get("traits", {})
            )
            
            # Add to context
            character_id = context.add_character(character)
            
            result = {
                "character_id": character_id,
                "name": character.name
            }
            
        elif command == "add_parameter":
            context_id = params.get("context_id")
            context = theory_fiction_simulation.narrative_contexts.get(context_id)
            
            if not context:
                return json.dumps({"error": f"No narrative context found with ID {context_id}"})
            
            # Create parameter from parameters
            from theory_fiction_module import WorldParameter
            parameter = WorldParameter(
                name=params.get("name", "Unnamed Parameter"),
                value=params.get("value", 0.5),
                description=params.get("description", "")
            )
            
            # Add to context
            parameter_id = context.add_parameter(parameter)
            
            result = {
                "parameter_id": parameter_id,
                "name": parameter.name,
                "value": parameter.value
            }
            
        elif command == "generate_theoretical_framework":
            name = params.get("name")
            num_concepts = params.get("num_concepts", 3)
            
            framework = theory_fiction_simulation.generate_theoretical_framework(
                name=name,
                num_concepts=num_concepts
            )
            
            # Extract concepts for result
            concepts = []
            for concept_id, concept in framework.concepts.items():
                concepts.append({
                    "id": concept_id,
                    "name": concept.name,
                    "domain": concept.domain.name
                })
            
            result = {
                "framework_id": framework.id,
                "framework_info": {
                    "name": framework.name,
                    "description": framework.description,
                    "core_principles": framework.core_principles,
                    "key_thinkers": framework.key_thinkers,
                    "concepts": concepts
                }
            }
            
        elif command == "generate_speculative_scenario":
            context_id = params.get("context_id")
            complexity = params.get("complexity", 3)
            
            # Generate scenario
            context = theory_fiction_simulation.generate_speculative_scenario(
                context_id=context_id,
                complexity=complexity
            )
            
            if not context:
                return json.dumps({"error": f"Failed to generate scenario for context {context_id}"})
            
            # Extract events for result
            events = []
            for event in context.events:
                events.append({
                    "id": event.id,
                    "description": event.description,
                    "category": event.category.name,
                    "probability": event.probability
                })
            
            result = {
                "context_id": context_id,
                "events_generated": len(events),
                "events": events
            }
            
        elif command == "apply_theoretical_framework":
            context_id = params.get("context_id")
            framework_id = params.get("framework_id")
            
            analysis = theory_fiction_simulation.apply_theoretical_framework(
                context_id=context_id,
                framework_id=framework_id
            )
            
            result = analysis
            
        elif command == "resolve_narrative":
            context_id = params.get("context_id")
            
            resolution = theory_fiction_simulation.resolve_narrative(
                context_id=context_id
            )
            
            result = resolution
            
        elif command == "analyze_narrative_dynamics":
            context_id = params.get("context_id")
            
            analysis = theory_fiction_simulation.analyze_narrative_dynamics(
                context_id=context_id
            )
            
            result = analysis
            
        elif command == "export_narrative":
            context_id = params.get("context_id")
            format = params.get("format", "text")
            
            narrative = theory_fiction_simulation.export_narrative(
                context_id=context_id,
                format=format
            )
            
            result = {
                "narrative": narrative
            }
            
        elif command == "generate_complete_narrative":
            # Generate a complete narrative with all components
            title = params.get("title")
            premise = params.get("premise")
            complexity = params.get("complexity", 5)
            
            # Create narrative context
            context = theory_fiction_simulation.generator.generate_narrative_context(
                title=title,
                core_premise=premise
            )
            
            # Register it
            theory_fiction_simulation.narrative_contexts[context.id] = context
            
            # Generate theoretical framework
            framework = theory_fiction_simulation.generate_theoretical_framework()
            
            # Generate speculative scenario
            theory_fiction_simulation.generator.generate_speculative_scenario(
                context=context,
                complexity=complexity
            )
            
            # Apply theoretical framework
            theory_fiction_simulation.apply_theoretical_framework(
                context_id=context.id,
                framework_id=framework.id
            )
            
            # Resolve narrative
            theory_fiction_simulation.resolve_narrative(context.id)
            
            # Extract key information for result
            characters = []
            for char in context.characters.values():
                characters.append({
                    "id": char.id,
                    "name": char.name,
                    "background": char.background,
                    "motivations": char.motivations
                })
            
            events = []
            for event in context.events:
                events.append({
                    "id": event.id,
                    "description": event.description,
                    "category": event.category.name,
                    "occurred": event.occurred
                })
            
            result = {
                "context_id": context.id,
                "title": context.title,
                "core_premise": context.core_premise,
                "theoretical_lens": context.theoretical_lens,
                "mode": context.mode.name,
                "character_count": len(characters),
                "event_count": len(events),
                "characters": characters,
                "events": events,
                "synopsis": context.generate_synopsis()
            }
            
        elif command == "generate_theory_fiction_response":
            user_message = params.get("user_message", "")
            conversation_context = params.get("conversation_context", {})
            
            # Generate a response using theory-fiction concepts
            # This is a placeholder implementation - in a real implementation,
            # this would use more sophisticated NLP techniques to generate
            # a response based on the theory fiction simulation
            
            # Try to extract themes from the user message
            lower_message = user_message.lower()
            
            # Check for theoretical domains
            themes = []
            domain_keywords = {
                "posthuman": "POSTHUMANISM",
                "technology": "POSTHUMANISM",
                "acceleration": "ACCELERATION",
                "speed": "ACCELERATION",
                "systems": "SYSTEMS_THEORY",
                "cybernetics": "CYBERNETICS",
                "feedback": "CYBERNETICS",
                "psychology": "PSYCHOANALYSIS",
                "mind": "PSYCHOANALYSIS",
                "experience": "PHENOMENOLOGY",
                "consciousness": "PHENOMENOLOGY",
                "material": "NEW_MATERIALISM",
                "matter": "NEW_MATERIALISM",
                "object": "OBJECT_ORIENTED_ONTOLOGY",
                "things": "OBJECT_ORIENTED_ONTOLOGY",
                "gender": "XENOFEMINISM",
                "feminism": "XENOFEMINISM",
                "colonial": "DECOLONIAL_THEORY",
                "decolonial": "DECOLONIAL_THEORY",
                "media": "MEDIA_ARCHAEOLOGY",
                "archive": "MEDIA_ARCHAEOLOGY",
                "complex": "SYSTEMS_THEORY",
                "emergence": "SYSTEMS_THEORY",
                "language": "DECONSTRUCTION",
                "text": "DECONSTRUCTION"
            }
            
            for keyword, domain in domain_keywords.items():
                if keyword in lower_message:
                    themes.append(domain)
            
            # If no themes detected, pick a random domain
            if not themes:
                from theory_fiction_module import TheoreticalDomain
                themes = [random.choice(list(TheoreticalDomain)).name]
            
            # Generate a response based on detected themes
            # Create a mini narrative framework for the response
            
            # Generate theoretical concept
            framework_name = None
            if themes:
                theme = themes[0]
                if theme == "POSTHUMANISM":
                    framework_name = "Posthumanist Perspective"
                    concept_content = "exploration of human-technology boundaries and how they reshape subjectivity"
                elif theme == "ACCELERATION":
                    framework_name = "Accelerationist Theory"
                    concept_content = "analysis of how technological and social acceleration transforms experience"
                elif theme == "CYBERNETICS":
                    framework_name = "Cybernetic Analysis"
                    concept_content = "application of feedback loops and systems thinking to human-machine relations"
                elif theme == "PSYCHOANALYSIS":
                    framework_name = "Psychoanalytic Interpretation"
                    concept_content = "exploration of unconscious processes and desires in technological engagement"
                elif theme == "PHENOMENOLOGY":
                    framework_name = "Phenomenological Approach"
                    concept_content = "focus on lived experience and embodied consciousness in a technological world"
                elif theme == "NEW_MATERIALISM":
                    framework_name = "New Materialist Framework"
                    concept_content = "understanding of matter as vibrant, active, and agential"
                elif theme == "OBJECT_ORIENTED_ONTOLOGY":
                    framework_name = "Object-Oriented Perspective"
                    concept_content = "flat ontology where humans and nonhumans share equal ontological status"
                elif theme == "XENOFEMINISM":
                    framework_name = "Xenofeminist Approach"
                    concept_content = "technology as a site for gender experimentation and emancipation"
                elif theme == "DECOLONIAL_THEORY":
                    framework_name = "Decolonial Reading"
                    concept_content = "challenge to colonial underpinnings of technological systems"
                elif theme == "MEDIA_ARCHAEOLOGY":
                    framework_name = "Media Archaeological Inquiry"
                    concept_content = "excavation of hidden histories and alternate genealogies of media"
                elif theme == "SYSTEMS_THEORY":
                    framework_name = "Complex Systems Analysis"
                    concept_content = "examination of emergent properties and self-organization in social-technical systems"
                else:  # DECONSTRUCTION
                    framework_name = "Deconstructive Reading"
                    concept_content = "interrogation of binaries and unstable meanings in technological discourse"
            
            # Generate response with theoretical framing
            if framework_name:
                # Create a response that incorporates theoretical framing
                response = f"From a {framework_name.lower()}, your question involves a {concept_content}. "
                
                # Add speculative elements based on the theme
                if "POSTHUMANISM" in themes:
                    response += "As we extend beyond traditional human limitations through technology, "
                    response += "we must reconsider what it means to be human in an era of technological integration. "
                    response += "The boundaries between human and machine become increasingly porous, "
                    response += "creating new hybrid identities and forms of embodiment."
                    
                elif "ACCELERATION" in themes:
                    response += "The accelerating pace of technological change produces recursive feedback loops "
                    response += "that transform social structures faster than they can be comprehended. "
                    response += "This acceleration creates a sense of temporal compression where the future "
                    response += "arrives before we've had time to adapt to the present."
                    
                elif "SYSTEMS_THEORY" in themes or "CYBERNETICS" in themes:
                    response += "Complex adaptive systems demonstrate emergent properties that cannot be "
                    response += "reduced to their component parts. These systems operate through feedback loops "
                    response += "that allow for self-regulation and autopoietic processes, creating "
                    response += "new organized complexities from apparent disorder."
                    
                else:
                    # Generic speculative response for other themes
                    response += "This perspective invites us to reconsider conventional understandings "
                    response += "and imagine alternative configurations of knowledge, power, and being. "
                    response += "By exploring speculative scenarios that extend from these theoretical premises, "
                    response += "we can develop new conceptual frameworks for engaging with emerging realities."
            else:
                # Generic theory-fiction response
                response = "Your question invites a speculative theoretical exploration. "
                response += "By examining this through multiple theoretical lenses, we can uncover "
                response += "hidden assumptions and imagine alternative possibilities that "
                response += "transcend conventional frameworks of understanding."
            
            result = {
                "response": response,
                "themes": themes,
                "framework": framework_name
            }
        
        else:
            result = {"error": f"Unknown command: {command}"}
            
        return json.dumps(result)
    except Exception as e:
        traceback_str = traceback.format_exc()
        return json.dumps({"error": str(e), "traceback": traceback_str})

# Initialize the module when imported
init_simulation()

# Example usage
if __name__ == "__main__":
    # Test creating a narrative context
    context_params = json.dumps({
        "title": "Technological Singularity",
        "core_premise": "Exploring the implications of exponential technological growth"
    })
    context_result = execute_theory_fiction("create_narrative_context", context_params)
    context_data = json.loads(context_result)
    context_id = context_data["context_id"]
    
    # Generate a speculative scenario
    scenario_params = json.dumps({
        "context_id": context_id,
        "complexity": 5
    })
    scenario_result = execute_theory_fiction("generate_speculative_scenario", scenario_params)
    print(f"Generated scenario: {scenario_result}")
    
    # Export the narrative
    export_params = json.dumps({
        "context_id": context_id,
        "format": "text"
    })
    export_result = execute_theory_fiction("export_narrative", export_params)
    export_data = json.loads(export_result)
    print(f"Narrative: {export_data['narrative']
