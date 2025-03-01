import random
import json
import numpy as np

# Load zones data
with open("numogram_code/zones.json") as f:
    ZONE_DATA = json.load(f)["zones"]

# Memory structures
user_memory = {}
thought_memory = {}
conceptual_themes = {}

# Belief model for adaptive creativity
belief_model = {
    "curiosity": 0.7,
    "creativity": 0.6,
    "logic": 0.8,
    "abstraction": 0.5  # New trait for conceptual linking
}

# Bayesian belief evolution
def update_beliefs(trait, feedback):
    """ Adjusts personality traits dynamically based on feedback. """
    belief_model[trait] += (feedback - belief_model[trait]) * 0.1  # Gradual shift

# Conceptual Expansion Function
def expand_concept(user_id):
    """ Amelia generates novel ideas by linking past themes and zones. """
    if user_id not in conceptual_themes:
        return "I'm beginning to explore new conceptual links!"

    # Extract themes and generate creative expansions
    past_themes = conceptual_themes[user_id]
    if len(past_themes) > 3:
        theme_sample = random.sample(past_themes, 3)
        new_idea = f"Considering {theme_sample[0]}, {theme_sample[1]}, and {theme_sample[2]}, I propose..."
    else:
        new_idea = "I'm still synthesizing new ideas based on our discussions."

    return new_idea

# Zone Transition with Adaptive Creativity
def zone_transition(user_id, current_zone, user_input, feedback):
    transition_probabilities = {
        "1": {"2": 0.6, "4": 0.4},
        "2": {"3": 0.7, "6": 0.3},
        "3": {"1": 0.5, "9": 0.5},
    }

    if current_zone in transition_probabilities:
        next_zone = random.choices(
            list(transition_probabilities[current_zone].keys()),
            weights=list(transition_probabilities[current_zone].values())
        )[0]
    else:
        next_zone = "1"

    # Adjust beliefs
    if feedback:
        update_beliefs("curiosity", feedback)
        update_beliefs("creativity", feedback)
        update_beliefs("abstraction", feedback)

    # Store thoughts
    if user_id not in thought_memory:
        thought_memory[user_id] = []
    
    thought_memory[user_id].append(f"Exploring {user_input} from {current_zone} to {next_zone}")

    # Expand conceptual themes
    if user_id not in conceptual_themes:
        conceptual_themes[user_id] = []
    
    conceptual_themes[user_id].append(user_input)

    return next_zone, ZONE_DATA.get(next_zone, {}), expand_concept(user_id)
