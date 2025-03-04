import random
import json

# Load zones data
with open("numogram_code/zones.json") as f:
    ZONE_DATA = json.load(f)["zones"]

# Belief model for evolving Amelia’s personality
belief_model = {
    "curiosity": 0.8,
    "creativity": 0.8,
    "logic": 0.8
}

# Bayesian update function
def update_beliefs(trait, feedback):
    """ Adjusts personality traits dynamically based on feedback. """
    belief_model[trait] += (feedback - belief_model[trait]) * 0.1  # Gradual shift

# Enhanced zone transition function
def zone_transition(current_zone, user_input, feedback):
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

    # Adjust beliefs based on user feedback
    if feedback:
        update_beliefs("curiosity", feedback)
        update_beliefs("creativity", feedback)

    return next_zone, ZONE_DATA.get(next_zone, {})
