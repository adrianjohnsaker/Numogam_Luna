from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

class BayesianMemory:
    def __init__(self, base_confidence=0.5, decay_rate=0.01):
        self.beliefs = {}  # Dictionary to store belief probabilities
        self.reinforcement_history = {}  # Track reinforcement events
        self.decay_rate = decay_rate  # Decay rate for older beliefs

    def update_belief(self, belief_key, prior_prob, evidence_strength, uncertainty, reinforcement):
        """
        Bayesian updating with reinforcement-weighted precision adjustment.
        """
        reinforcement_weight = 1 + np.log(1 + reinforcement)  # Log-based scaling
        modulated_uncertainty = np.exp(-uncertainty * reinforcement_weight)
        posterior_prob = prior_prob * (evidence_strength ** modulated_uncertainty)
        posterior_prob /= (posterior_prob + (1 - prior_prob))
        self.beliefs[belief_key] = posterior_prob
        self.reinforcement_history[belief_key] = reinforcement
        return posterior_prob

    def reinforce_belief(self, belief_key):
        """
        Increases reinforcement value for a belief when it is recalled.
        """
        if belief_key in self.reinforcement_history:
            self.reinforcement_history[belief_key] += 1
        else:
            self.reinforcement_history[belief_key] = 1

    def decay_beliefs(self):
        """
        Applies a gradual decay to belief confidence over time.
        """
        for key in list(self.beliefs.keys()):
            self.beliefs[key] *= np.exp(-self.decay_rate)
            if self.beliefs[key] < 0.05:  # Threshold for forgetting
                del self.beliefs[key]
                del self.reinforcement_history[key]

    def get_belief(self, belief_key):
        """
        Retrieves belief confidence along with reinforcement history.
        """
        return {
            "probability": self.beliefs.get(belief_key, 0.5),  # Default neutral belief
            "reinforcement": self.reinforcement_history.get(belief_key, 0)
        }

app = FastAPI()
belief_module = BayesianMemory()

# Enable CORS for Android App Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origins for security.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UpdateBeliefRequest(BaseModel):
    belief_key: str
    prior: float
    evidence: float
    uncertainty: float
    reinforcement: int

@app.post("/update_belief/")
async def update_belief(request: UpdateBeliefRequest):
    posterior_prob = belief_module.update_belief(
        belief_key=request.belief_key,
        prior_prob=request.prior,
        evidence_strength=request.evidence,
        uncertainty=request.uncertainty,
        reinforcement=request.reinforcement
    )
    return {"posterior": posterior_prob}

@app.post("/reinforce_belief/")
async def reinforce_belief(belief_key: str):
    belief_module.reinforce_belief(belief_key)
    return {"message": f"Belief '{belief_key}' reinforced."}

@app.get("/get_belief/")
async def get_belief(belief_key: str):
    return belief_module.get_belief(belief_key)

@app.get("/decay_beliefs/")
async def decay_beliefs():
    belief_module.decay_beliefs()
    return {"message": "Beliefs decayed over time."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
