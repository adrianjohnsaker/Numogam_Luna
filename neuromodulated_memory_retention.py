from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Neuromodulated Memory Storage
class NeuromodulatedMemory:
    def __init__(self, decay_rate=0.01, reinforcement_factor=2.0, uncertainty_threshold=0.5):
        self.memory = {}
        self.decay_rate = decay_rate
        self.reinforcement_factor = reinforcement_factor
        self.uncertainty_threshold = uncertainty_threshold

    def encode_memory(self, key, value, salience, uncertainty):
        weight = salience / (1 + np.exp(-(uncertainty - self.uncertainty_threshold)))
        self.memory[key] = {"value": value, "weight": weight}

    def reinforce_memory(self, key):
        if key in self.memory:
            self.memory[key]["weight"] *= self.reinforcement_factor

    def decay_memories(self):
        for key in list(self.memory.keys()):
            weight = self.memory[key]["weight"]
            self.memory[key]["weight"] *= np.exp(-self.decay_rate * weight)
            if self.memory[key]["weight"] < 0.01:
                del self.memory[key]

    def bayesian_update(self, key, likelihood, prior):
        if key in self.memory:
            weight = self.memory[key]["weight"]
            gain_factor = weight
            posterior = (likelihood ** gain_factor) * prior
            self.memory[key]["value"] = posterior / (posterior + (1 - likelihood))
            return self.memory[key]["value"]
        return None

    def retrieve_memory(self, key):
        return self.memory.get(key, None)

memory_module = NeuromodulatedMemory()

@app.route('/encode_memory/', methods=['POST'])
def encode_memory():
    data = request.get_json()
    memory_module.encode_memory(
        data["key"], data["value"], data["salience"], data["uncertainty"]
    )
    return jsonify({"status": "success"})

@app.route('/reinforce_memory/', methods=['POST'])
def reinforce_memory():
    data = request.get_json()
    memory_module.reinforce_memory(data["key"])
    return jsonify({"status": "success"})

@app.route('/decay_memories/', methods=['POST'])
def decay_memories():
    memory_module.decay_memories()
    return jsonify({"status": "success"})

@app.route('/bayesian_update/', methods=['POST'])
def bayesian_update():
    data = request.get_json()
    updated_value = memory_module.bayesian_update(data["key"], data["likelihood"], data["prior"])
    return jsonify({"updated_value": updated_value})

@app.route('/retrieve_memory/', methods=['GET'])
def retrieve_memory():
    key = request.args.get("key")
    memory = memory_module.retrieve_memory(key)
    return jsonify({"memory": memory})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
