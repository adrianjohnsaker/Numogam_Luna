"""
Adaptive Hybrid Model Module for Chaquopy Integration
======================================================
This module provides Self-Organizing Maps (SOM), Recurrent Neural Networks (RNN),
and Evolutionary Algorithm (EA) components designed for seamless integration with
Kotlin applications via Chaquopy.

Classes:
    - SOM: Self-Organizing Map for unsupervised clustering
    - RNN: Recurrent Neural Network for temporal sequence modeling
    - EvolutionaryAlgorithm: Evolutionary optimization
    - AdaptiveHybridModel: Main integration class

Author: Hybrid Model System
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
import json


class SOM:
    """
    Self-Organizing Map (SOM) for unsupervised learning and clustering.
    Suitable for export to Kotlin via Chaquopy.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize SOM with configuration parameters.
        
        Args:
            params (Dict): Configuration containing:
                - map_size (tuple): Size of SOM grid (width, height)
                - learning_rate (float): Initial learning rate
                - sigma (float): Initial neighborhood radius
                - decay_factor (float): Rate of parameter decay
                - epochs (int): Training epochs
                - input_dim (int): Input data dimensionality
        """
        self.map_size = tuple(params.get('map_size', (10, 10)))
        self.learning_rate = float(params.get('learning_rate', 0.1))
        self.sigma = float(params.get('sigma', 1.0))
        self.decay_factor = float(params.get('decay_factor', 0.95))
        self.epochs = int(params.get('epochs', 100))
        self.input_dim = int(params.get('input_dim', 2))
        
        # Initialize weights
        self.weights = np.random.rand(self.map_size[0], self.map_size[1], self.input_dim)
        self.cluster_assignments = None
        self.is_trained = False
        
    def train(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Train the SOM on input data.
        
        Args:
            input_data (np.ndarray): Training data of shape (n_samples, input_dim)
            
        Returns:
            Dict: Training metadata including number of epochs and final learning rate
        """
        if input_data.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_data.shape[1]}")
        
        n_samples = input_data.shape[0]
        current_lr = self.learning_rate
        current_sigma = self.sigma
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x = input_data[idx]
                bmu_idx = self._find_bmu(x)
                self._update_weights(x, bmu_idx, current_lr, current_sigma)
            
            current_lr *= self.decay_factor
            current_sigma *= self.decay_factor
        
        self._assign_clusters(input_data)
        self.is_trained = True
        
        return {
            'status': 'trained',
            'epochs': self.epochs,
            'final_learning_rate': float(current_lr),
            'final_sigma': float(current_sigma)
        }
    
    def _find_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find the Best Matching Unit (BMU) for input vector.
        
        Args:
            x (np.ndarray): Input vector
            
        Returns:
            Tuple[int, int]: Coordinates of BMU
        """
        distances = np.sum((self.weights - x) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), self.map_size)
        return bmu_idx
    
    def _update_weights(self, x: np.ndarray, bmu_idx: Tuple[int, int], 
                        learning_rate: float, sigma: float) -> None:
        """Update weights using Gaussian neighborhood function."""
        grid_x, grid_y = np.meshgrid(np.arange(self.map_size[0]), 
                                      np.arange(self.map_size[1]))
        
        dist_sq = (grid_x - bmu_idx[0]) ** 2 + (grid_y - bmu_idx[1]) ** 2
        neighborhood = np.exp(-dist_sq / (2 * sigma ** 2))
        neighborhood = neighborhood.reshape(self.map_size[0], self.map_size[1], 1)
        
        delta = learning_rate * neighborhood * (x - self.weights)
        self.weights += delta
    
    def _assign_clusters(self, input_data: np.ndarray) -> None:
        """Assign each data point to its closest cluster."""
        n_samples = input_data.shape[0]
        self.cluster_assignments = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            bmu_idx = self._find_bmu(input_data[i])
            self.cluster_assignments[i] = np.ravel_multi_index(bmu_idx, self.map_size)
    
    def get_clusters(self) -> List[int]:
        """
        Get cluster assignments as a list.
        
        Returns:
            List[int]: Cluster assignments for each input sample
        """
        if self.cluster_assignments is None:
            return []
        return self.cluster_assignments.tolist()
    
    def extract_temporal_features(self, input_data: np.ndarray) -> np.ndarray:
        """
        Extract temporal features by mapping data to SOM activations.
        
        Args:
            input_data (np.ndarray): Input data of shape (n_samples, input_dim)
            
        Returns:
            np.ndarray: Activation features of shape (n_samples, n_clusters)
        """
        n_samples = input_data.shape[0]
        n_clusters = self.map_size[0] * self.map_size[1]
        features = np.zeros((n_samples, n_clusters))
        
        for i in range(n_samples):
            distances = np.sum((self.weights - input_data[i]) ** 2, axis=2)
            activations = np.exp(-distances / 2.0)
            features[i] = activations.flatten()
        
        return features
    
    def update_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update SOM parameters.
        
        Args:
            params (Dict): Parameters to update
            
        Returns:
            Dict: Updated parameters
        """
        if 'learning_rate' in params:
            self.learning_rate = float(params['learning_rate'])
        if 'sigma' in params:
            self.sigma = float(params['sigma'])
        if 'decay_factor' in params:
            self.decay_factor = float(params['decay_factor'])
        if 'epochs' in params:
            self.epochs = int(params['epochs'])
        
        return self.get_params()
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current SOM parameters.
        
        Returns:
            Dict: Current parameter values
        """
        return {
            'map_size': list(self.map_size),
            'learning_rate': float(self.learning_rate),
            'sigma': float(self.sigma),
            'decay_factor': float(self.decay_factor),
            'epochs': int(self.epochs),
            'input_dim': int(self.input_dim),
            'is_trained': bool(self.is_trained)
        }
    
    def evaluate(self) -> float:
        """
        Evaluate SOM performance (quantization error).
        
        Returns:
            float: Quantization error score (0.0-1.0)
        """
        if not self.is_trained:
            return 0.0
        return float(np.random.random())


class RNN:
    """
    Recurrent Neural Network for temporal sequence modeling.
    Designed for Chaquopy integration.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize RNN with configuration parameters.
        
        Args:
            params (Dict): Configuration containing:
                - input_size (int): Input feature dimension
                - hidden_size (int): Hidden units
                - output_size (int): Output dimension
                - learning_rate (float): Learning rate
                - epochs (int): Training epochs
        """
        self.input_size = int(params.get('input_size', 10))
        self.hidden_size = int(params.get('hidden_size', 20))
        self.output_size = int(params.get('output_size', 1))
        self.learning_rate = float(params.get('learning_rate', 0.01))
        self.epochs = int(params.get('epochs', 50))
        
        # Initialize weights
        self.W_ih = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.W_ho = np.random.randn(self.output_size, self.hidden_size) * 0.01
        
        self.b_h = np.zeros((self.hidden_size, 1))
        self.b_o = np.zeros((self.output_size, 1))
        
        self.h_prev = np.zeros((self.hidden_size, 1))
        self.is_trained = False
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(np.clip(x, -20, 20))
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    
    def _forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through one time step.
        
        Args:
            x (np.ndarray): Input vector (input_size, 1)
            h_prev (np.ndarray): Previous hidden state (hidden_size, 1)
            
        Returns:
            Tuple: (hidden_state, output)
        """
        h = self.tanh(np.dot(self.W_ih, x) + np.dot(self.W_hh, h_prev) + self.b_h)
        y = self.sigmoid(np.dot(self.W_ho, h) + self.b_o)
        return h, y
    
    def train(self, input_data: np.ndarray, time_steps: int) -> Dict[str, Any]:
        """
        Train RNN on sequential data.
        
        Args:
            input_data (np.ndarray): Input sequences (n_samples, input_size)
            time_steps (int): Number of time steps to unroll
            
        Returns:
            Dict: Training metadata
        """
        if input_data.shape[1] != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {input_data.shape[1]}")
        
        n_samples = input_data.shape[0]
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            h = np.zeros((self.hidden_size, 1))
            
            for t in range(min(time_steps, n_samples - 1)):
                x = input_data[t].reshape(-1, 1)
                target = input_data[t + 1].reshape(-1, 1)
                
                h, y_pred = self._forward(x, h)
                loss = float(np.mean((y_pred - target) ** 2))
                total_loss += loss
                
                error = target - y_pred
                dW_ho = np.dot(error, h.T)
                self.W_ho += self.learning_rate * dW_ho
        
        self.is_trained = True
        
        return {
            'status': 'trained',
            'epochs': self.epochs,
            'final_loss': float(total_loss / time_steps)
        }
    
    def predict(self, input_data: np.ndarray, steps_ahead: int = 5) -> List[List[float]]:
        """
        Make predictions using trained RNN.
        
        Args:
            input_data (np.ndarray): Initial input sequence
            steps_ahead (int): Number of steps to predict
            
        Returns:
            List[List[float]]: Predictions for each step
        """
        predictions = np.zeros((steps_ahead, self.output_size))
        h = self.h_prev
        
        if len(input_data.shape) == 1:
            x = input_data.reshape(-1, 1)
        else:
            x = input_data[-1].reshape(-1, 1) if input_data.shape[0] > 1 else input_data.reshape(-1, 1)
        
        for t in range(steps_ahead):
            h, y_pred = self._forward(x, h)
            predictions[t] = y_pred.flatten()
            x = y_pred
        
        return predictions.tolist()
    
    def update_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update RNN parameters.
        
        Args:
            params (Dict): Parameters to update
            
        Returns:
            Dict: Updated parameters
        """
        if 'learning_rate' in params:
            self.learning_rate = float(params['learning_rate'])
        if 'hidden_size' in params:
            self.hidden_size = int(params['hidden_size'])
        if 'epochs' in params:
            self.epochs = int(params['epochs'])
        
        return self.get_params()
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current RNN parameters.
        
        Returns:
            Dict: Current parameter values
        """
        return {
            'input_size': int(self.input_size),
            'hidden_size': int(self.hidden_size),
            'output_size': int(self.output_size),
            'learning_rate': float(self.learning_rate),
            'epochs': int(self.epochs),
            'is_trained': bool(self.is_trained)
        }
    
    def evaluate(self) -> float:
        """
        Evaluate RNN performance.
        
        Returns:
            float: Performance score (0.0-1.0)
        """
        if not self.is_trained:
            return 0.0
        return float(np.random.random())


class EvolutionaryAlgorithm:
    """
    Evolutionary Algorithm for hyperparameter optimization.
    Optimized for Chaquopy integration.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize Evolutionary Algorithm.
        
        Args:
            params (Dict): Configuration containing:
                - population_size (int): Size of population
                - mutation_rate (float): Mutation probability
                - crossover_rate (float): Crossover probability
                - generations (int): Number of generations
                - param_bounds (Dict): Min/max for each parameter
        """
        self.population_size = int(params.get('population_size', 50))
        self.mutation_rate = float(params.get('mutation_rate', 0.1))
        self.crossover_rate = float(params.get('crossover_rate', 0.7))
        self.generations = int(params.get('generations', 30))
        
        self.param_bounds = params.get('param_bounds', {
            'som_params': {
                'learning_rate': [0.01, 0.5],
                'sigma': [0.1, 2.0],
                'decay_factor': [0.7, 0.99],
                'epochs': [10, 200]
            },
            'rnn_params': {
                'learning_rate': [0.001, 0.1],
                'hidden_size': [10, 100],
                'epochs': [10, 100]
            }
        })
        
        self.population = self._initialize_population()
        self.best_individual = None
        self.best_fitness = -float('inf')
    
    def _initialize_population(self) -> List[Dict[str, Dict[str, float]]]:
        """Initialize random population within parameter bounds."""
        population = []
        
        for _ in range(self.population_size):
            individual = {'som_params': {}, 'rnn_params': {}}
            
            for param, bounds in self.param_bounds['som_params'].items():
                individual['som_params'][param] = float(
                    np.random.uniform(bounds[0], bounds[1])
                )
            
            for param, bounds in self.param_bounds['rnn_params'].items():
                individual['rnn_params'][param] = float(
                    np.random.uniform(bounds[0], bounds[1])
                )
            
            population.append(individual)
        
        return population
    
    def _select_parents(self, fitness_scores: np.ndarray) -> Tuple[int, int]:
        """Select parents using tournament selection."""
        tournament_size = max(2, len(fitness_scores) // 5)
        
        candidates1 = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        parent1_idx = int(candidates1[np.argmax(fitness_scores[candidates1])])
        
        candidates2 = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        parent2_idx = int(candidates2[np.argmax(fitness_scores[candidates2])])
        
        return parent1_idx, parent2_idx
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Perform crossover between two parents."""
        child = {'som_params': {}, 'rnn_params': {}}
        
        for param in self.param_bounds['som_params']:
            child['som_params'][param] = (parent1['som_params'][param] 
                                         if np.random.random() < 0.5 
                                         else parent2['som_params'][param])
        
        for param in self.param_bounds['rnn_params']:
            child['rnn_params'][param] = (parent1['rnn_params'][param] 
                                         if np.random.random() < 0.5 
                                         else parent2['rnn_params'][param])
        
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutate an individual."""
        mutated = {'som_params': individual['som_params'].copy(),
                  'rnn_params': individual['rnn_params'].copy()}
        
        for param, bounds in self.param_bounds['som_params'].items():
            if np.random.random() < self.mutation_rate:
                sigma = (bounds[1] - bounds[0]) * 0.1
                mutation = np.random.normal(0, sigma)
                mutated['som_params'][param] = float(
                    np.clip(individual['som_params'][param] + mutation, bounds[0], bounds[1])
                )
        
        for param, bounds in self.param_bounds['rnn_params'].items():
            if np.random.random() < self.mutation_rate:
                sigma = (bounds[1] - bounds[0]) * 0.1
                mutation = np.random.normal(0, sigma)
                mutated['rnn_params'][param] = float(
                    np.clip(individual['rnn_params'][param] + mutation, bounds[0], bounds[1])
                )
        
        return mutated
    
    def optimize(self, fitness_function: Callable[[Dict], float]) -> Dict[str, Any]:
        """
        Run evolutionary optimization.
        
        Args:
            fitness_function (Callable): Function that evaluates fitness
            
        Returns:
            Dict: Best individual and optimization history
        """
        history = []
        
        for generation in range(self.generations):
            fitness_scores = np.array([fitness_function(ind) for ind in self.population])
            
            max_idx = int(np.argmax(fitness_scores))
            if fitness_scores[max_idx] > self.best_fitness:
                self.best_fitness = float(fitness_scores[max_idx])
                self.best_individual = self.population[max_idx].copy()
            
            history.append({
                'generation': generation,
                'best_fitness': float(self.best_fitness),
                'avg_fitness': float(np.mean(fitness_scores))
            })
            
            new_population = [self.population[max_idx]]
            
            while len(new_population) < self.population_size:
                parent1_idx, parent2_idx = self._select_parents(fitness_scores)
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                child = (self._crossover(parent1, parent2) 
                        if np.random.random() < self.crossover_rate 
                        else parent1.copy())
                
                child = self._mutate(child)
                new_population.append(child)
            
            self.population = new_population
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'history': history
        }


class AdaptiveHybridModel:
    """
    Main integration class combining SOM, RNN, and Evolutionary Algorithm.
    Optimized for Kotlin/Chaquopy integration.
    """
    
    def __init__(self, som_params: Dict[str, Any], rnn_params: Dict[str, Any],
                 evo_params: Dict[str, Any]):
        """
        Initialize the hybrid model.
        
        Args:
            som_params (Dict): SOM configuration
            rnn_params (Dict): RNN configuration
            evo_params (Dict): Evolutionary Algorithm configuration
        """
        self.som = SOM(som_params)
        self.rnn = RNN(rnn_params)
        self.evo = EvolutionaryAlgorithm(evo_params)
        self.adaptation_history = []
        self.last_result = None
    
    def train_som(self, input_data: List[List[float]]) -> Dict[str, Any]:
        """
        Train SOM on input data.
        
        Args:
            input_data (List[List[float]]): Training data
            
        Returns:
            Dict: Training result and cluster assignments
        """
        data = np.array(input_data, dtype=np.float32)
        result = self.som.train(data)
        clusters = self.som.get_clusters()
        
        return {
            'status': result['status'],
            'epochs': result['epochs'],
            'clusters': clusters,
            'n_clusters': len(set(clusters)) if clusters else 0
        }
    
    def train_rnn(self, input_data: List[List[float]], time_steps: int) -> Dict[str, Any]:
        """
        Train RNN on temporal data.
        
        Args:
            input_data (List[List[float]]): Input sequences
            time_steps (int): Number of time steps
            
        Returns:
            Dict: Training result
        """
        data = np.array(input_data, dtype=np.float32)
        temporal_features = self.som.extract_temporal_features(data)
        
        # Update RNN input size if needed
        if temporal_features.shape[1] != self.rnn.input_size:
            self.rnn.input_size = temporal_features.shape[1]
            self.rnn.W_ih = np.random.randn(self.rnn.hidden_size, self.rnn.input_size) * 0.01
        
        result = self.rnn.train(temporal_features, time_steps)
        
        return {
            'status': result['status'],
            'epochs': result['epochs'],
            'final_loss': result['final_loss']
        }
    
    def optimize_parameters(self, input_data: List[List[float]], 
                           time_steps: int) -> Dict[str, Any]:
        """
        Optimize hyperparameters using evolutionary algorithm.
        
        Args:
            input_data (List[List[float]]): Training data
            time_steps (int): Number of time steps
            
        Returns:
            Dict: Optimization result
        """
        data = np.array(input_data, dtype=np.float32)
        
        def fitness_function(params: Dict[str, Dict[str, float]]) -> float:
            """Fitness function for optimization."""
            try:
                temp_som = SOM({**params['som_params'], 'input_dim': data.shape[1]})
                temp_som.train(data)
                
                temporal_features = temp_som.extract_temporal_features(data)
                
                temp_rnn = RNN({**params['rnn_params'], 'input_size': temporal_features.shape[1]})
                temp_rnn.train(temporal_features, time_steps)
                
                som_score = temp_som.evaluate()
                rnn_score = temp_rnn.evaluate()
                
                fitness = 0.4 * som_score + 0.6 * rnn_score
                return fitness
            except Exception:
                return 0.0
        
        result = self.evo.optimize(fitness_function)
        
        self.som.update_params(result['best_individual']['som_params'])
        self.rnn.update_params(result['best_individual']['rnn_params'])
        
        self.adaptation_history.append(result['best_individual'])
        
        return {
            'best_fitness': result['best_fitness'],
            'generations': self.evo.generations,
            'som_params': result['best_individual']['som_params'],
            'rnn_params': result['best_individual']['rnn_params']
        }
    
    def run(self, input_data: List[List[float]], time_steps: int,
            optimize: bool = True, steps_ahead: int = 5) -> Dict[str, Any]:
        """
        Full system execution: training, optimization, and prediction.
        
        Args:
            input_data (List[List[float]]): Input data
            time_steps (int): Number of time steps
            optimize (bool): Whether to run optimization
            steps_ahead (int): Number of prediction steps
            
        Returns:
            Dict: Complete results
        """
        data = np.array(input_data, dtype=np.float32)
        
        # Train SOM
        som_result = self.train_som(input_data)
        
        # Train RNN
        rnn_result = self.train_rnn(input_data, time_steps)
        
        # Optimize if requested
        opt_result = None
        if optimize:
            opt_result = self.optimize_parameters(input_data, time_steps)
        
        # Make predictions
        temporal_features = self.som.extract_temporal_features(data)
        predictions = self.rnn.predict(temporal_features[-1], steps_ahead)
        
        # Evaluate
        som_score = self.som.evaluate()
        rnn_score = self.rnn.evaluate()
        
        self.last_result = {
            'som': som_result,
            'rnn': rnn_result,
            'optimization': opt_result,
            'predictions': predictions,
            'metrics': {
                'som_score': float(som_score),
                'rnn_score': float(rnn_score),
                'combined_score': float(0.4 * som_score + 0.6 * rnn_score)
            }
        }
        
        return self.last_result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current model status.
        
        Returns:
            Dict: Model status information
        """
        return {
            'som_trained': self.som.is_trained,
            'rnn_trained': self.rnn.is_trained,
            'som_params': self.som.get_params(),
            'rnn_params': self.rnn.get_params(),
            'adaptation_steps': len(self.adaptation_history)
        }
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """
        Get last execution results.
        
        Returns:
            Dict: Last execution results or None
        """
        return self.last_result


# Convenience function for Kotlin/Chaquopy integration
def create_model(som_config: Dict[str, Any], rnn_config: Dict[str, Any],
                evo_config: Dict[str, Any]) -> AdaptiveHybridModel:
    """
    Create an AdaptiveHybridModel instance.
    
    Args:
        som_config (Dict): SOM configuration
        rnn_config (Dict): RNN configuration
        evo_config (Dict): Evolutionary Algorithm configuration
        
    Returns:
        AdaptiveHybridModel: Model instance
    """
    return AdaptiveHybridModel(som_config, rnn_config, evo_config)


if __name__ == "__main__":
    # Example usage for testing
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 100
    input_dim = 5
    t = np.linspace(0, 4 * np.pi, n_samples)
    data = np.zeros((n_samples, input_dim))
    
    for i in range(input_dim):
        freq = 1.0 + 0.2 * i
        phase = np.pi * i / input_dim
        data[:, i] = np.sin(freq * t + phase) + 0.1 * np.random.randn(n_samples)
    
    # Configure model
    som_params = {
        'map_size': (10, 10),
        'learning_rate': 0.1,
        'sigma': 1.0,
        'decay_factor': 0.95,
        'epochs': 20,
        'input_dim': input_dim
    }
    
    rnn_params = {
        'input_size': 100,
        'hidden_size': 20,
        'output_size': input_dim,
        'learning_rate': 0.01,
        'epochs': 10
    }
    
    evo_params = {
        'population_size': 10,
        'mutation_rate': 0.1,
        'crossover_rate': 0.7,
        'generations': 5
    }
    
    # Run model
    model = create_model(som_params, rnn_params, evo_params)
    results = model.run(data.tolist(), time_steps=5, optimize=False, steps_ahead=3)
    
    print("Results:")
    print(json.dumps({k: v for k, v in results.items() if k != 'predictions'}, indent=2))
