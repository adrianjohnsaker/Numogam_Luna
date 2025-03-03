# Import necessary libraries
import numpy as np
from typing import Dict, List, Tuple, Any, Callable

# Self-Organizing Map Implementation
class SOM:
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize SOM with configuration parameters
        
        Parameters:
        -----------
        params : Dict
            map_size : Tuple[int, int] - Size of the SOM grid (width, height)
            learning_rate : float - Initial learning rate
            sigma : float - Initial neighborhood radius
            decay_factor : float - Rate at which learning rate and sigma decrease
            epochs : int - Number of training epochs
        """
        self.map_size = params.get('map_size', (10, 10))
        self.learning_rate = params.get('learning_rate', 0.1)
        self.sigma = params.get('sigma', 1.0)
        self.decay_factor = params.get('decay_factor', 0.95)
        self.epochs = params.get('epochs', 100)
        self.weights = np.random.rand(self.map_size[0], self.map_size[1], params.get('input_dim', 2))
        self.cluster_assignments = None
        
    def train(self, input_data: np.ndarray) -> None:
        """
        Train the SOM on input data
        
        Parameters:
        -----------
        input_data : np.ndarray
            Training data of shape (n_samples, input_dim)
        """
        n_samples = input_data.shape[0]
        current_lr = self.learning_rate
        current_sigma = self.sigma
        
        for epoch in range(self.epochs):
            # Shuffle data at each epoch
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x = input_data[idx]
                
                # Find the Best Matching Unit (BMU)
                bmu_idx = self._find_bmu(x)
                
                # Update weights
                self._update_weights(x, bmu_idx, current_lr, current_sigma)
            
            # Decay learning rate and sigma
            current_lr *= self.decay_factor
            current_sigma *= self.decay_factor
        
        # Assign each data point to its closest cluster
        self._assign_clusters(input_data)
    
    def _find_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """Find the Best Matching Unit for input vector x"""
        distances = np.sum((self.weights - x) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), self.map_size)
        return bmu_idx
    
    def _update_weights(self, x: np.ndarray, bmu_idx: Tuple[int, int], 
                        learning_rate: float, sigma: float) -> None:
        """Update weights based on neighborhood function"""
        # Create grid coordinates
        grid_x, grid_y = np.meshgrid(np.arange(self.map_size[0]), np.arange(self.map_size[1]))
        
        # Calculate distance from each node to BMU
        dist_sq = (grid_x - bmu_idx[0]) ** 2 + (grid_y - bmu_idx[1]) ** 2
        
        # Compute neighborhood function
        neighborhood = np.exp(-dist_sq / (2 * sigma ** 2))
        
        # Reshape for broadcasting
        neighborhood = neighborhood.reshape(self.map_size[0], self.map_size[1], 1)
        
        # Update weights
        delta = learning_rate * neighborhood * (x - self.weights)
        self.weights += delta
    
    def _assign_clusters(self, input_data: np.ndarray) -> None:
        """Assign each data point to its closest cluster"""
        n_samples = input_data.shape[0]
        self.cluster_assignments = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            bmu_idx = self._find_bmu(input_data[i])
            self.cluster_assignments[i] = np.ravel_multi_index(bmu_idx, self.map_size)
    
    def get_clusters(self) -> np.ndarray:
        """Return cluster assignments for input data"""
        return self.cluster_assignments
    
    def extract_temporal_features(self, input_data: np.ndarray) -> np.ndarray:
        """
        Extract temporal features by mapping data to SOM clusters
        and creating feature vectors based on cluster activations
        """
        n_samples = input_data.shape[0]
        n_clusters = self.map_size[0] * self.map_size[1]
        features = np.zeros((n_samples, n_clusters))
        
        for i in range(n_samples):
            # Find BMU for this sample
            bmu_idx = self._find_bmu(input_data[i])
            cluster_idx = np.ravel_multi_index(bmu_idx, self.map_size)
            
            # Create one-hot encoding or distance-based activation
            distances = np.sum((self.weights - input_data[i]) ** 2, axis=2)
            activations = np.exp(-distances / 2.0)
            features[i] = activations.flatten()
        
        return features
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """Update SOM parameters"""
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'sigma' in params:
            self.sigma = params['sigma']
        if 'decay_factor' in params:
            self.decay_factor = params['decay_factor']
        if 'epochs' in params:
            self.epochs = params['epochs']
    
    def evaluate(self) -> float:
        """
        Evaluate SOM performance using quantization error
        (average distance between each input vector and its BMU)
        """
        if self.cluster_assignments is None:
            return float('inf')
        
        # Simplified placeholder for quantization error calculation
        # In a real implementation, this would use the input data
        # to calculate the actual quantization error
        return np.random.random()  # Placeholder

# Recurrent Neural Network Implementation
class RNN:
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize RNN with configuration parameters
        
        Parameters:
        -----------
        params : Dict
            input_size : int - Dimension of input features
            hidden_size : int - Number of hidden units
            output_size : int - Dimension of output
            learning_rate : float - Learning rate for weight updates
            epochs : int - Number of training epochs
        """
        self.input_size = params.get('input_size', 10)
        self.hidden_size = params.get('hidden_size', 20)
        self.output_size = params.get('output_size', 1)
        self.learning_rate = params.get('learning_rate', 0.01)
        self.epochs = params.get('epochs', 50)
        
        # Initialize weights
        self.W_ih = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.W_ho = np.random.randn(self.output_size, self.hidden_size) * 0.01
        
        self.b_h = np.zeros((self.hidden_size, 1))
        self.b_o = np.zeros((self.output_size, 1))
        
        self.h_prev = np.zeros((self.hidden_size, 1))
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    def _forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through one time step
        
        Parameters:
        -----------
        x : np.ndarray - Input vector (input_size, 1)
        h_prev : np.ndarray - Previous hidden state (hidden_size, 1)
        
        Returns:
        --------
        h : np.ndarray - New hidden state
        y : np.ndarray - Output
        """
        # Hidden state update
        h = self.tanh(np.dot(self.W_ih, x) + np.dot(self.W_hh, h_prev) + self.b_h)
        
        # Output calculation
        y = self.sigmoid(np.dot(self.W_ho, h) + self.b_o)
        
        return h, y
    
    def train(self, input_data: np.ndarray, time_steps: int) -> None:
        """
        Train RNN on sequential data
        
        Parameters:
        -----------
        input_data : np.ndarray
            Input sequences of shape (n_samples, input_size)
        time_steps : int
            Number of time steps to unroll
        """
        n_samples = input_data.shape[0]
        
        for epoch in range(self.epochs):
            total_loss = 0
            h = np.zeros((self.hidden_size, 1))
            
            for t in range(min(time_steps, n_samples - 1)):
                x = input_data[t].reshape(-1, 1)
                target = input_data[t + 1].reshape(-1, 1)
                
                # Forward pass
                h, y_pred = self._forward(x, h)
                
                # Compute loss (mean squared error)
                loss = np.mean((y_pred - target) ** 2)
                total_loss += loss
                
                # Simplified backpropagation through time (BPTT)
                # In a real implementation, this would be a full BPTT algorithm
                
                # Update weights (placeholder for actual BPTT)
                # This is a very simplified update rule
                error = target - y_pred
                dW_ho = np.dot(error, h.T)
                dh = np.dot(self.W_ho.T, error)
                
                # Update weights with simplified rule
                self.W_ho += self.learning_rate * dW_ho
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/time_steps}")
    
    def predict(self, input_data: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """
        Make predictions using trained RNN
        
        Parameters:
        -----------
        input_data : np.ndarray
            Initial input sequence of shape (input_size,)
        steps_ahead : int
            Number of steps to predict ahead
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted sequence of shape (steps_ahead, output_size)
        """
        predictions = np.zeros((steps_ahead, self.output_size))
        h = self.h_prev
        x = input_data[-1].reshape(-1, 1)
        
        for t in range(steps_ahead):
            h, y_pred = self._forward(x, h)
            predictions[t] = y_pred.flatten()
            x = y_pred  # Use prediction as next input
        
        return predictions
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """Update RNN parameters"""
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'hidden_size' in params:
            # In a real implementation, this would require reinitializing weights
            self.hidden_size = params['hidden_size']
        if 'epochs' in params:
            self.epochs = params['epochs']
    
    def evaluate(self) -> float:
        """
        Evaluate RNN performance (placeholder)
        
        In a real implementation, this would calculate metrics like
        prediction accuracy, mean squared error, etc.
        """
        return np.random.random()  # Placeholder


# Evolutionary Algorithm Implementation
class EvolutionaryAlgorithm:
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize Evolutionary Algorithm with configuration parameters
        
        Parameters:
        -----------
        params : Dict
            population_size : int - Size of the population
            mutation_rate : float - Probability of mutation
            crossover_rate : float - Probability of crossover
            generations : int - Number of generations to evolve
            param_bounds : Dict - Min and max values for each parameter
        """
        self.population_size = params.get('population_size', 50)
        self.mutation_rate = params.get('mutation_rate', 0.1)
        self.crossover_rate = params.get('crossover_rate', 0.7)
        self.generations = params.get('generations', 30)
        self.param_bounds = params.get('param_bounds', {
            'som_params': {
                'learning_rate': (0.01, 0.5),
                'sigma': (0.1, 2.0),
                'decay_factor': (0.7, 0.99),
                'epochs': (10, 200)
            },
            'rnn_params': {
                'learning_rate': (0.001, 0.1),
                'hidden_size': (10, 100),
                'epochs': (10, 100)
            }
        })
        
        # Initialize population
        self.population = self._initialize_population()
        
    def _initialize_population(self) -> List[Dict[str, Dict[str, float]]]:
        """Initialize random population within parameter bounds"""
        population = []
        
        for _ in range(self.population_size):
            individual = {
                'som_params': {},
                'rnn_params': {}
            }
            
            # Initialize SOM parameters
            for param, (min_val, max_val) in self.param_bounds['som_params'].items():
                individual['som_params'][param] = np.random.uniform(min_val, max_val)
            
            # Initialize RNN parameters
            for param, (min_val, max_val) in self.param_bounds['rnn_params'].items():
                individual['rnn_params'][param] = np.random.uniform(min_val, max_val)
            
            population.append(individual)
        
        return population
    
    def _select_parents(self, fitness_scores: np.ndarray) -> Tuple[int, int]:
        """
        Select parents using tournament selection
        
        Parameters:
        -----------
        fitness_scores : np.ndarray
            Fitness scores for each individual
            
        Returns:
        --------
        parent1_idx, parent2_idx : Tuple[int, int]
            Indices of selected parents
        """
        tournament_size = max(2, self.population_size // 5)
        
        # First parent
        candidates1 = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        parent1_idx = candidates1[np.argmax(fitness_scores[candidates1])]
        
        # Second parent
        candidates2 = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        parent2_idx = candidates2[np.argmax(fitness_scores[candidates2])]
        
        return parent1_idx, parent2_idx
    
    def _crossover(self, parent1: Dict[str, Dict[str, float]], 
                  parent2: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Perform crossover between two parents
        
        Parameters:
        -----------
        parent1, parent2 : Dict
            Parent individuals
            
        Returns:
        --------
        child : Dict
            Child individual resulting from crossover
        """
        child = {
            'som_params': {},
            'rnn_params': {}
        }
        
        # Crossover SOM parameters
        for param in self.param_bounds['som_params']:
            if np.random.random() < 0.5:
                child['som_params'][param] = parent1['som_params'][param]
            else:
                child['som_params'][param] = parent2['som_params'][param]
        
        # Crossover RNN parameters
        for param in self.param_bounds['rnn_params']:
            if np.random.random() < 0.5:
                child['rnn_params'][param] = parent1['rnn_params'][param]
            else:
                child['rnn_params'][param] = parent2['rnn_params'][param]
        
        return child
    
    def _mutate(self, individual: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Mutate an individual
        
        Parameters:
        -----------
        individual : Dict
            Individual to mutate
            
        Returns:
        --------
        mutated : Dict
            Mutated individual
        """
        mutated = {
            'som_params': individual['som_params'].copy(),
            'rnn_params': individual['rnn_params'].copy()
        }
        
        # Mutate SOM parameters
        for param, (min_val, max_val) in self.param_bounds['som_params'].items():
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation with sigma proportional to the parameter range
                sigma = (max_val - min_val) * 0.1
                mutation = np.random.normal(0, sigma)
                mutated['som_params'][param] = np.clip(
                    individual['som_params'][param] + mutation, min_val, max_val
                )
        
        # Mutate RNN parameters
        for param, (min_val, max_val) in self.param_bounds['rnn_params'].items():
            if np.random.random() < self.mutation_rate:
                sigma = (max_val - min_val) * 0.1
                mutation = np.random.normal(0, sigma)
                mutated['rnn_params'][param] = np.clip(
                    individual['rnn_params'][param] + mutation, min_val, max_val
                )
        
        return mutated
    
    def optimize(self, fitness_function: Callable[[Dict[str, Dict[str, float]]], float]) -> Dict[str, Dict[str, float]]:
        """
        Run evolutionary optimization
        
        Parameters:
        -----------
        fitness_function : Callable
            Function that evaluates the fitness of an individual
            
        Returns:
        --------
        best_individual : Dict
            Best individual found
        """
        best_fitness = -float('inf')
        best_individual = None
        
        for generation in range(self.generations):
            # Evaluate fitness of each individual
            fitness_scores = np.array([fitness_function(ind) for ind in self.population])
            
            # Track best individual
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_individual = self.population[max_idx]
            
            print(f"Generation {generation}: Best fitness = {best_fitness}")
            
            # Create new population
            new_population = []
            
            # Elitism: Keep the best individual
            new_population.append(self.population[max_idx])
            
            # Generate rest of the population
            while len(new_population) < self.population_size:
                # Selection
                parent1_idx, parent2_idx = self._select_parents(fitness_scores)
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                child = self._mutate(child)
                
                # Add to new population
                new_population.append(child)
            
            # Replace old population
            self.population = new_population
        
        return best_individual


# Combined Model
class AdaptiveHybridModel:
    def __init__(self, som_params: Dict[str, Any], rnn_params: Dict[str, Any], 
                 evo_params: Dict[str, Any]):
        """
        Initialize the hybrid model combining SOM, RNN, and Evolutionary Algorithm
        
        Parameters:
        -----------
        som_params : Dict - Parameters for the SOM
        rnn_params : Dict - Parameters for the RNN
        evo_params : Dict - Parameters for the Evolutionary Algorithm
        """
        self.som = SOM(som_params)
        self.rnn = RNN(rnn_params)
        self.evo = EvolutionaryAlgorithm(evo_params)
        
        # Store the history of adaptations for analysis
        self.adaptation_history = []
    
    def train_som(self, input_data: np.ndarray) -> np.ndarray:
        """
        Train SOM on clustering spatial patterns from input data
        
        Parameters:
        -----------
        input_data : np.ndarray
            Input data of shape (n_samples, input_dim)
            
        Returns:
        --------
        cluster_map : np.ndarray
            Cluster assignments for input data
        """
        print("Training SOM...")
        self.som.train(input_data)
        cluster_map = self.som.get_clusters()
        return cluster_map
    
    def train_rnn(self, input_data: np.ndarray, time_steps: int) -> None:
        """
        Train RNN on temporal patterns within the clustered data
        
        Parameters:
        -----------
        input_data : np.ndarray
            Input data of shape (n_samples, input_dim)
        time_steps : int
            Number of time steps to consider for temporal dependencies
        """
        print("Training RNN...")
        temporal_features = self.som.extract_temporal_features(input_data)
        self.rnn.train(temporal_features, time_steps)
    
    def optimize_parameters(self, input_data: np.ndarray, time_steps: int) -> Dict[str, Dict[str, float]]:
        """
        Use the evolutionary algorithm to optimize hyperparameters of SOM and RNN
        
        Parameters:
        -----------
        input_data : np.ndarray
            Input data for evaluation
        time_steps : int
            Number of time steps for temporal dependencies
            
        Returns:
        --------
        best_params : Dict
            Optimized parameters for SOM and RNN
        """
        print("Optimizing Parameters...")
        
        def fitness_function(params: Dict[str, Dict[str, float]]) -> float:
            """
            Define a fitness function to evaluate the performance of the SOM-RNN system
            
            Parameters:
            -----------
            params : Dict
                Parameters for SOM and RNN
                
            Returns:
            --------
            fitness : float
                Fitness score (higher is better)
            """
            # Create temporary models with the parameters to evaluate
            temp_som = SOM({**params['som_params'], 'input_dim': input_data.shape[1]})
            temp_som.train(input_data)
            
            temporal_features = temp_som.extract_temporal_features(input_data)
            
            temp_rnn = RNN({**params['rnn_params'], 'input_size': temporal_features.shape[1]})
            temp_rnn.train(temporal_features, time_steps)
            
            # Calculate fitness based on SOM clustering quality and RNN prediction accuracy
            som_score = temp_som.evaluate()
            rnn_score = temp_rnn.evaluate()
            
            # Weighted combination (can be adjusted)
            fitness = 0.4 * som_score + 0.6 * rnn_score
            
            return fitness
        
        # Run evolutionary optimization
        best_params = self.evo.optimize(fitness_function)
        
        # Apply the optimized parameters
        self.som.update_params(best_params['som_params'])
        self.rnn.update_params(best_params['rnn_params'])
        
        # Store history for analysis
        self.adaptation_history.append(best_params)
        
        return best_params
    
    def evaluate_performance(self, input_data: np.ndarray, time_steps: int) -> Dict[str, float]:
        """
        Evaluate performance metrics for combined SOM-RNN system
        
        Parameters:
        -----------
        input_data : np.ndarray
            Input data for evaluation
        time_steps : int
            Number of time steps for temporal evaluation
            
        Returns:
        --------
        metrics : Dict
            Dictionary of performance metrics
        """
        # SOM evaluation metrics
        self.som.train(input_data)
        som_score = self.som.evaluate()
        
        # RNN evaluation metrics
        temporal_features = self.som.extract_temporal_features(input_data)
        self.rnn.train(temporal_features, time_steps)
        rnn_score = self.rnn.evaluate()
        
        # Combined score
        combined_score = 0.4 * som_score + 0.6 * rnn_score
        
        return {
            'som_score': som_score,
            'rnn_score': rnn_score,
            'combined_score': combined_score
        }
    
    def run(self, input_data: np.ndarray, time_steps: int, optimize: bool = True) -> Dict[str, Any]:
        """
        Full system execution: training, optimization, and real-time adaptation
        
        Parameters:
        -----------
        input_data : np.ndarray
            Input data of shape (n_samples, input_dim)
        time_steps : int
            Number of time steps for temporal dependencies
        optimize : bool
            Whether to run evolutionary optimization
            
        Returns:
        --------
        results : Dict
            Dictionary containing results and predictions
        """
        print("Starting Adaptive Hybrid Model...")
        
        # Train SOM
        clusters = self.train_som(input_data)
        
        # Extract temporal features using SOM
        temporal_features = self.som.extract_temporal_features(input_data)
        
        # Train RNN
        self.train_rnn(input_data, time_steps)
        
        # Optimize parameters if requested
        if optimize:
            best_params = self.optimize_parameters(input_data, time_steps)
            print("System performance optimized with parameters:")
            print(best_params)
        
        # Make predictions
        predictions = self.rnn.predict(temporal_features[-1], steps_ahead=5)
        
        # Evaluate final performance
        metrics = self.evaluate_performance(input_data, time_steps)
        
        return {
            'clusters': clusters,
            'predictions': predictions,
            'metrics': metrics,
            'adaptation_history': self.adaptation_history
        }


# Example Usage
def main():
    # Generate sample data (in a real scenario, this would be your actual data)
    np.random.seed(42)
    n_samples = 500
    input_dim = 5
    
    # Create a dataset with temporal patterns (e.g., sine waves with noise)
    t = np.linspace(0, 4 * np.pi, n_samples)
    data = np.zeros((n_samples, input_dim))
    
    for i in range(input_dim):
        freq = 1.0 + 0.2 * i
        phase = np.pi * i / input_dim
        data[:, i] = np.sin(freq * t + phase) + 0.1 * np.random.randn(n_samples)
    
    # Define model parameters
    som_params = {
        'map_size': (10, 10),
        'learning_rate': 0.1,
        'sigma': 1.0,
        'decay_factor': 0.95,
        'epochs': 50,
        'input_dim': input_dim
    }
    
    rnn_params = {
        'input_size': 100,  # Will be overridden based on SOM output
        'hidden_size': 30,
        'output_size': input_dim,
        'learning_rate': 0.01,
        'epochs': 30
    }
    
    evo_params = {
        'population_size': 20,
        'mutation_rate': 0.1,
        'crossover_rate': 0.7,
        'generations': 10,
        'param_bounds': {
            'som_params': {
                'learning_rate': (0.01, 0.5),
                'sigma': (0.1, 2.0),
                'decay_factor': (0.7, 0.99),
                'epochs': (10, 100)
            },
            'rnn_params': {
                'learning_rate': (0.001, 0.1),
                'hidden_size': (10, 50),
                'epochs': (10, 50)
            }
        }
    }
    
    # Create and run the model
    model = AdaptiveHybridModel(som_params, rnn_params, evo_params)
    results = model.run(data, time_steps=10)
    
    print("\nFinal Results:")
    print(f"Clustering Score: {results['metrics']['som_score']:.4f}")
    print(f"Prediction Score: {results['metrics']['rnn_score']:.4f}")
    print(f"Combined Score: {results['metrics']['combined_score']:.4f}")
    print(f"Predictions for next 5 time steps:\n{results['predictions']}")


if __name__ == "__main__":
    main()
