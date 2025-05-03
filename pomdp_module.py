    def set_model(self, 
                  states: List[Any], 
                  actions: List[Any], 
                  observations: List[Any],
                  transition_probs: List[List[List[float]]],
                  observation_probs: List[List[List[float]]],
                  rewards: List[List[float]],
                  initial_belief: List[float]) -> None:
        """
        Set the POMDP model components.
        
        Args:
            states: List of states
            actions: List of actions
            observations: List of observations
            transition_probs: Transition probabilities P(s'|s,a) as [action][state][next_state]
            observation_probs: Observation probabilities P(o|s',a) as [action][next_state][observation]
            rewards: Immediate rewards R(s,a) as [state][action]
            initial_belief: Initial belief distribution over states
        """
        # Validate inputs
        if not all(math.isclose(sum(b), 1.0, abs_tol=1e-5) for b in initial_belief):
            raise ValueError("Initial belief must sum to 1")
            
        if len(initial_belief) != len(states):
            raise ValueError("Initial belief size must match number of states")
            
        # Convert to numpy arrays for efficient computation
        self.states = states
        self.actions = actions
        self.observations = observations
        self.initial_belief = np.array(initial_belief, dtype=np.float32)
        
        # Convert matrices to numpy arrays with appropriate memory optimization
        if self.use_sparse_matrices:
            import scipy.sparse as sp
            self.transition_matrix = [sp.csr_matrix(t) for t in transition_probs]
            self.observation_matrix = [sp.csr_matrix(o) for o in observation_probs]
            self.reward_matrix = sp.csr_matrix(rewards)
        else:
            self.transition_matrix = np.array(transition_probs, dtype=np.float32)
            self.observation_matrix = np.array(observation_probs, dtype=np.float32)
            self.reward_matrix = np.array(rewards, dtype=np.float32)
            
        # Clear any existing solution
        self._reset_solution()
        
    def _reset_solution(self) -> None:
        """Reset all solution-related data structures"""
        self.alpha_vectors = []
        self.policy = {}
        self.belief_points = []
        self.convergence_history = []
        self.iterations_performed = 0
        self.solve_time_ms = 0
        self._belief_cache.clear()
        self._action_value_cache.clear()
        
    def solve(self) -> Dict[str, Any]:
        """
        Solve the POMDP using the configured solver.
        
        Returns:
            Dictionary with solution information
        """
        start_time = time.time()
        
        try:
            if self.solver_type == "point_based_value_iteration":
                self._point_based_value_iteration()
            else:
                raise ValueError(f"Unknown solver type: {self.solver_type}")
                
            self.solve_time_ms = (time.time() - start_time) * 1000
            return self._prepare_result_for_kotlin_bridge()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "iterations_performed": self.iterations_performed,
                    "solve_time_ms": (time.time() - start_time) * 1000
                }
            }
            
    def _point_based_value_iteration(self) -> None:
        """Point-based value iteration algorithm"""
        # Initialize belief points
        self._sample_belief_points()
        
        # Initialize alpha vectors
        self._initialize_alpha_vectors()
        
        # Main iteration loop
        for iteration in range(self.max_iterations):
            delta = self._pbvi_iteration()
            
            # Record convergence
            self.convergence_history.append(delta)
            self.iterations_performed += 1
            
            # Check for convergence
            if delta < self.convergence_threshold:
                break
                
        # Extract policy from alpha vectors
        self._extract_policy()
        
    def _sample_belief_points(self) -> None:
        """Sample belief points using random exploration"""
        num_states = len(self.states)
        self.belief_points = [self.initial_belief.copy()]
        
        # Generate random belief points
        for _ in range(self.num_belief_points - 1):
            # Generate random belief (Dirichlet distribution)
            belief = np.random.dirichlet(np.ones(num_states))
            self.belief_points.append(belief)
            
    def _initialize_alpha_vectors(self) -> None:
        """Initialize alpha vectors using immediate rewards"""
        num_states = len(self.states)
        
        # Create one alpha vector per action
        for action_idx in range(len(self.actions)):
            alpha = np.zeros(num_states)
            
            # Initialize with immediate rewards
            for state_idx in range(num_states):
                alpha[state_idx] = self.reward_matrix[state_idx, action_idx]
                
            self.alpha_vectors.append((alpha, action_idx))
            
    def _pbvi_iteration(self) -> float:
        """Perform one iteration of point-based value iteration"""
        new_alpha_vectors = []
        max_delta = 0.0
        
        for belief in self.belief_points:
            # Find best action and alpha vector for this belief
            best_alpha, best_action, delta = self._backup(belief)
            new_alpha_vectors.append((best_alpha, best_action))
            max_delta = max(max_delta, delta)
            
        # Update alpha vectors
        self.alpha_vectors = new_alpha_vectors
        return max_delta
        
    def _backup(self, belief: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """Backup operation for a single belief point"""
        best_value = -np.inf
        best_alpha = None
        best_action = None
        
        for action_idx in range(len(self.actions)):
            # Compute new alpha vector for this action
            new_alpha = self._compute_alpha_vector_for_action(belief, action_idx)
            
            # Compute value at current belief
            value = np.dot(belief, new_alpha)
            
            # Track best
            if value > best_value:
                best_value = value
                best_alpha = new_alpha
                best_action = action_idx
                
        # Compute improvement
        if self.alpha_vectors:
            old_value = max(np.dot(belief, alpha) for alpha, _ in self.alpha_vectors)
            delta = abs(best_value - old_value)
        else:
            delta = np.inf
            
        return best_alpha, best_action, delta
        
    def _compute_alpha_vector_for_action(self, belief: np.ndarray, action_idx: int) -> np.ndarray:
        """Compute new alpha vector for a given action"""
        num_states = len(self.states)
        num_observations = len(self.observations)
        new_alpha = np.zeros(num_states)
        
        # Compute immediate reward component
        for state_idx in range(num_states):
            new_alpha[state_idx] += self.reward_matrix[state_idx, action_idx]
            
        # Compute future reward component
        for next_state_idx in range(num_states):
            future_value = 0.0
            
            # Sum over possible observations
            for obs_idx in range(num_observations):
                # Compute belief update
                updated_belief = self._update_belief(belief, action_idx, obs_idx)
                
                if updated_belief is not None:
                    # Find best alpha vector for the updated belief
                    best_value = max(np.dot(updated_belief, alpha) for alpha, _ in self.alpha_vectors)
                    
                    # Weight by observation probability
                    obs_prob = self.observation_matrix[action_idx][next_state_idx, obs_idx]
                    future_value += obs_prob * best_value
                    
            # Weight by transition probability and discount
            trans_prob = self.transition_matrix[action_idx][:, next_state_idx]
            new_alpha += self.discount_factor * trans_prob * future_value
            
        return new_alpha
        
    def _update_belief(self, belief: np.ndarray, action_idx: int, obs_idx: int) -> Optional[np.ndarray]:
        """Update belief state after taking an action and receiving an observation"""
        cache_key = (tuple(belief), action_idx, obs_idx)
        
        # Check cache first
        if cache_key in self._belief_cache:
            return self._belief_cache[cache_key]
            
        num_states = len(self.states)
        new_belief = np.zeros(num_states)
        total_prob = 0.0
        
        # Compute new belief: P(s'|b,a,o) ~ P(o|s',a) * sum_s P(s'|s,a) * b(s)
        for next_state_idx in range(num_states):
            obs_prob = self.observation_matrix[action_idx][next_state_idx, obs_idx]
            trans_probs = self.transition_matrix[action_idx][:, next_state_idx]
            new_belief[next_state_idx] = obs_prob * np.dot(trans_probs, belief)
            total_prob += new
