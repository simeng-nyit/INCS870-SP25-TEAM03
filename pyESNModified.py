# pyESN.py
import numpy as np
from itertools import combinations

class ESN():
    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001,
                 teacher_scaling=1.0, teacher_forcing=True,
                 random_state=None, use_bundling=False, K=3):
        
        # Initialize parameters
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.teacher_scaling = teacher_scaling
        self.teacher_forcing = teacher_forcing
        self.use_bundling = use_bundling
        self.K = K
        self.bundle_matrix = None

        # Initialize random generator
        self.random_state = np.random.RandomState(random_state)
        
        # Initialize weights
        self.W = self._initialize_reservoir_weights()
        self.W_in = self._initialize_input_weights()
        self.W_feedb = self._initialize_feedback_weights()
        self.W_out = None

    def _initialize_reservoir_weights(self):
        weights = self.random_state.rand(self.n_reservoir, self.n_reservoir) - 0.5
        weights[self.random_state.rand(*weights.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(weights)))
        return weights * (self.spectral_radius / radius)

    def _initialize_input_weights(self):
        return self.random_state.rand(self.n_reservoir, self.n_inputs) * 2 - 1

    def _initialize_feedback_weights(self):
        return self.random_state.rand(self.n_reservoir, self.n_outputs) * 2 - 1

    def _create_bundles(self, inputs):
        n_features = inputs.shape[1]
        conflict = np.zeros((n_features, n_features), dtype=np.float64)
        
        # Calculate feature correlations
        for i, j in combinations(range(n_features), 2):
            if np.std(inputs[:,i]) == 0 or np.std(inputs[:,j]) == 0:
                correlation = 0
            else:
                correlation = abs(np.corrcoef(inputs[:,i], inputs[:,j])[0,1])
            conflict[i,j] = conflict[j,i] = correlation
        
        # Create conflict graph
        conflict = (conflict > 0.5).astype(int)
        
        # Greedy bundling algorithm
        degrees = conflict.sum(axis=1)
        order = np.argsort(-degrees)
        bundles = []
        available = np.ones(n_features, dtype=bool)
        
        for feat in order:
            if available[feat]:
                bundle = [feat]
                available[feat] = False
                neighbors = np.where(conflict[feat])[0]
                
                for neighbor in neighbors[available[neighbors]]:
                    if sum(conflict[neighbor][bundle]) <= self.K:
                        bundle.append(neighbor)
                        available[neighbor] = False
                
                bundles.append(bundle)
        
        # Create bundle matrix
        self.bundle_matrix = np.zeros((n_features, len(bundles)), dtype=np.float64)
        for i, bundle in enumerate(bundles):
            self.bundle_matrix[bundle, i] = 1.0
        
        # Normalize columns
        col_sums = self.bundle_matrix.sum(axis=0)
        valid_cols = col_sums > 0
        self.bundle_matrix[:, valid_cols] /= col_sums[valid_cols]
        
        return inputs @ self.bundle_matrix

    def fit(self, inputs, outputs):
        # Apply greedy bundling
        if self.use_bundling:
            inputs = self._create_bundles(inputs.astype(np.float64))
            self.W_in = self._initialize_input_weights()[:, :inputs.shape[1]]

        # Collect reservoir states
        states = np.zeros((inputs.shape[0], self.n_reservoir), dtype=np.float64)
        for n in range(1, inputs.shape[0]):
            states[n] = np.tanh(
                self.W @ states[n-1] + 
                self.W_in @ inputs[n] + 
                self.W_feedb @ outputs[n-1] + 
                self.noise * (self.random_state.rand(self.n_reservoir) - 0.5)
            )

        # Train output weights
        transient = min(int(inputs.shape[0]/10), 100)
        self.W_out = np.linalg.pinv(states[transient:]) @ outputs[transient:]
        return self

    def predict(self, inputs):
        if self.use_bundling:
            inputs = inputs @ self.bundle_matrix
            
        # Generate predictions
        states = np.zeros((inputs.shape[0], self.n_reservoir), dtype=np.float64)
        outputs = np.zeros((inputs.shape[0], self.n_outputs), dtype=np.float64)
        
        for n in range(inputs.shape[0]):
            states[n] = np.tanh(
                self.W @ states[n-1] + 
                self.W_in @ inputs[n] + 
                self.W_feedb @ outputs[n-1] + 
                self.noise * (self.random_state.rand(self.n_reservoir) - 0.5)
            )
            outputs[n] = self.W_out.T @ states[n]
            
        return outputs