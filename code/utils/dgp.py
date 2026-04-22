import numpy as np

class DataGenerator:
    def __init__(self, N: int, J: int, K: int, rng: np.random.Generator):
        self.N = N
        self.J = J
        self.K = K
        self.rng = rng
        self.x = None
        self.beta = None
        self.epsilon = None
        self.u = None
        self.y = None

    def _default_discrete_distribution_generator(self) -> np.ndarray:
        # Some black magic way to create {0,1}^K (cartesian product)
        support = np.array(np.meshgrid(*([np.array([0, 1])] * self.K))).T.reshape(-1, self.K)
        # Give each support point equal probability
        probs = probs = np.full(2**self.K, 1 / 2**self.K)
        return support, probs

    def _discrete_sampler_generator(self, support: np.ndarray, probs: np.ndarray) -> callable:
        rng = self.rng
        def draw(size):
            indices = rng.choice(len(probs), size=size[0], p=probs)
            return support[indices]
        return draw

    def _generate_x(self, sampler: callable) -> np.ndarray:
        # Default x is zero (for outside good)
        x = np.zeros((self.N, self.J + 1, self.K))
        # Change everything but the outside good according to sampler
        x[:, 1:, :] = sampler(size = (self.N, self.J, self.K))
        
        self.x = x
        return x
    
    def _generate_beta(self, sampler: callable) -> np.ndarray:
        beta = sampler(size = (self.N, self.K))
        
        self.beta = beta
        return beta
    
    def _generate_epsilon(self, sampler: callable) -> np.ndarray:
        epsilon = sampler(size = (self.N, self.J + 1))
        
        self.epsilon = epsilon
        return epsilon
    
    def _generate_latent_utility(self, x_sampler: callable = None, beta_sampler: callable = None, epsilon_sampler: callable = None, beta_support: np.ndarray = None, beta_probs: np.ndarray = None):
        if x_sampler is None:
            x_sampler = self.rng.normal
        if beta_sampler is None:
            if beta_support is None or beta_probs is None:
                support, probs = self._default_discrete_distribution_generator()
                beta_sampler = self._discrete_sampler_generator(support=support,probs=probs)
            else:
                beta_sampler = self._discrete_sampler_generator(support=beta_support,probs=beta_probs)
        if epsilon_sampler is None:
            epsilon_sampler = self.rng.gumbel

        if self.x is None:
            x = self._generate_x(sampler = x_sampler)
        else:
            x = self.x
        if self.beta is None:
            b = self._generate_beta(sampler = beta_sampler)
        else:
            b = self.beta
        if self.epsilon is None:
            e = self._generate_epsilon(sampler = epsilon_sampler)
        else:
            e = self.epsilon
        
        # Vectorized computation of u_ij = Σ_k x_ijk * beta_ik (+ epsilon_ij). This is in Einstein notation
        u = np.einsum('ijk,ik->ij', x, b) + e
        
        self.u = u
        return u
    
    def _generate_choices(self, x_sampler: callable = None, beta_sampler: callable = None, epsilon_sampler: callable = None, beta_support: np.ndarray = None, beta_probs: np.ndarray = None) -> np.ndarray:
        if self.u is None:
            u = self._generate_latent_utility(x_sampler=x_sampler, beta_sampler=beta_sampler, epsilon_sampler=epsilon_sampler, beta_support=beta_support, beta_probs=beta_probs)
        else:
            u = self.u
        
        y = (u == u.max(axis=1, keepdims=True))
        y = y.astype(int)

        self.y = y
        return y

    def generate(self, x_sampler: callable = None, beta_sampler: callable = None, epsilon_sampler: callable = None, beta_support: np.ndarray = None, beta_probs: np.ndarray = None):
        self._generate_choices(x_sampler=x_sampler, beta_sampler=beta_sampler, epsilon_sampler=epsilon_sampler, beta_support=beta_support, beta_probs=beta_probs)

        return self.y, self.x
    
def heiss_x_sampler(size=None):
    # The class passes size=(N, J, K). 
    # We unpack it to separate the observations/alternatives from the features (K)
    *base_dims, K = size
    
    if K != 2:
        raise ValueError(f"This specific sampler requires K=2, but got K={K}")
        
    result = np.zeros(size)
    
    # Fill the first feature with U(0, 5)
    result[:,:, 0] = np.random.uniform(0, 5, size=base_dims)
    
    # Fill the second feature with U(-3, 1)
    result[:,:, 1] = np.random.uniform(-3, 1, size=base_dims)
    
    return result

def heiss_beta_support_probs(R: int):
    # Determine points per dimension (r = sqrt(R))
    r = int(np.sqrt(R))
    
    grid_1d = np.linspace(-4.5, 3.5, r)
    beta_1, beta_2 = np.meshgrid(grid_1d, grid_1d)
    
    # Flatten and combine into an (R, 2) array of all possible coordinates
    full_grid = np.column_stack((beta_1.ravel(), beta_2.ravel()))
    
    # Define the masks for the two target regions
    in_region_1 = (full_grid[:, 0] <= -0.5) & (full_grid[:, 1] <= -0.5)
    in_region_2 = (full_grid[:, 0] >= -0.5) & (full_grid[:, 1] >= -0.5)
    
    # Filter the grid to keep only points in Region 1 OR Region 2
    support = full_grid[in_region_1 | in_region_2]
    
    # Calculate probabilities theta_s = 1/S
    S = len(support)
    probs = np.full(S, 1 / S)
    
    return full_grid, support, probs

# Test
if __name__ == '__main__':
    test = DataGenerator(10, 2, 2, np.random.default_rng(10))
    y, x = test.generate()

    print(y.shape)
    print(x.shape)
    print(x[0])
    print(x[0][0])
    print(x[0].shape)
    print(test.beta)