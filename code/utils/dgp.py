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

    def _discrete_sampler_generator(self, support: np.ndarray, probs: np.ndarray, rng: np.random.Generator) -> callable:
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
    
    def _generate_latent_utility(self, x_sampler: callable = None, beta_sampler: callable = None, epsilon_sampler: callable = None):
        if x_sampler is None:
            x_sampler = self.rng.normal
        if beta_sampler is None:
            support, probs = self._default_discrete_distribution_generator()
            beta_sampler = self._discrete_sampler_generator(support=support,probs=probs,rng=self.rng)
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
    
    def _generate_choices(self) -> np.ndarray:
        if self.u is None:
            u = self._generate_latent_utility()
        else:
            u = self.u
        
        y = (u == u.max(axis=1, keepdims=True))
        y = y.astype(int)

        self.y = y
        return y

    def generate(self):
        self._generate_choices()

        return self.y, self.x

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