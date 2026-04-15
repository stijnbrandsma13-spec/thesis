import numpy as np
from scipy.special import softmax
import cvxpy as cp

class FKRBEstimator:
    def __init__(self, beta_support: np.ndarray):
        self.beta_support = beta_support
        self.theta = dict()

    def _calculate_Z(self, x: np.ndarray):
        N, J, K = x.shape
        R = self.beta_support.shape[0]
        
        # Calculate v_ijr = x'_ij beta^(r) for all i,j,r with some numpy wizardry
        v = np.einsum('ijk,rk->ijr', x, self.beta_support)
        # Calculate g_j(x_i,beta^(r)) for all j,i,r. g_j is just a softmax
        g = softmax(v, axis = 1)

        Z = g[:, 1:, :].reshape(N*(J-1), R) # Remove outside good and reshape
        return Z
    
    def estimate(self, y: np.ndarray, x: np.ndarray, constrained: bool = True) -> np.ndarray:
        R = self.beta_support.shape[0]
        
        Z = self._calculate_Z(x)
        y = y[:, 1:].ravel()

        if constrained:
            # Compute constrained least squares
            theta_cp = cp.Variable(R)
            objective = cp.Minimize(cp.sum_squares(y - Z @ theta_cp))
            constraints = [
                theta_cp >= 0, 
                cp.sum(theta_cp) == 1
            ]

            problem = cp.Problem(objective, constraints)
            problem.solve()

            theta = theta_cp.value
            self.theta['constrained'] = theta
        else:
            # Compute OLS
            # Grab first item as this is the estimate, the rest are additional stuff such as residuals
            theta = np.linalg.lstsq(Z, y)[0]
            self.theta['unconstrained'] = theta
        
        return theta
    
    def get_cdf(self) -> callable:
        theta_unconstrained = self.theta['unconstrained']
        theta_constrained = self.theta['constrained']

        def cdf(b: np.ndarray, constrained: bool = True) -> float:
            if constrained:
                theta = theta_constrained
            else:
                theta = theta_unconstrained

            # Calculate prod_i 1{beta_i^(r) <= b_i} for all support points r
            mask = (self.beta_support <= b).all(axis = 1)
            # Then select all support points for which the above holds and sum with estimated probabilities
            return theta[mask].sum()
        
        return cdf