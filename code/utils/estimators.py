import numpy as np
from scipy.special import softmax
from scipy.stats import norm
from scipy.optimize import nnls
from sklearn.linear_model import enet_path
import cvxpy as cp

class FKRBEstimator:
    def __init__(self, beta_support: np.ndarray):
        self.beta_support = beta_support
        self.theta = dict()
        self.se = dict()
        self.ci = None

    def _calculate_Z(self, x: np.ndarray):
        N, J, K = x.shape
        R = self.beta_support.shape[0]
        
        # Calculate v_ijr = x'_ij beta^(r) for all i,j,r with some numpy wizardry
        v = np.einsum('ijk,rk->ijr', x, self.beta_support)
        # Calculate g_j(x_i,beta^(r)) for all j,i,r. g_j is just a softmax
        g = softmax(v, axis = 1)

        Z = g[:, 1:, :].reshape(N*(J-1), R) # Remove outside good and reshape
        return Z
    
    def _unconstrained_estimator(self, Z: np.ndarray, y: np.ndarray, N: int, J: int, R: int) -> np.ndarray:
        # Compute OLS
        # Grab first item as this is the estimate, the rest are additional stuff such as residuals
        theta = np.linalg.lstsq(Z, y)[0]
        self.theta['unconstrained'] = theta

        # Calculate standard errors
        u = y - (Z @ theta)
        u_clustered = u.reshape(N, J - 1)
        Z_clustered = Z.reshape(N, J - 1, R)

        scores = np.einsum('ijr,ij->ir', Z_clustered, u_clustered)
        omega_hat = scores.T @ scores

        ZprimeZinv = np.linalg.inv(Z.T @ Z)
        
        bias_correction = (N / (N - 1)) * ((N * (J - 1) - 1) / (N * (J - 1) - R))

        vcov = bias_correction * ZprimeZinv @ omega_hat @ ZprimeZinv
        se = np.sqrt(np.diag(vcov))
        self.se['unconstrained'] = se

        return theta
    
    def _constrained_estimator(self, Z: np.ndarray, y: np.ndarray, R: int, NNL: bool = True) -> np.ndarray:
        if NNL:
            # Reparametrize by fixing theta_R = 1 - sum(theta_1..R-1), absorbing the
            # sum-to-one constraint. This transforms the CLS problem into a NNL problem.
            y_tilde = y - Z[:, -1]
            Z_tilde = Z[:, :-1] - Z[:, [-1]]

            # Check if the NNLS solution (ignoring the sum <= 1 constraint) is already
            # feasible. If so, no regularization is needed and we skip the path entirely.
            theta_nnls, _ = nnls(Z_tilde, y_tilde)

            if theta_nnls.sum() <= 1.0:
                theta_partial = theta_nnls
            else:
                # Compute the full LASSO path in one coordinate descent sweep.
                # alpha decreases from alpha_max (all-zero solution) to alpha_min (dense
                # solution), so coef sums increase monotonically left to right.
                _, coefs, _ = enet_path(Z_tilde, y_tilde, l1_ratio=1, positive=True)

                sums = coefs.sum(axis=0)
                # Find the first index where the sum crosses 1, i.e. where the
                # constraint sum(theta) <= 1 becomes binding.
                idx = np.searchsorted(sums, 1.0)

                if idx == 0:
                    theta_partial = coefs[:, 0]
                elif idx >= coefs.shape[1]:
                    theta_partial = coefs[:, -1]
                else:
                    # Linearly interpolate between the two bracketing path solutions.
                    # This approximates the exact alpha where sum(theta) = 1.
                    t = (1.0 - sums[idx - 1]) / (sums[idx] - sums[idx - 1])
                    theta_partial = coefs[:, idx - 1] + t * (coefs[:, idx] - coefs[:, idx - 1])

            # Recover theta_R from the sum-to-one constraint and reassemble full theta.
            theta = np.append(theta_partial, 1.0 - theta_partial.sum())
            self.theta['constrained'] = theta
        
        else: # Compute constrained least squares
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

        return theta
    
    def estimate(self, y: np.ndarray, x: np.ndarray, constrained: bool = True, estimate_both: bool = True, NNL: bool = True) -> np.ndarray:
        N, J, K = x.shape
        R = self.beta_support.shape[0]
        
        Z = self._calculate_Z(x)
        y = y[:, 1:].ravel()

        if estimate_both:
            if constrained:
                # Estimate both but return constrained
                self._unconstrained_estimator(Z=Z, y=y, N=N, J=J, R=R)
                theta = self._constrained_estimator(Z=Z, y=y, R=R, NNL=NNL)
            else:
                # Estimate both but return unconstrained
                self._constrained_estimator(Z=Z, y=y, N=N, J=J, R=R, NNL=NNL)
                theta = self._unconstrained_estimator(Z=Z, y=y, N=N, J=J, R=R)
        else:
            if constrained:
                theta = self._constrained_estimator(Z=Z, y=y, R=R, NNL=NNL)
            else:
                theta = self._unconstrained_estimator(Z=Z, y=y, N=N, J=J, R=R)

        return theta
    
    def confidence_interval(self, alpha: float = 0.05) -> np.ndarray:
        theta_constrained = self.theta['constrained']
        se_ols = self.se['unconstrained']

        z_score = norm.ppf(1 - alpha / 2)

        lower_bound = theta_constrained - z_score * se_ols
        upper_bound = theta_constrained + z_score * se_ols

        ci_ols = np.column_stack((lower_bound, upper_bound))

        ci_fkrb = np.clip(ci_ols, a_min=0.0, a_max=1.0)
        
        self.ci = ci_fkrb
        return ci_fkrb
        
    
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