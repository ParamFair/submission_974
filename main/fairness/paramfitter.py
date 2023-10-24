"""
Module that contains classes to find coefficients for the minimum
expected wasserstein distance problems
"""
import numpy as np
from scipy.optimize import minimize
import scipy.stats as scs
import ot

class LocationScaleEMDW1:
    def __init__(self) -> None:
        self.mu = None
        self.sigma = None

    def _optimize_dist(self, theta):
        estimates_tmp = self.noise_terms*theta[1] +theta[0]
        wasserstein_d = np.abs(estimates_tmp - self.barycenter)
        return np.mean(wasserstein_d)
    
    def fit(self, X, sampling_obs=40000, mc_sampled=20, maxit=10000):
        bary_enlarged = np.repeat(X, sampling_obs/X.shape[0])
        bary_enlarged.sort()

        noise_terms = scs.norm(0,1).rvs(size=(mc_sampled, bary_enlarged.shape[0]))
        noise_terms.sort()

        self.barycenter = np.broadcast_to(bary_enlarged, noise_terms.shape)
        self.noise_terms = noise_terms

        theta_init = np.random.uniform(0,1,(2,))
        response = minimize(self._optimize_dist,
                            theta_init,
                            method='nelder-mead',
                            options={'xatol': 1e-8,
                                     'maxiter': maxit,
                                     'disp': True})
        
        self.mu, self.sigma = response.x

    def sample(self, n, mc_samples=20):
        samples_ = scs.norm(loc=self.mu,
                            scale=self.sigma).rvs(size=(mc_samples,n))
        samples_.sort(axis=1)
        samples_avg = samples_.mean(axis=0)
        return samples_avg
    

class GammaMEWD:
    """
    Gamme MEWD, based on simulation approach
    """
    def __init__(self) -> None:
        self.a = None
        self.rate = None

    def _optimize_dist(self, theta):
        estimates_tmp = (scs.gamma(a=np.exp(theta[0]),
                                   scale=1/np.exp(theta[1]))
                         .rvs(size=(self.mc_sample, self.sampling_obs)))
        
        estimates_tmp.sort()
        wasserstein_d = np.abs(estimates_tmp - self.barycenter)
        return np.mean(wasserstein_d)
    
    def fit(self, X, sampling_obs=40000, mc_sampled=20, maxit=10000,
            theta_init = np.array([0,1])):
        self.sampling_obs = sampling_obs
        self.mc_sample = mc_sampled

        bary_enlarged = np.repeat(X, sampling_obs/X.shape[0])
        bary_enlarged.sort()

        self.barycenter = np.broadcast_to(bary_enlarged, (mc_sampled,
                                                          bary_enlarged.shape[0]))

        response = minimize(self._optimize_dist,
                            theta_init,
                            method='nelder-mead',
                            options={'xatol': 1e-8,
                                     'maxiter': maxit,
                                     'disp': True})
        
        self.a, self.rate = response.x

    def sample(self, n, mc_samples=20):
        gamma_samples = (scs.gamma(a=np.exp(self.a),
                                  scale=1/np.exp(self.rate))
                         .rvs(size=(mc_samples, n)))
        
        gamma_samples.sort(axis=1)
        samples_avg = gamma_samples.mean(axis=0)
        return samples_avg


class BetaEMWD:
    """
    Beta emwd through simulation
    """
    def __init__(self) -> None:
        self.a = None
        self.b = None

    def _optimize_dist(self, theta):
        estimates_tmp = (scs.beta(a=np.exp(theta[0]), 
                                 b=np.exp(theta[1]))
                        .rvs(size=(self.mc_sample, self.sampling_obs)))
        
        estimates_tmp.sort()
        wasserstein_d = np.abs(estimates_tmp - self.barycenter)
        return np.mean(wasserstein_d)
    
    def fit(self, X, sampling_obs=40000, mc_sampled=20, maxit=10000,
            theta_init = np.array([0,1])):
        self.sampling_obs = sampling_obs
        self.mc_sample = mc_sampled

        bary_enlarged = np.repeat(X, sampling_obs/X.shape[0])
        bary_enlarged.sort()

        self.barycenter = np.broadcast_to(bary_enlarged, (mc_sampled,
                                                          bary_enlarged.shape[0]))

        response = minimize(self._optimize_dist,
                            theta_init,
                            method='nelder-mead',
                            options={'xatol': 1e-8,
                                     'maxiter': maxit,
                                     'disp': True})
        
        self.a, self.b = response.x

    def sample(self, n, mc_samples=20):
        gamma_samples = (scs.beta(a=np.exp(self.a), 
                                 b=np.exp(self.b))
                         .rvs(size=(mc_samples, n)))
        
        gamma_samples.sort(axis=1)
        samples_avg = gamma_samples.mean(axis=0)
        return samples_avg
    
class LocationScaleGumbel:
    """
    Location scale model where the shape is determined
    by a left gumbel distribution
    """
    def __init__(self) -> None:
        self.mu = None
        self.sigma = None

    def _optimize_dist(self, theta):
        estimates_tmp = self.noise_terms*theta[1] +theta[0]
        wasserstein_d = np.abs(estimates_tmp - self.barycenter)
        return np.mean(wasserstein_d)
    
    def fit(self, X, sampling_obs=40000, mc_sampled=20, maxit=10000):
        bary_enlarged = np.repeat(X, sampling_obs/X.shape[0])
        bary_enlarged.sort()

        noise_terms = scs.gumbel_l().rvs(size=(mc_sampled, bary_enlarged.shape[0]))
        noise_terms.sort()

        self.barycenter = np.broadcast_to(bary_enlarged, noise_terms.shape)
        self.noise_terms = noise_terms

        theta_init = np.random.uniform(0,1,(2,))
        response = minimize(self._optimize_dist,
                            theta_init,
                            method='nelder-mead',
                            options={'xatol': 1e-8,
                                     'maxiter': maxit,
                                     'disp': True})
        
        self.mu, self.sigma = response.x

    def sample(self, n, mc_samples=20):
        samples_ = scs.gumbel_l(loc=self.mu,
                            scale=self.sigma).rvs(size=(mc_samples,n))
        samples_.sort(axis=1)
        samples_avg = samples_.mean(axis=0)
        return samples_avg
    

class LocationScaleEMWD2:
    """
    EMWD with individually calcualted optimal transport
    matrix - to compare
    """
    def __init__(self) -> None:
        pass

    def _loss_function(self, y_obs, y_hat, weights='uniform'):
        if weights == 'uniform':
            weight_vec = ot.unif(len(y_obs))
        else: 
            raise NotImplementedError
        # Calcualte OT loss
        dist_matrix = ot.dist(y_obs.reshape(-1,1),
                              y_hat.reshape(-1,1),
                              metric='euclidean')
        dist_matrix /= dist_matrix.max()

        # run transport to get loss
        transport_loss = ot.emd2(weight_vec,
                                 weight_vec,
                                 dist_matrix)
        
        print(transport_loss)

        return transport_loss
    
    def _transform_samples(self,
                           y,
                           num_samples, 
                           num_mc):
        # Strech observation vector
        y_stretch = np.repeat(y, num_samples/y.shape[0])

        # Get noise obs
        noise_sample = scs.norm(0,1).rvs(size=(num_mc, y_stretch.shape[0]))

        # Set as attributes
        self.y_extended = np.broadcast_to(y_stretch, noise_sample.shape)
        self.noise_sample = noise_sample

    def objective_function(self):
        raise NotImplementedError
    
    def fit(self,
            y,
            num_samples,
            num_mc,
            param_upper,
            param_lower=0,
            maxiter=10000,
            tolerance=1e-8,
            verbose=True):
        # Init samples
        self._transform_samples(y, num_samples, num_mc)
        # init params
        theta_init = np.random.uniform(param_lower, param_upper, (2,))
        # run optim
        response = minimize(self.objective_function, 
                            theta_init, 
                            method='nelder-mead', 
                            options={
                                     'maxiter': maxiter, 
                                     'xatol':tolerance, 
                                     'disp': verbose})
        
        self.mu, self.sigma = response.x
    

class WassersteinNormal(LocationScaleEMWD2):
    def __init__(self) -> None:
        super().__init__()

    def objective_function(self, params):
        estimates_tmp = params[0] + params[1]*self.noise_sample

        loss_vals = np.zeros((estimates_tmp.shape[0]))

        for i, (obs,est) in enumerate(zip(self.y_extended, estimates_tmp)):
            loss_vals[i] = self._loss_function(obs, est)
        
        return np.mean(loss_vals)


