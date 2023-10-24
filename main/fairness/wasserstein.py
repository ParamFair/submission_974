import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

from .utils import EQF

class WassersteinBinary:
    def __init__(self, 
                 sigma=0.001, 
                 seed=42) -> None:
        self.sigma = sigma
        self.seed=seed

    def fit(self, 
            y_calib, 
            s_calib) -> None:
        
        # Isolate binary senstive attributes
        try:
            self.s0, self.s1 = set(s_calib)
        except ValueError:
            raise ValueError('The vector of sensitive features ' \
                             'you supplied is not binary')
        
        # Get idx
        iw0 = np.where(s_calib == self.s0)[0]
        iw1 = np.where(s_calib == self.s1)[0]

        # Set weights 
        self.w0 = len(iw0)/len(y_calib)
        self.w1 = 1 - self.w0

        # Fit ECDF/EQF 
        np.random.seed(self.seed)
        epsilon = np.random.uniform(-self.sigma, 
                                    self.sigma, 
                                    len(y_calib))
        
        # EQF
        self.EQF0 = EQF(y_calib[iw0]+epsilon[iw0])
        self.EQF1 = EQF(y_calib[iw1]+epsilon[iw1])

        # ECDF 
        self.ECDF0 = ECDF(y_calib[iw0]+epsilon[iw0])
        self.ECDF1 = ECDF(y_calib[iw1]+epsilon[iw1])

    def transform(self, 
                  y_test, 
                  s_test) -> np.ndarray:
        
        iw0 = np.where(s_test == self.s0)[0]
        iw1 = np.where(s_test == self.s1)[0]

        # Isolate predictions
        y_test_0 = y_test[iw0]
        y_test_1 = y_test[iw1]

        # Initialise fair predictions 
        y_fair_0 = np.zeros_like(y_test_0)
        y_fair_1 = np.zeros_like(y_test_1)
        y_fair = np.zeros_like(y_test)

        np.random.seed(self.seed)
        epsilon = np.random.uniform(-self.sigma,
                                    self.sigma,
                                    len(y_test))
        
        # Run fairness transform
        y_fair_0 += (self.w0 * 
                     self.EQF0(self.ECDF0(y_test_0 + epsilon[iw0])))
        y_fair_0 += (self.w1 * 
                     self.EQF1(self.ECDF0(y_test_0 + epsilon[iw0])))
        
        y_fair_1 += (self.w0 * 
                     self.EQF0(self.ECDF1(y_test_1 + epsilon[iw1])))
        y_fair_1 += (self.w1 * 
                     self.EQF1(self.ECDF1(y_test_1 + epsilon[iw1])))
        
        # Recombine 
        y_fair[iw0] = y_fair_0
        y_fair[iw1] = y_fair_1

        return y_fair

