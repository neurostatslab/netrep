import numpy as np
from sklearn.base import BaseEstimator

from netrep.metrics.hsic import HSIC

class EKD:
    """
    Euclidean Kernel Distance, equivalent to average linear decoding distance
        Given two kernels K, H, EKD(K, H) = sqrt(HSIC(K, K) + HSIC(H, H) - 2*HSIC(K, H)) = ||K - H||_F
        Equivalent to the Euclidean distance between two kernels
    """

    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def score(self, X, Y, mode = 'biased'):
        """
        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons_x) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons_y) matrix of activations.

        mode : str
            particular methods of calculating EKD
                default = 'biased'

            'biased': EKD estimator constructed via biased HSIC estimator from Gretton et al 2005
            'unbiased': EKD estimator constructed via debiased HSIC estimator from Song et al 2012

        Returns
        -------
        dist : float
            Distance between X and Y.
        """

        assert X.shape[0] == Y.shape[0], f'Number of samples do not align! (X.shape = {X.shape}, Y.shape = {Y.shape})'

        if mode == 'biased' or mode == 'unbiased':
            # HSIC-based computation of CKA inherits bias/debiased structure of HSIC
            hsic = HSIC()
            X_at_Y = hsic.score(X, Y, mode = mode)
            X_at_X = hsic.score(X, X, mode = mode)
            Y_at_Y = hsic.score(Y, Y, mode = mode)
            return np.sqrt(X_at_X + Y_at_Y - 2*X_at_Y)
        
        else:
            raise ValueError(f'mode {mode} does not exist')
