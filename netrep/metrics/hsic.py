import numpy as np
from sklearn.base import BaseEstimator

class HSIC:
    """
    Methods to calculate the HSIC score between two sets of neural data
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
            particular methods of calculating CKA
                default = 'biased'

            'biased': biased HSIC estimator from Gretton et al 2005
            'unbiased': debiased HSIC estimator from Song et al 2012

        Returns
        -------
        dist : float
            Distance between X and Y.
        """

        assert X.shape[0] == Y.shape[0], f'Number of samples do not align! (X.shape = {X.shape}, Y.shape = {Y.shape})'

        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

        if mode == 'biased':
            # Compute Tr(X @ Y)/M^2 for M = num_samples
            return np.einsum('ij, ji ->', X, Y)/(len(X) ** 2)

        elif mode == 'unbiased':
            # Fill diagonals with 0
            # Compute (1/(M*(M - 3))) * (trace(X @ Y) - (2/(M - 2)) * sum(X @ Y) + (sum(X)*sum(Y))/((M - 1)*(M - 2)))
            M = X.shape[0]

            X_copy = X.copy()
            Y_copy = Y.copy()

            np.fill_diagonal(X_copy, 0)
            np.fill_diagonal(Y_copy, 0)
            X_at_Y = np.matmul(X_copy, Y_copy)

            sum = np.trace(X_at_Y) - 2*np.sum(X_at_Y)/(M - 2) + np.sum(X_copy)*np.sum(Y_copy)/((M - 1)*(M - 2))
            return sum/(M*(M - 3))

        else:
            raise ValueError(f'mode {mode} does not exist')
