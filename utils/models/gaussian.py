import numpy as np
from utils import squared_exponential_kernel

class GaussianProcess:
    def __init__(self, method="inv", kernel=squared_exponential_kernel):
        self.method = method
        self.kernel = kernel

    def _calculate_covmatrix(self, X1, X2):
        return self.kernel(X1, X2)

    def fit(self, X_train, y_train=None, sigma_noise=0.1):

        # Save training data
        self.X_train = X_train

        # Compute number of data points
        data_points = X_train.shape[0]

        # If no labels are given, assume zero mean
        if y_train is None:
            y_train = np.zeros(data_points)

        # Compute the covariance matrix (Cxx)
        self.Cxx = self._calculate_covmatrix(X_train, X_train)

        # Compute diagonal noise matrix (σ²I)
        diagonal_noise = sigma_noise**2 * np.eye(self.Cxx.shape[0])

        # Compute matrix with diagonal noise (Cxx + σ²I)
        self.Cxx_sigma = self.Cxx + diagonal_noise

        if self.method == "inv":
            # Compute the inverse of the covariance matrix with diagonal noise
            self.Cxx_sigma_inv = np.linalg.inv(self.Cxx_sigma)
        else:
            raise NotImplementedError("Chosen method is not implemented")

        # Compute bayesian posterior mean of training data
        self.posterior_mean = self.Cxx_sigma_inv @ y_train

    def predict(self, X_test, return_cov=True):

        # Compute the cross-covariance matrix of the training and test data
        self.Cux = self._calculate_covmatrix(X_test, self.X_train)

        # Compute the mu (mean) as the product between the cross-covariance matrix and the posterior mean
        self.mu = self.Cux.T @ self.posterior_mean

        if return_cov is False:
            return self.mu

        # Compute the covariance matrix of the test data
        self.Cuu = self._calculate_covmatrix(X_test, X_test)

        # Compute covarience (Q)
        if self.method == "inv":
            self.Q = self.Cuu - self.Cux.T @ self.Cxx_sigma_inv @ self.Cux
        else:
            raise NotImplementedError("Chosen method is not implemented")

        return self.mu, self.Q

    def fit_and_predict(self, X_train, X_test, y_train=None, sigma_noise=0.1):
        self.fit(X_train, y_train, sigma_noise)
        return self.predict(X_test)
    