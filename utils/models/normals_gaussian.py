from utils import apply_gaussian_covariance
from time import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import plotly.express as px

class NormalsGaussianProcess:
    def __init__(self, method="inv", kernel=apply_gaussian_covariance, verbose=True, R_val=False, R=None):
        self.method = method
        self.kernel = kernel
        self.verbose = verbose
        self.R_val = R_val
        self.R = R

    def _calculate_covmatrix(self, X1, X2, n_jobs=-1, save=False):
        if not self.R_val:
          return self.kernel(X1, X2, n_jobs)
        if save and self.R is None:
          val, self.R = self.kernel(X1, X2, n_jobs)
        elif self.R:
          val,_ = self.kernel(X1, X2, n_jobs, self.R)
        else:
           val,_ = self.kernel(X1, X2, n_jobs)
        return val

    def fit(self, X_train, dy_train, y_train=None, sigma_noise=0.01, n_jobs=-1):

        if self.verbose:
          print("Starting training...")
          start_fit = time()

        # Save training data
        self.X_train = X_train

        # Compute number of data points
        data_points = X_train.shape[0]

        # If no labels are given, assume zero mean
        if y_train is None:
            y_train = np.zeros(data_points)

        self.y_train = np.concatenate((y_train, dy_train.flatten('F')))

        # Compute the covariance matrix (Cxx)
        self.Cxx = self._calculate_covmatrix(X_train, X_train, n_jobs, True)
        # Compute diagonal noise matrix (σ²I)
        diagonal_noise = sigma_noise**2 * np.eye(self.Cxx.shape[0])

        # Compute matrix with diagonal noise (Cxx + σ²I)
        self.Cxx_sigma = self.Cxx + diagonal_noise

        if self.method == "inv":
            # Compute the inverse of the covariance matrix with diagonal noise
            self.Cxx_sigma_inv = np.linalg.inv(self.Cxx_sigma)

            # Compute bayesian posterior mean of training data
            self.posterior_mean = self.Cxx_sigma_inv @ self.y_train

        elif self.method == "cholesky":
            # Compute Cholesky factor
            self.cho = cho_factor(self.Cxx_sigma)

            # Compute bayesian posterior mean of training data
            self.posterior_mean = cho_solve(self.cho, self.y_train)

        else:
            raise NotImplementedError("Chosen method is not implemented")



        if self.verbose:
          end_fit = time()
          print(f"Finished training in {end_fit-start_fit} seconds")


    def predict(self, X_test, return_cov=True, n_jobs=-1):

        if self.verbose:
          print("Starting predictions...")
          start_pred = time()

        # Compute the cross-covariance matrix of the training and test data
        self.Cux = self._calculate_covmatrix(self.X_train, X_test, n_jobs)

        if self.verbose:
          end_cux = time()
          print(f"Computed cross-covariance matrix in {end_cux-start_pred} seconds")

        # Compute the mu (mean) as the product between the cross-covariance matrix and the posterior mean
        self.mu = self.Cux.T @ self.posterior_mean

        if self.verbose:
          end_mu = time()
          print(f"Computed mean in {end_mu-end_cux} seconds")

        if return_cov is False:
            return self.mu

        # Compute the covariance matrix of the test data
        self.Cuu = self._calculate_covmatrix(X_test, X_test, n_jobs)

        if self.verbose:
          end_cuu = time()
          print(f"Computed test data covariance matrix in {end_cuu-end_cux} seconds")

        # Compute covarience (Q)
        if self.method == "inv":
            self.Q = self.Cuu - self.Cux.T @ self.Cxx_sigma_inv @ self.Cux

        elif self.method == "cholesky":
            self.Q = self.Cuu - self.Cux.T @ cho_solve(self.cho, self.Cux)

        else:
            raise NotImplementedError("Chosen method is not implemented")

        if self.verbose:
          end_q = time()
          print(f"Computed variance in {end_q-end_cuu} seconds")

          end_total = time()
          print(f"Total time: {end_total-start_pred} seconds")


        return self.mu, self.Q

    def fit_and_predict(self, X_train, X_test, dy_train, y_train=None, sigma_noise=0.1, return_cov=True, n_jobs=-1):
        self.fit(X_train, dy_train, y_train, sigma_noise, n_jobs)
        return self.predict(X_test, return_cov, n_jobs)

    def visualize_cxx(self):
      if hasattr(self,'Cxx'):
        fig = px.imshow(self.Cxx)
        fig.show()

    def visualize_cux(self):
      if hasattr(self,'Cux'):
        fig = px.imshow(self.Cux)
        fig.show()

    def visualize_cuu(self):
      if hasattr(self,'Cuu'):
        fig = px.imshow(self.Cuu)
        fig.show()
