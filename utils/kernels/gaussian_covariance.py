import numpy as np
from joblib import Parallel, delayed

def gaussian_covariance_function(r, sigma_w=0.2, sigma_p=1.0, sigma_n=0):
    sigma_w = 1/(sigma_w**2.0)
    squared_distance = np.sum((r * r),axis=1)
    return  sigma_p*np.exp(-0.5*squared_distance*sigma_w) + sigma_n

def first_derivative_u_gaussian_covariance_function(r, base_kernel, i, j, sigma_w=0.2, sigma_n=0):
    sigma_w = 1/(sigma_w**2.0)
    return -sigma_w*r[:, i] * (base_kernel + sigma_n*(i==j))

def first_derivative_v_gaussian_covariance_function(r, base_kernel, i, j, sigma_w=0.2, sigma_n=0):
    sigma_w = 1/(sigma_w**2.0)

    return sigma_w*r[:, j] * (base_kernel + sigma_n*(i==j))

def second_derivative_gaussian_covariance_function(r, base_kernel, i, j, sigma_w=0.2, sigma_n=0):
    sigma_w = 1/(sigma_w**2.0)
    return (((i==j)*sigma_w - (r[:,i])*(r[:,j])) * (base_kernel + sigma_n*(i==j)))*sigma_w

def apply_gaussian_covariance(X1, X2, n_jobs=-1):
    # Define shapes
    n1, n2 = X1.shape[0], X2.shape[0]
    dims = X1.shape[1]+1

    # Format data with Kronecker product (https://en.wikipedia.org/wiki/Kronecker_product)
    formated_X1 = np.kron(np.ones((n2,1)),X1) # formated X1 is the product of each value of ones(n2,1) by X1 = [1*X1,...,1*X1]
    formated_X2 = np.kron(X2,np.ones((n1,1))) # formated X2 is the product of each value of X2 by one(n1,1) [a*ones(n1,1),...,z*ones(n1,1)]

    # Calculate r
    r = formated_X1 - formated_X2

    # Compute base kernel (Kg)
    base_kernel = gaussian_covariance_function(r)

    # Define a helper function for parallel processing
    def compute_cov(i, j):
        i -=1
        j -=1
        # Initial Position -> Default Covariance Function
        if i == -1 and j == -1:
            return np.reshape(base_kernel, (n1,n2), order='F')

        # First order for Initial Positions iters.
        elif i == -1:
            return np.reshape(first_derivative_v_gaussian_covariance_function(r, base_kernel, i, j), (n1,n2), order='F')
        elif j == -1:
            return np.reshape(first_derivative_u_gaussian_covariance_function(r, base_kernel, i, j), (n1,n2), order='F')

        # Second order derivatives
        else:
            return np.reshape(second_derivative_gaussian_covariance_function(r, base_kernel, i, j), (n1,n2), order='F')

    # Generate all pairs of dimensions
    dim_pairs = [(i, j) for i in range(dims) for j in range(dims)]

    # Compute covariances in parallel
    cov_vals = Parallel(n_jobs=n_jobs)(delayed(compute_cov)(i, j) for i, j in dim_pairs)

    # Reshape the list of covariances to the desired output shape
    cov_mats = np.vstack([np.hstack(cov_vals[i:i+dims]) for i in range(0, len(cov_vals), dims)])

    return cov_mats