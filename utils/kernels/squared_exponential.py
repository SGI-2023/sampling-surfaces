import numpy as np

def squared_exponential_kernel(y, x, length_scale=1.0, sigma=1.0):
    # Compute the squared Euclidean distance between every pair of points
    # ||x-y||² = ||x||² + ||y||² - 2x·y
    sqdist = np.sum(x**2, 1).reshape(-1, 1) + np.sum(y**2, 1) - 2 * np.dot(x, y.T)

    # Compute the kernel
    # K(x, y) = σ² exp(-0.5 / l² * ||x-y||²)
    kernel = sigma**2 * np.exp(-0.5 / length_scale**2 * sqdist)
    return kernel
