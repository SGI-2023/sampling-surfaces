import numpy as np
from joblib import Parallel, delayed

def thin_plate_covariance_kernel(r,r2, R2, logr2R2):
  # log_part = (r2 * (logr2R2- 1))
  return (r2 * (logr2R2- 1)) + R2 #kernel
def thin_plate_covariance_derivative_u(r, i, logr2R2):
  dc_dui =  2*(logr2R2) * (r[:,i])
  return dc_dui #1st order wrt u
def thin_plate_covariance_derivative_v(r, j,logr2R2):
  dc_dvj =  -2*(logr2R2) * (r[:,j])
  return dc_dvj #1st order wrt u
def thin_plate_covariance_derivative_uv(r, i, j,r2,logr2R2):
  d_dvj_dc_dui = 0
  #d_dvj_dc_dui = -4*r[:,i]*r[:,j]/r2 - (i==j)*2*logr2R2
  if i == j:
    d_dvj_dc_dui = -4*r[:,i]**2/r2-2*logr2R2
  else:
    d_dvj_dc_dui = -4*r[:,0]*r[:,1]/r2
  return d_dvj_dc_dui #2nd order u,v

def calculate_r(X1, X2):
    # Define shapes
    n1, n2 = X1.shape[0], X2.shape[0]
    dims = X1.shape[1]+1
    # Format data with Kronecker product (https://en.wikipedia.org/wiki/Kronecker_product)
    formated_X1 = np.kron(np.ones((n2,1)),X1) # formated X1 is the product of each value of ones(n2,1) by X1 = [1*X1,...,1*X1]
    formated_X2 = np.kron(X2,np.ones((n1,1))) # formated X2 is the product of each value of X2 by one(n1,1) [a*ones(n1,1),...,z*ones(n1,1)]

    # Calculate r
    r = formated_X1 - formated_X2
    return n1, n2, dims, r

def apply_thin_plate_spline_covariance(X1, X2, n_jobs=-1,R=None):
  n1,n2,dims,r = calculate_r(X1,X2)
  reg = 1e-15
  old_r = r
  # r[r==0] += reg
  r += reg
  r2 = r[:,0]**2 + r[:,1]**2
  R2 = 0
  if not R:
    R2 = np.max(r2)*2
    R = R2**0.5
  else:
    R2 = R**2
  logr2R2 =np.log(r2/R2)
  # Define a helper function for parallel processing
  def compute_cov(i, j):
      i -=1
      j -=1
      # Initial Position -> Default Covariance Function
      if i == -1 and j == -1:
          return np.reshape(thin_plate_covariance_kernel(r,r2, R2, logr2R2), (n1,n2), order='F')

      # First order for Initial Positions iters.
      elif i == -1:
          return np.reshape(thin_plate_covariance_derivative_v(r, j,logr2R2), (n1,n2), order='F')
      elif j == -1:
          return np.reshape(thin_plate_covariance_derivative_u(r, i, logr2R2), (n1,n2), order='F')

      # Second order derivatives
      else:
          return np.reshape(thin_plate_covariance_derivative_uv(r, i, j,r2, logr2R2), (n1,n2), order='F')
  # Generate all pairs of dimensions
  dim_pairs = [(i, j) for i in range(dims) for j in range(dims)]

  # Compute covariances in parallel
  cov_vals = Parallel(n_jobs=n_jobs)(delayed(compute_cov)(i, j) for i, j in dim_pairs)
  # Reshape the list of covariances to the desired output shape
  cov_mats = np.vstack([np.hstack(cov_vals[i:i+dims]) for i in range(0, len(cov_vals), dims)])

  return cov_mats, R