import numpy as np

def thin_plate_spline_kernel(X1, X2, reg=3e-7):
  X1_grid_x, X2_grid_x = np.meshgrid(X1[:,0], X2[:,0])
  X1_grid_y, X2_grid_y = np.meshgrid(X1[:,1], X2[:,1])

  r = np.sqrt((X1_grid_x - X2_grid_x+reg)**2 + (X1_grid_y - X2_grid_y + reg)**2)
  R = np.max(r) * 50

  return 2*r**2 * np.log(r) - (1 + 2*np.log(R)) * r**2 + R**2
