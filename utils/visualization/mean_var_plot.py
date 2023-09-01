
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def gp_mean_variance_plot(Xs, Ys, mu, Q, shape, training_data=None, contour=False):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.set_title('Gaussian Process mean')

    p1 = ax1.pcolormesh(Xs,Ys,mu.reshape(shape))

    if training_data is not None:
      ax1.scatter(training_data[:,0],training_data[:,1])

    if contour:
      ax1.contour(Xs,Ys,mu.reshape(shape),levels=[0])

    ax1.set_aspect('equal')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p1,cax=cax1)

    # -------------------------------------------------------------------#

    ax2.set_title('Gaussian Process variance')

    p2 = ax2.pcolormesh(Xs,Ys,np.diag(Q).reshape(shape), cmap='plasma')

    if training_data is not None:
      ax2.scatter(training_data[:,0],training_data[:,1])

    if contour:
      ax1.contour(Xs,Ys,mu.reshape(shape),levels=[0])

    ax2.set_aspect('equal')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(p2,cax=cax2)

    plt.show()
    