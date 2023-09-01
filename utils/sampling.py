import gpytoolbox as gpy
import numpy as np
import matplotlib.pyplot as plt

def data_from_img(file_name, num_samples=(50), random_seed=2, plot=True):
    poly_all = gpy.png2poly(file_name)
    polys = np.concatenate(poly_all)

    if plot:
        for i in range(len(poly_all)):
            plt.plot(poly_all[i][:, 0], poly_all[i][:, 1], '-')
            _ = plt.axis('equal')
        plt.show()

    polys = polys - np.min(polys)
    polys = polys/np.max(polys)
    polys = 0.5*polys + 0.25
    polys = 3*polys - 1.5
    Xs = []
    Ns = []
    start = 0

    for i in range(len(poly_all)):
        length = len(poly_all[i])
        poly = polys[start:start+length,:]
        start += length
        EC = gpy.edge_indices(poly.shape[0],closed=False)
        X,I,_ = gpy.random_points_on_mesh(poly, EC, num_samples[i], return_indices=True,rng=np.random.default_rng(random_seed))
        vecs = poly[EC[:,0],:] - poly[EC[:,1],:]
        vecs /= np.linalg.norm(vecs, axis=1)[:,None]
        J = np.array([[0., -1.], [1., 0.]])
        N = vecs @ J.T
        N = N[I,:]
        Xs.append(X)
        Ns.append(N)
    return np.concatenate(Xs),np.concatenate(Ns)

def sample_surface(xs, ys, mean, variance, shape, training_points=None, normal_points=None, contour=True, plot=True, line_color='k'):
    random_sample = np.random.multivariate_normal(mean, variance)

    if plot:
        plt.pcolormesh(xs,ys,random_sample.reshape(shape),shading='gouraud',cmap='RdBu')
        plt.colorbar()

        if training_points is not None:
            plt.scatter(training_points[:,0],training_points[:,1])
            if normal_points is not None:
                plt.quiver(training_points[:,0],training_points[:,1],normal_points[:,0],normal_points[:,1])
        plt.axis("equal")
        plt.show()
        if contour:
            plt.contour(xs,ys,random_sample.reshape(shape),levels=[0], colors=line_color,linewidths=1, alpha=0.5)
            plt.axis("equal")

    return random_sample
