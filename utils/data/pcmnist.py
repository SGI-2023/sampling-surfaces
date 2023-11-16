import torch
from torchvision import datasets
import numpy as np
import gpytoolbox as gpy

from skimage import measure
from skimage.transform import rotate
from skimage.color.colorconv import rgb2gray, rgba2rgb
import gpytoolbox as gpy


from utils.data.rectangles import generate_pts_from_normals

def img2poly(polypic):
    """Export polylines from np.ndarray image
    
    Reads a png file and outputs a list of polylines that constitute the contours of the png, using marching squares. This is useful for generating 2D "realworld" data. 

    Parameters
    ----------
    polypic : np.ndarray
        numpy array that represents an image
    
    Returns
    -------
    poly : list of numpy double arrays
        Each list element is a matrix of ordered polyline vertex coordinates

    Notes
    -----
    This often results in "duplicate" polylines (one is the white->black contour, other is the black->white contour.
    """

    # For some reason reading the image flips it by 90 degrees. This fixes it
    polypic = rotate(polypic, angle=-90, resize=True)
   
    # convert to greyscale and remove alpha if neccessary
    if len(polypic.shape)>2:
        if polypic.shape[2]==4:
            polypic = rgb2gray(rgba2rgb(polypic))
        elif polypic.shape[2]==3:
            polypic = rgb2gray(polypic)
    # find contours
    polypic = polypic/np.max(polypic)
    poly = measure.find_contours(polypic, 0.5)
    return poly


class PointCloudMNISTDataset(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=True, num_samples=(50), random_seed=2, augment=False):
        super(PointCloudMNISTDataset, self).__init__(root, train=train, transform=transform, download=download)
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.augment = augment


    def __getitem__(self, index):
        img, target = super(PointCloudMNISTDataset, self).__getitem__(index)

        try:
        
            if self.transform is not None:
                img = self.transform(img)
                
            poly_all = img2poly(np.array(img))
            polys = np.concatenate(poly_all)

            polys = polys - np.min(polys)
            polys = polys/np.max(polys)
            polys = 0.5*polys + 0.25
            polys = 3*polys - 1.5
            Xs = []
            Ns = []
            start = 0

            # Calculate total length of all polygons
            total_length = sum([len(poly) for poly in poly_all])

            # Calculate number of samples for each polygon based on its proportion
            samples_per_poly = [int(len(poly) / total_length * self.num_samples) for poly in poly_all]

            # Adjust the last polygon's samples to make up for any lost samples
            samples_per_poly[-1] += self.num_samples - sum(samples_per_poly)

            for i in range(len(poly_all)):
                length = len(poly_all[i])
                poly = polys[start:start+length,:]
                start += length
                EC = gpy.edge_indices(poly.shape[0],closed=False)
                try:
                    if samples_per_poly[i] == 1:
                        X = np.asarray([poly[0]])
                        I = 0
                    else:
                        X,I,_ = gpy.random_points_on_mesh(poly, EC, samples_per_poly[i], return_indices=True,rng=np.random.default_rng(self.random_seed))
                except Exception as e:
                    print(e)
                    print(samples_per_poly[i])
                    print(poly.shape)
                    print(poly)
                    print(EC.shape)
                    raise e
                vecs = poly[EC[:,0],:] - poly[EC[:,1],:]
                vecs /= np.linalg.norm(vecs, axis=1)[:,None]
                J = np.array([[0., 1.], [-1., 0.]])
                N = vecs @ J.T

                if samples_per_poly[i] == 1:
                    N = np.asarray([N[I,:]])
                else:
                    N = N[I,:]
                if X.size > 0:
                    Xs.append(X)
                    Ns.append(N)

            try:
                coords = torch.tensor(np.concatenate(Xs), dtype=torch.float32)
            except Exception as e:
                print(e)
                print(Xs)
                print(len(Xs))
                raise e
            
            try:
                normals = torch.tensor(np.concatenate(Ns), dtype=torch.float32)
            except Exception as e:
                print(e)
                print(Ns)
                print(len(Ns))
                raise e
            
            if self.augment:
                coords, Y = generate_pts_from_normals(np.asarray(coords), np.asarray(normals))
                return torch.tensor(coords, dtype=torch.float32), torch.tensor(Y,dtype=torch.float32)
            
            Y = torch.zeros(coords.shape[0])

            return coords, Y, normals, target
        
        except Exception as e:
            print(f"Skipping invalid item due to: {e}")
            return self.__getitem__(index+1)

    