from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from utils.data.rectangles import generate_random_rectangle, generate_pts_from_normals

class RectData(Dataset):
    def __init__(self, num_samples=10000, size=500, num_seed=None):
        self.num_samples = num_samples
        self.size = size
        self.num_seed = num_seed
        self.x_dim = 2  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        self.data = [self.generate_data() for _ in tqdm(range(self.num_samples))]

    def generate_data(self):
        # Generate rectangle coordinates
        x, normals = generate_random_rectangle(self.size)

        # Generate target values for rect coordinates
        X, Y = generate_pts_from_normals(x, normals)

        return torch.Tensor(X), torch.Tensor(Y)

    def __getitem__(self, index):
      return self.data[index]

    def __len__(self):
        return self.num_samples