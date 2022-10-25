import numpy as np
import torch
from torch.utils.data import Dataset
import gzip
import matplotlib.pyplot as plt

class point_cloud_dataset(Dataset):
    def __init__(self, df, point_cloud_path):
        self.annotations = df
        self.point_cloud_path = point_cloud_path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        point_cloud_path = self.point_cloud_path + '/' + self.annotations.iloc[index, 0]
        protein = torch.tensor(np.loadtxt(point_cloud_path))
        y = torch.tensor(self.annotations.iloc[index, 2])
        return (protein, y)

