import numpy as np
import torch
from torch.utils.data import Dataset
import gzip
import matplotlib.pyplot as plt

# open pdb file, save coordinates and atom types
def get_protein_voxels(pdb):
    '''
    takes a pdb file and saves the atom coordinates of carbon, oxygen,
    nitrogen and sulfur atoms as stacked 3 dimensional np arrays
    '''
    xs = []
    ys = []
    zs = []
    atom_types = []
    CAs = []
    atoms = ['C', 'O', 'N', 'S', 'CA']
    with gzip.open(pdb, 'rb') as f:
        i = 0
        for line in f:
            if line.startswith(b'ATOM'):
                line = line.decode()
                xs.append(round(float(line[30:38]))) # x
                ys.append(round(float(line[38:46]))) # y
                zs.append(round(float(line[46:54]))) # z
                if line[13] == atoms[0]: atom_types.append(0)
                if line[13] == atoms[1]: atom_types.append(1)
                if line[13] == atoms[2]: atom_types.append(2)
                if line[13] == atoms[3]: atom_types.append(3)
                if line[13:15] == atoms[4]: CAs.append(i)
                i += 1

    # move the origin, so there are no negative coordinates
    xs = [x - min(xs) for x in xs]
    ys = [y - min(ys) for y in ys]
    zs = [z - min(zs) for z in zs]

    max_x = max(xs) + 1
    max_y = max(ys) + 1
    max_z = max(zs) + 1

    # create the boxes
    carbon_box = np.zeros((max_x, max_y, max_z), dtype=int)
    oxygen_box = np.zeros((max_x, max_y, max_z), dtype=int)
    nitrogen_box = np.zeros((max_x, max_y, max_z), dtype=int)
    sulfur_box = np.zeros((max_x, max_y, max_z), dtype=int)

    for i, atom_type in enumerate(atom_types):
        if atom_type == 0:
            carbon_box[xs[i], ys[i], zs[i]] = 1
        if atom_type == 1:
            oxygen_box[xs[i], ys[i], zs[i]] = 1
        if atom_type == 2:
            nitrogen_box[xs[i], ys[i], zs[i]] = 1
        if atom_type == 3:
            sulfur_box[xs[i], ys[i], zs[i]] = 1

    atom_boxes = np.stack([carbon_box, oxygen_box, nitrogen_box, sulfur_box], axis=3)
    return atom_boxes

class point_cloud_dataset(Dataset):
    def __init__(self, df, point_cloud_path):
        self.annotations = df
        self.point_cloud_path = point_cloud_path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        point_cloud_path = self.point_cloud_path + '/' + self.annotations.iloc[index, 0]
        protein = torch.tensor(np.loadtxt(point_cloud_path))
        label = torch.tensor(self.annotations.iloc[index, 2])
        return (protein, label)
