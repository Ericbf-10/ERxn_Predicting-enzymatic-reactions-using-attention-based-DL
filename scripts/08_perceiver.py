import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import torch
from perceiver_pytorch import Perceiver
import pandas as pd

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
dataset_path = os.path.join(data_dir, 'datasets/07_protein_voxelization_dataset.csv')

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

def one_hot_encoder(y_list):
    '''
    takes a list of objects and returns a list of one-hot-encoded vectors
    '''
    unique_labels = []
    for y in y_list:
        if y in unique_labels:
            pass
        else:
            unique_labels.append(y)
    unique_labels = sorted(unique_labels)

    encoder = {}
    for i, y in enumerate(unique_labels):
        if y in encoder.keys():
            pass
        else:
            encoder[y] = i

    zero_vector = [0 for i in range(len(unique_labels))]
    encoded_ys = []
    for y in y_list:
        zero_vector[encoder[y]] = 1

        encoded_y = zero_vector
        encoded_ys.append(encoded_y)
        zero_vector[encoder[y]] = 0

    return encoded_ys

model = Perceiver(
    input_channels = 4,          # number of channels for each token of the input - in my case 4 atom chanels
    input_axis = 3,              # number of axis for input data (2 for images, 3 for video) - in my case 3 dimensional box
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                 #   depth * (cross attention -> self_per_cross_attn * self attention)
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,         # number of dimensions per cross attention head
    latent_dim_head = 64,        # number of dimensions per latent self attention head
    num_classes = 1000,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    self_per_cross_attn = 2      # number of self attention blocks per cross attention
)

# load dataset
dataset = pd.read_csv(dataset_path)

# one hot encoding of y-values
dataset['y'] = one_hot_encoder(dataset['EC'].to_list())
print(dataset)
asdf

# 80 - 15 - 5 split
training_data = dataset.sample(frac=0.8)
test_data = dataset.drop(training_data.index).sample(frac=0.15)
validation_data = dataset.drop(training_data.index).drop(test_data.index)

print(dataset)
print(test_data)
print(validation_data)
asdf

# example file
pdb_files = [f for f in os.listdir(pdb_files_path) if f.endswith('pdb.gz')]
example_file_name = np.random.choice(pdb_files)
example_file = os.path.join(pdb_files_path, example_file_name)
MICRO_BOX_SIZE = 15


# get atom boxes
atom_boxes = get_protein_voxels(example_file)





atom_boxes = np.expand_dims(atom_boxes, axis=0)
atom_boxes = torch.from_numpy(atom_boxes)

print(atom_boxes.shape)
#atom_boxes = torch.randn(1, 512, 512, 2, 4)
print(model(atom_boxes))
print(model(atom_boxes).shape)

#img = torch.randn(1, 224, 224, 3) # 1 imagenet image, pixelized

#model(img) # (1, 1000)