import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from perceiver_pytorch import Perceiver
import sys
sys.path.append('functions/')
from functions.pytorchtools import EarlyStopping, invoke, one_hot_encoder, my_collate
from functions.customDataset import point_cloud_dataset

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
dataset_path = os.path.join(data_dir, 'datasets/08_point_cloud_dataset.csv')
point_cloud_path = os.path.join(data_dir, 'point_cloud_dataset')

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
dataset = pd.read_csv(dataset_path)

# one hot encoding of y-values
dataset['y'] = one_hot_encoder(dataset['EC'].to_list())

# 80 - 15 - 5 split
training_data = dataset.sample(frac=0.8)
test_data = dataset.drop(training_data.index).sample(frac=0.15)
validation_data = dataset.drop(training_data.index).drop(test_data.index)


# dataset and data loader
train = point_cloud_dataset(df=training_data, point_cloud_path=point_cloud_path)

train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=10,
    collate_fn=my_collate,
    pin_memory=True)

for i, (x,y) in enumerate(train_loader):
    print(i, x,y)

#it = iter(train)
#first = next(it)
aasdf
model = Perceiver(
    input_channels = 1,          # number of channels for each token of the input - in my case 4 atom chanels
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video) - in my case 3 dimensional box
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

# example file
pdb_files = [f for f in os.listdir(pdb_files_path) if f.endswith('pdb.gz')]
example_file_name = np.random.choice(pdb_files)
example_file = os.path.join(pdb_files_path, example_file_name)
MICRO_BOX_SIZE = 15


# # get atom boxes
# atom_boxes = np.expand_dims(atom_boxes, axis=0)
# atom_boxes = torch.from_numpy(atom_boxes)
#
# print(atom_boxes.shape)
# #atom_boxes = torch.randn(1, 512, 512, 2, 4)
# print(model(atom_boxes))
# print(model(atom_boxes).shape)
#
# #img = torch.randn(1, 224, 224, 3) # 1 imagenet image, pixelized
#
# #model(img) # (1, 1000)