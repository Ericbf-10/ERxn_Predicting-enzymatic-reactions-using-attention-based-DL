import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from perceiver_pytorch import Perceiver
import sys
sys.path.append('functions/')
from functions.pytorchtools import EarlyStopping, invoke, one_hot_encoder, collate_voxels
from functions.customDataset import point_cloud_dataset, voxel_dataset
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
point_cloud_path = os.path.join(data_dir, 'point_cloud_dataset')
results_dir = os.path.join(script_path, '../results')

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Choose data set
dataset_path = os.path.join(data_dir, 'datasets/08_point_cloud_dataset.csv')

# use GPU if available - my data is to big
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# load dataset
dataset = pd.read_csv(dataset_path)

# one hot encoding of y-values
dataset['y'] = one_hot_encoder(dataset['EC'].to_list())
N_CLASSES = len(dataset['y'][0])

# 80 - 15 - 5 split - with random seed
training_data = dataset.sample(frac=0.8, random_state=1)
test_data = dataset.drop(training_data.index).sample(frac=0.15, random_state=1)
validation_data = dataset.drop(training_data.index).drop(test_data.index)

# Hyper parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 0.01
NUM_EPOCHS = 3000
BATCH_SIZE = 1
PIN_MEMORY = False

# dataset and data loader
#train = point_cloud_dataset(df=training_data, point_cloud_path=point_cloud_path)
train = voxel_dataset(df=training_data, point_cloud_path=point_cloud_path)
test = voxel_dataset(df=test_data, point_cloud_path=point_cloud_path)
valid = voxel_dataset(df=validation_data, point_cloud_path=point_cloud_path)

train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=BATCH_SIZE,
    collate_fn=collate_voxels,
    pin_memory=PIN_MEMORY)

test_loader = torch.utils.data.DataLoader(
    test,
    batch_size=len(test),
    collate_fn=collate_voxels,
    pin_memory=PIN_MEMORY)

valid_loader = torch.utils.data.DataLoader(
    valid,
    batch_size=len(valid),
    collate_fn=collate_voxels,
    pin_memory=PIN_MEMORY)

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
    num_classes = N_CLASSES,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    self_per_cross_attn = 2      # number of self attention blocks per cross attention
).to(device)

# training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
train_loss, test_loss = [], []

early_stopping = EarlyStopping(patience=PATIENCE)

summary = []
for epoch in range(NUM_EPOCHS):
    batch_loss = 0
    model.train()
    for i, (x_train, y_train, _, _) in enumerate(train_loader):
        # attach to device
        x_train = x_train.to(device)
        y_train = y_train.to(device).reshape([len(x_train), -1])
        optimizer.zero_grad()
        x_train.detach()

        # forward + backward + optimize
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        batch_loss += loss.data
    train_loss.append(batch_loss / len(train_loader))

    batch_loss = 0
    model.eval()
    for x_test, y_test in test_loader:
        # attach to device
        x_test = x_test.to(device)
        y_test = y_test.to(device).reshape([len(x_test), -1])

        pred = model(x_test)
        loss = criterion(pred, y_test)
        batch_loss += loss.data

    test_loss.append(batch_loss / len(test_loader))

    if epoch % (NUM_EPOCHS // 10) == 0:
        summary.append('Train Epoch: {}\tLoss: {:.6f}\tVal Loss: {:.6f}'.format(epoch, train_loss[-1], test_loss[-1]))
        print('Train Epoch: {}\tLoss: {:.6f}\tVal Loss: {:.6f}'.format(epoch, train_loss[-1], test_loss[-1]))

    if invoke(early_stopping, test_loss[-1], model, implement=True):
        model.load_state_dict(torch.load('checkpoint.pt'))
        summary.append(f'Early stopping after {epoch} epochs')
        break

    train_loss.append(train_loss)
    test_loss.append(test_loss)
    torch.save(model.state_dict(), os.path.join(results_dir, f'09_voxel_perceiver'))

# performance evaluation
def plot_losses(train_loss, test_loss, i,burn_in=20):
    plt.figure(figsize=(15, 4))
    plt.plot(list(range(burn_in, len(train_loss))), train_loss[burn_in:], label='Training loss')
    plt.plot(list(range(burn_in, len(test_loss))), test_loss[burn_in:], label='Test loss')

    # find position of lowest testation loss
    minposs = test_loss.index(min(test_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Minimum Test Loss')

    plt.legend(frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(results_dir, f'09_losses'))
    plt.close()


train_loss = [x.detach().cpu().numpy() for x in train_loss]
test_loss = [x.detach().cpu().numpy() for x in test_loss]
plot_losses(train_loss, test_loss)

with torch.no_grad():
    n_points = valid_loader.batch_size
    i = 0
    y_predictions = np.zeros(n_points)

    model.eval()

    for x_valid, y_valid in valid_loader:
        # attach to device
        x_valid = x_valid.to(device)
        y_valid = y_valid.to(device).reshape([len(y_valid), -1])

        pred = model(x_valid)
        y_predictions = pred.detach().cpu().numpy()
        loss = criterion(pred, y_valid)
        i += 1

    CLASS_THRESHOLD = 0.5
    y_pred_class = np.zeros(n_points)
    for pred in enumerate(y_predictions):
        y_pred_class[i] = [1 if x >= CLASS_THRESHOLD else 0 for x in pred]

    y_valid_class = np.where(y_valid.detach().cpu().numpy().flatten() >= CLASS_THRESHOLD, 1, 0)


mcc = matthews_corrcoef(y_valid_class, y_pred_class)
summary.append('\nmathews correlation coefficient: ' + str(mcc))
print(mcc)

# ROC and AUC
fpr, tpr, threshold = roc_curve(y_valid_class, y_pred_class)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label=f'AUC = {roc_auc}')
plt.legend(loc='best')
plt.plot(linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(results_dir, f'09_roc.png'))
plt.close()

with open(os.path.join(results_dir, '09_summary.txt'), 'w') as f:
    for line in summary:
        f.write(str(line) + '\n')