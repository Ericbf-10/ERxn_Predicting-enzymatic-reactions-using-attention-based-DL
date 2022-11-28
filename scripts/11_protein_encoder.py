import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import sys
sys.path.append('functions/')
from functions.protein_encoder import ProteinEncoder
from functions.pytorchtools import EarlyStopping, invoke, one_hot_encoder, collate_point_cloud, get_acc, load_progress
from functions.customDataset import point_cloud_dataset
from sklearn.metrics import matthews_corrcoef

# Define hyper parameters with Argument Parser
from argparse import ArgumentParser

parser = ArgumentParser(description="ProteinEncoder")

parser.add_argument("-lr", action="store", dest="LEARNING_RATE", type=float, default=0.000001, help="Learning Rate (default: 0.000001)")
parser.add_argument("-wd", action="store", dest="WEIGHT_DECAY", type=float, default=0.01, help="Weight Decay (default: 0.01)")
parser.add_argument("-epoch", action="store", dest="NUM_EPOCHS", type=int, default=1000, help="Numner of epochs (default: 1000)")
parser.add_argument("-pati", action="store", dest="PATIENCE", type=int, default=10, help="Patience (default: 10)")
parser.add_argument("-bs", action="store", dest="BATCH_SIZE", type=int, default=100, help="Batch Size (default: 100)")
parser.add_argument("-m", action="store", dest="MOMENTUM", type=float, default=0.9, help="Momentum (default: 0.9)")
parser.add_argument("-pin", action="store_false", dest="PIN_MEMORY", default=False, help="Pin Memory (default: False)")
parser.add_argument("-plen", action="store", dest="PATCH_LENGTH", type=int, default=400, help="Patch Length (default: 400)")
parser.add_argument("-embed", action="store", dest="EMBED_DIM", type=int, default=768, help="Embedding Dimension (default: 768)")
parser.add_argument("-depth", action="store", dest="DEPTH", type=int, default=12, help="Number of Blocks (default: 12)")
parser.add_argument("-heads", action="store", dest="N_HEADS", type=int, default=12, help="Number of attention heads (default: 12)")
parser.add_argument("-mlp", action="store", dest="MLP_RATIO", type=float, default=4.0, help="Hidden dimension of the MLP module (default: 4.0)")
parser.add_argument("-qkvbias", action="store_true", dest="QKV_BIAS", help="Include bias to Q, K and V projections (default: True)")
parser.add_argument("-p", action="store", dest="P_DROP", type=float, default=0.1, help="Dropout probability (default: 0.1)")
parser.add_argument("-attnp", action="store", dest="ATTN_P", type=float, default=0.1, help="Dropout probability (default: 0.1)")
parser.add_argument("-fout", action="store", dest="out_file", default="out", type=str, help="Output summary file")
parser.add_argument("-optim", action="store", dest="OPTIM", type=str, default="adam", help="Optimizer to use. Choose between: [sgd, adam] (default: adam)")

args = parser.parse_args()
out_file = args.out_file
plot_file = out_file + "_losses"

RESUME_TRAINING = False

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
point_cloud_path = os.path.join(data_dir, 'point_cloud_dataset')
results_dir = os.path.join(script_path, '../results')
hyperparam_dir = os.path.join(results_dir, 'hyper_param_benchmark')

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if not os.path.exists(hyperparam_dir):
    os.makedirs(hyperparam_dir)

# Choose data set
dataset_path = os.path.join(data_dir, 'datasets/09_balanced_data_set.csv')

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
dataset = pd.read_csv(dataset_path)
_, encoder = one_hot_encoder(dataset['EC'].to_list())

# get max length of sequence in dataset - for current data set 21166, set to 21200 in collate function
MAX_LENGTH = 10000 # Change that in the collate function as well if needed
# for file in dataset['protein']:
#     path = point_cloud_path + '/' + file
#     x = np.loadtxt(path)
#     length = x.shape[0]
#     if length > MAX_LENGTH:
#         MAX_LENGTH = length

# 80 - 15 - 5 split - with random seed
training_data = pd.read_csv(os.path.join(data_dir, 'datasets/09_train.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'datasets/09_test.csv'))
validation_data = pd.read_csv(os.path.join(data_dir, 'datasets/09_valid.csv'))

# one hot encoding of y-values
training_data['y'], _ = one_hot_encoder(training_data['EC'].to_list(), _encoder=encoder)
test_data['y'], _ = one_hot_encoder(test_data['EC'].to_list(), _encoder=encoder)
validation_data['y'], _ = one_hot_encoder(validation_data['EC'].to_list(), _encoder=encoder)
N_CLASSES = len(training_data['y'].to_list()[0])

# Hyper parameters
LEARNING_RATE = args.LEARNING_RATE
WEIGHT_DECAY = args.WEIGHT_DECAY
NUM_EPOCHS = args.NUM_EPOCHS
PATIENCE = args.PATIENCE
BATCH_SIZE = args.BATCH_SIZE
MOMENTUM = args.MOMENTUM
PIN_MEMORY = args.PIN_MEMORY
PATCH_LENGTH = args.PATCH_LENGTH
EMBED_DIM = args.EMBED_DIM
DEPTH = args.DEPTH
N_HEADS = args.N_HEADS
MLP_RATIO = args.MLP_RATIO
QKV_BIAS = args.QKV_BIAS
P_DROP = args.P_DROP
ATTN_P = args.ATTN_P
OPTIM = args.OPTIM

# dataset and data loader
train = point_cloud_dataset(df=training_data, point_cloud_path=point_cloud_path)
test = point_cloud_dataset(df=test_data, point_cloud_path=point_cloud_path)
valid = point_cloud_dataset(df=validation_data, point_cloud_path=point_cloud_path)

train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=BATCH_SIZE,
    collate_fn=collate_point_cloud,
    pin_memory=PIN_MEMORY,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test,
    batch_size=BATCH_SIZE,
    collate_fn=collate_point_cloud,
    pin_memory=PIN_MEMORY)

valid_loader = torch.utils.data.DataLoader(
    valid,
    batch_size=BATCH_SIZE,
    collate_fn=collate_point_cloud,
    pin_memory=PIN_MEMORY)

model = ProteinEncoder(
    enz_shape=(MAX_LENGTH,7),
    patch_length=PATCH_LENGTH,
    in_chans=1,
    n_classes=N_CLASSES,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    n_heads=N_HEADS,
    mlp_ratio=MLP_RATIO,
    qkv_bias=QKV_BIAS,
    p=P_DROP,
    attn_p=ATTN_P
).to(device)

#model.load_state_dict(torch.load(os.path.join(results_dir, '10_protein_autoencoder')))

## Training loop
# Loss
criterion = nn.CrossEntropyLoss()
# Optimizer
if OPTIM == "adam":
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
elif OPTIM == "sgd":
    optimizer = torch.optim.SGD(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY,
                            momentum=MOMENTUM)
else:
    sys.exit("Optimizer not available, please choose between sgd or adam.")

train_loss, test_loss = [], []

early_stopping = EarlyStopping(patience=PATIENCE)

summary = []
if RESUME_TRAINING:
    # load model state
    state_dict = os.path.join(results_dir, f'11_protein_encoder')
    summary_path = os.path.join(results_dir, '11_summary.txt')
    model, summary, test_loss, train_loss, EPOCH = load_progress(model, state_dict, summary_path)

else:
    EPOCH = 0


for epoch in range(EPOCH, NUM_EPOCHS):
    batch_loss = 0
    model.train()
    for i, (x_train, y_train, _, _) in enumerate(train_loader):
        # attach to device
        x_train = x_train.to(device)
        y_train = y_train.to(device).reshape([len(x_train), -1])
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, _ = model(x_train)
        x_train.detach()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        batch_loss += loss.data

    train_loss.append(batch_loss / len(train_loader))

    batch_loss = 0
    model.eval()
    acc = 0
    for i, (x_test, y_test, _, _) in enumerate(test_loader):
        # attach to device
        x_test = x_test.to(device)
        y_test = y_test.to(device).reshape([len(x_test), -1])

        pred, _ = model(x_test)
        loss = criterion(pred, y_test)
        batch_loss += loss.data

        acc += get_acc(pred, y_test)

    test_loss.append(batch_loss / len(test_loader))
    acc = acc / len(test_loader)
    # turn if condition on for real run
    if epoch % (1) == 0:
        summary.append('Train Epoch: {}\tLoss: {:.6f}\tTest Loss: {:.6f}\tTest Acc: {:.6f} %'.format(epoch, train_loss[-1], test_loss[-1], acc))
        print('Train Epoch: {}\tLoss: {:.6f}\tTest Loss: {:.6f}\tTest Acc: {:.6f} %'.format(epoch, train_loss[-1], test_loss[-1], acc))

    if invoke(early_stopping, test_loss[-1], model, implement=True):
        model.load_state_dict(torch.load(os.path.join(results_dir,'11_protein_encoder'), map_location=device))
        summary.append(f'Early stopping after {epoch} epochs')
        break

    torch.save(model.state_dict(), os.path.join(results_dir, f'11_protein_encoder'))

    with open(os.path.join(results_dir, '11_summary.txt'), 'w') as f:
        for line in summary:
            f.write(str(line) + '\n')

# performance evaluation
def plot_losses(train_loss, test_loss,burn_in=20):
    plt.figure(figsize=(15, 4))
    plt.plot(list(range(burn_in, len(train_loss))), train_loss[burn_in:], label='Training loss')
    plt.plot(list(range(burn_in, len(test_loss))), test_loss[burn_in:], label='Test loss')

    # find position of lowest testation loss
    minposs = test_loss.index(min(test_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Minimum Test Loss')

    plt.legend(frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(hyperparam_dir, plot_file))
    plt.close()


train_loss = [x.detach().cpu().numpy() if not type(x) == float else np.array(x, dtype='f') for x in train_loss]
test_loss = [x.detach().cpu().numpy() if not type(x) == float else np.array(x, dtype='f') for x in test_loss]
plot_losses(train_loss, test_loss)

with torch.no_grad():
    n_points = len(validation_data)

    model.eval()
    acc = 0
    for i, (x_valid, y_valid, _, _) in enumerate(valid_loader):
        # attach to device
        x_valid = x_valid.to(device)
        y_valid = y_valid.to(device).reshape([len(y_valid), -1])

        pred, _ = model(x_valid)

        accuracy, y_hat, y_true = get_acc(pred, y_valid, return_classes=True)
        if i == 0:
            y_class_pred = y_hat
            y_valid_class = y_true
        else:
            y_class_pred = np.concatenate((y_class_pred, y_hat))
            y_valid_class = np.concatenate((y_valid_class, y_true))
        acc += accuracy

acc = acc / len(valid_loader)
mcc = matthews_corrcoef(y_valid_class, y_class_pred)
summary.append('\nValidation Acc: ' + str(acc) + ' %')
summary.append('\nmathews correlation coefficient: ' + str(mcc))
print(mcc)

header = "The parameters used: " + "-lr=" + str(LEARNING_RATE) + "; -wd=" + str(WEIGHT_DECAY) + "; -epoch=" + str(NUM_EPOCHS) \
         + "; -pati=" + str(PATIENCE) + "; -bs=" + str(BATCH_SIZE) + "; -m=" + str(MOMENTUM) + "; -pin=" + str(PIN_MEMORY) \
         + "; -plen=" + str(PATCH_LENGTH) + "; -embed=" + str(EMBED_DIM) + "; -depth=" + str(DEPTH) + "; -heads=" + str(N_HEADS) \
         + "; -mlp=" + str(MLP_RATIO) + "; -qkvbias=" + str(QKV_BIAS) + "; -p=" + str(P_DROP) + "; -attnp=" + str(ATTN_P)

with open(os.path.join(hyperparam_dir, out_file + ".txt"), 'w') as f:
    f.write(header + '\n' + '\n')
    for line in summary:
        f.write(str(line) + '\n')