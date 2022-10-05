import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence

# voxels to big
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def one_hot_encoder(y_list, _encoder=None):
    '''
    takes a list of objects and returns a list of one-hot-encoded vectors
    '''


    if _encoder is not None:
        encoder = _encoder
        unique_labels = _encoder.keys()
    else:
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

    encoded_ys = []
    for y in y_list:
        zero_vector = [0 for i in range(len(unique_labels))]
        zero_vector[encoder[y]] = 1
        encoded_y = zero_vector
        encoded_ys.append(encoded_y)

    return encoded_ys, encoder

def collate_voxels(batch, add_noise=False, VOXEL_DATA=False):
    (xx, yy) = zip(*batch)
    x_lens = [x.shape for x in xx]
    y_lens = [y.shape for y in yy]

    x_max = max([x[0] for x in x_lens])
    y_max = max([y[1] for y in x_lens])
    if VOXEL_DATA:
        z_max = max([z[2] for z in x_lens])

    xx_pad = []
    yy_pad = []
    for i in range(len(xx)):
        x = xx[i]
        y = yy[i]
        if VOXEL_DATA:
            target = torch.zeros(x_max, y_max, z_max, 4)
            target[:x.shape[0],:x.shape[1],:x.shape[2],:] = x
        else:
            x = torch.t(x)
            target = torch.zeros(y_max, x_max)
            target[:x.shape[0], :x.shape[1]] = x[:,:]
            target = target[:,:,None]
        xx_pad.append(target.to(device))
        yy_pad.append(y.to(device))

        # TODO: test
        if add_noise:
            target = target + (0.1**0.5)*torch.randn(target.shape)


    yy_pad = torch.stack(yy).type(torch.float).to(device)
    xx_pad = torch.stack(xx_pad).type(torch.float).to(device)

    return xx_pad, yy_pad, x_lens, y_lens

def collate_point_cloud(batch, max_length=21200):
    (xx, yy) = zip(*batch)
    x_lens = [x.shape for x in xx]
    y_lens = [y.shape for y in yy]

    x_max=max_length
    y_max = max([y[1] for y in x_lens])

    xx_pad = []
    yy_pad = []
    for i in range(len(xx)):
        x = xx[i]
        y = yy[i]
        target = torch.zeros(x_max, y_max)
        target[:x.shape[0], :x.shape[1]] = x[:,:]
        target = target[None,:, :] # add batch dim
        xx_pad.append(target.to(device))
        yy_pad.append(y.to(device))

    yy_pad = torch.stack(yy).type(torch.float).to(device)
    xx_pad = torch.stack(xx_pad).type(torch.float).to(device)
    return xx_pad, yy_pad, x_lens, y_lens

def get_acc(y_pred, y_target, return_classes=False):
    pred_class = [pred.argmax() for pred in y_pred]
    target_class = [t.argmax() for t in y_target]
    correct = 0
    false = 0
    y_hat = np.zeros(y_pred.shape[0])
    y_true = np.zeros(y_pred.shape[0])
    for i in range(len(pred_class)):
        y_hat[i] = pred_class[i]
        y_true[i] = target_class[i]
        if pred_class[i] == target_class[i]:
            correct += 1
        else:
            false += 1
    acc = 100 * correct / (correct + false)

    if return_classes:
        return acc, y_hat, y_true
    else:
        return acc


def load_progress(model, state_dict, summary):
    '''Load previous training stage and results to resume training.
    Parameters
    ----------
    summary : str
        path to summary file

    model : str
        path to model state dict

    Returns
    -------
    model : pytroch model
        state of a model.

    summary, train_loss, test_loss : list
        summary and loss of training.
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(state_dict, map_location=device))

    train_loss = []
    test_loss = []
    # continue summary
    with open(summary, 'r') as f:
        summary = f.readlines()
        summary = [s.replace('\n', '') for s in summary]
        for line in summary:
            if line.startswith('Train Epoch'):
                train_loss.append(float(line.split(':')[2].split('\t')[0]))
                test_loss.append(float(line.split(':')[3].split('\t')[0]))
                EPOCH = int(line.split(':')[1].split('\t')[0])
    return model, summary, test_loss, train_loss, EPOCH