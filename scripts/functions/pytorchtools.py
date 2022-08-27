import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence

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


    encoded_ys = []
    for y in y_list:
        zero_vector = [0 for i in range(len(unique_labels))]
        zero_vector[encoder[y]] = 1
        encoded_y = zero_vector
        encoded_ys.append(encoded_y)

    return encoded_ys

def collate_point_cloud(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    data = pack_sequence(data, enforce_sorted=False)
    targets = [item[1] for item in batch]
    return [data, targets]

def collate_voxels(batch):
    (xx, yy) = zip(*batch)
    x_lens = [x.shape for x in xx]
    y_lens = [y.shape for y in yy]

    x_max = max([x[0] for x in x_lens])
    y_max = max([y[1] for y in x_lens])
    z_max = max([z[2] for z in x_lens])

    xx_pad = []
    yy_pad = []
    for i in range(len(xx)):
        x = xx[i]
        y = yy[i]
        target = torch.zeros(x_max, y_max, z_max, 4).float()
        target[:x.shape[0],:x.shape[1],:x.shape[2],:] = x.float()
        xx_pad.append(target)
        yy_pad.append(y.float())
        #target = target + (0.1**0.5)*torch.randn(target.shape)


    yy_pad = torch.stack(yy).to(device)
    xx_pad = torch.stack(xx_pad).to(device)

    return xx_pad, yy_pad, x_lens, y_lens
