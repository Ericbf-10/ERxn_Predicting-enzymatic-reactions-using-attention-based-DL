import numpy as np
import torch
from torch.utils.data import Dataset
import linecache

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

class RxnDataset(Dataset):
    def __init__(self, src_path, tgt_path, vocab):
        self.src_data_path = src_path
        self.tgt_data_path = tgt_path
        self.vocab = vocab

    def __len__(self):
        with open(self.src_data_path, 'r') as f:
            return len(f.readlines())

    def __getitem__(self, index, sos_token=0):
        src_path = self.src_data_path
        tgt_path = self.tgt_data_path
        vocab = self.vocab

        src = linecache.getline(src_path, index + 1)  # linecache indexing starts at 1 for some reason
        tgt = linecache.getline(tgt_path, index + 1)

        src = torch.tensor(
            [vocab[token] for token in src.replace('\n', '').split(' ')]
        ).to(int)

        tgt = torch.tensor(
            [vocab[token] for token in tgt.replace('\n', '').split(' ')]
        )
        sos_token = torch.Tensor([sos_token])
        tgt = torch.concat((sos_token, tgt), dim=0).to(int)
        return (src, tgt)