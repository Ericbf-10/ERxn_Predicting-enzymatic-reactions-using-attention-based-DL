import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
import matplotlib.pyplot as plt
from functions.pytorchtools import EarlyStopping, invoke, pad_collate
from functions.customDataset import RxnDataset
from torch.utils.data import Dataset


warnings.simplefilter("ignore")
print(torch.__version__)

script_path = os.path.dirname(__file__)
data_dir = os.path.join(script_path, '../data')
processed_data_dir = os.path.join(data_dir, 'processed')
pdb_files_path = os.path.join(data_dir, 'pdbs')
point_cloud_path = os.path.join(data_dir, 'point_cloud_dataset')
results_dir = os.path.join(script_path, '../results')
hyperparam_dir = os.path.join(results_dir, 'hyper_param_benchmark')

src_training_data_path = os.path.join(data_dir, 'mol_transformer/data/STEREO_mixed_augm/src-train.txt')
src_test_data_path = os.path.join(data_dir, 'mol_transformer/data/STEREO_mixed_augm/src-test.txt')
src_valid_data_path = os.path.join(data_dir, 'mol_transformer/data/STEREO_mixed_augm/src-val.txt')
tgt_training_data_path = os.path.join(data_dir, 'mol_transformer/data/STEREO_mixed_augm/tgt-train.txt')
tgt_test_data_path = os.path.join(data_dir, 'mol_transformer/data/STEREO_mixed_augm/tgt-test.txt')
tgt_valid_data_path = os.path.join(data_dir, 'mol_transformer/data/STEREO_mixed_augm/tgt-val.txt')

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

files = [src_training_data_path,
         src_test_data_path,
         src_valid_data_path,
         tgt_training_data_path,
         tgt_test_data_path,
         tgt_valid_data_path
        ]

def yield_tokens():
    for file in files:
        with open(file, 'r') as f:
            for example in f:
                tokens = example.replace('\n','').split(' ')
                yield tokens

token_generator = yield_tokens()
vocab = build_vocab_from_iterator(token_generator)


train_dataset = RxnDataset(src_training_data_path,
                           tgt_training_data_path,
                           vocab
                          )

test_dataset = RxnDataset(src_test_data_path,
                           tgt_test_data_path,
                           vocab
                          )

valid_dataset = RxnDataset(src_valid_data_path,
                           tgt_valid_data_path,
                           vocab
                          )

BATCH_SIZE = 256

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=pad_collate,
    shuffle=True,
    pin_memory=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=pad_collate,
    shuffle=True,
    pin_memory=False
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=pad_collate,
    shuffle=True,
    pin_memory=False
)


def one_hot_encoder(v: Tensor, vocab_size: int) -> Tensor:
    '''
    Takes tokenized sentences and one hot encodes
    them. Tokens have to be integer values.
    Args:
    -----
    v : Tensor
        shape (batch_size, seq_length)
    Out:
    ----
    out : Tensor
        shape (batch_size, seq_length, vocab_size)
    '''

    out = torch.zeros((v.size(0), v.size(1), vocab_size))
    for batch in range(v.size(0)):
        for i, token in enumerate(v[0, :]):
            out[batch, i, token] = 1

    return out


class Embedding(nn.Module):
    '''
    embeds sentence
    Args:
    -----
    vocab_size : int
        size of vocabulary

    embed_dim : int
        embedding dimension

    '''

    def __init__(self, vocab_size: int, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x) -> Tensor:
        '''
        forward pass
        Args:
        -----
        x : Tensor
            shape [batch_size, seq_length]

        Returns:
        --------
        out : Tensor
            shape [seq_length, batch_size, embed_dim]
        '''
        out = self.embed(x)  # (batch_size, seq_length, embed_dim)
        out = out.permute(1, 0, 2)  # (seq_length, batch_size, embed_dim)
        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, embed_dim: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SelfAttention(nn.Module):
    '''SelfAttention mechanism.
    Args:
    -----
    dim : int
        The out dimension of the query, key and value.
    n_heads : int
        Number of self-attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.
    '''

    def __init__(self, dim: int = 512, n_heads: int = 8, qkv_bias: bool = True,
                 attn_p: float = 0.1, proj_p: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x, mask: Tensor = None) -> Tensor:
        """Run forward pass.
        Args:
        -----
        x : Tensor
            shape [seq_len, batch_size, embedding_dim].
        Returns:
        --------
        x : Tensor
            x shape [seq_len, batch_size, embedding_dim].
        q, k, v : Tensor
            q, k, v shape [batch_size, n_heads, tgt_seq_length, head_dim]
        """
        seq_len, batch_size, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (seq_length, batch_size, 3 * dim)
        qkv = qkv.permute(1, 0, 2)  # (batch_size, seq_len, 3 * dim)

        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.n_heads, self.head_dim
        )  # (batch_size, seq_length + 1, 3, n_heads, head_dim)

        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, n_heads, seq_length, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # (batch_size, n_heads, seq_length, head_dim)

        dp = torch.einsum('ijkl,ijml->ijkm', q, k) * self.scale  # k_t @ q (batch_size, n_heads, seq_length, seq_length)

        if mask is not None:
            dp = dp.masked_fill(mask == float('-inf'), float('-inf'))

        scores = dp.softmax(dim=-1)  # (batch_size, n_heads, seq_length, seq_length)
        scores = self.attn_drop(scores)

        weighted_avg = torch.einsum('ijkl,ijlm->kijm', scores, v)  # (seq_len, batch_size, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (seq_length, batch_size, dim)

        x = self.proj(weighted_avg)  # (seq_length, batch_size, dim)
        x = self.proj_drop(x)  # (seq_length, batch_size, dim)

        return x, q, k, v


class EncoderDecoderAttention(nn.Module):
    '''SelfAttention mechanism.
    Args
    ----
    dim : int
        The out dimension of the query, key and value.
    n_heads : int
        Number of self-attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.
    '''

    def __init__(self, dim: int = 512, n_heads: int = 8, qkv_bias: bool = True,
                 attn_p: float = 0.1, proj_p: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_matrix = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x, k: Tensor = None, v: Tensor = None, mask: Tensor = None) -> Tensor:
        """Run forward pass.
        Args
        ----
        x : Tensor
            shape [seq_len, batch_size, embedding_dim].
        Returns
        -------
        x : Tensor
            x shape [seq_len, batch_size, embedding_dim].
        q, k, v : Tensor
            q, k, v shape [batch_size, n_heads, tgt_seq_length, head_dim]
        """
        seq_len, batch_size, dim = x.shape

        if dim != self.dim:
            raise ValueError

        q = self.q_matrix(x)  # (tgt_seq_length, batch_size, dim)
        q = q.permute(1, 0, 2)  # (batch_size, tgt_seq_length, dim)

        q = q.reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        )  # (batch_size, tgt_seq_len, n_heads, head_dim)

        q = q.permute(
            0, 2, 1, 3
        )  # (batch_size, n_heads, tgt_seq_length, head_dim)

        dp = torch.einsum('ijml,ijkl->ijmk', q,
                          k) * self.scale  # k_t @ q (batch_size, n_heads, tgt_seq_len, src_seq_len)

        if mask is not None:
            dp = dp.masked_fill(mask == float('-inf'), float('-inf'))

        scores = dp.softmax(dim=-1)  # (batch_size, n_heads, seq_length + 1, seq_length + 1)
        scores = self.attn_drop(scores)

        weighted_avg = torch.einsum('ijkl,ijlm->kijm', scores, v)  # (seq_length, batch_size, n_heads, head_dim)

        weighted_avg = weighted_avg.flatten(2)  # (seq_length, batch_size, dim)

        x = self.proj(weighted_avg)  # (seq_length, batch_size, dim)
        x = self.proj_drop(x)  # (seq_length, batch_size, dim)

        return x, q, k, v

class MLP(nn.Module):
    """Multilayer perceptron.
    Args
    ----
    in_features : int
        Number of input features.
    hidden_features : int
        Number of nodes in the hidden layer.
    out_features : int
        Number of output features.
    p : float
        Dropout probability.
    """

    def __init__(self, in_features: int = 512, hidden_features: int = 4*512,
                 out_features: int = 512, p: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x) -> Tensor:
        """Run forward pass.
        Args
        ----
        x : torch.Tensor
            Shape `(batch_size, n_patches + 1, in_features)`.
        Returns
        -------
        x : torch.Tensor
            Shape `(batch_size, n_patches +1, out_features)`
        """
        x = self.fc1(
            x
        )  # (batch_size, n_patches + 1, hidden_features)
        x = self.act(x)  # (batch_size, n_patches + 1, hidden_features)
        x = self.drop(x)  # (batch_size, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (batch_size, n_patches + 1, out_features)
        x = self.drop(x)  # (batch_size, n_patches + 1, out_features)

        return x


class EncoderBlock(nn.Module):
    """Transformer block.
    Parameters
    ----------
    dim : int
        Embeddinig dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attention
        Attention module.
    mlp : MLP
        MLP module.
    """

    def __init__(self, dim: int = 512, n_heads: int = 8, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, p: float = 0., attn_p: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(batch_size, n_patches + 1, dim)`.
        """
        attn, q, k, v = self.attn(x, mask)
        attn_add_norm = self.norm1(attn + x)
        z = self.mlp(attn_add_norm)
        out = self.norm2(z + attn_add_norm)

        return out, k, v


class DecoderBlock(nn.Module):
    """Transformer block.
    Parameters
    ----------
    dim : int
        Embeddinig dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attention
        Attention module.
    mlp : MLP
        MLP module.
    """

    def __init__(self, dim: int = 512, n_heads: int = 8, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, p: float = 0., attn_p: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = SelfAttention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.encoder_decoder_attn = EncoderDecoderAttention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor, k: Tensor, v: Tensor,
                mask_self_attn: Tensor = None, mask_cross_attn: Tensor = None) -> Tensor:
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(batch_size, n_patches + 1, dim)`.
        """
        attn, _, _, _ = self.self_attn(x, mask_self_attn)

        attn_add_norm = self.norm1(attn + x)
        attn, _, _, _ = self.encoder_decoder_attn(attn_add_norm, k, v, mask_cross_attn)
        attn_add_norm = self.norm2(attn + x)
        z = self.mlp(attn_add_norm)
        out = self.norm3(z + attn_add_norm)

        return out


class Transformer(nn.Module):
    """The enzyme transformer.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    encoder_depth : int
        Number of blocks.
    decoder_depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    """

    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            embed_dim=512,
            encoder_depth=8,
            decoder_depth=8,
            n_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
            src_masking=False,
            tgt_masking=True
    ):
        super().__init__()
        self.src_embedding = Embedding(vocab_size=src_vocab_size,
                                       embed_dim=embed_dim)
        self.tgt_embedding = Embedding(vocab_size=tgt_vocab_size,
                                       embed_dim=embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim=embed_dim,
                                               dropout=p)
        self.pos_drop = nn.Dropout(p=p)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(encoder_depth)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.src_masking = src_masking
        self.tgt_masking = tgt_masking

        self.head = nn.Linear(embed_dim, tgt_vocab_size)
        self.softmax = F.softmax
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_mask(self, s1: int, s2: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(s1, s2) * float('-inf'), diagonal=1)

    def forward(self, src, tgt):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, in_chans, num_atoms, num_encoding_dimensions)`.
        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(batch_size, n_classes)`.
        """
        if self.src_masking:
            src_mask = self.generate_mask(src.size(1), src.size(1)).to(self.device)
        else:
            src_mask = None

        if self.tgt_masking:
            tgt_mask_self_attn = self.generate_mask(tgt.size(1), tgt.size(1)).to(self.device)
            tgt_mask_cross_attn = self.generate_mask(tgt.size(1), src.size(1)).to(self.device)

        else:
            tgt_mask_self_attn = None
            tgt_mask_cross_attn = None

        src_embed = self.src_embedding(src)
        src = self.pos_encoding(src_embed)
        src = self.pos_drop(src)

        for block in self.encoder_blocks:
            src, k, v = block(src, mask=src_mask)

        tgt_embed = self.tgt_embedding(tgt)
        tgt = self.pos_encoding(tgt_embed)
        tgt = self.pos_drop(tgt)

        for block in self.decoder_blocks:
            tgt = block(tgt, k, v,
                        mask_self_attn=tgt_mask_self_attn,
                        mask_cross_attn=tgt_mask_cross_attn)

        out = self.head(tgt)
        out = self.softmax(out, dim=-1)
        out = out.permute(1, 0, 2)

        return out


def get_acc(out, tgt):
    '''
    calculate prediction accuracy
    '''

    out_cls = torch.argmax(out[-1], dim=1)
    tgt_cls = torch.argmax(tgt[-1], dim=1)

    matches = 0
    for i in range(len(out_cls)):
        if out_cls[i] == tgt_cls[i]:
            matches += 1

    return matches / len(out_cls)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = len(vocab)

model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    embed_dim=256,
    encoder_depth=6,
    decoder_depth=6,
    n_heads=8,
    mlp_ratio=4.,
    qkv_bias=True,
    p=0.,
    attn_p=0.,
    src_masking=False,
    tgt_masking=True
).to(device)

criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=10)
optimizer = torch.optim.Adam(model.parameters(),
                                         betas=(0.9, 0.998),
                                         lr=1e-3,
                                         weight_decay=0.01
                             )

num_epochs = 1000
train_loss, test_loss = [], []
summary = []
for epoch in range(num_epochs):
    batch_loss = 0
    model.train()
    for i, (src, tgt, _, _) in enumerate(train_loader):
        # attach to device
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()

        # forward + backward + optimize
        out = model(src, tgt)
        loss = criterion(out, one_hot_encoder(tgt, vocab_size).to(device))
        loss.backward()
        optimizer.step()
        batch_loss += loss.data
        print(out)

    train_loss.append(batch_loss / len(train_loader))

    batch_loss = 0
    model.eval()
    acc = 0
    for i, (src, tgt, _, _) in enumerate(test_loader):
        # attach to device
        src = src.to(device)
        tgt = tgt.to(device)

        pred = model(src, tgt)
        loss = criterion(pred, one_hot_encoder(tgt, vocab_size).to(device))
        batch_loss += loss.data

        acc += get_acc(pred, one_hot_encoder(tgt, vocab_size).to(device))

    test_loss.append(batch_loss / len(test_loader))
    acc = acc / len(test_loader)

    if epoch % (1) == 0:
        summary.append(
            'Train Epoch: {}\tLoss: {:.6f}\tTest Loss: {:.6f}\tTest Acc: {:.6f} %'.format(epoch, train_loss[-1],
                                                                                          test_loss[-1], acc))
        print('Train Epoch: {}\tLoss: {:.6f}\tTest Loss: {:.6f}\tTest Acc: {:.6f} %'.format(epoch, train_loss[-1],
                                                                                            test_loss[-1], acc))

    if invoke(early_stopping, test_loss[-1], model, implement=True):
        model.load_state_dict(
            torch.load(os.path.join(results_dir, '13_mol_transformer'),
                       map_location=device))
        summary.append(f'Early stopping after {epoch} epochs')

    torch.save(model.state_dict(), os.path.join(results_dir, '13_mol_transformer'))

    with open(os.path.join(results_dir, '13_summary.txt'), 'w') as f:
        for line in summary:
            f.write(str(line) + '\n')

    torch.save(model.state_dict(), os.path.join(results_dir, '13_mol_transformer'))

plot_file = '13_losses.png'

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
    plt.savefig(os.path.join(results_dir, plot_file))
    plt.close()


train_loss = [x.detach().cpu().numpy() if not type(x) == float else np.array(x, dtype='f') for x in train_loss]
test_loss = [x.detach().cpu().numpy() if not type(x) == float else np.array(x, dtype='f') for x in test_loss]
plot_losses(train_loss, test_loss)