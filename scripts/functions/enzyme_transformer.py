import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split enzyme into patches and then embed them.
    Parameters
    ----------
    enz_shape : tuple
        Shape of the enzyme encoding '(number of atom + padding x x,y,z coodinates + atomic encoding)'.
    patch_length : int
        Number of atoms per patch.
    in_chans : int
        Number of input channels. (potentially interessting for future applications,
        like including hydrophobicity, charge, ...)
    embed_dim : int
        The emmbedding dimension.
    Attributes
    ----------
    n_patches : int
        Number of patches of enzyme.
    patch_size : tuple
        Shape of patch (patch_size, enz_shape[1])
    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """

    def __init__(self, enz_shape, patch_length, in_chans=1, embed_dim=768):
        super().__init__()
        self.enz_shape = enz_shape
        self.patch_size = (patch_length, enz_shape[1])
        self.n_patches = enz_shape[0] // patch_length

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=patch_length,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, enz_shape[0], enz_shape[1])`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(
            x
        )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class OutEmbed(nn.Module):
    """Embed output.
    Parameters
    ----------
    out_shape : tuple
        Shape of output.
    patch_length : int
        Number of words per patch.
    in_chans : int
        Number of input channels.
    embed_dim : int
        The emmbedding dimension.
    Attributes
    ----------
    n_patches : int
        Number of patches (default 1, one word at a time).
    patch_size : tuple
        Shape of patch (patch_size, enz_shape[1])
    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """

    def __init__(self, out_shape, patch_length=1, in_chans=1, embed_dim=768):
        super().__init__()
        self.out_shape = out_shape
        self.patch_size = (patch_length, out_shape[1])
        self.n_patches = out_shape[0] // patch_length

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=patch_length,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, enz_shape[0], enz_shape[1])`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = torch.unsqueeze(x, 1)  # adds channel dimension
        x = self.proj(
            x
        )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class SelfAttention(nn.Module):
    """SelfAttention mechanism.
    Parameters
    ----------
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
    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all self-attention
        heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim=768, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
                     q @ k_t
             ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x, q, k, v


class EncoderDecoderAttention(nn.Module):
    """Encoder-Decoder Attention mechanism.
    Parameters
    ----------
    dim : int
        The out dimension of the query, key and value.
    n_heads : int
        Number of encoder-decoder-attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.
    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all encoder-decoder-attention
        heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim=768, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x, q, k, v):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
                     q @ k_t
             ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    """Multilayer perceptron.
    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Number of nodes in the hidden layer.
    out_features : int
        Number of output features.
    p : float
        Dropout probability.
    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.
    act : nn.GELU
        GELU activation function.
    fc2 : nn.Linear
        The second linear layer.
    drop : nn.Dropout
        Dropout layer.
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(
            x
        )  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features)

        return x


class EncoderBlock(nn.Module):
    """Encoder block.
    Parameters
    ----------
    dim : int
        Embeddinig dimension.
    n_heads : int
        Number of self-attention heads.
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
    attn : SelfAttention
        SelfAttention module.
    mlp : MLP
        MLP module.
    """

    def __init__(self, dim=768, n_heads=12, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
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

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x_ = x.clone()
        x, q, k, v = self.attn(self.norm1(x))
        x = x + x_
        x = x + self.mlp(self.norm2(x))

        return x, k, v


class DecoderBlock(nn.Module):
    """Decoder block.
    Parameters
    ----------
    dim : int
        Embeddinig dimension.
    n_heads : int
        Number of self-attention heads.(same for self-attention and encoder-
        decoder-attention)
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
    attn : SelfAttention
        SelfAttention module.
    mlp : MLP
        MLP module.
    """

    def __init__(self, dim=768, n_heads=12, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.enocder_decoder_attn = EncoderDecoderAttention(
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

    def forward(self, x, k, v):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x_ = x.clone()
        x, q, _, _ = self.attn(self.norm1(x))
        x = x + x_
        x = x + self.enocder_decoder_attn(self.norm1(x), q, k, v)
        x = x + self.mlp(self.norm2(x))

        return x


class EnzymeTransformer(nn.Module):
    """The enzyme transformer.
    Parameters
    ----------
    enz_shape : tuple
        Shape of enzyme encoding.
    query_len : int
        Length of output query
    patch_length : int
        Number of atoms per embedding.
    in_chans : int
        Number of input channels.
    vocab_size : int
        Size of vocabulary.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    encoder_depth : int
        Number of encoder blocks.
    decoder_depth : int
        Number of decoder blocks.
    n_heads : int
        Number of self-attention heads (same for encoder and decoder).
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.
    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.
    pos_drop : nn.Dropout
        Dropout layer.
    blocks : nn.ModuleList
        List of `Block` modules.
    norm : nn.LayerNorm
        Layer normalization.
    """

    def __init__(
            self,
            enz_shape,
            patch_length=20,
            in_chans=1,
            vocab_size=1000,
            embed_dim=768,
            encoder_depth=6,
            decoder_depth=6,
            n_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            enz_shape=enz_shape,
            patch_length=patch_length,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.out_embed = OutEmbed(
            out_shape=(query_len, vocab_size),
            patch_length=1,
            in_chans=1,
            embed_dim=embed_dim,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, embed_dim)
        )  # learned embedding
        self.pos_embed_out = nn.Parameter(
            torch.zeros(1, self.out_embed.n_patches, embed_dim)
        )  # learned embedding

        self.pos_drop = nn.Dropout(p=p)

        self.encoderblocks = nn.ModuleList(
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

        self.decoderblocks = nn.ModuleList(
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
        self.norm = nn.LayerNorm(embed_dim * query_len, eps=1e-6)  # normalize final output of decoder
        self.head = nn.Linear(embed_dim * query_len, vocab_size)  # final layer
        self.softmax = nn.functional.softmax
        self.eos_token = torch.tensor([0 if x < vocab_size - 1 else 1 for x in range(vocab_size)])

    def forward(self, x, out):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, enz_shape[0], enz_shape[1])`.
        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]

        # Encoder
        x = self.patch_embed(x)
        x = x + self.pos_embed  # (n_samples, n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.encoderblocks:
            x, k, v = block(x)
        # Decoder - repeat until max sentence length has been reached or break when stop token was outputted
        # optional: add masking
        i = 0
        while i <= out.shape[1] - 1:
            x = self.out_embed(out)
            # x = self.out_embed(torch.randn((out.shape[0],out.shape[1])))
            x = x + self.pos_embed_out  # (n_samples, n_patches, embed_dim)
            x = self.pos_drop(x)
            for block in self.decoderblocks:
                x = block(x, k, v)
            x = x.flatten(start_dim=1)
            x = self.head(self.norm(x))
            x = self.softmax(x, dim=-1)
            print(out.shape)
            out[:, i, :] = x
            i += 1
            # TODO: break if end of sentence token is predicted
            # if x_.argmax() == self.eos_token.argmax():
            #    break

        return out