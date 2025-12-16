import math
import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones
from torch_geometric.nn.conv import GATv2Conv, MessagePassing
from torch_geometric.utils import softmax


class GATConvEncoderLayer(nn.Module):
    def __init__(self, d_model, d_edge, n_heads, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_edge = d_edge
        self.conv = GATv2Conv(
            in_channels=d_model,
            out_channels=d_model // n_heads,
            heads=n_heads,
            edge_dim=d_edge,
            dropout=dropout,
            residual=True
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm(x)
        return x

class GATConvEncoderLayerManualResidual(nn.Module):
    def __init__(self, d_model, d_edge, n_heads, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_edge = d_edge
        self.conv = GATv2Conv(
            in_channels=d_model,
            out_channels=d_model // n_heads,
            heads=n_heads,
            edge_dim=d_edge,
            dropout=dropout,
            residual=False
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = x + self._sa_block((self.norm1(x)), edge_index, edge_attr)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class GATConvEncoder(nn.Module):
    num_species = 100
    cutoff = 8.0

    def __init__(self, d_model=384, d_edge=64, num_heads=1, num_layers=1, dropout=0.):
        super().__init__()
        self.edge_basis = GaussianBasis(start=0.0, end=self.cutoff, step=0.2)
        self.edge_embed = nn.Linear(self.edge_basis.num_basis, d_edge)
        self.atom_embed = Embeddings(self.num_species, d_model)
        self.layers = _get_clones(
            GATConvEncoderLayer(d_model, d_edge, num_heads, dropout),
            # GATConvEncoderLayerManualResidual(d_model, d_edge, num_heads, dropout),
            num_layers)

        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_embed(x)
        edge_attr = self.edge_basis(edge_attr)
        edge_attr = self.edge_embed(edge_attr)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        x = self.norm(x)
        return x

class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class GraphAttention(MessagePassing):
    def __init__(self, d_model, n_heads=1, qkv_bias=True, attn_drop=0.):
        super().__init__(node_dim=0, aggr='add')
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.fc_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_drop)

    def message(self, q_i, k_j, v_j, edge_attr, index):
        attn = (q_i * (k_j + edge_attr)).sum(dim=-1) * (self.d_head ** -0.5)
        attn = softmax(attn, index)
        attn = self.attn_drop(attn)
        out = v_j * attn.view(-1, self.n_heads, 1)
        return out

    def forward(self, x, edge_index, edge_attr):
        B, C = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.n_heads, self.d_head).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = self.propagate(edge_index, q=q, k=k, v=v, edge_attr=edge_attr).view(-1, C)
        x = self.fc_out(x)
        return x
    
# def gaussian_basis(x, start, end, step):
#     num_basis = int((end - start) / step) + 1
#     values = torch.linspace(start, end, num_basis, dtype=x.dtype, device=x.device)
#     diff = (x[..., None] - values) / step
#     return diff.pow(2).neg().exp().div(1.12)

class GaussianBasis(nn.Module):
    def __init__(self, start, end, step):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.num_basis = int((end - start) / step) + 1

    def forward(self, x):
        values = torch.linspace(self.start, self.end, self.num_basis, dtype=x.dtype, device=x.device)
        diff = (x[..., None] - values) / self.step
        return diff.pow(2).neg().exp().div(1.12)

class GraphAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias=True, dropout=0.):
        super().__init__()
        self.attn = GraphAttention(d_model, n_heads, qkv_bias, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self.linear1 = nn.Linear(d_model, d_model)
        # self.linear2 = nn.Linear(d_model, d_model)
        # self.dropout2 = nn.Dropout(dropout)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)
        # self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = x + self._sa_block(self.norm1(x), edge_index, edge_attr)
        # x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x, edge_index, edge_attr):
        x = self.attn(x, edge_index, edge_attr)
        return self.dropout1(x)

    # def _ff_block(self, x):
    #     x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    #     return self.dropout2(x)

class GraphAttentionEncoder(nn.Module):
    num_species = 100
    cutoff = 8.0

    def __init__(self, d_model=384, num_heads=1, num_layers=1, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.edge_basis = GaussianBasis(start=0.0, end=self.cutoff, step=0.2)
        self.edge_embed = nn.Linear(self.edge_basis.num_basis, d_model)
        self.atom_embed = Embeddings(self.num_species, d_model)
        self.layers = _get_clones(GraphAttentionEncoderLayer(d_model, num_heads, True, dropout), num_layers)

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_embed(x)
        edge_attr = self.edge_basis(edge_attr)
        edge_attr = self.edge_embed(edge_attr).view(-1, self.num_heads, self.d_model // self.num_heads)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x
