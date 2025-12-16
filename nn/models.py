import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import to_dense_batch
import torch_geometric.nn.pool as pool

from .encoder import GATConvEncoder
from .decoder import Decoder, DecoderFFT, MLP

models = {}


def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def get_model(name, **args):
    net = models[name](**args)
    return net

@register_model('bandformer')
class Bandformer(nn.Module):
    def __init__(self, d_model, enc_heads, enc_num_layers, dec_heads, dec_num_layers, dropout):
        super().__init__()

        self.d_model = d_model
        self.enc_heads = enc_heads
        self.enc_num_layers = enc_num_layers
        self.dec_heads = dec_heads
        self.dec_num_layers = dec_num_layers
        self.dropout = dropout

        self.encoder = GATConvEncoder(
            d_model=self.d_model,
            num_heads=self.enc_heads,
            num_layers=self.enc_num_layers,
            dropout=self.dropout
        )

        # self.encoder = GraphAttentionEncoder(
        #     d_model=self.d_model,
        #     num_heads=self.enc_heads,
        #     num_layers=self.enc_num_layers,
        #     dropout=self.dropout
        # )

        self.decoder = DecoderFFT(
            d_model=self.d_model,
            num_heads=self.dec_heads,
            num_layers=self.dec_num_layers,
            attn_drop=self.dropout,
            pe_drop=self.dropout
        )

        self.mlp1 = MLP(self.d_model, self.d_model // 2, 6)
        self.mlp2 = MLP(self.d_model, self.d_model // 2, 6)

    def forward(self, data):
        x = self.encoder(data.x, data.edge_index, data.edge_len)
        x, mask = to_dense_batch(x, data.batch)
        x = self.decoder(data, x, ~mask)

        xr = self.mlp1(x).swapaxes(1, 2)
        xi = self.mlp2(x).swapaxes(1, 2)
        xi[:, :, 0] = 0
