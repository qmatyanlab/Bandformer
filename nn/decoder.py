import math
import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones

class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class ReciprocalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        dim = d_model // 6
        omega = torch.exp(torch.arange(dim) * -(math.log(100.0) / (dim - 1)))
        self.register_buffer('omega', omega)

    def forward(self, x):
        x = x[..., None] * self.omega * 2 * math.pi
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        x = x.view(x.size(0), x.size(1), -1)
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, attn_drop=0.0, pe_drop=0.0):
        super().__init__()
        assert d_model // 3 * 3 == d_model
        self.d_model = d_model
        self.pe = PositionalEncoding(d_model, pe_drop, max_len=128)
        self.kpe = ReciprocalPositionalEncoding(d_model, pe_drop)

        self.layers = _get_clones(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=int(d_model * 2),
                dropout=attn_drop,
                batch_first=True,
                norm_first=True,
                bias=True
            ),
            num_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, data, nodes, mask):
        x = self.kpe(data.kpoints)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(tgt=x, memory=nodes, memory_key_padding_mask=mask)
        x = self.norm(x)
        return x

class DecoderFFT(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, attn_drop=0.0, pe_drop=0.0):
        super().__init__()
        seq_len = 65  # 64 + 1 for the DC component
        assert d_model // 3 * 3 == d_model
        self.d_model = d_model
        self.pe = PositionalEncoding(d_model, pe_drop, max_len=seq_len)
        self.k_proj = nn.Linear(2 * d_model, d_model)
        self.kpe = ReciprocalPositionalEncoding(d_model, pe_drop)
        self.layers = _get_clones(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=int(d_model * 2),
                dropout=attn_drop,
                batch_first=True,
                norm_first=True,
                bias=True
            ),
            num_layers
        )
        self.norm = nn.LayerNorm(d_model)

        # self.kfc1 = nn.Linear(128, seq_len)
        # self.kfc2 = nn.Linear(3, d_model)

    def forward(self, data, nodes, mask):
        # x = data.kpoints.swapaxes(1, 2)
        # x = self.kfc1(x)
        # x = self.kfc2(x.swapaxes(1, 2))
        x = self.kpe(data.kpoints)
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        x = torch.cat([x.real, x.imag], dim=-1)
        x = self.k_proj(x)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(tgt=x, memory=nodes, memory_key_padding_mask=mask)

        x = self.norm(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    #     self.init_weights()
    #
    # def init_weights(self):
    #     initrange = 0.1
    #     self.fc1.bias.data.zero_()
    #     self.fc1.weight.data.uniform_(-initrange, initrange)
    #     self.fc2.bias.data.zero_()
    #     self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
