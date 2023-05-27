import torch
import torch.nn as nn

from WeightedBatchNorm1d import *

torch.manual_seed(0)


class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.out = nn.Linear(embedding_dim * num_heads, embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embedding_dim * num_heads, num_heads, batch_first=True)
        self.bn = WeightedBatchNorm1d(embedding_dim * num_heads)
        self.sp = nn.Softplus()

    def forward(self, x, weights):
        x = self.embedding(x)
        weighted_x = weights * x
        torch.abs(x)
        mask = torch.sum(torch.abs(x), dim=-1) == 0.
        att_output = self.multihead_attention(x, x, weighted_x, need_weights=False, key_padding_mask=mask)[0]
        x = x + self.sp(att_output)
        return self.out(self.bn(x, weights))


class PDDNet(nn.Module):

    def __init__(self, initial_fea_len, embed_dim, num_heads, n_encoders=3, decoder_layers=1):
        super(PDDNet, self).__init__()
        self.embedding_layer = nn.Linear(initial_fea_len - 1, embed_dim)
        self.encoders = nn.ModuleList([Encoder(embed_dim, num_heads) for _ in range(n_encoders)])
        self.decoder = nn.ModuleList([nn.Linear(embed_dim, embed_dim)
                                      for _ in range(decoder_layers - 1)])
        self.activations = nn.ModuleList([nn.Softplus()
                                          for _ in range(decoder_layers - 1)])
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, features):
        weights = features[:, :, 0, None]
        features = features[:, :, 1:]
        x = self.embedding_layer(features)
        for encoder in self.encoders:
            x = encoder(x, weights)

        x = torch.sum(weights * x, dim=1)

        for layer, activation in zip(self.decoder, self.activations):
            x = layer(x)
            x = activation(x)

        return self.out(x)
