import torch
import torch.nn as nn
import json
from WeightedBatchNorm1d import *
import numpy as np

torch.manual_seed(0)


class PeriodicSetTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(PeriodicSetTransformerEncoder, self).__init__()
        self.embedding = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.out = nn.Linear(embedding_dim * num_heads, embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embedding_dim * num_heads, num_heads, batch_first=True)
        self.bn = WeightedBatchNorm1d(embedding_dim * num_heads)
        self.ln = torch.nn.LayerNorm(embedding_dim * num_heads)
        self.sp = nn.Softplus()
        self.sp2 = nn.Softplus()
        self.W_q = nn.Linear(embedding_dim * num_heads, embedding_dim * num_heads)
        self.W_k = nn.Linear(embedding_dim * num_heads, embedding_dim * num_heads)
        self.W_v = nn.Linear(embedding_dim * num_heads, embedding_dim * num_heads)

    def forward(self, x, weights):
        x = self.embedding(x)
        mask = torch.sum(torch.abs(x), dim=-1) == 0.
        att_output, att_weights = self.multihead_attention(self.W_q(x), self.W_k(x), self.W_v(x), key_padding_mask=mask)
        att_weights = att_weights * torch.transpose(weights, -2, -1)
        att_weights = att_weights / torch.sum(att_weights, -1, keepdim=True)
        att_output = torch.einsum('b i j , b j d -> b i d', att_weights, self.W_v(x))
        x = x + self.sp(att_output)
        #x = self.bn(x, weights)
        x = self.ln(x)
        return self.out(x)


class PeriodicSetTransformer(nn.Module):

    def __init__(self, initial_fea_len, embed_dim, num_heads, n_encoders=3, decoder_layers=1, components=["pdd", "composition"], expansion_size=10):
        super(PeriodicSetTransformer, self).__init__()
        self.embedding_layer = nn.Linear((initial_fea_len - 2) * expansion_size, embed_dim)
        self.composition = "composition" in components
        self.pdd_encoding = "pdd" in components
        self.comp_embedding_layer = nn.Linear(92, embed_dim)
        self.af = AtomFeaturizer()
        self.de = DistanceExpansion(size=expansion_size)
        self.ln = nn.LayerNorm(embed_dim)
        self.softplus = nn.Softplus()
        self.encoders = nn.ModuleList([PeriodicSetTransformerEncoder(embed_dim, num_heads) for _ in range(n_encoders)])
        self.decoder = nn.ModuleList([nn.Linear(embed_dim, embed_dim)
                                      for _ in range(decoder_layers - 1)])
        self.activations = nn.ModuleList([nn.Softplus()
                                          for _ in range(decoder_layers - 1)])
        #self.bn2 = nn.BatchNorm(embed_dim)
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, features):
        weights = features[:, :, 0, None]
        features = features[:, :, 1:]
        comp_features = self.af(features[:, :, -1:])
        comp_features = self.comp_embedding_layer(comp_features)
        str_features = features[:, :, :-1]
        str_features = self.embedding_layer(self.de(str_features))
        # x = comp_features + str_features
        if self.composition and self.pdd_encoding:
            x = self.ln(comp_features + str_features)
        elif self.composition:
            x = comp_features
        elif self.pdd_encoding:
            x = str_features
        x_init = x
        for encoder in self.encoders:
            x = encoder(x, weights)

        x = torch.sum(weights * (x + x_init), dim=1)
        x = self.ln(x)
        for layer, activation in zip(self.decoder, self.activations):
            x = layer(x)
            x = activation(x)

        return self.out(x)


class AtomFeaturizer(nn.Module):
    def __init__(self, id_prop_file="/home/jon/Desktop/pdd-graph-cgcnn/root_dir/atom_init.json"):
        super(AtomFeaturizer, self).__init__()
        with open(id_prop_file) as f:
            atom_fea = json.load(f)
        self.atom_fea = torch.Tensor(np.vstack([i for i in atom_fea.values()])).cuda()

    def forward(self, x):
        return torch.squeeze(self.atom_fea[x.long()])


class DistanceExpansion(nn.Module):
    def __init__(self, size=10):
        super(DistanceExpansion, self).__init__()
        self.size = size
        self.starter = torch.Tensor([i for i in range(size)]).cuda()
        self.starter /= size

    def forward(self, x):
        out = (1 - (x.flatten().reshape((-1, 1)) - self.starter)) ** 2
        return out.reshape((x.shape[0], x.shape[1], x.shape[2] * self.size))

