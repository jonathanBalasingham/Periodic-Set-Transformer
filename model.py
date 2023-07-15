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
        self.pre_norm = nn.LayerNorm(embedding_dim)
        #self.ln = torch.nn.LayerNorm(embedding_dim * num_heads)
        self.ln = torch.nn.LayerNorm(embedding_dim)
        self.sp = nn.Softplus()
        self.sp2 = nn.Softplus()
        #self.W_q = nn.Linear(embedding_dim * num_heads, embedding_dim * num_heads)
        #self.W_k = nn.Linear(embedding_dim * num_heads, embedding_dim * num_heads)
        #self.W_v = nn.Linear(embedding_dim * num_heads, embedding_dim * num_heads)
        self.W_q = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.W_k = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.W_v = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.ffn = nn.Linear(embedding_dim, embedding_dim)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                 nn.Softplus())

    def forward(self, x, weights):
        x_norm = self.ln(x)
        mask = torch.sum(weights, dim=-1) == 0.
        att_output, att_weights = self.multihead_attention(self.W_q(x_norm), self.W_k(x_norm), self.W_v(x_norm), key_padding_mask=mask)
        att_weights = att_weights * torch.transpose(weights, -2, -1)
        att_weights = att_weights / torch.sum(att_weights, -1, keepdim=True)
        att_output = torch.einsum('b i j , b j d -> b i d', att_weights, self.W_v(x_norm))

        output1 = x + self.out(att_output)
        output2 = self.ln(output1)
        output2 = self.ffn(output2)
        return self.ln(output1 + output2)


class PeriodicSetTransformer(nn.Module):

    def __init__(self, str_fea_len, embed_dim, num_heads, n_encoders=3, decoder_layers=1, components=["pdd", "composition"], expansion_size=10):
        super(PeriodicSetTransformer, self).__init__()
        self.embedding_layer = nn.Linear((str_fea_len - 1) * 10, embed_dim)
        #self.embedding_layer = nn.Linear((str_fea_len - 1), embed_dim)
        self.composition = "composition" in components
        self.pdd_encoding = "pdd" in components
        self.comp_embedding_layer = nn.Linear(92, embed_dim)
        self.af = AtomFeaturizer()
        self.de = DistanceExpansion(size=expansion_size)
        self.ln = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cell_embed = nn.Linear(6, 32)
        self.softplus = nn.Softplus()
        self.encoders = nn.ModuleList([PeriodicSetTransformerEncoder(embed_dim, num_heads) for _ in range(n_encoders)])
        self.decoder = nn.ModuleList([nn.Linear(embed_dim, embed_dim)
                                      for _ in range(decoder_layers - 1)])
        self.activations = nn.ModuleList([nn.Softplus()
                                          for _ in range(decoder_layers - 1)])
        #self.bn2 = nn.BatchNorm(embed_dim)
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, features, mode="pretrain"):
        str_fea, comp_fea, cell_fea = features
        weights = str_fea[:, :, 0, None]
        comp_features = self.af(comp_fea)
        comp_features = self.comp_embedding_layer(comp_features)
        str_features = str_fea[:, :, 1:]
        str_features = self.embedding_layer(self.de(str_features))
        #str_features = self.embedding_layer(str_features)
        # x = comp_features + str_features
        if self.composition and self.pdd_encoding:
            #x = self.ln(comp_features + str_features)
            x = comp_features + str_features
        elif self.composition:
            x = comp_features
        elif self.pdd_encoding:
            x = str_features
        x_init = x
        for encoder in self.encoders:
            x = encoder(x, weights)

        x = torch.sum(weights * (x + x_init), dim=1)
        #x = torch.concat([x, self.cell_embed(cell_fea)], dim=1)
        x = self.ln2(x)
        for layer, activation in zip(self.decoder, self.activations):
            x = layer(x)
            x = activation(x)

        return self.out(x)



class PeSTEncoder(nn.Module):

    def __init__(self, str_fea_len, embed_dim, num_heads, n_encoders=3, expansion_size=10):
        super(PeSTEncoder, self).__init__()
        self.embedding_layer = nn.Linear((str_fea_len - 1) * expansion_size, embed_dim)
        self.comp_embedding_layer = nn.Linear(92, embed_dim)
        self.af = AtomFeaturizer()
        self.de = DistanceExpansion(size=expansion_size)
        self.ln = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim + 6)
        self.softplus = nn.Softplus()
        self.encoders = nn.ModuleList([PeriodicSetTransformerEncoder(embed_dim, num_heads) for _ in range(n_encoders)])

    def forward(self, features, pool=False):
        str_fea, comp_fea, cell_fea = features
        weights = str_fea[:, :, 0, None]
        comp_features = self.af(comp_fea)
        comp_features = self.comp_embedding_layer(comp_features)
        str_features = str_fea[:, :, 1:]
        str_features = self.embedding_layer(self.de(str_features))
        # x = comp_features + str_features
        x = self.ln(comp_features + str_features)
        for encoder in self.encoders:
            x = encoder(x, weights)

        if pool:
            return torch.sum(weights * x, dim=1)

        return weights, x


class AtomFeaturizer(nn.Module):
    def __init__(self, id_prop_file="atom_init.json"):
        super(AtomFeaturizer, self).__init__()
        with open(id_prop_file) as f:
            atom_fea = json.load(f)
        af = np.vstack([i for i in atom_fea.values()])
        af = np.vstack([np.zeros(92), af, np.ones(92)])  # last is the mask, first is for padding
        self.atom_fea = torch.Tensor(af).cuda()

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


class ElementMasker(nn.Module):
    def __init__(self):
        super(ElementMasker, self).__init__()

    def forward(self, input, masked_values, mask_type="composition"):
        x = input.clone()
        if mask_type == "composition":
            x[torch.arange(x.shape[0]), masked_values] = -1  # depends on AtomFeaturizer
        else:
            x[torch.arange(x.shape[0]), masked_values, 1:] = -1
        return x


class CompositionDecoder(nn.Module):
    """
    100 possible elements

    """
    def __init__(self, input_dim, predict_indv_props=True):
        super(CompositionDecoder, self).__init__()
        self.pip = predict_indv_props
        if predict_indv_props:
            self.dense = nn.Linear(input_dim, 92)
        else:
            self.dense = nn.Linear(input_dim, 100)
        self.group_num = nn.Softmax(dim=-1)
        self.period_num = nn.Softmax(dim=-1)
        self.electronegativity = nn.Softmax(dim=-1)
        self.cov_radius = nn.Softmax(dim=-1)
        self.val_electrons = nn.Softmax(dim=-1)
        self.first_ion = nn.Softmax(dim=-1)
        self.elec_aff = nn.Softmax(dim=-1)
        self.block = nn.Softmax(dim=-1)
        self.atomic_vol = nn.Softmax(dim=-1)
        self.element = nn.Softmax(dim=-1)

    def forward(self, x, masked_values):
        x = x[torch.arange(x.shape[0]), masked_values]
        embedded = self.dense(x)
        if self.pip:
            gn = embedded[:, :19]
            pn = embedded[:, 19:26]
            en = embedded[:, 26:36]
            cr = embedded[:, 36:46]
            ve = embedded[:, 46:58]
            fi = embedded[:, 58:68]
            ea = embedded[:, 68:78]
            bl = embedded[:, 78:82]
            av = embedded[:, 82:92]
            return gn, pn, en, cr, ve, fi, ea, bl, av
        else:
            element = self.element(embedded)
            return element


class DistanceDecoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DistanceDecoder, self).__init__()
        self.out = nn.Linear(input_dim, output_dim)

    def forward(self, x, masked_values):
        x = x[torch.arange(x.shape[0]), masked_values]
        return self.out(x)


class NeighborDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeighborDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class FineTuner(nn.Module):
    def __init__(self, input_dim, num_heads=1, n_encoders=1):
        super(FineTuner, self).__init__()
        self.encoders = nn.ModuleList([PeriodicSetTransformerEncoder(input_dim, num_heads) for _ in range(n_encoders)])
        self.embed = nn.Linear(input_dim, input_dim)
        self.relu = nn.Softplus()
        self.ln = nn.LayerNorm(input_dim)
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x, weights=None):
        if weights is not None:
            for encoder in self.encoders:
                x = self.ln(x + encoder(x, weights))
            x = torch.sum(weights * x, dim=1)

        x = self.relu(self.embed(x))
        return self.out(x)


