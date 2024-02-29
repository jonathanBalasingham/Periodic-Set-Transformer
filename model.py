import torch
import torch.nn as nn
import json
from WeightedBatchNorm1d import *
import numpy as np
import math

torch.manual_seed(0)


def weighted_softmax(x, dim=-1, weights=None):
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    if weights is not None:
        x_exp = weights * x_exp
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    probs = x_exp/x_exp_sum
    return probs


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MHA(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.0, use_kv_bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.use_kv_bias = use_kv_bias
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.delta_mul = nn.Linear(embed_dim, embed_dim)
        self.delta_bias = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        #  From original torch implementation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, mask=None, weights=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        #if weights is not None:
        #    attn_logits = attn_logits * weights

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = weighted_softmax(attn_logits, dim=-1, weights=weights)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x, mask=None, return_attention=False, weights=None, bias=None):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        if bias is not None:
            biases = bias[:, :, None, :] - bias[:, None, :, :]

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        if weights is not None:
            weights = torch.transpose(weights, -2, -1)
            weights = weights[:, None, :, :].expand(-1, self.num_heads, -1, -1)

        values, attention = self.scaled_dot_product(q, k, v, mask=mask, weights=weights)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        values = self.dropout(values)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class PeriodicSetTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, attention_dropout=0.0, dropout=0.0, activation=nn.Mish):
        super(PeriodicSetTransformerEncoder, self).__init__()
        self.embedding = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.out = nn.Linear(embedding_dim * num_heads, embedding_dim)
        self.multihead_attention = MHA(embedding_dim, embedding_dim * num_heads, num_heads, dropout=attention_dropout)
        self.pre_norm = nn.LayerNorm(embedding_dim)
        self.ln = torch.nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.W_q = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.W_k = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.W_v = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.ffn = nn.Linear(embedding_dim, embedding_dim)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                 activation())

    def forward(self, x, weights, use_weights=True):
        x_norm = self.ln(x)
        keep = weights > 0
        keep = keep * torch.transpose(keep, -2, -1)

        if use_weights:
            att_output = self.multihead_attention(x_norm, weights=weights, mask=keep)
        else:
            att_output = self.multihead_attention(x_norm, mask=keep)

        output1 = x + self.out(att_output)
        output2 = self.ln(output1)
        output2 = self.ffn(output2)
        return self.ln(output1 + output2)


class PeriodicSetTransformer(nn.Module):

    def __init__(self, str_fea_len, embed_dim, num_heads, n_encoders=3, decoder_layers=1, components=None,
                 expansion_size=10, dropout=0., attention_dropout=0., use_cuda=True, atom_encoding="mat2vec",
                 use_weighted_attention=True, use_weighted_pooling=True, activation=nn.Mish, sigmoid_out=False,
                 expand_distances=False):
        super(PeriodicSetTransformer, self).__init__()
        if components is None:
            components = ["pdd", "composition"]

        if atom_encoding not in ["mat2vec", "cgcnn"]:
            raise ValueError(f"atom_encoding_dim must be in {['mat2vec', 'cgcnn']}")
        else:
            atom_encoding_dim = 200 if atom_encoding == "mat2vec" else 92
            id_prop_file = "mat2vec.csv" if atom_encoding == "mat2vec" else "atom_init.json"

        self.composition = "composition" in components
        self.pdd_encoding = "pdd" in components
        self.use_weighted_attention = use_weighted_attention
        self.use_weighted_pooling = use_weighted_pooling
        self.expand_distances = expand_distances
        if expand_distances:
            self.pdd_embedding_layer = nn.Linear((str_fea_len - 1) * expansion_size, embed_dim)
        else:
            self.pdd_embedding_layer = nn.Linear(str_fea_len - 1, embed_dim)
        self.comp_embedding_layer = nn.Linear(atom_encoding_dim, embed_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.af = AtomFeaturizer(use_cuda=use_cuda, id_prop_file=id_prop_file)
        if expand_distances:
            self.de = DistanceExpansion(size=expansion_size, use_cuda=use_cuda)
        self.ln = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cell_embed = nn.Linear(6, 32)
        self.softplus = nn.Softplus()
        self.encoders = nn.ModuleList(
            [PeriodicSetTransformerEncoder(embed_dim, num_heads, attention_dropout=attention_dropout, activation=activation) for _ in
             range(n_encoders)])
        self.decoder = nn.ModuleList([nn.Linear(embed_dim, embed_dim)
                                      for _ in range(decoder_layers - 1)])
        self.activations = nn.ModuleList([activation()
                                          for _ in range(decoder_layers - 1)])
        self.out = nn.Linear(embed_dim, 1)
        self.so = sigmoid_out
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        str_fea, comp_fea, cell_fea = features
        weights = str_fea[:, :, 0, None]
        comp_features = self.af(comp_fea)
        comp_features = self.comp_embedding_layer(comp_features)
        comp_features = self.dropout_layer(comp_features)
        str_features = str_fea[:, :, 1:]
        str_features = self.pdd_embedding_layer(self.de(str_features) if self.expand_distances else str_features)

        if self.composition and self.pdd_encoding:
            x = comp_features + str_features
        elif self.composition:
            x = comp_features
        elif self.pdd_encoding:
            x = str_features
        x_init = x
        for encoder in self.encoders:
            x = encoder(x, weights, use_weights=self.use_weighted_attention)

        if self.use_weighted_pooling:
            x = torch.sum(weights * (x + x_init), dim=1)
        else:
            x = torch.mean(x + x_init, dim=1)

        x = self.ln2(x)
        for layer, activation in zip(self.decoder, self.activations):
            x = layer(x)
            x = activation(x)

        if self.so:
            return self.sigmoid(self.out(x))
        return self.out(x)


class PeSTEncoder(nn.Module):

    def __init__(self, str_fea_len, embed_dim, num_heads, n_encoders=3, expansion_size=10, expand_distances=False):
        super(PeSTEncoder, self).__init__()
        print(f"inp size: {str_fea_len}")
        self.expand_distances = expand_distances
        if expand_distances:
            self.embedding_layer = nn.Linear((str_fea_len - 1) * expansion_size, embed_dim)
        else:
            self.embedding_layer = nn.Linear(str_fea_len - 1, embed_dim)
        self.comp_embedding_layer = nn.Linear(92, embed_dim)
        self.af = AtomFeaturizer()
        if expand_distances:
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
        str_features = self.embedding_layer(self.de(str_features) if self.expand_distances else str_features)
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


class CloudDecoder(nn.Module):
    def __init__(self, fea_len, num_heads, num_layers):
        super(CloudDecoder, self).__init__()
        self.fc = nn.Linear(3, fea_len)
        self.fc2 = nn.Linear(fea_len, 3)
        cloud_decoder_layer = nn.TransformerDecoderLayer(d_model=fea_len, nhead=num_heads, batch_first=True)
        self.cloud_decoder = nn.TransformerDecoder(cloud_decoder_layer, num_layers=num_layers)

    def forward(self, tgt, mem):
        tgt = self.fc(tgt)
        return self.fc2(self.cloud_decoder(tgt, mem))


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


