import torch
import torch.nn as nn
from tree import TreeTransformerEncoderLayer, BinaryTreeConv, NeoTreeLayerNorm
import numpy as np
from copy import deepcopy


def FC(d_in, d_out, fc_nlayers, drop):
    dims = torch.linspace(d_in, d_out, fc_nlayers+1, dtype=torch.long)
    layers = []
    for i in range(fc_nlayers-1):
        layers.extend([nn.Linear(int(dims[i]), int(dims[i+1])),
                       nn.Dropout(drop), nn.LayerNorm([int(dims[i+1])]), nn.ReLU()])
    layers.append(nn.Linear(int(dims[-2]), d_out))
    return nn.Sequential(*layers)


def FC_from_dims(dims, drop):
    fc_nlayers = len(dims)-1
    layers = []
    for i in range(fc_nlayers-1):
        layers.extend([nn.Linear(int(dims[i]), int(dims[i+1])),
                       nn.Dropout(drop), nn.LayerNorm([int(dims[i+1])]), nn.ReLU()])
    layers.append(nn.Linear(int(dims[-2]), dims[-1]))
    return nn.Sequential(*layers)



class TransformerNet(nn.Module):
    def __init__(self, d_emb, d_query, d_model, nhead, ffdim, nlayers, fc_nlayers, drop, pretrained_path=False, fit_pretrained_layers=[], **kwargs):
        super().__init__()
        self.args = {k: v for k, v in locals().items() if k not in [
                                             'self', '__class__']}
        # Tree transformer
        self.enc = nn.Linear(d_emb, (d_model+1)//2)
        self.trans_enc = nn.TransformerEncoder(
            TreeTransformerEncoderLayer(d_model, nhead, ffdim, drop), nlayers)
        self.cls = nn.Parameter(torch.empty((1, 1, d_model)))
        torch.nn.init.xavier_uniform_(self.cls, gain=1.0)
        # Transformer encoder for combining forest repr (bunch of vectors) into one vector
        self.many_to_one = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 1, ffdim, drop), 1)
        self.cls2 = nn.Parameter(torch.empty((1, 1, d_model)))
        torch.nn.init.xavier_uniform_(self.cls2, gain=1.0)
        # FC layers
        self.fc = FC(d_model, 1, fc_nlayers, drop)
        # Query level
        d_q = d_model // 2
        self.qn = nn.Sequential(nn.Linear(d_query, d_q))
        self.pretrained_path = pretrained_path
        self.fit_pretrained_layers = fit_pretrained_layers

    def forward(self, inputs):
        q, t = inputs
        q = self.qn(q).unsqueeze(0)  # [1, (n1+n2+...), d_model // 2]
        x, indices, lens = t
        # [L, (n1+n2+...), d_model // 2]; ni = number of trees in i-th forest
        x = self.enc(x)
        x = torch.cat((x, q.expand(x.shape[0], -1, -1)), -1) # [L, (n1+n2+...), d_model]
        x = torch.cat((self.cls.expand(-1, x.shape[1], -1), x), 0)
        x, _ = self.trans_enc((x, indices))  # [1, (n1+n2+...), d_model], ...
        l = torch.split(x[0], lens)
        x = torch.nn.utils.rnn.pad_sequence(l)  # [max(ni), Nf, d_model]
        x = torch.cat((self.cls2.expand(1, x.shape[1], -1), x), 0)
        pad_mask = torch.tensor(np.arange(x.shape[0]).reshape(
            1, -1) > np.array(lens).reshape(-1, 1), device=x.device)
        x = self.many_to_one(x, src_key_padding_mask=pad_mask)[
                             0]  # [N_f, d_model]
        x = self.fc(x)
        return x

    def new(self):
        if self.pretrained_path:
            model = deepcopy(self)
            model.load_state_dict(torch.load(self.pretrained_path))
            return model
        else:
            return self.__class__(**self.args)



class NeoTreeConvNet(nn.Module):
    def __init__(self, d_emb, d_query, drop, pretrained_path=False, fit_pretrained_layers=[], **kwargs):
        super().__init__()
        self.args = {k: v for k, v in locals().items() if k not in [
                                             'self', '__class__']}
        # query-level
        self.qn = FC_from_dims([d_query, 128, 64, 32], drop)
        # tree conv
        self.tree_conv = nn.Sequential(BinaryTreeConv(32+d_emb, 512),
                                       NeoTreeLayerNorm(),
                                       BinaryTreeConv(512, 256),
                                       NeoTreeLayerNorm(),
                                       BinaryTreeConv(256,128),
                                       )
        # fc
        self.fc = FC_from_dims([128, 128, 64, 32, 1], drop)
        self.pretrained_path = pretrained_path
        self.fit_pretrained_layers = fit_pretrained_layers

    def forward(self, inputs):
        q, t = inputs
        q = self.qn(q).unsqueeze(0)  # [1, (n1+n2+...), 32]
        x, indices, lens = t
        x = torch.cat((x, q.expand(x.shape[0], -1, -1)), -1) # [L, (n1+n2+...), 32+d_emb]
        # [L, (n1+n2+...), d_model],
        x, _ = self.tree_conv((x.transpose(0,1), indices.transpose(0,1)))
        x = torch.max(x, dim=1).values # [(n1+n2+...), d_model]
        l = torch.split(x, lens)
        x = torch.nn.utils.rnn.pad_sequence(l)  # [max(ni), Nf, d_model]
        x = torch.max(x, dim=0).values
        x = self.fc(x)
        return x

    def new(self):
        if self.pretrained_path:
            model = deepcopy(self)
            model.load_state_dict(torch.load(self.pretrained_path))
            return model
        else:
            return self.__class__(**self.args)
