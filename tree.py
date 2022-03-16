from collections import deque
from typing import Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn


class Tree(nn.Module):
    def __init__(self, G, root):
        super().__init__()
        self.G = G
        self.root = root

    def __len__(self):
        return len(self.G)

    def full_tree_encoding(self):
        """
        Mapping from the nodes to the positions in a complete binary tree (BFS).
        """
        pos_map = {self.root : 0}
        q = deque([self.root])
        while len(q) > 0:
            n = q.popleft()
            children = self.G[n]
            for i, c in enumerate(children):
                idx = 2*pos_map[n]+1+i
                pos_map.update({c: idx})
                q.append(c)
        return pos_map

    def to_torch(self):
        """
        Transform all ndarray attributes to tensors.
        """
        G = self.G.copy()
        for n in G:
            attrs = G.nodes[n]
            for k,v in attrs.items():
                if isinstance(v, np.ndarray):
                    attrs[k] = torch.tensor(v)
        return Tree(G, self.root)

    def to(self, *args, **kwargs):
        """
        Apply "to" method to all tensor attributes. Returns a copy
        """
        G = self.G.copy()
        for n in G:
            attrs = G.nodes[n]
            for k,v in attrs.items():
                if isinstance(v, torch.Tensor):
                    attrs[k] = v.to(*args, **kwargs)
        return Tree(G, self.root)

    def get_cat_attr(self, attr, order=None):
        """
        Return concatenated torch.Tensor attributes as a dict.
        """
        order = order if order else self.G
        return torch.cat([self.G.nodes[n][attr] for n in order], 0)


def compose_feature_trees(trees, pad_value = 0):
    """
    Combine multiple trees with "feature" attribute into a single one with concatenated features.
    All the trees will be contained in the combined one as subtrees.
    Node features in the trees should be Torch.tensors of the same shape = [1, N_features].
    Also the "mask" attribute will be added to the nodes. It indicates which original trees contain this node.
    """
    feature_example = next(iter(trees[0].G.nodes.values()))["feature"]
    pad_array = torch.full_like(feature_example, pad_value)
    graphs = []
    for t in trees:
        graphs.append(nx.relabel_nodes(t.G, t.full_tree_encoding()))
    G = nx.compose_all(graphs)
    for n in G.nodes():
        features = [g.nodes[n]["feature"] if n in g else pad_array for g in graphs]
        G.nodes[n]["feature"] = torch.cat(features, 0)
        G.nodes[n]["mask"] = torch.tensor([True if n in g else False for g in graphs], device=features[0].device)
    return Tree(G, 0)


# ---------------------------------------------------------------------------


def flatten_batch_TreeConv(feature_trees, padding_value=0.0, batch_first=True):
    """
    Get batch with flatten representations of trees needed for Tree Conv.
    """
    feature_example = next(iter(feature_trees[0].G.nodes.values()))["feature"]
    features = []
    indices = []
    for i,ft in enumerate(feature_trees):
        graph = ft.G
        mapping = {n : i+1 for i,n in enumerate(nx.bfs_tree(graph, ft.root))}
        features.append(ft.get_cat_attr("feature", mapping).to(torch.float))
        g_indices = []
        for n,j in mapping.items():
            g_indices.append(j)
            ch = tuple(mapping[c] for c in graph[n])
            g_indices.extend(ch)
            g_indices.extend([0]*(2-len(ch)))
        g_indices = torch.tensor(g_indices, device=feature_example.device)
        indices.append(g_indices)
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=batch_first, padding_value=padding_value)
    indices = torch.nn.utils.rnn.pad_sequence(indices, batch_first=batch_first, padding_value=0)
    return features,  indices.unsqueeze(-1)


class BinaryTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        # we can think of the tree conv as a single dense layer
        # that we "drag" across the tree.
        self.linear = torch.nn.Linear(3*in_channels, out_channels)
        self.register_buffer('pad_value', torch.zeros(1, 1, self.__in_channels))

    def forward(self, flat_data):
        # features.shape = [batch_size, num_nodes, feature_dim]
        features, idxes = flat_data
        padded_features = torch.cat([self.pad_value.expand(features.shape[0], -1, -1), features], dim=1)
        expanded = torch.gather(padded_features, 1, idxes.expand(-1, -1, self.__in_channels))
        results = self.linear(expanded.view(features.shape[0], -1, 3*self.__in_channels))
        return (results, idxes)


class NeoTreeLayerNorm(nn.Module):
    def forward(self, x):
        data, idxes = x
        mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        normd = (data - mean) / (std + 0.00001)
        return (normd, idxes)


class TreeConvCompatible(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        a, *b = x
        return (self.module(a), *b)


class TreeBatchNorm(nn.BatchNorm1d):
    """
    Batchnorm over feature dim in batch of tree features.
    """
    def forward(self, x):
        data, idxes = x
        normd = super().forward(data.view(-1, data.shape[-1])).view(data.shape)
        return (normd, idxes)


class DynamicPooling(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x):
        if self.t == 'max':
            return torch.max(x[0], dim=1).values
        elif self.t == 'average':
            return torch.mean(x[0], dim=1)


# ---------------------------------------------------------------------------


class TreeLSTM(nn.Module):
    def __init__(self, rnn_type, in_dim, mem_dim, nlayers, N=2, attn_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        if rnn_type == "ChildSum":
            self.cells = nn.ModuleList([ChildSumTreeLSTMCell(in_dim, mem_dim)])
            self.cells.extend([ChildSumTreeLSTMCell(mem_dim, mem_dim) for l in range(nlayers-1)])
        elif rnn_type == "N-ary":
            self.cells = nn.ModuleList([N_TreeLSTMCell(in_dim, mem_dim, N)])
            self.cells.extend([N_TreeLSTMCell(mem_dim, mem_dim, N) for l in range(nlayers-1)])
        elif rnn_type == "Attn":
            self.cells = nn.ModuleList([Attentive_TreeLSTMCell(in_dim, mem_dim, attn_dim)])
            self.cells.extend([Attentive_TreeLSTMCell(mem_dim, mem_dim, attn_dim) for l in range(nlayers-1)])
        else:
            raise NotImplementedError
        self.register_buffer("mask_value", torch.zeros(1, 1, mem_dim, dtype=torch.float))

    def _forward(self, G, layer, node):
        child_outs = [self._forward(G, layer, n) for n in G[node]]
        x = G.nodes[node]["feature"]
        x = x.view(-1, 1, x.shape[-1])
        if len(G[node]) == 0:
            child_c, child_h = (self.mask_value.repeat(x.shape[0], 1, 1).requires_grad_() for _ in range(2))
        else:
            child_c, child_h = zip(*child_outs)
            child_c, child_h = torch.cat(child_c, dim=1), torch.cat(child_h, dim=1)
        c, h = self.cells[layer].forward(x, child_c, child_h)
        if "mask" in G.nodes[node]:
            mask = G.nodes[node]["mask"]
            c = torch.where(mask.view(-1, 1, 1), c, self.mask_value)
            h = torch.where(mask.view(-1, 1, 1), h, self.mask_value)
        G.nodes[node]["feature"] = h.clone()
        return c, h

    def forward(self, feature_tree):
        G = feature_tree.G.copy()
        for l in range(len(self.cells)):
            c, h = self._forward(G, l, feature_tree.root)
        return c, h


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super().__init__()
        self.ioux = nn.Linear(in_dim, 3 * mem_dim)
        self.iouh = nn.Linear(mem_dim, 3 * mem_dim, bias=False)
        self.fx = nn.Linear(in_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim, bias=False)

    def forward(self, inputs, child_c, child_h):
        # inputs.shape = [N_batch, 1, in_dim]
        # child_c.shape = child_h.shape = [N_batch, N_children, mem_dim]
        child_h_sum = torch.sum(child_h, dim=1, keepdim=True)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(-1) // 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u) # [N_batch, 1, mem_dim]
        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).expand(-1, child_h.shape[1], -1)
        )
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=1, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h


class N_TreeLSTMCell(nn.Module):
    def __init__(self, in_dim, mem_dim, N):
        super().__init__()
        self.N = N
        self.ioux = nn.Linear(in_dim, 3 * mem_dim)
        self.iouh = nn.Linear(N * mem_dim, 3 * mem_dim, bias=False)
        self.fx = nn.Linear(in_dim, mem_dim)
        self.fh = nn.Linear(N * mem_dim, N * mem_dim, bias=False)
        self.register_buffer("pad_value", torch.zeros(1, 1, mem_dim, dtype=torch.float))

    def forward(self, inputs, child_c, child_h):
        # inputs.shape = [N_batch, 1, in_dim]
        # child_c.shape = child_h.shape = [N_batch, N_children, mem_dim]
        Nb, Nc, mem_dim = child_h.shape
        assert(Nc <= self.N)
        if Nc < self.N:
            child_c = torch.cat([child_c, self.pad_value.expand(Nb, self.N-Nc,-1)], dim = 1)
            child_h = torch.cat([child_h, self.pad_value.expand(Nb, self.N-Nc,-1)], dim = 1)
        child_h = child_h.view(Nb, -1)
        iou = self.ioux(inputs) + self.iouh(child_h).view(Nb, 1, -1)
        i, o, u = torch.split(iou, iou.size(-1) // 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u) # [N_batch, 1, mem_dim]
        f = torch.sigmoid(
            self.fh(child_h).view(Nb, self.N, -1) +
            self.fx(inputs).expand(-1, self.N, -1)
        )
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=1, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h


class Attentive_TreeLSTMCell(nn.Module):
    def __init__(self, in_dim, mem_dim, attn_dim):
        super().__init__()
        self.ioux = nn.Linear(in_dim, 3 * mem_dim)
        self.iouh = nn.Linear(mem_dim, 3 * mem_dim, bias=False)
        self.fx = nn.Linear(in_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim, bias=False)
        self.W_k = nn.Linear(mem_dim, attn_dim)
        self.q = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, inputs, child_c, child_h):
        # inputs.shape = [N_batch, 1, in_dim]
        # child_c.shape = child_h.shape = [N_batch, N_children, mem_dim]
        WH = torch.tanh(self.W_k(child_h))
        # WH = self.W_k(child_h)/np.sqrt(child_h.shape[-1])
        weights = torch.softmax(self.q(WH), 1) # [N_batch, N_children, 1]
        h_tilda = (weights.expand(-1, -1, child_h.shape[-1]) * child_h).sum(dim=1, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(h_tilda)
        i, o, u = torch.split(iou, iou.size(-1) // 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u) # [N_batch, 1, mem_dim]
        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).expand(-1, child_h.shape[1], -1)
        )
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=1, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h


# ---------------------------------------------------------------------------


def apply_positional_encoding(tree, max_depth, max_degree=2):
    g, root = tree.G, tree.root
    emb = g.nodes[root]["feature"]
    g.nodes[root]["pos_encoding"] = torch.zeros((1, max_depth*max_degree), device=emb.device)
    for parent, children in nx.bfs_successors(g, root):
        par_enc = g.nodes[parent]["pos_encoding"]
        enc = torch.cat((par_enc.new_zeros((1, max_degree)), par_enc[..., :-max_degree]), -1)
        for i,c in enumerate(children):
            g.nodes[c]["pos_encoding"] =  enc.clone()
            g.nodes[c]["pos_encoding"][...,i] = 1
    return tree


class PositionalEncoding(nn.Module):
    def __init__(self, d_emb, d_model, depth, degree, d_param, dropout=0.1):
        super().__init__()
        d_pos = depth * degree * d_param
        self.d_model = d_model
        self.p = nn.Parameter(torch.empty((1, 1, d_param)))
        torch.nn.init.xavier_uniform_(self.p, gain=1.0)
        self.register_buffer('depths', torch.arange(depth).view(depth,1,1))
        self.demb_to_dmodel = nn.Linear(d_emb, d_model)
        self.dpos_to_dmodel = nn.Linear(d_pos, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, pos):
        # emb.shape = [L, batch_size, d_model], pos.shape = [L, batch_size, depth*degree]
        p = torch.tanh(self.p)
        weights = torch.pow(p, self.depths) * torch.sqrt((1-p**2) * self.d_model / 2)
        # weights.shape = [depth, 1, d_param]
        emb = self.demb_to_dmodel(emb) * np.sqrt(self.d_model)
        pos = pos.view(*pos.shape[:2], weights.shape[0], -1, 1) * weights
        x = emb + self.dpos_to_dmodel(pos.view(*pos.shape[:2], -1))
        return self.dropout(x)


# ---------------------------------------------------------------------------


class TreeTransformerEncoderLayer(nn.Module):
    """
    Transformer with Tree Conv block instead of FF.
    """
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # There is TCB block instead of Feedforward model
        self.tcb = nn.Sequential(BinaryTreeConv(d_model, dim_feedforward),
                                  TreeConvCompatible(nn.Dropout(dropout)),
                                  TreeConvCompatible(nn.ReLU()),
                                  BinaryTreeConv(dim_feedforward, d_model),
                                  TreeConvCompatible(nn.Dropout(dropout)))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src_ids: (torch.Tensor, torch.Tensor), src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # src.shape = [1+L, N, C], indices.shape = [3*L, N, 1]
            src, indices = src_ids
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout(src2)
            src = self.norm1(src)
            # first token is cls
            cls_token = src[:1]
            tree_features = src[1:].transpose(0, 1)
            src2, _ = self.tcb((tree_features, indices.transpose(0, 1)))
            src = src + torch.cat((cls_token, src2.transpose(0, 1)))
            src = self.norm2(src)
            return src, indices
