# -*- coding: utf-8 -*-
"""
Defines multiple reusable GNN and Transformer models for heterogeneous drug–protein interaction (DPI) graphs.
Created on Mon Jun 30 19:04:15 2025
@author: laila
"""

import torch
from torch import nn
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, HGTConv, HANConv,
    HeteroConv, Linear, TransformerConv
)

# ---------------------------------------------
# HeteroGNN using SAGEConv with HeteroConv
# ---------------------------------------------
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = HeteroConv({
            ('drug', 'interacts', 'protein'): SAGEConv((-1, -1), hidden_dim),
            ('protein', 'rev_interacts', 'drug'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='mean')
        self.lin = Linear(hidden_dim * 2, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = self.conv1(x_dict, edge_index_dict)
        drug_emb = x_dict['drug'][edge_label_index[0]]
        prot_emb = x_dict['protein'][edge_label_index[1]]
        edge_emb = torch.cat([drug_emb, prot_emb], dim=1)
        return self.lin(edge_emb)

# ---------------------------------------------
# GCN for homogeneous graphs (not Hetero!)
# ---------------------------------------------
class HeteroGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x = x_dict['drug']  # Assumes all nodes are 'drug'
        edge_index = edge_index_dict[('drug', 'interacts', 'protein')]
        x = self.conv1(x, edge_index)
        drug_emb = x[edge_label_index[0]]
        prot_emb = x[edge_label_index[1]]
        edge_emb = torch.cat([drug_emb, prot_emb], dim=1)
        return self.lin(edge_emb)

# ---------------------------------------------
# GraphSAGE for heterogeneous graphs
# ---------------------------------------------
class HeteroGraphSAGE(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, node_feature_dims):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.lin = nn.Linear(node_feature_dims['drug'] + hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        prot_emb_all = self.conv1((x_dict['drug'], x_dict['protein']),
                                  edge_index_dict[('drug', 'interacts', 'protein')])
        drug_feats = x_dict['drug']
        drug_emb = drug_feats[edge_label_index[0]]
        prot_emb = prot_emb_all[edge_label_index[1]]
        edge_emb = torch.cat([drug_emb, prot_emb], dim=1)
        return self.lin(edge_emb)

# ---------------------------------------------
# GAT for heterogeneous graphs
# ---------------------------------------------
class HeteroGAT(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, node_feature_dims):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_dim, heads=2, concat=False)
        self.lin = nn.Linear(node_feature_dims['drug'] + hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        prot_emb_all = self.conv1((x_dict['drug'], x_dict['protein']),
                                  edge_index_dict[('drug', 'interacts', 'protein')])
        drug_feats = x_dict['drug']
        drug_emb = drug_feats[edge_label_index[0]]
        prot_emb = prot_emb_all[edge_label_index[1]]
        edge_emb = torch.cat([drug_emb, prot_emb], dim=1)
        return self.lin(edge_emb)

# ---------------------------------------------
# HGT for heterogeneous graphs
# ---------------------------------------------
class HeteroHGT(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, node_feature_dims):
        super().__init__()
        self.input_proj = nn.ModuleDict()
        for node_type in metadata[0]:
            self.input_proj[node_type] = nn.Linear(node_feature_dims[node_type], hidden_dim)
        self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata)
        self.lin = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = {node: self.input_proj[node](x) for node, x in x_dict.items()}
        x_dict = self.conv1(x_dict, edge_index_dict)
        drug_emb = x_dict['drug'][edge_label_index[0]]
        prot_emb = x_dict['protein'][edge_label_index[1]]
        edge_emb = torch.cat([drug_emb, prot_emb], dim=1)
        return self.lin(edge_emb)

# ---------------------------------------------
# HAN for heterogeneous graphs
# ---------------------------------------------
class HeteroHAN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, node_feature_dims):
        super().__init__()
        self.input_proj = nn.ModuleDict()
        for node_type in metadata[0]:
            self.input_proj[node_type] = nn.Linear(node_feature_dims[node_type], hidden_dim)
        self.conv1 = HANConv(hidden_dim, hidden_dim, metadata, heads=2)
        self.lin = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = {node: self.input_proj[node](x) for node, x in x_dict.items()}
        x_dict = self.conv1(x_dict, edge_index_dict)
        drug_emb = x_dict['drug'][edge_label_index[0]]
        prot_emb = x_dict['protein'][edge_label_index[1]]
        edge_emb = torch.cat([drug_emb, prot_emb], dim=1)
        return self.lin(edge_emb)

# ---------------------------------------------
# HeteroGraphTransformer with stacked HGTConvs
# ---------------------------------------------
class HeteroGraphTransformer(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, node_feature_dims, num_heads=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.ModuleDict()
        for node_type in metadata[0]:
            self.input_proj[node_type] = nn.Linear(node_feature_dims[node_type], hidden_dim)
        self.layers = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.lin = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = {node: self.input_proj[node](x) for node, x in x_dict.items()}
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        drug_emb = x_dict['drug'][edge_label_index[0]]
        prot_emb = x_dict['protein'][edge_label_index[1]]
        edge_emb = torch.cat([drug_emb, prot_emb], dim=1)
        return self.lin(edge_emb)

# ---------------------------------------------
# Simple TransformerConv-based model for homogeneous graphs
# ---------------------------------------------
class SimpleGraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.layers.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.layers.append(TransformerConv(hidden_channels * heads, out_channels, heads=heads))

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = torch.relu(conv(x, edge_index))
        return x

# ---------------------------------------------
# Wrapper for homogeneous graph Transformer
# ---------------------------------------------
class GraphTransformerWrapper(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()
        self.model = SimpleGraphTransformer(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=3
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)
