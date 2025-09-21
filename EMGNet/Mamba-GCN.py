import os
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd



# Mamba实现

class SelectiveScan(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        self.proj_in = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        B, L, D = x.size()
        x = self.proj_in(x)
        h = torch.zeros(B, L, self.d_state, device=x.device)
        for t in range(L):
            if t == 0:
                h_t = self.activation(torch.matmul(x[:, t], self.B.t()))
                h = h.clone()
                h[:, t] = h_t
            else:
                prev_state = torch.matmul(h[:, t-1], self.A)
                curr_input = torch.matmul(x[:, t], self.B.t())
                h_t = self.activation(prev_state + curr_input)
                h = h.clone()
                h[:, t] = h_t
        y = torch.matmul(h, self.C.t())
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        y = self.proj_out(y)
        return self.dropout(y)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.scan = SelectiveScan(d_model, d_state, dropout)

    def forward(self, x):
        return x + self.scan(self.norm(x))

class CustomMamba(nn.Module):
    def __init__(self, d_model, n_layers=4, d_state=16, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MambaAttention(nn.Module):
    def __init__(self, d_model, n_layers=2, d_state=16, dropout=0.1):
        super().__init__()
        self.mamba = CustomMamba(d_model, n_layers, d_state, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.size()
        x_reshaped = x.reshape(B * N, 1, D)
        x_mamba = self.mamba(x_reshaped)
        x_mamba = x_mamba.reshape(B, N, D)
        x = x + x_mamba
        x = self.norm(x)
        x = self.proj(x)
        return x


# GCN

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        return self.linear(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.input_proj = nn.Linear(nfeat, nhid)
        self.norm1 = nn.LayerNorm(nhid)
        self.mamba_attn = MambaAttention(d_model=nhid, n_layers=4, d_state=32, dropout=dropout)
        self.gcn1 = GCNLayer(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gcn2 = GCNLayer(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.gcn3 = GCNLayer(nhid, nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.gcn4 = GCNLayer(nhid, nhid)
        self.bn4 = nn.BatchNorm1d(nhid)
        self.gcn5 = GCNLayer(nhid, nhid)
        self.dropout = dropout
        self.class_specific_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nhid * 4, nhid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nhid, 1)
            ) for _ in range(nclass)
        ])

    def forward(self, x, adj):
        x = self.input_proj(x)
        x = self.norm1(x)
        x_attn = self.mamba_attn(x)
        x_gcn = self.gcn1(x, adj)
        x_gcn = x_gcn.transpose(1,2)
        x_gcn = self.bn1(x_gcn)
        x_gcn = x_gcn.transpose(1,2)
        x_gcn = F.relu(x_gcn)
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gcn = self.gcn2(x_gcn, adj)
        x_gcn = x_gcn.transpose(1,2)
        x_gcn = self.bn2(x_gcn)
        x_gcn = x_gcn.transpose(1,2)
        x_gcn = F.relu(x_gcn)
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gcn = self.gcn3(x_gcn, adj)
        x_gcn = x_gcn.transpose(1,2)
        x_gcn = self.bn3(x_gcn)
        x_gcn = x_gcn.transpose(1,2)
        x_gcn = F.relu(x_gcn)
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gcn = self.gcn4(x_gcn, adj)
        x_gcn = x_gcn.transpose(1,2)
        x_gcn = self.bn4(x_gcn)
        x_gcn = x_gcn.transpose(1,2)
        x_gcn = F.relu(x_gcn)
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        x_gcn = self.gcn5(x_gcn, adj)
        x_attn_mean = x_attn.mean(dim=1)
        x_attn_max = x_attn.max(dim=1)[0]
        x_gcn_mean = x_gcn.mean(dim=1)
        x_gcn_max = x_gcn.max(dim=1)[0]
        x_fused = torch.cat([x_attn_mean, x_attn_max, x_gcn_mean, x_gcn_max], dim=-1)
        class_logits = [self.class_specific_fusion[i](x_fused) for i in range(len(self.class_specific_fusion))]
        x = torch.cat(class_logits, dim=1)
        return x


# Dataset类

class GraphDataset(Dataset):
    def __init__(self, data_dir, label_dict):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.label_dict = label_dict
        print(f"Total samples (one per file): {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        subject = os.path.basename(file_path).split('_')[0]
        npz = np.load(file_path)
        X_all = torch.tensor(npz['X_all'][:9], dtype=torch.float32)
        A_all = torch.tensor(npz['A_all'][:9], dtype=torch.float32)
        features = X_all.mean(dim=0)
        feats_mean = features.mean(dim=0, keepdim=True)
        feats_std = features.std(dim=0, keepdim=True) + 1e-6
        features = (features - feats_mean) / feats_std
        adj = A_all.mean(dim=0)
        adj = F.relu(adj)
        N = adj.size(0)
        adj = adj + torch.eye(N, dtype=adj.dtype, device=adj.device)
        deg = adj.sum(dim=1).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt
        label = torch.tensor(self.label_dict.get(subject, -1), dtype=torch.long)
        return features, adj, label, subject


