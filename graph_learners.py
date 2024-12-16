import dgl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from layers import Attentive, GCNConv_dense, GCNConv_dgl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
from utils import *
class FGP_learner(nn.Module):
    def __init__(self, features, k, knn_metric, i, sparse):
        super(FGP_learner, self).__init__()

        self.k = k
        self.knn_metric = knn_metric
        self.i = i
        self.sparse = sparse

        self.Adj = nn.Parameter(
            torch.from_numpy(nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i)))

    def forward(self, h):
        if not self.sparse:
            Adj = F.elu(self.Adj) + 1
        else:
            Adj = self.Adj.coalesce()
            Adj.values = F.elu(Adj.values()) + 1
        return Adj


class ATT_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act):
        super(ATT_learner, self).__init__()

        self.i = i
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentive(isize))
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def forward(self, features):

            print("features",features.shape)
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities


class SVD_learner(nn.Module):
    def __init__(self, adj1, svd_q=15):
        super(SVD_learner, self).__init__()
        self.svd_q = svd_q

        # 新增可学习参数
        self.learnable_param = nn.Parameter(torch.ones_like(adj1))  # 初始化为与 adj 形状相同且元素全为 1 的张量

    def forward(self, adj1):
        svd_u, s, svd_v = torch.svd_lowrank(adj1, q=self.svd_q)


        # 截断奇异值并重构低秩近似的邻接矩阵
        adj_reconstructed = torch.matmul(svd_u[:, :self.svd_q], torch.matmul(torch.diag(s[:self.svd_q]), svd_v[:, :self.svd_q].t()))

        # 使用可学习参数
        adj_updated = adj_reconstructed * self.learnable_param

        adj_original = torch.cat((torch.cat((adj_updated, torch.zeros(462, 462).to(device)), dim=1),
                                  torch.cat((torch.zeros(102, 102).to(device), adj_updated.t()), dim=1)), dim=0)

        return adj_original

