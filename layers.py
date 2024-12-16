# import dgl.function as fn
import dgl.nn.functional as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
EOS = 1e-10

class GCNConv_dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNConv_dense, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj):
        # 执行线性变换
        x = self.linear(x)
        # 执行图卷积操作
        x = torch.matmul(adj, x)
        return x

class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']



class LightGCNConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(LightGCNConv, self).__init__()
        self.linear = nn.Linear(input_size, output_size)#2888*512

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input) #input 1444*2888 hidden:1444*512
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden) #A
        return output

class LightGCNConv(nn.Module):
    def __init__(self, num_users, num_items, embed_size):
        super(LightGCNConv, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_size = embed_size
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.item_embedding = nn.Embedding(num_items, embed_size)
        self.init_para()

    def init_para(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_indices, item_indices, A):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)

        # Perform graph convolution
        user_output = torch.matmul(A, user_embed)
        item_output = torch.matmul(A, item_embed)

        return user_output, item_output


class Attentive(nn.Module):
    def __init__(self, isize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(isize))

    def forward(self, x):
        return x @ torch.diag(self.w)



class SparseDropout(nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - dprob

    def forward(self, x):
        mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(torch.bool)
        rc = x._indices()[:,mask]
        val = x._values()[mask]*(1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)
class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GATConv, self).__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(in_dim, out_dim * num_heads)
        self.attn_fc = nn.Linear(2 * out_dim, num_heads)

    def edge_attention(self, edges):
        z = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.attn_fc(z)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        with g.local_scope():
            h = self.linear(h).view(-1, self.num_heads, h.shape[1])
            g.ndata['h'] = h
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            return g.ndata.pop('h')