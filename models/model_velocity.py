#partially adapted from the Vector Neuron: https://github.com/FlyingGiraffe/vnn

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter, degree
from torch_geometric.nn import LayerNorm
from model import NodeModelIn, EdgeNodeMP

class VN_Lin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VN_Lin, self).__init__()
        self.Lin = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self,x):
        ''' 
        x: (N, d_in, 3)
        return (N, d_out, 3)
        '''
        x_out = self.Lin(x.permute(0,2,1))
        return x_out.permute(0,2,1)

class VN_ReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, eps=1e-6):
        super(VN_ReLU, self).__init__()
        if share_nonlinearity:
            self.proj_dir = VN_Lin(in_channels, 1)
        else:
            self.proj_dir = VN_Lin(in_channels, in_channels)
        self.eps = eps
    
    def forward(self,x):
        ''' 
        x: (N, d_in, 3)
        return (N, d_in, 3)
        '''
        dir = self.proj_dir(x) #(N, d_in, 3)
        dir_norm = (dir * dir).sum(2, keepdim=True).sqrt()
        dir_unit = dir / dir_norm
        dotprod = (x * dir_unit).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        x_out = (mask * x) + (1-mask)*(x - dotprod * dir_unit)
        return x_out


class VN_EdgeNodeMP(EdgeNodeMP):
    def __init__(self, node_dim=None, edge_dim=None, hid_dim=None, inter_dim=None, reduce='mean', 
                 norm=False, linear=False):
        EdgeNodeMP.__init__(self, node_dim, edge_dim, hid_dim, inter_dim, reduce, norm, linear)
        ## inherit other init params
        ## monkey patch the layers to ensure O(d) equivariance
        if linear: # perserve O(d) equivariance for input X = (N, d_in, 3), by Lin(X.permuta(0,2,1)) (acting d_in -> d_out)
            self.edge_mlp_1 = VN_Lin(node_dim + edge_dim, inter_dim)
            self.edge_mlp_1.bias.data.fill_(0)
        else:
            self.edge_mlp_1 = Seq(VN_Lin(node_dim + edge_dim, hid_dim), 
                              VN_ReLU(hid_dim, hid_dim), 
                              VN_Lin(hid_dim, inter_dim))

class VN_Node_GNN(nn.Module):
    def __init__(self, node_dim=1, edge_dim=1, hid_dim=16, out_dim=1, 
                 reduce='mean', norm=False, linear=False, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(VN_EdgeNodeMP(node_dim, edge_dim, hid_dim, hid_dim, reduce, norm))
        for l in range(1,n_layers):
            self.layers.append(VN_EdgeNodeMP(node_dim, edge_dim, hid_dim, hid_dim, reduce, norm))
        self.output_layer = Seq(Lin(hid_dim, hid_dim), 
                              ReLU(),
                            #   Lin(hid_dim, hid_dim),
                            #   ReLU(), 
                              Lin(hid_dim, out_dim))
    def forward(self, data):
        h, edge_index, edge_attr, node_weight = data.x, data.edge_index, data.edge_attr, data.node_weight
        for layer in self.layers:
            h, edge_attr = layer(h, edge_index, edge_attr, node_weight)
        out = self.output_layer(h)
        return out


## simple testing code
# in_channels = 1
# out_channels = 5
# lin_layer = VN_Lin(in_channels, out_channels)
# relu_layer = VN_ReLU(out_channels, out_channels)
# x = torch.rand(10,1,3)
# x_lin = lin_layer(x)
# out = relu_layer(x_lin)
# print(x_lin.shape, out.shape)
# model = VN_EdgeNodeMP(node_dim=in_channels, edge_dim=1, hid_dim=out_channels, inter_dim=out_channels)
# edge_index = torch.tensor([[0,4],
#                            [1,4],
#                            [2,5],
#                            [3,5],
#                            [4,6],
#                            [5,6]], dtype=torch.int64).T
# edge_attr = torch.rand((6,1,3)) ##Must be a 3D tensor
# node_new, edge_new = model(x, edge_index, edge_attr)
# print(node_new.shape, edge_new.shape)