import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter, degree
from torch_geometric.nn import LayerNorm, BatchNorm

#src: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MetaLayer.html#torch_geometric.nn.models.MetaLayer
#src: https://github.com/PabloVD/CosmoGraphNet/blob/master/Source/metalayer.py
#simplified as much as possible

# class EdgeModel(nn.Module):
#     def __init__(self, edge_dim, hid_dim):
#         super().__init__()
#         self.edge_dim = edge_dim
#         self.edge_mlp = Seq(Lin(edge_dim, hid_dim), 
#                             ReLU(), 
#                             Lin(hid_dim, edge_dim))

#     def forward(self, src, dst, edge_attr):
#         # src, dst: [E, F_x], where E is the number of edges.
#         # edge_attr: [E, F_e]
#         # batch: [E] with max entry B - 1.
#         if src is None or dst is None: #initialization with no node features
#             return edge_attr
#         else: #always take e^{(l-1)} while storing all intermediate edge features
#             out = torch.cat([src, dst, edge_attr[:,-self.edge_dim:]], 1)
#             return self.edge_mlp(out)


class NodeModelIn(nn.Module):
    def __init__(self, reduce='mean'):
        super().__init__()
        self.reduce = reduce

    def forward(self, x, edge_index, edge_attr):
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        row, col = edge_index
        n = x.size(0)
        if self.reduce == 'all':
            node_mean = scatter(edge_attr, col, dim=0, dim_size=n, reduce='mean')
            node_max = scatter(edge_attr, col, dim=0, dim_size=n, reduce='max')
            node_sum = scatter(edge_attr, col, dim=0, dim_size=n, reduce='sum')
            out = torch.cat([node_mean, node_max, node_sum], dim=-1)
        else:
            out = scatter(edge_attr, col, dim=0, dim_size=n, reduce=self.reduce) #[N, F_e]
        return out #self.linear(out) #[N, F_x]

class EdgeNodeMP(nn.Module):
    def __init__(self, node_dim=None, edge_dim=None, hid_dim=None, inter_dim=None, reduce='mean', 
                 norm=False, linear=False, mark=False):
        super().__init__()
        self.reduce = reduce
        self.edge_dim = edge_dim
        if inter_dim is None:
            inter_dim = edge_dim
        self.norm = norm
        if self.norm:
            self.norm_layer = LayerNorm(node_dim + edge_dim, mode='graph')
        if linear:
            self.edge_mlp_1 = Lin(node_dim + edge_dim, inter_dim)
            self.edge_mlp_1.bias.data.fill_(0)
            with torch.no_grad():
                self.edge_mlp_1.weight.copy_(torch.ones(inter_dim, node_dim + edge_dim))
        else:
            self.edge_mlp_1 = Seq(Lin(node_dim + edge_dim, hid_dim), 
                              ReLU(), 
                              Lin(hid_dim, inter_dim))
        self.mark = mark
        # self.node_mlp_2 = Seq(Lin(node_dim + edge_dim, hid_dim), 
        #                       ReLU(), 
        #                       Lin(hid_dim, node_dim))

    def forward(self, x, edge_index, edge_attr, node_weight=None):
        # x: [n, F_x], where n is the number of nodes. If self.mark: x is assumed to be the node degree [n,1]
        # edge_index: [2, E] with max entry n - 1.
        # edge_attr: [E, F_e]
        row, col = edge_index
        n = x.size(0)
        edge_new = torch.cat([x[row], edge_attr], dim=-1) #[E, F_x+F_e]
        if self.norm:
            edge_new = self.norm_layer(edge_new)
        edge_new = self.edge_mlp_1(edge_new) #[E, Fout]
        #edge_new = x[row] * edge_attr #TODO: relax the assumption F_x = F_e
        if self.reduce == 'all':
            node_mean = scatter(edge_new, col, dim=0, dim_size=n, reduce='mean')
            node_max = scatter(edge_new, col, dim=0, dim_size=n, reduce='max')
            node_sum = scatter(edge_new, col, dim=0, dim_size=n, reduce='sum')
            node_new = torch.cat([node_mean, node_max, node_sum], dim=-1)
        else:
            node_new = scatter(edge_new, col, dim=0, dim_size=n, #dim_size = n
                      reduce=self.reduce) #[n, Fout] #TODO: may need a better node model on sparse graphs
        
        if self.mark: #up-weight sparse nodes
            node_new = node_weight * node_new #[n, Fout]

        return node_new, edge_new  #[n, Fout*dim_factor]; [E, Fout]


class GNN(nn.Module):
    def __init__(self, n_layers, edge_dim, hid_dim, out_dim=None, inter_dim=None,
                 norm=False, reduce='mean', reduce_edge='mean', mark=False):
        super().__init__()
        self.n_layers = n_layers #Number of MP layers
        self.edge_dim = edge_dim
        self.hid_dim = hid_dim 
        self.out_dim = out_dim
        if inter_dim is None:
            inter_dim = edge_dim
        self.norm = norm
        self.reduce = reduce
        if reduce == 'all':
            node_factor = 3 #[max, mean, sum]
        else:
            node_factor = 1 #one agg function

        self.reduce_edge = reduce_edge
        self.mark = mark
        self.input_layer = NodeModelIn(reduce=self.reduce) #[n, 3F_n]
        self.layers = nn.ModuleList()
        self.layers.append(EdgeNodeMP(node_factor * edge_dim, edge_dim, hid_dim, inter_dim, self.reduce, norm, 
                                      mark=mark))
        for l in range(1,n_layers):
            self.layers.append(EdgeNodeMP(node_factor * inter_dim, inter_dim, hid_dim, inter_dim, self.reduce, norm, 
                                          mark=mark))
        if out_dim is not None:
            all_dim = (1+node_factor)*edge_dim + (1 + node_factor)*inter_dim*n_layers
            self.output_layer = Seq(Lin(all_dim, hid_dim), 
                              ReLU(),
                              Lin(hid_dim, hid_dim),
                              ReLU(), 
                              Lin(hid_dim, out_dim))
            #self.output_layer = Lin(edge_dim*(1+n_layers),out_dim)
            #self.output_layer.weight.data.fill_(1/(1+n_layers))
            #self.output_layer.bias.data.fill_(0)
    
    # def scatter_multi(self, data, batch, reduce='mean'):
    #     if reduce == 'all':
    #         data_mean = scatter(data, batch, reduce='mean')
    #         data_max = scatter(data, batch, reduce='max')
    #         data_sum = scatter(data, batch, reduce='sum')
    #         return torch.cat([data_mean, data_max, data_sum], dim=-1)
    #     else:
    #         return scatter(data, batch, reduce=reduce)

    
    def forward(self, data):
        h, edge_index, edge_attr, node_weight = data.x, data.edge_index, data.edge_attr, data.node_weight
        #node_batch, edge_batch = data.batch, data.edge_attr_batch
        h = self.input_layer(h, edge_index, edge_attr)
        global_node_attr = [scatter(h, data.batch, reduce='mean')]
        global_edge_attr = [scatter(edge_attr, data.edge_attr_batch, reduce=self.reduce_edge)]
        for layer in self.layers:
            h, edge_attr = layer(h, edge_index, edge_attr, node_weight)
            global_node_attr.append(scatter(h, data.batch, reduce='mean'))
            global_edge_attr.append(scatter(edge_attr, data.edge_attr_batch, reduce=self.reduce_edge))
        #all_edge_attr = torch.FloatTensor(global_edge_attr).unsqueeze(0).to(h.device) #(1,edge_dim*(1+n_layers))
        all_node_attr = torch.cat(global_node_attr, dim=-1)
        all_edge_attr = torch.cat(global_edge_attr, dim=-1) 
        all_attr = torch.cat([all_node_attr, all_edge_attr], dim=-1)
        if self.out_dim:
            out = self.output_layer(all_attr)
            #out = self.output_layer(all_edge_attr)
        else:
            out = None
        return all_node_attr, all_edge_attr, out



    
    

# class GlobalModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

#     def forward(self, x, edge_index, edge_attr, u, batch):
#         # x: [N, F_x], where N is the number of nodes.
#         # edge_index: [2, E] with max entry N - 1.
#         # edge_attr: [E, F_e]
#         # u: [B, F_u]
#         # batch: [N] with max entry B - 1.
#         out = torch.cat([
#             u,
#             scatter(x, batch, dim=0, reduce='mean'),
#         ], dim=1)
#         return self.global_mlp(out)
