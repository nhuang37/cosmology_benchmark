import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter, degree
from torch_geometric.nn import LayerNorm
#from model import NodeModelIn, EdgeNodeMP
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import radius_graph, knn_graph
import torch.nn.functional as F
from torch_geometric.data import Data

def count_parameters(model, trainable_only=True):
    """Count the number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        trainable_only (bool): If True, only count parameters that require gradients.

    Returns:
        int: Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

### Simple GNN
class MLP(nn.Module):
    def __init__(self, widths, activation='gelu'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = getattr(F, activation)
        for i in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[i], widths[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class VelocityEdgeConv(MessagePassing):
    #https://arxiv.org/pdf/2411.19484, eqn15 inspired: v(x_i) = C sum_{j} MLP(x_j) * (x_j - x_i) / |x_j - x_i|^3
    def __init__(self, in_channels, d_hidden, out_channels, edge_dim):
        super().__init__(aggr='mean') #  aggregation. => can try "add"
        self.density_mlp = Seq(Lin(in_channels, d_hidden),
                       ReLU(),
                       Lin(d_hidden, d_hidden),
                       ReLU(),
                       Lin(d_hidden, out_channels))
        self.edge_mlp = Seq(Lin(edge_dim, d_hidden),
                       ReLU(),
                       Lin(d_hidden, d_hidden),
                       ReLU(),
                       Lin(d_hidden, out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) #sum over neighbor messages

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        edge_feat = self.edge_mlp(edge_attr)
        return self.density_mlp(x_j)*edge_feat #[E, out_channels]
    
class VelocityGNN(torch.nn.Module):
    def __init__(self, node_dim=3, edge_dim=3, 
                 d_hidden=24, 
                 message_passing_steps=3, activation='relu'):
        super().__init__()
        self.message_passing_steps = message_passing_steps

        self.gnn_layers = nn.ModuleList([VelocityEdgeConv(node_dim, d_hidden, d_hidden, edge_dim)])
        #self.gnn_layers = nn.ModuleList()
        for i in range(message_passing_steps-1):
            self.gnn_layers.append(VelocityEdgeConv(d_hidden, d_hidden, d_hidden, edge_dim))
        readout_dims = [d_hidden, d_hidden, node_dim]
        self.readout_mlp = MLP(readout_dims, activation=activation)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index, edge_attr)

        return self.readout_mlp(x)  # shape [num_nodes, node_dim]


class VelocityHierarchicalGNN(torch.nn.Module):
    def __init__(self, node_dim=3, edge_dim=3, 
                 d_hidden=24, 
                 message_passing_steps=3, activation='relu'):
        super().__init__()
        self.message_passing_steps = message_passing_steps

        self.gnn_coarse = nn.ModuleList([VelocityEdgeConv(node_dim, d_hidden, d_hidden, edge_dim)])
        for i in range(message_passing_steps-1):
            self.gnn_coarse.append(VelocityEdgeConv(d_hidden, d_hidden, d_hidden, edge_dim))
        
        self.gnn_fine = nn.ModuleList([VelocityEdgeConv(node_dim, d_hidden, d_hidden, edge_dim)])
        for i in range(message_passing_steps-1):
            self.gnn_fine.append(VelocityEdgeConv(d_hidden, d_hidden, d_hidden, edge_dim))
        
        readout_dims = [d_hidden, d_hidden, node_dim]
        self.readout_mlp_coarse = MLP(readout_dims, activation=activation)
        self.readout_mlp_fine = MLP(readout_dims, activation=activation)

    def forward(self, data, data_coarse):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_c, edge_index_c, edge_attr_c, cluster = data_coarse.x, data_coarse.edge_index, data_coarse.edge_attr, data_coarse.cluster_idx
        for gnn in self.gnn_coarse:
            x_c = gnn(x_c, edge_index_c, edge_attr_c)
        out_c = self.readout_mlp_coarse(x_c) #shape [num_coarse_nodes, node_dim]
        out_c_tofine = out_c[cluster] #shape [num_nodes, node_dim]

        for gnn in self.gnn_fine:
            x = gnn(x, edge_index, edge_attr)
        correct_term = self.readout_mlp_fine(x)  # shape [num_nodes, node_dim]
        out = out_c_tofine + correct_term 

        return out, out_c 