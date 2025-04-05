from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, mask_to_index, index_to_mask
from torch_geometric.data import Data, DataLoader, InMemoryDataset, Dataset
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch import optim
from torch_scatter import scatter_sum, scatter_mean
import pickle
import matplotlib.pyplot as plt
import copy
from torch.nn.functional import one_hot


class TreeGINConv(MessagePassing):
    def __init__(self, node_dim, hid_dim, out_dim, loop_flag=True, cut=0):
        super(TreeGINConv, self).__init__()
        self.loop_flag = loop_flag
        self.cut = cut
        self.nn = Seq(Lin(node_dim, hid_dim), 
                              ReLU(), 
                              Lin(hid_dim, out_dim))
    
    def get_orders(self, pos, cut):
        mask_cut = (pos > cut).bool().squeeze()
        return [mask_cut, ~mask_cut] 

    def forward(self, x, edge_index, pos):
        h = x
        if self.loop_flag:
            self_loops = torch.arange(x.shape[0]).repeat(2,1).to(x.device)
        if self.cut > 0:
            orders = self.get_orders(pos, self.cut)
        else:
            orders = [torch.ones(x.shape[0]).bool().to(x.device)]

        for node_mask in orders:
            source_node_idx = mask_to_index(node_mask)
            edge_subset_mask = torch.isin(edge_index[0], source_node_idx)
            subedge_index = edge_index[:,edge_subset_mask]
            if self.loop_flag:
                subedge_loops = torch.cat([subedge_index, self_loops], dim=1)
                h = self.propagate(edge_index=subedge_loops, x=h)
            else:
                h = self.propagate(edge_index=subedge_index, x=h)
        
        h = self.nn(h) #wait till all nodes are transformed!
        #print(h.shape)
            #print(h)
        return h
    
class TreeRegressor(nn.Module):
    def __init__(self, node_dim, hid_dim, out_dim, n_layer=1, loop_flag=True, cut=0):
        super(TreeRegressor, self).__init__()
        self.conv_layers = nn.ModuleList([TreeGINConv(node_dim, hid_dim, hid_dim, loop_flag, cut)])
        for layer in range(n_layer - 1):
            self.conv_layers.append(TreeGINConv(hid_dim, hid_dim, hid_dim, loop_flag, cut))
        self.regressor = Seq(Lin(hid_dim, hid_dim), 
                              ReLU(), 
                              Lin(hid_dim, out_dim)) #Lin(hid_dim, out_dim)

    def forward(self, x, edge_index, pos, x_batch):
        for layer in self.conv_layers:
            x = layer(x, edge_index, pos)
        tree_out = scatter_mean(x, x_batch, dim=0)
        tree_pred = self.regressor(tree_out)
        return tree_pred

class MLPAgg(nn.Module):
    def __init__(self, node_dim, hid_dim, out_dim):
        super(MLPAgg, self).__init__()
        self.regressor =Seq(Lin(node_dim, hid_dim), 
                              ReLU(), 
                              Lin(hid_dim, hid_dim),
                              ReLU(),
                              Lin(hid_dim, out_dim)) #Lin(hid_dim, out_dim)
    
    def forward(self, x, x_batch):
        node_out = self.regressor(x) 
        tree_pred = scatter_mean(node_out, x_batch, dim=0)
        return tree_pred
    


def train_model(model, train_loader, mlp_only=False,
                n_epochs=100, lr=1e-2, target_id=1, scaling_factor=1e13):
    criterion = nn.L1Loss(reduce='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    step = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    for i in range(n_epochs):
        loss_ep = 0
        #for data in data_list:
        for data in train_loader:
            data = data.to(device)
            x = data.x[:,:1] / scaling_factor 
            if mlp_only:
                om_pred = model(x, data.x_batch)
            else:
                #orders = [torch.ones(x.shape[0]).bool().to(device)]
                om_pred = model(x, data.edge_index, data.pos, data.x_batch) 
            #om_pred = scatter_mean(om_pred_node, data.x_batch, dim=0)
            #print(om_pred, data.y.float())
            loss = criterion(om_pred, data.y[:,target_id:target_id+1].float()) #predicting omega_matter
            #loss_hist.append(loss.item())
            loss_ep += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step +=1
        if (i+1) % 10 == 0:
            print(f'epoch={i}, loss={(loss_ep/len(train_loader)):.4f}')
    
def eval_model(model, eval_loader,  mlp_only=False,
                target_id=1, scaling_factor=1e13):
    ##eval - assume full batch
    criterion = nn.L1Loss(reduce='sum')
    model.eval()
    data = next(iter(eval_loader))
    device = next(model.parameters()).device
    data = data.to(device)
    x = data.x[:,:1] / scaling_factor 
    if mlp_only:
        om_pred = model(x, data.x_batch)
    else:
        om_pred = model(x, data.edge_index, data.pos, data.x_batch) 
    loss = criterion(om_pred, data.y[:,target_id:target_id+1]).item()
    return data.y[:,target_id:target_id+1].detach().cpu(), om_pred.detach().cpu(), loss