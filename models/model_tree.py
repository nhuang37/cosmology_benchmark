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
from torch_geometric.nn.aggr import DeepSetsAggregation

def MSE_loss(ypred, y):
    return torch.mean((ypred - y)**2)

def variance(y):
    #compute mean vector (per feat), then average over variance per element
    mean = y.mean(axis=0)
    return torch.mean((y - mean)**2) 

class TreeGINConv(MessagePassing):
    def __init__(self, node_dim, hid_dim, out_dim, loop_flag=True):
        super(TreeGINConv, self).__init__()
        self.loop_flag = loop_flag
        self.nn = Seq(Lin(node_dim, hid_dim), 
                              ReLU(), 
                              Lin(hid_dim, out_dim))

    
    def forward(self, x, edge_index):
        h = x
        if self.loop_flag:
            self_loops = torch.arange(x.shape[0]).repeat(2,1).to(x.device)
            edge_loops = torch.cat([edge_index, self_loops], dim=1)
            h = self.propagate(edge_index=edge_loops, x=h)
        else:
            h = self.propagate(edge_index=edge_index, x=h)
        h = self.nn(h) #wait till all nodes are transformed!
        return h

    
    # def get_orders(self, pos, cut):
    #     mask_cut = (pos > cut).bool().squeeze()
    #     return [mask_cut, ~mask_cut] 

    # def forward(self, x, edge_index, pos):
    #     h = x
    #     if self.loop_flag:
    #         self_loops = torch.arange(x.shape[0]).repeat(2,1).to(x.device)
    #     if self.cut > 0:
    #         orders = self.get_orders(pos, self.cut)
    #     else:
    #         orders = [torch.ones(x.shape[0]).bool().to(x.device)]

    #     for node_mask in orders:
    #         source_node_idx = mask_to_index(node_mask)
    #         edge_subset_mask = torch.isin(edge_index[0], source_node_idx)
    #         subedge_index = edge_index[:,edge_subset_mask]
    #         if self.loop_flag:
    #             subedge_loops = torch.cat([subedge_index, self_loops], dim=1)
    #             h = self.propagate(edge_index=subedge_loops, x=h)
    #         else:
    #             h = self.propagate(edge_index=subedge_index, x=h)
        
    #     h = self.nn(h) #wait till all nodes are transformed!
    #     #print(h.shape)
    #         #print(h)
    #     return h
    
class TreeRegressor(nn.Module):
    def __init__(self, node_dim, hid_dim, out_dim, n_layer=1, loop_flag=True, 
                 node_level=False):
        super(TreeRegressor, self).__init__()
        self.conv_layers = nn.ModuleList([TreeGINConv(node_dim, hid_dim, hid_dim, loop_flag)])
        for layer in range(n_layer - 1):
            self.conv_layers.append(TreeGINConv(hid_dim, hid_dim, hid_dim, loop_flag))
        self.regressor = Seq(Lin(hid_dim, hid_dim), 
                              ReLU(), 
                              Lin(hid_dim, out_dim)) #Lin(hid_dim, out_dim)
        self.node_level = node_level

    def forward(self, x, edge_index, x_batch=None, pos=None, global_feat=None):
        for layer in self.conv_layers:
            x = layer(x, edge_index)
        if self.node_level:
            tree_pred = self.regressor(x)
        else:
            tree_out = scatter_mean(x, x_batch, dim=0)
            if global_feat is not None: #(bs, global_feat_dim)
                tree_out = torch.cat((tree_out, global_feat), dim=0)
            tree_pred = self.regressor(tree_out)
        return tree_pred

class MLPAgg(nn.Module):
    def __init__(self, node_dim, hid_dim, out_dim, agg_first=True):
        super(MLPAgg, self).__init__()
        self.regressor =Seq(Lin(node_dim, hid_dim), 
                              ReLU(), 
                              Lin(hid_dim, hid_dim),
                              ReLU(),
                              Lin(hid_dim, out_dim)) #Lin(hid_dim, out_dim)
        self.agg_first = agg_first
    
    def forward(self, x, x_batch):
        if self.agg_first:
            x_agg = scatter_mean(x, x_batch, dim=0)
            tree_pred = self.regressor(x_agg)
        else:
            node_out = self.regressor(x) 
            tree_pred = scatter_mean(node_out, x_batch, dim=0)
        return tree_pred

class DeepSet(nn.Module):
    def __init__(self, node_dim, hid_dim, out_dim, reduce='mean'):
        super(DeepSet, self).__init__()
        self.local_phi = Seq(Lin(node_dim, hid_dim), 
                        ReLU(), 
                        Lin(hid_dim, hid_dim),
                        )
        self.global_rho = Seq(Lin(hid_dim, hid_dim), 
                        ReLU(), 
                        Lin(hid_dim, out_dim),
                        )
        self.reduce = reduce
        #self.deepset = DeepSetsAggregation(local_nn=phi, global_nn=rho) #sum agg
    def forward(self, x, x_batch):
        if self.reduce == 'mean':
            hid = scatter_mean(self.local_phi(x), x_batch, dim=0)
        else:
            hid = scatter_sum(self.local_phi(x), x_batch, dim=0)
        out = self.global_rho(hid)
        #out = self.deepset(x, x_batch) 

        return out
    
def train_eval_model(model, train_loader, val_loader, 
                     mlp_only=False, edge_mp=False,
                     n_epochs=100, lr=1e-2, 
                     eval_every=1,
                     target_id=0, save_path=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    train_loss_steps, val_loss_eps = [], []
    for i in range(n_epochs):
        train_loss = train_model(model, train_loader, optimizer,
                                 mlp_only, edge_mp, target_id)
        train_loss_steps.extend(train_loss)
        if (i+1) % eval_every == 0:
            with torch.no_grad():
                _, _, val_loss, R2_om, R2_s8 = eval_model(model, val_loader, 
                                    mlp_only, edge_mp, target_id)
                val_loss_eps.append(val_loss)
                train_loss_avg = sum(train_loss)/len(train_loss)
                print(f'epoch={i}, train_loss={train_loss_avg:.4f}, val_loss={val_loss:.4f}, R2_om={R2_om:.4f}, R2_s8={R2_s8:.4f}')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return train_loss_steps, val_loss_eps
    
def train_model(model, train_loader, optimizer,
                mlp_only, edge_mp=False, target_id=None, 
                criterion=nn.MSELoss()):
                #criterion=nn.L1Loss(reduction='mean')):
    model.train()
    loss_hist = []
    device = next(model.parameters()).device
    #for data in data_list:
    for data in train_loader:
        data = data.to(device)
        if mlp_only:
            pred = model(data.x, data.batch)
        elif edge_mp:
            pred = model(data.x, data.edge_index, data.edge_attr)
        else:
            #orders = [torch.ones(x.shape[0]).bool().to(device)]
            pred = model(data.x, data.edge_index, data.batch) 
        #print(om_pred, data.y.float())
        if target_id is not None:
            loss = criterion(pred, data.y[:,target_id].unsqueeze(1).float()) #predicting omega_matter
        else:
            loss = criterion(pred, data.y)
        #loss_hist.append(loss.item())
        loss_hist.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_hist


def eval_model(model, eval_loader, 
               mlp_only, edge_mp=False, target_id=None, 
               criterion=nn.MSELoss(reduction='sum'), R2_sep=True):
               #criterion=nn.L1Loss(reduction='sum')):
    ##eval - assume full batch
    model.eval()
    device = next(model.parameters()).device
    target, pred_all = [], []
    loss = 0
    samples = 0
    with torch.no_grad():
        for data in eval_loader:
            data = data.to(device)
            samples += data.y.shape[0]

            if mlp_only:
                pred = model(data.x, data.batch)
            elif edge_mp:
                pred = model(data.x, data.edge_index, data.edge_attr)
            else:
                pred = model(data.x, data.edge_index, data.batch)

            if target_id is not None:
                y = data.y[:,target_id]
                loss += criterion(pred, y.unsqueeze(1)).item()
                target.extend(y.detach().cpu())
                pred_all.extend(pred.flatten().detach().cpu())
            else:
                loss += criterion(pred, data.y).item()
                target.append(data.y.detach().cpu())
                pred_all.append(pred.detach().cpu())
    #MSE
    loss_avg = loss/samples
    #R2
    pred_all = torch.cat(pred_all)
    target = torch.cat(target)
    if (R2_sep == True) and (target_id is None):
        mse_om = MSE_loss(pred_all[:,0], target[:,0])
        mse_s8 = MSE_loss(pred_all[:,1], target[:,1])
        var_om = variance(target[:,0])
        var_s8 = variance(target[:,1])
        R2_om = 1 - mse_om/var_om 
        R2_s8 = 1 - mse_s8/var_s8
        return target, pred_all, loss_avg, R2_om, R2_s8 
    else:
        mse = MSE_loss(pred_all, target)
        var = variance(target)
        R2 = 1 - mse/var

        return target, pred_all, loss_avg, R2

# def train_model(model, train_loader, mlp_only=False,
#                 n_epochs=100, lr=1e-2, target_id=1, save_path=None):
#     criterion = nn.L1Loss(reduction='mean') #nn.MSELoss() #
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     step = 0
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     model = model.to(device)
#     model.train()
#     loss_hist = []
#     for i in range(n_epochs):
#         loss_ep = 0
#         #for data in data_list:
#         for data in train_loader:
#             data = data.to(device)
#             if mlp_only:
#                 om_pred = model(data.x, data.x_batch)
#             else:
#                 #orders = [torch.ones(x.shape[0]).bool().to(device)]
#                 om_pred = model(data.x, data.edge_index, data.pos, data.x_batch) 
#             #om_pred = scatter_mean(om_pred_node, data.x_batch, dim=0)
#             #print(om_pred, data.y.float())
#             loss = criterion(om_pred, data.y[:,target_id].unsqueeze(1).float()) #predicting omega_matter
#             #loss_hist.append(loss.item())
#             loss_ep += loss.item()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             step +=1
#         loss_avg = loss_ep/len(train_loader)
#         loss_hist.append(loss_avg)
#         if (i+1) % 10 == 0:
#             print(f'epoch={i}, loss={loss_avg:.4f}')
#     if save_path is not None:
#         torch.save(model.state_dict(), save_path)
#     return loss_hist
    

def plot_result(train_target, train_pred, train_loss, 
                val_target, val_pred, val_loss, val_R2_om, val_R2_s8,
                model_name, target_id, fig_path, s=5):
    model_name_list = model_name.split("_")
    if target_id is not None: #create one plot for one specific target feature
        plt.scatter(train_target, train_pred, s=s, label='training', color='tab:blue', alpha=0.6)
        plt.scatter(val_target, val_pred, s=s, label="validation", color='tab:orange', alpha=0.6)
        plt.axline((0, 0), slope=1, color='gray', linestyle=':', label='y=x')
        plt.legend()
        target_name = r"$\Omega_m$" if target_id == 0 else r"$\sigma_8$"
        if target_id == 1:
            plt.xlim(0.6,1.0)
            plt.ylim(0.6,1.0)
        plt.xlabel(f"target {target_name}")
        plt.ylabel(f"predicted {target_name}")
        plt.title(f"{model_name_list[0]}: depth={model_name_list[2]},  \n target={target_name}, train_size={len(train_target)}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        fig.suptitle(f"{model_name_list[0]}: depth={model_name_list[2]},  \n target={target_name}, train_size={len(train_target)}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        if fig_path is not None:
            plt.savefig(fig_path, dpi=150)
    else: #two subplots for both (om, sigma8)
        # train_target = torch.cat(train_target)
        # train_pred = torch.cat(train_pred)
        # val_target = torch.cat(val_target)
        # val_pred = torch.cat(val_pred)
        print(train_target.shape, train_pred.shape, val_target.shape, val_pred.shape)
        fig, axs = plt.subplots(ncols=2, figsize=(12,4), dpi=150)
        target_name = {0: r"$\Omega_m$" , 1: r"$\sigma_8$"}
        for i in range(2):
            axs[i].scatter(train_target[:,i], train_pred[:,i], s=s, label='training', color='tab:blue', alpha=0.6)
            axs[i].scatter(val_target[:,i], val_pred[:,i], s=s, label="validation", color='tab:orange', alpha=0.6)
            axs[i].axline((0, 0), slope=1, color='gray', linestyle=':', label='y=x')
            axs[i].legend()
            axs[i].set_xlabel(f"target {target_name[i]}")
            axs[i].set_ylabel(f"predicted {target_name[i]}")
        axs[0].set_xlim(0,0.6)
        axs[0].set_ylim(0,0.6)
        axs[1].set_xlim(0.57,1.03)
        axs[1].set_ylim(0.57,1.03)
        fig.suptitle(f"{model_name_list[0]}: depth={model_name_list[2]},  \n  train_size={len(train_target)}, val_R2_om={val_R2_om:.4f}, val_R2_s8={val_R2_s8:.4f}")
        if fig_path is not None:
            fig.savefig(fig_path, dpi=150)


def eval_and_plot(model, train_loader, val_loader, target_id=0, 
                  mlp_only=False, edge_mp=False, model_name='MPNN', fig_path=None): 
    train_target, train_pred, train_loss, train_R2_om, train_R2_s8 = eval_model(model, train_loader, mlp_only, edge_mp, target_id)
    val_target, val_pred, val_loss, val_R2_om, val_R2_s8 = eval_model(model, val_loader, mlp_only, edge_mp, target_id)
    plot_result(train_target, train_pred, train_loss, \
                 val_target, val_pred, val_loss, val_R2_om, val_R2_s8, \
                model_name, target_id, fig_path)
    return train_loss, val_loss

def plot_train_val_loss(results, save_path):
    fig, axs =  plt.subplots(ncols=2, figsize=(12,4), dpi=150, sharey=True)
    axs[0].plot(results['train_steps'], color='tab:blue')
    axs[0].set_xlabel('gradient steps')
    axs[0].set_ylabel('loss')
    axs[0].set_title('training loss')
    axs[1].plot(results['val'], color='tab:orange', alpha=0.8)
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('loss')
    axs[1].set_title('validation loss')

    for ax in axs:
        ax.set_yscale('log')
    fig.savefig(save_path, dpi=150)

