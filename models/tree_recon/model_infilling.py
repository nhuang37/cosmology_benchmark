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
import numpy as np

class TreeConv(MessagePassing):
    def __init__(self, node_dim, hid_dim, out_dim, loop_flag=True):
        #Simple linear graph conv, without learnable params here
        super(TreeConv, self).__init__()
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
        return self.nn(h)


class TreeNodeClassifier(nn.Module):
    def __init__(self, node_dim, hid_dim, out_dim, n_layer=1, loop_flag=True):
        super(TreeNodeClassifier, self).__init__()
        self.conv_layers = nn.ModuleList([TreeConv(node_dim, hid_dim, hid_dim, loop_flag)])
        for layer in range(n_layer - 1):
            self.conv_layers.append(TreeConv(hid_dim, hid_dim, hid_dim, loop_flag))

        self.classifier = Seq(Lin(hid_dim, hid_dim), 
                              ReLU(), 
                              Lin(hid_dim, out_dim)) 

    def forward(self, x, edge_index):
        for layer in self.conv_layers:
            x = layer(x, edge_index)
        node_pred = self.classifier(x)
        return node_pred


def bootstrap_acc(y_true, y_pred, n_bootstrap=1000, seed=None):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    acc_values = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        acc = np.mean(y_true[indices] == y_pred[indices])
        acc_values.append(acc)

    acc_values = np.array(acc_values)
    return acc_values.mean(), acc_values.std()

def eval_classifier(model, data_list, 
                    criterion=torch.nn.CrossEntropyLoss(), device="cuda", boot_flag=False):
    model.eval()
    loss_all, acc_num, acc_denom = 0, 0, 0
    y_true, y_pred = [], []
    #print(f"data list has {len(data_list)}")
    with torch.no_grad():
        for data in data_list:
            mask = data.vn_mask #get all vn nodes
            out = model(data.x.to(device), data.edge_index.to(device))
            pred = out[mask]
            target = data.label[mask].to(device).flatten()
            loss = criterion(pred, target)
            loss_all += loss.item()
            pred_labels = torch.argmax(pred, dim=-1)
            #print(torch.unique(pred_labels, return_counts=True))
            acc_num += (target == pred_labels).sum() 
            acc_denom += target.shape[0]
            #print(pred_labels.shape, target.shape)
            y_true.extend(target.cpu().numpy().astype(int))
            y_pred.extend(pred_labels.detach().cpu().numpy().astype(int))
    acc = acc_num / acc_denom
    if boot_flag:
        # print(len(y_true), y_true[:5])
        # print(len(y_pred), y_pred[:5])
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc_boot, acc_boot_std = bootstrap_acc(y_true, y_pred, n_bootstrap=100)
        return loss_all/len(data_list), acc, acc_boot, acc_boot_std
    else:
        return loss_all/len(data_list), acc



def train_eval_classifier(model, train_list, val_list, save_dir, 
                          num_epochs = 100, lr=0.01, weight_decay=0, device="cuda"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss() #class weights makes the optim more weird...
    model = model.to(device)
    train_loss, val_loss_out = [], []
    best_acc_out = 0 #ultimately evaluate at hold on set
    # Training loop
    for epoch in range(num_epochs):
        ep_loss = 0
        model.train()  # Set model to training mode
        for data in train_list:
            optimizer.zero_grad()  # Clear gradients from the previous step
            # Forward pass - full batch
            out = model(data.x.to(device), data.edge_index.to(device))
            pred = out[data.vn_mask]
            target = data.label[data.vn_mask].to(device).flatten()
            # Alternative; outdated
            # pred = out[data.vn_train_idx]
            # target = data.label[data.vn_train_idx].to(device).flatten()
            
            # Compute loss only for training nodes
            loss = criterion(pred, target)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        train_loss.append(ep_loss/len(train_list))
        # Print progress
        model.eval()
        with torch.no_grad():
            #loss_val, acc_val = eval_classifier(model, train_list, criterion=criterion, mode='val')
            loss_val_out, acc_val_out = eval_classifier(model, val_list, criterion=criterion)
            val_loss_out.append(loss_val_out)
            if acc_val_out > best_acc_out:
                torch.save(model.state_dict(), f"{save_dir}/model.pt")
                best_acc_out = acc_val_out

            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}; Val Loss Hold-Out: {loss_val_out:.4f}, Val Acc. Hold-Out: {acc_val_out:.4f}')
    return train_loss, val_loss_out, best_acc_out