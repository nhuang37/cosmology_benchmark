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

class TreeConv(MessagePassing):
    def __init__(self, loop_flag=True):
        #Simple linear graph conv, without learnable params here
        super(TreeConv, self).__init__()
        self.loop_flag = loop_flag

    def forward(self, x, edge_index):
        h = x
        if self.loop_flag:
            self_loops = torch.arange(x.shape[0]).repeat(2,1).to(x.device)
            edge_loops = torch.cat([edge_index, self_loops], dim=1)
            h = self.propagate(edge_index=edge_loops, x=h)
        else:
            h = self.propagate(edge_index=edge_index, x=h)
        return h


class TreeNodeClassifier(nn.Module):
    def __init__(self, node_dim, hid_dim, out_dim, n_layer=1, loop_flag=True):
        super(TreeNodeClassifier, self).__init__()
        self.conv_layers = nn.ModuleList([TreeConv(loop_flag)])
        for layer in range(n_layer - 1):
            self.conv_layers.append(TreeConv(loop_flag))
        self.classifier = Seq(Lin(node_dim, hid_dim), 
                              ReLU(), 
                              Lin(hid_dim, out_dim)) 

    def forward(self, x, edge_index):
        for layer in self.conv_layers:
            x = layer(x, edge_index)
        node_pred = self.classifier(x)
        return node_pred

def eval_classifier(model, data_list, mode='val', criterion=torch.nn.CrossEntropyLoss(), device="cuda"):
    model.eval()
    loss_all, acc_num, acc_denom = 0, 0, 0
    with torch.no_grad():
        for data in data_list:
            if mode == 'val':
                mask = data.vn_val_idx
            else:
                mask = data.vn_test_idx
            out = model(data.x.to(device), data.edge_index.to(device))
            pred = out[mask]
            target = data.label[mask].to(device).flatten()
            loss = criterion(pred, target)
            loss_all += loss.item()
            pred_labels = torch.argmax(pred, dim=-1)
            #print(torch.unique(pred_labels, return_counts=True))
            acc_num += (target == pred_labels).sum() 
            acc_denom += target.shape[0]
    acc = acc_num / acc_denom
    return loss_all/len(data_list), acc


def train_eval_classifier(model, data_list, save_dir, 
                          num_epochs = 100, lr=0.01, weight_decay=0, device="cuda"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #TODO: deal with imbalanced data
    # total = data.vn_train_idx.shape[0]
    # pos = data.label[data.vn_train_idx].sum()
    # neg = total - pos
    class_weights = torch.FloatTensor([1.0, 1.0]).to(device)
    # print(class_weights)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights) #class weights makes the optim more weird...
    model = model.to(device)
    train_loss, val_loss = [], []
    best_acc = 0
    # Training loop
    for epoch in range(num_epochs):
        ep_loss = 0
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear gradients from the previous step
        for data in data_list:
            # Forward pass - full batch
            out = model(data.x.to(device), data.edge_index.to(device))
            pred = out[data.vn_train_idx]
            target = data.label[data.vn_train_idx].to(device).flatten()
            
            # Compute loss only for training nodes
            loss = criterion(pred, target)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        train_loss.append(ep_loss/len(data_list))
        # Print progress
        model.eval()
        with torch.no_grad():
            loss_val, acc_val = eval_classifier(model, data_list, criterion=criterion)
            val_loss.append(loss_val)
            if acc_val > best_acc:
                torch.save(model.state_dict(), f"{save_dir}/model.pt")
                best_acc = acc_val

            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}; Val Loss: {loss_val:.4f}, Val Acc.: {acc_val:.4f}')
    return train_loss, val_loss, best_acc