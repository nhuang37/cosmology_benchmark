import numpy as np
import scipy.spatial as SS
import random
from torch_geometric.utils import scatter, mask_to_index, index_to_mask
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn.models import EdgeCNN
from torch_geometric.nn.pool import voxel_grid, avg_pool_x, avg_pool
from torch_geometric.transforms import GridSampling
from torch_geometric.nn.conv import MessagePassing
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch import optim
from torch_scatter import scatter_sum, scatter_mean
import pickle
import matplotlib.pyplot as plt
import copy
from models.cloud_velocity.model_velocity import VelocityGNN, VelocityHierarchicalGNN, count_parameters
from models.cloud_velocity.simple_velocity import load_point_cloud_h5
from utils.graph_util import build_graph, pbc_distance, coarsen_graph
import h5py
import os
import argparse
import pathlib
from itertools import product
import json
import time

def compute_mse_R2(pred, target):
    var = torch.mean((target - target.mean())**2)
    mse = torch.mean((pred - target)**2)
    R2 =  1 - mse/var
    return mse, R2

def train_model(model, train_loader, optimizer, criterion, device):
    ep_loss = 0
    model.train()  # Set model to training mode
    for data in train_loader:
        optimizer.zero_grad()  # Clear gradients from the previous step -- per cloud!
        data = data.to(device)
        # Forward pass - full batch
        pred = model(data)    
        #pred = model(data.x, data.edge_index)
        # Compute loss only for training nodes
        loss = criterion(pred, data.y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
    return ep_loss

def eval_model(model, val_loader, device, return_pred=False):
    model.eval()
    mse_all, R2_all = [], []
    if return_pred:
        pred_all, y_all = [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            pred = model(data)
            mse, R2 = compute_mse_R2(pred, data.y)
            mse_all.append(mse.item())
            R2_all.append(R2.item())
            if return_pred:
                pred_all.append(pred.detach().cpu())
                y_all.append(data.y.cpu())
    if return_pred:
        return mse_all, R2_all, pred_all, y_all #torch.stack(pred_all), torch.stack(y_all) #(num, 5000, 3)
    else:
        return mse_all, R2_all, None, None


def train_hierarchical_model(model, pair_train_loader, optimizer, criterion, device, alpha=0.5):
    ep_loss, ep_loss_coarse = 0, 0
    model.train()  # Set model to training mode
    for data, data_coarse in pair_train_loader:
        optimizer.zero_grad()  # Clear gradients from the previous step -- per cloud!
        data = data.to(device)
        data_coarse = data_coarse.to(device)
        # Forward pass - full batch
        pred, pred_coarse = model(data, data_coarse)    
        # Compute loss only for training nodes
        loss_fine = criterion(pred, data.y)
        loss_coarse = criterion(pred_coarse, data_coarse.y)
        loss = alpha*loss_fine + (1-alpha)*loss_coarse
        # Backpropagation
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
        ep_loss_coarse += loss.item()
    return ep_loss, ep_loss_coarse



def eval_hierarchical_model(model, pair_val_loader, device):
    model.eval()
    mse_all, R2_all, mse_coarse_all, R2_coarse_all = [], [], [], []
    with torch.no_grad():
        for data, data_coarse in pair_val_loader:
            data = data.to(device)
            data_coarse = data_coarse.to(device)
            pred, pred_coarse = model(data, data_coarse)    
            mse, R2 = compute_mse_R2(pred, data.y)
            mse_all.append(mse.item())
            R2_all.append(R2.item())

            mse_c, R2_c = compute_mse_R2(pred_coarse, data_coarse.y)
            mse_coarse_all.append(mse_c.item())
            R2_coarse_all.append(R2_c.item())

    return mse_all, R2_all, mse_coarse_all, R2_coarse_all

def train_eval_node_regressor(model, train_loader, val_loader, save_dir=None, 
                          num_epochs = 100, lr=0.01, weight_decay=0, device="cuda"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # print(class_weights)
    criterion = torch.nn.MSELoss()
    model = model.to(device)
    train_loss, val_loss, R2_all = [], [], []
    R2_best = 0
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        ep_loss = train_model(model, train_loader, optimizer, criterion, device)
        end_time = time.time()
        train_loss.append(ep_loss/len(train_loader))
        val_mse, val_R2, _, _ = eval_model(model, val_loader, device)
        val_loss.append(sum(val_mse) / len(val_mse))
        R2 = sum(val_R2)/len(val_R2)
        R2_all.append(R2)
        if R2 > R2_best:
            if save_dir is not None:
                torch.save(model.state_dict(), f"{save_dir}/model.pt")
            R2_best = R2
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss[-1]:.4f}; Val Loss: {val_loss[-1]:.4f}; R2: {R2:.4f}, train_1ep_time={(end_time - start_time):.4f}')
    return train_loss, val_loss, R2_best

def train_eval_node_regressor_hierarchical(model, train_loader, val_loader, save_dir=None, 
                          num_epochs = 100, lr=0.01, weight_decay=0, device="cuda", alpha=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # print(class_weights)
    criterion = torch.nn.MSELoss()
    model = model.to(device)
    train_loss, val_loss, R2_all = [], [], []
    train_loss_c, val_loss_c, R2_all_c = [], [], []
    R2_best = 0
    # Training loop
    for epoch in range(num_epochs):
        ep_loss, ep_loss_c = train_hierarchical_model(model, train_loader, optimizer, criterion, device, 
                                                      alpha)
        train_loss.append(ep_loss/len(train_loader))
        train_loss_c.append(ep_loss_c/len(train_loader))

        val_mse, val_R2, val_mse_coarse, val_R2_coarse = eval_hierarchical_model(model, val_loader, device)
        val_loss.append(sum(val_mse) / len(val_mse))
        val_loss_c.append(sum(val_mse_coarse) / len(val_mse_coarse))

        R2 = sum(val_R2)/len(val_R2)
        R2_all.append(R2)
        R2_c = sum(val_R2_coarse)/len(val_R2_coarse)
        R2_all_c.append(R2_c)

        if R2 > R2_best:
            if save_dir is not None:
                torch.save(model.state_dict(), f"{save_dir}/model.pt")
            R2_best = R2
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss[-1]:.4f}; Val Loss: {val_loss[-1]:.4f}; R2: {R2:.4f}')
    return train_loss, val_loss, R2_best 

class PairedDataset(Dataset):
    def __init__(self, list1, list2):
        assert len(list1) == len(list2), "Datasets must be the same length"
        self.list1 = list1
        self.list2 = list2

    def __len__(self):
        return len(self.list1)

    def __getitem__(self, idx):
        return self.list1[idx], self.list2[idx]

def build_dataloaders(train_dataset, valid_dataset, test_dataset=None, shuffle=True, batch_size=128):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=['edge_attr'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, follow_batch=['edge_attr'])
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, follow_batch=['edge_attr'])
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader


def compute_X_V_mean_std(dataset):
    X_mean, X_std = [], []
    V_mean, V_std = [], []
    for data in dataset:
        X_mean.append(data.x.mean(axis=0))
        X_std.append(data.x.std(axis=0))
        V_mean.append(data.y.mean(axis=0))
        V_std.append(data.y.std(axis=0))
    return torch.stack(X_mean, dim=0).mean(axis=0), torch.stack(X_std, dim=0).mean(axis=0), \
            torch.stack(V_mean, dim=0).mean(axis=0), torch.stack(V_std, dim=0).mean(axis=0)

def standardize_data(data, X_mean, X_std, V_mean, V_std):
    data.x = (data.x - X_mean)/X_std 
    data.y = (data.y - V_mean)/V_std 
    return data

def standardize_dataset(PyGdataset, train_ranks, val_ranks, test_ranks=None):
    train_set = [data for i, data in enumerate(PyGdataset) if i in train_ranks]
    X_mean, X_std, V_mean, V_std = compute_X_V_mean_std(train_set)
    train_set = [standardize_data(data, X_mean, X_std, V_mean, V_std) for data in train_set]
    val_set = [standardize_data(data, X_mean, X_std, V_mean, V_std) for i, data in enumerate(PyGdataset) if i in val_ranks]
    if test_ranks is None:
        return train_set, val_set
    else:
        test_set = [standardize_data(data, X_mean, X_std, V_mean, V_std) for i, data in enumerate(PyGdataset) if i in test_ranks]
        return train_set, val_set, test_set


def run_training(train_path, val_path, train_all,
                 hid_dim, n_layer, lr, save_dir, args,
                 coarse_train_path=None, coarse_val_path=None):
    if train_all:
        start_time = time.time()
        train_set = []
        for (start_idx, end_idx) in zip([0,5000,10000,15000],[5000,10000,15000,19651]):
            file_path = f"{args.data_dir}/Quijote_Rc=0.1_graph_coarsen=False_train_start={start_idx}_end={end_idx}.pt"
            cur_set = torch.load(file_path)
            train_set = train_set + cur_set
        end_time = time.time()
        print(f"loaded {len(train_set)} training clouds, used {(end_time - start_time):.4f} secs!")
    else:
        train_set = torch.load(train_path)
    val_set = torch.load(val_path) 
    if args.HGNN_flag:
        coarse_train_set = torch.load(coarse_train_path) 
        coarse_val_set = torch.load(coarse_val_path)

    X_mean, X_std, V_mean, V_std = compute_X_V_mean_std(train_set)
    train_set = [standardize_data(data, X_mean, X_std, V_mean, V_std) for data in train_set]
    val_set = [standardize_data(data, X_mean, X_std, V_mean, V_std) for data in val_set]
    print(len(train_set), len(val_set))
    if args.HGNN_flag:
        train_set_c = [standardize_data(data, X_mean, X_std, V_mean, V_std) for data in coarse_train_set]
        val_set_c = [standardize_data(data, X_mean, X_std, V_mean, V_std) for data in coarse_val_set]
        train_set_pair = PairedDataset(train_set, train_set_c)
        val_set_pair = PairedDataset(val_set, val_set_c)
        train_loader, val_loader, _ = build_dataloaders(train_set_pair, val_set_pair, 
                                                        batch_size=1, shuffle=True)
        model = VelocityHierarchicalGNN(node_dim=3, d_hidden=hid_dim, message_passing_steps=n_layer, activation='relu')
    else:
        train_loader, val_loader, _ = build_dataloaders(train_set, val_set, batch_size=1, shuffle=True)
        model = VelocityGNN(node_dim=3, d_hidden=hid_dim, message_passing_steps=n_layer, activation='relu')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"model has {count_parameters(model)} params!")

    if args.HGNN_flag:
        _, _, R2_best = train_eval_node_regressor_hierarchical(model, train_loader, val_loader, 
                                                              num_epochs=args.num_epochs, lr=lr,
                                                              device=device, save_dir=save_dir)
    
    else:
        _, _, R2_best = train_eval_node_regressor(model, train_loader, val_loader, 
                                                 num_epochs=args.num_epochs, lr=lr,
                                                 device=device, save_dir=save_dir)
    return R2_best


def hyperparameter_search(args):
    n_layers_list = [5] 
    d_hidden_list = [64]
    lr_list = [1e-3]
    filename = args.processed_train_path            # Get the file name
    prefix = filename.split('_')[0]    
    train_path = f"{args.data_dir}/{args.processed_train_path}"
    val_path =  f"{args.data_dir}/{args.processed_val_path}"

    results = []
    best_R2 = float('-inf')
    best_config = None

    for n_layer, d_hidden, lr in product(n_layers_list, d_hidden_list, lr_list):
        print(f"Running n_layer={n_layer}, d_hidden={d_hidden}, lr={lr}")
        save_dir = f"/mnt/home/thuang/playground/velocity_prediction/GNN_search/{prefix}/hid={d_hidden}_layers={n_layer}_lr={lr:.4f}"      
        os.makedirs(save_dir, exist_ok=True)

        R2_val = run_training(train_path, val_path, train_all=args.train_all, 
                              hid_dim=d_hidden, n_layer=n_layer, lr=lr, save_dir=save_dir, args=args)
        result_entry = {'n_layer': n_layer, 'd_hidden': d_hidden, 'lr': lr,
                        'val_R2': R2_val, 'save_dir': save_dir}
        results.append(result_entry)
        print(f"\nBest validation for this hyper-param R²: {R2_val:.4f}")

        if R2_val > best_R2:
            best_R2 = R2_val
            best_config = result_entry

    print(f"\nBest validation R²: {best_R2:.4f}")
    # === Save all search results ===
    results_path = os.path.join("/mnt/home/thuang/playground/velocity_prediction/GNN_search", "hyperparameter_search_results.json")
    # with open(results_path, 'w') as f:
    #     json.dump(results, f, indent=4)
    # print(f"Search results saved to: {results_path}")

    # Load existing results if file exists
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
    else:
        existing_results = []
    
    # Append new result and save
    existing_results.append(results)
    with open(results_path, 'w') as f:
        json.dump(existing_results, f, indent=4)

    print(f"Appended test evaluation results to: {results_path}")

    # === Save best config ===
    best_config_path = os.path.join(args.output_dir, "best_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=4)
    print(f"Best config saved to: {best_config_path}")

    return best_config, results

def eval_pretrained_model(args):
    ##load data
    train_path = f"{args.data_dir}/{args.processed_train_path}"
    if args.train_all:
        start_time = time.time()
        train_set = []
        for (start_idx, end_idx) in zip([0,5000,10000,15000],[5000,10000,15000,19651]):
            file_path = f"{args.data_dir}/Quijote_Rc=0.1_train_start={start_idx}_end={end_idx}.pt"
            cur_set = torch.load(file_path)
            train_set = train_set + cur_set
        end_time = time.time()
        print(f"loaded {len(train_set)} training clouds, used {(end_time - start_time):.4f} secs!")
    else:
        train_path = f"{args.data_dir}/{args.processed_train_path}"
        train_set = torch.load(train_path)

    test_path =  f"{args.data_dir}/{args.processed_test_path}"
    test_set = torch.load(test_path) 
    
    X_mean, X_std, V_mean, V_std = compute_X_V_mean_std(train_set)
    torch.save([X_mean, X_std, V_mean, V_std], f"{args.output_dir}/XV_mean_std.pt")
    print(f"saved training set mean / std !")
    if args.test_sample_idx_end is not None: #save the test set prediction on the first cloud
        test_set = [standardize_data(data, X_mean, X_std, V_mean, V_std) for i, data in enumerate(test_set) if i < args.test_sample_idx_end] #TODO: trick by using train_set for plotting purpose
        return_pred = True    
    else:
        test_set = [standardize_data(data, X_mean, X_std, V_mean, V_std) for data in test_set]
        return_pred = False

    if args.HGNN_flag:
        train_path_c = f"{args.data_dir}/{args.processed_train_path_coarse}"
        train_set_c = torch.load(train_path_c)
        test_path_c = f"{args.data_dir}/{args.processed_test_path_coarse}"
        test_set_c = torch.load(test_path_c)
        X_mean, X_std, V_mean, V_std = compute_X_V_mean_std(train_set_c)
        test_set_c = [standardize_data(data, X_mean, X_std, V_mean, V_std) for data in test_set_c]
        test_set_pair = PairedDataset(test_set, test_set_c)
        test_loader = DataLoader(test_set_pair, batch_size=1, shuffle=False, follow_batch=['edge_attr'])
    else:
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, follow_batch=['edge_attr'])

    ##load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(args.output_dir, "best_config.json"), 'r') as f:
        best_config = json.load(f)
    if args.HGNN_flag:
        model = VelocityHierarchicalGNN(
            node_dim=3,
            d_hidden=best_config['d_hidden'],
            message_passing_steps=best_config['n_layer'],
            activation='relu'
        )
    else:
        model = VelocityGNN(
            node_dim=3,
            d_hidden=best_config['d_hidden'],
            message_passing_steps=best_config['n_layer'],
            activation='relu'
        )

    model.load_state_dict(torch.load(best_config['save_dir']))
    model.to(device)
    model.eval()

    ##Run eval
    print(f"Evaluating best model with size = {count_parameters(model)} on test set...")
    if args.HGNN_flag:
        _, R2_all, _, _ = eval_hierarchical_model(model, test_loader, device)
    else:
        _, R2_all, pred, target = eval_model(model, test_loader, device, return_pred=return_pred)
    
    R2_test = sum(R2_all) / len(R2_all)
    print(f"Best Model Test R²: {R2_test:.4f}")

    if return_pred:
        prefix = args.processed_train_path.split('_')[0]  
        test_pred_path = os.path.join(args.output_dir, f"{prefix}_test_pred_{args.test_sample_idx_end}.pt")
        torch.save([pred, target, X_mean, X_std, V_mean, V_std], test_pred_path)
        print(f"Saved test velocity predictions to: {test_pred_path}")
    else:
        ## Save to file
        results_dict = {
            'd_hidden': best_config['d_hidden'],
            'n_layer': best_config['n_layer'],
            'lr': best_config['lr'],
            'val_R2': best_config['val_R2'],
            'test_R2': R2_test,
            'save_dir': best_config['save_dir']
        }

        test_result_path = os.path.join(args.output_dir, "test_R2_result.json")
        with open(test_result_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"Saved test evaluation results to: {test_result_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, \
                        default='/mnt/home/thuang/ceph/playground/datasets/point_clouds/velocity_Quijote', help='data dir')

    parser.add_argument('--processed_train_path', type=str, \
                        default='Quijote_Rc=0.1_train.pt', help='fine-grained train data path')
    parser.add_argument('--processed_val_path', type=str, \
                        default='Quijote_Rc=0.1_val.pt', help='fine-grained val data path')
    parser.add_argument('--processed_test_path', type=str, \
                        default='Quijote_Rc=0.1_test.pt', help='fine-grained test data path')
    #optiona hierarhical MPNN -- not used for baseline
    parser.add_argument('--processed_coarsen_dataset_path', type=str, \
                        default='/mnt/home/thuang/ceph/playground/datasets/point_clouds/Rc=0.4_graph_coarsen=True.pt', help='coarsen data path')
    parser.add_argument('--output_dir', type=pathlib.Path, \
                        default='/mnt/home/thuang/playground/velocity_prediction/GNN_search/Quijote_all', help='hyper-param search path')
    parser.add_argument('--HGNN_flag', action="store_true", help='if true: fit Hierarchical GNN')
    
    parser.add_argument('--hid_dim', type=int, default=64, help='hidden dim')
    parser.add_argument('--n_layer', type=int, default=6, help='number of MP layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--search', action='store_true', help='Enable hyperparameter search')
    group.add_argument('--eval_test', action='store_true', help='Eval on test set only')
    parser.add_argument('--train_all', action='store_true', help='[ONLY USED FOR QUIJOTE] train on the full set with 19k clouds')
    parser.add_argument('--test_sample_idx_end', type=int, default=None, help='if specified, only test on \
                        test_sample test clouds, and save the predictions')

    args = parser.parse_args()
    print(args)
    if args.search:
        best_config, results = hyperparameter_search(args)
        for res in results:
            print(res)
    elif args.eval_test:
        start_time = time.time()
        result_test = eval_pretrained_model(args)
        end_time = time.time()
        print(f"finish eval in {(start_time - end_time):.4f} secs!")
    else:
        save_dir = f"{args.output_dir}/hid={args.hid_dim}_layers={args.n_layer}_lr={args.lr:.4f}"      
        os.makedirs(save_dir, exist_ok=True)
        train_path = f"{args.data_dir}/{args.processed_train_path}"
        val_path =  f"{args.data_dir}/{args.processed_val_path}"
        start_time = time.time()
        R2_best = run_training(train_path, val_path, train_all=args.train_all,
                     hid_dim=args.hid_dim, n_layer=args.n_layer, lr=args.lr, save_dir=save_dir, args=args)
        torch.cuda.synchronize()  # Make sure all training GPU ops are done
        end_time = time.time()
        print(f"R2_val = {R2_best:.4f}, training time = {(end_time - start_time):.4f}")
