import numpy as np
import random
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, mask_to_index, index_to_mask
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch import optim
from torch_scatter import scatter_sum, scatter_mean
import pickle
import matplotlib.pyplot as plt
import copy
from tree_util import load_single_h5_trees, load_merged_h5_trees, split_dataloader, split_dataset_corr
from model_tree import TreeGINConv, TreeRegressor, MLPAgg, train_model, eval_model, eval_and_plot
import argparse
import pathlib
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=pathlib.Path, \
                        default="datasets/merger_trees_1000/merged_data.hdf5", help='data path')
    parser.add_argument('--save_path', type=pathlib.Path, \
                        default='tree_regression_model_ckpt', help='model checkpoint dir')
    parser.add_argument('--model_type', type=str, default='MPNN', 
                        choices=['MPNN', 'TreeMP', 'MLPAgg'], help='model type')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--target_id', type=int, default=0, help='0: omega_m; 1: sigma_8')
    parser.add_argument('--hid_dim', type=int, default=16, help='hidden dim')
    parser.add_argument('--n_layer', type=int, default=1, help='number of MP layers')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')

    parser.add_argument('--normalize_mode',type=str, default='particle',
                        choices=["particle","first"], help='normalize by particle weight (roughly ~omega_m), or root particle masss')
    parser.add_argument('--feat_idx', type=list, default=[0], help='feature dimension: 0 - mass; 1 - concentration')
    parser.add_argument('--log_flag', action="store_true", help='normalize the node mass by taking log')
    parser.add_argument('--leaf_threshold',type=float, default=3e9, help='subset nodes based on their mass (to remove spurious correlation with labels)')

    args = parser.parse_args()

    dataset = load_merged_h5_trees(args.data_path, normalize_mode=args.normalize_mode, 
                                   feat_idx=args.feat_idx, log_mass=args.log_flag)
    print(len(dataset))       # Total number of trees
    print(dataset[0].x[:10], dataset[0].y)

    #leaf_threshold = math.log10(args.leaf_threshold) if args.log_flag else args.leaf_threshold
    train_loader, val_loader = split_dataloader(dataset, args.batch_size)
    node_dim = len(args.feat_idx)
    out_dim = 1
    
    mlp_only = False
    if args.model_type == 'MPNN':
    #model = TreeGINConv(node_dim,hid_dim, out_dim)
        model = TreeRegressor(node_dim, args.hid_dim, out_dim, args.n_layer, loop_flag=True, cut=0 )
    elif args.model_type == 'TreeMP':
        model = TreeRegressor(node_dim, args.hid_dim, out_dim, args.n_layer, loop_flag=True, cut=30)
    elif args.model_type == 'MLPAgg':
        model = MLPAgg(node_dim, args.hid_dim, out_dim)
        mlp_only = True
    else:
        raise NotImplementedError

    model_name = f"{args.model_type}_target_{args.target_id}_norm={args.normalize_mode}_log={args.log_flag}_input={node_dim}_hid={args.hid_dim}_lr={args.lr}_ep={args.num_epochs}_bs={args.batch_size}"
    train_model(model, train_loader, target_id=args.target_id, mlp_only=mlp_only,
                n_epochs=args.num_epochs, lr=args.lr, save_path=f"{args.save_path}/{model_name}.pt")
    eval_and_plot(model, train_loader, val_loader, target_id=args.target_id, 
              mlp_only=mlp_only, model_name=model_name, fig_path=f"{args.save_path}/{model_name}.png")
