import numpy as np
import random
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, mask_to_index, index_to_mask, degree, to_networkx
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.functional as F
from torch import optim
from torch_scatter import scatter_sum, scatter_mean
import pickle
import matplotlib.pyplot as plt
import copy
from models.model_infilling import TreeNodeClassifier, train_eval_classifier, eval_classifier
from utils.tree_util import subset_data_features
import math
import pathlib
import argparse
import os
from models.model_velocity import count_parameters
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=pathlib.Path, \
    #                     default="datasets/merger_trees_1000_feat_1e13/merged_data.hdf5", help='data path')
    parser.add_argument('--dataset_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/playground/datasets/pruned_trimmed_tree_small/infilling_trees_25k_100.pkl', help='training data path')
    parser.add_argument('--merger_ratio_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/playground/datasets/pruned_trimmed_tree_small/infilling_merger_ratios_25k_100.pkl', help='training data stats')
    
    parser.add_argument('--save_path', type=pathlib.Path, \
                        default='infilling_classification_25k_100', help='model checkpoint dir')
    parser.add_argument('--num_trees', type=int, default=100, help='number of trees')
    #parser.add_argument('--node_dim', type=int, default=4, help='node feat dim')
    parser.add_argument('--feat_idx', '--list', nargs='+', type=int, help='feature dimension: 0 - mass; 1 - concentration; 2-vmax')

    parser.add_argument('--hid_dim', type=int, default=16, help='hidden dim')
    parser.add_argument('--n_layer', type=int, default=4, help='number of MP layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')

    args = parser.parse_args()
    print(args)
    
    save_dir = f"{args.save_path}/{args.num_trees}/MPNN_depth_{args.n_layer}_hid={args.hid_dim}_lr={args.lr}_ep={args.num_epochs}_feat={str(args.feat_idx)}" 
    os.makedirs(save_dir, exist_ok=True)
    #Note: data is already normalized!
    dataset = pickle.load(open(args.dataset_path,"rb"))
    dataset = subset_data_features(dataset, args.feat_idx)
    print(f"taking {len(args.feat_idx)} features")
    data_ratios =  pickle.load(open(args.merger_ratio_path,"rb"))
    all_trees = dataset[-args.num_trees:] #ranked from smallest to largest trees, subset the largest ones
    all_merger_ratios = data_ratios[-args.num_trees:]
    mean_ratios, std_ratios = all_merger_ratios.mean(axis=0), all_merger_ratios.std(axis=0)
    val_minus_1std = mean_ratios[1] - std_ratios[1]
    val_plus_1std = mean_ratios[1] + std_ratios[1]
    test_minus_1std = mean_ratios[2] - std_ratios[2]
    test_plus_1std = mean_ratios[2] + std_ratios[2]
    title_ratio = f"val_ratio=[{val_minus_1std:.4f}, {val_plus_1std:.4f}], test_ratio=[{test_minus_1std:.4f}, {test_plus_1std:.4f}]"

    out_dim = 2 #binary classification
    node_dim = len(args.feat_idx) #input feature dim
    model = TreeNodeClassifier(node_dim, args.hid_dim, out_dim, args.n_layer, loop_flag=True)
    print(f"model has {count_parameters(model)} params")
    start_time = time.time()
    train_loss, val_loss, best_val_acc = train_eval_classifier(model, all_trees, save_dir, num_epochs=args.num_epochs, lr=args.lr)
    end_time = time.time()
    print(f"finish training, used {(end_time - start_time):.4f} seconds!")
                                                                                                                  
    model.load_state_dict(torch.load(f"{save_dir}/model.pt"))
    test_loss, test_acc = eval_classifier(model, all_trees, mode='test')

    #save results
    plt.plot(train_loss, label="train", color="tab:blue")
    plt.plot(val_loss, label="validation", color="tab:orange")
    title = f"best_val_acc={best_val_acc:.4f}, test_acc={test_acc:.4f} \n {title_ratio}"
    plt.legend()
    plt.title(title)
    plt.savefig(f"{save_dir}/results.png", dpi=150)
