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
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=pathlib.Path, \
    #                     default="datasets/merger_trees_1000_feat_1e13/merged_data.hdf5", help='data path')
    parser.add_argument('--trainset_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/ceph/playground/datasets/SAM_trees/infilling_trees_25k_200_train.pt', help='training data path')
    parser.add_argument('--valset_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/ceph/playground/datasets/SAM_trees/infilling_trees_25k_200_val.pt', help='val data path')
    parser.add_argument('--testset_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/ceph/playground/datasets/SAM_trees/infilling_trees_25k_200_test.pt', help='test data path')
    
    parser.add_argument('--merger_ratio_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/playground/datasets/pruned_trimmed_tree_small/infilling_merger_ratios_25k_100.pkl', help='training data stats')
    
    parser.add_argument('--save_path', type=pathlib.Path, \
                        default='infilling_classification_25k_200', help='model checkpoint dir')
    #parser.add_argument('--num_trees', type=int, default=200, help='number of trees')
    #parser.add_argument('--node_dim', type=int, default=4, help='node feat dim')
    parser.add_argument('--feat_idx', '--list', nargs='+', type=int, help='feature dimension: 0 - mass; 1 - concentration; 2-vmax')

    parser.add_argument('--hid_dim', type=int, default=16, help='hidden dim')
    parser.add_argument('--n_layer', type=int, default=4, help='number of MP layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')

    parser.add_argument('--eval_test', action='store_true', help='Eval on test set only')

    args = parser.parse_args()
    print(args)
    
    save_dir = f"{args.save_path}/MPNN_depth_{args.n_layer}_hid={args.hid_dim}_lr={args.lr}_ep={args.num_epochs}_feat={str(args.feat_idx)}" 
    os.makedirs(save_dir, exist_ok=True)
    #Note: data is already normalized!
    train_trees =  subset_data_features(torch.load(args.trainset_path), args.feat_idx)
    val_trees = subset_data_features(torch.load(args.valset_path), args.feat_idx)
    test_trees = subset_data_features(torch.load(args.testset_path), args.feat_idx)

    print(f"taking {len(args.feat_idx)} features")
    
    #check ratios
    all_merger_ratios =  pickle.load(open(args.merger_ratio_path,"rb"))
    mean_ratios, std_ratios = all_merger_ratios.mean(axis=0), all_merger_ratios.std(axis=0)
    val_minus_1std = mean_ratios[1] - std_ratios[1]
    val_plus_1std = mean_ratios[1] + std_ratios[1]
    test_minus_1std = mean_ratios[2] - std_ratios[2]
    test_plus_1std = mean_ratios[2] + std_ratios[2]
    title_ratio = f"val_ratio=[{val_minus_1std:.4f}, {val_plus_1std:.4f}], test_ratio=[{test_minus_1std:.4f}, {test_plus_1std:.4f}]"

    #train/val/test splits over trees

    out_dim = 2 #binary classification
    node_dim = len(args.feat_idx) #input feature dim
    model = TreeNodeClassifier(node_dim, args.hid_dim, out_dim, args.n_layer, loop_flag=True)
    print(f"model has {count_parameters(model)} params")

    if args.eval_test == False:
        start_time = time.time()
        train_loss, val_loss_out, best_val_acc = train_eval_classifier(model, train_trees, val_trees, save_dir, num_epochs=args.num_epochs, lr=args.lr)
        end_time = time.time()
        print(f"finish training, used {(end_time - start_time):.4f} seconds!")
        #save results
        plt.plot(train_loss, label="train", color="tab:blue")
        #plt.plot(val_loss, label="validation", color="tab:orange")
        plt.plot(val_loss_out, label="val_hold_out", color="tab:purple")

        title = f"best_val_acc={best_val_acc:.4f},  \n {title_ratio}"
        plt.legend()
        plt.title(title)
        plt.savefig(f"{save_dir}/results.png", dpi=150)

    else:                                                                                                    
        model.load_state_dict(torch.load(f"{save_dir}/model.pt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = model.to(device)
    test_loss, test_acc, acc_boot, acc_boot_std = eval_classifier(model, test_trees, boot_flag=True)
    print(test_acc.item(), acc_boot, acc_boot_std)
     ## Save to file
    results_dict = {
        #'feat_idx': str(args.feat_idx),
        'test_acc': test_acc.item(),
        'test_acc_boot': acc_boot,
        'test_acc_std': acc_boot_std,
    }

    test_result_path = os.path.join(save_dir, f"test_bootstrap_result.json")
    # Load existing results if file exists
    if os.path.exists(test_result_path):
        with open(test_result_path, 'r') as f:
            existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
    else:
        existing_results = []
    
    # Append new result and save
    existing_results.append(results_dict)
    with open(test_result_path, 'w') as f:
        json.dump(existing_results, f, indent=4)

    print(f"Appended test set evaluation results to: {test_result_path}")
