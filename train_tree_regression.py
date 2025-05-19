import numpy as np
import random
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, mask_to_index, index_to_mask
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch import optim
from torch_scatter import scatter_sum, scatter_mean
import pickle
import matplotlib.pyplot as plt
import copy
from utils.tree_util import read_split_indices, split_dataloader, dataset_to_dataloader
from models.tree_param.model_tree import TreeGINConv, TreeRegressor, MLPAgg, DeepSet, train_eval_model, \
    eval_and_plot, plot_train_val_loss, eval_model
from models.cloud_velocity.model_velocity import count_parameters

import argparse
import pathlib
import math
import os
import json
import re
import time 
from sklearn.metrics import r2_score

def bootstrap_r2(y_true, y_pred, n_bootstrap=1000, seed=None):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    r2_values = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        r2 = r2_score(y_true[indices], y_pred[indices])
        r2_values.append(r2)

    r2_values = np.array(r2_values)
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    return mean_r2, std_r2

def plot_data_check(data, save_path):
    feat = scatter_mean(data.x[:,0], data.batch, dim=0)
    y = data.y[:,0]
    plt.scatter(y, feat, s=3)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=pathlib.Path, \
                        default="/mnt/home/thuang/ceph/playground/datasets/merger_trees_1000_feat_1e13/SAM_trees", help='data path')
    parser.add_argument('--trainset_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/ceph/playground/datasets/SAM_trees/SAM_tree_train.pt', help='training data path')
    parser.add_argument('--valset_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/ceph/playground/datasets/SAM_trees/SAM_tree_val.pt', help='validation data path')
    parser.add_argument('--testset_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/ceph/playground/datasets/SAM_trees/SAM_tree_test.pt', help='test data path')
    parser.add_argument('--save_path', type=pathlib.Path, \
                        default='tree_regression_0509', help='model checkpoint dir')

    parser.add_argument('--model_type', type=str, default='MPNN', 
                        choices=['MPNN', 'MLPAgg', "DeepSet"], help='model type')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--target_id', type=int, default=None, help='None: both; 0: omega_m; 1: sigma_8')
    parser.add_argument('--hid_dim', type=int, default=16, help='hidden dim')
    parser.add_argument('--n_layer', type=int, default=5, help='number of MP layers')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')

    # parser.add_argument('--normalize_mode',type=str, default='identity',
    #                     choices=["vmax_threshold","identity"], help='preprocess features: vmax - threshold at 20; mass - normalize by particle weight (roughly ~omega_m), or root particle masss')
    parser.add_argument('--feat_idx', '--list', nargs='+', type=int, help='feature dimension: 0 - mass; 1 - concentration; 2-vmax')
    #parser.add_argument('--log_flag', action="store_true", help='normalize the node mass by taking log')
    parser.add_argument('--train_n_sample',type=int, default=3, help='number of training tree per LH label (to remove spurious correlation with labels); if -1 then use all trees')
    
    parser.add_argument('--subset_mode',type=str, default='full',
                        choices=["full","main_branch","leaves"], help='using full tree, main branch, or leaves only')
    parser.add_argument('--eval_test', action="store_true", help='if true: only eval model')
    parser.add_argument('--eval_model_path',type=pathlib.Path, default=None, help='pretrained model path for eval')

    args = parser.parse_args()
    print(args)
    ### full dataset with n_samples = 25
    trainset = torch.load(args.trainset_path)
    valset = torch.load(args.valset_path)
    testset = torch.load(args.testset_path)
    train_loader, val_loader, test_loader = dataset_to_dataloader(trainset, valset, testset,
                                                                   batch_size=args.batch_size,
                                                         normalize=True, time=True,
                                                         no_mass =True, feat_idx=args.feat_idx)

    batch = next(iter(train_loader))
    print(batch.x[:3])

    node_dim = batch.x.shape[1]
    out_dim = 1 if args.target_id is not None else 2
    
    mlp_only = False
    if args.model_type == 'MPNN':
        model = TreeRegressor(node_dim, args.hid_dim, out_dim, args.n_layer, loop_flag=True)
    elif args.model_type == 'MLPAgg':
        model = MLPAgg(node_dim, args.hid_dim, out_dim)
        mlp_only = True
    elif args.model_type == 'DeepSet':
        model = DeepSet(node_dim, args.hid_dim, out_dim)
        mlp_only = True
    else:
        raise NotImplementedError
    
    params = count_parameters(model)
    print(f"model has {params} parameters!")

    ##eval
    if args.eval_test:
        assert args.eval_model_path is not None, 'must pass in model weight path to run eval'
        model.load_state_dict(torch.load(args.eval_model_path))
        print(args.feat_idx)
        target, pred, test_loss, test_R2_om, test_R2_s8 = eval_model(model, test_loader, mlp_only, target_id=args.target_id)
        target = target.numpy()
        pred = pred.numpy()
        print(target.shape, pred.shape)
        r2_om, r2_om_std = bootstrap_r2(target[:,0], pred[:,0], n_bootstrap=100)
        r2_s8, r2_s8_std = bootstrap_r2(target[:,1], pred[:,1], n_bootstrap=100)

        print(f"test loss={test_loss:.4f}, test_R2_om={test_R2_om:.4f}, test_R2_s8={test_R2_s8:.4f}")
        results_dict = {
            'params': params,
            'test_R2_om': test_R2_om.item(),
            'test_R2_s8': test_R2_s8.item(),
            'save_dir': str(args.eval_model_path),
            'test_R2_om_boot': r2_om,
            'test_R2_om_boot_std': r2_om_std,
            'test_R2_s8_boot': r2_s8,
            'test_R2_s8_boot_std': r2_s8_std,
        }

        test_result_path = os.path.join(args.eval_model_path.parent, "test_R2_result.json")
        with open(test_result_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"Saved test evaluation results to: {test_result_path}")


    else: #TRAIN
        model_name = (f"{args.model_type}_depth_{args.n_layer}_target_{args.target_id}"
                    f"_input={node_dim}_hid={args.hid_dim}"
                    f"_lr={args.lr}_ep={args.num_epochs}"
                    f"_bs={args.batch_size}_n={args.train_n_sample}_feat={str(args.feat_idx)}")
                
        #save_dir = f"trim_{args.save_path}/{model_name}" if args.trim_mass else f"{args.subset_mode}_{args.save_path}/{model_name}"
        save_dir = f"SAM_Trees/{args.subset_mode}/{model_name}" 
        os.makedirs(save_dir, exist_ok=True)
        plot_data_check(next(iter(train_loader)), f"{save_dir}/feat_check.png")
        start_time = time.time()
        train_loss_steps, val_loss_eps = train_eval_model(model, train_loader, val_loader, 
                                                        mlp_only=mlp_only, n_epochs=args.num_epochs,
                                                        lr=args.lr, target_id=args.target_id, 
                                                        save_path=f'{save_dir}/model.pt')
        end_time = time.time()
        print(f"finish training, used {(end_time - start_time):.4f} sec!")
        train_loss, val_loss = eval_and_plot(model, train_loader, val_loader, target_id=args.target_id, 
                mlp_only=mlp_only, model_name=model_name, fig_path=f"{save_dir}/pred.png")
        
        results = {'train_steps': train_loss_steps, 'val': val_loss_eps}
        plot_train_val_loss(results, save_path= f"{save_dir}/results.png")

        ##Testing
        _, _, test_loss, test_R2_om, test_R2_s8 = eval_model(model, test_loader, mlp_only, target_id=args.target_id)
        print(f"test loss={test_loss:.4f}, test_R2_om={test_R2_om:.4f}, test_R2_s8={test_R2_s8:.4f}")
        results['test_loss'] = test_loss 
        pickle.dump(results, open(f"{save_dir}/results.pkl", 'wb'))
        results_dict = {
            'params': params,
            'test_R2_om': test_R2_om.item(),
            'test_R2_s8': test_R2_s8.item(),
            'save_dir': str(args.eval_model_path)
        }

        test_result_path = os.path.join(save_dir, "test_R2_result.json")
        with open(test_result_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"Saved test evaluation results to: {test_result_path}")


