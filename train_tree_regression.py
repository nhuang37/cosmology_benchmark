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
from tree_util import load_merged_h5_trees, split_dataloader, split_dataset_corr, cut_tree
from model_tree import TreeGINConv, TreeRegressor, MLPAgg, DeepSet, train_eval_model, eval_and_plot, plot_train_val_loss
import argparse
import pathlib
import math
import os

def plot_data_check(data, save_path):
    feat = scatter_mean(data.x[:,0], data.x_batch, dim=0)
    y = data.y[:,0]
    plt.scatter(y, feat, s=3)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=pathlib.Path, \
                        default="datasets/merger_trees_1000_feat_npart_cut5/merged_data.hdf5", help='data path')
    parser.add_argument('--save_path', type=pathlib.Path, \
                        default='tree_regression', help='model checkpoint dir')
    parser.add_argument('--model_type', type=str, default='MPNN', 
                        choices=['MPNN', 'TreeMP', 'MLPAgg', "DeepSet"], help='model type')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--target_id', type=int, default=0, help='0: omega_m; 1: sigma_8')
    parser.add_argument('--hid_dim', type=int, default=16, help='hidden dim')
    parser.add_argument('--n_layer', type=int, default=1, help='number of MP layers')

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')

    parser.add_argument('--normalize_mode',type=str, default='identity',
                        choices=["vmax_threshold","identity"], help='preprocess features: vmax - threshold at 20; mass - normalize by particle weight (roughly ~omega_m), or root particle masss')
    parser.add_argument('--feat_idx', '--list', nargs='+', type=int, help='feature dimension: 0 - mass; 1 - concentration; 2-vmax')
    parser.add_argument('--log_flag', action="store_true", help='normalize the node mass by taking log')
    parser.add_argument('--train_n_sample',type=int, default=5, help='number of training tree per LH label (to remove spurious correlation with labels); if -1 then use all trees')
    parser.add_argument('--node_feature_mode',type=str, default='cosmo',
                        choices=["cosmo","random","constant"], help='using cosmological features or uninformative node feature')
    parser.add_argument('--prune_flag', action="store_true", help='if true: use the pruned dataset')
    parser.add_argument('--trim_mass', action="store_false", help='default: use the trimmed dataset, where each tree is trimmed away of nodes with mass below threshold')
    parser.add_argument('--subset_mode',type=str, default='full',
                        choices=["full","main_branch","leaves"], help='using full tree, main branch, or leaves only')
    parser.add_argument('--log_mass_threshold',type=float, default=math.log10(3e10), help='subset nodes based on their mass (to remove spurious correlation with labels)')

    args = parser.parse_args()
    print(args)

    if args.prune_flag:
        dataset = load_merged_h5_trees("datasets/prune_merger_trees_1000_feat_npart_cut5/merged_data.hdf5",
                                       prune_flag=True,
                                       normalize_mode=args.normalize_mode, feat_idx=args.feat_idx,
                                       log_flag=True,
                                       node_feature_mode=args.node_feature_mode,
                                       subset_mode=args.subset_mode
                                       )
    else:
        dataset = load_merged_h5_trees(args.data_path, normalize_mode=args.normalize_mode, 
                                   feat_idx=args.feat_idx, log_flag=True,
                                   node_feature_mode=args.node_feature_mode,
                                   subset_mode=args.subset_mode)
    
    if args.trim_mass:
        print(f"cutting trees...")
        dataset = cut_tree(dataset, log_mass_node_threshold=args.log_mass_threshold)
        
    print(len(dataset))       # Total number of trees
    print(dataset[0].x.shape)

    #leaf_threshold = math.log10(args.leaf_threshold) if args.log_flag else args.leaf_threshold
    train_loader, val_loader, test_loader = split_dataloader(dataset, args.batch_size, 
                                                             train_n_sample=args.train_n_sample,
                                                             normalize=True)
    batch = next(iter(train_loader))
    print(batch.x[:10])

    if args.feat_idx is None: #random feat/ const. feat ->TODO: higher-dim random feat?
        node_dim = 1
    else:
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
    elif args.model_type == 'DeepSet':
        model = DeepSet(node_dim, args.hid_dim, out_dim)
        mlp_only = True
    else:
        raise NotImplementedError

    #model_name = f"{args.model_type}_target_{args.target_id}_norm={args.normalize_mode}_log={args.log_flag}_input={node_dim}_hid={args.hid_dim}_lr={args.lr}_ep={args.num_epochs}_bs={args.batch_size}_n={args.train_n_sample}_feat={args.node_feat_mode}"

    model_name = (f"{args.model_type}_depth_{args.n_layer}_target_{args.target_id}"
                f"_norm={args.normalize_mode}_log={args.log_flag}"
                f"_input={node_dim}_hid={args.hid_dim}"
                f"_lr={args.lr}_ep={args.num_epochs}"
                f"_bs={args.batch_size}_n={args.train_n_sample}"
                f"_feat={args.node_feature_mode}_masscut={args.log_mass_threshold}")
               
    save_dir = f"trim_{args.save_path}/{model_name}" if args.trim_mass else f"{args.subset_mode}_{args.save_path}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    plot_data_check(next(iter(train_loader)), f"{save_dir}/feat_check.png")


    train_loss_steps, val_loss_eps = train_eval_model(model, train_loader, val_loader, 
                                                      mlp_only=mlp_only, n_epochs=args.num_epochs,
                                                      lr=args.lr, target_id=args.target_id, 
                                                      save_path=f'{save_dir}/model.pt')

    train_loss, val_loss = eval_and_plot(model, train_loader, val_loader, target_id=args.target_id, 
              mlp_only=mlp_only, model_name=model_name, fig_path=f"{save_dir}/pred.png")
    
    results = {'train_steps': train_loss_steps, 'val': val_loss_eps}
    plot_train_val_loss(results, save_path= f"{save_dir}/results.png")
    pickle.dump(results, open(f"{save_dir}/results.pkl", 'wb'))
