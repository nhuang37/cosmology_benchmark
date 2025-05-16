import ytree
import numpy as np
import torch 
from torch.nn.functional import one_hot, threshold
try:
    from torch_geometric.data import Data, DataLoader 
    from torch_geometric.utils import subgraph, degree
    import torch_geometric.transforms as T
    PyG_EXISTS = True
except ImportError:
    PyG_EXISTS = False
print(f"Pytorch Geometric is available = {PyG_EXISTS}. Return list of PyG.Data = {PyG_EXISTS}")
import random
import time
import argparse
import pathlib
import pickle
import h5py
from itertools import compress
import copy 
import math
from collections import defaultdict
import pickle
import ast

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True



def trim_tree_subtree_mass_check(data: Data, threshold: float) -> Data:
    edge_index = data.edge_index
    mass = data.x[:, 0]

    # Build parent ➝ [children] map
    parent_to_children = defaultdict(list)
    for child, parent in edge_index.t().tolist():
        parent_to_children[parent].append(child)

    root = 0
    keep_nodes = set()

    def dfs(node):
        """Returns True if this subtree has any node with mass >= threshold"""
        subtree_has_high_mass = mass[node] >= threshold

        for child in parent_to_children.get(node, []):
            child_has_high_mass = dfs(child)
            subtree_has_high_mass |= child_has_high_mass

        if subtree_has_high_mass:
            keep_nodes.add(node)

        return subtree_has_high_mass

    dfs(root)

    # Map old ➝ new indices
    keep_nodes = sorted(list(keep_nodes))
    return keep_nodes

def trim_tree(data, log_mass_node_threshold=math.log10(3e10), connect_trim=True):
    if connect_trim: #Richard's approach
        node_indices = trim_tree_subtree_mass_check(data, log_mass_node_threshold)
    else:
        mass_mask = data.x[:,0] > log_mass_node_threshold
        node_indices = mass_mask.nonzero().flatten()
    subset_edge_index, subset_edge_attr = subgraph(node_indices, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True)
    subset_node_features = data.x[node_indices]
    subset_halo_id = data.node_halo_id[node_indices]
    subset_data = Data(x=subset_node_features, 
                    edge_index=subset_edge_index, 
                    edge_attr=subset_edge_attr,
                    y=data.y,
                    lh_id=data.lh_id,
                    mask_main=data.mask_main, 
                    node_halo_id = subset_halo_id)
    return subset_data

# # Toy test case to validate trim_tree
# #  Edge list: child ➝ parent
# edges = [
#     (1, 0),
#     (2, 0),
#     (3, 1),
#     (4, 1)
# ]
# edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# # Node features: [mass]
# x = torch.tensor([
#     [2.0],  # Node 0
#     [2.0],  # Node 1
#     [5.0],  # Node 2
#     [1.0],  # Node 3
#     [1.0],  # Node 4
# ], dtype=torch.float)

# test_data = Data(x=x, edge_index=edge_index)
# test_data.lh_id = 0
# trim_test_data = trim_tree(test_data, 3.0, connect_trim=True)
# trim_test_data.x

def find_leaf_nodes(data: Data):
    """ 
    Return the leaf nodes from data (torch_geometric.data.Data): The graph data object.
    """
    edge_index = data.edge_index  # shape [2, num_edges]
    num_nodes = data.x.shape[0]

    # Nodes with incomings edges (appear in source/index 0) (ancestor -> target)
    src_nodes = edge_index[1]
    all_nodes = torch.arange(num_nodes)

    # Leaf nodes = nodes not in source list
    leaf_mask = ~torch.isin(all_nodes, src_nodes)
    leaf_nodes = all_nodes[leaf_mask]

    return leaf_nodes

def prune_linear_nodes(data: Data, use_threshold=False, threshold = math.log10(3e10)) -> Data:
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]

    # Step 1: Compute in-degree (number of children) and out-degree (number of parents)
    child_nodes = edge_index[0]
    parent_nodes = edge_index[1]
    in_degree = degree(parent_nodes, num_nodes=num_nodes)  # is parent of someone
    out_degree = degree(child_nodes, num_nodes=num_nodes)  # has a parent

    # Step 2: Identify nodes with exactly one parent and one child
    # These are linear, "pass-through" nodes we want to prune
    linear_nodes_mask = (in_degree == 1) & (out_degree == 1)
    if use_threshold:
        linear_nodes_mask = linear_nodes_mask & (data.x[:,0] < threshold)
    linear_nodes = linear_nodes_mask.nonzero(as_tuple=False).flatten().tolist()

    # Step 3: Build mapping from child -> parent (edge direction)
    child_to_parent = {int(c): int(p) for c, p in edge_index.t().tolist()}

    # Also build reverse mapping: parent -> list of children
    parent_to_children = {}
    for c, p in edge_index.t().tolist():
        parent_to_children.setdefault(p, []).append(c)

    # Step 4: Traverse from each non-linear node and skip over linear nodes
    new_edges = []
    edge_lengths = []  # Store number of nodes skipped (path length)
    visited = set()

    for c, p in edge_index.t().tolist():
        if c in visited or c in linear_nodes:
            continue

        path = [c] #add the start node
 
        # Walk forward while encountering linear nodes -> skip linear nodes while appending them
        current = p
        while current in linear_nodes and current in child_to_parent:
            path.append(current)
            current = child_to_parent[current]

        path.append(current)  #add the end node
        visited.update(path[1:-1])  # intermediate nodes are skipped
        new_edges.append((path[0], path[-1])) #only store two end points of a path to the new edge set
        edge_lengths.append(len(path))

    # Step 5: Determine which nodes to keep (those still used in edges)
    kept_nodes = sorted(set([n for edge in new_edges for n in edge]))
    old_to_new = {old: new for new, old in enumerate(kept_nodes)}

    # Remap edge indices to new node indices
    edge_index_new = torch.tensor(
        [[old_to_new[c], old_to_new[p]] for c, p in new_edges],
        dtype=torch.long
    ).t().contiguous()
    edge_attr = torch.tensor(edge_lengths, dtype=torch.float).unsqueeze(1)  # shape [num_edges, 1]

    # Remap node features if present
    x_new = data.x[kept_nodes] 
    #pos_new = data.pos[kept_nodes]
    subset_halo_id = data.node_halo_id[kept_nodes]

    # Create new Data object
    data_pruned = Data(x=x_new, edge_index=edge_index_new, edge_attr=edge_attr,
                        num_nodes=len(kept_nodes), y=data.y, lh_id=data.lh_id,
                        mask_main=data.mask_main, 
                        node_halo_id = subset_halo_id)

    return data_pruned

def get_subset(data, mode='prune', log_mass_node_threshold=math.log10(3e10)):
    """
    Creates a subset tree (subgraph) based on given node feature (mass) threshold

    Args:
        data (torch_geometric.data.Data): The graph data object.
        mode: choices in ['prune, 'trim', 'leaves', 'main_branch'] where
        - 'prune': prune tree to coarsen linear paths
        - 'leaves': retain only the leaf nodes
        - 'main_branch': retian only the main branch

    Returns:
        torch_geometric.data.Data: The subgraph (remapped nodes and edges)
    """
    if mode == 'prune':
        return prune_linear_nodes(data)
    else:
        if mode == 'leaves':
            node_indices = find_leaf_nodes(data) #disconnected leaf components
    
        elif mode == 'main_branch':
            mask = torch.isin(data.node_halo_id.flatten(), torch.LongTensor(data.mask_main)) #NOTE: a halo id may appear > 1 if the halo splits
            node_indices = torch.nonzero(mask).flatten()
        # node_indices = torch.where(data.x[:,0] > leaf_threshold)[0]
        subset_edge_index, subset_edge_attr = subgraph(node_indices, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True)
        
        subset_node_features = data.x[node_indices]
        
        subset_data = Data(x=subset_node_features, 
                        edge_index=subset_edge_index, 
                        edge_attr=subset_edge_attr,
                        y=data.y,
                        lh_id=data.lh_id)
        return subset_data
   
    

def select_tree_per_LH(sample_lh, lh_id, train_n_sample=-1, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if train_n_sample > 0:
        rand_idx = (np.random.permutation(np.where(sample_lh == lh_id)[0]))[:train_n_sample] #first randomly permute, then select only train_n_samples
    else:
        rand_idx = np.where(sample_lh == lh_id)[0]
    return rand_idx

def subset_data_features(data_list, feat_idx=[0,1,2,3]):
    for data in data_list:
        data.x = data.x[:,feat_idx]
    return data_list


def normalize_data_all(data_list, mean, std, time=False):
    for data in data_list:
        if time:
            if mean.shape[0] > 1:
                data.x[:,:-1] = (data.x[:,:-1] - mean[:-1]) / std[:-1]
            else: 
                pass
        else:
            data.x = (data.x - mean)/std 

    return data_list

def dataset_to_dataloader(dataset_train, dataset_val, dataset_test=None, 
                          batch_size=128, shuffle=True, normalize=True, 
                          time=False, no_mass=False, feat_idx=None):
    ''' 
    given pre-splitted datasets (train/val/test), construct dataloaders with optional preprocessing
    '''
    if feat_idx is not None:
        dataset_train = subset_data_features(dataset_train, feat_idx)
        dataset_val = subset_data_features(dataset_val, feat_idx)
        dataset_test = subset_data_features(dataset_test, feat_idx)

    if normalize:
        print(f"normalizing for mean 0 , std 1 across all trees!")
        all_train_x = torch.cat([data.x for data in dataset_train], dim=0)
        #do not normalize the time feature stored as the last dimension
        mean_x, std_x = all_train_x.mean(dim=0), all_train_x.std(dim=0)
        time_flag = True if 3 in feat_idx else False
        dataset_train = normalize_data_all(dataset_train, mean_x, std_x, time=time_flag)
        dataset_val = normalize_data_all(dataset_val, mean_x, std_x, time=time_flag)
        if dataset_test is not None:
            dataset_test = normalize_data_all(dataset_test, mean_x, std_x, time=time_flag)
    test_size = len(dataset_test) if dataset_test is not None else 0

    print(f'train_size={len(dataset_train)}, val_size={len(dataset_val)}, test_size={test_size}')
    print(f'sampled train data view = {dataset_train[0]}')
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    if dataset_test is not None:
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader, None

def read_split_indices(filename):
    """
    Reads a split_indices.txt file written by `write_split_indices` and returns
    train, val, and test LH ID lists.

    Args:
        filename (str): Path to the split_indices.txt file

    Returns:
        tuple: (train_lhs, val_lhs, test_lhs) – each is a list of LH string IDs
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    train_lhs = ast.literal_eval(lines[1].strip())  # line after 'Train Indices'
    val_lhs   = ast.literal_eval(lines[4].strip())  # line after 'Validation Indices'
    test_lhs  = ast.literal_eval(lines[7].strip())  # line after 'Test Indices'

    return train_lhs, val_lhs, test_lhs

def split_dataloader(dataset, train_lh_ids, val_lh_ids, test_lh_ids,
                     batch_size=128, shuffle=True, 
                     seed=0, 
                     train_n_sample=1, normalize=True, eps=1e-8, 
                     feat_idx=[0,1,2,3], time=False, no_mass=False,
                     save_datasplit=False):
    ''' 
    60/20/20 train/val/test split based on disjoint cosmo (note: y cosmo is arranged in random order!)
    '''
    sample_lh = np.array([data.lh_id for data in dataset])
    # values, count = np.unique(sample_lh, return_counts=True)
    # split_train = int(len(values) * train_ratio)
    # split_val = int(len(values) * (train_ratio+val_ratio))
    # train_lh_ids = values[:split_train]
    # val_lh_ids = values[split_train:split_val]
    # test_lh_ids = values[split_val:]
    idx_train, idx_val, idx_test = [], [], []

    for lh_id in train_lh_ids:
        selected_idx = select_tree_per_LH(sample_lh, lh_id, train_n_sample, seed=seed)
        idx_train.extend(selected_idx)
    for lh_id in val_lh_ids:
        selected_idx = select_tree_per_LH(sample_lh, lh_id, train_n_sample, seed=seed)
        idx_val.extend(selected_idx)
    for lh_id in test_lh_ids:
        selected_idx = select_tree_per_LH(sample_lh, lh_id, train_n_sample, seed=seed)
        idx_test.extend(selected_idx)

    if feat_idx is not None:
        dataset = subset_data_features(dataset, feat_idx)
    dataset_train = [dataset[i] for i in idx_train]
    dataset_val = [dataset[i] for i in idx_val]
    dataset_test =  [dataset[i] for i in idx_test]
    if save_datasplit:
        torch.save(dataset_train, "/mnt/home/rstiskalek/ceph/graps4science/SAM_tree_train.pt")
        torch.save(dataset_val, "/mnt/home/rstiskalek/ceph/graps4science/SAM_tree_val.pt")
        torch.save(dataset_test, "/mnt/home/rstiskalek/ceph/graps4science/SAM_tree_test.pt")

    if normalize:
        print(f"normalizing for mean 0 , std 1 across all trees!")
        all_train_x = torch.cat([data.x for data in dataset_train], dim=0)
        mean_x, std_x = all_train_x[:,:-1].mean(dim=0), all_train_x[:,:-1].std(dim=0)
        dataset_train = normalize_data_all(dataset_train, mean_x, std_x, time, no_mass)
        dataset_val = normalize_data_all(dataset_val, mean_x, std_x, time, no_mass)
        dataset_test = normalize_data_all(dataset_test, mean_x, std_x, time, no_mass)

    print(f'train_size={len(dataset_train)}, val_size={len(dataset_val)}, test_size={len(dataset_test)}')
    print(f'sampled train data view = {dataset_train[0]}')
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def split_dataset_corr(dataset, seed=0, cut=5, train_n_sample=4, eval_n_sample=1):
    ''' 
    Take in the full dataset, return training set, evaluation set (in-dist), evaluation set (out-dist)
    where 
    - eval (out) consists of unseen trees with unseen cosmological parameters from training set [but these have smaller number of large roots]
    - eval (in) consists of unseen trees with training cosmological parameters  [correlation]
    :param: cut - number of trees per training cosmological parameters
    '''
    random.seed(seed)
    np.random.seed(seed)
    sample_lh = np.array([data.lh_id for data in dataset])
    values, count = np.unique(sample_lh, return_counts=True)
    mask = count > cut
    print(f"number of unique LH parameters = {len(values)}, trainin LH parameter = {mask.sum()}")
    #in distribution set vs out distribution split
    selected_unique_values = values[mask]
    subset_mask = np.isin(sample_lh, selected_unique_values)
    subset_indist = sample_lh[subset_mask]
    dataset_eval_out = list(compress(dataset, ~subset_mask))
    #further split indist
    idx_train_in, idx_eval_in = [], []
    for lh_id in values: 
        lh_id_in = np.where(subset_indist == lh_id)[0]
        lh_id_in = np.random.permutation(lh_id_in)
        idx_train_in.extend(lh_id_in[:train_n_sample])
        idx_eval_in.extend(lh_id_in[train_n_sample:(train_n_sample+eval_n_sample)])
    #assert len(dataset_eval_out) + len(idx_train_in) + len(idx_eval_in) == len(dataset), 'must sum up!'
    dataset_train = [dataset[i] for i in idx_train_in]
    dataset_eval_in = [dataset[i] for i in idx_eval_in]
    print(f"trainset size = {len(dataset_train)}, eval_in size = {len(dataset_eval_in)}, eval_out size = {len(dataset_eval_out)}")
    return dataset_train, dataset_eval_in, dataset_eval_out


def mass_particle(omega_m, log=False):
    ''' 
    compute particle mass as a function of omega_m
    '''
    rho_crit = 277.53663
    L = 100e3
    nres = 640
    out = L**3 * omega_m * rho_crit / nres**3
    if log:
        return math.log10(out)
    else:
        return out


def load_merged_h5_trees(h5_path, prune_flag=False, max_trees=None, feat_idx=[0,1,2], 
                         normalize_mode='vmax_threshold',
                         log_flag=True, node_feature_mode="cosmo",
                         subset_mode="full"):
    ''' 
    Load the merged h5 file created from merge_h5_rank_files
    Return:
    if PyG_EXIST, return a list of PyG.Data objects
    otherwise return a list of dictionaries, each dict has keys (node_feat, edge_index, edge_feat, node_order, node_halo_id )
        for original trees: key main_branch with values indicating the main branch node IDs
        for pruned trees: key edge_attr with values indicating the length of the pruned path (integral values)
    - param prune_flag: 
      if True: process the pruned trees; otherwise process the original trees
    - param max_trees: subset number of trees
    - param feat_idx: subset number of features, columns corresopnd to
      0 - Mass/Mpart
      1 - concentration
      2 - vmax
      3 - spin
      4 - scale (time), ranging from [0,1] where 1 represents current halo
      5-7 - x, y, z position
      8-10 - vx, vy, vz velocity
      11-13 - Jx, Jy, Jz actions
    - param normalize_mode: normalize features
    - param log_flag: log10 of features
    - param node_feature_mode: 
      "cosmo": using cosmological features from feat_idx above
      "random": random Gaussian features (investigate the role of tree topology)
      "constant": constant all-ones features (investigate the role of tree topology)
    - param subset_mode:
      "full": use the whole tree
      "main_branch": use the main branch (path) only
      "leaves": use the leaves only
    '''
    if subset_mode != "full":
        assert PyG_EXISTS, "must have pytorch geometric installed to subset trees!"
    data_list = []
    with h5py.File(h5_path, 'r') as f:
        for lh_group_name in f.keys():  # e.g. 'LH_0', 'LH_1', ...
            lh_group = f[lh_group_name]
            y = torch.tensor(lh_group['y'][()], dtype=torch.float32)
            if len(y.shape) == 1:
                y = y.unsqueeze(0)
            for tree_name in lh_group.keys():
                if tree_name == 'y':
                    continue
                tree_group = lh_group[tree_name]
                #read features
                node_name = torch.tensor(tree_group['node_name'][()], dtype=torch.long).unsqueeze(1)
                node_order = torch.tensor(tree_group['node_order'][()], dtype=torch.long).unsqueeze(1)
                edge_index = torch.tensor(tree_group['edge_index'][()], dtype=torch.long).T.contiguous()
                
                if node_feature_mode == 'cosmo':
                    node_feats = torch.tensor(tree_group['node_feats'][()]).float()
                    # if normalize_mode == 'vmax_threshold': #truncate vmax by thresholding at vmax=20
                    #     node_feats[:,2] = threshold(node_feats[:,2], threshold=20, value=0) 
                    # else:
                    #     pass
                    # if log_flag: #normalize by log -> apply it to all features [0-4]
                    #     node_feats[:,0] = torch.log10(node_feats[:,0])
                    node_feats = node_feats[:, feat_idx]
                    if log_flag:
                        node_feats = torch.log10(node_feats)
                    #node_feats = standardize_feature(node_feats[:, feat_idx], mode='std') #subset feature dimension (mass, concen, x, y, z, vx, vy, vz)
                    #node_feats = standardize_feature(node_feats, mode='log', std_dim=[1]) #TODO: all normalizations are worse than unnormalizaed...
                elif node_feature_mode == 'random':
                    node_feats = torch.rand(node_order.shape[0], 1)
                elif node_feature_mode == 'constant':
                    node_feats = torch.ones(node_order.shape[0], 1)
                else:
                    raise NotImplementedError
                
                ## return PyG if available
                lh_halo_ids = tree_name.split("_")
                if PyG_EXISTS:
                    data = Data(
                        x=node_feats, #(log mass, log concen, log vmax, ...)
                        edge_index=edge_index,
                        pos=node_order,  # DFS depth order as node position
                        y=y,              # cosmology label
                        node_halo_id=node_name,

                    )
                    if prune_flag:
                        edge_attr = torch.tensor(tree_group['edge_attr'], dtype=torch.float)
                        data.edge_attr = edge_attr.unsqueeze(1)
                    else:
                        main_branch = tree_group['main_branch'][()]
                        data.mask_main = main_branch #mask if the node is on the main branch

                    data.root_halo_id = int(lh_halo_ids[-1])  # add halo ID attribute -> corresponding to uid in ytree
                    data.lh_id = int(lh_halo_ids[1]) # also store LH simulation ID
                    #subsetting
                    if subset_mode != "full":
                        data = get_subset(data, mode=subset_mode)

                else:
                    data = {"x": node_feats,
                            "edge_index": edge_index,
                            "pos": node_order,
                            "y": y,
                            "node_halo_id": node_name}
                    if prune_flag:
                        data["edge_attr"] = torch.tensor(tree_group['edge_attr'], dtype=torch.float).unsqueeze(1)
                    else:
                        data["mask_main"] = tree_group['main_branch'][()]
                    data["root_halo_id"] = int(lh_halo_ids[-1]) 
                    data["lh_id"] = int(lh_halo_ids[1]) 
                
                data_list.append(data)
                if max_trees is not None and len(data_list) >= max_trees:
                    return data_list
    return data_list

def read_one_tree_from_lh_group(lh_group, tree_name, y, node_feature_mode='cosmo',
                                feat_idx=[0,1,2,4], log_flag=True, prune_flag=False,
                                subset_mode='full'):
    ''' 
    Load a tree from a group of the H5 file (i.e. group = all trees with root_mass > threshold for a particular LH id)
    feat_idx: 
    - 0: mass
    - 1: concentration
    - 2: vmax
    - 4: scale (time from 1 current to 0 the start of universe)
    '''
    tree_group = lh_group[tree_name]
    #read features
    node_name = torch.tensor(tree_group['node_name'][()], dtype=torch.long).unsqueeze(1)
    node_order = torch.tensor(tree_group['node_order'][()], dtype=torch.long).unsqueeze(1)
    edge_index = torch.tensor(tree_group['edge_index'][()], dtype=torch.long).T.contiguous()
    
    if node_feature_mode == 'cosmo':
        node_feats = torch.tensor(tree_group['node_feats'][()]).float()
        node_feats = node_feats[:, feat_idx]
        if log_flag:
            node_feats[:, :-1] = torch.log10(node_feats[:, :-1])
    elif node_feature_mode == 'random':
        node_feats = torch.rand(node_order.shape[0], 1)
    elif node_feature_mode == 'constant':
        node_feats = torch.ones(node_order.shape[0], 1)
    else:
        raise NotImplementedError
    
    ## return PyG if available
    lh_halo_ids = tree_name.split("_")
    data = Data(
            x=node_feats, #(log mass, log concen, log vmax, ...)
            edge_index=edge_index,
            pos=node_order,  # DFS depth order as node position
            y=y,              # cosmology label
            node_halo_id=node_name,

        )
    if prune_flag:
        edge_attr = torch.tensor(tree_group['edge_attr'], dtype=torch.float)
        data.edge_attr = edge_attr.unsqueeze(1)
    else:
        main_branch = tree_group['main_branch'][()]
        data.mask_main = main_branch #mask if the node is on the main branch

    data.root_halo_id = int(lh_halo_ids[-1])  # add halo ID attribute -> corresponding to uid in ytree
    data.lh_id = int(lh_halo_ids[1]) # also store LH simulation ID
    #subsetting
    if subset_mode != "full":
        data = get_subset(data, mode=subset_mode)
    return data


def random_sample_per_rank_h5(data_path, n_sample=5, seed=42):
    ''' 
    Return a subset of n_sample trees from a per_rank H5 file (containing multiple LH ids)
    '''
    random.seed(seed)
    np.random.seed(seed)
    data_samples = []
    #1. get the total number of trees per lh
    with h5py.File(data_path, 'r') as f:
        for lh_group_name in f.keys():  # e.g. 'LH_0', 'LH_1', ...
            lh_group = f[lh_group_name]
            #keys = [key for key in list(lh_group.keys()) if (key!= 'y') & (lh_group[key]['node_order'].shape[0] < 2e5) ]
            keys = [key for key in list(lh_group.keys()) if key!= 'y' ]
            num_trees = len(keys) - 1 #keys contain all trees and y (always comes last), thus minus 1
            #2. draw a random n_sample of indexes
            rand_idx = np.random.permutation(np.arange(num_trees))[:n_sample]
            selected_keys = [keys[i] for i in rand_idx]
            #3. append he random subset of trees
            y = torch.tensor(lh_group['y'][()], dtype=torch.float32)
            if len(y.shape) == 1:
                y = y.unsqueeze(0)
            for key in selected_keys:
                data = read_one_tree_from_lh_group(lh_group, key, y)
                data_samples.append(data)
    return data_samples

def gather_samples(save_path, ranks, n_sample=5, seed=42):
    ''' 
    Return subset from all chosen per_rank h5 files
    '''
    subset = []
    for lh_id in ranks:
        start = time.time()
        data_path = f"{save_path}/full_data_rank_{lh_id}.hdf5"
        data_samples = random_sample_per_rank_h5(data_path, n_sample, seed=seed)
        subset.extend(data_samples)
        end = time.time()
        print(f"processed lh_{lh_id}, with time={end - start} seconds!")
    return subset

def dataset_split_from_per_rank_h5(save_path="/mnt/home/thuang/ceph/playground/datasets/merger_trees_1000_feat_1e13", 
                                   num_ranks=384, n_sample=5, train_ratio=0.6, val_ratio=0.2, produce_test_only=False):
    ''' 
    Split on the file level as per-rank files contain disjoint sets of LH ids
    Return training/validation/test set of trees, where each LH id has (at most) n_sample of trees
    '''
    np.random.seed(42)
    all_ranks = np.arange(num_ranks)
    values = np.random.permutation(all_ranks)
    split_train = int(len(values) * train_ratio)
    split_val = int(len(values) * (train_ratio+val_ratio))
    train_ranks = values[:split_train]
    val_ranks = values[split_train:split_val]
    test_ranks = values[split_val:]
    print(train_ranks[:10], val_ranks[:10], test_ranks[:10])
    if produce_test_only:
        testset = gather_samples(save_path, test_ranks, n_sample)
        return testset
    else:
        trainset = gather_samples(save_path, train_ranks, n_sample)
        valset =  gather_samples(save_path, val_ranks, n_sample)
        testset = gather_samples(save_path, test_ranks, n_sample)
        return trainset, valset, testset
    

def prune_trim_dataset(dataset, save_flag=True, mode='train'):
    trim_dataset = [trim_tree(data, connect_trim=True) for data in dataset]
    if save_flag:
        lh_num = 600 if mode == 'train' else 200
        pickle.dump(trim_dataset,  open(f"{args.local_data_path}/trimmed_{mode}set_n={args.n_sample}_lh={lh_num}.pkl", 'wb') )
    prune_trim_dataset = [prune_linear_nodes(trim_connected_data, use_threshold=True) for trim_connected_data in trim_dataset]
    return prune_trim_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=pathlib.Path, \
                        default='/mnt/ceph/users/camels/PUBLIC_RELEASE/Rockstar/CAMELS-SAM/LH/', help='dataset parent dir')
    parser.add_argument('--file_name', type=str, \
                        default='tree_0_0_0.dat', help='graph dataset file')
    parser.add_argument('--h5_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/ceph/playground/datasets/merger_trees_1000_feat_1e13', help='full tree dataset directory') #""
    parser.add_argument('--local_data_path', type=pathlib.Path, \
                        default='trim_tree_regression/feature_with_time', help='save pruned trimmed tree dataset directory') #""
    parser.add_argument('--prune_flag', action="store_true", help='prune trees')
    parser.add_argument('--pre_split', action="store_true", help='splitting files')
    parser.add_argument('--n_sample', type=int, default=3, help='number of trees per LH')
    
    args = parser.parse_args()
    if args.prune_flag:
        dataset = load_merged_h5_trees("datasets/merger_trees_1000_feat/merged_data.hdf5", 
                                normalize_mode="mass_particle", feat_idx=[0,1,2,3], #(mass, c, vmax, spin)
                                log_mass=True)
        truncate_mask = pickle.load(open(f"datasets/merger_trees_1000_feat/subset_Npart_575.pkl","rb"))
        indices = np.where(truncate_mask)[0].tolist()
        dataset_truncate = [data for i, data in enumerate(dataset) if i in indices]
        start = time.time()
        dataset_prune_truncate = [prune_linear_nodes(data) for data in dataset_truncate]
        end = time.time()
        print(f"finish pruning after {end-start} seconds on {len(dataset_prune_truncate)} trees!")
        pickle.dump(dataset_prune_truncate, open(f"datasets/merger_trees_1000_feat/truncate_prune.pkl","wb"))

    else: #create samples from full Camel-Sam collection of trees with node mass > 1e13
        print("loading pre-split train/val/test files")
        if args.pre_split:
            trainset = pickle.load(open(f"{args.local_data_path}/trainset_n={args.n_sample}_lh=600.pkl", 'rb'))
            valset = pickle.load(open(f"{args.local_data_path}/valset_n={args.n_sample}_lh=200.pkl", 'rb'))
            testset = pickle.load(open(f"{args.local_data_path}/testset_n={args.n_sample}_lh=200.pkl", 'rb'))

        else:
            print("creating new split datasets!")
            trainset, valset, testset = dataset_split_from_per_rank_h5(args.h5_path, num_ranks=384, n_sample=args.n_sample)
            pickle.dump(trainset, open(f"{args.local_data_path}/trainset_n={args.n_sample}_lh=600.pkl", 'wb') )
            pickle.dump(valset, open(f"{args.local_data_path}/valset_n={args.n_sample}_lh=200.pkl", 'wb') )
            pickle.dump(testset, open(f"{args.local_data_path}/testset_n={args.n_sample}_lh=200.pkl", 'wb') )

        prune_trim_trainset = prune_trim_dataset(trainset, mode='train')
        pickle.dump(prune_trim_trainset,  open(f"{args.local_data_path}/pruned_trimmed_trainset_n={args.n_sample}_lh=600.pkl", 'wb') )

        prune_trim_valset = prune_trim_dataset(valset, mode='val')
        pickle.dump(prune_trim_valset,  open(f"{args.local_data_path}/pruned_trimmed_valset_n={args.n_sample}_lh=200.pkl", 'wb') )

        prune_trim_testset = prune_trim_dataset(testset, mode='test')
        pickle.dump(prune_trim_testset,  open(f"{args.local_data_path}/pruned_trimmed_testset_n={args.n_sample}_lh=200.pkl", 'wb') )




    
