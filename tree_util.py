import ytree
import numpy as np
import torch 
from torch.nn.functional import one_hot, threshold
try:
    from torch_geometric.data import Data, DataLoader 
    from torch_geometric.utils import subgraph, degree
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

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def find_leaf_nodes(data: Data):
    """ 
    Return the leaf nodes from data (torch_geometric.data.Data): The graph data object.
    """
    edge_index = data.edge_index  # shape [2, num_edges]
    num_nodes = data.num_nodes

    # Nodes with outgoing edges (appear in source/index 0)
    src_nodes = edge_index[0]
    all_nodes = torch.arange(num_nodes)

    # Leaf nodes = nodes not in source list
    leaf_mask = ~torch.isin(all_nodes, src_nodes)
    leaf_nodes = all_nodes[leaf_mask]

    return leaf_nodes

def prune_linear_nodes(data: Data) -> Data:
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
    pos_new = data.pos[kept_nodes]

    # Create new Data object
    data_pruned = Data(x=x_new, edge_index=edge_index_new, edge_attr=edge_attr,
                        num_nodes=len(kept_nodes), pos=pos_new, y=data.y)

    return data_pruned

def get_subset(data, mode='prune', leaf_threshold=math.log10(3e9)):
    """
    Creates a subset tree (subgraph) based on given node feature (mass) threshold

    Args:
        data (torch_geometric.data.Data): The graph data object.
        mode: choices in ['subtree', 'root', 'leaf'] where
        - 'subtree': get a subtree with node mass > leaf_threshold
        - 'root': retain only the root node
        - 'leaf': retain only the leaf node
        - 'prune': prune tree to coarsen linear paths

    Returns:
        torch_geometric.data.Data: The subgraph (remapped nodes and edges)
    """
    if mode == 'prune':
        return prune_linear_nodes(data)

    else:
        if mode == 'subtree':
            node_indices = torch.where(data.x[:,0] > leaf_threshold)[0]
            subset_edge_index, subset_edge_attr = subgraph(node_indices, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True)
            
            subset_node_features = data.x[node_indices]
            
            subset_data = Data(x=subset_node_features, 
                            edge_index=subset_edge_index, 
                            edge_attr=subset_edge_attr,
                            y=data.y)
            subset_leaf_idx = find_leaf_nodes(subset_data)
            subset_data['leaf_idx'] = subset_leaf_idx

        else:
            subset_data = copy.deepcopy(data)
            subset_data['edge_index'] = None
            subset_data['pos'] = None
            if mode == 'root':
                subset_data['x'] = subset_data['x'][0, :].unsqueeze(0)
            elif mode == 'leaf':
                subset_data['x'] = subset_data['x'][-1, :].unsqueeze(0)
            else:
                raise NotImplementedError
    
        return subset_data

def select_tree_per_LH(sample_lh, lh_id, train_n_sample=-1, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if train_n_sample > 0:
        rand_idx = (np.random.permutation(np.where(sample_lh == lh_id)[0]))[:train_n_sample] #first randomly permute, then select only train_n_samples
    else:
        rand_idx = np.where(sample_lh == lh_id)[0]
    return rand_idx

def split_dataloader(dataset, batch_size=128, shuffle=True, train_ratio=0.6, val_ratio=0.2, seed=0, 
                     train_n_sample=1):
    ''' 
    60/20/20 train/val/test split based on disjoint cosmo (note: y cosmo is arranged in random order!)
    '''
    sample_lh = np.array([data.lh_id for data in dataset])
    values, count = np.unique(sample_lh, return_counts=True)
    split_train = int(len(values) * train_ratio)
    split_val = int(len(values) * (train_ratio+val_ratio))
    train_lh_ids = values[:split_train]
    val_lh_ids = values[split_train:split_val]
    test_lh_ids = values[split_val:]
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

    # if subset_flag:
    #     dataset_train = [get_subset(dataset[i], mode, leaf_threshold) for i in idx_train]
    #     dataset_val = [get_subset(dataset[i], mode, leaf_threshold) for i in idx_val] 
    #     dataset_test = [get_subset(dataset[i], mode, leaf_threshold) for i in idx_test] 
    # else:
    dataset_train = [dataset[i] for i in idx_train]
    dataset_val = [dataset[i] for i in idx_val]
    dataset_test =  [dataset[i] for i in idx_test]

    print(f'train_size={len(dataset_train)}, val_size={len(dataset_val)}, test_size={len(dataset_test)}')
    print(f'sampled train data view = {dataset_train[0]}')
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, follow_batch=['x'])
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, follow_batch=['x'])
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, follow_batch=['x'])

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



###Helper function that generate the tree data / PyG format from ytree (outdated, replaced by tree_h5parallel.py)

def compute_concentration(halo):
    return halo["Rvir"] / halo["rs"]

def extract_node_feat(node):
    ''' 
    x, y, z: position, Mpc/h
    vx, vy, vz: velocity, km/s
    Mvir: mass, Msun/h
    concentration: 
    '''
    concentration = compute_concentration(node)
    feat_np = np.hstack([node['Mvir'].value, concentration.value,
            node['x'].value, node['y'].value, node['z'].value, 
            node['vx'].value, node['vy'].value, node['vz'].value])
    return torch.from_numpy(feat_np).unsqueeze(0).float()

def traverse_tree(halo, height, node_id_map=None, edge_index=None, node_order=None, node_feat=None, counter=None):
    """
    Recursively traverse tree and assign unique node IDs using a shared counter (start with counter=[0]).
    """
    if edge_index is None:
        edge_index = []
        node_order = []
        node_feat = []
        node_id_map = {}
        counter = [0]  # mutable counter shared across recursion

    node_key = halo['Orig_halo_ID']
    if node_key in node_id_map:
        return  # already visited
    curr_id = counter[0]
    node_id_map[node_key] = curr_id
    counter[0] += 1

    node_order.append(height)
    node_feat.append(extract_node_feat(halo))

    ancestors = list(halo.ancestors)
    if ancestors is None:
        return

    for anc in ancestors:
        anc_key = anc['Orig_halo_ID']
        traverse_tree(anc, height + 1, node_id_map, edge_index, node_order, node_feat, counter)
        anc_id = node_id_map[anc_key]
        edge_index.append((anc_id, curr_id))  # edge: ancestor → current

    return list(node_id_map.values()), list(node_id_map.keys()), node_order, node_feat, edge_index


def save_tree_data(halo):
    #traverse the tree
    node_id, node_name, node_order, node_feats, edges = traverse_tree(halo, 0)
    #export the main branch
    main_branch = list(halo["prog", "Orig_halo_ID"])
    #mask_main_branch = [node in main_branch for node in node_name]
    return node_id, node_name, node_order, node_feats, edges, main_branch


def construct_PyG_data(node_name, node_order, node_feats, edges, main_branch):
    #traverse the tree
    # node_name, node_order, node_feats, edges = traverse_tree(halo, 0)
    #export the main branch
    mask_main_branch = torch.tensor([node in main_branch for node in node_name], dtype=bool).unsqueeze(1)
    #map node names to indices
    node_to_index = {name: index for index, name in enumerate(node_name)}
    edge_list = [(node_to_index[source], node_to_index[target]) for source, target in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).T.contiguous()
    # Create PyTorch Geometric Data object
    data = Data(x=torch.cat(node_feats, dim=0),
                edge_index=edge_index, pos=torch.LongTensor(node_order).unsqueeze(1),
                mask_main=mask_main_branch)
    return data


def read_subset_LH(LH_path, root_mass_min, root_mass_max, n_samples):
    ''' 
    Given a LH folder path LH_path with a fixed label (sigma_8, omega_m), 
    read into the ytree data 'tree_0_0_0.dat', which contains ~1e5 trees
    extract the subset with root mass ranging from (root_mass_min, root_mass_max)
    then further randomly subset n_samples
    return: n_samples of tree_samples (i.e. list of roots), and cosmological param y
    '''
    tree_collection = ytree.load(LH_path)
    y = torch.tensor([tree_collection.hubble_constant, tree_collection.omega_matter], dtype=float).view(1,-1)
    subset = []
    for root in tree_collection:
        if (root['Mvir']  > root_mass_min) & (root['Mvir'] < root_mass_max):
            subset.append(root)
    if len(subset) > n_samples:
        tree_samples = random.sample(subset, n_samples)
        return tree_samples, y, tree_collection #avoid garbage collection / ReferenceError
    else:
        return subset, y, tree_collection  #avoid garbage collection / ReferenceError


def build_PyGdata_fromytree(root_mass_min, root_mass_max, n_samples, id_start=0, id_end=1000,
                            all_LH_paths='/mnt/ceph/users/camels/PUBLIC_RELEASE/Rockstar/CAMELS-SAM/LH/',
                            file_name='tree_0_0_0.dat',
                            save_path='datasets/merger_trees/'):
    ''' 
    Loop over each LH_path from all_LH_paths directory
    then apply read_subset_LH(LH_path, kargs**) to extract n_samples trees per LH_path
    '''
    data_list = []
    count = 0
    start = time.time()
    for LH_id in range(id_start,id_end):
        print(LH_id)
        path = f'{all_LH_paths}/LH_{LH_id}/ConsistentTrees/{file_name}'
        count += 1

        try:
            tree_samples, y, tree_collection = read_subset_LH(path, root_mass_min, root_mass_max, n_samples)
            for root in tree_samples:
                print(root)
                data = construct_PyG_data(root)
                data['y'] = y #add label attribute 
                data['root_id'] = root['Orig_halo_ID']
                data_list.append(data)

        except IOError:
            print(f"fail to read LH_{LH_id}")
            continue
        
        # time
        end = time.time()
        duration = end - start
        if count % 10 == 0:
            print(f"processed {count} number of trees! used {duration:.4f}")

        pickle.dump(data_list, open(f'{save_path}/data_min={int(root_mass_min/1e13)}e13_max={int(root_mass_max/1e14)}e14_n={n_samples}_start={id_start}_end={id_end}.pkl', 'wb'))

    return data_list


# def load_single_h5_trees(h5_path, max_trees=None, return_dict=False, normalize_first=True, eps=1e-8):
#     """
#     Reads all merger trees from a single HDF5 file returned from build_h5_fromytree_per_rank().

#     Args:
#         h5_path (str): Path to the HDF5 file.
#         max_trees (int, optional): Max number of trees to read (for debugging).
#         return_dict (bool): If True, return a nested dict; otherwise return list of PyG Data objects.

#     Returns:
#         list or dict: List of PyG Data objects or dict with structure {LH_id: [trees]}.
#     """
#     f = h5py.File(h5_path, 'r')
#     data_list = [] if not return_dict else {}

#     count = 0
#     for group_name in f.keys():  # LH_0, LH_1, ...
#         lh_group = f[group_name]
#         y = torch.tensor(lh_group['y'][()], dtype=torch.float32)

#         if return_dict:
#             data_list[group_name] = []

#         for tree_key in lh_group.keys():
#             if tree_key == 'y':
#                 continue
#             tree_group = lh_group[tree_key]

#             #read features
#             main_branch = tree_group['main_branch'][()]
#             node_name = torch.tensor(tree_group['node_name'][()], dtype=torch.long).unsqueeze(1)
#             node_feats = torch.tensor(tree_group['node_feats'][()]).float()
#             if normalize_first: #normalize mass, concentration by the root node
#                 node_feats[:,:2] = node_feats[:,:2] / (node_feats[0,:2]+eps)
#             node_order = torch.tensor(tree_group['node_order'][()], dtype=torch.long).unsqueeze(1)
#             edge_index = torch.tensor(tree_group['edge_index'][()], dtype=torch.long).T.contiguous()

#             data = Data(
#                 x=node_feats,
#                 edge_index=edge_index,
#                 pos=node_order,  # DFS order as node position
#                 mask_main=main_branch, #mask if the node is on the main branch
#                 y=y,              # cosmology label
#                 node_halo_id=node_name,
#             )

#             if return_dict:
#                 data_list[group_name].append(data)
#             else:
#                 data_list.append(data)

#             count += 1
#             if max_trees is not None and count >= max_trees:
#                 return data_list

#     return data_list

def mass_particle(omega_m):
    ''' 
    compute particle mass as a function of omega_m
    '''
    rho_crit = 277.53663
    L = 100e3
    nres = 640
    return L**3 * omega_m * rho_crit / nres**3

def standardize_feature(X, mode='log', std_dim=[1], eps=1e-7):
    ''' 
    Given input tensor (bs, d), noramlize per mode
    - mode == 'log': hstack(X[:,~std_dim], log10(X[:,std_dim]) (log norm vmax, keep concentration)
    - mode == 'standard': normalize to mean 0, std 1 per feature column
    '''
    if mode == 'log':
        return torch.hstack((X[:,:1], torch.log(X[:,1:])+eps))
    else:
        return (X - torch.mean(X, dim=0))/(torch.std(X, dim=0)+eps)


def load_merged_h5_trees(h5_path, PyG_EXIST, prune_flag=False, max_trees=None, feat_idx=[0,1,2,3,4], 
                         normalize_mode='vmax_threshold',
                         log_mass=True, node_feature_mode="cosmo",):
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
    - param log_mass: log10 of Mass/Mpart
    - param node_feature_mode: 
      "cosmo": using cosmological features from feat_idx above
      "random": random Gaussian features (investigate the role of tree topology)
      "constant": constant all-ones features (investigate the role of tree topology)
    '''
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
                    if normalize_mode == 'vmax_threshold': #truncate vmax by thresholding at vmax=20
                        node_feats[:,2] = threshold(node_feats[:,2], threshold=20, value=0) 
                    else:
                        pass
                    if log_mass: #normalize by log
                        node_feats[:,0] = torch.log10(node_feats[:,0])
                    node_feats = node_feats[:, feat_idx]
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
                if PyG_EXIST:
                    data = Data(
                        x=node_feats,
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

                    data.root_halo_id = int(lh_halo_ids[-1])  # add halo ID attribute
                    data.lh_id = int(lh_halo_ids[1]) # also store LH simulation ID
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=pathlib.Path, \
                        default='/mnt/ceph/users/camels/PUBLIC_RELEASE/Rockstar/CAMELS-SAM/LH/', help='dataset parent dir')
    parser.add_argument('--file_name', type=str, \
                        default='tree_0_0_0.dat', help='graph dataset file')
    parser.add_argument('--save_path', type=pathlib.Path, \
                        default='datasets/merger_trees_1000_feat/', help='save pruned tree dataset directory')

    args = parser.parse_args()
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
