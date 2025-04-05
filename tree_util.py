import ytree
import numpy as np
import torch 
from torch.nn.functional import one_hot
from torch_geometric.data import Data, DataLoader
import random
import time
import argparse
import pathlib
import pickle
import h5py
from itertools import compress
import copy 

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def subset_root(data):
    data_root = copy.deepcopy(data)
    data_root['x'] = data_root['x'][0, :].unsqueeze(0)
    data_root['edge_index'] = None 
    data_root['pos'] = None 
    return data_root
    
def get_root_only(data_list):
    ''' 
    input: list of Data objects from PyG
    root node is the first node of each Data.x
    '''
    subset_roots = [subset_root(data) for data in data_list]
    return subset_roots

def split_dataloader(dataset, batch_size=128, shuffle=True, train_ratio=0.6, seed=0, root_only=False):
    ''' 
    80/10/10 train/val/test split based on disjoint cosmo
    '''
    random.seed(seed)
    np.random.seed(seed)
    sample_lh = np.array([data.lh_id for data in dataset])
    values, count = np.unique(sample_lh, return_counts=True)
    perm_values = np.random.permutation(values)
    split_train = int(len(perm_values) * train_ratio)
    train_lh_ids = perm_values[:split_train]
    eval_lh_ids = perm_values[split_train:]
    idx_train_in, idx_eval_in = [], []
    for lh_id in train_lh_ids: 
        idx_train_in.extend(np.where(sample_lh == lh_id)[0])
    for lh_id in eval_lh_ids:
        idx_eval_in.extend(np.where(sample_lh == lh_id)[0])

    dataset_train = [dataset[i] for i in idx_train_in]
    dataset_eval = [dataset[i] for i in idx_eval_in]
    if root_only:
        dataset_train = get_root_only(dataset_train)
        dataset_eval = get_root_only(dataset_eval)
    print(f'train_size={len(dataset_train)}, val_size={len(dataset_eval)}')
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, follow_batch=['x'])
    val_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, follow_batch=['x'])

    return train_loader, val_loader

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

# def traverse_tree(halo, height, node_id_map=None, edge_index=None, node_name=None, node_order=None, node_feat=None):
#     ''' 
#     recursively travese the tree from root to leaves (start from root, height=0, id=0)
#     and save the traversed edges to the edge_index
#     together with 
#     - the DFS traversed order (height) to the node_order
#     - associated node features

#     '''
#     if edge_index is None:
#         edge_index = []
#         node_name = []
#         node_order = []
#         node_feat = []
    
#     node_name.append(halo['Orig_halo_ID'])    
#     node_order.append(height)
#     node_feat.append(extract_node_feat(halo))
#     ancestors = list(halo.ancestors)
#     if ancestors is None:
#         return #bottom of the tree
#     for j, anc in enumerate(ancestors):
#         anc_id = id + j
#         node_ids.append(anc_id)
#         edge_index.append((anc_id, id))
#         traverse_tree(anc, height+1, anc_id, edge_index, node_name, node_order, node_feat)
#     return node_ids, node_name, node_order, node_feat, edge_index

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


def load_single_h5_trees(h5_path, max_trees=None, return_dict=False):
    """
    Reads all merger trees from a single HDF5 file returned from build_h5_fromytree_per_rank().

    Args:
        h5_path (str): Path to the HDF5 file.
        max_trees (int, optional): Max number of trees to read (for debugging).
        return_dict (bool): If True, return a nested dict; otherwise return list of PyG Data objects.

    Returns:
        list or dict: List of PyG Data objects or dict with structure {LH_id: [trees]}.
    """
    f = h5py.File(h5_path, 'r')
    data_list = [] if not return_dict else {}

    count = 0
    for group_name in f.keys():  # LH_0, LH_1, ...
        lh_group = f[group_name]
        y = torch.tensor(lh_group['y'][()], dtype=torch.float32)

        if return_dict:
            data_list[group_name] = []

        for tree_key in lh_group.keys():
            if tree_key == 'y':
                continue
            tree_group = lh_group[tree_key]

            #read features
            main_branch = tree_group['main_branch'][()]
            node_name = torch.tensor(tree_group['node_name'][()], dtype=torch.long).unsqueeze(1)
            node_feats = torch.tensor(tree_group['node_feats'][()]).float()
            node_order = torch.tensor(tree_group['node_order'][()], dtype=torch.long).unsqueeze(1)
            edge_index = torch.tensor(tree_group['edge_index'][()], dtype=torch.long).T.contiguous()

            data = Data(
                x=node_feats,
                edge_index=edge_index,
                pos=node_order,  # DFS order as node position
                mask_main=main_branch, #mask if the node is on the main branch
                y=y,              # cosmology label
                node_halo_id=node_name,
            )

            if return_dict:
                data_list[group_name].append(data)
            else:
                data_list.append(data)

            count += 1
            if max_trees is not None and count >= max_trees:
                return data_list

    return data_list

def load_merged_h5_trees(h5_path, max_trees=None):
    ''' 
    Load the merged h5 file created from merge_h5_rank_files
    '''
    data_list = []
    with h5py.File(h5_path, 'r') as f:
        for lh_group_name in f.keys():  # e.g. 'LH_0', 'LH_1', ...
            lh_group = f[lh_group_name]
            y = torch.tensor(lh_group['y'][()], dtype=torch.float32)

            for tree_name in lh_group.keys():
                if tree_name == 'y':
                    continue
                tree_group = lh_group[tree_name]
                #read features
                main_branch = tree_group['main_branch'][()]
                node_name = torch.tensor(tree_group['node_name'][()], dtype=torch.long).unsqueeze(1)
                node_feats = torch.tensor(tree_group['node_feats'][()]).float()
                node_order = torch.tensor(tree_group['node_order'][()], dtype=torch.long).unsqueeze(1)
                edge_index = torch.tensor(tree_group['edge_index'][()], dtype=torch.long).T.contiguous()

                data = Data(
                    x=node_feats,
                    edge_index=edge_index,
                    pos=node_order,  # DFS order as node position
                    mask_main=main_branch, #mask if the node is on the main branch
                    y=y,              # cosmology label
                    node_halo_id=node_name,

                )

                lh_halo_ids = tree_name.split("_")
                data.root_halo_id = int(lh_halo_ids[-1])  # add halo ID attribute
                data.lh_id = int(lh_halo_ids[1]) # also store LH simulation ID

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
                        default='datasets/merger_trees/', help='save tree dataset directory')
    parser.add_argument('--mass_min', type=float, default=5e13, help='minimum mass of root halo (node)')
    parser.add_argument('--mass_max',  type=float, default=1e14, help='minimum mass of root halo (node)')
    parser.add_argument('--n_samples', type=int, default=1, help='random subset of samples')
    parser.add_argument('--id_start', type=int, default=0, help='LH folder start id')
    parser.add_argument('--id_end', type=int, default=1000, help='LH folder end id')


    args = parser.parse_args()
    dataset = build_PyGdata_fromytree(args.mass_min, args.mass_max, args.n_samples, args.id_start, args.id_end,
                                      args.dataset_path, args.file_name)

#bug:   File "/mnt/home/thuang/playground/.venv/lib/python3.10/site-packages/ytree/data_structures/tree_node.py", line 322, in query
#     self.arbor._node_io.get_fields(self, fields=[key],
# ReferenceError: weakly-referenced object no longer exists