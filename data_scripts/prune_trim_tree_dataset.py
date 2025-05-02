from mpi4py import MPI
import h5py
import numpy as np
import time
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, subgraph
import torch_geometric.transforms as T
import os 
import glob
import random
import pickle
import math
from collections import defaultdict

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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

def read_one_tree_from_lh_group(lh_group, tree_name, y,
                                feat_idx=[0,1,2,4], log_flag=True,
                                pos_idx=[5,6,7], vel_idx=[8,9,10]):
    ''' 
    Load a tree from a group of the H5 file (i.e. group = all trees with root_mass > threshold for a particular LH id)
    feat_idx: 
    - 0: mass
    - 1: concentration
    - 2: vmax
    - 4: scale (time from 1 current to 0 the start of universe)
    - 5-7: x, y, z
    - 8-10: vx, vy, vz
    '''
    tree_group = lh_group[tree_name]
    #read features
    node_name = torch.tensor(tree_group['node_name'][()], dtype=torch.long).unsqueeze(1)
    node_order = torch.tensor(tree_group['node_order'][()], dtype=torch.long).unsqueeze(1)
    edge_index = torch.tensor(tree_group['edge_index'][()], dtype=torch.long).T.contiguous()

    node_feats = torch.tensor(tree_group['node_feats'][()]).float()
    node_feats = node_feats[:, feat_idx + pos_idx + vel_idx ]
    if log_flag: #log mass, concentration, vmax
        node_feats[:, :3] = torch.log10(node_feats[:, :3])
    
    ## return PyG if available
    lh_halo_ids = tree_name.split("_")
    data = Data(
            x=node_feats, #(log mass, log concen, log vmax, ...)
            edge_index=edge_index,
            pos=node_order,  # DFS depth order as node position
            y=y,              # cosmology label
            node_halo_id=node_name,

        )
    main_branch = tree_group['main_branch'][()]
    data.mask_main = main_branch #mask if the node is on the main branch

    data.root_halo_id = int(lh_halo_ids[-1])  # add halo ID attribute -> corresponding to uid in ytree
    data.lh_id = int(lh_halo_ids[1]) # also store LH simulation ID
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

# def gather_samples(save_path, ranks, n_sample=5, seed=42):
#     ''' 
#     Return subset from all chosen per_rank h5 files
#     '''
#     subset = []
#     for lh_id in ranks:
#         start = time.time()
#         data_path = f"{save_path}/full_data_rank_{lh_id}.hdf5"
#         data_samples = random_sample_per_rank_h5(data_path, n_sample, seed=seed)
#         subset.extend(data_samples)
#         end = time.time()
#         print(f"processed lh_{lh_id}, with time={end - start} seconds!")
#     return subset

def distribute_files(all_files):
    return [f for i, f in enumerate(all_files) if i % size == rank]

def main():
    #input files and use control
    seed = 42
    n_sample = 25
    h5_path = "/mnt/home/thuang/ceph/playground/datasets/merger_trees_1000_feat_1e13"
    #id_split_path = "/mnt/home/thuang/ceph/playground/datasets/merger_trees_1000_feat_1e13/SAM_trees/split_indices.txt"
    h5_file_names='full_data_rank_*.hdf5'
    all_files = sorted(glob.glob(os.path.join(h5_path, h5_file_names)))
    print(f"gathered {len(all_files)} rank h5 files")
    local_files = distribute_files(all_files)
    output_file = f"{h5_path}/SAM_trees/prune_trim_dataset_nsample={n_sample}.pt"
    start_time = time.time()

    local_data_list = []
    for data_path in local_files:
        try:
            data_samples = random_sample_per_rank_h5(data_path, n_sample, seed=seed)
            for data in data_samples:
                trim_connected_data = trim_tree(data, connect_trim=True)
                prune_trim_data = prune_linear_nodes(trim_connected_data, use_threshold=True)
                local_data_list.append(prune_trim_data)
        except Exception as e:
            print(f"[Rank {rank}] Failed to process {data_path}: {e}")
            continue

    # Synchronize and report timing
    gathered_data = comm.gather(local_data_list, root=0)
    
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"[Rank {rank}] Finished in {elapsed:.2f} seconds")

    # Let rank 0 report global finish
    if rank == 0:
        full_data_list = [d for sublist in gathered_data for d in sublist]
        torch.save(full_data_list, output_file)
        print(f"All ranks finished pruning and writing: total {len(full_data_list)} trees!")


def combine_saved_data_files(save_dir, output_file, pattern="local_list_{}.pkl", num_ranks=None):
    """
    Combines individual rank-wise saved PyTorch .pt files into a single dataset.
    
    Args:
        save_dir (str): Directory where per-rank files are stored.
        output_file (str): Path to save the combined dataset.
        pattern (str): Filename pattern for rank files. Should include one '{}' for rank ID.
        num_ranks (int): Optional number of ranks to process. If None, will auto-detect.
    """
    combined = []
    rank_files = []

    if num_ranks is not None:
        rank_files = [os.path.join(save_dir, pattern.format(i)) for i in range(num_ranks)]
        print(len(rank_files))
    else:
        # Auto-detect files
        rank_files = sorted([
            os.path.join(save_dir, f) for f in os.listdir(save_dir)
            if f.startswith("local_list_") and f.endswith(".pkl")
        ])

    print(f"Found {len(rank_files)} files to combine.")

    for fpath in rank_files:
        try:
            data_list = pickle.load(open(fpath,"r"))
            assert isinstance(data_list, list), f"File {fpath} did not contain a list"
            combined.extend(data_list)
            print(f"Loaded {len(data_list)} items from {fpath}")
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")

    #torch.save(combined, output_file)
    pickle.dump(combined, open(f"{save_dir}/{output_file}","w"))
    print(f"Saved combined dataset with {len(combined)} items to {output_file}")

if __name__ == "__main__":
    main()
    # combine_saved_data_files(save_dir="/mnt/home/thuang/ceph/playground/datasets/merger_trees_1000_feat_1e13/SAM_trees",
    #                          output_file="prune_trim_dataset_nsample=10.pt",
    #                          num_ranks=384)