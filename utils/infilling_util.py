import numpy as np
import random
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import subgraph, degree, to_networkx
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

def exclude_topk_mask_1d(tensor, k):
    """
    Creates a boolean mask where the top k elements of a 1D tensor are True, and the rest are False.

    Args:
        tensor (torch.Tensor): The input 1D tensor.
        k (int): The number of top elements to mask.

    Returns:
        torch.Tensor: A boolean mask of the same shape as the input tensor.
    """
    if k > tensor.numel():
        raise ValueError("k cannot be larger than the tensor size")
    
    topk_values, topk_indices = torch.topk(tensor, k)
    mask = torch.ones_like(tensor, dtype=torch.bool)
    mask[topk_indices] = False
    return mask

def trim_tree_to_binary(data: Data, feat_idx: int = 0) -> Data:
    edge_index = data.edge_index  # shape [2, num_edges], where child → parent
    num_nodes = data.x.shape[0]
    child_nodes = edge_index[0]
    parent_nodes = edge_index[1]
    in_degree = degree(parent_nodes, num_nodes=num_nodes)  # is parent of someone
    mask_multi = in_degree > 2
    count_multi = mask_multi.sum()
    if count_multi == 0:
        print("it is already a binary tree!")
        return data
    else:
        print(f"pruning to binary tree...with {count_multi} nodes")
        exclude_nodes = []
        for node in torch.arange(num_nodes)[mask_multi]:
            children = child_nodes[parent_nodes == node]
            children_feat = data.x[children,feat_idx]
            exclude_mask = exclude_topk_mask_1d(children_feat, k=2)
            exclude_indices = children[exclude_mask]
            #print(f"processed node = {node}, original children = {children_feat.shape[0]}, exclude = {exclude_indices}")
            #print(children_feat, data.x[exclude_indices, feat_idx])
            exclude_nodes.extend(exclude_indices.tolist())
        
        kept_nodes = list(set(range(num_nodes)).difference(set(exclude_nodes)))
        #print(len(kept_nodes), num_nodes)
        subset_edge_index, subset_edge_attr = subgraph(kept_nodes, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True)
            
        subset_node_features = data.x[kept_nodes]
        
        subset_halo_id = data.node_halo_id[kept_nodes]
        subset_data = Data(x=subset_node_features, 
                        edge_index=subset_edge_index, 
                        edge_attr=subset_edge_attr,
                        y=data.y,
                        lh_id=data.lh_id,
                        mask_main=data.mask_main, 
                        node_halo_id = subset_halo_id)
        transform = T.LargestConnectedComponents() #ensure connectedness
        final_data = transform(subset_data)
        print(f"original size = {num_nodes}, binary size = {final_data.x.shape[0]}")
        return final_data


def subset_up_to_decimals(tensor, subset_value, decimals=2):
  """
  Compares two PyTorch tensors for equality up to a specified number of decimal places.

  Args:
    tensor: target tensor to subset
    subset_value: target values to include
    decimals: The number of decimal places to consider (default: 3).

  Returns:
    True if the tensors are equal up to the specified decimals, False otherwise.
  """
  rounded_tensor = torch.round(tensor * (10**decimals)) / (10**decimals)
  rounded_value = torch.round(subset_value * (10**decimals)) / (10**decimals)
  return torch.isin(rounded_tensor, rounded_value)

def find_parent_node(edge_index, child_node):
    child_nodes = edge_index[0]
    parent_nodes = edge_index[1]
    return parent_nodes[child_nodes == child_node].item() 

def find_merger_nodes(tree):
    child_nodes = tree.edge_index[0]
    parent_nodes = tree.edge_index[1]
    num_nodes = tree.x.shape[0]
    in_degree = degree(parent_nodes, num_nodes=num_nodes)  # is parent of someone
    merger_mask = in_degree == 2
    return set(torch.arange(num_nodes)[merger_mask].tolist())

def coarse_grain_tree(data: Data, subset_times: torch.tensor) -> Data:
    ''' 
    keep even nodes, mask out odd nodes
    if edge attributes are available:
    new edge attributes = sum of old edge attributes (i.e. sum of coarsened path lengths)
    '''
    num_nodes = data.x.shape[0]
    edge_index = data.edge_index
    keep_mask = subset_up_to_decimals(data.x[:,-1], subset_times)
    kept_nodes = set(torch.nonzero(keep_mask, as_tuple=False).flatten().tolist())
    dropped_nodes = set((range(0,num_nodes))).difference(kept_nodes)
    merger_nodes = find_merger_nodes(data)
    dropped_merger_nodes = dropped_nodes.intersection(merger_nodes)
    drop_merger_dict = {data.node_halo_id[find_parent_node(edge_index, node)]: data.node_halo_id[node] for node in dropped_merger_nodes}

    # Optional edge attributes
    has_edge_attr = hasattr(data, 'edge_attr') and data.edge_attr is not None
    if has_edge_attr:
        edge_attr = data.edge_attr.squeeze()
    else:
        edge_attr = None

    # Build child ➝ (parent, weight) or ➝ parent map
    child_to_parent = {}
    for i, (child, parent) in enumerate(edge_index.t().tolist()):
        if has_edge_attr:
            weight = edge_attr[i].item()
            child_to_parent[child] = (parent, weight)
        else:
            child_to_parent[child] = parent

    # Walk from each kept node up to next kept ancestor, summing edge_attr
    new_edges = []
    new_edge_attr = []
    visited_pairs = set()

    for child in kept_nodes:
        current = child
        total_weight = 0.0

        while current in child_to_parent:
            if has_edge_attr:
                parent, weight = child_to_parent[current]
                total_weight += weight
            else:
                parent = child_to_parent[current]
            if parent in kept_nodes:
                edge = (child, parent)
                if edge not in visited_pairs:
                    new_edges.append(edge)
                    if has_edge_attr:
                        new_edge_attr.append(total_weight)
                    visited_pairs.add(edge)
                break  # stop at first kept ancestor

            current = parent  # continue up

    # Remap node indices
    kept_nodes_sorted = sorted(kept_nodes)
    old_to_new = {old: new for new, old in enumerate(kept_nodes_sorted)}

    edge_index_new = torch.tensor([
        [old_to_new[src], old_to_new[dst]] for src, dst in new_edges
    ], dtype=torch.long).t().contiguous()

    x_new = data.x[kept_nodes_sorted]
    subset_halo_id = data.node_halo_id[kept_nodes_sorted]

    # Build new data object
    data_new = Data(
        x=x_new,
        edge_index=edge_index_new,
        edge_attr=None,
        y=data.y,
        lh_id=data.lh_id,
        mask_main=data.mask_main, 
        node_halo_id = subset_halo_id
    )
    if data.edge_attr is not None:
        data_new.edge_attr = torch.tensor(new_edge_attr, dtype=torch.float).unsqueeze(1)  # shape [E, 1]        

    return data_new, drop_merger_dict

def log_norm_standardize(x):
    x_log = torch.log10(x)
    mean = x_log.mean(dim=0)
    std = x_log.std(dim=0)
    return (x_log - mean)/std

def compute_kept_times(largest_tree, k=2):
    time_steps = torch.unique(largest_tree.x[:,-1])
    kept_steps = [i for i in range(len(time_steps)-1,0,-k)]
    kept_times = time_steps[kept_steps]
    return kept_times

def build_infill_tree(data, kept_times, verbose=False):
    #step 1: obtain binary tree
    binary_tree = trim_tree_to_binary(data)
    #step 2: coarsen binary tree by keeping every k steps | TODO: use a universal kept_times for many trees case
    # time_steps = torch.unique(binary_tree.x[:,-1])
    # kept_steps = [i for i in range(len(time_steps)-1,0,-k)]
    # kept_times = time_steps[kept_steps]
    coarsen_tree, dropped_merger_nodes = coarse_grain_tree(binary_tree, kept_times)
    #step 3: binarize the coarsen-binary tree and obtain T
    infill_tree = trim_tree_to_binary(coarsen_tree)
    #step 4: find merger nodes in T
    num_nodes = infill_tree.x.shape[0]
    child_nodes = infill_tree.edge_index[0]
    parent_nodes = infill_tree.edge_index[1]
    merger_nodes = find_merger_nodes(infill_tree)
    num_merger_nodes = len(merger_nodes)
    # standardize data here
    infill_tree.x[:,:-1] = log_norm_standardize(infill_tree.x[:,:-1])
    if verbose:
        print(f"found {num_merger_nodes} merger nodes out of {num_nodes} nodes")
        print(f"features x = {infill_tree.x[0]}")
    #step 5: add virtual nodes and edges
    d = infill_tree.x.shape[1]
    infill_tree.x = torch.cat([infill_tree.x, torch.zeros(num_merger_nodes, d)])
    infill_tree.label = -1 * torch.ones(num_nodes+num_merger_nodes).long() #ignore non-virtual nodes by setting ignore_index = -1
    infill_tree.vn_mask = torch.BoolTensor([False]* (num_nodes+num_merger_nodes))
    for i, merger_node in enumerate(merger_nodes):
        virtual_node = i + num_nodes
        ancestors = child_nodes[parent_nodes == merger_node]
        assert ancestors.shape[0] == 2, "must only have two ancestors by design of binary tree!"
        new_edges = torch.tensor([[merger_node, virtual_node],
                                  [ancestors[0], virtual_node],
                                  [ancestors[1], virtual_node]], dtype=torch.long).T
        #print(new_edges.shape, infill_tree.edge_index.shape)
        infill_tree.edge_index = torch.hstack([infill_tree.edge_index, new_edges])
        if infill_tree.node_halo_id[merger_node] in list(dropped_merger_nodes.keys()): #use node_halo_id as unique identifiers!
            infill_tree.label[num_nodes+i] = 1 
        else:
            infill_tree.label[num_nodes+i] = 0
        infill_tree.vn_mask[num_nodes+i] = True
    assert parent_nodes.shape[0] + 3*num_merger_nodes == infill_tree.edge_index.shape[1], "must add 3V' edges!"
    if verbose:
        print(f"finish constructing the infill tree with {infill_tree.x.shape[0]} nodes, {infill_tree.edge_index.shape[1]} edges!")
    return infill_tree


def label_train_val_test_split(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    torch.manual_seed(seed)

    node_ids = torch.arange(data.x.shape[0])
    subset_nodes = node_ids[data.vn_mask]

    # Extract labels only for the subset
    subset_labels = data.label[subset_nodes]

    # Separate indices for each class
    class0_nodes = subset_nodes[subset_labels == 0]
    class1_nodes = subset_nodes[subset_labels == 1]

    # Shuffle within each class
    class0_perm = class0_nodes[torch.randperm(len(class0_nodes))]
    class1_perm = class1_nodes[torch.randperm(len(class1_nodes))]

    # Helper to split each class separately
    def split_class_nodes(class_nodes):
        n = len(class_nodes)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val  # Remaining for test
        train_idx = class_nodes[:n_train]
        val_idx = class_nodes[n_train:n_train+n_val]
        test_idx = class_nodes[n_train+n_val:]
        return train_idx, val_idx, test_idx

    class0_train, class0_val, class0_test = split_class_nodes(class0_perm)
    class1_train, class1_val, class1_test = split_class_nodes(class1_perm)

    # Combine class 0 and class 1 splits
    data.vn_train_idx = torch.cat([class0_train, class1_train])
    data.vn_val_idx = torch.cat([class0_val, class1_val])
    data.vn_test_idx = torch.cat([class0_test, class1_test])
    train_ratio = len(class1_train) / len(data.vn_train_idx)
    val_ratio = len(class1_val) / len(data.vn_val_idx)
    test_ratio = len(class1_test) / len(data.vn_test_idx)

    # Shuffle final splits to avoid order bias
    data.vn_train_idx = data.vn_train_idx[torch.randperm(len(data.vn_train_idx))]
    data.vn_val_idx = data.vn_val_idx[torch.randperm(len(data.vn_val_idx))]
    data.vn_test_idx = data.vn_test_idx[torch.randperm(len(data.vn_test_idx))]

    return data, np.array([train_ratio, val_ratio, test_ratio])

