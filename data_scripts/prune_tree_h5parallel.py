from mpi4py import MPI
import h5py
import numpy as np
import time
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
import os 
import glob

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
    #print(edge_lengths)
    edge_index_new = torch.tensor(
        [[old_to_new[c], old_to_new[p]] for c, p in new_edges],
        dtype=torch.long
    ).t().contiguous()
    edge_attr = torch.tensor(edge_lengths, dtype=torch.float).contiguous()  # shape [num_edges] -> need to unsqueeze later for faster I/O

    # Remap node features if present
    x_new = data.x[kept_nodes] 
    pos_new = data.pos[kept_nodes]
    halo_id_new = data.node_halo_id[kept_nodes]

    # Create new Data object
    data_pruned = Data(x=x_new, edge_index=edge_index_new, edge_attr=edge_attr,
                        pos=pos_new, 
                        node_halo_id=halo_id_new)

    return data_pruned


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# File paths per rank
input_file = f"datasets/merger_trees_1000_feat_1e13/full_data_rank_{rank}.hdf5"
output_file = f"datasets/prune_merger_trees_1000_feat_1e13/prune_data_rank_{rank}.hdf5"
save_path =  f"datasets/prune_merger_trees_1000_feat_1e13/"
final_filename = "merged_data.hdf5"
h5_file_names='full_data_rank_*.hdf5'

# input_path = "datasets/merger_trees_1000_feat/merged_data.hdf5"
# output_path = "datasets/merger_trees_1000_feat/merged_data_prune.hdf5"
if not os.path.exists(input_file):
    print(f"[Rank {rank}] Input file {input_file} not found.")
    MPI.Finalize()
    exit()


# Distribute the LH_x group IDs across all ranks
start_time = time.time()

# Each rank builds its part of the HDF5 file separately
with h5py.File(input_file, 'r') as f_in, \
     h5py.File(output_file, 'w') as f_out:

    #for lh_id in my_lh_ids:
    for group_name in f_in:
        grp_in = f_in[group_name]
        grp_out = f_out.create_group(group_name)

        # Copy 'y' dataset
        y = torch.tensor(grp_in['y'][()], dtype=torch.float32)
        grp_out.create_dataset('y', data=grp_in['y'][()])
        count = 0

        for tree_name in grp_in.keys(): 
            # if count > 1: #trouble-shoot only
            #     break
            if tree_name == 'y':
                continue
            try:
                subgrp_in = grp_in[tree_name]
                # Load tree data
                node_feats = torch.tensor(subgrp_in['node_feats'][()]).float()
                edge_index = torch.tensor(subgrp_in['edge_index'][()], dtype=torch.long).T.contiguous()
                node_name = torch.tensor(subgrp_in['node_name'][()], dtype=torch.long)
                node_order = torch.tensor(subgrp_in['node_order'][()], dtype=torch.long)

                # Build PyG Data object
                data = Data(x=node_feats, edge_index=edge_index,
                            pos=node_order, y=y, node_halo_id=node_name)

                # Apply pruning
                data_pruned = prune_linear_nodes(data) #drop main_branch, have additional edge_attr representing length of the path pruned

                # Write pruned data
                subgrp_out = grp_out.create_group(tree_name)
                subgrp_out.create_dataset('node_feats', data=data_pruned.x.numpy())
                subgrp_out.create_dataset('edge_index', data=data_pruned.edge_index.t().numpy())
                subgrp_out.create_dataset('edge_attr', data=data_pruned.edge_attr.numpy())
                subgrp_out.create_dataset('node_order', data=data_pruned.pos.numpy())
                subgrp_out.create_dataset('node_name', data=data_pruned.node_halo_id.numpy())

                print(f"[Rank {rank}]  Finished {tree_name} (pruned nodes: {data_pruned.x.shape[0]})")
                count +=1
            except Exception as e:
                print(f"[Rank {rank}] ERROR in {group_name}/{tree_name}: {e}")

# Synchronize and report timing
comm.Barrier()
end_time = time.time()
elapsed = end_time - start_time

print(f"[Rank {rank}] Finished in {elapsed:.2f} seconds")

# Let rank 0 report global finish
comm.Barrier()
if rank == 0:
    print("All ranks finished pruning and writing.")

# output_path = os.path.join(save_path, final_filename)
# rank_files = sorted(glob.glob(os.path.join(save_path, h5_file_names)))

# with h5py.File(output_path, 'w') as f_out:
#     for rank_file in rank_files:
#         with h5py.File(rank_file, 'r') as f_in:
#             for group_name in f_in.keys():
#                 f_in.copy(group_name, f_out)

# print(f"Merged {len(rank_files)} files into {output_path}")
