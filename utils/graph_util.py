import torch 
import numpy as np
import argparse
import pathlib
import os
import pickle
import scipy.spatial as SS
import random
from torch_geometric.data import Data, DataLoader, InMemoryDataset, Dataset
from torch_geometric.utils import scatter, degree
from torch_geometric.nn.pool import voxel_grid, avg_pool_x, avg_pool

import h5py
import time


seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def coarsen_graph(pos, vel, size=200, period=1000, start=0, inverse=False):
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    if isinstance(vel, np.ndarray):
        vel = torch.from_numpy(vel)
    #use PyG utils on tensors
    cluster = voxel_grid(pos, size=size, start=start, end=period)
    batch = torch.ones(pos.shape[0])
    pos_coarse, _ = avg_pool_x(cluster, pos, batch)
    vel_coarse, _ = avg_pool_x(cluster, vel, batch)
    unique_clusters, cluster_index = torch.unique(cluster, return_inverse=True)
    if (torch.isnan(pos_coarse).any()) or (torch.isnan(vel_coarse).any()):
        print("detecting nan")
    if inverse:
        return pos_coarse.numpy(), vel_coarse.numpy(), cluster, unique_clusters, cluster_index
    else:
        return pos_coarse.numpy(), vel_coarse.numpy()
# idx = 10
# all(pos[cluster_idx == idx].mean(axis=0) == pos_coarse[idx])

def pbc_distance(a, b, L:float, r_link:float, norm=False):
    """
    Computes minimum toroidal distances between vectors a and b under periodic boundary conditions (PBC).

    Args:
        a (np.ndarray): shape (N, D) or (D,) — first point(s)
        b (np.ndarray): shape (N, D), (M, D), or (D,) — second point(s)

    Returns:
        distances (np.ndarray): Euclidean distances with PBC wrapping.
    """
    # Compute displacement with minimum image convention
    diff = a - b
    # Take into account periodic boundary conditions, correcting the distances
    for i, pos_i in enumerate(diff):
        for j, coord in enumerate(pos_i):
            if coord >= r_link:
                diff[i,j] -= L  # Boxsize
            elif -coord >= r_link:
                diff[i,j] += L  # Boxsize
    if norm:
        return np.linalg.norm(diff, axis=-1)
    else: 
        return diff

def build_graph(pos, r_link=90, L=1000, leafsize=16, epsilon=0.00001, multi_edge_feat=True):
    #nearest-neighbor graph with radius r_link
    kd_tree = SS.KDTree(pos, leafsize=leafsize, boxsize=(1+epsilon)*L)
    edge_index = kd_tree.query_pairs(r=r_link, output_type="ndarray")

    # Add reverse pairs
    reversepairs = np.zeros((edge_index.shape[0],2))
    for i, pair in enumerate(edge_index):
        reversepairs[i] = np.array([pair[1], pair[0]])
    edge_index = np.append(edge_index, reversepairs, 0)

    edge_index = edge_index.astype(int)

    # 2. Get edge attributes

    row, col = edge_index[:,0], edge_index[:,1]
    
    #edge feature as wrapped-around distance, 
    d_L = pbc_distance(pos[row], pos[col], L, r_link)
    
    # edge feature as squared Distance
    if multi_edge_feat:
        edge_dist = (pos[row] - pos[col])/r_link #d_L**2 #(E,3) TODO-CHECK
    else:
        edge_dist = np.linalg.norm(d_L, axis=1)/r_link

    return edge_index, edge_dist


def split_dataloader(dataset, batch_size=2000, shuffle=True):
    split_train = int(len(dataset) * 0.8)
    split_valid = int(len(dataset) * 0.9)

    train_dataset = dataset[:split_train]
    valid_dataset = dataset[split_train:split_valid]
    test_dataset = dataset[split_valid:len(dataset)]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=['edge_attr'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, follow_batch=['edge_attr'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, follow_batch=['edge_attr'])

    return train_loader, valid_loader, test_loader


def mapH5_to_PyGData(g, labels=None, 
                     edge_flag=True, mode='BSQ', predict_velocity=True, 
                     r_ratio=0.4, leafsize=16, epsilon=0.00001,
                     coarsen=True):
    '''
    input: g: dataset in the H5 file, storing point clouds 
    return PyG.Data'''
    L_dict = {'BSQ': 1000, 'CAMELS-SAM': 100, 'CAMELS': 25}
    L = L_dict[mode]
    r_link = r_ratio*L

    # Extract input node features
    #Mvir = torch.tensor(g['Mvir'][:], dtype=torch.float).view(-1, 1)
    #concentration = torch.tensor(g['c_klypin'][:], dtype=torch.float).view(-1,1)
    pos = np.stack( [g['X'][:], g['Y'][:], g['Z'][:]], axis=-1).astype(np.float64)

    # Construct labels
    if predict_velocity: #point-wise labels
        y = np.stack( [g['VX'][:], g['VY'][:], g['VZ'][:]], axis=-1).astype(np.float64)
    else:
    # Read the cloud-level labels: cosmological parameters:
        Omega_m = np.array(labels['Omega_m'][:]).reshape(-1,1)
        sigma_8 = np.array(labels['sigma_8'][:]).reshape(-1,1)
        y = np.hstack((Omega_m, sigma_8)).astype(np.float64)
    #build coarsen graphs
    if coarsen: #TODO: save the inverse and use it as additional feats
        pos, y, _, _, cluster_idx = coarsen_graph(pos, y, size=int(L//5), 
                                period=L, start=0,
                                inverse=True)
    else:
        cluster_idx = None
        
    # create edges based on position
    if edge_flag:
        edge_index, edge_dist = build_graph(pos, 
                                            r_link=r_link, 
                                            L=L, leafsize=leafsize, 
                                            epsilon=epsilon, multi_edge_feat=True)
        #print(edge_dist.shape)
    else:
        edge_index, edge_dist = None, None

    # Create graph
    data = Data(x=torch.DoubleTensor(pos), 
                y=torch.DoubleTensor(y), 
                edge_index=torch.tensor(edge_index.T, dtype=torch.long),
                edge_attr=torch.DoubleTensor(edge_dist),
                cluster_idx=cluster_idx)
    return data


def build_PyGdata_fromH5(h5_path, output_dir, edge_flag=True,
                         predict_velocity=True, data_name='BSQ', subset_ids=list(range(3072)),
                         r_ratio=0.4, leafsize=16, epsilon=0.00001, coarsen=True):
    ''' 
    Load the h5 file where groups (i.e. point clouds) contain (halo) point features (e.g., position, mass), 
    if predict_velocity == False:
    - predict the cloud labels (i.e. cosmological parameters)
    otherwise:
    - predict the point velocity
    '''
    data_list = []
    count = 0
    start = time.time()
    with h5py.File(h5_path, 'r') as f:
        group = f[data_name]
        labels = f["params"]
        if subset_ids is not None:
            for key in subset_ids:
                graph_name = f"BSQ_{key}"
                g = group[graph_name]
                data = mapH5_to_PyGData(g, labels, r_ratio=r_ratio, 
                     edge_flag=edge_flag, mode=data_name, predict_velocity=predict_velocity, 
                     leafsize=leafsize, epsilon=epsilon, coarsen=coarsen)
                data_list.append(data)
                count += 1
                # time
                end = time.time()
                duration = end - start
                if count % 100 == 0:
                    print(f"processed {count} number of clouds! used {duration:.4f}")
                    
        else:
            for graph_name in group:
                g = group[graph_name] #of the form '/BSQ/BSQ_0'
                data = mapH5_to_PyGData(g, labels, r_ratio=r_ratio,
                     edge_flag=edge_flag, mode=data_name, predict_velocity=predict_velocity, 
                     leafsize=leafsize, epsilon=epsilon, coarsen=coarsen)
                data_list.append(data)
                count += 1
                # time
                end = time.time()
                duration = end - start
                if count % 100 == 0:
                    print(f"processed {count} number of clouds! used {duration:.4f}")
    output_file = f'{output_dir}/Rc={r_ratio}_graph_coarsen={coarsen}_start={subset_ids[0]}_end={subset_ids[-1]}.pt'
    #output_file = f'{output_dir}/Rc={r_ratio}_graph.pt'
    torch.save(data_list, output_file)
    #return data_list

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', #default='/mnt/home/rstiskalek/ceph/CAMELS-SAM/LH_rockstar_99.hdf5',
                        default='/mnt/home/rstiskalek/ceph/graps4science/Quijote_BSQ_rockstar_10_top5000.hdf5',
                         help='h5 path to load the data')
    parser.add_argument('--data_name', default='BSQ', type=str,
                         help='data group name in the h5 file') #TODO: sync across BSQ and LH? 
    parser.add_argument('--output_dir', #default='/mnt/home/rstiskalek/ceph/CAMELS-SAM/LH_rockstar_99.hdf5',
                        default='/mnt/home/thuang/ceph/playground/datasets/point_clouds',
                         help='save path')
    parser.add_argument('--velocity_flag', action="store_true", help='if true: predict velocity')
    parser.add_argument('--coarsen_flag', action="store_true", help='if true: build coarsen graph')
    parser.add_argument('--r_ratio', default=0.1, type=float, help='linkage r_ratio (over a normalized box length 1.0)')
    parser.add_argument('--start_idx', default=500, type=int,help='starting point cloud idx')
    parser.add_argument('--end_idx', default=1000, type=int,help='ending point cloud idx')

    #the same script applies to position_only and position_velocity point clouds, 
    # as we only use position features to construct the graph

    args = parser.parse_args()
    print(args)
    subset_ids = list(range(args.start_idx, args.end_idx))
    dataset = build_PyGdata_fromH5(args.h5_path, args.output_dir, 
                                   predict_velocity=True, data_name=args.data_name, 
                                   subset_ids=subset_ids,
                                   r_ratio=args.r_ratio, 
                                   coarsen=args.coarsen_flag)