from mpi4py import MPI
import h5py
import numpy as np
import os
import ytree 
import time
import argparse
import pathlib

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
            node['vx'].value, node['vy'].value, node['vz'].value], dtype=float)
    return feat_np.view(1,-1) #torch.from_numpy(feat_np).unsqueeze(0).float()

def traverse_tree(halo, height, edge_index=None, node_name=None, node_order=None, node_feat=None):
    ''' 
    recursively travese the tree from root to leaves
    and save the traversed edges to the edge_index
    together with 
    - the DFS traversed order (height) to the node_order
    - associated node features

    '''
    if edge_index is None:
        edge_index = []
        node_name = []
        node_order = []
        node_feat = []
    
    node_ID = halo['Orig_halo_ID']
    node_name.append(node_ID)    
    node_order.append(height)
    node_feat.append(extract_node_feat(halo))
    ancestors = list(halo.ancestors)
    if ancestors is None:
        return #bottom of the tree
    for anc in ancestors:
        edge_index.append((anc['Orig_halo_ID'], node_ID))
        traverse_tree(anc, height+1, edge_index, node_name, node_order, node_feat)
    return node_name, node_order, node_feat, edge_index

def save_tree_data(halo):
    #traverse the tree
    node_name, node_order, node_feats, edges = traverse_tree(halo, 0)
    #export the main branch
    main_branch = list(halo["prog", "Orig_halo_ID"])
    #mask_main_branch = [node in main_branch for node in node_name]
    return node_name, node_order, node_feats, edges, main_branch

def read_subset_LH(LH_path, root_mass_min):
    ''' 
    Given a LH folder path LH_path with a fixed label (sigma_8, omega_m), 
    read into the ytree data 'tree_0_0_0.dat', which contains ~1e5 trees
    extract the subset with root mass greater than root_mass_min
    return: subset of tree_samples (i.e. list of roots), and cosmological param y
    '''
    tree_collection = ytree.load(LH_path)
    y = np.hstack([tree_collection.hubble_constant, tree_collection.omega_matter], dtype=float).view(1,-1)
    subset = []
    for root in tree_collection:
        if root['Mvir']  > root_mass_min:
            subset.append(root)
    return subset, y, tree_collection  #avoid garbage collection / ReferenceError


def build_h5_fromytree(root_mass_min, id_start=0, id_end=1000,
                        all_LH_paths='/mnt/ceph/users/camels/PUBLIC_RELEASE/Rockstar/CAMELS-SAM/LH/',
                        file_name='tree_0_0_0.dat',
                        save_path='datasets/merger_trees/'):
    ''' 
    Loop over each LH_path from all_LH_paths directory
    then apply read_subset_LH(LH_path, kargs**) to extract n_samples trees per LH_path
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Comm size is {size}.", verbose=rank == 0)

    fname_out = f'{save_path}/data_min={int(root_mass_min/1e13)}e13_start={id_start}_end={id_end}.hdf5'

    # Create list of valid LH file paths with IDs
    lh_ids = list(range(id_end))
    lh_info = [
        (lh_id, os.path.join(all_LH_paths, f"LH_{lh_id}/ConsistentTrees/{file_name}"))
        for lh_id in lh_ids
    ]
    lh_info = [(lh_id, path) for lh_id, path in lh_info if os.path.exists(path)]

    # Split work across MPI ranks
    local_info = lh_info[rank::size]
    
    start = time.time()

    with h5py.File(fname_out, 'w', driver='mpio', comm=comm) as f:
        for lh_id, path in local_info:
            try:
                tree_samples, y, tree_collection = read_subset_LH(path, root_mass_min)
                group_name= f"LH_{lh_id}"
                grp = f.create_group(group_name)
                grp.create_dataset('y', data=y)
                for root in tree_samples:
                    #load the ytree data into np arrays
                    node_name, node_order, node_feats, edges, main_branch = save_tree_data(root)
                    #data['y'] = y #add label attribute 
                    #write to h5 subgroups
                    sub_grp = grp.create_group(f"{group_name}_{root['Orig_halo_ID']}")
                    sub_grp.create_dataset('node_name', data=np.array(node_name, dtype='i8'))
                    sub_grp.create_dataset('node_order', data=np.array(node_order, dtype='i8'))
                    sub_grp.create_dataset('node_feats', data=np.concatenate(node_feats, axis=0))
                    sub_grp.create_dataset('edge_index', data=np.array(edges, dtype='i8'))
                    sub_grp.create_dataset('main_branch', data=np.array(main_branch, dtype='i8'))


            except IOError:
                print(f"fail to read LH {lh_id}")
                continue
    comm.Barrier()

    if rank == 0:
        # time
        end = time.time()
        duration = end - start
        print(f"All halos saved to {fname_out}, used {duration:.4f}")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=pathlib.Path, \
                        default='/mnt/ceph/users/camels/PUBLIC_RELEASE/Rockstar/CAMELS-SAM/LH/', help='dataset parent dir')
    parser.add_argument('--file_name', type=str, \
                        default='tree_0_0_0.dat', help='graph dataset file')
    parser.add_argument('--save_path', type=pathlib.Path, \
                        default='datasets/merger_trees/', help='save tree dataset directory')
    parser.add_argument('--mass_min', type=float, default=5e13, help='minimum mass of root halo (node)')
    parser.add_argument('--id_start', type=int, default=0, help='LH folder start id')
    parser.add_argument('--id_end', type=int, default=1000, help='LH folder end id')


    args = parser.parse_args()
    build_h5_fromytree(args.root_mass_min, args.id_start, args.id_end, 
                       args.dataset_path, args.file_name, args.save_path)