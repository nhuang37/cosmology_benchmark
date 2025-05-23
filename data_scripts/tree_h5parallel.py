from mpi4py import MPI
import h5py
import numpy as np
import os
import ytree 
import time
import argparse
import pathlib
import glob
import math

def mass_particle(omega_m):
    ''' 
    compute particle mass as a function of omega_m
    '''
    rho_crit = 277.53663
    L = 100e3
    nres = 640
    return L**3 * omega_m * rho_crit / nres**3

def compute_log_mass_by_mpart(mass, omega_m, eps=1e-8):
    '''
    return log(mass / mpart), where mpart=mass_particle(omega_m)
    '''
    mpart = mass_particle(omega_m)
    return math.log10(mass/ (mpart + eps))

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
                         node['vmax'].value, node['Spin'], node['scale'], #note Spin and scale are dimensionless quantity!
            node['x'].value, node['y'].value, node['z'].value, 
            node['vx'].value, node['vy'].value, node['vz'].value,
            node['Jx'].value, node['Jy'].value, node['Jz'].value], dtype=np.float32)
    return feat_np.reshape(1,-1) #torch.from_numpy(feat_np).unsqueeze(0).float()


def traverse_tree(halo, height, parent_id=None,counter=None,
                  edge_index=None, node_order=None, node_feat=None, node_names=None):
    """
    Recursively traverse tree and assign unique node IDs using a shared counter (start with counter=[0]).
    """
    if edge_index is None:
        edge_index = []
        node_order = []
        node_feat = []
        node_names = []
        counter = [0]  # mutable counter shared across recursion (list mutable, integer is not) / increasing order

    curr_id = counter[0]
    counter[0] += 1
    node_names.append(halo['uid'])
    node_order.append(height)
    node_feat.append(extract_node_feat(halo))

    #record edge from current to parent (directed)
    if parent_id is not None:
        edge_index.append((curr_id, parent_id))

    #recurse into children (ancestor)
    for anc in list(halo.ancestors):
        traverse_tree(anc, height + 1, curr_id, counter,
                      edge_index, node_order, node_feat, node_names)
        
    return node_names, node_order, node_feat, edge_index

def save_tree_data(halo):
    #traverse the tree
    node_name, node_order, node_feats, edges = traverse_tree(halo, 0)
    #export the main branch
    main_branch = list(halo["prog", "uid"])
    #mask_main_branch = [node in main_branch for node in node_name]
    return node_name, node_order, node_feats, edges, main_branch


def read_subset_LH(LH_path, mass_min=1e13):
    ''' 
    Given a LH folder path LH_path with a fixed label (sigma_8, omega_m), 
    read into the ytree data 'tree_0_0_0.dat', which contains ~1e5 trees
    NEW: extract the subset with root mass greater than root_mass_min 1e13
    OLD: extract the subset with log(Nparticle) > 0.5
    return: subset of tree_samples (i.e. list of roots), and cosmological param y
    '''
    tree_collection = ytree.load(LH_path)
    subset = tree_collection[tree_collection["Mvir"] > mass_min]
    return subset, tree_collection  #avoid garbage collection / ReferenceError

def build_h5_fromytree_per_rank(mass_min, id_start=0, id_end=1000,
                                 all_LH_paths='/mnt/ceph/users/camels/PUBLIC_RELEASE/Rockstar/CAMELS-SAM/LH/',
                                 label_path="/mnt/home/rstiskalek/ceph/CAMELS-SAM/LH_rockstar_99.hdf5",
                                 save_path='datasets/merger_trees_h5/',
                                 save_name='full_data_rank'):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"[Rank {rank}] Comm size is {size}")

    os.makedirs(save_path, exist_ok=True)
    fname_out = f'{save_path}/{save_name}_{rank}.hdf5'

    # Read labels
    with h5py.File(label_path, 'r') as f:
        grp = f["params"]
        lh_keys = grp['LH'][:]
        Omega_m = np.array(grp['Omega_m'][:]).reshape(-1,1)
        sigma_8 = np.array(grp['sigma_8'][:]).reshape(-1,1)
        y = np.hstack((Omega_m, sigma_8))
        y_dict =  dict(zip(lh_keys, y))

    all_files = ['tree_0_0_0.dat', 'tree_0_1_0.dat', 'tree_1_0_0.dat',  'tree_1_1_0.dat', 'tree_0_0_1.dat', 'tree_0_1_1.dat', 'tree_1_0_1.dat', 'tree_1_1_1.dat']
    lh_ids = list(range(id_start, id_end))
    local_lh_ids = lh_ids[rank::size]  # divide work acrossranks

    start = time.time()
    with h5py.File(fname_out, 'w') as f:
        #for lh_id, path in local_info:
        for lh_id in local_lh_ids:
            for file_name in all_files:
                path = os.path.join(all_LH_paths, f"LH_{lh_id}/ConsistentTrees/{file_name}")
                if not os.path.exists(path): #some LH ids do not have all 8 dat files
                    continue

                try:
                    tree_samples, tree_collection = read_subset_LH(path, mass_min)
                    group_name = f"LH_{lh_id}"
                    if group_name in f:
                        grp = f[group_name]
                    else:
                        grp = f.create_group(group_name)

                    if lh_id in y_dict and 'y' not in grp:
                        grp.create_dataset('y', data=y_dict[lh_id])

                    print(f"started LH_{lh_id} from {file_name}")
                    lh_count = 0
                    for root in tree_samples:
                        node_name, node_order, node_feats, edges, main_branch = save_tree_data(root)
                        node_feats = np.concatenate(node_feats, axis=0)
                        # mpart = mass_particle(tree_collection.omega_matter) #
                        # node_feats[:,0] = node_feats[:,0] / (mpart+eps) #DEPRECIATED: write off mass with Npart
                        subgroup_name = f"{group_name}_{root['uid']}"
                        if subgroup_name in grp:
                            sub_grp = grp[subgroup_name]
                        else:
                            sub_grp = grp.create_group(subgroup_name)
                        sub_grp.create_dataset('node_name', data=np.array(node_name, dtype='i8'))
                        sub_grp.create_dataset('node_order', data=np.array(node_order, dtype='i8'))
                        sub_grp.create_dataset('node_feats', data=node_feats)
                        sub_grp.create_dataset('edge_index', data=np.array(edges, dtype='i8'))
                        sub_grp.create_dataset('main_branch', data=np.array(main_branch, dtype='i8'))
                        lh_count += 1
                    
                    print(f"processed LH_{lh_id} from {file_name}, with {lh_count} trees!")

                except IOError:
                    print(f"[Rank {rank}] Failed to read LH_{lh_id}")
                    continue

    comm.Barrier()
    if rank == 0:
        duration = time.time() - start
        print(f"[Rank 0] Finished writing per-rank HDF5 files in {duration:.2f}")

def merge_h5_rank_files(save_path, final_filename='merged_data.hdf5', 
                        h5_file_names='full_data_rank_*.hdf5'):
    output_path = os.path.join(save_path, final_filename)
    rank_files = sorted(glob.glob(os.path.join(save_path, h5_file_names)))

    with h5py.File(output_path, 'w') as f_out:
        for rank_file in rank_files:
            with h5py.File(rank_file, 'r') as f_in:
                for group_name in f_in.keys():
                    f_in.copy(group_name, f_out)

    print(f"Merged {len(rank_files)} files into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=pathlib.Path, \
                        default='/mnt/ceph/users/camels/PUBLIC_RELEASE/Rockstar/CAMELS-SAM/LH/', help='dataset parent dir')
    # parser.add_argument('--file_name', type=str, \
    #                     default='tree_0_0_0.dat', help='graph dataset file')
    parser.add_argument('--label_path', type=str, \
                        default="/mnt/home/rstiskalek/ceph/CAMELS-SAM/LH_rockstar_99.hdf5", help='label dataset file')
    parser.add_argument('--save_path', type=pathlib.Path, \
                        default='/mnt/home/thuang/ceph/playground/datasets/merger_trees_1000_feat_1e13/', help='save tree dataset directory')
    parser.add_argument('--save_name', type=str, default='full_data_rank', help='output h5 file name')
    #parser.add_argument('--logNpart_min', type=float, default=5.0, help='minimum log10 Nparticle=mass/mpart of root halo (node); 5.0 -> 20k+ tree')
    parser.add_argument('--mass_min', type=float, default=1e13, help='minimum mass root halo (node), 1e13')
    parser.add_argument('--id_start', type=int, default=0, help='LH folder start id')
    parser.add_argument('--id_end', type=int, default=1000, help='LH folder end id')
    parser.add_argument('--post_merge', action="store_true", help='to merge processed files only')
    parser.add_argument('--h5_name', type=str, default='full_data_rank_*.hdf5', help='saved h5 file name')


    args = parser.parse_args()

    # merge after writing
    if args.post_merge:
        merge_h5_rank_files(args.save_path, h5_file_names=args.h5_name)
    # safe write
    else:
        build_h5_fromytree_per_rank(args.mass_min, args.id_start, args.id_end, 
                        args.dataset_path, args.label_path,
                        args.save_path, args.save_name)

    