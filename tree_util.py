import ytree
import numpy as np
import torch 
from torch.nn.functional import one_hot
from torch_geometric.data import Data
import random
import time
import argparse
import pathlib
import pickle

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


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

def construct_PyG_data(halo):
    #traverse the tree
    node_name, node_order, node_feats, edges = traverse_tree(halo, 0)
    #map node names to indices
    node_to_index = {name: index for index, name in enumerate(node_name)}
    edge_list = [(node_to_index[source], node_to_index[target]) for source, target in edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).T.contiguous()
    # Create PyTorch Geometric Data object
    data = Data(x=torch.cat(node_feats, dim=0),
                edge_index=edge_index, pos=torch.LongTensor(node_order).unsqueeze(1))
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
    y = torch.tensor([tree_collection.hubble_constant, tree_collection.omega_matter, 
                      tree_collection.omega_lambda], dtype=float).view(-1,1)
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