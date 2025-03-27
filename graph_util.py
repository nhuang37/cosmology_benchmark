import torch 
import numpy as np
import argparse
import pathlib
import os
import pickle
import scipy.spatial as SS
import random
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.utils import scatter, degree

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def build_graph(pos, r_link=90, L=1000, leafsize=16, epsilon=0.00001):
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
    d_0 = np.abs(pos[row]-pos[col]) #(E, 3)
    d_L = np.where(d_0 < r_link, d_0, L-d_0)
    
    # edge feature as squared Distance
    edge_dist = np.linalg.norm(d_L, axis=1)**2

    return edge_index, edge_dist

def build_graphset(args):
    if args.neutrinos:
        assert str(args.dataset_path.name) == 'data_m=2000_n=5000_neutrinos=True.pkl'

    data = pickle.load(open(args.dataset_path, 'rb'))
    edges = []
    edge_features = []
    for i, (X,y) in enumerate(zip(data['Xs'], data['y'])):
        if i % 100 == 0:
            print(f"processing {i+1} cloud...")
        edge_index, edge_dist = build_graph(X, r_link=args.r_link)
        edges.append(edge_index)
        edge_features.append(edge_dist)
    dataset = {'edges': edges, 'edge_features': edge_features}
    parent_dir = args.dataset_path.parent
    pickle.dump(dataset, open(f'{parent_dir}/Rc={args.r_link}_graph_'+ str(args.dataset_path.name)[5:], 'wb'))


def mark(node_deg, p=1):
    n = node_deg.shape[0]
    avg_deg = node_deg.sum()/n
    node_weight = (avg_deg + 1) / (node_deg + 1)
    return torch.pow(node_weight, p)

def build_PyGdata(data_path, graph_path, r_link, p=1):
    data = pickle.load(open(data_path, 'rb'))
    Xs, ys = data['Xs'], data['y']
    n = Xs[0].shape[0]
    graphs = pickle.load(open(graph_path, 'rb'))
    edges, edge_feats = graphs['edges'], graphs['edge_features']
    PyGdataset = []
    for i, (edge_index, edge_dist) in enumerate(zip(edges, edge_feats)):
        # node weight: assume self-loops;  soft(avg_deg / x_deg)
        row = torch.LongTensor(edge_index[:,0])
        node_deg = degree(row, num_nodes = n, dtype=torch.long)
        node_weight = mark(node_deg, p=p)
        # Construct the graph in PyG
        graph = Data(x=torch.ones(n,1), #can change to have node feature later
                    y=torch.tensor(ys[i], dtype=torch.float32).unsqueeze(0),
                    edge_index=torch.tensor(edge_index.T, dtype=torch.long), #(2, E) in PyG format
                    edge_attr=torch.tensor(edge_dist/r_link**2, dtype=torch.float32).unsqueeze(1), #(E, k) in PyG format
                    node_weight = node_weight.unsqueeze(1))
        PyGdataset.append(graph)
    return PyGdataset

def build_PyGdata_velocity(data_path, graph_path, r_link, p=1):
    data = pickle.load(open(data_path, 'rb'))
    Xs, ys = data['Xs'], data['y']
    n = Xs[0].shape[0]
    graphs = pickle.load(open(graph_path, 'rb'))
    edges, edge_feats = graphs['edges'], graphs['edge_features']
    PyGdataset = []
    for i, (edge_index, edge_dist) in enumerate(zip(edges, edge_feats)):
        # node weight: assume self-loops;  soft(avg_deg / x_deg)
        row = torch.LongTensor(edge_index[:,0])
        node_deg = degree(row, num_nodes = n, dtype=torch.long)
        node_weight = mark(node_deg, p=p)
        # Construct the graph in PyG
        graph = Data(x=torch.tensor(Xs[i], dtype=torch.float32).unsqueeze(1), #(N, 1, 3)
                    y=torch.tensor(ys[i], dtype=torch.float32).unsqueeze(1), #(N, 1, 3)
                    edge_index=torch.tensor(edge_index.T, dtype=torch.long), #(2, E) in PyG format
                    edge_attr=torch.tensor(edge_dist/r_link**2, dtype=torch.float32).unsqueeze(1), #(E, k) in PyG format
                    node_weight = node_weight.unsqueeze(1))
        PyGdataset.append(graph)
    return PyGdataset

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

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=pathlib.Path, \
                        default='datasets/position_only/data_m=2000_n=5000.pkl', help='dataset file')
    # parser.add_argument('--graph_path', type=pathlib.Path, \
    #                     default='datasets/position_only/graph_m=2000_n=5000.pkl', help='graph dataset file')
    parser.add_argument('--r_link', type=float, default=90, help='knn graph distance cut-off threshold')
    parser.add_argument('--neutrinos', action='store_true', help='use neutrino dataset')
    #the same script applies to position_only and position_velocity point clouds, 
    # as we only use position features to construct the graph

    args = parser.parse_args()

    dataset = build_graphset(args)
    #train_loader, valid_loader, test_loader = split_dataloader(dataset)
    #print(f"train={len(train_loader)}, valid={len(valid_loader)}, test={len(test_loader)}")