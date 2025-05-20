import torch 
import numpy as np
import math
import argparse
import pathlib
import h5py
import time
import os
import matplotlib.pyplot as plt
import pickle
import json
from itertools import product
from math import ceil, floor

torch.set_printoptions(precision=4,sci_mode=False,linewidth=150)
torch.set_default_dtype(torch.float64)

Rc_dict_om = {
    'Quijote': torch.tensor([112, 6, 23, 73, 105, 115, 76, 89, 7, 86, 99, 88]),
    'CAMELS-SAM': torch.tensor([60.5, 1, 8, 1.5, 44, 3.5, 48.5, 52, 0.5, 11, 8.5, 53]),
    'CAMELS-TNG': torch.tensor([0.4, 7.1, 0.6, 0.1, 9.6, 2.3, 0.2, 13.7, 2.4, 1.7, 3.1, 2.5]),
    'name': 'om',
    'limits': [0.1, 0.5]
}

Rc_dict_s8 = {
    'Quijote': torch.tensor([15, 53, 6, 27, 100, 8, 83, 31, 67, 7, 79, 110]),
    'CAMELS-SAM': torch.tensor([3, 20, 1, 58.5, 15.5, 8.5, 15, 0.5, 26, 22, 22.5, 35.5]),
    'CAMELS-TNG': torch.tensor([3.3, 3.7, 0.4, 20.5, 7.1, 11.6, 0.2, 0.1, 1.9, 1.2, 0.3, 2]),
    'name': 's8',
    'limits': [0.6, 1.0]
}


def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile. Source: https://github.com/pytorch/pytorch/issues/64947

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): 1D tensor
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Logic
    k = inter(q * (input.shape[0] - 1)) + 1
    out = torch.kthvalue(input, k)[0]
    return out

# x = torch.arange(1., 6.)
# print(torch_quantile(x, 0.5), torch_quantile(x, 1/3), torch_quantile(x, 2/3))


def load_position_h5(h5_path, idx, data_name='BSQ', device="cuda"):
    # period_dict = {'BSQ': 1000, 'CAMELS-SAM': 100, 'CAMELS': 25}
    # period = period_dict[data_name]
    with h5py.File(h5_path, 'r') as f:
        group = f[data_name]
        #labels = f["params"]
        g = group[f"{data_name}_{idx}"]
        x = torch.tensor(g['X'][:], dtype=torch.float64).to(device) 
        y = torch.tensor(g['Y'][:], dtype=torch.float64).to(device) 
        z = torch.tensor(g['Z'][:], dtype=torch.float64).to(device)  
    return x,y,z

def load_param_h5(h5_path, target_name='om', device="cuda"):
    with h5py.File(h5_path, 'r') as f:
        labels = f["params"]
        if target_name == 'om':
            Y = torch.tensor(labels['Omega_m'][:], dtype=torch.float64)#.reshape(-1,1)
        else:
            Y =torch.tensor(labels['sigma_8'][:], dtype=torch.float64)#.reshape(-1,1)
        #Y = torch.vstack([Omega_m, sigma_8]) #torch.concatenate([Omega_m, sigma_8], dim=-1).to(device)
    return Y.to(device)


def MSE_loss(ypred, y):
    return torch.mean((ypred - y)**2)

def variance(y):
    #compute mean vector (per feat), then average over variance per element
    mean = y.mean(axis=0)
    return torch.mean((y - mean)**2) 

def compute_R2(ypred, y):
    mse = MSE_loss(ypred, y)
    var = variance(y)
    return 1 - mse/var

def bootstrap_r2(Y_pred_test, Y_test, num_bootstrap=1000, seed=42):
    """
    Compute bootstrapped R² scores from predictions and ground truth.

    Args:
        Y_pred_test (torch.Tensor): Predicted values, shape [n]
        Y_test (torch.Tensor): Ground truth values, shape [n]
        num_bootstrap (int): Number of bootstrap samples
        seed (int, optional): Random seed for reproducibility

    Returns:
        torch.Tensor: Bootstrap R² scores mean and std
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = Y_test.shape[0]
    r2_scores = torch.zeros(num_bootstrap)

    for i in range(num_bootstrap):
        idx = torch.randint(0, n, (n,), device=Y_test.device)  # sample with replacement
        y_pred_sample = Y_pred_test[idx]
        y_true_sample = Y_test[idx]
        r2_scores[i] = compute_R2(y_pred_sample, y_true_sample)

    return r2_scores.mean(), r2_scores.std()


def pdm(x, L):
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # shape: [N, N]
    return L/2 - torch.abs(L/2 - torch.abs(diff))

def compute_dataset_features(prefix, Rc_dict, device, args):
    if prefix == 'Quijote':
        n_train, n_val, n_test, period = 19651, 6551, 6550, 1000
        data_name = 'BSQ'
    elif prefix == 'CAMELS-SAM':
        n_train, n_val, n_test, period = 600, 204, 196, 100
        data_name = 'LH'
    elif prefix == 'CAMELS-TNG':
        n_train, n_val, n_test, period = 600, 200, 200, 25
        data_name = 'LH'
    else:
        raise NotImplementedError 
    Rc = Rc_dict[prefix]
    target_name = Rc_dict['name']

    Xt, _ = compute_h5_features(f"{args.data_dir}/{args.h5_path_train}", 
                             data_name, n_train, Rc, period, device)
    torch.save(Xt, f"{args.feature_dir}/{prefix}_{target_name}_train.pt") 

    Xv, _ = compute_h5_features(f"{args.data_dir}/{args.h5_path_val}", 
                             data_name, n_val, Rc, period, device)
    torch.save(Xv, f"{args.feature_dir}/{prefix}_{target_name}_val.pt") 

    Xe, _ = compute_h5_features(f"{args.data_dir}/{args.h5_path_test}", 
                             data_name, n_test, Rc, period, device)
    torch.save(Xe, f"{args.feature_dir}/{prefix}_{target_name}_test.pt") 


def compute_h5_features(h5file, data_name, num_catalog, Rc, period, device='cuda'):
    """
    Compute features for each catalog in an HDF5 file.

    Args:
        h5file (str): Path to HDF5 file.
        base_path (str): Path inside HDF5 file to datasets.
        num_catalog (int): Number of catalogs.
        Rc (torch.Tensor): Cutoff radius tensor.
        period (float): Periodic boundary box size.
        device (str): 'cuda' or 'cpu'.

    Returns:
        distF (torch.Tensor): Shape (4, len(Rc), num_catalog).
        numPts (torch.Tensor): Shape (num_catalog,)
    """
    num_stats = 4
    Rc = Rc.to(device)
    distF = torch.zeros((num_stats, len(Rc), num_catalog), device=device)
    numPts = torch.zeros(num_catalog, dtype=torch.long)

    print(f"\n{h5file}")
    print("  catalog  time (sec)")

    for c in range(num_catalog):
        start_time = time.time()

        # Load catalog from HDF5
        x, y, z = load_position_h5(h5file, c, data_name=data_name, device=device)
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        distF[:, :, c] = compute_catalog_features(x, y, z, Rc, period)
        numPts[c] = x.shape[0]

        elapsed = time.time() - start_time
        print(f"  [{c:05d}]   {elapsed:.3f}")

    return distF, numPts


def compute_catalog_features(x, y, z, Rc, L):
    """
    Compute catalog (point cloud simulation) features 
    based on periodic boundary conditions in 3D.

    Args:
        x, y, z: 1D tensors of coordinates (assumed to be same length, shape [n])
        Rc: 1D tensor of cutoff radii
        L: scalar box size
    Returns:
        distF: Tensor of shape (4, len(Rc)) with mean, std, and 1/3 and 2/3 quantiles of distances
    """
    device = x.device  # Assume x, y, z are already on GPU if needed
    dxM = pdm(x, L) #(n,n)
    dyM = pdm(y, L) #(n,n)
    dzM = pdm(z, L) #(n,n)

    # Compute 3D squared distances
    sqDstM = dxM**2 + dyM**2 + dzM**2
    sqDstM.fill_diagonal_(float('inf'))  # Exclude self-pairs

    # Flatten and filter by maximum Rc
    sqDstA = sqDstM[sqDstM < Rc.max()**2]
    sqDstA = torch.sort(sqDstA)[0]  # Sort ascending

    # Compute features
    distF = torch.zeros((4, len(Rc)), device=device)
    for j in range(len(Rc)):
        thresh = Rc[j]**2
        v = sqDstA[sqDstA <= thresh]
        #print(v.shape, device)
        if v.numel() > 0:
            distF[0, j] = v.mean()
            distF[1, j] = v.std()
            distF[2, j] = torch_quantile(v, 1/3) #torch.quantile(v, torch.tensor([1/3, 2/3], device=device))
            distF[3, j] = torch_quantile(v, 2/3)
        else:
            distF[:, j] = float('nan')  # Handle case of no valid distances

    return distF

# x = torch.linspace(0, 1, 10) #torch.rand(10,1).squeeze(1)
# y = torch.linspace(0, 1, 10) #torch.rand(10,1).squeeze(1)
# z = torch.linspace(0, 1, 10) #torch.rand(10,1).squeeze(1)
# Rc = torch.tensor([0.1, 0.2, 0.3])
# distF = compute_catalog_features(x, y, z, Rc, L=1)
# print(distF)


def fit_least_squares(X: torch.Tensor, Y: torch.Tensor):
    """
    Solve linear least squares (with additional bias term) with feature normalization 
    Args:
        X (torch.Tensor): Shape [d, n] -- features x samples
        Y (torch.Tensor): Shape [n] or [1, n] -- target values
    Returns:
        w (torch.Tensor): Weight vector [d]
        b (torch.Tensor): Bias scalar
    """
    if Y.ndim == 2:
        Y = Y.flatten()

    # Normalize X
    m = X.mean(dim=1, keepdim=True)        # [d, 1]
    s = X.std(dim=1, unbiased=True, keepdim=True)  # [d, 1]
    dX = (X - m) / s                        # [d, n]

    # Demean Y
    Y_mean = Y.mean()
    dY = Y - Y_mean                         # [n]

    # Least-squares solution using pseudo-inverse
    G = dX @ dX.T                           # [d, d]
    rhs = dX @ dY                           # [d]
    w = torch.linalg.lstsq(G, rhs).solution  # [d]

    # Scale back weights
    w = w / s.squeeze()

    # Compute bias
    b = Y_mean - w @ m.squeeze()           # scalar

    return w, b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default='/mnt/home/rstiskalek/ceph/graps4science', help='data dir')
    parser.add_argument('--h5_path_train', 
                        default='CAMELS-SAM_LH_gal_99_top5000_train.hdf5', #'/mnt/home/rstiskalek/ceph/graps4science/Quijote_BSQ_rockstar_10_top5000.hdf5',
                         help='h5 path to load the training data')
    parser.add_argument('--h5_path_val', 
                        default='CAMELS-SAM_LH_gal_99_top5000_val.hdf5', 
                         help='h5 path to load the test data')
    parser.add_argument('--h5_path_test', 
                        default='CAMELS-SAM_LH_gal_99_top5000_test.hdf5', 
                         help='h5 path to load the test data')
    parser.add_argument('--data_name', default='LH', type=str,
                         help='data group name in the h5 file') #TODO: sync across BSQ and LH? 
    
    parser.add_argument('--feature_dir', type=str, \
                        default='/mnt/home/thuang/ceph/playground/datasets/pos_param_lls', help='data feature dir')

    parser.add_argument('--output_dir', #default='/mnt/home/rstiskalek/ceph/CAMELS-SAM/LH_rockstar_99.hdf5',
                        default='/mnt/home/thuang/playground/param_prediction',
                        #default='/mnt/home/thuang/ceph/playground/datasets/point_clouds/invPwrFeat',
                         help='save path')
    parser.add_argument('--target_name', type=str, default='om', help='select target variable')
    parser.add_argument('--test_sample_idx', type=int, default=None, help='if specified, only test on \
                        test_sample test clouds, and save the predictions')
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dir_train = f"{args.data_dir}/{args.h5_path_train}"
    dir_val = f"{args.data_dir}/{args.h5_path_val}"
    dir_test = f"{args.data_dir}/{args.h5_path_test}"
    filename = dir_train.split('/')[-1]             # Get the file name
    prefix = filename.split('_')[0]            # Extract 'CAMELS-SAM'

    feat_path = f"{args.feature_dir}/{prefix}_{args.target_name}"
    if not os.path.exists(f"{feat_path}_train.pt"):
        print(f"precomputing features...")
        #Pre-compute features...
        compute_dataset_features(prefix, Rc_dict_om, "cpu", args)
        compute_dataset_features(prefix, Rc_dict_s8, "cpu", args)
    
    X_train = torch.load(f"{feat_path}_train.pt")
    X_val =  torch.load(f"{feat_path}_val.pt")
    X_test =  torch.load(f"{feat_path}_test.pt")
     
    X_train = X_train.flatten(start_dim=0, end_dim=1)
    X_train = X_train.to(device) #shape (4*12, n_cloud_train)
    #X_val = (X_val.flatten(start_dim=0, end_dim=1)).to(device)
    X_test = X_test.flatten(start_dim=0, end_dim=1) 
    X_test = X_test.to(device)  #shape (4*12, n_cloud_test)
 
    #load labels 
    Y_train = load_param_h5(dir_train, args.target_name, device=device)
    #Y_val = load_param_h5(dir_val, device=device) --used to select Rc
    Y_test = load_param_h5(dir_test, args.target_name, device=device)

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    w, b = fit_least_squares(X_train, Y_train)
    #pred: clip into the range
    lim = Rc_dict_om['limits'] if args.target_name == 'om' else Rc_dict_s8['limits']
    Y_pred_train = torch.clamp( w @ X_train + b , min=lim[0], max=lim[1])
    Y_pred_test = torch.clamp( w @ X_test + b , min=lim[0], max=lim[1])

    #Compute metrics
    R2_train = compute_R2(Y_pred_train, Y_train)
    R2_test = compute_R2(Y_pred_test, Y_test)
    R2_boot, R2_boot_std = bootstrap_r2(Y_pred_test, Y_test, )

    print(f"{prefix} on {args.target_name}: R2_train={R2_train:.4f}, R2_test={R2_test:.4f}, R2_boot_std={R2_boot_std:.4f}")
    results = {
        'prefix': prefix,
        'target_name': args.target_name,
        'R2_train': R2_train.item(),
        'R2_test': R2_test.item(),
        'R2_boot_std': R2_boot_std.item()
    }
    
    result_dir = args.feature_dir if args.output_dir is None else args.output_dir
    results_path = f"{result_dir}/lls_results.json" 
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
    else:
        existing_results = []
        
    # Append new result and save
    existing_results.append(results)
    with open(results_path, 'w') as f:
        json.dump(existing_results, f, indent=4)

    print(f"Appended test evaluation results to: {results_path}")