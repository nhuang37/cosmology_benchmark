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

torch.set_printoptions(precision=4,sci_mode=False,linewidth=150)
torch.set_default_dtype(torch.float64)


def load_point_cloud_h5(h5_path, idx, data_name='BSQ', device="cuda"):
    # period_dict = {'BSQ': 1000, 'CAMELS-SAM': 100, 'CAMELS': 25}
    # period = period_dict[data_name]
    with h5py.File(h5_path, 'r') as f:
        group = f[data_name]
        #labels = f["params"]
        g = group[f"{data_name}_{idx}"]
        x = torch.tensor(g['X'][:], dtype=torch.float64).to(device) 
        y = torch.tensor(g['Y'][:], dtype=torch.float64).to(device) 
        z = torch.tensor(g['Z'][:], dtype=torch.float64).to(device)  

        # Construct labels
        vx = torch.tensor(g['VX'][:], dtype=torch.float64).to(device)
        vy = torch.tensor(g['VY'][:], dtype=torch.float64).to(device) 
        vz = torch.tensor(g['VZ'][:], dtype=torch.float64).to(device) 
        
    return x,y,z,vx,vy,vz



def MSE_loss(ypred, y):
    return torch.mean((ypred - y)**2)

def variance(y):
    #compute mean vector (per feat), then average over variance per element
    mean = y.mean(axis=0)
    return torch.mean((y - mean)**2) 

def pdm(x, L):
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # shape: [N, N]
    return L/2 - torch.abs(L/2 - torch.abs(diff))

def compute_invPwrLaw_features(x,y,z, K, P, period=1000):
    # Fourier Features
    xp = x * (2 * math.pi / period)  # (n,)
    k_vals = torch.arange(1, K + 1, device=device).view(-1, 1)  # (K, 1)
    Sx = torch.sin(k_vals * xp)  # (K, n)
    Cx = torch.cos(k_vals * xp)  # (K, n)
    n = x.shape[0]

    # Pairwise distance of periodict boundary box of period = 100
    dx = pdm(x, period)
    dy = pdm(y, period)
    dz = pdm(z, period)

    # 3D pairwise distance (inverse, power)
    dist = torch.sqrt(dx**2 + dy**2 + dz**2)
    # Exclude self-pairs (set diagonal to Inf) and any degenerate pairs
    dist[dist==0] = float('inf')
    #dist.fill_diagonal_(float('inf'))
    # Compute 1/r
    invPwr = period / dist #shape (n, n)
    # Initialize Feature
    Feat = torch.zeros((K*P, n), dtype=x.dtype, device=x.device)
    # Construct and store features
    for p in range(P): #iterate through powers
        block = Sx * (Cx @ invPwr) - Cx * (Sx @ invPwr) # shape (K, n)
        Feat[p*K : (p+1)*K, :] = block 
        invPwr = invPwr * (period / dist)
    return Feat
    
def get_h5_group_size(dir, data_name='LH'):
    with h5py.File(dir, 'r') as f:
        group = f[data_name]
        size = len(group.keys())
    return size


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
    parser.add_argument('--output_dir', #default='/mnt/home/rstiskalek/ceph/CAMELS-SAM/LH_rockstar_99.hdf5',
                        default='/mnt/home/thuang/ceph/playground/datasets/point_clouds/invPwrFeat',
                         help='save path')
    #parser.add_argument('--num_clouds', default=3072, type=int, help="number of point clouds")
    parser.add_argument('--K', default=10, type=int, help="number of x frequencies")
    parser.add_argument('--P', default=4, type=int, help="number of powers of inv_dist")

    parser.add_argument('--train_clouds', default=2048, type=int, help="number of training point clouds")
    parser.add_argument('--val_clouds', default=512, type=int, help="number of VAL point clouds")
    parser.add_argument('--test_clouds', default=512, type=int, help="number of test point clouds")

    parser.add_argument('--test_sample_idx', type=int, default=None, help='if specified, only test on \
                        test_sample test clouds, and save the predictions')
    args = parser.parse_args()
    print(args)

    #Data split: 
    dir = f"{args.data_dir}/Quijote_BSQ_rockstar_10_top5000.hdf5"
    print(dir)
    dir_train = f"{args.data_dir}/{args.h5_path_train}"
    dir_val =  f"{args.data_dir}/{args.h5_path_val}"
    dir_test = f"{args.data_dir}/{args.h5_path_test}"

    train_size = get_h5_group_size(dir_train, args.data_name)
    val_size = get_h5_group_size(dir_val, args.data_name)
    test_size = get_h5_group_size(dir_test, args.data_name)

    # n = train_size + val_size + test_size
    idxTrain = list(range(0, min(train_size, args.train_clouds))) # list(range(0, 19651)) 
    idxVal =  list(range(0, min(val_size, args.val_clouds))) #list(range(19651, 26202)) #
    idxTest = list(range(0, min(test_size, args.test_clouds))) #list(range(26202, 32752))#

    filename = dir_train.split('/')[-1]             # Get the file name
    prefix = filename.split('_')[0]            # Extract 'CAMELS-SAM'
    print(prefix)  # Output: CAMELS-SAM
    print(f"train on {len(idxTrain)} clouds, test on {len(idxTest)} clouds")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    period_dict = {'Quijote': 1000, 'CAMELS-SAM': 100, 'CAMELS-TNG': 25000, 'fiducial': 1000}
    period = period_dict[prefix]
    #Feature Order
    K = args.K
    P = args.P
    A = torch.zeros((K*P, K*P)).to(device)
    b = torch.zeros((K*P,1)).to(device)
    output_dir = f"{args.output_dir}_K={K}_P={P}"
    os.makedirs(output_dir, exist_ok=True)

    if args.test_sample_idx is not None: #eval test sample idx = 0
        w = torch.load(f"{output_dir}/weight.pt")
        x,y,z,vx,vy,vz = load_point_cloud_h5(dir_test, args.test_sample_idx, args.data_name) 
        target = torch.stack([vx,vy,vz], dim=-1).cpu()
        Fx = compute_invPwrLaw_features(x,y,z,K,P, period) #(d, n)
        Fy = compute_invPwrLaw_features(y,z,x,K,P, period)
        Fz = compute_invPwrLaw_features(z,x,y,K,P, period)
        #feat_time = time.time()
        F_xyz =  torch.stack([Fx, Fy, Fz], dim=0)  # shape: (3, d, n)
        v_pred = torch.einsum('d,cdn->cn', w.squeeze(), F_xyz).T.detach().cpu()
        mse = MSE_loss(v_pred, target)
        var = variance(target)
        R2 = 1 - mse/var
        test_pred_path = os.path.join(args.output_dir, f"{prefix}_test_pred_{args.test_sample_idx}.pt")
        torch.save([v_pred,target], test_pred_path)
        print(f"Saved test velocity predictions to: {test_pred_path}, R2_test = {R2:.4f}")
    
    else:
        start_time = time.time()
        ## Train + Val
        # compute all features
        for t in idxTrain:
            x,y,z,vx,vy,vz = load_point_cloud_h5(dir_train, t, args.data_name) #each is a (n,1) vector or (1,n) ? CHECK
            Fx = compute_invPwrLaw_features(x,y,z,K,P, period) #(d, n)
            Fy = compute_invPwrLaw_features(y,z,x,K,P, period)
            Fz = compute_invPwrLaw_features(z,x,y,K,P, period)
            A = A + Fx @ Fx.T + Fy @ Fy.T + Fz @ Fz.T #(d,d)
            b = b + Fx @ vx.unsqueeze(1) + Fy @ vy.unsqueeze(1) + Fz @ vz.unsqueeze(1)  #(d,1)
        
        # for t in idxVal: #-> use it for K,P selection
        #     x,y,z,vx,vy,vz = load_point_cloud_h5(dir_val, t, args.data_name) #each is a (n,1) vector or (1,n) ? CHECK
        #     Fx = compute_invPwrLaw_features(x,y,z,K,P, period) #(d, n)
        #     Fy = compute_invPwrLaw_features(y,z,x,K,P, period)
        #     Fz = compute_invPwrLaw_features(z,x,y,K,P, period)
        #     A = A + Fx @ Fx.T + Fy @ Fy.T + Fz @ Fz.T #(d,d)
        #     b = b + Fx @ vx.unsqueeze(1) + Fy @ vy.unsqueeze(1) + Fz @ vz.unsqueeze(1)  #(d,1)
        # Compute w after seeing all train+val clouds    
        w = torch.linalg.lstsq(A, b).solution #(d,1)
        torch.cuda.synchronize()  # Make sure all training GPU ops are done
        train_time = time.time()
        ## Test
        mse_all = []
        var_all = []
        for idx in idxTest:
            #idx_time = time.time()
            x,y,z,vx,vy,vz = load_point_cloud_h5(dir_test, idx, args.data_name) 
            Fx = compute_invPwrLaw_features(x,y,z,K,P, period) #(d, n)
            Fy = compute_invPwrLaw_features(y,z,x,K,P, period)
            Fz = compute_invPwrLaw_features(z,x,y,K,P, period)
            #feat_time = time.time()
            F_xyz =  torch.stack([Fx, Fy, Fz], dim=0)  # shape: (3, d, n)
            v_pred = torch.einsum('d,cdn->cn', w.squeeze(), F_xyz).T  # shape: (n, 3)
            ## the above einsum computation is a (slightly) fast version of the following 4 lines
            # vx_pred = w.T @ Fx #(1,n)
            # vy_pred = w.T @ Fy #(1,n)
            # vz_pred = w.T @ Fz #(1,n)
            # v_pred = torch.vstack([vx_pred, vy_pred, vz_pred]).T #(n,3)
            #mat_time = time.time()
            v_target = torch.stack([vx, vy, vz], dim=-1)
            var = variance(v_target) #torch.mean((v_target - v_target.mean())**2)
            var_all.append(var) #for computing R2 later
            mse = MSE_loss(v_pred, v_target)
            mse_all.append(mse)
            #print(f"{(feat_time - idx_time):.4f}, {(mat_time - feat_time):.4f}")
        torch.cuda.synchronize()  # Make sure all training GPU ops are done
        end_time = time.time()
        final_mse = sum(mse_all)/len(mse_all)
        final_var = sum(var_all)/len(var_all)
        R2 = 1 - final_mse/final_var

        print(f"R2_test = {R2:.4f}, Train time = {(train_time - start_time):.4f}, \
            Test time = {(end_time - train_time):.4f}") #R

        # save trained weights
        torch.save(w, f"{output_dir}/weight.pt")
        print(f"saved trained weights to: {output_dir}")
        ## Save to file
        results_dict = {
            'K': args.K,
            'P': args.P,
            'data_file': prefix,
            'train_clouds': len(idxTrain)+len(idxVal),
            'test_clouds': len(idxTest),
            'R2': R2.item()
        }

        test_result_path = os.path.join(output_dir, f"{prefix}_test_R2_result.json")
        # Load existing results if file exists
        if os.path.exists(test_result_path):
            with open(test_result_path, 'r') as f:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
        else:
            existing_results = []
        
        # Append new result and save
        existing_results.append(results_dict)
        with open(test_result_path, 'w') as f:
            json.dump(existing_results, f, indent=4)

        print(f"Appended test evaluation results to: {test_result_path}")

