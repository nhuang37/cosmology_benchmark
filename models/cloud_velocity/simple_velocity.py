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
from utils.get_redshift_pos import pos_redshift_space

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



def load_cosmo_param_h5(h5_path, idx):
    with h5py.File(h5_path, 'r') as f:
        grp = f[f"params"]
        cosmo = {}
        for k in ['h', 'Omega_m', 'Omega_b', 'sigma_8', 'n_s']:
            if k in grp:
                cosmo[k] = grp[k][idx]
            else:
                if k == 'h':
                    cosmo[k] = 0.6711 #set to 1 (constant in CAMELS/CAMELS-SAM)
                else:
                    pass
    return cosmo

def MSE_loss(ypred, y):
    return torch.mean((ypred - y)**2)

def variance(y):
    #compute mean vector (per feat), then average over variance per element
    mean = y.mean(axis=0)
    return torch.mean((y - mean)**2) 

def pdm(x, L):
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # shape: [N, N]
    return L/2 - torch.abs(L/2 - torch.abs(diff))

def compute_invPwrLaw_features(x,y,z, K, P, period=1000, device="cuda", 
                               x_shrink=1.0, y_shrink=1.0, z_shrink=1.0, adaptive=False):
    # Fourier Features
    xp = x * (2 * math.pi / period)  # (n,)
    k_vals = torch.arange(1, K + 1, device=device).view(-1, 1)  # (K, 1)
    Sx = torch.sin(k_vals * xp)  # (K, n)
    Cx = torch.cos(k_vals * xp)  # (K, n)
    n = x.shape[0]

    # Pairwise distance of periodict boundary box of period = 100
    dx = pdm(x, period) * x_shrink
    dy = pdm(y, period) * y_shrink
    dz = pdm(z, period) * z_shrink #TODO: check - undo redshift effect on LOS w/ z-axis (anistropic dist)

    # 3D pairwise distance (inverse, power)
    dist = torch.sqrt(dx**2 + dy**2 + dz**2)
    ## TODO: if adaptive
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

def eval_pretrained_weight(w, idxVal, dir_val, K, P, period, data_name, device, 
                           eval_redshift=False, shrink=1.0, eval_per_axis=False):
    ## EVAL
    mse_all = []
    var_all = []
    if eval_per_axis:
        mse_all_x, mse_all_y, mse_all_z = [], [], []
        var_all_x, var_all_y, var_all_z = [], [], []

    for idx in idxVal:
        #idx_time = time.time()
        x,y,z,vx,vy,vz = load_point_cloud_h5(dir_val, idx, data_name)
        if eval_redshift: #use redshift positions instead
            cosmo = load_cosmo_param_h5(dir_val, idx)
            z = pos_redshift_space(z, vz, period, cosmo["h"])
            #assert shrink < 1.0, 'need to adjust for redshift LOS distortion by shrinking distance!'
        Fx = compute_invPwrLaw_features(x,y,z,K,P, period, device, z_shrink=shrink) #(d, n)
        Fy = compute_invPwrLaw_features(y,z,x,K,P, period, device, y_shrink=shrink)
        Fz = compute_invPwrLaw_features(z,x,y,K,P, period, device, x_shrink=shrink)
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
        if eval_per_axis:
            var_all_x.append(variance(vx))
            var_all_y.append(variance(vy))
            var_all_z.append(variance(vz))
            mse_all_x.append(MSE_loss(v_pred[:,0], vx))
            mse_all_y.append(MSE_loss(v_pred[:,1], vy))
            mse_all_z.append(MSE_loss(v_pred[:,2], vz))

    final_mse = sum(mse_all)/len(mse_all)
    final_var = sum(var_all)/len(var_all)
    R2 = (1 - final_mse/final_var).item()
    if eval_per_axis:
        var_axis = [var_all_x, var_all_y, var_all_z]
        mse_axis = [mse_all_x, mse_all_y, mse_all_z]
        R2x, R2y, R2z = [1- (sum(mse)/len(mse))/(sum(var)/len(var)) for mse, var in zip(mse_axis, var_axis)]
        return R2, R2x.item(), R2y.item(), R2z.item()
    else:
        return R2, None, None, None

def run_training(args, K=10, P=4):
    dir_train = f"{args.data_dir}/{args.h5_path_train}"
    dir_val =  f"{args.data_dir}/{args.h5_path_val}"

    train_size = get_h5_group_size(dir_train, args.data_name)
    val_size = get_h5_group_size(dir_val, args.data_name)

    # n = train_size + val_size + test_size
    idxTrain = list(range(0, min(train_size, args.train_clouds))) # list(range(0, 19651)) 
    idxVal =  list(range(0, min(val_size, args.val_clouds))) #list(range(19651, 26202)) #

    #filename = dir_train.split('/')[-1]             # Get the file name
    #prefix = filename.split('_')[0]            # Extract 'CAMELS-SAM'
    prefix = args.prefix
    print(prefix)  # Output: CAMELS-SAM
    print(f"train on {len(idxTrain)} clouds, eval on {len(idxVal)} clouds")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    period_dict = {'Quijote': 1000, 'CAMELS-SAM': 100, 'CAMELS-TNG': 25, 'fiducial': 1000}
    period = period_dict[prefix]
    #Feature Order
    A = torch.zeros((K*P, K*P)).to(device)
    b = torch.zeros((K*P,1)).to(device)
    output_dir = f"{args.output_dir}/{prefix}/_K={K}_P={P}_shrink={args.shrink_factor}"
    os.makedirs(output_dir, exist_ok=True)
    #TRAIN
    start_time = time.time()
    ## Train + Val
    # compute all features
    shrink = 1.0
    for t in idxTrain:
        x,y,z,vx,vy,vz = load_point_cloud_h5(dir_train, t, args.data_name) #each is a (n,1) vector or (1,n) ? CHECK
        if args.eval_redshift: #use redshift positions instead
            # cosmo = load_cosmo_param_h5(dir_train, t)
            z = pos_redshift_space(z, vz, period) # , cosmo["h"])
            shrink = args.shrink_factor #TODO: check
        Fx = compute_invPwrLaw_features(x,y,z,K,P, period, device, z_shrink=shrink) #(d, n)
        Fy = compute_invPwrLaw_features(y,z,x,K,P, period, device, y_shrink=shrink)
        Fz = compute_invPwrLaw_features(z,x,y,K,P, period, device, x_shrink=shrink)
        # guard against inf values
        for name, F in [('Fx',Fx),('Fy',Fy),('Fz',Fz)]:
            if not torch.isfinite(F).all():
                bad = (~torch.isfinite(F)).sum().item()
                print(f"[{t}] {name} has {bad} non-finite entries"); break
        A = A + Fx @ Fx.T + Fy @ Fy.T + Fz @ Fz.T #(d,d)
        b = b + Fx @ vx.unsqueeze(1) + Fy @ vy.unsqueeze(1) + Fz @ vz.unsqueeze(1)  #(d,1)
        if (t+1)%1000 == 0: #eval on a subset of val set
            w = torch.linalg.lstsq(A, b).solution
            R2 = eval_pretrained_weight(w, idxVal, dir_val[:512], K, P, period, 
                                args.data_name, device, args.eval_redshift, shrink)
            print(f"finished {t+1} clouds, subset val_R2 = {R2:.4f}")
        
    
    w = torch.linalg.lstsq(A, b).solution #(d,1)
    torch.cuda.synchronize()  # Make sure all training GPU ops are done
    train_time = time.time()
    # save trained weights
    weight_path = f"{output_dir}/weight.pt"
    torch.save(w, weight_path)
    print(f"saved trained weights to: {output_dir}")

    ## EVAL
    R2 = eval_pretrained_weight(w, idxVal, dir_val, K, P, period, 
                                args.data_name, device, args.eval_redshift, shrink)
    end_time = time.time()

    print(f"R2_val = {R2:.4f}, Train time = {(train_time - start_time):.4f}, \
        Eval time = {(end_time - train_time):.4f}") #R

    ## Save to file
    results_dict = {
        'K': args.K,
        'P': args.P,
        'data_file': prefix,
        'train_clouds': len(idxTrain),
        'val_clouds': len(idxVal),
        'R2_val': R2,
        'redshift': args.eval_redshift,
        'shrink_factor': shrink,
    }

    test_result_path = os.path.join(output_dir, f"{prefix}_val_R2_result.json")
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

    print(f"Appended validation set evaluation results to: {test_result_path}")
    return R2, w, weight_path, K, P, period

def hyperparameter_search(args):
    Ks = [10,15,25]#[10,15,25]
    Ps = [2,3,4]
    best_R2 = -100
    best_config = None
    best_w = None
    best_path = None
    for P, K in product(Ps, Ks):
        print(f"Running P={P}, K={K}")
        start_time = time.time()
        R2_val, w, w_path, K, P, period = run_training(args, K=K, P=P)
        end_time = time.time()
        if R2_val > best_R2:
            best_R2 = R2_val 
            best_w = w
            best_path = w_path
            best_config = {'P': P,
                           'K': K,
                           'train time': end_time - start_time,
                           'w_path': w_path}
    print(f"\nBest validation R²: {best_R2:.4f}, used training time = {(end_time - start_time):.4f}")
    # === Save best config ===
    output_dir = f"{args.output_dir}/{args.prefix}"
    best_config_path = os.path.join(output_dir, "best_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=4)
    print(f"Best config saved to: {best_config_path}")
    return best_R2, best_w, best_path, best_config['K'], best_config['P'], period


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
                        default='/mnt/home/thuang/playground/velocity_prediction/LLS',
                        #default='/mnt/home/thuang/ceph/playground/datasets/point_clouds/invPwrFeat',
                         help='save path')
    #parser.add_argument('--num_clouds', default=3072, type=int, help="number of point clouds")
    parser.add_argument('--K', default=25, type=int, help="number of x frequencies")
    parser.add_argument('--P', default=3, type=int, help="number of powers of inv_dist")

    parser.add_argument('--train_clouds', default=2048, type=int, help="number of training point clouds")
    parser.add_argument('--val_clouds', default=512, type=int, help="number of VAL point clouds")
    parser.add_argument('--test_clouds', default=512, type=int, help="number of test point clouds")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--search', action='store_true', help='Enable hyperparameter search, K:[10,20,25], P:[2,3,4]')
    group.add_argument('--eval_test', action='store_true', help='Eval on test set only')
    parser.add_argument('--test_sample_idx', type=int, default=None, help='if specified, only test on \
                        test_sample test clouds, and save the predictions')
    parser.add_argument('--eval_redshift', action='store_true', help='Train/Eval on test set - redshift position on z axis (OOD)')
    parser.add_argument('--shrink_factor', default=1.0, type=float, help="shrink redshift axis distance")
    parser.add_argument('--eval_per_axis', action='store_true', help='eval per-axis R2')

    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dir_train = f"{args.data_dir}/{args.h5_path_train}"
    filename = dir_train.split('/')[-1]             # Get the file name
    prefix = filename.split('_')[0]            # Extract 'CAMELS-SAM'
    args.prefix = prefix
    dir_test = f"{args.data_dir}/{args.h5_path_test}"
    K = args.K
    P = args.P
    period_dict = {'Quijote': 1000, 'CAMELS-SAM': 100, 'CAMELS-TNG': 25, 'fiducial': 1000}
    period = period_dict[prefix]
    shrink = args.shrink_factor  #args.eval_redshift else 1.0

    if args.test_sample_idx is not None: #eval test sample idx = 0
        output_dir = f"{args.output_dir}/{prefix}/_K={K}_P={P}_shrink={args.shrink_factor}"
        w = torch.load(f"{output_dir}/weight.pt")
        x,y,z,vx,vy,vz = load_point_cloud_h5(dir_test, args.test_sample_idx, args.data_name) 
        target = torch.stack([vx,vy,vz], dim=-1).cpu()        #adjust for redshift on eval
        if args.eval_redshift: #use redshift positions instead
            z = pos_redshift_space(z, vz, period)
        Fx = compute_invPwrLaw_features(x,y,z,K,P, period, z_shrink=shrink)  #(d, n)
        Fy = compute_invPwrLaw_features(y,z,x,K,P, period, y_shrink=shrink) 
        Fz = compute_invPwrLaw_features(z,x,y,K,P, period, x_shrink=shrink) 
        #feat_time = time.time()
        F_xyz =  torch.stack([Fx, Fy, Fz], dim=0)  # shape: (3, d, n)
        v_pred = torch.einsum('d,cdn->cn', w.squeeze(), F_xyz).T.detach().cpu()
        mse = MSE_loss(v_pred, target)
        var = variance(target)
        R2 = 1 - mse/var
        test_pred_path = os.path.join(output_dir, f"test_pred_{args.test_sample_idx}.pt")
        #torch.save([v_pred,target], test_pred_path)
        print(f"Saved test velocity predictions to: {test_pred_path}, R2_test = {R2:.4f}")
    
    else:
        if args.search:
            best_R2, w, w_path, K, P, period = hyperparameter_search(args)
        elif args.eval_test:
            output_dir = f"{args.output_dir}/{prefix}/_K={K}_P={P}_shrink={args.shrink_factor}"
            w = torch.load(f"{output_dir}/weight.pt")
        else:
            R2_val, w, w_path, K, P, period = run_training(args, args.K, args.P)
        
        ##EVAL on test set
        dir_test = f"{args.data_dir}/{args.h5_path_test}"
        test_size = get_h5_group_size(dir_test, args.data_name)
        idxTest = list(range(0, min(test_size, args.test_clouds))) #list(range(26202, 32752))#
        print(f"test on {len(idxTest)} clouds...(redshift={args.eval_redshift})")
        R2_test, R2x, R2y, R2z = eval_pretrained_weight(w, idxTest, dir_test, K, P, period, 
                                         args.data_name, device, args.eval_redshift, 
                                         shrink, args.eval_per_axis)

        ## Save to file
        results_dict = {
            'dir_test': dir_test,
            'test_size': test_size,
            'K': K,
            'P': P,
            'R2_test': R2_test,
            'redshift': args.eval_redshift,
            'shrink_factor': shrink,
        }
        if args.eval_per_axis:
            results_dict['test_R2x'] = R2x
            results_dict['test_R2y'] = R2y
            results_dict['test_R2z'] = R2z

        test_result_path = os.path.join(args.output_dir, f"{prefix}_test_R2_result.json")
        data = []
        if os.path.exists(test_result_path):
            try:
                data = json.load(open(test_result_path))
                if isinstance(data, dict):
                    data = [data]
            except json.JSONDecodeError:
                pass
        data.append(results_dict)
        json.dump(data, open(test_result_path, 'w'), indent=4)
        print(f"R2_test = {R2_test:.4f}, appended test evaluation results to: {test_result_path}")