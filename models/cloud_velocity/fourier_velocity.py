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

torch.set_printoptions(precision=4,sci_mode=False,linewidth=150)
torch.set_default_dtype(torch.float64)

def load_point_cloud_h5(h5_path, idx, data_name='BSQ', device="cuda"):
    # period_dict = {'BSQ': 1000, 'CAMELS-SAM': 100, 'CAMELS': 25}
    # period = period_dict[data_name]
    with h5py.File(h5_path, 'r') as f:
        group = f[data_name]
        #labels = f["params"]
        g = group[f"{data_name}_{idx}"]
        x = torch.tensor(g['X'][:], dtype=torch.float64).to(device) #/ period
        y = torch.tensor(g['Y'][:], dtype=torch.float64).to(device) #/ period
        z = torch.tensor(g['Z'][:], dtype=torch.float64).to(device) #/ period  

        # Construct labels
        vx = torch.tensor(g['VX'][:], dtype=torch.float64).to(device) #/ period
        vy = torch.tensor(g['VY'][:], dtype=torch.float64).to(device) #/ period
        vz = torch.tensor(g['VZ'][:], dtype=torch.float64).to(device) #/ period
        
    return x,y,z,vx,vy,vz


def sum_over_j_term(Cx, Cy, Cz, k):
    """ 
    For a given k (interger), with input matrices Cx, Cy, Cz in shape(d_, n)
    first extract the k-th row Cx[k], reshaped it to (1,n) and obtain row vector c
    then broacast elementwise product c with rows in Cy, yielding a matrix of shape (d_,n)
    finally matrix multiplication with Cz^T, 
    return shape (d_, d_)
    """
    c = Cx[k]
    elem_prod = c * Cy #(d_, n)
    return elem_prod @ Cz.T #(d_, d_)


def compute_fourier_features(x, y, z, K, L, period=1000, normalize=True):
    """
    Compute Fourier features from 3D coordinates using trigonometric basis functions.
    P = period
    assuming symmetries of (y,z) coordinates w.r.t x
    $vx_i = \sum_{j=1}^n \sum_{k=1}^K \sum_{l=0}^L \sum_{m=0}^l A_{klm}
              [\sin(\frac{2pi}{P} k(x_i - x_j)) \cos(\frac{2pi}{P} l(y_i - y_j)) \cos(\frac{2pi}{P} m(z_i - z_j)) 
               + \sin(\frac{2pi}{P} k(x_i - x_j)) \cos(\frac{2pi}{P} m(y_i - y_j)) \cos(\frac{2pi}{P} l(z_i - z_j)) ]$
    The code implement the equation by expanding and collecting the sum_over_j_terms, 
    for an order of n = 5000 speedup.
    Args:
        x, y, z: Tensors of shape (n,) – the input coordinates
        K: number of frequency components in x-direction (starts from 1)
        L: number of frequency components in y- and z-directions (starts from 0)
        period: spatial period for frequency scaling
    Returns:
        F: Tensor of shape (d, n), where d = K * (L+1) * (L+2) // 2
    """

    n = x.shape[0]
    device = x.device
    xp = x * (2 * math.pi / period)  # (n,)
    yp = y * (2 * math.pi / period)
    zp = z * (2 * math.pi / period)

    # Frequency indices
    k_vals = torch.arange(1, K + 1, device=device).view(-1, 1)  # (K, 1)
    l_vals = torch.arange(0, L + 1, device=device).view(-1, 1)  # (L+1, 1)

    # Sine and cosine bases (shape: (K or L+1, n))
    Sx = torch.sin(k_vals * xp)  # (K, n)
    Cx = torch.cos(k_vals * xp)  # (K, n)
    Sy = torch.sin(l_vals * yp)  # (L+1, n)
    Cy = torch.cos(l_vals * yp)
    Sz = torch.sin(l_vals * zp)
    Cz = torch.cos(l_vals * zp)

    # Output dimension
    d = K * (L + 1) * (L + 2) // 2
    Feat = torch.zeros((d, n), device=device)

    idx = 0
    for k in range(len(k_vals)):
        ccc = sum_over_j_term(Cx, Cy, Cz, k)
        csc = sum_over_j_term(Cx, Sy, Cz, k)
        ccs = sum_over_j_term(Cx, Cy, Sz, k)
        css = sum_over_j_term(Cx, Sy, Sz, k)
        scc = sum_over_j_term(Sx, Cy, Cz, k)
        ssc = sum_over_j_term(Sx, Sy, Cz, k)
        scs = sum_over_j_term(Sx, Cy, Sz, k)
        sss = sum_over_j_term(Sx, Sy, Sz, k)
        for l in range(L + 1): #1:L+1 in matlab (L+1 number, inclusive)
            for m in range(l + 1): #wanna stop at l -> 1:l in matlab (l number)
                term1 = (
                    Cx[k] * (
                        Cy[l] * (Cz[m] * scc[l, m] + Sz[m] * ssc[l, m]) +
                        Sy[l] * (Cz[m] * scs[l, m] + Sz[m] * sss[l, m]) +
                        Cy[m] * (Cz[l] * scc[m, l] + Sz[l] * ssc[m, l]) +
                        Sy[m] * (Cz[l] * scs[m, l] + Sz[l] * sss[m, l])
                    )
                )
                term2 = (
                    Sx[k] * (
                        Cy[l] * (Cz[m] * ccc[l, m] + Sz[m] * ccs[l, m]) +
                        Sy[l] * (Cz[m] * csc[l, m] + Sz[m] * css[l, m]) +
                        Cy[m] * (Cz[l] * ccc[m, l] + Sz[l] * ccs[m, l]) +
                        Sy[m] * (Cz[l] * csc[m, l] + Sz[l] * css[m, l])
                    )
                )
                Feat[idx] = term1 - term2
                idx += 1
    #return Feat
    if normalize:
        return Feat/period 
    else:
        return Feat

def online_least_squares(w0, X, y, eps=1e-8,  max_step=1.0): #w0 shape (1,n); X shape (d,n); y shape (n);
    sqrNorm = (X * X).sum(dim=0) #shape (n)
    sqrNorm = torch.clamp(sqrNorm, min=eps)  # avoid division by zero
    residuals = y - w0 @ X             # shape: (n)
    steps = residuals / sqrNorm    # shape: (n,)
    steps = torch.clamp(steps, min=-max_step, max=max_step)  # prevent exploding updates
    grad = X * steps
    w = w0 + grad.sum(dim=1)       # shape: (d,)
    return w


def MSE_loss(ypred, y):
    return torch.mean((ypred - y)**2)

def variance(y):
    #compute mean vector (per feat), then average over variance per element
    mean = y.mean(axis=0)
    return torch.mean((y - mean)**2) 

def compute_MSE_all(wa, Fx, Fy, Fz, v_target):
    vx_pred = wa @ Fx  #double-check; (1,d) (d,n) -> (1,n)
    vy_pred = wa @ Fy
    vz_pred = wa @ Fz 
    v_pred = torch.stack([vx_pred, vy_pred, vz_pred], dim=-1)
    mse = MSE_loss(v_pred, v_target) #torch.mean((v_pred - v_target)**2)
    return mse

def eval(output_dir, data_dir, idxTest, wa, 
         data_name, device="cuda",
         K=20, L=20):
    mse_all = []
    var_all = []
    for idx in idxTest:
        x,y,z,vx,vy,vz = load_point_cloud_h5(data_dir, idx, data_name) 
        v_target = torch.stack([vx, vy, vz], dim=-1)
        var = variance(v_target) #torch.mean((v_target - v_target.mean())**2)
        var_all.append(var) #for computing R2 later

        feat_file = f"{output_dir}/{idx}.pt"
        if os.path.isfile(feat_file):
            list_of_ff = torch.load(feat_file, map_location=device)
            Fx = list_of_ff[0]
            Fy = list_of_ff[1]
            Fz = list_of_ff[2]
        else:
            Fx = compute_fourier_features(x,y,z,K,L, period).to(device)
            Fy = compute_fourier_features(y,z,x,K,L, period).to(device)
            Fz = compute_fourier_features(z,x,y,K,L, period).to(device)
            os.makedirs(output_dir, exist_ok=True)
            torch.save([Fx,Fy,Fz], feat_file)
        mse = compute_MSE_all(wa, Fx, Fy, Fz, v_target)
        mse_all.append(mse)

    final_mse = sum(mse_all)/len(mse_all)
    final_var = sum(var_all)/len(var_all)
    R2 = 1 - final_mse/final_var
    return final_mse, R2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', #default='/mnt/home/rstiskalek/ceph/CAMELS-SAM/LH_rockstar_99.hdf5',
                        default='/mnt/home/rstiskalek/ceph/graps4science/CAMELS-SAM_LH_gal_99_top5000.hdf5',
                         help='h5 path to load the data')
    parser.add_argument('--data_name', default='LH', type=str,
                         help='data group name in the h5 file') #TODO: sync across BSQ and LH? 
    parser.add_argument('--output_dir', #default='/mnt/home/rstiskalek/ceph/CAMELS-SAM/LH_rockstar_99.hdf5',
                        default='/mnt/home/thuang/ceph/playground/datasets/point_clouds/fourier_features',
                         help='save path')
    parser.add_argument('--num_clouds', default=125, type=int, help="number of point clouds")
    parser.add_argument('--K', default=20, type=int, help="number of x frequencies")
    parser.add_argument('--L', default=20, type=int, help="number of y/z frequencies")

    args = parser.parse_args()

    #Data split
    dir = args.h5_path
    n = args.num_clouds

    filename = dir.split('/')[-1]             # Get the file name
    prefix = filename.split('_')[0]            # Extract 'CAMELS-SAM'

    idxTrain = list(range(0, int(n*0.8))) #list(range(0,2048+512))
    idxTest = list(range(int(n*0.8), n))#list(range(2048+512, 2048+512+512))
    #print(idxTrain[-10:], idxTest[:10])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    period_dict = {'Quijote': 1000, 'CAMELS-SAM': 100, 'CAMELS-TNG': 25000, 'fiducial': 1000}
    period = period_dict[prefix]
    print(prefix, period)  # Output: CAMELS-SAM
    #Fourier Feature Order
    K = args.K
    L = args.L
    d = int(K*(L+1)*(L+2)/2)
    output_dir = f"{args.output_dir}/{prefix}_K={K}_L={L}"

    #Weights
    wt = torch.zeros(d).to(device) #last update
    wa = torch.zeros(d).to(device) #TIME-AVERAGED

    #Main Trianing Loop
    t0 = time.time()
    print(f"Training...")
    ff_time_sum, ls_time_sum = 0, 0
    for t in idxTrain:
        x,y,z,vx,vy,vz = load_point_cloud_h5(dir, t, args.data_name) 
        start_time = time.time()
        feat_file = f"{output_dir}/{t}.pt"
        # if os.path.isfile(feat_file):
        # # if args.load_features:
        #     if t % 100 == 0:
        #         print(f"loading {t}")
        #     list_of_ff = torch.load(feat_file, map_location=device)
        #     Fx = list_of_ff[0]
        #     Fy = list_of_ff[1]
        #     Fz = list_of_ff[2]
        # else:
        Fx = compute_fourier_features(x,y,z,K,L, period).to(device)
        Fy = compute_fourier_features(y,z,x,K,L, period).to(device)
        Fz = compute_fourier_features(z,x,y,K,L, period).to(device)
        os.makedirs(output_dir, exist_ok=True)
        torch.save([Fx,Fy,Fz], feat_file)
        ff_time = time.time()
        ff_time_sum += ff_time - start_time

        wt = online_least_squares(wt,Fx,vx)
        wt = online_least_squares(wt,Fy,vy)
        wt = online_least_squares(wt,Fz,vz)
        wa = (1-1/(t+1))*wa + (1/(t+1))*wt
        ls_time = time.time()
        ls_time_sum += ls_time - ff_time
        if t % 5 == 0:
            print(wa[:10])
            print(f"processed {t+1} clouds, ff_time={ff_time_sum:.4f}, ls_time={ls_time_sum:.4f}")

    #EVAL
    print(f"Evaluating....")
    #mse_train, R2_train = eval(output_dir, dir, idxTrain, wa, K=K, L=L)
    mse_test, R2_test = eval(output_dir, dir, idxTest, wa, args.data_name, K=K,L=L)
    t1 = time.time()
    #print(f"MSE_train={mse_train.item():.4f}, R2_train={R2_train.item():.4f}") 
    print(f"MSE_test={mse_test.item():.4f}, R2_test={R2_test.item():.4f}")
    torch.save(wa, f"{output_dir}/wa_{n}.pt")

    #save results
    plt.plot(wa.flatten().detach().cpu())
    title = f"R2_test={R2_test.item():.4f}, n={n} clouds"
    #plt.legend()
    plt.title(title)
    plt.savefig(f"{output_dir}/wa_{n}_results.png", dpi=150)



#=======TESTING FOURIER FEATURES============
# K = 3
# L = 3
# n = 5000
# period = 1000
# x = torch.linspace(0,period,steps=n)
# y = torch.linspace(0,period,steps=n)
# z = torch.linspace(0,period,steps=n)
# print(x.shape)
# print(x[0:20])
# Feat = compute_fourier_features(x,y,z, K, L, period=period)*period
# print(Feat.shape)
# print(Feat[0:30,0:10])
#==========================================
