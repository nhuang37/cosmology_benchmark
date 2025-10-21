import numpy as np
import h5py
from h5py import File
import Pk_library as PKL
import MAS_library as MASL
from scipy.optimize import minimize
import numpy as np
from tqdm import tqdm
import argparse
from utils.get_redshift_pos import pos_redshift_space
import json
import os 
import pandas as pd
from pathlib import Path
import csv

# Try to import MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mpi_enabled = True
except ImportError:
    print("mpi4py not available. Running in single-process mode.")
    comm = None
    rank = 0
    size = 1
    mpi_enabled = False

MAS = 'TSC'
verbose = False
threads = 1

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)  # residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # total sum of squares
    
    return 1 - ss_res / ss_tot

def r2_axes_for_params(bc, f, grid, pos, vel, BoxSize, use_biased_LT):
    '''
    Use the best grid
    '''
    vx, vy, vz = pred_v(bc, f, grid, pos, vel, BoxSize, use_biased_LT)
    r2x = r2_score(vel[:, 0], vx)
    r2y = r2_score(vel[:, 1], vy)
    r2z = r2_score(vel[:, 2], vz)
    return r2x, r2y, r2z


def pred_v(bc, f, grid, pos, vel, BoxSize,
           use_biased_LT=False, z=0.0,  H_kms_per_mpch=100.0):
    b,c=bc
    # f = Om**0.55
    kx = np.fft.fftfreq(grid, BoxSize/grid) * 2 * np.pi
    ky = np.fft.fftfreq(grid, BoxSize/grid) * 2 * np.pi
    kz = np.fft.rfftfreq(grid, BoxSize/grid) * 2 * np.pi

    # Create 3D grids
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')

    kmag = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)
    kmag[0,0,0] = 1    # does nothing... avoid divide by 0
    
    delta_h = np.zeros((grid,grid,grid), dtype=np.float32)
    MASL.MA(pos, delta_h, BoxSize, MAS, verbose=verbose)
    delta_h /= np.mean(delta_h, dtype=np.float64);  delta_h -= 1.0
    delta_h = delta_h.astype(np.float64)   # FIXME?
    delta_h_k = np.fft.rfftn(delta_h, norm='ortho')
    # Kaiser inversion: δ_h^s = (b + f μ^2) δ_m  -> δ_m = δ_h^s / (b + f μ^2)
    mu = kz3d/kmag
    denom = (b + c * f * mu**2)
    delta_m_k = delta_h_k / denom

    # --- amplitude choice ---
    if use_biased_LT:
        a = 1.0 / (1.0 + z)
        pref = a * H_kms_per_mpch * f   # physical amplitude
    else:
        pref = 1.0                      # old baseline amplitude

    vel_h_x_k = 1j * pref / kmag**2 * kx3d * delta_m_k    # H f
    vel_h_y_k = 1j * pref / kmag**2 * ky3d * delta_m_k
    vel_h_z_k = 1j * pref / kmag**2 * kz3d * delta_m_k

    vel_h_x = np.fft.irfftn(vel_h_x_k, norm='ortho').astype(np.float32)
    vel_h_y = np.fft.irfftn(vel_h_y_k, norm='ortho').astype(np.float32)
    vel_h_z = np.fft.irfftn(vel_h_z_k, norm='ortho').astype(np.float32)
    vel_linear_x = np.zeros(pos.shape[0], dtype=np.float32)
    vel_linear_y = np.zeros(pos.shape[0], dtype=np.float32)
    vel_linear_z = np.zeros(pos.shape[0], dtype=np.float32)
    MASL.CIC_interp(vel_h_x, BoxSize, pos, vel_linear_x)
    MASL.CIC_interp(vel_h_y, BoxSize, pos, vel_linear_y)
    MASL.CIC_interp(vel_h_z, BoxSize, pos, vel_linear_z)

    return vel_linear_x, vel_linear_y, vel_linear_z

def mean_r2_pred_v(b, f, grid, pos, vel, BoxSize, use_biased_LT=False):
    vel_linear_x, vel_linear_y, vel_linear_z = pred_v(b, f, grid, pos, vel, BoxSize, use_biased_LT)
    vel_linear = np.vstack((vel_linear_x, vel_linear_y, vel_linear_z)).T
    r2 = r2_score(vel, vel_linear)
    return r2

def loss(b, f, grid, pos, vel, BoxSize, use_biased_LT=False):
    return 1 - mean_r2_pred_v(b, f, grid, pos, vel, BoxSize, use_biased_LT)

def read_pos_vel(fname, one_seed=10, redshift=False, BoxSize=None):   # one_seed = -1 for all seeds. else give the seed you want
    if "Quijote_BSQ" in fname:
        grp_key = "BSQ"
    elif "Quijote_Fiducial" in fname:
        grp_key = "fiducial"
    elif "CAMELS" in fname:
        grp_key = "LH"
    else:
        raise ValueError("`fname` is not recognized.")

    with File(fname, 'r') as f:
        Om = f["params/Omega_m"][...]
        sigma8 = f["params/sigma_8"][...]
        if redshift:
            if grp_key == 'BSQ':
                h = f["params/h"][...]

        pos, vel = [], []

        keys = list(f[grp_key].keys())
        if one_seed != -1:
            keys = [k for k in keys if k.endswith(f"_{one_seed:d}")]
        for key in keys: # tqdm(keys):
            grp = f[f"{grp_key}/{key}"]
            #TODO: adjust for z_redshift
            if one_seed != -1:
                pos = np.vstack([grp[p] for p in "XYZ"]).T.astype(np.float32)
                vel = np.vstack([grp[f"V{p}"] for p in "XYZ"]).T.astype(np.float64)
                if redshift: #use redshift positions instead
                    print("computing redshift positions z_r")
                    cosmo_h = h[one_seed] if grp_key == 'BSQ' else 0.6711
                    pos[:,-1] = pos_redshift_space(pos[:,-1], vel[:,-1], BoxSize, cosmo_h)
            else:
                pos.append(np.vstack([grp[p] for p in "XYZ"]).T)
                vel.append(np.vstack([grp[f"V{p}"] for p in "XYZ"]).T)
                print("no redshift corrections supported!")

    if one_seed == -1:
        pos = np.asarray(pos).astype(np.float32)
        vel = np.asarray(vel).astype(np.float64)
    else:
        Om = Om[one_seed]
        sigma8 = sigma8[one_seed]
    return pos, vel, Om, sigma8

def append_results(R2_mean, R2x_mean, R2y_mean, R2z_mean, args):
    """
    Safe single-writer append for MPI jobs.
    - Creates args.output_dir if missing
    - Writes CSV header only once
    - Avoids read/concat/write races
    """
    # Gate on rank 0 (single writer)
    is_writer = (comm is None) or (rank == 0)

    # Ensure the directory exists (only writer creates it)
    if is_writer:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not is_writer:
        return  # non-writers do nothing

    row = [
        args.suite,
        f"{float(R2_mean):.6f}",
        f"{float(R2x_mean):.6f}",
        f"{float(R2y_mean):.6f}",
        f"{float(R2z_mean):.6f}",
        int(args.eval_redshift),
        int(args.bias_LT),
    ]
    out_path = os.path.join(args.output_dir, "result.csv")
    header_needed = not os.path.exists(out_path)

    # Append atomically from a single writer; add header on first write.
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["suite", "testR2", "testR2_x", "testR2_y", "testR2_z", "redshift", "biasLT"])
        w.writerow(row)

    print(f"[rank {rank}] appended result to {out_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_redshift', action='store_true', help='Train/Eval on test set - redshift position on z axis (OOD)')
    parser.add_argument('--bias_LT', action='store_true', help='whether to add f=om**0.55 for biased linear theory')
    parser.add_argument('--suite', default='Quijote', type=str, choices=['Quijote', 'CAMELS-SAM', 'CAMELS'])
    parser.add_argument('--output_dir', #default='/mnt/home/rstiskalek/ceph/CAMELS-SAM/LH_rockstar_99.hdf5',
                        default='/mnt/home/thuang/playground/velocity_prediction/linear_theory_H0')
    args = parser.parse_args()
    print(args)
    data_dir = "/mnt/home/rstiskalek/ceph/graps4science"
    if args.suite == 'Quijote':
        fname = f"{data_dir}/Quijote_BSQ_rockstar_10_top5000_test.hdf5"
        BoxSize = 1000
        N = 6550
    elif args.suite == 'CAMELS-SAM':
        fname = f"{data_dir}/CAMELS-SAM_LH_gal_99_top5000_test.hdf5"
        BoxSize = 100
        N = 196
    elif args.suite == 'CAMELS':
        fname = f"{data_dir}/CAMELS-TNG_galaxy_90_ALL_test.hdf5"
        BoxSize = 25
        N = 200


    # ---- Main Loop ----
    if rank == 0:
        print(f'\n{args.suite}')

    seeds = list(range(N))
    seeds_rank = seeds[rank::size]

    R2s_local = []
    R2xs_local, R2ys_local, R2zs_local = [], [], []
    for one_seed in seeds_rank:
        if rank == 0:
            print(f"Rank {rank} processing seed {one_seed}", flush=True)
        pos, vel, Om, sigma8 = read_pos_vel(fname, one_seed=one_seed, 
                                            redshift=args.eval_redshift, BoxSize=BoxSize)
        f = Om**0.55 if args.bias_LT else 0
        R2_best = -100
        b_best, c_best, grid_best = None, None, None
        for grid in range(10, 100, 10):
            res = minimize(loss, x0=[0.02, 0.02], args=(f, grid, pos, vel, BoxSize, args.bias_LT), method='L-BFGS-B')
            R2 = -res.fun + 1
            if R2 > R2_best:
                R2_best = R2
                b_best = float(np.atleast_1d(res.x)[0])
                c_best = float(np.atleast_1d(res.x)[1])
                bc_best = [b_best, c_best]
                grid_best = grid
        #print(f"sim={one_seed}, R2={R2_best:.4f}")
        r2x, r2y, r2z = r2_axes_for_params(bc_best, f, grid_best, pos, vel, BoxSize, args.bias_LT)
        R2s_local.append(R2_best)
        R2xs_local.append(r2x)
        R2ys_local.append(r2y)
        R2zs_local.append(r2z)

    # Gather results
    if mpi_enabled:
        R2s_all = comm.gather(R2s_local, root=0)
        R2xs_all  = comm.gather(R2xs_local,  root=0)
        R2ys_all  = comm.gather(R2ys_local,  root=0)
        R2zs_all  = comm.gather(R2zs_local,  root=0)
        if rank == 0:
            R2s = [val for sublist in R2s_all for val in sublist]
            R2_mean = np.mean(R2s)
            R2x_mean  = np.mean([val for sublist in R2xs_all for val in sublist])
            R2y_mean  = np.mean([val for sublist in R2ys_all for val in sublist])
            R2z_mean  = np.mean([val for sublist in R2zs_all for val in sublist])
            print("R2_mean:", R2_mean, "R2x_mean:", R2x_mean, "R2y_mean:", R2y_mean, "R2z_mean:", R2z_mean)
            append_results(R2_mean, R2x_mean, R2y_mean, R2z_mean, args)
    else:
        R2_mean = np.mean(np.array(R2s_local))
        R2x_mean  = np.mean(np.array(R2xs_local))
        R2y_mean  = np.mean(np.array(R2ys_local))
        R2z_mean  = np.mean(np.array(R2zs_local))
        print("R2_mean:", R2_mean)
        append_results(R2_mean, args)

    

