import numpy as np
import h5py
from h5py import File
import Pk_library as PKL
import MAS_library as MASL
from scipy.optimize import minimize
import numpy as np
from tqdm import tqdm

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

def pred_v(b, grid, pos, vel, BoxSize):
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
    delta_m = delta_h / b
    delta_m_k = np.fft.rfftn(delta_m, norm='ortho')
    vel_h_x_k = 1j / kmag**2 * kx3d * delta_m_k    # H f
    vel_h_y_k = 1j / kmag**2 * ky3d * delta_m_k
    vel_h_z_k = 1j / kmag**2 * kz3d * delta_m_k
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

def mean_r2_pred_v(b, grid, pos, vel, BoxSize):
    vel_linear_x, vel_linear_y, vel_linear_z = pred_v(b, grid, pos, vel, BoxSize)
    vel_linear = np.vstack((vel_linear_x, vel_linear_y, vel_linear_z)).T
    r2 = r2_score(vel, vel_linear)
    return r2

def loss(b, grid, pos, vel, BoxSize):
    return 1 - mean_r2_pred_v(b, grid, pos, vel, BoxSize)

def read_pos_vel(fname, one_seed=10):   # one_seed = -1 for all seeds. else give the seed you want
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

        pos, vel = [], []

        keys = list(f[grp_key].keys())
        if one_seed != -1:
            keys = [k for k in keys if k.endswith(f"_{one_seed:d}")]
        for key in keys: # tqdm(keys):
            grp = f[f"{grp_key}/{key}"]
            if one_seed != -1:
                pos = np.vstack([grp[p] for p in "XYZ"]).T.astype(np.float32)
                vel = np.vstack([grp[f"V{p}"] for p in "XYZ"]).T.astype(np.float64)
            else:
                pos.append(np.vstack([grp[p] for p in "XYZ"]).T)
                vel.append(np.vstack([grp[f"V{p}"] for p in "XYZ"]).T)

    if one_seed == -1:
        pos = np.asarray(pos).astype(np.float32)
        vel = np.asarray(vel).astype(np.float64)
    else:
        Om = Om[one_seed]
        sigma8 = sigma8[one_seed]
    return pos, vel, Om, sigma8
    
# ---- Configuration on ALL 3 datasets ----
# fnames = [
#     "/mnt/home/rstiskalek/ceph/graps4science/Quijote_BSQ_rockstar_10_top5000_test.hdf5",
#     "/mnt/home/rstiskalek/ceph/graps4science/CAMELS-SAM_LH_gal_99_top5000_test.hdf5",
#     "/mnt/home/rstiskalek/ceph/graps4science/CAMELS-TNG_galaxy_90_ALL_test.hdf5"
# ]
# BoxSizes = [1000., 100., 25.]
# labels = ["Quijote", "CAMELS-SAM", "CAMELS"]
# Nsims = [6550, 196, 200]

# Config on CAMELS only for demo
fnames = [
    "/mnt/home/rstiskalek/ceph/graps4science/CAMELS-TNG_galaxy_90_ALL_test.hdf5"
]
BoxSizes = [25.]
labels = ["CAMELS"]
Nsims = [200]

# ---- Main Loop ----
for i, fname in enumerate(fnames):
    if rank == 0:
        print(f'\n{labels[i]}')
    BoxSize = BoxSizes[i]
    N = Nsims[i]

    seeds = list(range(N))
    seeds_rank = seeds[rank::size]

    R2s_local = []
    for one_seed in seeds_rank:
        if rank == 0:
            print(f"Rank {rank} processing seed {one_seed}", flush=True)
        pos, vel, Om, sigma8 = read_pos_vel(fname, one_seed=one_seed)
        R2_best = -100
        for grid in range(10, 100, 10):
            res = minimize(loss, x0=0.02, args=(grid, pos, vel, BoxSize), method='L-BFGS-B')
            R2 = -res.fun + 1
            if R2 > R2_best:
                R2_best = R2
        R2s_local.append(R2_best)

    # Gather results
    if mpi_enabled:
        R2s_all = comm.gather(R2s_local, root=0)
        if rank == 0:
            R2s = [val for sublist in R2s_all for val in sublist]
            R2_mean = np.mean(R2s)
            print("R2_mean:", R2_mean)
    else:
        R2s = R2s_local
        R2_mean = np.mean(R2s)
        print("R2_mean:", R2_mean)
