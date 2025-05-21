from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
import argparse

def get_2pcf_periodic(x, y, z, boxsize, nrand_mult, rbins, nthreads=1,
                      seed=42):
    """
    Compute the two-point correlation function (2PCF) in a periodic box
    using Corrfunc.

    Parameters
    ----------
    x, y, z : ndarray
        Cartesian coordinates of the data points.
    boxsize : float
        Size of the periodic box.
    nrand_mult : int
        Number of random points = nrand_mult × number of data points.
    rbins : ndarray
        Radial bin edge.
    nthreads : int
        Number of OpenMP threads to use.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    xi : ndarray
        The 2PCF ξ(r) evaluated at the center of each bin.
    """
    try:
        from Corrfunc.theory.DD import DD
        from Corrfunc.utils import convert_3d_counts_to_cf
    except ImportError as e:
        raise ImportError("Corrfunc is not installed.") from e

    if nrand_mult < 10:
        warn("Small number of randoms may bias the result.", UserWarning)

    rng = np.random.default_rng(seed)
    n_data = len(x)
    n_rand = nrand_mult * n_data

    rand_x = rng.uniform(0, boxsize, n_rand).astype(x.dtype)
    rand_y = rng.uniform(0, boxsize, n_rand).astype(x.dtype)
    rand_z = rng.uniform(0, boxsize, n_rand).astype(x.dtype)

    DD_counts = DD(1, nthreads, rbins, x, y, z, boxsize=boxsize)
    DR_counts = DD(0, nthreads, rbins, x, y, z,
                   X2=rand_x, Y2=rand_y, Z2=rand_z, boxsize=boxsize)
    RR_counts = DD(1, nthreads, rbins, rand_x, rand_y, rand_z,
                   boxsize=boxsize)

    return convert_3d_counts_to_cf(
        n_data, n_data, n_rand, n_rand, DD_counts, DR_counts, DR_counts,
        RR_counts)

# Recommended radial binning (rbins) and box sizes for datasets used in the paper:
#
# - Quijote_BSQ:
#     Used in the paper.
#     Bins: log-spaced from 2 to 80 Mpc/h
#     rbins = np.logspace(np.log10(2.), np.log10(80), 25)
#
# - CAMELS-SAM_LG_gal_top5000:
#     Used in the paper.
#     Box size: 100 Mpc/h
#     Bins: log-spaced from 1 to 40 Mpc/h
#     rbins = np.logspace(0, np.log10(40), 20)
#
# - CAMELS-TNG_galaxy_90_ALL:
#     Used in the paper.
#     Box size: 25 Mpc/h
#     Bins: log-spaced from 0.1 to 12 Mpc/h
#     rbins = np.logspace(np.log10(0.1), np.log10(12), 20)


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
                        default='/mnt/home/thuang/playground/param_predictions',
                        #default='/mnt/home/thuang/ceph/playground/datasets/point_clouds/invPwrFeat',
                         help='save path')
    parser.add_argument('--start_idx', default=0, type=int, help="start cloud ix")
    parser.add_argument('--end_idx', default=10, type=int, help="end cloud idx")
    parser.add_argument('--make_plot', action='store_true', help='plot 2PCF')

    args = parser.parse_args()
    dir_train = f"{args.data_dir}/{args.h5_path_train}"
    dir_val = f"{args.data_dir}/{args.h5_path_val}"
    dir_test = f"{args.data_dir}/{args.h5_path_test}"
    filename = dir_train.split('/')[-1]             # Get the file name
    prefix = filename.split('_')[0]            # Extract 'CAMELS-SAM'

   
    # === Parameters ===
    if prefix == 'Quijote':
        rbins = np.logspace(np.log10(2.), np.log10(80), 25)
        n_train, n_val, n_test, boxsize = 19651, 6551, 6550, 1000
        data_name = 'BSQ'

    elif prefix == 'CAMELS-SAM':
        rbins = np.logspace(0, np.log10(40), 20)
        n_train, n_val, n_test, boxsize = 600, 204, 196, 100
        data_name = 'LH'
    elif prefix == 'CAMELS':
        rbins = np.logspace(np.log10(0.1), np.log10(12), 20)
        n_train, n_val, n_test, boxsize = 600, 200, 200, 25
        data_name = 'LH'

    nrand_mult = 100
    make_plot = args.make_plot
    #index = 0  # Catalog index
    hdf5_path = dir_train #"/mnt/home/rstiskalek/ceph/graps4science/Quijote_BSQ_rockstar_10_ALL_train.hdf5"  # noqa
    results = []

    # === Load data ===
    with File(hdf5_path, "r") as f:
        for index in range(args.start_idx, args.end_idx):
            group = f[f"{data_name}/{data_name}_{index}"]
            x = group["X"][...]
            y = group["Y"][...]
            z = group["Z"][...]

            # === Compute 2PCF ===
            xi = get_2pcf_periodic(x, y, z, boxsize, nrand_mult, rbins)
            results.append(xi)
            print(xi.shape)
            print(f"2PCF ξ(r):\n{xi}")

            # === Plot ===
            if make_plot:
                r_centers = 0.5 * (rbins[:-1] + rbins[1:])

                plt.figure()
                plt.plot(r_centers, xi)
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel(r"$r\ [\mathrm{Mpc}/h]$")
                plt.ylabel(r"$\xi(r)$")
                plt.title(f"2PCF for BSQ_{index}")

                out_fname = f"BSQ_tpcf_{index}.png"
                print(f"Saving plot to {out_fname}")
                plt.savefig(f"{args.output_dir}/{out_fname}", dpi=300)
                plt.close()

        #collect all clouds's 2PCF        
        results = np.array(results)
        # Save
        np.save(f"{args.output_dir}/{prefix}_start={args.start_idx}_end={args.end_idx}.npy", results)
        print(results.shape)
        print("Done.")