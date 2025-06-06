#!/bin/bash

#SBATCH -p ccm
#SBATCH --mail-user=thuang@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH --ntasks=128
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH -J subset_tree_h5
#SBATCH -o sbatch_logs/subset_tree_h5.o%j
#SBATCH -e sbatch_logs/subset_tree_h5.e%j

module --force purge
module load modules/2.3-20240529
module load gcc
module load slurm
module load python/3.11.7
module load openmpi/4.0.7

python_exec="/mnt/home/thuang/playground/.venv_mpi/bin/python"

# Use SLURM-provided number of tasks
if [ -z "$SLURM_NTASKS" ]; then
    NCPU=1
else
    NCPU=$SLURM_NTASKS
fi

echo "Running with $NCPU MPI processes..."

srun -n "$NCPU" "$python_exec" /mnt/home/thuang/playground/data_scripts/prune_trim_tree_dataset.py