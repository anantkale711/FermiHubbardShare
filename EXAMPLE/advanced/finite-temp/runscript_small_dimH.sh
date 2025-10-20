#!/bin/bash
#SBATCH -J anant_test # A single job name for the array
#SBATCH -c 8 # Number of cores
#SBATCH -p sapphire # Partition
#SBATCH --mem-per-cpu=4000 # Memory request (4 GB)
#SBATCH -t 0-18:00 # Maximum execution time (D-HH:MM)
#SBATCH -o console_output_OBC/small_%A_%a.out # Standard output
#SBATCH -e console_error_OBC/small_%A_%a.err # Standard error
module load intelpython
module load intel/23.2.0-fasrc01
module load intel-mkl/23.2.0-fasrc01
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd ~/FermiHubbardED/square_lattice/U8_OBC_batch
echo "$PWD"
sector_no=$SLURM_ARRAY_TASK_ID   # for sectors < 59, dimH < 1e7
echo "${SLURM_ARRAY_TASK_ID} ${sector_no}"
srun -n 1 -c $SLURM_CPUS_PER_TASK python runarray.py "${sector_no}" 1000 400