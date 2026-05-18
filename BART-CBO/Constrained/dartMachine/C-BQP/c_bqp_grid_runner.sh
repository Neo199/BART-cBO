#!/bin/bash -l
#SBATCH --job-name="c_bqp_grid"
#SBATCH --array=1-180             # 9 configs x 2 methods x 10 reps
#SBATCH --cpus-per-task=8         # Enough cores for BART+GA
#SBATCH -t 520:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niyati.seth@ucdconnect.ie
# Load intel libraries
module load java/25.0.1
module load intel-oneapi-compilers/2023.2.4-gcc-11.5.0-6uvfkah
module load intel-oneapi-mpi/2021.14.0-gcc-11.5.0-hjmtgxa
module load intel-oneapi-mkl/2024.2.2-gcc-11.5.0-hjitxos
module load gcc/11.5.0-gcc-11.5.0-vdl6dwy
# Run from current directory
cd $SLURM_SUBMIT_DIR
# Create output directory if it doesn't exist
mkdir -p c_bqp_grid_results
# Run worker script with array task ID
srun /home/people/20204013/r450/bin/Rscript c_bqp_grid_runner.R $SLURM_ARRAY_TASK_ID
