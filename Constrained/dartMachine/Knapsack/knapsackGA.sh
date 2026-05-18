#!/bin/bash -l
#SBATCH --job-name="knapsack_bartbo"
#SBATCH --array=1-10              # 10 tasks 
#SBATCH --cpus-per-task=1          # Enough coresfor GA
#SBATCH -t 20:00:00
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
mkdir -p knapsack_results_GA

# Run worker script with array task ID
srun /home/people/20204013/r450/bin/Rscript knapsack_GA_traces.R $SLURM_ARRAY_TASK_ID
