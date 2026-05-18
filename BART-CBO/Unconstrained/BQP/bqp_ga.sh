#!/bin/bash -l
#SBATCH --job-name="bqp_ga_patch"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niyati.seth@ucdconnect.ie
 
# Load intel libraries
module load java/25.0.1
module load intel-oneapi-compilers/2023.2.4-gcc-11.5.0-6uvfkah
module load intel-oneapi-mpi/2021.14.0-gcc-11.5.0-hjmtgxa
module load intel-oneapi-mkl/2024.2.2-gcc-11.5.0-hjitxos
module load gcc/11.5.0-gcc-11.5.0-vdl6dwy
 
cd $SLURM_SUBMIT_DIR
mkdir -p bqp_results
 
# Run all 10 GA reps sequentially in one node (no array needed)
/home/people/20204013/r450/bin/Rscript ga_traces_bqp.R