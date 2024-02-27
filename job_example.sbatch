#!/bin/bash

#SBATCH --account=nn9997k --job-name=MyJob
#SBATCH --partition=a100 --gpus-per-node=1
#SBATCH --time=0-00:05:00
#SBATCH --ntasks-per-node=1 --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --open-mode=truncate
#SBATCH -o block.out

set -o errexit   ## Exit the script on any error
set -o nounset   ## Treat any unset variables as an error

ml purge
module --force swap StdEnv Zen2Env
ml OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0

nvcc -o vectorAdd vector_add.cu
# srun vectorAdd

nsys profile --stats=true srun vectorAdd
# nsys profile --stats=true --gpu-metrics-device=all --trace=cuda,nvtx srun vectorAdd
