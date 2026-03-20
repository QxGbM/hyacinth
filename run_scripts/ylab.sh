#!/bin/bash
#YBATCH -r rtx6000-bw_4
#SBATCH --job-name=hyacine-cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --time=01:00:00

module load cuda nccl intel/2022/mkl

#NCCL_DEBUG=WARN srun --cpu-bind=cores --gres-flags=allow-task-sharing /home/ma/hyacinth/build/examples/bootstrap_check.app
#rm -f "${SLURM_JOB_ID}.id.out"

NCCL_DEBUG=WARN srun --cpu-bind=cores --gres-flags=allow-task-sharing /home/ma/hyacinth/build/examples/xsvd_2d_example.app tilem=2 tilen=2 M=65536 N=4096
rm -f "${SLURM_JOB_ID}.id.out"
