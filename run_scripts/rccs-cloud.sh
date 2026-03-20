#!/bin/bash
#SBATCH --partition=ai-l40s
#SBATCH --job-name=hyacine-cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --time=01:00:00

module load system/ai-l40s nvhpc

nvidia-smi topo -m

srun --cpu-bind=cores \
  bash -lc 'echo "rank=$SLURM_PROCID local=$SLURM_LOCALID host=$(hostname) cvd=$CUDA_VISIBLE_DEVICES"'

#NCCL_DEBUG=WARN srun --cpu-bind=cores --gres-flags=allow-task-sharing /home/users/qianxiang.ma/hyacinth/build/examples/bootstrap_check.app
#rm -f "${SLURM_JOB_ID}.id.out"

NCCL_DEBUG=WARN srun --cpu-bind=cores --gres-flags=allow-task-sharing /home/users/qianxiang.ma/hyacinth/build/examples/xsvd_2d_example.app tilem=2 tilen=2 M=65536 N=4096
rm -f "${SLURM_JOB_ID}.id.out"
