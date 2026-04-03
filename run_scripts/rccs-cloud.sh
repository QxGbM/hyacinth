#!/bin/bash
#SBATCH --partition=ai-l40s
#SBATCH --job-name=hyacine-cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --time=01:00:00

module load system/ai-l40s nvhpc

nvidia-smi topo -m

srun --cpu-bind=cores bash -lc 'echo "rank=$SLURM_PROCID local=$SLURM_LOCALID host=$(hostname) cvd=$CUDA_VISIBLE_DEVICES"'

HYACIN_JOB_ID=1 NCCL_DEBUG=WARN srun --cpu-bind=cores --gres-flags=allow-task-sharing /home/users/qianxiang.ma/hyacinth/build-l40s/examples/bootstrap_check.app
rm -f "${SLURM_JOB_ID}-1.id.out"

HYACIN_JOB_ID=2 srun --cpu-bind=cores --gres-flags=allow-task-sharing /home/users/qianxiang.ma/hyacinth/build-l40s/examples/xsvd_2d_example.app M=65536 N=4096 tilem=4 tilen=2
rm -f "${SLURM_JOB_ID}-2.id.out"
