#!/bin/bash
#YBATCH -r rtx6000-ada_8
#SBATCH --job-name=hyacine-cluster
#SBATCH --nodes=1
#SBATCH --time=04:00:00

module load intel/2022/mkl cuda/12.8 nccl/cuda-12.9/2.28.7 openmpi/5.0-cuda-12.8

mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=100 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=200 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=400 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=600 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=800 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=1200 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=1400 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=1600 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=1800 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
mpirun -n 4 ./examples/xsvd_1drow_example.app M=1281167 N=2048 K=2000 epi=1.e-3 data=S file=/mnt/nfs/Users/ma/imagenet1k_train_resnet50_features_2048d.csv
