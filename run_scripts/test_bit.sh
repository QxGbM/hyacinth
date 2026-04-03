#!/bin/bash
#YBATCH -r rtx6000-ada_8
#SBATCH --job-name=hyacine-cluster
#SBATCH --nodes=1
#SBATCH --time=04:00:00

module load cuda nccl intel/2022/mkl openmpi

mpirun -n 1 /home/ma/hyacinth/build/examples/xsvd_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=A out=mat.csv
mpirun -n 1 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=A ref=mat.csv
mpirun -n 2 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=A ref=mat.csv
mpirun -n 3 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=A ref=mat.csv
mpirun -n 4 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=A ref=mat.csv
mpirun -n 5 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=A ref=mat.csv
mpirun -n 6 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=A ref=mat.csv
mpirun -n 7 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=A ref=mat.csv
mpirun -n 8 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=A ref=mat.csv

mpirun -n 1 /home/ma/hyacinth/build/examples/xsvd_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=N out=mat.csv
mpirun -n 1 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=N ref=mat.csv
mpirun -n 2 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=N ref=mat.csv
mpirun -n 3 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=N ref=mat.csv
mpirun -n 4 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=N ref=mat.csv
mpirun -n 5 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=N ref=mat.csv
mpirun -n 6 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=N ref=mat.csv
mpirun -n 7 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=N ref=mat.csv
mpirun -n 8 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=N ref=mat.csv

mpirun -n 1 /home/ma/hyacinth/build/examples/xsvd_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F out=mat.csv
mpirun -n 1 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F ref=mat.csv
mpirun -n 2 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F ref=mat.csv
mpirun -n 3 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F ref=mat.csv
mpirun -n 4 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F ref=mat.csv
mpirun -n 5 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F ref=mat.csv
mpirun -n 6 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F ref=mat.csv
mpirun -n 7 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F ref=mat.csv
mpirun -n 8 /home/ma/hyacinth/build/examples/xsvd_1drow_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F ref=mat.csv
