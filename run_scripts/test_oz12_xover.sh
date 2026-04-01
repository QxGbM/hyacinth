#!/bin/bash
#YBATCH -r dgx-b200_1
#SBATCH --job-name=hyacine-cluster
#SBATCH --nodes=1
#SBATCH --time=01:00:00

module load cuda nccl intel/2022/mkl

/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-4 data=Z algo=F
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=F
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-8 data=Z algo=F
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-10 data=Z algo=F

/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-1 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-2 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-3 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-4 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-5 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-7 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-8 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-9 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-10 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-11 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-12 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-13 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-14 data=Z algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-15 data=Z algo=L

/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-1 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-2 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-3 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-4 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-5 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-6 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-7 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-8 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-9 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-10 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-11 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-12 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-13 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-14 data=Z algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-15 data=Z algo=C
