#!/bin/bash
#YBATCH -r dgx-b200_1
#SBATCH --job-name=hyacine-cluster
#SBATCH --nodes=1
#SBATCH --time=01:00:00

module load cuda nccl intel/2022/mkl

/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-1 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-2 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-3 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-4 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-5 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-6 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-7 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-8 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-9 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-10 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-11 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-12 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-13 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-14 algo=L
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-15 algo=L

/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-1 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-2 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-3 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-4 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-5 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-6 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-7 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-8 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-9 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-10 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-11 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-12 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-13 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-14 algo=C
/home/ma/hyacinth/build/examples/xlra_example.app M=262144 N=4096 epi=1.e-15 algo=C
