#!/bin/bash
#PBS -P iu60
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l walltime=20:00:00
#PBS -l wd
#PBS-l storage=gdata/ma05+gdata/ub7

module load python3/3.7.4

python3  --n_thread 0 --batch_size 4

