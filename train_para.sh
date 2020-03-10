#!/bin/bash
#PBS -P iu60
#PBS -q gpuvolta
#PBS -l ngpus=2
#PBS -l ncpus=24
#PBS -l mem=64GB
#PBS -l jobfs=1GB
#PBS -l walltime=00:20:00
#PBS -l storage=gdata/ub7+gdata/ma05
#PBS -l wd

module load python3/3.7.4

python3 train_para.py --n_threads 2 --batch_size 8
