#!/bin/bash
#PBS -P iu60
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l walltime=00:40:00
#PBS -l storage=gdata/ub7+gdata/ma05
#PBS -l wd

module load python3/3.7.4

python3 calculate_mean_access-s1.py
