#!/bin/bash
#PBS -P iu60
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12

#PBS -l mem=8GB
#PBS -l jobfs=1GB
#PBS -l walltime=00:10:00
#PBS -l software=python
#PBS -l storage=gdata/ub7+gdata/ma05
#PBS -l wd

module load python3/3.7.4

python3 dataloader_test.py  --n_threads 0 --batch_size 8 --n_resgroups 10 --n_resblocks 20 --patch_size 192 --pre_train ./model/RCAN_BIX4.pt --tasmax --zg --tasmin --psl --dem
