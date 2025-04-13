#!/bin/bash
#PBS -N myjobname
#PBS -l select=1:ncpus=4:mem=8GB
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o output.log
#PBS -P personal-ankur013
#PBS -q normal

module load miniforge3
source activate myenv

export LD_PRELOAD=/home/users/ntu/ankur013/.conda/envs/myenv/lib/libiomp5.so

cd $PBS_O_WORKDIR

python abc_1.py
