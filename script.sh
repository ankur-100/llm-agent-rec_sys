#!/bin/bash
#PBS -N slurm_script                      # Job name
#PBS -l select=1:ncpus=16:mem=110GB       # Resources: 1 chunk, 16 CPUs, 110 GB RAM (for 1 GPU if needed)
#PBS -l walltime=02:00:00                 # Job wall time (HH:MM:SS)
#PBS -j oe                                # Combine stdout and stderr
#PBS -o output.log                        # Output file
#PBS -P personal-ankur01                  # Your NSCC Project ID
#PBS -q ai                                # Queue to submit to (normal or ai)

# Load necessary modules
module load miniforge3

# Activate conda environment (if applicable)
source activate myenv

# Move to the working directory (PBS sets $PBS_O_WORKDIR)
cd $PBS_O_WORKDIR

# Run your script
python main.py
