#!/bin/bash
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=100GB           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o observation_error%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e observation_error%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules

#module load python
# run code


python get_observations_mse.py
