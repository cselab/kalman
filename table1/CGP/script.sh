#!/bin/bash
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_compute   # Partition to submit to
#SBATCH --mem=100GB           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o predict/predict%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e predict/predict%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --array=1-5

# load modules

#module load python
# run code


python predict.py