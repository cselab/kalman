#!/bin/bash
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=100GB           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o evaluate_and_test%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e evaluate_and_test%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules

#module load python
# run code


python evaluate_and_test.py
