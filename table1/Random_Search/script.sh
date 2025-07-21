#!/bin/bash
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_compute   # Partition to submit to
#SBATCH --mem=100GB           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o predict_plus_three_lines2/predict_plus_three_lines_output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e predict_plus_three_lines2/predict_plus_three_lines_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --array=1-15

# load modules

#module load python
# run code


python  predict_plus_three_lines.py