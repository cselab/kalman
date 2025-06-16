#!/bin/bash
#!/bin/bash
#SBATCH -c 64
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH --partition seas_gpu
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:4
#SBATCH --mem=80Gb
#SBATCH --constraint h100
#SBATCH -o pytorch_%j.out 
#SBATCH -e pytorch_%j.err 

# Load software modules and source conda environment
module load python
mamba activate env2

# Run program
srun -u python -u  funsearch.py 
