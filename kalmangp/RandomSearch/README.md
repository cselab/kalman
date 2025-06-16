Instrall numpy
```
module load python
mamba create -y -q -n numpy numpy
```

Run
```
sbatch --array 0-1999 -t 0-3 --mem 1Gb --wrap 'srun mamba run --no-capture-output -n numpy sh -xc "OMP_NUM_THREADS=1 python Kalmangp.nodes.py"'
```
