#!/bin/bash
 
### #SBATCH directives need to be in the first part of the jobscript

### Job name
#SBATCH --job-name=ex1

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=5-00:00:00

### Request the amount of memory you need for your job.
### You can specify this in either MB (1024M) or GB (4G).

#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1

### Request a host with a Volta GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:volta:2

### if needed: switch to your working directory (where you saved your program)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/.bashrc
cd $HOME/ganslate/

### your code goes here, the second part of the jobscript
# DONT FORGET TO UPDATE THE SBATCH jobname
python -m torch.distributed.launch --use_env --nproc_per_node 2 tools/train.py config=projects/brats_flair_to_t1w/experiments/ex1.yaml
