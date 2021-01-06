#!/bin/bash
 
### #SBATCH directives need to be in the first part of the jobscript

### Job name
#SBATCH --job-name=RIRE_MR_to_CT_preprocessed_train

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=1-00:00:00

### Request the amount of memory you need for your job.
### You can specify this in either MB (1024M) or GB (4G).

#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1

### Request a host with a Volta GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:volta:2

### Set number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
### Activate conda environment
source $HOME/miniconda3/bin/activate
conda activate gan_env
### Run training
cd $HOME/midaGAN2/midaGAN/
CUDA_VISIBLE_DEVICES=0 python tools/train.py config=projects/RIRE_MR_to_CT/experiments/RIRE_MR_to_CT_preprocessed_train.yaml