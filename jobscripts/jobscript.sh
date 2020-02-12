#!/usr/local_rwth/bin/zsh
 
### #SBATCH directives need to be in the first part of the jobscript

### Job name
#SBATCH --job-name=ganIbro

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
conda activate maastro

### your code goes here, the second part of the jobscript
# DONT FORGET TO UPDATE THE SBATCH jobname
python train.py --name NAME --niter 20 --niter_decay 20 --lr_G 0.0001 --lr_D 0.0004 --batchSize 8 --model unpaired_revgan3d --which_model_netG revResVnet_generator_3d  --which_model_netD n_layers --n_layers_D 3 --dataset_mode npy_unaligned_3d --dataroot /hpcwork/ft002207/lung1_lidc_gan --gpu_ids 0,1 --save_epoch_freq 10
