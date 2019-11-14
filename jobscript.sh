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

### Load modules
module switch intel gcc
module load python/3.6.8
module load cuda/100
module load cudnn/7.4
pip install --user -r requirements.txt

### your code goes here, the second part of the jobscript

#python3 train.py --niter 20 --niter_decay 20 --save_epoch_freq 10 --dataset_mode npy_aligned_3d --dataroot ../3D_460_patchified_norm/ --model paired_revgan3d --name 3d_460 --which_model_netG edsrF_generator_3d --gpu_ids 0,1 --batchSize 2 --which_model_netD n_layers --n_layers_D 2 --lr_G 0.0001 --lr_D 0.0004

python3 train.py --niter 100 --niter_decay 100 --save_epoch_freq 10 --dataset_mode npy_aligned_2d --dataroot ../2D_460_norm/ --model paired_revgan --name 2d_460 --gpu_ids 0,1 --batchSize 16 --lr_G 0.0001 --lr_D 0.0004
