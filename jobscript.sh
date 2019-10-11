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
#SBATCH --mem-per-cpu=16G

### Request a host with a Volta GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:volta:2

### if needed: switch to your working directory (where you saved your program)
#cd $HOME/a/

### Load modules
module load python/3.6.8
module load cuda/100
module load cudnn/7.4
pip install --user -r requirements.txt

### your code goes here, the second part of the jobscript
python3 train.py --niter 20 --niter_decay 20 --save_epoch_freq 10 --dataset_mode npy_aligned_3d --dataroot ../patchified/ --model paired_revgan3d --name 3d_460 --which_model_netG srcnn_generator_3d --gpu_ids 0,1 --loadSize 256 --fineSize 256 --batchSize 2 
