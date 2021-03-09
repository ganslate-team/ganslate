#!/usr/local_rwth/bin/zsh


# Job configuration ---

#SBATCH --job-name=label2photo_cyclegan-color
#SBATCH --output=slurm/%j.log

## OpenMP settings
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

## Request for a node with 2 Tesla P100 GPUs
#SBATCH --gres=gpu:pascal:2

#SBATCH --time=5:00:00

## TO use the UM DKE project account
# #SBATCH --account=um_dke


# Load CUDA 
module load cuda

# Debug info
echo; echo
nvidia-smi
echo; echo

# Execute training
python_interpreter="/home/zk315372/miniconda3/envs/gan_env/bin/python3"
training_file="/home/zk315372/Chinmay/Git/midaGAN/tools/train.py"
config_file="/home/zk315372/Chinmay/Git/midaGAN/projects/cityscapes_label2photo/experiments/cyclegan.yaml"

CUDA_VISIBLE_DEVICES=0 $python_interpreter $training_file config=$config_file
