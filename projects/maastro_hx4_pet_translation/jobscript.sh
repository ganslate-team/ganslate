#!/usr/local_rwth/bin/zsh


# Job configuration ---

#SBATCH --job-name=hx4_pet_pix2pix
#SBATCH --output=/home/zk315372/Chinmay/Git/ganslate/projects/maastro_hx4_pet_translation/slurm_logs/%j.log

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
training_file="/home/zk315372/Chinmay/Git/ganslate/tools/train.py"
config_file="/home/zk315372/Chinmay/Git/ganslate/projects/maastro_hx4_pet_translation/experiments/pix2pix.yaml"

CUDA_VISIBLE_DEVICES=0 $python_interpreter $training_file config=$config_file


# ----------------------
# Run single GPU example: 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py config="./projects/maastro_hx4_pet_translation/experiments/pix2pix.yaml"

# Run distributed example:
# python -m torch.distributed.launch --use_env --nproc_per_node 2 tools/train.py config="./projects/maastro_hx4_pet_translation/experiments/pix2pix.yaml"