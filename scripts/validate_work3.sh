#!/bin/bash
#BSUB -q gpuv100
#BSUB -J protopnet_train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
# Run everything from /work3
#BSUB -cwd /work3/s204075/ProtoPNet
# Write logs to /work3
#BSUB -o /work3/s204075/ProtoPNet/logs/%J.out
#BSUB -e /work3/s204075/ProtoPNet/logs/%J.err
# Notifications
#BSUB -u s204075@dtu.dk
#BSUB -B
#BSUB -N

set -e

PROJECT_DIR=/work3/s204075/ProtoPNet
VENV_DIR=/work3/s204075/ProtoPNet/.venv
CACHE_DIR=/work3/s204075/.cache

mkdir -p "$CACHE_DIR" "$PROJECT_DIR/logs" "$(dirname $VENV_DIR)"

# Avoid Python writing to zhome caches
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=$CACHE_DIR
export PIP_CACHE_DIR=$CACHE_DIR/pip
export TORCH_HOME=$CACHE_DIR/torch
export HF_HOME=$CACHE_DIR/huggingface

# Load modules
module load cuda/11.6
module load python3/3.10.14

# Create fresh venv
# rm -rf "$VENV_DIR"
# python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Install dependencies fully inside /work3
python -m pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0

python validate_prototypes.py   --dataset data   --annotation_file data/part_locs.csv   --model_path saved_models/resnet34/cub200_experiment_1/checkpoints/300push74.51.pth   --architecture resnet34   --img_size 224   --batch_size 64   --threshold_mu 0.8   --bbox_size 50 50   --num_classes 200   --num_prototypes 2000   --prototype_shape 2000 128 1 1   --perturb   --seed 42 --prototype_activation_function log --results_dir validation_results/test