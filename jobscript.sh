#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J protopnet_train
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 16GB of system-memory (adjust based on dataset size)
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u mail@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# Navigate to project directory
cd ~/Desktop/02517/ProtoPNet

# Load necessary modules
module load cuda/11.6
module load python3/3.10

# Activate virtual environment (if you have one)
# source .venv/bin/activate

# Or install dependencies if needed
# pip install --user -r requirements.txt

# Set CUDA device (will be set automatically by LSF, but can override)
# export CUDA_VISIBLE_DEVICES=0

# Run training
python train.py \
    --dataset datasets/cub200 \
    --exp_name cub200_experiment_1 \
    --architecture resnet34 \
    --num_prototypes 2000 \
    --epochs 10000 \
    --batch_size 32 \
    --warm_epochs 150 \
    --test_interval 30 \
    --push_interval 300 \
    --gpus 0 \
    --num_workers 4 \
    --seed 42