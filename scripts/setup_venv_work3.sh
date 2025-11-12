set -e

PROJECT_DIR=/work3/s204075/ProtoPNet
VENV_DIR=/work3/s204075/.venvs/protopnet
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
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Install dependencies fully inside /work3
python -m pip install -r requirements.txt