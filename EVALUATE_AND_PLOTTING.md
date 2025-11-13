# Evaluate.py and Plotting Guide

## What `evaluate.py` Does

The `evaluate.py` script provides **visualization and analysis** of learned prototypes. It has **three different modes**:

### 1. Global Analysis (`evaluate.py ... global`)

**What it does:**
- Finds the **most activated image patches** across the entire dataset for each prototype
- Shows which image regions each prototype matches best
- Generates visualizations of prototype images with bounding boxes

**Output:**
- For each prototype, saves the top-K most similar image patches from train/test sets
- Saves prototype images with bounding boxes showing where they activate
- Creates directory structure: `analysis/{architecture}/{exp_name}/{model_name}/global/nearest_prototypes/`

**Usage:**
```bash
python evaluate.py \
  --model saved_models/resnet34/cub200_experiment_1/checkpoints/600nopush78.86.pth \
  global \
  --dataset datasets/cub200/data \
  --top_imgs 5
```

**Why use it:**
- Visual inspection of what each prototype learned
- See which image patches prototypes match
- Understand prototype diversity

---

### 2. Local Analysis (`evaluate.py ... local`)

**What it does:**
- Analyzes prototype activations for **specific images** or a directory of images
- Shows which prototypes activate most strongly for given images
- Visualizes activation patterns overlaid on images
- Creates alignment matrices showing prototype-to-part relationships

**Output:**
- For each analyzed image:
  - Top-K most activated prototypes
  - Activation heatmaps overlaid on images
  - Bounding boxes showing activation locations
  - Alignment matrices (prototype vs part locations)
- Creates directory: `analysis/{architecture}/{exp_name}/{model_name}/local/`

**Usage:**
```bash
python evaluate.py \
  --model saved_models/resnet34/cub200_experiment_1/checkpoints/600nopush78.86.pth \
  local \
  --img datasets/cub200/data/test/001.Black_footed_Albatross \
  --top_prototypes 10
```

**Why use it:**
- Understand which prototypes activate for specific images
- See activation patterns visually
- Analyze prototype-part alignment for individual images

---

### 3. Alignment Analysis (`evaluate.py ... alignment`)

**What it does:**
- Computes **alignment matrices** showing how well prototypes align with annotated parts
- For each class, creates heatmaps showing prototype-to-part distances
- Generates alignment scores (Euclidean distance between prototype activation and part locations)

**Output:**
- Alignment matrices (heatmaps) for each class
- Shows which prototypes align with which parts
- Creates directory: `analysis/{architecture}/{exp_name}/{model_name}/alignment/{class_name}/`

**Usage:**
```bash
python evaluate.py \
  --model saved_models/resnet34/cub200_experiment_1/checkpoints/600nopush78.86.pth \
  alignment \
  --dataset datasets/cub200/data
```

**Why use it:**
- Quantitative analysis of prototype-part alignment
- Visual heatmaps showing alignment patterns
- Compare alignment across different classes

---

## How to Generate Plots for Validation Results

The validation pipeline (`validate_prototypes.py`) produces CSV/JSON files with metrics. To generate plots from these results, use the **separate plotting script**:

### Plotting Script: `ppnet/plot_prototype_metrics.py`

**What it does:**
- Generates histograms of consistency metrics (`max_freq`)
- Generates histograms of stability metrics (`frac_same`)
- Optionally creates scatter plots of class accuracy vs consistency

**Usage:**
```bash
python ppnet/plot_prototype_metrics.py \
  --results_dir validation_results/test \
  --output_dir validation_results/test/plots
```

**Output:**
- `hist_max_freq.png` - Histogram showing distribution of prototype consistency
- `hist_frac_same.png` - Histogram showing distribution of prototype stability
- `scatter_acc_vs_consistency.png` - (Optional) Scatter plot if class stats provided

**Example with your current results:**
```bash
python ppnet/plot_prototype_metrics.py \
  --results_dir validation_results/test \
  --output_dir validation_results/test/plots
```

This will create:
- `validation_results/test/plots/hist_max_freq.png`
- `validation_results/test/plots/hist_frac_same.png`

---

## Key Differences

| Script | Purpose | Output |
|--------|---------|--------|
| `evaluate.py` | Visualize prototypes and activations | Images, heatmaps, alignment matrices |
| `validate_prototypes.py` | Quantitative validation metrics | CSV/JSON with consistency/stability scores |
| `ppnet/plot_prototype_metrics.py` | Plot validation metrics | Histograms, scatter plots |

---

## Quick Reference

### Generate validation plots (from your validation results):
```bash
python ppnet/plot_prototype_metrics.py \
  --results_dir validation_results/test \
  --output_dir validation_results/test/plots
```

### Visualize prototype activations (global):
```bash
python evaluate.py \
  --model saved_models/resnet34/cub200_experiment_1/checkpoints/600nopush78.86.pth \
  global \
  --dataset datasets/cub200/data \
  --top_imgs 5
```

### Analyze alignment (prototype-to-part):
```bash
python evaluate.py \
  --model saved_models/resnet34/cub200_experiment_1/checkpoints/600nopush78.86.pth \
  alignment \
  --dataset datasets/cub200/data
```

