# ProtoPNet Validation Pipeline Documentation

## Overview

This document describes the validation pipeline for assessing the quality of explanations provided by the ProtoPNet model. The pipeline evaluates two key aspects of prototype-based explanations:

1. **Consistency (S_con)**: Measures how consistently each prototype activates on the same semantic part across different images
2. **Stability (S_sta)**: Measures how stable prototype activations are under small image perturbations

## Motivation

ProtoPNet models learn interpretable prototypes (image patches) that are used for classification. To validate that these prototypes are meaningful and useful explanations, we need to assess:

- **Do prototypes consistently identify the same semantic parts?** A good prototype should activate on the same part (e.g., "beak", "wing", "tail") across multiple images, not randomly on different parts.
- **Are prototype activations stable?** A robust explanation should not change dramatically under small perturbations (noise), as this would indicate the model is relying on spurious features rather than meaningful visual concepts.

## Pipeline Steps

### Step 1: Load Part Annotations

**What we do:**
- Load part location annotations from `part_locs.csv` which contains `(image_id, x, y, part_name)` tuples
- These annotations specify where semantic parts (beak, crown, wing, etc.) are located in each image

**Why:**
- We need ground truth part locations to evaluate whether prototypes activate on meaningful parts
- The annotations are in cropped image coordinates (images are cropped to bounding boxes during dataset preparation)

**Technical details:**
- Annotations are point coordinates `(x, y)` for each part
- We convert these to bounding boxes by creating a small box (15×15 pixels in resized space) around each point
- **Coordinate scaling**: Since images are resized to 224×224 during training/validation, but annotations are in original cropped image coordinates, we must:
  1. Load each image to get its original cropped dimensions
  2. Scale annotation coordinates proportionally: `x_resized = x_original * (224 / width_original)`
  3. This ensures annotations align with the resized images used by the model

**Output:** Dictionary mapping `image_id → list of (part_name, bbox)` where bbox is in 224×224 coordinate space

---

### Step 2: Load Model and Prepare Data

**What we do:**
- Load the trained ProtoPNet model from checkpoint (`.pth` file)
- Extract model configuration (number of prototypes, classes, architecture, etc.)
- Prepare test dataset with image transformations (resize to 224×224, normalize)

**Why:**
- We need the trained model to compute prototype activations on test images
- Images must be preprocessed identically to how they were during training

**Technical details:**
- Model is loaded with `torch.load()` and state dict is extracted
- Images are transformed: `Resize(224×224) → ToTensor() → Normalize(mean, std)`
- Test dataset uses `ImageFolder` with custom wrapper to extract image IDs from filenames

---

### Step 3: Extract Prototype Activations (Original Images)

**What we do:**
For each test image:
1. Pass image through model to get feature maps
2. Compute L2 distances between feature maps and all prototypes
3. For each prototype, find the location of minimum distance (maximum activation)
4. Map this activation location from feature map space to image pixel space
5. Match the activation bounding box to annotated parts

**Why:**
- We need to know where each prototype activates in each image
- Mapping to pixel space allows us to compare with part annotations
- Matching to parts tells us which semantic part (if any) the prototype is detecting

**Technical details:**

**Activation extraction:**
- Forward pass: `feature_maps = conv_features(images)` → shape `[batch, channels, H_feat, W_feat]`
- Distance computation: `distances = _l2_convolution(feature_maps)` → shape `[batch, num_prototypes, H_feat, W_feat]`
- For each prototype `j`, find minimum distance location: `argmin(distance_map[j])` → `(row, col)` in feature space

**Coordinate mapping:**
- Feature map coordinates `(row, col)` are mapped to image coordinates using:
  - `center_x = (col + 0.5) * (img_width / W_feat)`
  - `center_y = (row + 0.5) * (img_height / H_feat)`
- Create bounding box around activation center: `[center_x ± bbox_w/2, center_y ± bbox_h/2]`
- Default bbox size: 50×50 pixels

**Part matching:**
- For each activation bbox, find overlapping annotated parts using IoU (Intersection over Union)
- IoU threshold: 0.2 (allows loose matching)
- If no IoU match, fallback to checking if activation center point is inside any part bbox
- Assign part label: matched part name, or "none" if no match

**Output:** For each prototype `j` and each image, we record the assigned part label → `proto_assignments[j] = [part_label_1, part_label_2, ..., part_label_N]`

---

### Step 4: Compute Consistency Metric (S_con)

**What we do:**
For each prototype:
1. Count how many times it matched each part label across all test images
2. Compute the maximum frequency: `max_freq = max(counts) / total_images`
3. A prototype is "consistent" if `max_freq ≥ threshold_μ` (default: 0.8)
4. Overall consistency score: `S_con = (# consistent prototypes) / (total prototypes)`

**Why:**
- Measures whether prototypes consistently identify the same semantic part
- High `max_freq` means the prototype activates on the same part across most images
- `S_con` tells us what fraction of prototypes are semantically consistent

**Mathematical definition:**
- For prototype `j`, let `L_j = [l_1, l_2, ..., l_N]` be the part labels assigned across `N` images
- Let `counts_j(part)` be the frequency of each part in `L_j`
- `max_freq_j = max(counts_j) / N`
- Prototype `j` is consistent if `max_freq_j ≥ μ` (threshold, default 0.8)
- `S_con = (1/P) * Σ_j [max_freq_j ≥ μ]` where `P` is total number of prototypes

**Output:**
- `per_proto_max_freq.csv`: Each prototype's maximum frequency
- `per_proto_hist.json`: Full histogram of part label counts for each prototype
- `S_con`: Overall consistency score (0.0 to 1.0)

---

### Step 5: Stability Test (Optional, if `--perturb` flag is set)

**What we do:**
1. Apply small random noise to test images: `images_pert = clamp(images + N(0, 0.05), 0, 1)`
2. Repeat Step 3 on perturbed images to get part labels
3. For each prototype and each image, compare: did it match the same part label as before?
4. Compute fraction of images where part label remained the same

**Why:**
- Tests robustness of prototype activations
- If activations change dramatically under small perturbations, the model may be relying on noise/spurious features
- Stable activations indicate the model learned meaningful, robust visual concepts

**Technical details:**
- Noise: Gaussian noise with standard deviation 0.05, clipped to [0, 1] range
- Same activation extraction and part matching process as Step 3
- For each prototype `j` and image `i`: `same[i] = 1` if `part_label_original[i] == part_label_perturbed[i]`, else `0`
- Per-prototype stability: `frac_same_j = mean(same)` across all images

**Mathematical definition:**
- For prototype `j` and image `i`, let `l_orig[i]` and `l_pert[i]` be part labels on original and perturbed images
- Stability indicator: `s_j[i] = 1` if `l_orig[i] == l_pert[i]`, else `0`
- Per-prototype stability: `frac_same_j = (1/N) * Σ_i s_j[i]`
- Overall stability: `S_sta = (1/P) * Σ_j frac_same_j`

**Output:**
- `per_proto_frac_same.csv`: Fraction of images where each prototype matched the same part under perturbation
- `S_sta`: Overall stability score (0.0 to 1.0)

---

## Key Design Decisions

### 1. Coordinate System Handling

**Problem:** Annotations are in cropped image coordinates, but model uses 224×224 resized images.

**Solution:** Load each image to get original dimensions, then scale annotations proportionally. This ensures accurate alignment between activation locations and part annotations.

**Trade-off:** Slower execution (must load images), but necessary for accuracy.

### 2. Bounding Box Size

**Choice:** 50×50 pixel boxes around activation centers.

**Rationale:** 
- Prototypes activate on small image patches (typically 1×1 in feature space, ~7×7 pixels in image space)
- 50×50 provides reasonable tolerance for matching while not being too permissive
- Can be adjusted via `--bbox_size` parameter

### 3. IoU Threshold

**Choice:** 0.2 for part matching.

**Rationale:**
- Prototype activations are point locations, converted to small boxes
- Part annotations are also small boxes (15×15 pixels)
- Low threshold (0.2) allows matching even with imperfect alignment
- Fallback to center-point check if no IoU match

### 4. Consistency Threshold (μ)

**Choice:** Default 0.8 (80% of activations must match the same part).

**Rationale:**
- High threshold ensures prototypes are truly consistent
- Allows some flexibility for edge cases or ambiguous images
- Can be adjusted via `--threshold_mu` parameter

### 5. Perturbation Magnitude

**Choice:** Gaussian noise with σ = 0.05.

**Rationale:**
- Small enough to not drastically change image appearance
- Large enough to test robustness to noise
- Represents realistic image variations (sensor noise, compression artifacts)

---

## Output Files

### `per_proto_max_freq.csv`
- Columns: `proto_idx`, `max_freq`
- Each row: prototype index and its maximum part label frequency
- Used to analyze distribution of consistency across prototypes

### `per_proto_hist.json`
- Dictionary: `{proto_idx: {part_name: count, ...}, ...}`
- Full histogram of part label counts for each prototype
- Allows detailed analysis of which parts each prototype matches

### `per_proto_frac_same.csv` (if `--perturb` enabled)
- Columns: `proto_idx`, `frac_same`
- Each row: prototype index and fraction of images where part label remained the same under perturbation
- Used to analyze stability distribution

### Metrics
- **S_con**: Consistency score (0.0 = no prototypes consistent, 1.0 = all prototypes consistent)
- **S_sta**: Stability score (0.0 = all activations change under perturbation, 1.0 = all remain the same)

---

## Interpretation of Results

### High S_con (e.g., > 0.5)
- Many prototypes consistently activate on the same semantic parts
- Indicates the model learned meaningful, interpretable concepts
- Prototypes are useful as explanations

### Low S_con (e.g., < 0.1)
- Prototypes activate on different parts across images
- May indicate:
  - Prototypes are too generic or not well-aligned with semantic parts
  - Model learned spurious features
  - Coordinate matching issues (should be rare after fixes)

### High S_sta (e.g., > 0.7)
- Prototype activations are robust to small perturbations
- Indicates stable, meaningful features rather than noise
- Good for reliable explanations

### Low S_sta (e.g., < 0.5)
- Activations change significantly under small perturbations
- May indicate:
  - Model relies on fragile, noise-sensitive features
  - Prototypes are not robust
  - Potential overfitting to training data

### Distribution Analysis
- Histogram of `max_freq` shows spread of consistency
- If many prototypes have `max_freq ≈ 1.0`: very consistent prototypes
- If many have `max_freq ≈ 0.3-0.5`: prototypes match multiple parts (less consistent)
- If many have `max_freq` for "none": prototypes don't match annotated parts (may match unannotated regions or background)

---

## Limitations and Future Improvements

### Current Limitations

1. **Part annotation coverage**: Only evaluates against annotated parts. Prototypes may activate on valid but unannotated regions (e.g., background, unlabeled parts).

2. **Binary matching**: Part matching is binary (match/no match). Doesn't capture partial matches or confidence.

3. **Single activation per prototype**: Only considers the maximum activation location. Prototypes may have multiple strong activations.

4. **Coordinate scaling overhead**: Loading images for size information is slow. Could be optimized with caching or pre-computed size database.

5. **Perturbation type**: Only tests Gaussian noise. Could test other perturbations (rotation, brightness, contrast).

### Potential Improvements

1. **Multi-activation matching**: Consider top-K activation locations per prototype
2. **Confidence scores**: Weight matches by activation strength
3. **Class-specific analysis**: Compute metrics per class to see if some classes have more consistent prototypes
4. **Visualization**: Generate visualizations showing prototype activations overlaid on images with part annotations
5. **Additional perturbations**: Test rotation, scaling, color jitter, etc.
6. **Temporal stability**: For video data, test stability across frames

---

## Usage Example

```bash
python validate_prototypes.py \
  --dataset datasets/cub200/data \
  --annotation_file datasets/cub200/data/part_locs.csv \
  --model_path saved_models/resnet34/cub200_experiment_1/checkpoints/600nopush78.86.pth \
  --architecture resnet34 \
  --img_size 224 \
  --batch_size 64 \
  --threshold_mu 0.8 \
  --bbox_size 50 50 \
  --perturb \
  --seed 42 \
  --prototype_activation_function log \
  --results_dir validation_results/test
```

This will:
1. Load annotations and model
2. Process all test images
3. Compute consistency (S_con)
4. Run stability test with perturbations
5. Compute stability (S_sta)
6. Save all results to `validation_results/test/`

---

## Summary

The validation pipeline provides quantitative metrics to assess the quality of ProtoPNet explanations:

- **Consistency (S_con)**: Measures semantic alignment between prototypes and annotated parts
- **Stability (S_sta)**: Measures robustness of prototype activations under perturbations

These metrics help answer the key question: *Are the learned prototypes meaningful, interpretable, and useful as explanations?*

High scores indicate the model learned semantically consistent and robust visual concepts, making the explanations valuable for understanding model behavior. Low scores suggest the prototypes may not be capturing meaningful semantic information, limiting their usefulness as explanations.

