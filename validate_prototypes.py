#!/usr/bin/env python3
import argparse
import csv
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Import your model utilities
from ppnet.model import construct_PPNet
from ppnet.preprocess import preprocess_input_function, mean, std
from ppnet.helpers import set_seed

def load_part_annotations(annotation_file, img_size, dataset_path=None):
    """
    Load part annotations for CUB-200 dataset.
    annotation_file: path to CSV with columns: image_id, x, y, part_name
    img_size: int, the size to which images are resized (img_size × img_size)
    dataset_path: path to dataset root (to load actual image sizes for scaling)
    Returns:
      dict of image_id (int) -> list of (part_name, (x_min, y_min, x_max, y_max))
    """
    from PIL import Image
    annots = {}
    # If using point annotations (x,y) rather than full boxes, we define a small box around point
    box_half = 10  # pixels in resized image space; you may adjust (original was 10)
    image_sizes = {}  # cache image sizes
    
    # Load annotations first
    with open(annotation_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = int(row['image_id'])
            x = float(row['x'])
            y = float(row['y'])
            part_name = row['part_name']
            
            # Get original image size for scaling (annotations are in cropped image coordinates)
            if dataset_path and img_id not in image_sizes:
                # Try to find the image file
                for split in ['test', 'train']:
                    split_dir = os.path.join(dataset_path, split)
                    if os.path.exists(split_dir):
                        for class_dir in os.listdir(split_dir):
                            img_path = os.path.join(split_dir, class_dir, f"{img_id}.jpg")
                            if os.path.exists(img_path):
                                try:
                                    img = Image.open(img_path)
                                    orig_w, orig_h = img.size
                                    image_sizes[img_id] = (orig_w, orig_h)
                                    break
                                except:
                                    pass
                # If not found, assume a reasonable default (most CUB images are roughly square after cropping)
                if img_id not in image_sizes:
                    image_sizes[img_id] = (224, 224)  # fallback: assume already resized
            
            orig_w, orig_h = image_sizes.get(img_id, (224, 224))
            
            # Scale coordinates from original cropped size to resized size
            scale_x = img_size / orig_w
            scale_y = img_size / orig_h
            x_s = x * scale_x
            y_s = y * scale_y
            
            x_min = max(0.0, x_s - box_half)
            y_min = max(0.0, y_s - box_half)
            x_max = min(img_size, x_s + box_half)
            y_max = min(img_size, y_s + box_half)
            bbox = (x_min, y_min, x_max, y_max)
            annots.setdefault(img_id, []).append((part_name, bbox))
    return annots

def bbox_overlaps(b1, b2, threshold=0.3):
    """
    Compute IoU between two boxes and return True if IoU >= threshold.
    """
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    if area1 + area2 - interArea <= 0:
        return False
    iou = interArea / float(area1 + area2 - interArea + 1e-6)
    return iou >= threshold

def determine_part_label_for_bbox(bbox, part_annots, iou_threshold=0.2):
    """
    Assigns the part label whose bbox loosely overlaps with activation bbox.
    If no IoU > threshold, then fallback to checking activation center point inside part bbox.
    """
    x1, y1, x2, y2 = bbox
    best_part = None
    best_iou = 0.0
    for part_name, p_bbox in part_annots:
        px1, py1, px2, py2 = p_bbox
        # compute IoU
        inter_x1 = max(x1, px1)
        inter_y1 = max(y1, py1)
        inter_x2 = min(x2, px2)
        inter_y2 = min(y2, py2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            continue
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (x2 - x1) * (y2 - y1)
        area_b = (px2 - px1) * (py2 - py1)
        iou = inter_area / (area_a + area_b - inter_area)
        if iou > best_iou:
            best_iou = iou
            best_part = part_name
    if best_iou >= iou_threshold:
        return best_part
    else:
        # fallback: center point check
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        for part_name, p_bbox in part_annots:
            px1, py1, px2, py2 = p_bbox
            if (px1 <= x_center <= px2) and (py1 <= y_center <= py2):
                return part_name
        # debug print
        # print(f"No match: bbox {bbox}, parts {part_annots}, best_iou={best_iou:.3f}")
        return None

def find_max_activation_location(activation_map):
    flat_idx = activation_map.argmax()
    row = flat_idx // activation_map.shape[1]
    col = flat_idx % activation_map.shape[1]
    return row.item(), col.item()

def map_activation_to_bbox(row, col, img_shape, feat_shape, bbox_size):
    """
    Maps activation map coordinate (row, col) to image bbox.
    img_shape: (img_h, img_w)
    feat_shape: (H_feat, W_feat)
    bbox_size: (bbox_h, bbox_w) in image‐pixel units
    Returns: (x_min, y_min, x_max, y_max)
    """
    img_h, img_w = img_shape
    H_feat, W_feat = feat_shape
    center_y = (row + 0.5) * (img_h / H_feat)
    center_x = (col + 0.5) * (img_w / W_feat)
    bbox_h, bbox_w = bbox_size
    y_min = max(0, center_y - bbox_h / 2)
    x_min = max(0, center_x - bbox_w / 2)
    y_max = min(img_h, center_y + bbox_h / 2)
    x_max = min(img_w, center_x + bbox_w / 2)
    return (x_min, y_min, x_max, y_max)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load annotations
    part_annotations = load_part_annotations(args.annotation_file, img_size=args.img_size, dataset_path=args.dataset)


    # Build model
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        num_classes = checkpoint.get('num_classes', args.num_classes)
        num_prototypes = checkpoint.get('num_prototypes', args.num_prototypes)
        prototype_shape = tuple(checkpoint.get('prototype_shape', args.prototype_shape))
        proto_act_fn = checkpoint.get('prototype_activation_function', args.prototype_activation_function)
        add_on_type = checkpoint.get('add_on_layers_type', args.add_on_layers_type)
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        num_classes = args.num_classes
        num_prototypes = args.num_prototypes
        prototype_shape = tuple(args.prototype_shape)
        proto_act_fn = args.prototype_activation_function
        add_on_type = args.add_on_layers_type

    if num_classes is None:
        raise ValueError("num_classes must be provided either in checkpoint or via --num_classes")
    if num_prototypes is None:
        raise ValueError("num_prototypes must be provided either in checkpoint or via --num_prototypes")
    if len(prototype_shape) != 4:
        raise ValueError("prototype_shape must be provided properly")

    ppnet = construct_PPNet(
        base_architecture=args.architecture,
        pretrained=False,
        img_size=args.img_size,
        prototype_shape=prototype_shape,
        num_classes=num_classes,
        prototype_activation_function=proto_act_fn,
        add_on_layers_type=add_on_type
    )
    ppnet.load_state_dict(state_dict, strict=False)
    ppnet = ppnet.to(device)
    ppnet = torch.nn.DataParallel(ppnet)

    print(f"Loaded model with architecture {args.architecture}, img_size {args.img_size}, num_classes {num_classes}, "
          f"num_prototypes {num_prototypes}, prototype_shape {prototype_shape}, activation_fn {proto_act_fn}, add_on_layers_type {add_on_type}")

    # Dataset & loader
    normalize = T.Normalize(mean=mean, std=std)
    transform = T.Compose([T.Resize((args.img_size,args.img_size)), T.ToTensor(), normalize])
    from torchvision.datasets import ImageFolder
    class IdFolder(ImageFolder):
        def __getitem__(self, index):
            image, label = super().__getitem__(index)
            path, _ = self.samples[index]
            image_id = os.path.splitext(os.path.basename(path))[0]
            return image, label, image_id

    test_dir = os.path.join(args.dataset, 'test')
    test_dataset = IdFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print("Part annotation keys sample:", list(part_annotations.keys())[:10])
    # Inspect sample image filenames from loader
    sample_paths = [p for p, _ in test_loader.dataset.samples[:10]]
    sample_ids = []
    for p in sample_paths:
        fname = os.path.basename(p)
        # assumes e.g., '001.Black_footed_Albatross_0001.jpg' → image_id 1
        id_str = os.path.splitext(fname)[0].split('_')[-1]
        try:
            sample_ids.append(int(id_str))
        except ValueError:
            sample_ids.append(fname)
    print("Loader image_ids sample:", sample_ids)
    print("Sample annotation image_ids (first10):", list(part_annotations.keys())[:10])
    sample_ids = [os.path.splitext(os.path.basename(p))[0] for p, _ in test_dataset.samples[:5]]
    print("Sample loader image_ids:", sample_ids)

    # Extract activations
    print("Extracting prototype activations (original images)…")
    results_orig = []
    proto_assignments = {j: [] for j in range(num_prototypes)}
    image_count = 0
    for images, labels, image_ids in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        feature_maps = ppnet.module.conv_features(images)
        distances_full = ppnet.module._l2_convolution(feature_maps)
        batch_size, num_protos, H_feat, W_feat = distances_full.shape
        for b in range(batch_size):
            image_id_str = image_ids[b]
            # Convert image_id to int for annotation lookup
            try:
                image_id = int(image_id_str)
            except (ValueError, TypeError):
                # If it's not a simple integer, try extracting from filename
                image_id = int(os.path.splitext(os.path.basename(str(image_id_str)))[0])
            cls = labels[b].item()
            for j in range(num_protos):
                amap = distances_full[b, j, :, :].cpu()
                flat_idx = amap.argmin()
                row = flat_idx // W_feat
                col = flat_idx % W_feat
                bbox = map_activation_to_bbox(row, col, (args.img_size, args.img_size),
                                              (H_feat, W_feat), bbox_size=tuple(args.bbox_size))
                part_annots = part_annotations.get(image_id, [])
                part_label = determine_part_label_for_bbox(bbox, part_annots, iou_threshold=0.2)
                if part_label is None:
                    part_label = "none"
                proto_assignments[j].append(part_label)
                if image_count < 5 and j == 0:
                    print(f"Image {image_id} (from '{image_id_str}'), proto {j}: bbox {bbox}, part_annots {len(part_annots)} parts, assigned {part_label}")
                results_orig.append({
                    "image_id": image_id,
                    "class": cls,
                    "proto_idx": j,
                    "part_label": part_label
                })
        image_count += batch_size
        if args.num_samples and image_count >= args.num_samples:
            break

    # Compute consistency metric
    per_proto_hist = {}
    per_proto_max_freq = {}
    consistent_count = 0
    for j, labels in proto_assignments.items():
        if not labels:
            per_proto_hist[j] = {}
            per_proto_max_freq[j] = 0.0
            continue
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        per_proto_hist[j] = counts
        max_count = max(counts.values())
        freq = max_count / len(labels)
        per_proto_max_freq[j] = freq
        if freq >= args.threshold_mu:
            consistent_count += 1

    S_con = consistent_count / float(num_prototypes)
    print(f"Computing consistency metric…\nConsistency score S_con = {S_con:.4f}")

    # Save outputs
    os.makedirs(args.results_dir, exist_ok=True)
    df_max_freq = pd.DataFrame({"proto_idx": list(per_proto_max_freq.keys()),
                                "max_freq": list(per_proto_max_freq.values())})
    csv_path = os.path.join(args.results_dir, "per_proto_max_freq.csv")
    df_max_freq.to_csv(csv_path, index=False)
    print(f"Saved per‐prototype max_freq CSV to {csv_path}")

    hist_path = os.path.join(args.results_dir, "per_proto_hist.json")
    with open(hist_path, 'w') as f:
        json.dump(per_proto_hist, f, indent=2)
    print(f"Saved per‐prototype hist JSON to {hist_path}")

    non_none = sum(1 for labels in per_proto_hist.values() if any(lbl != "none" for lbl in labels))
    print(f"How many prototypes ever matched a non‐‘none’ label? {non_none}/{num_prototypes}")

    # Stability (optional)
    if args.perturb:
        print("Applying perturbations and extracting activations (perturbed)…")
        results_pert = []
        per_proto_same = {j: [] for j in range(num_prototypes)}
        pert_image_count = 0  # Track position in perturbed sequence (matches original image_count)
        for images, labels, image_ids in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            noise = torch.randn_like(images) * 0.05
            images_pert = torch.clamp(images + noise, min=0.0, max=1.0)
            feature_maps = ppnet.module.conv_features(images_pert)
            distances_full = ppnet.module._l2_convolution(feature_maps)
            batch_size, num_protos, H_feat, W_feat = distances_full.shape
            for b in range(batch_size):
                image_id_str = image_ids[b]
                # Convert image_id to int for annotation lookup
                try:
                    image_id = int(image_id_str)
                except (ValueError, TypeError):
                    # If it's not a simple integer, try extracting from filename
                    image_id = int(os.path.splitext(os.path.basename(str(image_id_str)))[0])
                cls = labels[b].item()
                for j in range(num_protos):
                    amap = distances_full[b, j, :, :].cpu()
                    flat_idx = amap.argmin()
                    row = flat_idx // W_feat
                    col = flat_idx % W_feat
                    bbox = map_activation_to_bbox(row, col, (args.img_size, args.img_size),
                                                  (H_feat, W_feat), bbox_size=tuple(args.bbox_size))
                    part_annots = part_annotations.get(image_id, [])
                    part_label = determine_part_label_for_bbox(bbox, part_annots, iou_threshold=0.3)
                    if part_label is None:
                        part_label = "none"
                    results_pert.append({
                        "image_id": image_id,
                        "class": cls,
                        "proto_idx": j,
                        "part_label": part_label
                    })
                    if pert_image_count < 5 and j == 0:
                        print(f"Image {image_id} (from '{image_id_str}'), proto {j}: bbox {bbox}, part_annots {len(part_annots)} parts, assigned {part_label}")
                    # Match by image index: pert_image_count should match original image_count position
                    if pert_image_count < len(proto_assignments[j]):
                        per_proto_same[j].append(1 if part_label == proto_assignments[j][pert_image_count] else 0)
                    else:
                        per_proto_same[j].append(0)
                pert_image_count += 1
            if args.num_samples and pert_image_count >= args.num_samples:
                break

        per_proto_frac = {j: (sum(vals)/len(vals) if len(vals)>0 else 0.0) for j, vals in per_proto_same.items()}
        S_sta = sum(per_proto_frac.values()) / float(num_prototypes)
        print(f"Computing stability metric…\nStability score S_sta = {S_sta:.4f}")

        df_frac_same = pd.DataFrame({"proto_idx": list(per_proto_frac.keys()),
                                     "frac_same": list(per_proto_frac.values())})
        csv_frac_path = os.path.join(args.results_dir, "per_proto_frac_same.csv")
        df_frac_same.to_csv(csv_frac_path, index=False)
        print(f"Saved per‐prototype stability CSV to {csv_frac_path}")

    print("Validation pipeline completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate prototypes of a trained ProtoPNet model")
    parser.add_argument('--dataset', type=str, required=True,
                        help='path of dataset root (with train/test folders)')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='path to part annotations CSV file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to trained model checkpoint')
    parser.add_argument('--architecture', type=str, default='resnet34',
                        help='backbone architecture (default: %(default)s)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image resize size (default: %(default)d)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for validation loader')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for DataLoader')
    parser.add_argument('--threshold_mu', type=float, default=0.8,
                        help='consistency threshold µ')
    parser.add_argument('--bbox_size', type=int, nargs=2, default=(50,50),
                        help='bounding box size (h, w) in pixels around activation')
    parser.add_argument('--perturb', action='store_true',
                        help='run stability test with perturbations')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--results_dir', type=str, default="validation_results/default",
                        help='directory where validation results (CSV/JSON) will be saved')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='(optional) number of images to process for quicker test')
    parser.add_argument('--num_classes', type=int, required=False,
                        help='number of classes in model (if not in checkpoint)')
    parser.add_argument('--num_prototypes', type=int, required=False,
                        help='number of prototypes in model (if not in checkpoint)')
    parser.add_argument('--prototype_shape', type=int, nargs=4, required=False,
                        help='shape of prototypes (num_prototypes, channels, h, w), if not in checkpoint')
    parser.add_argument('--prototype_activation_function', type=str, default='log',
                        help='prototype activation function (if not in checkpoint)')
    parser.add_argument('--add_on_layers_type', type=str, required=False,
                        help='add‐on layers type (if not in checkpoint)')

    args = parser.parse_args()
    main(args)
