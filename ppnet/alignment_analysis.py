import os
import re
import shutil
import cv2
import numpy as np
import pandas as pd
import glob
from argparse import Namespace
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

from .helpers import makedir, find_high_activation_crop
from .log import create_logger
from .global_analysis import save_prototype_original_img_with_bbox


def save_alignment_matrix(fname, alignment_matrix):
    fig, ax = plt.subplots(figsize=(5, 5))
    heatmap = alignment_matrix.replace(np.inf, np.nanmax(alignment_matrix[alignment_matrix < np.inf])).astype(float)
    heatmap = heatmap.div(heatmap.min(axis=1), axis=0)
    ax.imshow(heatmap, norm=LogNorm(), cmap='viridis_r')
    ax.set_yticks(range(len(alignment_matrix.index)))
    ax.set_yticklabels(alignment_matrix.index)
    ax.set_xticks(range(len(alignment_matrix.columns)))
    ax.set_xticklabels(alignment_matrix.columns)
    for i in range(len(alignment_matrix.index)):
        for j in range(len(alignment_matrix.columns)):
            txt = f'{alignment_matrix.iloc[i, j]:.0f}' if alignment_matrix.iloc[i, j] < np.inf else '-'
            ax.text(j, i, txt, ha='center', va='center', color='w', size='small')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fig.tight_layout()
    fig.savefig(fname + '.png')
    plt.close(fig)
    np.save(fname + '.npy', alignment_matrix)


def alignment_score(part_locs, high_act_patch_indices):
    """Compute alignemnt score between a vector fo part location and a bound box of an highly activated pathch."""
    high_act_y, high_act_x = np.mean(high_act_patch_indices[0:2], dtype=int), np.mean(high_act_patch_indices[2:4], dtype=int)
    dist = ((part_locs['x'] - high_act_x) ** 2 + (part_locs['y'] - high_act_y) ** 2) ** .5  # Euclidean distances
    return dist


def scale_part_locs(part_locs, img_paths, img_resize):
    """Scale part locations to the corresponding cropped image scale."""
    img_ids = []
    for img_path in tqdm(img_paths):
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])
        img_width, img_height = Image.open(img_path).size
        part_locs.loc[part_locs.image_id == img_id, 'x'] *= img_resize / img_width
        part_locs.loc[part_locs.image_id == img_id, 'y'] *= img_resize / img_height
        img_ids.append(img_id)
    part_locs = part_locs.astype({'x': int, 'y': int})
    assert np.all(part_locs.loc[part_locs.image_id.isin(img_ids), ['x', 'y']] <= img_resize), 'Part locations are outside of image boundaries'
    return part_locs, img_ids


def run_analysis(args: Namespace):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    dataset_split_path = os.path.join(args.dataset, 'test')

    model_path = os.path.abspath(args.model)  # ./saved_models/vgg19/003/checkpoints/10_18push0.7822.pth
    model_base_architecture, experiment_run, _, model_name = re.split(r'\\|/', model_path)[-4:]
    start_epoch_number = int(re.search(r'\d+', model_name).group(0))
    if model_base_architecture == 'pruned_prototypes':
        model_base_architecture, experiment_run = re.split(r'\\|/', model_path)[-6:-4]
        model_name = f'pruned_{model_name}'

    # load the model
    save_analysis_path = os.path.join(args.out, model_base_architecture, experiment_run, model_name, 'alignment')
    makedir(save_analysis_path)
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'alignment_analysis.log'))
    
    log(f'\nLoad model from: {args.model}')
    log(f'Model epoch: {start_epoch_number}')
    log(f'Model base architecture: {model_base_architecture}')
    log(f'Experiment run: {experiment_run}')
    log(f'Output path: {os.path.abspath(save_analysis_path)}\n')

    ppnet = torch.load(args.model)
    ppnet = ppnet.cuda() if torch.cuda.is_available() else ppnet
    ppnet.eval()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_resize = ppnet_multi.module.img_size
    batch_size = 100

    # Dataset with samples only from the target class
    log('Loading dataset...')
    dataset = datasets.ImageFolder(
        dataset_split_path,
        transforms.Compose([
            transforms.Resize(size=(img_resize, img_resize)),
            transforms.ToTensor(),
        ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # Load part locations
    log('Loading test split part locations...')
    part_locs = pd.read_csv(os.path.join(os.path.dirname(dataset_split_path), 'part_locs.csv'))
    part_locs, image_ids = scale_part_locs(part_locs, [img_path for img_path, _ in dataset.samples], img_resize)
    
    # Scale also training part location to visalize prototypes
    log('Loading train split part locations...')
    part_locs, train_img_ids = scale_part_locs(part_locs, [img_path for img_path in glob.glob(os.path.join(args.dataset, 'train', '*/*.jpg'))], img_resize)

    # Load prototypes info
    load_img_dir = os.path.join(os.path.dirname(args.model), '..', 'img')
    assert os.path.exists(load_img_dir), f'Folder "{load_img_dir}" does not exist'
    prototype_info = np.load(os.path.join(load_img_dir, f'epoch-{start_epoch_number}', 'bb.npy'))

    # Compute prototypes activations for the class images
    prototype_activation_patterns = []
    for (imgs, _) in tqdm(dataloader, desc=f'Computing prototypes activations'):
        imgs = imgs.cuda() if torch.cuda.is_available() else imgs
        _, distances = ppnet.push_forward(imgs)
        batch_prototype_activation_patterns = ppnet.distance_2_similarity(distances)
        if ppnet.prototype_activation_function == 'linear':
            max_dist = ppnet.prototype_shape[1] * ppnet.prototype_shape[2] * ppnet.prototype_shape[3]
            batch_prototype_activation_patterns = batch_prototype_activation_patterns + max_dist
        prototype_activation_patterns.append(batch_prototype_activation_patterns.detach().cpu().numpy())
    prototype_activation_patterns = np.concatenate(prototype_activation_patterns, axis=0)

    # Process each class
    for class_name in dataset.classes:
        class_dir = os.path.join(save_analysis_path, class_name)
        if os.path.exists(class_dir):
            shutil.rmtree(class_dir)
        makedir(class_dir)
        class_idx = dataset.class_to_idx[class_name]
        class_prototypes_indices = np.argwhere(prototype_info[:, -1] == class_idx).flatten()

        # Save prototypes for each class
        alignment_matrix = pd.DataFrame(0, index=class_prototypes_indices, columns=part_locs.part_name.unique())
        for i, prototype_index in enumerate(tqdm(class_prototypes_indices, desc=f'Saving learned prototypes for class "{class_name}"')):
            image_id = train_img_ids[prototype_info[prototype_index][0]]
            prototype_part_locs = part_locs.loc[part_locs.image_id == image_id].set_index('part_name')
            align_score = alignment_score(prototype_part_locs, prototype_info[prototype_index][1:5]).reindex(alignment_matrix.columns)
            alignment_matrix.loc[prototype_index, :] = align_score.fillna(np.inf)
            closest_part = align_score.idxmin()
            save_prototype_original_img_with_bbox(
                load_img_dir=load_img_dir,
                fname=os.path.join(class_dir, f'{i+1}_prototype_{prototype_index}_bbox.png'),
                epoch=start_epoch_number,
                index=prototype_index,
                bbox_height_start=prototype_info[prototype_index][1],
                bbox_height_end=prototype_info[prototype_index][2],
                bbox_width_start=prototype_info[prototype_index][3],
                bbox_width_end=prototype_info[prototype_index][4],
                color=(0, 255, 255),
                markers=prototype_part_locs.loc[[closest_part], ['x', 'y']].values.tolist()
            )
        save_alignment_matrix(os.path.join(class_dir, f'alignment_matrix_prototypes'), alignment_matrix)

        # Compute alignment matrix on class images
        class_image_ids = [(i, img_id) for i, img_id in enumerate(image_ids) if dataset.samples[i][1] == class_idx]
        alignment_matrix = pd.DataFrame(0, index=class_prototypes_indices, columns=part_locs.part_name.unique())
        alignment_counts = alignment_matrix.copy()
        for idx, image_id in tqdm(class_image_ids, desc=f'Computing alignment matrix for class "{class_name}"'):
            image_part_locs = part_locs[part_locs.image_id == image_id].set_index('part_name')
            for prototype_index in class_prototypes_indices:
                activation_pattern = prototype_activation_patterns[idx][prototype_index]
                upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_resize, img_resize), interpolation=cv2.INTER_CUBIC)
                high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
                # Compute alignment matrix
                align_score = alignment_score(image_part_locs, high_act_patch_indices).reindex(alignment_matrix.columns)
                alignment_matrix.loc[prototype_index, :] += align_score.fillna(0)
                alignment_counts.loc[prototype_index, :] += align_score.notna().astype(int)
        alignment_matrix /= alignment_counts
        alignment_matrix[alignment_counts == 0] = np.inf
        # Plot alignment matrix
        save_alignment_matrix(os.path.join(class_dir, f'alignment_matrix_samples'), alignment_matrix)
    logclose()
