import os
import glob
import shutil
import gdown
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image

CUB200_URL = 'https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'  # Data taken from https://www.vision.caltech.edu/datasets/cub_200_2011/
CELEB_A_HQ_URL = 'https://drive.google.com/uc?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv'  # Data taken from http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_FOLDER = f'{PROJECT_ROOT}/datasets'


def download_dataset(url: str, filename: str):
    """Download dataset from gdrive."""
    out_path = os.path.join(DATASETS_FOLDER, os.path.splitext(filename)[0], filename)
    if not os.path.exists(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gdown.download(url, out_path)
    return out_path


def unpack_dataset(filepath: str):
    """Unpack dataset."""
    out_path = os.path.join(DATASETS_FOLDER, os.path.splitext(os.path.basename(filepath))[0], 'original')
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        gdown.extractall(filepath, out_path)
    return out_path


def generate_cub200():
    """Generate cub200 dataset."""
    print('Downloading dataset...')
    dataset_archive_path = download_dataset(CUB200_URL, 'cub200.tgz')
    dataset_path = os.path.dirname(dataset_archive_path)
    print('Unpacking folder...')
    original_path = unpack_dataset(dataset_archive_path)
    print('Generating parts locations...')
    part_names = pd.read_csv(f'{original_path}/CUB_200_2011/parts/parts.txt', header=None)[0].str.split(' ', n=1, expand=True).rename(columns={0: 'part_id', 1: 'part_name'}).astype({'part_id': int})
    part_locs = pd.read_csv(f'{original_path}/CUB_200_2011/parts/part_locs.txt', header=None, names=['image_id', 'part_id', 'x', 'y', 'visible'], sep=' ')
    bboxes = pd.read_csv(f'{original_path}/CUB_200_2011/bounding_boxes.txt', header=None, sep=' ', names=['image_id', 'bb_x', 'bb_y', 'bb_w', 'bb_h'])
    part_locs = part_locs[part_locs.visible == 1]  # Remove parts that are not visible
    part_locs = part_locs.merge(part_names, on='part_id').drop(['part_id', 'visible'], axis=1)
    part_locs = part_locs.merge(bboxes, on='image_id')
    part_locs['x'] -= part_locs['bb_x']  # Adjust x and y coordinates to be relative to the cropped bounding box
    part_locs['y'] -= part_locs['bb_y']
    part_locs = part_locs.drop(['bb_x', 'bb_y', 'bb_w', 'bb_h'], axis=1).astype({'x': int, 'y': int})
    part_locs.to_csv(f'{dataset_path}/part_locs.csv', index=False)
    print('Cropping images...')
    img_count = 0
    with open(f'{original_path}/CUB_200_2011/images.txt') as f:
        for line in f: img_count += 1
    with open(f'{original_path}/CUB_200_2011/images.txt') as images_file, \
         open(f'{original_path}/CUB_200_2011/bounding_boxes.txt') as bboxes_file, \
         open(f'{original_path}/CUB_200_2011/train_test_split.txt') as split_file:
        for images_line, bboxes_line, split_line in tqdm(zip(images_file, bboxes_file, split_file), total=img_count):
            # Read lines
            id1, path = images_line.strip().split(' ')
            id2, x, y, w, h = [int(float(x)) for x in bboxes_line.strip().split(' ')]
            id3, is_training = split_line.strip().split(' ')
            if int(id1) != int(id2) or int(id1) != int(id3):
                raise ValueError(f'ids in images.txt and bounding_boxes.txt do not match for {id1}, {id2} and {id3}.')
            # Crop image
            img = Image.open(f'{original_path}/CUB_200_2011/images/{path}')
            cropped = img.crop((x, y, x + w, y + h))
            # Save image
            class_folder, _ = path.split('/')
            out_path = f'{dataset_path}/{"train" if bool(int(is_training)) else "test"}/{class_folder}'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            cropped.save(f'{out_path}/{id1}.jpg')

def generate_cub200_fromfile(dataset_archive_path):
    """Generate cub200 dataset."""
    print('Downloading dataset...')
    dataset_path = os.path.dirname(dataset_archive_path)
    print('Unpacking folder...')
    original_path = unpack_dataset(dataset_archive_path)
    print('Generating parts locations...')
    part_names = pd.read_csv(f'{original_path}/CUB_200_2011/parts/parts.txt', header=None)[0].str.split(' ', n=1, expand=True).rename(columns={0: 'part_id', 1: 'part_name'}).astype({'part_id': int})
    part_locs = pd.read_csv(f'{original_path}/CUB_200_2011/parts/part_locs.txt', header=None, names=['image_id', 'part_id', 'x', 'y', 'visible'], sep=' ')
    bboxes = pd.read_csv(f'{original_path}/CUB_200_2011/bounding_boxes.txt', header=None, sep=' ', names=['image_id', 'bb_x', 'bb_y', 'bb_w', 'bb_h'])
    part_locs = part_locs[part_locs.visible == 1]  # Remove parts that are not visible
    part_locs = part_locs.merge(part_names, on='part_id').drop(['part_id', 'visible'], axis=1)
    part_locs = part_locs.merge(bboxes, on='image_id')
    part_locs['x'] -= part_locs['bb_x']  # Adjust x and y coordinates to be relative to the cropped bounding box
    part_locs['y'] -= part_locs['bb_y']
    part_locs = part_locs.drop(['bb_x', 'bb_y', 'bb_w', 'bb_h'], axis=1).astype({'x': int, 'y': int})
    part_locs.to_csv(f'{dataset_path}/part_locs.csv', index=False)
    print('Cropping images...')
    img_count = 0
    with open(f'{original_path}/CUB_200_2011/images.txt') as f:
        for line in f: img_count += 1
    with open(f'{original_path}/CUB_200_2011/images.txt') as images_file, \
         open(f'{original_path}/CUB_200_2011/bounding_boxes.txt') as bboxes_file, \
         open(f'{original_path}/CUB_200_2011/train_test_split.txt') as split_file:
        for images_line, bboxes_line, split_line in tqdm(zip(images_file, bboxes_file, split_file), total=img_count):
            # Read lines
            id1, path = images_line.strip().split(' ')
            id2, x, y, w, h = [int(float(x)) for x in bboxes_line.strip().split(' ')]
            id3, is_training = split_line.strip().split(' ')
            if int(id1) != int(id2) or int(id1) != int(id3):
                raise ValueError(f'ids in images.txt and bounding_boxes.txt do not match for {id1}, {id2} and {id3}.')
            # Crop image
            img = Image.open(f'{original_path}/CUB_200_2011/images/{path}')
            cropped = img.crop((x, y, x + w, y + h))
            # Save image
            class_folder, _ = path.split('/')
            out_path = f'{dataset_path}/{"train" if bool(int(is_training)) else "test"}/{class_folder}'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            cropped.save(f'{out_path}/{id1}.jpg')

def _get_img_part_loc(filepath):
    """Get part location centroid from a given binary mask."""
    filename, _ = os.path.basename(filepath).split('.')
    image_id, part_name = filename.split('_', maxsplit=1)
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if np.any(mask > 0):
        y, x = np.mean(np.nonzero(mask), axis=1).astype(int)  # Compute mask centroid
    else:
        y, x = np.nan, np.nan
    return (int(image_id), x, y, part_name)

def generate_celeb_a():
    """Generate celeb_a dataset."""
    print('Downloading dataset...')
    dataset_archive_path = download_dataset(CELEB_A_HQ_URL, 'celeb_a.zip')
    dataset_path = os.path.dirname(dataset_archive_path)
    print('Unpacking folder...')
    original_path = unpack_dataset(dataset_archive_path)
    print('Generating parts locations...')
    masks_filepaths = glob.glob(f'{original_path}/CelebAMask-HQ//CelebAMask-HQ-mask-anno/*/*.png')
    with mp.Pool(mp.cpu_count()) as pool:
        part_locs = list(tqdm(pool.imap_unordered(_get_img_part_loc, masks_filepaths), total=len(masks_filepaths)))
    part_locs = pd.DataFrame(part_locs, columns=['image_id', 'x', 'y', 'part_name'])
    part_locs[['x', 'y']] *= 2  # Rescale locations since annotated masks are 512x512, images are 1024x1024
    part_locs = part_locs[~part_locs[['x', 'y']].isna().any(axis=1)].astype({'x': int, 'y': int})  # Remove parts with NaN location
    part_locs.to_csv(f'{dataset_path}/part_locs.csv', index=False)
    print('Generating attributes subsplits...')
    attributes = pd.read_csv(f'{original_path}/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt', skiprows=1, sep='\s+')
    attributes = (attributes == 1)  # Convert to boolean
    test_split_fraction = 0.3
    subplits = {
        'hair': dict(cols=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Bald']),
        'attractive': dict(col='Attractive', other='Not_Attractive'),
        'young': dict(col='Young', other='Old'),
        'gender': dict(col='Male', other='Female'),
        'smiling': dict(col='Smiling', other='Not_Smiling'),
        'makeup': dict(col='Heavy_Makeup', other='No_Makeup'),
        'multi': ['hair', 'young', 'gender', 'makeup'], # Combination of attributes
    }
    rng = np.random.default_rng(0)
    for subsplit_name, values in subplits.items():
        # Generate dataset root folder
        subsplit_path = os.path.join(dataset_path, subsplit_name)
        if os.path.exists(subsplit_path):
            shutil.rmtree(subsplit_path)
        os.makedirs(subsplit_path)
        # Handle multi-attributes cases
        if isinstance(values, list):
            groups = [subplits[prop] for prop in values]
        else:
            groups = [values]
        # Generate dataset
        final_filtered_attributes = None
        for properties in groups:
            if 'cols' in properties:  # Multi-class case
                filtered_attributes = attributes[properties['cols']]
                filtered_attributes = filtered_attributes[filtered_attributes.sum(axis=1) == 1]  # Discard cases where more than one class is true
            else:  # Binary-class case
                filtered_attributes = attributes[properties['col']].to_frame()
                filtered_attributes[properties['other']] = ~attributes[properties['col']]
            filtered_attributes = filtered_attributes.idxmax(axis=1).rename('class_name').to_frame()  # Convert from one-hot encoding to class names
            if final_filtered_attributes is None:
                final_filtered_attributes = filtered_attributes
            else:
                final_filtered_attributes['class_name'] += '-' + filtered_attributes['class_name']
        final_filtered_attributes = final_filtered_attributes.dropna()
        is_training = rng.choice([True, False], size=len(final_filtered_attributes), p=[1 - test_split_fraction, test_split_fraction])
        final_filtered_attributes['is_training'] = is_training
        # Create dataset subsplit as hard-links of original images to save space
        os.link(f'{dataset_path}/part_locs.csv', f'{subsplit_path}/part_locs.csv')
        for sample in tqdm(final_filtered_attributes.itertuples(), desc=f'Generating "{subsplit_name}" subsplit', total=len(final_filtered_attributes)):
            sample_path = f'{subsplit_path}/{"train" if sample.is_training else "test"}/{sample.class_name}/{sample.Index}'
            if not os.path.exists(os.path.dirname(sample_path)):
                os.makedirs(os.path.dirname(sample_path))
            os.link(f'{original_path}/CelebAMask-HQ/CelebA-HQ-img/{sample.Index}', sample_path)


parser = argparse.ArgumentParser(description='Download and prepare the dataset')
parser.add_argument('dataset', type=str, choices=['cub200', 'celeb_a', 'cub200_fromfile'], help='The dataset to download %(choices)s')

if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Generating "{args.dataset}" dataset...')
    if args.dataset == 'cub200':
        generate_cub200()
    elif args.dataset == 'cub200_fromfile':
        generate_cub200_fromfile('data/cub200.tgz')
    elif args.dataset == 'celeb_a':
        generate_celeb_a()
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
