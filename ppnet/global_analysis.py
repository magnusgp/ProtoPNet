import os
import re
from argparse import Namespace
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .helpers import makedir
from .find_nearest import find_k_nearest_patches_to_prototypes
from .log import create_logger
from .preprocess import preprocess_input_function


def save_prototype_original_img_with_bbox(load_img_dir, fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end,
                                          color=(0, 255, 255), markers=None):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), str(index) + '_prototype-img-original.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    if markers is not None:
        for marker in markers:
            cv2.circle(p_img_bgr, (int(marker[0]), int(marker[1])), 5, (0, 0, 255), -1)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255

    plt.axis('off')
    plt.imsave(fname, p_img_rgb)


def run_analysis(args: Namespace):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    train_dir = os.path.join(args.dataset, 'train')
    test_dir = os.path.join(args.dataset, 'test')

    model_path = os.path.abspath(args.model)  # ./saved_models/vgg19/003/checkpoints/10_18push0.7822.pth
    model_base_architecture, experiment_run, _, model_name = re.split(r'\\|/', model_path)[-4:]
    start_epoch_number = int(re.search(r'\d+', model_name).group(0))
    if model_base_architecture == 'pruned_prototypes':
        model_base_architecture, experiment_run = re.split(r'\\|/', model_path)[-6:-4]
        model_name = f'pruned_{model_name}'

    # load the model
    save_analysis_path = os.path.join(args.out, model_base_architecture, experiment_run, model_name, 'global')
    makedir(save_analysis_path)
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'global_analysis.log'))
    
    log(f'\nLoad model from: {args.model}')
    log(f'Model epoch: {start_epoch_number}')
    log(f'Model base architecture: {model_base_architecture}')
    log(f'Experiment run: {experiment_run}\n')

    # Load the pickled model. Newer PyTorch versions default to weights-only loading
    # which may fail for older checkpoints that contain pickled module instances.
    # Try to allowlist the PPNet class for safe unpickling when available, otherwise
    # fall back to loading with weights_only=False (trusted checkpoint).
    try:
        from .model import PPNet as _PPNetClass
        import torch.serialization as _ts
        if hasattr(_ts, 'safe_globals'):
            with _ts.safe_globals([_PPNetClass]):
                ppnet = torch.load(args.model, weights_only=False, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            # Older torch may provide add_safe_globals or only allow weights_only flag
            if hasattr(_ts, 'add_safe_globals'):
                _ts.add_safe_globals([_PPNetClass])
            ppnet = torch.load(args.model, weights_only=False, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        # As a last resort, attempt to load with weights_only=False without modifications.
        # This can execute arbitrary code from the checkpoint — only proceed if the file
        # is trusted.
        log(f'Warning: safe unpickle attempt failed ({e}). Falling back to torch.load(weights_only=False).')
        ppnet = torch.load(args.model, weights_only=False)
    ppnet = ppnet.cuda() if torch.cuda.is_available() else ppnet
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet_multi.module.img_size
    batch_size = 100

    # train set: do not normalize
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)

    # test set: do not normalize
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)

    root_dir_for_saving_train_images = os.path.join(save_analysis_path, 'nearest_prototypes', 'train')
    root_dir_for_saving_test_images = os.path.join(save_analysis_path, 'nearest_prototypes', 'test')
    makedir(root_dir_for_saving_train_images)
    makedir(root_dir_for_saving_test_images)

    # save prototypes in original images
    try:
        load_img_dir = os.path.join(os.path.dirname(args.model), '..', 'img')
        assert os.path.exists(load_img_dir), f'Folder "{load_img_dir}" does not exist'
    except Exception as e:
        load_img_dir = "./img"
        assert os.path.exists(load_img_dir), f'Folder "{load_img_dir}" does not exist'
    prototype_info = np.load(os.path.join(load_img_dir, f'epoch-{start_epoch_number}', 'bb.npy'))

    for j in tqdm(range(ppnet.num_prototypes), desc='Saving learned prototypes'):
        makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
        makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
        save_prototype_original_img_with_bbox(
            load_img_dir=load_img_dir,
            fname=os.path.join(root_dir_for_saving_train_images, str(j), 'prototype_bbox.png'),
            epoch=start_epoch_number,
            index=j,
            bbox_height_start=prototype_info[j][1],
            bbox_height_end=prototype_info[j][2],
            bbox_width_start=prototype_info[j][3],
            bbox_width_end=prototype_info[j][4],
            color=(0, 255, 255)
        )
        save_prototype_original_img_with_bbox(
            load_img_dir=load_img_dir,
            fname=os.path.join(root_dir_for_saving_test_images, str(j), 'prototype_bbox.png'),
            epoch=start_epoch_number,
            index=j,
            bbox_height_start=prototype_info[j][1],
            bbox_height_end=prototype_info[j][2],
            bbox_width_start=prototype_info[j][3],
            bbox_width_end=prototype_info[j][4],
            color=(0, 255, 255)
        )

    log('\nSaving nearest prototypes of train set')
    find_k_nearest_patches_to_prototypes(
        dataloader=train_loader,  # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
        k=args.top_imgs + 1,
        preprocess_input_function=preprocess_input_function,  # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=log)
    log('\nSaving nearest prototypes of test set')
    find_k_nearest_patches_to_prototypes(
        dataloader=test_loader,  # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
        k=args.top_imgs,
        preprocess_input_function=preprocess_input_function,  # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        log=log)

    logclose()