import os
import shutil
from argparse import Namespace
from collections import Counter
import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .helpers import makedir
from . import prune, find_nearest, train_and_test as tnt, save
from .log import create_logger
from .preprocess import mean, std, preprocess_input_function



def prune_prototypes(
        dataloader,
        prototype_network_parallel,
        k,
        prune_threshold,
        preprocess_input_function,
        original_model_dir,
        epoch_number,
        log=print,
        copy_prototype_imgs=True
    ):
    
    # run global analysis
    nearest_train_patch_class_ids = find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=dataloader,
        prototype_network_parallel=prototype_network_parallel,
        k=k,
        preprocess_input_function=preprocess_input_function,
        full_save=False,
        log=log
    )

    # find prototypes to prune
    original_num_prototypes = prototype_network_parallel.module.num_prototypes

    prototypes_to_prune = []
    for j in range(prototype_network_parallel.module.num_prototypes):
        class_j = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
        nearest_train_patch_class_counts_j = Counter(nearest_train_patch_class_ids[j])
        # if no such element is in Counter, it will return 0
        if nearest_train_patch_class_counts_j[class_j] < prune_threshold:
            prototypes_to_prune.append(j)

    log('k = {}, prune_threshold = {}'.format(k, prune_threshold))
    log('{} prototypes will be pruned'.format(len(prototypes_to_prune)))

    # bookkeeping of prototypes to be pruned
    class_of_prototypes_to_prune = \
        torch.argmax(
            prototype_network_parallel.module.prototype_class_identity[prototypes_to_prune],
            dim=1).numpy().reshape(-1, 1)
    prototypes_to_prune_np = np.array(prototypes_to_prune).reshape(-1, 1)
    prune_info = np.hstack((prototypes_to_prune_np, class_of_prototypes_to_prune))
    pruned_dir = os.path.join(original_model_dir, '..', 'pruned_prototypes', f'epoch{epoch_number:03d}_k{k}_pt{prune_threshold}')
    makedir(pruned_dir)
    np.save(os.path.join(pruned_dir, 'prune_info.npy'), prune_info)

    # prune prototypes
    prototype_network_parallel.module.prune_prototypes(prototypes_to_prune)
    if copy_prototype_imgs:
        original_img_dir = os.path.join(original_model_dir, '..', 'img', 'epoch-%d' % epoch_number)
        dst_img_dir = os.path.join(pruned_dir, 'img', 'epoch-%d' % epoch_number)
        makedir(dst_img_dir)
        prototypes_to_keep = list(set(range(original_num_prototypes)) - set(prototypes_to_prune))

        for idx in range(len(prototypes_to_keep)):
            shutil.copyfile(src=os.path.join(original_img_dir, str(prototypes_to_keep[idx]) + '_prototype-img.png'),
                            dst=os.path.join(dst_img_dir, str(idx) + '_prototype-img.png'))

            shutil.copyfile(src=os.path.join(original_img_dir, str(prototypes_to_keep[idx]) + '_prototype-img-original.png'),
                            dst=os.path.join(dst_img_dir, str(idx) + '_prototype-img-original.png'))

            shutil.copyfile(src=os.path.join(original_img_dir, str(prototypes_to_keep[idx]) + '_prototype-img-original_with_self_act.png'),
                            dst=os.path.join(dst_img_dir, str(idx) + '_prototype-img-original_with_self_act.png'))

            shutil.copyfile(src=os.path.join(original_img_dir, str(prototypes_to_keep[idx]) + '_prototype-self-act.npy'),
                            dst=os.path.join(dst_img_dir, str(idx) + '_prototype-self-act.npy'))

            bb = np.load(os.path.join(original_img_dir, 'bb.npy'))
            bb = bb[prototypes_to_keep]
            np.save(os.path.join(dst_img_dir, 'bb.npy'), bb)

            bb_rf = np.load(os.path.join(original_img_dir, 'bb-receptive_field.npy'))
            bb_rf = bb_rf[prototypes_to_keep]
            np.save(os.path.join(dst_img_dir, 'bb-receptive_field.npy'), bb_rf)

    return prune_info



def run_pruning(args: Namespace):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    train_dir = os.path.join(args.dataset, 'train')
    test_dir = os.path.join(args.dataset, 'test')

    k = args.k_nearest

    model_path = os.path.abspath(args.model)
    original_model_dir = os.path.dirname(model_path)  # './saved_models/densenet161/003/'
    original_model_name = os.path.basename(model_path)  # '10_16push0.8007.pth'

    assert 'nopush' not in original_model_name, 'Pruning must happen after pushing prototypes'
    epoch = int(original_model_name.split('push')[0].split('_')[0])
    model_dir = os.path.join(original_model_dir, '..', 'pruned_prototypes', f'epoch{epoch:03d}_k{k}_pt{args.prune_threshold}')
    makedir(model_dir)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'prune.log'))

    ppnet = torch.load(args.model)
    ppnet = ppnet.cuda() if torch.cuda.is_available() else ppnet
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    img_size = ppnet_multi.module.img_size

    normalize = transforms.Normalize(mean=mean, std=std)

    # train set
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)

    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)

    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(args.batch_size))

    # push set: needed for pruning because it is unnormalized
    train_push_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)

    log('push set size: {0}'.format(len(train_push_loader.dataset)))

    tnt.test(model=ppnet_multi, dataloader=test_loader,
            class_specific=class_specific, log=log)

    # prune prototypes
    log('prune')
    prune.prune_prototypes(
        dataloader=train_push_loader,
        prototype_network_parallel=ppnet_multi,
        k=k,
        prune_threshold=args.prune_threshold,
        preprocess_input_function=preprocess_input_function,  # normalize
        original_model_dir=original_model_dir,
        epoch_number=epoch,
        log=log,
        copy_prototype_imgs=True
    )
    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                model_name=original_model_name.split('push')[0] + 'prune',
                                accu=accu,
                                target_accu=0.70, log=log)

    # last layer optimization
    if args.n_iter > 0:
        last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        coefs = {
            'crs_ent': 1,
            'clst': 0.8,
            'sep': -0.08,
            'l1': 1e-4,
            'diversity': 0
        }

        log('optimize last layer')
        tnt.last_only(model=ppnet_multi, log=log)
        for i in range(args.n_iter):
            log('iteration: \t{0}'.format(i))
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                        model_name=original_model_name.split('push')[0] + '_' + str(i) + 'prune',
                                        accu=accu,
                                        target_accu=0.70, log=log)
    logclose()
