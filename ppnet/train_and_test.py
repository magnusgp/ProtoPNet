import os
import re
import time
from tqdm import tqdm
from argparse import Namespace
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets


from .helpers import set_seed, pairwise_dist
from . import model, push, save
from .log import create_logger
from .preprocess import mean, std, preprocess_input_function


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True, min_prototypes_dist=0.1,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_diversity_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in enumerate(tqdm(dataloader)):
        input = image.cuda() if torch.cuda.is_available() else image
        target = label.cuda() if torch.cuda.is_available() else label

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda() if torch.cuda.is_available() else torch.t(model.module.prototype_class_identity[:, label])
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate prototypes diversity cost
                prototypes_pairwise_dist = pairwise_dist(model.module.prototype_vectors.squeeze(), squared=True)
                prototypes_pairwise_dist = torch.clamp(min_prototypes_dist - prototypes_pairwise_dist, min=0)  # Kepp only distances lower than `min_prototypes_dist
                prototypes_pairwise_dist = prototypes_pairwise_dist * (1 - torch.eye(model.module.prototype_shape[0], device=prototypes_pairwise_dist.device))  # Remove diagonal values
                diversity_cost = torch.sum(prototypes_pairwise_dist) / 2  # Sum up and divide by 2 because each distance is counted twice

                # calculate avg sepration cost
                avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)         

                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda() if torch.cuda.is_available() else 1 - torch.t(model.module.prototype_class_identity)
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_diversity_cost += diversity_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['diversity'] * diversity_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0:.2f}'.format(end - start))
    log('\tcross ent: \t{0:.5f}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0:.5f}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0:.5f}'.format(total_separation_cost / n_batches))
        log('\tdiversity:\t{0:.5f}'.format(total_diversity_cost / n_batches))
        log('\tavg separation:\t{0:.5f}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0:.5f}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0:.2f}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(pairwise_dist(p, squared=False))
    log('\tavg proto dist:\t{0:.5f}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, min_prototypes_dist=0.1, coefs=None, log=print):
    assert(optimizer is not None)

    log('train')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, min_prototypes_dist=min_prototypes_dist, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('test')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tjoint')




def run_training(args: Namespace):
    # Set default values
    prototype_shape = (args.num_prototypes, 128, 1, 1)
    joint_optimizer_lrs = dict(
        features=1e-4,
        add_on_layers=3e-3,
        prototype_vectors=3e-3
    )
    warm_optimizer_lrs = dict(
        add_on_layers=3e-3,
        prototype_vectors=3e-3
    )
    last_layer_optimizer_lr = 1e-4
    coefs = dict(
        crs_ent=1,
        clst=0.8,
        sep=-0.08,
        diversity=args.diversity_coeff,
        l1=1e-4,
    )

    # Init environment
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('GPUs:', os.environ['CUDA_VISIBLE_DEVICES'])

    base_architecture_type = re.match('^[a-z]*', args.architecture).group(0)
    model_dir = os.path.join('./saved_models', args.architecture, args.exp_name)
    if os.path.exists(model_dir):
        print(f'Warning: model directory "{args.exp_name}" already exists, overwriting...')
    else:
        os.makedirs(model_dir)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    with open(os.path.join(model_dir, 'args.yaml'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    img_dir = os.path.join(model_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # load the data
    train_dir = os.path.join(args.dataset, 'train')
    test_dir = os.path.join(args.dataset, 'test')

    normalize = T.Normalize(mean=mean, std=std)

    # all datasets
    # train set
    train_dataset = datasets.ImageFolder(
        train_dir,
        T.Compose([
            T.RandomAffine(degrees=20, shear=15),
            T.RandomPerspective(distortion_scale=0.25, p=0.25),
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(size=(args.img_size, args.img_size)),
            T.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)
    # push set
    train_push_dataset = datasets.ImageFolder(
        train_dir,
        T.Compose([
            T.Resize(size=(args.img_size, args.img_size)),
            T.ToTensor(),
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        T.Compose([
            T.Resize(size=(args.img_size, args.img_size)),
            T.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False)

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(args.batch_size))

    # construct the model
    ppnet = model.construct_PPNet(base_architecture=args.architecture,
                                  pretrained=True, img_size=args.img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=len(train_dataset.classes),
                                  prototype_activation_function=args.prototype_activation_function,
                                  add_on_layers_type=args.add_on_layers)
    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda() if torch.cuda.is_available() else ppnet
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # define optimizer
    joint_optimizer_specs = \
        [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},  # bias are now also being regularized
         {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
            {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
         ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=args.step_size, gamma=0.1)

    warm_optimizer_specs = \
        [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
         ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # train the model
    log('\nStart training')
    for epoch in range(args.epochs + 1):
        log('epoch: \t{0}'.format(epoch))

        if epoch < args.warm_epochs:
            warm_only(model=ppnet_multi, log=log)
            _ = train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, min_prototypes_dist=args.min_diversity, coefs=coefs, log=log)
        else:
            joint(model=ppnet_multi, log=log)
            _ = train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                          class_specific=class_specific, min_prototypes_dist=args.min_diversity, coefs=coefs, log=log)
            joint_lr_scheduler.step()

        if epoch % args.test_interval == 0:
            accu = test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=f'{epoch:03d}nopush', accu=accu,
                                        target_accu=0.70, log=log, epoch=epoch)

        if epoch > 0 and epoch % args.push_interval == 0:
            push.push_prototypes(
                train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function,  # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,  # if not None, prototypes will be saved here
                epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            accu = test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=f'{epoch:03d}push', accu=accu,
                                        target_accu=0.70, log=log)

            if args.prototype_activation_function != 'linear':
                last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                  class_specific=class_specific, min_prototypes_dist=args.min_diversity, coefs=coefs, log=log)
                    accu = test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=f'{epoch:03d}_{i:02d}push', accu=accu,
                                                target_accu=0.70, log=log)
        log('------------------------------------------\n')
    logclose()
