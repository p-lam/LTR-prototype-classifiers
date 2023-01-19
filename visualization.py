import argparse
import os
import random
import warnings
import numpy as np
import pprint
import pandas as pd 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from modules.cb_loss import CB_loss
from sklearn.manifold import TSNE
from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
from datasets.places import Places_LT
from datasets.imagenet import ImageNet_LT
from datasets.ina2018 import iNa2018

from models import resnet
from models import resnet_places
from models import resnet_cifar
from models import vgg 

from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration
from lda import LDA
from vos import vos_sampling_step

from methods import mixup_data, mixup_criterion
from sklearn.mixture import GaussianMixture 
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-1)')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


best_acc1 = 0
its_ece = 100
t = 0.5

def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    if config.deterministic:
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, logger, model_dir)


def main_worker(gpu, ngpus_per_node, config, logger, model_dir):
    global best_acc1, its_ece
    config.gpu = gpu
#     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    if config.dataset == 'cifar10' or config.dataset == 'cifar100':
        model = getattr(resnet_cifar, config.backbone)()
        classifier = getattr(resnet_cifar, 'Classifier')(feat_in=64, num_classes=config.num_classes)

    elif config.dataset == 'imagenet' or config.dataset == 'ina2018':
        model = getattr(resnet, config.backbone)()
        classifier = getattr(resnet, 'Classifier')(feat_in=2048, num_classes=config.num_classes)

    elif config.dataset == 'places':
        model = getattr(resnet_places, config.backbone)(pretrained=True)
        classifier = getattr(resnet_places, 'Classifier')(feat_in=2048, num_classes=config.num_classes)
        block = getattr(resnet_places, 'Bottleneck')(2048, 512, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d)

    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            classifier.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[config.gpu])
            if config.dataset == 'places':
                block.cuda(config.gpu)
                block = torch.nn.parallel.DistributedDataParallel(block, device_ids=[config.gpu])
        else:
            model.cuda()
            classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier)
            if config.dataset == 'places':
                block.cuda()
                block = torch.nn.parallel.DistributedDataParallel(block)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        classifier = classifier.cuda(config.gpu)
        if config.dataset == 'places':
            block.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()
        if config.dataset == 'places':
            block = torch.nn.DataParallel(block).cuda()

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.resume, map_location=loc)
            # config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if config.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(config.gpu)
            model.load_state_dict(checkpoint['state_dict_model'])
            classifier.load_state_dict(checkpoint['state_dict_classifier'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    # Data loading code
    if config.dataset == 'cifar10':
        dataset = CIFAR10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                             batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'cifar100':
        dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'places':
        dataset = Places_LT(config.distributed, root=config.data_path,
                            batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'imagenet':
        dataset = ImageNet_LT(config.distributed, root=config.data_path,
                              batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'ina2018':
        dataset = iNa2018(config.distributed, root=config.data_path,
                          batch_size=config.batch_size, num_works=config.workers)

    train_loader = dataset.train_instance
    imgs, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            images = images.cuda()
            feats = model(images)
            # feats = images
            # feats = classifier(feats)
            imgs.append(feats.detach().cpu().numpy())
            labels.append(target.numpy())
    images = np.concatenate(imgs, axis=0)
    labels = np.concatenate(labels, axis=0)

    # filtered = [] 
    # cnt1, cnt2, cnt3 = 0, 0, 0 
    # for i, j in zip(images, labels):
    #     if j == 3:
    #         if cnt1 <= 100:
    #             filtered.append((i,j))
    #             cnt1 += 1 
    #     if j == 2:
    #         if cnt2 <= 100:
    #             filtered.append((i,j))
    #             cnt2 += 1
    #     if j == 7: 
    #         if cnt3 <= 100:
    #             filtered.append((i,j))
    #             cnt3 += 1 
    #     else:
    #         pass

    filtered = [(i,j) for i,j in zip(images, labels) if j == 3 or j == 2 or j == 7]
    images, labels = zip(*filtered)
    images = np.array(list(images))
    labels = np.array(list(labels))

    print(images.shape) # x_train
    print(labels.shape) # y_train 
    print(np.unique(labels, return_counts=True))

    # pca = PCA(n_components=10)
    # pca.fit(images.reshape((len(images), 32*32*3)))
    # pca.fit(images)
    # plt.figure(figsize=(8,8))
    # plt.scatter(x=pca.transform(images.reshape((len(images), 32*32*3)))[:,0],
    #             y=pca.transform(images.reshape((len(images), 32*32*3)))[:,1],
    #             c=labels.reshape(len(images)))
    # plt.scatter(x=pca.transform(images)[:,0],
    #             y=pca.transform(images)[:,1],
    #             c=labels.reshape(len(images)))
    # plt.title("PCA on raw values")
    # plt.colorbar()
    # plt.plot()
    # plt.show()

    x_train = images
    tsne_model = TSNE(n_components=2, random_state=0)
    # tsne = tsne_model.fit_transform(x_train.reshape((len(x_train),32*32*3)))
    tsne = tsne_model.fit_transform(x_train)
    x_tsne = tsne[:,0]
    y_tsne = tsne[:,1]
    # x_tsne = scale_to_01_range(x_tsne)
    # y_tsne = scale_to_01_range(y_tsne) 

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for label in np.unique(labels): 
        if label in [2, 3, 7]:
            indices = [i for i, l in enumerate(labels) if l == label]
            current_tx = np.take(x_tsne, indices)
            current_ty = np.take(y_tsne, indices)
            ax.scatter(current_tx, current_ty, label=label)
    ax.legend(loc='best')
    plt.show() 

    # print(x_tsne.shape, y_tsne.shape)
    # plt.figure(figsize=(16,16))
    # plt.scatter(x=x_tsne,y=y_tsne,c=labels.reshape(len(x_train)))
    # plt.title("t-sne on raw pixelvalues cifar10")
    # plt.colorbar()
    # plt.plot()
    # plt.show()

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


if __name__ == '__main__':
    main()
