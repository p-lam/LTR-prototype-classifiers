import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pprint
import math
import pandas as pd 
import torchvision.transforms as transforms 

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
from torch.optim import lr_scheduler 

from functools import partial
from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
from datasets.places import Places_LT
from datasets.imagenet import ImageNet_LT
from datasets.ina2018 import iNa2018
from ray.tune import CLIReporter
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

from models import resnet
from models import resnet_places
from models import resnet_cifar
from models import vgg 

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours 
import ray 
from ray import tune 
from ray.air import session 
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler

from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration
from lda import LDA
from vos import vos_sampling_step

from methods import mixup_data, mixup_criterion
from sklearn.mixture import GaussianMixture 
from sklearn.model_selection import GridSearchCV


param_grid = {
    "n_components": 30,
    "covariance_type": ["tied"], # "spherical", "tied", "diag", "full"
    "random_state": 0
}


def gmm_bic_score(estimator, X):
    return -estimator.bic(X)

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

    # hacky way of combining model + classifier
    full_model = RNClassifier(model, classifier)
    full_model = full_model.cuda() 

    # # load data with instance sampling
    # train_loader = dataset.train_instance
    # val_loader = dataset.eval

    # criterion = nn.CrossEntropyLoss().cuda()

    # train/val loop
    # def train_evaluate(parametrization):
    #     trained_net = train(train_loader, full_model, criterion, config, parameters=parametrization)
    #     return evaluate(net=trained_net, data_loader=val_loader, dtype=torch.float, device='cuda')

    # best_params, values, experiment, model = optimize(parameters=[{"name": "weight_decay", "type": "range", "bounds":[1e-3, 1e-2]}],
    #                                                 evaluation_function=train_evaluate,
    #                                                 objective_name='accuracy')
    
    algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    # scheduler = AsyncHyperBandScheduler(metric="loss", mode="min")
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        tune.with_parameters(train, config=config),
        # tune_config = tune.TuneConfig(
        #     metric="mean_accuracy",
        #     mode="min",
        #     search_alg=algo,
        #     scheduler=scheduler
        # ),
        resources_per_trial={"cpu": 16, "gpu": 1},
        config={"wd": tune.uniform(5e-4, 5e-2)},
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best hyperparameters found were: ", result.best_result().config)


def train(cfg, config):
    # switch to train mode
    if config.dataset == 'cifar100':
        m = getattr(resnet_cifar, config.backbone)()
        classifier = getattr(resnet_cifar, 'Classifier')(feat_in=64, num_classes=config.num_classes)
        dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers)
    torch.cuda.set_device(config.gpu)
    m = m.cuda(config.gpu)
    classifier = classifier.cuda(config.gpu)
    
    model = RNClassifier(m, classifier)
    model = model.cuda() 

    # load data with instance sampling
    train_loader = dataset.train_instance
    val_loader = dataset.eval

    criterion = nn.CrossEntropyLoss().cuda()
    
    model.train()
    total_num, total_loss = 0.0, 0.0
    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)

    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                    momentum=config.momentum, weight_decay=cfg["wd"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs, eta_min=0)

    # end = time.time()
    for epoch in range(config.num_epochs):
        for i, (images, target) in enumerate(train_loader):
            if i > end_steps:
                break
            if torch.cuda.is_available():
                images = images.cuda(config.gpu, non_blocking=True)
                target = target.cuda(config.gpu, non_blocking=True)
                output = model(images)
                loss = criterion(output, target)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        train_acc1, acc5 = accuracy(output, target, topk=(1, 5))
        total_num += train_loader.batch_size 
        total_loss += loss.item() * train_loader.batch_size 

        # val_loss, val_acc = validate(val_loader, model, criterion, config, block=None)
        """
        Validation code 
        """
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        # top5 = AverageMeter('Acc@5', ':6.3f')
        progress = ProgressMeter(
            len(val_loader),
            [losses, top1],
            prefix='Eval: ')

        # switch to evaluate mode
        model.eval()
        # classifier.eval()
        total_loss, total_num = 0.0, 0

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if config.gpu is not None:
                    images = images.cuda(config.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(config.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                total_num += val_loader.batch_size 
                total_loss += loss.item() * val_loader.batch_size 

                # measure accuracy and record loss
                val_acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(val_acc1[0], images.size(0))
                # top5.update(acc5[0], images.size(0))

            ret_loss = total_loss / total_num

        """
        Raytune logging
        """
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=ret_loss, accuracy=top1.avg)    


class RNClassifier(nn.Module):
    def __init__(self, model, classifier) -> None:
        super(RNClassifier, self).__init__()
        self.model = model
        self.classifier = classifier
    
    def forward(self, x):
        return self.classifier(self.model(x))


if __name__ == '__main__':
    ray.init(num_gpus=1, num_cpus=16)
    main()
