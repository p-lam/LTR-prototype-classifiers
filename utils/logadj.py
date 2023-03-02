import os
import torch
import numpy as np
from torch.utils.data import DataLoader

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, labels):
    """Computes accuracy for given outputs and ground truths"""

    _, predicted = torch.max(outputs, 1)
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    return acc


def class_accuracy(test_loader, model, args):
    """ Computes the accuracy for each class"""

    classes = args.class_names
    num_class = len(args.class_names)
    with torch.no_grad():
        n_class_correct = [0 for _ in range(num_class)]
        n_class_samples = [0 for _ in range(num_class)]
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            output = model(images)

            if args.logit_adj_post:
                output = output - args.logit_adjustments

            _, predicted = torch.max(output, 1)

            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        results = {}
        avg_acc = 0
        for i in range(num_class):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            avg_acc += acc
            results["class/" + classes[i]] = acc
        results["AA"] = avg_acc / num_class
        return results


def make_dir(log_dir):
    """ Makes a directory """

    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass


def log_hyperparameter(args, tro):
    """Logs the hyperparameter values"""

    whole_dict = vars(args)
    hyperparam = {}
    keys = ['logit_adj_post', 'logit_adj_train']
    for key in keys:
        hyperparam[key] = whole_dict[key]
    hyperparam['tro'] = tro * (hyperparam['logit_adj_post'] or hyperparam['logit_adj_train'])
    return hyperparam


def log_folders(args):
    """logs the folder"""

    log_dir = 'logs'
    exp_dir = 'dataset_{}_adjtrain_{}'.format(
        args.dataset,
        args.logit_adj_train)
    exp_loc = os.path.join(log_dir, exp_dir)
    model_loc = os.path.join(exp_loc, "model_weights")
    make_dir(log_dir)
    make_dir(exp_loc)
    make_dir(model_loc)
    return exp_loc, model_loc


def compute_adjustment(train_loader, tro):
    """compute the base probabilities"""

    label_freq = {}
    for i, (inputs, target) in enumerate(train_loader):
        target = target.cuda()
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()
    return adjustments
