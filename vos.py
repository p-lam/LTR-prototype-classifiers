from typing import List
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance
import numpy as np
import torch

def fit_ccg(feat_train, labels_train, num_classes=100):
    """
    fit class conditional gaussian on given features/labels from training set
    """
    train_means = []
    train_feat_centered = []
    for i in tqdm(range(num_classes)):
        fs = feat_train[labels_train == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))

    means = torch.from_numpy(np.array(train_means)).cuda().float()
    prec = torch.from_numpy(ec.precision_).cuda().float()
    return means, prec

def sample_ccg(means, prec, class_index=0, num_samples=1000, likelihood="lowest"):
    """
    given parameters to ccg (per-class means and shared precision matrix)
    return num_samples lowest or highest likelihood samples from the ccg
    """
    if num_samples == 0:
        return []
    assert likelihood in ["lowest", "highest"]
    dist = torch.distributions.MultivariateNormal(loc=means[class_index], precision_matrix=prec)
    samples = dist.rsample((num_samples * 100,))
    log_prob = dist.log_prob(samples)
    
    # flip criterion if needed
    if likelihood == "lowest":
        log_prob = -log_prob

    # take num_samples out of whole
    topk_prob, indices = torch.topk(log_prob, num_samples)

    return samples[indices]

def vos_sampling_step(feat_train, label_train, num_samples_per_class: List, likelihood="lowest"):
    """
    Use this after stage 1 of training, or at least partway through stage 1
        (VOS does not make sense if clusters are not somewhat well defined)
    feat_train: np array of training features
    label_train: np array of associated labels (integers)
    num_samples_per_class: list of length n_classes telling how many of each class label to
        sample from synthetically
    likelihood: whether or not to sample features with high or low likelihood 
        VOS uses "lowest" to get synthetic outliers for each class
        in our application it might be interesting to try "highest" 
    returns:
        list of torch tensors containing the resampled features for each class
    """
    num_classes = len(num_samples_per_class)
    ccg_means, prec = fit_ccg(feat_train, label_train)

    sampled_data = []
    for c_idx in range(num_classes):
        samples = sample_ccg(ccg_means, prec, c_idx, num_samples=num_samples_per_class[c_idx], likelihood=likelihood)
        if len(samples) > 0:
            sampled_data.append(samples)
    
    # the end result is a list of samples for each class in the datase
    return sampled_data

if __name__ == "__main__":
    # sample usage
    # params for generating synthetic training data (instead of cifar10 features)
    N = 50000
    FEAT_DIM = 32
    N_CLASSES = 10
    # params for resampling method
    N_SAMPLES_PER_CLASS = [100] * 10
    feat_train = np.random.normal(loc=0, scale=1, size=(50000, FEAT_DIM))
    labels_train = np.random.randint(low=0, high=N_CLASSES, size=N)

    # try out method, sampling lowest likelihood parts of each ccg
    resampled_data = vos_sampling_step(feat_train, labels_train, N_SAMPLES_PER_CLASS, likelihood="highest")

    for item in resampled_data:
        print(item.shape)