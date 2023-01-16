import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1. - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)

# def mixup_criterion(criterion, pred, y_a, y_b, lam, x1, x2, pred_mixed, y_mixed):
#     mixup = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
#     # mixup = lam * criterion(x1, y_a) + (1 - lam) * criterion(x2, y_b)
#     y_mixed = y_mixed.type(torch.LongTensor).cuda()
#     mixup2 = criterion(pred, y_mixed)

#     # mixup = criterion(pred, y)
#     # x1_crit = lam * criterion(x1, y_a)
#     # x2_crit = (1. - lam) * criterion(x2, y_b)
#     # hinge = max(0, (mixup - x1_crit - x2_crit))
#     hinge  = max(0, (mixup2 - mixup))
#     return mixup + hinge
#     # return mixup2


# def mixup_criterion(criterion, pred, y_a, y_b, lam, pred_mixed, y_mixed, y, gamma=1.0, rho=0.0):
#     loss_mixup = lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b) # lamda * L(f(x'),y1) + (1-lamda)* L(f(x'),y2)
#     y_mixed = y_mixed.type(torch.LongTensor).cuda()
#     label_mixup = criterion(pred_mixed, y_mixed) # L(f(x'),y')
#     sup = criterion(pred, y) # L(f(x), y)
#     hinge  = max(0, (label_mixup - loss_mixup))
#     # correct term: return sup + mixup
#     return rho * sup + label_mixup + gamma * hinge


# def mixup_criterion(criterion, pred, y_a, y_b, lam, pred_mixed, y_mixed, y, x1, x2, gamma=0.9, rho=0.0, mu=0.5):
#     # lamda * L(f(x'),y1) + (1-lamda)* L(f(x'),y2)
#     y_a = y_a.type(torch.FloatTensor).cuda()
#     y_b = y_b.type(torch.FloatTensor).cuda()
#     y_mixed = y_mixed.type(torch.FloatTensor).cuda()
#     loss_mixup = lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b) 
#     # L(f(x'),y')
#     label_mixup = criterion(pred_mixed, y_mixed)
#     # L(f(x), y)
#     sup = criterion(pred, y) 
#     # new term
#     l2_dist = (y_mixed - y_b).pow(2).sum(dim=1).sqrt()
#     dist = ((lam*(1.0-lam)*mu)/2) * l2_dist 
#     hinge = max(torch.Tensor([0.0]).cuda().requires_grad_(True), (label_mixup - loss_mixup + dist[0]))

#     return label_mixup + gamma*hinge

# def mixup_criterion2(criterion, pred, y_a, y_b, lam, pred_mixed, y_mixed, y, x1, x2, gamma=0.9, rho=0.0, mu=0.5):
#     # lamda * L(f(x'),y1) + (1-lamda)* L(f(x'),y2)
#     y_a = y_a.type(torch.FloatTensor).cuda()
#     y_b = y_b.type(torch.FloatTensor).cuda()
#     y_mixed = y_mixed.type(torch.FloatTensor).cuda()
#     y1_mixup = criterion(pred_mixed, y_a)
#     y2_mixup = criterion(pred_mixed, y_b)
#     # L(f(x'),y')
#     label_mixup = criterion(pred_mixed, y_mixed)
#     # L(f(x), y)
#     sup = criterion(pred, y) 
#     # new term
#     l2_dist = (y_mixed - y_a).pow(2).sum(dim=1).sqrt()
#     dist = ((lam*(1.0-lam)*mu)/2) * l2_dist 

#     hinge = max(0, (label_mixup - y1_mixup + (mu/2)*dist[0]))
#     return sup + gamma*hinge
    
# class LabelAwareSmoothing(nn.Module):
#     def __init__(self, cls_num_list, smooth_head, smooth_tail, shape='concave', power=None):
#         super(LabelAwareSmoothing, self).__init__()

#         n_1 = max(cls_num_list)
#         n_K = min(cls_num_list)

#         if shape == 'concave':
#             self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

#         elif shape == 'linear':
#             self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

#         elif shape == 'convex':
#             self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

#         elif shape == 'exp' and power is not None:
#             self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

#         self.smooth = torch.from_numpy(self.smooth)
#         self.smooth = self.smooth.float()
#         if torch.cuda.is_available():
#             self.smooth = self.smooth.cuda()

#     def forward(self, x, target):
#         smoothing = self.smooth[target]
#         confidence = 1. - smoothing
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = confidence * nll_loss + smoothing * smooth_loss

#         return loss.mean()


class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x