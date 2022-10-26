import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import sys
import math

sys.path.append('./')
from utils import adjust_learning_rate

class RegularizedLogit(nn.Module):
    def __init__(self, W, features, criterion, labels, lamb, lr, wd, config):
        super(RegularizedLogit, self).__init__()
        self.W = W
        self.features = features
        self.criterion = criterion
        self.labels = labels
        self.lamb = lamb 
        self.config = config 
        self.lr = lr

        # self.W_norm = F.normalize(self.W, p=2.0, dim=1).detach().requires_grad_(True)
        self.optimizer = torch.optim.SGD([self.W], lr=self.lr, weight_decay=wd, momentum=0.9)
        self.grad = 0.0 
        self.loss = 0.0
        self.num_classes = self.W.shape[0]
        hidden_dim = self.W.shape[1]
        self.new_W = torch.zeros((self.num_classes, hidden_dim))
    
    def update_W(self, i, k):
        schedule=[35,45]

        self.adjust_learning_rate(self.optimizer, k, 50, schedule=schedule)

        max_val = self.compute_cos(i)

        # compute logits
        x = torch.einsum('ij,jk->ik', self.features, self.W.T)
        sup_loss = self.criterion(x, self.labels)
        loss = (self.lamb / 2) * max_val + sup_loss

        loss.sum().backward()
        self.loss = loss 
        
        for j in range(self.num_classes):
            if j != i:
                self.W.grad[j].zero_()

        self.optimizer.step()
        self.grad = self.W.grad[i]

        return self.W, self.loss, self.grad

    def compute_cos(self, idx):
        # take max value in row that is not on diagonal of matrix
        with torch.no_grad():
            dist_mat = torch.einsum('ij,jk->ik', self.W, self.W.T)
            mask = torch.ones(dist_mat.shape, dtype=bool)
            mask.fill_diagonal_(0)
            dist_row, mask_row = dist_mat[idx], mask[idx]
            max_val = dist_row[mask_row].max()

        return max_val
    # cosine annealing lr scheduler for training
    def adjust_learning_rate(self,optimizer, epoch, w_epochs, schedule,cos=False):
        """
        Decay the learning rate based on schedule
        """
        lr = self.lr
        if cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / w_epochs))
        else:  # stepwise lr schedule
            for milestone in schedule:
                if epoch >= milestone:
                    lr *= 0.1 
                else:
                    lr *= 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def forward(self, i, k):
        W, loss, grad = self.update_W(i, k)
        return W, loss, grad 

