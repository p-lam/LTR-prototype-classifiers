import torch
import torch.nn as nn
from hyptorch.nn import * 
from hyptorch.pmath import * 
        
class Pseudo_Huber(nn.Module):
    """Pseudo-Huber function"""

    def __init__(self, delta=1):
        super(Pseudo_Huber, self).__init__()
        self.delta = delta

    def forward(self, input):
        out = (input / self.delta) ** 2
        out = torch.sqrt(out + 1)
        out = self.delta * (out - 1)
        return out


class LearnedPrototypes(nn.Module):
    def __init__(
        self,
        model,
        n_prototypes,
        prototypes=None,
        squared=False,
        ph=None,
        dist="euclidean",
        device="cuda",
    ):
        super(LearnedPrototypes, self).__init__()
        self.model = model
        self.prototypes = (
           nn.Parameter(prototypes).requires_grad_(True)
        )
        self.n_prototypes = n_prototypes
        self.squared = squared
        self.dist = dist
        self.ph = None if ph is None else Pseudo_Huber(delta=ph)
        if self.dist == "hyperbolic":
            self.e2p = ToPoincare(c=0.2, clip_r=2.3, ball_dim=256, riemannian=False, train_c=True)
    def forward(self, x, temp):

        with torch.no_grad():
            embeddings = self.model(x)
     
        if self.dist == "cosine":
            dists = 1 - nn.CosineSimilarity(dim=-1)(
                embeddings[:, None, :], self.prototypes[None, :, :]
            )
        elif self.dist == "euclidean":
            # efficient solution for euclidean dist
            # goal: have n x m x d distances to embeddings
            n = embeddings.size(0)  # number of samples in batch
            d = embeddings.size(1)  # feature dim
            m = self.prototypes.size(1)  # number of class centroids
            # insert empty batch axis for prototypes and reshape
            proto = self.prototypes.T.unsqueeze(0).expand(n, m, d)
            # insert empty class axis for embeddings and reshape
            embed = embeddings.unsqueeze(1).expand(n, m, d)
            # dists = (embed @ proto.T) * temp * (embed @ proto.T)
            dists = torch.pow(embed - proto, exponent=2).sum(2).sqrt()
        # class dependent temp
        elif self.dist == "CDT":
            n = embeddings.size(0)  # number of samples in batch
            d = embeddings.size(1)  # feature dim
            m = self.prototypes.size(1)  # number of class centroids
            # insert empty batch axis for prototypes and reshape
            proto = self.prototypes.T.unsqueeze(0).expand(n, m, d)
            # insert empty class axis for embeddings and reshape
            embed = embeddings.unsqueeze(1).expand(n, m, d)
            # dists = (embed @ proto.T) * temp * (embed @ proto.T)
            diff = embed - proto
            dists = (torch.matmul(diff, temp) * diff).sum(2).sqrt()
        elif self.dist == "hyperbolic":
            emb = self.e2p(embeddings)
            proto = self.e2p(self.prototypes)
            # dists = dist_matrix(emb, proto.T, self.e2p.c)
            dists = dist_matrix(emb, proto.T, self.e2p.c)
        if self.squared:
            dists = dists ** 2
        
        return -(0.5) * dists