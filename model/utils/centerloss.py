import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        embedding_dim (int): feature dimension.
    """
    def __init__(self, num_classes, embedding_dim=512, device=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.embedding_dim))

        if device is not None:
            self.centers = self.centers.to(device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, embedding_dim).
            labels: ground truth labels with shape (batch_size).
        """

        centers = self.centers.to(dtype=x.dtype, device=x.device)

        batch_size = x.size(0)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat = distmat.to(x.dtype)


        distmat.addmm_(1, -2, x, centers.t())

        classes = torch.arange(self.num_classes).long().to(labels.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss