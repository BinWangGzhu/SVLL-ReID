import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothCELoss(_Loss):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1-self.smooth_ratio+self.v)

        loss = - torch.sum(F.log_softmax(input, 1) *
                           (one_hot.detach())) / input.size(0)
        return loss


def D(p, z):
    # [N, E]
    z = z.detach() # stop gradient
    p = p / p.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    # [N E] [N E] -> [N]
    return (p * z).sum(dim=1).mean() # dot product & batch coeff normalization

def D_minimize(p, z):  # ..., X, size; ..., Y, size; choose the minimize one
    z = z.detach()
    p = p / p.norm(dim=-1, keepdim=True)
    z = (z / z.norm(dim=-1, keepdim=True)).permute(0, 2, 1)
    sim = torch.bmm(p, z)
    return sim.max(dim=-1)[0].mean(dim=-1).mean()


class SimsiamLoss(nn.Module):
    def __init__(self, symmetry=True):
        super(SimsiamLoss, self).__init__()
        self.symmetry = symmetry

    def forward(self, outputs, minimize_loss=False,):
        p1 = outputs['aug1_embed']
        z1 = outputs['z1']
        p2 = outputs['aug2_embed']
        z2 = outputs['z2']
        if self.symmetry:
            if minimize_loss:
                D1 = D_minimize(p1, z2)
                D2 = D_minimize(p2, z1)
                # import ipdb
                # ipdb.set_trace()
                return -0.5 * (D1.mean() + D2.mean())
            else:
                D1 = D(p1, z2)
                D2 = D(p2, z1)
                return -0.5 * (D(p1, z2)  + D(p2, z1) )
