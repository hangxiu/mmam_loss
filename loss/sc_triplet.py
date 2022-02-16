# Implementation of SoftTriple Loss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from utils import accuracy
from basic.evaluate import equal_error_rate


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses,  la=20, gamma=0.1, tau=0.2, K=10,margin=0.01, scale=15, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = nClasses
        self.K = K
        self.fc = Parameter(torch.Tensor(nOut, nClasses * K))
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        output = self.la*(simClass-marginM)
        output_tem = output.detach().cpu()
        label_tem = target.detach().cpu()
        prec1, _ = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1, 5))
        eer = equal_error_rate(label_tem.numpy(), output_tem.numpy())
        return lossClassify, prec1, eer