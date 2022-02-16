
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
from utils import accuracy
from basic.evaluate import equal_error_rate

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses,  la=20, r=1, easy_margin=False, weight_lambda=0.03, gamma=0.1, tau=0.2, K=5,margin=0.01, scale=15, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        dim=nOut
        self.C = nClasses
        self.N = K
        self.r = r
        self.m = margin
        self.s = scale
        self.weight_lambda = weight_lambda
        self.instance_label = torch.tensor(np.repeat(np.arange(self.C), self.N)).cuda()
        self.y_instance_onehot = self.to_one_hot(self.instance_label, n_dims=self.C).cuda()
        self.class_label = torch.tensor(np.repeat(np.arange(self.C), 1)).cuda()
        self.centers = torch.nn.Parameter(torch.FloatTensor(nOut, self.C*self.N), requires_grad=True)
        nn.init.xavier_normal_(self.centers, gain=1)
        self.ce = nn.CrossEntropyLoss()
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        return

    def to_one_hot(self, y, n_dims=None):
        y_tensor = y.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot

    def scale_mask_softmax(self,tensor,mask,softmax_dim,scale=1.0):
        scale_mask_exp_tensor = torch.exp(tensor* scale) * mask.detach()
        scale_mask_softmax_tensor = scale_mask_exp_tensor / (1e-8 + torch.sum(scale_mask_exp_tensor, dim=softmax_dim)).unsqueeze(softmax_dim)
        return scale_mask_softmax_tensor

    def forward(self, input, target):
        centers = F.normalize(self.centers, p=2, dim=0)
        input = F.normalize(input, p=2, dim=1)
        similarity= input.matmul(centers)
        #Masking of similar relationships
        positive_mask=torch.eq(target.view(-1,1).cuda()-self.instance_label.view(1,-1),0.0).float().cuda() #obtain positive mask
        topk = math.ceil(self.r * self.C * self.N)
        _, indices = torch.topk(similarity + positive_mask, topk, dim=1)
        mask = torch.zeros_like(similarity)
        mask = mask.scatter(1, indices, 1)
        prob_a =mask*similarity
        #Reverse Label Propagation
        logits=torch.matmul(prob_a , self.y_instance_onehot)
        y_target_onehot = self.to_one_hot(target, n_dims=self.C).cuda()
        logits_mask=1-torch.eq(logits,0.0).float().cuda()
        predict=self.scale_mask_softmax(logits, logits_mask,1).cuda()
        #Margin-Based Optimization
        cosine = predict
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        output_tem = output.detach().cpu()
        label_tem = target.detach().cpu()
        loss = self.ce(output, target)
        eer = equal_error_rate(label_tem.numpy(), output_tem.numpy())
        prec1, _    = accuracy(output_tem,label_tem, topk=(1, 5))
        if self.weight_lambda  > 0:
            simCenter = centers.t().matmul(centers)
            centers_logits = torch.matmul(simCenter , self.y_instance_onehot)
            y_target_onehot = self.to_one_hot(self.instance_label, n_dims=self.C).cuda()
            logits_mask=1-torch.eq(centers_logits,0.0).float().cuda()
            predict=self.scale_mask_softmax(centers_logits, logits_mask,1).cuda()
            cosine = predict
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, self.instance_label.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
            reg=self.ce(output, self.instance_label)
            return loss+self.weight_lambda*reg, prec1,eer
        else:
            return loss, prec1, eer
