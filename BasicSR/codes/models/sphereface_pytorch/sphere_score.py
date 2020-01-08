'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import argparse
'''
import numpy as np
import models.sphereface_pytorch.net_sphere as net_sphere
#import net_sphere
import torch

trained_file = 'models/sphereface_pytorch/model/sphere20a.pth'
#trained_file = 'model/sphere20a.pth'

net = getattr(net_sphere,'sphere20a')()
net.load_state_dict(torch.load(trained_file))
net.cuda()
net.eval()
#for param in net.parameters():
#    param.requires_grad = False
net.feature = True

def batch_dot(a, b):
    return torch.sum(a*b, dim=-1)

def cos_d(f1, f2):
    return batch_dot(f1,f2)/(f1.norm(dim=-1)*f2.norm(dim=-1))

def sphereface(output, target):
    output_vec = net(output).data
    target_vec = net(target).data
    #f = output.data
    #f1,f2 = f[0],f[2]
    f1,f2 = output_vec, target_vec
    #cosdistance = batch_dot(f1,f2)/(f1.norm(dim=-1)*f2.norm(dim=-1)+1e-5)
    cosdistance = cos_d(f1, f2)
    return cosdistance

if __name__ == '__main__':
    v1 = torch.Tensor([[0,1],[1,1],[.5,0]])
    v2 = torch.Tensor([[0,1],[1,-1],[.5,.5]])
    
    print(batch_dot(v1, v2))
    print(cos_d(v1, v2))
