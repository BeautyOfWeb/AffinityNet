import numpy as np
from PIL import Image
import os
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models 

def dist(params1, params2=None, dist_fn=torch.norm): #pylint disable=no-member
    """Calculate the norm of params1 or the distance between params1 and params2; 
        Common usage calculate the distance between two model state_dicts.
    Args:
        params1: dictionary; with each item a torch.Tensor
        params2: if not None, should have the same structure (data types and dimensions) as params1
    """
    if params2 is None:
        return dist_fn(torch.Tensor([dist_fn(params1[k]) for k in params1]))
    d = torch.Tensor([dist_fn(params1[k] - params2[k]) for k in params1])
    return dist_fn(d)
    
class AverageMeter(object):
    def __init__(self):
        self._reset()
    
    def _reset(self):
        self.val = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def pil_loader(path, format = 'RGB'):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(format)

class ImageFolder(data.Dataset):
    def __init__(self, root, imgs, transform = None, target_transform = None, 
                 loader = pil_loader, is_test = False):
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.is_test = is_test
    
    def __getitem__(self, idx):
        if self.is_test:
            img = self.imgs[idx]
        else:
            img, target = self.imgs[idx]
        img = self.loader(os.path.join(self.root, img))
        if self.transform is not None:
            img = self.transform(img)
        if not self.is_test and self.target_transform is not None:
            target = self.target_transform(target)
        if self.is_test:
            return img
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs)

        
def check_acc(output, target, topk=(1,)):
    if isinstance(output, tuple):
        output = output[0]
    maxk = max(topk)
    _, pred = output.topk(maxk, 1)
    res = []
    for k in topk:
        acc = (pred.eq(target.contiguous().view(-1,1).expand(pred.size()))[:, :k]
               .float().contiguous().view(-1).sum(0))
        acc.mul_(100 / target.size(0))
        res.append(acc)
    return res