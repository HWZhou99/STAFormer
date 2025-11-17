import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling 
        - At p = 1, one gets Average Pooling   
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """
    def __init__(self,norm,output_size=1,eps=1e-6):
        super(GeneralizedMeanPooling,self).__init__()
        assert norm>0
        self.p=float(norm)
        self.output_size=output_size
        
        self.eps=eps
    
    def forward(self,x):
        x=x.clamp(min=self.eps)
        x=x.pow(self.p)
        return F.adaptive_avg_pool2d(x,self.output_size).pow(1./ self.p)
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
class GeneralizedMeanPoolongP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """
    def __init__(self,norm=3,output_size=1,eps=1e-6):
        super(GeneralizedMeanPoolongP,self).__init__(norm,output_size,eps)
        self.p=nn.Parameter(torch.ones(1)*norm)   
        
        
'''
GAP
input: bt x c x h x w
output:bt x c 
'''
class Global_avg_pool(nn.Module):
    def __init__(self):
        super(Global_avg_pool,self).__init__()
    def forward(self,inputs):
        return F.adaptive_avg_pool2d(inputs,1).view(inputs.size(0),-1)
'''GMP
input: bt x c x h x w
output:bt x c 
'''
class Global_max_pool(nn.Module):
    def __init__(self) :
        super(Global_max_pool,self).__init__()
    def forward(self,inputs):
        return F.adaptive_max_pool2d(inputs,1).view(inputs.size(0),-1)