import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.register as register


@register.setmodelname("DenseNet")
class DenseNet(nn.Module):
    def __init__(self, in_size=100):
        super(DenseNet, self).__init__()

        
    def output_size(self, in_size, kernel_size, stride, padding):

        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1

        return(output)

    def forward(self, x):
        
        
        return x
