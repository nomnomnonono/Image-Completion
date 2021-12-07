import torch
import torch.nn as nn
import torch.nn.functional as F



class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate, self).__init__()
        self.dim = dim
    
    def forward(self,x):
        return torch.cat(x, dim=self.dim)