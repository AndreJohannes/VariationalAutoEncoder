import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self, n_latent_units, drop_ratio):
        '''
        Constructor
        :param n_latent_units: number of latent units
        :param drop_ratio: not used
        '''
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(n_latent_units, 100)
        self.linear2 = torch.nn.Linear(100, 28*28)

    def forward(self, x):
        '''
        the forward method consists of a simple two layered net
        :param x: the input variable (latent variable)
        :return: the output as 28x28 image
        '''
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x)).view(-1, 1, 28, 28)
