import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy

class Encoder(nn.Module):

    def __init__(self, n_latent_units, drop_ratio):

        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(28*28, 100)
        self.linear2 = torch.nn.Linear(100, 100)
        self._enc_mu = torch.nn.Linear(100, n_latent_units)
        self._enc_log_std = torch.nn.Linear(100, n_latent_units)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self._enc_mu(x)
        log_std = self._enc_log_std(x)
        std = torch.exp(log_std)
        std_z = torch.from_numpy(numpy.random.normal(0, 1, size=std.size())).float()
        return mu + std * Variable(std_z, requires_grad=False), mu, log_std

