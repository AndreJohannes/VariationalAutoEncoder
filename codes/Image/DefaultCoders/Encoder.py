import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy

class Encoder(nn.Module):
    '''
    Simple 2 Layer encoder
    '''
    def __init__(self, n_latent_units, drop_ratio):
        '''
        Constructor
        :param n_latent_units:
        :param drop_ratio: not used
        '''
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(28*28, 100)
        self.linear2 = torch.nn.Linear(100, 100)
        self._enc_mu = torch.nn.Linear(100, n_latent_units)
        self._enc_log_std = torch.nn.Linear(100, n_latent_units)
        self.has_cuda = False

    def forward(self, x):
        '''
        forward function, simple 2 layers followed by a random sample variable according to the
        mean and variance specified my the neural net
        :param x: the input variable
        :return: the latent output variable
        '''
        x = x.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self._enc_mu(x)
        log_std = self._enc_log_std(x)
        std = torch.exp(log_std)
        std_z = torch.from_numpy(numpy.random.normal(0, 1, size=std.size())).float()
        eps = Variable(std_z, requires_grad=False)
        if self.has_cuda:
            eps = eps.cuda()
        return mu + std * eps, mu, log_std

    def cuda(self):
        '''
        The random variable generated in the forward function needs to be gpu compliant if
        cuda is used.
        :return: nothing
        '''
        super(Encoder, self).cuda()
        self.has_cuda = True