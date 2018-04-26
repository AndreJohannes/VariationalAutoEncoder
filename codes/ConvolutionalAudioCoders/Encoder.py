import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy

class Encoder(nn.Module):
    '''
    Convolution based encoder
    '''
    def __init__(self, n_latent_units, drop_ratio):
        '''
        Constructor
        :param n_latent_units: the number of latent variables, good default value might be 8
        :param drop_ratio: the drop ratio for the drop out.
        '''
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 256, stride=128, padding = 0)
        self.conv1_drop = nn.Dropout(p = drop_ratio)
        self.conv2 = nn.Conv2d(1, 64, 4, stride=1, padding = 0)
        self.pool =  nn.MaxPool2d(2,2)
        self.conv2_drop = nn.Dropout(p = drop_ratio)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=1, padding = 0)
        self.conv3_drop = nn.Dropout(p = drop_ratio)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=1, padding = 0)
        self.conv4_drop = nn.Dropout(p = drop_ratio)
        self.num_flat_features = 4160
        self.linear1 = nn.Linear(self.num_flat_features, n_latent_units)
        self.linear2 = nn.Linear(self.num_flat_features, n_latent_units)
        self.get_random_variable = self._get_random_variable

    def forward(self, x):
        '''
        The forward function
        :param x: the input vector (expects 28x28 matrix)
        :return: the output vector
        '''
        x = x / 2. + 0.5
        x = F.leaky_relu(self.conv1_drop(self.conv1(x)))
        x= x.view(-1, 1 , 128, 64)
        x = self.pool(F.leaky_relu(self.conv2_drop(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.conv3_drop(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.conv4_drop(self.conv4(x))))
        x = x.view(-1, self.num_flat_features)
        mu = self.linear1(x)
        logstd = 0.5*self.linear2(x)
        eps = self.get_random_variable(logstd.size()) #Function called depends on cuda or cpu version
        return eps.mul(torch.exp(logstd)).add_(mu), mu, logstd

    def cuda(self):
        '''
        This function is needed because we have to make sure
        that the variable created in the forward method is cuda compliant
        :return:
        '''
        super(Encoder, self).cuda()
        self.get_random_variable = self._get_random_variable_cuda

    def _get_random_variable(self, size):
        '''
        Function generates a random variable with mean 0 and variance 1
        :param size: the dimension of the variable
        :return: the variable / sample
        '''
        eps = torch.from_numpy(numpy.random.normal(0, 1, size=size)).float()
        eps = Variable(eps, requires_grad=False)
        return eps

    def _get_random_variable_cuda(self, size):
        '''
        Function generates a random variable for the gpu with mean 0 and variance 1
        :param size: the dimension of the variable
        :return: the variable / sample
        '''
        eps = torch.from_numpy(numpy.random.normal(0, 1, size=size)).float()
        eps = Variable(eps, requires_grad=False).cuda()
        return eps