import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy


class Encoder(nn.Module):
    '''
    Convolution based encoder
    '''

    def __init__(self, n_latent_units, drop_ratio, input_length=8192):
        '''
        Constructor
        :param n_latent_units: the number of latent variables, good default value might be 8
        :param drop_ratio: the drop ratio for the drop out.
        '''
        super(Encoder, self).__init__()
        if (input_length != 8192) & (input_length != 12288):
            raise Exception("input length not supported")
        self.conv1 = nn.Conv1d(1, 16, 32, stride=2, padding=15)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=4, padding=14)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv1d(32, 64, 32, stride=4, padding=14)
        self.conv4 = nn.Conv1d(64, 128, 16, stride=4, padding=7)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(1, 32, 5, stride=1, padding=4)
        self.conv5_drop = nn.Dropout2d(p=drop_ratio)
        self.mp5 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv6_drop = nn.Dropout2d(p=drop_ratio)
        self.mp6 = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv7_drop = nn.Dropout2d(p=drop_ratio)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 8, 5, stride=1, padding=2)
        self.num_flat_features = 8 * 33 * 17 if input_length == 8192 \
            else 8 * 33 * 25
        self.linear1 = nn.Linear(self.num_flat_features,
                                 self.num_flat_features // 4)
        self.linear2a = nn.Linear(self.num_flat_features // 4, n_latent_units)
        self.linear2b = nn.Linear(self.num_flat_features // 4, n_latent_units)
        self.get_random_variable = self._get_random_variable
        self._const = 64 if input_length == 8192 else 96

    def forward(self, x):
        '''
        The forward function
        :param x: the input vector (expects 28x28 matrix)
        :return: the output vector
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = x.view(-1, 1, 128, self._const)
        x = F.leaky_relu(self.conv5_drop(self.conv5(x)))
        x = self.mp5(x)
        x = F.leaky_relu(self.conv6_drop(self.bn6(self.conv6(x))))
        x = self.mp6(x)
        x = F.leaky_relu(self.conv7_drop(self.bn7(self.conv7(x))))
        x = F.leaky_relu(self.conv8(x))
        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.linear1(x))
        mu = self.linear2a(x)
        logstd = 0.5 * self.linear2b(x)
        eps = self.get_random_variable(
            logstd.size())  # Function called depends on cuda or cpu version
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