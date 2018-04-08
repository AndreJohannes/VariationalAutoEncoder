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

        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, stride=2, padding = 1)
        self.conv1_drop = nn.Dropout2d(p = drop_ratio)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2, padding = 1)
        self.conv2_drop = nn.Dropout2d(p = drop_ratio)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=1, padding = 2)
        self.conv3_drop = nn.Dropout2d(p = drop_ratio)
        self.num_flat_features = 64 * 8 * 8
        self.linear1 = nn.Linear(self.num_flat_features, n_latent_units)
        self.linear2 = nn.Linear(self.num_flat_features, n_latent_units)

    def forward(self, x):
        #y = nn.Linear(28*28, 8)(x.view(-1, 28*28))
         #print(y.shape)
        x = F.leaky_relu(self.conv1_drop(self.conv1(x)))
        x = F.leaky_relu(self.conv2_drop(self.conv2(x)))
        x = F.leaky_relu(self.conv3_drop(self.conv3(x)))
        x = x.view(-1, self.num_flat_features)
        mu = self.linear1(x)
        logstd = 0.5*self.linear2(x)
        eps = torch.from_numpy(numpy.random.normal(0, 1, size=logstd.size())).float()
        eps = Variable(eps, requires_grad = False)

        return eps.mul(torch.exp(logstd)).add_(mu), mu, logstd

