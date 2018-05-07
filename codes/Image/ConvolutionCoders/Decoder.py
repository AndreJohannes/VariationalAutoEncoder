import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    '''
    Convolution based decoder
    '''

    def __init__(self, n_latent_units, drop_ratio):
        '''
        Constructor
        :param n_latent_units: the number of latent variables, good default value might be 8
        :param drop_ratio: the drop ratio for the drop out (Not used)
        '''
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(n_latent_units, 140)
        self.linear2 = nn.Linear(140, 64*8*8)

        self.conv1 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv1_drop = nn.Dropout2d(p=drop_ratio)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.up2 = nn.UpsamplingBilinear2d(size=(16, 16))
        self.conv2_drop = nn.Dropout2d(p=drop_ratio)
        self.conv3 = nn.Conv2d(64, 32, 5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.up3 = nn.UpsamplingBilinear2d(size=(32, 32))
        self.conv4 = nn.Conv2d(32, 1, 5, stride=1, padding=0)

    def forward(self, x):
        '''
        The forward function.
        :param x: the input vector (expects vector with dimension as specified by n_latent_units)
        :return: the output vector
        '''
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = x.view(-1, 64, 8, 8)
        x = self.conv1_drop((F.relu(self.conv1(x))))
        x = self.up2(self.conv2_drop(F.relu(self.bn2(self.conv2(x)))))
        x = self.up3(self.conv2_drop(F.relu(self.bn3(self.conv3(x)))))
        x = (F.sigmoid(self.conv4(x)))

        return x.view(-1, 1, 28, 28)
