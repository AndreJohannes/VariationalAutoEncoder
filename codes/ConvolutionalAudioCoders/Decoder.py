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
        self.linear1 = nn.Linear(n_latent_units, 24)
        self.linear2 = nn.Linear(24, 115)

        self.conv1 = nn.ConvTranspose1d(1, 64, 256, stride=4, padding=0)
        self.conv1_drop = nn.Dropout(p=drop_ratio)
        self.conv2 = nn.ConvTranspose1d(64, 64, 256, stride=4, padding=0)
        self.conv2_drop = nn.Dropout(p=drop_ratio)
        self.conv3 = nn.ConvTranspose1d(64, 1, 256, stride=4, padding=2)


    def forward(self, x):
        '''
        The forward function.
        :param x: the input vector (expects vector with dimension as specified by n_latent_units)
        :return: the output vector
        '''
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = x.view(-1, 1, 115)
        x = F.relu(self.conv1_drop(self.conv1(x)[:, :, 254:-254]))
        x = F.relu(self.conv2_drop(self.conv2(x)[:, :, 252:-252]))
        x = F.sigmoid((self.conv3(x)[:, :, 252:-252]))
        return 2*x-1

