import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    '''
    Convolution based decoder
    '''

    def __init__(self, n_latent_units, drop_ratio):
        '''

        :param n_latent_units:
        :param drop_ratio:
        '''
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(n_latent_units, 24)
        self.linear2 = nn.Linear(24, 49)

        self.conv1 = nn.ConvTranspose2d(1, 64, 4, stride=2, padding=1)
        self.conv1_drop = nn.Dropout2d(p=drop_ratio)
        self.conv2 = nn.ConvTranspose2d(64, 64, 4, stride=1, padding=2)
        self.conv2_drop = nn.Dropout2d(p=drop_ratio)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, stride=1, padding=1)

        self.num_flat_features = 14 * 14 * 64
        self.linear3 = nn.Linear(self.num_flat_features, 28 * 28)

    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = x.view(-1, 1, 7, 7)
        x = F.relu(self.conv1_drop(self.conv1(x)))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = x.view(-1, self.num_flat_features)
        x = F.sigmoid(self.linear3(x))
        return x.view(-1, 1, 28, 28)
