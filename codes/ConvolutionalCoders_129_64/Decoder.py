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
        self.linear1 = nn.Linear(n_latent_units, 60)
        self.linear2 = nn.Linear(60, 60)

        self.convt1 = nn.ConvTranspose2d(1, 64, 4, stride=2, padding=2)
        self.convt1_drop = nn.Dropout2d(p=drop_ratio)
        self.convt2 = nn.ConvTranspose2d(64, 128, 4, stride=2, padding=2)
        self.convt2_drop = nn.Dropout2d(p=drop_ratio)
        self.convt3 = nn.ConvTranspose2d(128, 256, 4, stride=2, padding=2)
        self.convt4 = nn.ConvTranspose2d(256, 1, 4, stride=2, padding=(2, 3))

        self.num_flat_features = 14 * 14 * 64
        self.linear3 = nn.Linear(self.num_flat_features, 28 * 28)

    def forward(self, x):
        '''
        The forward function.
        :param x: the input vector (expects vector with dimension as specified by n_latent_units)
        :return: the output vector
        '''
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = x.view(-1, 1, 10, 6)
        x = F.relu(self.convt1_drop(self.convt1(x)))
        x = F.relu(self.convt2_drop(self.convt2(x)))
        x = F.relu((self.convt3(x)))
        x = F.sigmoid(self.convt4(x))
        return x[:,:,0:129,:]
