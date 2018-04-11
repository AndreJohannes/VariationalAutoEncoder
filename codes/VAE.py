import numpy
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.multiprocessing import Queue
from torchvision import datasets, transforms

import codes.ConvolutionCoders.Decoder as ConvDecoder
import codes.ConvolutionCoders.Encoder as ConvEncoder
import codes.DefaultCoders.Decoder as Decoder
import codes.DefaultCoders.Encoder as Encoder
from codes.utility.Multiprocessing import Counter, Signal
from codes.utility.DataLoader import DataIterator

class VariationalAutoEncoder(nn.Module):
    '''
    Implementation of a Variational AutoEncoder in pytorch. Currently two
    decoder/encoder units are supported. The first unit features a two layer
    dense neural network and the second a deep convolutional net.
    The number of laternt units can be specified and is usually around 4-12 units.
    The implmentation supports cuda. If cuda is not used the multiprocessing
    framework is/can be used to send the computation to the background, so the
    jupyter notebook it runs in will not be blocked.
    '''
    def __init__(self, n_latent_units, drop_ratio, convolutional=False):
        '''
        Constructor
        :param n_latent_units:
        :param drop_ratio:
        '''
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder.Encoder(n_latent_units, drop_ratio) if not convolutional \
            else ConvEncoder.Encoder(n_latent_units, drop_ratio)
        self.decoder = Decoder.Decoder(n_latent_units, drop_ratio) if not convolutional \
            else ConvDecoder.Decoder(n_latent_units, drop_ratio)
        self.proc = None

        self.counter_epoch = Counter()
        self.counter_interation = Counter()
        self.loss_queue = Queue()
        self.stop_signal = Signal()
        self.losses = []

    def forward(self, x):
        '''
        The forward method, calles the encoder and decoder
        :param x:
        :return:
        '''
        z, mu, log_std = self.encoder.forward(x)
        self.mu = mu
        self.log_std = log_std
        return self.decoder.forward(z)

    def loss(self, _in, _out, mu, log_std):
        '''
        The loss function, the loss is calculated as the reconstruction error and
        the error given by the deviation of latent variable from the normal distirbution
        :param _in:
        :param _out:
        :param mu:
        :param log_std:
        :return:
        '''
        # img_loss = self.img_loss_func(_in, _out)
        # img_loss = F.mse_loss(_in, _out)
        img_loss = _in.sub(_out).pow(2).sum()
        mean_sq = mu * mu
        # -0.5 * tf.reduce_sum(1.0 + 2.0 * logsd - tf.square(mn) - tf.exp(2.0 * logsd), 1)
        latent_loss = -0.5 * torch.sum(1.0 + 2.0 * log_std - mean_sq - torch.exp(2.0 * log_std))
        return img_loss + latent_loss

    def start(self, train = None):
        '''
        This runs the training in the background. Currently only works with the cpu version
        (cuda not supported atm)
        :param train:
        :return:
        '''
        if self.proc is not None:
            raise Exception("Process already started.")
        self.share_memory()
        self.losses = []
        if train is None:
            train = VariationalAutoEncoder._get_training_test_method()
        self.proc = mp.Process(target=train, args=(self, self.train_loader,
                                                   self.test_loader,
                                                   self.counter_epoch,
                                                   self.counter_interation,
                                                   self.loss_queue,
                                                   self.stop_signal))
        self.proc.start()

    def restart(self, train = None):
        '''
        Running in the background can be stopped. This method should be used if
        the computation should be resumed. As with start(), does currently not work with cuda.
        :param train:
        :return:
        '''
        if self.proc is None:
            raise Exception("Process has not been started before.")
        if self.proc.is_alive():
            raise Exception("Process is still active.")
        self.stop_signal.set_signal(False)
        if train is None:
            train = VariationalAutoEncoder._get_training_test_method()
        self.proc = mp.Process(target=train, args=(self, self.train_loader,
                                                   self.test_loader,
                                                   self.counter_epoch,
                                                   self.counter_interation,
                                                   self.loss_queue,
                                                   self.stop_signal))
        self.proc.start()

    def stop(self):
        '''
        This functions sends a stop signal to the background process.
        :return:
        '''
        if self.proc is None:
            raise Exception("Process has been started.")
        if not self.proc.is_alive():
            raise Exception("Process is not alive.")
        self.stop_signal.set_signal(True)
        self.proc.join()
        self.stop_signal.set_signal(False)

    def get_progress(self):
        '''
        Functions gets the progress of the computation running in the background.
        :return:
        '''
        while self.loss_queue.qsize() > 0:
            self.losses.append(self.loss_queue.get())
        return self.losses

    def set_train_loader(self, train_loader, test_loader = None):
        self.train_loader = train_loader
        self.test_loader = test_loader

    def cuda(self):
        super(VariationalAutoEncoder, self).cuda()
        self.decoder.cuda()
        self.encoder.cuda()

    @staticmethod
    def _get_training_test_method():
        def train(model, train_loader, test_loader, counter_epoch,
                  counter_iterations, loss_queue, stop_signal):
            print("started", stop_signal.value)
            train_op = optim.Adam(model.parameters(), lr=0.0005)
            while not stop_signal.value:
                loss_train = []
                loss_test = []
                n_train = []
                n_test = []
                for _, data in enumerate(train_loader):
                    # data = Variable(data.view(-1,784))
                    data = Variable(data)
                    train_op.zero_grad()
                    dec = model(data)
                    loss = model.loss(data, dec, model.mu, model.log_std)
                    loss_train.append(loss.data[0])
                    n_train.append(len(data))
                    loss.backward()
                    train_op.step()
                    counter_iterations.increment()

                for _, data in enumerate(test_loader):
                    # data = Variable(data.view(-1,784))
                    data = Variable(data)
                    dec = model(data)
                    loss = model.loss(data, dec, model.mu, model.log_std)
                    loss_test.append(loss.data[0])
                    n_test.append(len(data))

                counter_epoch.increment()

                epoch = counter_epoch.value
                loss_train_mean = numpy.sum(loss_train) / numpy.sum(n_train)
                loss_test_mean = numpy.sum(loss_test) / numpy.sum(n_test)
                loss_queue.put((epoch, loss_train_mean, loss_test_mean))
                #print("{}: ".format(epoch),  loss_train_mean, loss_test_mean)

        return train

    @staticmethod
    def get_MINST_train_loader():
        batch_size = 32
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=batch_size)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
            batch_size=batch_size)

        return DataIterator(train_loader), DataIterator(test_loader)
