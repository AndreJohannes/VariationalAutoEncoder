import numpy
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.multiprocessing import Queue

import codes.ConvolutionCoders.Decoder as ConvDecoder
import codes.ConvolutionCoders.Encoder as ConvEncoder
import codes.DefaultCoders.Decoder as Decoder
import codes.DefaultCoders.Encoder as Encoder


class Counter(object):
    def __init__(self):
        from torch.multiprocessing import Value
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    def reset(self):
        with self.val.get_lock():
            self.val.value = 0

    @property
    def value(self):
        return self.val.value

class Signal(object):
    def __init__(self):
        from torch.multiprocessing import Value
        self.val = Value('i', False)

    def set_signal(self, boolean):
        with self.val.get_lock():
            self.val.value = boolean

    @property
    def value(self):
        return bool(self.val.value)


class VariationalAutoEncoder(nn.Module):
    '''

    '''

    def __init__(self, n_latent_units, drop_ratio, convolutional=False, cuda = False):
        '''

        :param n_latent_units:
        :param drop_ratio:
        '''
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder.Encoder(n_latent_units, drop_ratio) if not convolutional \
            else ConvEncoder.Encoder(n_latent_units, drop_ratio)
        self.decoder = Decoder.Decoder(n_latent_units, drop_ratio) if not convolutional \
            else ConvDecoder.Decoder(n_latent_units, drop_ratio)
        self.proc = None

        self.has_cuda = cuda
        if cuda:
            self.cuda()

        self.counter_epoch = Counter()
        self.counter_interation = Counter()
        self.loss_queue = Queue()
        self.stop_signal = Signal()
        self.losses = []
        self.train_loader, self.test_loader = VariationalAutoEncoder._get_train_loader()
        # self.img_loss_func = nn.MSELoss()

    def forward(self, x):
        z, mu, log_std = self.encoder.forward(x)
        self.mu = mu
        self.log_std = log_std
        return self.decoder.forward(z)

    def loss(self, _in, _out, mu, log_std):
        # img_loss = self.img_loss_func(_in, _out)
        # img_loss = F.mse_loss(_in, _out)
        img_loss = _in.sub(_out).pow(2).sum()
        mean_sq = mu * mu
        # -0.5 * tf.reduce_sum(1.0 + 2.0 * logsd - tf.square(mn) - tf.exp(2.0 * logsd), 1)
        latent_loss = -0.5 * torch.sum(1.0 + 2.0 * log_std - mean_sq - torch.exp(2.0 * log_std))
        return img_loss + latent_loss

    def start(self):
        if self.proc is not None:
            raise Exception("Process already started.")
        self.share_memory()
        self.losses = []
        train = VariationalAutoEncoder._get_training_test_method(self.has_cuda)
        self.proc = mp.Process(target=train, args=(self, self.train_loader,
                                                   self.test_loader,
                                                   self.counter_epoch,
                                                   self.counter_interation,
                                                   self.loss_queue,
                                                   self.stop_signal))
        self.proc.start()

    def restart(self):
        if self.proc is None:
            raise Exception("Process has not been started before.")
        if self.proc.is_alive():
            raise Exception("Process is still active.")
        #self.share_memory()
        self.stop_signal.set_signal(False)
        train = VariationalAutoEncoder._get_training_test_method(self.has_cuda)
        self.proc = mp.Process(target=train, args=(self, self.train_loader,
                                                   self.test_loader,
                                                   self.counter_epoch,
                                                   self.counter_interation,
                                                   self.loss_queue,
                                                   self.stop_signal))
        self.proc.start()

    def stop(self):
        if self.proc is None:
            raise Exception("Process has been started.")
        if not self.proc.is_alive():
            raise Exception("Process is not alive.")
        self.stop_signal.set_signal(True)
        self.proc.join()
        self.stop_signal.set_signal(False)

    def get_progress(self):
        while self.loss_queue.qsize() > 0:
            self.losses.append(self.loss_queue.get())
        return self.losses

    def set_train_loader(self, train_loader, test_loader = None):
        self.train_loader = train_loader
        self.test_loader = test_loader

    @staticmethod
    def _get_training_method():
        def train(model, train_loader, counter_epoch,
                  counter_iterations, loss_queue, stop_signal):
            print("start", stop_signal.value)
            train_op = optim.Adam(model.parameters(), lr=0.0005)
            while not stop_signal.value:
                loss_ = []
                n = []
                print("1")
                for _, (data, target) in enumerate(train_loader):
                    # data = Variable(data.view(-1,784))
                    print("2")
                    data = Variable(data)
                    print("3")
                    train_op.zero_grad()
                    dec = model(data)
                    print("4")
                    loss = model.loss(data, dec, model.mu, model.log_std)
                    print("5")
                    loss_.append(loss.data[0])
                    print("6")
                    n.append(len(data))
                    print("7")
                    loss.backward()
                    print("8")
                    train_op.step()
                    counter_iterations.increment()
                    print("9")
                counter_epoch.increment()
                print("10")

                epoch = counter_epoch.value
                loss_mean = numpy.sum(loss_) / numpy.sum(n)
                loss_queue.put((epoch, loss_mean))
                print("{}: ".format(epoch), loss_mean)

        return train

    @staticmethod
    def _get_training_test_method(cuda):
        def train(model, train_loader, test_loader, counter_epoch,
                  counter_iterations, loss_queue, stop_signal):
            print("started", stop_signal.value)
            train_op = optim.Adam(model.parameters(), lr=0.0005)
            while not stop_signal.value:
                loss_train = []
                loss_test = []
                n_train = []
                n_test = []
                print("1")
                for _, data in enumerate(train_loader):
                    # data = Variable(data.view(-1,784))
                    data = Variable(data)
                    if(cuda):
                        data = data.cuda()
                    train_op.zero_grad()
                    dec = model(data)
                    loss = model.loss(data, dec, model.mu, model.log_std)
                    loss_train.append(loss.data[0])
                    n_train.append(len(data))
                    loss.backward()
                    train_op.step()
                    counter_iterations.increment()
                print("2")
                for _, data in enumerate(test_loader):
                    # data = Variable(data.view(-1,784))
                    data = Variable(data)
                    if(cuda):
                        data = data.cuda()
                    dec = model(data)
                    loss = model.loss(data, dec, model.mu, model.log_std)
                    loss_test.append(loss.data[0])
                    n_test.append(len(data))

                print("3")
                counter_epoch.increment()

                epoch = counter_epoch.value
                loss_train_mean = numpy.sum(loss_train) / numpy.sum(n_train)
                loss_test_mean = numpy.sum(loss_test) / numpy.sum(n_test)
                loss_queue.put((epoch, loss_train_mean, loss_test_mean))
                #print("{}: ".format(epoch),  loss_train_mean, loss_test_mean)

        return train

    @staticmethod
    def _get_train_loader():
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

        return train_loader, test_loader
