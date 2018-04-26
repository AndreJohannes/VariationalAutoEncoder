from torch.autograd import Variable
import torch, math, numpy
from codes.utility.FFT import FFT


class Spectrogram:
    def __init__(self, L):
        self.L = L
        self.window = Variable(torch.FloatTensor(numpy.hamming(L)))

    def get_spectrogram(self, _input):
        length = _input.shape[-1]
        L = self.L
        n_max = int(math.floor(length / (L / 2) - 2))
        ret = []
        for n in range(n_max + 1):
            fft_ = FFT()(self.window * _input[..., int(L / 2 * n):int(L / 2 * n + L)])
            ret.append(FFT._abs(fft_))
        ret_tensor = torch.stack(ret)
        ndim = ret_tensor.dim()
        return ret_tensor.permute([(i + 1) % ndim for i in range(ndim)])

    def __call__(self, _input):

        return self.get_spectrogram(_input)