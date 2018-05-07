from torch.autograd import Variable
import torch, math, numpy
from codes.utility.FFT import fft, FFT



class Spectrogram:
    def __init__(self, L):
        self.L = L
        self.window = Variable(torch.FloatTensor(numpy.hamming(L)))
        self.fft = fft

    def get_spectrogram(self, _input):
        _input = _input
        length = _input.shape[-1]
        L = self.L
        n_max = int(math.floor(length / (L / 2) - 2))
        ret = []
        for n in range(n_max + 1):
            fft_r, fft_i = self.fft(self.window * _input[..., int(L / 2 * n):int(L / 2 * n + L)])
            ret.append(FFT._abs(fft_r.float(), fft_i))
        ret_tensor = torch.stack(ret)
        ndim = ret_tensor.dim()
        return ret_tensor.permute([(i + 1) % ndim for i in range(ndim)])

    def __call__(self, _input):

        return self.get_spectrogram(_input)

    def cuda(self): # Use pytorch_fft if we porcessing on the gpu
        self.window = self.window.cuda()
        from pytorch_fft.fft.autograd import Rfft
        def fft_cuda(_input):
            re, im = Rfft()(_input.double())
            return re.float(), im.float()
        self.fft = fft_cuda
        return self
