import torch, numpy
from torch.autograd import Function
from numpy.fft import rfft, irfft
from torch.autograd import Variable

class FFT(Function):

    def forward(self, input):
        numpy_input = input.numpy()
        result = rfft(numpy_input)
        n = result.shape[-1]
        result = numpy.concatenate((numpy.real(result), numpy.imag(result[..., 1:n - 1])), -1)
        return torch.FloatTensor(result)

    def backward(self, grad_output):
        numpy_go = grad_output.numpy()
        n = numpy_go.shape[-1]
        n_h = int(n / 2)
        oo = (numpy_go[..., 0:n_h + 1]).astype(numpy.complex128)
        oo[...,1:n_h] += 1j * numpy_go[...,n_h + 1:]
        oo[...,1:n_h] *= 0.5
        result = irfft(oo)
        return torch.FloatTensor(result * n)

    @staticmethod
    def _abs(_in):
        n = _in.shape[-1]
        n_h = int(n / 2)

        oo_ = Variable(torch.zeros(_in[..., 0:n_h + 1].shape))
        oo_[..., 1:n_h] = torch.pow(_in[..., n_h + 1:], 2)
        oo = torch.pow(_in[..., 0: n_h + 1], 2) + oo_

        return torch.sqrt(oo)

def fft(input):
    return FFT()(input)