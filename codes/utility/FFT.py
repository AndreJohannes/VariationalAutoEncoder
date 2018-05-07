import torch, numpy
from torch.autograd import Function
from numpy.fft import rfft, irfft

class FFT(Function):

    def forward(self, input):
        numpy_input = input.numpy()
        result = rfft(numpy_input)
        return torch.FloatTensor(numpy.real(result)), torch.FloatTensor(numpy.imag(result))

    def backward(self, grad_output_r, grad_output_i):
        numpy_go = grad_output_r.numpy() + 1j*grad_output_i.numpy()
        numpy_go[...,1:-1] *= 0.5
        result = irfft(numpy_go)
        return torch.FloatTensor(result * (numpy_go.shape[-1]-1)*2)

    @staticmethod
    def _abs(_in_r , _in_i):
        return torch.sqrt(torch.pow(_in_r,2)+torch.pow(_in_i, 2))

def fft(input):
    return FFT()(input)