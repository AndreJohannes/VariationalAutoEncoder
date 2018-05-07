import scipy.io.wavfile as wf
import numpy, torch
from torch.autograd import Variable
from codes.VAE import VariationalAutoEncoder
from codes.Audio.ConvolutionalAudioCoders2D.Encoder import Encoder
from codes.Audio.ConvolutionalAudioCoders2D.Decoder import Decoder
from codes.utility.Spectrogram import Spectrogram
import torch.optim as optim
import logging

# os.remove('/output/runtime.log')

class VAE_Wrapper:
    def __init__(self, vae, training_batches):
        self.vae = vae
        self.training_batches = training_batches
        self.vae.cuda()
        self.train_op = optim.Adam(self.vae.parameters(), lr=0.0005)

    def train_batch(self):
        model = self.vae
        train_op = self.train_op
        loss_ = []
        n = []
        for _, data in enumerate(self.training_batches):
            # data = Variable(data.view(-1,784))
            data = Variable(data)  # .cuda()

            train_op.zero_grad()
            dec = model(data)
            loss, _, _ = model.loss(data, dec, model.mu, model.log_std)
            loss_.append(loss.data[0])
            n.append(len(data))
            loss.backward()
            train_op.step()

        loss_train = numpy.sum(loss_) / numpy.sum(n)

        return loss_train



spec_256 = Spectrogram(256).cuda()
spec_4t256 = Spectrogram(4*256).cuda()

def loss_function( _in, _out, mu, log_std):
        assert(_in.shape==_out.shape)
        img_loss = torch.sqrt(torch.sum(torch.pow(spec_4t256(_in)-spec_4t256(_out),2)))
        img_loss += 0.25* torch.sqrt(torch.sum(torch.pow(spec_256(_in)-spec_256(_out),2)))
        mean_sq = mu * mu
        # -0.5 * tf.reduce_sum(1.0 + 2.0 * logsd - tf.square(mn) - tf.exp(2.0 * logsd), 1)
        latent_loss = -0.5 * torch.sum(1.0 + 2.0 * log_std - mean_sq - torch.exp(2.0 * log_std))
        return img_loss.cuda() + latent_loss, img_loss.cuda(), latent_loss

def make_batch(sound, chunk_length=8192, overlap=128):
    n = numpy.ceil((fs * 10 - overlap) / (chunk_length - overlap)).astype(int)
    chunks = []
    for i in numpy.arange(n):
        hop = chunk_length - overlap
        chunks.append(torch.FloatTensor(sound[i * hop: i * hop + chunk_length]))

    batch = torch.stack(chunks).unsqueeze(1).cuda()
    return batch


# Get the sound file
print("get sound file")
fs, sound = wf.read('/data/barackobama2004.wav')
sound = sound.astype(float)
sound *= 1.0/numpy.max(numpy.abs(sound))

print("make training set")
# make the batch
train_loader = [make_batch(sound[fs * 11:, 0])]

print("training set finished, {} batches of size {}".format(len(train_loader), train_loader[0].shape[0]))

#######
# Make the VAE
#######
vae = VariationalAutoEncoder(38, 0)
vae.encoder = Encoder(38, 0.00)
vae.decoder = Decoder(38, 0.00)
#spec = Spectrogram(256)
vae.loss = loss_function
vae.train_loader = train_loader

wrapper = VAE_Wrapper(vae,
                      train_loader)

for i in range(10000):
    loss = wrapper.train_batch()
    if i % 1 == 0:
        print("Epoch {}, loss {}".format(i, loss))
        print('{{"metric": "loss", "value": {} }}'.format(min(loss, 26000)))
        print([(name, torch.max(torch.abs(_)).data[0]) for name, _ in vae.named_parameters()][ \
            numpy.argmax([torch.max(torch.abs(params)).data[0] for name, params in (vae.named_parameters())])])
        if numpy.isnan(loss):
            break;
        if i%100==0:
            torch.save(vae.state_dict(), '/output/checkpoint_{}.pt'.format(i))
        #    logging.info("saved state")
torch.save(vae.state_dict(), '/output/checkpoint_{}.pt'.format(i))
print("finished - hasta luego")