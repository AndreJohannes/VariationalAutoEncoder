import scipy.io.wavfile as wf
import numpy, torch, os
from torch.autograd import Variable
from codes.VAE import VariationalAutoEncoder
from codes.ConvolutionalCoders_129_64.Encoder import Encoder
from codes.ConvolutionalCoders_129_64.Decoder import Decoder
from codes.utility.Spectrogram import Spectrogram
import torch.optim as optim
import logging

# os.remove('/output/runtime.log')
logging.basicConfig(filename='/output/runtime.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')


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


# Define function for generating the batch
def get_random_chunk(train=True):
    start = numpy.random.randint(0, int(len(sound[:, 0]) * 3 / 4) - 8320) if train else \
        numpy.random.randint(int(len(sound[:, 0]) / 2), len(sound[:, 0]) - 8320)

    _sound = sound[start:start + 8320, 0]
    _sound *= 0.9 / numpy.max(numpy.abs(_sound))
    return _sound


# Get the sound file
logging.info("get sound file")
fs, sound = wf.read('/data/barackobama2004.wav')
sound = sound.astype(float)
sound /= numpy.max(numpy.abs(sound))

logging.info("make training set")
# make the batch
train_loader = []


def spec(_in):
    v = torch.pow(Spectrogram(256)(Variable(_in)).data, 1 / 10.).cuda()
    return v / torch.max(v)


for i in range(10):
    train_loader.append(torch.stack([spec(torch.FloatTensor(get_random_chunk())).unsqueeze(0) for i in range(128)]))
logging.info("training set finished, {} batches of size {}".format(len(train_loader), train_loader[0].shape[0]))

#######
# Make the VAE
#######
vae = VariationalAutoEncoder(18, 0)
vae.encoder = Encoder(48, 0)
vae.decoder = Decoder(48, 0)
spec = Spectrogram(256)
vae.train_loader = train_loader

wrapper = VAE_Wrapper(vae,
                      train_loader)

for i in range(10):
    loss = wrapper.train_batch()
    if i % 1 == 0:
        logging.info("Epoch {}, loss {}".format(i, loss))
        print("Epoch {}, loss {}".format(i, loss))
        # if i%1000==0:
        #    torch.save(vae.state_dict(), 'checkpoint_{}.pt'.format(i))
        #    logging.info("saved state")
torch.save(vae.state_dict(), 'checkpoint_{}.pt'.format(i))
logging.info("finished - hasta luego")