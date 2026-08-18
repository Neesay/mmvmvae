

import torch
import torch.nn as nn
import numpy as np

def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)


class Encoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 7, 7)
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 4, 4)
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            Flatten(),  # -> (2048)
            nn.Linear(2048, self.latent_dim),
            nn.ReLU(),
        )
        # latent representation
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return (
            self.mu(h),
            self.logvar(h),
        )


class Decoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),  # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),  # -> (128, 4, 4)
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (3, 28, 28)
            # nn.BatchNorm2d(3),
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat, 0.75  # NOTE: consider learning scale param, too


class ResnetBlock(nn.Module):
    # fhidden=None in all experiments conducted by E. Orme, but keeping option to specify fhidden for flexibility
    # is_bias=True in all experiments conducted by E. Orme, but keeping option to specify is_bias for flexibility
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        # if the input and output number of channels differ, we need a learned shortcut connection (residual connection) - this is a convolutional layer with kernel size 1 that transforms the input to the correct dimension for the residual connection
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
            # in the encoder, fhidden is always fin
            # in the decoder, fhidden is always fout
        else:
            self.fhidden = fhidden

        # Submodules
        # conv2d(input_channels, output_channels, kernel_size, stride=1, padding=0, bias=True)
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias
        )
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class ResnetEncoder(nn.Module):
    def __init__(self, cfg):  # , z_dim, size, nfilter=64, nfilter_max=1024, **kwargs):
        super().__init__()
        s0 = self.s0 = 7  # kwargs['s0'] - this is the size of the feature maps after the last ResnetBlock, and determines how many times we need to apply downsampling (average pooling) to go from the input image size to s0
        nf = self.nf = 64  # nfilter - the number of initial filters
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28 # input image size (height and width)

        # Submodules
        # nlayers is the number of times we need to apply a downsampling operation (average pooling) to go from the input image size to the size s0 (the size of the feature maps after the last ResnetBlock)
        nlayers = int(np.log2(size / s0)) # e.g. for size=28 and s0=7, nlayers=2 (we need to apply average pooling twice to go from 28 to 7)
        self.nf0 = min(nf_max, nf * 2**nlayers) # 256 # number of channels in the feature maps after the last ResnetBlock - this is determined by the number of layers and the base number of filters, and is capped at nf_max

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            # nf0<=nf1 by defintion
            # number of hidden channels is equal to nf0
            # number of output channels is equal to nf1
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc_mu = nn.Linear(self.nf0 * s0 * s0, cfg.model.latent_dim)
        self.fc_lv = nn.Linear(self.nf0 * s0 * s0, cfg.model.latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        # out = self.fc(actvn(out))
        return self.fc_mu(out), self.fc_lv(out)


class ResnetDecoder(nn.Module):
    def __init__(self, cfg):  # , z_dim, size, nfilter=64, nfilter_max=512, **kwargs):
        super().__init__()
        self.latent_dim = cfg.model.latent_dim

        # NOTE: I've modified/set below variables according to Kieran's suggestions
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(self.latent_dim, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [ResnetBlock(nf0, nf1), nn.Upsample(scale_factor=2)]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        out = self.fc(z)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        return out, 0.75  # NOTE: consider learning scale param, too
        # return torch.sigmoid(out)  # torch.tanh(out), torch.sigmoid(out)
