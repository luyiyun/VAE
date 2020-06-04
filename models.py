import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaVAE(nn.Module):
    def __init__(self, inpt, latentNum, **kwargs):
        super().__init__()
        self.inpt = inpt
        self.latentNum = latentNum
        self.kwargs = kwargs

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

    def forward(self, x):
        mu, logvar = self.encode(x)
        dec_inp = self.reparameter(mu, logvar)
        rec = self.decode(dec_inp)
        return rec, mu, logvar

    def reparameter(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        std = (0.5 * logvar).exp()
        return mu + std * epsilon

    def criterion(self, rec, ori, mu, logvar, kl_weight, loss_type="mse"):
        kl_loss = torch.mean(
            -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1))
        if loss_type == "mse":
            rec_loss = F.mse_loss(rec, ori)
        elif loss_type == "ce":
            rec_loss = F.binary_cross_entropy(rec, ori)
        else:
            raise NotImplementedError
        total = kl_loss * kl_weight + rec_loss

        return total, rec_loss, kl_loss


class VanillaVAEfc(VanillaVAE):

    def __init__(self, inpt, latentNum, hiddens):
        super().__init__(inpt, latentNum)

        hiddens = list(hiddens)
        encoders = []
        for i, j in zip([inpt] + hiddens[:-1], hiddens):
            encoders.append(nn.Linear(i, j))
            encoders.append(nn.BatchNorm1d(j))
            encoders.append(nn.LeakyReLU())
        encoders.append(nn.Linear(hiddens[-1], self.latentNum*2))
        self.encoders = nn.Sequential(*encoders)

        hiddens.reverse()
        decoders = []
        for i, j in zip([latentNum] + hiddens[:-1], hiddens):
            decoders.append(nn.Linear(i, j))
            decoders.append(nn.BatchNorm1d(j))
            decoders.append(nn.ReLU())
        decoders.append(nn.Linear(hiddens[-1], inpt))
        decoders.append(nn.Sigmoid())
        self.decoders = nn.Sequential(*decoders)

    def encode(self, x):
        enc_out = self.encoders(x)
        mu, logvar = enc_out[:, :self.latentNum], enc_out[:, self.latentNum:]
        return mu, logvar

    def decode(self, x):
        out = self.decoders(x)
        return out


class VanillaVAEcnn(VanillaVAE):
    """ 因为涉及到维度的计算，所以只适用于MNIST """

    def __init__(self, inpt, latentNum, hiddens):
        super().__init__(inpt, latentNum)

        encoders = []
        for i, j in zip([inpt] + hiddens[:-1], hiddens):
            encoders.append(
                nn.Sequential(
                    nn.Conv2d(i, j, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(j),
                    nn.LeakyReLU()
                )
                # floor((28 + 2 x padding(1) - dilation(1) x
                #   (kernel_size(3) -1) -1) / stride(2) + 1)
                #   ==> 14
                #   ==> 7
                #   ==> 4
                #   ==> 2
                #   ==> 1
            )
        self.encoders = nn.Sequential(*encoders)

        self.fc_mu = nn.Linear(hiddens[-1], latentNum)
        self.fc_var = nn.Linear(hiddens[-1], latentNum)

        hiddens.reverse()
        self.decode_inpt = nn.Linear(latentNum, hiddens[0])
        decoders = []
        for i, j in zip(hiddens[:-1], hiddens[1:]):
            decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        i, j, kernel_size=3, stride=2, padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(j),
                    nn.LeakyReLU()
                )
            )
        decoders.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hiddens[-1], hiddens[-1], kernel_size=3, stride=2,
                    padding=1, output_padding=1
                ),
                nn.BatchNorm2d(hiddens[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hiddens[-1], 1, kernel_size=5, padding=0)
            )
        )
        decoders.append(nn.Sigmoid())
        self.decoders = nn.Sequential(*decoders)

    def encode(self, x):
        enc_out = self.encoders(x).reshape(x.size(0), -1)
        mu = self.fc_mu(enc_out)
        logvar = self.fc_var(enc_out)
        return mu, logvar

    def decode(self, code):
        code = self.decode_inpt(code).reshape(code.size(0), -1, 1, 1)
        rec = self.decoders(code)
        return rec
