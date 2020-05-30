import torch
import torch.nn as nn


class VanillaVAE(nn.Module):

    def __init__(self, inpt, enHiddens, outHiddens, LatentNum, useSigma=True):
        super().__init__()
        self.useSigma = useSigma
        self.LatentNum = LatentNum
        if self.useSigma:
            bottle_neck_num = LatentNum * 2
        else:
            bottle_neck_num = LatentNum

        encoders = []
        for i, j in zip([inpt] + enHiddens[:-1], enHiddens):
            encoders.append(nn.Linear(i, j))
            encoders.append(nn.BatchNorm1d(j))
            encoders.append(nn.ReLU())
        encoders.append(nn.Linear(enHiddens[-1], bottle_neck_num))
        self.encoders = nn.Sequential(*encoders)

        decoders = []
        for i, j in zip([LatentNum] + outHiddens[:-1], outHiddens):
            decoders.append(nn.Linear(i, j))
            decoders.append(nn.BatchNorm1d(j))
            decoders.append(nn.ReLU())
        decoders.append(nn.Linear(outHiddens[-1], inpt))
        # decoders.append(nn.Sigmoid())
        self.decoders = nn.Sequential(*decoders)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc_out = self.encoders(x)
        dec_inp, mu, logvar = self.reparameter(enc_out)
        rec = self.decoders(dec_inp)
        return rec, mu, logvar

    def encode(self, x):
        enc_out = self.encoders(x)
        mu = enc_out[:, :self.LatentNum]
        if self.useSigma:
            return mu, enc_out[:, self.LatentNum:]
        return mu, torch.ones_like(mu)

    def decode(self, mu, noise=False, logvar=None):
        if noise:
            if logvar is None:
                std = torch.ones_like(mu)
            else:
                std = (0.5*logvar).exp()
            epsilon = torch.randn_like(mu)
            dec_in = mu + std * epsilon
        else:
            dec_in = mu
        return self.sigmoid(self.decoders(dec_in))

    def reparameter(self, enc_out):
        epsilon = torch.randn(
            enc_out.size(0), self.LatentNum).to(enc_out.device)
        mu = enc_out[:, :self.LatentNum]
        if self.useSigma:
            logvar = enc_out[:, self.LatentNum:]
            std = (0.5 * logvar).exp()
        else:
            std = torch.ones_like(mu)
            logvar = torch.zeros_like(mu)

        return mu + std * epsilon, mu, logvar


class VanillaCNNVAE(nn.Module):

    def __init__(self, inpt, hiddens, LatentNum, useSigma=True):
        super().__init__()
        self.useSigma = useSigma
        self.LatentNum = LatentNum
        self.sigmoid = nn.Sigmoid()

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
            )
        self.encoders = nn.Sequential(*encoders)

        self.fc_mu = nn.Linear(hiddens[-1], LatentNum)
        if useSigma:
            self.fc_var = nn.Linear(hiddens[-1], LatentNum)

        hiddens.reverse()
        self.decode_inpt = nn.Linear(LatentNum, hiddens[0])

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
        self.decoders = nn.Sequential(*decoders)

    def encode(self, x):
        enc_out = self.encoders(x).reshape(x.size(0), -1)
        mu = self.fc_mu(enc_out)
        if self.useSigma:
            logvar = self.fc_var(enc_out)
        else:
            logvar = torch.zeros_like(mu)
        return mu, logvar

    def reparameter(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        std = (0.5 * logvar).exp()
        return mu + std * epsilon

    def decode(self, code, sigmoid=True):
        code = self.decode_inpt(code).reshape(code.size(0), -1, 1, 1)
        rec = self.decoders(code)
        if sigmoid:
            rec = self.sigmoid(rec)
        return rec

    def forward(self, x):
        mu, logvar = self.encode(x)
        code = self.reparameter(mu, logvar)
        rec = self.decode(code, False)
        return rec, mu, logvar
