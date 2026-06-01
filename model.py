import torch.nn as nn
from torch.nn.functional import normalize
import torch
import numpy as np
from torch.nn import Linear


class Encoder(nn.Module):
    def __init__(self, n_layers, n_input, n_z):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        dims = []
        for idim in range(n_layers - 2):
            dim = round(n_input * 0.5)
            dim = int(dim)
            dims.append(dim)
        dim = 1000
        dim = int(dim)
        dims.append(dim)

        for i in range(n_layers):
            if i == 0:
                self.encoder.add_module('Linear%d' % i, nn.Linear(n_input, dims[i]))
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], n_z))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, n_layers, n_input, n_z):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential()
        dims = []
        for idim in range(n_layers - 2):
            dim = round(n_input * 0.5)
            dim = int(dim)
            dims.append(dim)
        dim = 1000
        dim = int(dim)
        dims.append(dim)
        dims = list(reversed(dims))
        for i in range(n_layers):
            if i == 0:
                self.decoder.add_module('Linear%d' % i, nn.Linear(n_z, dims[i]))
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], n_input))
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


class AEC_v2_m(nn.Module):

    def __init__(self,
                 n_layers,
                 n_input,
                 n_z,
                 n_h,
                 v=1):
        super(AEC_v2_m, self).__init__()

        self.encoders_v0 = Encoder(n_layers, n_input[0], n_z)
        self.encoders_v1 = Encoder(n_layers, n_input[1], n_z)
        self.decoders_v0 = Decoder(n_layers, n_input[0], n_z)
        self.decoders_v1 = Decoder(n_layers, n_input[1], n_z)

        self.v = v

        self.cluster_layer_v0 = nn.Parameter(torch.Tensor(n_h, n_z))
        self.cluster_layer_v1 = nn.Parameter(torch.Tensor(n_h, n_z))

        torch.nn.init.xavier_normal_(self.cluster_layer_v0.data)
        torch.nn.init.xavier_normal_(self.cluster_layer_v1.data)

    def forward(self, x0, x1, w):
        z0 = self.encoders_v0(x0)
        z1 = self.encoders_v1(x1)

        xb0 = self.decoders_v0(z0)
        xb1 = self.decoders_v1(z1)

        z0c = z0[torch.where(w[:, 0] == 1)]
        z1c = z1[torch.where(w[:, 1] == 1)]

        q0 = 1.0 / (1.0 + torch.sum(torch.pow(z0c.unsqueeze(1) - self.cluster_layer_v0, 2), 2) / self.v)
        q0 = q0.pow((self.v + 1.0) / 2.0)
        q0 = (q0.t() / torch.sum(q0, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z1c.unsqueeze(1) - self.cluster_layer_v1, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        return xb0, xb1, q0, q1, z0, z1


class AEC_v2_u(nn.Module):

    def __init__(self,
                 n_layers,
                 n_input,
                 n_z,
                 n_h,
                 v=1):
        super(AEC_v2_u, self).__init__()

        self.encoders_v0 = Encoder(n_layers, n_input[0], n_z)
        self.encoders_v1 = Encoder(n_layers, n_input[1], n_z)
        self.decoders_v0 = Decoder(n_layers, n_input[0], n_z)
        self.decoders_v1 = Decoder(n_layers, n_input[1], n_z)

        self.v = v

        self.cluster_layer_v0 = nn.Parameter(torch.Tensor(n_h, n_z))
        self.cluster_layer_v1 = nn.Parameter(torch.Tensor(n_h, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer_v0.data)
        torch.nn.init.xavier_normal_(self.cluster_layer_v1.data)

    def forward(self, x0, x1):
        z0 = self.encoders_v0(x0)
        z1 = self.encoders_v1(x1)

        xb0 = self.decoders_v0(z0)
        xb1 = self.decoders_v1(z1)

        q0 = 1.0 / (1.0 + torch.sum(torch.pow(z0.unsqueeze(1) - self.cluster_layer_v0, 2), 2) / self.v)
        q0 = q0.pow((self.v + 1.0) / 2.0)
        q0 = (q0.t() / torch.sum(q0, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z1.unsqueeze(1) - self.cluster_layer_v1, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        return xb0, xb1, q0, q1, z0, z1
