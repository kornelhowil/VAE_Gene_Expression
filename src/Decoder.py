import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, sizes):
        super(Decoder, self).__init__()
        layers = []
        for i in range(1, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i - 1], sizes[i]))
            layers.append(nn.BatchNorm1d(sizes[i]))
            layers.append(nn.Dropout())
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.ReLU())
        self.decoder_layers = nn.Sequential(*layers)

    @staticmethod
    def log_prob(x, p):
        dist = torch.distributions.Normal(p, 1)
        log_prob_tensor = dist.log_prob(x)
        log_prob_sum = torch.sum(log_prob_tensor)
        return log_prob_sum

    def forward(self, z):
        return self.decoder_layers(z)
