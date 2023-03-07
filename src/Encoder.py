import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, sizes):
        super(Encoder, self).__init__()
        layers = []
        for i in range(1, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i-1], sizes[i]))
            layers.append(nn.BatchNorm1d(sizes[i]))
            layers.append(nn.Dropout())
            layers.append(nn.ReLU())
        self.encoder_layers = nn.Sequential(*layers)
        self.output_layer1 = nn.Linear(sizes[-2], sizes[-1])  # mu
        self.output_layer2 = nn.Linear(sizes[-2], sizes[-1])  # var

    def forward(self, x):
        x = self.encoder_layers(x)
        mu = self.output_layer1(x)
        var = nn.functional.softplus(self.output_layer2(x)) + 1e-4
        return mu, var
