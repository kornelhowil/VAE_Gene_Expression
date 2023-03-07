import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 sizes: tuple
                 ) -> None:
        """
        Initializes Decoder object.

        Parameters
        ----------
        sizes : tuple
            Tuple of sizes of linear layers.
        """
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
    def log_prob(x: torch.Tensor,
                 mu: torch.Tensor
                 ) -> torch.Tensor:
        """
        Calculated log probability.

        This function calculates log probability of x coming from a
        normal distribution defined by means  mu and variance equal to 1.

        Parameters
        ----------
        x : torch.Tensor
            1D or 2D tensor of numbers.
        mu : torch.Tensor
            1D or 2D tensor of means which defines probability distribution.
            Has the same shape as x.

        Returns
        ----------
        torch.Tensor
            Log probability of x coming from normal distribution defined by mu.
        """
        dist = torch.distributions.Normal(mu, 1)
        log_prob_tensor = dist.log_prob(x)
        log_prob_sum = torch.sum(log_prob_tensor)
        return log_prob_sum

    def forward(self,
                z: torch.Tensor
                ) -> torch.Tensor:
        """
        Calculated forward step of the decoder.

        Parameters
        ----------
        z : torch.Tensor
            1D or 2D tensor in the latent space.
            Shape must agree with the first layer of the decoder.

        Returns
        ----------
        torch.Tensor
            Result of the forward step of the decoder.
        """
        return self.decoder_layers(z)
