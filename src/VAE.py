import torch
import torch.nn as nn
import torch.distributions as td
from src.Encoder import Encoder
from src.Decoder import Decoder


class VAE(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder
                 ) -> None:
        """
        Initializes VAE object.

        Parameters
        ----------
        encoder : Encoder
            Encoder to be used in VAE.
        decoder : Decoder
            Decoder to be used in VAE.
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def sampling(mu: torch.Tensor,
                 var: torch.Tensor
                 ) -> torch.Tensor:
        """
        Samples from normal distribution.

        This function uses the reparameterization trick from original paper (section 2.4).
        Samples n numbers from normal distribution with means mu and variances var.

        Parameters
        ----------
        mu : torch.Tensor
            1D Tensor of size n with means to be used in sampling.
        var : torch.Tensor
            1D Tensor of size n with variances to be used in sampling.

        Returns
        ----------
        torch.Tensor
            1D Tensor of size n with numbers sampled from normal distribution.
        """
        eps = torch.randn_like(var)
        return eps * var + mu

    @staticmethod
    def kl_div(mu: torch.Tensor,
               var: torch.Tensor
               ) -> torch.Tensor:
        """
        Calculated Kullback–Leibler divergence.

        This function calculated KL divergence between normal distribution
        defined by a tensor of n means mu and a tensor of variances var and
        a standard normal distribution with mean 0 and variance 1.

        Parameters
        ----------
        mu : torch.Tensor
            1D Tensor of size n with means defining distribution.
        var : torch.Tensor
            1D Tensor of size n with variances defining distribution.

        Returns
        ----------
        torch.Tensor
            Calculated Kullback–Leibler divergence.
        """
        dist1 = td.Normal(mu, var)
        dist2 = td.Normal(torch.zeros_like(mu), 1)
        return torch.sum(td.kl.kl_divergence(dist1, dist2))

    def loss_function(self,
                      data: torch.Tensor,
                      beta: int
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates loss of the VAE on the given data.

        This function calculated loss of the model on the given data.
        Used in backpropagation.

        Parameters
        ----------
        data : torch.Tensor
            1D Tensor of size n with means defining distribution.
        beta : int
            Positive integer used for scaling regularization loss.

        Returns
        ----------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of total, regularization loss and reconstruction loss.
        """
        p, mu, var = self.forward(data)
        kl_loss = beta * self.kl_div(mu, var)
        recon_loss = -self.decoder.log_prob(data, p)
        return recon_loss + kl_loss, kl_loss, recon_loss

    def generate(self,
                 z: torch.Tensor
                 ) -> torch.Tensor:
        """
        Decodes vector z.

        This function decodes vector z from latent space
        and returns tensor in the output space.

        Parameters
        ----------
        z : torch.Tensor
            1D Tensor of number in the latent space.
            Dimension must agree with the first layer of the decoder.

        Returns
        ----------
        torch.Tensor
            Tensor generated from z.
        """
        with torch.no_grad():
            return self.decoder(z)

    def calc_latent(self,
                    x: torch.Tensor
                    ) -> torch.Tensor:
        """
        Encodes tensor x and samples vector in the latent space.

        This function Encodes tensor x and samples vector in the latent space.
        Uses reparametrization trick.

        Parameters
        ----------
        x : torch.Tensor
            2D Tensor in the input space.
            Dimensions must agree with the first layer of the encoder.

        Returns
        ----------
        torch.Tensor
            1D tensor in latent space calculated from tensor in input space.
            Has the same size as the last layer of the encoder.
        """
        with torch.no_grad():
            mu, var = self.encoder(x)
            return self.sampling(mu, var)

    def forward(self,
                x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs forward step of the VAE.

        This function runs forward step of the VAE.
        It encodes x in the latent space and then decodes it back to the input space.

        Parameters
        ----------
        x : torch.Tensor
            2D Tensor in the input space.
            Dimensions must agree with the first layer of the encoder.

        Returns
        ----------
        torch.Tensor
            Result of the forward step of the VAE. Has the same shape as x.
        """
        mu, var = self.encoder(x)
        z = self.sampling(mu, var)
        mu_d = self.decoder(z)
        return mu_d, mu, var
