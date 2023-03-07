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
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def sampling(mu: torch.Tensor,
                 var: torch.Tensor
                 ) -> torch.Tensor:
        eps = torch.randn_like(var)
        return eps * var + mu

    @staticmethod
    def kl_div(mu: torch.Tensor,
               sigma: torch.Tensor
               ) -> torch.Tensor:
        dist1 = td.Normal(mu, sigma)
        dist2 = td.Normal(torch.zeros_like(mu), 1)
        return torch.sum(td.kl.kl_divergence(dist1, dist2))

    def loss_function(self,
                      data: torch.Tensor,
                      beta: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p, mu, var = self.forward(data)
        kl_loss = beta * self.kl_div(mu, var)
        recon_loss = -self.decoder.log_prob(data, p)
        return recon_loss + kl_loss, kl_loss, recon_loss

    def generate(self,
                 z: torch.Tensor
                 ) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(z)

    def calc_latent(self,
                    x: torch.Tensor
                    ) -> torch.Tensor:
        with torch.no_grad():
            mu, var = self.encoder(x)
            return self.sampling(mu, var)

    def forward(self,
                x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, var = self.encoder(x)
        z = self.sampling(mu, var)
        mu_d = self.decoder(z)
        return mu_d, mu, var

