import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def sampling(mu, var):
        eps = torch.randn_like(var)
        return eps * var + mu

    @staticmethod
    def kl_div(mu, sigma):
        dist1 = torch.distributions.Normal(mu, sigma)
        dist2 = torch.distributions.Normal(torch.zeros_like(mu), 1)
        return torch.sum(torch.distributions.kl.kl_divergence(dist1, dist2))

    def loss_function(self, data, beta):
        p, mu, var = self.forward(data)
        kl_loss = beta * self.kl_div(mu, var)
        recon_loss = -self.decoder.log_prob(data, p)
        return recon_loss + kl_loss, kl_loss, recon_loss

    def generate(self, z):
        with torch.no_grad():
            return self.decoder(z)

    def calc_latent(self, x):
        with torch.no_grad():
            mu, var = self.encoder(x)
            return self.sampling(mu, var)

    def forward(self, x):
        mu, var = self.encoder(x)
        z = self.sampling(mu, var)
        mu_d = self.decoder(z)
        return mu_d, mu, var

