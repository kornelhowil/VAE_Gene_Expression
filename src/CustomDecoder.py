import torch
from src.Decoder import Decoder


class CustomDecoder(Decoder):
    @staticmethod
    def log_prob(x, p):
        dist = torch.distributions.Normal(p, 1)
        log_prob_tensor = dist.log_prob(x)
        log_prob_sum = torch.sum(log_prob_tensor)
        return log_prob_sum
