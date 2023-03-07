import torch
from src.Decoder import Decoder


class CustomDecoder(Decoder):
    @staticmethod
    def log_prob(x: torch.Tensor,
                 p: torch.Tensor
                 ) -> torch.Tensor:
        """
        Calculated log probability.

        This function calculates log probability of x coming from a
        Poisson distribution defined by probabilities p.

        Parameters
        ----------
        x : torch.Tensor
            1D tensor of numbers.
        p : torch.Tensor
            1D tensor of probabilities which defines probability distribution.
            Has the same shape as x.

        Returns
        ----------
        torch.Tensor
            Log probability of x coming from normal distribution defined by p.
        """
        dist = torch.distributions.poisson.Poisson(p + 1e-4)
        log_prob_tensor = dist.log_prob(x)
        log_prob_sum = torch.sum(log_prob_tensor)
        return log_prob_sum
