import torch.nn as nn
import torch


class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class EdgeDensity(Density):
    def __init__(self, params_init={"beta": 0.1}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, e_density, beta=None):
        if beta is None:
            beta = self.get_beta()

        return 1e4 * beta / (1 + torch.exp(-(e_density - 0.8) * 10)), beta

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta


