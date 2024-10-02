import torch
import numpy as np
from tqdm import tqdm
from utils.sde_lib import VPSDE

class DDNM:
    def __init__(self, env, gamma=1.):
        self.env = env
        self.gamma = gamma
        
    @torch.no_grad()
    def solve(self, xT):
        # generate trajectory w.r.t. nominal actions
        xt = xT
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1)
        for t in pbar:
            x_0_hat = self.env.denoise(t, xt).reshape(-1, *self.env.shape)

            ATy = self.env.operator.transpose(self.env.measurement, mask=self.env.mask)
            ATAx = self.env.operator.transpose(self.env.operator.forward(x_0_hat, mask=self.env.mask))
            x_0 = x_0_hat + (ATy - ATAx) * self.gamma

            mean, std = VPSDE().marginal_prob(x_0, t[None])
            xt = mean + std * torch.randn_like(mean)

            xt = xt.reshape(-1, np.prod(self.env.shape))

        return xt