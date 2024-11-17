import torch
import numpy as np
from tqdm import tqdm
import os

class LGDMC:
    """
    Like DPS but log sum exp over candidate x0s?
    """
    def __init__(self, env, latent, n, gamma=1.):
        self.env = env
        self.gamma = gamma
        self.latent = latent
        self.n = n

    @torch.no_grad()
    def solve(self, xT):
        
        xt = xT.requires_grad_(True)
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1, disable=os.environ.get("DISABLE_TQDM", False))
        for t in pbar:
            
            rt = self.env.sigma(t)/np.sqrt(1 + self.env.sigma(t)**2)
            eps = self.env.eps(t, xt)
            xt.data = self.env.step(t, xt, eps)

            xt.grad = None
            with torch.enable_grad():
                x0s = []
                x0 = self.env.denoise(t, xt)
                for i in range(self.n):
                    x0s.append(x0 + rt*torch.randn_like(x0))
                x0s = torch.stack(x0s, 1)
                loss = (-self.env.logsump_y_x(x0s)).clip(min=1e-7).sqrt()
                loss.backward()
            pbar.set_postfix(loss=loss.item(), norm=(xt.grad * self.gamma).norm().item())
            xt.data -= xt.grad * self.gamma
        
        return xt.detach()
