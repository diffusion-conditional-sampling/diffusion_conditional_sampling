import torch
import numpy as np
from tqdm import tqdm
from diffusion.diffusion import sqrt

class RED:
    def __init__(self, env, latent=False, lam=.25):
        self.env = env
        self.lam = lam
        self.latent = latent

    def add_noise(self, x0, t):
        alphat = self.env.alpha(t)
        eps = torch.randn_like(x0)
        xt = sqrt(alphat) * x0 + sqrt(1 - alphat) * eps
        return xt, eps

    def eps_weight(self, t):
        alphat = self.env.alpha(t)
        snr_inv = sqrt(1 - alphat) / sqrt(alphat)
        return snr_inv * self.lam

    def mse(self, x):
        x = x.reshape(-1, *self.env.shape)
        return 0.5 * ((self.env.measurement - self.env.operator.forward(x, mask=self.env.mask)) ** 2).mean()

    @torch.no_grad()
    def solve(self, _):
        ATy = self.env.operator.transpose(self.env.measurement, mask=self.env.mask).reshape(-1, np.prod(self.env.shape))
        x0 = ATy.requires_grad_(True)
        
        optim = torch.optim.Adam([x0], lr=0.1, betas=(0.9, 0.99))
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1)
        for t in pbar:
            xt, eps = self.add_noise(x0, t)
            eps_hat = self.env.eps(t, xt)

            x0.grad = None
            with torch.enable_grad():
                eps_loss = ((eps_hat - eps) * x0).mean()
              
                if self.latent:
                    assert False
                else:
                    consistency_loss = self.mse(x0)
                
                assert consistency_loss >= 0., f"consistency loss: {consistency_loss}"
                loss = consistency_loss + self.eps_weight(t) * eps_loss
                loss.backward()
                optim.step()
            pbar.set_postfix(loss=loss.item(), norm=(x0.grad).norm().item())
        
        return x0.detach()
