import torch
import numpy as np
from tqdm import tqdm

class LGDMCJF:
    """
    Like LGDMC but jacobian free
    """
    def __init__(self, env, latent, gamma=1.):
        self.env = env
        self.gamma = gamma
        self.latent = latent
        self.n = 10

    @torch.no_grad()
    def solve(self, xT):
        
        xt = xT.requires_grad_(True)
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1)
        for t in pbar:
            
            rt = self.env.sigma(t)/np.sqrt(1 + self.env.sigma(t)**2)
            eps = self.env.eps(t, xt)
            xt.data = self.env.step(t, xt, eps)
            
            #Stom gradient computation on the model
            with torch.no_grad():
                x0 = self.env.denoise(t, xt)
            
            #Retain gradient to avoid jacobian calculation
            x0.requires_grad = True
            x0.retain_grad()

            #monte-carlo
            x0s = []
            for i in range(self.n):
                x0s.append(x0 + rt*torch.randn_like(x0))
            x0s = torch.stack(x0s, 1)

            
            with torch.enable_grad():
                loss = (-self.env.logsump_y_x(x0s)).clip(min=1e-7).sqrt()
                loss.backward()
            pbar.set_postfix(loss=loss.item(), norm=(xt.grad * self.gamma).norm().item())
            xt.data -= xt.grad * self.gamma
        
        return xt.detach()
