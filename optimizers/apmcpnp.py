import torch
import numpy as np
from tqdm import tqdm

class APMCPNP:
    """
    See https://arxiv.org/pdf/2310.10835
    """
    def __init__(self, env):
        self.env = env

    @torch.no_grad()
    def solve(self, xT):
        xt = xT.requires_grad_(True)
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1)
        for t in pbar:
            gamma = self.env.sigma(t)**2

            #compute gradient nll
            xt.requires_grad = True
            with torch.enable_grad():
                loss = (-self.env.logp_y_x(xt)).clip(min=1e-7).sqrt()
                loss.backward()

            xt_imp = xt - gamma*xt.grad
            
            eps = self.env.eps(t, xt_imp)
            score = self.env.eps_to_score(t, eps)
            
            xt.data += -gamma*(xt.grad - self.env.alpha(t)*score) #move accoring to gradient
            xt.data += np.sqrt(2*gamma)*self.env.sigma(t)*torch.randn_like(xt.data) #noising

        
        return xt.detach()
