import torch
import numpy as np
from tqdm import tqdm

class DPSJF:
    """
    Jacobian-Free DPS
    """
    def __init__(self, env, latent, gamma=1.):
        self.env = env
        self.gamma = gamma
        self.latent = latent

    @torch.no_grad()
    def solve(self, xT):
        xt = xT.requires_grad_(True)
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1)
        for t in pbar:
            eps = self.env.eps(t, xt)
            xt.data = self.env.step(t, xt, eps)

            #Stom gradient computation on the model
            with torch.no_grad():
                x0 = self.env.denoise(t, xt)
            
            #Retain gradient to avoid jacobian calculation
            x0.requires_grad = True
            x0.retain_grad()
            
            with torch.enable_grad():
                if self.latent:
                    from PIL import Image
                    x_ = self.env.decode(x0)
                    x_ = (x_[0].reshape(*self.env.x_shape).detach().cpu().permute(1, 2, 0) * .5 + .5) * 255.
                    im = Image.fromarray(np.array(x_.int(), dtype=np.uint8))
                    im.save(f'/data/inverse/results/x0/{t:.3f}.png')
                    loss = (-self.env.logp_y_x(x0, differentiable=True)).sqrt()
                else:
                    loss = (-self.env.logp_y_x(x0)).clip(min=1e-7).sqrt()
                loss.backward()
            pbar.set_postfix(loss=loss.item(), norm=(x0.grad * self.gamma).norm().item())
            xt.data -= x0.grad * self.gamma
        
        return xt.detach()
