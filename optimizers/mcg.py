import torch
import numpy as np
from tqdm import tqdm

class MCG:
    def __init__(self, env, latent, gamma=1.):
        self.env = env
        self.gamma = gamma
        self.latent = latent
        
        if self.latent:
            ###In the case of latent diffusion, instead of running a decode -> project -> encode to compute f(Ax - y)
            self.latent_manifold_projection = lambda xtr, yi: self.env.operator.project(self.env.decode(xtr).reshape(-1,*self.env.x_shape),
                                                                                    self.env.measurement,
                                                                                    mask = self.env.mask
                                                                                   )
            
            self.manifold_projection = lambda xtr, yi: self.env.encode(self.latent_manifold_projection(xtr, yi)).reshape(-1,np.prod(self.env.shape))
        else:
            self.manifold_projection = lambda xtr, yi: self.env.operator.project(xtr.reshape(-1,*self.env.shape),
                                                                                 yi,
                                                                                 mask = self.env.mask
                                                                                ).reshape(-1,np.prod(self.env.shape))

    @torch.no_grad()
    def solve(self, xT):
        
        xt = xT
        
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1)
        
        for t in pbar:
            #Take unconditional step, and move along action
            eps = self.env.eps(t, xt)
                
            xt_prime = self.env.step(t, xt, eps).requires_grad_(True)

            #Compute -\rho \nabla \frac{1}{2} || A(\hat{x}_0) - y ||^2
            xt_prime.grad = None
            with torch.enable_grad():
                x0 = self.env.denoise(t, xt_prime)
                if self.latent:
                    loss = (-self.env.logp_y_x(x0, differentiable=True)).clip(min=1e-7).sqrt()
                else:
                    loss = (-self.env.logp_y_x(x0)).clip(min=1e-7).sqrt()
                loss = loss
                loss.backward()

            #Move along "action"
            xt_prime.data -= xt_prime.grad * self.gamma
            
            ###MCG Unique step: project onto "manifold" (lol ok) constraints
            #alpha_t = self.env.alpha(t)
            #yi = float(np.sqrt(alpha_t)) * self.env.measurement + float(np.sqrt(1.-alpha_t)) * torch.randn_like(self.env.measurement, device=self.env.device)
            yi = self.env.measurement
            xt = self.manifold_projection(xt_prime.data, yi)
                                            
        return xt


