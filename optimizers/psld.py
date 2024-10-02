import torch
import numpy as np
from tqdm import tqdm

class PSLD:
    def __init__(self, env, gamma=1.):
        self.env = env
        self.gamma = gamma

    @torch.no_grad()
    def solve(self, xT):
        # generate trajectory w.r.t. nominal actions
        xt = xT.requires_grad_(True)
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1)
        for t in pbar:
            eps = self.env.eps(t, xt)
            xt.data = self.env.step(t, xt, eps)

            xt.grad = None
            with torch.enable_grad():
                x0 = self.env.denoise(t, xt)

                # conditional loss
                loss = (-self.env.logp_y_x(x0, differentiable=True)).sqrt()

                # consistency loss
                pixel_x0 = self.env.decode(x0, differentiable=True).reshape(-1, *self.env.x_shape)
                Ax0 = self.env.operator.forward(pixel_x0).type(xt.dtype)

                ATy = self.env.operator.transpose(self.env.measurement)
                ATAx0 = self.env.operator.transpose(Ax0).type(xt.dtype)
                
                x0_recon = self.env.encode(ATy + pixel_x0 - ATAx0).reshape(-1, self.env.ndims)

                loss = loss + 0.1 * (x0 - x0_recon).norm()
                loss.backward()
            pbar.set_postfix(loss=loss.item(), norm=(xt.grad * self.gamma).norm().item())
            xt.data -= xt.grad * self.gamma

        return xt.detach()
