import torch
import numpy as np
from tqdm import tqdm
from optimizers.dcs import *

class RDPSEps(DCS):
    
    def compute_eps(self, t, xt, iters, verbose=False):
        """
        Main function for computing \nabla \log p_t(x_t | x_0). We output it in epsilon form.

        Inputs:
            t:        continuous time variable t \in [0, 1]
            xt:       current diffusion iterate xt
            iters:    number of inner optimization step
            verbose:  print extra stuff

        Outputs:
            eps:      original epsilon output of model (i.e., \nabla \log p_t(x_t))
            cond_eps: epsilon correction of our model (i.e., v such that v + eps = \nabla \log p_t(x_t | x_0))
        """

        x0 = self.env.denoise(t, xt)
        eps = self.env.x0_to_eps(t, x0, xt)

        cond_eps = torch.zeros_like(xt, requires_grad=True)
        optim = torch.optim.Adam([cond_eps], lr=1.)
        pbar = tqdm(range(iters), total=iters, disable=not verbose)

        early_stop = EarlyStop()
        for i in pbar:
            optim.zero_grad()
            with torch.enable_grad():
                x0_hat = self.env.eps_to_x0(t, eps + cond_eps, xt)
                if self.env.mask is not None:
                    x0_hat = self.blur(x0_hat.reshape(-1, *self.env.shape)).reshape(-1, self.env.ndims)
                nll = -self.env.logp_y_x(x0_hat) + 1e-3 * np.sqrt(self.env.alpha(t).item()) * cond_eps.norm() ** 2
                nll.backward(retain_graph=True)

            optim.step()

            if early_stop(nll / np.prod(cond_eps.shape)):
                break
            pbar.set_postfix(nll=nll.item() / np.prod(cond_eps.shape))

        return eps, cond_eps
