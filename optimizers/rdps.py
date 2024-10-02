import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
from torch.distributions.normal import Normal
from diffusion.diffusion import sqrt, clip

class EarlyStop:
    def __init__(self, env, sigma=0.1):
        self.env = env
        self.sigma = sigma

    def get_residual(self, x):
        x = x.reshape(-1, *self.env.shape)
        return self.env.measurement - self.env.operator.forward(x, mask=self.env.mask)

    def __call__(self, t, x):
        res = self.get_residual(x)
        s = self.sigma / np.sqrt(np.prod(res.shape))
        z = res.mean().abs() / s
        dist = Normal(loc=0., scale=1.)
        cdf = 2 * (1 - dist.cdf(z))
        thresh = np.sqrt(1 - self.env.alpha(t))
        return cdf > thresh

class RDPS:
    def __init__(self, env, iters=1, sigma=None, lr=5.):
        self.env = env
        self.iters = iters
        self.eta = 1.
        self.lr = lr
        self.sigma = sigma
        # self.blur = GaussianBlur(3, .25)
        self.blur = GaussianBlur(11, .25)
        
    def prepare_variables(self, t, xt, eps):
        if not hasattr(self, 'last_pred'):
            self.cond_score = torch.zeros_like(xt, requires_grad=True)
            self.optim = torch.optim.AdamW(
              [self.cond_score], 
              lr=self.lr, 
              weight_decay=self.lr * 10,
              eps=1e-4 if xt.dtype == torch.half else 1e-8,
            )
            self.last_pred = self.cond_score.data
        else:
            last_eps = self.env.x0_to_eps(t, self.last_pred, xt)
            last_cond_eps = (last_eps - eps)
            self.cond_score.data = self.env.eps_to_score(t, last_cond_eps)
           
        t = t.cpu()
        self.early_stop = EarlyStop(env=self.env, sigma=self.sigma)
    
    @torch.no_grad()
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
        
        bpd = 0.
        eps = self.env.eps(t, xt)
        self.prepare_variables(t, xt, eps)
        
        
        pbar = tqdm(range(iters), total=iters, disable=not verbose)
        for i in pbar:
            self.optim.zero_grad()
            with torch.enable_grad():
                cond_eps = self.env.score_to_eps(t, self.cond_score)
                x0_hat = self.env.eps_to_x0(t, eps + cond_eps, xt)
                
                if self.env.mask is not None:
                    x0_hat = self.blur(x0_hat.reshape(-1, *self.env.shape)).reshape(-1, self.env.ndims)
                
                nll = -self.env.logp_y_x(x0_hat)
                bpd = nll.item() / np.prod(xt.shape)
                nll.backward(retain_graph=True)
                
            self.optim.step()

            if self.early_stop(t, x0_hat):
                break
                
            pbar.set_postfix(nll=bpd, grad_norm=self.cond_score.grad.norm().item())

        cond_eps = self.env.score_to_eps(t, self.cond_score)
        self.last_pred = self.env.eps_to_x0(t, eps + cond_eps, xt)
        return eps + cond_eps, cond_eps, bpd

    def get_update(self, t, cond_eps):
        coef1, coef2, _ = self.env.get_coef(t, eta=self.eta)
        update = coef1 * -cond_eps * np.sqrt(1 - self.env.alpha(t)) / np.sqrt(self.env.alpha(t))
        update = update + coef2 * cond_eps
        return update

    def rewind(self, t, s, xt):
        x0 = self.env.denoise(t, xt)
        alphas = self.env.alpha(s)
        return sqrt(alphas) * x0 + sqrt(1 - alphas) * torch.randn_like(x0)

    @torch.no_grad()
    def solve(self, xT, back_steps=0, skip=5):
        """
        RDPS Solver.

        Inputs:
            xT:              initial state

        Outputs:
            cond_eps_list:   all the optimized perturbations v
            xt:              a.k.a. x0 | y
        """

        xt = xT
        pbar = tqdm(self.env.timesteps, total=self.env.num_steps)
        for i, t in enumerate(pbar):
            if i % max(1, back_steps * skip) == 0:
                start = max(0, i - back_steps * skip)
                idxs = list(range(start, i, skip))
                for j, s in zip(idxs, self.env.timesteps[idxs]):
                    if j == start:
                        xt = self.rewind(t, s, xt)
                    eps, cond_eps, loss = self.compute_eps(s, xt, self.iters)
                    xt = self.env.step(s, xt, eps, eta=self.eta, s=s - (self.env.dt * skip))

            eps, cond_eps, loss = self.compute_eps(t, xt, self.iters)
            xt = self.env.step(t, xt, eps, eta=self.eta, s=t - self.env.dt)
            pbar.set_postfix(guidance=self.get_update(t, cond_eps).norm().item(), loss=loss)

        return xt