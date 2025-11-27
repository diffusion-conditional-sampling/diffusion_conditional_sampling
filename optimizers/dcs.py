import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
from torch.distributions.normal import Normal
from diffusion.diffusion import sqrt, clip
from utils.measurements import InpaintingOperator

class EarlyStop:
    def __init__(self, env, sigma=0.1, max_cdf=None, noise_distribution='gaussian'):
        self.env = env
        self.sigma = sigma
        self.max_cdf = max_cdf if max_cdf is not None else 1.
        self.noise_distribution = noise_distribution

        if noise_distribution == 'laplace':
            self.noise_distribution = torch.distributions.laplace.Laplace(loc=0., scale=1.)
        elif noise_distribution == 'gaussian':
            self.noise_distribution = torch.distributions.normal.Normal(loc=0., scale=1.)
        else:
            raise ValueError(f"Unknown noise distribution: {noise_distribution}")

    def get_residual(self, x):
        N, _ = x.shape
        x = x.reshape(N, *self.env.shape)
        return (self.env.measurement - self.env.operator.forward(x, mask=self.env.mask)).reshape(N, -1)

    def __call__(self, t, x):
        cdf, thresh = self._get_cdf_thresh(t, x)
        return cdf > thresh

    def _get_cdf_thresh(self, t, x):
        res = self.get_residual(x)
        s = self.sigma
        z = res.abs() / s
        cdf = 2 * self.noise_distribution.cdf(-z).mean(dim=-1)
        thresh = np.sqrt(1 - self.env.alpha(t)) * self.max_cdf
        return cdf, thresh

class DCS:
    def __init__(self, env, iters=1, sigma=None, lr=5., const=1., patience=0, noise_distribution='gaussian'):
        self.env = env
        self.iters = iters
        self.lr = lr
        self.sigma = sigma
        self.blur = GaussianBlur(11, .25)
        self.max_cdf = None
        self.const = const
        self.patience = patience
        self.noise_distribution = noise_distribution

    def eta(self, t):
        return (1 - self.env.alpha(t - self.env.dt) - 1e-7) / (self.env.dsigma(t) + 1e-7)

    def snr(self, t):
        alphat = self.env.alpha(t)
        return sqrt(alphat) / sqrt(1 - alphat)
    
    def get_update(self, t, cond_eps):
        coef1, coef2, _ = self.env.get_coef(t, eta=self.eta(t))
        update = coef1 * -cond_eps * np.sqrt(1 - self.env.alpha(t)) / np.sqrt(self.env.alpha(t))
        update = update + coef2 * cond_eps
        return update
        
    def prepare_variables(self, t, xt, eps):
        self.early_stop = EarlyStop(env=self.env, sigma=self.sigma, max_cdf=self.max_cdf, noise_distribution=self.noise_distribution)
        if not hasattr(self, 'last_pred'):
            self.cond_eps = torch.zeros_like(xt, requires_grad=True)
            self.optim = torch.optim.AdamW(
              [self.cond_eps], 
              lr=self.lr, 
              weight_decay=1.
            )
            self.last_pred = self.cond_eps.data
        else:
            last_eps = self.env.x0_to_eps(t, self.last_pred, xt)
            self.cond_eps.data = last_eps - eps
            if self.env.mask is not None:
                masked = (1 - self.env.mask).tile(1, 3, 1, 1).reshape(-1, np.prod(self.env.shape))
                self.cond_eps.data = torch.where(masked != 0, torch.zeros_like(self.cond_eps.data), self.cond_eps.data)
                norm = (masked * self.cond_eps.data).norm()
                assert norm < 1e-3, f"{norm}, {masked.shape}, {self.env.mask.mean()}"
          
        self.optim.param_groups[0]['lr'] = self.lr * self.snr(t)
        self.stopped = torch.full(size=(len(xt),), fill_value=False, dtype=bool, device=xt.device)
        
    def optimize_and_check_if_stopped(self, t, x0_hat, i=0):
        # batched early stopping
        early_stop = torch.logical_and(
            self.early_stop(t, x0_hat),
            torch.full_like(self.stopped, fill_value=i > self.patience)
        )
        self.stopped = torch.where(early_stop, torch.ones_like(self.stopped), self.stopped)
        cond_eps_before = self.cond_eps.data.clone()
        self.optim.step()
        self.cond_eps.data = torch.where(self.stopped[:, None], cond_eps_before, self.cond_eps.data)
            
        return self.stopped.all()
    
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
                x0_hat = self.env.eps_to_x0(t, eps + self.cond_eps, xt)
                
                if self.env.mask is not None and self.env.mask.mean() < 0.5:
                    x0_hat = self.blur(x0_hat.reshape(-1, *self.env.shape)).reshape(-1, self.env.ndims)
                
                nll = -self.env.logp_y_x(x0_hat).mean()
                bpd = nll.item() / np.prod(xt.shape)
                nll.backward(retain_graph=True)
                
            if self.optimize_and_check_if_stopped(t, x0_hat, i=i):
                break
                
            pbar.set_postfix(nll=bpd, grad_norm=self.cond_eps.grad.norm().item())        

        self.last_pred = self.env.eps_to_x0(t, eps + self.cond_eps, xt)
        cdf, thresh = self.early_stop._get_cdf_thresh(t, self.last_pred)
        self.max_cdf = cdf if self.max_cdf is None else torch.where(cdf > self.max_cdf, cdf, self.max_cdf)
        return eps, self.cond_eps.data / self.snr(t), bpd

    @torch.no_grad()
    def solve(self, xT):
        """
        DCS Solver.

        Inputs:
            xT:              initial state

        Outputs:
            cond_eps_list:   all the optimized perturbations v
            xt:              a.k.a. x0 | y
        """

        xt = xT
        pbar = tqdm(self.env.timesteps, total=self.env.num_steps)
        cond_eps = torch.zeros_like(xT)
        loss = 0.
        for i, t in enumerate(pbar):
            eps, cond_eps, loss = self.compute_eps(t, xt, self.iters)
            eps = eps + cond_eps
            xt = self.env.step(t, xt, eps, eta=self.eta(t), s=t - self.env.dt)
            pbar.set_postfix(guidance=self.get_update(t, cond_eps).norm().item(), loss=loss)

        return xt

