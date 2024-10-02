import torch
import numpy as np
from tqdm import tqdm

def get_hutchinson_fn(fn):
  """Estimate the trace of the Hessian of `fn` using the Hutchinson-Skilling trace estimator."""

  def hutchinson_fn(t, x, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(t, x, eps) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    return grad_fn_eps

  return hutchinson_fn

class STSL:
    def __init__(self, env, gamma=2., eta=0.02):
        self.env = env
        self.gamma = gamma
        self.eta = eta
        
    def fn(self, t, xt, eps):
        fn_x = self.env.eps_to_score(t, self.env.eps(t, xt))
        fn_x_plus_eps = self.env.eps_to_score(t, self.env.eps(t, xt + eps))
        return fn_x_plus_eps - fn_x
        
    def stochastic_averaging(self, t, xt, k=5):
        for _ in range(k):
            xt.grad = None
            with torch.enable_grad():
                x0 = self.env.denoise(t, xt)
                loss = (-self.env.logp_y_x(x0, differentiable=True)).clip(min=1e-7).sqrt()
                
                # consistency loss
                pixel_x0 = self.env.decode(x0, differentiable=True).reshape(-1, *self.env.x_shape)
                Ax0 = self.env.operator.forward(pixel_x0)

                ATy = self.env.operator.transpose(self.env.measurement)
                ATAx0 = self.env.operator.transpose(Ax0)

                x0_recon = self.env.encode(ATy + pixel_x0 - ATAx0).reshape(-1, self.env.ndims)
                
                loss = loss + 0.1 * (x0 - x0_recon).norm()
                loss.backward()
            xt_grad = xt.grad
            
            if self.eta > 0:
                eps = torch.randn_like(xt)
                hutchinson_fn = get_hutchinson_fn(self.fn)
                correction = hutchinson_fn(t, xt, eps) / np.prod(xt.shape)

                assert xt_grad.shape == correction.shape and xt_grad.shape == xt.shape
            else:
                correction = 0.
            
            return xt_grad * self.gamma + correction * self.eta
            

    @torch.no_grad()
    def solve(self, xT):
        xt = xT.requires_grad_(True)
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1)
        for t in pbar:
            eps = self.env.eps(t, xt)
            xt.data = self.env.step(t, xt, eps)
            xt.data -= self.stochastic_averaging(t, xt)

        return xt.detach()
