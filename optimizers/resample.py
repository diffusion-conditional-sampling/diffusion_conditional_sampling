import torch
import numpy as np
from tqdm import tqdm

class ReSample:
    def __init__(self, env, gamma=.5):
        self.env = env
        self.gamma = gamma

    def operator_fn(self, x):
        x = x.reshape(-1, *self.env.x_shape)
        y = self.env.operator.forward(x, mask=self.env.mask)
        return y

    def stochastic_resample(self, t, pseudo_x0, xt, sigma):
        noise = torch.randn_like(pseudo_x0)
        a_t = self.env.alpha(t)
        x_t = xt
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t)/(sigma + 1 - a_t) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t)))

    @torch.no_grad()
    def pseudo_tweedies(self, t, eps, xt):
        """
        Pseudo x_0 computed in ReSample. Rather than
        x0 = xt + sigma ** 2 score
        they compute
        x0 = xt + sigma ** 3 score
    https://github.com/soominkwon/resample/blob/03f5d069953cad42f8e0f8f44cddb6bed375ce91/ldm/models/diffusion/ddim.py#L541
        """
        alpha = self.env.alpha(t)
        sqrt_alpha = torch.sqrt(alpha)

        return (xt - (1 - alpha) * eps) / sqrt_alpha

    def pixel_optimization(self, measurement, x_prime, operator_fn, eps=1e-3, max_iters=2000):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            x_prime:               Estimation of \hat{x}_0 using Tweedie's formula
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss() # MSE loss

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=1e-2) # Initializing optimizer
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop

        for _ in range(max_iters):
            optimizer.zero_grad()

            with torch.enable_grad():
                measurement_loss = loss(measurement, operator_fn(opt_var))
                measurement_loss.backward()

            optimizer.step()

            # Convergence criteria
            if measurement_loss < eps ** 2: # needs tuning according to noise level for early stopping
                break

        return opt_var


    def latent_optimization(self, measurement, z_init, operator_fn, eps=1e-3, max_iters=500, lr=None):

        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations

        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()

        if lr is None:
            lr_val = 5e-3
        else:
            lr_val = lr.item()

        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=lr_val) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop
        init_loss = 0
        losses = []

        for itr in range(max_iters):
            optimizer.zero_grad()

            with torch.enable_grad():
                x_init = self.env.decode(z_init, differentiable=True)
                output = loss(measurement, operator_fn(x_init))
                output.backward() # Take GD step
            optimizer.step()

            cur_loss = output.detach().cpu().numpy()

            # Convergence criteria

            if itr < 200: # may need tuning for early stopping
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)

            if cur_loss < eps**2:  # needs tuning according to noise level for early stopping
                break


        return z_init

    @torch.no_grad()
    def solve(self, xT):
        xt = xT.requires_grad_(True)
        pbar = tqdm(self.env.timesteps[:-1], total=self.env.num_steps - 1)
        for i, t in enumerate(pbar):
            pbar.set_postfix(mode='dps')
            # unconditional step
            eps = self.env.eps(t, xt)
            xt.data = self.env.step(t, xt, eps)

            # DPS step (not described in paper, but present in code)
            xt.grad = None
            with torch.enable_grad():
                x0 = self.env.denoise(t, xt)
                loss = (-self.env.logp_y_x(x0, differentiable=True)).clip(min=1e-7).sqrt()
                loss.backward()
            xt.data -= xt.grad * self.gamma * self.env.alpha(t)

            # (maybe) perform hard consistency optimization
            index = self.env.get_index(t)
            timesteps = self.env.ddim_timesteps
            inter_timesteps = 5
            a_t = self.env.alpha(t)
            a_prev = self.env.alpha(t - self.env.dt)

            splits = 3
            index_split = self.env.num_steps // splits

            # Performing time-travel if in selected indices
            if index <= (self.env.num_steps - index_split) and index > 0:
                x = xt.detach().clone()

                # Performing only every 10 steps (or so)
                # TODO: also make this not hard-coded
                if index % 10 == 0 :
                    xs = x
                    assert self.env.timesteps[i] == t, f"{self.env.timesteps[i]}, {t}"
                    start, end = i + 1, min(i + 1 + inter_timesteps, len(self.env.timesteps[:-1]))
                    for s in self.env.timesteps[:-1][start:end]:
                        eps_s = self.env.eps(s, xs)
                        xs = self.env.step(s, xs, eps_s)

                    pseudo_x0 = self.pseudo_tweedies(t, eps_s, xs)

                    # Some arbitrary scheduling for sigma
                    if index >= 0:
                        sigma = 40 * (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
                    else:
                        sigma = 0.5

                    # Pixel-based optimization for second stage
                    if index >= index_split:
                        pbar.set_postfix(mode='pixel_consistency')
                        # Enforcing consistency via pixel-based optimization
                        pseudo_x0 = pseudo_x0.detach()
                        pseudo_x0 = self.env.decode(pseudo_x0) # Get \hat{x}_0 into pixel space

                        pseudo_x0 = self.pixel_optimization(measurement=self.env.measurement,
                                                          x_prime=pseudo_x0,
                                                          operator_fn=self.operator_fn)

                        pseudo_x0 = self.env.encode(pseudo_x0) # Going back into latent space

                    # Latent-based optimization for third stage
                    elif index < index_split: # Needs to (possibly) be tuned
                        pbar.set_postfix(mode='latent_consistency')
                        # Enforcing consistency via latent space optimization
                        pseudo_x0 = self.latent_optimization(measurement=self.env.measurement,
                                                             z_init=pseudo_x0.detach(),
                                                             operator_fn=self.operator_fn)

                        sigma = 40 * (1-a_prev)/(1 - a_t) * (1 - a_t / a_prev) # Change the 40 value for each task


                    xt.data = self.stochastic_resample(t=t - self.env.dt, pseudo_x0=pseudo_x0.detach(), xt=x, sigma=sigma).detach()

        return xt.detach()
