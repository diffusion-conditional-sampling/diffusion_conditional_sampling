import numpy as np
import torch
from utils.sde_lib import VPSDE
from diffusion.latent_pipeline import retrieve_timesteps

def sqrt(x):
    if isinstance(x, float) or isinstance(x, np.float32):
        return np.sqrt(x)
    elif isinstance(x, np.ndarray):
        return np.sqrt(x)
    elif isinstance(x, torch.Tensor):
        return torch.sqrt(x)
    else:
        print("Unknown type!", type(x))
        raise NotImplementedError()

def clip(x, *args, **kwargs):
    if isinstance(x, float):
        return np.clip(x, *args, **kwargs)
    elif isinstance(x, np.ndarray):
        return np.clip(x, *args, **kwargs)
    elif isinstance(x, torch.Tensor):
        return torch.clip(x, *args, **kwargs)
    else:
        print("Unknown type!", type(x))
        raise NotImplementedError()

class Diffusion:
    def __init__(
        self,
        shape,
        measurement,
        operator,
        mask,
        dtype,
        num_steps=100,
        device='cpu',
        eps=1e-3,
    ):
        self.shape = shape
        self.ndims = np.prod(shape) # eg. a shape of (1, 28, 28) would give 1*28*28=784
        self.measurement = measurement
        self.operator = operator
        self.mask = mask
        self.num_steps = num_steps
        self.timesteps = torch.linspace(1, eps, num_steps, device=device)
        self.inv_timesteps = lambda t: ((1 - t) * num_steps).long()
        self.device = device
        self.dt = (1 - eps) / self.num_steps
        self.dtype = dtype
        

    def sample_xT(self, n, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        device = self.device

        init_x = torch.randn((n, self.ndims), device=device, dtype=self.dtype)

        return init_x

    def logp_y_x(self, x):
        x = x.reshape(-1, *self.shape)
        logp = -0.5 * ((self.measurement - self.operator.forward(x, mask=self.mask)) ** 2).sum()
        return logp

    def sigma(self, t):
        return sqrt(1 - self.alpha(t))

    def score_to_eps(self, t, score, scale=1):
        # computes epsilon from score
        return -score * self.sigma(t) * scale

    def eps_to_score(self, t, eps):
        # and vice versa
        return -eps / self.sigma(t)

    def eps_to_x0(self, t, eps, xt):
        # computes x0 from epsilon
        alpha = self.alpha(t)
        sqrt_alpha = sqrt(alpha)
        x0 = (xt - sqrt(1 - alpha) * eps) / sqrt_alpha
        return x0.clip(-1, 1)

    def x0_to_eps(self, t, x0, xt):
        # computes epsilon from x0
        alpha = self.alpha(t)
        sqrt_alpha = sqrt(alpha)

        return (xt - sqrt_alpha * x0) / sqrt(1 - alpha)

class DDPMDiffusion(Diffusion):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dt = 1 / self.num_steps

    def get_eta(self, target_sigma):
        # experimental: adjusting the noise strength in DDIM based on measurement noise eta
        t = self.timesteps[-1]
        eta = target_sigma / self.sigma(t)
        return eta

    def get_coef(self, t, eta=1.):
        alphatm1 = self.alpha(t - self.dt)

        dsigma = self.dsigma(t)
        dsigma = clip(dsigma * eta, 0., (1 - alphatm1 - 1e-7).item())

        coef1 = sqrt(alphatm1)
        coef2 = sqrt(1 - alphatm1 - dsigma ** 2)
        return coef1, coef2, dsigma
    
    def dsigma(self, t):
        # computes dsigma_t in https://arxiv.org/pdf/2010.02502.pdf
        alpha = self.alpha(t)
        alphatm1 = self.alpha(t - self.dt)

        dsigma = sqrt((1 - alphatm1) * (1 - alpha)) * sqrt(1 - alpha / alphatm1)

        return dsigma

    def step(self, t, xt, eps, eta=1., s=None):
        """
        computes one (potentially noisy) DDIM step (eq. 12 in https://arxiv.org/pdf/2010.02502.pdf)

        Inputs:
            t:        continuous time variable t \in [0, 1]
            xt:       current diffusion iterate x_t
            eps:      epsilon noise function
        Outputs:
            xtm1:     x_{t-1}
        """
        if s is None:
            s = t - self.dt
        
        x0 = self.eps_to_x0(t, eps, xt)

        alphatm1 = self.alpha(s)

        dsigma = self.dsigma(t)
        dsigma = clip(dsigma * eta, 0., (1 - alphatm1 - 1e-7).item())

        coef1 = sqrt(alphatm1)
        coef2 = sqrt(1 - alphatm1 - dsigma ** 2)
        
        xtm1 = coef1 * x0 + coef2 * eps + dsigma * torch.randn_like(xt)

        return xtm1

    def discretize_time(self, t):
        t_int = (t * (self.get_model_steps() - 1)).int().reshape(-1)
        return t_int
    
class HFDiffusion(DDPMDiffusion):
    """
    Diffusion class compatible with Hugging Face SD models.
    """
    def __init__(
        self,
        pipeline,
        sampler=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline
        self.pipeline.scheduler.set_timesteps(self.num_steps)
        self.pipeline.unet.eval()
        timesteps = retrieve_timesteps(self.pipeline.scheduler, self.num_steps - 1, 'cpu')[0].numpy()[::-1]
        self.alphas = self.pipeline.scheduler.alphas_cumprod.numpy()[timesteps]
        self.model_t = timesteps
        self.dtype = pipeline.dtype
        self.measurement = self.measurement.type(self.dtype)
            
    def alpha(self, t):
        t = self.discretize_time(t)
        return self.alphas[t]

    def get_model_steps(self):
        return self.num_steps - 1
    
    def denoise(self, t, x):
        eps = self.eps(t, x)
        return self.eps_to_x0(t, eps, x)
    
    def eps(self, t, x):
        input_t = torch.tensor(self.model_t[self.discretize_time(t)], device=x.device)
        x = x.reshape(-1, *self.shape)

        # predict the noise residual
        model_output = self.pipeline.unet(x, input_t).sample
        
        if self.pipeline.scheduler.config.prediction_type == 'epsilon':
            eps = model_output
        elif self.pipeline.scheduler.config.prediction_type == 'sample':
            eps = self.x0_to_eps(t, x0, x)
        elif self.pipeline.scheduler.config.prediction_type == 'v_prediction':
            eps = sqrt(self.alpha(t)) * model_output + sqrt(1 - self.alpha(t)) * x
        else:
            raise NotImplementedError()
                
        eps = eps.reshape(-1, self.ndims)
        return eps
    
class SimpleLatentDiffusion(DDPMDiffusion):
    """
    Diffusion class compatible with Ommer Lab Latent Diffusion models.
    """
    def __init__(
        self,
        *args,
        loss_type='latent',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type

    def alpha(self, t):
        # computes alpha_t in https://arxiv.org/pdf/2010.02502.pdf
        index = self.get_index(t)
        return self.sampler.ddim_alphas[index]

    def eps(self, t, x):
        x0 = self.denoise(t, x)
        return self.x0_to_eps(t, x0, x)

    def logp_y_x(self, x, differentiable=False):
        logp = 0.
        if self.loss_type == 'latent' or self.loss_type == 'both':
            logp += -0.5 * ((self.z_measurement - x.reshape(-1, *self.shape)) ** 2).sum()

        if self.loss_type == 'pixel' or self.loss_type == 'both':
            x = self.decode(x, differentiable=differentiable)
            x = x.reshape(-1, *self.x_shape)
            x = self.operator.forward(x, mask=self.mask)
            logp += -0.5 * ((self.measurement - x) ** 2).sum()

        return logp

class LatentDiffusion(SimpleLatentDiffusion):
    """
    Diffusion class compatible with Ommer Lab Latent Diffusion models.
    """
    def __init__(
        self,
        sampler,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
        self.sampler.make_schedule(ddim_num_steps=self.num_steps, ddim_eta=1., verbose=False)
        self.ddim_timesteps = self.sampler.ddim_timesteps
        self.time_range = torch.tensor(np.flip(self.ddim_timesteps).copy()).to(self.device)
        assert self.ddim_timesteps.shape[0] == self.num_steps
        self.x_shape = self.raw_x_shape = self.shape
        self.x_ndims = np.prod(self.x_shape)
        self.shape = (
             sampler.model.model.diffusion_model.in_channels,
             sampler.model.model.diffusion_model.image_size,
             sampler.model.model.diffusion_model.image_size
        )
        self.ndims = np.prod(self.shape)
        self.num_timesteps = torch.tensor(self.sampler.model.num_timesteps, device=self.device)

        if self.loss_type == 'latent' or self.loss_type == 'both':
            ATy = self.operator.transpose(self.measurement, mask=self.mask)
            self.z_measurement = self.encode(ATy).reshape(-1, *self.shape)

    def get_index(self, t):
        return (t * (self.num_steps - 1)).int().reshape(-1)

    def denoise(self, t, x):
        x = x.reshape(-1, *self.shape)
        index = self.get_index(t)
        t = self.time_range[self.num_steps - index - 1]
        xtm1, x0, eps = self.sampler.p_sample_ddim(x, c=None, t=t, index=index)
        return x0.reshape(-1, self.ndims)

    def get_model_steps(self):
        return self.num_timesteps

    def decode(self, x, differentiable=False):
        if differentiable:
            decode_fn = self.sampler.model.differentiable_decode_first_stage
        else:
            decode_fn = self.sampler.model.decode_first_stage

        return decode_fn(x.reshape(-1, *self.shape)).reshape(-1, self.x_ndims)

    def encode(self, x):
        encode_fn = self.sampler.model.encode_first_stage

        return encode_fn(x.reshape(-1, *self.x_shape)).reshape(-1, self.ndims)

    def logp_y_x(self, x, differentiable=False):
        logp = 0.
        if self.loss_type == 'latent' or self.loss_type == 'both':
            logp += -0.5 * ((self.z_measurement - x.reshape(-1, *self.shape)) ** 2).sum()

        if self.loss_type == 'pixel' or self.loss_type == 'both':
            x = self.decode(x, differentiable=differentiable)
            x = x.reshape(-1, *self.x_shape)
            x = self.operator.forward(x, mask=self.mask)
            logp += -0.5 * ((self.measurement - x) ** 2).sum()

        return logp
    
class StableDiffusion(SimpleLatentDiffusion):
    """
    Diffusion class compatible with Hugging Face SD models.
    """
    def __init__(
        self,
        pipeline,
        sampler=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline
        self.pipeline.scheduler.set_timesteps(self.num_steps)
        self.pipeline.unet.eval()
        self.x_shape = self.shape
        self.x_ndims = self.ndims
        self.shape = (
            self.pipeline.unet.config.in_channels,
            self.pipeline.unet.config.sample_size,
            self.pipeline.unet.config.sample_size,
        )
        self.ndims = np.prod(self.shape)
        self.raw_x_shape = (
            self.x_shape[0],
            self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor,
            self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor,
        )
        self.raw_x_ndims = np.prod(self.raw_x_shape)
        timesteps = retrieve_timesteps(self.pipeline.scheduler, self.num_steps - 1, 'cpu')[0].numpy()[::-1]
        self.alphas = self.pipeline.scheduler.alphas_cumprod.numpy()[timesteps]
        self.model_t = timesteps
        self.measurement = self.measurement.type(self.dtype)
        
        if self.loss_type == 'latent' or self.loss_type == 'both':
            ATy = self.operator.transpose(self.measurement, mask=self.mask)
            self.z_measurement = self.encode(ATy).reshape(-1, *self.shape)
            
    def alpha(self, t):
        t = self.discretize_time(t)
        return self.alphas[t]

    def get_model_steps(self):
        return self.num_steps - 1
    
    def decode(self, x, differentiable=False):
        x = x.reshape(-1, *self.shape)
        x = self.pipeline.decode(x)
        x = torch.nn.functional.interpolate(x, self.x_shape[-1])
        return x.reshape(-1, self.x_ndims)

    def encode(self, x):
        x = x.reshape(-1, *self.x_shape)
        x = torch.nn.functional.interpolate(x, self.raw_x_shape[-1])
        x = self.pipeline.encode(x)
        return x.reshape(-1, self.ndims)
    
    def denoise(self, t, x):
        eps = self.eps(t, x)
        return self.eps_to_x0(t, eps, x)
    
    def eps(self, t, x):
        input_t = torch.tensor(self.model_t[self.discretize_time(t)], device=x.device)
        x = x.reshape(-1, *self.shape)
        latent_model_input = self.pipeline.scheduler.scale_model_input(x, input_t)

        # predict the noise residual
        model_output = self.pipeline.eps(input_t, latent_model_input)
        
        if self.pipeline.scheduler.config.prediction_type == 'epsilon':
            eps = model_output
        elif self.pipeline.scheduler.config.prediction_type == 'sample':
            eps = self.x0_to_eps(t, x0, x)
        elif self.pipeline.scheduler.config.prediction_type == 'v_prediction':
            eps = sqrt(self.alpha(t)) * model_output + sqrt(1 - self.alpha(t)) * x
        else:
            raise NotImplementedError()
                
        eps = eps.reshape(-1, self.ndims)
        return eps
    
    def logp_y_x(self, x, differentiable=False):
        logp = 0.
        if self.loss_type == 'latent' or self.loss_type == 'both':
            logp = logp + -0.5 * ((self.z_measurement - x.reshape(-1, *self.shape)) ** 2).sum()

        if self.loss_type == 'pixel' or self.loss_type == 'both':
            x = self.decode(x, differentiable=differentiable)
            x = x.reshape(-1, *self.x_shape)
            x = self.operator.forward(x, mask=self.mask)
            logp = logp + -0.5 * ((self.measurement - x) ** 2).sum()

        return logp

class DPSDiffusion(DDPMDiffusion):
    """
    Diffusion class compatible with DPS DDPM models.
    """
    def __init__(
        self,
        sampler,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sampler = sampler

    def get_model_steps(self):
        return self.num_steps

    def alpha(self, t):
        # computes alpha_t in https://arxiv.org/pdf/2010.02502.pdf
        t_int = self.discretize_time(t)
        return self.sampler.alphas_cumprod[t_int]

    def denoise(self, t, xt):
        """
        Computes x0 from xt and modeled score function using Tweedie's formula (in backend)
        (x0 in eq. 12 in https://arxiv.org/pdf/2010.02502.pdf)

        Inputs:
            t:        continuous time variable t \in [0, 1]
            xt:       current diffusion iterate xt
            eps:      epsilon noise function
        Outputs:
            x0:       predicted x0
        """
        t_int = self.discretize_time(t)

        xt = xt.reshape(-1, *self.shape)
        out = self.sampler.p_mean_variance(self.sampler.model, xt, t_int)

        x0_hat = out['pred_xstart'].reshape(-1, np.prod(self.shape))

        return x0_hat

    def eps(self, t, xt):
        x0 = self.denoise(t, xt)
        return self.x0_to_eps(t, x0, xt)

    def logsump_y_x(self, x_all):
        n=x_all.shape[1]
        x_all = x_all.reshape(-1, n, *self.shape)
        res = 0.
        for i in range(n):
            res += self.measurement - self.operator.forward(x_all[:,i], mask=self.mask)
        res /= float(n)
        logp = -0.5 * (res ** 2).sum()
        return logp

    