import torch
import yaml
import lpips
import os

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from torchvision import transforms, datasets
from torchmetrics.image import StructuralSimilarityIndexMeasure
from functools import partial

from optimizers.dps import DPS
from optimizers.lgdmc import LGDMC
from optimizers.lgdmcjf import LGDMCJF
from optimizers.dpsjf import DPSJF
from optimizers.ddnm import DDNM
from optimizers.rdps import RDPS
from optimizers.rdps_eps import RDPSEps
from optimizers.mcg import MCG
from optimizers.ddrm import DDRM
from optimizers.psld import PSLD
from optimizers.stsl import STSL
from optimizers.resample import ReSample

from utils.unet import create_model
from utils.ldm import load_model, DDIMSampler
from utils.gaussian_diffusion import create_sampler
from utils.measurements import get_noise, get_operator
from utils.dataloader import get_dataset, get_dataloader
from utils.img_utils import mask_generator
from utils.condition_methods import get_conditioning_method

from diffusion.diffusion import *

from diffusers import DiffusionPipeline
from diffusion.latent_pipeline import SimpleStableDiffusionPipeline

def process(x):
    x = (x.clip(-1, 1).permute(1, 2, 0).float().cpu().numpy() + 1) * .5
    x = (x * 255.).astype(np.uint8)
    if x.shape[-1] == 1:
        x = np.tile(x, (1, 1, 3))
    return x
scale = lambda x: x

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

class AttrDict(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

class Metrics:
    def __init__(self, shape, outdir=None, device='cpu'):
        self.shape = shape
        self.outdir = outdir
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex', verbose=True).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(reduction=None, data_range=(-1, 1))
        self.metric_names = ['LPIPS', 'PSNR', 'SSIM', 'MSE']
        self.metric_fns = [self.lpips_fn, self.psnr_fn, self.ssim_fn, self.mse_fn]

    def get(self, ref_img, states, shape=None):
        shape = self.shape if shape is None else shape
        n_batch = len(ref_img)
        ref_img = ref_img.cpu().reshape(n_batch, *shape).float()
        states = states.reshape(n_batch, *shape).cpu().float()
        metrics = []

        for fn in self.metric_fns:
            metric = fn(scale(ref_img), scale(states)).reshape(n_batch)
            metrics.append(metric)

        metrics = torch.stack(metrics, axis=0)
        return metrics.detach().cpu().numpy()

    def print(self, metrics, extra_str="metrics in x-space"):
        all_time_metrics = np.stack(metrics, axis=1)
        strings = [f"#{'-' * 41}#", f"|{extra_str:^41}|", f"#{'-' * 41}#"]
        strings.append(f" {'metric':<6}: {'cur':>10} | {'all time':^25}")
        for name, metric in zip(self.metric_names, all_time_metrics):
            strings.append(f" {name:<6}: {metric[-1].mean():10.3f} | {metric.mean():>10.3f} +/- {metric.std():<10.3f}")

        print("\n".join(strings))

    def plot(self, metrics_over_times):
        assert self.outdir is not None
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        metrics_over_time = np.stack(metrics_over_times).mean(axis=0)
        for i, (ax, name) in enumerate(zip(axs, ['LPIPS', 'PSNR', 'SSIM', 'NMSE'])):
            metric = metrics_over_time[:, i]
            ax.plot(metric, label=name)
            ax.legend()
        plt.savefig(os.path.join(self.outdir, 'average_metrics_over_time.png'))
        plt.close(fig)
        del fig
        del axs

    def lpips_fn(self, x, y):
        with torch.no_grad():
            return self.lpips_model(x.to(self.device), y.to(self.device)).cpu()

    def ssim_fn(self, x, y):
        return self.ssim.forward(x, y)

    def mse_fn(self, x, y):
        mse = torch.mean((x - y) ** 2, axis=(1, 2, 3)) / torch.mean(x ** 2, axis=(1, 2, 3))
        return mse

    def psnr_fn(self, x, y):
        mse = torch.mean((x - y) ** 2, axis=(1, 2, 3))
        max_pixel = 2.
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse + 1e-7))

        return psnr

def get_config(args, device, transform=None):
    if not args.latent and (args.dataset in ['celeba', 'imagenet'] or 'ffhq' in args.dataset):
        diffusion_cls = DPSDiffusion
    elif args.latent:
        if args.dataset == 'imagenet':
            diffusion_cls = StableDiffusion
        else:
            diffusion_cls = LatentDiffusion
    else:
        diffusion_cls = HFDiffusion
        

    if args.algorithm == 'dps':
        alg_kwargs = {
            # 'gamma': 0.1 if args.latent else 1.0,
            'gamma': args.scale,
            'latent': args.latent,
        }
        alg_cls = DPS
        if args.latent:
            diffusion_cls = partial(diffusion_cls, loss_type='pixel')
    
    elif args.algorithm == 'lgdmc':
        alg_kwargs = {
            # 'gamma': 0.1 if args.latent else 1.0,
            'gamma': args.scale,
            'latent': args.latent,
            'n':args.nmc
        }
        alg_cls = LGDMC
        
    elif args.algorithm == 'dpsjf':
        alg_kwargs = {
            'gamma': args.scale,
            'latent': args.latent,
        }
        alg_cls = DPSJF
        if args.latent:
            diffusion_cls = partial(diffusion_cls, loss_type='pixel')
    elif args.algorithm == 'psld':
        assert args.latent
        alg_kwargs = {}
        alg_cls = PSLD
        if args.latent:
            diffusion_cls = partial(diffusion_cls, loss_type='pixel')
    elif args.algorithm == 'stsl':
        assert args.latent
        alg_kwargs = {}
        alg_cls = STSL
        if args.latent:
            diffusion_cls = partial(diffusion_cls, loss_type='pixel')
    elif args.algorithm == 'resample':
        assert args.latent
        alg_kwargs = {}
        alg_cls = ReSample
        if args.latent:
            diffusion_cls = partial(diffusion_cls, loss_type='pixel')
    elif args.algorithm == 'ddnm':
        alg_kwargs = {}
        alg_cls = DDNM
    elif args.algorithm == 'rdps':
        alg_kwargs = {
            'iters': args.iters,
            'lr': args.scale,
            'sigma': args.sigma
        }
        alg_cls = RDPS
        if args.latent:
            diffusion_cls = partial(diffusion_cls, loss_type='pixel')
    elif args.algorithm == 'rdps_eps':
        alg_kwargs = {
            'iters': args.iters,
            'lr': args.scale
        }
        alg_cls = RDPSEps
        if args.latent:
            diffusion_cls = partial(diffusion_cls, loss_type='latent')
    elif args.algorithm == 'mcg':
        alg_kwargs = {
            'latent': args.latent,
        }
        alg_cls = MCG
    elif args.algorithm == 'ddrm':
        alg_kwargs = {
            'noise_std_est': args.sigma
        }
        alg_cls = DDRM
        
    alg_cls = partial(alg_cls, **alg_kwargs)
    
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(256),
        ])

    if args.task == 'super_resolution':
        task_config = load_yaml('inverse_configs/super_resolution_config.yaml')
    elif args.task == 'box_inpainting':
        task_config = load_yaml('inverse_configs/box_inpainting_config.yaml')
    elif args.task == 'random_inpainting':
        task_config = load_yaml('inverse_configs/inpainting_config.yaml')
    elif args.task == 'motion_deblur':
        task_config = load_yaml('inverse_configs/motion_deblur_config.yaml')
    elif args.task == 'gaussian_deblur':
        task_config = load_yaml('inverse_configs/gaussian_deblur_config.yaml')
    elif args.task == 'phase_retrieval':
        task_config = load_yaml('inverse_configs/phase_retrieval_config.yaml')
    elif args.task == 'denoise':
        task_config = load_yaml('inverse_configs/denoise_config.yaml')

    # Prepare dataloader
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    shuffle = True
    data_config = task_config['data']
    shape = (3, 256, 256)
    if args.dataset == 'ffhq':
        # data_config['root'] = os.path.join(data_root, 'ffhq')
        data_config['root'] = os.path.join(args.data_root, 'ffhq1024')

        if args.latent:
            model_config = load_yaml('inverse_configs/ffhq_ldm_config.yaml')
        else:
            model_config = load_yaml('inverse_configs/model_config.yaml')
        model_config['model_path'] = os.path.join(args.model_root, 'ffhq_10m.pt')

        loader = get_dataloader(
          get_dataset(**data_config, transforms=transform), batch_size=args.batch_size,
            num_workers=0, train=False, shuffle=shuffle)
    elif args.dataset == 'ffhq_subset':
        data_config['root'] = os.path.join(args.data_root, 'ffhq_subset')
        shuffle = False
        if args.latent:
            model_config = load_yaml('inverse_configs/ffhq_ldm_config.yaml')
        else:
            model_config = load_yaml('inverse_configs/model_config.yaml')
        model_config['model_path'] = os.path.join(args.model_root, 'ffhq_10m.pt')

        loader = get_dataloader(
          get_dataset(**data_config, transforms=transform), batch_size=args.batch_size,
            num_workers=0, train=False, shuffle=shuffle)
    elif args.dataset == 'ffhq_subset2':
        data_config['root'] = os.path.join(args.data_root, 'ffhq_subset2')
        shuffle = False
        if args.latent:
            model_config = load_yaml('inverse_configs/ffhq_ldm_config.yaml')
        else:
            model_config = load_yaml('inverse_configs/model_config.yaml')
        model_config['model_path'] = os.path.join(args.model_root, 'ffhq_10m.pt')

        loader = get_dataloader(
          get_dataset(**data_config, transforms=transform), batch_size=args.batch_size,
            num_workers=0, train=False, shuffle=shuffle)
    elif args.dataset == 'celeba':
        data_config['root'] = os.path.join(args.data_root, 'celeba512')

        if args.latent:
            model_config = load_yaml('inverse_configs/ffhq_ldm_config.yaml')
        else:
            model_config = load_yaml('inverse_configs/model_config.yaml')
        model_config['model_path'] = os.path.join(args.model_root, 'ffhq_10m.pt')

        loader = get_dataloader(
          get_dataset(**data_config, transforms=transform), batch_size=args.batch_size,
            num_workers=0, train=False, shuffle=shuffle)
    elif args.dataset == 'imagenet':
        data_config['root'] = os.path.join(args.data_root, 'imagenet256')
        model_config = load_yaml('inverse_configs/imagenet_model_config.yaml')
        model_config['model_path'] = os.path.join(args.model_root, 'imagenet256.pt')

        loader = get_dataloader(
          get_dataset(**data_config, transforms=transform), batch_size=args.batch_size,
            num_workers=0, train=False, shuffle=shuffle)
    elif args.dataset == 'ct':
        shape = (1, 256, 256)
        data_config['root'] = os.path.join(args.data_root, 'ct/IMG/')
        
        # create single-channel image
        transform = transforms.Compose([
            transform,
            lambda x: x.mean(axis=0, keepdims=True)
        ])
                        
        loader = get_dataloader(
          get_dataset(**data_config, transforms=transform), batch_size=args.batch_size,
            num_workers=0, train=False, shuffle=shuffle)
    else:
        raise NotImplementedError()

    if args.latent:
        if args.dataset == 'celeba' or 'ffhq' in args.dataset:
            ckpt = os.path.join(args.model_root, 'ffhq_ldm.ckpt')
            # config = OmegaConf.load(os.path.join(args.model_root, 'ffhq_ldm_config.yaml'))
            model_config = AttrDict(model_config)

            model, global_step = load_model(model_config, ckpt, device=device)
            sampler = DDIMSampler(model)
            dtype = torch.float
        else:
            model_id = "stabilityai/stable-diffusion-2-1"
            # dtype = torch.half
            dtype = torch.float32
            pipeline = SimpleStableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
            sampler = pipeline
    else:
        if args.dataset == 'celeba' or args.dataset == 'imagenet' or 'ffhq' in args.dataset:
            diffusion_config = load_yaml('inverse_configs/diffusion_config.yaml')
            diffusion_config['sampler'] = 'ddpm'
            diffusion_config['timestep_respacing'] = args.diffusion_steps
            diffusion_config['model_var_type'] = 'fixed_small'
            diffusion_config['sampler'] = 'ddim'

            model = create_model(**model_config)
            model = model.to(device)
            model.eval()

            sampler = create_sampler(model=model, **diffusion_config)
            dtype = torch.float
        elif args.dataset == 'ct':
            model_id = '/home/projects/inverse/ddpm-ct-256/'
            dtype = torch.float
            pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
            sampler = pipeline

    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    measure_config['noise']['sigma'] = args.sigma
    noiser = get_noise(**measure_config['noise'])

    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    else:
        mask_gen = None
        
    env_cls = partial(
        diffusion_cls,
        sampler,
        shape=shape,
        num_steps=args.diffusion_steps,
        device=device,
        dtype=dtype
        )

    return_dict = AttrDict({
        'sampler': sampler,
        'mask_gen': mask_gen,
        'noiser': noiser,
        'operator': operator,
        'loader': loader,
        'diffusion_cls': diffusion_cls,
        'alg_cls': alg_cls,
        'env_cls': env_cls,
        'shape': shape,
    })

    return return_dict
