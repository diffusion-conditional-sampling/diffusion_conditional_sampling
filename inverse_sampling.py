import os
import gc
import argparse
import pickle
import yaml
from PIL import Image
import matplotlib.pyplot as plt

import torch
import numpy as np

from simple_utils import *

parser = argparse.ArgumentParser(description='Compute DDP on MNIST.')
parser.add_argument('-i','--iters', help='Number of optimizer iterations', default=1000, type=int)
parser.add_argument('-s','--diffusion_steps', help='Number of diffusion iterations', default=100, type=int)
parser.add_argument('-b','--batch_size', help='Model batch size', default=1, type=int)
parser.add_argument('-a','--algorithm', help='Algorithm to use: bayes', default='dcs', type=str)
parser.add_argument('-t','--task', help='Task to benchmark: box_inpainting / random_inpainting / motion_deblur / gaussian_deblur / super_resolution / phase_retrieval', default='super_resolution', type=str)
parser.add_argument('--outdir', help='Output directory', default='./out', type=str)
parser.add_argument('--dataset', help='Dataset: ffhq / imagenet', default='ffhq', type=str)
parser.add_argument('--seed', help='Seed', type=int, default=-1)
parser.add_argument('--scale', help='Guidance scale on algorithm (for DCS, it is the learning rate)', type=float, default=1e-2)
parser.add_argument('--sigma', help='Measurement noise (e.g., variance of eta)', type=float, default=0.01)
parser.add_argument('--nmc', help='LGDMC number of monte-carlo iterations', type=int, default=10)
parser.add_argument('--data_root', help='Data location', default='/data/inverse/', type=str)
parser.add_argument('--model_root', help='Model location', default='/data/inverse/models', type=str)
parser.add_argument('--latent', help='Latent models', action='store_true')
parser.add_argument('--device', help='Device cpu/cuda/cuda:x', default='cuda', type=str)
parser.add_argument('--hush', help='Print less metrics?', action='store_true')
parser.add_argument('--noise_distribution', help='Measurement noise (e.g., variance of eta)', type=str, default='gaussian')

args = parser.parse_args()

if args.seed == -1:
    args.seed = np.random.randint(1000)

if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)

        
roots = []
for dir_name in ['ref/', 'jpg/', 'obs/']:
    root = os.path.join(args.outdir, dir_name)
    roots.append(root)
    if not os.path.isdir(root):
        os.mkdir(root)


device = args.device

config = get_config(args, device)
metrics = Metrics(config.shape, args.outdir, device)
    
print_every = max(1, len(config.loader) // 10)
save_idx = 0
x_metrics = []
y_metrics = []
for i, ref_img in enumerate(config.loader):

    ref_img = ref_img.to(device)

    if 'inpainting' in args.task:
        mask = config.mask_gen(ref_img)
        mask = mask[:, 0:1, :, :]
    else:
        mask = None

    # Forward measurement model (Ax + n)
    y = config.operator.forward(ref_img, mask=mask)
    y_n = config.noiser(y).detach()
    
    # initialize classes
    env = config.env_cls(
        measurement=y_n,
        operator=config.operator,
        mask=mask
    )
    solver = config.alg_cls(env)
    
    # solve inverse problem
    xT = env.sample_xT(n=len(y_n))
    x0 = solver.solve(xT)
    
    if args.latent:
        with torch.no_grad():
            x0 = env.decode(x0)
            
    sample = x0.reshape(-1, *config.shape)

    # print metrics
    ATAsample = config.operator.transpose(config.operator.forward(sample, mask=mask), mask=mask)
    ATy = config.operator.transpose(y, mask=mask)
    y_metrics.append(metrics.get(ATAsample, ATy, shape=config.shape))
    x_metrics.append(metrics.get(ref_img, sample))

    if not args.hush:
        metrics.print(x_metrics)
        metrics.print(y_metrics, extra_str="metrics in y-space")

    # save images
    for root, im_list in zip(roots, [ref_img, sample, y_n]):
        for j, x in enumerate(im_list):
            x = process(x)
            im = Image.fromarray(x)
            im.save(os.path.join(root, f"{save_idx + j:04d}.png"))
    save_idx += len(ref_img)
    
    # save
    if (i + 1) % print_every == 0:
        print('', '*' * 41, '\n' * 2, f"Finished {i + 1} / {len(config.loader)} batches.", '\n' * 2, '*' * 41)
        with open(os.path.join(args.outdir, 'results.pkl'), 'wb') as f:
            obj = {
                'x_metrics': np.stack(x_metrics),
                'y_metrics': np.stack(y_metrics)
            }
            pickle.dump(obj, f)

        if args.hush:
            metrics.print(x_metrics)
            metrics.print(y_metrics, extra_str="metrics in y-space")

        gc.collect()
