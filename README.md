# Injecting Measurement Information Yields a Fast and Noise-Robust Diffusion-Based Inverse Problem Solver

**Jonathan Patsenker\*, Henry Li\*, Myeongseob Ko, Ruoxi Jia, Yuval Kluger**
*Yale University · Virginia Tech*

*Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS) 2026, Tangier, Morocco. PMLR: Volume 300.*

\* Equal contribution

<!-- [![Paper](https://img.shields.io/badge/paper-AISTATS%202026-blue)](https://proceedings.mlr.press/v300/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![website: up](https://img.shields.io/website?url=https://diffusion-conditional-sampling.github.io
)](https://diffusion-conditional-sampling.github.io)

---

## Abstract

Diffusion models have been firmly established as principled zero-shot solvers for linear and nonlinear inverse problems, owing to their powerful image prior and iterative sampling algorithm. These approaches often rely on Tweedie's formula, which relates the diffusion variate **x**<sub>t</sub> to the posterior mean **E**[**x**<sub>0</sub>|**x**<sub>t</sub>], in order to guide the diffusion trajectory with an estimate of the final denoised sample **x**<sub>0</sub>. However, this does not consider information from the measurement **y**, which must then be integrated downstream. In this work, we propose to estimate the conditional posterior mean **E**[**x**<sub>0</sub>|**x**<sub>t</sub>, **y**], which can be formulated as the solution to a lightweight, single-parameter maximum likelihood estimation problem. The resulting prediction can be integrated into any standard sampler, resulting in a fast and memory-efficient inverse solver. Our optimizer is amenable to a noise-aware likelihood-based stopping criterion that is robust to measurement noise in **y**. We demonstrate comparable or improved performance against a wide selection of contemporary inverse solvers across multiple datasets and tasks.

---

## Method Overview

**DCS** corrects the standard (unconditional) Tweedie estimate of **x**<sub>0</sub> by incorporating measurement information **y** at each diffusion step. Rather than using **E**[**x**<sub>0</sub>|**x**<sub>t</sub>], which averages over all data consistent with **x**<sub>t</sub>, we optimize a single scalar correction ε_<sub>**y**</sub> to approximate **E**[**x**<sub>0</sub>|**x**<sub>t</sub>, **y**] via maximum likelihood. This corrected score is then fed into any off-the-shelf DDPM sampler.

| | DCS (Ours) | MCG | DPS | DPS-JF | DDNM | RED-Diff | LGD-MC |
|---|---|---|---|---|---|---|---|
| **No NFE Backprop** | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| **Runtime** | **1×** | 2.6× | 2.5× | 1.2× | 1.5× | 1.5× | 2× |
| **Memory** | **1×** | 3.2× | 3.2× | 1.1× | 1× | 1× | 3.2× |

DCS achieves the fastest runtime and lowest memory footprint while matching or exceeding the reconstruction quality of all compared methods.
- **Measurement-consistent Tweedies**: Rather than predicting **E**[**x**<sub>0</sub>|**x**<sub>t</sub>] via the unconditional score, we predict **E**[**x**<sub>0</sub>|**x**<sub>t</sub>, **y**] by solving a single-parameter MLE problem estimating the forward process score at each diffusion step.
- **Noise-aware maximization (NAM)**: A statistically-grounded early stopping criterion based on a two-sided z-test prevents overfitting to measurement noise across all noise levels.
- **No score network backpropagation**: DCS never differentiates through the neural score network, cutting runtime by 2–3× compared to posterior-based methods.

---

## Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-repo>/dcs.git
cd dcs
pip install -r requirements.txt
```

### 2. Latent diffusion support

To run latent diffusion experiments, install the Latent Diffusion and Taming Transformers libraries:

```bash
git clone https://github.com/CompVis/latent-diffusion.git
cd latent-diffusion
git reset --hard a506df5   # optional: pin to a known-good commit
pip install -e .
cd ..

git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers
git reset --hard 3ba01b2   # optional: pin to a known-good commit
pip install -e .
cd ..
```

> **Note:** If you encounter an import error in the Latent Diffusion library, change line 19 of
> `latent-diffusion/ldm/models/diffusion/ddpm.py` from:
> ```python
> from pytorch_lightning.utilities.distributed import rank_zero_only
> ```
> to:
> ```python
> from pytorch_lightning.utilities.rank_zero import rank_zero_only
> ```

---

## Downloading Models and Data

### Pretrained models

Create a `models/` directory and download the following checkpoints:

| Model | Dataset | Source |
|---|---|---|
| `ffhq_10m.pt` | FFHQ 256×256 | [DPS repo](https://github.com/DPS2022/diffusion-posterior-sampling) |
| `imagenet256.pt` | ImageNet 256×256 | [Guided Diffusion](https://github.com/openai/guided-diffusion) |
| `ffhq_ldm.ckpt` | FFHQ LDM | [Latent Diffusion](https://github.com/CompVis/latent-diffusion) |

The first two models should be accessible through this [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh)

### Datasets

Download the following datasets and place them under a shared data root (default: `/data/inverse/`):

| Dataset | Expected path | Download Link |
|---|---|---|
| ImageNet 256×256 | `<data_root>/imagenet256/` | [Download] |
| FFHQ subset (100 images) | `<data_root>/ffhq_subset/` | [Download](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only) |

The code uses images numbered ≥ 69000 from FFHQ as the evaluation split (1K images).

---

## Running Experiments

All experiments are launched via `inverse_sampling.py`. The key arguments are:

| Argument | Description | Default |
|---|---|---|
| `--algorithm` | Solver to run (`dcs`, `dps`, `ddnm`, `mcg`, `lgdmc`, …) | `rdps` |
| `--task` | Inverse problem (`super_resolution`, `gaussian_deblur`, `motion_deblur`, `random_inpainting`, `box_inpainting`) | `super_resolution` |
| `--dataset` | Dataset (`ffhq`, `imagenet`, `ffhq_subset`) | `ffhq` |
| `--sigma` | Measurement noise level σ_**y** | `0.01` |
| `--diffusion_steps` | Number of diffusion steps T | `100` |
| `--scale` | Learning rate for DCS (or guidance scale for other methods) | `1e-2` |
| `--iters` | NAM inner optimization steps | `1000` |
| `--outdir` | Output directory for images and metrics | `./out` |
| `--data_root` | Root directory for datasets | `/data/inverse/` |
| `--model_root` | Directory containing pretrained model weights | `/data/inverse/models` |

### Example commands

```bash
# DCS — Super-resolution on FFHQ, low noise
python inverse_sampling.py \
    --algorithm dcs \
    --task super_resolution \
    --dataset ffhq \
    --sigma 0.01 \
    --diffusion_steps 50 \
    --scale 1.0 \
    --iters 50 \
    --outdir ./out/dcs_sr_ffhq

# DCS — Motion deblurring on ImageNet, high noise
python inverse_sampling.py \
    --algorithm dcs \
    --task motion_deblur \
    --dataset imagenet \
    --sigma 0.1 \
    --diffusion_steps 50 \
    --scale 1.0 \
    --outdir ./out/dcs_mdeblur_imagenet

# DPS baseline — Gaussian deblurring on FFHQ
python inverse_sampling.py \
    --algorithm dps \
    --task gaussian_deblur \
    --dataset ffhq \
    --sigma 0.01 \
    --diffusion_steps 100 \
    --scale 1.0 \
    --outdir ./out/dps_gdeblur_ffhq

# Latent DCS — Super-resolution on ImageNet
python inverse_sampling.py \
    --algorithm dcs \
    --task super_resolution \
    --dataset imagenet \
    --sigma 0.01 \
    --latent \
    --outdir ./out/dcs_sr_imagenet_latent
```

### Reproducibility

Results are sensitive to the random seed. Use `--seed <int>` to fix it. By default a random seed is chosen and printed at the start of each run.

---

## Code Structure

```
dcs/
├── inverse_sampling.py       # Main entry point for all experiments
├── simple_utils.py           # Config loading, metrics, dataset/model setup
├── diffusion/
│   ├── diffusion.py          # Diffusion process classes (DDPM, DPS, Latent, etc.)
│   └── latent_pipeline.py    # HuggingFace Stable Diffusion pipeline wrapper
├── optimizers/
│   ├── dcs.py                # DCS solver + EarlyStop / NAM logic
│   ├── dps.py                # DPS baseline
│   ├── dpsjf.py              # Jacobian-free DPS
│   ├── ddnm.py               # DDNM baseline
│   ├── mcg.py                # MCG baseline
│   ├── lgdmc.py              # LGD-MC baseline
│   ├── lgdmcjf.py            # Jacobian-free LGD-MC
│   ├── red.py                # RED-Diff baseline
│   └── resample.py           # ReSample baseline
├── utils/
│   ├── measurements.py       # Forward operators (SR, blur, inpainting, …)
│   ├── gaussian_diffusion.py # Gaussian diffusion core (DDIM/DDPM samplers)
│   ├── unet.py               # OpenAI UNet architecture
│   ├── img_utils.py          # Image utilities and mask generators
│   └── dataloader.py         # Dataset classes (FFHQ, ImageNet, CIFAR)
└── inverse_configs/          # YAML configs for each task and model
```

---

If you find this work useful, please cite:

```bibtex
@misc{patsenker2025injecting,
      title={Injecting Measurement Information Yields a Fast and Noise-Robust Diffusion-Based Inverse Problem Solver}, 
      author={Jonathan Patsenker and Henry Li and Myeongseob Ko and Ruoxi Jia and Yuval Kluger},
      year={2025},
      eprint={2508.02964},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!-- ## Citation for when publisher actually does its thing

```bibtex
@inproceedings{patsenker2026dcs,
  title     = {Injecting Measurement Information Yields a Fast and Noise-Robust
               Diffusion-Based Inverse Problem Solver},
  author    = {Patsenker, Jonathan and Li, Henry and Ko, Myeongseob and
               Jia, Ruoxi and Kluger, Yuval},
  booktitle = {Proceedings of the 29th International Conference on
               Artificial Intelligence and Statistics},
  series    = {Proceedings of Machine Learning Research},
  volume    = {300},
  year      = {2026},
  publisher = {PMLR}
}
``` -->
