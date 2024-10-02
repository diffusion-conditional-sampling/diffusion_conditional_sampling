# Diffusion Conditional Sampling (DCS)

Step 1:
Install requirements.
```
pip install -r requirements
```

Step 2:
Download models and data.
```
TODO
```

To run latent diffusion code, please also install the ldm and taming-transformers repos. 
```
git clone https://github.com/CompVis/latent-diffusion.git
cd latent-diffusion
git reset --hard a506df5 # optional: in case new commits are not backward compatible
pip install -e .
```

```
git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers
git reset --hard 3ba01b2 # optional: in case new commits are not backward compatible
pip install -e .
```

You may need to change L19 in `latent-diffusion/ldm/models/diffusion/ddpm.py` from
```
from pytorch_lightning.utilities.distributed import rank_zero_only
```
to
```
from pytorch_lightning.utilities.rank_zero import rank_zero_only
```
