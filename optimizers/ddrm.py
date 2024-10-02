import torch
import numpy as np
from tqdm import tqdm
from utils.sde_lib import VPSDE
from utils.svd_replacements import *

def nz_div(a, b, zt):
    """
    Helper function for vectorized non-zero division- if denominator is 0, instead make that position 0 for output vector
    """
    assert a.device == b.device, f'numerator device {a.device} != denominator device {b.device}'
    adivb = torch.zeros_like(a, device=a.device)
    adivb[b>zt] = a[b>zt]/b[b>zt]
    return adivb

class DDRM:
    """
    Basic DDRM model as detailed with hyperparameters from (https://arxiv.org/abs/2201.11793).
    See readable algorithm implementation in appendix of DDNM paper (https://arxiv.org/pdf/2212.00490.pdf)
    """
    def __init__(self, env, eta = .85, noise_std_est = 1e-2, zero_thresh = 1e-8):
        self.env = env
        self.svd = get_most_relevant_svd_op(self.env.operator)(self.env)

        #Save svs locally for easier access in a array of "full-rank" size (i.e. all null-space dimensions get 0s)
        self.svs = torch.zeros(np.prod(self.env.shape), device=self.env.device)
        svs_nz = self.svd.singulars()
        self.svs[:len(svs_nz)] = svs_nz
        
        self.eta = eta
        self.sigma_y = noise_std_est
        self.zero_thresh = zero_thresh
        
    
    def solve(self, xT):
        with torch.no_grad():
            betas = torch.FloatTensor(self.env.sampler.betas).to(self.env.device)

            xt = xT
            x_bar_t = self.svd.Vt(xt)
            y_bar = torch.zeros_like(xt, device=self.env.device)
            y_bar[:,:np.prod(self.env.measurement.shape)] = self.svd.S_pinv(self.svd.Ut(self.env.measurement.reshape(-1,np.prod(self.env.measurement.shape))))

            
            #sigma_tm1 depends on alpha_tm2, so truncate 2 here.
            pbar = tqdm(enumerate(zip(self.env.timesteps[:-1], #t
                                      self.env.timesteps[1:],  #t-1
                                      torch.cat([self.env.timesteps[2:],-torch.ones(1, device=self.env.device)]) #t-2 with indicator -1 at end
                                     )), total=self.env.num_steps - 1)
            
            for i, (t, tm1, tm2) in pbar:
                #Compute betas, alphas, sigmas for easier access.
                sigma_t = self.env.sigma(t)
                alpha_t = self.env.alpha(t)
                alpha_tm1 = self.env.alpha(tm1)
                
                if i == len(pbar)-1:
                    alpha_tm2 = 0.
                    sigma_tm1 = 0.
                else:
                    alpha_tm2 = self.env.alpha(tm2)
                    sigma_tm1 = self.env.sigma(tm1)

                
                #initialize tmp variables
                x_bar_tm1 = torch.zeros_like(xt, device=self.env.device)
                    
                
                #Algorithm 7 Line 5
                eps_theta = self.env.eps(t, xt)
                x_0lt = np.sqrt(1./alpha_t) * (xt - np.sqrt(1.-alpha_t) * eps_theta) #tweedie
                #x_0lt = self.env.step(t,xt,eps_theta)
                x_bar_0lt = self.svd.Vt(x_0lt)
                

                #Create noise sample (final sigma_tm1 is 0, so no need to set to 0 on last iter)
                eps = torch.randn_like(xt)
                
                #Algorithm 7 Line 7-8
                crit1 = (self.svs<self.zero_thresh)
                
                sigma_ratio = sigma_tm1/sigma_t.clip(self.zero_thresh)
                x_diff = (x_bar_t[:,crit1] - x_bar_0lt[:,crit1])
                x_bar_tm1[:,crit1] = x_bar_0lt[:,crit1] + np.sqrt(1.-self.eta**2) * sigma_ratio * x_diff
                x_bar_tm1[:,crit1] += self.eta*sigma_tm1*eps[:,crit1]



                #Algorithm 7 Line 9-10
                crit2 = (sigma_tm1  < self.sigma_y/self.svs.clip(self.zero_thresh)) * (~crit1)

                sigma_sv_ratio = self.svs[crit2]*sigma_tm1/self.sigma_y
                yx_diff = y_bar[:,crit2] - x_bar_0lt[:,crit2]
                x_bar_tm1[:,crit2] = x_bar_0lt[:,crit2] + np.sqrt(1.-self.eta**2) * sigma_sv_ratio * yx_diff
                x_bar_tm1[:,crit2] += self.eta*sigma_tm1*eps[:,crit2]
        
                #Algorithm 7 Line 11-12
                crit3 = (~crit1) * (~crit2) #all other svs
                x_bar_tm1[:,crit3] = y_bar[:,crit3] + torch.sqrt(sigma_tm1**2 - (self.sigma_y**2)/(self.svs[crit3]**2)) * eps[:,crit3]

                
                #Algorithm 7 Line 13
                xt = self.svd.V(x_bar_tm1)
                x_bar_t = torch.clone(x_bar_tm1)
                

        return xt