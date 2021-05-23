import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np 

def grad_gauss(x, m, var):
    xout = (x - m) / var
    return -xout

def ornstein_ulhenbeck(x, gradx, gamma):
    xout = x + gamma * gradx + torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
    return xout

class Langevin(torch.nn.Module):

    def __init__(self, num_steps, shape, gammas, time_sampler, device = None, 
                 mean_final=torch.tensor([0.,0.]), var_final=torch.tensor([.5, .5]), fast_sampling=True, corrector=True, score_matching=False):
        super().__init__()
        self.score_matching = score_matching
        self.mean_final = mean_final
        self.var_final = var_final
        
        self.num_steps = num_steps # num diffusion steps
        self.d = shape # shape of object to diffuse
        self.gammas = gammas.float() # schedule
        gammas_vec = torch.ones(self.num_steps,*self.d,device=device)
        for k in range(num_steps):
            gammas_vec[k] = gammas[k].float()
        self.gammas_vec = gammas_vec    

        if device is not None:
            self.device = device
        else:
            self.device = gammas.device

        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas,0).to(self.device).float()
        self.time_sampler = time_sampler

        self.corrector = corrector

        # compute sampling quantities Ornstein-Ulhenbeck (if fast_sampling=True)
        self.fast_sampling = fast_sampling
        a = torch.zeros(self.num_steps,*self.d,device=device)
        b = torch.zeros(self.num_steps,*self.d,device=device)
            

    def record_init_langevin(self, init_samples):
        mean_final = self.mean_final
        var_final = self.var_final        
        
        x = init_samples
        N = x.shape[0]
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))
        time = self.time.reshape((1,self.num_steps,1)).repeat((N,1,1))
        gammas = self.gammas.reshape((1,self.num_steps,1)).repeat((N,1,1))


        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        store_steps = self.steps
        num_iter = self.num_steps
        steps_expanded = steps
        
        for k in range(num_iter):
            gamma = self.gammas[k]
            gradx = grad_gauss(x, mean_final, var_final)
            t_old = x + gamma * gradx
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma)*z
            gradx = grad_gauss(x, mean_final, var_final)
            t_new = x + gamma * gradx
            
            x_tot[:, k, :] = x
            if self.score_matching:
                out[:, k, :] = -z
            else:
                out[:, k, :] = (t_old - t_new) #/ (2 * gamma)
            
        return x_tot, out, steps_expanded

    def record_langevin_seq(self, net, init_samples, t_batch=None, ipf_it=0):
        mean_final = self.mean_final
        var_final = self.var_final        
    
        x = init_samples
        N = x.shape[0]
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))
        time = self.time.reshape((1,self.num_steps,1)).repeat((N,1,1))
        gammas = self.gammas.reshape((1,self.num_steps,1)).repeat((N,1,1))

        
        x_tot = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d).to(x.device)
        store_steps = self.steps
        steps_expanded = steps
        num_iter = self.num_steps
        
        for k in range(num_iter):
            gamma = self.gammas[k]                    
            t_old = x + net(x, steps[:, k, :])#* 2 * gamma 
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma) * z
            t_new = x + net(x, steps[:, k, :])#* 2 * gamma 
            
            if self.corrector: 
               if ipf_it == 1:
                    bx_corr = net(x, steps[:, k, :])#*(2 * gamma)
                    bx_corr += gamma * grad_gauss(x, mean_final, var_final)
                    t_corr = x + bx_corr
                    x = t_corr + torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
            
            x_tot[:, k, :] = x
            if self.score_matching:
                out[:, k, :] = -z
            else:
                out[:, k, :] = (t_old - t_new) #/ (2 * gamma)
            

        return x_tot, out, steps_expanded


    def forward(self, net, init_samples, t_batch, ipf_it):
        return self.record_langevin_seq(net, init_samples, t_batch, ipf_it)
    



