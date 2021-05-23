

import torch
from .base import Diffusion

def grad_gauss(x, m, var):
    xout = (x - m) / var
    return -xout

def ornstein_ulhenbeck(x, gradx, gamma):
    xout = x + gamma * gradx + torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
    return xout

class NetSampler(Diffusion):

    def __init__(self, num_steps, shape, gammas, 
                    time_sampler, device = None, 
                    num_classes=0,
                    score_matching=False):

        super().__init__()
        self.num_classes = num_classes
        self.score_matching = score_matching
        self.time_sampler = time_sampler
        
        self.num_steps = num_steps # num diffusion steps
        self.d = shape # shape of object to diffuse
        self.gammas = gammas.float() # schedule          
           

        if device is not None:
            self.device = device
        else:
            self.device = gammas.device

        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time_sampler = time_sampler
    
    def compute_loss_terms(self, init_samples, labels, t_batch=None, net=None):
        with torch.no_grad():
            return forward(init_samples, labels, t_batch, net)

    def sample(self, init_samples, labels, t_batch=None, net=None):
        with torch.no_grad():
            x_tot, _, _, _ = self.forward(init_samples, labels, t_batch, net)
        return x_tot

    def forward(self, init_samples, labels, t_batch=None, net=None):

        if t_batch is None:
            t_batch = self.num_steps
        if self.time_sampler is not None:
            levels, _ = torch.sort(self.time_sampler.sample(t_batch))
        else: 
            levels = self.steps  
    
        x = init_samples
        N = x.shape[0]
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))
        gammas = self.gammas.reshape((1,self.num_steps)).repeat((N,1))
        if self.num_classes > 0:
            labels_expanded = labels.reshape((N,1)).repeat((1,t_batch))
        else:
            labels_expanded = None
        
        x_tot = torch.Tensor(N, t_batch, *self.d).to(x.device)
        out = torch.Tensor(N, t_batch, *self.d).to(x.device)
        store_steps = self.steps
        steps_expanded = steps[:, levels,:]
        num_iter = self.num_steps
        
        j = 0
        for k in range(num_iter):
            gamma = self.gammas[k]
            if self.num_classes > 0:
                t_old = x + net(x, steps[:, k, :], labels)#* 2 * gamma 
                z = torch.randn(x.shape, device=x.device)
                x = t_old + torch.sqrt(2 * gamma) * z
                t_new = x + net(x, steps[:, k, :], labels)#* 2 * gamma 
            else:
                t_old = x + net(x, steps[:, k, :])#* 2 * gamma 
                z = torch.randn(x.shape, device=x.device)
                x = t_old + torch.sqrt(2 * gamma) * z
                t_new = x + net(x, steps[:, k, :])#* 2 * gamma 
            
            store_idx = torch.eq(levels, k)
            if torch.any(store_idx):
                x_tot[:, j, :] = x
                out[:, j, :] = (t_old - t_new)# / (2 * gamma)
                j+=1
            

        return x_tot, out, steps_expanded, labels_expanded