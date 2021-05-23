

import torch
from .base import Diffusion

def grad_gauss(x, m, var):
    xout = (x - m) / var
    return -xout

def ornstein_ulhenbeck(x, gradx, gamma):
    xout = x + gamma * gradx + torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
    return xout

class OUSampler(Diffusion):

    def __init__(self, num_steps, shape, gammas, 
                    time_sampler, device = None, 
                    mean_final=torch.tensor([0.,0.]), 
                    var_final=torch.tensor([.5, .5]), 
                    num_classes=0,
                    score_matching=False):

        super().__init__()
        self.num_classes = num_classes
        self.score_matching = score_matching
        self.time_sampler = time_sampler
        self.mean_final = mean_final
        self.var_final = var_final
        
        self.num_steps = num_steps # num diffusion steps
        self.shape = shape # shape of object to diffuse
        self.gammas = gammas.float() # schedule          
           

        if device is not None:
            self.device = device
        else:
            self.device = gammas.device

        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas,0).to(self.device).float()
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

        mean_final = self.mean_final
        var_final = self.var_final        
        
        x = init_samples
        N = x.shape[0]
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))
        time = self.time.reshape((1,self.num_steps,1)).repeat((N,1,1))
        gammas = self.gammas.reshape((1,self.num_steps,1)).repeat((N,1,1))
        if self.num_classes > 0:
            labels_expanded = labels.reshape((N,1)).repeat((1,t_batch))
        else:
            labels_expanded = None


        x_tot = torch.Tensor(N, t_batch, *self.shape).to(x.device)
        out = torch.Tensor(N, t_batch, *self.shape).to(x.device)
        num_iter = self.num_steps
        steps_expanded = steps[:, levels,:]
        
        j = 0
        for k in range(num_iter):
            gamma = self.gammas[k]
            gradx = grad_gauss(x, mean_final, var_final)
            t_old = x + gamma * gradx
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma)*z
            gradx = grad_gauss(x, mean_final, var_final)
            t_new = x + gamma * gradx
            
            store_idx = torch.eq(levels, k)
            if torch.any(store_idx):
                x_tot[:, j, :] = x
                out[:, j, :] = (t_old - t_new) #/ (2 * gamma)
                j+=1
            
        return x_tot, out, steps_expanded, labels_expanded