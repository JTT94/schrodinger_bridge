import torch
from .base import Diffusion
from .ou_sampler import grad_gauss, ornstein_ulhenbeck

class FastSampler(Diffusion):

    def __init__(self, num_steps, shape, gammas, num_classes, time_sampler, mean_final, var_final, device=None):
        super().__init__()

        if device is not None:
            self.device = device
        else:
            self.device = gammas.device


        self.num_classes = num_classes
        self.time_sampler = time_sampler
        self.mean_final = mean_final.to(self.device)
        self.var_final = var_final.to(self.device)
        
        self.num_steps = num_steps # num diffusion steps
        self.d = shape # shape of object to diffuse
        self.gammas = gammas.float() # schedule          
           

        

        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas,0).to(self.device).float()

        gammas_expanded = torch.ones(self.num_steps,*self.d,device=device)
        for k in range(num_steps):
            gammas_expanded[k] = gammas[k].float()  
        self.gammas_expanded = gammas_expanded

        # compute sampling quantities Ornstein-Ulhenbeck (if fast_sampling=True)
        a = torch.zeros(self.num_steps,*self.d,device=device)
        b = torch.zeros(self.num_steps,*self.d,device=device)

        ak = 1
        bk = 0
        for k in range(num_steps):
            gam = gammas[k]
            mul = 1 - gam / var_final.mean()
            ak = mul * ak
            bk = torch.sqrt(2 * gam + (bk * mul) ** 2)
            a[k] = ak
            b[k] = bk

        self.a = a
        self.b = b


    def compute_loss_terms(self, init_samples, labels, t_batch=None, net=None):
        return self.forward(init_samples, labels, t_batch, net)

    def sample(self, init_samples, labels, t_batch=None, net=None):
        x_tot, _, _, _ = self.forward(init_samples, labels, t_batch, net)
        return x_tot

    def forward(self, init_samples, labels, t_batch=None, net=None):
        x = init_samples
        N = x.shape[0]

        levels = self.time_sampler.sample(N, replacement=True)
        

        mean_final = self.mean_final
        var_final = self.var_final        
        
        
        steps = self.steps.reshape((self.num_steps,1))
        if self.num_classes > 0:
            labels_expanded = labels.reshape((N,1))
        else:
            labels_expanded = None


        x_tot = torch.Tensor(N, *self.d).to(self.device)  
        out = torch.Tensor(N, *self.d).to(self.device)
        steps_expanded = steps[levels]
        
        x0 = init_samples

        gamma_batch = self.gammas_expanded[levels]
        z = torch.randn(x.shape, device=self.device)

        e1 = self.a[levels]
        e2 = self.b[levels]
        x = e1 * x0 + e2 * z

        gradx = grad_gauss(x, mean_final, var_final)        
        t_old = x + gamma_batch * gradx
        x = ornstein_ulhenbeck(x, gradx, gamma_batch)
        gradx = grad_gauss(x, mean_final, var_final)
        t_new = x + gamma_batch * gradx

        x_tot = x
        out = (t_old - t_new) #/ (2 * gamma_batch)

        return x_tot, out, steps_expanded, labels_expanded
