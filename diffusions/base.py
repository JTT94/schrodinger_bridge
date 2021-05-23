import torch

class Diffusion(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

    def sample(self, init_samples, labels, t_batch=None, net=None):
        pass

    def compute_loss_terms(self, init_samples, labels, t_batch=None, net=None):
        pass