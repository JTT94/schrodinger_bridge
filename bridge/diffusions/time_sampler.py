import torch

class TimeSampler:

    def __init__(self, weights):
        self.weights=weights
    
    def sample(self, n, replacement=False):
        return torch.multinomial(self.weights, n, replacement=replacement)
