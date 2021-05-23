
import torch

class PriorSampler:

    """Iterator that counts upward forever."""

    def __init__(self, batch_size, shape, mean_final, std_final, num_classes, device):
        self.mean_final = mean_final.to(device)
        self.std_final = std_final.to(device)
        self.shape = shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device=device
        if num_classes > 0 :
            self.weights = torch.ones(num_classes, device=device) 
            
        
    def __iter__(self):
        return self

    def __next__(self):
        batch =  self.mean_final + self.std_final*torch.randn((self.batch_size, *self.shape), device=self.device)
        if self.num_classes > 0:
            labels = torch.multinomial(self.weights, self.batch_size, replacement=True)
            return batch, labels
        else:
            return batch, None
        