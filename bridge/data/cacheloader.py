import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time

class CacheLoader(Dataset):
    def __init__(self, 
                 sample_net, 
                 data_loader, 
                 num_batches, 
                 forward_diffusion, 
                 batch_size, 
                 device='cpu', 
                 t_batch=None
                 ): 

        super().__init__()
        shape = forward_diffusion.shape
        num_steps = forward_diffusion.num_steps
        if t_batch is None:
            t_batch = num_steps
        else:
            t_batch = min(t_batch, num_steps)
        self.forward_model = sample_net
        self.forward_diffusion = forward_diffusion
        self.data = torch.zeros((num_batches, batch_size*t_batch, 2, *shape)).to(device)#.cpu()
        self.steps_data = torch.zeros((num_batches, batch_size*t_batch,1), dtype=torch.long).to(device)#.cpu() # steps
        self.labels = torch.zeros((num_batches, batch_size*t_batch,1), dtype=torch.long).to(device)#.cpu() # steps
        self.classes = False
        with torch.no_grad():
            for b in range(num_batches):
                batch, labels = next(data_loader)
                x, target, steps, labels = self.forward_diffusion.compute_loss_terms(batch, labels, 
                                                                                     net=self.forward_model, 
                                                                                     t_batch=t_batch)
                x = x.unsqueeze(2)
                target = target.unsqueeze(2)
                batch_data = torch.cat((x, target), dim=2)
                flat_data = batch_data.flatten(start_dim=0, end_dim=1)#.to('cpu')
                self.data[b] = flat_data #torch.cat((self.data, flat_data),0)
                
                # store steps
                flat_steps = steps.flatten(start_dim=0, end_dim=1)#.to('cpu')
                self.steps_data[b] = flat_steps # = torch.cat((self.steps_data, flat_steps),0)

                if labels is not None:
                    self.classes = True
                    labels_flat = labels.flatten(start_dim=0, end_dim=1)
                    self.labels[b] = labels_flat
        
        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)
    
    def __getitem__(self, index):
        item = self.data[index]
        x = item[0]
        out = item[1]
        steps = self.steps_data[index]
        if self.classes:
            labels = self.labels[index]
        else:
            labels = -1
        return x, out, steps, labels

    def __len__(self):
        return self.data.shape[0]
