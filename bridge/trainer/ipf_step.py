from .ipf_base import IPFStepBase
from bridge.data import repeater
from torch.utils.data import DataLoader
import copy
import functools
import os
from ..data.cacheloader import CacheLoader
import torch.nn.functional as F
import blobfile as bf
import torchvision.utils as vutils
import numpy as np
import torch as th
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from ..models.unet.fp16_util import zero_grad
from tqdm  import tqdm
from ..utils import dist_util
import matplotlib.pyplot as plt

class IPFStep(IPFStepBase):

    def __init__(self,    
                    model,
                    forward_diffusion,
                    backward_diffusion,
                    data_loader,
                    prior_loader,
                    cache_data_loader = None,
                    forward_model = None,
                    args=None, 
                    cache_loader = False, 
                    resume_checkpoint = 0,
                    checkpoint_directory = './',
                    plot_directory = './'):
        
        super().__init__(model=model,
                        forward_diffusion=forward_diffusion,
                        backward_diffusion=backward_diffusion,
                        data_loader=data_loader,
                        prior_loader=prior_loader,
                        cache_data_loader=cache_data_loader,
                        forward_model=forward_model,
                        args=args, 
                        cache_loader = cache_loader, 
                        resume_checkpoint = resume_checkpoint,
                        checkpoint_directory = checkpoint_directory,
                        plot_directory = plot_directory)

    def run_loop(self):
        if dist.get_rank() == 0:
            print('Training Loop ..............')
            pbar = tqdm(total =self.num_iter)
        
        if self.cache_loader:
            cache_dl = None
            torch.cuda.empty_cache()
            cache_dl, cache_ds = self.get_cacheloader()


        while (
            self.step + self.resume_step < self.num_iter
        ):
            init_samples, labels = next(self.data_loader)
            init_samples = init_samples.to(dist_util.dev())
            labels = None if not self.classes else labels.to(dist_util.dev()) 

            if not self.cache_loader:
                x, target, steps, labels = self.forward_diffusion.compute_loss_terms(init_samples, labels)
            else:
                if self.classes:
                    x, target, steps, labels = next(cache_dl)
                else:
                    x, target, steps = next(cache_dl)
                    labels = None
            self.run_step(x, target, steps, labels)
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1
            if (dist.get_rank() == 0) & ((self.step) % 1==0):
                    pbar.update()

            if (self.step % self.cache_refresh == 0) & (self.step > 0) & self.cache_loader:
                cache_ds.deconstruct()
                cache_dl = None
                cache_ds = None
                torch.cuda.empty_cache()
                cache_dl, cache_ds = self.get_cacheloader()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            torch.save(self.model, os.path.join(self.checkpoint_dir, 'final_model.pt'))
            zero_grad(self.master_params)
            if self.cache_loader:
                cache_ds.deconstruct()
            cache_dl = None
            torch.cuda.empty_cache()
            

    def run_step(self, x, target, steps, labels):
        eval_steps = self.num_steps - 1 - steps
        self.forward_backward(x, target, eval_steps, labels)
        self.optimize_step()
        self.log_step()

    def forward_backward(self, x, target, eval_steps, labels):
        zero_grad(self.master_params)
        pred = self.model(x, eval_steps, labels) - x
        loss = F.mse_loss(pred, target)
        loss.backward() 
         
    def get_cacheloader(self):
        cache_ds = CacheLoader(self.forward_model, 
                                self.cache_data_loader, 
                                self.args.num_cache_batches, 
                                self.forward_diffusion, 
                                self.args.cache_npar, 
                                device=dist_util.dev(),
                                t_batch = self.args.t_batch,
                                classes = self.classes)
        cache_dl = DataLoader(cache_ds, batch_size=self.args.batch_size)
        cache_dl = repeater(cache_dl)
        return cache_dl, cache_ds