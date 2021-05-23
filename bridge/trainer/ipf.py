import os,sys
import numpy as np
import torch

from bridge.utils import dist_util
from bridge.diffusions.time_sampler import TimeSampler
from bridge.trainer.config_getters import get_model, get_datasets, get_schedule, get_dataloader
from bridge.diffusions import FastSampler, NetSampler, PriorSampler
from bridge.trainer.ipf_step import IPFStep

class IPF(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_ipf = args.n_ipf
        self.num_steps =args.nit
        self.batch_size =args.batch_size
        self.num_iter =args.num_iter
        self.grad_clipping =args.grad_clipping   
        self.lr =args.lr
        self.out_dir = './' if args.out_dir is None else args.out_dir
        self.num_classes = args.num_data_classes

        dist_util.setup_dist()

        # marginal loaders
        self.data_loader, mean_final, var_final = get_dataloader(args)

        # get shape
        batch, _ = next(self.data_loader)
        self.shape = batch[0].shape

        #prior loader
        self.prior_loader = PriorSampler(batch_size=args.batch_size, 
                                    shape=self.shape, 
                                    mean_final=mean_final, 
                                    var_final = var_final, 
                                    num_classes=self.num_classes, 
                                    device=dist_util.dev())

        gammas = get_schedule(args)
        gammas = torch.tensor(gammas)
        self.gammas = gammas.to(dist_util.dev())


        # langevin
        if args.weight_distrib:
            alpha = args.weight_distrib_alpha
            prob_vec = (1 + alpha) * torch.sum(gammas) - torch.cumsum(gammas, 0)            
        else:
            prob_vec = gammas * 0 + 1
        self.time_sampler = TimeSampler(prob_vec)

    def ipf_loop(self):
        for i in range(self.n_ipf):
            for train_direction in ['backward', 'forward']:

                checkpoint_dir = os.path.join(self.out_dir, train_direction, str(i), 'checkpoints')
                plot_dir = os.path.join(self.out_dir, train_direction, str(i), 'im')
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)

                if train_direction == 'forward':
                    prev_checkpoint_dir =  os.path.join(self.out_dir, 'backward', str(i), 'checkpoints')
                else:
                    prev_checkpoint_dir = os.path.join(self.out_dir, 'forward', str(i-1), 'checkpoints')

                if (i == 0) & (train_direction=='backward'):
                    cache = False
                    forward_diffusion = FastSampler(num_steps=self.num_steps, 
                                                    shape=self.shape,  
                                                    gammas=self.gammas, 
                                                    num_classes=self.num_classes, 
                                                    time_sampler=self.time_sampler, 
                                                    mean_final=self.prior_loader.mean_final, 
                                                    var_final=self.prior_loader.var_final, 
                                                    device=dist_util.dev())
                else:
                    cache = True
                    forward_diffusion = NetSampler(num_steps=self.num_steps, 
                                                    shape=self.shape,
                                                    gammas=self.gammas, 
                                                    num_classes=self.num_classes, 
                                                    time_sampler=self.time_sampler, 
                                                    device = dist_util.dev()
                                                    )

                backward_diffusion = NetSampler(num_steps=self.num_steps, 
                                        shape=self.shape,
                                        gammas=self.gammas, 
                                        num_classes=self.num_classes, 
                                        time_sampler=self.time_sampler, 
                                        device = dist_util.dev()
                                        )
                if train_direction == 'backward':
                    data_loader = self.data_loader
                    prior_loader = self.prior_loader
                    
                else: #forward becomes backward and vice versa
                    data_loader = self.prior_loader
                    prior_loader = self.data_loader

                # get model, to train
                model = get_model(self.args)
                model.to(dist_util.dev())

                # model to sample from
                if (i == 0) & (train_direction == 'backward'):
                    sample_model = None
                else:
                    file_path = self.find_last_checkpoint(prev_checkpoint_dir)
                    sample_model = get_model(self.args)
                    sample_model.load_state_dict(
                            dist_util.load_state_dict(file_path, map_location="cpu")
                        )
                    sample_model.to(dist_util.dev())



                IPFStep(model=model,
                        forward_diffusion=forward_diffusion,
                        forward_model = sample_model,
                        backward_diffusion=backward_diffusion,
                        data_loader=data_loader,
                        prior_loader=prior_loader,
                        args=self.args, 
                        cache_loader = cache, 
                        checkpoint_directory=checkpoint_dir,
                        plot_directory= plot_dir).run_loop()
                

    def find_last_checkpoint(self, checkpoint_directory, ema=True):
        """
        filename = f"ema_{rate}_{(step):06d}.pt"
        """
        
        tag = 'ema_' if ema else 'model'
        checkpoints = os.listdir(checkpoint_directory)
        step = 0
        rate = 0
        for fn in checkpoints:
            if tag in fn:
                current_step = int(fn.split('_')[-1].split('.')[0])
                current_rate = int(fn.split('_')[1].split('.')[0])
                if current_step > step:
                    filename = fn

        path = os.path.join(checkpoint_directory, filename)
        return path

