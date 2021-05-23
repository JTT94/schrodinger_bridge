import hydra
import os,sys
import numpy as np
import torch


sys.path.append('..')

from bridge.utils import dist_util
from bridge.diffusions.time_sampler import TimeSampler
from bridge.trainer.config_getters import get_model, get_datasets, get_schedule, get_dataloader
from bridge.trainer.ipf_step import IPFStep
from bridge.diffusions import FastSampler, NetSampler, PriorSampler


@hydra.main(config_path="../conf", config_name="config")
def main(args):

    num_steps =args.nit
    batch_size =args.batch_size
    num_iter =args.num_iter
    grad_clipping =args.grad_clipping   
    lr =args.lr
    num_classes = args.num_data_classes

    dist_util.setup_dist()

    # get model
    model = get_model(args)
    model.to(dist_util.dev())

    # get steps
    
    # get time sampler

    # batch data laoder
    data_loader, mean_final, var_final = get_dataloader(args)
    batch, _ = next(data_loader)
    shape = batch[0].shape

    #prior loader
    prior_loader = PriorSampler(batch_size=args.batch_size, 
                                shape=shape, 
                                mean_final=mean_final, 
                                std_final = torch.sqrt(var_final), 
                                num_classes=num_classes, 
                                device=dist_util.dev())


    gammas = get_schedule(args)
    gammas = torch.tensor(gammas)
    gammas = gammas.to(dist_util.dev())

    # langevin
    if args.weight_distrib:
        alpha = args.weight_distrib_alpha
        prob_vec = (1 + alpha) * torch.sum(gammas) - torch.cumsum(gammas, 0)            
    else:
        prob_vec = gammas * 0 + 1
    time_sampler = TimeSampler(prob_vec)

    forward_diffusion = FastSampler(num_steps=num_steps, 
                                    shape=shape,  
                                    gammas=gammas, 
                                    num_classes=num_classes, 
                                    time_sampler=time_sampler, 
                                    mean_final=mean_final, 
                                    var_final=var_final, 
                                    device=dist_util.dev())

    backward_diffusion = NetSampler(num_steps=num_steps, 
                                    shape=shape,
                                    gammas=gammas, 
                                    num_classes=num_classes, 
                                    time_sampler=time_sampler, 
                                    device = dist_util.dev()
                                    )

    IPFStep(model=model,
            forward_diffusion=forward_diffusion,
            backward_diffusion=backward_diffusion,
            data_loader=data_loader,
            prior_loader = prior_loader,
            forward_model = None, args=args).run_loop()

if __name__ == '__main__':
    main()  