

class IPF(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_ipf = n_ipf
        self.num_steps =args.nit
        self.batch_size =args.batch_size
        self.num_iter =args.num_iter
        self.grad_clipping =args.grad_clipping   
        self.lr =args.lr
        self.num_classes = args.num_data_classes

        dist_util.setup_dist()

        # marginal loaders
        data_loader, mean_final, var_final = get_dataloader(args)

        # get shape
        batch, _ = next(data_loader)
        self.shape = batch[0].shape

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
        forward_gammas = gammas
        backward_gammas, _ = torch.sort(gammas, ascendi)


        # langevin
    if args.weight_distrib:
        alpha = args.weight_distrib_alpha
        prob_vec = (1 + alpha) * torch.sum(gammas) - torch.cumsum(gammas, 0)            
    else:
        prob_vec = gammas * 0 + 1
    time_sampler = TimeSampler(prob_vec)

    def ipf_loop(self):
        for i in range(self.n_ipf):
            for train_direction in ['backward', 'forward']:

                if (i == 0) & (train_direction=='backward'):
                    forward_diffusion = FastSampler(num_steps=num_steps, 
                                                    shape=shape,  
                                                    gammas=gammas, 
                                                    num_classes=num_classes, 
                                                    time_sampler=time_sampler, 
                                                    mean_final=mean_final, 
                                                    var_final=var_final, 
                                                    device=dist_util.dev())
                else:

                    forward_diffusion = NetSampler(num_steps=num_steps, 
                                                    shape=shape,
                                                    gammas=gammas, 
                                                    num_classes=num_classes, 
                                                    time_sampler=time_sampler, 
                                                    device = dist_util.dev()
                                                    )

                    backward_diffusion = NetSampler(num_steps=num_steps, 
                                            shape=shape,
                                            gammas=gammas, 
                                            num_classes=num_classes, 
                                            time_sampler=time_sampler, 
                                            device = dist_util.dev()
                                            )
                if train_direction == 'backward':
                    data_loader = self.data_loader
                    prior_loader = self.prior_loader
                    
                else:
                    data_loader = self.prior_loader
                    prior_loader = self.data_loader

                # get model, to train
                model = get_model(args)
                model.to(dist_util.dev())

                # to sample from
                if (i == 0) & (train_direction == 'backward'):
                    sample_model = None
                else:
                    file_path = find_last_checkpoint(train_direction, checkpoint_directory)
                    sample_model.load_state_dict(
                            dist_util.load_state_dict(args.model_path, map_location="cpu")
                        )
                    sample_model.to(dist_util.dev())



                IPFStep(model=model,
                        forward_diffusion=forward_diffusion,
                        forward_model = sample_model
                        backward_diffusion=backward_diffusion,
                        data_loader=data_loader,
                        prior_loader=prior_loader,
                        batch_size = args.batch_size,
                        lr = lr,
                        ema_rate = 0.9990,
                        log_interval = 10,
                        save_interval = 5000,
                        weight_decay=0.0,
                        num_iter=args.num_iter,
                        lr_anneal_steps = 0,
                        resume_checkpoint = False,
                        ).run_loop()

    def find_last_checkpoint(train_direction, checkpoint_directory, ema=True):
        """
        filename = f"ema_{rate}_{(step):06d}.pt"
        """
        
        tag = 'ema_' if ema else 'model'
        checkpoints = os.listdir(checkpoint_directory)
        step = 0
        rate = 0
        for fn in checkpoints:
            if tag in fn:
                current_step = fn.split('.')[0].split('_')[-1]
                current_rate = fn.split('.')[0].split('_')[1]
                if current_step > step:
                    step = current_step
                    rate = current_rate
        
        filename = f"ema_{rate}_{step}.pt"

        path = bf.join(bf.dirname(checkpoint_directory), filename)
        return path

