
from ..trainer.config_getters import get_model, get_datasets
from ..trainer.train_util import IPFStep
from ..diffusions.fast_sampler import FastSampler


@hydra.main(config_path="../conf", config_name="config")
def main(args):

    self.n_ipf =self.args.n_ipf
    self.nit =self.args.nit
    self.batch_size =self.args.batch_size
    self.num_iter =self.args.num_iter
    self.grad_clipping =self.args.grad_clipping   
    self.lr =self.args.lr
    self.classes = self.args.data_classes

    # get model
    model = get_model(args)

    # get steps
    
    # get time sampler

    # batch data laoder
    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    self.init_dl = repeater(DataLoader(init_ds, batch_size=self.args.batch_size, shuffle=True, **self.kwargs))
    self.init_dl = self.accelerator.prepare(self.init_dl)
    self.init_dl = repeater(self.init_dl)

    forward_diffusion = FastSampler(gammas, num_classes, time_sampler, mean_final, var_final)
    backward_diffusion = NetSampler(num_steps, shape, gammas, time_sampler, device = None, num_classes=0)

    IPFStep(model,
            forward_diffusion,
            backward_diffusion
            data_loader,
            batch_size = args.batch_size,
            lr = self.lr,
            ema_rate = 0.9990,
            log_interval = 10,
            save_interval = 10,
            weight_decay=0.0,
            lr_anneal_steps = 0,
            resume_checkpoint = False,
            forward_model = None,).run_loop()