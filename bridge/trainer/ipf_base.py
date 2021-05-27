import copy
import functools
import os
import random
import torch
import torch.nn.functional as F
import blobfile as bf
import torchvision.utils as vutils
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Adam
from ..models.unet.fp16_util import zero_grad
from tqdm  import tqdm
from ..utils import dist_util
import matplotlib.pyplot as plt
from .plotters import ImPlotter
from .config_getters import get_model


class IPFStepBase(th.nn.Module):
    def __init__(
        self,
        model,
        forward_diffusion,
        backward_diffusion,
        data_loader,
        prior_loader,
        cache_data_loader = None,
        args = None,
        forward_model = None,
        cache_loader = False, 
        resume_checkpoint = 0,
        checkpoint_directory = './',
        plot_directory = './',
    ):

        super().__init__()

        self.set_seed(dist.get_rank()+0)
        ema_rate = args.ema_rate
        save_interval=args.save_interval
        lr_anneal_steps = 0
        self.args = args
        self.model = model
        self.forward_diffusion = forward_diffusion
        self.backward_diffusion = backward_diffusion

        self.forward_model = forward_model
        self.prior_loader = prior_loader
        self.data_loader = data_loader
        self.cache_data_loader = cache_data_loader

        self.num_steps = self.args.nit
        self.num_iter = self.args.num_iter
        self.lr_anneal_steps = lr_anneal_steps
        self.batch_size = self.args.batch_size
        self.cache_loader = cache_loader
        self.cache_refresh = self.args.cache_refresh_stride
        self.lr = self.args.lr
        self.classes = self.args.num_data_classes > 0
        self.weight_decay = self.args.weight_decay
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_directory
        self.plot_dir = plot_directory
        self.plotter = ImPlotter(im_dir=self.plot_dir, plot_level=1)

        self.step = 0
        self.resume_step = resume_checkpoint
        self.resume_checkpoint = resume_checkpoint
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()

        # Optimizers
        self.opt = Adam(self.master_params, lr=self.lr)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                self.use_ddp = False
                self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def optimize_step(self):
        #self._anneal_lr()
        # if self.args.grad_clipping:
        #     clipping_param = self.args.grad_clip
        #     total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_param)
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr


    def save(self):
        if dist.get_rank() == 0:
            self.set_seed(0)
            init_samples, labels = next(self.prior_loader)
            init_samples = init_samples.to(dist_util.dev())
            labels = labels.to(dist_util.dev()) if labels is not None else None

            sample_model = get_model(self.args)
            for rate, params in zip(self.ema_rate, self.ema_params):
                state_dict = self._master_params_to_state_dict(params)
                sample_model.load_state_dict(state_dict)
                sample_model = sample_model.to(dist_util.dev())
                x_tot_plot = self.backward_diffusion.sample(init_samples, labels, t_batch=None, net=sample_model)
                filename = 'ema{0}_step{1}.png'.format(rate, self.step)
                self.plotter.plot(init_samples, x_tot_plot, filename)
            sample_model = None
            torch.cuda.empty_cache()

            # init_samples, labels = next(self.data_loader)
            # init_samples = init_samples.to(dist_util.dev())
            # labels = labels.to(dist_util.dev()) if labels is not None else None
            # x_tot_plot = self.forward_diffusion.sample(init_samples, labels, t_batch=None, net=self.forward_model)
            # filename = 'sample{0}_step{1}.png'.format(rate, self.step)
            # self.plotter.plot(init_samples, x_tot_plot, filename)
            
        def save_checkpoint(rate, params):

            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(self.checkpoint_dir, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.checkpoint_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return params

    def log_step(self):
        return 
    
    def get_blob_logdir(self):
        return self.plot_dir

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0





def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

