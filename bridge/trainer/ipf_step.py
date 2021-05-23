import copy
import functools
import os

import torch.nn.functional as F
import blobfile as bf
import torchvision.utils as vutils
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from ..models.unet.fp16_util import zero_grad
from tqdm  import tqdm
from ..utils import dist_util
import matplotlib.pyplot as plt


class IPFStep:
    def __init__(
        self,
        *,
        model,
        forward_diffusion,
        backward_diffusion,
        data_loader,
        prior_loader,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        cache_loader = False,
        num_iter = 1000,
        weight_decay=0.0,
        lr_anneal_steps = 0,
        resume_checkpoint = False,
        forward_model = None,
    ):
        self.model = model
        self.num_steps = forward_diffusion.num_steps
        self.prior_loader = prior_loader
        self.forward_diffusion = forward_diffusion
        self.backward_diffusion = backward_diffusion
        self.data_loader = data_loader
        self.num_iter = num_iter
        self.lr_anneal_steps = lr_anneal_steps
        self.batch_size = batch_size
        self.resume_checkpoint = resume_checkpoint
        self.cache_loader = cache_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.master_params = list(self.model.parameters())
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()

        # Optimizers
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
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

    def run_loop(self):
        if dist.get_rank() == 0:
            print('Training Loop ..............')
            pbar = tqdm(total =self.num_iter)
        while (
            self.step + self.resume_step < self.num_iter
        ):
            init_samples, labels = next(self.data_loader)
            init_samples = init_samples.to(dist_util.dev())
            labels = labels.to(dist_util.dev()) 

            x, target, steps, labels = self.forward_diffusion.compute_loss_terms(init_samples, labels)
            self.run_step(x, target, steps, labels)
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1
            if (dist.get_rank() == 0) & ((self.step) % 100):
                    pbar.update(self.step)
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            

    def run_step(self, x, target, steps, labels):
        eval_steps = self.num_steps - 1 - steps
        self.forward_backward(x, target, eval_steps, labels)
        self.optimize_step()
        self.log_step()

    def forward_backward(self, x, target, eval_steps, labels):
        zero_grad(self.master_params)
        pred = self.model(x, eval_steps, labels)
        loss = F.mse_loss(pred, target)
        loss.backward()


    def optimize_step(self):
        self._anneal_lr()
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
        init_samples, labels = next(self.prior_loader)
        init_samples = init_samples.to(dist_util.dev())
        labels = labels.to(dist_util.dev()) if labels is not None else None
        x = self.backward_diffusion.sample(init_samples, labels, net=self.model).detach().cpu()
        filename = 'final.png'
        plt.plot(x[:,-1,0], x[:,-1,1], 'ro')
        plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=200)
        plt.close()

        def save_checkpoint(rate, params):

            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
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


def get_blob_logdir():
    return './'


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

