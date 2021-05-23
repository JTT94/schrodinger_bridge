import torch
import numpy as np
from ..models import *
from ..data.two_dim import two_dim_ds
from ..data.mnist import MNIST
#from ..data.emnist import EMNIST
from ..data.celeba  import CelebA
from .plotters import TwoDPlotter, ImPlotter
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import os
from bridge.data import repeater
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision.datasets import CIFAR10
cmp = lambda x: transforms.Compose([*x])

def get_plotter(runner, args):
    dataset_tag = getattr(args, DATASET)
    if dataset_tag == DATASET_2D:
        return TwoDPlotter(nit=runner.nit, gammas=runner.langevin.gammas)
    else:
        return ImPlotter(plot_level = args.plot_level)

# Model
#--------------------------------------------------------------------------------

MODEL = 'Model'
BASIC_MODEL = 'Basic'
UNET_MODEL = 'UNET'


def get_model(args):
    model_tag = getattr(args, MODEL)
    
    if model_tag == BASIC_MODEL:
        net = ScoreNetwork()

    if model_tag == UNET_MODEL:
        image_size=args.data.image_size

        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 28:
            channel_mult = (1, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = []
        for res in args.model.attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        
        kwargs = {
                    "in_channels": args.data.channels,
                    "model_channels": args.model.num_channels,
                    "out_channels": args.data.channels,
                    "num_res_blocks": args.model.num_res_blocks,
                    "attention_resolutions": tuple(attention_ds),
                    "dropout": args.model.dropout,
                    "channel_mult": channel_mult,
                    "num_classes": args.num_data_classes,
                    "use_checkpoint": args.model.use_checkpoint,
                    "num_heads": args.model.num_heads,
                    "num_heads_upsample": args.model.num_heads_upsample,
                    "use_scale_shift_norm": args.model.use_scale_shift_norm
                }

        net = UNetModel(**kwargs)  
    
    return net

# Optimizer
#--------------------------------------------------------------------------------
def get_optimizers(net_f, net_b, lr):
    return torch.optim.Adam(net_f.parameters(), lr=lr), torch.optim.Adam(net_b.parameters(), lr=lr)

# Dataset
#--------------------------------------------------------------------------------

DATASET = 'Dataset'
DATASET_TRANSFER = 'Dataset_transfer'
DATASET_2D = '2d'
DATASET_CELEBA = 'celeba'
DATASET_STACKEDMNIST = 'stackedmnist'
DATASET_CIFAR10 = 'cifar10'
DATASET_EMNIST = 'emnist'


def get_datasets(args):
    dataset_tag = getattr(args, DATASET)
    if args.transfer:
        dataset_transfer_tag = getattr(args, DATASET_TRANSFER)
    else:
        dataset_transfer_tag = None
    
    # INITIAL (DATA) DATASET

    # 2D DATASET
    
    if dataset_tag == DATASET_2D:
        data_tag = args.data
        npar = max(args.npar, args.cache_npar)
        init_ds = two_dim_ds(npar, data_tag)

    if dataset_transfer_tag == DATASET_2D:
        data_tag = args.data_transfer
        npar = max(args.npar, args.cache_npar)
        final_ds = two_dim_ds(npar, data_tag)
        mean_final = torch.tensor(0.)
        var_final = torch.tensor(1.*10**3) #infty like

    # CELEBA DATASET

    if dataset_tag == DATASET_CELEBA:

        train_transform = [transforms.CenterCrop(140), transforms.Resize(args.data.image_size), transforms.ToTensor()]
        test_transform = [transforms.CenterCrop(140), transforms.Resize(args.data.image_size), transforms.ToTensor()]
        if args.data.random_flip:
            train_transform.insert(2, transforms.RandomHorizontalFlip())


        root = os.path.join(args.data_dir, 'celeba')
        init_ds = CelebA(root, split='train', transform=cmp(train_transform), download=False)

    # MNIST DATASET

    if dataset_tag ==  DATASET_STACKEDMNIST:
        root = os.path.join(args.data_dir, 'mnist')
        saved_file = os.path.join(root, "data.pt")
        load = os.path.exists(saved_file) 
        load = args.load
        init_ds = MNIST(root, load=load, source_root=root, 
                        train=True, num_channels = args.data.channels, 
                        imageSize=args.data.image_size,
                        device=args.device)

    if dataset_transfer_tag == DATASET_STACKEDMNIST:
        root = os.path.join(args.data_dir, 'mnist')
        saved_file = os.path.join(root, "data.pt")
        load = os.path.exists(saved_file)
        load = args.load
        final_ds = MNIST(root, load=load, source_root=root,
                        train=True, num_channels = args.data.channels,
                        imageSize=args.data.image_size,
                        device=args.device)
        mean_final = torch.tensor(0.)
        var_final = torch.tensor(1.*10**3)

    # EMNIST DATASET

    if dataset_tag == DATASET_EMNIST:
        root = os.path.join(args.data_dir, 'EMNIST')
        saved_file = os.path.join(root, "data.pt")
        load = os.path.exists(saved_file)
        load = args.load
        init_ds = EMNIST(root, load=load, source_root=root,
                                train=True, num_channels = args.data.channels,
                                imageSize=args.data.image_size,
                                device=args.device)

    if dataset_transfer_tag == DATASET_EMNIST:
        root = os.path.join(args.data_dir, 'EMNIST')
        saved_file = os.path.join(root, "data.pt")
        load = os.path.exists(saved_file)
        load = args.load
        final_ds = EMNIST(root, load=load, source_root=root,
                                train=True, num_channels = args.data.channels,
                                imageSize=args.data.image_size,
                                device=args.device)
        mean_final = torch.tensor(0.)
        var_final = torch.tensor(1.*10**3)

    # CIFAR 10 DATASET
        
    if dataset_tag == DATASET_CIFAR10:

        train_transform = [transforms.Resize(args.data.image_size), transforms.ToTensor()]
        test_transform = [transforms.Resize(args.data.image_size), transforms.ToTensor()]
        if args.data.random_flip:
            train_transform.insert(1, transforms.RandomHorizontalFlip())

        path = os.path.join(args.data_dir, 'CIFAR10')
        init_ds = CIFAR10(path, train=True, download=True, transform=cmp(train_transform))
        #test_dataset = CIFAR10(path, train=False, download=True, transform=cmp(test_transform))

    # FINAL (GAUSSIAN) DATASET (if no transfer)

    if not(args.transfer):
        if args.adaptive_mean:
            NAPPROX = 100
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX)))[0]
            mean_final = vec.mean()
            mean_final = vec[0] * 0 + mean_final
            var_final = eval(args.var_final)
            final_ds = None
        elif args.final_adaptive:
            NAPPROX = 100
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX)))[0]
            mean_final = vec.mean(axis=0)
            var_final = vec.var()
            final_ds = None
        else:
            mean_final = eval(args.mean_final)
            var_final = eval(args.var_final)
            final_ds = None
        

    return init_ds, final_ds, mean_final, var_final


def get_schedule(args):
    num_diffusion_timesteps = args.nit
    n = num_diffusion_timesteps//2
    if args.gamma_space == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            max_beta=args.gamma_max
        )
    elif args.gamma_space == 'linspace':
        gamma_half = np.linspace(args.gamma_min,args.gamma_max, n)
        return np.concatenate([gamma_half, np.flip(gamma_half)])
    elif args.gamma_space == 'geomspace':
        gamma_half = np.geomspace(args.gamma_min, args.gamma_max, n)
        return np.concatenate([gamma_half, np.flip(gamma_half)])


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_dataloader(args):
    def worker_init_fn(worker_id):                                                          
            np.random.seed(np.random.get_state()[1][0] + worker_id + dist.get_rank())
    kwargs = {"num_workers": args.num_workers, 
            "pin_memory": args.pin_memory, 
            "worker_init_fn": worker_init_fn,
            "drop_last": True}
    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    data_loader = repeater(DataLoader(init_ds, batch_size=args.batch_size, shuffle=True, **kwargs))
    data_loader = repeater(data_loader)
    return data_loader, mean_final, var_final