import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision.utils as vutils
from PIL import Image
import os, sys

DPI = 200

def save_quiver(net, gammas, nit, xlim, ylim, im_dir='./im', gif_dir='./gif', name='', freq=1, ipf_it=None):
    if not os.path.isdir(im_dir):
        os.mkdir(im_dir)
    if not os.path.isdir(gif_dir):
        os.mkdir(gif_dir)

    nit = nit - 1
    name_gif = name + 'quiver'

    discx = torch.arange(xlim[0], xlim[1], 1)
    discy = torch.arange(ylim[0], ylim[1], 1)
    gridx, gridy = torch.meshgrid(discx, discy)
    Npt = discx.shape[0] * discy.shape[0]
    xval = torch.Tensor(Npt, 2).to(gammas.device)

    xval[:, 0] = gridx.flatten()
    xval[:, 1] = gridy.flatten()

    gridx_np = gridx.cpu().detach().cpu().numpy()
    gridy_np = gridy.cpu().detach().cpu().numpy()

    plot_paths = []    
    for k in range(nit):
        if k % freq == 0:
            filename =  name + 'quiver_' + str(k) + '.png'
            filename = os.path.join(im_dir, filename)

            t = torch.tensor([k - 1] * Npt).unsqueeze(-1).to(gammas.device)
            out = (net(xval, t) - xval) / gammas[k - 1]
            out_np = out.cpu().detach().cpu().cpu().numpy()

            out_np = out_np.reshape(discx.shape[0], discy.shape[0], 2)
            outx_np = out_np[:, :, 0]
            outy_np = out_np[:, :, 1]

            plt.clf()
            plt.quiver(gridx_np, gridy_np, outx_np, outy_np)

            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)
            
            plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)

            plot_paths.append(filename)

    make_gif(plot_paths, output_directory=gif_dir, gif_name=name+'quiver')


def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    frames = [Image.open(fn) for fn in plot_paths]

    frames[0].save(os.path.join(output_directory, f'{gif_name}.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100,
                   loop=0)

def save_sequence(nit, x, name='', im_dir='./im', gif_dir = './gif', xlim=None, ylim=None, ipf_it=None, freq=1):
    if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
    if not os.path.isdir(gif_dir):
        os.mkdir(gif_dir)

    # PARTICLES (INIT AND FINAL DISTRIB)

    plot_paths = []
    for k in range(nit):
        if k % freq == 0:
            filename =  name + 'particle_' + str(k) + '.png'
            filename = os.path.join(im_dir, filename)
            plt.clf()
            if (xlim is not None) and (ylim is not None):
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            plt.plot(x[-1, :, 0], x[-1, :, 1], '*')
            plt.plot(x[0, :, 0], x[0, :, 1], '*')
            plt.plot(x[k, :, 0], x[k, :, 1], '*')
            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)
                
            #plt.axis('equal')
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths.append(filename)

    # TRAJECTORIES

    N_part = 10
    filename = name + 'trajectory.png'
    filename = os.path.join(im_dir, filename)
    plt.clf()
    plt.plot(x[-1, :, 0], x[-1, :, 1], '*')
    plt.plot(x[0, :, 0], x[0, :, 1], '*')
    for j in range(N_part):
        xj = x[:, j, :]
        plt.plot(xj[:, 0], xj[:, 1], 'g', linewidth=2)
        plt.plot(xj[0,0], xj[0,1], 'rx')
        plt.plot(xj[-1,0], xj[-1,1], 'rx')
    plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)

    make_gif(plot_paths, output_directory=gif_dir, gif_name=name)

    # REGISTRATION

    colors = np.cos(0.1 * x[0, :, 0]) * np.cos(0.1 * x[0, :, 1])

    name_gif = name + 'registration'
    plot_paths_reg = []
    for k in range(nit):
        if k % freq == 0:
            filename =  name + 'registration_' + str(k) + '.png'
            filename = os.path.join(im_dir, filename)
            plt.clf()
            if (xlim is not None) and (ylim is not None):
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            plt.plot(x[-1, :, 0], x[-1, :, 1], '*', alpha=0)
            plt.plot(x[0, :, 0], x[0, :, 1], '*', alpha=0)
            plt.scatter(x[k, :, 0], x[k, :, 1], c=colors)
            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)            
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths_reg.append(filename)

    make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)

    # DENSITY

    name_gif = name + 'density'
    plot_paths_reg = []
    npts = 100
    for k in range(nit):
        if k % freq == 0:
            filename =  name + 'density_' + str(k) + '.png'
            filename = os.path.join(im_dir, filename)
            plt.clf()
            if (xlim is not None) and (ylim is not None):
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            else:
                xlim = [-15, 15]
                ylim = [-15, 15]
            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)                            
            plt.hist2d(x[k, :, 0], x[k, :, 1], range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]], bins=npts)
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths_reg.append(filename)

    make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)    
            


def save_ot(W_list, npar, data, name='', output_dir='./im'):

    filename =  name + 'animate.png'
    filename = os.path.join(output_dir, filename)    

    W_array = np.array(W_list)
    x = np.arange(W_array.shape[0])
    mean_W = np.mean(W_array, axis = 1)
    std_W = np.std(W_array, axis = 1)

    W = 0
    n_repeat = 10
    for k_repeat in range(n_repeat):
        init_sample_1 = data_distrib(npar, data)
        init_sample_2 = data_distrib(npar, data)
        init_1 = init_sample_1.detach().cpu().numpy()
        init_2 = init_sample_2.detach().cpu().numpy()
        W += ot_empirical(init_1, init_2)
    W /= n_repeat

    plt.clf()
    plt.plot(x, mean_W)
    plt.plot(x, x * 0 + W)
    plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)


class Plotter(object):

    def __init__(self):
        pass

    def plot(self, x_tot_plot, net, i, n, forward_or_backward):
        pass

    def __call__(self, initial_sample, x_tot_plot, net, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, net, i, n, forward_or_backward)


class ImPlotter(object):

    def __init__(self, im_dir = './im', gif_dir='./gif', plot_level=3):
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)
        self.im_dir = im_dir
        self.gif_dir = gif_dir
        self.num_plots = 50
        self.num_digits = 20
        self.plot_level = plot_level
        

    def plot(self, initial_sample, x_tot_plot, net, i, n, forward_or_backward, ipf_it):
        if self.plot_level > 0:
            x_tot_plot = x_tot_plot[:,:self.num_plots]
            name = '{0}_{1}_{2}'.format(forward_or_backward, n, i)
            im_dir = os.path.join(self.im_dir, name)
            
            if not os.path.isdir(im_dir):
                os.mkdir(im_dir)         
            
            if self.plot_level > 0:
                plt.clf()
                filename_grid_png = os.path.join(im_dir, 'im_grid_first.png')
                vutils.save_image(initial_sample, filename_grid_png, nrow=10)
                filename_grid_png = os.path.join(im_dir, 'im_grid_final.png')
                vutils.save_image(x_tot_plot[-1], filename_grid_png, nrow=10)

            if self.plot_level >= 2:
                plt.clf()
                plot_paths = []
                num_steps, num_particles, channels, H, W = x_tot_plot.shape
                x_tot_plot_tiff = x_tot_plot.cpu().numpy()
                x_tot_plot_tiff = np.moveaxis(x_tot_plot_tiff, 2, -1)        
                if channels == 1:
                    x_tot_plot_tiff = x_tot_plot_tiff.reshape(num_steps, num_particles, H, W)
                x_tot_plot_tiff =  np.concatenate([x_tot_plot_tiff[:,i] for i in range(self.num_digits)],1)
                plot_steps = np.linspace(0,num_steps-1,self.num_plots, dtype=int) 

                for k in plot_steps:
                    
                    
                    # filename_tiff = os.path.join(im_dir, 'im_{0}.tiff'.format(k))
                    # if channels == 1:
                    #     plt.imshow(x_tot_plot_tiff[k].astype('uint8'), cmap='gray')
                    # else:
                    #     plt.imshow(x_tot_plot_tiff[k].astype('uint8'))
                    
                    # # plot tiff
                    # if self.plot_level >= 3:
                    #     # save tiff            
                    #     im = Image.fromarray(x_tot_plot_tiff[k])
                    #     im.save(filename_tiff)
                    # plt.savefig(filename_png, bbox_inches = 'tight', transparent = False)
                    # save png
                    filename_grid_png = os.path.join(im_dir, 'im_grid_{0}.png'.format(k))    
                    # filename_png = os.path.join(im_dir, 'im_{0}.png'.format(k))
                    plot_paths.append(filename_grid_png)
                    vutils.save_image(x_tot_plot[k], filename_grid_png, nrow=10)
                    

                make_gif(plot_paths, output_directory=self.gif_dir, gif_name=name)

    def __call__(self, initial_sample, x_tot_plot, net, i, n, forward_or_backward, ipf_it):
        self.plot(initial_sample, x_tot_plot, net, i, n, forward_or_backward, ipf_it)


class TwoDPlotter(Plotter):

    def __init__(self, nit, gammas, im_dir = './im', gif_dir='./gif'):

        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)

        self.im_dir = im_dir
        self.gif_dir = gif_dir

        self.nit = nit
        self.gammas = gammas

    def plot(self, initial_sample, x_tot_plot, net, i, n, forward_or_backward, ipf_it):
        fb = forward_or_backward
        x_tot_plot = x_tot_plot.cpu().numpy()
        name = str(i) + '_' + fb +'_' + str(n) + '_'

        save_sequence(nit=self.nit, x=x_tot_plot, name=name, xlim=(-15,15),
                      ylim=(-15,15), ipf_it=ipf_it, freq=self.nit//min(self.nit,50),
                      im_dir=self.im_dir, gif_dir=self.gif_dir)

        save_quiver(net, self.gammas, self.nit, (-10,10), (-10,10),
                    name=name, im_dir=self.im_dir, gif_dir=self.gif_dir,  freq=self.nit//min(self.nit,50),
                    ipf_it=ipf_it)

    def __call__(self, initial_sample, x_tot_plot, net, i, n, forward_or_backward, ipf_it):
        self.plot(initial_sample, x_tot_plot, net, i, n, forward_or_backward, ipf_it)
