import torch
import numpy as np
import h5py as h5
from configs import Config
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import os
import sys
import scipy.io as sio
import random
from PIL import Image

conf = Config().parse()

class TrackConf:
    def __init__(self, conf):
        self._conf = conf
        self._used = set()
        
    def __getattr__(self, k):
        self._used.add(k)
        return getattr(self._conf, k)
    
    @property
    def used_dict(self):
        return {k: getattr(self._conf, k) for k in self._used}


def prepare_folders():
    if not os.path.exists(conf.out_path):
        os.makedirs(conf.out_path+'/train')
        os.makedirs(conf.out_path+'/train/outputs')
        os.makedirs(conf.out_path+'/train/weights')
        print(f"Directory '{conf.out_path}' created successfully.")
    else:
        print(f"Directory '{conf.out_path}' already exists. If you want to change this, please see the config file.")
        while True:
            write_over = input("Do you want to print over the existing results? (y/n): ")
            if write_over.lower() == "y":
                print("Printing over the existing results!")
                break
            elif write_over.lower() == "n":
                print("Exiting the code!")
                sys.exit()
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")


def load_data(ntrain_slices, batchSize):  # please customize it
    if conf.data_type == 'PD':
        kspace_train = h5.File("./dataset/samples/train/sample_PD.h5", "r")['kspace']
        maps_train = h5.File("./dataset/samples/train/sample_PD.h5", "r")['trnCsm']
    elif conf.data_type == 'PDFS':
        kspace_train = h5.File("__data_route__.h5", "r")['kspace']
        maps_train = h5.File("__data_route__.h5", "r")['trnCsm']
    elif conf.data_type == 'AXT2': 
        kspace_train = h5.File("__data_route__.h5", "r")['kspace']
        maps_train = h5.File("__data_route__.h5", "r")['coil_sens_maps']
        kspace_train = kspace_train.astype(np.complex64)
        maps_train = maps_train.astype(np.complex64)

    ntotal_slices = kspace_train.shape[0]
    if ntrain_slices!=1:
        random_slices_indices = np.random.choice(ntotal_slices, size=ntrain_slices, replace=False)
        random_slices_indices = np.sort(random_slices_indices)
    else:
        random_slices_indices = np.arange(conf.idx_slice, conf.idx_slice +1)

    kspace_train = kspace_train[random_slices_indices]
    maps_train = maps_train[random_slices_indices]
    ksp = torch.from_numpy(np.asarray(kspace_train))
    maps = torch.from_numpy(np.asarray(maps_train))    

    if conf.train_shuffle:
        dataset = TensorDataset(ksp, maps)
        train_data = DataLoader(dataset, batch_size=batchSize, shuffle=True)
        train_slices = DataLoader([torch.squeeze(batch[0], dim=0) for batch in train_data], 
                                    batch_size=batchSize, shuffle=False)
        maps = torch.cat([batch[1] for batch in train_data], dim=0)
    else:
        train_slices = DataLoader(ksp, batch_size=batchSize, shuffle=False)  
    return train_slices, maps

def load_data_test(batchSize, slices=None): # please customize it
    if conf.data_type == 'AXT2':
        kspace_valid = h5.File("__data_route__.h5", "r")['kspace']
        maps_valid = h5.File("__data_route__.h5", "r")['coil_sens_maps']

        kspace_valid = kspace_valid.astype(np.complex64)
        maps_valid = maps_valid.astype(np.complex64)
    else:
        kspace_valid = h5.File("./dataset/samples/test/sample_PD.h5", "r")['kspace']
        maps_valid = h5.File("./dataset/samples/test/sample_PD.h5", "r")['testCsm']

    ksp_valid = torch.from_numpy(np.asarray(kspace_valid)).permute(0,3,1,2)  # N, 15, 320, 368
    maps_valid = torch.from_numpy(np.asarray(maps_valid)).permute(0,3,1,2)   # N, 15, 320, 368

    ksp_loader = DataLoader(ksp_valid, batch_size=batchSize, shuffle=False)
    maps_loader = DataLoader(maps_valid, batch_size=batchSize, shuffle=False)
    print("Loading Test Data is completed ...")
    return ksp_loader, maps_loader
    

def cus_ifft(img, dims):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(img, dim=dims), dim=dims, norm = 'ortho'), dim=dims)
    

def cus_fft(kspace, dims):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(kspace, dim=dims), dim=dims, norm = 'ortho'), dim=dims)


def mask_gen(ACS_length, Nro, Npe, Nch, R):    # please customize it 
    ACS_start = int((Npe - ACS_length)/2) - 1
    ACS_end = ACS_start + ACS_length
    mask = torch.zeros(size = (Nro, Npe, Nch), dtype = torch.complex64)
    mask[:, 0::R, :] = 1
    mask[:, ACS_start:ACS_end, :] = 1
    # k-space regions with no signal
    if conf.data_type == 'AXT2':
        pass
    else:
        mask[:, 0:17, :] = 1
        mask[:, 352:Npe, :] = 1
    return mask.permute(2,0,1).unsqueeze(0)

def mask_ksp_gen(Nro, Npe, Nch):    # please customize it
    mask = torch.ones(size = (Nro, Npe, Nch), dtype = torch.complex64)
    # k-space regions with no signal
    if conf.data_type == 'AXT2':
        pass
    else:
        mask[:, 0:17, :] = 0
        mask[:, 352:Npe, :] = 0
    return mask.permute(2,0,1).unsqueeze(0)

def cal_SSIM(ref, recon):
    return structural_similarity(ref, recon, data_range=recon.max() - recon.min())


def cal_PSNR(ref, recon):
    
    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    return 20 * np.log10(ref.max() / (np.sqrt(mse) + 1e-10))


def l1_L2_loss(reference, output):
    return torch.norm((reference-output), p=2)/torch.norm(reference, p=2) + torch.norm((reference-output), p=1)/torch.norm(reference, p=1)
    

def plot_loss(loss, epochs):
    plt.close('all')
    plt.plot(range(1, epochs+1), loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Epochs')
    plt.grid(True)
    plt.savefig(conf.out_path+'/train/train_loss_vs_epochs.png')
    sio.savemat(conf.out_path +'/train/loss_pd_r4.mat', {'loss_pd_r4': loss})


def plot_mu(mu_values, epochs):
    plt.close('all')
    plt.plot(range(1, epochs+1), mu_values)
    plt.xlabel('Epochs')
    plt.ylabel('mu')
    plt.title('mu vs Epochs')
    plt.grid(True)
    plt.savefig(conf.out_path+'/train/mu_vs_epochs.png')
    sio.savemat(conf.out_path +'/train/mu_pd_r4.mat', {'mu_pd_r4': mu_values})

def plot_gamma(gamma_values, epochs):
    plt.close('all')
    plt.plot(range(1, epochs+1), gamma_values)
    plt.xlabel('Epochs')
    plt.ylabel('gamma')
    plt.title('gamma vs Epochs')
    plt.grid(True)
    plt.savefig(conf.out_path+'/train/gamma_vs_epochs.png')
    sio.savemat(conf.out_path +'/train/gamma_pd_r4.mat', {'gamma_pd_r4': gamma_values})


def plot_mu_multi(mu_values, epochs, nb_unroll_blocks=conf.nb_unroll_blocks):
    plt.close('all')
    mu_values = np.array(mu_values)  
    for stage_idx in range(nb_unroll_blocks):
        plt.plot(range(1, epochs+1), mu_values[:, stage_idx], label=f'Stage {stage_idx+1}')
    plt.xlabel('Epochs')
    plt.ylabel('mu')
    plt.title('mu vs Epochs')
    plt.legend(bbox_to_anchor=(1.3, 1), loc='upper right')  # Move legend to upper right outside the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(conf.out_path+'/train/mu_vs_epochs.png', bbox_inches='tight')
    sio.savemat(conf.out_path +'/train/mu_pd_r4.mat', {'mu_pd_r4': mu_values})

def plot_gamma_multi(gamma_values, epochs, nb_unroll_blocks=conf.nb_unroll_blocks):
    plt.close('all')
    gamma_values = np.array(gamma_values)  
    for stage_idx in range(nb_unroll_blocks):
        plt.plot(range(1, epochs+1), gamma_values[:, stage_idx], label=f'Stage {stage_idx+1}')
    plt.xlabel('Epochs')
    plt.ylabel('gamma_values')
    plt.title('gamma_values vs Epochs')
    plt.legend(bbox_to_anchor=(1.3, 1), loc='upper right')  # Move legend to upper right outside the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(conf.out_path+'/train/gamma_vs_epochs.png', bbox_inches='tight')
    sio.savemat(conf.out_path +'/train/gamma_pd_r4.mat', {'gamma_pd_r4': gamma_values})


def plot_quantitative(psnr_values, ssim_values, epochs):
    plt.close('all')
    plt.plot(range(1, epochs+1, conf.save_freq), psnr_values)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Epochs')
    plt.grid(True)
    plt.savefig(conf.out_path+'/train/psnr_vs_epochs.png')
    plt.close('all')
    plt.plot(range(1, epochs+1, conf.save_freq), ssim_values)
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.title('SSIM vs Epochs')
    plt.grid(True)
    plt.savefig(conf.out_path+'/train/ssim_vs_epochs.png')
    sio.savemat(conf.out_path +'/train/psnr_pd_r4.mat', {'psnr_pd_r4': psnr_values})
    sio.savemat(conf.out_path +'/train/ssim_pd_r4.mat', {'ssim_pd_r4': ssim_values})


def print_h5py_structure(name, obj):
    if isinstance(obj, h5.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")


def add_boundaries(stage_list, boundary_width, device):
    with_boundaries = []
    for i, stage in enumerate(stage_list):
        with_boundaries.append(stage)
        if i < len(stage_list) - 1:  
            H, W = stage.size()[-2:]
            boundary = torch.ones((H, boundary_width), dtype=stage.dtype, device=device) * 255 # abs(stage).max()
            with_boundaries.append(boundary.unsqueeze(0))
    return torch.cat(with_boundaries, dim=-1)  


def flip_and_crop(img, flip, crop):
    if flip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if crop:
        width, height = img.size
        img = img.crop((24, 0, width - 24, height))  
    return img

def save_stages_output(output_stage, sense1, slice_num, epoch, rssq_sens_map, flip=True, crop=False):
    output_stage_ = []
    output_stage_error_hot_ = []

    if flip:
        sense1 = np.flipud(sense1).copy()
        rssq_sens_map = torch.flip(rssq_sens_map, dims=[1])

    if crop:
        sense1 = sense1[..., 24:-24]
        rssq_sens_map = rssq_sens_map[..., 24:-24]

    for i, output_i in enumerate(output_stage):
        if flip:
            output_i = torch.flip(output_i, dims=[2])
        if crop:
            output_i = output_i[..., 24:-24]
        output_stage1 = (output_i[:,0,:,:] + 1j*output_i[:,1,:,:])
        output_stage = output_stage1 * (abs(rssq_sens_map)!=0)
        output_stage_.append(output_stage)

        sense1_tensor = torch.from_numpy(sense1).to(output_stage.device) 
        output_stage_error_hot = 5*(abs(sense1_tensor - output_stage))
        output_stage_error_hot_.append(output_stage_error_hot)


    #############################
    # w/o mask stage-wise results
    #############################

    # Add boundaries
    output_stage = add_boundaries(output_stage_, boundary_width=1, device=output_stage_[0].device)
    output_stage_error_hot = add_boundaries(output_stage_error_hot_, boundary_width=1, device=output_stage_error_hot_[0].device)

    output_stage = output_stage.squeeze().detach().cpu().numpy()
    output_stage_error_hot = output_stage_error_hot.squeeze().detach().cpu().numpy()

    target_dpi = 100
    figsize = (int(output_stage.shape[1]  / 1.5) / target_dpi, int(output_stage.shape[0]*2 / 1.5 / target_dpi))
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    v_max = 0.7    
    axes[0].imshow(abs(output_stage), cmap='gray', vmin=0, vmax=v_max * abs(sense1).max())
    axes[0].axis('off')
    axes[1].imshow(output_stage_error_hot, cmap='hot', vmin=0, vmax=abs(sense1).max())
    axes[1].axis('off') 


    plt.tight_layout(pad=0)
    plt.savefig(conf.out_path + '/test/' + f'{slice_num:03}_recon_stage_{epoch}.png', 
                bbox_inches='tight',
                dpi=target_dpi)
    plt.close(fig)



def plot_l2(l2_loss, stage, slice_num, domain, mean=False):
    plt.close('all')
    plt.plot(range(0, stage), l2_loss)
    plt.xlabel('stage')
    plt.ylabel('l2_norm')
    plt.title('l2_norm vs stage')
    plt.grid(True)
    if mean:
        plt.savefig(conf.out_path+f'/l2_vs_stage_{slice_num}_{domain}.png')
    else:
        plt.savefig(conf.out_path+f'/test/{slice_num:03}_l2_vs_stage_{domain}.png')


def set_global_seed(seed=2025):
    # For PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For NumPy
    np.random.seed(seed)
    random.seed(seed)

