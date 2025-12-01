import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from Unrolled_Net.UnrolledNet import UnrolledNet
from Unrolled_Net.data_consistency import Data_consistency
from utils import mask_gen, cal_SSIM, cal_PSNR, load_data_test, save_stages_output
from configs import Config
import tqdm
import scipy.io as sio
from utils import set_global_seed
import json

set_global_seed(2025)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
conf = Config().parse()

if not os.path.exists(conf.out_path+'/test'):
    os.makedirs(conf.out_path+'/test')

''' Loading Model '''
dc = Data_consistency()
epoch = str(conf.weight_epoch).zfill(4)
model = UnrolledNet()
model.load_state_dict(torch.load(conf.out_path+f'/train/weights/model_weights_at_epoch{epoch}.pth'))
print(conf.model)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
print(f"Total number of trainable parameters: {trainable_params}")

if conf.data_type == 'PD':      # Please adjust the following code to match your dataset.
    model = model.cuda().eval().double()
elif conf.data_type == 'PDFS' or conf.data_type == 'AXT2':
    model = model.cuda().eval()

''' Data Loading (Raw kspace, coil sens map, undersampling mask)'''
test_slice, maps = load_data_test(conf.batchSize)
mask = mask_gen(conf.ACS_length, conf.nrow_GLOB, conf.ncol_GLOB, conf.ncoil_GLOB, conf.acc_rate)

n_slices = len(test_slice)
all_psnr = np.zeros((n_slices,1))
all_ssim = np.zeros((n_slices,1))
print(f"Total number of test data: {len(test_slice)}")

l2_norm_ksp_ = []
l2_norm_img_ = []

for slice_num, (test_ksp, test_map) in tqdm.tqdm(enumerate(zip(test_slice, maps)), total=n_slices):

    ksp_valid = test_ksp    # B, C, H, W
    maps_valid = test_map   # B, C, H, W

    ksp_valid = ksp_valid / torch.max(torch.abs(ksp_valid))    
    rssq_sens_map = torch.sqrt(torch.sum(maps_valid**2, dim=1)).cuda()
    sense1 = torch.sum(np.conj(maps_valid) * torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(ksp_valid, dim=[2,3]), dim=[2,3], norm = 'ortho'), dim=[2,3]), dim=1).numpy().squeeze()
        
    zerofilled_ksp = ksp_valid*mask
    zerofilled_img = dc.EH(zerofilled_ksp, maps_valid, mask)
    net_input = torch.stack((zerofilled_img.real, zerofilled_img.imag), axis=1) # B, C, H, W

    net_input = net_input.cuda()
    maps_valid = maps_valid.cuda()

    with torch.no_grad():
        train_output, mu, gamma, output_stage = model(net_input, maps_valid, mask.cuda())
        
    out = (train_output[:,0,:,:] + 1j*train_output[:,1,:,:])
    out = out * (abs(rssq_sens_map)!=0)
    out = out.squeeze().detach().cpu().numpy()

    sense1 = sense1.copy(); out = out.copy()

    psnr_val = cal_PSNR(abs(sense1), abs(out))
    ssim_val = cal_SSIM(abs(sense1), abs(out))
    all_psnr[slice_num,:] = psnr_val
    all_ssim[slice_num,:] = ssim_val

    #############################
    # Save into "/test" directory
    #############################   
    plt.imsave(conf.out_path+'/test/'+f'{slice_num:03}'+'_sense1.png', np.flipud(abs(sense1.squeeze())), cmap='gray', vmax=0.8*abs(sense1).max())
    plt.close('all')

    plt.imsave(conf.out_path+'/test/'+f'{slice_num:03}'+f'_recon_{epoch}_wo_text.png', np.flipud(abs(out)), cmap='gray', vmin = 0, vmax=0.8*abs(sense1).max())
    plt.close('all')


    # Comment out below line if necessary.

    # plt.imshow(np.flipud(abs(out)), cmap='gray', vmin = 0, vmax=0.8*abs(sense1).max())
    # plt.text(7, 313, f'PSNR: {psnr_val:.3f}\nSSIM: {ssim_val:.3f}', color='white', fontsize=15, fontweight='bold', ha='left', va='bottom')
    # plt.axis('off')    
    # plt.savefig(conf.out_path+'/test/'+f'{slice_num:03}'+f'_recon_{epoch}.png', bbox_inches='tight', pad_inches=0)
    # plt.close('all')

    # plt.imsave(conf.out_path+'/test/'+f'{slice_num:03}'+f'_error_{epoch}.png', 5*(np.flipud(abs(sense1.squeeze() - out))), cmap='gray', vmin = 0, vmax = abs(sense1).max())
    # plt.imsave(conf.out_path+'/test/'+f'{slice_num:03}'+f'_error_hot_{epoch}.png', 5*(np.flipud(abs(sense1.squeeze() - out))), cmap='hot', vmin = 0, vmax = abs(sense1).max())
    # plt.close('all')

    # sio.savemat(conf.out_path +f'/test/ref_{slice_num}.mat', {'ref': sense1.squeeze()})
    # sio.savemat(conf.out_path +f'/test/recon_{slice_num}.mat', {'recon': out})

    ################################
    # Save outputs from each stage 
    ################################
    save_stages_output(output_stage, sense1, 
                        slice_num, epoch, 
                        rssq_sens_map)


# Save the PSNR, SSIM 
all_psnr_flat = all_psnr.flatten()
all_ssim_flat = all_ssim.flatten()

if conf.data_type == 'PD':
    # num_drop = 0             # set to the number of drop incdices (e.g., 12) if needed
    # worst_indices = np.argsort(all_psnr_flat)[:num_drop]    
    worst_indices = []
elif conf.data_type == 'PDFS':
    # num_drop = 0             # set to the number of drop incdices (e.g., 14) if needed
    # worst_indices = np.argsort(all_psnr_flat)[:14]  
    worst_indices = []

elif conf.data_type == 'AXT2':
    worst_indices = []

sorted_psnrs = np.delete(all_psnr_flat, worst_indices)
sorted_ssims = np.delete(all_ssim_flat, worst_indices)

results = {
    "overall": {
        "PSNR_mean": float(np.mean(all_psnr)),
        "SSIM_mean": float(np.mean(all_ssim)),
        "PSNR_std": float(np.std(all_psnr)),
        "SSIM_std": float(np.std(all_ssim)),
        "PSNR_median": float(np.median(all_psnr)),
        "SSIM_median": float(np.median(all_ssim)),
    },
    "sorted_removed_worst": {
        "PSNR_mean": float(np.mean(sorted_psnrs)),
        "SSIM_mean": float(np.mean(sorted_ssims)),
        "PSNR_std": float(np.std(sorted_psnrs)),
        "SSIM_std": float(np.std(sorted_ssims)),
        "PSNR_median": float(np.median(sorted_psnrs)),
        "SSIM_median": float(np.median(sorted_ssims)),
        }
    }


with open(conf.out_path + "/Test_results.json", "w") as f:
    json.dump(results, f, indent=4)


sio.savemat(conf.out_path+'/ssims.mat', {'ssims': all_ssim})
sio.savemat(conf.out_path+'/psnrs.mat', {'psnrs': all_psnr})
