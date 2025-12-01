import torch 
import numpy as np
import os
from configs import Config
import tqdm
import matplotlib.pyplot as plt
from Unrolled_Net.UnrolledNet import UnrolledNet
from Unrolled_Net.data_consistency import Data_consistency
from utils import *
import json

from utils import set_global_seed
set_global_seed(2025)

def main():
    conf = Config().parse()
    conf = TrackConf(conf)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    prepare_folders()

    
    ''' Data Loading (Raw kspace, coil sens map, undersampling mask)'''
    train_slices, maps = load_data(conf.ntrain_slices, conf.batchSize)
    mask = mask_gen(conf.ACS_length, conf.nrow_GLOB, conf.ncol_GLOB, conf.ncoil_GLOB, conf.acc_rate)
    mask_ksp = mask_ksp_gen(conf.nrow_GLOB, conf.ncol_GLOB, conf.ncoil_GLOB) 
    mask = mask.cuda(); mask_ksp = mask_ksp.cuda()
    
    plt.imsave(conf.out_path+"/train/outputs/mask.png", abs(mask.squeeze()[0].cpu().numpy()), cmap= 'gray')  # Save a mask for visualization
    
    ''' Model '''
    dc = Data_consistency()
    model = UnrolledNet().cuda()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print(f"Total number of trainable parameters: {trainable_params}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = conf.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr = conf.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)   # Set conf.LR_sch to True if scheduler needed (Default : False)

    # check configuration
    print(
        '*' * 60,
        f"\nUnroll Algo   : {conf.Unroll_algo}\n"
        f"Acc           : {conf.acc_rate}\n"
        f"Dataset       : {conf.data_type}\n"
        f"Dataset size  : {len(train_slices)}\n"
        f"Model         : {conf.model}\n"
        f"Domain        : {conf.domain}\n"
        f"DC_unshared   : {conf.DC_unshared}\n"
        f"Unroll Blocks : {conf.nb_unroll_blocks}\n"
    )

    
    ''' Training Process '''
    train_loss_epoch, mu_values, psnr_values, ssim_values = [], [], [], []
    gamma_values = []
    num_val = 0

    for epoch in tqdm.tqdm(range(conf.epochs), ncols=60):
        train_slice_loss = []
        for slice_num, slice_data in enumerate(train_slices):
            slice_data = slice_data.cuda()

            ### 1. Raw kspace & Sens_map###
            full_ksp = slice_data.permute(0,3,1,2) # (B,H,W,C) -> (B,C,H,W)
            normalized_ksp = full_ksp / torch.max(torch.abs(full_ksp)) # Normalize between -1-1
            sens_map = maps[slice_num, :, :, :].unsqueeze(0).permute(0,3,1,2).cuda() # (B,C,H,W)
            sense1 = torch.sum(torch.conj(sens_map) * cus_ifft(normalized_ksp, [2,3]), dim=1)
                       
            ### 2. Zero-filled data with undersampled mask ###
            zerofilled_ksp = normalized_ksp * mask # (B,C,H,W)
            zerofilled_img = dc.EH(zerofilled_ksp, sens_map, mask) # (B,H,W)  

            ### 3. Feeding input into unrolled network ###
            net_input = torch.stack((zerofilled_img.real, zerofilled_img.imag), axis=1) # (B,2,H,W)
            train_output, mu, gamma, stage_output = model(net_input, sens_map, mask) # (B,2,H,W)
            train_output = (train_output[:,0,:,:] + 1j*train_output[:,1,:,:]) # (B,2,H,W) -> # (B,1,H,W) complex
            
            ### 4. Calculate the {ksp, img} loss ###
            # Choose loss domain = {ksp, img} / default == "img"
            if conf.domain == "ksp":
                train_output_ = train_output.repeat(sens_map.shape[0],1,1)
                train_output_ = cus_fft(train_output_*sens_map, [2,3])
                loss = l1_L2_loss(normalized_ksp * mask_ksp, train_output_ * mask_ksp) 
            elif conf.domain == "img":
                loss = l1_L2_loss(sense1, train_output)

            train_slice_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if conf.LR_sch:    
                scheduler.step() 
        train_loss_epoch.append(np.mean(torch.tensor(train_slice_loss).detach().numpy()))


        # For visualization
        if conf.DC_unshared:
            mu_values.append([m.detach().cpu().numpy() for m in mu])
        else:
            mu_values.append(mu.detach().cpu().numpy())

        if gamma is not None:
            if conf.gamma_unshared:
                gamma_values.append([g.detach().cpu().numpy() for g in gamma])
            else:
                gamma_values.append(gamma.detach().cpu().numpy())

        if (epoch+1) % conf.save_freq == 0:

            torch.save(model.state_dict(), conf.out_path+'/train/weights/model_weights_at_epoch{:04}.pth'.format(epoch+1))
            
            plt.imsave(conf.out_path+"/train/outputs/output.png", abs(train_output[0].squeeze().detach().cpu().numpy()), cmap= 'gray', vmin = 0, vmax = abs(train_output[0]).max())
            plt.imsave(conf.out_path+"/train/outputs/sense1.png".format(epoch), abs(sense1.squeeze().cpu().numpy()), cmap= 'gray', vmin = 0, vmax = abs(sense1).max())
            plt.imsave(conf.out_path+"/train/outputs/zerofilled_img.png".format(epoch), abs(zerofilled_img.squeeze().cpu().numpy()), cmap= 'gray', vmin = 0, vmax = abs(sense1).max())

            if conf.DC_unshared:
                plot_mu_multi(np.array(mu_values), (epoch + 1))
            else:
                plot_mu(np.array(mu_values), (epoch + 1))

            if gamma is not None: 
                gamma_array = np.array(gamma_values)
                if conf.gamma_unshared: 
                    plot_gamma_multi(gamma_array, epoch + 1)
                else:
                    plot_gamma(gamma_array, epoch + 1)

            plot_loss(np.array(train_loss_epoch), (epoch + 1))
   

    plot_loss(train_loss_epoch, conf.epochs)

    if conf.DC_unshared:
        plot_mu_multi(np.array(mu_values), (conf.epochs))
    else:
        plot_mu(np.array(mu_values), (conf.epochs))
                
    with open(conf.out_path + "/config_used.json", "w") as f:
        json.dump(conf.used_dict, f, indent=4)

if __name__ == '__main__':
    main()
