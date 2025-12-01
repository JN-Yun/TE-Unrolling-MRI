import torch
from configs import Config
from utils import cus_fft, cus_ifft


class Data_consistency(torch.nn.Module):

    def __init__(self, mu = 0.05):
        super().__init__()
        self.conf = Config().parse()
        self.mu = torch.nn.Parameter(torch.tensor(mu), requires_grad = True)
        self.warm_start = False

    def E(self, image, coil, mask):
        image = image.repeat(coil.shape[0],1,1)
        return cus_fft(image*coil, [2,3]) * mask
        
    def EH(self, kspace, coil, mask):
        return torch.sum(cus_ifft(kspace*mask, [2,3]) * torch.conj(coil), dim=1)

    def EHE(self, image, coil, mask):
        return self.EH(self.E(image, coil, mask), coil, mask) 

    def forward(self, zerofilled, coil, mask, denoiser_out):
        if self.conf.Unroll_algo == "PGD":
            z = denoiser_out[:,0,:,:] + 1j*denoiser_out[:,1,:,:]
            x_approx = z + self.mu * (zerofilled[:,0,:,:] + 1j*zerofilled[:,1,:,:]) - self.mu * self.EHE(z, coil, mask)
            return torch.stack((torch.real(x_approx), torch.imag(x_approx)), axis=1), self.mu

        else:
            if self.conf.DC_unshared:
                self.mu.data = torch.clamp(self.mu.data, min=1e-5)

            ''' CG Initialization '''
            if self.warm_start:
                x_approx = (zerofilled[:,0,:,:] + 1j*zerofilled[:,1,:,:])   # x_approx -> x_0 = E^Hy
                b_now = (zerofilled[:,0,:,:] + 1j*zerofilled[:,1,:,:]) + (self.mu*(denoiser_out[:,0,:,:] + 1j*denoiser_out[:,1,:,:]))  # b -> (E^Hy + mu*z)
                r_now = b_now - self.EHE(x_approx, coil, mask) - self.mu*x_approx   # r_0 = b - Ax -> b - (E^HE + mu*I)x_0
                p_now = torch.clone(r_now)       
            else:
                r_now = (zerofilled[:,0,:,:] + 1j*zerofilled[:,1,:,:]) + (self.mu*(denoiser_out[:,0,:,:] + 1j*denoiser_out[:,1,:,:])) 
                p_now = torch.clone(r_now)
                x_approx = torch.zeros_like(p_now)

            ''' CG Iteration '''
            for _ in range(self.conf.CG_Iter):
                
                q = self.EHE(p_now, coil, mask) + self.mu * p_now # A * p = (E^HE + mu*I) * p = E^HE(p) + mu*p
                alpha = torch.sum(r_now*torch.conj(r_now)) / torch.sum(q*torch.conj(p_now))
                x_next = x_approx + alpha*p_now
                r_next = r_now - alpha*q
                p_next = r_next + torch.sum(r_next*torch.conj(r_next)) / torch.sum(r_now*torch.conj(r_now)) * p_now
                x_approx = x_next
        
                p_now = torch.clone(p_next)
                r_now = torch.clone(r_next)
                
            return torch.stack((torch.real(x_approx), torch.imag(x_approx)), axis=1), self.mu