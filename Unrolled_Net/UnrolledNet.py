import torch
import torch.nn as nn
from configs import Config

from Unrolled_Net.algo.admm import admm_unrolling
from Unrolled_Net.algo.vamp import vamp_unrolling
from Unrolled_Net.algo.pgd import pgd_unrolling
from Unrolled_Net.algo.vsqp import vsqp_unrolling

from Unrolled_Net.build_architecture import build_regularizer, build_dc, build_gamma

conf = Config().parse()
    
class UnrolledNet(nn.Module):
    def __init__(self):
        super(UnrolledNet, self).__init__()
        self.conf = conf
        
        self.R = build_regularizer(conf)     # 1. define Regularizer #
        self.dc = build_dc(conf)             # 2. define DC (Default : shared DC). if set conf.DC_unshared, using unshared DC #
        self.gamma = build_gamma(conf)       # 3. define gamma for ADMM and VAMP #

        # 4. Unrolling method #
        self.method_map = {
            "ADMM": admm_unrolling,
            "VAMP": vamp_unrolling,
            "PGD": pgd_unrolling,
            "VSQP": vsqp_unrolling,
        }

    def forward(self, zerofilled, coil, mask):
        method = self.method_map.get(self.conf.Unroll_algo, None)
        if method is None:
            raise ValueError(f"Unknown method: {self.conf.Unroll_algo}")

        return method(self, zerofilled, coil, mask, self.conf)
