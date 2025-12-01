import torch
import torch.nn as nn
from Unrolled_Net.unet.unet import create_model as UNet
from Unrolled_Net.data_consistency import Data_consistency
from Unrolled_Net.resnet.ResNet import ResNet
from Unrolled_Net.resnet.ResNet_time_emb import ResNet_time_emb


def build_regularizer(conf):
    if "Unet_share" in conf.model or "Unet_time_emb" in conf.model:
        return UNet(
            use_time_emb=conf.use_time_emb,
            use_norm=conf.use_norm,
        )

    elif "Unet_multi" in conf.model:
        nets = [UNet(use_time_emb=False,
                     use_norm=conf.use_norm)
                for _ in range(conf.nb_unroll_blocks)]
        return nn.ModuleList(nets)

    elif "ResNet_multi" in conf.model:
        nets = [ResNet(conf.nb_res_blocks) for _ in range(conf.nb_unroll_blocks)]
        return nn.ModuleList(nets)

    elif "ResNet_time_emb" in conf.model:
        return ResNet_time_emb(conf.nb_res_blocks)

    elif "ResNet_share" in conf.model:
        return ResNet(conf.nb_res_blocks)

    else:
        raise ValueError(f"Unknown regularizer type: {conf.model}")


def build_dc(conf):
    if conf.DC_unshared:
        blocks = conf.nb_unroll_blocks + 1 if conf.Unroll_algo in ["ADMM", "VAMP"] else conf.nb_unroll_blocks
        return nn.ModuleList([Data_consistency(mu=conf.mu_init) for _ in range(blocks)])
    else:
        return Data_consistency(mu=conf.mu_init)


def build_gamma(conf):
    if conf.Unroll_algo not in ["ADMM", "VAMP"]:
        return None

    if conf.gamma_unshared:
        return nn.ParameterList([
            nn.Parameter(torch.tensor(conf.gamma_init), requires_grad=True)
            for _ in range(conf.nb_unroll_blocks)
        ])
    else:
        return nn.Parameter(torch.tensor(conf.gamma_init), requires_grad=True)
