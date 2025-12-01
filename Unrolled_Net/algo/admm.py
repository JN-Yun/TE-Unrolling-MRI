import torch

def admm_unrolling(self, zerofilled, coil, mask, conf):
    u = zerofilled.clone()
    eta = torch.zeros_like(zerofilled)

    output_stage = [u]
    mu = []
    gamma = []

    for i in range(conf.nb_unroll_blocks + 1):

        # --- 1. Data Fidelity ---
        if conf.DC_unshared:
            output, mu_ = self.dc[i](zerofilled, coil, mask, (u - eta))
            if i > 0:
                mu.append(mu_)
        else:
            output, mu = self.dc(zerofilled, coil, mask, (u - eta))

        # --- Exit on last block ---
        if i == conf.nb_unroll_blocks:
            output_stage.append(output)
            break

        # --- gamma selection ---
        if conf.gamma_unshared:
            gamma_ = self.gamma[i]
            gamma.append(gamma_)
        else:
            gamma_ = self.gamma

        # --- 2. Regularizer ---
        if "time_emb" in conf.model:
            t = output.new_tensor([i+1] * output.shape[0]).long().to(output.device)
            u = self.R(output + eta, t)  # t = i
        elif "multi" in conf.model:
            u = self.R[i](output + eta)
        else:
            u = self.R(output + eta)

        # --- 3. Dual Update ---
        eta = eta - gamma_ * (u - output)

        output_stage.append(output)

    if conf.gamma_unshared:
        return output, mu, gamma, output_stage
    else:
        return output, mu, self.gamma, output_stage
