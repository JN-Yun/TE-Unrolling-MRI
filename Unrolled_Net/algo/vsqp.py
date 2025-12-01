import torch

def vsqp_unrolling(self, zerofilled, coil, mask, conf):
    output = zerofilled.clone()
    output_stage = [output]
    mu = []

    for i in range(conf.nb_unroll_blocks):

        # --- 1. Regularizer ---
        if "time_emb" in conf.model:
            t = output.new_tensor([i+1] * output.shape[0]).long().to(output.device)
            output = self.R(output, t)  # t = i
        elif "multi" in conf.model:
            output = self.R[i](output)
        else:
            output = self.R(output)

        # --- 2. Data Fidelity ---
        if conf.DC_unshared:
            output, mu_ = self.dc[i](zerofilled, coil, mask, output)
            mu.append(mu_)
        else:
            output, mu = self.dc(zerofilled, coil, mask, output)

        output_stage.append(output)

    return output, mu, None, output_stage
