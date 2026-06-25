import torch
from mv_vaes.mv_vae import MVVAE


class MVJointVAE(MVVAE):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.model.aggregation == "moe":
            self.aggregate_latents = self.aggregate_latents_moe
        elif cfg.model.aggregation == "poe":
            self.aggregate_latents = self.aggregate_latents_poe
        elif cfg.model.aggregation == "mopoe":
            self.aggregate_latents = self.aggregate_latents_mopoe
        else:
            self.aggregate_latents = self.aggregate_latents_avg
        self.save_hyperparameters()

    def log_additional_values(self, out):
        pass

    def log_additional_values_val(self):
        pass

    def forward(self, batch):
        data = batch[0]
        labels = batch[1]

        mus = []
        lvs = []
        dists_enc_out = {}
        # encode views
        # for m in range(0, self.cfg.dataset.num_views):
        for m, key in enumerate(self.modality_names):
            mod_m = data[key]
            mu_m, lv_m = self.encoders[m](mod_m)
            # [OPTIMISATION] Clone tensors outputted by compiled modules to prevent PyTorch from overwriting static memory pools when using CUDA Graphs.
            mu_m, lv_m = mu_m.clone(), lv_m.clone()
            mus.append(mu_m.unsqueeze(1))
            lvs.append(lv_m.unsqueeze(1))
            dists_enc_out[key] = [mu_m, lv_m]
        mus = torch.cat(mus, dim=1)
        lvs = torch.cat(lvs, dim=1)

        mu_out, lv_out = self.aggregate_latents(mus, lvs)
        z_out = self.reparametrize(mu_out, lv_out)

        # decode views
        dists_out = {}
        mods_rec = {}
        # for m in range(0, self.cfg.dataset.num_views):
        for m, key in enumerate(self.modality_names):
            mod_hat_m = self.decoders[m](z_out)
            # [OPTIMISATION] Clone decoder outputs to prevent inplace CUDA Graphs memory errors during compilation.
            if isinstance(mod_hat_m, tuple):
                mod_hat_m = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in mod_hat_m)
            elif isinstance(mod_hat_m, torch.Tensor):
                mod_hat_m = mod_hat_m.clone()
            mods_rec[key] = mod_hat_m

            dist_out_m = [mu_out, lv_out]
            dists_out[key] = dist_out_m
        return (mods_rec, dists_out, dists_enc_out)

    def get_reconstructions(self, mods_out, key, n_samples):
        mod_rec = mods_out[key][0][:n_samples]
        return mod_rec

    def cond_generate_samples(self, m, z):
        mod_c_gen_m_tilde = self.decoders[m](z)
        return mod_c_gen_m_tilde

    def compute_loss(self, str_set, batch, forward_out):
        data, _ = batch
        data_rec = forward_out[0]
        dists_out = forward_out[1]
        
        if self.cfg.model.beta_annealing:
            init_temp = self.cfg.model.init_beta_value
            final_temp = self.cfg.model.final_beta_value
            annealing_steps = self.cfg.model.beta_annealing_steps
            M = self.cfg.model.beta_M
            R = self.cfg.model.beta_R
            beta_weight = self.compute_current_beta(
                init_temp, final_temp, annealing_steps, M, R
            )
        else:
            beta_weight = self.cfg.model.final_beta_value
        self.log("beta annealing", beta_weight)
        
        # kl divergence of latent distribution
        all_mus = torch.stack([dists_out[k][0] for k in self.modality_names], dim=1)
        all_lvs = torch.stack([dists_out[k][1] for k in self.modality_names], dim=1)
        klds_sum = (-0.5 * torch.sum(1 - all_lvs.exp() - all_mus.pow(2) + all_lvs, dim=-1)).sum(dim=1)

        ## compute reconstruction loss/ conditional log-likelihood out data
        ## given latents
        loss_rec, loss_rec_mods, loss_rec_mods_weighted = self.compute_rec_loss(
            data, data_rec
        )
        for m, key in enumerate(self.modality_names):
            self.log(
                f"{str_set}/loss/weighted_rec_loss_{key}",
                loss_rec_mods_weighted[key],
            )
            self.log(
                f"{str_set}/loss/rec_loss_{key}",
                loss_rec_mods[key],
            )

        loss_mv_vae = (loss_rec + beta_weight * klds_sum).mean(dim=0)
        total_loss = loss_mv_vae
        # logging
        self.log(str_set + "/loss/klds_avg", klds_sum.mean(dim=0))
        self.log(str_set + "/loss/loss_rec", loss_rec.mean(dim=0))
        self.log(str_set + "/loss/mv_vae", loss_mv_vae)
        self.log(str_set + "/loss/loss", total_loss)
        return total_loss, loss_rec
