import sys
import math
from itertools import combinations

import torch
import pytorch_lightning as pl
import wandb

from torchvision.utils import make_grid
import torchvision.transforms.functional as t_f
from torchvision.transforms import v2

from utils.eval import train_clf_lr_PM, eval_clf_lr_PM
from utils.eval import train_clf_lr_celeba, eval_clf_lr_celeba
from utils.eval import train_clf_lr_scMNC, eval_clf_lr_scMNC
from utils.eval import generate_samples
from utils.eval import conditional_generation
from utils.eval import conditional_generation_cov
from utils.eval import calc_coherence_acc, calc_coherence_ap
from utils.eval import load_modality_clfs
from utils.eval import from_preds_to_acc
from utils.eval import from_preds_to_ap

from utils.text import create_txt_image
from utils.fid import FrechetInceptionDistance, NoTrainInceptionV3
from utils.vae import get_networks


class MVVAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoders, self.decoders, self.cov_mat, self.covariance, self.mu = get_networks(cfg)
        self._update_cov_diag_chol()
        # fixed, data-independent list of non-empty modality-index subsets used
        # by MoPoE aggregation; precomputed once instead of every forward call
        xs = range(0, cfg.dataset.num_views)
        self._mopoe_subsets = [
            tuple(sorted(s))
            for n in range(1, len(xs) + 1)
            for s in combinations(xs, n)
        ]

        if cfg.dataset.name.startswith("PM"):
            self.train_clf_lr = train_clf_lr_PM
            self.eval_clf_lr = eval_clf_lr_PM
            self.eval_downstream_task = self.eval_downstream_task_PM
            self.from_preds_to_clf_metric = from_preds_to_acc
            self.calc_coherence = calc_coherence_acc
            self.modality_names = [
                "m" + str(m) for m in range(0, cfg.dataset.num_views)
            ]
            mod_list = [n for n in range(cfg.dataset.num_views)]
            mod_list.remove(cfg.dataset.modalities_order)
            mod_order = [cfg.dataset.modalities_order] + mod_list 
            self.modality_names_ordered = [
                "m" + str(m) for m in mod_order
            ]
            self.ref_mod_d_size = 3 * 28 * 28
            self.modalities_size = {
                "m" + str(m): 3 * 28 * 28 for m in range(cfg.dataset.num_views)
            }
        elif cfg.dataset.name.startswith("sc"):
            self.train_clf_lr = train_clf_lr_scMNC
            self.eval_clf_lr = eval_clf_lr_scMNC
            self.eval_downstream_task = self.eval_downstream_task_scMNC
            self.from_preds_to_clf_metric = from_preds_to_acc
            self.calc_coherence = calc_coherence_acc
            self.modality_names = ["exp", "feat"]
            mod_list = [n for n in range(cfg.dataset.num_views)]
            self.ref_mod_d_size = 1302
            self.modalities_size = {
                "exp": 1302, "feat": 39
            }
        elif cfg.dataset.name.startswith("celeba"):
            self.train_clf_lr = train_clf_lr_celeba
            self.eval_clf_lr = eval_clf_lr_celeba
            self.eval_downstream_task = self.eval_downstream_task_celeba
            self.calc_coherence = calc_coherence_ap
            self.from_preds_to_clf_metric = from_preds_to_ap
            self.modality_names = ["img", "text"]
            if cfg.dataset.use_rec_weight and cfg.dataset.include_channels_rec_weight:
                self.ref_mod_d_size = 3 * cfg.dataset.img_size * cfg.dataset.img_size
                self.modalities_size = {
                    "img": 3 * cfg.dataset.img_size * cfg.dataset.img_size,
                    "text": cfg.dataset.len_sequence,
                }
            elif cfg.dataset.use_rec_weight:
                self.ref_mod_d_size = cfg.dataset.img_size * cfg.dataset.img_size
                self.modalities_size = {
                    "img": cfg.dataset.img_size * cfg.dataset.img_size,
                    "text": cfg.dataset.len_sequence,
                }
            else:
                self.ref_mod_d_size = cfg.dataset.img_size * cfg.dataset.img_size
                self.modalities_size = {
                    "img": cfg.dataset.img_size * cfg.dataset.img_size,
                    "text": cfg.dataset.img_size * cfg.dataset.img_size,
                }

        self.transforms = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
            ]
        )
        # Plain dicts on purpose: nn.Module.__setattr__ auto-registers any
        # Module assigned directly to an attribute, which would pull the
        # frozen InceptionV3 backbone and the pretrained coherence classifiers
        # into self.parameters() (so the optimizer would train them), into
        # every checkpoint, and into the model summary. Holding them inside a
        # dict keeps them out of the module tree, matching how self.fid_scores
        # already worked.
        self._fid_shared = {}
        self.fid_scores = {}
        self._coherence_clf_cache = {}
        self.initialize_fid_scores()
        
        if cfg.model.schedule == "cyclical":
            self.compute_current_beta = self.beta_cyc_annealing
        if cfg.model.temp_annealing == "cosine":
            self.compute_current_temperature = self.cos_annealing
        elif cfg.model.temp_annealing == "linear":
            self.compute_current_temperature = self.linear_annealing
        elif cfg.model.temp_annealing == "exp":
            self.compute_current_temperature = self.exp_annealing
        else:
            print("annealing schedule not known...exit")
            sys.exit()

        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.save_hyperparameters()

        # buffer for final scores
        self.register_buffer("final_scores_rec_loss", torch.tensor(0))
        self.register_buffer("final_scores_cond_rec_loss", torch.tensor(0))
        self.register_buffer("final_scores_cond_rec_loss_cov", torch.tensor(0))
        self.register_buffer(
            "final_scores_lr_unimodal", torch.zeros(cfg.dataset.num_views)
        )
        self.register_buffer(
            "final_scores_lr_aggregated", torch.zeros(cfg.dataset.num_views)
        )
        self.register_buffer(
            "final_scores_lr_unimodal_alllabels",
            torch.zeros(cfg.dataset.num_views, 1),
        )
        self.register_buffer(
            "final_scores_lr_aggregated_alllabels",
            torch.zeros(cfg.dataset.num_views, 1),
        )
        self.register_buffer(
            "final_scores_coh",
            torch.zeros(
                (cfg.dataset.num_views, cfg.dataset.num_views,1)
            ),
        )
        self.register_buffer(
            "final_scores_coh_cov",
            torch.zeros(
                (cfg.dataset.num_views, cfg.dataset.num_views, 1)
            ),
        )
        self.register_buffer("best_val_loss", torch.tensor(float("inf")))
        self.register_buffer("best_scores_rec_loss", torch.tensor(0.0))
        self.register_buffer("best_scores_cond_rec_loss", torch.tensor(0.0))
        self.register_buffer("best_scores_cond_rec_loss_cov", torch.tensor(0.0))
        self.register_buffer(
            "best_scores_lr_unimodal", torch.zeros(cfg.dataset.num_views)
        )
        self.register_buffer(
            "best_scores_lr_aggregated", torch.zeros(cfg.dataset.num_views)
        )
        self.register_buffer(
            "best_scores_lr_unimodal_alllabels",
            torch.zeros(cfg.dataset.num_views, cfg.dataset.n_clfs_outputs),
        )
        self.register_buffer(
            "best_scores_lr_aggregated_alllabels",
            torch.zeros(cfg.dataset.num_views, cfg.dataset.n_clfs_outputs),
        )
        self.register_buffer(
            "best_scores_coh",
            torch.zeros(
                (cfg.dataset.num_views, cfg.dataset.num_views, 1)
            ),
        )
        self.register_buffer(
            "best_scores_coh_cov",
            torch.zeros(
                (cfg.dataset.num_views, cfg.dataset.num_views, 1)
            ),
        )

    def _snapshot_best_scores(self):
        self.best_scores_rec_loss.copy_(self.final_scores_rec_loss.detach())
        self.best_scores_cond_rec_loss.copy_(self.final_scores_cond_rec_loss.detach())
        self.best_scores_cond_rec_loss_cov.copy_(
            self.final_scores_cond_rec_loss_cov.detach()
        )
        self.best_scores_lr_unimodal.copy_(self.final_scores_lr_unimodal.detach())
        self.best_scores_lr_aggregated.copy_(self.final_scores_lr_aggregated.detach())
        self.best_scores_lr_unimodal_alllabels.copy_(
            self.final_scores_lr_unimodal_alllabels.detach()
        )
        self.best_scores_lr_aggregated_alllabels.copy_(
            self.final_scores_lr_aggregated_alllabels.detach()
        )
        self.best_scores_coh.copy_(self.final_scores_coh.detach())
        self.best_scores_coh_cov.copy_(self.final_scores_coh_cov.detach())

    def _get_current_beta_weight(self):
        if self.cfg.model.beta_annealing:
            return self.compute_current_beta(
                self.cfg.model.init_beta_value,
                self.cfg.model.final_beta_value,
                self.cfg.model.beta_annealing_steps,
                self.cfg.model.beta_M,
                self.cfg.model.beta_R,
            )
        return self.cfg.model.final_beta_value

    def _get_shared_inception(self):
        """One InceptionV3 backbone shared by every FID metric.

        Constructing FrechetInceptionDistance with an int `feature` builds its
        own InceptionV3 and loads the weights file from disk. PolyMNIST needs
        num_views^2 * 2 = 50 of these metrics, so the default path put 50
        identical ~25M-parameter networks on the GPU (~5 GB) and did 50 disk
        loads. The metric also accepts a pre-built Module, so build one and
        hand the same instance to all of them: identical weights, identical
        features, one copy.
        """
        inception = self._fid_shared.get("inception")
        if inception is None:
            inception = NoTrainInceptionV3(
                name="inception-v3-compat",
                features_list=["2048"],
                feature_extractor_weights_path=self.cfg.eval.path_inception_weights,
            )
            # Lets FrechetInceptionDistance size its buffers directly instead of
            # running a probe forward pass to discover the feature width. That
            # probe builds its dummy tensor on the CPU, so it would also fail
            # outright once this backbone has been moved to the GPU.
            inception.num_features = 2048
            self._fid_shared["inception"] = inception
        return inception

    def initialize_fid_scores(self):
        # Called once at construction and again at the start of every
        # FID-logging epoch. Rebuilding the metrics each time meant re-loading
        # InceptionV3 50x per FID epoch; the metrics are stateless apart from
        # their accumulated features, so reset() gives the same clean slate.
        if self.fid_scores:
            for fid in self.fid_scores.values():
                fid.reset()
            return
        inception = self._get_shared_inception()
        for key in self.modality_names:
            for key_tilde in self.modality_names:
                if key_tilde in ("text", "exp", "feat"):
                    continue
                self.fid_scores[key + "_to_" + key_tilde] = FrechetInceptionDistance(
                    feature=inception,
                    compute_on_cpu=True,
                ).to(self.cfg.model.device)
                # storage of results using covariance matrix
                self.fid_scores[key + "_to_" + key_tilde + "_cov"] = FrechetInceptionDistance(
                    feature=inception,
                    compute_on_cpu=True,
                ).to(self.cfg.model.device)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, _ = self.compute_loss("train", batch, out)
        bs = self.cfg.model.batch_size
        if len(self.training_step_outputs) * bs < self.cfg.eval.num_samples_train:
            # detach before storing: these dists are only read back in
            # on_validation_epoch_end, and keeping them attached to the
            # autograd graph for a whole epoch wastes memory
            detached_dists = [
                {key: [mu.detach(), lv.detach()] for key, (mu, lv) in dists.items()}
                for dists in out[1:]
            ]
            self.training_step_outputs.append([detached_dists, batch])
        return loss
      
      
    def _cov_needed_this_epoch(self):
        """Whether the empirical covariance/mu are about to be needed.

        Recomputing them is a second full pass over the training set through
        the encoders, so it is worth skipping on epochs where nothing reads
        them. Three validation branches consume them via
        cond_generate_samples_cov: coherence, FID, and the conditional-
        generation image grids (conditional_generation_cov), each behind its
        own logging frequency.

        The offsets: this hook fires adjacent to the validation loop, and
        which side it lands on -- and therefore whether the estimate produced
        here serves this epoch's validation or the next one's -- depends on
        the Lightning version's epoch-loop ordering. Rather than depend on
        that, cover both. At the default frequency of 50 this still skips
        ~96% of epochs, and it can never leave a stale estimate in place.
        """
        freqs = [
            self.cfg.log.fid_logging_frequency,
            self.cfg.log.img_plotting_frequency,
        ]
        if self.cfg.eval.coherence:
            freqs.append(self.cfg.log.coherence_logging_frequency)
        return any(
            (self.current_epoch + offset) % freq == 0
            for freq in freqs
            for offset in (1, 2)
        )

    def on_train_epoch_end(self):
        if not self._cov_needed_this_epoch():
            return
        self.eval()  # set to eval mode
        dataloader = self.trainer.train_dataloader # doesn't work
        # device = self.device  # current device (CPU or GPU)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = self.cfg.model.device
        total_num_latents = self.cfg.model.latent_dim * self.cfg.dataset.num_views# get correct dim
        mu = torch.zeros(total_num_latents).to(self.cfg.model.device)
        cov = torch.zeros(total_num_latents, total_num_latents).to(self.cfg.model.device)
        num_samples = torch.zeros(1).to(self.cfg.model.device)
        with torch.no_grad():
            for batch in dataloader:
                # batch_outputs = []
                if isinstance(batch, (tuple, list)):
                    x = batch[0]  # assumes first item is input
                else:
                    x = batch
                x = {key: value.to(device) for key, value in x.items()}
                out = self.get_latent_representations(x).to(self.cfg.model.device)
                num_batch = out.shape[0]
                num_samples += num_batch
                # concatenate the outputs
                batch_mu = out.sum(dim=0).to(self.cfg.model.device)
                # mu.append(batch_mu)
                mu += batch_mu
                # outputs.append(out)
                batch_cov = torch.mm(out.T, out)
                cov += batch_cov

            cov /= num_samples - 1
            cov_est = cov - 1 / ((num_samples - 1) * num_samples) * torch.outer(mu, mu)  # empirical covariance

        # to calculate the empirical covariance matrix
        # see if it works at this point
        self.covariance = cov_est.to(self.cfg.model.device)
        mu = mu / num_samples
        self.mu = mu.to(self.cfg.model.device)
        self._update_cov_diag_chol()
        self.train()  # reset to training mode

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss, rec_loss = self.compute_loss("val", batch, out)

        if (self.current_epoch + 1) % self.cfg.log.coherence_logging_frequency == 0:
            if self.cfg.eval.coherence:
                pred_coh, cond_rec_loss, pred_coh_cov, cond_rec_loss_cov = self.evaluate_conditional_generation(
                    out, batch
                )
            else:
                pred_coh, cond_rec_loss, pred_coh_cov, cond_rec_loss_cov = None, None, None, None
        else:
            pred_coh, cond_rec_loss, pred_coh_cov, cond_rec_loss_cov = None, None, None, None
        if (self.current_epoch + 1) % self.cfg.log.fid_logging_frequency == 0:
            if batch_idx == 0:
                self.initialize_fid_scores()
            self.update_fid_scores(out, batch)

        self.last_val_batch = batch
        self.validation_step_outputs.append(
            [out[1:], batch[1], pred_coh, cond_rec_loss, rec_loss, pred_coh_cov, cond_rec_loss_cov]
        )

        if (self.current_epoch + 1) % self.cfg.log.img_plotting_frequency == 0:
            if len(self.validation_step_outputs) >= self.trainer.num_val_batches[0]:
                n_samples_plot = min(100, self.cfg.model.batch_size_eval)
                n_samples_row = int(math.sqrt(n_samples_plot))
                # reconstructions
                for _, key in enumerate(self.modality_names):
                    mod_m = self.last_val_batch[0][key][:n_samples_plot]
                    mod_rec_m = self.get_reconstructions(out[0], key, n_samples_plot)
                    if key == "text" and self.cfg.dataset.name.startswith("celeba"):
                        mod_rec_m, txt_samples_rec = create_txt_image(
                            self.cfg, mod_rec_m
                        )
                        mod_m, txt_samples_gt = create_txt_image(self.cfg, mod_m)
                        self.log_txt_samples(
                            txt_samples_rec, "val/txt_samples", "reconstructions"
                        )
                        self.log_txt_samples(txt_samples_gt, "val/txt_samples", "gt")
                    elif key == "text":
                        continue
                    mod_grid_m = make_grid(
                        torch.cat([mod_m, mod_rec_m], dim=0), nrow=n_samples_row
                    )
                    # bf16-mixed precision means mod_rec_m (a decoder output)
                    # can be bfloat16; to_pil_image() goes through numpy,
                    # which doesn't support that dtype.
                    mod_grid_m = t_f.to_pil_image(mod_grid_m.float())
                    self.logger.log_image(
                        key="reconstructions " + key, images=[wandb.Image(mod_grid_m)]
                    )
        self.log_additional_values(out)
        return loss

    def _get_coherence_clfs(self):
        """Pretrained coherence classifiers, loaded once per process.

        evaluate_conditional_generation runs per validation *batch*, and used
        to call load_modality_clfs() every time -- re-reading the checkpoint
        from disk and rebuilding the classifier ~156x per validation epoch.
        The weights are identical every time, so load once and hold on to it.
        """
        clfs = self._coherence_clf_cache.get("clfs")
        if clfs is None:
            clfs = load_modality_clfs(self.cfg)
            clfs.eval()
            for param in clfs.parameters():
                param.requires_grad_(False)
            self._coherence_clf_cache["clfs"] = clfs
        return clfs

    def evaluate_conditional_generation(self, out, batch):
        dists_enc_out = out[2]
        labels = batch[1]
        data = batch[0]
        n_views = self.cfg.dataset.num_views
        clfs_coherence = self._get_coherence_clfs()

        preds = torch.zeros(
            (
                self.cfg.model.batch_size_eval,
                n_views,
                n_views,
                self.cfg.dataset.n_clfs_outputs,
            ),
            device=self.cfg.model.device,
        )
        preds_cov = torch.zeros(
            (
                self.cfg.model.batch_size_eval,
                n_views,
                n_views,
                self.cfg.dataset.n_clfs_outputs,
            ),
            device=self.cfg.model.device,
        )
        # for m in range(n_views):
        cond_rec = {}
        cond_rec_cov = {}
        for m, key in enumerate(self.modality_names):
            mu_m, lv_m = dists_enc_out[key]
            mods_m_gen = {}
            mods_m_cov_gen = {}
            # sample once per source modality m: every target m_tilde should be
            # conditioned on the same z_m draw, not an independent resample
            z_m = self.reparametrize(mu_m, lv_m)
            # for m_tilde in range(n_views):
            for m_tilde, key_tilde in enumerate(self.modality_names):
                mod_c_gen_m_tilde = self.cond_generate_samples(m_tilde, z_m)
                mod_c_cor_gen_m_tilde = self.cond_generate_samples_cov(m, m_tilde, z_m)
                mods_m_gen[key_tilde] = mod_c_gen_m_tilde[0]
                mods_m_cov_gen[key_tilde] = mod_c_cor_gen_m_tilde[0]
                if m_tilde == m:
                    cond_rec[key] = mod_c_gen_m_tilde
                    cond_rec_cov[key] = mod_c_cor_gen_m_tilde
            preds_m = self.calc_coherence(self.cfg, clfs_coherence, mods_m_gen, labels)
            preds_m_cov = self.calc_coherence(self.cfg, clfs_coherence, mods_m_cov_gen, labels)
            preds[:, m] = preds_m
            preds_cov[:, m] = preds_m_cov
        cond_rec_loss, _, _ = self.compute_rec_loss(data, cond_rec)
        cond_rec_loss_cov, _, _ = self.compute_rec_loss(data, cond_rec_cov)
        return preds, cond_rec_loss, preds_cov, cond_rec_loss_cov

    def get_reconstructions(self, mods_out, key, n_samples):
        raise NotImplementedError

    def cond_generate_samples(self, m, z):
        raise NotImplementedError
    # def cond_generate_samples_cov(self, m_in, m_out, z):
    #     raise NotImplementedError
    
    def get_latent_representations(self, batch):
        data = batch
        # z_ms = {}
        z_ms = []
        for m, key in enumerate(data.keys()):
            # encode views: img_m -> z_m
            mod_m = data[key]
            mu_m, lv_m = self.encoders[m](mod_m)
            # dists_enc_out[key] = [mu_m, lv_m]
            z_m = self.reparametrize(mu_m, lv_m)
            # z_ms[key] = z_m
            z_ms.append(z_m)
        z = torch.cat(z_ms, dim=1) # [z_m1, z_m2, ...]
        return (z)
      
    def extract_relevant_cov(self, m_in, m_out):
        # extract the covariance matrix for the two modalities
        col_start = m_in * self.cfg.model.latent_dim
        col_end = (m_in + 1) * self.cfg.model.latent_dim
        row_start = m_out * self.cfg.model.latent_dim
        row_end = (m_out + 1) * self.cfg.model.latent_dim
        C_m_in_m_out = self.covariance[row_start:row_end, col_start:col_end]
        return C_m_in_m_out

    def _update_cov_diag_chol(self):
        # Precompute the Cholesky factor of each modality's own diagonal
        # covariance block once per covariance update, instead of inverting it
        # from scratch on every single conditional_z call. The same block is
        # reused across every (m_in, m_tilde) pair, for every batch, for the
        # whole epoch, so factorizing once amortizes an O(d^3) op that used to
        # happen up to num_views^2 times per validation batch.
        d = self.cfg.model.latent_dim
        self._cov_diag_chol = [
            torch.linalg.cholesky(
                self.covariance[m * d : (m + 1) * d, m * d : (m + 1) * d]
            )
            for m in range(self.cfg.dataset.num_views)
        ]

    def conditional_z(self, m_in, m_out, z_in):
        C_m_in_m_out = self.extract_relevant_cov(m_in, m_out)
        # assuming mu=0
        z_shift = z_in - self.mu[m_in * self.cfg.model.latent_dim : (m_in + 1) * self.cfg.model.latent_dim]
        # C_m_in_m_out @ inv(C_m_in_m_in) @ z_shift.T, solved via the cached
        # Cholesky factor of C_m_in_m_in rather than forming the inverse
        # explicitly (same result, avoids a redundant O(d^3) factorization).
        rhs = torch.cholesky_solve(torch.transpose(z_shift, 0, 1), self._cov_diag_chol[m_in])
        z_med = torch.mm(C_m_in_m_out, rhs)
        z_out = torch.transpose(z_med, 0, 1) + self.mu[m_out * self.cfg.model.latent_dim : (m_out + 1) * self.cfg.model.latent_dim]
        return z_out

    def cond_generate_samples_cov(self, m_in, m_out, z_in):
        z_out = self.conditional_z(m_in, m_out, z_in)
        mod_c_gen_m_tilde = self.decoders[m_out](z_out)
        return mod_c_gen_m_tilde

    def update_fid_scores(self, out, batch):
        dists_enc_out = out[2]
        imgs = batch[0]
        for m, key in enumerate(self.modality_names):
            mu_m, lv_m = dists_enc_out[key]
            # sample once per source modality m, reused for every target m_tilde
            z_m = self.reparametrize(mu_m, lv_m)
            for m_tilde, key_tilde in enumerate(self.modality_names):
                if key_tilde in ("text", "exp", "feat"):
                    continue
                imgs_m_tilde = imgs[key_tilde]
                fid = self.fid_scores[key + "_to_" + key_tilde]
                # mod_c_gen_m_tilde = self.decoders[m_tilde](z_m)
                mod_c_gen_m_tilde = self.cond_generate_samples(m_tilde, z_m)
                # compute fids between original and generated samples from key_tilde modality_names
                # conditioned on key modality
                fid.update(self.transforms(imgs_m_tilde), real=True)
                fid.update(self.transforms(mod_c_gen_m_tilde[0]), real=False)
                self.fid_scores[key + "_to_" + key_tilde]
                # conditional generation using covaraince matrix
                mod_c_cor_gen_m_tilde = self.cond_generate_samples_cov(m, m_tilde, z_m)
                # fid.reset()
                fid_cov = self.fid_scores[key + "_to_" + key_tilde + "_cov"]
                fid_cov.update(self.transforms(imgs_m_tilde), real=True)
                fid_cov.update(self.transforms(mod_c_cor_gen_m_tilde[0]), real=False)
                self.fid_scores[key + "_to_" + key_tilde + "_cov"]

    def on_validation_epoch_end(self):
        enc_mu_out_train = {key: [] for key in self.modality_names}
        enc_mu_enc_train = {key: [] for key in self.modality_names}
        labels_train = []
        if len(self.training_step_outputs) == 0:
            return
        
        for _, train_out in enumerate(self.training_step_outputs):
            out, batch = train_out
            data, labels = batch
            dists_out = out[0]
            dists_enc = out[1]
            for m, key in enumerate(self.modality_names):
                mu_out_m, lv_out_m = dists_out[key]
                mu_enc_m, lv_enc_m = dists_enc[key]
                enc_mus_out_m = enc_mu_out_train[key]
                enc_mus_enc_m = enc_mu_enc_train[key]
                enc_mus_out_m.append(mu_out_m)
                enc_mus_enc_m.append(mu_enc_m)
                enc_mu_out_train[key] = enc_mus_out_m
                enc_mu_enc_train[key] = enc_mus_enc_m
            labels_train.append(labels)
        for m, key in enumerate(self.modality_names):
            enc_mu_out_m_train = enc_mu_out_train[key]
            enc_mu_out_m_train = torch.cat(enc_mu_out_m_train, dim=0)
            enc_mu_out_train[key] = enc_mu_out_m_train
            enc_mu_enc_m_train = enc_mu_enc_train[key]
            enc_mu_enc_m_train = torch.cat(enc_mu_enc_m_train, dim=0)
            enc_mu_enc_train[key] = enc_mu_enc_m_train
        labels_train = torch.cat(labels_train, dim=0)
          # do everything using training output before this line
        self.training_step_outputs.clear()  # free memory
        
        # select samples for training of classifier
        if (self.current_epoch + 1) % self.cfg.log.downstream_logging_frequency == 0:
            clfs_out = []
            clfs_enc = []
            if self.cfg.eval.eval_downstream_task:
                # train linear clfs on representations that are fed into decoder
                # i.e. after aggregation for joint models
                for m, key in enumerate(data.keys()):
                    enc_mu_out_m_train = enc_mu_out_train[key]
                    clf_out_m = self.train_clf_lr(
                        enc_mu_out_m_train,
                        labels_train,
                    )
                    clfs_out.append(clf_out_m)

                # train linear clfs on representations right after encoders
                # i.e. before aggregation for joint models
                for m, key in enumerate(data.keys()):
                    enc_mu_enc_m_train = enc_mu_enc_train[key]
                    clf_enc_m = self.train_clf_lr(
                        enc_mu_enc_m_train,
                        labels_train,
                    )
                    clfs_enc.append(clf_enc_m)

        enc_mu_out_val = {key: [] for key in self.modality_names}
        enc_lv_out_val = {key: [] for key in self.modality_names}
        enc_mu_enc_val = {key: [] for key in self.modality_names}
        enc_lv_enc_val = {key: [] for key in self.modality_names}
        labels_val = []
        preds_coherence = []
        cond_rec_loss = []
        preds_coherence_cov = []
        preds_coherence_cov = []
        cond_rec_loss_cov = []
        rec_loss = []
        for _, val_out in enumerate(self.validation_step_outputs):
            (
                out,
                labels,
                pred_coh,
                cond_rec_loss_batch,
                rec_loss_batch,
                pred_coh_cov,
                cond_rec_loss_batch_cov,
            ) = val_out
            # imgs, labels = batch
            dists_out = out[0]
            dists_enc_out = out[1]
            for m, key in enumerate(self.modality_names):
                mu_out_m, lv_out_m = dists_out[key]
                mu_enc_m, lv_enc_m = dists_enc_out[key]
                enc_mus_out_m = enc_mu_out_val[key]
                enc_lvs_out_m = enc_lv_out_val[key]
                enc_mus_enc_m = enc_mu_enc_val[key]
                enc_lvs_enc_m = enc_lv_enc_val[key]
                enc_mus_out_m.append(mu_out_m)
                enc_lvs_out_m.append(lv_out_m)
                enc_mus_enc_m.append(mu_enc_m)
                enc_lvs_enc_m.append(lv_enc_m)
                enc_mu_out_val[key] = enc_mus_out_m
                enc_lv_out_val[key] = enc_lvs_out_m
                enc_mu_enc_val[key] = enc_mus_enc_m
                enc_lv_enc_val[key] = enc_lvs_enc_m
            labels_val.append(labels)
            preds_coherence.append(pred_coh)
            cond_rec_loss.append(cond_rec_loss_batch)
            rec_loss.append(rec_loss_batch)
            preds_coherence_cov.append(pred_coh_cov)
            cond_rec_loss_cov.append(cond_rec_loss_batch_cov)
        self.log_additional_values_val()
        self.validation_step_outputs.clear()  # free memory

        self.log("val/loss/avg_rec_loss_epoch", torch.cat(rec_loss).mean())
        self.final_scores_rec_loss = torch.cat(rec_loss).mean()

        for m, key in enumerate(self.modality_names):
              enc_mu_out_m_val = enc_mu_out_val[key]
              enc_mu_out_m_val = torch.cat(enc_mu_out_m_val, dim=0)
              enc_mu_out_val[key] = enc_mu_out_m_val
              enc_lv_out_m_val = enc_lv_out_val[key]
              enc_lv_out_m_val = torch.cat(enc_lv_out_m_val, dim=0)
              enc_lv_out_val[key] = enc_lv_out_m_val
              enc_mu_enc_m_val = enc_mu_enc_val[key]
              enc_mu_enc_m_val = torch.cat(enc_mu_enc_m_val, dim=0)
              enc_mu_enc_val[key] = enc_mu_enc_m_val
              enc_lv_enc_m_val = enc_lv_enc_val[key]
              enc_lv_enc_m_val = torch.cat(enc_lv_enc_m_val, dim=0)
              enc_lv_enc_val[key] = enc_lv_enc_m_val
            
        labels_val = torch.cat(labels_val, dim=0)
        # log coherence results
        if (self.current_epoch + 1) % self.cfg.log.coherence_logging_frequency == 0:
            if self.cfg.eval.coherence:
                # coherence of conditional generation
                pred_coherence = torch.cat(preds_coherence)
                acc_coh = self.from_preds_to_clf_metric(
                    pred_coherence, labels_val, self.modality_names
                )
                self.final_scores_coh = acc_coh
                preds_coherence_cov = torch.cat(preds_coherence_cov)
                acc_coh_cov = self.from_preds_to_clf_metric(
                    preds_coherence_cov, labels_val, self.modality_names
                )
                self.final_scores_coh_cov = acc_coh_cov
                for m, key in enumerate(self.modality_names):
                    for m_tilde, key_tilde in enumerate(self.modality_names):
                        accs_m_m_tilde = acc_coh[m, m_tilde, :].mean()
                        self.log(
                            "val/coherence/" + key + "_to_" + key_tilde,
                            accs_m_m_tilde,
                        )
                        accs_m_m_tilde_cov = acc_coh_cov[m, m_tilde, :].mean()
                        self.log(
                            "val/coherence/" + key + "_to_" + key_tilde + "_cov",
                            accs_m_m_tilde_cov,
                        )

                if self.cfg.dataset.name == "celeba":
                    self.coherence_plot_all_labels_celeba(acc_coh)
                    self.coherence_plot_all_labels_celeba(acc_coh_cov, True)
                    
                self.log(
                    "val/condition_generation/avg_rec_loss",
                    torch.cat(cond_rec_loss).mean(),
                )
                self.log(
                    "val/condition_generation_cov/avg_rec_loss",
                    torch.cat(cond_rec_loss_cov).mean(),
                )
                self.final_scores_cond_rec_loss = torch.cat(cond_rec_loss).mean()
                self.final_scores_cond_rec_loss_cov = torch.cat(cond_rec_loss_cov).mean()

        if (self.current_epoch + 1) % self.cfg.log.downstream_logging_frequency == 0:
          if self.cfg.eval.eval_downstream_task:
              scores_agg = self.eval_downstream_task(
                  "aggregated", clfs_out, enc_mu_out_val, labels_val
              )
              scores_unimodal = self.eval_downstream_task(
                  "unimodal", clfs_enc, enc_mu_enc_val, labels_val
              )
              # Save current final scores
              self.final_scores_lr_unimodal = scores_unimodal.mean(dim=1)
              self.final_scores_lr_aggregated = scores_agg.mean(dim=1)
              self.final_scores_lr_unimodal_alllabels = scores_unimodal
              self.final_scores_lr_aggregated_alllabels = scores_agg

        # check whether this is best validation loss yet and if yes, then save the scores for this epoch
        current_beta = self._get_current_beta_weight()
        if isinstance(current_beta, torch.Tensor):
            current_beta_value = float(current_beta.detach().cpu().item())
        else:
            current_beta_value = float(current_beta)
        is_beta_one = math.isclose(current_beta_value, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        current_val_loss = self.trainer.callback_metrics.get("val/loss/loss")
        if current_val_loss is None:
            current_val_loss = self.final_scores_rec_loss
        if not isinstance(current_val_loss, torch.Tensor):
            current_val_loss = torch.tensor(
                current_val_loss,
                device=self.best_val_loss.device,
                dtype=self.best_val_loss.dtype,
            )
        else:
            current_val_loss = current_val_loss.detach().to(self.best_val_loss.device)

        if is_beta_one and torch.lt(current_val_loss, self.best_val_loss).item():
            self.best_val_loss.copy_(current_val_loss)
            self._snapshot_best_scores()
        
        # log images
        if (self.current_epoch + 1) % self.cfg.log.img_plotting_frequency == 0:
            n_samples_plot = min(100, self.cfg.model.batch_size_eval)
            n_samples_row = int(math.sqrt(n_samples_plot))
            # plotting samples
            # generate samples
            z = torch.randn(
                (100, self.cfg.model.latent_dim), device=self.cfg.model.device
            )
            random_samples = generate_samples(self.decoders, z)
            # for m in range(self.cfg.dataset.num_views):
            for m, key in enumerate(self.modality_names):
                random_gen_m = random_samples[m][:n_samples_plot]
                if key == "text" and self.cfg.dataset.name.startswith("celeba"):
                    random_gen_m, random_txt_samples = create_txt_image(
                        self.cfg, random_gen_m
                    )
                    self.log_txt_samples(
                        random_txt_samples,
                        "val/txt_samples/random",
                        "random generations",
                    )
                elif key == "text":
                    continue
                imgs_grid_m = make_grid(random_gen_m, nrow=n_samples_row)
                # on_validation_epoch_end runs outside autocast, so z and this
                # decoder output are already float32; .float() here is just a
                # defensive no-op guard against that changing later.
                imgs_grid_m = t_f.to_pil_image(imgs_grid_m.float())
                self.logger.log_image(
                    key="random generations " + key,
                    images=[wandb.Image(imgs_grid_m)],
                )

            # conditional generations
            # to start with: we only do conditional generation based on a single modality
            # and generate the remaining modalities
            for m, key in enumerate(self.modality_names):
                mod_m = self.last_val_batch[0][key][-n_samples_plot:]
                # on_validation_epoch_end runs outside Lightning's bf16-mixed
                # autocast context (unlike validation_step, where these mu/lv
                # were originally produced), but decoder weights are always
                # float32 -- so a stored bfloat16 mu/lv fed straight into a
                # decoder here would hit "mat1 and mat2 must have the same
                # dtype". Cast back to float32 at this boundary.
                mu_m_val = enc_mu_enc_val[key][-n_samples_plot:].float()
                lv_m_val = enc_lv_enc_val[key][-n_samples_plot:].float()
                dist_m = [mu_m_val, lv_m_val]
                mod_gen_m = conditional_generation(self, [dist_m])[0]
                mod_gen_m_cov = conditional_generation_cov(self, [dist_m])[0]
                if key == "text" and self.cfg.dataset.name.startswith("celeba"):
                    mod_m, _ = create_txt_image(self.cfg, mod_m)
                elif key == "text":
                    continue
                for m_tilde, key_tilde in enumerate(self.modality_names):
                    mod_gen_m_m_tilde = mod_gen_m[m_tilde]
                    mod_gen_m_m_tilde_cov = mod_gen_m_cov[m_tilde]
                    # replaced key with key_tilde
                    if key_tilde == "text" and self.cfg.dataset.name.startswith("celeba"):
                        mod_gen_m_m_tilde, txt_m_m_tilde = create_txt_image(
                            self.cfg, mod_gen_m_m_tilde
                        )
                        mod_gen_m_m_tilde = mod_gen_m_m_tilde.to(self.cfg.model.device)
                        self.log_txt_samples(
                            txt_m_m_tilde,
                            "val/txt_samples",
                            f"cond_gen_{key}_{key_tilde}",
                        )
                        mod_gen_m_m_tilde_cov, txt_m_m_tilde_cov = create_txt_image(
                            self.cfg, mod_gen_m_m_tilde_cov
                        )
                        mod_gen_m_m_tilde_cov = mod_gen_m_m_tilde_cov.to(
                            self.cfg.model.device
                        )
                        self.log_txt_samples(
                            txt_m_m_tilde_cov,
                            "val/txt_samples",
                            f"cond_gen_cov_{key}_{key_tilde}",
                        )
                    elif key_tilde in ("text", "exp", "feat"):
                        continue
                    mod_grid_m_m_tilde = make_grid(
                        torch.cat(
                            [mod_m.to(self.cfg.model.device), mod_gen_m_m_tilde], dim=0
                        ),
                        nrow=n_samples_row,
                    )
                    mod_grid_m_m_tilde = t_f.to_pil_image(mod_grid_m_m_tilde.float())
                    self.logger.log_image(
                        key="cond_gen_" + key + "_to_" + key_tilde,
                        images=[wandb.Image(mod_grid_m_m_tilde)],
                    )
                    mod_grid_m_m_tilde_cov = make_grid(
                        torch.cat(
                            [mod_m.to(self.cfg.model.device), mod_gen_m_m_tilde_cov],
                            dim=0,
                        ),
                        nrow=n_samples_row,
                    )
                    mod_grid_m_m_tilde_cov = t_f.to_pil_image(mod_grid_m_m_tilde_cov.float())
                    self.logger.log_image(
                        key="cond_gen_cov_" + key + "_to_" + key_tilde,
                        images=[wandb.Image(mod_grid_m_m_tilde_cov)],
                    )
                    # log the images

        if (self.current_epoch + 1) % self.cfg.log.fid_logging_frequency == 0:
            for m, key in enumerate(self.modality_names):
                for m_tilde, key_tilde in enumerate(self.modality_names):
                    if key_tilde in ("text", "exp", "feat"):
                        continue
                    fid = self.fid_scores[key + "_to_" + key_tilde]
                    score_m_m_tilde = fid.compute()
                    self.log(
                        f"val/fid/{key}_to_{key_tilde}",
                        score_m_m_tilde,
                    )
                    fid_cov = self.fid_scores[key + "_to_" + key_tilde + "_cov"]
                    score_m_m_tilde_cov = fid_cov.compute()
                    self.log(
                        f"val/fid/{key}_to_{key_tilde}_cov",
                        score_m_m_tilde_cov,
                    )

    def log_txt_samples(self, txt_samples, str_txt, str_title):
        sample_ids = range(0, len(txt_samples))
        data = [[label, val] for (label, val) in zip(sample_ids, txt_samples)]
        table = wandb.Table(data=data, columns=["id", "txt_sample"])
        wandb.log(
            {
                str_txt: wandb.plot.bar(
                    table,
                    "id",
                    "txt_sample",
                    title=str_title,
                )
            }
        )

    def coherence_plot_all_labels_celeba(self, scores, cov=False):
        for m, key in enumerate(self.modality_names):
            for m_tilde, tilde_key in enumerate(self.modality_names):
                data = [
                    [label, val]
                    for (label, val) in zip(self.label_names, scores[m, m_tilde, :])
                ]
                table = wandb.Table(data=data, columns=["label", "AP"])
                if cov:
                  wandb.log(
                      {
                          "val/coherence/all_labels_"
                          + key: wandb.plot.bar(
                              table,
                              "label",
                              "AP",
                              title="Coherence " + key + " to " + tilde_key,
                          )
                      }
                  )
                  for k, l_name in enumerate(self.label_names):
                      self.log(
                          f"val/coherence/{key}_{tilde_key}/{l_name}",
                          scores[m, m_tilde, k],
                      )
                else:
                  wandb.log(
                      {
                          "val/coherence/all_labels_"
                          + key: wandb.plot.bar(
                              table,
                              "label",
                              "AP",
                              title="Coherence " + key + " to " + tilde_key + "_cov",
                          )
                      }
                  )
                  for k, l_name in enumerate(self.label_names):
                      self.log(
                          f"val/coherence/{key}_{tilde_key}_cov/{l_name}",
                          scores[m, m_tilde, k],
                   )                  
                   

    def eval_downstream_task_PM(self, str_ds, clfs, enc_mu_val, labels_val):
        scores = torch.zeros((self.cfg.dataset.num_views, 1))
        for m, key in enumerate(self.modality_names):
            clf_m = clfs[m]
            enc_mu_m_val = enc_mu_val[key]
            bal_acc_m = self.eval_clf_lr(
                clf_m,
                enc_mu_m_val,
                labels_val,
            )
            self.log("val/downstream/" + str_ds + "/" + key, bal_acc_m.mean())
            scores[m, 0] = bal_acc_m.mean()
        return scores
      
    def eval_downstream_task_scMNC(self, str_ds, clfs, enc_mu_val, labels_val):
        scores = torch.zeros((self.cfg.dataset.num_views, 1))
        for m, key in enumerate(self.modality_names):
            clf_m = clfs[m]
            enc_mu_m_val = enc_mu_val[key]
            bal_acc_m = self.eval_clf_lr(
                clf_m,
                enc_mu_m_val,
                labels_val,
            )
            self.log("val/downstream/" + str_ds + "/" + key, bal_acc_m.mean())
            scores[m, 0] = bal_acc_m.mean()
        return scores

    def eval_downstream_task_celeba(self, str_ds, clfs, enc_mu_val, labels_val):
        n_labels = labels_val.shape[1]
        scores = torch.zeros((self.cfg.dataset.num_views, n_labels))
        for m, key in enumerate(self.modality_names):
            clf_m = clfs[m]
            enc_mu_m_val = enc_mu_val[key]
            scores_m = self.eval_clf_lr(
                clf_m,
                enc_mu_m_val,
                labels_val,
            )
            scores[m, :] = scores_m
            self.log("val/downstream/" + str_ds + "/" + key, scores_m.mean())
            data = [[label, val] for (label, val) in zip(self.label_names, scores_m)]
            table = wandb.Table(data=data, columns=["label", "AP"])
            wandb.log(
                {
                    "val/downstream_lr/"
                    + str_ds
                    + "/all_labels_"
                    + key: wandb.plot.bar(
                        table,
                        "label",
                        "AP",
                        title=str_ds + " Latent Representation Evaluation " + key,
                    )
                }
            )
            # remove saving all labels
            # for k, l_name in enumerate(self.label_names):
            #     # self.log(f"val/downstream/{str_ds}/{key}/{l_name}", scores_m[k])
                
        return scores
      
    def kl_div_cov(self, dist, cov_inv):
        mu, lv = dist
        # Compute the KL divergence between the two distributions
        kld = self.calc_kl_divergence_cov(mu, lv, cov_inv)
        return kld
      
    def kl_div_orthog(self, dist, cov_scalar):
        mu, lv = dist
        # Compute the KL divergence between the two distributions
        kld = self.calc_kl_divergence_orthog(mu, lv, cov_scalar)
        return kld

    def kl_div_z(self, dist):
        mu, lv = dist
        kld = self.calc_kl_divergence(mu, lv)
        return kld

    def kl_div_z_two_dists(self, dist1, dist2):
        mu1, lv1 = dist1
        mu2, lv2 = dist2
        kld = self.calc_kl_divergence(mu1, lv1, mu2, lv2)
        return kld
    

    def calc_kl_divergence(self, mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
        if mu1 is None or logvar1 is None:
            kld = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0, dim=-1)
        else:
            kld = -0.5 * (
                torch.sum(
                    1
                    - logvar0.exp() / logvar1.exp()
                    - (mu0 - mu1).pow(2) / logvar1.exp()
                    + logvar0
                    - logvar1,
                    dim=-1,
                )
            )
        if norm_value is not None:
            kld = kld / float(norm_value)
        return kld
    
    def calc_kl_divergence_orthog(self, mu0, logvar0, cov_scalar,           
                                  norm_value=None):
        if cov_scalar is None:
            kld = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0, dim=-1)
        else:
            kld = -0.5 * (
                torch.sum(
                    1
                    - logvar0.exp() / (cov_scalar)
                    - (mu0).pow(2) / (cov_scalar)
                    + logvar0
                    - math.exp(cov_scalar),
                    dim=-1,
                )
            )
        if norm_value is not None:
            kld = kld / float(norm_value)
        return kld
    
    def calc_kl_divergence_cov(self, mu0, logvar0, cov_inv=None,
                               norm_value=None):
       
       cov_inv_device = cov_inv.to(self.device)
       mu0_device = mu0.to(self.device)
       mu_term =  torch.matmul(torch.matmul(mu0_device, cov_inv_device), mu0_device)
       kld_sum_term = -0.5 * torch.sum(1 + logvar0, dim=-1)
       det_term = torch.linalg.slogdet(cov_inv)
       trace_term = torch.einsum('ii,i->', cov_inv_device, logvar0.exp())
       kld = - det_term + kld_sum_term + mu_term + trace_term 
       if norm_value is not None:
          kld = kld / float(norm_value)
       return kld

    def compute_rec_loss(self, data, data_rec):
        rec_loss_all = []
        rec_loss_mods = {}
        rec_loss_mods_weighted = {}
        # output probability x_m
        for m, key in enumerate(data.keys()):
            mod_gt_m = data[key]
            mod_rec_m = data_rec[key]
            rec_weight_m = float(self.ref_mod_d_size / self.modalities_size[key])
            if key == "text":
                mod_d_out_m = (
                        torch.distributions.one_hot_categorical.OneHotCategorical(
                            logits=mod_rec_m[0], validate_args=False
                        )
                    )
                log_p_mod_m = mod_d_out_m.log_prob(mod_gt_m).sum(dim=[1])
            elif key in ("exp", "feat"):
                mod_d_out_m = torch.distributions.laplace.Laplace(
                    mod_rec_m[0], 0.75
                )
                log_p_mod_m = mod_d_out_m.log_prob(mod_gt_m).sum(dim=[1])
            else:
                mod_d_out_m = torch.distributions.laplace.Laplace(
                    mod_rec_m[0], 0.75
                )
                log_p_mod_m = mod_d_out_m.log_prob(mod_gt_m).sum(dim=[1, 2, 3])
                
            rec_loss_mods[key] = log_p_mod_m.mean(dim=0)
            rec_loss_mods_weighted[key] = (rec_weight_m * log_p_mod_m).mean(dim=0)
            rec_loss_all.append(rec_weight_m * log_p_mod_m.unsqueeze(1))
        rec_loss_avg = -torch.cat(rec_loss_all, dim=1).sum(dim=1)
        return rec_loss_avg, rec_loss_mods, rec_loss_mods_weighted

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.lr,
        )
        return {
            "optimizer": optimizer,
        }

    def aggregate_latents_avg(self, mus, lvs):
        num_views = mus.shape[1]
        mu_agg = (mus.sum(dim=1) / float(num_views)).squeeze(1)
        lv_agg = (lvs.exp().sum(dim=1) / float(num_views)).log().squeeze(1)
        return mu_agg, lv_agg

    def aggregate_latents_moe(self, mus, lvs):
        num_components = mus.shape[1]
        num_samples = mus.shape[0]
        w_mods = float(1 / num_components)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k - 1])
            if k == num_components - 1:
                i_end = num_samples
            else:
                i_end = i_start + int(num_samples * w_mods)
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples
        mu_sel = torch.cat(
            [mus[idx_start[k] : idx_end[k], k, :] for k in range(num_components)], dim=0
        )
        logvar_sel = torch.cat(
            [lvs[idx_start[k] : idx_end[k], k, :] for k in range(num_components)], dim=0
        )
        return mu_sel, logvar_sel

    def aggregate_latents_poe(self, mus, lvs):
        log_precisions = -lvs
        precisions = log_precisions.exp()
        joint_log_precision = torch.logsumexp(log_precisions, dim=1)
        joint_lv = -joint_log_precision
        joint_var = joint_lv.exp()
        joint_mu = precisions.mul(mus).sum(dim=1) * joint_var
        return joint_mu, joint_lv

    def aggregate_latents_mopoe(self, mus, lvs):
        # self._mopoe_subsets (precomputed once in __init__) is the fixed list
        # of non-empty modality-index subsets -- it only depends on num_views,
        # not on data, so there is no need to rebuild it via itertools on every
        # forward call. Each subset is also gathered in one indexing op instead
        # of a per-modality Python loop + concat.
        mus_subsets = []
        lvs_subsets = []
        for mod_indices in self._mopoe_subsets:
            mus_sub = mus[:, mod_indices, :]
            lvs_sub = lvs[:, mod_indices, :]
            mu_sub, lv_sub = self.aggregate_latents_poe(mus_sub, lvs_sub)
            mus_subsets.append(mu_sub.unsqueeze(1))
            lvs_subsets.append(lv_sub.unsqueeze(1))
        mus_subsets = torch.cat(mus_subsets, dim=1)
        lvs_subsets = torch.cat(lvs_subsets, dim=1)
        mu_agg, lv_agg = self.aggregate_latents_moe(mus_subsets, lvs_subsets)
        return mu_agg, lv_agg

    def reparametrize(self, mu, logvar):
        """
        Reparametrized sampling from gaussian
        """
        # dist = torch.distributions.normal.Normal(mu, log_sigma.exp() + 1e-6)
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
        # return dist.rsample()

    # def compute_current_temperature(self, init_temp=None, final_temp=None, num_steps_annealing=None):

    def exp_annealing(self, init_temp=None, final_temp=None, num_steps_annealing=None):
        """
        Compute temperature based on current step
        -> exponential temperature annealing
        """
        if init_temp is None:
            init_temp = self.cfg.init_temp
        if final_temp is None:
            final_temp = self.cfg.model.final_temp
        if num_steps_annealing is None:
            num_steps_annealing = self.cfg.model.num_steps_annealing
        rate = (math.log(final_temp + 1e-10) - math.log(init_temp + 1e-10)) / float(
            num_steps_annealing
        )
        curr_temp = max(init_temp * math.exp(rate * self.global_step), final_temp)
        return curr_temp
    
    def beta_cyc_annealing(
          self, init_temp=None, final_temp=None, num_steps_annealing=None, M=None, R=None
      ):
          if init_temp is None:
              init_temp = self.cfg.init_temp
          if final_temp is None:
              final_temp = self.cfg.model.final_temp
          if num_steps_annealing is None:
              num_steps_annealing = self.cfg.model.num_steps_annealing
          if M is None:
              M = self.cfg.model.M
          if R is None:
              R = self.cfg.model.R

          interval = math.ceil(num_steps_annealing / M)
          tau = (self.global_step % interval) 
          if tau < (R * interval):              
              curr_temp = (1 - tau / interval) * init_temp + (
                  tau / interval
              ) * final_temp

          else:
              curr_temp = final_temp
          return curr_temp

    def cos_annealing(self, init_temp=None, final_temp=None, num_steps_annealing=None):
        """
        Compute temperature based on current step
        -> cosine temperature annealing
        """
        if init_temp is None:
            init_temp = self.cfg.init_temp
        if final_temp is None:
            final_temp = self.cfg.model.final_temp
        if num_steps_annealing is None:
            num_steps_annealing = self.cfg.model.num_steps_annealing
        curr_temp_cos = final_temp + 0.5 * (init_temp - final_temp) * (
            1
            + torch.cos(
                torch.tensor((self.global_step / num_steps_annealing) * math.pi)
            )
        )
        if self.global_step < num_steps_annealing:
            curr_temp = curr_temp_cos
        else:
            curr_temp = final_temp 
        # curr_temp = torch.cos((self.global_step / num_steps_annealing) * math.pi / 2)
        # rate = (math.log(final_temp + 1e-10) - math.log(init_temp + 1e-10)) / float(num_steps_annealing)
        # curr_temp = max(init_temp * math.exp(rate * self.global_step), final_temp)
        return curr_temp

    def linear_annealing(
        self, init_temp=None, final_temp=None, num_steps_annealing=None
    ):
        if init_temp is None:
            init_temp = self.cfg.init_temp
        if final_temp is None:
            final_temp = self.cfg.model.final_temp
        if num_steps_annealing is None:
            num_steps_annealing = self.cfg.model.num_steps_annealing

        if self.global_step < num_steps_annealing:
            curr_temp = (1 - self.global_step / num_steps_annealing) * init_temp + (
                self.global_step / num_steps_annealing
            ) * final_temp

        else:
            curr_temp = final_temp
        return curr_temp

    def assign_label_names(self, label_names):
        self.label_names = label_names
